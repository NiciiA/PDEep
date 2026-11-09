"""B0/B1: independent learned inverse refinement, final fields are physical solves.

Both variants share source patterns/coefficient semantics, never parameter storage.
All recurrence is local to one infer() call, reset for each new observation set.
"""
from dataclasses import dataclass,replace
import math
import torch
from torch import nn
from .batches import masked_values,validate_prediction
from .core import require
from .diagnostics import RandomStreams
from .encoding import EquationEncoder,EquationVocabulary,ObservationEncoder,pack_graphs,masked_pool,module_input
from .geometry import geometry_for
from .memory import masked_softmax
from .b_physics import build_system,cpu64,physical_solution,anchor_projections


@dataclass(frozen=True)
class BAblation:
    spatial: bool=True
    spatial_attention: bool=True
    physical_projections: bool=True

    def __post_init__(self):
        require(all(type(v) is bool for v in (self.spatial,self.spatial_attention,self.physical_projections)),'B ablation flags must be bool')


class TokenLayer(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(96,288);self.output=nn.Linear(96,96)
        self.norm=nn.LayerNorm(96);self.ff=nn.Sequential(nn.Linear(96,192),nn.SiLU(),nn.Linear(192,96));self.ff_norm=nn.LayerNorm(96)

    def forward(self,tokens,valid):
        B,N,_=tokens.shape
        q,k,v=(x.reshape(B,N,4,24).transpose(1,2) for x in self.qkv(tokens).chunk(3,-1))
        weights=masked_softmax(q@k.transpose(-1,-2)/math.sqrt(24),valid[:,None,None,:]&valid[:,None,:,None])
        x=self.norm(tokens+self.output((weights@v).transpose(1,2).reshape(B,N,96)))
        return masked_values(self.ff_norm(x+self.ff(x)),valid)


class CoefficientMemory(nn.Module):
    def __init__(self):
        super().__init__();self.mode=nn.Embedding(9,16);self.block=nn.Embedding(2,16)
        self.numeric=nn.Sequential(nn.Linear(40,96),nn.SiLU(),nn.Linear(96,96))
        self.context=nn.Linear(192,96);self.norm=nn.LayerNorm(96)
        self.layers=nn.ModuleList([TokenLayer(),TokenLayer()])
        self.diagonal=nn.Linear(96,1);self.step=nn.Sequential(nn.Linear(192,96),nn.SiLU(),nn.Linear(96,1))

    def tokenize(self,xi,mean,scale,g,previous,active,equation,observations,k):
        # Physical values + mean/scale + stabilized standardized current gradient.
        features=torch.stack((xi,mean,scale,(xi-mean)/scale,torch.asinh(g*scale),previous/scale,scale.log(),torch.full_like(xi,k/3)),dim=-1)
        ids=torch.arange(18,device=xi.device);mode=self.mode(ids%9)[None].expand(len(xi),-1,-1);block=self.block(ids//9)[None].expand(len(xi),-1,-1)
        tokens=self.norm(self.numeric(torch.cat((features,mode,block),-1))+self.context(torch.cat((equation,observations),-1))[:,None])
        return masked_values(tokens,active)

    def refine(self,tokens,active,equation):
        for layer in self.layers:tokens=layer(tokens,active)
        d=.25+1.5*self.diagonal(tokens)[...,0].sigmoid() # (.25,1.75)
        fraction=.05+.9*self.step(torch.cat((masked_pool(tokens,active),equation),-1))[:,0].sigmoid()
        return d,fraction,tokens


class SpatialBlock(nn.Module):
    def __init__(self,neighbors):
        super().__init__();self.register_buffer('neighbors',neighbors)
        self.q,self.k,self.v=(nn.Linear(96,96) for _ in range(3));self.output=nn.Linear(96,96)
        self.bias=nn.Sequential(nn.Linear(7,32),nn.SiLU(),nn.Linear(32,4));self.norm=nn.LayerNorm(96)
        self.local=nn.Linear(96,96);self.global_mix=nn.Linear(96,96);self.mix_norm=nn.LayerNorm(96)
        self.ff=nn.Sequential(nn.Linear(96,192),nn.SiLU(),nn.Linear(192,96));self.ff_norm=nn.LayerNorm(96)

    def forward(self,memory,obs,anchors,*,attention_enabled):
        B,M,_=memory.shape;N=obs.tokens.shape[1];weights=None
        if attention_enabled:
            q=self.q(memory).reshape(B,M,4,24).transpose(1,2);k=self.k(obs.tokens).reshape(B,N,4,24).transpose(1,2);v=self.v(obs.tokens).reshape(B,N,4,24).transpose(1,2)
            delta=anchors[None,:,None]-obs.position[:,None]
            meta=torch.cat((obs.physical_time,obs.time_present[...,None].float(),obs.quantity_id[...,None].float(),obs.log_noise),-1)
            bias=self.bias(torch.cat((delta,delta.square().sum(-1,keepdim=True),meta[:,None].expand(-1,M,-1,-1)),-1)).permute(0,3,1,2)
            weights=masked_softmax(q@k.transpose(-1,-2)/math.sqrt(24)+bias,obs.valid[:,None,None,:])
            update=self.output((weights@v).transpose(1,2).reshape(B,M,96))
            update=torch.where(obs.valid.any(1)[:,None,None],update,torch.zeros_like(update));memory=self.norm(memory+update)
        # No attention parameters are evaluated in the attention-disabled ablation.
        memory=self.mix_norm(memory+self.local(self.neighbors@memory)+self.global_mix(memory.mean(1))[:,None])
        return self.ff_norm(memory+self.ff(memory)),weights


class SpatialPath(nn.Module):
    def __init__(self,geometry):
        super().__init__();self.geometry_provider=geometry;layout=geometry.anchor_layout().validate()
        require(layout.positions.shape==(80,2) and torch.bincount(layout.levels).tolist()==[16,64],'B1 V1 requires16/64 anchors')
        self.register_buffer('positions',layout.positions);self.register_buffer('levels',layout.levels)
        self.geometry=nn.Sequential(nn.Linear(9,96),nn.SiLU(),nn.Linear(96,96)) # xy,4side,min,2level
        self.context=nn.Linear(192,96);self.physical=nn.Sequential(nn.Linear(8,96),nn.SiLU(),nn.Linear(96,96))
        self.previous=nn.Linear(96,96,bias=False);self.norm=nn.LayerNorm(96)
        self.residual_tokens=nn.Sequential(nn.Linear(3,96),nn.SiLU(),nn.Linear(96,96))
        self.blocks=nn.ModuleList([SpatialBlock(geometry.neighbor_weights()),SpatialBlock(geometry.neighbor_weights())])
        self.exchange_q,self.exchange_k,self.exchange_v=(nn.Linear(96,96) for _ in range(3))
        self.exchange_out=nn.Linear(96,96);self.exchange_norm=nn.LayerNorm(96)

    def forward(self,tokens,obs,eq,system,xi,gradient,predicted,residual,previous,ablation,*,return_diagnostics):
        device=tokens.device;p=self.positions;g=self.geometry_provider
        features=torch.cat((p,g.boundary_features(p),g.boundary_distance(p),torch.nn.functional.one_hot(self.levels,2).float()),-1)
        memory=self.geometry(features)[None]+self.context(torch.cat((eq,obs.summary),-1))[:,None]
        fields=projected=None
        if ablation.physical_projections:
            fields,projected,physical=anchor_projections(system,xi,gradient,p)
            memory=memory+self.physical(physical.to(dtype=torch.float32).to(device))
        if previous is not None:memory=memory+self.previous(previous)
        memory=self.norm(memory)
        attention=[]
        if ablation.spatial_attention:
            # y/sigma are already encoded; add candidate prediction, normalized
            # residual and noise-standardized residual without hidden clean values.
            # sigma denominator is physical, strictly positive for valid entries.
            extra=torch.stack((predicted/system.sigma,residual/system.sigma,system.sigma.log()),-1)
            encoded=replace(obs,tokens=masked_values(obs.tokens+self.residual_tokens(torch.asinh(extra).to(dtype=torch.float32).to(device)),obs.valid))
        else:encoded=obs
        for block in self.blocks:
            memory,weights=block(memory,encoded,p,attention_enabled=ablation.spatial_attention)
            if return_diagnostics:attention.append(weights)
        B,J,_=tokens.shape;M=memory.shape[1]
        q=self.exchange_q(tokens).reshape(B,J,4,24).transpose(1,2);k=self.exchange_k(memory).reshape(B,M,4,24).transpose(1,2);v=self.exchange_v(memory).reshape(B,M,4,24).transpose(1,2)
        weights=torch.softmax(q@k.transpose(-1,-2)/math.sqrt(24),-1)
        exchanged=self.exchange_out((weights@v).transpose(1,2).reshape(B,J,96))
        output=self.exchange_norm(tokens+exchanged)
        diagnostics={'tokens':memory,'fields':fields,'projected_gradient':projected,
                     'attention':attention,'coarse_summary':memory[:,self.levels==0].mean(1),'fine_summary':memory[:,self.levels==1].mean(1),
                     'exchange_update_norm':exchanged.norm(dim=-1)} if return_diagnostics else None
        return output,memory,diagnostics


@dataclass(frozen=True)
class BContext:
    specs: tuple
    system: object
    graph: object
    observations: object
    owner: object


@dataclass(frozen=True)
class BEstimate:
    xi_hat: torch.Tensor # CPU64 physical units
    xi_active: torch.Tensor
    trajectory: torch.Tensor # [B,K+1,18], CPU64
    predictions: torch.Tensor # [B,K+1,N], CPU64
    residuals: torch.Tensor
    gradients: torch.Tensor
    objectives: torch.Tensor # [B,K+1]
    diagonal: torch.Tensor # [B,K,18], physical preconditioner, CPU64
    steps: torch.Tensor # [B,K], CPU64
    token_history: tuple
    spatial_history: tuple
    context: BContext


@dataclass(frozen=True)
class BPrediction:
    predicted_value: torch.Tensor # query device/dtype for existing tensor contract
    physical_value: torch.Tensor # CPU64 reference value retaining autograd
    valid: torch.Tensor


@dataclass(frozen=True)
class BOutput:
    predicted_value: torch.Tensor
    xi_hat: torch.Tensor
    xi_active: torch.Tensor
    estimate: BEstimate


class ArchitectureB(nn.Module):
    def __init__(self,approach,normalizers,vocabulary,*,iterations=3):
        super().__init__();require(approach in {'B0','B1'},'select B0 or B1 explicitly');require(iterations==3,'V1 B uses three refinement steps')
        self.approach=approach;self.iterations=iterations;self.normalizers=normalizers;self.vocabulary=vocabulary
        self.equation=EquationEncoder(vocabulary);self.observation=ObservationEncoder(normalizers)
        self.initializer=nn.Sequential(nn.Linear(192,128),nn.SiLU(),nn.Linear(128,18))
        self.coefficients=CoefficientMemory()
        self.spatial=SpatialPath(geometry_for()) if approach=='B1' else None
        self._owner=object()

    def prepare_context(self,problem,observations):
        specs=tuple(problem) if isinstance(problem,(tuple,list)) else (problem,)
        for s in specs:geometry_for(s.geometry)
        observations.validate(specs);module_input(self,observations.position,'B observations')
        system=build_system(specs,observations)
        eq=self.equation(pack_graphs(specs,self.vocabulary).to(observations.position.device));obs=self.observation(observations,specs)
        return BContext(specs,system,eq,obs,self._owner)

    def infer_unknowns(self,context,*,ablation=BAblation(),return_diagnostics=False):
        require(context.owner is self._owner,'B context belongs to another model')
        system=context.system;device=context.graph.context.device
        active=system.active.to(device);mean=system.mean.to(dtype=torch.float32).to(device);scale=system.scale.to(dtype=torch.float32).to(device)
        standardized=self.initializer(torch.cat((context.graph.context,context.observations.summary),-1))
        xi=masked_values(system.mean+system.scale*cpu64(standardized),system.active)
        estimates=[xi];predictions=[];residuals=[];gradients=[];objectives=[];diagonal=[];steps=[];tokens_history=[];spatial_history=[]
        previous=torch.zeros_like(xi);memory=None
        for k in range(self.iterations+1):
            predicted,residual,gradient,objective=system.state(xi)
            predictions.append(predicted);residuals.append(residual);gradients.append(gradient);objectives.append(objective)
            if k==self.iterations:break
            tokens=self.coefficients.tokenize(xi.to(dtype=torch.float32).to(device),mean,scale,gradient.to(dtype=torch.float32).to(device),previous.to(dtype=torch.float32).to(device),active,context.graph.context,context.observations.summary,k)
            if self.spatial is not None and ablation.spatial:
                tokens,memory,diagnostics=self.spatial(tokens,context.observations,context.graph.context,system,xi,gradient,predicted,residual,memory,ablation,return_diagnostics=return_diagnostics)
                if return_diagnostics:spatial_history.append(diagnostics)
            # No spatial module call, memory state or stale features when disabled.
            tokens=masked_values(tokens,active)
            d,fraction,tokens=self.coefficients.refine(tokens,active,context.graph.context)
            D=masked_values(system.scale.square()*cpu64(d),system.active)
            eta=cpu64(fraction)/(1.75*system.curvature)
            update=-eta[:,None]*D*gradient
            xi=masked_values(xi+update,system.active);previous=update
            diagonal.append(D);steps.append(eta);estimates.append(xi)
            if return_diagnostics:tokens_history.append(tokens)
        require(bool(torch.isfinite(xi).all()),'nonfinite B refinement')
        return BEstimate(xi,system.active,torch.stack(estimates,1),torch.stack(predictions,1),torch.stack(residuals,1),torch.stack(gradients,1),torch.stack(objectives,1),torch.stack(diagonal,1),torch.stack(steps,1),tuple(tokens_history),tuple(spatial_history),context)

    def predict_queries(self,estimate,queries):
        require(estimate.context.owner is self._owner,'B estimate belongs to another model')
        physical=physical_solution(estimate.context.system,estimate.xi_hat,queries)
        value=physical.to(dtype=queries.position.dtype).to(device=queries.position.device)
        validate_prediction(value,queries)
        return BPrediction(value,physical,queries.valid)

    def forward(self,problem,observations,queries,*,ablation=BAblation(),return_diagnostics=False):
        estimate=self.infer_unknowns(self.prepare_context(problem,observations),ablation=ablation,return_diagnostics=return_diagnostics)
        return BOutput(self.predict_queries(estimate,queries).predicted_value,estimate.xi_hat,estimate.xi_active,estimate)

    def parameter_inventory(self):return {name:sum(p.numel() for p in module.parameters()) for name,module in self.named_children()}


def build_architecture_b(approach,normalizers,vocabulary=None,*,root_seed=20261108,iterations=3):
    if vocabulary is None:
        from .benchmarks import benchmark
        vocabulary=EquationVocabulary.from_specs(tuple(benchmark(p) for p in ('poisson','heat','wave')))
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(RandomStreams(root_seed).torch('model_initialization',approach,'inverse-v1').get_state())
        return ArchitectureB(approach,normalizers,vocabulary,iterations=iterations)
