"""A: direct learned inverse field. No solver, hidden truth or query interaction.

Prepared contexts retain autograd during training; reuse them only until the next
optimizer update. A fresh model owns fresh graph/observation/memory parameters.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn

from .batches import QueryBatch, expected_active_mask, masked_values, validate_prediction
from .core import require, Status
from .diagnostics import RandomStreams
from .encoding import (WIDTH, CoordinateFeatures, EquationEncoder, EquationVocabulary,
                       ObservationEncoder, pack_graphs, module_input)
from .geometry import geometry_for, SquareDirichletTreatment
from .memory import MemoryState, MemoryRetriever, masked_softmax

FAMILIES=("poisson","heat","wave")
FREQUENCIES=(1.,2.,4.,8.)


def finite(tensor,label):
    require(bool(torch.isfinite(tensor).all()),f"nonfinite A {label}")
    return tensor


class ACrossAttention(nn.Module):
    """A-owned two-stage residual attention; no changes to Stage 3 defaults."""
    def __init__(self):
        super().__init__()
        self.q,self.k,self.v=(nn.Linear(96,96) for _ in range(3))
        self.output=nn.Linear(96,96)
        self.quantity=nn.Embedding(2,8)
        self.bias=nn.Sequential(nn.Linear(6+8,32),nn.SiLU(),nn.Linear(32,4))
        self.norm=nn.LayerNorm(96)
        self.ff=nn.Sequential(nn.Linear(96,192),nn.SiLU(),nn.Linear(192,96))
        self.ff_norm=nn.LayerNorm(96)

    def forward(self,memory,obs,anchors):
        b,m,_=memory.shape;n=obs.tokens.shape[1]
        q=self.q(memory).reshape(b,m,4,24).transpose(1,2)
        k=self.k(obs.tokens).reshape(b,n,4,24).transpose(1,2)
        v=self.v(obs.tokens).reshape(b,n,4,24).transpose(1,2)
        relative=anchors[None,:,None]-obs.position[:,None]
        distance=torch.linalg.vector_norm(relative,dim=-1,keepdim=True)
        meta=torch.cat((obs.physical_time,obs.time_present[...,None].to(memory.dtype),obs.log_noise,
                        self.quantity(obs.quantity_id)),dim=-1)
        bias_input=torch.cat((relative,distance,meta[:,None].expand(-1,m,-1,-1)),dim=-1)
        logits=q@k.transpose(-1,-2)/math.sqrt(24)+self.bias(bias_input).permute(0,3,1,2)
        weights=masked_softmax(logits,obs.valid[:,None,None,:])
        update=self.output((weights@v).transpose(1,2).reshape(b,m,96))
        update=torch.where(obs.valid.any(1)[:,None,None],update,torch.zeros_like(update))
        memory=self.norm(memory+update)
        return finite(self.ff_norm(memory+self.ff(memory)),"cross-attention"),weights


class LocalGlobalMixer(nn.Module):
    def __init__(self,geometry):
        super().__init__()
        neighbors=geometry.neighbor_weights()
        require(neighbors.shape==(80,80) and bool(torch.isfinite(neighbors).all()) and bool((neighbors>=0).all()),"invalid A geometry neighbors")
        require(torch.allclose(neighbors.sum(-1),torch.ones(80)),"neighbor weights must sum to one")
        self.register_buffer("neighbors",neighbors)
        self.local=nn.Linear(96,96)
        self.global_projection=nn.Linear(96,96)
        self.norm=nn.LayerNorm(96)
        self.ff=nn.Sequential(nn.Linear(96,192),nn.SiLU(),nn.Linear(192,96))
        self.ff_norm=nn.LayerNorm(96)

    def forward(self,memory):
        mixed=self.norm(memory+self.local(self.neighbors@memory)+self.global_projection(memory.mean(1))[:,None])
        return finite(self.ff_norm(mixed+self.ff(mixed)),"local/global memory")


class ASpatialMemory(nn.Module):
    def __init__(self,geometry):
        super().__init__()
        self.geometry_provider=geometry
        layout=geometry.anchor_layout().validate()
        require(layout.positions.shape==(80,2) and torch.bincount(layout.levels).tolist()==[16,64],"A V1 requires 16/64 anchors")
        self.register_buffer("positions",layout.positions)
        self.register_buffer("levels",layout.levels)
        # raw position, four boundary-region distances, minimum distance, Fourier xy
        coord=CoordinateFeatures(geometry,frequencies=FREQUENCIES)
        self.fourier=coord.fourier
        self.geometry=nn.Sequential(nn.Linear(2+4+1+16,96),nn.SiLU(),nn.Linear(96,96))
        self.equation=nn.Linear(96,96)
        # alpha, c, their presence, known homogeneous boundary value, field unknown mask
        self.known=nn.Linear(6,96)
        self.scale=nn.Embedding(2,96)
        self.norm=nn.LayerNorm(96)
        self.layers=nn.ModuleList([ACrossAttention(),ACrossAttention()])
        self.mixer=LocalGlobalMixer(geometry)
        self.global_projection=nn.Linear(3*96,96)

    def forward(self,eq,obs,specs,*,return_attention=False):
        p=self.positions;g=self.geometry_provider
        geom=torch.cat((p,g.boundary_features(p),g.boundary_distance(p),self.fourier(p)),dim=-1)
        known=[]
        for spec in specs:
            bindings={b.id:b for b in spec.bindings}
            alpha=bindings.get("alpha");c=bindings.get("c")
            known.append([alpha.value if alpha else 0.,c.value if c else 0.,float(alpha is not None),float(c is not None),0.,1.])
        physical=eq.new_tensor(known)
        memory=self.norm(self.geometry(geom)[None]+self.equation(eq)[:,None]+self.known(physical)[:,None]+self.scale(self.levels)[None])
        finite(memory,"initial memory")
        attention=[]
        for layer in self.layers:
            memory,weights=layer(memory,obs,p)
            if return_attention:attention.append(weights)
        memory=self.mixer(memory)
        global_context=finite(self.global_projection(torch.cat((eq,obs.summary,memory.mean(1)),dim=-1)),"global context")
        return MemoryState(memory,global_context,p,self.levels,torch.stack(attention,1) if return_attention else None)


class CoordinateDecoder(nn.Module):
    def __init__(self,geometry):
        super().__init__()
        self.coordinate=CoordinateFeatures(geometry,frequencies=FREQUENCIES)
        self.quantity=nn.Embedding(2,16)
        # Coordinates + q + exact coarse/fine/global + fused retrieval feature.
        self.input=nn.Linear(self.coordinate.size+16+4*96,128)
        self.layers=nn.ModuleList([nn.Linear(128,128) for _ in range(3)])
        self.activation=nn.SiLU()
        self.u,self.v=nn.Linear(128,1),nn.Linear(128,1)

    def forward(self,query,retrieval,global_context):
        q=query.masked();n=q.position.shape[1]
        x=torch.cat((self.coordinate(q.position,q.physical_time,q.time_present,q.valid),self.quantity(q.quantity_id),
                     retrieval.coarse_context,retrieval.fine_context,global_context[:,None].expand(-1,n,-1),retrieval.features),dim=-1)
        h=self.activation(self.input(x))
        for layer in self.layers:h=h+self.activation(layer(h))
        return torch.where(q.quantity_id[...,None]==0,self.u(h),self.v(h))


@dataclass(frozen=True)
class PreparedContext:
    specs: tuple
    graph: object
    observations: object
    memory: MemoryState
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    xi_active: torch.Tensor
    quantity_scales: torch.Tensor  # [B,2]
    owner: object  # identity guard; not serialized and contains no truth


@dataclass(frozen=True)
class PredictionBatch:
    predicted_value: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class UnknownEstimate:
    xi_hat: torch.Tensor  # inactive slots exactly zero
    xi_active: torch.Tensor


@dataclass(frozen=True)
class ModelOutput:
    predicted_value: torch.Tensor
    xi_hat: torch.Tensor
    xi_active: torch.Tensor
    context: PreparedContext


class ArchitectureA(nn.Module):
    """Deterministic T1 learned inverse model; context accepts observations only.

    Trainable scalar heads share a pointwise trunk. No dropout, batch statistics,
    query attention, coefficient-to-solver path, PDE loss or posterior semantics.
    """
    def __init__(self,normalizers,vocabulary,*,geometry=None):
        super().__init__()
        self.vocabulary=vocabulary
        self.normalizers=normalizers
        self.geometry_provider=geometry_for() if geometry is None else geometry
        self.boundary=SquareDirichletTreatment(self.geometry_provider)
        self.equation=EquationEncoder(vocabulary)
        self.observation=ObservationEncoder(normalizers,coordinate=CoordinateFeatures(self.geometry_provider,frequencies=FREQUENCIES))
        self.memory=ASpatialMemory(self.geometry_provider)
        self.retrieval=MemoryRetriever()
        self.decoder=CoordinateDecoder(self.geometry_provider)
        self.coefficient_head=nn.Sequential(nn.Linear(192,128),nn.SiLU(),nn.Linear(128,18))
        self._context_owner=object()

    def prepare_context(self,problem,observations,*,return_attention=False):
        specs=tuple(problem) if isinstance(problem,(tuple,list)) else (problem,)
        for spec in specs:
            spec.validate();geometry_for(spec.geometry)
            require(spec.name in FAMILIES,"A supports T1 Poisson, heat and wave only")
            require(all(b.kind=="dirichlet" and b.status==Status.KNOWN and b.value==0 for b in spec.boundaries),"A requires known homogeneous Dirichlet constraints")
            require((spec.physical_time is not None)==(spec.name!="poisson"),"A physical time conflicts with family")
            require(tuple((o.quantity_id,o.target) for o in spec.observations)==(((0,"u"),(1,"v")) if spec.name=="wave" else ((0,"u"),)),"A unsupported quantity identity/target mapping")
        observations.validate(specs);module_input(self,observations.position,"A observations")
        active=expected_active_mask(specs,observations.position.device)
        require(active.sum(1).tolist()==[18 if s.name=="wave" else 9 for s in specs],"A supports fully unknown canonical T1 blocks only")
        eq=self.equation(pack_graphs(specs,self.vocabulary).to(observations.position.device))
        obs=self.observation(observations,specs)
        finite(eq.nodes,"graph nodes");finite(eq.context,"graph context");finite(obs.tokens,"observation tokens")
        memory=self.memory(eq.context,obs,specs,return_attention=return_attention)
        mean=torch.zeros(len(specs),18,device=eq.context.device,dtype=torch.float32)
        std=torch.ones_like(mean)
        for i,spec in enumerate(specs):
            means=[v for block in spec.parameter_blocks for v in block.prior.mean]
            scales=[v for block in spec.parameter_blocks for v in block.prior.scale]
            require(tuple(block.id for block in spec.parameter_blocks)==(("a","b") if spec.name=="wave" else ("a",)),"A requires canonical coefficient slot order")
            mean[i,:len(means)]=mean.new_tensor(means);std[i,:len(scales)]=std.new_tensor(scales)
        family=torch.tensor([FAMILIES.index(s.name) for s in specs],device=eq.context.device)
        quantity_scales=self.observation.quantity_scales[family]
        return PreparedContext(specs,eq,obs,memory,mean,std,active,quantity_scales,self._context_owner)

    def _validate_context(self,context):
        require(isinstance(context,PreparedContext) and context.owner is self._context_owner,"A context belongs to a different model")
        module_input(self,context.memory.tokens,"A prepared context")

    def predict_queries(self,context,query_batch):
        self._validate_context(context)
        q=query_batch.validate(context.specs).masked();module_input(self,q.position,"A queries")
        # Existing retrieval API preserved. Its query slot uses fixed graph context;
        # raw coordinates/quantity are supplied explicitly to the decoder.
        features=context.graph.context[:,None].expand(-1,q.position.shape[1],-1)
        retrieval=self.retrieval(context.memory,q,features,context.specs)
        finite(retrieval.features,"retrieval")
        raw=self.decoder(q,retrieval,context.memory.global_context)
        scales=context.quantity_scales.gather(1,q.quantity_id)[...,None]
        prediction=masked_values(scales*self.boundary.factor(q.position)*raw,q.valid)
        finite(prediction,"prediction");validate_prediction(prediction,q)
        return PredictionBatch(prediction,q.valid)

    def infer_unknowns(self,context):
        self._validate_context(context)
        z=self.coefficient_head(torch.cat((context.memory.global_context,context.memory.tokens.mean(1)),dim=-1))
        xi=masked_values(context.prior_mean+context.prior_std*z,context.xi_active)
        return UnknownEstimate(finite(xi,"coefficient estimate"),context.xi_active)

    def forward(self,problem,observations,queries,*,return_attention=False):
        context=self.prepare_context(problem,observations,return_attention=return_attention)
        prediction=self.predict_queries(context,queries);xi=self.infer_unknowns(context)
        return ModelOutput(prediction.predicted_value,xi.xi_hat,xi.xi_active,context)

    def parameter_inventory(self):
        return {name:sum(p.numel() for p in module.parameters()) for name,module in self.named_children()}


def build_architecture_a(normalizers,vocabulary=None,*,root_seed=20261108,geometry=None):
    if vocabulary is None:
        from .benchmarks import benchmark
        vocabulary=EquationVocabulary.from_specs(tuple(benchmark(p) for p in FAMILIES))
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(RandomStreams(root_seed).torch("model_initialization","A","direct-v1").get_state())
        return ArchitectureA(normalizers,vocabulary,geometry=geometry)
