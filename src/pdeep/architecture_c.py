"""C: independently conditioned neural field, without spatial memory or solver.

Only observed data and validated public physics enter the context. The shared
u/v trunk is pointwise in query coordinates; physical constraints live elsewhere.
"""
from dataclasses import dataclass,replace,fields
import math
import torch
from torch import nn
from .batches import expected_active_mask,masked_values,validate_prediction
from .core import require,Status
from .diagnostics import RandomStreams
from .encoding import (EquationEncoder,EquationVocabulary,ObservationEncoder,CoordinateFeatures,
                       pack_graphs,masked_pool,module_input)
from .geometry import geometry_for,SquareDirichletTreatment
from .memory import masked_softmax

FAMILIES=("poisson","heat","wave")


def finite(t,label):
    require(bool(torch.isfinite(t).all()),f"nonfinite C {label}")
    return t


class ObservationInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv=nn.Linear(96,288)
        self.output=nn.Linear(96,96)
        self.norm=nn.LayerNorm(96)
        self.ff=nn.Sequential(nn.Linear(96,192),nn.SiLU(),nn.Linear(192,96))
        self.ff_norm=nn.LayerNorm(96)

    def forward(self,tokens,valid):
        b,n,_=tokens.shape
        q,k,v=(x.reshape(b,n,4,24).transpose(1,2) for x in self.qkv(tokens).chunk(3,-1))
        weights=masked_softmax(q@k.transpose(-1,-2)/math.sqrt(24),valid[:,None,None,:]&valid[:,None,:,None])
        x=self.norm(tokens+self.output((weights@v).transpose(1,2).reshape(b,n,96)))
        return masked_values(self.ff_norm(x+self.ff(x)),valid),weights


class GlobalConditioning(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph=nn.Linear(96,96);self.observation=nn.Linear(96,96)
        self.known=nn.Linear(7,96);self.geometry=nn.Linear(10,96)
        self.norm=nn.LayerNorm(96)

    def forward(self,graph,obs,known,geometry):
        return self.norm(self.graph(graph)+self.observation(obs)+self.known(known)+self.geometry(geometry))


class FiLMLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(128,128);self.norm=nn.LayerNorm(128)
        self.gamma=nn.Linear(96,128);self.beta=nn.Linear(96,128)

    def forward(self,h,conditioning):
        gamma=.1*self.gamma(conditioning).tanh()[:,None]
        beta=self.beta(conditioning)[:,None]
        return h+torch.nn.functional.silu(self.norm(self.linear(h))*(1+gamma)+beta)


class FiLMField(nn.Module):
    def __init__(self,geometry):
        super().__init__()
        self.coordinate=CoordinateFeatures(geometry,frequencies=(1.,2.,4.,8.))
        self.project=nn.Linear(self.coordinate.size,128)
        self.layers=nn.ModuleList([FiLMLayer() for _ in range(4)])
        self.u=nn.Linear(128,1);self.v=nn.Linear(128,1)

    def forward(self,query,conditioning):
        q=query.masked()
        h=self.project(self.coordinate(q.position,q.physical_time,q.time_present,q.valid))
        for layer in self.layers:h=layer(h,conditioning)
        return self.u(h),self.v(h)


@dataclass(frozen=True)
class CContext:
    specs: tuple
    graph: object
    observations: object
    interaction_tokens: torch.Tensor
    observation_context: torch.Tensor
    conditioning: torch.Tensor
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    xi_active: torch.Tensor
    quantity_scales: torch.Tensor
    attention: torch.Tensor | None
    owner: object

    def select(self,row):
        def sliced(obj):return replace(obj,**{f.name:getattr(obj,f.name)[row:row+1] for f in fields(obj)})
        return replace(self,specs=(self.specs[row],),graph=sliced(self.graph),observations=sliced(self.observations),
                       interaction_tokens=self.interaction_tokens[row:row+1],observation_context=self.observation_context[row:row+1],
                       conditioning=self.conditioning[row:row+1],prior_mean=self.prior_mean[row:row+1],prior_std=self.prior_std[row:row+1],
                       xi_active=self.xi_active[row:row+1],quantity_scales=self.quantity_scales[row:row+1],
                       attention=None if self.attention is None else self.attention[row:row+1])


@dataclass(frozen=True)
class CState:
    u: torch.Tensor
    v: torch.Tensor  # zero outside wave; only u is meaningful for P1/P2
    valid: torch.Tensor


@dataclass(frozen=True)
class CPrediction:
    predicted_value: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class CUnknownEstimate:
    xi_hat: torch.Tensor
    xi_active: torch.Tensor


@dataclass(frozen=True)
class COutput:
    predicted_value: torch.Tensor
    xi_hat: torch.Tensor
    xi_active: torch.Tensor
    context: CContext


class ArchitectureC(nn.Module):
    def __init__(self,normalizers,vocabulary):
        super().__init__()
        self.normalizers=normalizers;self.vocabulary=vocabulary
        self.geometry_provider=geometry_for();self.boundary=SquareDirichletTreatment(self.geometry_provider)
        self.equation=EquationEncoder(vocabulary)
        self.observation=ObservationEncoder(normalizers) # Stage 3 six-frequency tokenizer unchanged
        self.interaction=nn.ModuleList([ObservationInteraction(),ObservationInteraction()])
        self.conditioning=GlobalConditioning()
        self.field=FiLMField(self.geometry_provider)
        self.coefficient_head=nn.Sequential(nn.Linear(96,128),nn.SiLU(),nn.Linear(128,18))
        self._owner=object()

    def prepare_context(self,problem,observations,*,return_attention=False):
        specs=tuple(problem) if isinstance(problem,(tuple,list)) else (problem,)
        for s in specs:
            s.validate();geometry_for(s.geometry)
            require(s.name in FAMILIES and (s.physical_time is not None)==(s.name!="poisson"),"C supports canonical T1 family/time semantics only")
            require(all(b.kind=="dirichlet" and b.status==Status.KNOWN and b.value==0 for b in s.boundaries),"C requires known homogeneous Dirichlet boundary")
            require(tuple((o.quantity_id,o.target) for o in s.observations)==(((0,"u"),(1,"v")) if s.name=="wave" else ((0,"u"),)),"C unsupported quantity identities")
            require(tuple(b.id for b in s.parameter_blocks)==(("a","b") if s.name=="wave" else ("a",)),"C excludes scalar T2/unsupported slots")
        observations.validate(specs);module_input(self,observations.position,"C observations")
        active=expected_active_mask(specs,observations.position.device)
        require(active.sum(1).tolist()==[18 if s.name=="wave" else 9 for s in specs],"C requires all canonical modal slots unknown")
        graph=self.equation(pack_graphs(specs,self.vocabulary).to(observations.position.device))
        obs=self.observation(observations,specs);tokens=obs.tokens;attention=[]
        for layer in self.interaction:
            tokens,weights=layer(tokens,obs.valid)
            if return_attention:attention.append(weights)
        summary=masked_pool(tokens,obs.valid)
        known=[];geometry=[];means=[];stds=[]
        for s in specs:
            bindings={b.id:b for b in s.bindings};alpha=bindings.get("alpha");speed=bindings.get("c")
            known.append([alpha.value if alpha else 0.,speed.value if speed else 0.,float(alpha is not None),float(speed is not None),1.,float(s.name=="wave"),float(s.physical_time is not None)])
            g=geometry_for(s.geometry)
            center=graph.context.new_tensor([[[.5,.5]]])
            geometry.append([v for bounds in s.geometry.bounds for v in bounds]+g.boundary_features(center).flatten().tolist()+[2.,1.])
            m=[v for b in s.parameter_blocks for v in b.prior.mean];sd=[v for b in s.parameter_blocks for v in b.prior.scale]
            means.append(m+[0.]*(18-len(m)));stds.append(sd+[1.]*(18-len(sd)))
        c=self.conditioning(graph.context,summary,graph.context.new_tensor(known),graph.context.new_tensor(geometry))
        family=torch.tensor([FAMILIES.index(s.name) for s in specs],device=c.device)
        for label,value in (("graph",graph.nodes),("observation",obs.tokens),("interaction",tokens),("conditioning",c)):finite(value,label)
        return CContext(specs,graph,obs,tokens,summary,c,c.new_tensor(means),c.new_tensor(stds),active,
                        self.observation.quantity_scales[family],torch.stack(attention,1) if return_attention else None,self._owner)

    def _check(self,context):
        require(isinstance(context,CContext) and context.owner is self._owner,"C context belongs to another model")
        module_input(self,context.conditioning,"C context")

    def evaluate_state(self,context,queries):
        self._check(context);queries.validate(context.specs);module_input(self,queries.position,"C queries")
        q=queries.masked();raw_u,raw_v=self.field(q,context.conditioning)
        factor=self.boundary.factor(q.position)
        u=masked_values(context.quantity_scales[:,0,None,None]*factor*raw_u,q.valid)
        wave=torch.tensor([s.name=="wave" for s in context.specs],device=u.device)
        v=masked_values(context.quantity_scales[:,1,None,None]*factor*raw_v,q.valid&wave[:,None])
        return CState(finite(u,"u field"),finite(v,"v field"),q.valid)

    def predict_queries(self,context,queries):
        state=self.evaluate_state(context,queries)
        value=torch.where(queries.masked().quantity_id[...,None]==0,state.u,state.v)
        validate_prediction(value,queries)
        return CPrediction(value,queries.valid)

    def infer_unknowns(self,context):
        self._check(context)
        xi=masked_values(context.prior_mean+context.prior_std*self.coefficient_head(context.conditioning),context.xi_active)
        return CUnknownEstimate(finite(xi,"xi"),context.xi_active)

    def forward(self,problem,observations,queries,*,return_attention=False):
        c=self.prepare_context(problem,observations,return_attention=return_attention)
        xi=self.infer_unknowns(c)
        return COutput(self.predict_queries(c,queries).predicted_value,xi.xi_hat,xi.xi_active,c)

    def parameter_inventory(self):
        return {name:sum(p.numel() for p in m.parameters()) for name,m in self.named_children()}


def build_architecture_c(normalizers,vocabulary=None,*,root_seed=20261108):
    if vocabulary is None:
        from .benchmarks import benchmark
        vocabulary=EquationVocabulary.from_specs(tuple(benchmark(p) for p in FAMILIES))
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(RandomStreams(root_seed).torch("model_initialization","C","field-weak-v1").get_state())
        return ArchitectureC(normalizers,vocabulary)
