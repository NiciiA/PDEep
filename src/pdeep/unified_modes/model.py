"""Additional field-information pathway on the EXISTING A/C backbones."""
import numpy as np
import torch
from torch import nn
from ..core import require
from ..geometric.models import GeometryModel,ModelConfig
from ..geometric.representation import Attention,tensor,interpolate
from ..smooth.definition import geometry
from .conditioning import FieldCondition,ROLES_ALL,SCHEMA
from .architecture import model_config,inventory

class FieldEncoder(nn.Module):
    def __init__(self,d):
        super().__init__();self.quantity=nn.Embedding(3,d);self.status=nn.Embedding(3,d);self.numeric=nn.Sequential(nn.Linear(9,d),nn.SiLU(),nn.Linear(d,d));self.pool=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d))
    def forward(self,condition,problem,normalizers,like):
        f=condition;valid=f.valid;L=np.sqrt(problem.mesh.cell_areas.sum());scale=np.array([normalizers.scale(problem.name,ROLES_ALL[int(q)]) if app else 1. for q,app in zip(f.quantity,f.applicable)])
        distance=problem.mesh.features(f.centers)['distance']/L
        features=np.column_stack((f.centers/L,f.radius/L,np.full(len(valid),1/np.sqrt(32))/scale,f.values,f.available,f.applicable,distance,np.ones(len(valid))))
        t=lambda a,dtype=like.dtype:torch.as_tensor(np.array(a,copy=True),device=like.device,dtype=dtype)
        h=self.numeric(t(features))+self.quantity(t(f.quantity,torch.long))+self.status(t(f.status,torch.long));h=h*t(valid)[:,None]
        pooled=self.pool(h[t(valid,torch.bool)].mean(0));return h,pooled

class ConditionedModel(GeometryModel):
    conditioning_schema=SCHEMA
    def __init__(self,config,normalizers):
        super().__init__(config,normalizers);d=config.width
        self.known_field_encoder=FieldEncoder(d);self.known_field_cross=Attention(d,config.heads)
    def prepare_context(self,problems,observations,*,latent=None):
        require(all(isinstance(getattr(p,'field_condition',None),FieldCondition) for p in problems),'explicit field availability required; inverse-only checkpoints/models are different schemas')
        contexts=super().prepare_context(problems,observations,latent=latent)
        for ctx in contexts:
            f=ctx.problem.field_condition;require(f.pde==ctx.problem.name and f.geometry_identity==ctx.problem.mesh.geometry_id,'field geometry/PDE mismatch')
            tokens,pool=self.known_field_encoder(f,ctx.problem,self.normalizers,ctx.equation);ctx.global_context=ctx.global_context+pool
            d2=interpolate(ctx.problem.mesh,tensor(ctx.layout.distance2,ctx.equation),tensor(f.centers,ctx.equation)).T
            valid=torch.as_tensor(np.array(f.valid&f.applicable,copy=True),device=ctx.equation.device)
            ctx.memory,attention=self.known_field_cross(ctx.memory,tokens,bias=-d2/(2*.3**2*ctx.problem.mesh.cell_areas.sum()),valid=valid);ctx.attention.append(attention)
            ctx.latent=self.latent_projection(ctx.global_context) if latent is None else ctx.latent
            # Existing auxiliary heads become shared per-basis coefficient readouts.
            # Exact known coefficients are CLAMPED; only unknown coefficients are inferred.
            ctx.field_coefficients={};ctx.field_condition=f
            for role in ROLES_ALL:
                ids=np.flatnonzero(f.valid&(f.quantity==ROLES_ALL.index(role))&f.applicable)
                if not len(ids):continue
                idx=torch.as_tensor(ids,device=tokens.device);proposal=self.aux_heads[role](tokens[idx]+ctx.global_context)[:,0]
                mask=torch.as_tensor(np.array(f.available[ids],copy=True),device=tokens.device);value=tensor(f.values[ids],tokens)
                ctx.field_coefficients[role]=(ids,torch.where(mask,value,proposal))
        return contexts
    def evaluate(self,context,x,t=None,quantity='u'):
        if quantity not in ROLES_ALL:return super().evaluate(context,x,t,quantity)
        require((t is not None)==(context.problem.physical_time is not None),'stationary physical time absent; dynamics require time')
        require(quantity in context.field_coefficients,'not-applicable physical field')
        ids,a=context.field_coefficients[quantity];f=context.field_condition
        centers=tensor(f.centers[ids],x);radius=tensor(f.radius[ids],x)
        phi=(1-((x[:,None]-centers)**2).sum(-1)/radius.square()).clamp_min(0).pow(4)/np.sqrt(32)
        return (phi@a)[:,None]


def create(architecture,normalizers,capacity='medium',*,seed=42,device='cpu'):
    from ..runtime import resolve_device
    resolve_device(device,'float32')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=ConditionedModel(model_config(architecture,capacity),normalizers)
    inventory(model)
    return model.to(device)
