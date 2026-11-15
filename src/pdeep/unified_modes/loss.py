"""Unchanged established objective reductions, plus disclosed view diagnostics."""
import torch
from ..unified.loss import UnifiedAdapter

class Adapter(UnifiedAdapter):
    def loss(self,model,batch,*,detailed=False):
        result=super().loss(model,batch,detailed=detailed)
        for row,(c,r) in enumerate(zip(batch.cases,self.last['per_case'])):
            r.update(mode=c.mode,known_fraction=c.condition.known_fraction,observations=int(batch.observations.valid[row].sum()))
        if detailed or self.measure_gradients:
            parameters=tuple(model.parameters());g=torch.autograd.grad(result.weighted['field'],parameters,allow_unused=True,retain_graph=True)
            physics=sum((v for k,v in result.weighted.items() if k in ('kinematic','weak','initial','boundary')),result.total.new_zeros(()))
            if physics.requires_grad:
                h=torch.autograd.grad(physics,parameters,allow_unused=True,retain_graph=True)
                dot=sum(float((a.detach()*b.detach()).sum()) for a,b in zip(g,h) if a is not None and b is not None)
                norm=lambda z:sum(float(a.detach().square().sum()) for a in z if a is not None)**.5
                ng,nh=norm(g),norm(h);self.last['field_physics_gradient']={'field_norm':ng,'weighted_physics_norm':nh,'cosine':dot/(ng*nh) if ng*nh else None,'scope':'bounded diagnostic every500 successful updates; no adaptive loss weighting'}
        return result
