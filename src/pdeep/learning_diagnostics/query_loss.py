"""Exact quantity/case reductions with checkpointed independent query decoding.

One trainable equation/geometry/observation context is encoded per update.
Checkpointing replays decoder chunks in backward; it never detaches the context.
Physics integrals use the original whole-domain weak_loss, never per-chunk squares.
"""
from contextlib import nullcontext
import time
import torch
from torch.utils.checkpoint import checkpoint
from ..core import require
from ..geometric.data import Tokens
from ..geometric.physics import weak_loss
from ..train_types import LossReport
from .runtime import InputCache,using

class QueryAdapter:
    def __init__(self,architecture,weak,*,chunk=512,checkpointing=True,field_only=False,amp='off',cache=None):
        self.architecture=architecture;self.weak=weak;self.chunk=chunk;self.checkpointing=checkpointing;self.field_only=field_only;self.amp=amp;self.cache=cache or InputCache();self.last={}
    def autocast(self,device):return torch.autocast('cuda',dtype=torch.bfloat16) if self.amp=='bf16' else nullcontext()
    def loss(self,model,batch,*,detailed=False):
        self.cache.clear_queries()
        start=time.perf_counter();device=batch.queries.position.device
        with using(self.cache),self.autocast(device):
            contexts=model.prepare_context(batch.specs,batch.observations)
            raw_field=[];raw_aux=[]
            for row,(context,case) in enumerate(zip(contexts,batch.cases)):
                q=batch.queries;valid=q.valid[row];N=int(valid.sum());require(N>0,'empty supervised queries')
                quantities=(0,1) if case.pde=='wave' else (0,)
                sizes={k:int((valid&(q.quantity_id[row]==k)).sum()) for k in quantities}
                active=[k for k in quantities if sizes[k]]
                fields=case.fields
                # Hidden targets are used ONLY in supervised loss, never model inputs.
                cpu_x=q.position[row].detach().cpu().double().numpy()
                aux={name:torch.as_tensor(field(cpu_x),device=device,dtype=torch.float32)[:,None] for name,field in fields.items()}
                ff=[];aa=[]
                for a in range(0,valid.numel(),self.chunk):
                    end=min(a+self.chunk,valid.numel());data={k:None if v is None else v[row:row+1,a:end] for k,v in vars(q).items()};query=Tokens(**data)
                    if not bool(query.valid.any()):continue
                    truth={k:v[a:end] for k,v in aux.items()}
                    # Bind each chunk/context by factory: backward must not capture
                    # the final loop iteration's query/targets through a late closure.
                    def build(query,truth,context,case,N,sizes,active):
                        def evaluate():
                            take=query.valid[0];ids=query.quantity_id[0];terms=[]
                            for k in active:
                                choose=take&(ids==k)
                                if not bool(choose.any()):continue
                                xx=query.position[0,choose];tt=query.physical_time[0,choose] if case.pde!='poisson' else None
                                # FP32 loss accumulation also avoids indexed BF16->FP32
                                # assignment in the unchanged inference predict helper.
                                value=model.evaluate(context,xx,tt,'v' if k else 'u').float()
                                terms.append(((value-query.value[0,choose])/batch.normalizers.scale(case.pde,k)).square().sum()/sizes[k]/len(active))
                            x=query.position[0,take];t=x.new_zeros((len(x),1)) if case.pde!='poisson' else None
                            pieces=[((model.evaluate(context,x,t,name)-v[take])/batch.normalizers.scale(case.pde,name)).float().square().sum()/N/len(truth) for name,v in truth.items()]
                            return torch.stack((sum(terms),sum(pieces)))
                        return evaluate
                    fn=build(query,truth,context,case,N,sizes,active)
                    result=checkpoint(fn,use_reentrant=False,context_fn=lambda:(using(self.cache),using(self.cache))) if self.checkpointing and torch.is_grad_enabled() else fn()
                    ff.append(result[0]);aa.append(result[1])
                raw_field.append(sum(ff));raw_aux.append(sum(aa))
            raw={'field':torch.stack(raw_field).mean(),'auxiliary':torch.stack(raw_aux).mean()}
            weighted={'field':raw['field'],'auxiliary':self.weak.auxiliary_weight*raw['auxiliary']}
        if self.architecture=='C_GEO' and model.config.physics and not self.field_only:
            # Keep derivative-intensive weak integrals in FP32 even for AMP decoder.
            with using(self.cache):
                physical_context=model.prepare_context(batch.specs,batch.observations) if self.amp!='off' else contexts
                physical=weak_loss(model,physical_context,batch.normalizers,self.weak);physical.pop('diagnostics');raw.update(physical)
                weighted.update(weak=self.weak.weak_weight*raw['weak'],initial=self.weak.initial_weight*raw['initial'],boundary=self.weak.boundary_weight*raw['boundary'])
        self.last={'forward_CPU_enqueue_seconds':time.perf_counter()-start,'context_count':len(contexts),'checkpointing':self.checkpointing,'query_chunk':self.chunk,'amp':self.amp,'static_cache_hits':self.cache.hits,'static_cache_misses':self.cache.misses}
        return LossReport(sum(weighted.values()),raw,weighted,{k:'unchanged equal-case/quantity loss; sums divided by GLOBAL valid counts before chunk addition' for k in raw},{'cases':len(batch.cases),'valid_queries':int(batch.queries.valid.sum()),'active_coefficients':0,'valid_observations':int(batch.observations.valid.sum())},self.last.copy())
