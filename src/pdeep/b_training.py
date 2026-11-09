"""Matched B0/B1 coefficient-only objective and bounded AdamW protocol."""
from dataclasses import dataclass
import math,resource,sys,time
import torch
from .core import require
from .batches import masked_values,UnknownCoefficients
from .b_physics import cpu64
from .a_training import SupervisedBatch,gradient_report
from .representation_checks import sync


@dataclass(frozen=True)
class BTrainingConfig:
    approach: str='B0'
    updates: int=3
    iterations: int=3
    microbatch: int=1
    effective_batch: int=3
    learning_rate: float=3e-4
    weight_decay: float=1e-4
    gradient_clip: float=1.
    root_seed: int=20261108
    precision: str='float32'
    preset: str='smoke'
    tiny_overfit: bool=False

    def __post_init__(self):
        require(self.approach in {'B0','B1'} and self.iterations==3,'select B0/B1 and three inverse steps')
        require(self.precision=='float32','B neural precision is float32; physical bridge CPU64')
        require(self.preset in {'smoke','tiny_overfit'} and self.tiny_overfit==(self.preset=='tiny_overfit'),'unsupported B training preset')
        require(type(self.updates) is int and (3<=self.updates<=5 if self.preset=='smoke' else 1<=self.updates<=100),'bounded B budget:3–5 smoke or <=100 explicit overfit')
        require(type(self.microbatch) is int and type(self.effective_batch) is int and min(self.microbatch,self.effective_batch)>0,'positive B batch sizes required')
        require(all(math.isfinite(v) for v in (self.learning_rate,self.weight_decay,self.gradient_clip)) and self.learning_rate>0 and self.weight_decay>=0 and self.gradient_clip>0,'invalid B optimizer settings')


def coefficient_loss(estimate,truth,*,initial=False):
    truth.validate(estimate.context.specs)
    active=estimate.xi_active
    target=cpu64(truth.masked());xi=estimate.trajectory[:,0] if initial else estimate.xi_hat
    square=masked_values(((xi-target)/estimate.context.system.scale).square(),active)
    return (square.sum(1)/active.sum(1).clamp_min(1)).mean()


def make_training_batch(dataset,normalizers,step,*,count=3,device='cpu',fixed=False):
    """Variant-independent sampler: identical cases/O/query identities for B0/B1."""
    from .data import observations,training_queries,collate_inputs,case_coefficients
    from .reference import canonical_spec,evaluate_batch
    from .representations import float32_batch
    step=0 if fixed else step;rng=dataset.streams.numpy('optimization','B','matched-cases',step)
    ids=[]
    while len(ids)<count:ids.extend(rng.permutation(3).tolist())
    names=('poisson','heat','wave');cases=[dataset.case('train',names[j],int(rng.integers(12000))) for j in ids[:count]]
    specs=tuple(canonical_spec(c.key.pde) for c in cases)
    obs=collate_inputs([observations(dataset,c,normalizers,set_id=f'B-matched-{step}').inputs for c in cases])
    q=collate_inputs([training_queries(dataset,c,set_id=f'B-matched-{step}').inputs for c in cases]);xi=case_coefficients(cases)
    target=evaluate_batch(specs,xi,q)
    return SupervisedBatch(specs,float32_batch(obs,device=device),float32_batch(q,device=device),target.to(dtype=torch.float32).to(device),
                           UnknownCoefficients(xi.xi.to(dtype=torch.float32).to(device),xi.active.to(device)),tuple(c.identity for c in cases))


def train_update(model,optimizer,batch,config):
    require(model.approach==config.approach,'B training/model variant mismatch')
    model.train();optimizer.zero_grad(set_to_none=True);device=next(model.parameters()).device;N=len(batch.specs)
    totals=dict(loss=0.,initial_coefficient_MSE=0.,forward_seconds=0.,backward_seconds=0.,physical_construction_seconds=0.)
    for lo in range(0,N,config.microbatch):
        hi=min(lo+config.microbatch,N);b=batch.slice(lo,hi)
        sync(device);start=time.perf_counter()
        result=model.infer_unknowns(model.prepare_context(b.specs,b.observations))
        loss=coefficient_loss(result,b.unknowns)
        require(bool(torch.isfinite(loss)),'nonfinite B supervised loss')
        sync(device);totals['forward_seconds']+=time.perf_counter()-start
        totals['physical_construction_seconds']+=result.context.system.construction_seconds
        start=time.perf_counter();(loss*((hi-lo)/N)).backward();sync(device);totals['backward_seconds']+=time.perf_counter()-start
        totals['loss']+=float(loss.detach())*((hi-lo)/N);totals['initial_coefficient_MSE']+=float(coefficient_loss(result,b.unknowns,initial=True).detach())*((hi-lo)/N)
    gradients=gradient_report(model);require(all(not g['nonfinite'] for g in gradients.values()),'nonfinite B gradients')
    before=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip,error_if_nonfinite=True))
    after=sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None)**.5
    sync(device);start=time.perf_counter();optimizer.step();sync(device)
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'nonfinite B parameters')
    return {**totals,'update_seconds':time.perf_counter()-start,'gradient_before_clip':before,'gradient_after_clip':after,
            'gradients':gradients,'learning_rate':optimizer.param_groups[0]['lr'],'microbatch_sizes':[min(config.microbatch,N-lo) for lo in range(0,N,config.microbatch)]}


def run_bounded(model,dataset,normalizers,config,*,optimizer=None,start_step=0):
    require(model.approach==config.approach,'B run variant mismatch')
    device=next(model.parameters()).device;dev=device.type if device.type in {'cpu','mps'} else str(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay) if optimizer is None else optimizer
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    logs=[];counters=dict(attempted=0,successful=0,skipped=0,failed=0);began=time.perf_counter()
    for step in range(start_step,start_step+config.updates):
        batch=make_training_batch(dataset,normalizers,step,count=config.effective_batch,device=dev,fixed=config.tiny_overfit)
        counters['attempted']+=1
        try:log=train_update(model,optimizer,batch,config)
        except Exception:counters['failed']+=1;raise
        counters['successful']+=1;log.update(update=step+1,case_ids=list(batch.case_ids),family_counts={p:sum(s.name==p for s in batch.specs) for p in ('poisson','heat','wave')})
        log['rolling_loss']=(sum(v['loss'] for v in logs[-4:])+log['loss'])/min(5,len(logs)+1);logs.append(log)
    model.eval()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    memory={'process_peak_RSS_bytes':rss,'RSS_scope':'whole process; includes libraries/calibration and other models; not variant-isolated peak',
            'parameters_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'gradients_bytes':sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None),
            'Adam_tensor_bytes':sum(t.numel()*t.element_size() for state in optimizer.state.values() for t in state.values() if isinstance(t,torch.Tensor))}
    if device.type=='cuda':memory['CUDA_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
    if device.type=='mps':memory.update(MPS_current_allocated_bytes=torch.mps.current_allocated_memory(),MPS_peak='not exposed by API')
    return {'approach':model.approach,'status':'smoke-tested' if config.preset=='smoke' else 'tiny-overfit diagnostic','updates':logs,'counters':counters,
            'elapsed_seconds':time.perf_counter()-began,'memory':memory,'neural_device':dev,'neural_dtype':'float32','physics_device':'cpu','physics_dtype':'float64',
            'effective_batch':config.effective_batch,'accumulation_steps':math.ceil(config.effective_batch/config.microbatch),'next_step':start_step+config.updates,
            'interpretation':'execution check, not reconstruction quality; no posterior-objective term in loss'},optimizer
