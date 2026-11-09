"""Exact case/quantity-balanced A objective and bounded AdamW diagnostics.

Targets live here, never in PreparedContext. Research-scale orchestration is not
implemented; smoke and opt-in tiny-overfit use precisely this model/objective.
"""
from dataclasses import dataclass, fields
import math
import resource
import time
import torch

from .batches import masked_values
from .core import require
from .representation_checks import sync


@dataclass(frozen=True)
class ATrainingConfig:
    updates: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.
    coefficient_weight: float = .1
    microbatch: int = 8
    effective_batch: int = 16
    precision: str = "float32"
    preset: str = "smoke"
    tiny_overfit: bool = False

    def __post_init__(self):
        require(self.preset in {"smoke","tiny_overfit"},"research/large training is not implemented in Stage 4")
        require(type(self.updates) is int and (3<=self.updates<=5 if self.preset=="smoke" else self.tiny_overfit and 1<=self.updates<=100),"smoke requires 3–5 updates; tiny_overfit requires explicit opt-in, at most 100")
        require(type(self.microbatch) is int and type(self.effective_batch) is int and min(self.microbatch,self.effective_batch)>0,"positive batch sizes required")
        require(all(math.isfinite(v) for v in (self.learning_rate,self.weight_decay,self.gradient_clip,self.coefficient_weight)) and self.learning_rate>0 and self.weight_decay>=0 and self.gradient_clip>0 and self.coefficient_weight>=0,"invalid optimizer/objective settings")
        require(self.precision=="float32","only verified explicit float32 training supported")


@dataclass(frozen=True)
class SupervisedBatch:
    specs: tuple
    observations: object
    queries: object
    target: torch.Tensor
    unknowns: object
    case_ids: tuple[str,...]

    def slice(self,lo,hi):
        def sliced(batch):return type(batch)(**{f.name:getattr(batch,f.name)[lo:hi] for f in fields(batch)})
        return SupervisedBatch(self.specs[lo:hi],sliced(self.observations),sliced(self.queries),self.target[lo:hi],sliced(self.unknowns),self.case_ids[lo:hi])


def objective(output,batch,*,coefficient_weight=.1):
    """Mean cases, then available quantities, then valid queries; xi mean active.

    Empty query cases contribute differentiable zero field loss (no available
    quantities). Inactive slots are sanitized before every subtraction/division.
    """
    require(math.isfinite(coefficient_weight) and coefficient_weight>=0,"invalid coefficient weight")
    q=batch.queries;batch.unknowns.validate(batch.specs)
    require(batch.target.shape==output.predicted_value.shape and batch.target.dtype==output.predicted_value.dtype
            and batch.target.device==output.predicted_value.device,"target shape/dtype/device mismatch")
    require(bool(torch.isfinite(batch.target[q.valid]).all()),"nonfinite valid field target")
    require(torch.equal(output.xi_active,batch.unknowns.active),"coefficient target mask mismatch")
    pred=masked_values(output.predicted_value,q.valid);target=masked_values(batch.target,q.valid)
    scales=output.context.quantity_scales.gather(1,q.masked().quantity_id)[...,None]
    square=((pred-target)/scales).square()[...,0]
    case_field=[]
    for row in range(len(batch.specs)):
        quantities=[]
        for quantity in (0,1):
            valid=q.valid[row] & (q.quantity_id[row]==quantity)
            if bool(valid.any()):quantities.append(square[row,valid].mean())
        case_field.append(torch.stack(quantities).mean() if quantities else square[row].sum()*0)
    field=torch.stack(case_field)
    active=output.xi_active
    delta=(masked_values(output.xi_hat,active)-masked_values(batch.unknowns.xi,active))/output.context.prior_std
    xi=(masked_values(delta.square(),active).sum(1)/active.sum(1).clamp_min(1))
    return {"field":field.mean(),"xi":xi.mean(),"total":(field+coefficient_weight*xi).mean(),
            "field_per_case":field,"xi_per_case":xi}


def observation_agreement(model,context,observations):
    predicted=model.predict_queries(context,observations).predicted_value
    residual=predicted-observations.masked().value
    result=[]
    for i,spec in enumerate(context.specs):
        quantities={}
        for q in (0,1):
            valid=observations.valid[i] & (observations.quantity_id[i]==q)
            r=residual[i,valid,0];sigma=observations.sigma[i,valid,0]
            if len(r):quantities[str(q)]={"count":len(r),"physical_RMSE":float(r.square().mean().sqrt().detach()),
                                        "noise_standardized_RMSE":float((r/sigma).square().mean().sqrt().detach())}
        result.append({"pde":spec.name,"quantities":quantities})
    return result


def gradient_report(model):
    result={}
    for name,module in model.named_children():
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        result[name]={"l2_norm":math.sqrt(sum(float(g.detach().square().sum()) for g in grads)),
                      "missing":[n for n,p in module.named_parameters() if p.grad is None],
                      "zero":[n for n,p in module.named_parameters() if p.grad is not None and not bool(p.grad.any())],
                      "nonfinite":[n for n,p in module.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]}
    return result


def make_training_batch(dataset,normalizers,step,*,count=16,device="cpu",fixed=False):
    """Independent keyed family-uniform sampling, train only; monitoring is inert.

    Tiny overfit fixes the complete cases/observations/queries at step zero.
    Field truth is generated only here on CPU float64 then explicitly transferred.
    """
    from .data import observations,training_queries,case_coefficients,collate_inputs
    from .reference import canonical_spec,evaluate_batch
    from .representations import float32_batch
    from .batches import UnknownCoefficients
    names=("poisson","heat","wave");step=0 if fixed else step
    rng=dataset.streams.numpy("optimization","A","case-sampler",step)
    cases=[dataset.case("train",names[int(rng.integers(3))],int(rng.integers(12000))) for _ in range(count)]
    specs=tuple(canonical_spec(c.key.pde) for c in cases)
    obs=collate_inputs([observations(dataset,c,normalizers,set_id=f"A-smoke-{step}").inputs for c in cases])
    query=collate_inputs([training_queries(dataset,c,set_id=f"A-smoke-{step}").inputs for c in cases])
    unknown=case_coefficients(cases)
    target=evaluate_batch(specs,unknown,query)
    return SupervisedBatch(specs,float32_batch(obs,device=device),float32_batch(query,device=device),target.to(device=device,dtype=torch.float32),
                           UnknownCoefficients(unknown.xi.to(device=device,dtype=torch.float32),unknown.active.to(device)),tuple(c.identity for c in cases))


def train_update(model,optimizer,batch,config):
    """One actual update; unequal final microbatches weighted by physical cases."""
    model.train();optimizer.zero_grad(set_to_none=True)
    device=next(model.parameters()).device
    count=len(batch.specs);totals={k:0. for k in ("field","xi","total")}
    forward_seconds=backward_seconds=0.
    for lo in range(0,count,config.microbatch):
        hi=min(count,lo+config.microbatch);part=batch.slice(lo,hi)
        sync(device);start=time.perf_counter()
        output=model(part.specs,part.observations,part.queries)
        losses=objective(output,part,coefficient_weight=config.coefficient_weight)
        sync(device);forward_seconds+=time.perf_counter()-start
        require(bool(torch.isfinite(losses["total"])),"nonfinite A loss; update failed")
        start=time.perf_counter();(losses["total"]*((hi-lo)/count)).backward()
        sync(device);backward_seconds+=time.perf_counter()-start
        for key in totals:totals[key]+=float(losses[key].detach())*((hi-lo)/count)
    gradients=gradient_report(model)
    require(all(not g["nonfinite"] for g in gradients.values()),"nonfinite A gradients; update failed")
    before=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip,error_if_nonfinite=True))
    after=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in model.parameters() if p.grad is not None))
    sync(device);start=time.perf_counter();optimizer.step();sync(device)
    update_seconds=time.perf_counter()-start
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"nonfinite A weights after update")
    return {**totals,"forward_seconds":forward_seconds,"backward_seconds":backward_seconds,"update_seconds":update_seconds,
            "gradient_norm_before_clip":before,"gradient_norm_after_clip":after,"module_gradients":gradients,
            "learning_rate":optimizer.param_groups[0]["lr"],"physical_cases":count,
            "microbatch_sizes":[min(config.microbatch,count-lo) for lo in range(0,count,config.microbatch)]}


def run_bounded(model,dataset,normalizers,config=ATrainingConfig()):
    device=next(model.parameters()).device
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    logs=[];started=time.perf_counter();counts={"attempted":0,"successful":0,"skipped":0,"failed":0}
    for step in range(config.updates):
        batch=make_training_batch(dataset,normalizers,step,count=config.effective_batch,device=str(device),fixed=config.tiny_overfit)
        counts["attempted"]+=1
        try:log=train_update(model,optimizer,batch,config)
        except Exception:
            counts["failed"]+=1
            raise
        counts["successful"]+=1
        log.update(update=step+1,rolling_total_loss=sum(x["total"] for x in logs[-4:])+log["total"])
        log["rolling_total_loss"]/=min(len(logs)+1,5)
        log["family_counts"]={name:sum(s.name==name for s in batch.specs) for name in ("poisson","heat","wave")}
        logs.append(log)
    model.eval()
    with torch.no_grad():
        output=model(batch.specs,batch.observations,batch.queries)
        values=output.predicted_value[batch.queries.valid]
        predictions={"mean":float(values.mean()),"std":float(values.std(unbiased=False)),"max_abs":float(values.abs().max())}
        agreement=observation_agreement(model,output.context,batch.observations)
    # ru_maxrss bytes on macOS, KiB on Linux; process peak is not model allocation.
    import sys
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
    memory={"process_peak_RSS_bytes":rss,"RSS_scope":"entire process, includes calibration/libraries; not isolated model peak",
            "parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
            "gradient_bytes":sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None),
            "adam_state_tensor_bytes":sum(t.numel()*t.element_size() for state in optimizer.state.values() for t in state.values() if isinstance(t,torch.Tensor))}
    if device.type=="cuda":memory["CUDA_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device)
    if device.type=="mps":memory.update(MPS_current_allocated_bytes=torch.mps.current_allocated_memory(),MPS_driver_allocated_bytes=torch.mps.driver_allocated_memory(),MPS_peak="unavailable in this API; current and process peak reported")
    return {"status":"smoke-tested" if config.preset=="smoke" else "tiny-overfit diagnostic","updates":logs,"counters":counts,
            "device":str(device),"precision":"float32","mixed_precision":False,"elapsed_seconds":time.perf_counter()-started,
            "sampling_distribution":{"poisson":1/3,"heat":1/3,"wave":1/3},"schedule":"constant learning rate for bounded diagnostic; research warmup/cosine not implemented",
            "prediction_statistics":predictions,"observation_agreement":agreement,"memory":memory,
            "interpretation":"execution check; 3–5 updates are not evidence of learned reconstruction"},optimizer
