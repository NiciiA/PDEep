"""C objectives and bounded optimizer path. Truth is supervised data only."""
from dataclasses import dataclass
import math
import resource
import sys
import time
import torch
from .a_training import objective as supervised_objective,SupervisedBatch,gradient_report,observation_agreement
from .batches import UnknownCoefficients
from .constraints import physical_losses,strong_diagnostics
from .core import require
from .representation_checks import sync


@dataclass(frozen=True)
class CTrainingConfig:
    updates: int=3
    microbatch: int=1
    effective_batch: int=3  # explicit bounded preflight; three accumulated cases
    learning_rate: float=3e-4
    weight_decay: float=1e-4
    gradient_clip: float=1.
    coefficient_weight: float=.1
    weak_weight: float=.1
    initial_weight: float=.1
    spatial_order: int=16
    time_order: int=8
    spatial_chunk: int | None=None
    precision: str="float32"
    preset: str="smoke"
    tiny_overfit: bool=False

    def __post_init__(self):
        require(self.preset in {"smoke","tiny_overfit"},"C research training is not implemented")
        require(type(self.updates) is int and (3<=self.updates<=5 if self.preset=="smoke" else self.tiny_overfit and 1<=self.updates<=100),"C requires 3–5 smoke updates or explicit tiny_overfit <=100")
        require(self.tiny_overfit==(self.preset=="tiny_overfit"),"tiny_overfit flag/preset mismatch")
        require(type(self.microbatch) is int and type(self.effective_batch) is int and min(self.microbatch,self.effective_batch)>0,"positive C batch sizes required")
        require(self.spatial_order==16 and self.time_order==8,"C training quadrature fixed at 16x16 and all eight times")
        require(self.spatial_chunk is None or type(self.spatial_chunk) is int and self.spatial_chunk>0,"invalid C spatial chunk")
        require(self.precision=="float32","only explicit float32 C neural precision verified")
        require(all(math.isfinite(v) and v>=0 for v in (self.coefficient_weight,self.weak_weight,self.initial_weight,self.weight_decay)),"invalid C weights")
        require(math.isfinite(self.learning_rate) and self.learning_rate>0 and math.isfinite(self.gradient_clip) and self.gradient_clip>0,"invalid C optimizer settings")


def objective(model,output,batch,*,coefficient_weight=.1,weak_weight=.1,initial_weight=.1,spatial_chunk=None,physical=None):
    require(all(math.isfinite(w) and w>=0 for w in (coefficient_weight,weak_weight,initial_weight)),"nonnegative finite C weights required")
    supervised=supervised_objective(output,batch,coefficient_weight=coefficient_weight)
    physical=physical_losses(model,output.context,xi_hat=output.xi_hat,spatial_chunk=spatial_chunk) if physical is None else physical
    weak=physical.weak_per_case.mean();initial=physical.initial_per_case.mean()
    return {"field":supervised["field"],"xi":supervised["xi"],"weak":weak,"initial":initial,
            "weighted_field":supervised["field"],"weighted_xi":coefficient_weight*supervised["xi"],
            "weighted_weak":weak_weight*weak,"weighted_initial":initial_weight*initial,
            "total":supervised["total"]+weak_weight*weak+initial_weight*initial,"physical":physical}


def make_training_batch(dataset,normalizers,step,*,count=3,device="cpu",fixed=False):
    """Uniform stratified PDE families; train-only keyed cases, unchanged data risk."""
    from .data import observations,training_queries,collate_inputs,case_coefficients
    from .reference import canonical_spec,evaluate_batch
    from .representations import float32_batch
    step=0 if fixed else step
    rng=dataset.streams.numpy("optimization","C","case-sampler",step)
    families=("poisson","heat","wave");selected=[]
    while len(selected)<count:selected.extend(rng.permutation(3).tolist())
    cases=[dataset.case("train",families[index],int(rng.integers(12000))) for index in selected[:count]]
    specs=tuple(canonical_spec(c.key.pde) for c in cases)
    obs=collate_inputs([observations(dataset,c,normalizers,set_id=f"C-smoke-{step}").inputs for c in cases])
    queries=collate_inputs([training_queries(dataset,c,set_id=f"C-smoke-{step}").inputs for c in cases]);xi=case_coefficients(cases)
    target=evaluate_batch(specs,xi,queries)
    return SupervisedBatch(specs,float32_batch(obs,device=device),float32_batch(queries,device=device),target.to(device=device,dtype=torch.float32),
                           UnknownCoefficients(xi.xi.to(device=device,dtype=torch.float32),xi.active.to(device)),tuple(c.identity for c in cases))


def train_update(model,optimizer,batch,config):
    model.train();optimizer.zero_grad(set_to_none=True);device=next(model.parameters()).device;n=len(batch.specs)
    totals={k:0. for k in ("field","xi","weak","initial","weighted_field","weighted_xi","weighted_weak","weighted_initial","total")}
    timing={k:0. for k in ("forward_seconds","backward_seconds","quadrature_forward_seconds","coordinate_derivative_seconds")}
    physical_metrics=[]
    evaluations={k:0 for k in ("field_evaluations","weak_points","initial_points")}
    for lo in range(0,n,config.microbatch):
        hi=min(lo+config.microbatch,n);b=batch.slice(lo,hi)
        sync(device);start=time.perf_counter();out=model(b.specs,b.observations,b.queries)
        losses=objective(model,out,b,coefficient_weight=config.coefficient_weight,weak_weight=config.weak_weight,initial_weight=config.initial_weight,spatial_chunk=config.spatial_chunk)
        sync(device);timing["forward_seconds"]+=time.perf_counter()-start
        require(bool(torch.isfinite(losses["total"])),"nonfinite C objective")
        start=time.perf_counter();(losses["total"]*((hi-lo)/n)).backward();sync(device)
        timing["backward_seconds"]+=time.perf_counter()-start
        physical_metrics.extend({"pde":case["pde"],"weak_physical_MSE_by_equation":[float(v.detach()) for v in case["physical_weak_MSE"]],"initial_physical_MSE_by_quantity":[float(v.detach()) for v in case["physical_initial_MSE"]]} for case in losses["physical"].cases)
        for k in totals:totals[k]+=float(losses[k].detach())*((hi-lo)/n)
        for k in ("quadrature_forward_seconds","coordinate_derivative_seconds"):timing[k]+=losses["physical"].timing[k]
        for k in evaluations:evaluations[k]+=losses["physical"].timing[k]
    gradients=gradient_report(model)
    require(all(not g["nonfinite"] for g in gradients.values()),"nonfinite C gradients")
    raw=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip,error_if_nonfinite=True))
    clipped=sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None)**.5
    sync(device);start=time.perf_counter();optimizer.step();sync(device)
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"nonfinite C parameters after optimizer")
    return {**totals,**timing,**evaluations,"physical_metrics":physical_metrics,"update_seconds":time.perf_counter()-start,"gradient_before_clip":raw,"gradient_after_clip":clipped,
            "gradients":gradients,"learning_rate":optimizer.param_groups[0]["lr"],"microbatch_sizes":[min(config.microbatch,n-lo) for lo in range(0,n,config.microbatch)]}


def sensor_diagnostics(model,context,observations):
    """Gaussian likelihood with its case-level SUM, diagnostic only."""
    with torch.no_grad():
        prediction=model.predict_queries(context,observations).predicted_value
        result=observation_agreement(model,context,observations)
        for i,row in enumerate(result):
            mask=observations.valid[i];r=(prediction-observations.masked().value)[i,mask,0];sigma=observations.sigma[i,mask,0]
            row['Gaussian_negative_log_likelihood_sum']=float((.5*((r/sigma).square()+2*sigma.log()+math.log(2*math.pi))).sum())
        return result


def run_bounded(model,dataset,normalizers,config=CTrainingConfig()):
    device=next(model.parameters()).device;logs=[];counters=dict(attempted=0,successful=0,skipped=0,failed=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    for step in range(config.updates):
        batch=make_training_batch(dataset,normalizers,step,count=config.effective_batch,device=device.type if device.type in {'cpu','mps'} else str(device),fixed=config.tiny_overfit)
        counters["attempted"]+=1
        try:log=train_update(model,optimizer,batch,config)
        except Exception:
            counters["failed"]+=1
            raise
        counters["successful"]+=1
        log['update']=step+1;log['rolling_total_loss']=(sum(l['total'] for l in logs[-4:])+log['total'])/min(5,len(logs)+1)
        log['family_counts']={p:sum(s.name==p for s in batch.specs) for p in ("poisson","heat","wave")}
        log['case_ids']=list(batch.case_ids);logs.append(log)
    model.eval()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
    memory={"process_peak_RSS_bytes":rss,"RSS_scope":"whole process including calibration/libraries; not isolated model peak",
            "parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
            "gradient_bytes":sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None),
            "Adam_state_tensor_bytes":sum(t.numel()*t.element_size() for state in optimizer.state.values() for t in state.values() if isinstance(t,torch.Tensor))}
    if device.type=="cuda":memory["CUDA_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device)
    if device.type=="mps":memory.update(MPS_current_allocated_bytes=torch.mps.current_allocated_memory(),MPS_driver_allocated_bytes=torch.mps.driver_allocated_memory(),MPS_peak="not exposed by this API")
    return {"status":"smoke-tested" if config.preset=="smoke" else "tiny-overfit diagnostic","updates":logs,"counters":counters,
            "device":str(device),"precision":"float32","AMP":False,"effective_batch":config.effective_batch,
            "accumulation_steps":math.ceil(config.effective_batch/config.microbatch),"quadrature":"16x16 spatial, all 8 physical times; initial at exactly zero",
            "family_sampling":"uniform stratified independent blocks of three; equally weighted cases","elapsed_seconds":time.perf_counter()-start,
            "memory":memory,"interpretation":"execution check; no reconstruction/generalization claim; no research schedule"},optimizer
