"""Isolated measurements, device evidence and opt-in legacy engine timing policy.

No alternate optimizer loop: production calls the EXACT shared update code object
with a private globals dictionary changing only its explicit `sync` timing helper.
This avoids process-wide monkeypatches and preserves legacy/T1 source fingerprints
and execution. Scalar safety/report checks still entail some implicit fences.
"""
import copy,subprocess,time,types
from contextlib import nullcontext
import numpy as np
import torch
from ..core import require
from ..training import TrainingEngine
from ..artifact_tensors import state_identity
from .runtime import using

# Same engine bytecode, same AdamW, scheduling, clipping and failure transactions.
# Only explicit phase-timing barriers are removed; no training algorithm is copied.
asynchronous_update=types.FunctionType(TrainingEngine.update.__code__,{**TrainingEngine.update.__globals__,'sync':lambda device:None},'shared_update_without_timing_barriers',TrainingEngine.update.__defaults__,TrainingEngine.update.__closure__)


def memory(device):
    d=torch.device(device)
    if d.type=='cuda':
        i=d.index if d.index is not None else torch.cuda.current_device()
        return {'device':str(d),'GPU_name':torch.cuda.get_device_name(i),'allocated_bytes':torch.cuda.memory_allocated(i),'reserved_bytes':torch.cuda.memory_reserved(i),'peak_allocated_bytes':torch.cuda.max_memory_allocated(i),'total_bytes':torch.cuda.get_device_properties(i).total_memory,'available':True}
    return {'device':str(d),'available':False,'allocated_bytes':torch.mps.current_allocated_memory() if d.type=='mps' else None,'reserved_bytes':None,'peak_allocated_bytes':None,'total_bytes':None,'GPU_name':None}

_UTIL={}
def utilization(device):
    if torch.device(device).type!='cuda':return {'percent':None,'scope':'not CUDA'}
    now=time.monotonic();key=str(device)
    if key in _UTIL and now-_UTIL[key][0]<5:return _UTIL[key][1]
    try:
        i=torch.device(device).index
        if i is None:i=torch.cuda.current_device()
        p=subprocess.run(['nvidia-smi',f'--id={i}','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
        value=float(p.stdout.strip()) if p.returncode==0 else None
    except (OSError,ValueError,subprocess.TimeoutExpired):value=None
    result={'percent':value,'scope':'device-wide instantaneous nvidia-smi sample; NOT per-process utilization','sample_monotonic':now};_UTIL[key]=(now,result);return result


def device_audit(model,batch,adapter,optimizer=None,*,backward=False):
    seen={};handles=[]
    def hook(name):
        def collect(m,inputs,output):
            def tensors(v):
                if isinstance(v,torch.Tensor):return [str(v.device)]
                if isinstance(v,(tuple,list)):return [x for p in v for x in tensors(p)]
                return []
            seen[name]={'tensor_inputs':sorted(set(tensors(inputs))),'tensor_outputs':sorted(set(tensors(output)))}
        return collect
    for name in ('equation','geometry','observation','anchor_input','geometry_cross','observation_cross','context','decoder','u_head','v_head'):
        handles.append(getattr(model,name).register_forward_hook(hook(name)))
    for i,layer in enumerate(model.memory_layers):handles.append(layer.register_forward_hook(hook('spatial_memory_'+str(i))))
    try:
        with using(adapter.cache if hasattr(adapter,'cache') else None):loss=adapter.loss(model,batch)
        result={'model_parameters':sorted({str(p.device) for p in model.parameters()}),'observation':{k:str(v.device) for k,v in vars(batch.observations).items() if v is not None},'queries_and_targets':{k:str(v.device) for k,v in vars(batch.queries).items() if v is not None},'loss':str(loss.total.device),'modules':seen,'mesh_metadata':'immutable CPU topology/locator; registered graph/connectivity/geometry tensors transferred and reused on neural device','reference':'CPU float64; targets explicitly copied to neural device'}
        if hasattr(adapter,'cache'):
            result['cached_graph_tensors']=[{k:str(v.device) for k,v in vars(b).items() if isinstance(v,torch.Tensor)} for b in adapter.cache.graph_batches.values()]
            result['cached_mesh_input_tensor_devices']=[sorted({str(v.device) for v in item}) for item in adapter.cache.meshes.values()]
        if backward:
            model.zero_grad(set_to_none=True)
            with using(adapter.cache if hasattr(adapter,'cache') else None):loss.total.backward()
            result['parameter_gradients']=sorted({str(p.grad.device) for p in model.parameters() if p.grad is not None})
            model.zero_grad(set_to_none=True)
        result['parameter_gradients']=sorted({str(p.grad.device) for p in model.parameters() if p.grad is not None})
        result['backward_evidence']='existing gradients from completed isolated backward' if result['parameter_gradients'] else 'no backward requested/existing gradients'
        expected=str(batch.queries.position.device)
        require(result['model_parameters']==[expected] and result['loss']==expected,'neural model/loss device mismatch')
        require(all(str(v.device)==expected for t in (batch.observations,batch.queries) for v in vars(t).values() if isinstance(v,torch.Tensor)),'conditioning/query/target device mismatch')
        if backward or result['parameter_gradients']:require(result['parameter_gradients']==[expected],'backward gradient device mismatch')
        if optimizer is not None:
            result['optimizer_parameters']=sorted({str(p.device) for g in optimizer.param_groups for p in g['params']})
            result['optimizer_moments']=sorted({str(v.device) for state in optimizer.state.values() for k,v in state.items() if isinstance(v,torch.Tensor) and k!='step'})
            result['optimizer_step_scalars']=sorted({str(v.device) for state in optimizer.state.values() for k,v in state.items() if isinstance(v,torch.Tensor) and k=='step'})
        if optimizer is not None:require(result['optimizer_parameters']==[expected] and all(d==expected for d in result['optimizer_moments']),'optimizer parameter/moment device mismatch')
        return result
    finally:
        for h in handles:h.remove()
        if hasattr(adapter,'cache'):adapter.cache.clear_queries()


def resolve_amp(model,batch,adapter,requested):
    require(requested in {'off','auto','bf16'},'AMP supports off/auto/bf16; FP16 GradScaler is not enabled for coordinate derivative losses')
    if requested=='off':return {'resolved':'off','reason':'explicit FP32'}
    if batch.queries.position.device.type!='cuda':
        require(requested=='auto','BF16 requires a CUDA device and a numerical pilot');return {'resolved':'off','reason':'non-CUDA; preserve neural FP32'}
    if model.config.architecture=='C_GEO' and not adapter.field_only:
        require(requested=='auto','full C physics BF16 is not verified; choose off or auto (FP32)')
        return {'resolved':'off','reason':'full C coordinate-derivative physics uses verified FP32'}
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        require(requested=='auto','GPU does not support BF16');return {'resolved':'off','reason':'BF16 hardware unsupported'}
    # Compare current model loss AND parameter gradients on the SAME pilot data.
    from ..geometric.data import Batch
    from .data import select
    idx=torch.arange(min(128,batch.queries.valid.shape[1]),device=batch.queries.position.device)
    pilot=Batch(batch.cases,batch.observations,select(batch.queries,idx),batch.normalizers)
    outputs={}
    try:
        for mode in ('off','bf16'):
            adapter.amp=mode;model.zero_grad(set_to_none=True)
            with using(adapter.cache):r=adapter.loss(model,pilot);r.total.backward()
            outputs[mode]=(float(r.total.detach()),torch.cat([p.grad.detach().float().flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()]))
        l0,g0=outputs['off'];l1,g1=outputs['bf16'];ld=abs(l1-l0)/max(abs(l0),1e-8);gd=float((g1-g0).norm()/g0.norm().clamp_min(1e-8))
        passed=ld<=.005 and gd<=.02
        require(passed or requested=='auto',f'BF16 numerical pilot failed: loss relative {ld}, gradient relative {gd}')
        return {'resolved':'bf16' if passed else 'off','reason':'actual GPU FP32/BF16 loss-gradient comparison','loss_relative_discrepancy':ld,'gradient_relative_discrepancy':gd,'tolerances':{'loss':.005,'gradient':.02}}
    finally:adapter.amp='off';model.zero_grad(set_to_none=True)


def profile(model,batch,adapter,*,memory_target_gb=11,optimizer_step=False,cuda_kernel_profile=False):
    """Bounded ISOLATED clone; preview uses backward only, no optimizer step.

    CUDA phase events are synchronized ONLY here. An optional one-step smoke is
    explicit and disposable. Returned optimizer-state budget is conservative when
    no step runs; it is never described as measured Adam-state allocation.
    """
    d=batch.queries.position.device;clone=copy.deepcopy(model);before=state_identity(model)
    cuda=d.type=='cuda';sync=lambda:torch.cuda.synchronize(d) if cuda else torch.mps.synchronize() if d.type=='mps' else None
    sync();initial=memory(d)
    if cuda:torch.cuda.reset_peak_memory_stats(d)
    events={};wall={};opt=None
    def phase(name,fn):
        if cuda:events[name]=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[name][0].record()
        start=time.perf_counter();value=fn();wall[name]=time.perf_counter()-start
        if cuda:events[name][1].record()
        return value
    capture=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if cuda and cuda_kernel_profile else None
    if capture is not None:capture.start()
    start=time.perf_counter();optimizer_attempted=False;optimizer_completed=False
    try:
        with using(adapter.cache):
            report=phase('forward',lambda:adapter.loss(clone,batch))
            phase('backward',lambda:report.total.backward())
            if optimizer_step:
                opt=torch.optim.AdamW(clone.parameters(),lr=3e-4,weight_decay=1e-4)
                torch.nn.utils.clip_grad_norm_(clone.parameters(),1.)
                optimizer_attempted=True;phase('optimizer',opt.step);optimizer_completed=True
        sync();seconds=time.perf_counter()-start;mem=memory(d)
        gpu={k:a.elapsed_time(b)/1000 for k,(a,b) in events.items()}
        kernels=None
        if capture is not None:
            capture.stop()
            kernels=sum(e.self_device_time_total for e in capture.events() if e.device_type==torch.autograd.DeviceType.CUDA and not any(k in e.name.lower() for k in ('memcpy','memset')))/1e6
            capture=None
        weights=sum(p.numel()*p.element_size() for p in clone.parameters())
        # Budget lazy Adam moments, foreach temporaries and original model resident
        # during this isolated clone preflight. Do not force allocation to reach11GB.
        estimate=(mem['peak_allocated_bytes'] or 0)+(0 if optimizer_step else 3*weights)
        limit=min(memory_target_gb*1e9,.8*mem['total_bytes']) if cuda else None
        result={'status':'fits' if not cuda or estimate<=limit else 'over_budget','memory':mem,'memory_before_clone_step':initial,'estimated_peak_with_optimizer_bytes':estimate if cuda else None,'memory_target_bytes':limit,'optimizer_state_scope':'actual one disposable step' if optimizer_step else 'measured forward/backward + conservative3x parameter-byte optimizer allowance','phase_wall_enqueue_seconds':wall,'phase_CUDA_event_seconds':gpu,'sampled_CUDA_kernel_seconds':kernels,'kernel_time_scope':'optional isolated profiler sum of CUDA kernel durations, excludes copies; not whole-run compute time','step_wall_seconds':seconds,'query_count':int(batch.queries.valid.sum()),'queries_per_second':int(batch.queries.valid.sum())/seconds,'optimizer_steps':int(optimizer_completed),'optimizer_step_attempted':optimizer_attempted,'utilization':utilization(d),'device_audit':device_audit(clone,batch,adapter,opt),'precision':adapter.amp,'profile_scope':'isolated clone, warm/cold state reported; not full training throughput'}
        return result
    except torch.OutOfMemoryError:
        return {'status':'out_of_memory','memory':memory(d),'query_count':int(batch.queries.valid.sum()),'optimizer_steps':int(optimizer_completed),'optimizer_step_attempted':optimizer_attempted,'profile_scope':'isolated failed preflight; original model retained'}
    finally:
        if capture is not None:capture.stop()
        adapter.cache.clear_queries();clone.zero_grad(set_to_none=True);require(state_identity(model)==before,'preflight changed original model')
