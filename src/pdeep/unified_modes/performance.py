"""Bounded, real four-case accumulated backward; never an optimizer update.

Timing barriers and Python derivative-call instrumentation belong ONLY here.
The objective, query chunks, physics quadrature and accumulation weights are the
production ones. CUDA numbers are absent on CPU/MPS, never extrapolated to T4.
"""
import sys,time
from contextlib import contextmanager
import torch
from ..core import require
from ..learning_diagnostics.runtime import using
from .architecture import inventory
from .loss import Adapter

@contextmanager
def derivative_timer(cuda):
    from ..geometric.physics import weak_loss
    from ..learning_diagnostics.kinematic import kinematic_loss
    codes={weak_loss.__code__,kinematic_loss.__code__};active={};records=[];previous=sys.getprofile()
    def hook(frame,event,arg):
        if frame.f_code not in codes:return
        if event=='call':
            pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) if cuda else None
            if pair:pair[0].record()
            active[id(frame)]=(time.perf_counter(),pair)
        elif event=='return' and id(frame) in active:
            start,pair=active.pop(id(frame))
            if pair:pair[1].record()
            records.append((time.perf_counter()-start,pair))
    try:sys.setprofile(hook);yield records
    finally:sys.setprofile(previous)

def accumulated_backward(model,config,cases,make_batch,*,memory_target_gb=11.5):
    require(len(cases)==4 and len({c.case_id for c in cases})==4,'preflight requires four independent physical cases')
    d=next(model.parameters()).device;cuda=d.type=='cuda';info=inventory(model)
    sync=lambda:torch.cuda.synchronize(d) if cuda else torch.mps.synchronize() if d.type=='mps' else None
    model.zero_grad(set_to_none=True);sync()
    if cuda:torch.cuda.reset_peak_memory_stats(d)
    phases={k:[] for k in ('data','forward','backward','accumulation_scaling')};events={k:[] for k in phases};grads=set();seen=[];adapter=Adapter(config)
    def phase(name,fn):
        pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) if cuda else None
        if pair:pair[0].record()
        start=time.perf_counter();result=fn();phases[name].append(time.perf_counter()-start)
        if pair:pair[1].record();events[name].append(pair)
        return result
    start=time.perf_counter()
    try:
        with derivative_timer(cuda) as physics:
            for i,case in enumerate(cases):
                batch=phase('data',lambda:make_batch(case,('FORWARD','INVERSE','HYBRID','INVERSE')[i]))
                seen.append({'case_id':case.case_id,'pde':case.pde,'geometry':case.family,'mode':batch.cases[0].mode,'queries':int(batch.queries.valid.sum())})
                require(int(batch.queries.valid.sum())==config.query_count,'preflight cannot reduce supervised queries')
                with using(adapter.cache):
                    loss=phase('forward',lambda:adapter.loss(model,batch))
                    scaled=phase('accumulation_scaling',lambda:loss.total/4)
                    phase('backward',scaled.backward)
                require(bool(torch.isfinite(loss.total.detach())),'nonfinite accumulated loss')
                grads.update(k for k,p in model.named_parameters() if p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.detach().abs().max()>0))
                del scaled,loss,batch
                adapter.cache.clear_queries();adapter.window.clear()
        sync();seconds=time.perf_counter()-start
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'nonfinite accumulated gradients')
        missing=sorted(set(dict(model.named_children()))-{k.split('.')[0] for k in grads}-{'coordinates'})
        require(not missing,'disconnected applicable submodules: '+str(missing))
        allocated=torch.cuda.max_memory_allocated(d) if cuda else None;reserved=torch.cuda.max_memory_reserved(d) if cuda else None
        total=torch.cuda.get_device_properties(d).total_memory if cuda else None
        # Gradients are measured. Reserve two lazy Adam moments plus one parameter
        # buffer for foreach temporaries. This is a budget, not measured Adam time.
        budget=allocated+12*info['trainable_parameters'] if cuda else None
        limit=min(memory_target_gb*1e9,.8*total) if cuda else None
        gpu={k:sum(a.elapsed_time(b) for a,b in pairs)/1000 for k,pairs in events.items()} if cuda else None
        return {'status':'fits' if not cuda or budget<=limit else 'over_budget','device':str(d),'GPU_name':torch.cuda.get_device_name(d) if cuda else None,
                'inventory':info,'physical_cases':seen,'effective_cases':4,'supervised_queries':4*config.query_count,'query_chunk':config.query_chunk,
                'optimizer_steps':0,'optimizer_latency_seconds':None,'precision':'FP32; auto remains off; no unverified FP16 or T4 BF16',
                'step_wall_seconds':seconds,'cases_per_second':4/seconds,'queries_per_second':4*config.query_count/seconds,
                'phase_CPU_wall_enqueue_seconds':{k:sum(v) for k,v in phases.items()},'phase_CUDA_event_seconds':gpu,
                'physics_forward_CPU_wall_enqueue_seconds':sum(v for v,_ in physics),
                'physics_forward_CUDA_seconds':sum(a.elapsed_time(b) for _,(a,b) in physics)/1000 if cuda else None,
                'physics_timing_scope':'weak/kinematic call includes coordinate differentiation; parameter backward included in backward phase; timings instrumented',
                'accumulation_scope':'four real backwards, each objective divided by4; scaling overhead reported separately; no fabricated parallel-batch comparison',
                'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,'estimated_peak_with_Adam_bytes':budget,'memory_limit_bytes':limit,
                'CUDA_scope':'actual device only; CPU timings do not qualify T4','active_parameter_names':sorted(grads),'inactive_submodules':missing}
    finally:model.zero_grad(set_to_none=True);adapter.cache.clear_queries()
