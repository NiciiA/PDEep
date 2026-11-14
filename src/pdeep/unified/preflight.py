"""Bounded zero-update model/derivative/memory checks at the requested settings."""
from dataclasses import asdict
from pathlib import Path
import time
import torch
from ..core import digest,require
from ..artifact_io import atomic_json,write_artifact,read_payload
from ..artifact_rng import capture_rng,restore_rng
from ..artifact_tensors import state_identity,tree_identity
from ..geometric.models import create
from ..learning_diagnostics.performance import profile
from .loss import UnifiedAdapter
from .data import Provider
from .config import SCHEMA,implementation_identity


def inventory(configs,norms):
    out=[]
    for c in configs:
        model=create(c.architecture,norms,c.capacity,seed=c.root_seed,device='cpu')
        count=sum(p.numel() for p in model.parameters());out.append({'model':c.model_key,'capacity':c.capacity,'parameters':count,
            'parameter_float32_MB':count*4/1e6,'Adam_moments_float32_MB':count*8/1e6,
            'scope':'complete existing network, including its independent representation utilities; ONE shared network over all selected tasks',
            'instances':'independent trainable objects; C variants use matched seed and initially identical values, never shared parameters'})
        del model
    return out


def _run_preflight(configs,dataset,root,*,progress=None):
    """No optimizer construction/step. Each PDE loss at full per-case query count.

    Effective batch four uses one-case accumulation, so peak graph is one case.
    Other microbatch sizes are explicitly rejected until measured on that device.
    """
    progress=progress or (lambda stage,**kw:print(stage,kw,flush=True));dataset.progress=progress;dataset.require_ready('train');dataset.require_ready('validation')
    rng=capture_rng();reports=[];provider=Provider(dataset);start=time.perf_counter()
    try:
        for c in configs:
            require(c.microbatch==1,'preflight V1 supports measured single-case accumulation only; retain effective batch four')
            progress('Model construction',model=c.model_key);model=create(c.architecture,dataset.norms,c.capacity,seed=c.root_seed,device=c.device)
            before=state_identity(model);adapter=UnifiedAdapter(c);adapter.amp='off'
            for p in dataset.config.pdes:
                case=max((v for v in dataset.cases('train',training_only=True) if v.pde==p),key=lambda v:len(v.problem.mesh.cells))
                progress('Numerical preflight',model=c.model_key,pde=p,queries_per_case=c.query_count,effective_cases=4)
                data_started=time.perf_counter()
                batch,ids=provider.batch((case,),count=c.query_count,device=c.device,seed=c.data_seed,observation_count=128)
                data_seconds=time.perf_counter()-data_started
                measured=profile(model,batch,adapter,memory_target_gb=11.5,optimizer_step=False)
                if torch.device(c.device).type=='cuda':
                    measured['memory']['peak_reserved_bytes']=torch.cuda.max_memory_reserved(c.device)
                    measured['memory']['peak_allocated_bytes']=torch.cuda.max_memory_allocated(c.device)
                    weights=sum(v.numel()*v.element_size() for v in model.parameters())
                    measured['estimated_peak_with_optimizer_bytes']=max(measured['estimated_peak_with_optimizer_bytes'],measured['memory']['peak_allocated_bytes']+3*weights)
                    measured['status']='fits' if measured['estimated_peak_with_optimizer_bytes']<=measured['memory_target_bytes'] else 'over_budget'
                require(measured['status']=='fits','preflight exceeded memory: use a smaller query chunk, not fewer physical cases/losses: '+str(measured))
                # Separate bounded derivative path evidence; no optimizer and no running
                # statistics. Frozen metadata model uses layer norms, not batch norms.
                model.zero_grad(set_to_none=True)
                small,_=provider.batch((case,),count=min(32,c.query_count),device=c.device,seed=c.data_seed)
                loss=adapter.loss(model,small,detailed=True);loss.total.backward()
                require(all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None),'nonfinite preflight gradients')
                require(any(v.grad is not None and bool(v.grad.abs().max()>0) for v in model.parameters()),'disconnected gradients')
                reports.append({'config_identity':c.identity,'model':c.model_key,'pde':p,'geometry':case.family,'data_preparation_seconds':data_seconds,'performance':measured,'derivative_paths':adapter.last,
                                'tensor_shapes':{'observations':list(batch.observations.position.shape),'queries':list(batch.queries.position.shape)},
                                'reference_dtype':'CPU float64','model_dtype':'float32','device':c.device})
                model.zero_grad(set_to_none=True);adapter.cache.clear_queries();del loss,batch,small
            require(state_identity(model)==before,'preflight changed original model');del model,adapter
            if torch.cuda.is_available():torch.cuda.empty_cache()
    finally:restore_rng(rng,target_device=configs[0].device)
    result={'schema':SCHEMA,'passed':True,'source':implementation_identity(),'dataset_identity':dataset.config.identity,
            'configs':[c.identity for c in configs],'reports':reports,'seconds':time.perf_counter()-start,'optimizer_steps':0,
            'CUDA_Drive_scope':'CUDA memory only if reported on actual CUDA; no real Drive test implied'}
    path=write_artifact(Path(root)/'preflight',digest(result),{'artifact_type':'diagnostic_bundle','architecture_id':'PREFLIGHT','run_id':'G1-unified-preflight','scientific_schema':SCHEMA},
                        {'preflight':result},required=('preflight',))
    return str(path),result


def require_preflight(path,configs,dataset):
    require(path is not None,'run the bounded preflight before training')
    _,p=read_payload(path,kind='diagnostic_bundle',required=('preflight',));r=p['preflight']
    require(r['passed'] and r['source']==implementation_identity() and r['dataset_identity']==dataset.config.identity,'stale/incompatible preflight')
    require(set(c.identity for c in configs)<=set(r['configs']),'preflight settings/device differ from selected training configurations')
    return r


def run_preflight(configs,dataset,root,*,progress=None):
    from .monitor import Dashboard
    if progress is not None:return _run_preflight(configs,dataset,root,progress=progress)
    dashboard=Dashboard(dataset.config)
    with dashboard.live():
        dashboard.phase('Dataset cache loading',operation='preflight')
        return _run_preflight(configs,dataset,root,progress=dashboard.phase)
