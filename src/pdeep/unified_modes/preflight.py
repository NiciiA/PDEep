"""Bounded no-update evidence using the existing profiler and derivative objectives."""
from pathlib import Path
import time
import numpy as np
import torch
from ..core import require,digest
from ..artifact_io import write_artifact,read_payload
from ..artifact_rng import capture_rng,restore_rng
from ..artifact_tensors import state_identity
from ..geometric.data import Batch
from ..unified.data import Provider
from ..learning_diagnostics.performance import profile
from .conditioning import FieldCondition,CaseView,MODES
from .config import extension_identity
from .model import create
from .loss import Adapter
from .dashboard import Dashboard
from .architecture import inventory as model_inventory
from .compatibility import approved_extension
from .performance import accumulated_backward


def preflight_batch(dataset,case,mode,count,device,seed=42):
    b,_=Provider(dataset).batch((case,),count=count,device=device,seed=seed,observation_count=128)
    d=64 if case.pde=='wave' else 32;mask=np.ones(d,bool) if mode=='FORWARD' else np.zeros(d,bool)
    if mode=='HYBRID':mask[:d//2]=True
    view=CaseView(case,FieldCondition.from_case(case,mask),mode)
    if mode=='FORWARD':
        for v in vars(b.observations).values():
            if v is not None:v.zero_()
    return Batch((view,),b.observations,b.queries,b.normalizers)


def inventory(configs,norms):
    rows=[]
    for c in configs:
        model=create(c.architecture,norms,c.capacity,seed=c.root_seed);base=sum(p.numel() for k,p in model.named_parameters() if not k.startswith(('known_field_encoder.','known_field_cross.')));info=model_inventory(model)
        rows.append({'model':c.model_key,**info,'base_parameters':base,'added_field_parameters':info['trainable_parameters']-base})
    return rows


def run_preflight(configs,dataset,root,*,monitor=None,query_counts=None):
    dataset.require_ready('train');dataset.require_ready('validation');monitor=monitor or Dashboard(dataset.config);rng=capture_rng();rows=[];accumulation=[];start=time.perf_counter()
    try:
        with monitor.live():
            for c in configs:
                require(approved_extension(c.extension_source,c.capacity),'stale config');model=create(c.architecture,dataset.norms,c.capacity,seed=c.root_seed,device=c.device);before=state_identity(model)
                info=model_inventory(model);monitor.latest.update(model_capacity=c.capacity,trainable_parameters=info['trainable_parameters'],architecture_configuration=info['model_config'])
                for pde in dataset.config.pdes:
                    case=max((c for c in dataset.cases('train') if c.pde==pde),key=lambda c:len(c.problem.mesh.cells))
                    for mode in MODES:
                        count=c.query_count;monitor.phase('Memory preflight',model=c.model_key,pde=pde,geometry=case.family,mode=mode,queries=count,optimizer_steps=0)
                        batch=preflight_batch(dataset,case,mode,count,c.device,c.data_seed);adapter=Adapter(c)
                        measured=profile(model,batch,adapter,memory_target_gb=11.5,optimizer_step=False)
                        require(measured['status']=='fits','measured memory exceeds budget; explicitly adjust chunk/case settings and rerun preflight')
                        small=preflight_batch(dataset,case,mode,min(32,count),c.device,c.data_seed)
                        model.zero_grad(set_to_none=True);adapter.loss(model,small,detailed=True).total.backward()
                        grads={k:float(v.grad.detach().norm()) for k,v in model.named_parameters() if v.grad is not None}
                        require(any(v>0 for k,v in grads.items() if k.startswith('known_field_encoder')),'field encoder gradient disconnected')
                        if mode!='FORWARD':require(any(v>0 for k,v in grads.items() if k.startswith('observation.')),'observation gradient disconnected')
                        rows.append({'model':c.model_key,'pde':pde,'mode':mode,'geometry':case.family,'queries':count,'effective_cases':4,'profile':measured,'field_encoder_gradient_norm':sum(v*v for k,v in grads.items() if k.startswith('known_field_encoder'))**.5,'observation_gradient_norm':sum(v*v for k,v in grads.items() if k.startswith('observation.'))**.5,'field_physics_gradient_conflict':adapter.last.get('field_physics_gradient')})
                        model.zero_grad(set_to_none=True);adapter.cache.clear_queries()
                # Optional explicitly prepared throughput candidates, no change to main config.
                for count in (query_counts or ()):
                    require(count in (2048,4096,8192),'throughput candidates are8192/16384/32768 queries per four-case update')
                    case=next(v for v in dataset.cases('train') if v.pde=='wave');batch=throughput_batch(dataset,case,count,c.device,c.data_seed)
                    measured=profile(model,batch,Adapter(c),memory_target_gb=11.5,optimizer_step=False);rows.append({'model':c.model_key,'throughput_candidate_per_update':4*count,'profile':measured,'scope':'isolated backward/no optimizer; prefix/concatenation of distinct exact prepared draws for this physical case, not a new training sampler'})
                if c.capacity=='large':
                    available=list(dataset.cases('train'));chosen=[]
                    # Include all PDEs plus two wave cases, prefer irregular meshes.
                    for pde in ('wave','heat','poisson','wave'):
                        candidates=[v for v in available if v.pde==pde and v.case_id not in {x.case_id for x in chosen}]
                        require(candidates,'four-case preflight needs all PDEs and at least two wave cases')
                        chosen.append(max(candidates,key=lambda v:(v.family not in ('square','rectangle'),len(v.problem.mesh.cells))))
                    monitor.phase('Four-case accumulated backward preflight',model=c.model_key,queries_per_case=c.query_count,query_chunk=c.query_chunk,optimizer_steps=0)
                    r=accumulated_backward(model,c,chosen,lambda case,mode:preflight_batch(dataset,case,mode,c.query_count,c.device,c.data_seed))
                    require(r['status']=='fits','large four-case accumulated backward exceeds CUDA memory budget; training remains blocked')
                    accumulation.append({'model':c.model_key,**r})
                require(state_identity(model)==before,'preflight changed parameters');del model
    finally:restore_rng(rng,target_device=configs[0].device)
    result={'source':extension_identity(),'dataset':dataset.config.identity,'configs':[c.identity for c in configs],'reports':rows,'accumulation_reports':accumulation,'seconds':time.perf_counter()-start,'optimizer_steps':0,'passed':True,'CUDA_scope':'actual memory/throughput only on recorded CUDA device; no T4 or Drive claim from CPU tests'}
    path=write_artifact(Path(root)/'preflight-modes',digest(result),{'artifact_type':'diagnostic_bundle','architecture_id':'G1_MODES_PREFLIGHT','run_id':'G1-field-conditioning-preflight'},{'preflight':result},required=('preflight',));return str(path),result


def require_preflight(path,configs,dataset):
    require(path is not None,'run all model/mode/PDE preflight first');_,p=read_payload(path,kind='diagnostic_bundle');r=p['preflight'];require(r['passed'] and r['source']==extension_identity() and r['dataset']==dataset.config.identity and set(c.identity for c in configs)<=set(r['configs']),'stale/configuration-incompatible preflight');return r


def throughput_batch(dataset,case,count,device,seed=42):
    """Performance-only query concatenation from exact prepared independent draws.

    No reference solve, synthetic labels or repeated/recycled query set. Main
    scientific sampling remains untouched. A short prefix is explicitly a
    performance control, not a claim about changing training distributions.
    """
    from ..smooth.bank import physical_design
    from ..smooth.data import tokens
    from ..geometric.data import Tokens
    from ..unified.config import SCHEMA as DATA_SCHEMA
    require(hasattr(dataset.cache,'_payload'),'throughput sweep requires a prepared continuous-stream dataset')
    events=dataset.cache._payload(case)['stream_events'];needed=(count+dataset.config.stream_queries-1)//dataset.config.stream_queries
    require(len(events)>=needed,'not enough distinct prepared encounters for this physical case/candidate; choose another prepared case or plan enough offline encounters')
    pieces=[]
    for cursor,slot,n in events[:needed]:
        sid=(DATA_SCHEMA,'supervised',seed,cursor,slot);design=physical_design(case,dataset.config.stream_queries,set_id=sid);clean=dataset.cache.evaluate(case,*design[:3]);pieces.append(tokens(case,len(clean),dataset.norms,coordinates=design,clean=clean)[0])
    query=Tokens(**{k:None if v is None else torch.cat([getattr(t,k) for t in pieces],dim=1)[:,:count] for k,v in vars(pieces[0]).items()})
    obs=Provider(dataset).token_set(case,128,observation=True,set_id=(DATA_SCHEMA,'fixed-evaluation',seed))[0];d=64 if case.pde=='wave' else 32
    mask=np.arange(d)<d//2;view=CaseView(case,FieldCondition.from_case(case,mask),'HYBRID')
    return Batch((view,),obs,query,dataset.norms).to(device)
