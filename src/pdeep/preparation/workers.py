"""Importable spawn workers. Private artifact folders; no shared-index writes."""
import os,time,traceback
from pathlib import Path

QUEUE=None;STOP=None
THREAD_VARS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')

def initialize(queue,stop):
    global QUEUE,STOP
    QUEUE=queue;STOP=stop
    for key in THREAD_VARS:os.environ[key]='1'
    import torch
    torch.set_num_threads(1)

def _run_family(spec):
    import json,numpy as np
    from ..core import digest,require
    from ..artifact_io import read_payload,write_artifact,read_manifest,atomic_json
    from ..smooth.data import Case
    from ..smooth.certification import normalizers
    from ..smooth.definition import physics_identity
    from ..unified.config import DatasetConfig,SCHEMA
    from ..unified.dataset import quality_check_full
    from ..unified.streams import save_sampling_space,STREAM_SCHEMA
    from .engine import References
    from .labels import Journal,prepare_labels
    from .numerics import FastReference
    from .evolution import Evolution
    from .reuse import reuse_case,ReuseBlocked
    from .transfer import copy_file
    import psutil
    root=Path(spec['scratch']);root.mkdir(parents=True,exist_ok=True);config=DatasetConfig(**spec['config']);cert=spec['certificate'];norms=normalizers(cert)
    owner=root/'owner.json';import socket
    if owner.exists():
        old=json.loads(owner.read_text());alive=False
        if old.get('host')==socket.gethostname():
            try:alive=psutil.Process(old['pid']).is_running() and abs(psutil.Process(old['pid']).create_time()-old['created'])<.01
            except psutil.Error:pass
        require(not alive or old['pid']==os.getpid(),'private reference lane is still active')
        if not alive:
            # Only this dead worker's LOCAL private transaction locks. No Drive
            # discovery and no removal of committed artifacts or numerical data.
            for lock in root.rglob('*.lock'):lock.unlink()
    atomic_json(owner,{'pid':os.getpid(),'host':socket.gethostname(),'created':psutil.Process().create_time()})
    _,plan=read_payload(spec['plan'],kind='diagnostic_bundle',required=('events',));events=plan['events'];profile={};refs=References(root,profile);ops={};physics=physics_identity();completed=0
    def report(stage,**details):
        if STOP.is_set():raise InterruptedError('coordinator requested stop; committed cases preserved')
        details.setdefault('pde',pde);details.setdefault('geometry',spec['family'])
        QUEUE.put({'kind':'progress','family':spec['family'],'stage':stage,'details':details,'pid':os.getpid(),'RSS_bytes':psutil.Process().memory_info().rss})
    started_wall=time.perf_counter();started_cpu=time.process_time();entries=spec['entries'];pending=list(entries)
    while pending and not STOP.is_set():
        first=pending[0];pde=first['key'][0];group=[v for v in pending if v['key'][0]==pde][:spec['block_cases']]
        for v in group:pending.remove(v)
        fresh=[];results={};reuse_cost={};build_wall=build_cpu=0.
        for entry in group:
            c=Case(entry['key'][1],pde,entry['key'][2],entry['key'][3],config.conditioning_h);stem=digest((c.manifest,config.identity,config.reference_execution_source,cert['certification_id']));record=root/'cases'/f'{stem}.json'
            reuse_start=time.perf_counter();reuse_cpu=time.process_time()
            try:
                if record.exists():
                    meta,payload=read_payload(record);require(meta['case_id']==c.case_id and meta['dataset_identity']==config.identity and meta['certificate_id']==cert['certification_id'] and meta['physics']==physics and meta['execution_source']==config.reference_execution_source and bool(meta.get('control_only'))==bool(spec.get('_control')) and payload['case']==c.manifest and payload['qc']['passed'],'private committed record invalid');results[c.case_id]=(record,payload,root/payload['sampling_space'],True)
                elif spec.get('reuse_from'):
                    reused=reuse_case(spec['reuse_from'],c,config,events[c.case_id],record,root,cert,finer=entry['finer'])
                    if reused:results[c.case_id]=(*reused,True)
                    else:fresh.append((entry,c,record))
                else:fresh.append((entry,c,record))
                if c.case_id in results:reuse_cost[c.case_id]=(time.perf_counter()-reuse_start,time.process_time()-reuse_cpu)
            except ReuseBlocked as error:
                QUEUE.put({'kind':'blocked','case_id':c.case_id,'key':entry['key'],'error':str(error)});continue
            except Exception as error:
                QUEUE.put({'kind':'failure','case_id':c.case_id,'key':entry['key'],'error':traceback.format_exc(),'category':'artifact_or_reuse'});continue
        fulls={}
        if fresh:
            try:
                report('Reference block setup',pde=pde,geometry=spec['family'],cases=len(fresh))
                control=spec.get('_control')
                if control:
                    from ..smooth.reference import Reference
                    from ..wave_fidelity.reference import Space
                    from ..smooth.definition import geometry
                    if refs.r is None:
                        r=object.__new__(Reference);r.family=spec['family'];r.h=control['h'];r.degree=control['degree'];r.space=Space(geometry(r.family,r.h),r.degree,speed=1.);r.factors={};refs.r=r;refs.family=r.family
                    else:r=refs.r
                else:r=refs.get(spec['family'])
                wall=time.perf_counter();cpu=time.process_time()
                if pde!='poisson':
                    if pde not in ops:
                        ops[pde]=Evolution(r,pde)
                        profile[pde+'_step_factorization_wall_seconds']=ops[pde].factorization_timing['wall_seconds']
                        profile[pde+'_step_factorization_CPU_seconds']=ops[pde].factorization_timing['CPU_seconds']
                    values=ops[pde].propagate([c for _,c,_ in fresh],progress=report,memory_bytes=spec['state_memory_bytes']);fulls={f.case.case_id:f for f in values};del values
                else:
                    basis=refs.poisson(fresh[0][1],{'certificate_id':cert['certification_id'],'config':spec['config'],'files':{}}) if not control else None
                    fulls={c.case_id:FastReference(c,reference_object=r,poisson_basis=basis) for _,c,_ in fresh}
                build_wall=time.perf_counter()-wall;build_cpu=time.process_time()-cpu
            except InterruptedError:break
            except Exception:
                for entry,c,_ in fresh:QUEUE.put({'kind':'failure','case_id':c.case_id,'key':entry['key'],'error':traceback.format_exc(),'category':'execution'})
                fresh=[]
        for entry in group:
            c=Case(entry['key'][1],pde,entry['key'][2],entry['key'][3],config.conditioning_h);wall=time.perf_counter();cpu=time.process_time();metrics={}
            try:
                if c.case_id in results:record,payload,space,reused=results.pop(c.case_id)
                elif c.case_id in fulls:
                    full=fulls.pop(c.case_id);journal=Journal(root,c,config)
                    saved=spec.get('partial_cases',{}).get(c.case_id)
                    if saved and not journal.path.exists():
                        require(saved['identity']==journal.identity,'restored worker journal differs');journal.index=saved
                        for name in saved['shards']:
                            src=Path(spec['main_root'])/'.working'/c.case_id/name;meta=read_manifest(src)
                            for fn,info in meta['files'].items():copy_file(src.parent/fn,journal.root/fn,info)
                            from ..artifact_io import sha256
                            copy_file(src,journal.root/name,{'sha256':sha256(src),'size_bytes':src.stat().st_size})
                        atomic_json(journal.path,journal.index)
                    qc=journal.index['qc']
                    if qc is None:
                        t=time.perf_counter();qc=quality_check_full(c,full,norms,config,finer=entry['finer'] and not spec.get('_control'));metrics['validation_wall_seconds']=time.perf_counter()-t
                        require(qc['passed'],'per-case reference QC failed: '+str(qc['checks']));journal.qc(qc)
                    def label_report(stage,**details):
                        report(stage,**details)
                        if details.get('batches_completed',0)%4==0:QUEUE.put({'kind':'journal','case_id':c.case_id,'root':str(journal.root),'index':{**journal.index,'shards':list(journal.index['shards'])}})
                    labels,complete=prepare_labels(c,full,config,events[c.case_id],journal=journal,progress=label_report,profile=metrics)
                    require(complete,'incomplete worker labels');space=save_sampling_space(full,root/'reference_cache',config.reference_execution_source)
                    record=root/'cases'/(digest((c.manifest,config.identity,config.reference_execution_source,cert['certification_id']))+'.json')
                    meta={'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-unified-cases','scientific_schema':SCHEMA,'stream_schema':STREAM_SCHEMA,'case_id':c.case_id,'dataset_identity':config.identity,'normalizer_identity':norms.identity,'certificate_id':cert['certification_id'],'physics':physics,'execution_source':config.reference_execution_source,'execution_optimization_source':spec['source'],'preparation_implementation':'spawn-full-block-FEM-v1','control_only':bool(spec.get('_control'))}
                    payload={'case':c.manifest,'qc':qc,'labels':labels,'stream_events':events[c.case_id],'sampling_space':space.relative_to(root).as_posix()}
                    t=time.perf_counter();record=write_artifact(record.parent,record.stem,meta,payload,required=tuple(payload));metrics['serialization_wall_seconds']=time.perf_counter()-t
                    del full,labels;reused=False
                else:continue
                rw,rc=reuse_cost.get(c.case_id,(0.,0.))
                metrics.update(case_processing_wall_seconds=time.perf_counter()-wall+rw,case_processing_CPU_seconds=time.process_time()-cpu+rc,attributed_solve_wall_seconds=0. if reused else build_wall/max(1,len(fresh)),attributed_solve_CPU_seconds=0. if reused else build_cpu/max(1,len(fresh)),reused_completed_case=reused)
                if spec.get('_fault_case')==c.case_id:os._exit(71) # isolated crash-recovery test only
                QUEUE.put({'kind':'case','case_id':c.case_id,'key':entry['key'],'record':str(record),'space':str(space),'finer':entry['finer'] and not spec.get('_control'),'metrics':metrics,'operator_profile':dict(profile),'pid':os.getpid(),'RSS_bytes':psutil.Process().memory_info().rss});completed+=1
                del payload
            except InterruptedError:return {'interrupted':True,'completed':completed}
            except Exception:QUEUE.put({'kind':'failure','case_id':c.case_id,'key':entry['key'],'category':'QC_or_serialization','error':traceback.format_exc()})
        del fulls
    import resource,sys
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return {'completed':completed,'pid':os.getpid(),'operator_profile':profile,'wall_seconds':time.perf_counter()-started_wall,'CPU_seconds':time.process_time()-started_cpu,'peak_self_RSS_bytes':peak,'BLAS_thread_environment':{k:os.environ[k] for k in THREAD_VARS}}

def run_family(spec):
    try:
        result=_run_family(spec);QUEUE.put({'kind':'family_done','family':spec['family']});return result
    except BaseException:
        QUEUE.put({'kind':'family_done','family':spec['family']});raise
