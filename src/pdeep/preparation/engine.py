"""Incremental local CPU64 preparation; no neural construction or training."""
from dataclasses import asdict
from pathlib import Path
from contextlib import contextmanager
import cProfile,pstats,json,time,statistics
import numpy as np
import torch
from ..core import require,digest
from ..artifact_io import atomic_json,read_payload,write_artifact,read_manifest,sha256
from ..smooth.data import Case
from ..smooth.definition import POLICY,physics_identity,split_manifest
from ..smooth.certification import authorize,normalizers
from ..smooth.reference import Reference
from ..smooth.resolution import resolve_reference
from ..unified.config import SCHEMA,case_index
from ..unified.dataset import inventory,quality_check_full
from ..unified.streams import STREAM_SCHEMA,save_sampling_space
from .portable import Dataset,LAYOUT,read_index,commit,add_file,add_artifact,relative,compatible_certificates,verify_folder
from .planning import saved_plan
from .numerics import FastReference,PoissonBasis
from .labels import Journal,prepare_labels
from .transfer import sync_folder

@contextmanager
def timed(profile,name):
    start=time.perf_counter();cpu=time.process_time()
    try:yield
    finally:
        profile[name]=profile.get(name,0.)+time.perf_counter()-start
        profile['CPU/'+name]=profile.get('CPU/'+name,0.)+time.process_time()-cpu

class References:
    """One family resident; share original assembled operators and direct LUs."""
    def __init__(self,folder,profile):self.folder=Path(folder);self.profile=profile;self.family=None;self.r=None;self.basis=None
    def get(self,family):
        if self.family==family:return self.r
        self.r=None;self.basis=None
        with timed(self.profile,'mesh_loading_generation_validation'):resolve_reference(family)
        cpu=cProfile.Profile();start=time.perf_counter();cpu_start=time.process_time();cpu.enable()
        try:self.r=Reference(family,POLICY['reference_h'],POLICY['reference_degree'])
        finally:cpu.disable()
        seconds=time.perf_counter()-start;stats=pstats.Stats(cpu)
        factor=sum(v[3] for k,v in stats.stats.items() if k[2]=='splu')
        self.profile['operator_assembly']=self.profile.get('operator_assembly',0.)+seconds-factor
        self.profile['mass_factorization']=self.profile.get('mass_factorization',0.)+factor
        self.profile['CPU/operator_assembly_and_mass_factorization']=self.profile.get('CPU/operator_assembly_and_mass_factorization',0.)+time.process_time()-cpu_start
        self.family=family;return self.r
    def poisson(self,case,index):
        if self.basis is not None:return self.basis
        r=self.get(case.family);field=case.fields['f'];key=digest((r.space.identity,field.centers.tolist(),field.radius,field.amplitude,index['certificate_id'],index['config']['reference_execution_source'],'full-32-Poisson-response-v1'))
        path=self.folder/'reference_cache/poisson'/f'{key}.json'
        with timed(self.profile,'source_basis_response_setup_or_reload'):
            if path.exists():
                meta,p=read_payload(path,kind='diagnostic_bundle',required=('response',));require(meta['space_identity']==r.space.identity and meta['basis_key']==key,'Poisson response cache mismatch')
                b=object.__new__(PoissonBasis);b.reference=r;b.signature=(field.centers.copy(),field.radius,field.geometry_id,field.amplitude);b.response=p['response'].numpy();b.bytes=b.response.nbytes;b.seconds=0.
                require(b.response.shape==(len(r.space.free),32) and np.isfinite(b.response).all(),'invalid source responses')
            else:
                b=PoissonBasis(r,field)
                write_artifact(path.parent,path.stem,{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-full-Poisson-response','space_identity':r.space.identity,'basis_key':key,'scope':'all32 full P4 source responses; original quadrature8; independent per-case physical-load residual checked'},{'response':torch.from_numpy(b.response)},required=('response',))
        add_artifact(index,self.folder,path);self.basis=b;return b

def prepare_dataset(config,folder,certificate,*,manifest=None,progress=None,sync_to=None,max_cases=None,max_label_batches=None,reuse_poisson_basis=True):
    """Explicit manifest resume; complete records and partial label shards reused.

    The folder is the complete portable dataset, unlike the legacy global root.
    Continuation generation needs the SAME producer numerical runtime/certificate;
    consuming complete bytes supports independently qualified Mac/Linux runtimes.
    """
    progress=progress or (lambda stage,**kw:print(stage,kw,flush=True));folder=Path(folder).expanduser();path=folder/'manifest.json' if manifest is None else Path(manifest).expanduser()
    if path.is_dir():path=path/'manifest.json'
    require(path.parent.resolve()==folder.resolve(),'explicit manifest must be inside the selected dataset folder')
    profile={};start=time.perf_counter();new_cases=0;sync_cache={};cpu=cProfile.Profile();cpu.enable()
    try:
        with timed(profile,'certificate_validation_including_mesh_inputs'):authorize(certificate,pdes=config.pdes,families=config.families,normalizer_identity=certificate['normalizers']['identity'])
    finally:cpu.disable()
    profile['SUBSET_geometry_loading_during_authorization']=sum(v[3] for k,v in pstats.Stats(cpu).stats.items() if k[2]=='geometry' and k[0].endswith('smooth/definition.py'))
    norms=normalizers(certificate)
    if path.exists():
        _,index=read_index(path);require(index['dataset_identity']==config.identity,'resume dataset configuration mismatch')
        if index['status']=='complete':return Dataset(path,certificate,progress=progress)
        require(index['certificate_id']==certificate['certification_id'],'generation continuation requires original producer certificate/runtime; verified bytes may be loaded/transferred across qualified environments without solving')
        rows=index['cases']
    else:
        with timed(profile,'physical_inventory'):rows=inventory(config)
        index={'schema':SCHEMA,'stream_schema':STREAM_SCHEMA,'portable_layout':LAYOUT,'status':'partial','config':asdict(config),'dataset_identity':config.identity,
            'certificate_id':certificate['certification_id'],'normalizer_identity':norms.identity,'original_split_identity':split_manifest()['identity'],
            'cases':rows,'inventory_identity':digest(rows),'records':{},'assets':{},'qualification':{},'quarantine':{},'persistence':{},'files':{},
            'producer_certificate':'quality_reports/producer-certificate.json','preparation_profile':{},'reference_scope':'original full P4 states and fixed-grid Hermite; exact continuous draws; no compression or change to physical distribution'}
        certpath=folder/index['producer_certificate'];atomic_json(certpath,certificate);add_file(index,folder,certpath);commit(path,index)
    with timed(profile,'encounter_planning_or_reload'):events,plan,reused=saved_plan(config,rows,folder,progress=progress)
    add_artifact(index,folder,plan);commit(path,index);refs=References(folder,profile)
    previous=index.get('preparation_profile',{});task_times={k:list(v) for k,v in previous.get('task_case_seconds',{}).items()};prior_phases=previous.get('phase_seconds',{})
    physics=physics_identity();counts={}
    optimization_source=digest({p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    for row in rows:
        if row['case_id'] in index['records']:counts['/'.join(row['key'][:2])]=counts.get('/'.join(row['key'][:2]),0)+1
    def checkpoint():
        index['preparation_profile']={'phase_seconds':{k:profile.get(k,0.)+prior_phases.get(k,0.) for k in set(profile)|set(prior_phases)},'task_case_seconds':task_times,'latest_session_seconds':time.perf_counter()-start,'encounter_plan_reused':reused}
        with timed(profile,'manifest_serialization'):commit(path,index)
        if sync_to:
            try:
                with timed(profile,'Drive_or_filesystem_sync'):status=sync_folder(path,sync_to,progress=progress,verified_cache=sync_cache)
                index['persistence']={'local_status':'verified','last_sync_status':status['transfer_status'],'dataset_status':status['dataset_status'],'destination_kind':'explicit filesystem folder','last_sync_seconds':status['seconds']}
            except Exception as error:
                index['persistence']={'local_status':'verified','last_sync_status':'failed','error':str(error)};progress('DATASET SYNC FAILED — valid local work retained',error=str(error),local_manifest=str(path))
            index['preparation_profile']['phase_seconds']={k:profile.get(k,0.)+prior_phases.get(k,0.) for k in set(profile)|set(prior_phases)}
            with timed(profile,'manifest_serialization'):commit(path,index)
    def publish_journal(c,journal):
        index.setdefault('partial_cases',{})[c.case_id]=dict(journal.index)
        for rel in journal.index['shards']:add_artifact(index,folder,journal.root/rel)
        checkpoint()
    for g in config.families:
        for p in config.pdes:
            task='/'.join((p,g));required=sum(config.counts);task_times.setdefault(task,[])
            for split,n in zip(('train','validation','test'),config.counts):
                for ordinal in range(n):
                    c=Case(g,p,split,case_index(g,ordinal),config.conditioning_h)
                    if c.case_id in index['records']:continue
                    if c.case_id in index['quarantine']:continue
                    if max_cases is not None and new_cases>=max_cases:checkpoint();return Dataset(path,certificate,progress=progress)
                    case_start=time.perf_counter();finer=split=='train' and ordinal<config.fine_spot_checks_per_task;done=counts.get(task,0)
                    median=statistics.median(task_times[task][-32:]) if task_times[task] else None
                    known=[t for t in config.tasks if '/'.join(t) not in task_times or not task_times['/'.join(t)]]
                    progress('Dataset inspection',task=task,prepared=done,required=required,physical_cases_total=len(index['records']),target_total=sum(config.totals.values()),cases_per_second=1/median if median else None,task_ETA_seconds=(required-done)*median if median else None,whole_dataset_ETA_seconds=None if known else sum((sum(config.counts)-counts.get('/'.join(t),0))*statistics.median(task_times['/'.join(t)][-32:]) for t in config.tasks),ETA_scope='measured per-task medians; unmeasured tasks are unknown; initial finer/setup cost reported separately',elapsed_seconds=time.perf_counter()-start)
                    stem=digest((c.manifest,config.identity,config.reference_execution_source,certificate['certification_id']));record=folder/'cases'/f'{stem}.json';journal=Journal(folder,c,config)
                    saved=index.get('partial_cases',{}).get(c.case_id)
                    if saved and not journal.path.exists():
                        require(saved['identity']==journal.identity,'restored journal identity mismatch');journal.index=saved;atomic_json(journal.path,saved)
                    try:
                        if record.exists():
                            with timed(profile,'artifact_validation'):
                                meta,payload=read_payload(record,kind='diagnostic_bundle',required=('case','qc','labels','sampling_space'))
                                require(meta['dataset_identity']==config.identity and meta['certificate_id']==index['certificate_id'] and payload['case']==c.manifest and payload['qc']['passed'],'committed case artifact incompatible')
                            qc=payload['qc'];space=relative(folder,payload['sampling_space']);progress('Reference preparation',task=task,case=c.case_id,operation='reuse committed verified case; zero solves')
                        else:
                            progress('Reference preparation',task=task,case=c.case_id,method='unchanged full P4/CN/Gauss4; exact vectorized coordinate evaluation')
                            r=refs.get(g);basis=refs.poisson(c,index) if p=='poisson' and reuse_poisson_basis else None
                            cpu=cProfile.Profile();cpu.enable()
                            try:
                                with timed(profile,'numerical_solution_with_initialization'):full=FastReference(c,reference_object=r,poisson_basis=basis)
                            finally:cpu.disable()
                            stats=pstats.Stats(cpu)
                            profile['SUBSET_step_factorization']=profile.get('SUBSET_step_factorization',0.)+sum(v[3] for k,v in stats.stats.items() if k[2]=='splu')
                            profile['SUBSET_direct_linear_solves']=profile.get('SUBSET_direct_linear_solves',0.)+sum(v[3] for k,v in stats.stats.items() if 'SuperLU' in k[2] and 'solve' in k[2])
                            qc=journal.index['qc']
                            if qc is None:
                                with timed(profile,'numerical_validation_finer' if finer else 'numerical_validation'):
                                    qc=quality_check_full(c,full,norms,config,finer=finer);require(qc['passed'],'per-case numerical QC failed: '+str(qc['checks']))
                                journal.qc(qc)
                            def label_progress(stage,**details):
                                progress(stage,task=task,**details)
                                if details.get('batches_completed',0)%4==0:publish_journal(c,journal)
                            labels,complete=prepare_labels(c,full,config,events[c.case_id],journal=journal,progress=label_progress,max_batches=max_label_batches,profile=profile)
                            if not complete:
                                del full,labels;publish_journal(c,journal);return Dataset(path,certificate,progress=progress)
                            with timed(profile,'serialization_and_verified_readback'):
                                space=save_sampling_space(full,folder/'reference_cache',config.reference_execution_source)
                                meta={'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-unified-cases','scientific_schema':SCHEMA,'stream_schema':STREAM_SCHEMA,'case_id':c.case_id,'dataset_identity':config.identity,'normalizer_identity':norms.identity,'certificate_id':index['certificate_id'],'physics':physics,'execution_source':config.reference_execution_source,'execution_optimization_source':optimization_source,'preparation_implementation':'P4-values-only-batched-full-source-superposition-v1','reference_preparation_seconds':full.preparation_seconds,'transient_full_bytes':full.bytes,'stored_label_bytes':sum(v.numel()*v.element_size() for v in labels.values())}
                                payload={'case':c.manifest,'qc':qc,'labels':labels,'stream_events':events[c.case_id],'sampling_space':space.relative_to(folder).as_posix()}
                                record=write_artifact(record.parent,record.stem,meta,payload,required=tuple(payload),idempotent=True)
                            del full,labels
                        with timed(profile,'artifact_validation'):
                            add_artifact(index,folder,record);add_artifact(index,folder,space)
                        index['records'][c.case_id]=record.relative_to(folder).as_posix();index['assets'][space.stem]=space.relative_to(folder).as_posix()
                        if c.case_id in index.get('partial_cases',{}):
                            index['partial_cases'].pop(c.case_id)
                            for rel in tuple(index['files']):
                                if rel.startswith('.working/'+c.case_id+'/'):index['files'].pop(rel)
                        if finer:
                            old=index['qualification'].get(task,{'passed':True,'cases':[]});index['qualification'][task]={'passed':old['passed'] and qc['passed'],'cases':list(dict.fromkeys([*old['cases'],c.case_id])),'policy':'unchanged independent P3/.01/.001 + fixed-space dt halving'}
                        counts[task]=done+1;task_times[task].append(time.perf_counter()-case_start);del payload
                    except Exception as error:
                        index['quarantine'][c.case_id]={'task':task,'key':(p,g,split,c.index),'error':f'{type(error).__name__}: {error}'};progress('Reference quarantine',task=task,case=c.case_id,error=str(error))
                    new_cases+=1;checkpoint()
                    if c.case_id in index['records']:
                        # ETA includes final serialization/upload, not just solver
                        # compute. Persist the completed end-to-end timing sample.
                        task_times[task][-1]=time.perf_counter()-case_start
                        with timed(profile,'manifest_serialization'):commit(path,index)
                        median=statistics.median(task_times[task][-32:])
                        progress('Dataset case verified',task=task,prepared=counts[task],required=required,case_wall_seconds=task_times[task][-1],cases_per_second=1/median,task_ETA_seconds=(required-counts[task])*median,whole_dataset_ETA_seconds=None if any(not task_times.get('/'.join(t)) for t in config.tasks) else sum((sum(config.counts)-counts.get('/'.join(t),0))*statistics.median(task_times['/'.join(t)][-32:]) for t in config.tasks))
                    if c.case_id in index['records']:journal.clean()
    complete=len(index['records'])==sum(config.totals.values()) and not index['quarantine'] and all(index['qualification'].get('/'.join(t),{}).get('passed') for t in config.tasks)
    if complete:verify_folder(path,progress=progress)
    index['status']='complete' if complete else 'partial'
    checkpoint();return Dataset(path,certificate,progress=progress,verify=not complete)
