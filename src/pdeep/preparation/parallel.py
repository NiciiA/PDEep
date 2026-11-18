"""Bounded spawn processes and ONE verified dataset-manifest coordinator."""
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import os,json,time,queue,socket,statistics
import fcntl
from functools import wraps
import psutil
from ..core import require,digest
from ..artifact_io import atomic_json,read_payload,read_manifest,sha256
from ..smooth.definition import physics_identity,split_manifest
from ..smooth.certification import authorize
from ..unified.config import SCHEMA
from ..unified.dataset import inventory,case_from_key
from ..unified.streams import STREAM_SCHEMA
from .portable import Dataset,LAYOUT,read_index,commit,add_file,add_artifact,relative,verify_folder
from .planning import saved_plan
from .resources import budget
from .transfer import copy_file,sync_folder
from .workers import initialize,run_family,THREAD_VARS

def _single_coordinator(function):
    """Local macOS/Linux kernel lock, including initial manifest creation.

    The persistent lock inode is never unlinked: this avoids a replacement-file
    race between two launchers. Kernel ownership is released on process death.
    JSON owner metadata remains a human-readable/stale-process diagnostic.
    """
    @wraps(function)
    def locked(config,folder,certificate,**kwargs):
        root=Path(folder).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
        with (root/'.preparation.lock').open('a+') as guard:
            try:fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('this dataset already has an active preparation coordinator; do not launch another worker') from None
            return function(config,folder,certificate,**kwargs)
    return locked

def _publish(source,root,rel,*,equivalent_payload=False):
    source=Path(source);target=relative(root,rel);r=read_manifest(source)
    if target.exists():
        old=read_manifest(target)
        require(old['artifact_id']==r['artifact_id'] or equivalent_payload and old['payload_tree_identity']==r['payload_tree_identity'],'artifact collision; no overwrite: '+str(target))
        return target
    for name,info in r['files'].items():copy_file(source.parent/name,target.parent/name,info)
    copy_file(source,target,{'sha256':sha256(source),'size_bytes':source.stat().st_size});return target

@_single_coordinator
def prepare_dataset(config,folder,certificate,*,manifest=None,progress=None,sync_to=None,max_cases=None,workers=0,block_cases=2,feasibility=False,reuse_from=None,parent_config=None,_selected_keys=None,_control=None,_fault_case=None):
    require(isinstance(block_cases,int) and block_cases>0,'block_cases must be a positive integer')
    require(isinstance(workers,int) and workers>=0,'workers must be0 (automatic) or a positive integer')
    require(max_cases is None or isinstance(max_cases,int) and max_cases>=0,'max_cases must be nonnegative')
    progress=progress or (lambda stage,**kw:print(stage,kw,flush=True));root=Path(folder).expanduser().resolve();path=root/'manifest.json' if manifest is None else Path(manifest).expanduser().resolve()
    if path.is_dir():path=path/'manifest.json'
    require(path.parent==root,'exact manifest must belong to selected folder');root.mkdir(parents=True,exist_ok=True)
    authorize(certificate,pdes=config.pdes,families=config.families,normalizer_identity=certificate['normalizers']['identity'])
    from .feasibility import is_configuration
    feasibility=feasibility or is_configuration(config)
    if path.exists():
        _,index=read_index(path);require(index['dataset_identity']==config.identity and index['certificate_id']==certificate['certification_id'],'resume configuration/producer runtime differs')
        require(bool(index.get('control_only'))==bool(_control),'diagnostic control data cannot be promoted to production references')
        require(not feasibility or not index['records'] or index.get('feasibility'),'existing manifest has a different draw policy; no silent conversion of completed cases')
        verify_folder(path) # Fail on corrupt saved work before doing any new solves.
        if index['status']=='complete':return Dataset(path,certificate,progress=progress)
        feasibility=bool(index.get('feasibility')) or feasibility;rows=index['cases']
    else:
        rows=inventory(config);index={'schema':SCHEMA,'stream_schema':STREAM_SCHEMA,'portable_layout':LAYOUT,'status':'partial','config':asdict(config),'dataset_identity':config.identity,'certificate_id':certificate['certification_id'],'normalizer_identity':certificate['normalizers']['identity'],'original_split_identity':split_manifest()['identity'],'cases':rows,'inventory_identity':digest(rows),'records':{},'assets':{},'qualification':{},'quarantine':{},'persistence':{},'files':{},'producer_certificate':'quality_reports/producer-certificate.json','preparation_profile':{},'reference_scope':'unchanged full CPU64 P4/CN/Gauss4; independent case columns'}
        atomic_json(root/index['producer_certificate'],certificate);add_file(index,root,root/index['producer_certificate']);commit(path,index)
    lease=root/'.preparation-coordinator.json';pid=os.getpid();host=socket.gethostname()
    if lease.exists():
        old=json.loads(lease.read_text());alive=False
        if old.get('host')==host:
            try:alive=psutil.Process(old['pid']).is_running() and abs(psutil.Process(old['pid']).create_time()-old['created'])<.01
            except psutil.Error:pass
        require(not alive,'this dataset already has an active preparation coordinator; do not launch another worker')
    atomic_json(lease,{'pid':pid,'host':host,'created':psutil.Process().create_time()})
    pool=None;old_env={k:os.environ.get(k) for k in THREAD_VARS};start=time.perf_counter();cache={};profile=index.get('preparation_profile',{});times=profile.setdefault('task_case_seconds',{});cpu_times=profile.setdefault('task_CPU_seconds',{});records=profile.setdefault('case_metrics',{});resources=None
    try:
        t=time.perf_counter()
        if feasibility:
            from .feasibility import save_plan,SCHEMA as F_SCHEMA
            plan,plan_path,reused=save_plan(config,rows,root,parent_manifest=reuse_from,parent_config=parent_config,progress=progress)
            index['feasibility']={'schema':F_SCHEMA,'parent_dataset_identity':__import__('pdeep.unified.config',fromlist=['DatasetConfig']).DatasetConfig(**plan['parent_config']).identity,'mapping_identity':plan['mapping_identity'],'sampling':'unique per-case parent encounter prefix; no replacement or recycling'}
        else:_,plan_path,reused=saved_plan(config,rows,root,progress=progress)
        add_artifact(index,root,plan_path);profile.setdefault('phase_seconds',{})['latest_plan_or_reload_wall_seconds']=time.perf_counter()-t
        index['preparation_profile']=profile
        if _control:index['control_only']=True
        commit(path,index)
        tasks={};lookup={r['case_id']:r for r in rows}
        for row in rows:
            cid=row['case_id'];key=tuple(row['key'])
            if cid in index['records'] or cid in index['quarantine']:continue
            if _selected_keys is not None and key not in set(map(tuple,_selected_keys)):continue
            ordinal=(key[3]-__import__('pdeep.unified.config',fromlist=['case_index']).case_index(key[1],0))//4
            # Recover a fully committed main artifact if the previous coordinator
            # died before its mutable index update. Never solve that case again.
            c=case_from_key(key,config.conditioning_h);stem=digest((c.manifest,config.identity,config.reference_execution_source,certificate['certification_id']));existing=root/'cases'/f'{stem}.json'
            if existing.exists():
                r,p=read_payload(existing,required=('qc','case','sampling_space'))
                require(r['case_id']==cid and r['dataset_identity']==config.identity and r['certificate_id']==certificate['certification_id'] and r['physics']==certificate['source_revision'] and bool(r.get('control_only'))==bool(_control) and p['case']==c.manifest and p['qc']['passed'],'committed case recovery failed')
                add_artifact(index,root,existing);space=relative(root,p['sampling_space']);add_artifact(index,root,space);index['records'][cid]=existing.relative_to(root).as_posix();index['assets'][space.stem]=space.relative_to(root).as_posix()
                if key[2]=='train' and ordinal<config.fine_spot_checks_per_task:
                    require(p['qc']['checks'].get('finer_spatial') is True,'committed case lacks required finer qualification')
                    task='/'.join(key[:2]);old=index['qualification'].get(task,{'passed':True,'cases':[]});index['qualification'][task]={'passed':True,'cases':list(dict.fromkeys([*old['cases'],cid]))}
                commit(path,index);continue
            tasks.setdefault(key[1],[]).append({'key':key,'case_id':cid,'finer':key[2]=='train' and ordinal<config.fine_spot_checks_per_task})
        # Fair bounded pilot: one round through tasks before a second case/task.
        if max_cases is not None:
            flattened=sorted([v for x in tasks.values() for v in x],key=lambda v:(v['key'][3]//4,('train','validation','test').index(v['key'][2]),config.families.index(v['key'][1]),config.pdes.index(v['key'][0])))[:max_cases]
            allowed={v['case_id'] for v in flattened};tasks={g:[v for v in x if v['case_id'] in allowed] for g,x in tasks.items()};tasks={g:x for g,x in tasks.items() if x}
        if not tasks:return Dataset(path,certificate,progress=progress)
        cert_for_budget=certificate
        if _control:
            cert_for_budget=json.loads(json.dumps(certificate))
            for g in tasks:cert_for_budget['mesh_records'][g]['production']['free_DOFs']=2000
        resources=budget(cert_for_budget,tuple(tasks),requested=workers,block_cases=block_cases,needs_finer=not _control and any(v['finer'] for group in tasks.values() for v in group));count=min(resources['workers'],len(tasks));resources['effective_geometry_lanes']=count;profile['resource_policy']=resources
        context=mp.get_context('spawn');messages=context.Queue(maxsize=max(2,2*count));stop=context.Event();peak=psutil.Process().memory_info().rss;active={};failed=[];blocked=[];verified_new=0;sync_status=None
        def snapshot(stage='Dataset progress',**extra):
            totals={};fail_counts={}
            active_tasks={v['details'].get('pde','')+'/'+v['family'] for v in active.values()}
            for cid in index['records']:
                k=lookup[cid]['key'];name='/'.join(k[:2]);totals[name]=totals.get(name,0)+1
            for item in index['quarantine'].values():fail_counts[item['task']]=fail_counts.get(item['task'],0)+1
            table=[]
            for p,g in config.tasks:
                name=p+'/'+g;n=totals.get(name,0);samples=times.get(name,[]);median=statistics.mean(samples[-32:]) if samples else None
                table.append({'task':name,'completed':n,'total':sum(config.counts),'state':'complete' if n==sum(config.counts) else 'running' if name in active_tasks else 'partial' if n else 'pending','failed':fail_counts.get(name,0),'seconds_per_case':median,'ETA_seconds':(sum(config.counts)-n)*median if median else None})
            whole=None
            if all(v['ETA_seconds'] is not None for v in table):
                lane={}
                for row in table:lane[row['task'].split('/')[1]]=lane.get(row['task'].split('/')[1],0.)+row['ETA_seconds']
                io_samples=[v['coordinator_wall_seconds'] for v in records.values() if 'coordinator_wall_seconds' in v]
                io=(sum(config.totals.values())-len(index['records']))*statistics.mean(io_samples[-32:]) if io_samples else 0.
                whole=max(sum(v['ETA_seconds'] for v in table)/max(1,count),max(lane.values()),io)
            progress(stage,physical_cases_total=len(index['records']),target_total=sum(config.totals.values()),task_rows=table,active_workers=len(active),configured_workers=count,cases_per_second=verified_new/max(time.perf_counter()-start,1e-9),peak_aggregate_RSS_bytes=peak,failed_cases=len(index['quarantine']),execution_failures=len(failed),whole_dataset_ETA_seconds=whole,ETA_scope='observed per-task amortized means / configured concurrency; unknown until all task costs observed',**extra)
        def checkpoint():
            nonlocal sync_status
            profile.update(latest_session_seconds=time.perf_counter()-start,peak_aggregate_RSS_bytes=peak,execution_failures=failed,reuse_blockers=blocked);index['preparation_profile']=profile;commit(path,index)
            if sync_to:
                t=time.perf_counter()
                try:sync_status=sync_folder(path,sync_to,verified_cache=cache);index['persistence']={'local_status':'verified','last_sync_status':sync_status['transfer_status'],'dataset_status':sync_status['dataset_status'],'last_synced_cases':len(index['records'])}
                except Exception as error:index['persistence']={'local_status':'verified','last_sync_status':'failed','error':str(error)};snapshot('DATASET SYNC FAILED — local references retained',error=str(error))
                profile.setdefault('phase_seconds',{})['Drive_or_filesystem_sync']=profile['phase_seconds'].get('Drive_or_filesystem_sync',0.)+time.perf_counter()-t;commit(path,index)
        for name in THREAD_VARS:os.environ[name]='1'
        pool=ProcessPoolExecutor(max_workers=count,mp_context=context,initializer=initialize,initargs=(messages,stop))
        pending=list(tasks.items());futures={};submitted=set();done=set();checkpoint();snapshot('CPU reference workers starting',resource_policy=resources)
        def submit():
            while pending and len(futures)<count:
                family,entries=pending.pop(0);spec={'config':asdict(config),'family':family,'certificate':certificate,'scratch':str(root/'.workers'/family),'entries':entries,'plan':str(plan_path),'block_cases':block_cases,'state_memory_bytes':int(2*501*max(certificate['mesh_records'][g]['production']['free_DOFs'] for g in tasks)*8*block_cases+1),'reuse_from':str(reuse_from) if reuse_from else None,'source':digest({p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}),'_control':_control}
                spec.update(main_root=str(root),_fault_case=_fault_case,partial_cases={v['case_id']:index.get('partial_cases',{}).get(v['case_id']) for v in entries if index.get('partial_cases',{}).get(v['case_id'])})
                futures[pool.submit(run_family,spec)]=family;submitted.add(family)
        submit();last_render=0.
        while futures or done!=submitted:
            # Sample actual coordinator+worker RSS (shared library pages may be
            # counted more than once; this is not a fabricated PSS measurement).
            rss=psutil.Process().memory_info().rss
            # Known owned worker PIDs only; no machine-wide process discovery
            # (also works where macOS disallows enumerating unrelated PIDs).
            for process in (getattr(pool,'_processes',None) or {}).values():
                try:rss+=psutil.Process(process.pid).memory_info().rss
                except (psutil.Error,OSError):rss+=active.get(process.pid,{}).get('RSS_bytes',0)
            peak=max(peak,rss)
            try:msg=messages.get(timeout=.15)
            except queue.Empty:msg=None
            if msg:
                kind=msg['kind']
                if kind=='progress':active[msg['pid']]=msg
                elif kind=='family_done':done.add(msg['family'])
                elif kind=='journal':
                    cid=msg['case_id'];j=msg['index'];target=root/'.working'/cid
                    for name in j['shards']:
                        p=_publish(Path(msg['root'])/name,root,('.working/'+cid+'/'+name));add_artifact(index,root,p)
                    atomic_json(target/'progress.json',j);index.setdefault('partial_cases',{})[cid]=j;checkpoint()
                elif kind=='case':
                    coord_start=time.perf_counter();coord_cpu=time.process_time()
                    cid=msg['case_id'];r,payload=read_payload(msg['record'],required=('qc','case','labels','sampling_space'));require(r['case_id']==cid and r['dataset_identity']==config.identity and r['certificate_id']==certificate['certification_id'] and r['physics']==certificate['source_revision'] and r['execution_source']==config.reference_execution_source and payload['case']==case_from_key(msg['key'],config.conditioning_h).manifest and payload['qc']['passed'],'worker result identity/QC invalid')
                    space=_publish(msg['space'],root,payload['sampling_space'],equivalent_payload=True);record=_publish(msg['record'],root,'cases/'+Path(msg['record']).name)
                    add_artifact(index,root,space);add_artifact(index,root,record);index['records'][cid]=record.relative_to(root).as_posix();index['assets'][space.stem]=space.relative_to(root).as_posix()
                    index.get('partial_cases',{}).pop(cid,None)
                    for rel in tuple(index['files']):
                        if rel.startswith('.working/'+cid+'/'):index['files'].pop(rel)
                    task='/'.join(msg['key'][:2]);m=msg['metrics'];sample=m['case_processing_wall_seconds']+m['attributed_solve_wall_seconds'];cpu_sample=m['case_processing_CPU_seconds']+m['attributed_solve_CPU_seconds'];times.setdefault(task,[]).append(sample);cpu_times.setdefault(task,[]).append(cpu_sample);records[cid]=m
                    if msg['finer']:
                        old=index['qualification'].get(task,{'passed':True,'cases':[]});index['qualification'][task]={'passed':True,'cases':list(dict.fromkeys([*old['cases'],cid])),'policy':'unchanged independent finer controls'}
                    profile.setdefault('worker_operator_profiles',{})[msg['key'][1]]=msg.get('operator_profile',{})
                    verified_new+=1;checkpoint()
                    m['coordinator_wall_seconds']=time.perf_counter()-coord_start;m['coordinator_CPU_seconds']=time.process_time()-coord_cpu
                    times[task][-1]+=m['coordinator_wall_seconds'];cpu_times[task][-1]+=m['coordinator_CPU_seconds'];commit(path,index)
                    snapshot('Dataset case verified',case=cid,task=task,case_CPU_seconds=cpu_times[task][-1],case_wall_seconds=times[task][-1])
                    # Local index is now authoritative. Remove only duplicate
                    # private case artifacts/journals, retaining reusable assets.
                    private=Path(msg['record']);require(private.resolve().is_relative_to((root/'.workers').resolve()),'worker path escapes private area')
                    meta=read_manifest(private)
                    for name in meta['files']:(private.parent/name).unlink()
                    private.unlink()
                    from .labels import Journal
                    for work_root in (private.parent.parent,root):
                        journal=Journal(work_root,case_from_key(msg['key'],config.conditioning_h),config)
                        if journal.path.exists():journal.clean()
                elif kind=='blocked':blocked.append(msg);checkpoint();snapshot('Completed reference reuse blocked — no regeneration',error=msg['error'])
                elif kind=='failure':
                    if msg['category']=='QC_or_serialization':index['quarantine'][msg['case_id']]={'task':'/'.join(msg['key'][:2]),'key':msg['key'],'error':msg['error']}
                    else:failed.append(msg)
                    checkpoint();snapshot('Reference worker failure; verified work retained',error=msg['error'])
            for future in list(futures):
                if future.done():
                    family=futures.pop(future)
                    try:profile.setdefault('worker_profiles',{})[family]=future.result()
                    except Exception as error:failed.append({'family':family,'error':repr(error),'unfinished_only':True});done.add(family);checkpoint()
                    for pid in tuple(active):
                        if active[pid].get('family')==family:active.pop(pid)
            if any(isinstance(getattr(f,'_exception',None),BrokenProcessPool) for f in futures):raise RuntimeError('worker process crashed; resume the exact manifest to recover committed private artifacts')
            submit()
            if time.perf_counter()-last_render>5:snapshot();last_render=time.perf_counter()
        # Drain messages published just before a future's completion. Queue's
        # producer feeder is joined by process shutdown; next resume also recovers
        # private committed records if a disconnect precedes coordinator commit.
        pool.shutdown(wait=True);pool=None
        complete=len(index['records'])==sum(config.totals.values()) and not index['quarantine'] and not _control
        if complete:verify_folder(path)
        index['status']='complete' if complete else 'partial';checkpoint();snapshot('Reference preparation finished')
        return Dataset(path,certificate,progress=progress)
    finally:
        if pool is not None:
            try:stop.set()
            except UnboundLocalError:pass
            for process in list((getattr(pool,'_processes',None) or {}).values()):process.terminate()
            pool.shutdown(wait=True,cancel_futures=True)
        for name,value in old_env.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
        if lease.exists():lease.unlink()
