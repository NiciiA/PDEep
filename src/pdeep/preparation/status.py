"""Read only exact manifests; distinguish recorded, audited and mirrored work."""
from pathlib import Path
import statistics,time
from ..artifact_io import read_manifest,read_payload,sha256
from ..core import require
from .portable import read_index,relative

def status(manifest,*,verify='index',mirror=None):
    start=time.perf_counter();path,index=read_index(manifest);config=__import__('pdeep.unified.config',fromlist=['DatasetConfig']).DatasetConfig(**index['config'])
    require(verify in ('index','checksums','payloads'),'unknown verification level')
    rows=[];totals={};checked=0;failures=[];profile=index.get('preparation_profile',{})
    for row in index['cases']:
        if row['case_id'] in index['records']:totals['/'.join(row['key'][:2])]=totals.get('/'.join(row['key'][:2]),0)+1
    if verify!='index':
        for cid,rel in index['records'].items():
            try:
                artifact=relative(path.parent,rel);expected=index['files'][rel]
                require(artifact.stat().st_size==expected['size_bytes'] and sha256(artifact)==expected['sha256'],'case manifest differs from dataset inventory')
                if verify=='payloads':
                    r,p=read_payload(artifact,kind='diagnostic_bundle',required=('qc','case','labels'));require(p['qc']['passed'] and p['case']['physical_case_id']==cid,'case payload/QC mismatch')
                else:r=read_manifest(artifact,kind='diagnostic_bundle')
                for name,info in r['files'].items():require(index['files'].get((artifact.parent/name).relative_to(path.parent).as_posix())==info,'case payload differs from dataset inventory')
                require(r['case_id']==cid and r['dataset_identity']==index['dataset_identity'],'case identity mismatch');checked+=1
            except Exception as error:failures.append({'case_id':cid,'error':str(error)})
    for task in config.tasks:
        name='/'.join(task);n=totals.get(name,0);samples=profile.get('task_case_seconds',{}).get(name,[]);cpu=profile.get('task_CPU_seconds',{}).get(name,[])
        median=statistics.median(samples[-32:]) if samples else None
        rows.append({'task':name,'completed':n,'total':sum(config.counts),'state':'complete' if n==sum(config.counts) else 'partial' if n else 'pending','median_wall_seconds_per_case':median,'median_CPU_seconds_per_case':statistics.median(cpu[-32:]) if cpu else None,'task_ETA_seconds':(sum(config.counts)-n)*median if median else None})
    mirrored=None
    if mirror:
        _,remote=read_index(mirror);require(remote['dataset_identity']==index['dataset_identity'],'mirror dataset identity mismatch');mirrored=sum(remote['records'].get(k)==v and remote['files'].get(v)==index['files'][v] for k,v in index['records'].items())
    phases=dict(profile.get('phase_seconds',{}))
    for metric in profile.get('case_metrics',{}).values():
        for key,value in metric.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):phases[key]=phases.get(key,0.)+value
    operators=profile.get('worker_operator_profiles',{}) or {g:v.get('operator_profile',{}) for g,v in profile.get('worker_profiles',{}).items()}
    return {'manifest':str(path),'manifest_identity':index['manifest_identity'],'recorded_completed':len(index['records']),'independently_verified_completed':checked if verify!='index' else None,'verification_level':verify,'verification_failures':failures,'total_cases':sum(config.totals.values()),'remaining':sum(config.totals.values())-len(index['records']),'per_task':rows,'phase_seconds':phases,'phase_scope':'recorded case metrics; nested subsets are not additive; cold operator profiles are latest worker per geometry','latest_worker_operator_seconds':operators,'persistence':index.get('persistence',{}),'mirror_indexed_cases':mirrored,'Drive_scope':'last recorded sync status; not an independent remote audit unless exact mirror supplied','seconds':time.perf_counter()-start}
