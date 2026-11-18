"""Derive new-plan label artifacts from verified completed physical cases.

Never evaluate a donor's numerical solution again. Missing exact coordinate
labels are an explicit blocker, not permission to silently regenerate a case.
"""
from pathlib import Path
from ..core import require
from ..artifact_io import read_payload,read_manifest,write_artifact,sha256
from ..unified.streams import designs,coordinates_id
from .portable import read_index,relative
from .transfer import copy_file

class ReuseBlocked(ValueError):pass

def reuse_case(source,case,config,events,target_record,target_root,certificate,*,finer=False):
    source,index=read_index(source);rel=index['records'].get(case.case_id)
    require(not index.get('control_only'),'diagnostic controls cannot supply production reference labels')
    if rel is None:return None
    require(index['certificate_id']==certificate['certification_id'],'deriving new labels requires the original producer runtime; no relabeling foreign results')
    r,p=read_payload(relative(source.parent,rel),kind='diagnostic_bundle',required=('case','qc','labels','sampling_space'))
    require(r['case_id']==case.case_id and r['dataset_identity']==index['dataset_identity'] and r['physics']==certificate['source_revision'],'donor artifact provenance differs from its manifest/certificate')
    require(p['case']==case.manifest and p['qc']['passed'] and r['execution_source']==config.reference_execution_source and r['normalizer_identity']==certificate['normalizers']['identity'],'incompatible donor physical case/reference')
    if finer:
        task=case.pde+'/'+case.family
        require(case.case_id in index['qualification'].get(task,{}).get('cases',[]) and index['qualification'][task]['passed'],'donor lacks required stratified finer check')
    needed={coordinates_id(*d[:3]) for d in designs(case,config,events)};missing=needed-set(p['labels'])
    if missing:raise ReuseBlocked(f'{case.case_id}: completed donor lacks{len(missing)} exact designs; no case regeneration or recycled labels allowed')
    target_root=Path(target_root);space=relative(source.parent,p['sampling_space']);dst=relative(target_root,p['sampling_space']);sp=read_manifest(space)
    if not dst.exists():
        for name,expected in sp['files'].items():copy_file(space.parent/name,dst.parent/name,expected)
        copy_file(space,dst,{'sha256':sha256(space),'size_bytes':space.stat().st_size})
    else:require(read_manifest(dst)['payload_tree_identity']==sp['payload_tree_identity'],'sampling-space collision')
    meta={k:r[k] for k in ('artifact_type','architecture_id','run_id','scientific_schema','stream_schema','case_id','normalizer_identity','certificate_id','physics','execution_source')}
    meta.update(dataset_identity=config.identity,parents=[r['artifact_id']],preparation_implementation='verified-independent-parent-draw-subset-v1',reference_solves=0)
    payload={'case':p['case'],'qc':p['qc'],'labels':{k:p['labels'][k] for k in p['labels'] if k in needed},'stream_events':events,'sampling_space':p['sampling_space']}
    path=write_artifact(Path(target_record).parent,Path(target_record).stem,meta,payload,required=tuple(payload),idempotent=True)
    return path,payload,dst
