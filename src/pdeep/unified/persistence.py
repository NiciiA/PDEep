"""Known-path verified copies only. No glob/rglob/checkpoint discovery."""
from dataclasses import replace
from pathlib import Path
import json, os, shutil, tempfile, warnings
from ..artifact_drive import PersistencePolicy, sync_verified
from ..artifact_io import read_manifest, atomic_json
from ..core import digest,require


def policy_at(policy,*parts):
    if not policy.drive_enabled:return policy
    return replace(policy,drive_root=str(Path(policy.drive_root or policy.drive_mount)/'G1_unified'/Path(*parts)))


def sync(path,policy):
    status=sync_verified(path,policy)
    if status['Drive_status']=='failed':warnings.warn('DRIVE SYNC FAILED: valid local artifact retained; local Colab storage is temporary. '+str(status['error']),RuntimeWarning)
    return status


def copy_verified(source,destination):
    source=Path(source);destination=Path(destination);r=read_manifest(source);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():require(read_manifest(destination)['artifact_id']==r['artifact_id'],'destination artifact collision');return destination
    with tempfile.TemporaryDirectory(prefix='.restoring-',dir=destination.parent) as tmp:
        tmp=Path(tmp)
        for name in (*r['files'],source.name):shutil.copyfile(source.parent/name,tmp/name)
        read_manifest(tmp/source.name)
        for name in r['files']:
            target=destination.parent/name
            if target.exists():
                from ..artifact_io import sha256
                require(sha256(target)==r['files'][name]['sha256'],'payload collision on restore')
            else:os.link(tmp/name,target)
        os.link(tmp/source.name,destination)
    read_manifest(destination);return destination


class DatasetPersistence:
    def __init__(self,root,dataset_id,policy):self.root=Path(root);self.dataset_id=dataset_id;self.policy=policy_at(policy,'datasets',dataset_id)
    def __call__(self,path,*,index_only=False):
        if not index_only:return sync(path,self.policy)
        if not self.policy.drive_enabled:return {'Drive_status':'disabled'}
        target=Path(self.policy.drive_root)/'index.json'
        try:
            data=json.loads(Path(path).read_text());atomic_json(target,data)
            require(digest(json.loads(target.read_text()))==digest(data),'dataset index copy verification failed')
            return {'Drive_status':'completed','Drive_index':str(target),'Drive_verified':True}
        except Exception as error:
            warnings.warn('DRIVE DATASET INDEX SYNC FAILED; valid local cache retained: '+str(error),RuntimeWarning)
            return {'Drive_status':'failed','error':str(error)}


def restore_dataset(index_path,local_root,*,case_ids=None):
    """Explicit known Drive index; only requested case records + their task assets.

    Selection may be expanded later. Restored local index retains the entire target
    inventory, but readiness gates require every case needed by the requested split.
    """
    from .config import DatasetConfig
    index_path=Path(index_path);index=json.loads(index_path.read_text());cfg=DatasetConfig(**index['config'])
    require(index['dataset_identity']==cfg.identity,'dataset index identity mismatch')
    root=Path(local_root);requested=set(index['records']) if case_ids is None else set(case_ids)
    require(requested<=set(index['records']),'requested case unprepared/quarantined in explicit index')
    tasks={tuple(r['key'][:2]) for r in index['cases'] if r['case_id'] in requested}
    paths={index['records'][k] for k in requested}
    # Asset metadata identifies geometry/PDE without scanning any directory.
    for rel in index['assets'].values():
        saved=next((v for k,v in index['persistence'].items() if k.endswith('/'+rel)),None)
        require(saved and saved.get('Drive_verified'),'missing verified persisted asset: '+rel)
        record=read_manifest(saved['Drive_manifest'])
        if (record['pde'],record['family']) in tasks or record['pde']=='shared' and record['family'] in {g for _,g in tasks}:paths.add(rel)
    for rel in sorted(paths):
        saved=next((v for k,v in index['persistence'].items() if k.endswith('/'+rel)),None)
        require(saved and saved.get('Drive_verified'),'missing verified persisted case: '+rel)
        copy_verified(saved['Drive_manifest'],root/rel)
    target=root/'datasets'/cfg.identity/'index.json';atomic_json(target,index);return target
