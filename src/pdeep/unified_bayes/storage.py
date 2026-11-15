"""Explicit verified artifact registry, shared persistence; no directory discovery."""
from pathlib import Path
import json,warnings
from ..artifact_io import atomic_json,sha256
from ..core import require,digest
from ..unified.persistence import policy_at,sync,copy_verified


class ArtifactStore:
    def __init__(self,root,policy):
        self.root=Path(root);self.policy=policy_at(policy,'baselines');self.path=self.root/'persistence.json'
        self.record=json.loads(self.path.read_text()) if self.path.exists() else {'schema':'G1-baseline-persistence-v1','artifacts':{},'indices':{}}
    def __call__(self,path):
        path=Path(path);relative=str(path.relative_to(self.root));status=sync(path,self.policy)
        self.record['artifacts'][relative]={'local_manifest':str(path),**status};self._save();return status
    def _save(self):
        atomic_json(self.path,self.record)
        if self.policy.drive_enabled:
            try:
                target=Path(self.policy.drive_root)/'persistence.json';atomic_json(target,self.record);require(digest(json.loads(target.read_text()))==digest(self.record),'baseline registry copy differs')
            except Exception as error:warnings.warn('BASELINE DRIVE REGISTRY FAILED; valid local artifacts retained: '+str(error),RuntimeWarning)
    def index(self,path):
        path=Path(path);rel=str(path.relative_to(self.root));entry={'local_path':str(path),'sha256':sha256(path)}
        if self.policy.drive_enabled:
            try:
                target=Path(self.policy.drive_root)/'indices'/rel;atomic_json(target,json.loads(path.read_text()));require(sha256(target)==sha256(path),'baseline index copy differs');entry['Drive_path']=str(target)
            except Exception as error:entry['Drive_error']=str(error);warnings.warn('BASELINE DRIVE INDEX FAILED; local index retained: '+str(error),RuntimeWarning)
        self.record['indices'][rel]=entry;self._save()


def restore_index(index_path,registry_path,local_root):
    """Restore exactly one known index plus its listed verified artifacts.

    Operator assets can be restored individually using restore_artifacts(). No
    recursive lookup is permitted. Existing immutable local artifacts are retained.
    """
    index_path=Path(index_path);registry=json.loads(Path(registry_path).read_text());entry=next((v for v in registry['indices'].values() if str(index_path) in (v.get('local_path'),v.get('Drive_path'))),None)
    require(entry and sha256(index_path)==entry['sha256'],'index missing from explicit registry or checksum differs')
    index=json.loads(index_path.read_text());root=Path(local_root);paths=[]
    for original in index['case_results']:
        matches=[(k,v) for k,v in registry['artifacts'].items() if v['local_manifest']==original];require(len(matches)==1,'result not listed in registry')
        rel,saved=matches[0];source=original if Path(original).exists() else saved.get('Drive_manifest');require(source is not None,'no verified local/Drive result');paths.append(str(copy_verified(source,root/rel)))
    index['case_results']=paths;target=root/'restored-indices'/index_path.name;atomic_json(target,index);return target


def restore_artifacts(registry_path,relative_paths,local_root):
    registry=json.loads(Path(registry_path).read_text());out=[]
    for rel in relative_paths:
        require(rel in registry['artifacts'],'artifact not in explicit registry');saved=registry['artifacts'][rel];source=saved['local_manifest'] if Path(saved['local_manifest']).is_file() else saved.get('Drive_manifest');require(source is not None,'no verified source');out.append(str(copy_verified(source,Path(local_root)/rel)))
    return out
