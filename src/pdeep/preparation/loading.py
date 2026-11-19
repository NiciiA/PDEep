"""Bounded LOCAL caches of verified immutable bytes, never numerical solutions.

Cold processes SHA256 every indexed file. Warm hits require the same expected
hash AND dev/inode/size/mtime/ctime fingerprint. Deserialization retains the
existing weights-only loader and tensor-tree verification. Changed files fail.
"""
from collections import OrderedDict
from pathlib import Path
import time
import numpy as np
import torch
from ..core import require,ValidationError
from ..artifact_io import read_payload,sha256
from ..unified.streams import PlannedReference,coordinates_id

_FILES=OrderedDict();_DATASETS=OrderedDict()

def stamp(path):
    s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def checked_file(path,expected,profile):
    path=Path(path);start=time.perf_counter()
    try:fingerprint=stamp(path)
    except OSError:raise ValidationError('dataset file missing/corrupt: '+str(path)) from None
    require(fingerprint[2]==expected['size_bytes'],'dataset file missing/corrupt (size): '+str(path))
    key=(str(path.resolve()),expected['sha256'])
    if _FILES.get(key)==fingerprint:
        _FILES.move_to_end(key);profile['verified_file_cache_hits']=profile.get('verified_file_cache_hits',0)+1
    else:
        require(sha256(path)==expected['sha256'],'dataset file missing/corrupt (checksum): '+str(path))
        _FILES[key]=fingerprint;profile['files_hashed']=profile.get('files_hashed',0)+1;profile['bytes_hashed']=profile.get('bytes_hashed',0)+fingerprint[2]
        while len(_FILES)>4096:_FILES.popitem(last=False)
    profile['file_verification_seconds']=profile.get('file_verification_seconds',0.)+time.perf_counter()-start
    return fingerprint

def verified_folder(path,profile,progress=None):
    from .portable import read_index,relative
    start=time.perf_counter();path,index=read_index(path);profile['manifest_seconds']=time.perf_counter()-start
    start=time.perf_counter();total=len(index['files'])
    for i,(rel,expected) in enumerate(index['files'].items()):
        checked_file(relative(path.parent,rel),expected,profile)
        if progress and (i%64==0 or i+1==total):
            seconds=time.perf_counter()-start;rate=(i+1)/max(seconds,1e-9)
            progress('Local dataset integrity',verified_files=i+1,total_files=total,seconds=seconds,files_per_second=rate,ETA_seconds=(total-i-1)/rate)
    return path,index

def tensor_bytes(payload):
    if torch.is_tensor(payload):return payload.numel()*payload.element_size()
    if isinstance(payload,dict):return sum(map(tensor_bytes,payload.values()))
    if isinstance(payload,(list,tuple)):return sum(map(tensor_bytes,payload))
    return 0

class MissingPreparedDraw(ValidationError):
    def __init__(self,dataset,case,x,t,q):
        key=coordinates_id(x,t,q)
        self.request={'schema':'G1-exact-missing-label-request-v1','dataset_identity':dataset.config.identity,'manifest_identity':dataset.index['manifest_identity'],'producer_certificate':dataset.index['certificate_id'],'case_key':[case.pde,case.family,case.split,case.index],'case_id':case.case_id,'coordinates_id':key,'points':len(x),'position':np.asarray(x,np.float64).tolist(),'physical_time':np.asarray(t,np.float64).tolist(),'quantity_id':np.asarray(q,np.int64).tolist()}
        super().__init__(f'Continuous draw not prepared: case={case.case_id}, PDE={case.pde}, geometry={case.family}, split={case.split}, points={len(x)}, coordinates_id={key}. No label substitution or online PDE solve is permitted. Export this exact missing-label request for explicit offline preparation in the original producer runtime.')

class LocalReference(PlannedReference):
    def __init__(self,dataset,*,max_cases=64,max_bytes=128*1024**2):
        super().__init__(dataset,max_cases=max_cases);self.max_bytes=max_bytes;self.resident_bytes=0;self.sizes={};self.fingerprints={}
    def _payload(self,c):return self.dataset._load_case(c)
    def put(self,cid,payload,fingerprints):
        self.drop(cid);n=tensor_bytes(payload)
        if n>self.max_bytes:return
        self.loaded[cid]=payload;self.sizes[cid]=n;self.fingerprints[cid]=fingerprints;self.resident_bytes+=n
        while len(self.loaded)>self.max_cases or self.resident_bytes>self.max_bytes:self.drop(next(iter(self.loaded)))
    def drop(self,cid):
        self.loaded.pop(cid,None);self.resident_bytes-=self.sizes.pop(cid,0);self.fingerprints.pop(cid,None)
    def evaluate(self,c,x,t,q,**kwargs):
        p=self._payload(c);key=coordinates_id(x,t,q)
        if key in p['labels']:y=p['labels'][key].numpy()
        else:
            rel=self.dataset.index.get('supplemental_labels',{}).get(c.case_id,{}).get(key)
            if rel is None:raise MissingPreparedDraw(self.dataset,c,x,t,q)
            checked_file(self.root/rel,self.dataset.index['files'][rel],self.dataset.load_profile)
            r,p=read_payload(self.root/rel,kind='diagnostic_bundle',required=('labels','position','physical_time','quantity_id','qc'))
            require(r['dataset_identity']==self.dataset.config.identity and r['case_id']==c.case_id and r['coordinates_id']==key and r['certificate_id']==self.dataset.index['certificate_id'] and r['execution_source']==self.dataset.config.reference_execution_source and r['normalizer_identity']==self.dataset.norms.identity and r['physics']==self.physics and p['qc']['passed'],'supplemental label provenance/QC mismatch')
            require(coordinates_id(p['position'].numpy(),p['physical_time'].numpy(),p['quantity_id'].numpy())==key,'supplement coordinates differ')
            y=p['labels'].numpy()
        require(len(y)==len(x),'label/coordinate size mismatch')
        return (y,np.zeros(len(y))) if kwargs.get('return_bound') else y

def load_dataset(path,certificate,*,progress=None):
    """Reuse one of at most two local datasets; recheck identity/file stamps."""
    from .portable import Dataset,compatible_certificates
    profile={};path,index=verified_folder(path,profile,progress);key=(str(path.resolve()),index['manifest_identity'],certificate['certification_id'])
    if key in _DATASETS:
        data=_DATASETS[key];start=time.perf_counter();compatible_certificates(data.producer_certificate,certificate,data.config)
        profile['certificate_validation_seconds']=time.perf_counter()-start;profile['storage_paths']=data.load_profile.get('storage_paths',{})
        data.progress=progress or (lambda *a,**k:None);data.load_profile=profile;data.load_profile['dataset_object_reused']=True;_DATASETS.move_to_end(key)
    else:
        data=Dataset(path,certificate,progress=progress,verify=False);data.load_profile.update(profile,dataset_object_reused=False);_DATASETS[key]=data
        while len(_DATASETS)>2:_DATASETS.popitem(last=False)
    return data
