"""Self-contained relative-path dataset manifests; legacy scientific IDs intact."""
from collections import OrderedDict
from pathlib import Path,PurePosixPath
import json
from ..core import digest,require
from ..artifact_io import atomic_json,sha256,read_payload,read_manifest
from ..smooth.certification import authorize,normalizers
from ..smooth.definition import split_manifest
from ..unified.config import DatasetConfig,SCHEMA
from ..unified.dataset import Dataset as LegacyDataset
from ..unified.streams import PlannedReference,STREAM_SCHEMA

LAYOUT='G1-portable-reference-dataset-v1'

def relative(root,name):
    name=str(name);p=PurePosixPath(name)
    require(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name,'unsafe dataset relative path')
    root=Path(root).resolve();path=root/name
    require(path.resolve().is_relative_to(root),'dataset path escapes folder (possibly symlink)')
    return path

def resolve_manifest(path):
    p=Path(path).expanduser()
    if p.is_dir():
        p=p/'manifest.json' if (p/'manifest.json').exists() else p/'index.json'
    require(p.is_file(),'dataset manifest not found: '+str(p));return p

def read_index(path):
    path=resolve_manifest(path);d=json.loads(path.read_text())
    require(d.get('portable_layout')==LAYOUT,'legacy index: export it once using the explicit legacy-export command')
    require(d.get('manifest_identity')==digest({k:v for k,v in d.items() if k!='manifest_identity'}),'dataset manifest checksum mismatch')
    config=DatasetConfig(**d['config'])
    require(d['schema']==SCHEMA and d['stream_schema']==STREAM_SCHEMA and d['dataset_identity']==config.identity,'dataset/config/stream identity mismatch')
    require(d['original_split_identity']==split_manifest()['identity'] and d['inventory_identity']==digest(d['cases']),'dataset split/inventory changed')
    require(len(set(d['records'].values()))==len(d['records']),'case artifact aliases')
    ids={r['case_id'] for r in d['cases']};require(set(d['records'])<=ids,'record not in physical case inventory')
    for rel in [*d['files'],*d['records'].values(),*d['assets'].values(),d['producer_certificate']]:relative(path.parent,rel)
    require(set(d['records'].values())|set(d['assets'].values())|{d['producer_certificate']}<=set(d['files']),'untracked required dataset files')
    if d['status']=='complete':
        require(len(d['records'])==sum(config.totals.values()) and not d['quarantine'],'incomplete dataset marked complete')
        require(all(len(set(d['qualification'].get('/'.join(t),{}).get('cases',[])))>=config.fine_spot_checks_per_task and d['qualification']['/'.join(t)]['passed'] for t in config.tasks),'missing finer numerical qualification')
    else:require(d['status']=='partial','unknown dataset status')
    return path,d

def commit(path,index):
    index.pop('manifest_identity',None);index['manifest_identity']=digest(index);atomic_json(path,index)

def add_file(index,root,path):
    rel=Path(path).relative_to(root).as_posix();p=relative(root,rel)
    index['files'][rel]={'sha256':sha256(p),'size_bytes':p.stat().st_size};return rel

def add_artifact(index,root,path):
    r=read_manifest(path)
    add_file(index,root,path)
    for name in r['files']:add_file(index,root,Path(path).parent/name)

def verify_folder(path,*,require_complete=False,progress=None):
    path,d=read_index(path)
    require(not require_complete or d['status']=='complete','dataset remains partial; completed records are reusable but full training is blocked')
    for i,(rel,expected) in enumerate(d['files'].items()):
        p=relative(path.parent,rel)
        require(p.is_file() and p.stat().st_size==expected['size_bytes'] and sha256(p)==expected['sha256'],'dataset file missing/corrupt: '+rel)
        if progress and i%100==0:progress('Dataset file verification',verified=i,total=len(d['files']))
    return path,d

def compatible_certificates(producer,current,config):
    """Reuse numerical BYTES, never authorize a foreign runtime to solve PDEs.

    The consumer still needs its existing native certificate. Both certificates
    must descend from the same frozen scientific parent with identical sources,
    numerical policy, represented meshes, priors, splits and TRAIN scales.
    Generation always authorizes the CURRENT runtime separately.
    """
    require(producer['certification_id']==digest({k:v for k,v in producer.items() if k!='certification_id'}),'producer certificate checksum mismatch')
    authorize(current,pdes=config.pdes,families=config.families,normalizer_identity=producer['normalizers']['identity'])
    parent=lambda r:r.get('parent_certification_id',r['certification_id'])
    require(parent(producer)==parent(current),'unrelated numerical certificate lineage')
    for key in ('reference_policy','reference_solver_identity','known_coefficients','certified_field_prior','certified_boundary_policies','splits','normalizers','additional_scientific_sources'):
        require(producer.get(key)==current.get(key),'producer/consumer scientific mismatch: '+key)
    require(producer['source_revision']['files']==current['source_revision']['files'],'producer scientific source mismatch')
    for family in config.families:
        for key in ('mesh_id','geometry_id','polygon_id','h','degree'):
            require(producer['mesh_records'][family]['production'][key]==current['mesh_records'][family]['production'][key],'producer/consumer mesh mismatch')
    caps={(v['pde'],v['family']) for v in producer['capabilities'] if v['passed']}
    require(set(config.tasks)<=caps,'producer certificate lacks selected tasks')
    require(all(producer.get(k,{}).get('passed') for k in ('independent_controls','boundary_controls','analytical_square_controls')),'producer certificate incomplete')

class Dataset(LegacyDataset):
    """Same prepared-label provider/training interface, with portable provenance."""
    def __init__(self,path,certificate,*,progress=None,verify=True):
        from .loading import verified_folder,LocalReference
        self.load_profile={}
        self.path,self.index=(verified_folder(path,self.load_profile,progress) if verify else read_index(path))
        self.root=self.path.parent;self.config=DatasetConfig(**self.index['config']);self.progress=progress or (lambda *a,**k:None)
        self.producer_certificate=json.loads(relative(self.root,self.index['producer_certificate']).read_text())
        require(self.producer_certificate['certification_id']==self.index['certificate_id'],'producer certificate ID mismatch')
        import time
        start=time.perf_counter()
        compatible_certificates(self.producer_certificate,certificate,self.config)
        self.load_profile['certificate_validation_seconds']=time.perf_counter()-start
        from ..geometric.portable import ROOT as numerical_inputs
        self.load_profile['storage_paths']={'dataset':str(self.root.resolve()),'numerical_inputs':str(numerical_inputs.resolve()),'loader_source':str(Path(__file__).resolve())}
        self.certificate=certificate;self.norms=normalizers(certificate)
        require(self.index['normalizer_identity']==self.norms.identity,'normalization changed')
        self.cache=LocalReference(self);self.cache.physics=self.producer_certificate['source_revision']
        self.rows={r['case_id']:r for r in self.index['cases']};self.verified=set()
        self._case_lists={};self._record_metadata={}
    def cases(self,split,*,training_only=False):
        key=(split,training_only)
        if key not in self._case_lists:self._case_lists[key]=super().cases(split,training_only=training_only)
        return self._case_lists[key]
    def require_case(self,c):
        self._load_case(c)
    def _load_case(self,c):
        from .loading import stamp,checked_file
        import time
        require(c.case_id in self.rows and self.rows[c.case_id]['key'][2]==c.split,'case not in dataset partition')
        if c.case_id in self.cache.loaded:
            try:unchanged=all(stamp(self.root/rel)==v for rel,v in self.cache.fingerprints[c.case_id].items())
            except OSError:unchanged=False
            if unchanged:
                self.cache.loaded.move_to_end(c.case_id);self.load_profile['case_cache_hits']=self.load_profile.get('case_cache_hits',0)+1;return self.cache.loaded[c.case_id]
            self.cache.drop(c.case_id);self.verified.discard(c.case_id)
        rel=self.index['records'].get(c.case_id);require(rel is not None,'unprepared or quarantined physical case: '+c.case_id)
        checked_file(relative(self.root,rel),self.index['files'][rel],self.load_profile)
        start=time.perf_counter()
        r,p=read_payload(relative(self.root,rel),kind='diagnostic_bundle',required=('qc','case','labels','sampling_space'))
        self.load_profile['case_deserialization_and_validation_seconds']=self.load_profile.get('case_deserialization_and_validation_seconds',0.)+time.perf_counter()-start
        self.load_profile['case_payload_loads']=self.load_profile.get('case_payload_loads',0)+1
        require(r['case_id']==c.case_id and r['dataset_identity']==self.config.identity and r['normalizer_identity']==self.norms.identity and r['certificate_id']==self.index['certificate_id'] and p['qc']['passed'],'case QC/provenance failed')
        start=time.perf_counter()
        require(p['case']==c.manifest and r['physics']==self.cache.physics and r['execution_source']==self.config.reference_execution_source,'case/reference specification changed')
        self.load_profile['physical_case_identity_validation_seconds']=self.load_profile.get('physical_case_identity_validation_seconds',0.)+time.perf_counter()-start
        space=relative(self.root,p['sampling_space']);require(space.is_file() and p['sampling_space'] in self.index['files'],'sampling space missing')
        fingerprints={rel:stamp(self.root/rel)}
        for name,info in r['files'].items():
            payload=(Path(rel).parent/name).as_posix();require(self.index['files'].get(payload)==info,'case payload/inventory mismatch');fingerprints[payload]=stamp(self.root/payload)
        self.cache.put(c.case_id,p,fingerprints)
        self.verified.add(c.case_id)
        return p
    def require_ready(self,split='train'):
        require(not self.index.get('control_only'),'diagnostic control data are not production training references')
        # The constructor verified the full file inventory. Readiness checks the
        # selected split's signed metadata; payload/QC/physical identity validation
        # is lazy and mandatory before a case is actually consumed.
        import time
        from .loading import checked_file
        start=time.perf_counter();cases=self.cases(split,training_only=split=='train')
        for i,c in enumerate(cases):
            rel=self.index['records'].get(c.case_id);require(rel is not None,'unprepared physical case: '+c.case_id)
            fingerprint=checked_file(self.root/rel,self.index['files'][rel],self.load_profile)
            saved=self._record_metadata.get(rel)
            if saved is not None and saved[0]==fingerprint:r=saved[1]
            else:
                r=read_manifest(self.root/rel,kind='diagnostic_bundle',verify_files=False);self._record_metadata[rel]=(fingerprint,r)
            require(r['case_id']==c.case_id and r['dataset_identity']==self.config.identity and r['certificate_id']==self.index['certificate_id'] and r['normalizer_identity']==self.norms.identity and r['physics']==self.cache.physics and r['execution_source']==self.config.reference_execution_source,'case readiness provenance mismatch')
            for name,info in r['files'].items():
                payload=(Path(rel).parent/name).as_posix();require(self.index['files'].get(payload)==info,'case payload/inventory mismatch')
                checked_file(self.root/payload,info,self.load_profile)
            if i%32==0 or i+1==len(cases):
                elapsed=time.perf_counter()-start;rate=(i+1)/max(elapsed,1e-9);self.progress('Dataset readiness (local metadata; payloads lazy)',split=split,checked=i+1,total=len(cases),seconds=elapsed,cases_per_second=rate,ETA_seconds=(len(cases)-i-1)/rate)
        for task in self.config.tasks:
            qualification=self.index['qualification'].get('/'.join(task),{})
            require(qualification.get('passed') is True and len(set(qualification.get('cases',[])))>=self.config.fine_spot_checks_per_task,'stratified finer-reference check missing/failed: '+str(task))
    def summary(self):return {**super().summary(),'status':self.index['status'],'portable_folder':str(self.root),'producer_certificate_id':self.index['certificate_id'],'consumer_certificate_id':self.certificate['certification_id'],'persistence':self.index.get('persistence',{}),'profile':self.index.get('preparation_profile',{})}
