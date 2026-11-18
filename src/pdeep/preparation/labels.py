"""Batched continuous evaluation and verified in-progress per-case shards."""
from pathlib import Path
import json,time
import numpy as np
import torch
from ..core import require,digest
from ..artifact_io import atomic_json,write_artifact,read_payload
from ..artifact_tensors import native_scalars
from ..unified.streams import designs,coordinates_id

class Journal:
    def __init__(self,folder,case,config):
        self.root=Path(folder)/'.working'/case.case_id;self.path=self.root/'progress.json'
        self.identity=digest((case.manifest,config.identity,config.reference_execution_source,'exact-label-shards-v1'))
        self.index=json.loads(self.path.read_text()) if self.path.exists() else {'identity':self.identity,'shards':[],'qc':None}
        require(self.index['identity']==self.identity,'in-progress case journal identity mismatch')
    def load(self):
        out={}
        for rel in self.index['shards']:
            r,p=read_payload(self.root/rel,kind='diagnostic_bundle',required=('labels',));require(r['journal']==self.identity,'journal shard mismatch')
            for key,v in p['labels'].items():require(key not in out,'duplicate journal label shard');out[key]=v
        return out
    def save(self,labels):
        if not labels:return
        stem='labels-'+digest(tuple(labels));path=self.root/(stem+'.json')
        if path.exists():
            r,p=read_payload(path);require(r['journal']==self.identity and set(p['labels'])==set(labels),'journal commit collision')
        else:write_artifact(self.root,stem,{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-in-progress-labels','journal':self.identity},{'labels':labels},required=('labels',))
        self.index['shards'].append(path.name);atomic_json(self.path,self.index)
    def qc(self,value):self.index['qc']=native_scalars(value);atomic_json(self.path,self.index)
    def clean(self):
        # Only this journal's verified, committed shards; never case checkpoints.
        for rel in self.index['shards']:
            path=self.root/rel;r,_=read_payload(path)
            for name in r['files']:(path.parent/name).unlink()
            path.unlink()
        if self.path.exists():self.path.unlink()
        if self.root.exists() and not any(self.root.iterdir()):self.root.rmdir()

def prepare_labels(case,full,config,events,*,journal=None,progress=None,max_batches=None,profile=None,point_batch=32768):
    output=journal.load() if journal else {};pending=[];pending_keys=set();points=0;batches=0
    def record(name,seconds,cpu_seconds=0.):
        if profile is not None:
            profile[name]=profile.get(name,0.)+seconds
            profile['CPU/'+name]=profile.get('CPU/'+name,0.)+cpu_seconds
    def flush():
        nonlocal pending,points,batches
        if not pending:return
        start=time.perf_counter();cpu=time.process_time();x=np.concatenate([d[0] for _,d in pending]);t=np.concatenate([d[1] for _,d in pending]);q=np.concatenate([d[2] for _,d in pending]);y=full.evaluate(x,t,q)
        record('continuous_reference_evaluation',time.perf_counter()-start,time.process_time()-cpu)
        require(y.dtype==np.float64 and np.isfinite(y).all(),'reference labels require finite CPU64')
        new={};offset=0
        for key,design in pending:
            n=len(design[0]);new[key]=torch.from_numpy(y[offset:offset+n].copy());offset+=n
        start=time.perf_counter();cpu=time.process_time()
        if journal:journal.save(new)
        record('partial_serialization_validation',time.perf_counter()-start,time.process_time()-cpu)
        output.update(new);pending=[];points=0;pending_keys.clear();batches+=1
        if progress:progress('Reference preparation',operation='batched exact continuous label evaluation',case=case.case_id,designs_completed=len(output),points_completed=sum(v.numel() for v in output.values()),encounters=len(events),batches_completed=batches)
    iterator=iter(designs(case,config,events))
    while True:
        start=time.perf_counter();cpu=time.process_time()
        try:d=next(iterator)
        except StopIteration:record('continuous_query_generation',time.perf_counter()-start,time.process_time()-cpu);break
        key=coordinates_id(*d[:3]);record('continuous_query_generation',time.perf_counter()-start,time.process_time()-cpu)
        if key in output or key in pending_keys:continue
        pending.append((key,d));pending_keys.add(key);points+=len(d[0])
        if points>=point_batch:
            flush()
            if max_batches is not None and batches>=max_batches:return output,False
    flush();return output,True
