"""Verified finite continuous-query banks for bounded, manually launched studies.

A bank stores evaluated point labels, not trajectory frames. Reference warm-up
is explicit and may be CPU expensive. The physical prior is unchanged; a bounded
run minimizes an explicitly declared finite-query empirical objective. Observation
noise is fixed per case/set; selecting training queries never changes it.
"""
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import torch
from ..core import digest,require
from ..artifact_io import write_artifact,read_payload
from ..geometric.data import Batch,Tokens
from .definition import *
from .data import Case,design,tokens,evaluate_pairs,pack

SCHEMA='G1_smooth-point-bank-v1'

def physical_design(c,n,**kwargs):
    # Sampling triangulation is frozen canonical geometry, independent of neural
    # conditioning/remesh choices. This changes no area-uniform physical law.
    return design(replace(c,conditioning_h=.08),n,**kwargs)

def grid(c,size=64):
    mesh=c.problem.mesh;lo,hi=mesh.vertices.min(0),mesh.vertices.max(0)
    a,b=np.meshgrid(np.linspace(lo[0],hi[0],size),np.linspace(lo[1],hi[1],size),indexing='ij');x=np.column_stack((a.ravel(),b.ravel()))
    x=x[mesh.features(x)['inside']]
    # Boundary coordinates may round outside polygon under float32 conversion.
    # Use only strict interior grid points for field scores; BC scored separately.
    x=x[mesh.features(x)['distance']>1e-7]
    T=(0.,) if c.pde=='poisson' else (0.,.02,.05,.1,.2,.35,.5,.75,1.)
    qset=(0,1) if c.pde=='wave' else (0,)
    xx=np.tile(x,(len(T)*len(qset),1));t=np.tile(np.repeat(T,len(x)),len(qset));q=np.repeat(qset,len(T)*len(x))
    return xx,t,q,c.pde!='poisson'

class Bank:
    def __init__(self,root,record,*,train_queries=2048,grid_size=64):
        self.root=Path(root);self.record=record;self.norms=__import__('pdeep.smooth.certification',fromlist=['normalizers']).normalizers(record)
        self.train_queries=train_queries;self.grid_size=grid_size;self.cache={}
    def key(self,c,set_id=0):
        return digest((SCHEMA,c.manifest,self.record['certification_id'],self.norms.identity,self.train_queries,self.grid_size,set_id))
    def get(self,c,set_id=0):
        key=self.key(c,set_id)
        if key in self.cache:return self.cache[key]
        path=self.root/(key+'.json')
        if path.exists():
            record,payload=read_payload(path,kind='diagnostic_bundle',required=('observation','training','validation','identities'))
            require(record['bank_identity']==key and record['certification_id']==self.record['certification_id'],'incompatible reference bank')
        else:
            print(f'Reference warm-up (CPU64, no optimizer): {c.pde}/{c.family}/{c.split}/{c.index}',flush=True)
            obs=physical_design(c,{'poisson':32,'heat':48,'wave':64}[c.pde],observation=True,set_id=set_id)
            train=physical_design(c,self.train_queries,set_id=713)
            val=grid(c,self.grid_size)
            lengths=[len(v[0]) for v in (obs,train,val)];xx=np.concatenate([v[0] for v in (obs,train,val)]);tt=np.concatenate([v[1] for v in (obs,train,val)]);qq=np.concatenate([v[2] for v in (obs,train,val)])
            clean=evaluate_pairs(c,xx,tt,qq);a,b=lengths[0],sum(lengths[:2])
            ot,oi=tokens(c,a,self.norms,observation=True,set_id=set_id,coordinates=obs,clean=clean[:a])
            tr,ti=tokens(c,lengths[1],self.norms,set_id=713,coordinates=train,clean=clean[a:b]);va,vi=tokens(c,lengths[2],self.norms,set_id=917,coordinates=val,clean=clean[b:])
            payload={'observation':vars(ot),'training':vars(tr),'validation':vars(va),'identities':{'observation':oi['identity'],'training':ti['identity'],'validation':vi['identity'],'physical_case':c.case_id,'bank':key}}
            meta={'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'smooth-reference-banks','scientific_schema':SCHEMA,'benchmark':BENCHMARK_ID,'bank_identity':key,'case':c.manifest,'certification_id':self.record['certification_id']}
            write_artifact(self.root,key,meta,payload,required=tuple(payload),idempotent=True)
        self.cache[key]=payload;return payload
    def batch(self,cases,*,cursor=0,count=128,validation=False,device='cpu'):
        obs=[];qry=[];identities=[]
        for c in cases:
            payload=self.get(c);obs.append(Tokens(**payload['observation']));data=payload['validation' if validation else 'training']
            if not validation:
                N=data['valid'].shape[1];idx=random_stream('training-query-selection',c.case_id,cursor).choice(N,size=count,replace=count>N)
                data={k:None if v is None else v[:,idx] for k,v in data.items()}
            qry.append(Tokens(**data));identities.append(payload['identities'])
        return Batch(tuple(cases),pack(obs),pack(qry),self.norms).to(device),identities
