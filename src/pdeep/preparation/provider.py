"""Unique independent parent-counter coordinates, shared by all neural models."""
from pathlib import Path
import numpy as np
from ..core import require
from ..unified.data import Provider
from ..unified.config import SCHEMA
from ..smooth.bank import physical_design
from ..smooth.data import tokens,pack
from ..geometric.data import Batch
from ..artifact_io import read_payload

class FeasibilityProvider(Provider):
    def __init__(self,dataset):
        super().__init__(dataset)
        _,p=read_payload(dataset.root/'reference_cache/feasibility-plan.json',kind='diagnostic_bundle',required=('mapping','case_ids'))
        require(p['mapping_identity']==dataset.index['feasibility']['mapping_identity'],'dataset/reference mapping differs')
        self.mapping=p['mapping'].numpy();self.case_ids=p['case_ids']
    def token_set(self,c,count,*,observation=False,set_id=0,coordinates=None,noise_multiplier=1.):
        if coordinates is None:
            from .coordinates import prepared_design
            coordinates=prepared_design(self.dataset,c,count,observation=observation,set_id=set_id)
        return super().token_set(c,count,observation=observation,set_id=set_id,coordinates=coordinates,noise_multiplier=noise_multiplier)
    def case_batch(self,c,*,cursor,slot,count=4096,seed=42,device='cpu'):
        require(c.case_id==self.case_ids[cursor][slot],'logical sampler/case mapping mismatch')
        require(count==self.dataset.config.stream_queries and seed==self.dataset.config.seed,'feasibility needs its exact declared parent draw')
        a,b,n=map(int,self.mapping[cursor,slot]);sid=(SCHEMA,'encounter',seed,a,b)
        ot,oi=self.token_set(c,n,observation=True,set_id=sid);qt,qi=self.token_set(c,count,set_id=(SCHEMA,'supervised',seed,a,b))
        identity={'case_id':c.case_id,'observations':oi['identity'],'queries':qi['identity'],'observation_count':n,'logical_cursor':cursor,'logical_slot':slot,'reference_cursor':a,'reference_slot':b,'sampling_policy':'unique independent parent draw; never recycled'}
        return Batch((c,),ot,qt,self.norms).to(device),identity
    def batch(self,cases,*,cursor=0,count=4096,seed=42,training=True,device='cpu',observation_count=None):
        if not training:return super().batch(cases,cursor=cursor,count=count,seed=seed,training=False,device=device,observation_count=observation_count)
        require(tuple(c.case_id for c in cases)==tuple(self.case_ids[cursor]),'logical sampler/case mapping mismatch')
        require(count==self.dataset.config.stream_queries and seed==self.dataset.config.seed and observation_count is None,'feasibility training needs its exact declared parent draws')
        obs=[];queries=[];ids=[]
        for slot,c in enumerate(cases):
            batch,identity=self.case_batch(c,cursor=cursor,slot=slot,count=count,seed=seed)
            obs.append(batch.observations);queries.append(batch.queries);ids.append(identity)
        return Batch(tuple(cases),pack(obs),pack(queries),self.norms).to(device),ids
