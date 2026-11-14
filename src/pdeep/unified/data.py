"""Counter-keyed fair multi-task order and fresh continuous physical observations."""
from dataclasses import replace
import numpy as np
import torch
from ..core import digest, require
from ..geometric.data import Batch, Tokens
from ..smooth.data import Case, tokens, pack
from ..smooth.bank import physical_design
from ..smooth.definition import random_stream
from .config import SCHEMA, case_index

OBSERVATION_COUNTS=(8,16,32,64,128)
DEFAULT_COUNTS={'poisson':32,'heat':48,'wave':64}


class Provider:
    def __init__(self,dataset):self.dataset=dataset;self.norms=dataset.norms;self.cache=dataset.cache
    def token_set(self,c,count,*,observation=False,set_id=0,coordinates=None,noise_multiplier=1.):
        self.dataset.require_case(c)
        design=physical_design(c,count,observation=observation,set_id=set_id) if coordinates is None else coordinates
        clean=self.cache.evaluate(c,*design[:3])
        token,meta=tokens(c,count,self.norms,observation=observation,set_id=set_id,coordinates=design,clean=clean)
        if observation and noise_multiplier!=1:
            require(noise_multiplier>0,'noise multiplier must be strictly positive')
            old=token.value.clone();truth=torch.as_tensor(clean,dtype=old.dtype)[None,:,None]
            token.value=truth+noise_multiplier*(old-truth);token.sigma*=noise_multiplier
            meta['sigma']=meta['sigma']*noise_multiplier
        return token,meta
    def batch(self,cases,*,cursor=0,count=4096,seed=42,training=True,device='cpu',observation_count=None):
        obs=[];qry=[];identities=[]
        for j,c in enumerate(cases):
            sid=(SCHEMA,'encounter',seed,cursor,j) if training else (SCHEMA,'fixed-evaluation',seed)
            n=int(random_stream('unified-count',seed,cursor,j).choice(OBSERVATION_COUNTS)) if training else DEFAULT_COUNTS[c.pde]
            if observation_count is not None:n=observation_count
            ot,oi=self.token_set(c,n,observation=True,set_id=sid)
            qt,qi=self.token_set(c,count,set_id=(SCHEMA,'supervised' if training else 'validation-query',seed,cursor if training else 0,j if training else 0))
            obs.append(ot);qry.append(qt);identities.append({'case_id':c.case_id,'observations':oi['identity'],'queries':qi['identity'],'observation_count':n})
        return Batch(tuple(cases),pack(obs),pack(qry),self.norms).to(device),identities


class BalancedSampler:
    def __init__(self,config,dataset,provider=None):
        self.config=config;self.dataset=dataset;self.provider=provider or Provider(dataset);self.tasks=dataset.config.training_tasks
    def state_dict(self,cursor):
        return {'schema':SCHEMA,'cursor':cursor,'seed':self.config.data_seed,'dataset':self.dataset.config.identity,
                'tasks':self.tasks,'effective_batch':4,'policy':'permuted full task cycles, independent uniform physical cases; fresh continuous observations/queries'}
    def keys(self,cursor):
        c=self.config;D=self.dataset.config;keys=[];used=set()
        for j in range(4):
            ordinal=4*cursor+j;cycle,offset=divmod(ordinal,len(self.tasks))
            permutation=random_stream('unified-task-order',c.data_seed,cycle).permutation(len(self.tasks))
            p,g=self.tasks[int(permutation[offset])]
            index=int(random_stream('unified-physical-index',c.data_seed,ordinal,p,g).integers(D.counts[0]))
            for _ in range(D.counts[0]):
                key=(p,g,'train',case_index(g,index))
                if key not in used:break
                index=(index+1)%D.counts[0]
            require(key not in used,'cannot form four independent physical cases');used.add(key);keys.append(key)
        return tuple(keys)
    def batch(self,cursor):
        D=self.dataset.config;cases=tuple(Case(g,p,s,i,D.conditioning_h) for p,g,s,i in self.keys(cursor))
        return self.provider.batch(cases,cursor=cursor,count=self.config.query_count,seed=self.config.data_seed,device=self.config.device)
