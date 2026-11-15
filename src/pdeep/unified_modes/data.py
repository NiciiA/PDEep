"""Information-only views; unchanged physical IDs, labels and encounter ordering."""
from collections import Counter
from fractions import Fraction
from math import lcm
import numpy as np
from ..core import digest,require
from ..smooth.definition import random_stream
from ..geometric.data import Batch
from ..unified.data import BalancedSampler as OriginalSampler
from .conditioning import MODES,SCHEMA,FieldCondition,CaseView,mask_for


def mode_at(config,cursor,slot,task):
    cycle=(4*cursor+slot)//len(config.dataset_tasks) if hasattr(config,'dataset_tasks') else None
    fractions=[Fraction(float(v)).limit_denominator(100) for v in config.mode_mixture];length=lcm(*(v.denominator for v in fractions))
    require(length<=100 and sum(fractions)==1,'mode proportions need an exact rational cycle of <=100')
    bag=np.repeat(np.arange(3),[int(v*length) for v in fractions])
    return bag,length

class Sampler(OriginalSampler):
    def plan(self,cursor,slot,pde,family):
        bag,length=mode_at(self.config,cursor,slot,(pde,family));cycle=(4*cursor+slot)//len(self.tasks)
        rng=random_stream(SCHEMA,'mode-order',self.config.data_seed,pde,family,cycle//length);mode=MODES[int(rng.permutation(bag)[cycle%length])]
        rng=random_stream(SCHEMA,'known-mask',self.config.data_seed,cursor,slot,pde,family);mask,pattern=mask_for(pde,mode,rng)
        empty=mode=='FORWARD' or (mode=='HYBRID' and rng.random()<self.config.hybrid_zero_probability)
        return mode,mask,pattern,empty
    def state_dict(self,cursor):return {**super().state_dict(cursor),'conditioning_schema':SCHEMA,'mode_mixture':self.config.mode_mixture,'hybrid_zero_probability':self.config.hybrid_zero_probability,'mode_policy':'per-task permuted rational cycles; mask/empty streams independent of physical/coordinate/noise streams'}
    def batch(self,cursor):
        batch,identities=super().batch(cursor);views=[]
        for slot,c in enumerate(batch.cases):
            mode,mask,pattern,empty=self.plan(cursor,slot,c.pde,c.family);condition=FieldCondition.from_case(c,mask);views.append(CaseView(c,condition,mode))
            if empty:
                for name,v in vars(batch.observations).items():
                    if v is not None:v[slot].zero_()
            identities[slot].update(mode=mode,known_fraction=condition.known_fraction,known_pattern=pattern,field_condition_identity=condition.identity,valid_observations=int(batch.observations.valid[slot].sum()))
        return Batch(tuple(views),batch.observations,batch.queries,batch.normalizers),identities
