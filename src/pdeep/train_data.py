"""Stateless matched training sampler and immutable named held-out manifests."""
from dataclasses import dataclass,asdict
import torch
from .a_training import SupervisedBatch
from .batches import UnknownCoefficients
from .core import require,digest
from .data import CaseDataset,observations,training_queries,collate_inputs,case_coefficients
from .reference import canonical_spec
from .representations import float32_batch


def case_batch(dataset,normalizers,keys,*,observation_name,query_name,query_count,device):
    cases=tuple(dataset.case(split,pde,index) for split,pde,index in keys)
    specs=tuple(canonical_spec(c.key.pde) for c in cases)
    obs=tuple(observations(dataset,c,normalizers,set_id=observation_name) for c in cases)
    query=tuple(training_queries(dataset,c,count=query_count,set_id=query_name) for c in cases)
    xi=case_coefficients(cases)
    batch=SupervisedBatch(specs,float32_batch(collate_inputs([o.inputs for o in obs]),device=device),
                          float32_batch(collate_inputs([q.inputs for q in query]),device=device),
                          torch.cat([q.target for q in query]).to(dtype=torch.float32).to(device),
                          UnknownCoefficients(xi.xi.to(dtype=torch.float32).to(device),xi.active.to(device)),tuple(c.identity for c in cases))
    identities={'cases':batch.case_ids,'observations':tuple(o.metadata.identity for o in obs),'queries':tuple(q.identity for q in query)}
    return batch,identities


class TrainingSampler:
    def __init__(self,config,normalizers):
        self.config=config;self.normalizers=normalizers;self.dataset=CaseDataset(config.root_seed)

    def keys(self,cursor):
        rng=self.dataset.streams.numpy('optimization','shared-engine-v1','window',cursor)
        pool=self.config.training_case_ids
        def index():return int(rng.integers(12000)) if pool is None else pool[int(rng.integers(len(pool)))]
        return tuple(('train',self.config.pdes[int(rng.choice(len(self.config.pdes),p=self.config.probabilities))],index()) for _ in range(self.config.effective_batch))

    def batch(self,cursor):
        return case_batch(self.dataset,self.normalizers,self.keys(cursor),observation_name=f'engine-train-{cursor}',query_name=f'engine-query-{cursor}',query_count=self.config.query_count,device=self.config.device)

    def state_dict(self,cursor):return {'version':'stateless-case-key-v1','cursor':cursor,'root_seed':self.config.root_seed,'dataset_id':self.dataset.identity,'pdes':list(self.config.pdes),'probabilities':list(self.config.probabilities),'selected_training_case_ids':None if self.config.training_case_ids is None else list(self.config.training_case_ids)}


@dataclass(frozen=True)
class ValidationManifest:
    name: str
    root_seed: int
    dataset_id: str
    normalizer_id: str
    keys: tuple
    observation_name: str
    query_name: str
    query_count: int
    observation_ids: tuple
    query_ids: tuple
    case_ids: tuple
    domain: str='fixed validation query-risk points; not full64x64/time-grid evaluation'

    @property
    def identity(self):return digest(asdict(self))

    def batches(self,normalizers,*,device='cpu'):
        require(normalizers.identity==self.normalizer_id,'validation normalization identity mismatch')
        dataset=CaseDataset(self.root_seed);require(dataset.identity==self.dataset_id,'validation dataset identity mismatch')
        for i,key in enumerate(self.keys):
            batch,ids=case_batch(dataset,normalizers,(key,),observation_name=self.observation_name,query_name=self.query_name,query_count=self.query_count,device=device)
            require(ids['observations']==(self.observation_ids[i],) and ids['queries']==(self.query_ids[i],) and ids['cases']==(self.case_ids[i],),'fixed validation identities changed')
            yield batch


def make_validation_manifest(config,normalizers):
    dataset=CaseDataset(config.root_seed);keys=[];obsids=[];qids=[];caseids=[]
    # Independent named validation stream selects IDs once. Never training/test.
    for pde in config.pdes:
        rng=dataset.streams.numpy('validation','manifest',config.validation_name,pde)
        ids=rng.permutation(1024)[:config.validation_cases_per_pde]
        keys.extend(('validation',pde,int(i)) for i in ids)
    obsname=config.validation_name+'-observations';qname=config.validation_name+'-queries'
    for key in keys:
        _,ids=case_batch(dataset,normalizers,(key,),observation_name=obsname,query_name=qname,query_count=config.validation_query_count,device='cpu')
        obsids.extend(ids['observations']);qids.extend(ids['queries']);caseids.extend(ids['cases'])
    return ValidationManifest(config.validation_name,config.root_seed,dataset.identity,normalizers.identity,tuple(keys),obsname,qname,config.validation_query_count,tuple(obsids),tuple(qids),tuple(caseids))
