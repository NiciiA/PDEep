"""Feasibility draw adapter for the EXISTING research memory/gradient preflight.

Only data access and TRAIN-only readiness differ. The profiler, objective,
gradient checks, chunks and four-case accumulation execute the original code.
No global monkeypatches, mutable training sampler or optimizer are used.
"""
from dataclasses import asdict
from pathlib import Path
import types
import numpy as np
import torch
from ..core import require,digest
from ..artifact_io import sha256,write_artifact
from ..geometric.data import Batch,Tokens
from ..smooth.data import Case
from ..unified.data import BalancedSampler
from ..unified_modes import preflight as original
from .provider import FeasibilityProvider

def _bind(function,**bindings):
    bound=types.FunctionType(function.__code__,{**function.__globals__,**bindings},function.__name__,function.__defaults__,function.__closure__)
    bound.__kwdefaults__=function.__kwdefaults__;return bound

def _tokens(value,slot,count=None):
    return Tokens(**{k:None if v is None else v[slot:slot+1,:count].clone() for k,v in vars(value).items()})

class TrainingDraws:
    def __init__(self,dataset):
        self.dataset=dataset;self.provider=FeasibilityProvider(dataset);self.visits={};self.used={}
        self.sampler=BalancedSampler(type('Seed',(),{'data_seed':dataset.config.seed})(),dataset,provider=self.provider)
        for cursor,ids in enumerate(self.provider.case_ids):
            for slot,cid in enumerate(ids):self.visits.setdefault(cid,[]).append((cursor,slot))
        require(len(self.provider.case_ids)==dataset.config.stream_updates==3000,'feasibility preflight requires the exact3000-update mapping')
    def batch(self,cases,*,count,device,seed,**unused):
        require(len(cases)==1 and cases[0].split=='train','optimizer preflight only accepts TRAIN cases')
        case=cases[0];require(seed==self.dataset.config.seed,'preflight seed differs from prepared training schedule')
        require(case.case_id in self.visits,'TRAIN case has no recorded feasibility encounter: '+case.case_id)
        cursor,slot=self.visits[case.case_id][0]
        require(count<=self.dataset.config.stream_queries,'preflight prefix exceeds the declared prepared query count')
        keys=self.sampler.keys(cursor);cohort=tuple(Case(g,p,s,i,self.dataset.config.conditioning_h) for p,g,s,i in keys)
        batch,ids=self.provider.batch(cohort,cursor=cursor,count=self.dataset.config.stream_queries,seed=seed,device='cpu')
        self.used[case.case_id]={'case_id':case.case_id,'split':'train','logical_cursor':cursor,'logical_slot':slot,**ids[slot]}
        selected=Batch((case,),_tokens(batch.observations,slot),_tokens(batch.queries,slot,count),batch.normalizers).to(device)
        return selected,[ids[slot]]
    def preflight_batch(self,dataset,case,mode,count,device,seed=42):
        return _bind(original.preflight_batch,Provider=lambda ignored:self)(dataset,case,mode,count,device,seed)

class _TrainingOnly:
    def __init__(self,dataset):self.base=dataset
    def __getattr__(self,name):return getattr(self.base,name)
    def require_ready(self,split='train'):
        # Original preflight requests validation readiness too. It is not used
        # by any optimizer/memory batch and must not materialize validation here.
        if split=='train':return self.base.require_ready('train')
        require(split=='validation','unexpected non-TRAIN preflight access')
    def cases(self,split,**kwargs):
        require(split=='train','optimizer preflight may not select validation/test')
        return self.base.cases('train',**kwargs)

def run_preflight(configs,dataset,root,*,monitor=None,query_counts=None):
    require(dataset.index.get('feasibility'),'feasibility preflight needs its verified mapping')
    require(not query_counts,'optional throughput sweeps need explicitly prepared designs; use the declared4096-query preflight first')
    require(all(c.query_count==dataset.config.stream_queries==4096 and c.updates==3000 for c in configs),'preflight must use4096 queries and the exact3000-update schedule')
    draws=TrainingDraws(dataset);view=_TrainingOnly(dataset)
    def save(directory,stem,metadata,payload,**kwargs):
        payload['preflight'].update(provider='FeasibilityProvider',split='train',mapping_identity=dataset.index['feasibility']['mapping_identity'],preflight_adapter_source=sha256(Path(__file__)),prepared_draws=list(draws.used.values()),sampler_policy='separate stateless lookup; no training cursor advanced; small gradient smoke uses an explicit prefix of the same prepared4096 draw')
        return write_artifact(directory,digest(payload['preflight']),metadata,payload,**kwargs)
    execute=_bind(original.run_preflight,preflight_batch=draws.preflight_batch,write_artifact=save)
    try:return execute(configs,view,root,monitor=monitor,query_counts=())
    except Exception as error:
        from .loading import MissingPreparedDraw
        if isinstance(error,MissingPreparedDraw):
            request=Path(root)/'preflight-missing'/(error.request['coordinates_id']+'.json')
            from ..artifact_io import atomic_json
            atomic_json(request,error.request)
            raise ValueError(str(error)+f'\nExplicit offline repair: python -m pdeep.preparation prepare-label --manifest {dataset.path} --request {request}') from error
        raise

def four_case_control(config,dataset,*,monitor=None):
    """Bounded real accumulated backward for the three-model CPU/CUDA audit.

    Uses the original case selection and exact accumulated_backward function.
    No optimizer, validation/test data or reference solve.
    """
    from ..artifact_rng import capture_rng,restore_rng
    from ..artifact_tensors import state_identity
    from ..unified_modes.model import create
    from ..unified_modes.performance import accumulated_backward
    state=capture_rng();draws=TrainingDraws(dataset)
    try:
        dataset.require_ready('train');chosen=[]
        for pde in ('wave','heat','poisson','wave'):
            candidates=[v for v in dataset.cases('train') if v.pde==pde and v.case_id not in {x.case_id for x in chosen}]
            chosen.append(max(candidates,key=lambda v:(v.family not in ('square','rectangle'),len(v.problem.mesh.cells))))
        model=create(config.architecture,dataset.norms,config.capacity,seed=config.root_seed,device=config.device);before=state_identity(model)
        result=accumulated_backward(model,config,chosen,lambda case,mode:draws.preflight_batch(dataset,case,mode,config.query_count,config.device,config.data_seed))
        require(state_identity(model)==before,'preflight changed model weights');result.update(model=config.model_key,prepared_draws=list(draws.used.values()))
        return result
    finally:restore_rng(state,target_device=config.device)
