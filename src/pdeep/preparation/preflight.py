"""Feasibility draw adapter for the EXISTING research memory/gradient preflight.

Only data access and TRAIN-only readiness differ. The profiler, objective,
gradient checks, chunks and four-case accumulation execute the original code.
No global monkeypatches, mutable training sampler or optimizer are used.
"""
from dataclasses import asdict
from pathlib import Path
import types
import copy,time
from contextlib import contextmanager
import numpy as np
import torch
from ..core import require,digest
from ..artifact_io import sha256,write_artifact
from ..geometric.data import Batch,Tokens
from ..smooth.data import Case
from ..unified.data import BalancedSampler
from ..unified_modes import preflight as original
from .provider import FeasibilityProvider
from .monitor import Dashboard

@contextmanager
def _operation(monitor,name,**details):
    if hasattr(monitor,'operation'):
        with monitor.operation(name,**details):yield
    else:
        monitor.phase(name,**details);yield

def inventory(configs,norms,*,monitor=None):
    monitor=monitor or Dashboard(notebook=False)
    with _operation(monitor,'Architecture inventory'):return original.inventory(configs,norms)

def _bind(function,**bindings):
    bound=types.FunctionType(function.__code__,{**function.__globals__,**bindings},function.__name__,function.__defaults__,function.__closure__)
    bound.__kwdefaults__=function.__kwdefaults__;return bound

def _tokens(value,slot,count=None):
    return Tokens(**{k:None if v is None else v[slot:slot+1,:count].clone() for k,v in vars(value).items()})

class TrainingDraws:
    def __init__(self,dataset):
        self.dataset=dataset;self.provider=FeasibilityProvider(dataset);self.visits={};self.used={};self.prepared={}
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
        if case.case_id not in self.prepared:
            self.prepared[case.case_id]=self.provider.case_batch(case,cursor=cursor,slot=slot,count=self.dataset.config.stream_queries,seed=seed)
        batch,identity=self.prepared[case.case_id]
        self.used[case.case_id]={'split':'train',**identity}
        selected=Batch((case,),_tokens(batch.observations,0),_tokens(batch.queries,0,count),batch.normalizers).to(device)
        return selected,[identity]
    def preflight_batch(self,dataset,case,mode,count,device,seed=42):
        return _bind(original.preflight_batch,Provider=lambda ignored:self)(dataset,case,mode,count,device,seed)
    def audit(self,cases,root):
        """Enumerate ALL dependencies before any neural model is allocated."""
        from .coordinates import prepared_design
        from .loading import MissingPreparedDraw
        from ..unified.config import SCHEMA
        from ..unified.streams import coordinates_id
        from ..artifact_io import atomic_json
        required=[];missing=[];unique={c.case_id:c for c in cases}
        for cid,case in sorted(unique.items()):
            cursor,slot=self.visits[cid][0];a,b,n=map(int,self.provider.mapping[cursor,slot]);valid=True
            for observation,count,role in ((True,n,'encounter'),(False,self.dataset.config.stream_queries,'supervised')):
                sid=(SCHEMA,role,self.dataset.config.seed,a,b);info={'case_id':cid,'PDE':case.pde,'geometry':case.family,'split':case.split,'logical_cursor':cursor,'logical_slot':slot,'reference_cursor':a,'reference_slot':b,'role':role,'points':count}
                try:
                    design=prepared_design(self.dataset,case,count,observation=observation,set_id=sid)
                    info['coordinates_id']=coordinates_id(*design[:3]);info['status']='verified'
                except MissingPreparedDraw as error:
                    info.update(status='missing',coordinates_id=error.request['coordinates_id']);missing.append({**error.request,**info});valid=False
                required.append(info)
            if valid:self.batch((case,),count=self.dataset.config.stream_queries,device='cpu',seed=self.dataset.config.seed)
        report={'schema':'G1-feasibility-preflight-dependencies-v1','dataset_identity':self.dataset.config.identity,'mapping_identity':self.dataset.index['feasibility']['mapping_identity'],'independent_designs_required':0,'required':required,'missing':missing,'model_allocations':0,'reference_solves':0}
        path=Path(root)/'preflight-draw-audits'/(digest(report)+'.json');atomic_json(path,report)
        require(not missing,f'Preflight dependency audit found ALL {len(missing)} missing exact training designs before model allocation. Batch manifest: {path}. No automatic solving or one-at-a-time repair. Check the producer dataset/coordinate replay; original labels remain untouched.')
        return str(path),report

def diverse_cases(dataset):
    available=dataset.cases('train');chosen=[]
    for pde,family in (('wave','annulus'),('heat','rectangle'),('poisson','hole'),('wave','square')):
        matching=[c for c in available if c.pde==pde and c.family==family]
        require(matching,'representative preflight needs '+pde+'/'+family)
        chosen.append(min(matching,key=lambda c:c.index))
    return chosen

def required_cases(dataset):
    available=dataset.cases('train');chosen=[]
    for pde in ('wave','heat','poisson','wave'):
        matching=[c for c in available if c.pde==pde and c.case_id not in {x.case_id for x in chosen}]
        chosen.append(max(matching,key=lambda c:(c.family not in ('square','rectangle'),len(c.problem.mesh.cells))))
    return chosen

class _TrainingOnly:
    def __init__(self,dataset,monitor):self.base=dataset;self.monitor=monitor
    def __getattr__(self,name):return getattr(self.base,name)
    def require_ready(self,split='train'):
        # Original preflight requests validation readiness too. It is not used
        # by any optimizer/memory batch and must not materialize validation here.
        if split=='train':
            with _operation(self.monitor,'TRAIN readiness'):return self.base.require_ready('train')
        require(split=='validation','unexpected non-TRAIN preflight access')
    def cases(self,split,**kwargs):
        require(split=='train','optimizer preflight may not select validation/test')
        return self.base.cases('train',**kwargs)

def run_preflight(configs,dataset,root,*,monitor=None,query_counts=None):
    require(dataset.index.get('feasibility'),'feasibility preflight needs its verified mapping')
    require(not query_counts,'optional throughput sweeps need explicitly prepared designs; use the declared4096-query preflight first')
    require(all(c.query_count==dataset.config.stream_queries==4096 and c.updates==3000 for c in configs),'preflight must use4096 queries and the exact3000-update schedule')
    from ..artifact_rng import capture_rng,restore_rng
    from .feasibility import source_identity
    require(all(c.feasibility_source==source_identity() for c in configs),'feasibility adapter source changed; restart/reload and rerun configuration before preflight')
    invocation_start=time.perf_counter()
    monitor=monitor or Dashboard(dataset.config);monitor.latest['preceding_dataset_load']=copy.deepcopy(getattr(dataset,'load_profile',{}))
    rng=capture_rng();previous_progress=getattr(dataset,'progress',None);dataset.progress=monitor.phase
    draws=None;view=_TrainingOnly(dataset,monitor)
    def save(directory,stem,metadata,payload,**kwargs):
        representative=[]
        for c in configs:
            r=four_case_control(c,dataset,monitor=monitor,diverse=True,_draws=draws);require(r['status']=='fits','representative four-case memory preflight exceeds budget');representative.append(r)
        payload['preflight'].update(provider='FeasibilityProvider',split='train',mapping_identity=dataset.index['feasibility']['mapping_identity'],preflight_adapter_source=sha256(Path(__file__)),prepared_draws=list(draws.used.values()),representative_accumulation_reports=representative,dependency_audit=audit_path,total_invocation_seconds=time.perf_counter()-invocation_start,operation_timings=monitor.latest.get('completed_operation_timings',[]),sampler_policy='separate stateless lookup; full verified CPU batches materialized before models; small diagnostics slice both coordinates and values with no label lookup')
        return write_artifact(directory,digest(payload['preflight']),metadata,payload,**kwargs)
    def construct(*args,**kwargs):
        with _operation(monitor,'Model construction',architecture=args[0]):return original.create(*args,**kwargs)
    def profile(*args,**kwargs):
        with _operation(monitor,'Full memory forward/backward',**monitor.latest.get('details',{})):return original.profile(*args,**kwargs)
    def accumulated(*args,**kwargs):
        with _operation(monitor,'Worst-mesh four-case backward',**monitor.latest.get('details',{})):return original.accumulated_backward(*args,**kwargs)
    try:
        draws=TrainingDraws(dataset)
        with _operation(monitor,'Verify ALL prepared preflight draws'):
            audit_path,_=draws.audit([*required_cases(dataset),*diverse_cases(dataset)],root)
        execute=_bind(original.run_preflight,preflight_batch=draws.preflight_batch,write_artifact=save,create=construct,profile=profile,accumulated_backward=accumulated)
        return execute(configs,view,root,monitor=monitor,query_counts=())
    finally:
        if previous_progress is None:del dataset.progress
        else:dataset.progress=previous_progress
        restore_rng(rng,target_device=configs[0].device)

def four_case_control(config,dataset,*,monitor=None,diverse=False,_draws=None):
    """Bounded real accumulated backward for the three-model CPU/CUDA audit.

    Uses the original case selection and exact accumulated_backward function.
    No optimizer, validation/test data or reference solve.
    """
    from ..artifact_rng import capture_rng,restore_rng
    from ..artifact_tensors import state_identity
    state=capture_rng();draws=_draws or TrainingDraws(dataset);monitor=monitor or Dashboard(dataset.config,notebook=False)
    try:
        dataset.require_ready('train');chosen=diverse_cases(dataset) if diverse else required_cases(dataset)
        if _draws is None:draws.audit(chosen,Path('reports/g1_preflight_draw_fix'))
        model=original.create(config.architecture,dataset.norms,config.capacity,seed=config.root_seed,device=config.device);before=state_identity(model)
        with _operation(monitor,'Representative four-case backward' if diverse else 'Worst-mesh four-case backward',model=config.model_key,queries_per_case=config.query_count,query_chunk=config.query_chunk):
            result=original.accumulated_backward(model,config,chosen,lambda case,mode:draws.preflight_batch(dataset,case,mode,config.query_count,config.device,config.data_seed))
        require(state_identity(model)==before,'preflight changed model weights');result.update(model=config.model_key,prepared_draws=list(draws.used.values()),coverage='representative geometries' if diverse else 'original worst-mesh control')
        return result
    finally:restore_rng(state,target_device=config.device)
