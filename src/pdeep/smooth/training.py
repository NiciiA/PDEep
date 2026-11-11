"""Certified smooth specialization of the existing successful-update engine.

TrainingEngine.update/run and SuccessfulScheduler remain the optimizer loop.
No alternate optimizer implementation. Every construction/update/run checks the
numerical capability record. No uncertified research bypass is provided.
"""
from dataclasses import dataclass,asdict,fields
from pathlib import Path
import math,time,json,warnings
import numpy as np
import torch
from ..core import require,digest
from ..train_types import ResolvedTrainConfig,TrainingState,ValidationReport
from ..training import TrainingEngine,SuccessfulScheduler
from ..train_monitor import LiveMonitor
from ..artifact_drive import PersistencePolicy,sync_verified
from ..artifact_io import write_artifact,read_payload,atomic_json,sha256
from ..artifact_rng import capture_rng,restore_rng
from ..artifact_tensors import state_identity,tree_cpu
from ..geometric.models import create,GeometryModel,ModelConfig
from ..geometric.physics import Adapter,WeakConfig,weak_loss
from .definition import *
from .data import Case
from .bank import Bank
from .certification import load_record,authorize,normalizers

SCHEMA='G1_smooth-training-v1'

def select_device(device='auto'):
    if device!='auto':return device
    if torch.cuda.is_available():return 'cuda'
    if torch.backends.mps.is_available():return 'mps'
    return 'cpu'

@dataclass(frozen=True)
class Config(ResolvedTrainConfig):
    architecture: str='A_GEO'
    benchmark: str=BENCHMARK_ID
    inverse_track: str=BENCHMARK_ID
    preset: str='bounded'
    capacity: str='medium'
    device: str='auto'
    updates: int=1000
    effective_batch: int=1
    microbatch: int=1
    accumulation: int=1
    validation_cases_per_pde: int=4
    training_families: tuple=SEEN
    cases_per_geometry: int | None=None
    conditioning_h: float=.12
    bank_queries: int=2048
    validation_grid: int=64
    spatial_order: int=3
    time_order: int=3
    weak_tests: int=16
    physics_chunk: int=256
    scheduler: str='constant'
    full_training_enabled: bool=False
    def __post_init__(self):
        require(self.architecture in {'A_GEO','C_GEO'},'select independent A_GEO or C_GEO')
        require(self.benchmark==self.inverse_track==BENCHMARK_ID,'original rough G1 remains uncertified and blocked')
        require(self.capacity in {'medium','large','verification'},'unknown geometry capacity')
        require(self.capacity!='verification' or self.preset=='smoke','verification capacity is a nonresearch smoke fixture only')
        require(self.preset in {'smoke','tiny_overfit','bounded'},'only explicit smoke/tiny_overfit/bounded budgets are prepared')
        require(type(self.updates) is int and 0<self.updates<=({'smoke':3,'tiny_overfit':1000,'bounded':1000}[self.preset]),'invalid manual successful-update budget')
        require(bool(self.training_families) and set(self.training_families)<=set(SEEN),'training requires certified seen instances; no held-out leakage')
        require(self.cases_per_geometry is None or type(self.cases_per_geometry) is int and 1<=self.cases_per_geometry<=128,'choose full lazy split (None) or an explicit finite case pool1..128 per geometry')
        require(self.conditioning_h in {.04,.08,.12},'conditioning mesh policy must be .04/.08/.12; reference remains certified P4/.02')
        require(self.bank_queries>=self.query_count and self.validation_grid>=4,'invalid finite query bank/grid')
        require(self.spatial_order>=3 and self.time_order>=3 and self.weak_tests>0 and self.physics_chunk>0,'invalid actual weak-objective quadrature')
        if self.preset=='tiny_overfit':require(len(self.training_families)>=2 and self.cases_per_geometry is not None,'tiny overfit requires a finite fixed pool on at least2 geometries')
        device=select_device(self.device)
        keys={f.name for f in fields(ResolvedTrainConfig)};kwargs={k:v for k,v in asdict(self).items() if k in keys}
        kwargs.update(architecture='A' if self.architecture=='A_GEO' else 'C',benchmark='unit_square_modal_v1',inverse_track='T1',capacity='canonical_v1',device=device,preset='bounded',updates=min(self.updates,100),source_id=self.source_id or implementation_identity())
        # Original base validates devices/AdamW/mixture/batching/reporting. Validate
        # a surrogate budget only; actual smooth budget is independently bounded.
        if kwargs['scheduler']=='warmup_then_cosine':kwargs['warmup_updates']=min(self.warmup_updates,kwargs['updates']-1)
        base=ResolvedTrainConfig(**kwargs)
        require(self.scheduler!='warmup_then_cosine' or self.warmup_updates<self.updates,'warmup must fit actual budget')
        for f in fields(base):
            if f.name not in {'architecture','benchmark','inverse_track','capacity','preset','updates','training_case_ids','warmup_updates'}:object.__setattr__(self,f.name,getattr(base,f.name))
        require(self.training_case_ids is None,'smooth bounded pool uses explicit cases_per_geometry, not legacy T1 case indices')
    @property
    def case_pool(self):return None if self.cases_per_geometry is None else tuple((p,g,4*i+SEEN.index(g)) for p in self.pdes for g in self.training_families for i in range(self.cases_per_geometry))
    @property
    def case_pool_size(self):return len(self.pdes)*len(self.training_families)*(3000 if self.cases_per_geometry is None else self.cases_per_geometry)


def implementation_identity():
    root=Path(__file__).resolve().parents[3]
    paths=[*Path(__file__).parent.glob('*.py'),* (root/'src/pdeep/geometric').glob('*.py'),root/'src/pdeep/training.py',root/'src/pdeep/artifact_rng.py',root/'src/pdeep/artifact_drive.py']
    return digest({str(p.relative_to(root)):sha256(p) for p in sorted(paths)})


def register():
    from ..train_registry import REGISTRY,ArchitectureRegistration
    for name in ('A_GEO','C_GEO'):
        REGISTRY[name]=ArchitectureRegistration(name,True,tracks=('G1',BENCHMARK_ID),inputs=('GeometryProblem','ObservationTokens','QueryTokens'),outputs=('field','auxiliary_fields'),coordinate_derivatives=name=='C_GEO',checkpoint_version=SCHEMA,objective='field + .1 auxiliary'+(' + .1 weak + .1 initial + .1 boundary' if name=='C_GEO' else ''),objective_implementation='pdeep.geometric.physics.Adapter')

class Sampler:
    def __init__(self,config,bank):self.config=config;self.bank=bank
    def state_dict(self,cursor):return {'schema':SCHEMA,'cursor':cursor,'root_seed':self.config.root_seed,'config':self.config.identity,'pool':self.config.case_pool if self.config.case_pool is not None else {'index_range_per_pde':[0,11999],'seen_instance_catalog':SEEN,'selected_families':self.config.training_families},'bank_policy':(self.config.bank_queries,self.config.validation_grid),'observation_set':0,'split':split_manifest()['identity']}
    def batch(self,cursor):
        c=self.config;s=random_stream(c.root_seed,'order',cursor);cases=[]
        for j in range(c.effective_batch):
            p=str(s.choice(c.pdes,p=c.probabilities))
            if c.case_pool is None:
                g=str(s.choice(c.training_families));i=4*int(s.integers(3000))+SEEN.index(g)
            else:
                pool=[v for v in c.case_pool if v[0]==p];p,g,i=pool[int(s.integers(len(pool)))]
            cases.append(Case(g,p,'train',i,c.conditioning_h))
        b,ids=self.bank.batch(cases,cursor=cursor,count=c.query_count,device=c.device)
        return b,{'reference_banks':ids,'query_selection_cursor':cursor,'fixed_observations':True}

@dataclass(frozen=True)
class Manifest:
    cases: tuple
    normalizer_identity: str
    certification_id: str
    name: str
    @property
    def identity(self):return digest(asdict(self))

class Engine(TrainingEngine):
    def _authorize(self,families=None):return authorize(getattr(self,'execution_certificate',self.certificate),benchmark=self.config.benchmark,pdes=self.config.pdes,families=self.config.training_families if families is None else families,normalizer_identity=self.normalizers.identity)
    def __init__(self,config,run_root,*,certificate=None,monitor=None,model=None,persistence=None,allow_existing_run=False,execution_certificate=None):
        self.execution_certificate=execution_certificate
        self.config=config;self.certificate=load_record() if certificate is None else certificate;self.normalizers=normalizers(self.certificate)
        if self.execution_certificate is None:self.execution_certificate=self.certificate
        self._authorize()
        from .compatibility import approved_predecessor
        require(approved_predecessor(config.source_id),'Training implementation changed after configuration/preview; restart/review configuration before launch')
        register();self.state=TrainingState(config.architecture,config.run_id,stage='G1_smooth-manual');self.monitor=monitor or LiveMonitor(enabled=False)
        self.run_dir=Path(run_root)/BENCHMARK_ID/'runs'/config.architecture/config.run_id
        require(allow_existing_run or not self.run_dir.exists(),'run exists; select explicit resume or new run_tag')
        self.run_dir.mkdir(parents=True,exist_ok=True);self._started=time.monotonic();self._elapsed_base=0.;self.latest_step=None;self.latest_validation=None
        self.persistence_policy=persistence or PersistencePolicy();self.rng_generators={};self.parent_checkpoints=[];self.scaler=None;self.best_validation_model_state=None
        self.model=model or create(config.architecture,self.normalizers,config.capacity,seed=config.root_seed,device=config.device)
        require(self.model.config==ModelConfig.preset(config.architecture,config.capacity),'model/config mismatch; prepared experiments use the unchanged full architecture preset')
        require(self.model.normalizers.identity==self.normalizers.identity,'model normalization differs from certified training RMS')
        self.adapter=Adapter(config.architecture,WeakConfig(config.spatial_order,config.time_order,config.weak_tests,config.physics_chunk))
        self.bank=Bank(Path(run_root)/BENCHMARK_ID/'reference_banks',self.certificate,train_queries=config.bank_queries,grid_size=config.validation_grid)
        self.sampler=Sampler(config,self.bank);self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay);self.scheduler=SuccessfulScheduler(self.optimizer,config)
        if config.preset=='tiny_overfit':cs=tuple((p,g,'train',i) for p,g,i in config.case_pool);label='fixed TRAIN overfit grid; NOT generalization'
        else:cs=tuple((p,SEEN[i%len(SEEN)],'validation',i) for p in config.pdes for i in range(config.validation_cases_per_pde));label='paired seen-geometry validation; untouched test'
        self.manifest=Manifest(cs,self.normalizers.identity,self.certificate['certification_id'],label)
        self.state.rng_sampler_state=self.sampler.state_dict(0);self.state.scheduler_state=self.scheduler.state_dict();self.state.phase='ready'
        self.capability={'policy':'explicit actual adapter forward/backward available; no automatic optimizer probe','certification_id':self.certificate['certification_id']}
        if not allow_existing_run:
            atomic_json(self.run_dir/'config.json',asdict(config));atomic_json(self.run_dir/'benchmark.json',manifest(self.certificate));atomic_json(self.run_dir/'validation_manifest.json',{'identity':self.manifest.identity,**asdict(self.manifest)})
    def update(self,batch=None):self._authorize();return super().update(batch)
    def run(self,*,launch_full=False):
        self._authorize()
        require(self.state.phase!='complete','completed run cannot be launched again')
        # Manual caller/button invokes run; notebook never invokes automatically.
        if not self.state.validation_history:self.validate()
        reports=super().run(launch_full=launch_full)
        if self.config.preset=='tiny_overfit':atomic_json(self.run_dir/'overfit_assessment.json',self.overfit_assessment())
        return reports
    def overfit_assessment(self):
        h=self.state.validation_history
        if len(h)<2:return {'status':'pending user execution','criterion':'at least75% common field MSE reduction; every PDE/quantity improves or reaches1e-5 MSE floor; training memorization only'}
        before,after=h[0],h[-1];initial=before['value'];final=after['value'];reduction=1-final/max(initial,1e-30)
        old={tuple(r[k] for k in ('pde','quantity','family','case_id')):r['standardized_MSE'] for r in before['rows']}
        all_improve=all(r['standardized_MSE']<=max(.25*old[tuple(r[k] for k in ('pde','quantity','family','case_id'))],1e-5) for r in after['rows'])
        return {'status':'passed' if reduction>=.75 and all_improve else 'failed','initial_field_MSE':initial,'final_field_MSE':final,'fractional_reduction':reduction,'per_quantity_criterion':all_improve,'successful':self.state.successful,'failed':self.state.failed,'skipped':self.state.skipped,'criterion':'>=75% common and each quantity reduction or1e-5 floor; no generalization claim'}
    def validate(self,*,geometry_held_out=False,diagnostics=True):
        require(not self.state.accumulation_active,'validate only at completed update boundary');start=time.perf_counter();previous=self.model.training;self.model.eval();rows=[];physical=[];timings=[]
        cases=tuple((p,g,'geometry_held_out',i) for p in self.config.pdes for i,g in enumerate(HELD_OUT)) if geometry_held_out else self.manifest.cases
        self._authorize(tuple(sorted({g for _,g,_,_ in cases})))
        try:
            for p,g,split,i in cases:
                c=Case(g,p,split,i,self.config.conditioning_h);b,ids=self.bank.batch((c,),validation=True,device=self.config.device)
                from ..representation_checks import sync
                sync(torch.device(self.config.device));st=time.perf_counter()
                with torch.no_grad():
                    contexts=self.model.prepare_context(b.specs,b.observations);pred=self.model.predict(contexts,b.queries,chunk=1024);sensor=self.model.predict(contexts,b.observations,chunk=1024)
                sync(torch.device(self.config.device));timings.append(time.perf_counter()-st)
                for q in (0,1) if p=='wave' else (0,):
                    take=b.queries.valid[0]&(b.queries.quantity_id[0]==q);err=(pred[0,take]-b.queries.value[0,take]);scale=self.normalizers.scale(p,q)
                    obs=b.observations.valid[0]&(b.observations.quantity_id[0]==q);sensor_err=sensor[0,obs]-b.observations.value[0,obs]
                    bytime={str(float(t)):float((err[b.queries.physical_time[0,take,0]==t]/scale).square().mean()) for t in b.queries.physical_time[0,take,0].unique()}
                    rows.append({'pde':p,'quantity':'v' if q else 'u','family':g,'case_id':c.case_id,'standardized_MSE':float((err/scale).square().mean()),'physical_RMSE':float(err.square().mean().sqrt()),'sensor_standardized_MSE':float((sensor_err/scale).square().mean()),'time_standardized_MSE':bytime,'bank':ids[0]['bank']})
                if diagnostics:
                    # Recompute differentiable context for first coordinate derivatives.
                    contexts=self.model.prepare_context(b.specs,b.observations)
                    d=weak_loss(self.model,contexts,self.normalizers,self.adapter.weak)
                    physical.append({'pde':p,'family':g,**{k:float(v.detach()) for k,v in d.items() if k!='diagnostics'},'strong_residual':'not globally defined for C0/H1 path; weak residual is the valid physical diagnostic'})
                    del d,contexts
            value=float(np.mean([r['standardized_MSE'] for r in rows]));seconds=time.perf_counter()-start
            mid=state_identity(self.model);identity=digest((cases,self.normalizers.identity,self.certificate['certification_id'],self.config.validation_grid))
            report=ValidationReport(self.config.architecture,mid,identity,'geometry-held-out' if geometry_held_out else self.manifest.name,self.state.successful,{'common':{'field_normalized_MSE':value}},{'per_case_quantity':rows,'physical':physical,'inference_seconds':timings,'benchmark':BENCHMARK_ID,'scope':'fixed polygon instances; held-out families/topologies' if geometry_held_out else self.manifest.name},{'cases':len(cases)},seconds,self.config.device,'float32')
            if not geometry_held_out:
                self.state.validation_history.append({'step':self.state.successful,'value':value,'rows':rows,'checkpoint':mid,'manifest':identity});self.state.validation_seconds+=seconds
                if self.state.successful>0 and (self.state.best_validation_metric is None or value<self.state.best_validation_metric):self.state.best_validation_metric=value;self.state.best_validation_step=self.state.successful;self.best_validation_model_state=tree_cpu(self.model.state_dict())
            self._log('geometry_held_out.jsonl' if geometry_held_out else 'validation.jsonl',asdict(report));self._notify(validation=report);return report
        finally:self.model.train(previous)
    def save_checkpoint(self):
        start=time.monotonic();self._authorize();require(not self.state.accumulation_active and self.state.phase!='failed','checkpoint requires completed optimizer boundary')
        from ..diagnostics import source_identity
        meta={'artifact_type':'training_checkpoint','architecture_id':self.config.architecture,'run_id':self.config.run_id,'scientific_schema':SCHEMA,'benchmark':BENCHMARK_ID,'config':asdict(self.config),'model_config':asdict(self.model.config),'normalizers':self.normalizers.asdict(),'validation':asdict(self.manifest),'certificate':self.certificate,'execution_qualification':self.execution_certificate,'implementation_identity':implementation_identity(),'source':source_identity(Path(__file__).resolve().parents[3]),'runtime':__import__('pdeep.runtime',fromlist=['environment_report']).environment_report(),'successful_update':self.state.successful,'split_identity':split_manifest()['identity'],'experiment_kind':self.config.preset,'result_status':'untrained nonresearch verification artifact' if self.state.successful==0 or self.config.capacity=='verification' else 'manual full-train-split bounded experiment; finite per-case query banks; generalization requires evaluation' if self.config.cases_per_geometry is None else 'manual finite diagnostic pool; generalization requires evaluation','numerical_reference_certified':True}
        payload={'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'training_state':asdict(self.state),'rng':capture_rng(self.rng_generators),'sampler':self.sampler.state_dict(self.state.sampler_cursor),'best_validation_model':self.best_validation_model_state,'model_config':asdict(self.model.config),'normalizers':self.normalizers.asdict()}
        path=write_artifact(self.run_dir/'checkpoints',f'smooth-update-{self.state.successful:08d}',meta,payload,required=tuple(payload),idempotent=True)
        self.state.last_checkpoint=str(path);self.state.checkpoint_history.append({'step':self.state.successful,'path':str(path)})
        status=sync_verified(path,self.persistence_policy);self.state.persistence.update(local_status='verified',Drive_status=status['Drive_status'],last_sync=status)
        if status['Drive_status']=='failed':warnings.warn('DRIVE SYNC FAILED. Valid local checkpoint retained; Colab storage is temporary. '+status['error'],RuntimeWarning)
        atomic_json(self.run_dir/'persistence.json',self.state.persistence);self.state.saving_seconds+=time.monotonic()-start;self._notify();return path


def resume(path,run_root,*,device=None,persistence=None,monitor=None):
    record,payload=read_payload(path,kind='training_checkpoint',required=('model','optimizer','scheduler','training_state','rng','sampler','normalizers'))
    require(record.get('scientific_schema')==SCHEMA,'checkpoint is not G1_smooth_v1; old G1/T1 artifacts are never reinterpreted')
    require(record['runtime']['dependencies']['torch']==torch.__version__,'Torch version differs; exact resume is not certified')
    from .compatibility import approved_predecessor,execution_certificate
    require(approved_predecessor(record['implementation_identity']),'implementation/model/loss changed; exact resume requires matching source or the explicit verified input-loader repair')
    qualification=execution_certificate(record.get('execution_qualification',record['certificate']))
    cfg=record['config'];cfg={**cfg,**{k:tuple(cfg[k]) for k in ('pdes','probabilities','training_families') if k in cfg}}
    require(device is None or select_device(device)==cfg['device'],'exact continuation requires same device type; cross-device migration is a separate experiment')
    config=Config(**cfg);n=normalizers(record['certificate']);require(n.identity==payload['normalizers']['identity'],'checkpoint normalization mismatch')
    model=GeometryModel(ModelConfig(**payload['model_config']),n).to(config.device);model.load_state_dict(payload['model'],strict=True)
    engine=Engine(config,run_root,certificate=record['certificate'],execution_certificate=qualification,model=model,persistence=persistence,monitor=monitor,allow_existing_run=True)
    require(digest(record['validation'])==digest(asdict(engine.manifest)) and record['split_identity']==split_manifest()['identity'],'checkpoint dataset/config/validation mismatch')
    engine.optimizer.load_state_dict(payload['optimizer']);engine.scheduler.load_state_dict(payload['scheduler']);engine.state=TrainingState(**payload['training_state']);engine.state.phase='ready'
    require(digest(payload['sampler'])==digest(engine.sampler.state_dict(engine.state.sampler_cursor)),'checkpoint data ordering mismatch')
    engine.best_validation_model_state=payload.get('best_validation_model');engine.parent_checkpoints=[str(path)];engine._elapsed_base=engine.state.elapsed_seconds
    engine.rng_generators,limitations=restore_rng(payload['rng'],target_device=config.device);require(not limitations,'exact RNG continuation unavailable: '+str(limitations))
    return engine
