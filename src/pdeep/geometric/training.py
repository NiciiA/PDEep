"""Geometry specialization of the EXISTING successful-update engine.

No second optimizer loop: TrainingEngine.update/run/scheduler/logging/monitor are
inherited unchanged. Only data, scientific adapter, validation and versioned
checkpoint metadata differ. T1 source fingerprints/checkpoint semantics are intact.
"""
from dataclasses import dataclass,asdict,fields
from pathlib import Path
import json,time,warnings
import numpy as np
import torch
from ..core import require,digest
from ..train_types import ResolvedTrainConfig,TrainingState,ValidationReport
from ..training import TrainingEngine,SuccessfulScheduler
from ..train_monitor import LiveMonitor
from ..artifact_drive import PersistencePolicy,sync_verified
from ..artifact_io import write_artifact,read_payload,atomic_json
from ..artifact_rng import capture_rng,restore_rng
from .data import VERSION,PDES,physical_case,batch,rng,Normalizers
from .models import create,GeometryModel,ModelConfig
from .physics import Adapter,WeakConfig


@dataclass(frozen=True)
class Config(ResolvedTrainConfig):
    benchmark: str='G1'
    inverse_track: str='G1'
    capacity: str='medium'
    microbatch: int=1
    accumulation: int=1
    effective_batch: int=1
    reference_h: float=.04
    reference_dt: float=.002
    sensors: int | None=None
    spatial_order: int=3
    time_order: int=3
    weak_tests: int=16
    physics_chunk: int=256
    def __post_init__(self):
        require(self.architecture in {'A_GEO','C_GEO'} and self.benchmark==self.inverse_track=='G1','explicit G1/A_GEO/C_GEO configuration required')
        require(self.capacity in {'medium','large','verification'},'unknown geometry capacity')
        # Reuse all established budget/device/precision/accumulation validation.
        keys={f.name for f in fields(ResolvedTrainConfig)};kwargs={k:v for k,v in asdict(self).items() if k in keys}
        kwargs.update(architecture='A' if self.architecture=='A_GEO' else 'C',benchmark='unit_square_modal_v1',inverse_track='T1',capacity='canonical_v1')
        base=ResolvedTrainConfig(**kwargs)
        for f in fields(base):
            if f.name not in {'architecture','benchmark','inverse_track','capacity'}:object.__setattr__(self,f.name,getattr(base,f.name))
        require(self.reference_h>0 and self.reference_dt>0 and (self.sensors is None or 0<=self.sensors<=256),'invalid G1 discretization/sensors')
        require(self.spatial_order>=2 and self.time_order>=2 and self.weak_tests>0 and self.physics_chunk>0,'invalid G1 weak quadrature')


@dataclass(frozen=True)
class Manifest:
    cases: tuple
    normalizer_identity: str
    name: str='G1-ID-validation-v1'
    @property
    def identity(self):return digest(asdict(self))


class Sampler:
    def __init__(self,config,normalizers):self.config=config;self.normalizers=normalizers
    def state_dict(self,cursor):return {'version':VERSION,'cursor':cursor,'root_seed':self.config.root_seed,'config':self.config.identity,'recipe':'stateless keyed order/PDE/field/observations/queries'}
    def batch(self,cursor):
        c=self.config;s=rng(c.root_seed,'training_order',cursor);cases=[]
        choices=c.training_case_ids or tuple(range(12000))
        for _ in range(c.effective_batch):
            pde=s.choice(c.pdes,p=c.probabilities);index=int(s.choice(choices));cases.append(physical_case('train',str(pde),index,h=c.reference_h,dt=c.reference_dt))
        b=batch(cases,self.normalizers,sensors=c.sensors,queries=c.query_count,set_id=cursor,device=c.device)
        return b,{'physical_cases':b.case_ids,'observation_query_set':cursor,'version':VERSION}


def register():
    from ..train_registry import REGISTRY,ArchitectureRegistration
    for name in ('A_GEO','C_GEO'):
        REGISTRY[name]=ArchitectureRegistration(name,True,tracks=('G1',),inputs=('GeometryProblem','ObservationTokens','QueryTokens'),outputs=('field','auxiliary_fields'),coordinate_derivatives=name=='C_GEO',checkpoint_version='geometry-inverse-v1/pdeep-artifact-v2',objective='field + .1 auxiliary'+(' + .1 weak + .1 initial + .1 boundary' if name=='C_GEO' else ''),objective_implementation='pdeep.geometric.physics.Adapter')


class Engine(TrainingEngine):
    def run(self,*,launch_full=False):
        # G1 v1 references are not certified at the observation-noise floor.
        # Enforce this in the shared-engine entry point, not just a notebook UI.
        require(self.config.preset=='smoke','STOP: G3 canonical-P1 wave reference fidelity is BLOCKED; geometry research/bounded training cannot launch')
        return super().run(launch_full=launch_full)

    def __init__(self,config,normalizers,run_root,*,monitor=None,model=None,persistence=None,allow_existing_run=False):
        register();self.config=config;self.normalizers=normalizers;self.state=TrainingState(config.architecture,config.run_id,stage='phaseG')
        self.monitor=LiveMonitor(enabled=False) if monitor is None else monitor
        self.run_dir=Path(run_root)/config.architecture/config.run_id
        require(allow_existing_run or not self.run_dir.exists(),'G1 run directory exists; select resume or new run_tag')
        self.run_dir.mkdir(parents=True,exist_ok=True);self._started=time.monotonic();self._elapsed_base=0.;self.latest_step=None;self.latest_validation=None
        self.persistence_policy=persistence or PersistencePolicy();self.rng_generators={};self.parent_checkpoints=[];self.scaler=None;self.best_validation_model_state=None
        self.model=model or create(config.architecture,normalizers,config.capacity,seed=config.root_seed,device=config.device)
        require(self.model.config.architecture==config.architecture and self.model.config.capacity==config.capacity,'G1 model/config mismatch')
        self.adapter=Adapter(config.architecture,WeakConfig(config.spatial_order,config.time_order,config.weak_tests,config.physics_chunk))
        self.sampler=Sampler(config,normalizers);self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay);self.scheduler=SuccessfulScheduler(self.optimizer,config)
        cases=tuple((p,i) for p in config.pdes for i in range(config.validation_cases_per_pde))
        self.manifest=Manifest(cases,normalizers.identity,config.validation_name)
        self.state.rng_sampler_state=self.sampler.state_dict(0);self.state.scheduler_state=self.scheduler.state_dict();self.state.phase='ready'
        self.capability={'policy':'actual production adapter forward/backward on first bounded update; explicit device, no fallback'}
        if not allow_existing_run:
            atomic_json(self.run_dir/'config.json',asdict(config));atomic_json(self.run_dir/'validation_manifest.json',{'identity':self.manifest.identity,**asdict(self.manifest)})

    def validate(self):
        require(not self.state.accumulation_active,'validation at optimizer boundary only');start=time.perf_counter();previous=self.model.training;self.model.eval();rows=[]
        try:
            with torch.no_grad():
                for p,i in self.manifest.cases:
                    c=physical_case('validation',p,i,h=self.config.reference_h,dt=self.config.reference_dt);b=batch((c,),self.normalizers,sensors=self.config.sensors,queries=self.config.validation_query_count,set_id=917,device=self.config.device)
                    pred=self.model(b.specs,b.observations,b.queries);values=[]
                    for q in (0,1) if p=='wave' else (0,):
                        select=b.queries.valid&(b.queries.quantity_id==q)
                        if select.any():values.append(float(((pred[select]-b.queries.value[select])/self.normalizers.scale(p,q)).square().mean()))
                    rows.append({'pde':p,'field_normalized_MSE':float(np.mean(values)),'case_id':c.case_id,'mesh_id':c.problem.mesh.mesh_id})
            metric=float(np.mean([r['field_normalized_MSE'] for r in rows]));seconds=time.perf_counter()-start
            from ..artifact_tensors import state_identity
            report=ValidationReport(self.config.architecture,state_identity(self.model),self.manifest.identity,self.manifest.name,self.state.successful,{'common':{'field_normalized_MSE':metric}},{'per_case':rows,'benchmark':'G1'}, {'cases':len(rows)},seconds,self.config.device,'float32')
            self.state.validation_history.append({'step':self.state.successful,'manifest':self.manifest.identity,'value':metric});self.state.validation_seconds+=seconds
            if self.state.best_validation_metric is None or metric<self.state.best_validation_metric:
                from ..artifact_tensors import tree_cpu
                self.state.best_validation_metric=metric;self.state.best_validation_step=self.state.successful;self.best_validation_model_state=tree_cpu(self.model.state_dict())
            self._log('validation.jsonl',asdict(report));self._notify(validation=report);return report
        finally:self.model.train(previous)

    def save_checkpoint(self):
        require(not self.state.accumulation_active and self.state.phase!='failed','G1 checkpoint requires completed optimizer boundary')
        from ..artifact_metadata import scientific_identity
        from ..diagnostics import source_identity
        from ..artifact_tensors import tree_cpu
        mesh_manifests=[physical_case('validation',p,i,h=self.config.reference_h,dt=self.config.reference_dt).manifest for p,i in self.manifest.cases]
        meta={'artifact_type':'training_checkpoint','architecture_id':self.config.architecture,'run_id':self.config.run_id,
              'scientific_schema':'geometry-inverse-v1','benchmark':'G1','field_prior':VERSION,'config':asdict(self.config),'model_config':asdict(self.model.config),
              'normalizers':self.normalizers.asdict(),'validation':{'identity':self.manifest.identity,**asdict(self.manifest)},'meshes':mesh_manifests,
              'legacy_scientific_identity':scientific_identity(),'geometry_source':geometry_identity(),'source':source_identity(Path(__file__).resolve().parents[3]),
              'successful_update':self.state.successful,'boundary_schema':'G1-known-homogeneous-Dirichlet','temporal_output':'Hermite-C1',
              'device':self.config.device,'precision':'float32'}
        import scipy,shapely,triangle,platform
        meta['dependency_versions']={'Python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'shapely':shapely.__version__,'triangle':triangle.__version__}
        payload={'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),
                 'training_state':asdict(self.state),'rng':capture_rng(self.rng_generators),'sampler':self.sampler.state_dict(self.state.sampler_cursor),
                 'best_validation_model':self.best_validation_model_state,'model_config':asdict(self.model.config),'normalizers':self.normalizers.asdict()}
        path=write_artifact(self.run_dir/'checkpoints',f'geometry-update-{self.state.successful:08d}',meta,payload,required=tuple(payload),idempotent=True)
        self.state.last_checkpoint=str(path);self.state.checkpoint_history.append({'step':self.state.successful,'path':str(path)})
        status=sync_verified(path,self.persistence_policy);self.state.persistence.update(local_status='verified',Drive_status=status['Drive_status'],last_sync=status)
        if status['Drive_status']=='failed':warnings.warn('DRIVE SYNC FAILED; local checkpoint retained; Colab local storage is temporary: '+status['error'],RuntimeWarning)
        atomic_json(self.run_dir/'persistence.json',self.state.persistence);self._notify();return path


def geometry_identity():
    from ..artifact_io import sha256
    directory=Path(__file__).parent
    return {p.name:sha256(p) for p in sorted(directory.glob('*.py'))}


def resume(path,run_root,*,device=None,persistence=None):
    record,payload=read_payload(path,kind='training_checkpoint',required=('model','optimizer','scheduler','training_state','rng','sampler','model_config','normalizers'))
    require(record.get('scientific_schema')=='geometry-inverse-v1' and record['architecture_id'] in {'A_GEO','C_GEO'},'use explicit geometry checkpoint; legacy A/C are incompatible')
    require(record['field_prior']==VERSION and record['geometry_source']==geometry_identity(),'G1 source/prior changed; exact continuation not authorized')
    cfg=dict(record['config'])
    for k in ('pdes','probabilities','training_case_ids'):
        if cfg.get(k) is not None:cfg[k]=tuple(cfg[k])
    original_device=cfg['device']
    if device is not None:cfg['device']=device
    require(cfg['device']==original_device,'exact G1 resume requires same device type; cross-device migration must be a new experiment')
    config=Config(**cfg);n=Normalizers.fromdict(payload['normalizers']);m=GeometryModel(ModelConfig(**payload['model_config']),n).to(config.device)
    m.load_state_dict(payload['model'],strict=True);engine=Engine(config,n,run_root,model=m,persistence=persistence,allow_existing_run=True)
    require(record['validation']['identity']==engine.manifest.identity and record['normalizers']['identity']==n.identity,'G1 resume data/config mismatch')
    engine.optimizer.load_state_dict(payload['optimizer']);engine.scheduler.load_state_dict(payload['scheduler']);engine.state=TrainingState(**payload['training_state']);engine.state.phase='ready'
    require(payload['sampler']==engine.sampler.state_dict(engine.state.sampler_cursor),'G1 sampler ordering mismatch')
    engine.best_validation_model_state=payload.get('best_validation_model');engine.parent_checkpoints=[str(path)];engine.rng_generators,limitations=restore_rng(payload['rng'],target_device=config.device)
    require(not limitations,'exact resume RNG limitations: '+str(limitations));return engine
