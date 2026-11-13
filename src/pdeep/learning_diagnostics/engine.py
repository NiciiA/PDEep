"""Diagnostic specialization of smooth.Engine; SAME successful-update loop.

Only sampling, validation, objective selection and artifact metadata specialize.
FULL uses the original Adapter unchanged. FIELD_ONLY keeps supervised auxiliary
weight .1 and skips physics evaluation, without changing C's ModelConfig.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import time, warnings
import numpy as np
import torch
from ..core import require,digest
from ..smooth.training import Config,Engine,Manifest,implementation_identity,select_device
from ..geometric.physics import Adapter,supervised
from ..train_types import LossReport,ValidationReport,TrainingState
from ..artifact_tensors import tree_cpu,state_identity
from ..artifact_io import atomic_json,sha256,write_artifact,read_payload
from ..artifact_drive import sync_verified
from ..artifact_rng import capture_rng,restore_rng
from .data import FixedSampler,EXPERIMENTS
from .metrics import reconstruction,gradient_terms,representation

SCHEMA='G1_smooth-fixed-TRAIN-learning-diagnostics-v1'
QUERY_FIELDS=('query_revision','query_mode','query_time_policy','query_sampler_seed','query_chunk','query_checkpointing','amp_mode','resolved_amp_mode','memory_target_gb','synchronize_timings')
LEGACY_DIAGNOSTIC_SOURCE='484f281114c3528d89c218d6cb6e9e61ea7f429b8315eaca9658587661678192'

def approved_diagnostic(config):
    from .suite_compatibility import approved_diagnostic_predecessor
    return config.diagnostic_source_id==diagnostic_source() or (config.diagnostic_source_id==LEGACY_DIAGNOSTIC_SOURCE and config.query_revision=='legacy-fixed-v1' and config.query_mode=='fixed' and config.amp_mode=='off') or approved_diagnostic_predecessor(config)


def diagnostic_source():
    return digest({p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})

@dataclass(frozen=True)
class DiagnosticConfig(Config):
    updates: int=500
    root_seed: int=42
    bank_queries: int=128
    experiment: str='poisson_square'
    objective_variant: str='FULL'
    data_identity: str=''
    diagnostic_source_id: str=''
    validation_every: int=50
    checkpoint_every: int=100
    instrumentation_every: int=50
    query_revision: str='legacy-fixed-v1'
    query_mode: str='fixed'
    query_time_policy: str='stratified'
    query_sampler_seed: int=42
    query_chunk: int=512
    query_checkpointing: bool=True
    amp_mode: str='off'
    resolved_amp_mode: str='off'
    memory_target_gb: float=11.
    synchronize_timings: bool=False
    physics_weights: tuple | None=None  # weak, initial, boundary; explicit study controls
    physics_schedule: str='constant'
    gradient_diagnostics_every: int=0
    kinematic_weight: float=0.
    kinematic_queries: int=512
    kinematic_seed: int=42
    kinematic_revision: str='wave-kinematic-v1'
    @property
    def identity(self):
        values=asdict(self)
        for key,default in [('physics_weights',None),('physics_schedule','constant'),('gradient_diagnostics_every',0),('kinematic_weight',0.),('kinematic_queries',512),('kinematic_seed',42),('kinematic_revision','wave-kinematic-v1')]:
            if values[key]==default:values.pop(key)
        if self.query_revision=='legacy-fixed-v1':
            for k in QUERY_FIELDS:values.pop(k,None)
        return digest(values)
    def __post_init__(self):
        super().__post_init__()
        require(self.experiment in EXPERIMENTS,'unknown learning diagnostic')
        require(self.objective_variant in {'KIN_001','FULL','FIELD_ONLY','WEAK_001','ALL_001','ALL_01','FULL_01','STAGED_001','BASELINE','CLIP_2','CLIP_5','EARLY_TIME','EARLY_TIME_BEST_CLIP'},'unknown objective variant')
        require(self.physics_weights is None or len(self.physics_weights)==3 and all(np.isfinite(w) and w>=0 for w in self.physics_weights),'nonnegative finite physics weights required')
        if self.physics_weights is not None:object.__setattr__(self,'physics_weights',tuple(self.physics_weights))
        require(self.physics_schedule in {'constant','staged_300_700','staged_fractional_30_40_30_v1'} and self.gradient_diagnostics_every>=0,'unknown physics schedule/gradient cadence')
        require(self.physics_weights is None or self.architecture=='C_GEO','physics weight studies require C_GEO')
        require(self.physics_schedule=='constant' or self.physics_weights is not None,'staged weights must be explicit')
        require(self.objective_variant!='FIELD_ONLY' or self.architecture=='C_GEO','field-only ablation is C_GEO')
        require(np.isfinite(self.kinematic_weight) and self.kinematic_weight>=0 and 5<=self.kinematic_queries<=4096,'invalid kinematic weight/count')
        require(self.kinematic_revision=='wave-kinematic-v1','unknown kinematic sampler revision')
        require(not self.kinematic_weight or (self.architecture=='C_GEO' and tuple(self.pdes)==('wave',) and self.objective_variant=='KIN_001' and self.physics_weights==(0.,0.,0.) and self.physics_schedule=='constant' and self.query_revision!='legacy-fixed-v1'),'kinematic objective requires explicit isolated C wave variant')
        require(self.objective_variant!='KIN_001' or self.kinematic_weight>0,'KIN_001 needs a positive declared weight')
        require(self.instrumentation_every>0,'positive instrumentation cadence required')
        require(len(self.pdes)==len(self.training_families)==1 and self.effective_batch==1,'fixed single TRAIN case required')
        require(self.query_revision in {'legacy-fixed-v1','continuous-queries-v2','continuous-queries-v3'},'unknown query execution revision')
        require(self.query_mode in {'fixed','resampled','dense_resampled'} and self.query_time_policy in {'stratified','mixture','early_stratified_v1'},'unknown training query policy')
        require(self.query_revision!='legacy-fixed-v1' or self.query_mode=='fixed','legacy checkpoints cannot silently change sampling')
        require(2<=self.query_count<=65536 and 0<self.query_chunk<=65536 and 0<self.memory_target_gb<=12,'invalid bounded query/memory budget')
        require(not (self.query_revision=='continuous-queries-v3' and self.query_mode!='fixed' and self.query_time_policy=='stratified' and self.pdes[0]!='poisson') or self.query_count>=(16 if self.pdes[0]=='wave' else 8),'v3 dynamic stratification needs >=8 records per quantity')
        require(self.amp_mode in {'off','auto','bf16'} and self.resolved_amp_mode in {'off','bf16'},'unsupported neural AMP policy')
        if not self.diagnostic_source_id:object.__setattr__(self,'diagnostic_source_id',diagnostic_source())


class FieldOnly(Adapter):
    def loss(self,model,batch,*,detailed=False):
        cs=model.prepare_context(batch.specs,batch.observations);f,a=supervised(model,batch,cs)
        raw={'field':f,'auxiliary':a}; weighted={'field':f,'auxiliary':self.weak.auxiliary_weight*a}
        return LossReport(sum(weighted.values()),raw,weighted,{k:'explicit supervised target; certified quantity-specific training RMS' for k in raw},{'cases':len(batch.cases),'valid_queries':int(batch.queries.valid.sum()),'active_coefficients':0,'valid_observations':int(batch.observations.valid.sum())},{'physics_weights':{'weak':0.,'initial':0.,'boundary':0.},'architecture_physics_flag_unchanged':model.config.physics})


class DiagnosticEngine(Engine):
    def __init__(self,config,root,*,data,**kwargs):
        require(config.data_identity==data.identity,'fixed data identity differs from reviewed configuration')
        require(approved_diagnostic(config),'diagnostic source changed after preview')
        self.data=data
        super().__init__(config,Path(root)/'learning_diagnostics'/config.experiment/config.objective_variant,**kwargs)
        self.sampler=FixedSampler(data,config)
        if config.query_revision!='legacy-fixed-v1':
            from .queries import QuerySampler
            from .study_controls import configured_adapter
            self.sampler=QuerySampler(data,config)
            self.adapter=configured_adapter(config)
        self._configure_physics(0)
        self.latest_performance={};self.performance_preflight={};self._device_events={};self._device_phase=None
        if config.objective_variant=='FIELD_ONLY' and config.query_revision=='legacy-fixed-v1':self.adapter=FieldOnly(config.architecture,self.adapter.weak)
        self.manifest=Manifest(((data.case.pde,data.case.family,'train',data.case.index),),self.normalizers.identity,self.certificate['certification_id'],'fixed TRAIN memorization; independent queries, NOT generalization' if config.query_revision=='legacy-fixed-v1' else 'fixed TRAIN physical case; '+config.query_mode+' queries; independent coordinates, NOT physical-case generalization')
        self.state.rng_sampler_state=self.sampler.state_dict(self.state.sampler_cursor)
        atomic_json(self.run_dir/'fixed_data.json',{'identity':data.identity,'identities':data.identities,'bank':data.path,'case':data.case.manifest})
        atomic_json(self.run_dir/'validation_manifest.json',asdict(self.manifest))

    def _configure_physics(self,successful):
        if self.config.query_revision=='legacy-fixed-v1':return
        self.adapter.kinematic_cursor=successful
        from dataclasses import replace
        from .study_controls import physics_weights
        w=physics_weights(self.config,successful)
        self.adapter.weak=replace(self.adapter.weak,weak_weight=w[0],initial_weight=w[1],boundary_weight=w[2])
        self.adapter.field_only=self.config.architecture!='C_GEO' or not any(w)

    def _notify(self,step=None,validation=None):
        if hasattr(self,'_device_events') and self.config.query_revision!='legacy-fixed-v1' and self.config.device.startswith('cuda'):
            phase=self.state.phase
            if phase!=self._device_phase and phase in {'forward','backward','optimizer','ready'}:
                if phase=='forward':self._device_events={}
                self._device_events[phase]=torch.cuda.Event(enable_timing=True);self._device_events[phase].record()
            self._device_phase=phase
        return super()._notify(step,validation)

    def update(self,batch=None):
        self._configure_physics(self.state.successful)
        if hasattr(self.adapter,'kinematic_cursor'):self.adapter.kinematic_cursor=self.state.sampler_cursor
        if self.config.gradient_diagnostics_every and self.state.successful%self.config.gradient_diagnostics_every==0:
            self._log('gradient_terms.jsonl',{'successful_update':self.state.successful,'physics_weights':__import__('pdeep.learning_diagnostics.study_controls',fromlist=['physics_weights']).physics_weights(self.config,self.state.successful),'report':gradient_terms(self.model,self.data.batch(self.config.device),self.adapter)})
        if self.config.query_revision!='legacy-fixed-v1' and self.config.device.startswith('cuda'):
            require(self.performance_preflight.get('config_identity')==self.config.identity,'CUDA training requires the matching bounded memory/AMP preflight before updates')
            torch.cuda.reset_peak_memory_stats(self.config.device)
        capture=self.state.successful%self.config.instrumentation_every==0
        before={n:p.detach().clone() for n,p in self.model.named_parameters()} if capture else {}
        start=time.perf_counter()
        if self.config.query_revision=='legacy-fixed-v1':report=super().update(batch)
        else:
            from .performance import asynchronous_update,memory,utilization
            from .runtime import using
            self._authorize()
            try:
                with using(self.adapter.cache):report=asynchronous_update(self,batch)
            finally:self.adapter.cache.clear_queries()
            seconds=time.perf_counter()-start
            spans={}
            for a,b in [('forward','backward'),('backward','optimizer'),('optimizer','ready')]:
                if a in self._device_events and b in self._device_events and self._device_events[b].query():spans[a]=self._device_events[a].elapsed_time(self._device_events[b])/1000
            self.latest_performance={'CUDA_event_phase_span_seconds':spans,'CUDA_event_scope':'stream elapsed spans include launch/CPU gaps; not kernel-active utilization','memory':memory(self.config.device),'utilization':utilization(self.config.device),'queries_per_update':report.counts['valid_queries'],'queries_per_second':report.counts['valid_queries']/seconds,'step_wall_seconds':seconds,'phase_timing_scope':'CPU wall/enqueue; CUDA phase timings only in isolated preflight','reference':self.sampler.last,'cache':self.adapter.last,'gradient_before_clip':report.gradient_before_clip,'gradient_after_clip':report.gradient_after_clip,'clipping_factor':report.gradient_after_clip/report.gradient_before_clip if report.gradient_before_clip else 1.,'physics_weights':__import__('pdeep.learning_diagnostics.study_controls',fromlist=['physics_weights']).physics_weights(self.config,max(0,self.state.successful-1)),'AMP_gradient_policy':'BF16 autocast or FP32; no GradScaler, gradients already in physical parameter scale before clipping','clipped_fraction':sum(r['gradient_before_clip']>self.config.grad_clip_norm for r in self.state.recent_steps)/max(1,len(self.state.recent_steps))}
            self._log('performance.jsonl',self.latest_performance);self.monitor.performance=self.latest_performance;self._notify(report)
        if capture:
            rows={}
            for n,p in self.model.named_parameters():
                group=n.split('.')[0];r=rows.setdefault(group,{'parameter_squared':0.,'update_squared':0.,'gradient_squared':0.})
                r['parameter_squared']+=float(before[n].square().sum());r['update_squared']+=float((p.detach()-before[n]).square().sum())
                r['gradient_squared']+=0 if p.grad is None else float(p.grad.square().sum())
            self._log('module_updates.jsonl',{'step':self.state.successful,'modules':{k:{n.replace('_squared','_norm'):v**.5 for n,v in r.items()} for k,r in rows.items()},'gradient_scope':'after engine clipping; per-loss unclipped norms in gradient_terms.json'})
        return report

    def validate(self,**kwargs):
        require(not self.state.accumulation_active,'validation only at optimizer boundary')
        start=time.perf_counter()
        if self.config.query_revision=='legacy-fixed-v1':m=reconstruction(self.model,self.data)
        else:
            from ..geometric.data import Batch
            from .runtime import using
            b=Batch((self.data.case,),self.sampler.observations,self.sampler.fixed_queries,self.normalizers)
            d=Batch((self.data.case,),self.sampler.observations,self.sampler.independent,self.normalizers)
            try:
                with using(self.adapter.cache):m=reconstruction(self.model,self.data,prepared_batches=(b,d))
            finally:self.adapter.cache.clear_queries()
        rows=[{'pde':self.data.case.pde,'family':self.data.case.family,'case_id':self.data.case.case_id,'quantity':q,**r} for q,r in m['field'].items()]
        value=float(np.mean([r['standardized_MSE'] for r in rows])); mid=state_identity(self.model)
        report=ValidationReport(self.config.architecture,mid,self.manifest.identity,self.manifest.name,self.state.successful,{'common':{'field_normalized_MSE':value}},m,{'cases':1},time.perf_counter()-start,self.config.device,'float32')
        self.state.validation_history.append({'step':self.state.successful,'value':value,'rows':rows,'checkpoint':mid,'manifest':self.manifest.identity})
        self.state.validation_seconds+=report.elapsed_seconds
        if self.state.successful>0 and (self.state.best_validation_metric is None or value<self.state.best_validation_metric):
            self.state.best_validation_metric=value;self.state.best_validation_step=self.state.successful;self.best_validation_model_state=tree_cpu(self.model.state_dict())
        self._log('validation.jsonl',asdict(report));self.monitor.diagnostic_metrics=m
        self._notify(validation=report);return report

    def save_checkpoint(self):
        self._authorize(); require(not self.state.accumulation_active and self.state.phase!='failed','completed optimizer boundary required')
        start=time.monotonic()
        from ..runtime import environment_report
        from ..diagnostics import source_identity
        meta={'artifact_type':'training_checkpoint','architecture_id':self.config.architecture,'run_id':self.config.run_id,'scientific_schema':SCHEMA,'benchmark':self.config.benchmark,'config':asdict(self.config),'certificate':self.certificate,'execution_qualification':self.execution_certificate,'implementation_identity':implementation_identity(),'diagnostic_source_id':self.config.diagnostic_source_id,'executing_diagnostic_source_id':diagnostic_source(),'fixed_data_identity':self.data.identity,'fixed_case':self.data.case.manifest,'fixed_data_path':self.data.path,'runtime':environment_report(),'source':source_identity(Path(__file__).resolve().parents[3]),'successful_update':self.state.successful,'result_status':'TRAIN memorization only; not research/generalization certification'}
        payload={'model':self.model.state_dict(),'model_config':asdict(self.model.config),'normalizers':self.normalizers.asdict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'training_state':asdict(self.state),'rng':capture_rng(self.rng_generators),'sampler':self.sampler.state_dict(self.state.sampler_cursor),'best_validation_model':self.best_validation_model_state}
        path=write_artifact(self.run_dir/'checkpoints',f'diagnostic-update-{self.state.successful:08d}',meta,payload,required=tuple(payload),idempotent=True)
        self.state.last_checkpoint=str(path);self.state.checkpoint_history.append({'step':self.state.successful,'path':str(path)})
        status=sync_verified(path,self.persistence_policy);self.state.persistence.update(local_status='verified',Drive_status=status['Drive_status'],last_sync=status)
        if status['Drive_status']=='failed':warnings.warn('DRIVE SYNC FAILED; valid local diagnostic checkpoint retained. Colab storage is temporary. '+status['error'],RuntimeWarning)
        # Preserve complete disk histories separately from the resumable tensor state.
        history={name:(self.run_dir/name).read_text() for name in ('steps.jsonl','validation.jsonl','performance.jsonl','module_updates.jsonl','gradient_terms.json','gradient_terms.jsonl','fixed_data.json') if (self.run_dir/name).exists()}
        history_path=write_artifact(self.run_dir/'history',f'history-{self.state.successful:08d}',{**meta,'artifact_type':'diagnostic_bundle'},{'history':history},required=('history',),idempotent=True)
        hs=sync_verified(history_path,self.persistence_policy);self.state.persistence['history_sync']=hs
        if hs['Drive_status']=='failed':warnings.warn('DRIVE HISTORY SYNC FAILED; local complete histories retained: '+hs['error'],RuntimeWarning)
        atomic_json(self.run_dir/'persistence.json',self.state.persistence);self.state.saving_seconds+=time.monotonic()-start;self._notify();return path

    def export(self):
        from ..diagnostics import source_identity
        meta={'artifact_type':'inference_export','architecture_id':self.config.architecture,'run_id':self.config.run_id,'scientific_schema':SCHEMA,'benchmark':self.config.benchmark,'config':asdict(self.config),'fixed_case':self.data.case.manifest,'certificate':self.certificate,'fixed_data_identity':self.data.identity,'implementation_identity':implementation_identity(),'diagnostic_source_id':diagnostic_source(),'source':source_identity(Path(__file__).resolve().parents[3])}
        payload={'model':self.model.state_dict(),'model_config':asdict(self.model.config),'normalizers':self.normalizers.asdict()}
        path=write_artifact(self.run_dir/'inference',f'inference-{self.state.successful:08d}',meta,payload,required=tuple(payload),idempotent=True)
        status=sync_verified(path,self.persistence_policy);atomic_json(self.run_dir/'inference_sync.json',status)
        if status['Drive_status']=='failed':warnings.warn('DRIVE INFERENCE EXPORT SYNC FAILED; local export retained: '+status['error'],RuntimeWarning)
        # Persist detailed final diagnostics separately from inference weights.
        reports={p.name:p.read_text() for p in self.run_dir.glob('*.json') if p.name not in {'config.json','persistence.json','diagnostic_sync.json','inference_sync.json'}}
        bundle=write_artifact(self.run_dir/'diagnostics',f'final-report-{self.state.successful:08d}',{**meta,'artifact_type':'diagnostic_bundle'},{'reports':reports,'validation':None if self.latest_validation is None else asdict(self.latest_validation)},required=('reports','validation'),idempotent=True)
        ds=sync_verified(bundle,self.persistence_policy);atomic_json(self.run_dir/'diagnostic_sync.json',ds)
        if ds['Drive_status']=='failed':warnings.warn('DRIVE DIAGNOSTICS SYNC FAILED; local report retained: '+ds['error'],RuntimeWarning)
        return path


def resume_diagnostic(path,root,*,data,certificate,config,persistence=None,monitor=None):
    meta,p=read_payload(path,kind='training_checkpoint',architecture=config.architecture,required=('model','optimizer','scheduler','training_state','sampler','rng','normalizers','model_config'))
    require(meta.get('scientific_schema')==SCHEMA,'select a diagnostic checkpoint; mixed-pool checkpoints are not implicit warm starts')
    saved={**meta['config'],**{k:tuple(meta['config'][k]) for k in ('pdes','probabilities','training_families')}}
    require(asdict(DiagnosticConfig(**saved))==asdict(config),'diagnostic resume configuration mismatch')
    from ..smooth.compatibility import approved_predecessor
    require(approved_diagnostic(config) and meta['diagnostic_source_id']==config.diagnostic_source_id and approved_predecessor(meta['implementation_identity']),'exact resume implementation mismatch')
    require(meta['fixed_data_identity']==data.identity,'fixed case/observations/queries differ')
    require(meta['execution_qualification']['certification_id']==certificate['certification_id'],'resume numerical runtime/certificate mismatch')
    require(meta['runtime']['dependencies']['torch']==torch.__version__,'exact resume Torch version mismatch')
    from ..geometric.models import GeometryModel,ModelConfig
    model=GeometryModel(ModelConfig(**p['model_config']),data.normalizers).to(config.device);model.load_state_dict(p['model'],strict=True)
    engine=DiagnosticEngine(config,root,data=data,certificate=meta['certificate'],execution_certificate=certificate,model=model,persistence=persistence,monitor=monitor,allow_existing_run=True)
    engine.optimizer.load_state_dict(p['optimizer']);engine.scheduler.load_state_dict(p['scheduler']);engine.state=TrainingState(**p['training_state']);engine.state.phase='ready' if engine.state.successful<config.updates else 'complete'
    engine.state.last_checkpoint=str(path);engine.state.persistence['local_status']='verified'
    if not engine.state.checkpoint_history or engine.state.checkpoint_history[-1]['step']!=engine.state.successful:engine.state.checkpoint_history.append({'step':engine.state.successful,'path':str(path)})
    require(digest(p['sampler'])==digest(engine.sampler.state_dict(engine.state.sampler_cursor)),'fixed sampler resume mismatch')
    engine._configure_physics(engine.state.successful)
    if hasattr(engine.adapter,'kinematic_cursor'):engine.adapter.kinematic_cursor=engine.state.sampler_cursor
    engine.best_validation_model_state=p.get('best_validation_model');engine.parent_checkpoints=[str(path)];engine._elapsed_base=engine.state.elapsed_seconds
    engine.rng_generators,limitations=restore_rng(p['rng'],target_device=config.device);require(not limitations,'exact RNG restore unavailable: '+str(limitations))
    return engine
