"""Specialize the existing G1 successful-update engine; no second update loop."""
from dataclasses import asdict
from pathlib import Path
import time
import torch
from ..core import require,digest
from ..artifact_io import read_payload,atomic_json
from ..artifact_tensors import tree_cpu
from ..artifact_rng import restore_rng
from ..train_types import TrainingState,ValidationReport
from ..unified.training import Engine as InverseEngine
from ..unified.config import implementation_identity
from ..unified_bayes.compatibility import approved_source
from .conditioning import SCHEMA
from .config import extension_identity
from .model import create
from .loss import Adapter
from .data import Sampler
from .dashboard import Dashboard
from .evaluation import evaluate
from .architecture import model_config,inventory,verify_definition
from .compatibility import approved_extension

class Engine(InverseEngine):
    def expected_model_config(self,config):return model_config(config.architecture,config.capacity)
    def __init__(self,config,dataset,root,*,monitor=None,persistence=None,allow_existing=False):
        import json
        registry=Path(root)/'runs'/config.experiment_id/'experiment.json'
        if registry.exists():
            existing=json.loads(registry.read_text())
            require(existing['dataset_identity']==dataset.config.identity and all(v.get('config',{}).get('conditioning_schema')==SCHEMA for v in existing['models'].values()),'existing experiment is inverse-only or has another data plan; use a NEW experiment ID, historical pointers preserved')
            previous=existing['models'].get(config.model_key)
            require(previous is None or digest(previous['config'])==config.identity,'experiment configuration changed; use a NEW ID instead of overwriting pointers')
        monitor=monitor or Dashboard(dataset.config);monitor.latest['current_model']=config.model_key;monitor.phase('Model construction',model=config.model_key,conditioning_schema=SCHEMA)
        model=create(config.architecture,dataset.norms,config.capacity,seed=config.root_seed,device=config.device)
        info=inventory(model);monitor.latest.update(model_capacity=config.capacity,trainable_parameters=info['trainable_parameters'],architecture_configuration=info['model_config'])
        super().__init__(config,dataset,root,monitor=monitor or Dashboard(dataset.config),persistence=persistence,model=model,allow_existing=allow_existing)
        self.adapter=Adapter(config);self.sampler=Sampler(config,dataset);self.state.stage=SCHEMA;self.state.rng_sampler_state=self.sampler.state_dict(0)
    def _authorize(self):
        require(self.config.conditioning_schema==SCHEMA and approved_extension(self.config.extension_source,self.config.capacity),'field-conditioning implementation/schema changed; exact resume rejected')
        return super()._authorize()
    def run_until(self,stop_after,*,launch=False):
        # Segment the EXISTING loop at validation/milestone boundaries. Its local
        # report list is released each segment instead of retaining100k reports.
        # No optimizer/update algorithm is duplicated; horizon/cursor never reset.
        require(launch,'explicit training launch required');require(0<stop_after<=self.config.updates,'invalid absolute session boundary')
        self.monitor.latest['session_stop']=stop_after
        milestones=(3000,15000,50000,100000)
        result=self.summary('already_at_requested_boundary')
        while self.state.successful<stop_after:
            next_validation=(self.state.successful//self.config.validation_every+1)*self.config.validation_every
            next_milestone=min((n for n in milestones if n>self.state.successful),default=stop_after)
            boundary=min(stop_after,next_validation,next_milestone)
            result=super().run_until(boundary,launch=True)
        return result
    def save_checkpoint(self):
        self._authorize();path=super().save_checkpoint()
        # Immutable verified history artifacts already contain these rows. Keep
        # raw local logs too, but rotate so the next archive reads only one window.
        if self.history_refs and self.history_refs[-1]['through_update']==self.state.successful:
            folder=self.run_dir/'history-log-shards';folder.mkdir(exist_ok=True)
            for name in ('steps.jsonl','case_losses.jsonl','validation.jsonl'):
                raw=self.run_dir/name
                if raw.exists():
                    target=folder/f'through-{self.state.successful:08d}-{name}'
                    require(not target.exists(),'raw history shard collision; checkpoint is valid and logs retained')
                    raw.rename(target)
        return path
    def validate(self):
        require(not self.state.accumulation_active,'validation only at successful update boundary');previous=self.state.phase;self.state.phase='validation';self._notify();start=time.perf_counter()
        try:
            r=evaluate(self.model,self.dataset,model_key=self.config.model_key,query_count=self.config.validation_query_count,max_cases_per_task=2,progress=self.monitor.phase)
            report=ValidationReport(self.config.architecture,__import__('pdeep.artifact_tensors',fromlist=['state_identity']).state_identity(self.model),digest((self.dataset.config.identity,SCHEMA,'validation')), 'unseen three-mode validation',self.state.successful,r['aggregate'],{'rows':r['rows']},{'physical_views':len(r['rows'])},time.perf_counter()-start,self.config.device,'float32')
            score=r['aggregate']['selection_standardized_MSE'];self.state.validation_seconds+=report.elapsed_seconds;self.state.validation_history.append({'step':self.state.successful,'value':score,'per_mode':r['aggregate']['per_mode']})
            if self.state.successful>0 and (self.state.best_validation_metric is None or score<self.state.best_validation_metric):self.state.best_validation_metric=score;self.state.best_validation_step=self.state.successful;self.best_validation_model_state=tree_cpu(self.model.state_dict())
            self._log('validation.jsonl',{'update':self.state.successful,**r});self._notify(validation=report);return report
        finally:self.state.phase=previous


def inspect_checkpoint(path,config,dataset):
    record,p=read_payload(path,kind='training_checkpoint',architecture=config.architecture,required=('model','optimizer','scheduler','scaler','training_state','rng','sampler','normalizers','best_validation_model'))
    require(record['model_config']==asdict(model_config(config.architecture,config.capacity)) and approved_source(config.source_id),'saved backbone/source configuration differs')
    require(record['config'].get('conditioning_schema')==SCHEMA,'inverse-only checkpoint lacks field encoder; choose explicit finetune_inverse, not resume')
    require(digest(record['config'])==config.identity,'configuration/horizon/initialization differs; no silent schedule stretching')
    require(record['config']['extension_source']==config.extension_source and approved_extension(config.extension_source,config.capacity) and approved_source(record['implementation_identity']),'source/schema changed')
    require(record['dataset_identity']==dataset.config.identity and p['normalizers']['identity']==dataset.norms.identity,'dataset/normalizers differ')
    require(record['certificate']['certification_id']==dataset.certificate['certification_id'] and record['runtime']['dependencies']['torch']==torch.__version__,'certificate/Torch differs')
    s=TrainingState(**p['training_state']);require(s.stage==SCHEMA and s.successful==record['successful_update']==p['scheduler']['completed'] and not s.accumulation_active,'invalid update/schema boundary')
    require(digest(p['sampler'])==digest(Sampler(config,dataset).state_dict(s.sampler_cursor)),'task/mode sampler changed');require(p['scaler'] is None,'verified FP32 path expects scaler=None')
    model=create(config.architecture,dataset.norms,config.capacity,seed=config.root_seed,device='cpu');verify_definition(record,model);expected=model.state_dict();require(set(expected)==set(p['model']),'field encoder/model state missing')
    for k,v in expected.items():require(v.shape==p['model'][k].shape and v.dtype==p['model'][k].dtype and bool(torch.isfinite(p['model'][k]).all()),'invalid model tensor '+k)
    groups=p['optimizer']['param_groups'];params=tuple(model.parameters());require(len(groups)==1 and len(groups[0]['params'])==len(params),'Adam parameter ordering differs')
    require(groups[0]['weight_decay']==config.weight_decay and groups[0]['lr']==p['scheduler']['next_lr'],'Adam/scheduler settings differ')
    for key,param in zip(groups[0]['params'],params):
        moment=p['optimizer']['state'].get(key)
        if moment:
            require(0<float(moment['step'])<=s.successful,'invalid Adam counter')
            for k in ('exp_avg','exp_avg_sq'):require(moment[k].shape==param.shape and bool(torch.isfinite(moment[k]).all()),'invalid Adam moment')
    return record,p


def resume(path,config,dataset,root,*,monitor=None,persistence=None):
    record,p=inspect_checkpoint(path,config,dataset);e=Engine(config,dataset,root,monitor=monitor,persistence=persistence,allow_existing=True)
    e.model.load_state_dict(p['model'],strict=True);e.optimizer.load_state_dict(p['optimizer']);e.scheduler.load_state_dict(p['scheduler']);e.state=TrainingState(**p['training_state']);e.state.phase='ready';e.state.last_checkpoint=str(path)
    e.history_refs=p.get('history_refs',[]);e.best_validation_model_state=p['best_validation_model'];e.parent_checkpoints=[str(path)];e._elapsed_base=e.state.elapsed_seconds
    e.monitor.restore_state(p.get('monitor_state'),run_id=e.state.run_id,successful=e.state.successful)
    e.rng_generators,limitations=restore_rng(p['rng'],target_device=config.device);require(not limitations,'exact RNG continuation unavailable: '+str(limitations));return e


def finetune_inverse(path,engine):
    """Explicit weights-only transfer; NEW optimizer/scheduler/RNG/update budget."""
    from ..geometric.models import create as old_create
    from ..unified.config import SCHEMA as OLD
    record,p=read_payload(path,kind='training_checkpoint',architecture=engine.config.architecture,required=('model','normalizers'))
    require(engine.config.initialization=='finetune_inverse' and engine.state.successful==0,'explicit fresh finetune experiment required')
    require(record['config']['experiment_id']!=engine.config.experiment_id,'fine-tuning requires a NEW experiment ID; preserve the parent registry')
    require(record.get('scientific_schema')==OLD and not record['config'].get('conditioning_schema'),'only inverse-only unified checkpoints accepted')
    require(approved_source(record['implementation_identity']) and p['normalizers']['identity']==engine.normalizers.identity and record['certificate']['certification_id']==engine.certificate['certification_id'],'incompatible legacy scientific source/prior/normalization/certificate')
    require(record['config']['capacity']==engine.config.capacity,'capacity differs')
    require(record['model_config']==asdict(engine.model.config),'architecture expansion is a separate experiment; full backbone definition differs')
    old=old_create(engine.config.architecture,engine.normalizers,engine.config.capacity,seed=0);old.load_state_dict(p['model'],strict=True)
    require(all(bool(torch.isfinite(v).all()) for v in p['model'].values()),'nonfinite legacy model')
    missing,unexpected=engine.model.load_state_dict(p['model'],strict=False);require(not unexpected and all(k.startswith(('known_field_encoder.','known_field_cross.')) for k in missing),'unexpected transfer mismatch')
    engine.parent_checkpoints=[str(path)];engine.monitor.phase('Fine-tuning initialization',parent=str(path),new_parameters=missing,optimizer_restored=False,scheduler_restored=False,old_updates=record['successful_update'],new_updates=0)
    return {'parent':str(path),'new_parameter_names':missing,'old_updates':record['successful_update'],'initialization':'weights-only transfer, new conditioning pathway and new optimizer/schedule; not exact resume'}


def train_models(configs,dataset,root,*,run_mode='preview',stop_after=50000,lookup='none',action='fresh',resume_paths=None,persistence=None,preflight_path=None,monitor=None):
    require(run_mode in ('preview','train') and action in ('fresh','resume','finetune_inverse') and lookup in ('none','explicit'),'invalid launch controls');paths=resume_paths or {}
    require(not paths or lookup=='explicit','exact checkpoint paths require explicit lookup')
    if run_mode=='preview':return {'mode':'preview','models':[c.model_key for c in configs],'modes':('FORWARD','INVERSE','HYBRID'),'mode_mixture':configs[0].mode_mixture,'main_target':configs[0].main_target,'planned_horizon':configs[0].updates,'session_stop':stop_after,'optimizer_steps':0,'reference_solves':0,'checkpoint_searches':0}
    require(dataset is not None and configs and len({c.model_key for c in configs})==len(configs),'prepared dataset/unique configurations required')
    require(action=='fresh' or lookup=='explicit','resume/finetune requires explicit paths');require(action!='fresh' or not paths,'fresh cannot load weights')
    from .preflight import require_preflight
    require_preflight(preflight_path,configs,dataset);dashboard=monitor or Dashboard(dataset.config);results={}
    if persistence and persistence.drive_enabled:require(not Path(dataset.root).resolve().is_relative_to(Path(persistence.drive_mount).resolve()),'restore data to fast local storage first')
    with dashboard.live():
        for i,c in enumerate(configs):
            require(0<stop_after<=c.updates,'session milestone exceeds immutable schedule');require((action=='finetune_inverse')==(c.initialization=='finetune_inverse') or action=='resume','initialization type mismatch')
            dashboard.latest.update(current_model=c.model_key,successful=0,losses={},weighted_losses={},gradient_norm=None,model_index=f'{i+1}/{len(configs)}',model_total=len(configs),completed_models=i,session_stop=stop_after,conditioning_schema=SCHEMA,mode_mixture=dict(zip(('FORWARD','INVERSE','HYBRID'),c.mode_mixture)),milestones=(3000,15000,50000,100000),main_target=c.main_target)
            if action=='resume' and c.model_key in paths:
                dashboard.phase('Checkpoint compatibility',model=c.model_key,exact_path=str(paths[c.model_key]))
                _,p=inspect_checkpoint(paths[c.model_key],c,dataset)
                dashboard.latest.update(successful=p['training_state']['successful'],last_verified_checkpoint=str(paths[c.model_key]))
                if p['training_state']['successful']>=stop_after:
                    s=p['training_state'];results[c.model_key]={'status':'already_at_requested_boundary','successful':s['successful'],'latest':str(paths[c.model_key]),'best':s['best_checkpoint'],'persistence':s['persistence']};continue
                engine=resume(paths[c.model_key],c,dataset,root,monitor=dashboard,persistence=persistence)
            else:
                require(action!='resume' or c.initialization!='finetune_inverse','a fine-tuned experiment needs its own unified checkpoint for resume; do not silently initialize missing parent weights')
                engine=Engine(c,dataset,root,monitor=dashboard,persistence=persistence)
                if action=='finetune_inverse':require(c.model_key in paths,'finetune requires one exact parent/model');finetune_inverse(paths[c.model_key],engine)
            results[c.model_key]=engine.run_until(stop_after,launch=True);del engine
            if torch.cuda.is_available():torch.cuda.empty_cache()
        dashboard.phase('Completed',models=len(results))
    return results
