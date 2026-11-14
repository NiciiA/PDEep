"""Successful-update engine specialization; shared AdamW/update code unchanged."""
from dataclasses import asdict
from pathlib import Path
import json,time,uuid
import torch
from ..core import digest,require
from ..training import TrainingEngine,SuccessfulScheduler
from ..train_types import TrainingState,ValidationReport
from ..train_monitor import LiveMonitor
from ..artifact_io import atomic_json,write_artifact,read_payload,read_manifest
from ..artifact_tensors import state_identity,tree_cpu
from ..artifact_rng import capture_rng,restore_rng
from ..artifact_drive import PersistencePolicy
from ..geometric.models import create,GeometryModel,ModelConfig
from ..smooth.certification import authorize
from ..runtime import environment_report
from ..diagnostics import source_identity
from ..learning_diagnostics.performance import asynchronous_update
from .config import SCHEMA,TrainConfig,implementation_identity
from .data import BalancedSampler
from .loss import UnifiedAdapter
from .evaluation import evaluate_model
from .monitor import Dashboard
from .persistence import policy_at,sync


class _SessionBoundary(Exception):pass


class Engine(TrainingEngine):
    def __init__(self,config,dataset,root,*,monitor=None,persistence=None,model=None,allow_existing=False):
        require(config.dataset_identity==dataset.config.identity,'training/dataset identity mismatch')
        require(config.source_id==implementation_identity(),'source changed after preview/configuration')
        self.monitor=monitor or Dashboard(dataset.config,notebook=False)
        self.monitor.phase('Dataset cache loading',model=config.model_key)
        dataset.progress=self.monitor.phase
        if hasattr(dataset,'index'):
            require(config.updates==dataset.config.stream_updates and config.query_count==dataset.config.stream_queries and config.data_seed==dataset.config.seed and config.validation_query_count==dataset.config.stream_validation_queries,'training differs from the exact prepared continuous stream plan')
        dataset.require_ready('train');dataset.require_ready('validation')
        self.config=config;self.dataset=dataset;self.normalizers=dataset.norms;self.certificate=dataset.certificate
        self._authorize()
        self.state=TrainingState(config.architecture,config.run_id,stage=SCHEMA)
        self.run_dir=Path(root)/'runs'/config.experiment_id/config.model_key.replace('/','_')/config.run_id
        require(allow_existing or not self.run_dir.exists(),'fresh run exists; use explicit resume or a NEW experiment ID')
        self.run_dir.mkdir(parents=True,exist_ok=True);self._started=time.monotonic();self._elapsed_base=0.;self.latest_step=None;self.latest_validation=None
        self.persistence_policy=policy_at(persistence or PersistencePolicy(),'runs',config.experiment_id,config.model_key.replace('/','_'))
        self.rng_generators={};self.parent_checkpoints=[];self.scaler=None;self.best_validation_model_state=None;self.session_stop=config.updates;self.history_refs=[]
        self.monitor.phase('Model construction',model=config.model_key)
        self.model=model if model is not None else create(config.architecture,self.normalizers,config.capacity,seed=config.root_seed,device=config.device)
        require(self.model.config==ModelConfig.preset(config.architecture,config.capacity),'architecture was modified')
        require(self.model.normalizers.identity==self.normalizers.identity,'model normalization mismatch')
        self.adapter=UnifiedAdapter(config);self.sampler=BalancedSampler(config,dataset)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        self.scheduler=SuccessfulScheduler(self.optimizer,config)
        self.state.scheduler_state=self.scheduler.state_dict();self.state.rng_sampler_state=self.sampler.state_dict(0);self.state.phase='ready'
        self.capability={'source':config.source_id,'reference':'per-case QC and stratified qualification required','AMP':'FP32; scaler=None'}
        if not allow_existing:atomic_json(self.run_dir/'config.json',asdict(config))
    def _authorize(self):
        return authorize(self.certificate,benchmark=self.config.benchmark,pdes=self.dataset.config.pdes,families=self.dataset.config.families,normalizer_identity=self.normalizers.identity)
    def update(self,batch=None):
        if self.state.successful>=self.session_stop:raise _SessionBoundary()
        self._update_start=time.perf_counter()
        self._authorize();self.adapter.cursor=self.state.sampler_cursor;self.adapter.window=[];self.adapter.measure_gradients=(self.state.successful+1)%500==0
        # Exact inherited optimizer/update bytecode; explicit timing barriers removed,
        # safety checks retained. CUDA profiling synchronization belongs to preflight.
        result=asynchronous_update(self,batch)
        self._log('case_losses.jsonl',{'update':self.state.successful,'microbatches':self.adapter.window})
        return result
    def _log(self,name,value):
        if name=='steps.jsonl' and hasattr(self,'_update_start'):
            self._last_update_wall=time.perf_counter()-self._update_start
            value.setdefault('details',{})['measured_update_wall_seconds']=self._last_update_wall
        return TrainingEngine._log(self,name,value)
    def _notify(self,step=None,validation=None):
        if step is not None and hasattr(self,'_last_update_wall'):
            step.details['measured_update_wall_seconds']=self._last_update_wall
            if self.state.recent_steps:self.state.recent_steps[-1]['details']['measured_update_wall_seconds']=self._last_update_wall
        return TrainingEngine._notify(self,step,validation)
    def run_until(self,stop_after,*,launch=False):
        require(launch,'training requires explicit launch=True; preview never constructs an engine')
        require(type(stop_after) is int and 0<stop_after<=self.config.updates,'invalid absolute session boundary')
        if self.state.successful>=stop_after:return self.summary('already_at_requested_boundary')
        self.session_stop=stop_after;self.state.phase='ready';self.monitor.phase('Training',model=self.config.model_key,stop_after=stop_after,schedule_budget=self.config.updates)
        try:TrainingEngine.run(self,launch_full=True)
        except _SessionBoundary:
            # The full scheduler still references immutable 30k config. No extra
            # attempted update, no scheduler extension, no optimizer at this boundary.
            if not self.state.validation_history or self.state.validation_history[-1]['step']!=self.state.successful:self.validate()
            if not self.state.checkpoint_history or self.state.checkpoint_history[-1]['step']!=self.state.successful:self.save_checkpoint()
            self.state.phase='ready';self._notify()
        return self.summary('complete' if self.state.successful==self.config.updates else 'planned_session_boundary')
    def validate(self):
        require(not self.state.accumulation_active,'validation requires completed update boundary');previous=self.state.phase
        self.state.phase='validation';self._notify();start=time.perf_counter()
        try:
            result=evaluate_model(self.model,self.dataset,model_key=self.config.model_key,seed=self.config.data_seed,
                grid_size=None,count=self.config.validation_query_count,diagnostics=False,progress=self.monitor.phase,familiar_only=True)
            score=result['aggregate']['selection_standardized_MSE'];identity=state_identity(self.model)
            report=ValidationReport(self.config.architecture,identity,digest((self.dataset.config.identity,'validation',self.config.data_seed,self.config.validation_query_count)),
                'G1 unified fixed unseen validation',self.state.successful,result['aggregate'],{'rows':result['rows']},{'cases':len(self.dataset.cases('validation'))},time.perf_counter()-start,self.config.device,'float32')
            self.state.validation_history.append({'step':self.state.successful,'value':score,'per_PDE':result['aggregate']['per_PDE'],'checkpoint':identity})
            self.state.validation_seconds+=report.elapsed_seconds
            if self.state.successful>0 and (self.state.best_validation_metric is None or score<self.state.best_validation_metric):
                self.state.best_validation_metric=score;self.state.best_validation_step=self.state.successful;self.best_validation_model_state=tree_cpu(self.model.state_dict())
            self._log('validation.jsonl',{'update':self.state.successful,**result});self._notify(validation=report);return report
        finally:self.state.phase=previous if previous!='validation' else 'ready'
    def save_checkpoint(self):
        require(not self.state.accumulation_active and self.state.phase!='failed','cannot checkpoint partial/failed update')
        require(self.config.source_id==implementation_identity(),'source changed under active process; preserving existing checkpoints')
        self.monitor.phase('Checkpoint saving',model=self.config.model_key,successful=self.state.successful);start=time.perf_counter()
        root=Path(__file__).resolve().parents[3]
        meta={'artifact_type':'training_checkpoint','architecture_id':self.config.architecture,'run_id':self.config.run_id,
              'scientific_schema':SCHEMA,'config':asdict(self.config),'model_config':asdict(self.model.config),'certificate':self.certificate,
              'normalizers':self.normalizers.asdict(),'dataset_identity':self.dataset.config.identity,'dataset_manifest':str(self.dataset.path),
              'original_split_identity':getattr(self.dataset,'index',{}).get('original_split_identity'),
              'case_inventory_identity':getattr(self.dataset,'index',{}).get('inventory_identity'),
              'source':source_identity(root),'implementation_identity':implementation_identity(),'runtime':environment_report(),
              'successful_update':self.state.successful,'result_status':'unified multicase training; no accuracy claim before independent evaluation',
              'reference_scope':'sampled certificate + per-case QC + separate stratified dataset qualification',
              'parents':self.parent_checkpoints}
        from .history import archive
        archive(self)
        payload={'history_refs':self.history_refs,'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'scaler':None,
                 'training_state':asdict(self.state),'rng':capture_rng(self.rng_generators),'sampler':self.sampler.state_dict(self.state.sampler_cursor),
                 'best_validation_model':self.best_validation_model_state,'normalizers':self.normalizers.asdict()}
        path=write_artifact(self.run_dir/'checkpoints',f'unified-update-{self.state.successful:08d}',meta,payload,required=tuple(payload))
        self.state.last_checkpoint=str(path);self.state.checkpoint_history.append({'step':self.state.successful,'path':str(path)})
        if self.state.best_validation_step==self.state.successful:self.state.best_checkpoint=str(path)
        status=sync(path,self.persistence_policy);self.state.persistence.update(local_status='verified',Drive_status=status['Drive_status'],last_sync=status)
        if self.state.best_validation_step==self.state.successful:self.state.persistence['Drive_best']=status.get('Drive_manifest')
        backlog=list(self.state.persistence.get('backlog',[]))
        if status['Drive_status']=='failed':backlog.append(str(path))
        self.state.persistence['backlog']=[p for p in dict.fromkeys(backlog) if p==str(path) or sync(p,self.persistence_policy)['Drive_status']=='failed']
        pointers={'latest':str(path),'best':self.state.best_checkpoint,'best_step':self.state.best_validation_step,'successful':self.state.successful,
                  'model':self.config.model_key,'dataset_identity':self.dataset.config.identity,'Drive_latest':status.get('Drive_manifest'),'Drive_best':self.state.persistence.get('Drive_best')}
        atomic_json(self.run_dir/'checkpoints.json',pointers);atomic_json(self.run_dir/'persistence.json',self.state.persistence)
        registry=self.run_dir.parent.parent/'experiment.json'
        existing=json.loads(registry.read_text()) if registry.exists() else {'schema':SCHEMA,'dataset_identity':self.dataset.config.identity,'models':{}}
        existing['models'][self.config.model_key]=self.summary('checkpointed')
        atomic_json(registry,existing)
        if self.persistence_policy.drive_enabled:
            # Predictable registry pointer, separate from immutable artifacts.
            try:
                remote=Path(self.persistence_policy.drive_root).parent/'experiment.json';atomic_json(remote,existing)
                require(digest(json.loads(remote.read_text()))==digest(existing),'Drive experiment pointer verification failed')
            except Exception as error:
                import warnings
                warnings.warn('DRIVE EXPERIMENT INDEX FAILED; local registry/checkpoints retained: '+str(error),RuntimeWarning)
        self.state.saving_seconds+=time.perf_counter()-start;self._notify();return path
    def summary(self,status):
        return {'status':status,'model':self.config.model_key,'config':asdict(self.config),'run_dir':str(self.run_dir),'successful':self.state.successful,
                'latest':self.state.last_checkpoint,'best':self.state.best_checkpoint,'persistence':self.state.persistence,
                'elapsed_seconds':self.state.elapsed_seconds,'validation_seconds':self.state.validation_seconds,'saving_seconds':self.state.saving_seconds}


def inspect_checkpoint(path,config,dataset):
    record,payload=read_payload(path,kind='training_checkpoint',architecture=config.architecture,
        required=('model','optimizer','scheduler','scaler','training_state','rng','sampler','normalizers','best_validation_model'))
    require(record.get('scientific_schema')==SCHEMA,'legacy single-case/T1 checkpoints cannot warm-start unified training')
    require(digest(record['config'])==config.identity,'scientific configuration/full schedule differs; not exact resume')
    require(record['implementation_identity']==implementation_identity(),'implementation changed; exact resume rejected')
    require(record['dataset_identity']==dataset.config.identity and payload['normalizers']['identity']==dataset.norms.identity,'dataset/normalizers differ')
    require(record['certificate']['certification_id']==dataset.certificate['certification_id'],'certificate differs')
    require(record['runtime']['dependencies']['torch']==torch.__version__,'Torch differs; exact continuation not claimed')
    state=TrainingState(**payload['training_state'])
    require(state.successful==record['successful_update']==payload['scheduler']['completed'] and not state.accumulation_active,'inconsistent saved update boundary')
    require(digest(payload['sampler'])==digest(BalancedSampler(config,dataset).state_dict(state.sampler_cursor)),'sampler/order mismatch')
    require(payload['scaler'] is None,'V1 FP32 checkpoint expects no AMP scaler')
    require(record['model_config']==asdict(ModelConfig.preset(config.architecture,config.capacity)),'saved architecture definition differs')
    probe=create(config.architecture,dataset.norms,config.capacity,seed=config.root_seed,device='cpu')
    expected=probe.state_dict();require(set(expected)==set(payload['model']),'saved model key mismatch')
    for k,v in expected.items():
        saved=payload['model'][k];require(saved.shape==v.shape and saved.dtype==v.dtype and bool(torch.isfinite(saved).all()),'saved model tensor invalid: '+k)
    groups=payload['optimizer']['param_groups'];parameters=tuple(probe.parameters())
    require(len(groups)==1 and len(groups[0]['params'])==len(parameters),'saved Adam parameter ordering mismatch')
    require(groups[0]['weight_decay']==config.weight_decay and groups[0]['lr']==payload['scheduler']['next_lr'],'saved Adam hyperparameters/scheduler mismatch')
    for i,param in zip(groups[0]['params'],parameters):
        moment=payload['optimizer']['state'].get(i)
        if state.successful>0:
            # Some architecture heads may be unused in a particular task history.
            if moment is None:continue
            require(float(moment['step'])<=state.successful and float(moment['step'])>0,'invalid Adam update counter')
            for k in ('exp_avg','exp_avg_sq'):
                require(moment[k].shape==param.shape and bool(torch.isfinite(moment[k]).all()),'invalid Adam moment shape/values')
    del probe
    return record,payload


def resume(path,config,dataset,root,*,monitor=None,persistence=None):
    record,payload=inspect_checkpoint(path,config,dataset)
    engine=Engine(config,dataset,root,monitor=monitor,persistence=persistence,allow_existing=True)
    engine.model.load_state_dict(payload['model'],strict=True);engine.optimizer.load_state_dict(payload['optimizer']);engine.scheduler.load_state_dict(payload['scheduler'])
    engine.state=TrainingState(**payload['training_state']);engine.state.phase='ready';engine.state.last_checkpoint=str(path)
    engine.state.checkpoint_history.append({'step':engine.state.successful,'path':str(path)})
    engine.state.best_checkpoint=str(path) if engine.state.best_validation_step==engine.state.successful else engine.state.persistence.get('Drive_best') or engine.state.best_checkpoint
    engine.history_refs=payload.get('history_refs',[])
    engine.best_validation_model_state=payload['best_validation_model'];engine.parent_checkpoints=[str(path)];engine._elapsed_base=engine.state.elapsed_seconds
    engine.rng_generators,limitations=restore_rng(payload['rng'],target_device=config.device);require(not limitations,'exact RNG continuation unavailable: '+str(limitations))
    return engine


def train_models(configs,dataset,root,*,run_mode='preview',stop_after=3000,lookup='none',action='fresh',resume_paths=None,persistence=None,monitor=None,preflight_path=None):
    """Single synchronous cell; independent model continuation, exact paths only."""
    require(run_mode in ('preview','train') and lookup in ('none','explicit') and action in ('fresh','resume'),'invalid launch/recovery controls')
    require(bool(configs) and len({c.model_key for c in configs})==len(configs),'duplicate selected model')
    paths=resume_paths or {};require(not paths or lookup=='explicit','explicit checkpoint lookup must be selected')
    require(action!='resume' or lookup=='explicit','resume requires explicit known paths')
    require(action!='fresh' or not paths,'fresh cannot consume old checkpoints')
    preview={'mode':run_mode,'models':[c.model_key for c in configs],'dataset':dataset.config.identity if dataset else configs[0].dataset_identity,
             'schedule_budget':[c.updates for c in configs],'absolute_session_stop':stop_after,'optimizer_constructors':0,'checkpoints_searched':0,
             'AMP':'auto -> FP32; no unverified T4 mixed-precision path','effective_cases_per_update':4}
    if run_mode=='preview':return preview
    require(dataset is not None,'prepare and qualify the dataset first')
    if persistence is not None and persistence.drive_enabled:
        require(not Path(dataset.root).resolve().is_relative_to(Path(persistence.drive_mount).resolve()),'restore the dataset to fast LOCAL storage before training; random-access Drive reads are prohibited')
    from .preflight import require_preflight
    dashboard=monitor or Dashboard(dataset.config);results={}
    with dashboard.live():
        dashboard.phase('Numerical preflight',artifact=preflight_path)
        require_preflight(preflight_path,configs,dataset)
        saved={}
        dashboard.phase('Checkpoint compatibility',lookup=lookup,explicit_paths=len(paths),recursive_search=False)
        for c in configs:
            require(stop_after<=c.updates,'session stop exceeds declared schedule')
            if c.model_key in paths:
                record,payload=inspect_checkpoint(paths[c.model_key],c,dataset)
                saved[c.model_key]=payload['training_state'];del record,payload
        for i,c in enumerate(configs):
            dashboard.latest.update(model_index=f'{i+1}/{len(configs)}',model_total=len(configs),completed_models=i,session_stop=stop_after,whole_study_ETA_seconds=None)
            engine=None
            dashboard.latest.update(current_model=c.model_key,successful=0,losses={},weighted_losses={},gradient_norm=None,clipping_fraction=None,updates_per_second=None,current_model_ETA_seconds=None)
            try:
                if c.model_key in saved and saved[c.model_key]['successful']>=stop_after:
                    state=saved[c.model_key]
                    results[c.model_key]={'status':'already_at_requested_boundary','model':c.model_key,'config':asdict(c),
                        'run_dir':str(Path(root)/'runs'/c.experiment_id/c.model_key.replace('/','_')/c.run_id),
                        'successful':state['successful'],'latest':str(paths[c.model_key]),'best':state['persistence'].get('Drive_best') or state['best_checkpoint'],
                        'persistence':state['persistence'],'elapsed_seconds':state['elapsed_seconds']}
                    continue
                engine=resume(paths[c.model_key],c,dataset,root,monitor=dashboard,persistence=persistence) if c.model_key in paths else Engine(c,dataset,root,monitor=dashboard,persistence=persistence)
                results[c.model_key]=engine.run_until(stop_after,launch=True)
                registry=Path(root)/'runs'/c.experiment_id/'experiment.json'
                existing=json.loads(registry.read_text()) if registry.exists() else {'schema':SCHEMA,'dataset_identity':dataset.config.identity,'models':{}}
                existing['models'].update(results);atomic_json(registry,existing)
            finally:
                if c.model_key in results:
                    registry=Path(root)/'runs'/c.experiment_id/'experiment.json'
                    existing=json.loads(registry.read_text()) if registry.exists() else {'schema':SCHEMA,'dataset_identity':dataset.config.identity,'models':{}}
                    existing['models'].update(results);atomic_json(registry,existing)
                if engine is not None:del engine
                if torch.cuda.is_available():torch.cuda.empty_cache()
        dashboard.phase('Completed',models=len(results),session_boundary=stop_after)
    return results
