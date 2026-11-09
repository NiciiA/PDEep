"""Complete v2 optimizer-boundary checkpoints and portable same-run resume."""
from dataclasses import asdict
from pathlib import Path
import json,time
import torch
from .core import require,decode,digest
from .artifact_io import write_artifact,read_payload,read_manifest,atomic_json,append_event
from .artifact_metadata import make_metadata,validate_metadata
from .artifact_tensors import tree_cpu,tree_identity,state_identity
from .artifact_rng import capture_rng,restore_rng
from .artifact_drive import PersistencePolicy,sync_verified

HISTORY_FILES=('steps.jsonl','validation.jsonl','evaluations.jsonl','checkpoints.jsonl','persistence.jsonl','failures.jsonl')


def histories(directory):
    result={}
    for name in HISTORY_FILES:
        path=Path(directory)/name
        result[name]=[json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    return result


def execution_state():
    return {'CPU_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark}


def restore_execution(settings):
    torch.set_num_threads(settings['CPU_threads'])
    if torch.get_num_interop_threads()!=settings['interop_threads']:
        try:torch.set_num_interop_threads(settings['interop_threads'])
        except RuntimeError as error:raise ValueError('cannot restore interop threads after parallel work; reload in a fresh process') from error
    torch.use_deterministic_algorithms(settings['deterministic_algorithms'])
    torch.backends.cudnn.deterministic=settings['cudnn_deterministic'];torch.backends.cudnn.benchmark=settings['cudnn_benchmark']


def persistence_event(engine,status,**values):
    event={'run_id':engine.config.run_id,'successful':engine.state.successful,'event':status,**values}
    append_event(engine.run_dir/'persistence.jsonl',event)
    engine.state.persistence.update(event)
    return event


def save_resumable(engine):
    local_started=time.monotonic()
    state=engine.state;config=engine.config
    require(state.successful>0 and not state.accumulation_active and state.phase!='failed','checkpoint requires a completed successful optimizer boundary; unfinished gradients are not resumable')
    require(engine.scheduler.completed==state.successful,'scheduler must match successful update boundary')
    require(getattr(engine,'scaler',None) is None,'AMP/scaler unsupported in this float32 contract')
    policy=getattr(engine,'persistence_policy',PersistencePolicy())
    directory=engine.run_dir/'checkpoints';stem=f'update-{state.successful:08d}';manifest=directory/(stem+'.json')
    persistence_event(engine,'local_save_started',destination=str(manifest),Drive_enabled=policy.drive_enabled)
    try:
        snapshot=asdict(state);snapshot.update(phase='ready',last_checkpoint=str(manifest))
        snapshot['checkpoint_history']=snapshot['checkpoint_history']+[{'step':state.successful,'path':str(manifest)}]
        current_model=state_identity(engine.model)
        is_best=(state.best_validation_step==state.successful and state.best_validation_weights==current_model)
        best_model=tree_cpu(engine.model.state_dict()) if is_best else getattr(engine,'best_validation_model_state',None)
        if state.best_validation_weights is not None:
            require(best_model is not None and digest(tree_identity(best_model))==state.best_validation_weights,'best-validation model snapshot missing or inconsistent')
        engine.best_validation_model_state=best_model
        if is_best:snapshot['best_checkpoint']=str(manifest)
        snapshot['persistence'].update(local_status='completed',last_verified_local_checkpoint=str(manifest))
        parents=getattr(engine,'parent_checkpoints',[])
        metadata=make_metadata('training_checkpoint',config.architecture,engine.normalizers,model=engine.model,run_id=config.run_id,
                               config=config,successful=state.successful,stage=state.stage,parents=parents,device=config.device)
        metadata.update(validation_manifest_identity=engine.manifest.identity,
                        safe_boundary={'accumulation_active':False,'successful':state.successful,'sampler_cursor':state.sampler_cursor},
                        persistence_policy=asdict(policy),persistence_policy_identity=policy.identity,
                        model_selection={'metric':state.validation_metric_name,'best_value':state.best_validation_metric,'best_step':state.best_validation_step,'is_best':is_best,'best_model_embedded':best_model is not None},
                        rng_policy={'state':'Python, NumPy global/registered, torch CPU/available CUDA/MPS, registered model generators',
                                    'project_streams':'stateless keyed root+cursor; case physics/observations/training queries/calibration/validation/monitoring independent',
                                    'exactness':'controlled single-process same-device CPU verified; no cross-hardware bitwise claim'})
        payload={'version':'train-engine-v2','architecture':config.architecture,'config':asdict(config),'state':snapshot,
                 'model':tree_cpu(engine.model.state_dict()),'optimizer':tree_cpu(engine.optimizer.state_dict()),'scheduler':engine.scheduler.state_dict(),
                 'scaler':None,'AMP':False,'rng':capture_rng(getattr(engine,'rng_generators',{})),
                 'execution':execution_state(),'sampler':engine.sampler.state_dict(state.sampler_cursor),
                 'validation_manifest':asdict(engine.manifest),'histories':histories(engine.run_dir),
                 'capability':engine.capability,'normalizers':asdict(engine.normalizers),'vocabulary':list(engine.model.vocabulary.quantities),
                 'latest_validation':None if engine.latest_validation is None else asdict(engine.latest_validation)}
        payload['best_validation_model']=best_model
        path=write_artifact(directory,stem,metadata,payload,required=('model','optimizer','scheduler','rng','sampler','state','histories','validation_manifest','execution'))
        record=read_manifest(path)
        pointer={'manifest':path.name,'artifact_id':record['artifact_id'],'successful':state.successful,'model_state_identity':current_model}
        atomic_json(directory/'last.json',pointer)
        if is_best:
            atomic_json(directory/'best_validation.json',pointer);state.best_checkpoint=str(path)
        append_event(engine.run_dir/'checkpoints.jsonl',{'run_id':config.run_id,'successful':state.successful,'artifact_id':record['artifact_id'],
                                                        'manifest':str(path),'model_state_identity':current_model,'is_best_validation':is_best})
        persistence_event(engine,'local_save_completed',local_status='completed',last_verified_local_checkpoint=str(path),last_verified_local_artifact_id=record['artifact_id'],local_saving_seconds=time.monotonic()-local_started)
    except Exception as error:
        persistence_event(engine,'local_save_failed',local_status='failed',error=f'{type(error).__name__}: {error}')
        raise
    if policy.drive_enabled:
        persistence_event(engine,'Drive_copy_started',Drive_status='started',backlog=[str(path)])
    status=sync_verified(path,policy)
    if status['Drive_status']=='completed':
        persistence_event(engine,'Drive_copy_completed',Drive_status='completed',last_verified_Drive_checkpoint=status['Drive_manifest'],backlog=[],synchronizing_seconds=status['seconds'])
    elif status['Drive_status']=='failed':
        persistence_event(engine,'Drive_copy_failed',Drive_status='failed',backlog=[str(path)],error=status['error'],synchronizing_seconds=status['seconds'])
    else:state.persistence.update(Drive_status='disabled',backlog=[])
    return path


def _restore_histories(engine,saved,*,relocated):
    for name,rows in saved.items():
        path=engine.run_dir/name
        if path.exists():
            existing=[json.loads(line) for line in path.read_text().splitlines() if line]
            if name in {'steps.jsonl','validation.jsonl'}:
                require(existing==rows,'existing run history has advanced/diverged; resume in an explicit new run_root instead of mixing histories')
            continue
        if rows:
            with path.open('w') as stream:
                for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')


def load_resumable(manifest,*,expected_architecture,monitor=None,restore_rng_state=True,probe=False,run_root=None):
    from .train_types import ResolvedTrainConfig,TrainingState,ValidationReport
    from .training import TrainingEngine
    from .models import select_architecture
    from .train_data import ValidationManifest
    record,payload=read_payload(manifest,kind='training_checkpoint',architecture=expected_architecture,
                                required=('config','state','model','optimizer','scheduler','rng','sampler','scaler','AMP','execution','validation_manifest','histories'))
    require(payload['version']=='train-engine-v2' and payload['architecture']==expected_architecture,'checkpoint payload architecture/version mismatch')
    norm,vocab=validate_metadata(record)
    config=decode(ResolvedTrainConfig,payload['config'])
    require(config.architecture==expected_architecture and config.identity==record['config_identity'] and config.run_id==record['run_id'],'resolved checkpoint configuration/run/architecture mismatch')
    state=TrainingState(**payload['state'])
    require(0<state.successful<=config.updates and state.attempted>=state.successful+state.skipped+state.failed,'invalid checkpoint budget/counters')
    require(state.successful==record['successful_updates']==payload['scheduler']['completed'] and not state.accumulation_active,
            'checkpoint contains unfinished/inconsistent successful-update boundary')
    require(state.architecture==expected_architecture and state.run_id==config.run_id,'training state architecture/run mismatch')
    require(not payload['AMP'] and payload['scaler'] is None,'unsupported AMP/scaler state')
    if restore_rng_state:restore_execution(payload['execution'])
    with torch.random.fork_rng(devices=[]):model=select_architecture(expected_architecture,normalizers=norm,vocabulary=vocab,root_seed=config.root_seed)
    validate_metadata(record,model=model);model.load_state_dict(payload['model'],strict=True)
    require(state_identity(model)==record['component_inventory']['model_state_identity'],'checkpoint model state identity mismatch')
    require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values() if t.is_floating_point()),'nonfinite checkpoint model weights')
    model.to(config.device)
    original=Path(manifest).resolve().parent.parent
    root=original.parent.parent if run_root is None else Path(run_root)
    engine=TrainingEngine(config,norm,root,model=model,monitor=monitor,probe=probe,allow_existing_run=True)
    require(engine.manifest.identity==record['validation_manifest_identity'] and asdict(engine.manifest)==decode(ValidationManifest,payload['validation_manifest']).__dict__, 'fixed validation manifest/specification mismatch')
    require(payload['sampler']==engine.sampler.state_dict(state.sampler_cursor),'sampler/root seed/next-case cursor mismatch')
    engine.optimizer.load_state_dict(payload['optimizer']);engine.scheduler.load_state_dict(payload['scheduler'])
    require(digest(tree_identity(tree_cpu(engine.optimizer.state_dict())))==digest(tree_identity(payload['optimizer'])),'optimizer moments/parameter groups failed restoration')
    engine.state=state;engine.state.phase='ready';engine._elapsed_base=state.elapsed_seconds;engine._started=time.monotonic()
    engine.capability=payload.get('capability',{});engine.parent_checkpoints=record['dependencies']['parent_checkpoints']+[{'artifact_id':record['artifact_id'],'path_hint':str(Path(manifest).resolve()),'kind':'training_checkpoint'}]
    engine.best_validation_model_state=payload.get('best_validation_model')
    if state.best_validation_weights is not None:
        require(engine.best_validation_model_state is not None and digest(tree_identity(engine.best_validation_model_state))==state.best_validation_weights,'embedded best-validation model state missing or inconsistent')
    engine.persistence_policy=decode(PersistencePolicy,record['persistence_policy'])
    if payload.get('latest_validation') is not None:engine.latest_validation=decode(ValidationReport,payload['latest_validation'])
    _restore_histories(engine,payload['histories'],relocated=run_root is not None)
    for name,value in (('config.json',asdict(config)),('validation_manifest.json',{'identity':engine.manifest.identity,**asdict(engine.manifest)})):
        if not (engine.run_dir/name).exists():atomic_json(engine.run_dir/name,value)
    engine.loaded_artifact_id=record['artifact_id'];engine.state.persistence.update(local_status='completed',last_verified_local_checkpoint=str(Path(manifest).resolve()),last_verified_local_artifact_id=record['artifact_id'])
    if restore_rng_state:engine.rng_generators,engine.rng_restore_limitations=restore_rng(payload['rng'],target_device=config.device)
    else:engine.rng_generators={};engine.rng_restore_limitations=['RNG restoration deliberately disabled; not an exact stochastic resume']
    return engine


def record_evaluation(manifest,*,plan_identity,result_identity,metrics):
    record=read_manifest(manifest)
    event={'run_id':record['run_id'],'artifact_id':record['artifact_id'],'successful':record['successful_updates'],
           'model_state_identity':record['component_inventory']['model_state_identity'],
           'plan_identity':plan_identity,'result_identity':result_identity,'metrics':metrics}
    append_event(Path(manifest).parent.parent/'evaluations.jsonl',event)
    return event


def export_selected_inference(engine,directory,*,selection='best_validation'):
    """Export embedded best weights after relocation, without an older file.

    Optimizer resume still uses the actual latest boundary, never best weights
    combined with the latest Adam moments.
    """
    from .models import select_architecture
    from .inference_artifacts import export_inference
    require(selection in {'best_validation','last'},'explicit best_validation/last selection required')
    state=engine.model.state_dict() if selection=='last' else engine.best_validation_model_state
    require(state is not None,'no post-update best-validation model available')
    with torch.random.fork_rng(devices=[]):
        model=select_architecture(engine.config.architecture,normalizers=engine.normalizers,vocabulary=engine.model.vocabulary,root_seed=engine.config.root_seed)
    model.load_state_dict(state,strict=True)
    step=engine.state.successful if selection=='last' else engine.state.best_validation_step
    return export_inference(model,directory,architecture=engine.config.architecture,normalizers=engine.normalizers,
                            run_id=engine.config.run_id,config=engine.config,successful=step,parents=engine.parent_checkpoints,
                            status='selected '+selection+'; '+engine.config.preset)
