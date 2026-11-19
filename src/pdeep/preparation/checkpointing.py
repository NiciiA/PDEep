"""Local-first validated boundary saves, using the existing verified payload writer.

Drive uses the existing bounded synchronous verified copier. A milestone waits
for its commit: no unbounded thread queue, no optimistic durability claim. The
full optimizer/RNG payload is committed locally BEFORE validation or Drive IO.
"""
import types,uuid,json,sys,time
from pathlib import Path
from dataclasses import replace
from ..core import require,digest
from ..artifact_io import write_artifact,atomic_json,read_manifest
from ..artifact_rng import capture_rng,restore_rng
from ..artifact_drive import PersistencePolicy
from ..unified.training import Engine as BaseEngine
from ..unified.persistence import sync
from ..unified_modes import training as original
from .validation import evaluate

def bind(function,**bindings):
    f=types.FunctionType(function.__code__,{**function.__globals__,**bindings},function.__name__,function.__defaults__,function.__closure__)
    f.__kwdefaults__=function.__kwdefaults__;return f

def storage_ready(root,policy=None):
    """Small real write/readback probe; never calls a numerical reference."""
    policy=policy or PersistencePolicy();root=Path(root)/'storage-readiness'
    path=write_artifact(root,'probe-'+uuid.uuid4().hex,{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-storage-probe'},{'probe':True},required=('probe',))
    status=sync(path,policy)
    require(not policy.drive_enabled or status.get('Drive_verified') is True,'Drive enabled but verified storage probe failed; no optimizer started. '+str(status))
    return {'local_status':'not_started','Drive_enabled':policy.drive_enabled,'Drive_status':'not_started' if policy.drive_enabled else 'disabled','storage_probe':status,'probe_is_training_checkpoint':False}

def save(engine,reason):
    started=time.perf_counter();previous_saving=engine.state.saving_seconds
    from .feasibility import source_identity
    engine._authorize()
    require(not engine.state.accumulation_active,'cannot checkpoint an unfinished accumulation')
    # Unique immutable names allow a pre-validation latest AND a post-validation
    # latest/best at the same successful update without overwriting either.
    def writer(directory,stem,meta,payload,**kwargs):
        meta={**meta,'feasibility_runtime_source':source_identity(),'checkpoint_reason':reason,'validation_pending':reason=='before-validation'}
        return write_artifact(directory,stem+'-'+reason+'-'+uuid.uuid4().hex,meta,payload,**kwargs)
    policy=engine.persistence_policy
    if reason=='validated':
        raw=engine.run_dir/'validation.jsonl';rows=[]
        if raw.exists():rows=[json.loads(v) for v in raw.read_text().splitlines() if json.loads(v).get('update')==engine.state.successful]
        if rows:
            artifact=write_artifact(engine.run_dir/'history','validation-'+uuid.uuid4().hex,{'artifact_type':'diagnostic_bundle','architecture_id':engine.config.architecture,'run_id':engine.config.run_id},{'history':{'steps.jsonl':[],'case_losses.jsonl':[],'validation.jsonl':rows}},required=('history',))
            engine.history_refs.append({'from_update_exclusive':engine.state.successful,'through_update':engine.state.successful,'local_manifest':str(artifact)})
    previous_backlog=list(engine.state.persistence.get('backlog',[]))
    engine.persistence_policy=replace(policy,drive_enabled=False)
    try:
        # Reuse base payload writer; avoid same-step raw-log rotation collisions.
        def local_commit(manifest,disabled_policy):
            status=sync(manifest,disabled_policy)
            if policy.drive_enabled:status.update(Drive_enabled=True,Drive_status='pending')
            return status
        path=bind(BaseEngine.save_checkpoint,write_artifact=writer,sync=local_commit)(engine)
    finally:engine.persistence_policy=policy
    # Every successful update is recoverable locally before any remote IO.
    engine.state.persistence.update(local_status='verified',Drive_enabled=policy.drive_enabled,Drive_status='syncing' if policy.drive_enabled else 'disabled')
    engine.monitor.phase('Verified local checkpoint; Drive commit' if policy.drive_enabled else 'Verified local checkpoint',last_verified_local_checkpoint=str(path),Drive_enabled=policy.drive_enabled)
    for ref in engine.history_refs:
        if not ref.get('Drive_verified') and policy.drive_enabled:ref.update(sync(ref['local_manifest'],policy))
    backlog=[]
    for previous in previous_backlog:
        if policy.drive_enabled and sync(previous,policy).get('Drive_verified') is not True:backlog.append(previous)
        elif not policy.drive_enabled:backlog.append(previous)
    engine.state.persistence['backlog']=backlog
    status=sync(path,policy)
    engine.state.persistence.update(local_status='verified',Drive_enabled=policy.drive_enabled,Drive_status=status['Drive_status'],last_sync=status)
    if status.get('Drive_verified'):
        engine.state.persistence['Drive_latest']=status['Drive_manifest']
        if reason=='validated' and engine.state.best_validation_step==engine.state.successful:engine.state.persistence['Drive_best']=status['Drive_manifest']
    elif status['Drive_status']=='failed':
        engine.state.persistence.setdefault('backlog',[]).append(str(path))
    pointers={'latest':str(path),'best':engine.state.best_checkpoint,'successful':engine.state.successful,'best_step':engine.state.best_validation_step,'Drive_latest':engine.state.persistence.get('Drive_latest'),'Drive_best':engine.state.persistence.get('Drive_best'),'validation_pending':reason=='before-validation'}
    atomic_json(engine.run_dir/'checkpoints.json',pointers);atomic_json(engine.run_dir/'persistence.json',engine.state.persistence)
    # Update discoverable experiment pointers only from read-back verified status.
    registry=engine.run_dir.parent.parent/'experiment.json'
    body=json.loads(registry.read_text()) if registry.exists() else {'models':{}}
    body['models'][engine.config.model_key]=engine.summary('checkpointed');atomic_json(registry,body)
    if policy.drive_enabled and status.get('Drive_verified'):
        try:
            remote=Path(policy.drive_root).parent/'experiment.json';atomic_json(remote,body)
            require(digest(json.loads(remote.read_text()))==digest(body),'Drive experiment pointer verification failed')
        except Exception as error:
            import warnings
            warnings.warn('DRIVE INDEX FAILED; immutable checkpoint and valid local copy retained: '+str(error),RuntimeWarning)
    duration=time.perf_counter()-started;engine.state.saving_seconds=previous_saving+duration
    engine.monitor.saved(duration,status);engine._notify();return path

def validated(engine):
    return bool(engine.state.validation_history and engine.state.validation_history[-1]['step']==engine.state.successful)

class SafeBoundaries:
    def validate(self):
        if self.state.successful>0:save(self,'before-validation')
        rng=capture_rng(self.rng_generators)
        try:
            result=bind(original.Engine.validate,evaluate=evaluate)(self)
        except BaseException:
            self.monitor.phase('Validation failed; recoverable local checkpoint retained',successful=self.state.successful,last_verified_local_checkpoint=self.state.last_checkpoint)
            raise
        finally:self.rng_generators,_=restore_rng(rng,target_device=self.config.device)
        if self.state.successful>0:save(self,'validated');self._skip_duplicate_save=self.state.successful
        return result
    def save_checkpoint(self):
        if getattr(self,'_skip_duplicate_save',None)==self.state.successful:self._skip_duplicate_save=None;return Path(self.state.last_checkpoint)
        return save(self,'validated' if validated(self) else 'milestone')
    def run_until(self,stop_after,*,launch=False):
        require(launch,'explicit launch required')
        # An interrupted pre-validation checkpoint resumes this boundary first,
        # even if the requested session stop equals its successful update count.
        if self.state.successful and not validated(self) and (self.state.successful%self.config.validation_every==0 or self.state.successful>=stop_after):self.validate()
        return super().run_until(stop_after,launch=launch)

def recover_live_failure(*,model_key,exception=None):
    """Explicit recovery BEFORE kernel restart, only from retained trusted frames.

    No weights/counters are reconstructed from logs. No frame locals are printed.
    """
    tb=exception.__traceback__ if exception is not None else getattr(sys,'last_traceback',None);engines={}
    while tb:
        for key in ('engine','self'):
            e=tb.tb_frame.f_locals.get(key)
            if isinstance(e,original.Engine) and e.config.model_key==model_key:engines[id(e)]=e
        tb=tb.tb_next
    require(len(engines)==1,'no unique live failed engine retained; supply original exception before restarting. Scalar logs cannot recover lost weights.')
    e=next(iter(engines.values()));require(e.state.successful>0 and not e.state.accumulation_active and e.scheduler.completed==e.state.successful,'engine is not at a complete resumable update boundary')
    # Old active instances retain their approved original implementation. This is
    # an explicit save, never a class mutation or optimizer reconstruction.
    path=e.save_checkpoint();read_manifest(path)
    return {'checkpoint':str(path),'successful':e.state.successful,'model':model_key,'configuration':e.config,'persistence':e.state.persistence}
