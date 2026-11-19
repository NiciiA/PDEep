"""Reuse the existing engine/update/launch bytecode with one dataset provider.

No optimizer algorithm, architecture, objective or experiment manager is added.
Private function globals bind the provider adapter without monkeypatching an
active module (same technique already used by performance.asynchronous_update).
"""
from dataclasses import dataclass,asdict
import types
from ..core import require
from ..unified_modes.config import TrainConfig as OriginalConfig
from ..unified_modes import training as original
from ..unified_modes.workflow import configurations as original_configs
from .feasibility import SCHEMA,parent_configuration,source_identity
from .provider import FeasibilityProvider
from .compatibility import approved_adapter
from .readiness import ensure_ready
from .checkpointing import SafeBoundaries,bind,storage_ready

@dataclass(frozen=True)
class TrainConfig(OriginalConfig):
    feasibility_schema:str=SCHEMA
    reference_parent_identity:str=''
    feasibility_source:str=''
    def __post_init__(self):
        super().__post_init__()
        require(self.feasibility_schema==SCHEMA and self.reference_parent_identity,'feasibility plan metadata required')
        require(not self.feasibility_source or approved_adapter(self.feasibility_source),'feasibility provider source changed; exact resume rejected')
        object.__setattr__(self,'feasibility_source',self.feasibility_source or source_identity())

def configurations(dataset_config,**kwargs):
    kwargs.setdefault('mode','pilot');kwargs.setdefault('updates',3000);kwargs.setdefault('main_target',3000);kwargs.setdefault('capacity','large')
    require(kwargs['updates']==kwargs['main_target']==3000,'feasibility is a separately declared3000-update schedule; do not stretch research checkpoints')
    return tuple(TrainConfig(**asdict(c),reference_parent_identity=parent_configuration(dataset_config).identity) for c in original_configs(dataset_config,**kwargs))

def configure_provider(engine):
    require(approved_adapter(engine.config.feasibility_source) and engine.dataset.index.get('feasibility',{}).get('parent_dataset_identity')==engine.config.reference_parent_identity,'training/dataset feasibility policy mismatch')
    engine.sampler.provider=FeasibilityProvider(engine.dataset);return engine

class Engine(SafeBoundaries,original.Engine):
    def __init__(self,config,dataset,root,**kwargs):
        # No model/optimizer allocation before the complete reference/storage gate.
        self.readiness=ensure_ready((config,),dataset,root)
        storage=storage_ready(root,kwargs.get('persistence'))
        super().__init__(config,dataset,root,**kwargs);configure_provider(self)
        self.state.persistence.update(storage)

def inspect_checkpoint(path,config,dataset):
    record,payload=original.inspect_checkpoint(path,config,dataset)
    # Saved state necessarily predates committing its own immutable filename.
    # Resolve latest/current-best pointers from the verified artifact on load.
    state={**payload['training_state'],'last_checkpoint':str(path)}
    if state['best_validation_step']==state['successful'] and payload['best_validation_model'] is not None:state['best_checkpoint']=str(path)
    return record,{**payload,'training_state':state}

def resume(*a,**k):
    e=bind(original.resume,Engine=Engine,inspect_checkpoint=inspect_checkpoint)(*a,**k);configure_provider(e)
    e.state.checkpoint_history.append({'step':e.state.successful,'path':e.state.last_checkpoint})
    e.state.persistence.update(local_status='verified',Drive_enabled=e.persistence_policy.drive_enabled)
    if not e.persistence_policy.drive_enabled:e.state.persistence['Drive_status']='disabled'
    return e

def train_models(configs,dataset,root,**kwargs):
    if not configs or not hasattr(configs[0],'feasibility_schema'):return original.train_models(configs,dataset,root,**kwargs)
    require(all(isinstance(c,TrainConfig) for c in configs),'mixed feasibility/research configurations')
    kwargs.setdefault('stop_after',3000)
    if kwargs.get('run_mode','preview')!='preview':
        ensure_ready(configs,dataset,root)
        from ..unified_modes.preflight import require_preflight
        require_preflight(kwargs.get('preflight_path'),configs,dataset)
        # Complete pending validation even when resume stops at this exact update.
        paths=dict(kwargs.get('resume_paths') or {})
        for config in configs:
            if config.model_key in paths:
                _,payload=inspect_checkpoint(paths[config.model_key],config,dataset)
                state=payload['training_state'];n=state['successful']
                pending=n and (not state['validation_history'] or state['validation_history'][-1]['step']!=n)
                if pending and n>=kwargs['stop_after']:
                    engine=resume(paths[config.model_key],config,dataset,root,monitor=kwargs.get('monitor'),persistence=kwargs.get('persistence'))
                    engine.run_until(n,launch=True);paths[config.model_key]=engine.state.last_checkpoint
        if paths:kwargs['resume_paths']=paths
    namespace={**original.train_models.__globals__,'Engine':Engine,'resume':resume,'inspect_checkpoint':inspect_checkpoint}
    launch=types.FunctionType(original.train_models.__code__,namespace,'existing_launch_with_feasibility_provider',original.train_models.__defaults__,original.train_models.__closure__)
    launch.__kwdefaults__=original.train_models.__kwdefaults__
    return launch(configs,dataset,root,**kwargs)
