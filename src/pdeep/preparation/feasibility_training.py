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

@dataclass(frozen=True)
class TrainConfig(OriginalConfig):
    feasibility_schema:str=SCHEMA
    reference_parent_identity:str=''
    feasibility_source:str=''
    def __post_init__(self):
        super().__post_init__()
        require(self.feasibility_schema==SCHEMA and self.reference_parent_identity,'feasibility plan metadata required')
        require(not self.feasibility_source or self.feasibility_source==source_identity(),'feasibility provider source changed; exact resume rejected')
        object.__setattr__(self,'feasibility_source',source_identity())

def configurations(dataset_config,**kwargs):
    kwargs.setdefault('mode','pilot');kwargs.setdefault('updates',3000);kwargs.setdefault('main_target',3000);kwargs.setdefault('capacity','large')
    require(kwargs['updates']==kwargs['main_target']==3000,'feasibility is a separately declared3000-update schedule; do not stretch research checkpoints')
    return tuple(TrainConfig(**asdict(c),reference_parent_identity=parent_configuration(dataset_config).identity) for c in original_configs(dataset_config,**kwargs))

def configure_provider(engine):
    require(engine.config.feasibility_source==source_identity() and engine.dataset.index.get('feasibility',{}).get('parent_dataset_identity')==engine.config.reference_parent_identity,'training/dataset feasibility policy mismatch')
    engine.sampler.provider=FeasibilityProvider(engine.dataset);return engine

class Engine(original.Engine):
    def __init__(self,*a,**k):super().__init__(*a,**k);configure_provider(self)

def resume(*a,**k):return configure_provider(original.resume(*a,**k))

def train_models(configs,dataset,root,**kwargs):
    if not configs or not hasattr(configs[0],'feasibility_schema'):return original.train_models(configs,dataset,root,**kwargs)
    require(all(isinstance(c,TrainConfig) for c in configs),'mixed feasibility/research configurations')
    kwargs.setdefault('stop_after',3000)
    namespace={**original.train_models.__globals__,'Engine':Engine,'resume':resume}
    launch=types.FunctionType(original.train_models.__code__,namespace,'existing_launch_with_feasibility_provider',original.train_models.__defaults__,original.train_models.__closure__)
    launch.__kwdefaults__=original.train_models.__kwdefaults__
    return launch(configs,dataset,root,**kwargs)
