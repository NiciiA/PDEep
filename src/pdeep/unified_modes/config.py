"""New experiment schema; old inverse-only schedules/configurations are untouched."""
from dataclasses import dataclass,asdict,fields
from pathlib import Path
from ..core import digest,require
from ..artifact_io import sha256
from ..unified.config import TrainConfig as InverseConfig,implementation_identity
from .conditioning import SCHEMA

def extension_identity():return digest({'base':implementation_identity(),'files':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}})

@dataclass(frozen=True)
class TrainConfig(InverseConfig):
    updates:int=100000
    main_target:int=50000
    conditioning_schema:str=SCHEMA
    mode_mixture:tuple=(.3,.4,.3)
    hybrid_zero_probability:float=.25
    extension_source:str=''
    initialization:str='fresh'
    def __post_init__(self):
        requested=asdict(self);base={k:requested[k] for k in {f.name for f in fields(InverseConfig)}}
        # Resolve using existing numerical/training checks; V2 smoke permits10-50,
        # and <=3 only for isolated unit fixtures. Final saved settings stay explicit.
        preset,capacity=self.preset,self.capacity
        require(capacity in ('verification','medium','large'),'unsupported model capacity')
        require(preset in ('smoke','pilot','research'),'unsupported experiment preset')
        require(capacity!='verification' or preset=='smoke','verification capacity only for bounded tests')
        require(self.updates>0 and (preset!='smoke' or self.updates<=50),'smoke horizon must be <=50')
        base.update(preset='research',capacity='medium');resolved=InverseConfig(**base)
        for k,v in asdict(resolved).items():object.__setattr__(self,k,v)
        object.__setattr__(self,'preset',preset);object.__setattr__(self,'capacity',capacity)
        require(self.conditioning_schema==SCHEMA and self.initialization in ('fresh','finetune_inverse'),'unsupported conditioning/init schema')
        require(len(self.mode_mixture)==3 and all(v>0 for v in self.mode_mixture) and abs(sum(self.mode_mixture)-1)<1e-12,'three positive mode proportions summing to1 required')
        require(0<=self.hybrid_zero_probability<=1 and 0<self.main_target<=self.updates,'invalid zero-observation policy/main target')
        if not self.extension_source:object.__setattr__(self,'extension_source',extension_identity())
