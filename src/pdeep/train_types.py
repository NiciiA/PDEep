"""Stage7A resolved configuration and structured training/validation reports."""
from dataclasses import dataclass,field,asdict
from pathlib import Path
import math
from .core import require,digest
from .runtime import resolve_device

NEURAL=('A','C','B0','B1')
FAMILIES=('poisson','heat','wave')


@dataclass(frozen=True)
class ResolvedTrainConfig:
    architecture: str
    preset: str='smoke'
    pdes: tuple[str,...]=FAMILIES
    probabilities: tuple[float,...]=(1/3,1/3,1/3)
    benchmark: str='unit_square_modal_v1'
    inverse_track: str='T1'
    capacity: str='canonical_v1'
    device: str='cpu'
    precision: str='float32'
    microbatch: int | None=None
    accumulation: int | None=None
    effective_batch: int | None=None
    query_count: int=128
    training_case_ids: tuple[int,...] | None=None
    learning_rate: float=3e-4
    weight_decay: float=1e-4
    grad_clip_norm: float=1.
    scheduler: str | None=None
    warmup_updates: int=250
    minimum_lr_fraction: float=.1
    updates: int | None=None
    validation_every: int=250
    checkpoint_every: int=250
    validation_cases_per_pde: int | None=None
    validation_query_count: int=128
    validation_name: str='validation-v1'
    root_seed: int=20261108
    verbosity: str='normal'
    rolling_window: int=50
    source_id: str=''
    run_tag: str='run'
    full_training_enabled: bool=False
    overwrite_checkpoint: bool=False

    def __post_init__(self):
        require(self.architecture in NEURAL,'neural trainer requires A/C/B0/B1; D is evaluation-only; B is ambiguous')
        require(self.benchmark=='unit_square_modal_v1' and self.inverse_track=='T1' and self.capacity=='canonical_v1','unsupported benchmark/track/capacity')
        require(self.precision=='float32','neural engine supports explicit float32 only; no AMP/fallback')
        resolve_device(self.device,self.precision)
        require(self.preset in {'smoke','bounded','research','large'},'unknown training preset')
        require(bool(self.pdes) and len(set(self.pdes))==len(self.pdes) and set(self.pdes)<=set(FAMILIES),'unsupported PDE families')
        require(len(self.probabilities)==len(self.pdes) and all(math.isfinite(p) and p>0 for p in self.probabilities) and abs(sum(self.probabilities)-1)<1e-12,'PDE probabilities must be positive and sum to one')
        default_micro=1 if self.architecture=='C' else 8
        micro=default_micro if self.microbatch is None else self.microbatch
        accum=(16 if self.architecture=='C' else 2) if self.accumulation is None else self.accumulation
        batch=micro*accum if self.effective_batch is None else self.effective_batch
        require(all(type(v) is int and v>0 for v in (micro,accum,batch)) and math.ceil(batch/micro)==accum,'effective batch must match declared accumulation including a smaller final microbatch')
        for name,value in (('microbatch',micro),('accumulation',accum),('effective_batch',batch)):object.__setattr__(self,name,value)
        if self.preset=='large':require(self.updates is not None,'large preset requires explicit update budget')
        updates={'smoke':3,'bounded':100,'research':10000,'large':20000}[self.preset] if self.updates is None else self.updates
        require(type(updates) is int and updates>0,'positive successful-update budget required')
        require((updates<=3 if self.preset=='smoke' else updates<=100 if self.preset=='bounded' else updates==10000 if self.preset=='research' else updates>=20000),'budget inconsistent with preset')
        object.__setattr__(self,'updates',updates)
        if self.training_case_ids is None and self.preset=='bounded':object.__setattr__(self,'training_case_ids',tuple(range(16)))
        if self.training_case_ids is not None:
            require(bool(self.training_case_ids) and len(set(self.training_case_ids))==len(self.training_case_ids) and all(type(i) is int and 0<=i<12000 for i in self.training_case_ids),'selected training IDs must be unique train-split indices')
        scheduler=('warmup_then_cosine' if self.preset in {'research','large'} else 'constant') if self.scheduler is None else self.scheduler
        require(scheduler in {'constant','warmup_then_cosine'},'unsupported scheduler')
        require(type(self.warmup_updates) is int and self.warmup_updates>=0 and 0<self.minimum_lr_fraction<=1,'invalid scheduler settings')
        require(scheduler!='warmup_then_cosine' or self.warmup_updates<updates,'cosine schedule requires warmup shorter than the successful-update budget')
        object.__setattr__(self,'scheduler',scheduler)
        val=(1 if self.preset=='smoke' else 128) if self.validation_cases_per_pde is None else self.validation_cases_per_pde
        require(type(val) is int and 1<=val<=1024,'fixed validation supports1–1024 cases per PDE')
        object.__setattr__(self,'validation_cases_per_pde',val)
        require(all(type(v) is int and v>0 for v in (self.validation_every,self.checkpoint_every,self.rolling_window)),'positive cadence/window required')
        require(type(self.query_count) is int and self.query_count>=2 and type(self.validation_query_count) is int and self.validation_query_count>=2,'at least2 queries required by declared sampling coverage')
        require(all(math.isfinite(v) for v in (self.learning_rate,self.weight_decay,self.grad_clip_norm)) and self.learning_rate>0 and self.weight_decay>=0 and self.grad_clip_norm>0,'invalid AdamW/clipping settings')
        require(type(self.root_seed) is int and 0<=self.root_seed<2**63,'invalid root seed')
        require(self.verbosity in {'normal','detailed','failure'} and bool(self.validation_name) and bool(self.run_tag),'invalid reporting/run identity')
        require(type(self.full_training_enabled) is bool and type(self.overwrite_checkpoint) is bool,'launch/overwrite controls must be bool')
        if not self.source_id:
            from .diagnostics import source_identity
            object.__setattr__(self,'source_id',source_identity(Path(__file__).resolve().parents[2])['sha256'])

    @property
    def identity(self):return digest(asdict(self))
    @property
    def run_id(self):return f'{self.architecture}-{self.identity[:24]}'


@dataclass
class TrainingState:
    architecture: str
    run_id: str
    stage: str='stage7A'
    phase: str='created'
    successful: int=0
    attempted: int=0
    skipped: int=0
    failed: int=0
    accumulation_active: bool=False
    accumulation_microbatches: int=0
    accumulation_cases: int=0
    cases_processed: int=0
    queries_processed: int=0
    microbatches_processed: int=0
    sampler_cursor: int=0
    pde_counts: dict=field(default_factory=dict)
    validation_history: list=field(default_factory=list)
    checkpoint_history: list=field(default_factory=list)
    scheduler_state: dict=field(default_factory=dict)
    rng_sampler_state: dict=field(default_factory=dict)
    optimizer_state: dict=field(default_factory=lambda:{'family':'AdamW','checkpoint_storage':'serialized with tensor state'})
    recent_steps: list=field(default_factory=list)
    elapsed_seconds: float=0.
    validation_seconds: float=0.
    saving_seconds: float=0.
    last_error: str | None=None
    last_checkpoint: str | None=None
    validation_metric_name: str='common.field_normalized_MSE'
    best_validation_metric: float | None=None
    best_validation_step: int | None=None
    best_validation_weights: str | None=None
    best_checkpoint: str | None=None
    persistence: dict=field(default_factory=lambda:{'local_status':'not_started','Drive_status':'disabled','backlog':[]})


@dataclass
class LossReport:
    total: object # differentiable scalar tensor, including B's CPU64 bridge
    unweighted: dict
    weighted: dict
    meanings: dict
    counts: dict
    diagnostics: dict=field(default_factory=dict)


@dataclass(frozen=True)
class StepReport:
    architecture: str
    run_id: str
    attempted: int
    successful: int
    skipped: int
    failed: int
    status: str
    total: float | None
    unweighted: dict
    weighted: dict
    meanings: dict
    counts: dict
    learning_rate: float
    next_learning_rate: float
    gradient_before_clip: float | None
    gradient_after_clip: float | None
    timings: dict
    finite: bool
    memory: dict
    case_ids: tuple
    details: dict=field(default_factory=dict)
    error: str | None=None


@dataclass(frozen=True)
class ValidationReport:
    architecture: str
    checkpoint_identity: str
    case_set_identity: str
    manifest_name: str
    successful_update: int
    metrics: dict
    architecture_metrics: dict
    counts: dict
    elapsed_seconds: float
    device: str
    precision: str
    units: dict=field(default_factory=lambda:{'physical_RMSE':'physical quantity units (V1 nondimensional)','normalized_MSE':'training quantity RMS standardized','coefficient_MSE':'active prior-standardized','sensor_NLL':'Gaussian negative log likelihood sum, diagnostic only'})
