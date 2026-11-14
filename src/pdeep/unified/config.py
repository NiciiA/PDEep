"""Versioned all-eight-instance experiment; does not redefine benchmark splits."""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import math
from ..core import digest, require
from ..artifact_io import sha256
from ..smooth.definition import PDES, GEOMETRIES, SEEN, HELD_OUT, BENCHMARK_ID, SPLIT_COUNTS
from ..train_types import ResolvedTrainConfig
from ..smooth.training import select_device

SCHEMA = 'G1-unified-multicase-v1'
MODELS = ('A_GEO/KIN_001', 'C_GEO/FIELD_ONLY', 'C_GEO/ALL_001')
FAMILIES = tuple(GEOMETRIES)
TASKS = tuple((p, g) for p in PDES for g in FAMILIES)
PRESETS = {'smoke': (2, 2, 2), 'pilot': (32, 16, 16), 'research': (128, 32, 64), 'larger': (512, 32, 64)}
TIMES = (0., .02, .05, .1, .2, .35, .5, .75, 1.)


def implementation_identity():
    root = Path(__file__).resolve().parents[3]
    # Include dependencies used for mathematics/training, without changing their identity rules.
    paths = [*Path(__file__).parent.glob('*.py'), * (root/'src/pdeep/geometric').glob('*.py'),
             * (root/'src/pdeep/smooth').glob('*.py'), * (root/'src/pdeep/learning_diagnostics').glob('*.py'),
             * (root/'src/pdeep').glob('artifact*.py'), root/'src/pdeep/training.py']
    return digest({str(p.relative_to(root)): sha256(p) for p in sorted(paths)})


def case_index(family, ordinal):
    # Preserve old four-seen cyclic catalog exactly; geometry is an explicit identity axis.
    group = SEEN if family in SEEN else HELD_OUT
    return 4*ordinal + group.index(family)


@dataclass(frozen=True)
class DatasetConfig:
    preset: str = 'research'
    pdes: tuple = PDES
    families: tuple = FAMILIES
    counts: tuple | None = None
    conditioning_h: float = .12
    seed: int = 42
    fine_spot_checks_per_task: int = 1
    qc_points: int = 128
    reference_tolerance: float = .002
    cache_tolerance: float = .0002
    # Continuous output error budget per unit response, verified over ALL intervals.
    response_relative_tolerance: float = 1e-6
    reference_execution_source: str = ''
    stream_updates: int | None = None
    stream_queries: int | None = None
    stream_validation_queries: int = 512
    stream_grid: int = 64
    holdout_family: str | None = None
    schema: str = SCHEMA
    def __post_init__(self):
        from .provenance import reference_execution_identity
        execution_source=reference_execution_identity()
        require(not self.reference_execution_source or self.reference_execution_source==execution_source,'reference execution source changed; use a new dataset manifest, preserve physical IDs')
        object.__setattr__(self,'reference_execution_source',execution_source)
        require(self.schema == SCHEMA and self.preset in PRESETS, 'unknown unified dataset schema/preset')
        object.__setattr__(self,'stream_updates',(3 if self.preset=='smoke' else 30000) if self.stream_updates is None else self.stream_updates)
        object.__setattr__(self,'stream_queries',(128 if self.preset=='smoke' else 4096) if self.stream_queries is None else self.stream_queries)
        require(all(type(n) is int and n>0 for n in (self.stream_updates,self.stream_queries,self.stream_validation_queries,self.stream_grid)),'invalid continuous stream plan')
        object.__setattr__(self, 'counts', tuple(PRESETS[self.preset] if self.counts is None else self.counts))
        for name in ('pdes', 'families'): object.__setattr__(self, name, tuple(getattr(self, name)))
        require(self.pdes and len(set(self.pdes)) == len(self.pdes) and set(self.pdes) <= set(PDES), 'invalid PDE selection')
        require(self.families and len(set(self.families)) == len(self.families) and set(self.families) <= set(FAMILIES), 'unqualified geometry instance')
        require(len(self.counts) == 3 and all(type(n) is int and n >= 2 for n in self.counts), 'need >=2 physical cases per task/split')
        for split, n in zip(('train', 'validation', 'test'), self.counts):
            require(all(case_index(g, n-1) < SPLIT_COUNTS[split] for g in self.families), 'subset exceeds existing split pool')
        require(self.conditioning_h in (.04, .08, .12) and self.qc_points >= 8, 'invalid mesh/QC sampling')
        require(self.fine_spot_checks_per_task >= 1, 'stratified finer-reference qualification is required')
        require(0 < self.cache_tolerance <= .0002 and 0 < self.reference_tolerance <= .002 and 0 < self.response_relative_tolerance <= .0002/128, 'cannot relax numerical budgets')
        require(self.holdout_family is None or self.holdout_family in self.families, 'holdout must be declared in dataset families')
    @property
    def identity(self): return digest(asdict(self))
    @property
    def tasks(self): return tuple((p,g) for p in self.pdes for g in self.families)
    @property
    def training_tasks(self): return tuple(t for t in self.tasks if t[1] != self.holdout_family)
    @property
    def totals(self): return dict(zip(('train','validation','test'), (len(self.tasks)*n for n in self.counts)))
    def keys(self, split):
        n = self.counts[('train','validation','test').index(split)]
        return tuple((p,g,split,case_index(g,i)) for p,g in self.tasks for i in range(n))
    def preview(self):
        return {**asdict(self), 'identity': self.identity, 'tasks': len(self.tasks), 'totals': self.totals,
                'scope': 'unseen physical fields on eight familiar fixed polygon instances; NOT geometry transfer',
                'prior_calibration': 'existing frozen TRAIN-only normalizers; never recalibrated',
                'split_policy': 'versioned all-eight explicit geometry axis; original lazy namespaces/ranges unchanged',
                'reference_status': 'requires per-case QC and separately recorded finer-reference checks'}


@dataclass(frozen=True)
class TrainConfig(ResolvedTrainConfig):
    architecture: str = 'A_GEO'
    model_key: str = MODELS[0]
    preset: str = 'research'
    benchmark: str = BENCHMARK_ID
    inverse_track: str = BENCHMARK_ID
    capacity: str = 'medium'
    device: str = 'auto'
    effective_batch: int = 4
    microbatch: int = 1
    accumulation: int = 4
    query_count: int = 4096
    query_chunk: int = 2048
    updates: int = 30000
    validation_every: int = 500
    checkpoint_every: int = 500
    validation_cases_per_pde: int = 1
    validation_query_count: int = 512
    root_seed: int = 42
    scheduler: str = 'warmup_then_cosine'
    full_training_enabled: bool = True
    dataset_identity: str = ''
    experiment_id: str = ''
    spatial_order: int = 3
    time_order: int = 3
    weak_tests: int = 16
    physics_chunk: int = 256
    kinematic_queries: int = 512
    amp: str = 'auto'
    data_seed: int = 42
    def __post_init__(self):
        require(self.model_key in MODELS and self.architecture == self.model_key.split('/')[0], 'model/objective mismatch')
        require(self.benchmark == self.inverse_track == BENCHMARK_ID, 'unified uses only G1_smooth_v1')
        require(self.preset in ('smoke','pilot','research') and self.capacity in ('medium','verification'), 'unsupported capacity/preset')
        require(self.capacity != 'verification' or self.preset == 'smoke', 'verification networks are smoke fixtures only')
        require(type(self.updates) is int and self.updates >= 1 and (self.preset != 'smoke' or self.updates <= 3), 'invalid declared schedule')
        require(self.effective_batch == 4 and math.ceil(4/self.microbatch) == self.accumulation, 'effective batch remains four independent cases')
        require(type(self.data_seed) is int and 0<=self.data_seed<2**63,'invalid shared data seed')
        require(self.amp in ('auto','off'), 'unified V1 uses FP32; FP16 scaler/derivative compatibility not verified; BF16 is not assumed on T4')
        require(self.dataset_identity and self.experiment_id and Path(self.experiment_id).name == self.experiment_id, 'explicit dataset/experiment identities required')
        require(self.training_case_ids is None,'unified case identities come from the dataset, never legacy T1 indices')
        require(self.query_chunk > 0 and self.kinematic_queries == 512, 'invalid chunk or changed kinematic coverage')
        require(self.spatial_order >= 3 and self.time_order >= 3 and self.weak_tests > 0 and self.physics_chunk > 0, 'invalid weak quadrature')
        keys = {f.name for f in fields(ResolvedTrainConfig)}
        kwargs = {k:v for k,v in asdict(self).items() if k in keys}
        kwargs.update(architecture='A' if self.architecture == 'A_GEO' else 'C', benchmark='unit_square_modal_v1',
                      inverse_track='T1', capacity='canonical_v1', device=select_device(self.device), preset='bounded',
                      updates=min(self.updates,100), warmup_updates=min(self.warmup_updates, min(self.updates,100)-1),
                      source_id=self.source_id or implementation_identity())
        base = ResolvedTrainConfig(**kwargs)
        require(self.scheduler != 'warmup_then_cosine' or self.warmup_updates < self.updates, 'warmup must fit full schedule')
        preserve = {'architecture','benchmark','inverse_track','capacity','preset','updates','warmup_updates','training_case_ids'}
        for f in fields(base):
            if f.name not in preserve: object.__setattr__(self, f.name, getattr(base,f.name))
    @property
    def run_id(self): return self.model_key.replace('/','_') + '-' + self.identity[:16]
