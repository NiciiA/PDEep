"""Explicit fresh/resume/inference suite planning; D never enters an optimizer."""
from dataclasses import dataclass,asdict
from pathlib import Path
from .core import require,digest,decode
from .artifact_io import read_manifest
from .artifact_drive import PersistencePolicy

@dataclass(frozen=True)
class SuiteEntry:
    architecture: str
    action: str # fresh, resume, inference
    run_root: str
    config: object=None
    checkpoint: str | None=None
    device: str='cpu'
    persistence: PersistencePolicy=PersistencePolicy()

    def __post_init__(self):
        require(self.architecture in {'A','C','B0','B1','D'} and self.action in {'fresh','resume','inference'},'exact architecture/action required')
        require(self.architecture!='D' or self.action=='inference','D is evaluation-only')
        if self.action=='fresh':require(self.config is not None and self.checkpoint is None,'fresh run needs resolved config and no checkpoint')
        else:require(self.checkpoint is not None,'resume/inference requires an explicit artifact')
        if self.config is not None:require(self.config.architecture==self.architecture,'suite config/architecture mismatch')


def preview_suite(entries):
    require(bool(entries) and len({e.architecture for e in entries})==len(entries),'suite must explicitly select unique architecture identities/order')
    rows=[]
    for i,entry in enumerate(entries):
        config=entry.config;existing=None;identity=None;completed=0
        if entry.checkpoint is not None:
            from .artifact_metadata import validate_metadata
            kind='training_checkpoint' if entry.action=='resume' else 'inference_export'
            existing=read_manifest(entry.checkpoint,kind=kind,architecture=entry.architecture);validate_metadata(existing)
            completed=existing['successful_updates'];identity=existing['artifact_id']
            if entry.action=='resume':
                from .train_types import ResolvedTrainConfig
                stored=decode(ResolvedTrainConfig,existing['training_config'])
                require(config is None or config.identity==stored.identity,'resume cannot silently change the run configuration/schedule')
                config=stored
            elif config is None and existing['training_config'] is not None:
                from .train_types import ResolvedTrainConfig
                config=decode(ResolvedTrainConfig,existing['training_config'])
        destination=None
        if config is not None:
            destination=Path(entry.run_root)/entry.architecture/config.run_id
            if entry.action=='fresh':require(not destination.exists(),'fresh run destination exists; explicitly choose resume or a new run_tag')
            if entry.action=='resume':require(Path(entry.checkpoint).resolve().parent.parent==destination.resolve(),'resume run_root differs; explicitly fork/relocate with load_resumable instead of mixing runs')
        rows.append({'order':i,'architecture':entry.architecture,'action':entry.action,'requires_training':entry.action!='inference',
                     'resolved_config':None if config is None else asdict(config),'config_identity':None if config is None else config.identity,
                     'training_budget':0 if entry.action=='inference' else config.updates,'completed_successful_updates':completed,
                     'origin_declared_training_budget':None if config is None else config.updates,
                     'remaining_successful_updates':0 if entry.action=='inference' else max(config.updates-completed,0),
                     'validation_budget':None if config is None else {'cases_per_PDE':config.validation_cases_per_pde,'queries_per_case':config.validation_query_count,'cadence':config.validation_every},
                     'microbatch':None if config is None else config.microbatch,'accumulation':None if config is None else config.accumulation,
                     'effective_batch':None if config is None else config.effective_batch,
                     'device':entry.device if entry.action=='inference' else config.device,
                     'precision':'CPU float64' if entry.architecture=='D' else 'float32; explicit CPU64 B physics',
                     'checkpoint_dependency':entry.checkpoint,'compatible_artifact_id':identity,
                     'normalizer_identity':None if existing is None else existing['normalization']['identity'],
                     'existing_compatible_checkpoint':entry.checkpoint,
                     'outputs':{'run_directory':None if destination is None else str(destination),'checkpoints':None if destination is None else str(destination/'checkpoints'),
                                'inference':None if destination is None else str(destination/'inference')},
                     'Drive_enabled':entry.persistence.drive_enabled,'persistence_policy':asdict(entry.persistence)})
    return {'version':'explicit-suite-v2','execution_order':[e.architecture for e in entries],'entries':rows,'identity':digest(rows),
            'policy':'no auto retrain; exact architecture + explicit fresh/resume choice; full training requires both guards'}


def execute_suite(entries,normalizers,*,execute_training=False,launch_full=False,expected_preview_identity=None):
    """Validate the ENTIRE suite before constructing/training its first model."""
    preview=preview_suite(entries)
    require(expected_preview_identity is None or expected_preview_identity==preview['identity'],'suite changed after preview')
    if any(r['remaining_successful_updates'] for r in preview['entries']):require(execute_training,'training disabled: explicitly set execute_training=True after reviewing suite')
    for row in preview['entries']:
        c=row['resolved_config']
        require(row['normalizer_identity'] is None or row['normalizer_identity']==normalizers.identity,'suite normalization differs from the selected artifacts')
        require(c is None or c['root_seed']==normalizers.root_seed,'suite data root seed differs from frozen normalization')
        if row['remaining_successful_updates'] and c['preset'] in {'research','large'}:
            require(c['full_training_enabled'] and launch_full,'full training requires config.full_training_enabled AND launch_full=True')
    result={}
    for entry,row in zip(entries,preview['entries']):
        if entry.action=='inference':
            from .inference_artifacts import load_inference
            result[entry.architecture]=load_inference(entry.checkpoint,expected_architecture=entry.architecture,device=entry.device)
            continue
        from .training import TrainingEngine
        from .train_checkpoint import load_checkpoint
        if entry.action=='fresh':engine=TrainingEngine(entry.config,normalizers,entry.run_root,persistence=entry.persistence)
        else:
            engine=load_checkpoint(entry.checkpoint,expected_architecture=entry.architecture,restore_rng=True,probe=False)
            require(engine.normalizers.identity==normalizers.identity,'suite normalizers differ from checkpoint')
            engine.persistence_policy=entry.persistence
        engine.run(launch_full=launch_full);result[entry.architecture]=engine
    return result
