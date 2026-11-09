"""Versioned checkpoint dispatch: v2 complete state; explicit v1 read support.

New writes use pdeep-artifact-v2. Historical train-engine-v1 artifacts remain
readable and are never silently relabeled as full-RNG v2 checkpoints.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,os,tempfile
import torch
from .core import require,decode
from .b_artifacts import tree_cpu
from .train_types import ResolvedTrainConfig,TrainingState
from .data import Normalizers


def save_checkpoint_v1(engine):
    s=engine.state;c=engine.config
    require(s.successful>0 and not s.accumulation_active,'checkpoint only after a completed successful update, never mid-accumulation')
    directory=engine.run_dir/'checkpoints';directory.mkdir(exist_ok=True)
    manifest=directory/f'update-{s.successful:08d}.json';path=directory/f'update-{s.successful:08d}.pt'
    require(c.overwrite_checkpoint or not path.exists() and not manifest.exists(),'checkpoint exists; deliberate overwrite configuration required')
    state=asdict(s);state.update(phase='ready',last_checkpoint=str(manifest))
    state['checkpoint_history']=state['checkpoint_history']+[{'step':s.successful,'path':str(manifest)}]
    payload={'version':'train-engine-v1','architecture':c.architecture,'config':asdict(c),'state':state,
             'model':tree_cpu(engine.model.state_dict()),'optimizer':tree_cpu(engine.optimizer.state_dict()),'scheduler':engine.scheduler.state_dict(),
             'normalizers':asdict(engine.normalizers),'vocabulary':list(engine.model.vocabulary.quantities),
             'sampler':engine.sampler.state_dict(s.sampler_cursor),'torch_cpu_rng':torch.random.get_rng_state().clone(),'AMP':False}
    with tempfile.NamedTemporaryFile(dir=directory,delete=False) as stream:tmp=stream.name
    try:torch.save(payload,tmp);os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
    metadata={'version':'train-engine-v1','architecture':c.architecture,'run_id':s.run_id,'config_identity':c.identity,'source_id':c.source_id,
              'successful_update':s.successful,'sampler_cursor':s.sampler_cursor,'validation_manifest':engine.manifest.identity,
              'payload':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with tempfile.NamedTemporaryFile(mode='w',dir=directory,delete=False) as stream:json.dump(metadata,stream,indent=2);stream.write('\n');tmp=stream.name
    os.replace(tmp,manifest);return manifest


def load_checkpoint_v1(manifest,*,expected_architecture,monitor=None,restore_rng=False,probe=True):
    manifest=Path(manifest);meta=json.loads(manifest.read_text())
    require(meta['version']=='train-engine-v1' and meta['architecture']==expected_architecture,'checkpoint format/architecture mismatch')
    path=manifest.parent/meta['payload'];require(path.name==meta['payload'] and path.parent==manifest.parent,'invalid checkpoint payload path')
    require(hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256'],'checkpoint checksum mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    config=decode(ResolvedTrainConfig,payload['config'])
    require(config.identity==meta['config_identity'] and config.run_id==meta['run_id'] and config.architecture==expected_architecture,'checkpoint config/identity mismatch')
    state=TrainingState(**payload['state']);require(not state.accumulation_active and state.successful==meta['successful_update'],'checkpoint contains unfinished or inconsistent update state')
    from .training import TrainingEngine
    from .models import select_architecture
    norm=decode(Normalizers,payload['normalizers']);model=select_architecture(expected_architecture,normalizers=norm,root_seed=config.root_seed).to(config.device)
    model.load_state_dict(payload['model'],strict=True)
    require(list(model.vocabulary.quantities)==payload['vocabulary'],'checkpoint graph vocabulary mismatch')
    root=manifest.parent.parent.parent.parent
    engine=TrainingEngine(config,norm,root,model=model,monitor=monitor,probe=probe,allow_existing_run=True)
    require(engine.run_dir==manifest.parent.parent,'checkpoint/run directory mismatch')
    require(engine.manifest.identity==meta['validation_manifest'],'fixed validation manifest mismatch')
    engine.optimizer.load_state_dict(payload['optimizer']);engine.scheduler.load_state_dict(payload['scheduler'])
    require(engine.scheduler.completed==state.successful and payload['sampler']==engine.sampler.state_dict(state.sampler_cursor),'checkpoint scheduler/sampler mismatch')
    engine.state=state;engine.state.phase='ready';engine._elapsed_base=state.elapsed_seconds
    engine._started=__import__('time').monotonic()
    if restore_rng:torch.random.set_rng_state(payload['torch_cpu_rng'])
    return engine


def save_checkpoint(engine):
    from .experiment_artifacts import save_resumable
    return save_resumable(engine)


def load_checkpoint(manifest,*,expected_architecture,monitor=None,restore_rng=None,probe=True,run_root=None):
    manifest=Path(manifest);record=json.loads(manifest.read_text())
    if 'manifest' in record and 'schema_version' not in record and 'version' not in record:
        from .artifact_io import read_manifest
        require(Path(record['manifest']).name==record['manifest'],'invalid checkpoint pointer path')
        target=manifest.parent/record['manifest']
        require(read_manifest(target)['artifact_id']==record['artifact_id'],'checkpoint pointer identity mismatch')
        return load_checkpoint(target,expected_architecture=expected_architecture,monitor=monitor,restore_rng=restore_rng,probe=probe,run_root=run_root)
    if record.get('version')=='train-engine-v1':
        require(run_root is None,'legacy checkpoint relocation requires an explicit migration; v1 is not full-RNG v2')
        engine=load_checkpoint_v1(manifest,expected_architecture=expected_architecture,monitor=monitor,
                                  restore_rng=False if restore_rng is None else restore_rng,probe=probe)
        engine.parent_checkpoints=[{'kind':'train-engine-v1','path_hint':str(manifest.resolve()),'sha256':record['sha256']}]
        engine.rng_restore_limitations=['legacy v1 records only CPU torch RNG; exact Python/NumPy/device continuation is not certified']
        return engine
    from .experiment_artifacts import load_resumable
    return load_resumable(manifest,expected_architecture=expected_architecture,monitor=monitor,
                          restore_rng_state=True if restore_rng is None else restore_rng,probe=probe,run_root=run_root)
