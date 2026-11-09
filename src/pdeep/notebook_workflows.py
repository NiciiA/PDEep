"""Everyday notebook orchestration around the existing verified package APIs.

No optimizer, update loop, scientific loss, sampler or solver is implemented here.
RUN_MODE is the single user-facing launch action; engine guards remain in force.
"""
from dataclasses import dataclass,asdict,fields
from pathlib import Path
import json,uuid
import torch
from .core import require,decode,digest
from .runtime import capabilities,resolve_device
from .diagnostics import source_identity
from .data import calibrate,Normalizers,CaseDataset,observations,case_coefficients,evaluation_grid
from .reference import canonical_spec,evaluate_batch
from .train_types import ResolvedTrainConfig
from .experiment_suite import SuiteEntry,preview_suite
from .artifact_drive import PersistencePolicy
from .artifact_io import read_manifest,read_payload,atomic_json
from .artifact_metadata import validate_metadata,model_inventory
from .artifact_tensors import state_identity
from .models import select_architecture

NEURAL=('A','B0','B1','C')
FAMILIES=('poisson','heat','wave')


def choose_device(requested='auto',*,saved_device=None):
    if requested=='auto':
        chosen=saved_device or next(name for name in ('cuda','mps','cpu') if next(c.available for c in capabilities() if c.device==name))
        reason='saved training device (resume)' if saved_device else 'first available: CUDA, MPS, CPU'
    else:chosen=requested;reason='explicit user selection'
    resolve_device(chosen,'float32')
    require(saved_device is None or chosen==saved_device,'resume preserves the saved device/configuration; select DEVICE='+repr(saved_device)+' (cross-device training migration is not implicit)')
    return chosen,f'DEVICE={requested!r} selected {chosen}/float32: {reason}. References and B/D physics stay CPU float64. No runtime fallback.'

@dataclass
class WorkflowContext:
    root: Path
    device: str
    device_message: str
    normalizers: Normalizers
    source_id: str
    implementation_id: str
    output_root: Path | None=None

    @property
    def output_directory(self):return self.output_root or self.root/'artifacts'

    def require_current(self):
        require(implementation_identity(source_identity(self.root))==self.implementation_id,
                'package/scientific implementation changed: rerun environment initialization and prepare the plan again')


def implementation_identity(source):
    # Editing RUN_MODE in an autosaved notebook is the deliberate start action,
    # not a numerical-code change. Full preview/launch sources remain recorded.
    return digest({p:h for p,h in source['files'].items() if p.startswith('src/') or p in {'docs/scientific_contract_v1.md','pyproject.toml'}})


def initialize(root=None,*,device='auto',root_seed=20261108,output_dir=None):
    root=Path(root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    require((root/'docs/scientific_contract_v1.md').is_file(),'complete editable PDEep project required')
    require(root==Path(__file__).resolve().parents[2],'initialize the checkout imported by this kernel; install the requested project and restart the kernel')
    chosen,message=choose_device(device)
    # Existing calibration API: training distribution only, frozen definitions.
    normalizers=calibrate(root_seed)
    source=source_identity(root)
    return WorkflowContext(root,chosen,message,normalizers,source['sha256'],implementation_identity(source),
                           None if output_dir is None else Path(output_dir).expanduser().resolve())


def mount_colab_drive(enabled=False,*,mount='/content/drive'):
    if not enabled:return PersistencePolicy()
    try:from google.colab import drive
    except ImportError as error:raise ValueError('ENABLE_DRIVE=True requires Colab; local training needs no Drive') from error
    drive.mount(mount)
    return PersistencePolicy(drive_enabled=True,drive_mount=mount,drive_explicitly_mounted=True)


def resolve_artifact(path):
    path=Path(path).expanduser().resolve()
    require(path.is_file(),f'checkpoint/export not found: {path}; no model substituted')
    value=json.loads(path.read_text())
    if 'manifest' in value and 'schema_version' not in value:
        require(Path(value['manifest']).name==value['manifest'],'invalid artifact pointer')
        target=path.parent/value['manifest']
        require(read_manifest(target)['artifact_id']==value['artifact_id'],'checkpoint pointer identity mismatch')
        path=target
    return path


def stage_resume_checkpoint(context,path,architecture):
    """Stage a portable/Drive artifact in the engine's existing local layout.

    Only verified bytes are copied; existing different payloads/manifests are
    never overwritten. The verified resume loader still restores all state.
    """
    import os,shutil,tempfile
    from .artifact_io import sha256
    path=resolve_artifact(path);meta=read_manifest(path,kind='training_checkpoint',architecture=architecture)
    standard=(path.parent.name=='checkpoints' and path.parent.parent.name==meta['run_id'] and path.parent.parent.parent.name==architecture)
    if standard and path.is_relative_to(context.output_directory):return path
    destination=context.output_directory/'training_runs'/architecture/meta['run_id']/'checkpoints'
    destination.mkdir(parents=True,exist_ok=True);target=destination/path.name
    if target.exists():
        require(read_manifest(target)['artifact_id']==meta['artifact_id'],'different local checkpoint exists; select a separate recovery root')
        return target
    with tempfile.TemporaryDirectory(prefix='.staging-',dir=destination) as tmp:
        tmp=Path(tmp)
        for name in (*meta['files'],path.name):shutil.copyfile(path if name==path.name else path.parent/name,tmp/name)
        read_manifest(tmp/path.name)
        for name,info in meta['files'].items():
            final=destination/name
            if final.exists():require(sha256(final)==info['sha256'],'different local payload exists; no overwrite')
            else:os.link(tmp/name,final)
        os.link(tmp/path.name,target) # Manifest last, after copied payload verification.
    print('Verified portable checkpoint staged locally:',target)
    return target


def training_status(metadata):
    count=metadata['successful_updates'];cfg=metadata.get('training_config')
    if metadata['architecture_id']=='D':return 'exact; no training'
    if count==0:return 'untrained'
    if cfg is None:return 'partially trained'
    if cfg['preset'] in {'smoke','bounded'}:return cfg['preset']
    return 'trained' if count>=cfg['updates'] else 'partially trained'


def load_selected(path,architecture,*,device='cpu',allow_partial=False):
    """Read an explicitly selected v2 export OR training snapshot, without an engine.

    Training snapshots reuse the same manifest validation, constructor and strict
    state loading primitives. Optimizer/RNG/history state is not restored for
    inference. No artifact or original run directory is written.
    """
    path=resolve_artifact(path);meta=read_manifest(path,architecture=architecture)
    require(meta['artifact_type'] in {'inference_export','training_checkpoint'},'select an inference export or a completed training checkpoint, not diagnostics')
    norm,vocab=validate_metadata(meta);status=training_status(meta)
    require(architecture=='D' or status=='trained' or allow_partial,
            f'{architecture}: {status}, {meta["successful_updates"]} updates; not a completed research model. Explicitly set ALLOW_PARTIAL=True to inspect it.')
    if meta['artifact_type']=='inference_export':
        from .inference_artifacts import load_inference
        loaded=load_inference(path,expected_architecture=architecture,device='cpu' if architecture=='D' else device)
        model=loaded.model
    else:
        require(architecture in NEURAL,'D has no neural resumable checkpoint')
        _,payload=read_payload(path,kind='training_checkpoint',architecture=architecture,required=('model','config','optimizer','scheduler','rng'))
        require(payload['architecture']==architecture and payload['version']=='train-engine-v2','training payload architecture/version incompatible')
        cfg=decode(ResolvedTrainConfig,payload['config'])
        require(cfg.identity==meta['config_identity'],'training configuration identity mismatch')
        with torch.random.fork_rng(devices=[]):model=select_architecture(architecture,normalizers=norm,vocabulary=vocab,root_seed=cfg.root_seed)
        validate_metadata(meta,model=model);model.load_state_dict(payload['model'],strict=True)
        require(state_identity(model)==meta['component_inventory']['model_state_identity'],'model state identity mismatch')
        resolved,_=resolve_device(device,'float32');model.to(resolved).eval()
    return model,norm,meta,status,str(path)

@dataclass
class TrainingPlan:
    context: WorkflowContext
    entries: tuple
    preview: dict
    settings_identity: str
    settings: dict

    @property
    def identity(self):return digest({'source':self.context.source_id,'normalizers':self.context.normalizers.identity,'settings':self.settings_identity,'preview':self.preview['identity']})


def _settings_value(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {k:_settings_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return tuple(_settings_value(v) for v in value)
    return value


def assert_plan_settings(plan,**settings):
    plan.context.require_current()
    settings={**settings,'persistence':asdict(settings['persistence'])}
    require(digest(_settings_value(settings))==plan.settings_identity,'training settings changed: rerun the resolved preview and model construction before launch')


def prepare_training(context,*,models=('B1',),pde_families=FAMILIES,track='T1',device='auto',preset='research',updates=10000,
                     action='fresh',resume_checkpoint=None,save_every=250,validate_every=250,persistence=None,advanced=None,run_name=None):
    context.require_current();require(track=='T1','only the verified square-domain T1 benchmark is implemented')
    require(bool(models) and len(set(models))==len(models) and set(models)<=set(NEURAL),'select unique A/B0/B1/C models; D is evaluation-only')
    require(action in {'fresh','resume'},'ACTION must be fresh or resume')
    require(action!='fresh' or resume_checkpoint is None,'fresh action requires RESUME_CHECKPOINT=None; choose ACTION=resume to continue a checkpoint')
    require(bool(pde_families) and len(set(pde_families))==len(pde_families) and set(pde_families)<=set(FAMILIES),'unsupported PDE selection')
    advanced=advanced or {};policy=persistence or PersistencePolicy();entries=[]
    allowed={'learning_rate','microbatch','accumulation','effective_batch','capacity','query_count','weight_decay','grad_clip_norm',
             'scheduler','warmup_updates','minimum_lr_fraction','validation_cases_per_pde','validation_query_count','validation_name',
             'training_case_ids','verbosity','rolling_window'}
    require(set(advanced)<=allowed|set(NEURAL),'unknown advanced setting')
    tag=run_name or ('notebook-'+uuid.uuid4().hex[:12])
    for name in models:
        extra={k:v for k,v in advanced.items() if k in allowed};extra.update(advanced.get(name,{}))
        require(set(extra)<=allowed,'unknown model-specific advanced setting')
        if action=='fresh':
            chosen,note=choose_device(device)
            cfg=ResolvedTrainConfig(name,preset=preset,pdes=tuple(pde_families),probabilities=tuple(1/len(pde_families) for _ in pde_families),
                 inverse_track=track,device=chosen,updates=updates,checkpoint_every=save_every,validation_every=validate_every,
                 root_seed=context.normalizers.root_seed,source_id=context.source_id,run_tag=tag,
                 full_training_enabled=preset in {'research','large'},**extra)
            entry=SuiteEntry(name,'fresh',str(context.output_directory/'training_runs'),config=cfg,persistence=policy)
        else:
            require(resume_checkpoint is not None,'ACTION=resume requires RESUME_CHECKPOINT')
            require(len(models)==1 or isinstance(resume_checkpoint,dict),'a suite resume requires {architecture: checkpoint} paths')
            selected=resume_checkpoint.get(name) if isinstance(resume_checkpoint,dict) else resume_checkpoint
            require(selected is not None,f'resume path missing for {name}')
            path=resolve_artifact(selected);meta=read_manifest(path,kind='training_checkpoint',architecture=name);norm,_=validate_metadata(meta)
            require(norm.identity==context.normalizers.identity,'resume normalizers differ from this environment')
            cfg=decode(ResolvedTrainConfig,meta['training_config']);chosen,note=choose_device(device,saved_device=cfg.device)
            require(tuple(pde_families)==cfg.pdes and track==cfg.inverse_track,'resume preserves the saved PDE families and track')
            require(not advanced,'resume uses saved hyperparameters/batches; clear ADVANCED instead of silently changing state')
            path=stage_resume_checkpoint(context,path,name)
            entry=SuiteEntry(name,'resume',str(path.parent.parent.parent.parent),config=cfg,checkpoint=str(path),persistence=policy)
        entries.append(entry)
    preview=preview_suite(tuple(entries))
    for row in preview['entries']:
        row['estimated_parameters']={'A':596956,'B0':399428,'B1':682924,'C':527796}[row['architecture']]
        row['selection_status']='fresh architecture' if action=='fresh' else 'saved training state; full RNG/Adam/scheduler restored at launch'
    settings={'models':tuple(models),'pde_families':tuple(pde_families),'track':track,'device':device,'preset':preset,'updates':updates,
              'action':action,'resume_checkpoint':resume_checkpoint,'save_every':save_every,'validate_every':validate_every,
              'persistence':asdict(policy),'advanced':advanced,'run_name':run_name}
    return TrainingPlan(context,tuple(entries),preview,digest(_settings_value(settings)),settings)


def construct_models(plan):
    plan.context.require_current();result={}
    for entry in plan.entries:
        with torch.random.fork_rng(devices=[]):
            model=select_architecture(entry.architecture,normalizers=plan.context.normalizers,root_seed=entry.config.root_seed)
        model.to(entry.config.device);result[entry.architecture]=model
    return result


def plan_table(plan):
    lines=['| Model / order | PDEs / track | Action | Updates (previous → target) | LR / schedule | Micro × accum / E | Validation / save | Device / precision | Params | Drive |',
           '|---|---|---|---|---|---|---|---|---:|---|']
    for r in plan.preview['entries']:
        c=r['resolved_config'];lines.append(f"| {r['architecture']} / {r['order']} | {', '.join(c['pdes'])} / {c['inverse_track']} | {r['action']} | {r['completed_successful_updates']} → {c['updates']} | {c['learning_rate']} / {c['scheduler']} | {c['microbatch']} × {c['accumulation']} / {c['effective_batch']} | {c['validation_every']} / {c['checkpoint_every']} | {c['device']} / {c['precision']} | {r['estimated_parameters']} | {r['Drive_enabled']} |")
    return '\n'.join(lines)


def launch_training(plan,*,run_mode='preview',models=None,notebook=True,monitor=None):
    """One explicit action calls the existing engine.run; no update loop here."""
    plan.context.require_current();require(run_mode in {'preview','train','resume'},'RUN_MODE must be preview, train or resume')
    if run_mode=='preview':return {}
    require(run_mode!='resume' or all(e.action=='resume' for e in plan.entries),'RUN_MODE=resume requires an ACTION=resume plan and checkpoint paths')
    # Revalidate destinations/checkpoint metadata immediately before any launch.
    live_preview=preview_suite(plan.entries)
    require(live_preview['identity']==plan.preview['identity'],'plan changed; rerun resolved preview before launch')
    from .training import TrainingEngine
    from .train_checkpoint import load_checkpoint
    from .workflow_plotting import TrainingMonitor
    from .inference_artifacts import export_inference
    selected=models if models is not None else construct_models(plan)
    require(set(selected)=={e.architecture for e in plan.entries},'constructed model selection differs from launch plan')
    monitor=monitor or TrainingMonitor(notebook=notebook,suite_budgets={e.architecture:e.config.updates for e in plan.entries})
    print('Launch source identity:',source_identity(plan.context.root)['sha256'],'| preview source:',plan.context.source_id)
    results={}
    for entry in plan.entries:
        if entry.action=='fresh':
            engine=TrainingEngine(entry.config,plan.context.normalizers,entry.run_root,model=selected[entry.architecture],monitor=monitor,persistence=entry.persistence)
        else:
            engine=load_checkpoint(entry.checkpoint,expected_architecture=entry.architecture,monitor=monitor,restore_rng=True,probe=False)
            engine.persistence_policy=entry.persistence
        if hasattr(monitor,'bind'):monitor.bind(engine)
        engine.state.stage='everyday_'+engine.config.preset
        try:steps=engine.run(launch_full=True) # Existing guard is authorized by RUN_MODE above.
        except KeyboardInterrupt:
            results[entry.architecture]={'engine':engine,'interrupted':True,'inference_export':None,'last_checkpoint':engine.state.last_checkpoint}
            print('Interrupted; resume the last completed checkpoint:',engine.state.last_checkpoint)
            break
        parent=read_manifest(engine.state.last_checkpoint)
        exported=export_inference(engine.model,engine.run_dir/'inference',architecture=entry.architecture,normalizers=engine.normalizers,
                                  run_id=engine.config.run_id,config=engine.config,successful=engine.state.successful,
                                  parents=[{'artifact_id':parent['artifact_id'],'path_hint':engine.state.last_checkpoint}],
                                  status=engine.config.preset if engine.config.preset in {'smoke','bounded'} else 'trained')
        results[entry.architecture]={'engine':engine,'interrupted':False,'inference_export':str(exported),
                                     'last_checkpoint':engine.state.last_checkpoint,'new_successful_updates':len([s for s in steps if s.status=='successful'])}
    return results


def load_evaluation_selection(context,checkpoints,*,allow_partial=False):
    from .evaluation_inference import Method
    context.require_current();require(set(checkpoints)<=set(NEURAL),'CHECKPOINTS maps A/B0/B1/C to explicit artifact paths; D is added exactly')
    methods=[]
    for name,path in checkpoints.items():
        model,norm,meta,status,resolved=load_selected(path,name,device=context.device,allow_partial=allow_partial)
        require(norm.identity==context.normalizers.identity,'selected models have incompatible frozen normalization')
        c=meta.get('training_config') or {}
        train={**c,'successful_updates':meta['successful_updates'],'declared_updates':c.get('updates'),
               'config_identity':meta.get('config_identity'),'source_identity':meta['source']['sha256']}
        methods.append(Method(name,name,status,model,resolved,train))
    methods.append(Method('D','D','exact; no training',checkpoint='exact-T1-Gaussian'))
    return methods


def make_evaluation_plan(context,*,pdes=FAMILIES,case_ids=(0,),plan_path=None,observation_diagnostics=False):
    from .evaluation import EvaluationConfig
    from .evaluation_plan import EvaluationPlan,make_plan
    context.require_current();cfg=EvaluationConfig(observation_diagnostics=observation_diagnostics)
    if plan_path is None:plan=make_plan(context.normalizers,source_id=context.source_id,config_id=cfg.identity,pdes=tuple(pdes),case_ids=tuple(case_ids))
    else:
        plan=EvaluationPlan.load(plan_path)
        require(plan.pdes==tuple(pdes) and plan.normalizer_identity==context.normalizers.identity and plan.config_identity==cfg.identity,
                'loaded plan differs from selected PDEs/normalizers/diagnostic settings; match its explicit settings')
    return plan,cfg


def evaluate_selected(context,methods,plan,config):
    from .evaluation import evaluate
    context.require_current()
    output=context.root/'reports/evaluation_runs'/('notebook-'+uuid.uuid4().hex[:12])
    return evaluate(plan,methods,context.normalizers,config=config,output=output)


def physical_example(context,pde='wave',case_id=0):
    require(pde in FAMILIES,'select poisson, heat or wave; irregular geometries and new PDEs are not implemented')
    ds=CaseDataset(context.normalizers.root_seed);case=ds.case('validation',pde,case_id)
    return case,observations(ds,case,context.normalizers,set_id='everyday-explore-v1')


def preserved_verification(context_root,*,fresh=False):
    """Preserve old proof creation provenance; revalidate unchanged implementation.

    Adding UI notebooks changes the whole-project hash, not the previously tested
    numerical implementation. Old numerical/local-writer files must still match.
    The optional Drive destination change is rechecked only on the disabled route
    actually covered by the old proof; hosted Drive is never inferred from it.
    """
    root=Path(context_root);active=source_identity(root)
    if fresh:
        from .persistence_checks import run_persistence_checks
        record=run_persistence_checks(root,fresh=True)
    else:
        pointer=json.loads((root/'reports/latest_persistence.json').read_text());record=json.loads((root/pointer['report']).read_text())
        checked={p:h for p,h in record['source']['files'].items() if p.startswith('src/') or p in {'docs/scientific_contract_v1.md','scripts/check_persistence.py'}}
        changed=[p for p,h in checked.items() if active['files'].get(p)!=h]
        require(set(changed)<={'src/pdeep/artifact_drive.py'},'verified numerical/persistence implementation changed; explicitly rerun its bounded verification')
        for path in record['verified_artifacts']:read_manifest(path)
        if changed:
            from .artifact_drive import sync_verified
            require(record['Drive']['actual_Drive_enabled'] is False and record['Drive']['real_Drive_tested'] is False,
                    'old proof used Drive; explicit persistence implementation verification required')
            for path in record['verified_artifacts']:
                meta=read_manifest(path)
                if meta['artifact_type']!='training_checkpoint':continue
                require(meta['persistence_policy']['drive_enabled'] is False,'old proof checkpoint enabled Drive')
                status=sync_verified(path,PersistencePolicy())
                require(status['local_verified'] and status['Drive_status']=='disabled' and not status['Drive_verified'],
                        'disabled persistence implementation changed; bounded verification required')
    return {**record,'verified_against_source_identity':active['sha256'],
            'proof_creation_source_identity':record['source_identity'],'UI_revalidation_only':not fresh,
            'persistence_revalidation_scope':'verified local artifacts and Drive-disabled route only; hosted Drive untested' if not fresh else 'fresh bounded suite'}


def checkpoint_catalog(root,*,include_partial=False,limit=12):
    rows=[]
    for path in (Path(root)/'artifacts').rglob('*.json'):
        try:
            m=json.loads(path.read_text())
            if m.get('schema_version')!='pdeep-artifact-v2' or m.get('architecture_id') not in NEURAL or m.get('artifact_type') not in {'inference_export','training_checkpoint'}:continue
            status=training_status(m)
            if status!='trained' and not include_partial:continue
            rows.append({'architecture':m['architecture_id'],'status':status,'successful_updates':m['successful_updates'],
                         'kind':m['artifact_type'],'path':str(path.resolve()),'created':m['created_UTC']})
        except (ValueError,KeyError):continue
    return sorted(rows,key=lambda v:v['created'],reverse=True)[:limit]


def exploration_method(context,*,architecture='D',checkpoint=None,allow_partial=False):
    from .evaluation_inference import Method
    if architecture=='D':
        require(checkpoint is None,'D recomputes the exact posterior for the selected case; no neural checkpoint needed')
        return Method('D','D','exact; no training',checkpoint='exact-T1-Gaussian')
    require(checkpoint is not None,'select an explicit model checkpoint; no trained neural model is substituted')
    model,norm,metadata,status,path=load_selected(checkpoint,architecture,device=context.device,allow_partial=allow_partial)
    require(norm.identity==context.normalizers.identity,'model normalizer differs from selected example')
    return Method(architecture,architecture,status,model,path,{'successful_updates':metadata['successful_updates']})
