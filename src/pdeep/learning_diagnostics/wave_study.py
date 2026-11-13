"""One focused wave comparison, configuring the existing synchronous suite.

Preview/recovery/evaluation never constructs an optimizer. Missing arms only
enter the existing successful-update engine after explicit train mode AND the
matching nonoptimizing derivative gate. No warm start between variants.
"""
from dataclasses import asdict,replace
from pathlib import Path
import json,pickle
from uuid import uuid4
from ..core import require,digest
from ..artifact_io import read_manifest,read_payload,atomic_json
from .engine import DiagnosticConfig,diagnostic_source,approved_diagnostic,SCHEMA
from .suite import SuiteSettings,prepare_diagnostic_suite,_snapshot,_checkpoint_path
from .study_controls import C_VARIANTS,STAGED

VARIANTS=('FIELD_ONLY','KIN_001','STAGED_001')


def prepare_wave_study(*,certificate,root,persistence=None,model='C_GEO',capacity='medium',device='auto',query_mode='dense_resampled',
                       query_count=16384,query_chunk=4096,query_time_policy='stratified',updates=1000,learning_rate=3e-4,grad_clip=1.,seed=42,
                       amp_mode='auto',kinematic_weight=.001,kinematic_queries=512,variants=VARIANTS,reuse=False,action='fresh',advanced=None,checkpoint_lookup='none',resume_paths=None,suite_path=None):
    require(model in {'A_GEO','C_GEO'},'select existing A_GEO/C_GEO architecture')
    supported=VARIANTS if model=='C_GEO' else ('BASELINE','KIN_001')
    require(variants and len(set(variants))==len(variants) and set(variants)<=set(supported),'select unique wave variants in execution order')
    require(query_count in {8192,16384,32768,65536} or capacity=='verification','supported supervised totals:8192/16384/32768/65536')
    require(0<query_chunk<=query_count,'chunk must be positive and no larger than supervised Q')
    require(amp_mode in {'auto','off'},'matched derivative comparison uses common FP32, no BF16')
    require(kinematic_weight>0,'declare a positive experimental kinematic weight')

    controls={
        'FIELD_ONLY':dict(C_VARIANTS['FIELD_ONLY']),
        'KIN_001':dict(physics_weights=(0.,0.,0.),kinematic_weight=kinematic_weight,kinematic_queries=kinematic_queries,kinematic_seed=seed),
        'STAGED_001':dict(STAGED),
    }
    # Legacy default1000 is byte-for-byte the original300/700 definition. Other
    # budgets use the explicitly named proportional version, never reinterpret
    # an old STAGED checkpoint saved with absolute300/700 thresholds.
    if updates!=1000:controls['STAGED_001']['physics_schedule']='staged_fractional_30_40_30_v1'
    if model=='A_GEO':controls={'BASELINE':{},'KIN_001':dict(kinematic_weight=kinematic_weight,kinematic_queries=kinematic_queries,kinematic_seed=seed)}
    experiment='c_physics_ablation' if model=='C_GEO' else 'wave_single'
    selected={k:dict(controls[k],grad_clip_norm=grad_clip,gradient_diagnostics_every=200) for k in variants}
    settings=SuiteSettings(query_mode=query_mode,query_count=query_count,query_chunk=query_chunk,query_time_policy=query_time_policy,
        query_sampler_seed=seed,updates=updates,learning_rate=learning_rate,device=device,amp_mode=amp_mode,seed=seed,
        query_checkpointing=False,variant_settings=selected,advanced={**(advanced or {}),**({'pde':'wave','family':'square'} if model=='C_GEO' else {})})
    require(checkpoint_lookup in {'none','explicit'},'wave checkpoint lookup must be none/explicit; automatic search was removed')
    require(action in {'fresh','resume'},'wave action fresh/resume only')
    paths=dict(resume_paths or {})
    require(set(paths)<=set(variants),'explicit checkpoint keys must be selected variant names')
    require(checkpoint_lookup=='explicit' or (not paths and suite_path is None and action=='fresh'),'resume requires checkpoint_lookup=explicit and specified paths')
    require(not (suite_path and paths),'choose a specified suite snapshot OR per-variant checkpoint paths')
    require(action!='resume' or suite_path or set(paths)==set(variants),'resume requires explicit paths for every selected arm; no fresh substitution')
    identifier=('wave-kinematic-' if model=='C_GEO' else 'wave-kinematic-A-')+uuid4().hex
    # Read only caller-selected files. No old-run enumeration, glob or Drive walk.
    if suite_path is not None:
        suite_path=_explicit_manifest(suite_path,root)
        plan=prepare_diagnostic_suite((experiment,),model=model,capacity=capacity,settings=settings,certificate=certificate,root=root,
            persistence=persistence,action='resume',resume_manifest=suite_path)
    else:
        headers={}
        for label,path in paths.items():
            paths[label]=str(_explicit_manifest(path,root))
            headers[label]=read_manifest(paths[label],kind='training_checkpoint',architecture=model)
            saved=headers[label].get('config',{})
            actual='BASELINE' if model=='A_GEO' and saved.get('objective_variant')=='FULL' else saved.get('objective_variant')
            require(actual==label and tuple(saved.get('pdes',()))==('wave',),'specified checkpoint is not the requested wave arm: '+str(path))
        plan=prepare_diagnostic_suite((experiment,),model=model,capacity=capacity,settings=settings,certificate=certificate,root=root,
            persistence=persistence,action='fresh',suite_name=identifier)
    if suite_path is None:
        for label,meta in headers.items():
            cfg=DiagnosticConfig(**{**meta['config'],**{k:tuple(meta['config'][k]) for k in ('pdes','probabilities','training_families')}})
            expected=member_config(plan,meta['fixed_data_identity'],label,cfg.run_tag)
            require(approved_diagnostic(cfg),'explicit checkpoint source incompatible')
            require(meta.get('benchmark')=='G1_smooth_v1' and meta.get('scientific_schema')==SCHEMA,'explicit checkpoint benchmark/schema mismatch')
            require(meta['fixed_case']['physical_case_id']==plan.recipes[0]['physical_case'],'explicit checkpoint physical case mismatch')
            require(meta['execution_qualification']['certification_id']==certificate['certification_id'],'explicit checkpoint numerical certificate mismatch')
            require(asdict(replace(cfg,objective_variant=label,diagnostic_source_id=expected.diagnostic_source_id))==asdict(expected),'explicit checkpoint training/query configuration mismatch')
    plan.wave_force_fp32=True
    plan.wave_lookup=checkpoint_lookup;plan.wave_resume_paths=paths
    plan.wave_requested_action=action;plan.wave_reuse=False
    plan.snapshot_recovery_failures=[]
    return plan


def _explicit_manifest(value,root):
    """An exact manifest, pointer, or local suite:<ID> pointer; never discovery."""
    if str(value).startswith('suite:'):
        identifier=str(value)[6:]
        require(Path(identifier).name==identifier and identifier not in {'','.','..'},'invalid exact suite identity')
        path=Path(root)/'learning_diagnostics'/'suites'/identifier/'latest.json'
    else:path=Path(value).expanduser()
    require(path.is_file(),'explicit checkpoint/suite path does not exist: '+str(path))
    if path.name=='latest.json':
        pointer=json.loads(path.read_text())
        # Only the two associated paths recorded in this explicitly selected pointer.
        found=next((Path(p).expanduser() for p in (pointer.get('manifest'),pointer.get('Drive_manifest')) if p and Path(p).expanduser().is_file()),None)
        require(found is not None,'specified suite pointer has no accessible associated snapshot; supply its full path')
        path=found
    return path



def member_config(plan,data_identity,label,run_tag):
    """Exact suite member settings, also used by read-only recovery/probes."""
    return DiagnosticConfig(architecture=plan.model,capacity=plan.capacity,preset='smoke' if plan.capacity=='verification' else 'bounded',
        pdes=('wave',),probabilities=(1.,),training_families=('square',),cases_per_geometry=1,updates=plan.settings.updates,device=plan.settings.device,
        root_seed=plan.settings.seed,experiment=plan.experiments[0],objective_variant=label,data_identity=data_identity,run_tag=run_tag,
        bank_queries=max(128,plan.settings.query_count),query_count=plan.settings.query_count,query_revision='continuous-queries-v3',query_mode=plan.settings.query_mode,
        query_time_policy=plan.settings.query_time_policy,query_sampler_seed=plan.settings.query_sampler_seed,query_chunk=plan.settings.query_chunk,
        query_checkpointing=plan.settings.query_checkpointing,amp_mode=plan.settings.amp_mode,resolved_amp_mode='off',memory_target_gb=plan.settings.memory_target_gb,
        learning_rate=plan.settings.learning_rate,scheduler=plan.settings.scheduler,**plan.settings.variant_settings[label],
        **{k:v for k,v in plan.settings.advanced.items() if k not in {'pde','family','grid_size'}})

def recover_wave_checkpoints(plan,verification):
    """No optimizer or reference solve; strict scientific/data/config matching.

    Partial arms must also satisfy existing exact Torch/device/RNG resume guards.
    Complete arms are used through the SAME suite verification, not filename
    guessing. A valid but different budget/query/schedule is not a matched arm.
    """
    require(verification.get('plan_identity')==plan.identity and verification.get('passed'),'matching derivative/data gate required for recovery')
    if getattr(plan,'wave_lookup','none')!='explicit':return {'status':'lookup disabled','selected':{},'missing':list(plan.settings.variant_settings),'optimizer_updates':0}
    if plan.action=='resume':
        runs=plan.snapshot['entries'][plan.experiments[0]].get('runs',{})
        for run in runs.values():
            if run.get('checkpoint'):
                meta,payload=read_payload(_checkpoint_path(run),required=('optimizer','scheduler','rng','sampler','model','model_config','training_state'))
                cfg=DiagnosticConfig(**{**meta['config'],**{k:tuple(meta['config'][k]) for k in ('pdes','probabilities','training_families')}})
                _validate_resume_state(meta,payload,cfg,plan.certificate)
        selected={label:{'path':_checkpoint_path(run),'successful':run['last_successful_update'],
            'origin':'restored from completed checkpoint' if run['last_successful_update']>=plan.settings.updates else 'resumed'}
            for label,run in runs.items() if run.get('checkpoint')}
        return {'status':'verified saved suite selected','selected':selected,'missing':[label for label in plan.settings.variant_settings if label not in selected],
            'rejected':getattr(plan,'snapshot_recovery_failures',[]),'roots':[],'optimizer_updates':0}

    paths={str(path):None for path in getattr(plan,'wave_resume_paths',{}).values()};rejected=[];selected={};roots=[]
    from ..smooth.compatibility import approved_predecessor
    from ..smooth.training import select_device
    import torch
    for path,run in paths.items():
        try:
            meta=read_manifest(path,kind='training_checkpoint')
            if meta['architecture_id']!=plan.model:continue
            cfg=meta.get('config',{});original_label=cfg.get('objective_variant')
            label='BASELINE' if plan.model=='A_GEO' and original_label=='FULL' else original_label
            require(label in plan.settings.variant_settings and tuple(cfg.get('pdes',()))==('wave',),'explicit checkpoint is not a selected wave arm')
            require(str(plan.wave_resume_paths.get(label))==path,'specified checkpoint key does not match its stored variant')
            meta,p=read_payload(path,kind='training_checkpoint',architecture=plan.model,required=('model','model_config','normalizers','optimizer','scheduler','rng','sampler','training_state'))
            c=DiagnosticConfig(**{**cfg,**{k:tuple(cfg[k]) for k in ('pdes','probabilities','training_families')}})
            require(meta.get('scientific_schema')==SCHEMA and meta.get('benchmark')=='G1_smooth_v1','benchmark/schema mismatch')
            require(approved_diagnostic(c) and approved_predecessor(meta['implementation_identity']),'source compatibility mismatch')
            require(meta['execution_qualification']['certification_id']==plan.certificate['certification_id'] and p['normalizers']['identity']==plan.certificate['normalizers']['identity'],'certificate/normalization mismatch')
            require(p['model_config']==verification['model_config'],'model configuration mismatch')
            require(meta['runtime']['dependencies']['torch']==torch.__version__ and c.device==select_device(plan.settings.device),'exact resume Torch/device mismatch')
            require(meta['fixed_case']['physical_case_id']==verification['physical_case'] and meta['fixed_data_identity']==verification['data_identity'],'physical case or fixed data mismatch')
            require(p['sampler'].get('observations')==verification['observation_identity'],'observation identity mismatch')
            # Reconstruct the exact prospective member with its original run tag.
            # Any remaining field is compared, not just a short allowlist.
            expected=member_config(plan,verification['data_identity'],label,c.run_tag)
            require(asdict(replace(c,objective_variant=label,diagnostic_source_id=expected.diagnostic_source_id))==asdict(expected),'training/query/optimizer/schedule configuration mismatch')
            n=int(meta['successful_update']);require(n==p['training_state']['successful'] and 0<=n<=c.updates,'successful update count mismatch')
            _validate_resume_state(meta,p,c,plan.certificate)
            init=(run or {}).get('initialization_identity',verification['initialization_identity'])
            require(init==verification['initialization_identity'],'initialization mismatch')
            if label not in selected or n>selected[label]['last_successful_update']:
                selected[label]={**(run or {}),'status':'completed' if n==c.updates else 'pending','checkpoint':str(path),'checkpoint_id':meta['artifact_id'],
                    'run_id':c.run_id,'config_identity':c.identity,'case_id':verification['physical_case'],'data_identity':verification['data_identity'],
                    'observation_identity':verification['observation_identity'],'initialization_identity':init,'last_successful_update':n,'reused_from':str(path),**({'objective_alias':'A_FULL_is_BASELINE'} if original_label!=label else {})}
        except (ValueError,OSError,KeyError,TypeError,RuntimeError,pickle.UnpicklingError) as error:
            raise ValueError(f'explicit wave checkpoint rejected: {path}: {error}') from error
    # Completed reuse requires an existing report for suite aggregation. A raw
    # final checkpoint is finalized evaluation-only by launch_diagnostic, no steps.
    for run in selected.values():
        if run['status']=='completed' and not run.get('report'):run['status']='pending'
    require(len(selected)==len(paths),'explicit checkpoint variant does not match the selected arm; no fallback search')
    plan.reused_runs=selected
    result={'status':'read-only verified inventory','selected':{k:{'path':v['checkpoint'],'successful':v['last_successful_update'],'origin':'restored from completed checkpoint' if v['last_successful_update']==plan.settings.updates else 'resumed'} for k,v in selected.items()},
        'missing':[v for v in plan.settings.variant_settings if v not in selected],'rejected':rejected,'roots':[{'path':str(p),'accessible':p.is_dir()} for p in roots],
        'note':'missing/incompatible arms start fresh ONLY on explicit train; no initialization from a trained sibling','optimizer_updates':0}
    atomic_json(Path(plan.root)/'learning_diagnostics'/'wave_kinematic'/'recovery.json',result)
    return result


def _validate_resume_state(meta,p,config,certificate):
    """Read-only state validation; no optimizer construction or RNG restoration."""
    import torch
    from ..geometric.models import GeometryModel,ModelConfig
    from ..smooth.certification import normalizers
    with torch.random.fork_rng(devices=[]):
        model=GeometryModel(ModelConfig(**p['model_config']),normalizers(certificate))
        model.load_state_dict(p['model'],strict=True)
    parameters=tuple(model.parameters());opt=p['optimizer']
    require(isinstance(opt,dict) and isinstance(opt.get('state'),dict) and isinstance(opt.get('param_groups'),list) and len(opt['param_groups'])==1,'invalid AdamW state/parameter groups')
    group=opt['param_groups'][0];ids=group.get('params',[])
    require(len(ids)==len(parameters) and len(set(ids))==len(ids),'optimizer/model parameter association mismatch')
    require(abs(group.get('lr',-1)-config.learning_rate)<1e-15 and abs(group.get('weight_decay',-1)-config.weight_decay)<1e-15,'optimizer LR/weight-decay mismatch')
    require(set(opt['state'])<=set(ids),'optimizer state refers to unknown parameters')
    for key,parameter in zip(ids,parameters):
        state=opt['state'].get(key,{})
        if not state:continue
        for name in ('exp_avg','exp_avg_sq'):
            require(isinstance(state.get(name),torch.Tensor) and state[name].shape==parameter.shape and bool(torch.isfinite(state[name]).all()),'optimizer moment shape/nonfinite mismatch')
    scheduler=p['scheduler'];n=p['training_state']['successful']
    require(scheduler.get('completed')==n and scheduler.get('kind')==config.scheduler and abs(scheduler.get('next_lr',-1)-group['lr'])<1e-15,'scheduler/update/LR state mismatch')
    rng=p['rng']
    require(isinstance(rng,dict) and {'python_random','numpy_global','torch_CPU','torch_CUDA','torch_MPS','registered'}<=set(rng),'incomplete RNG state')
    require(isinstance(rng['torch_CPU'],torch.Tensor) and rng['torch_CPU'].dtype==torch.uint8,'invalid CPU RNG tensor')
    require(p['sampler'].get('config')==config.identity and p['sampler'].get('cursor')==p['training_state'].get('sampler_cursor'),'sampler configuration/cursor mismatch')


def run_wave_study(plan,verification,*,run_mode='preview',notebook=True,monitor=None):
    if isinstance(plan,(tuple,list)):return run_wave_experiments(plan,verification,run_mode=run_mode,notebook=notebook,monitor=monitor)
    if monitor is None:
        from ..study_monitoring import WaveDashboard
        print('PHASE G — WAVE TRAINING | '+run_mode.upper(),flush=True)
        monitor=WaveDashboard((plan,),notebook=notebook)
        with monitor.live():return run_wave_study(plan,verification,run_mode=run_mode,notebook=notebook,monitor=monitor)
    from .suite import run_diagnostic_suite
    require(run_mode in {'preview','train','resume'},'wave run mode preview/train/resume only')
    if run_mode=='preview':
        print('Wave PREVIEW: '+' → '.join(plan.settings.variant_settings)+f' · {plan.settings.query_count:,} supervised records + {next((v.get("kinematic_queries",512) for k,v in plan.settings.variant_settings.items() if k=="KIN_001"),0)} separate KIN points · zero optimizer updates.')
        return {'status':'preview','optimizer_updates':0,'plan':plan.preview()}
    require(verification and verification.get('passed') and verification.get('plan_identity')==plan.identity and verification.get('source')==diagnostic_source(),'wave derivative verification absent/failed/stale: run Cell2 before training')
    require(run_mode!='resume' or getattr(plan,'wave_lookup','none')=='explicit','resume requires specified checkpoint/suite paths')
    recovery=(getattr(plan,'_explicit_recovery',None) or recover_wave_checkpoints(plan,verification)) if getattr(plan,'wave_lookup','none')=='explicit' else {'status':'lookup disabled','selected':{}}
    if getattr(plan,'wave_lookup','none')=='none':
        plan.reused_runs={}
        require(plan.action=='fresh' and plan.snapshot is None,'lookup disabled requires fresh initialization')
    directory=Path(plan.root)/'learning_diagnostics'/'suites'/plan.suite_id
    if plan.action=='fresh' and directory.exists():plan.suite_id=plan.suite_id+'-'+uuid4().hex
    from ..study_monitoring import VariantDashboard
    result=run_diagnostic_suite(plan,expected_identity=plan.identity,run_mode='resume' if plan.action=='resume' else 'train',monitor=monitor or VariantDashboard(suite_state={},notebook=notebook))
    result['recovery']=recovery
    return result


EXPERIMENTS=('A_GEO/BASELINE','A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/KIN_001','C_GEO/STAGED_001')

def prepare_wave_experiments(*,experiments=EXPERIMENTS,architectures=None,resume_paths=None,**settings):
    """Two existing suite plans, one shared configuration; no new saved format."""
    if isinstance(experiments,dict):
        order=list(architectures or experiments)
        require(order and len(set(order))==len(order) and set(order)<=set(experiments),'select available architectures once')
        experiments=tuple(a+'/'+v for a in order for v in experiments[a])
    elif architectures is not None:experiments=tuple(label for label in experiments if label.split('/')[0] in architectures)
    require(set(resume_paths or {})<=set(experiments)|{label.split('/')[0] for label in experiments},'explicit resume keys must be selected experiments or architectures')
    require(experiments and len(set(experiments))==len(experiments) and set(experiments)<=set(EXPERIMENTS),'select unique supported wave experiments')
    from itertools import groupby
    groups=[(architecture,list(items)) for architecture,items in groupby(experiments,key=lambda x:x.split('/')[0])]
    require(len({a for a,_ in groups})==len(groups),'keep each architecture contiguous to preserve the recorded execution order')
    selected_paths=dict(resume_paths or {})
    plans=tuple(prepare_wave_study(model=a,variants=[label.split('/')[1] for label in items],
        resume_paths={label.split('/')[1]:selected_paths[label] for label in items if label in selected_paths},suite_path=selected_paths.get(a),**settings) for a,items in groups)
    require(len({p.recipes[0]['physical_case'] for p in plans})==1,'wave physical cases differ')
    return plans


def verify_wave_experiments(plans,*,data=None,reference_count=12):
    """Shared certified case/cache; same derivative controls on each architecture."""
    from .data import prepare_data
    from .wave_verification import verify_wave
    first=plans[0]
    data=data or prepare_data(Path(first.root)/'learning_diagnostics'/'fixed_data',first.certificate,experiment=first.experiments[0],seed=first.settings.seed,
        pde='wave',family='square',count=min(first.settings.query_count,1024),conditioning_h=first.settings.advanced.get('conditioning_h',.12),
        grid_size=first.settings.advanced.get('grid_size',64),cache_reference=True,evaluation_times=(0.,.05,.1,.25,.5,.75,1.))
    reports={p.model:verify_wave(p,data=data,reference_count=reference_count) for p in plans}
    require(len({(r['physical_case'],r['data_identity'],r['observation_identity'],r['evaluation_identity']) for r in reports.values()})==1,'wave architectures do not share fixed case/sensors/evaluation')
    atomic_json(Path(first.root)/'learning_diagnostics'/'wave_kinematic'/'derivative-verification-A-C.json',reports)
    return reports


def recover_wave_experiments(plans,verification):
    return {p.model:(recover_wave_checkpoints(p,verification[p.model]) if getattr(p,'wave_lookup','none')=='explicit' else {'status':'lookup disabled','selected':{},'missing':list(p.settings.variant_settings),'optimizer_updates':0}) for p in plans}


def run_wave_experiments(plans,verification,*,run_mode='preview',notebook=True,monitor=None):
    """Same synchronous runner; immediate UI, no implicit checkpoint lookup."""
    print('PHASE G — WAVE TRAINING | '+run_mode.upper()+' | checkpoint lookup: '+('/'.join(sorted({getattr(p,'wave_lookup','none') for p in plans}))),flush=True)
    from ..study_monitoring import WaveDashboard
    monitor=monitor or WaveDashboard(plans,notebook=notebook)
    results={};current=None
    with monitor.live():
        monitor.phase('CONFIGURATION / DERIVATIVE GATE')
        try:
            require(run_mode in {'preview','train','resume'},'wave mode preview/train/resume only')
            if run_mode=='preview':
                monitor.phase('PREVIEW — zero optimizer updates')
                return {'status':'preview','architectures':{},'optimizer_updates':0}
            for p in plans:
                r=(verification or {}).get(p.model,{})
                require(r.get('passed') and r.get('plan_identity')==p.identity and r.get('source')==diagnostic_source(),'all selected architectures need matching derivative verification before any training')
                require(run_mode!='resume' or getattr(p,'wave_lookup','none')=='explicit','resume requires specified checkpoint paths for every architecture')
            from .wave_progress import observe
            for p in plans:
                if getattr(p,'wave_lookup','none')=='explicit':
                    p._explicit_recovery=observe('VALIDATING SPECIFIED CHECKPOINT / OPTIMIZER STATE',lambda:recover_wave_checkpoints(p,verification[p.model]),architecture=p.model)
            from ..smooth.training import select_device
            import torch
            device=select_device(plans[0].settings.device)
            monitor.device_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else device
            monitor.phase('DERIVATIVE GATE PASSED',derivatives='PASSED')
            for p in plans:
                current=p;monitor.architecture=p.model
                monitor.phase('EXPLICIT RESUME VALIDATION' if getattr(p,'wave_lookup','none')=='explicit' else 'FRESH INITIALIZATION — lookup disabled',architecture=p.model,variant=next(iter(p.settings.variant_settings)),derivatives='PASSED')
                results[p.model]=run_wave_study(p,verification[p.model],run_mode=run_mode,notebook=notebook,monitor=monitor)
                monitor.finish_architecture(p.model,results[p.model])
            monitor.phase('COMPLETED')
            return {'status':'completed','architectures':results,
                'session_optimizer_updates':sum(run.get('session_optimizer_updates',0) for r in results.values() for entry in r['entries'].values() for run in entry['runs'].values())}
        except BaseException as error:
            if current is not None and getattr(current,'last_result',None):results[current.model]=current.last_result
            failed_stage=monitor.failed_stage or monitor.phase_state.get('stage','unknown')
            monitor.phase('INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=f'{type(error).__name__}: {error}',failed_stage=failed_stage)
            print(f'Wave stopped at {failed_stage}; last successful optimizer update: {monitor.last_successful}. '+str(error),flush=True)
            error.wave_result={'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed','architectures':results}
            raise
