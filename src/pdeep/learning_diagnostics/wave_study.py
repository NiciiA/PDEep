"""One focused wave comparison, configuring the existing synchronous suite.

Preview/recovery/evaluation never constructs an optimizer. Missing arms only
enter the existing successful-update engine after explicit train mode AND the
matching nonoptimizing derivative gate. No warm start between variants.
"""
from dataclasses import asdict,replace
from pathlib import Path
import json,pickle
from ..core import require,digest
from ..artifact_io import read_manifest,read_payload,atomic_json
from .engine import DiagnosticConfig,diagnostic_source,approved_diagnostic,SCHEMA
from .suite import SuiteSettings,prepare_diagnostic_suite,_snapshot,_checkpoint_path
from .study_controls import C_VARIANTS,STAGED

VARIANTS=('FIELD_ONLY','KIN_001','STAGED_001')


def prepare_wave_study(*,certificate,root,persistence=None,model='C_GEO',capacity='medium',device='auto',query_mode='dense_resampled',
                       query_count=16384,query_chunk=4096,query_time_policy='stratified',updates=1000,learning_rate=3e-4,grad_clip=1.,seed=42,
                       amp_mode='auto',kinematic_weight=.001,kinematic_queries=512,variants=VARIANTS,reuse=True,action='fresh',advanced=None):
    require(model=='C_GEO','this experiment keeps the existing C_GEO architecture')
    require(variants and len(set(variants))==len(variants) and set(variants)<=set(VARIANTS),'select unique wave variants in execution order')
    require(query_count in {8192,16384,32768,65536} or capacity=='verification','supported supervised totals:8192/16384/32768/65536')
    require(0<query_chunk<=query_count,'chunk must be positive and no larger than supervised Q')
    require(amp_mode in {'auto','off'},'matched derivative comparison uses common FP32, no BF16')
    require(kinematic_weight>0,'declare a positive experimental kinematic weight')
    require(not (len(variants)==1 and variants[0]=='FIELD_ONLY' and amp_mode=='auto'),'single-arm FIELD_ONLY must explicitly use amp_mode=off to preserve the matched wave FP32 policy')
    controls={
        'FIELD_ONLY':dict(C_VARIANTS['FIELD_ONLY']),
        'KIN_001':dict(physics_weights=(0.,0.,0.),kinematic_weight=kinematic_weight,kinematic_queries=kinematic_queries,kinematic_seed=seed),
        'STAGED_001':dict(STAGED),
    }
    # Legacy default1000 is byte-for-byte the original300/700 definition. Other
    # budgets use the explicitly named proportional version, never reinterpret
    # an old STAGED checkpoint saved with absolute300/700 thresholds.
    if updates!=1000:controls['STAGED_001']['physics_schedule']='staged_fractional_30_40_30_v1'
    selected={k:dict(controls[k],grad_clip_norm=grad_clip,gradient_diagnostics_every=200) for k in variants}
    settings=SuiteSettings(query_mode=query_mode,query_count=query_count,query_chunk=query_chunk,query_time_policy=query_time_policy,
        query_sampler_seed=seed,updates=updates,learning_rate=learning_rate,device=device,amp_mode=amp_mode,seed=seed,
        query_checkpointing=False,variant_settings=selected,advanced={**(advanced or {}),'pde':'wave','family':'square'})
    identifier='wave-kinematic-'+digest((asdict(settings),capacity,certificate['certification_id'],diagnostic_source()))[:20]
    local=Path(root)/'learning_diagnostics'/'suites'/identifier/'latest.json'
    snapshots=list((local.parent/'snapshots').glob('suite-state-*.json'))
    if local.is_file():
        pointer=json.loads(local.read_text())
        snapshots += [Path(p) for p in (pointer.get('manifest'),pointer.get('Drive_manifest')) if p]
    if persistence is not None and persistence.drive_enabled:
        base=Path(persistence.drive_root)/'checkpoints' if persistence.drive_root else Path(persistence.drive_mount)/'PDEep_artifacts'
        snapshots+=list((base/identifier).glob('*/suite-state-*.json'))
    require(action in {'fresh','resume'},'wave action fresh/resume only')
    # Interrupted copies can leave an invalid newest candidate. Try immutable
    # snapshots in descending revision and retain all files/evidence untouched.
    failures=[];plan=None
    def revision(path):
        try:return int(path.stem.rsplit('-',1)[1])
        except ValueError:return -1
    for resume in sorted(set(snapshots),key=revision,reverse=True):
        try:
            plan=prepare_diagnostic_suite(('c_physics_ablation',),model=model,capacity=capacity,settings=settings,certificate=certificate,root=root,
                persistence=persistence,action='resume',resume_manifest=resume,suite_name=identifier)
            break
        except (ValueError,OSError,KeyError,TypeError,pickle.UnpicklingError) as error:
            failures.append({'path':str(resume),'reason':str(error)})
    if plan is None:
        require(not snapshots and not local.exists() and action!='resume','no valid matching suite snapshot/checkpoint available; restore manifest AND payload from Drive. Existing artifacts were not overwritten. '+str(failures[-3:]))
        plan=prepare_diagnostic_suite(('c_physics_ablation',),model=model,capacity=capacity,settings=settings,certificate=certificate,root=root,
            persistence=persistence,action='fresh',suite_name=identifier)
    plan.snapshot_recovery_failures=failures
    if failures:
        import warnings
        warnings.warn('Newest wave suite artifact unavailable/incompatible; restored the latest fully verified earlier snapshot. No files removed. '+str(failures[-3:])+f' ({len(failures)} rejected candidates; full reasons on plan.snapshot_recovery_failures)',RuntimeWarning)

    plan.wave_reuse=reuse;return plan



def member_config(plan,data_identity,label,run_tag):
    """Exact suite member settings, also used by read-only recovery/probes."""
    return DiagnosticConfig(architecture='C_GEO',capacity=plan.capacity,preset='smoke' if plan.capacity=='verification' else 'bounded',
        pdes=('wave',),probabilities=(1.,),training_families=('square',),cases_per_geometry=1,updates=plan.settings.updates,device=plan.settings.device,
        root_seed=plan.settings.seed,experiment='c_physics_ablation',objective_variant=label,data_identity=data_identity,run_tag=run_tag,
        bank_queries=max(128,plan.settings.query_count),query_count=plan.settings.query_count,query_revision='continuous-queries-v3',query_mode=plan.settings.query_mode,
        query_time_policy=plan.settings.query_time_policy,query_sampler_seed=plan.settings.query_sampler_seed,query_chunk=plan.settings.query_chunk,
        query_checkpointing=plan.settings.query_checkpointing,amp_mode=plan.settings.amp_mode,resolved_amp_mode='off',memory_target_gb=plan.settings.memory_target_gb,
        learning_rate=plan.settings.learning_rate,scheduler=plan.settings.scheduler,**plan.settings.variant_settings[label],
        **{k:v for k,v in plan.settings.advanced.items() if k not in {'pde','family','grid_size'}})

def _candidate_paths(plan):
    """Read-only metadata inventory. JSON without verified adjacent PT is rejected."""
    roots=[Path(plan.root)/'learning_diagnostics']
    if plan.persistence.drive_enabled:
        roots.append(Path(plan.persistence.drive_root)/'checkpoints' if plan.persistence.drive_root else Path(plan.persistence.drive_mount)/'PDEep_artifacts')
    candidates={};failures=[]
    for root in roots:
        if not root.is_dir():continue
        for path in root.rglob('*.json'):
            if path.name.startswith('suite-state-'):
                try:
                    state,_=_snapshot(path)
                    for entry in state.get('entries',{}).values():
                        for label,run in entry.get('runs',{}).items():
                            if label in plan.settings.variant_settings and run.get('checkpoint'):
                                try:candidates[_checkpoint_path(run)]=run
                                except (ValueError,OSError) as error:failures.append({'path':str(path),'reason':str(error)})
                except (ValueError,OSError,KeyError,pickle.UnpicklingError) as error:failures.append({'path':str(path),'reason':str(error)})
            elif path.name.startswith(('diagnostic-update-','smooth-update-')):
                candidates.setdefault(str(path),None)
    return candidates,failures,roots


def recover_wave_checkpoints(plan,verification):
    """No optimizer or reference solve; strict scientific/data/config matching.

    Partial arms must also satisfy existing exact Torch/device/RNG resume guards.
    Complete arms are used through the SAME suite verification, not filename
    guessing. A valid but different budget/query/schedule is not a matched arm.
    """
    require(verification.get('plan_identity')==plan.identity and verification.get('passed'),'matching derivative/data gate required for recovery')
    if plan.action=='resume':
        runs=plan.snapshot['entries']['c_physics_ablation'].get('runs',{})
        selected={label:{'path':_checkpoint_path(run),'successful':run['last_successful_update'],
            'origin':'restored from completed checkpoint' if run['last_successful_update']>=plan.settings.updates else 'resumed'}
            for label,run in runs.items() if run.get('checkpoint')}
        return {'status':'verified saved suite selected','selected':selected,'missing':[label for label in plan.settings.variant_settings if label not in selected],
            'rejected':getattr(plan,'snapshot_recovery_failures',[]),'roots':[],'optimizer_updates':0}

    paths,rejected,roots=_candidate_paths(plan);selected={}
    if not getattr(plan,'wave_reuse',True):return {'status':'reuse explicitly disabled','selected':{},'rejected':[],'roots':[str(p) for p in roots]}
    from ..smooth.compatibility import approved_predecessor
    from ..smooth.training import select_device
    import torch
    for path,run in paths.items():
        try:
            meta=read_manifest(path,kind='training_checkpoint',architecture='C_GEO');cfg=meta.get('config',{});label=cfg.get('objective_variant')
            if label not in plan.settings.variant_settings or tuple(cfg.get('pdes',()))!=('wave',):continue
            meta,p=read_payload(path,kind='training_checkpoint',architecture='C_GEO',required=('model','model_config','normalizers','optimizer','scheduler','rng','sampler','training_state'))
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
            require(asdict(replace(c,diagnostic_source_id=expected.diagnostic_source_id))==asdict(expected),'training/query/optimizer/schedule configuration mismatch')
            n=int(meta['successful_update']);require(n==p['training_state']['successful'] and 0<n<=c.updates,'successful update count mismatch')
            init=(run or {}).get('initialization_identity',verification['initialization_identity'])
            require(init==verification['initialization_identity'],'initialization mismatch')
            if label not in selected or n>selected[label]['last_successful_update']:
                selected[label]={**(run or {}),'status':'completed' if n==c.updates else 'pending','checkpoint':str(path),'checkpoint_id':meta['artifact_id'],
                    'run_id':c.run_id,'config_identity':c.identity,'case_id':verification['physical_case'],'data_identity':verification['data_identity'],
                    'observation_identity':verification['observation_identity'],'initialization_identity':init,'last_successful_update':n,'reused_from':str(path)}
        except (ValueError,OSError,KeyError,TypeError,pickle.UnpicklingError) as error:rejected.append({'path':str(path),'reason':str(error)})
    # Completed reuse requires an existing report for suite aggregation. A raw
    # final checkpoint is finalized evaluation-only by launch_diagnostic, no steps.
    for run in selected.values():
        if run['status']=='completed' and not run.get('report'):run['status']='pending'
    plan.reused_runs=selected
    result={'status':'read-only verified inventory','selected':{k:{'path':v['checkpoint'],'successful':v['last_successful_update'],'origin':'restored from completed checkpoint' if v['last_successful_update']==plan.settings.updates else 'resumed'} for k,v in selected.items()},
        'missing':[v for v in plan.settings.variant_settings if v not in selected],'rejected':rejected,'roots':[{'path':str(p),'accessible':p.is_dir()} for p in roots],
        'note':'missing/incompatible arms start fresh ONLY on explicit train; no initialization from a trained sibling','optimizer_updates':0}
    atomic_json(Path(plan.root)/'learning_diagnostics'/'wave_kinematic'/'recovery.json',result)
    return result


def run_wave_study(plan,verification,*,run_mode='preview',notebook=True):
    from .suite import run_diagnostic_suite
    require(run_mode in {'preview','train','resume'},'wave run mode preview/train/resume only')
    if run_mode=='preview':
        print('Wave PREVIEW: '+' → '.join(plan.settings.variant_settings)+f' · {plan.settings.query_count:,} supervised records + {next((v.get("kinematic_queries",512) for k,v in plan.settings.variant_settings.items() if k=="KIN_001"),0)} separate KIN points · zero optimizer updates.')
        return {'status':'preview','optimizer_updates':0,'plan':plan.preview()}
    require(verification and verification.get('passed') and verification.get('plan_identity')==plan.identity and verification.get('source')==diagnostic_source(),'wave derivative verification absent/failed/stale: run Cell2 before training')
    require(run_mode!='resume' or plan.action=='resume','resume requires a compatible saved suite; fresh training not substituted')
    recovery=recover_wave_checkpoints(plan,verification)
    if notebook:
        for label in plan.settings.variant_settings:
            r=recovery.get('selected',{}).get(label)
            print(label+': '+(r['origin']+f" ({r['successful']}/{plan.settings.updates})" if r else 'missing compatible checkpoint; fresh matched training'))
    from ..study_monitoring import VariantDashboard
    result=run_diagnostic_suite(plan,expected_identity=plan.identity,run_mode='resume' if plan.action=='resume' else 'train',monitor=VariantDashboard(suite_state={},notebook=notebook))
    result['recovery']=recovery
    return result
