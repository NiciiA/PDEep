"""Synchronous, local-first diagnostic orchestration; never a second training loop.

A preview validates recipes/certification without models, labels or optimizers.
Launch prepares and pilots ALL members before the first optimizer update. Only
one experiment's trainable model/optimizer is resident during execution. Suite
snapshots are verified artifacts, copied by the same mounted-Drive transaction as
checkpoints. Select a snapshot JSON (and adjacent payload) after a disconnect.
"""
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import time
import traceback
import warnings
import torch
from ..core import digest, require
from ..artifact_io import atomic_json, read_manifest, read_payload, write_artifact
from ..artifact_drive import PersistencePolicy, sync_verified
from ..artifact_tensors import state_identity
from ..smooth.certification import authorize
from ..smooth.data import Case
from ..smooth.definition import SEEN
from ..smooth.training import select_device
from .engine import DiagnosticConfig, diagnostic_source, approved_diagnostic, SCHEMA as CHECKPOINT_SCHEMA
from .workflows import prepare_diagnostic, performance_preflight, launch_diagnostic
from .dashboard import Dashboard

DEFAULT_EXPERIMENTS=('poisson_rectangle','heat_single','wave_single')
CATALOG={'poisson_square':('poisson','square'), 'poisson_rectangle':('poisson','rectangle'),
         'heat_single':('heat','square'), 'wave_single':('wave','square'),
         'c_physics_ablation':('poisson','square')}
SCHEMA='G1-smooth-diagnostic-suite-v1'

@dataclass(frozen=True)
class SuiteSettings:
    query_mode: str='dense_resampled'
    query_count: int=8192
    query_time_policy: str='stratified'
    query_sampler_seed: int=42
    updates: int=1000
    learning_rate: float=3e-4
    scheduler: str='constant'
    device: str='auto'
    amp_mode: str='auto'
    seed: int=42
    query_chunk: int=1024
    query_checkpointing: bool=False
    stop_on_failure: bool=True
    memory_target_gb: float=11.
    advanced: dict=field(default_factory=dict)
    variant_settings: dict=field(default_factory=dict)

@dataclass
class SuitePlan:
    experiments: tuple
    model: str
    capacity: str
    settings: SuiteSettings
    certificate: dict
    root: str
    persistence: PersistencePolicy
    action: str
    suite_id: str
    recipes: tuple
    source: str
    snapshot: dict | None=None
    @property
    def identity(self):
        # Local/Drive locations are deliberately portable. Execution/device/source,
        # initialization, query recipe and numerical identity are not.
        return digest({'schema':SCHEMA,'experiments':self.experiments,'model':self.model,'capacity':self.capacity,
                       'settings':{k:v for k,v in asdict(self.settings).items() if k!='variant_settings' or v},'certificate':self.certificate['certification_id'],
                       'recipes':self.recipes,'source':self.source})
    def preview(self):
        return {'suite_id':self.suite_id,'configuration_hash':self.identity,'action':self.action,
                'experiments':self.recipes,'settings':asdict(self.settings),
                'certificate':self.certificate['certification_id'],'qualification_executed':False,
                'initialization':'fresh deterministic independent instances per experiment; no warm starts',
                'optimizer_updates':0,'scope':'fixed TRAIN cases; independent-coordinate interpolation; new-case generalization NOT tested',
                'time_policy':('fixed original queries' if self.settings.query_mode=='fixed' else 'v3: per quantity 1/8 zero, 1/8 early, 1/4 middle, 1/4 late, 1/4 continuous uniform; rounded counts' if self.settings.query_time_policy=='stratified' else 'existing .125/.125/.75 zero/early/uniform mixture'),
                'wave_loss':'0.5 L_u + 0.5 L_v, fixed TRAIN RMS; auxiliary .1; C FULL additionally .1 weak/.1 initial/.1 boundary'}


def _variants(experiment,settings=None):
    if settings is not None and settings.variant_settings:return tuple(settings.variant_settings)
    return ('FULL','FIELD_ONLY') if experiment=='c_physics_ablation' else ('FULL',)

def _snapshot(path):
    meta,payload=read_payload(Path(path).expanduser(),kind='diagnostic_bundle',required=('suite_state','files'))
    require(meta.get('suite_schema')==SCHEMA,'select a verified diagnostic SUITE snapshot, not an individual checkpoint')
    state=payload['suite_state']
    require(state['configuration_hash']==meta['configuration_hash'] and state['suite_id']==meta['run_id'],'suite snapshot identity mismatch')
    return state,payload['files']


def prepare_diagnostic_suite(experiment_names=DEFAULT_EXPERIMENTS,*,model='A_GEO',capacity='medium',settings=None,
                             certificate,root='artifacts',persistence=None,action='fresh',resume_manifest=None,suite_name=None):
    settings=settings if isinstance(settings,SuiteSettings) else SuiteSettings(**(settings or {}))
    names=tuple(experiment_names)
    require(names and len(set(names))==len(names) and set(names)<=set(CATALOG),'suite requires unique supported diagnostic names in execution order')
    require(model in {'A_GEO','C_GEO'} and ('c_physics_ablation' not in names or model=='C_GEO'),'C physics ablation requires C_GEO')
    require(action in {'fresh','resume'},'suite action must be fresh/resume')
    require(settings.query_mode in {'fixed','resampled','dense_resampled'},'invalid suite query mode')
    require(settings.query_time_policy in {'stratified','mixture'},'invalid suite time policy')
    require(type(settings.stop_on_failure) is bool and type(settings.query_checkpointing) is bool,'suite flags must be boolean')
    require(type(settings.advanced.get('grid_size',64)) is int and settings.advanced.get('grid_size',64)>=4,'suite evaluation grid must be >=4')
    # Keep overrides explicit and bounded; no hidden replacement of fixed cases.
    allowed={'grid_size','conditioning_h','validation_every','checkpoint_every','instrumentation_every','pde','family'}
    require(set(settings.advanced)<=allowed,'unknown/unsafe suite advanced settings')
    require('pde' not in settings.advanced and 'family' not in settings.advanced or names==('c_physics_ablation',),'PDE/family overrides are only for the explicit C ablation suite')
    require(not settings.variant_settings or len(names)==1,'matched variant controls apply to one shared physical experiment')
    recipes=[]
    for i,name in enumerate(names):
        pde,family=CATALOG[name];pde=settings.advanced.get('pde',pde);family=settings.advanced.get('family',family)
        authorize(certificate,pdes=(pde,),families=(family,))
        # Stable experiment seeds govern independently initialized weights and
        # fixed physical/observation identities, all recorded separately below.
        seed=settings.seed+i
        for variant in _variants(name,settings):
            from .study_controls import validate_variant
            controls=settings.variant_settings.get(variant,{})
            if controls:validate_variant(model,variant,controls)
            options={'query_time_policy':settings.query_time_policy,**controls}
            DiagnosticConfig(architecture=model,capacity=capacity,preset='smoke' if capacity=='verification' else 'bounded',
                updates=settings.updates,device=settings.device,pdes=(pde,),probabilities=(1.,),training_families=(family,),
                cases_per_geometry=1,query_count=settings.query_count,bank_queries=max(128,settings.query_count),
                query_revision='continuous-queries-v3',query_mode=settings.query_mode,**options,
                query_sampler_seed=settings.query_sampler_seed,query_chunk=settings.query_chunk,query_checkpointing=settings.query_checkpointing,
                amp_mode=settings.amp_mode,memory_target_gb=settings.memory_target_gb,learning_rate=settings.learning_rate,
                scheduler=settings.scheduler,experiment=name,objective_variant=variant,
                **{k:v for k,v in settings.advanced.items() if k not in {'grid_size','pde','family'}})
        case=Case(family,pde,'train',4*(seed%3000)+SEEN.index(family),settings.advanced.get('conditioning_h',.12))
        recipes.append({'experiment':name,'pde':pde,'family':family,'seed':seed,'physical_case':case.case_id,'variants':_variants(name,settings)})
    saved=None;files={}
    if action=='resume':
        require(resume_manifest is not None,'select the verified suite snapshot manifest and keep its payload beside it')
        saved,files=_snapshot(resume_manifest);suite_id=saved['suite_id']
    else:
        require(resume_manifest is None,'fresh suite must not specify a resume snapshot')
        suite_id=suite_name or 'suite-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    require(Path(suite_id).name==suite_id and not suite_id.startswith('.'),'unsafe suite ID')
    plan=SuitePlan(names,model,capacity,settings,certificate,str(root),persistence or PersistencePolicy(),action,suite_id,tuple(recipes),diagnostic_source(),saved)
    if saved is not None:
        if saved['configuration_hash']!=plan.identity:
            from .suite_compatibility import approved_source
            require(approved_source(saved.get('source')),'suite identity/configuration/source/certificate mismatch; restore the saved settings and source')
            current=plan.source;plan.source=saved['source']
            require(saved['configuration_hash']==plan.identity,'suite identity/configuration/source/certificate mismatch; restore the saved settings and source')
            plan.source=current
            plan._saved_identity=saved['configuration_hash']
        require(saved['order']==list(names),'suite experiment order mismatch')
        # Verify completed checkpoints too, without constructing an optimizer.
        for entry in saved['entries'].values():
            for variant,run in entry.get('runs',{}).items():
                if run.get('checkpoint'):_verify_checkpoint(plan,run)
            require(entry.get('status')!='completed' or all(entry['runs'].get(v,{}).get('status')=='completed' for v in _variants(entry['experiment'],settings)),'incomplete member cannot be marked completed')
        plan._restored_files=files
    return plan


def _checkpoint_path(run):
    for path in (run.get('checkpoint'),run.get('Drive_checkpoint')):
        if path and Path(path).expanduser().is_file():
            record=read_manifest(Path(path).expanduser(),kind='training_checkpoint')
            require(record['artifact_id']==run['checkpoint_id'],'suite checkpoint identity mismatch')
            return str(Path(path).expanduser())
    raise ValueError('suite checkpoint unavailable locally/Drive; mount Drive and restore the verified manifest AND payload')


def _verify_checkpoint(plan,run):
    path=_checkpoint_path(run);meta,p=read_payload(path,kind='training_checkpoint',architecture=plan.model,
        required=('model','model_config','optimizer','scheduler','rng','sampler','normalizers','training_state'))
    c=DiagnosticConfig(**{**meta['config'],**{k:tuple(meta['config'][k]) for k in ('pdes','probabilities','training_families')}})
    require(meta.get('scientific_schema')==CHECKPOINT_SCHEMA and meta.get('benchmark')=='G1_smooth_v1','suite checkpoint benchmark/schema mismatch')
    from ..smooth.compatibility import approved_predecessor
    require(approved_diagnostic(c) and approved_predecessor(meta['implementation_identity']),'suite checkpoint implementation/source incompatible')
    require(c.identity==run['config_identity'] and c.run_id==run['run_id'],'suite checkpoint configuration mismatch')
    require(meta['execution_qualification']['certification_id']==plan.certificate['certification_id'] and p['normalizers']['identity']==plan.certificate['normalizers']['identity'],'suite checkpoint certification/normalization mismatch')
    require(meta['runtime']['dependencies']['torch']==torch.__version__,'suite resume Torch mismatch')
    require(c.device==select_device(plan.settings.device),'suite resume device mismatch')
    require(meta['fixed_data_identity']==run['data_identity'] and meta['fixed_case']['physical_case_id']==run['case_id'],'suite fixed data mismatch')
    require(meta['successful_update']==p['training_state']['successful']==run['last_successful_update'],'suite checkpoint update mismatch')
    require(run.get('status')!='completed' or meta['successful_update']==c.updates,'completed suite member lacks a final checkpoint')
    return path


class SuiteDashboard(Dashboard):
    """One handle across experiments; persistence callbacks are outside UI catch."""
    def __init__(self,*,suite_state,on_event=None,**kwargs):
        super().__init__(**kwargs);self.suite_state=suite_state;self.on_event=on_event
    def update(self,config,state,step=None,validation=None):
        v=super().update(config,state,step,validation)
        if self.on_event:self.on_event(v)
        return v
    def render(self,v):
        s=self.suite_state;completed=[k for k,e in s['entries'].items() if e['status']=='completed'];active=s.get('active_experiment')
        upcoming=[k for k in s['order'] if k not in completed and k!=active]
        current=len(completed)+1 if active else len(completed)
        arms=s['entries'].get(active,{}).get('runs',{})
        active_variant=next((label for label,run in arms.items() if run.get('run_id')==v.get('run_id')),None)
        budget=s['preview']['settings']['updates'] if 'preview' in s else 0
        total=budget*sum(len(e.get('runs',{})) or len(_variants(name)) for name,e in s['entries'].items())
        done=sum(run.get('observed_update',run.get('last_successful_update',0)) for e in s['entries'].values() for run in e.get('runs',{}).values())
        self.latest['suite']={'title':'G1_smooth_v1 — Diagnostic Suite','experiment':current,'total':len(s['order']),
            'current':active,'completed':completed,'upcoming':upcoming,'snapshot':s.get('snapshot'),
            'suite_Drive':s.get('suite_sync',{}),'configuration_hash':s['configuration_hash'],
            'overall_updates':done,'overall_requested':total,'overall_progress_percent':100*done/total if total else 0,
            'current_variant':active_variant,'variants_completed':sum(run.get('status')=='completed' for run in arms.values()),'variants_total':len(arms),
            'current_experiment_ETA_seconds':v.get('ETA_seconds'),'whole_suite_ETA':'unavailable until remaining PDE costs are measured'}
        # Use exactly the same display handle; no stdout wrapping or nested display.
        view=dict(v);view['experiment']=f"{current}/{len(s['order'])} · {active} · {active_variant or 'preparing'}"
        view['performance']={**v.get('performance',{}),'suite':self.latest['suite']}
        super().render(view)


def _release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    if torch.backends.mps.is_available():torch.mps.empty_cache()


def run_diagnostic_suite(plan,*,expected_identity,run_mode='preview',progress_callback=None,monitor=None):
    require(expected_identity==plan.identity and plan.source==diagnostic_source(),'suite plan changed; rerun preview')
    require(run_mode in {'preview','train','resume'},'suite run mode preview/train/resume only')
    if run_mode=='preview':return {'status':'preview','plan':plan.preview(),'optimizer_updates':0}
    require((run_mode=='resume')==(plan.action=='resume'),'suite action/run mode mismatch')
    root=Path(plan.root)/'learning_diagnostics'/'suites'/plan.suite_id
    require(plan.action=='resume' or not root.exists(),'suite exists; explicitly resume its verified snapshot or choose a new suite name')
    s=plan.snapshot or {'schema':SCHEMA,'suite_id':plan.suite_id,'configuration_hash':plan.identity,'certificate':plan.certificate['certification_id'],
        'order':list(plan.experiments),'entries':{n:{'experiment':n,'status':'pending','runs':{}} for n in plan.experiments},
        'active_experiment':None,'revision':0,'status':'preparing','preview':plan.preview(),'source':plan.source}
    if plan.action=='fresh' and getattr(plan,'reused_runs',None):
        name=plan.experiments[0]
        for variant,run in plan.reused_runs.items():
            _verify_checkpoint(plan,run)
            s['entries'][name]['runs'][variant]=run
        if all(s['entries'][name]['runs'].get(v,{}).get('status')=='completed' for v in _variants(name,plan.settings)):
            s['entries'][name]['status']='completed'
    # Session accounting is independent of immutable snapshot revisions.
    for entry in s['entries'].values():
        for run in entry.get('runs',{}).values():
            n=run.get('last_successful_update',0)
            run.update(session_start_successful=n,session_optimizer_updates=0,result_origin='restored from completed checkpoint' if n>=plan.settings.updates else 'resumed' if n else 'newly trained')
    root.mkdir(parents=True,exist_ok=True)
    for relative,content in getattr(plan,'_restored_files',{}).items():
        dest=root/relative;require(dest.resolve().is_relative_to(root.resolve()),'unsafe suite report path')
        if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(bytes.fromhex(content))
    def persist():
        s['revision']+=1
        files={str(p.relative_to(root)):p.read_bytes().hex() for p in root.rglob('*') if p.is_file() and p.suffix in {'.png','.json','.jsonl','.md'} and not {'snapshots','continuations'}&set(p.relative_to(root).parts) and p.name!='latest.json'}
        # Ephemeral synchronization status is published in latest.json, not nested
        # recursively inside successive immutable snapshots.
        state={k:v for k,v in s.items() if k not in {'snapshot','suite_sync'}}
        path=write_artifact(root/'snapshots',f'suite-state-{s["revision"]:08d}',
            {'artifact_type':'diagnostic_bundle','architecture_id':plan.model,'run_id':plan.suite_id,'suite_schema':SCHEMA,'configuration_hash':s['configuration_hash']},
            {'suite_state':state,'files':files},required=('suite_state','files'))
        status=sync_verified(path,plan.persistence);s['snapshot']=str(path);s['suite_sync']=status
        atomic_json(root/'latest.json',{'manifest':str(path),'Drive_manifest':status['Drive_manifest'],'sync':status})
        if status['Drive_status']=='failed':warnings.warn('SUITE DRIVE SYNC FAILED; verified local snapshot retained; Colab local storage is temporary. '+status['error'],RuntimeWarning)
        return path
    def event(v):
        name=s.get('active_experiment')
        if name:
            for run in s['entries'][name]['runs'].values():
                if run.get('run_id')==v['run_id']:
                    run['observed_update']=v['successful']
                    run['session_optimizer_updates']=max(0,v['successful']-run.get('session_start_successful',0))
                    if v.get('checkpoint') and v['checkpoint']!=run.get('checkpoint'):
                        meta=read_manifest(v['checkpoint'],kind='training_checkpoint');last=v.get('persistence',{}).get('last_sync',{})
                        # sync_verified already checked the remote payload. Do
                        # not turn a later Drive disconnect into a local-save
                        # failure by rereading that remote file in the callback.
                        fresh_sync=last.get('local_manifest')==v['checkpoint']
                        drive=last.get('Drive_manifest') if fresh_sync and last.get('Drive_verified') else run.get('Drive_checkpoint') if run.get('checkpoint_id')==meta['artifact_id'] else None
                        run.update(checkpoint=v['checkpoint'],checkpoint_id=meta['artifact_id'],last_successful_update=meta['successful_update'],
                                   Drive_checkpoint=drive,Drive_status=last.get('Drive_status','disabled'))
                        if fresh_sync:
                            hs=v.get('persistence',{}).get('history_sync',{})
                            run['history_manifest']=hs.get('Drive_manifest') or hs.get('local_manifest')
                        persist()
        if progress_callback:progress_callback(v)
    dashboard=monitor or SuiteDashboard(suite_state=s,on_event=event,progress_callback=None)
    if monitor is not None:
        require(isinstance(monitor,SuiteDashboard),'suite monitor must be SuiteDashboard for one persistent handle')
        dashboard.suite_state=s;dashboard.on_event=event
    prepared={}
    try:
        # Prepare/review every real case and model, including GPU/AMP pilots,
        # BEFORE entering any engine optimizer loop. Retain only CPU model states.
        for recipe in plan.recipes:
            name=recipe['experiment'];entry=s['entries'][name]
            if entry['status']=='completed':continue
            s['active_experiment']=name
            dashboard.render({'phase':'preparing fixed case / bounded zero-update pilot','performance':{}})
            advanced={**plan.settings.advanced,'learning_rate':plan.settings.learning_rate,'scheduler':plan.settings.scheduler}
            p=prepare_diagnostic(model=plan.model,experiment=name,capacity=plan.capacity,updates=plan.settings.updates,
                device=plan.settings.device,seed=recipe['seed'],certificate=plan.certificate,root=plan.root,persistence=plan.persistence,
                run_name=plan.suite_id+'-'+name,advanced=advanced,query_mode=plan.settings.query_mode,query_count=plan.settings.query_count,
                query_time_policy=plan.settings.query_time_policy,query_sampler_seed=plan.settings.query_sampler_seed,
                query_chunk=plan.settings.query_chunk,query_checkpointing=plan.settings.query_checkpointing,
                amp_mode=plan.settings.amp_mode,memory_target_gb=plan.settings.memory_target_gb,query_revision='continuous-queries-v3',evaluation_times=(0.,.05,.1,.25,.5,.75,1.),variant_settings=plan.settings.variant_settings or None)
            require(p.data.case.case_id==recipe['physical_case'],'suite physical recipe differs from prepared case')
            # Check all variants' exact requested settings. Existing completed
            # ablation arms are not retrained when the other arm was interrupted.
            fresh_configs=list(p.configs);fresh_models=list(p.models)
            for i,(c,m) in enumerate(zip(p.configs,p.models)):
                old=entry['runs'].get(c.objective_variant)
                if old and old.get('checkpoint'):
                    path=_verify_checkpoint(plan,old);meta,_=read_payload(path,required=('model',))
                    saved=DiagnosticConfig(**{**meta['config'],**{k:tuple(meta['config'][k]) for k in ('pdes','probabilities','training_families')}})
                    require(asdict(replace(saved,resolved_amp_mode=c.resolved_amp_mode,diagnostic_source_id=c.diagnostic_source_id,run_tag=c.run_tag if old.get('reused_from') else saved.run_tag))==asdict(c),'requested suite settings differ from saved member')
                    from .inspection import load_model
                    fresh_models[i],_=load_model(path,plan.certificate,device=c.device);fresh_configs[i]=saved
            p.configs=tuple(fresh_configs);p.models=tuple(fresh_models);p.initial_model_ids=tuple(state_identity(m) for m in p.models)
            # Pilots never update weights. Also resolves auto precision on CPU/MPS.
            performance_preflight(p)
            for c in p.configs:
                old=entry['runs'].get(c.objective_variant,{})
                if old.get('checkpoint'):require(c.identity==old['config_identity'],'resume numerical precision pilot differs from checkpoint')
            entry['runs'].update({c.objective_variant:{**entry['runs'].get(c.objective_variant,{}),
                'status':entry['runs'].get(c.objective_variant,{}).get('status','pending'),'run_id':c.run_id,'config_identity':c.identity,
                'initialization_identity':entry['runs'].get(c.objective_variant,{}).get('initialization_identity',mid),
                'initialization_seed':recipe['seed'],'data_identity':p.data.identity,'case_id':p.data.case.case_id,
                'observation_identity':p.data.identities['observation'],'evaluation_identity':p.data.identities['diagnostic'],
                'last_successful_update':entry['runs'].get(c.objective_variant,{}).get('last_successful_update',0),
                'session_start_successful':entry['runs'].get(c.objective_variant,{}).get('session_start_successful',0),
                'session_optimizer_updates':0,'result_origin':entry['runs'].get(c.objective_variant,{}).get('result_origin','newly trained'),
                'preflight':p.performance.get(c.objective_variant,{})} for c,mid in zip(p.configs,p.initial_model_ids)})
            # Release accelerator tensors; prepared CPU reference caches are reused.
            for m in p.models:m.cpu()
            prepared[name]=p
        s['active_experiment']=None;s['status']='running';persist()
        from .suite_reports import collect_run,combined_report
        for name in plan.experiments:
            entry=s['entries'][name]
            if entry['status']=='completed':continue
            s['active_experiment']=name;entry['status']='running';persist()
            p=prepared[name]
            try:
                for c,m in zip(p.configs,p.models):
                    variant=c.objective_variant;run=entry['runs'][variant]
                    if run['status']=='completed':continue
                    m.to(c.device);checkpoint=_verify_checkpoint(plan,run) if run.get('checkpoint') else None
                    # Restore into a new execution directory. Original local logs
                    # may contain an uncheckpointed tail; never append duplicate
                    # updates or overwrite any previous attempt's evidence.
                    execution_root=str(root/'continuations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')) if checkpoint else p.root
                    one=replace(p,configs=(c,),models=(m,),initial_model_ids=(state_identity(m),),
                                action='resume' if checkpoint else 'fresh',checkpoints=(checkpoint,),root=execution_root)
                    if checkpoint and run.get('history_manifest'):
                        _,history=read_payload(run['history_manifest'],kind='diagnostic_bundle',required=('history',))
                        destination=Path(execution_root)/'learning_diagnostics'/name/variant/'G1_smooth_v1'/'runs'/plan.model/c.run_id
                        destination.mkdir(parents=True,exist_ok=True)
                        for filename,content in history['history'].items():
                            require(Path(filename).name==filename,'unsafe history path')
                            target=destination/filename
                            if not target.exists():target.write_text(content)
                    result=launch_diagnostic(one,expected_identity=one.identity,run_mode='resume' if checkpoint else 'train',monitor=dashboard,save_initial_checkpoint=True)
                    engine=result[variant]
                    require(engine.state.successful==c.updates and engine.state.failed==0,'experiment did not finish its requested successful updates')
                    # Final checkpoint MUST verify before marking the member complete.
                    meta=read_manifest(engine.state.last_checkpoint,kind='training_checkpoint',architecture=plan.model)
                    last=engine.state.persistence.get('last_sync',{})
                    run.update(checkpoint=engine.state.last_checkpoint,checkpoint_id=meta['artifact_id'],last_successful_update=engine.state.successful,
                               Drive_checkpoint=last.get('Drive_manifest'),Drive_status=last.get('Drive_status','disabled'))
                    run['report']=collect_run(engine,root/name/variant,figures=False) if plan.settings.variant_settings else collect_run(engine,root/name/variant)
                    run['session_optimizer_updates']=engine.state.successful-run.get('session_start_successful',0)
                    run['status']='completed';persist()
                    engine.model.cpu();del engine,result,one;_release()
                entry['status']='completed';s['active_experiment']=None;persist()
            except Exception as error:
                entry['status']='failed'
                entry['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
                atomic_json(root/name/'failure.json',entry['error']);persist()
                if plan.settings.stop_on_failure:raise
                s['active_experiment']=None
            # Drop reference trajectories, prepared observations, optimizers and caches.
            del prepared[name],p;_release()
        s['status']='completed_with_failures' if any(e['status']=='failed' for e in s['entries'].values()) else 'completed'
        if plan.settings.variant_settings:
            from .suite_reports import study_report
            s['summary']=study_report(s,root)
        else:s['summary']=combined_report(s,root)
        persist()
        dashboard.render(dashboard.latest)
    except BaseException as error:
        name=s.get('active_experiment');s['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        s['error']={'experiment':name,'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        if name:s['entries'][name]['status']=s['status']
        atomic_json(root/f'failure-{s["revision"]:08d}.json',s['error']);persist()
        # Continue-on-failure needs a fresh invocation of the same verified state,
        # never a silent success or an in-process restart of the failed optimizer.
        raise
    finally:
        prepared.clear();_release()
    return s


def chunk_preflight(diagnostic_plan,chunk_sizes=(1024,2048,4096,8192)):
    """Optional isolated same-Q chunk comparison; never changes the selected plan.

    Use on a prepared single diagnostic plan. No qualification or optimizer step;
    all rows disclose cold/warm scope. A faster chunk is a suggestion, never an
    automatic configuration change or an attempt to occupy all VRAM.
    """
    from .queries import QuerySampler
    from .query_loss import QueryAdapter
    from .performance import profile,resolve_amp
    from ..geometric.physics import WeakConfig
    outcomes={}
    for c,m in zip(diagnostic_plan.configs,diagnostic_plan.models):
        sampler=QuerySampler(diagnostic_plan.data,c,reference_cache=diagnostic_plan.reference_cache)
        sampler.measure_transfer=True;batch,preparation=sampler.batch(0)
        adapter=QueryAdapter(c.architecture,WeakConfig(c.spatial_order,c.time_order,c.weak_tests,c.physics_chunk),chunk=c.query_chunk,checkpointing=c.query_checkpointing,field_only=c.objective_variant=='FIELD_ONLY')
        precision=resolve_amp(m,batch,adapter,c.amp_mode);adapter.amp=precision['resolved'];rows=[]
        for chunk in chunk_sizes:
            require(type(chunk) is int and chunk>0,'positive chunk size required')
            adapter.chunk=chunk
            r=profile(m,batch,adapter,memory_target_gb=c.memory_target_gb,optimizer_step=False)
            rows.append({**r,'query_chunk':chunk,'data_preparation':preparation})
        fits=[r for r in rows if r['status']=='fits']
        outcomes[c.objective_variant]={'profiles':rows,'precision':precision,
            'suggested_chunk':min(fits,key=lambda r:r['step_wall_seconds'])['query_chunk'] if fits else None,
            'configuration_changed':False,'optimizer_updates':0}
    return outcomes
