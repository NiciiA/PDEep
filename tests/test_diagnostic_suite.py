"""Suite control-flow tests use explicitly synthetic events; NO optimizer steps.

Actual models, reference agreement, losses/gradients and devices are independently
covered by continuous-query/scientific tests. Filesystem copies below are not
claims of live Google Drive verification.
"""
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
import json
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import torch
from pdeep.core import ValidationError,digest
from pdeep.artifact_io import write_artifact,read_payload,read_manifest
from pdeep.artifact_drive import PersistencePolicy,sync_verified
from pdeep.artifact_tensors import state_identity
from pdeep.train_types import TrainingState
from pdeep.smooth.training import implementation_identity
from pdeep.smooth.data import Case
from pdeep.learning_diagnostics import suite as module
from pdeep.learning_diagnostics.suite import SuiteSettings,prepare_diagnostic_suite,run_diagnostic_suite,SuiteDashboard,DEFAULT_EXPERIMENTS
from pdeep.learning_diagnostics.workflows import DiagnosticPlan
from pdeep.learning_diagnostics.engine import DiagnosticConfig,approved_diagnostic
from pdeep.learning_diagnostics.queries import coordinates
from pdeep.learning_diagnostics.suite_reports import summary_rows,assess
from pdeep.learning_diagnostics.certificates import resolve_certification

@pytest.fixture(autouse=True)
def no_updates(monkeypatch):
    def forbidden(*a,**k):pytest.fail('suite verification must never perform an optimizer step')
    for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):monkeypatch.setattr(cls,'step',forbidden)

@pytest.fixture(scope='module')
def certificate():
    r=resolve_certification(requested_capabilities={'certification_id':None})
    if not r.verified:pytest.skip('native certificate unavailable; no automatic qualification permitted')
    return r.require()


def test_defaults_validation_readonly_preview(certificate,tmp_path,monkeypatch):
    monkeypatch.setattr(module,'prepare_diagnostic',lambda **kw:pytest.fail('preview must not prepare numerical labels/models'))
    import pdeep.smooth.qualification as q
    monkeypatch.setattr(q,'ensure_compatible_certificate',lambda *a,**kw:pytest.fail('must reuse certificate'))
    s=SuiteSettings();assert (s.query_mode,s.query_count,s.query_chunk)==('dense_resampled',8192,1024)
    plan=prepare_diagnostic_suite(certificate=certificate,root=tmp_path)
    assert plan.experiments==DEFAULT_EXPERIMENTS and plan.preview()['optimizer_updates']==0
    assert [r['pde'] for r in plan.recipes]==['poisson','heat','wave']
    assert [r['family'] for r in plan.recipes]==['rectangle','square','square']
    assert len({r['seed'] for r in plan.recipes})==3
    assert run_diagnostic_suite(plan,expected_identity=plan.identity)['status']=='preview' and not list(tmp_path.iterdir())
    for names,kw in [(('poisson_square','poisson_square'),{}),(('absent',),{}),(('c_physics_ablation',),{}),
                     (('wave_single',),{'settings':{'query_count':8}}),(DEFAULT_EXPERIMENTS,{'settings':{'query_mode':'bad'}})]:
        with pytest.raises(ValidationError):prepare_diagnostic_suite(names,certificate=certificate,root=tmp_path,**kw)
    with pytest.raises(ValidationError,match='changed'):run_diagnostic_suite(plan,expected_identity='stale',run_mode='train')

@pytest.mark.parametrize('pde',['heat','wave'])
def test_v3_per_quantity_continuous_time_coverage(pde):
    case=Case('square',pde,'train',168);a=coordinates(case,8192,42,0,revision='continuous-queries-v3');b=coordinates(case,8192,42,1,revision='continuous-queries-v3')
    x,t,q,present=a
    assert present and len(t)==8192 and not np.array_equal(t,b[1]) and not np.array_equal(x,b[0])
    assert np.array_equal(t,coordinates(case,8192,42,0,revision='continuous-queries-v3')[1])
    for k in np.unique(q):
        times=t[q==k];n=len(times)
        assert n==(4096 if pde=='wave' else 8192)
        assert (times==0).sum()==n//8
        assert ((times>0)&(times<=.1)).sum()>=n//8
        assert ((times>.1)&(times<=.5)).sum()>=n//4
        assert ((times>.5)&(times<=1)).sum()>=n//4
        assert len(np.unique(times[times>0]))==n-n//8
    # v2 continues to use the previously established .125/.125/.75 mixture.
    old=coordinates(case,8192,42,0)
    assert np.array_equal(old[0],x) and np.array_equal(old[2],q) and not np.array_equal(old[1],t)
    assert (old[1]==0).sum()==1024


def _metrics(pde):
    quantities=('u','v') if pde=='wave' else ('u',)
    def record(value):return {'standardized_MSE':value,'physical_RMSE':value**.5*.1,'amplitude_ratio':.99}
    return {'training':{q:record(.001) for q in quantities},'field':{q:{**record(.002 if q=='u' else .004),
        'per_time':{str(t):record(.003) for t in (0,.05,.1,.25,.5,.75,1)} if pde!='poisson' else {}} for q in quantities},
        'sensors':{q:record(.01) for q in quantities},'auxiliary':{'f' if pde=='poisson' else 'u0':record(.02)},'boundary':{}}

@pytest.fixture
def synthetic_runtime(monkeypatch,certificate):
    calls=[];prepared=[];init=[];failure={'name':None,'once':True};launch_updates=[]
    def prepare(**kw):
        name=kw['experiment'];pde,family=module.CATALOG[name];seed=kw['seed'];case=Case(family,pde,'train',4*(seed%3000)+__import__('pdeep.smooth.definition',fromlist=['SEEN']).SEEN.index(family))
        data=SimpleNamespace(case=case,identity='data-'+case.case_id,identities={'observation':'obs-'+case.case_id,'diagnostic':'eval-'+case.case_id},cached_reference=None)
        configs=[];models=[]
        for variant in module._variants(name):
            c=DiagnosticConfig(architecture=kw['model'],capacity='verification',preset='smoke',updates=kw['updates'],device='cpu',
                pdes=(pde,),probabilities=(1.,),training_families=(family,),cases_per_geometry=1,query_count=kw['query_count'],bank_queries=kw['query_count'],
                query_revision=kw['query_revision'],query_mode=kw['query_mode'],query_time_policy=kw['query_time_policy'],query_sampler_seed=kw['query_sampler_seed'],
                query_chunk=kw['query_chunk'],query_checkpointing=kw['query_checkpointing'],amp_mode=kw['amp_mode'],root_seed=seed,
                data_identity=data.identity,experiment=name,objective_variant=variant,run_tag=kw['run_name']+'-'+variant)
            with torch.random.fork_rng():
                torch.manual_seed(seed);m=torch.nn.Linear(2,1)
            configs.append(c);models.append(m)
        prepared.append(name);init.extend((name,state_identity(m),m.weight.data_ptr()) for m in models)
        return DiagnosticPlan(tuple(configs),data,tuple(models),certificate,'fresh',tuple(None for _ in models),kw['root'],kw['persistence'],{},tuple(state_identity(m) for m in models))
    def pilot(plan,**kw):
        assert not launch_updates, 'all pilots must precede first optimizer launch' if len(calls)==0 else ''
        plan.performance={c.objective_variant:{'config_identity':c.identity,'profiles':[{'status':'fits','optimizer_steps':0}]} for c in plan.configs}
    # On resume there have already been historical launches; assert preparation
    # is before any launch IN THAT invocation, via the launch event list below.
    monkeypatch.setattr(module,'prepare_diagnostic',prepare)
    monkeypatch.setattr(module,'performance_preflight',lambda p,**kw:setattr(p,'performance',{c.objective_variant:{'config_identity':c.identity} for c in p.configs}))
    def load(path,cert,device):
        meta,p=read_payload(path);m=torch.nn.Linear(2,1);m.load_state_dict(p['model']);return m,meta
    monkeypatch.setattr('pdeep.learning_diagnostics.inspection.load_model',load)
    def launch(plan,expected_identity,run_mode,monitor,**kwargs):
        assert expected_identity==plan.identity
        name=plan.data.case.pde;c=plan.configs[0];experiment=c.experiment;calls.append((experiment,run_mode))
        assert len(prepared)>=len(DEFAULT_EXPERIMENTS) or experiment=='c_physics_ablation'
        start=0
        if run_mode=='resume':_,pp=read_payload(plan.checkpoints[0]);start=pp['training_state']['successful']
        state=TrainingState(c.architecture,c.run_id);state.phase='ready';state.successful=start
        for i in range(start+1,c.updates+1):
            state.successful=i;state.phase='ready';launch_updates.append((experiment,i))
            directory=Path(plan.root)/'synthetic-runs'/c.run_id
            meta={'artifact_type':'training_checkpoint','architecture_id':c.architecture,'run_id':c.run_id,'config':asdict(c),
                'implementation_identity':implementation_identity(),'diagnostic_source_id':c.diagnostic_source_id,'scientific_schema':module.CHECKPOINT_SCHEMA,'benchmark':'G1_smooth_v1',
                'execution_qualification':certificate,'fixed_data_identity':plan.data.identity,'fixed_case':plan.data.case.manifest,
                'runtime':{'dependencies':{'torch':torch.__version__}},'successful_update':i}
            payload={'model':plan.models[0].state_dict(),'model_config':{},'optimizer':{},'scheduler':{},'rng':{},'sampler':{},
                'normalizers':{'identity':certificate['normalizers']['identity']},'training_state':{'successful':i}}
            path=write_artifact(directory,f'update-{i}',meta,payload,required=tuple(payload))
            ss=sync_verified(path,plan.persistence);state.last_checkpoint=str(path);state.persistence={'local_status':'verified','Drive_status':ss['Drive_status'],'last_sync':ss}
            monitor.update(c,state)
            if failure['name']==experiment and failure['once']:
                failure['once']=False;raise KeyboardInterrupt('synthetic Colab disconnect after verified checkpoint')
        return {c.objective_variant:SimpleNamespace(config=c,state=state,model=plan.models[0],data=plan.data)}
    monkeypatch.setattr(module,'launch_diagnostic',launch)
    def report(e,directory):
        Path(directory).mkdir(parents=True,exist_ok=True)
        return {'model':e.config.architecture,'updates':e.state.successful,'metrics':_metrics(e.data.case.pde),'case':e.data.case.manifest,
                'assessment':{'status':'passed'},'stability':{'elapsed_seconds':1.,'peak_allocated_CUDA_bytes':None,'clipping_fraction':0.}}
    monkeypatch.setattr('pdeep.learning_diagnostics.suite_reports.collect_run',report)
    return SimpleNamespace(calls=calls,prepared=prepared,init=init,failure=failure,updates=launch_updates)


def test_order_independent_instances_checkpoint_drive_and_reports(certificate,tmp_path,synthetic_runtime):
    drive=tmp_path/'drive';drive.mkdir();policy=PersistencePolicy(True,str(drive),True,backend='filesystem_test')
    plan=prepare_diagnostic_suite(certificate=certificate,root=tmp_path/'local',capacity='verification',settings={'updates':2,'amp_mode':'off','device':'cpu'},persistence=policy)
    handles=[];handle=object()
    d=SuiteDashboard(suite_state={},interval=0,renderer=lambda text,previous:handles.append(previous) or handle)
    out=run_diagnostic_suite(plan,expected_identity=plan.identity,run_mode='train',monitor=d)
    assert out['status']=='completed' and [n for n,_ in synthetic_runtime.calls]==list(DEFAULT_EXPERIMENTS)
    assert len({h for _,h,_ in synthetic_runtime.init})==3 and len({ptr for _,_,ptr in synthetic_runtime.init})==3
    assert handles[0] is None and all(x is handle for x in handles[1:])
    assert out['suite_sync']['Drive_verified'] and out['suite_sync']['backend']=='filesystem_test'
    assert len(out['summary']['rows'])==4 and {r['quantity'] for r in out['summary']['rows']}=={'u','v'}
    assert all(r['checkpoint_verified'] for r in out['summary']['rows'])
    for name,e in out['entries'].items():
        run=e['runs']['FULL'];assert run['Drive_status']=='completed' and run['last_successful_update']==2
        assert read_manifest(run['Drive_checkpoint'])['artifact_id']==run['checkpoint_id']
    before=list(synthetic_runtime.updates)
    resumed=prepare_diagnostic_suite(certificate=certificate,root=tmp_path/'new-local',capacity='verification',settings={'updates':2,'amp_mode':'off','device':'cpu'},persistence=policy,action='resume',resume_manifest=out['suite_sync']['Drive_manifest'])
    again=run_diagnostic_suite(resumed,expected_identity=resumed.identity,run_mode='resume',monitor=SuiteDashboard(suite_state={},notebook=False))
    assert again['status']=='completed' and synthetic_runtime.updates==before


def test_interrupt_and_resume_skip_complete_restore_active(certificate,tmp_path,synthetic_runtime):
    rt=synthetic_runtime;rt.failure['name']='heat_single'
    drive=tmp_path/'drive';drive.mkdir();policy=PersistencePolicy(True,str(drive),True,backend='filesystem_test')
    settings={'updates':2,'amp_mode':'off','device':'cpu'}
    plan=prepare_diagnostic_suite(certificate=certificate,root=tmp_path/'local',capacity='verification',settings=settings,persistence=policy)
    with pytest.raises(KeyboardInterrupt):run_diagnostic_suite(plan,expected_identity=plan.identity,run_mode='train',monitor=SuiteDashboard(suite_state={},notebook=False))
    pointer=json.loads((tmp_path/'local'/'learning_diagnostics'/'suites'/plan.suite_id/'latest.json').read_text());state,_=module._snapshot(pointer['Drive_manifest'])
    assert state['entries']['poisson_rectangle']['status']=='completed' and state['active_experiment']=='heat_single'
    assert state['entries']['heat_single']['runs']['FULL']['last_successful_update']==1
    # Simulate loss of all original local Colab files, not just a changed root.
    (tmp_path/'local').rename(tmp_path/'lost-local')
    resumed=prepare_diagnostic_suite(certificate=certificate,root=tmp_path/'new-local',capacity='verification',settings=settings,persistence=policy,action='resume',resume_manifest=pointer['Drive_manifest'])
    out=run_diagnostic_suite(resumed,expected_identity=resumed.identity,run_mode='resume',monitor=SuiteDashboard(suite_state={},notebook=False))
    assert out['status']=='completed'
    assert rt.calls==[('poisson_rectangle','train'),('heat_single','train'),('heat_single','resume'),('wave_single','train')]
    assert rt.updates.count(('poisson_rectangle',1))==1 and rt.updates.count(('heat_single',1))==1
    assert ('heat_single',2) in rt.updates and ('wave_single',2) in rt.updates
    with pytest.raises(ValidationError,match='identity'):prepare_diagnostic_suite(certificate=certificate,root=tmp_path/'bad',capacity='verification',settings={**settings,'query_count':4096},action='resume',resume_manifest=out['snapshot'])


def test_continue_on_failure_has_explicit_failed_member(certificate,tmp_path,synthetic_runtime,monkeypatch):
    original=module.launch_diagnostic
    def fail(p,**kw):
        if p.configs[0].experiment=='heat_single':raise ValueError('synthetic heat failure')
        return original(p,**kw)
    monkeypatch.setattr(module,'launch_diagnostic',fail)
    plan=prepare_diagnostic_suite(certificate=certificate,root=tmp_path,capacity='verification',settings={'updates':2,'amp_mode':'off','device':'cpu','stop_on_failure':False})
    out=run_diagnostic_suite(plan,expected_identity=plan.identity,run_mode='train',monitor=SuiteDashboard(suite_state={},notebook=False))
    assert out['status']=='completed_with_failures' and out['entries']['heat_single']['status']=='failed'
    assert 'synthetic heat failure' in out['entries']['heat_single']['error']['traceback']
    assert out['entries']['wave_single']['status']=='completed'


def test_drive_failure_retains_local_checkpoint(certificate,tmp_path,synthetic_runtime):
    drive=tmp_path/'absent';policy=PersistencePolicy(True,str(drive),True,backend='filesystem_test')
    plan=prepare_diagnostic_suite(certificate=certificate,root=tmp_path/'local',capacity='verification',settings={'updates':2,'amp_mode':'off','device':'cpu'},persistence=policy)
    with pytest.warns(RuntimeWarning,match='DRIVE SYNC FAILED'):
        out=run_diagnostic_suite(plan,expected_identity=plan.identity,run_mode='train',monitor=SuiteDashboard(suite_state={},notebook=False))
    assert out['suite_sync']['Drive_status']=='failed' and read_manifest(out['snapshot'])
    assert all(Path(e['runs']['FULL']['checkpoint']).exists() for e in out['entries'].values())


def test_wave_assessment_and_summary_cannot_hide_velocity():
    before={'value':1.,'rows':[{'quantity':'u','standardized_MSE':1.},{'quantity':'v','standardized_MSE':1.}]}
    after={'value':.1,'rows':[{'quantity':'u','standardized_MSE':.001},{'quantity':'v','standardized_MSE':.4}]}
    assert assess(before,after)['status']=='partial' and not assess(before,after)['per_quantity']['v']
    assert assess(before,{'value':.01,'rows':after['rows'][:1]})['status']!='passed'


def test_previous_diagnostic_source_continuation(certificate):
    old='f262d19fdc5a49122a51c38e9433294b09ef58f4c356b0f1f8549008280f45af'
    config=DiagnosticConfig(architecture='A_GEO',device='cpu',pdes=('poisson',),probabilities=(1.,),training_families=('square',),cases_per_geometry=1,query_count=128,diagnostic_source_id=old,query_revision='continuous-queries-v2')
    assert approved_diagnostic(config)
    assert not approved_diagnostic(replace(config,diagnostic_source_id='arbitrary changed code'))


def test_paired_c_physics_suite_has_independent_completed_arms(certificate,tmp_path,synthetic_runtime):
    p=prepare_diagnostic_suite(('c_physics_ablation',),model='C_GEO',capacity='verification',certificate=certificate,root=tmp_path,
        settings={'updates':2,'amp_mode':'off','device':'cpu'})
    result=run_diagnostic_suite(p,expected_identity=p.identity,run_mode='train',monitor=SuiteDashboard(suite_state={},notebook=False))
    runs=result['entries']['c_physics_ablation']['runs']
    assert set(runs)=={'FULL','FIELD_ONLY'} and all(r['status']=='completed' for r in runs.values())
    # Matched ablation starts at identical values, distinct trainable storage.
    assert runs['FULL']['initialization_identity']==runs['FIELD_ONLY']['initialization_identity']
    assert synthetic_runtime.init[0][2]!=synthetic_runtime.init[1][2]
    assert runs['FULL']['run_id']!=runs['FIELD_ONLY']['run_id']


def test_notebook_direct_suite_launch_guard(monkeypatch):
    import nbformat
    root=Path(__file__).resolve().parents[1];nb=nbformat.read(root/'notebooks/02_train.ipynb',4)
    config=next(c.source for c in nb.cells if c.cell_type=='code' and 'DIAGNOSTIC_SUITE = ' in c.source)
    scope={'IS_COLAB':False};exec(config,scope)
    assert scope['DIAGNOSTIC_SUITE']==list(DEFAULT_EXPERIMENTS) and scope['SUITE_QUERY_COUNT']==8192
    assert scope['SUITE_RUN_MODE']=='preview'
    cell=next(c.source for c in nb.cells if c.cell_type=='code' and 'SUITE_RESULTS = None' in c.source)
    events=[];scope.update(run_diagnostic_suite=lambda *a,**kw:events.append(kw['run_mode']) or {'status':'complete'},display_suite_results=lambda r:None)
    exec(cell,scope);assert not events
    scope.update(SUITE_RUN_MODE='train',suite_settings=lambda:{'frozen':True},SUITE_SETTINGS={'frozen':True},CERTIFICATION=SimpleNamespace(require=lambda:None),SUITE_PLAN=object(),SUITE_REVIEWED_IDENTITY='reviewed')
    exec(cell,scope);assert events==['train']
    scope['SUITE_SETTINGS']={}
    with pytest.raises(RuntimeError,match='settings changed'):exec(cell,scope)
    source='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    for banned in ('input(', 'Button(', 'Thread(', 'redirect_stdout'):assert banned not in source


def test_heat_assessment_requires_early_and_initial_times():
    initial={'value':1.,'rows':[{'quantity':'u','standardized_MSE':1.,'per_time':{'0':{'standardized_MSE':1.},'1':{'standardized_MSE':1.}}}]}
    late_only={'value':.2,'rows':[{'quantity':'u','standardized_MSE':.2,'per_time':{'0':{'standardized_MSE':.9},'1':{'standardized_MSE':1e-7}}}]}
    assert assess(initial,late_only)['status']!='passed'


def test_real_initial_checkpoint_before_any_update(certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics.workflows import prepare_diagnostic,launch_diagnostic
    from pdeep.learning_diagnostics.engine import DiagnosticEngine,resume_diagnostic
    from pdeep.learning_diagnostics.dashboard import Dashboard
    p=prepare_diagnostic(certificate=certificate,root=tmp_path,capacity='verification',updates=1,device='cpu',experiment='poisson_rectangle',
        query_mode='dense_resampled',query_count=16,query_revision='continuous-queries-v3',advanced={'grid_size':4},amp_mode='off')
    class StopBeforeTraining(Exception):pass
    monkeypatch.setattr(DiagnosticEngine,'run',lambda *a,**kw:(_ for _ in ()).throw(StopBeforeTraining()))
    with pytest.raises(StopBeforeTraining):launch_diagnostic(p,expected_identity=p.identity,run_mode='train',monitor=module.Dashboard(notebook=False),save_initial_checkpoint=True)
    paths=list(tmp_path.rglob('diagnostic-update-00000000.json'));assert len(paths)==1
    meta,payload=read_payload(paths[0]);assert payload['training_state']['successful']==payload['training_state']['sampler_cursor']==0
    assert not payload['optimizer']['state']
    e=resume_diagnostic(paths[0],tmp_path/'restored',data=p.data,certificate=certificate,config=p.configs[0],monitor=Dashboard(notebook=False))
    assert e.state.successful==0 and state_identity(e.model)==p.initial_model_ids[0]
