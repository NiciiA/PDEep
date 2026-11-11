"""Bounded verification: no optimizer.step is allowed anywhere in this module."""
import copy,json
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from pdeep.core import digest,ValidationError
from pdeep.artifact_tensors import state_identity
from pdeep.learning_diagnostics.certificates import resolve_certification,copy_certificate,inspect_file,SUPPLIED_ID,ORIGINAL_SHA
from pdeep.learning_diagnostics.data import prepare_data,token_id,select,FixedSampler
from pdeep.learning_diagnostics.engine import DiagnosticConfig,DiagnosticEngine,FieldOnly,resume_diagnostic
from pdeep.learning_diagnostics.metrics import reconstruction,quantity_metrics,gradient_terms,representation
from pdeep.learning_diagnostics.sensitivity import perturbations,sensitivity
from pdeep.learning_diagnostics.dashboard import Dashboard
from pdeep.geometric.models import create
from pdeep.geometric.physics import Adapter,supervised,WeakConfig

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def forbid_updates(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Diagnostic verification permits NO optimizer step')
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    monkeypatch.setattr(torch.optim.SGD,'step',forbidden)

@pytest.fixture(scope='module')
def certificate(tmp_path_factory):
    from pdeep.smooth.qualification import ensure_compatible_certificate
    # Resolver reads only existing evidence, never calls the qualifier.
    r=resolve_certification(requested_capabilities={'certification_id':None},cache_dir=tmp_path_factory.mktemp('certs'))
    if not r.verified:pytest.skip('No existing native-runtime numerical qualification: '+str(r.mismatches))
    return r.require()

@pytest.fixture(scope='module')
def datasets(certificate,tmp_path_factory):
    root=tmp_path_factory.mktemp('fixed-data')
    return {p:prepare_data(root,certificate,experiment=e,count=8,grid_size=4,seed=42) for p,e in [('poisson','poisson_square'),('heat','heat_single'),('wave','wave_single')]}


def test_certificate_original_integrity_and_runtime_mismatch(tmp_path,monkeypatch):
    import pdeep.smooth.qualification as q
    monkeypatch.setattr(q,'ensure_compatible_certificate',lambda *a,**k:pytest.fail('unexpected expensive qualification'))
    src=ROOT/'certifications/G1_smooth_v1'/(SUPPLIED_ID+'.json');r,checksum=inspect_file(src)
    assert checksum==ORIGINAL_SHA[SUPPLIED_ID]
    copied=copy_certificate(src,tmp_path/'Drive'/'certifications'/'G1_smooth_v1')
    assert src.read_bytes()==copied.read_bytes()
    pointer=copied.parent/'pointer.json';pointer.write_text(json.dumps({'record':copied.name}))
    assert inspect_file(pointer)[1]==checksum
    assert copy_certificate(pointer,tmp_path/'pointer-copy').read_bytes()==src.read_bytes()
    resolved=resolve_certification(configured_path=copied,cache_dir=tmp_path/'cache')
    if r['runtime_identity']==q.runtime_identity():assert resolved.verified
    else:
        assert resolved.status=='INCOMPATIBLE' and 'runtime_identity' in str(resolved.mismatches)
        with pytest.raises(ValidationError):resolved.require()
    assert resolve_certification(configured_path=tmp_path/'missing.json',cache_dir=tmp_path/'none').status=='MISSING'
    altered=tmp_path/'altered.json';altered.write_text(src.read_text().replace('0.0004886613992471543','0.0004886613992471544'))
    assert resolve_certification(configured_path=altered,cache_dir=tmp_path/'bad').status=='INCOMPATIBLE'
    with pytest.raises(ValidationError,match='rough G1'):resolve_certification('G1')


def test_native_capabilities_fail_closed(certificate,tmp_path,monkeypatch):
    from pdeep.smooth.certification import save_record
    path=save_record(certificate,tmp_path)
    assert resolve_certification(requested_capabilities={'certification_id':certificate['certification_id']},configured_path=path,cache_dir=tmp_path/'cache').verified
    for caps in ({'families':['not-a-geometry']},{'pdes':['flow']},{'mesh_resolution':.04},{'normalization_id':'bad'},{'observation_policy':'bad'},{'prior':'bad'},{'solver':'bad'}):
        r=resolve_certification(requested_capabilities={'certification_id':certificate['certification_id'],**caps},configured_path=path,cache_dir=tmp_path/'cache')
        assert not r.verified,caps
    with pytest.raises(ValidationError,match='unknown requested'):resolve_certification(requested_capabilities={'misspelled_PDEs':['flow']})
    import pdeep.smooth.certification as cert
    monkeypatch.setattr(cert,'physics_identity',lambda:{'changed_source':True})
    r=resolve_certification(requested_capabilities={'certification_id':certificate['certification_id']},configured_path=path,cache_dir=tmp_path/'cache')
    assert r.status=='INCOMPATIBLE' and 'source' in str(r.mismatches)


def config_for(data,arch='A_GEO',**kw):
    return DiagnosticConfig(architecture=arch,capacity='verification',preset='smoke',updates=1,device='cpu',pdes=(data.case.pde,),probabilities=(1.,),training_families=(data.case.family,),cases_per_geometry=1,query_count=8,data_identity=data.identity,experiment='wave_single' if data.case.pde=='wave' else 'heat_single' if data.case.pde=='heat' else 'poisson_square',**kw)


def test_frozen_data_and_absent_time(datasets,certificate,tmp_path):
    for p,data in datasets.items():
        again=prepare_data(tmp_path,certificate,experiment='poisson_square',pde=p,seed=42,count=8,grid_size=4)
        assert again.identity==data.identity
        assert again.case.split=='train' and not hasattr(again.case.problem,'fields')
        assert not set(vars(again.case.problem)) & {'source','u0','v0','hidden'}
        s=FixedSampler(data,config_for(data))
        a,ai=s.batch(0);b,bi=s.batch(999)
        assert token_id(a.observations)==token_id(b.observations)
        assert token_id(a.queries)==token_id(b.queries) and data.identities['training']!=data.identities['diagnostic']
        assert bool(data.training.time_present.all())==(p!='poisson')
        if p!='poisson':assert set(data.diagnostic.physical_time.flatten().tolist())=={0.,.05,.25,.5,1.} or np.allclose(sorted(set(data.diagnostic.physical_time.flatten().tolist())),[0,.05,.25,.5,1])
        else:assert not data.diagnostic.time_present.any()
        assert data.observations.sigma.min()>0
        if p=='wave':assert (data.observations.quantity_id==0).sum()==(data.observations.quantity_id==1).sum()


@pytest.mark.parametrize('arch',['A_GEO','C_GEO'])
@pytest.mark.parametrize('pde',['poisson','heat','wave'])
def test_real_geometry_models_losses_metrics_and_derivatives(datasets,arch,pde):
    data=datasets[pde];m=create(arch,data.normalizers,'verification',seed=42);before=state_identity(m)
    metrics=reconstruction(m,data);assert set(metrics['field'])==({'u','v'} if pde=='wave' else {'u'})
    b=data.batch();cs=m.prepare_context(b.specs,b.observations);f,a=supervised(m,b,cs)
    pred=m.predict(cs,b.queries);expected=[]
    for q in (0,1) if pde=='wave' else (0,):
        take=b.queries.valid[0]&(b.queries.quantity_id[0]==q)
        expected.append(((pred[0,take]-b.queries.value[0,take])/data.normalizers.scale(pde,q)).square().mean())
    assert torch.allclose(f,sum(expected)/len(expected),rtol=2e-6,atol=2e-7)
    f.backward()
    for group in ('observation','decoder','u_head'):
        ps=[p for n,p in m.named_parameters() if n.startswith(group) and p.grad is not None]
        assert ps and all(torch.isfinite(p.grad).all() for p in ps) and sum(float(p.grad.square().sum()) for p in ps)>0
    if pde=='wave':assert m.v_head.weight.grad.abs().max()>0
    rep=representation(m,data);assert rep['query_coordinate_gradient_RMS']>0 and max(rep['retrieval_scale_sum_errors'])<4e-7
    assert state_identity(m)==before
    if pde=='wave':
        check=rep['wave_consistency']
        # Float32 central differences: small truncation relative to AD derivative
        # amplitude, plus finite-precision allowance. NOT a PDE accuracy gate.
        assert check['spatial_gradient_AD_FD_max_error'] < .002*check['spatial_gradient_AD_max'] + 1e-4
        assert check['ut_AD_FD_max_error'] < .002*data.normalizers.scale('wave','v')
    if pde=='heat':assert set(metrics['field']['u']['time_groups'])=={'t0','early','middle','late'}


def test_physics_ablation_isolation_same_architecture(datasets,monkeypatch):
    data=datasets['poisson'];a=create('C_GEO',data.normalizers,'verification',seed=42);b=create('C_GEO',data.normalizers,'verification',seed=42)
    assert state_identity(a)==state_identity(b) and a.config==b.config and a.config.physics
    assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(a.parameters(),b.parameters()))
    import pdeep.geometric.physics as physics
    monkeypatch.setattr(physics,'weak_loss',lambda *a,**k:pytest.fail('FIELD_ONLY evaluated physics'))
    r=FieldOnly('C_GEO').loss(a,data.batch());assert set(r.weighted)=={'field','auxiliary'}
    assert torch.allclose(r.total,r.unweighted['field']+.1*r.unweighted['auxiliary'])
    assert a.config.physics and state_identity(a)==state_identity(b)


def test_full_physics_gradients_are_separate_no_update(datasets):
    data=datasets['poisson'];m=create('C_GEO',data.normalizers,'verification',seed=42);before=state_identity(m)
    r=gradient_terms(m,data.batch(),Adapter('C_GEO',WeakConfig(tests=2)))
    assert set(r['raw'])=={'field','auxiliary','weak','boundary','initial'}
    assert all(np.isfinite(v) for v in r['weighted_gradient_norms'].values())
    assert r['weighted_gradient_norms']['field']>0 and r['weighted_gradient_norms']['weak']>0
    assert 'field:weak' in r['weighted_gradient_cosines'] and state_identity(m)==before
    assert all(p.grad is None for p in m.parameters())


def test_sensitivity_setup_actual_context_recomputation_and_fd(datasets):
    data=datasets['wave'];m=create('A_GEO',data.normalizers,'verification',seed=42)
    assert len(perturbations(data))==9
    r=sensitivity(m,data,query_limit=16)
    assert r['perturbations']['complete_record_permutation']['standardized_RMS_difference']<2e-5
    for k in ('one_value','one_valid_location','sigma_only','physical_time','wave_quantity'):
        assert r['perturbations'][k]['standardized_RMS_difference']>0 and r['perturbations'][k]['fresh_context']
    assert r['perturbations']['one_value']['context_identity']!=r['perturbations']['complete_record_permutation']['context_identity']
    assert abs(r['value_derivative']['AD_standardized']-r['value_derivative']['FD_standardized'])<2e-4
    assert r['value_derivative']['all_sensor_gradient_RMS']>0


def test_dashboard_one_handle_throttle_render_failure_preserves_state(monkeypatch):
    from pdeep.train_types import TrainingState
    from pdeep.smooth.training import Config
    calls=[];handle=object()
    def renderer(text,old):calls.append(old);return handle
    d=Dashboard(interval=0,renderer=renderer); c=Config(device='cpu');s=TrainingState(c.architecture,c.run_id)
    for i in range(200):s.successful=i;d.update(c,s)
    assert len(calls)==200 and calls[0] is None and all(h is handle for h in calls[1:]) and len(d.recent)==6
    throttled=Dashboard(interval=100,renderer=renderer)
    for _ in range(200):throttled.update(c,s)
    assert throttled.render_count==1
    bad=Dashboard(renderer=lambda *a:(_ for _ in ()).throw(RuntimeError('display broke')))
    with pytest.warns(RuntimeWarning,match='Dashboard'):bad.update(c,s)
    assert bad.errors and s.failed==0
    def optimizer_failure():
        try:raise ValueError('original optimizer exception')
        finally:bad.update(c,s)
    with pytest.raises(ValueError,match='original optimizer'):optimizer_failure()


def test_engine_persistence_resume_zero_updates(datasets,certificate,tmp_path):
    data=datasets['poisson'];cfg=config_for(data,run_tag='zero-update-fixture')
    m=create(cfg.architecture,data.normalizers,'verification',seed=42)
    e=DiagnosticEngine(cfg,tmp_path,data=data,certificate=certificate,model=m,monitor=Dashboard(notebook=False))
    assert isinstance(e,__import__('pdeep.smooth.training',fromlist=['Engine']).Engine)
    assert type(e.optimizer) is torch.optim.AdamW and e.state.successful==0
    path=e.save_checkpoint();r=resume_diagnostic(path,tmp_path,data=data,certificate=certificate,config=cfg,monitor=Dashboard(notebook=False))
    assert state_identity(r.model)==state_identity(e.model) and r.state.successful==0 and r.state.sampler_cursor==0
    assert r.scheduler.state_dict()==e.scheduler.state_dict()
    assert r.sampler.state_dict(0)==e.sampler.state_dict(0)
    with pytest.raises(ValidationError,match='configuration mismatch'):
        resume_diagnostic(path,tmp_path,data=data,certificate=certificate,config=replace(cfg,root_seed=43))
    with pytest.raises(ValidationError,match='run exists'):DiagnosticEngine(cfg,tmp_path,data=data,certificate=certificate)
    from pdeep.artifact_io import read_payload
    export=e.export();meta,payload=read_payload(export,kind='inference_export');assert set(payload)=={'model','model_config','normalizers'}


def test_direct_workflow_reuses_monitor_and_is_synchronous(monkeypatch):
    from pdeep.smooth import workflows as w
    calls=[];mon=Dashboard(notebook=False)
    class Stub:
        def __init__(self,c,root,**kw):self.c=c;assert kw['monitor'] is mon;calls.append(('construct',c.architecture))
        def run(self):calls.append(('start',self.c.architecture));calls.append(('finish',self.c.architecture))
    monkeypatch.setattr(w,'Engine',Stub)
    from pdeep.artifact_drive import PersistencePolicy
    cs=tuple(SimpleNamespace(architecture=a,device='cpu') for a in ('A_GEO','C_GEO'))
    p=SimpleNamespace(identity='reviewed',configs=cs,checkpoints=(None,None),action='fresh',run_root='unused',certificate={},persistence=PersistencePolicy())
    with pytest.raises(ValidationError):w.launch(p,expected_identity='changed',monitor=mon)
    assert not calls
    result=w.launch(p,expected_identity='reviewed',monitor=mon)
    assert calls==[('construct','A_GEO'),('start','A_GEO'),('finish','A_GEO'),('construct','C_GEO'),('start','C_GEO'),('finish','C_GEO')]
    assert set(result)=={'A_GEO','C_GEO'}


def test_saved_preview_and_direct_notebook_semantics(monkeypatch):
    import nbformat
    nb=nbformat.read(ROOT/'notebooks/02_train.ipynb',4);code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    assert 'DIAGNOSTIC_RUN_MODE = "preview"' in code and 'RUN_MODE = "preview"' in code
    assert 'offer_launch' not in code and 'input(' not in code and 'redirect_stdout' not in code and 'Thread(' not in code
    assert 'Run All with training mode enabled can train' in nb.cells[0].source
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'notebook','exec')
    cell=next(c for c in nb.cells if c.cell_type=='code' and 'RESULTS = launch(PLAN' in c.source)
    import pdeep.smooth.workflows as w
    calls=[];monkeypatch.setattr(w,'launch',lambda *a,**kw:calls.append('synchronous') or {'done':True})
    scope={'RUN_MODE':'preview'};exec(cell.source,scope);assert not calls
    cert=SimpleNamespace(verified=True,location='cert',require=lambda:None)
    settings={'certificate_path':'cert'}
    scope={'RUN_MODE':'train','ACTION':'fresh','training_settings':lambda:{},'SETTINGS':settings,'CERTIFICATION':cert,'PLAN':object(),'REVIEWED_IDENTITY':'id','Dashboard':lambda **kw:None}
    exec(cell.source,scope);assert calls==['synchronous'] and scope['RESULTS']=={'done':True}
    scope['SETTINGS']={'changed':True}
    with pytest.raises(RuntimeError,match='Configuration changed'):exec(cell.source,scope)
    assert calls==['synchronous']


def test_nonempty_optimizer_and_counters_resume_without_training(datasets,certificate,tmp_path):
    from pdeep.artifact_tensors import tree_identity
    data=datasets['poisson'];cfg=replace(config_for(data,run_tag='synthetic-resume-state'),updates=3)
    e=DiagnosticEngine(cfg,tmp_path,data=data,certificate=certificate,monitor=Dashboard(notebook=False))
    # Persistence fixture only: inject known nonzero Adam moments/counters;
    # never call step or label this as trained model evidence.
    p=e.model.u_head.weight
    e.optimizer.state[p]={'step':torch.tensor(1.),'exp_avg':torch.full_like(p,.125),'exp_avg_sq':torch.full_like(p,.5)}
    e.state.successful=1;e.state.attempted=1;e.state.sampler_cursor=1;e.scheduler.advance(1)
    checkpoint=e.save_checkpoint();r=resume_diagnostic(checkpoint,tmp_path,data=data,certificate=certificate,config=cfg,monitor=Dashboard(notebook=False))
    assert digest(tree_identity(e.optimizer.state_dict()))==digest(tree_identity(r.optimizer.state_dict()))
    assert r.scheduler.state_dict()==e.scheduler.state_dict() and r.state.successful==r.state.sampler_cursor==1
    assert token_id(r.sampler.batch(1)[0].queries)==token_id(e.sampler.batch(1)[0].queries)


def test_legacy_smooth_checkpoint_preserved_and_loadable(tmp_path):
    from pdeep.smooth.training import resume
    from pdeep.artifact_io import read_payload
    from pdeep.artifact_tensors import tree_identity
    paths=list((ROOT/'reports/colab_repair/bounded-verification').rglob('smooth-update-00000001.json'))
    if not paths:pytest.skip('historical smooth checkpoint is not distributed with this checkout')
    path=paths[0];meta,payload=read_payload(path,kind='training_checkpoint')
    if meta['runtime']['dependencies']['torch']!=torch.__version__:pytest.skip('historical exact resume needs its original Torch version')
    model_before=path.stat().st_mtime_ns
    e=resume(path,tmp_path,device='cpu',monitor=Dashboard(notebook=False))
    assert e.state.successful==1 and state_identity(e.model)==digest(tree_identity(payload['model']))
    assert digest(tree_identity(e.optimizer.state_dict()))==digest(tree_identity(payload['optimizer']))
    assert path.stat().st_mtime_ns==model_before


def test_query_labels_are_not_prediction_inputs(datasets):
    d=datasets['wave'];b=d.batch();m=create('C_GEO',d.normalizers,'verification',seed=42)
    c=m.prepare_context(b.specs,b.observations)
    with torch.no_grad():
        p=m.predict(c,b.queries)
        q=select(b.queries,torch.arange(b.queries.valid.shape[1]));q.value.fill_(float('nan'))
        actual=m.predict(c,q)
    assert torch.equal(p,actual)


def test_field_only_can_measure_physics_without_optimizing_it(datasets):
    from pdeep.learning_diagnostics.metrics import physical_diagnostics
    data=datasets['poisson'];m=create('C_GEO',data.normalizers,'verification',seed=42);before=state_identity(m)
    raw=physical_diagnostics(m,data,WeakConfig(tests=2))
    assert set(raw['unweighted'])=={'weak','boundary','initial'} and all(np.isfinite(v) for v in raw['unweighted'].values())
    report=FieldOnly('C_GEO').loss(m,data.batch())
    assert set(report.weighted)=={'field','auxiliary'} and state_identity(m)==before
    assert all(p.grad is None for p in m.parameters())


def test_all_six_real_medium_plans_and_preview_identity(certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics.workflows import prepare_diagnostic,launch_diagnostic,diagnostic_preview
    from pdeep.learning_diagnostics.data import EXPERIMENTS
    def forbidden(*a,**k):pytest.fail('preview constructed an optimizer')
    monkeypatch.setattr(torch.optim.AdamW,'__init__',forbidden)
    for name in EXPERIMENTS:
        arch='C_GEO' if name=='c_physics_ablation' else 'A_GEO'
        p=prepare_diagnostic(model=arch,experiment=name,device='cpu',certificate=certificate,root=tmp_path,advanced={'query_count':8,'grid_size':4})
        assert p.data.case.split=='train' and p.configs[0].experiment==name
        assert all(m.parameter_count==(6455429 if arch=='C_GEO' else 6158789) for m in p.models)
        if name=='poisson_rectangle':assert p.data.case.family=='rectangle'
        if name=='heat_single':assert p.data.case.pde=='heat'
        if name=='wave_single':assert p.data.case.pde=='wave'
        if name=='c_physics_ablation':
            assert len(p.models)==2 and p.configs[0].run_id!=p.configs[1].run_id
            assert state_identity(p.models[0])==state_identity(p.models[1]) and p.models[0].config==p.models[1].config
        review=p.identity
        assert diagnostic_preview(p)['optimizer_updates']==0
        assert launch_diagnostic(p,expected_identity=review,run_mode='preview')=={}
        with pytest.raises(ValidationError,match='plan changed'):launch_diagnostic(p,expected_identity='bad',run_mode='train')
        with torch.no_grad():next(p.models[0].parameters()).add_(.001)
        with pytest.raises(ValidationError,match='parameters changed'):launch_diagnostic(p,expected_identity=review,run_mode='train')


def test_genuine_paired_case_conditioning_controls(certificate,tmp_path):
    a=prepare_data(tmp_path,certificate,experiment='observation_sensitivity',count=8,grid_size=4,seed=42)
    b=prepare_data(tmp_path,certificate,experiment='observation_sensitivity',count=8,grid_size=4,seed=42,paired=True)
    assert a.case.case_id!=b.case.case_id and a.case.problem.identity==b.case.problem.identity
    m=create('A_GEO',a.normalizers,'verification',seed=42);before=state_identity(m)
    r=sensitivity(m,a,b)
    assert r['paired_case_ids']==[a.case.case_id,b.case.case_id]
    assert {'O1','O2','empty','Gaussian_prior_mean_zero','shuffled_values_O1','improvement_correct_vs_wrong'}<=set(r['paired'])
    assert r['paired']['O1']!=r['paired']['O2'] and state_identity(m)==before
    expected=float((a.diagnostic.value/a.normalizers.scale('poisson','u')).square().mean())
    assert abs(r['paired']['Gaussian_prior_mean_zero']['truth1_standardized_MSE']-expected)<1e-6


def test_inference_export_enforces_implementation_identity(certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics.inspection import load_model
    data=prepare_data(tmp_path/'data',certificate,count=8,grid_size=4)
    e=DiagnosticEngine(config_for(data,run_tag='inference-source-gate'),tmp_path,data=data,certificate=certificate,monitor=Dashboard(notebook=False))
    path=e.export();m,meta=load_model(path,certificate,device='cpu')
    assert state_identity(m)==state_identity(e.model) and 'implementation_identity' in meta
    monkeypatch.setattr('pdeep.smooth.compatibility.approved_predecessor',lambda identity:False)
    with pytest.raises(ValidationError,match='implementation is incompatible'):load_model(path,certificate,device='cpu')


def test_launch_progress_callback_is_wired_to_structured_events(monkeypatch):
    from pdeep.smooth import workflows as w
    from pdeep.smooth.training import Config
    from pdeep.train_types import TrainingState
    seen=[];monitors=[]
    class Stub:
        def __init__(self,c,root,**kw):self.config=c;self.monitor=kw['monitor'];monitors.append(self.monitor)
        def run(self):self.monitor.update(self.config,TrainingState(self.config.architecture,self.config.run_id))
    monkeypatch.setattr(w,'Engine',Stub)
    cs=tuple(Config(architecture=a,device='cpu') for a in ('A_GEO','C_GEO'))
    p=SimpleNamespace(identity='reviewed',configs=cs,checkpoints=(None,None),action='fresh',run_root='unused',certificate={},persistence=None)
    w.launch(p,expected_identity='reviewed',notebook=False,progress_callback=lambda v:seen.append((v['architecture'],v['successful'])))
    assert seen==[('A_GEO',0),('C_GEO',0)] and monitors[0] is monitors[1]
