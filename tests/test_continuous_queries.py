"""Bounded point-sampling/cache/device/gradient tests; never train experiment arms."""
from dataclasses import replace,asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from pdeep.core import ValidationError,digest
from pdeep.artifact_tensors import state_identity
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.learning_diagnostics.data import prepare_data,token_id
from pdeep.learning_diagnostics.queries import coordinates,QuerySampler
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.learning_diagnostics.engine import DiagnosticConfig,DiagnosticEngine,resume_diagnostic
from pdeep.learning_diagnostics.query_loss import QueryAdapter
from pdeep.learning_diagnostics.runtime import InputCache,using
from pdeep.learning_diagnostics.performance import asynchronous_update,profile,device_audit,resolve_amp
from pdeep.geometric.models import create
from pdeep.geometric.physics import Adapter,WeakConfig
from pdeep.learning_diagnostics.engine import FieldOnly
from pdeep.smooth.data import Case

@pytest.fixture(autouse=True)
def no_updates(monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW,'step',lambda *a,**k:pytest.fail('NO training steps authorized in unit verification'))

@pytest.fixture(scope='module')
def certificate():
    r=resolve_certification(requested_capabilities={'certification_id':None})
    if not r.verified:pytest.skip('existing native qualification unavailable')
    return r.require()

@pytest.fixture(scope='module')
def data(certificate,tmp_path_factory):return prepare_data(tmp_path_factory.mktemp('query-data'),certificate,count=17,grid_size=4)

def config(d,mode='resampled',**kwargs):
    return DiagnosticConfig(architecture='A_GEO',capacity='verification',preset='smoke',updates=1,device='cpu',pdes=(d.case.pde,),probabilities=(1.,),training_families=(d.case.family,),cases_per_geometry=1,query_count=17,data_identity=d.identity,query_revision='continuous-queries-v2',query_mode=mode,**kwargs)


def test_area_uniform_holes_and_stratified_times():
    from pdeep.geometric.domain import make_shape,mesh_shape
    mesh=mesh_shape(make_shape('hole'),.2);x=mesh.sample(24000,np.random.default_rng(42));idx,_=mesh.locate(x)
    counts=np.bincount(idx,minlength=len(mesh.cells));expected=len(x)*mesh.cell_areas/mesh.cell_areas.sum()
    # Statistical aggregate, not a fragile per-cell exact count threshold.
    chi=float(np.sum((counts-expected)**2/expected));df=len(expected)-1
    assert chi<df+7*np.sqrt(2*df)
    assert mesh.features(x)['inside'].all()
    c=Case('square','heat','train',168)
    xx,t,q,present=coordinates(c,1024,42,0)
    assert present and (t==0).sum()==128 and ((t>0)&(t<.1)).sum()>=128 and ((t>=0)&(t<=1)).all()
    # Uniform [0,1] stratum can ALSO generate early values: stratified mixture,
    # not an artificial removal of (0,.1) from its75% component.
    assert len(np.unique(xx,axis=0))==1024


def test_sampler_frozen_inputs_changed_queries_and_reproducible_resume(data):
    obs=token_id(data.observations);ev=token_id(data.diagnostic)
    fixed=QuerySampler(data,config(data,'fixed'));a,_=fixed.batch(0);b,_=fixed.batch(1)
    assert token_id(a.queries)==token_id(b.queries)==token_id(data.training)
    for mode in ('resampled','dense_resampled'):
        sampler=QuerySampler(data,config(data,mode));a,ai=sampler.batch(0);b,bi=sampler.batch(1);again,_=sampler.batch(0)
        assert token_id(a.queries)!=token_id(b.queries) and token_id(a.queries)==token_id(again.queries)
        assert sampler.reference.solves==1 and ai['physical_case']==bi['physical_case']
        assert token_id(a.observations)==token_id(b.observations)==obs
        assert token_id(data.diagnostic)==ev
        restored=QuerySampler(data,sampler.config,reference_cache=sampler.reference)
        assert restored.state_dict(2)==sampler.state_dict(2)
        assert token_id(restored.batch(2)[0].queries)==token_id(sampler.batch(2)[0].queries)
    original_value=data.observations.value.clone();data.observations.value.add_(1)
    try:
        with pytest.raises(ValidationError,match='frozen'):sampler.batch(3)
    finally:data.observations.value.copy_(original_value) # versions changed: fixture not used frozen assertions later


@pytest.mark.parametrize('pde',['poisson','heat','wave'])
def test_cached_full_operator_against_existing_reference(pde):
    from pdeep.smooth.reference import Reference
    # Small deterministic operator control; production always uses certified .02/P4.
    c=Case('square',pde,'train',168);r=Reference('square',h=.12,degree=2)
    cache=FixedReference(c,reference_object=r);rng=np.random.default_rng(9)
    x=r.space.mesh.sample(32,rng);t=np.r_[0,.00071,.02,.237,1,rng.random(27)];q=np.arange(32)%2 if pde=='wave' else np.zeros(32,dtype=int)
    f=c.fields
    if pde=='poisson':truth=r.poisson([f['f']],x)[0][0]
    else:
        u,v,_=r.dynamic(pde,[f['u0']],[f['v0']] if pde=='wave' else None,x,t)
        truth=np.where(q,v[0],u[0]) if v is not None else u[0]
    out=cache.evaluate(x,t,q);err=np.max(abs(out-truth))
    assert err<2e-11*max(1,np.max(abs(truth))),err
    before=cache.solves;cache.evaluate(x[::-1],t[::-1],q[::-1]);assert cache.solves==before
    if pde=='wave':
        tx=np.full(len(x),.237);h=1e-6
        fd=(cache.evaluate(x,tx+h)-cache.evaluate(x,tx-h))/(2*h);v=cache.evaluate(x,tx,np.ones(len(x),dtype=int))
        assert np.max(abs(fd-v))<2e-7*max(1,np.max(abs(v)))


def _grad(model):return {k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}

@pytest.mark.parametrize('arch',['A_GEO','C_GEO'])
@pytest.mark.parametrize('checkpointing',[False,True])
def test_global_chunk_loss_and_gradients_exact(arch,checkpointing,certificate,tmp_path):
    d=prepare_data(tmp_path,certificate,experiment='wave_single',count=17,grid_size=4)
    a=create(arch,d.normalizers,'verification',seed=42);b=create(arch,d.normalizers,'verification',seed=42);batch=d.batch()
    old=(FieldOnly if arch=='C_GEO' else Adapter)(arch).loss(a,batch);old.total.backward();ga=_grad(a)
    new=QueryAdapter(arch,WeakConfig(),chunk=6,checkpointing=checkpointing,field_only=arch=='C_GEO').loss(b,batch)
    new.total.backward();gb=_grad(b)
    assert torch.allclose(old.total,new.total,atol=3e-6,rtol=3e-6),(old.total,new.total)
    assert ga.keys()==gb.keys()
    maxerr=max(float((ga[k]-gb[k]).abs().max()) for k in ga)
    scale=max(float(g.abs().max()) for g in ga.values())
    assert maxerr<3e-5+3e-5*scale,(maxerr,scale)


def test_geometry_graph_cache_no_encoded_activation_reuse(certificate,tmp_path):
    d=prepare_data(tmp_path,certificate,count=8,grid_size=4);m=create('A_GEO',d.normalizers,'verification');b=d.batch();cache=InputCache()
    with torch.no_grad():
        first=m(b.specs,b.observations,b.queries)
        with using(cache):cached=m(b.specs,b.observations,b.queries)
        assert torch.equal(first,cached)
        with using(cache):again=m(b.specs,b.observations,b.queries)
        assert torch.equal(cached,again) and cache.hits>0
    assert all(not v.requires_grad and v.grad_fn is None for v in cache.arrays.values())
    before=cache.misses
    from dataclasses import replace
    with using(cache):m.prepare_context((replace(b.specs[0],coefficient=2.),),b.observations)
    assert cache.misses>before # numeric coefficient belongs to graph-cache identity


def test_shared_engine_bytecode_and_nonoptimizing_profile(certificate,tmp_path):
    from pdeep.training import TrainingEngine
    assert asynchronous_update.__code__ is TrainingEngine.update.__code__
    d=prepare_data(tmp_path,certificate,count=8,grid_size=4);m=create('A_GEO',d.normalizers,'verification');before=state_identity(m)
    adapter=QueryAdapter('A_GEO',WeakConfig(),chunk=3)
    r=profile(m,d.batch(),adapter)
    assert r['status']=='fits' and r['optimizer_steps']==0 and r['query_count']==8 and r['queries_per_second']>0
    assert r['device_audit']['model_parameters']==['cpu'] and r['device_audit']['loss']=='cpu'
    assert state_identity(m)==before
    assert resolve_amp(m,d.batch(),adapter,'auto')['resolved']=='off'


@pytest.mark.parametrize('mode',['fixed','resampled','dense_resampled'])
def test_modes_save_resume(certificate,tmp_path,mode):
    d=prepare_data(tmp_path/'data',certificate,count=17,grid_size=4);cfg=config(d,mode,run_tag='mode-'+mode)
    from pdeep.learning_diagnostics.dashboard import Dashboard
    e=DiagnosticEngine(cfg,tmp_path,data=d,certificate=certificate,monitor=Dashboard(notebook=False));batch,_=e.sampler.batch(0)
    path=e.save_checkpoint();r=resume_diagnostic(path,tmp_path,data=d,certificate=certificate,config=cfg,monitor=Dashboard(notebook=False))
    assert r.sampler.state_dict(0)==e.sampler.state_dict(0)
    assert token_id(r.sampler.batch(0)[0].queries)==token_id(batch.queries)


def test_masked_spectrum_does_not_fft_holes():
    from pdeep.learning_diagnostics.striping import directional_grid
    xx,yy=np.meshgrid(np.linspace(0,1,65),np.linspace(0,1,65),indexing='ij');x=np.column_stack((xx.ravel(),yy.ravel()));y=np.zeros(len(x));p=np.sin(2*np.pi*16*x[:,0])
    d=directional_grid(x,p,y,1,span=(1,1));assert d['spectrum']['axis_aligned_HF_error_fraction']>.9
    keep=np.linalg.norm(x-.5,axis=1)>.15;d=directional_grid(x[keep],p[keep],y[keep],1,span=(1,1));assert 'status' in d['spectrum']


def test_legacy_config_identity_is_exact(monkeypatch):
    import json
    p=Path.home()/'Downloads/diagnostic-update-00001000.json'
    if not p.exists():pytest.skip('user checkpoint manifest not distributed')
    m=json.loads(p.read_text());cfg={**m['config'],**{k:tuple(m['config'][k]) for k in ('pdes','probabilities','training_families')}}
    # Serialization-only check of a CUDA manifest, not a fake CUDA execution.
    monkeypatch.setattr('pdeep.train_types.resolve_device',lambda *a,**k:None)
    c=DiagnosticConfig(**cfg);assert c.identity==digest(m['config']) and c.run_id==m['run_id']


def test_independent_coordinates_excluded_even_if_float32_coincides(certificate,tmp_path,monkeypatch):
    import pdeep.learning_diagnostics.queries as module
    d=prepare_data(tmp_path,certificate,count=17,grid_size=4);original=module.coordinates
    def collision(*args,**kwargs):
        x,t,q,p=original(*args,**kwargs);x[0]=d.diagnostic.position[0,0].numpy();t[0]=0;q[0]=0;return x,t,q,p
    monkeypatch.setattr(module,'coordinates',collision)
    sampler=QuerySampler(d,config(d));b,_=sampler.batch(0)
    rows=np.column_stack((b.queries.position[0].numpy(),b.queries.physical_time[0].numpy(),b.queries.quantity_id[0].numpy())).astype(np.float32)
    assert not any(row.tobytes() in sampler.excluded for row in rows)


def test_cached_query_weights_preserve_coordinate_derivatives(certificate,tmp_path):
    d=prepare_data(tmp_path,certificate,count=8,grid_size=4);m=create('C_GEO',d.normalizers,'verification');b=d.batch();cache=InputCache()
    a=b.queries.position[0,:3].clone().requires_grad_(True);c=m.prepare_context(b.specs,b.observations)[0]
    p=m.evaluate(c,a,None);da=torch.autograd.grad(p.sum(),a)[0]
    x=a.detach().clone().requires_grad_(True)
    with using(cache):
        c=m.prepare_context(b.specs,b.observations)[0];q=m.evaluate(c,x,None);dx=torch.autograd.grad(q.sum(),x)[0]
    assert torch.equal(p,q) and torch.allclose(da,dx,atol=1e-8,rtol=1e-6)
    assert cache.query_geometry;cache.clear_queries();assert not cache.query_geometry


def test_C_full_weak_loss_preserved(certificate,tmp_path):
    d=prepare_data(tmp_path,certificate,count=9,grid_size=4);a=create('C_GEO',d.normalizers,'verification',seed=42);b=create('C_GEO',d.normalizers,'verification',seed=42);w=WeakConfig(tests=2)
    old=Adapter('C_GEO',w).loss(a,d.batch());old.total.backward();ga=_grad(a)
    adapter=QueryAdapter('C_GEO',w,chunk=4,checkpointing=True);new=adapter.loss(b,d.batch());new.total.backward();gb=_grad(b)
    for k in old.unweighted:assert torch.allclose(old.unweighted[k],new.unweighted[k],atol=3e-6,rtol=3e-6)
    scale=max(float(v.abs().max()) for v in ga.values());assert max(float((ga[k]-gb[k]).abs().max()) for k in ga)<1e-4+5e-5*scale


def test_comparison_equal_query_and_update_budgets(certificate,tmp_path):
    from pdeep.learning_diagnostics.comparison import prepare_comparison,launch_comparison
    from pdeep.artifact_io import read_payload
    # Real medium architecture, single common persisted initial state; preview only.
    for budget in ('equal_updates','equal_queries'):
        plan=prepare_comparison(model='A_GEO',experiment='poisson_square',updates=1000,device='cpu',certificate=certificate,root=tmp_path/budget,budget=budget,advanced={'grid_size':4})
        assert len(plan.arms)==4
        if budget=='equal_updates':assert {c.updates for c in plan.arms.values()}=={1000}
        else:assert {c.updates*c.query_count for c in plan.arms.values()}=={1024000}
        rows=plan.preview();assert len({r['initial_model'] for r in rows})==len({r['observation'] for r in rows})==len({r['evaluation'] for r in rows})==1
        meta,payload=read_payload(plan.initial_manifest,kind='diagnostic_bundle');assert meta['initial_model_state']==state_identity(plan.base.models[0])
        assert launch_comparison(plan,expected_identity=plan.identity,run_mode='preview')=={}
        with pytest.raises(ValidationError):launch_comparison(plan,expected_identity='bad',run_mode='train')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA hardware unavailable; no mocked utilization/memory claims')
def test_real_CUDA_device_peak_and_AMP_pilot(certificate,tmp_path):
    d=prepare_data(tmp_path,certificate,count=32,grid_size=4);m=create('A_GEO',d.normalizers,'verification',device='cuda')
    adapter=QueryAdapter('A_GEO',WeakConfig(),chunk=16);precision=resolve_amp(m,d.batch('cuda'),adapter,'auto');adapter.amp=precision['resolved']
    r=profile(m,d.batch('cuda'),adapter)
    assert r['memory']['peak_allocated_bytes']>=r['memory']['allocated_bytes']>0 and r['memory']['total_bytes']>0
    assert r['device_audit']['loss'].startswith('cuda') and r['phase_CUDA_event_seconds']['backward']>0
