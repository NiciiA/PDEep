"""Bounded nontraining controls: no optimizer step or recertification allowed."""
from dataclasses import replace,asdict
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from pdeep.core import ValidationError,digest
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.learning_diagnostics.engine import DiagnosticConfig
from pdeep.learning_diagnostics.study_controls import C_VARIANTS,HEAT_VARIANTS,STAGED,physics_weights,configured_adapter
from pdeep.learning_diagnostics.optimization_studies import prepare_study,run_study
from pdeep.learning_diagnostics.queries import coordinates
from pdeep.learning_diagnostics.consistency import ReferenceField,derivative_values,summarize,reference_floor,evaluate_wave_checkpoints
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.learning_diagnostics.data import FixedData,token_id
from pdeep.smooth.data import Case,tokens
from pdeep.smooth.reference import Reference
from pdeep.smooth.certification import normalizers
from pdeep.geometric.models import create
from pdeep.geometric.physics import WeakConfig
from pdeep.learning_diagnostics.query_loss import QueryAdapter

@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def forbidden(*a,**kw):pytest.fail('No training or recertification permitted')
    for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):monkeypatch.setattr(cls,'step',forbidden)
    monkeypatch.setattr('pdeep.smooth.qualification.ensure_compatible_certificate',forbidden)

@pytest.fixture(scope='module')
def certificate():
    r=resolve_certification(requested_capabilities={'certification_id':None})
    if not r.verified:pytest.skip('native certificate absent; no automatic numerical qualification')
    return r.require()

@pytest.fixture(scope='module')
def unit_data(certificate):
    # Explicit smaller propagation mesh for independent implementation control;
    # production always uses the unchanged certified P4/.02 policy.
    c=Case('square','wave','train',168);reference=Reference('square',h=.12,degree=4)
    cache=FixedReference(c,reference_object=reference);norms=normalizers(certificate)
    design=coordinates(c,32,42,0,revision='continuous-queries-v3');x,t,q,p=design
    clean=cache.evaluate(x,t,q);query,_=tokens(c,len(t),norms,coordinates=design,clean=clean)
    obs,_=tokens(c,len(t),norms,coordinates=design,clean=clean,observation=True,set_id=42)
    di=coordinates(c,32,7193,0,revision='continuous-queries-v3')
    independent,_=tokens(c,len(t),norms,coordinates=di,clean=cache.evaluate(*di[:3]))
    ids={'observation':token_id(obs),'training':token_id(query),'diagnostic':token_id(independent)}
    return FixedData(c,obs,query,independent,torch.tensor(clean)[None,:,None].float(),norms,ids,'unit-reference-control',cache)


def cfg(data,**kw):
    return DiagnosticConfig(architecture='C_GEO',capacity='verification',preset='smoke',updates=2,device='cpu',pdes=('wave',),probabilities=(1.,),training_families=('square',),cases_per_geometry=1,query_count=32,bank_queries=32,query_chunk=17,data_identity=data.identity,query_revision='continuous-queries-v3',query_mode='dense_resampled',**kw)


def test_matched_study_defaults_no_labels_or_optimizers(certificate,tmp_path,monkeypatch):
    monkeypatch.setattr('pdeep.learning_diagnostics.suite.prepare_diagnostic',lambda **kw:pytest.fail('preview must not prepare real labels/model'))
    for name,expected in [('c_wave',C_VARIANTS),('a_heat',HEAT_VARIANTS)]:
        p=prepare_study(name,certificate=certificate,root=tmp_path)
        assert p.settings.query_count==16384 and p.settings.query_chunk==4096
        assert tuple(p.settings.variant_settings)==tuple(expected)
        assert len(p.recipes)==1 and p.recipes[0]['seed']==42
        assert run_study(p,notebook=False)['status']=='preview'
    p=prepare_study('c_wave',certificate=certificate,root=tmp_path,staged=True)
    assert p.settings.variant_settings['STAGED_001']['physics_schedule']=='staged_300_700'
    p=prepare_study('a_heat',certificate=certificate,root=tmp_path,best_clip=2.)
    assert p.settings.variant_settings['EARLY_TIME_BEST_CLIP']['grad_clip_norm']==2.


def test_weights_staged_counter_resume_and_field_only(unit_data):
    data=unit_data
    for label,options in C_VARIANTS.items():
        c=cfg(data,objective_variant=label,**options)
        assert physics_weights(c)==options['physics_weights']
    staged=cfg(data,objective_variant='STAGED_001',**STAGED)
    assert physics_weights(staged,0)==physics_weights(staged,299)==(0.,0.,0.)
    assert physics_weights(staged,300)==(0.,0.,0.)
    assert physics_weights(staged,500)==(.0005,)*3 and physics_weights(staged,700)==(.001,)*3
    restored=DiagnosticConfig(**json.loads(json.dumps(asdict(staged))))
    assert restored.identity==staged.identity and physics_weights(restored,625)==physics_weights(staged,625)
    a=create('C_GEO',data.normalizers,'verification',seed=42);b=create('C_GEO',data.normalizers,'verification',seed=42)
    r=configured_adapter(cfg(data,objective_variant='FIELD_ONLY',physics_weights=(0.,0.,0.))).loss(a,data.batch())
    from pdeep.learning_diagnostics.engine import FieldOnly
    expected=FieldOnly('C_GEO').loss(b,data.batch())
    r.total.backward();expected.total.backward()
    assert set(r.unweighted)=={'field','auxiliary'}
    assert torch.allclose(r.total,expected.total,rtol=3e-6,atol=2e-7)
    for x,y in zip(a.parameters(),b.parameters()):
        if x.grad is not None:assert torch.allclose(x.grad,y.grad,rtol=4e-5,atol=2e-6)


def test_explicit_physics_loss_weights_and_gradient_paths(unit_data):
    c=cfg(unit_data,objective_variant='WEAK_001',physics_weights=(.001,0.,0.))
    m=create('C_GEO',unit_data.normalizers,'verification',seed=42)
    r=configured_adapter(c).loss(m,unit_data.batch())
    assert torch.allclose(r.total,r.unweighted['field']+.1*r.unweighted['auxiliary']+.001*r.unweighted['weak'])
    assert r.weighted['initial']==r.weighted['boundary']==0
    r.total.backward();assert m.u_head.weight.grad.isfinite().all() and m.v_head.weight.grad.abs().max()>0

@pytest.mark.parametrize('count',[16384,32768,65536])
def test_total_records_and_early_time_coverage(count):
    wave=Case('square','wave','train',168);x,t,q,p=coordinates(wave,count,42,0,revision='continuous-queries-v3')
    assert len(q)==count and np.bincount(q).tolist()==[count//2,count//2]
    for k in (0,1):
        z=t[q==k];assert (z==0).sum()==count//16
        assert ((z>0)&(z<=.1)).any() and ((z>.1)&(z<=.5)).any() and (z>.5).any()
    heat=Case('square','heat','train',168)
    x,t,q,p=coordinates(heat,count,42,0,'early_stratified_v1',revision='continuous-queries-v3')
    baseline=coordinates(heat,count,42,0,revision='continuous-queries-v3')
    assert np.array_equal(x,baseline[0]) and len(t)==count
    assert (t==0).sum()==count//4
    assert ((t>0)&(t<=.05)).sum()==count//5
    assert ((t>.05)&(t<=.1)).sum()==3*count//20
    assert ((t>.1)&(t<=.5)).any() and (t>.5).any()
    assert not np.array_equal(t,coordinates(heat,count,42,1,'early_stratified_v1',revision='continuous-queries-v3')[1])


def test_chunked_accumulation_and_large_query_validation(unit_data):
    c=cfg(unit_data)
    for count in (16384,32768,65536):
        assert replace(c,query_count=count,bank_queries=count,query_chunk=4096).query_count==count
    for values in ({'query_count':65537,'bank_queries':65537},{'query_chunk':0},{'query_chunk':65537}):
        with pytest.raises(ValidationError):replace(c,**values)
    # Odd count/chunks and balanced field reductions; actual backwards, zero steps.
    a=create('A_GEO',unit_data.normalizers,'verification',seed=42);b=create('A_GEO',unit_data.normalizers,'verification',seed=42)
    x=QueryAdapter('A_GEO',WeakConfig(),chunk=32,checkpointing=False).loss(a,unit_data.batch());x.total.backward()
    y=QueryAdapter('A_GEO',WeakConfig(),chunk=7,checkpointing=True).loss(b,unit_data.batch());y.total.backward()
    assert torch.allclose(x.total,y.total,rtol=3e-6,atol=3e-7)
    for p,q in zip(a.parameters(),b.parameters()):
        if p.grad is not None:assert torch.allclose(p.grad,q.grad,rtol=5e-5,atol=3e-6)


def test_reference_torch_values_and_independent_derivatives(unit_data):
    cache=unit_data.cached_reference;model=ReferenceField(cache);rng=np.random.default_rng(42)
    x=cache.reference.space.mesh.sample(12,rng);t=np.r_[0,.05,.1,.25,.5,.75,1,rng.uniform(.1,.9,5)]
    values=derivative_values(model,model.context,x,t,chunk=5)
    for name,q in [('u',np.zeros(len(t),int)),('v',np.ones(len(t),int))]:
        truth=cache.evaluate(x,t,q)
        assert np.max(abs(values[name]-truth))<2e-12*max(1,np.max(abs(truth)))
    assert np.max(abs(values['kinematic']))<3e-11
    dt=2e-6;middle=np.full(len(t),.237)
    val=derivative_values(model,model.context,x,middle,chunk=7)
    fd=(cache.evaluate(x,middle+dt)-cache.evaluate(x,middle-dt))/(2*dt)
    assert np.max(abs(fd-val['ut']))<2e-7*max(1,np.max(abs(val['ut'])))
    # Independent spatial finite difference on the SAME reference element.
    h=1e-6;lap=np.zeros(len(x));centers=cache.reference.space.mesh.vertices[cache.reference.space.mesh.cells[:12]].mean(1)
    val=derivative_values(model,model.context,centers,middle,chunk=3)
    for k in (0,1):
        e=np.zeros_like(centers);e[:,k]=h
        lap+=(cache.evaluate(centers+e,middle)-2*cache.evaluate(centers,middle)+cache.evaluate(centers-e,middle))/h**2
    assert np.max(abs(lap-val['laplacian']))<2e-3*max(1,np.max(abs(val['laplacian'])))
    floor,_=reference_floor(unit_data,cache=cache,count=4,weak=WeakConfig(tests=2))
    assert floor['weak']['weak']>=0 and np.isfinite(floor['weak']['weak'])


def test_manufactured_wave_physical_time_and_no_missing_substitution(tmp_path,certificate):
    class Exact:
        def evaluate(self,c,x,t,name):
            omega=.7*np.pi*np.sqrt(2);phi=torch.sin(np.pi*x[:,0:1])*torch.sin(np.pi*x[:,1:2])
            return phi*torch.cos(omega*t) if name=='u' else -omega*phi*torch.sin(omega*t)
    c=SimpleNamespace(equation=torch.zeros(1,dtype=torch.float64),problem=SimpleNamespace(coefficient=.49))
    rng=np.random.default_rng(7);v=derivative_values(Exact(),c,rng.random((16,2)),rng.random(16),chunk=3)
    assert np.max(abs(v['kinematic']))<1e-13 and np.max(abs(v['dynamic']))<1e-12
    r=evaluate_wave_checkpoints({'A_GEO/FULL':tmp_path/'absent.json'},certificate,tmp_path/'reports',count=4)
    assert r['optimizer_updates']==0 and not r['models'] and 'A_GEO/FULL' in r['missing']


def test_exactly_four_safe_notebook_cells():
    import nbformat
    nb=nbformat.read(Path(__file__).resolve().parents[1]/'notebooks/02_train.ipynb',4)
    cells=[c for c in nb.cells if 'optimization_diagnostics' in c.metadata.get('tags',[])]
    assert len(cells)==4 and all(c.cell_type=='code' for c in cells)
    source='\n'.join(c.source for c in cells)
    assert 'G_DIAG_RUN_MODE = "preview"' in source and 'G_DIAG_QUERY_COUNT = 16384' in source
    assert 'G_DIAG_QUERY_CHUNK_SIZE = 4096' in source
    assert 'run_study(' in cells[1].source and 'run_study(' in cells[2].source
    assert 'evaluate_wave_checkpoints(' in cells[3].source and 'run_study(' not in cells[3].source
    for banned in ('Button(', 'input(', 'Thread(', 'redirect_stdout'):assert banned not in source


def test_actual_named_variant_plans_and_query_pairing(unit_data,certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics.workflows import prepare_diagnostic
    from pdeep.artifact_tensors import state_identity
    from pdeep.learning_diagnostics.queries import QuerySampler
    monkeypatch.setattr('pdeep.learning_diagnostics.workflows.prepare_data',lambda *a,**kw:unit_data)
    plan=prepare_diagnostic(model='C_GEO',experiment='c_physics_ablation',certificate=certificate,root=tmp_path,capacity='verification',updates=2,device='cpu',
        query_mode='dense_resampled',query_count=32,query_chunk=17,query_revision='continuous-queries-v3',variant_settings=C_VARIANTS)
    assert tuple(c.objective_variant for c in plan.configs)==tuple(C_VARIANTS)
    assert len(set(plan.initial_model_ids))==1
    assert len({m.u_head.weight.data_ptr() for m in plan.models})==5
    assert len({c.run_id for c in plan.configs})==5 and len({c.data_identity for c in plan.configs})==1
    ids=[]
    for c in plan.configs:
        sampler=QuerySampler(unit_data,c,reference_cache=unit_data.cached_reference)
        b,_=sampler.batch(7);ids.append(token_id(b.queries));assert token_id(b.observations)==unit_data.identities['observation']
    assert len(set(ids))==1


def test_memory_preflight_rejects_unsafe_before_training(unit_data,certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics.workflows import prepare_diagnostic,performance_preflight
    monkeypatch.setattr('pdeep.learning_diagnostics.workflows.prepare_data',lambda *a,**kw:unit_data)
    p=prepare_diagnostic(model='C_GEO',experiment='c_physics_ablation',certificate=certificate,root=tmp_path,capacity='verification',updates=2,device='cpu',
        query_mode='dense_resampled',query_count=32,query_chunk=17,query_revision='continuous-queries-v3',variant_settings={'FIELD_ONLY':C_VARIANTS['FIELD_ONLY']})
    monkeypatch.setattr('pdeep.learning_diagnostics.performance.profile',lambda *a,**kw:{'status':'over_budget','estimated_peak_with_optimizer_bytes':12e9,'memory_target_bytes':11e9})
    with pytest.raises(ValidationError,match='exceeds measured memory allowance'):performance_preflight(p)

@pytest.mark.parametrize('count',[16384,32768,65536])
def test_actual_bounded_large_query_forward_backward(unit_data,count):
    from pdeep.geometric.data import Batch
    # Verification-size network, full count and same chunked adapter. Does NOT
    # claim that medium CUDA64k fits; each real requested model is preflighted.
    c=unit_data.case;design=coordinates(c,count,42,9,revision='continuous-queries-v3')
    query,_=tokens(c,count,unit_data.normalizers,coordinates=design,clean=unit_data.cached_reference.evaluate(*design[:3]))
    batch=Batch((c,),unit_data.observations,query,unit_data.normalizers)
    m=create('A_GEO',unit_data.normalizers,'verification',seed=42)
    adapter=QueryAdapter('A_GEO',WeakConfig(),chunk=4096,checkpointing=True)
    r=adapter.loss(m,batch);r.total.backward()
    assert r.counts['valid_queries']==count and m.u_head.weight.grad.isfinite().all() and m.v_head.weight.grad.abs().max()>0
    assert [q['records'] for q in adapter.last['per_quantity_supervision'][0].values()]==[count//2,count//2]

# Shared synthetic-event fixture, extended for named controls; original suite
# checkpoint/Drive tests exercise the identical transaction with two/default arms.
from test_diagnostic_suite import synthetic_runtime


def test_weight_study_progress_resume_and_optional_reuse(certificate,tmp_path,synthetic_runtime,monkeypatch,unit_data):
    from pdeep.learning_diagnostics import suite
    from pdeep.learning_diagnostics.suite import SuiteDashboard
    monkeypatch.setattr('pdeep.learning_diagnostics.data.prepare_data',lambda *a,**kw:unit_data)
    monkeypatch.setattr('pdeep.learning_diagnostics.consistency.reference_floor',lambda *a,**kw:({'weak':{'weak':0.,'boundary':0.,'initial':0.},'scope':'synthetic orchestration control only'},None))
    # Avoid IPython rendering while retaining the real suite and report/artifact path.
    monkeypatch.setattr(suite,'SuiteDashboard',lambda **kw:SuiteDashboard(notebook=False,**kw))
    p=prepare_study('c_wave',certificate=certificate,root=tmp_path,capacity='verification',device='cpu',amp_mode='off',updates=2)
    r=run_study(p,run_mode='train',notebook=False)
    runs=r['entries']['c_physics_ablation']['runs']
    assert set(runs)==set(C_VARIANTS) and all(v['status']=='completed' for v in runs.values())
    before=list(synthetic_runtime.updates)
    # Adding only the explicit staged arm must not repeat the five completed controls.
    p2=prepare_study('c_wave',certificate=certificate,root=tmp_path,capacity='verification',device='cpu',amp_mode='off',updates=2,staged=True)
    assert set(p2.reused_runs)==set(C_VARIANTS)
    r2=run_study(p2,run_mode='train',notebook=False)
    assert len(synthetic_runtime.updates)==len(before)+2
    assert r2['entries']['c_physics_ablation']['runs']['STAGED_001']['status']=='completed'
    p3=prepare_study('c_wave',certificate=certificate,root=tmp_path,capacity='verification',device='cpu',amp_mode='off',updates=2,staged=True)
    assert p3.action=='resume'
    run_study(p3,run_mode='train',notebook=False)
    assert len(synthetic_runtime.updates)==len(before)+2


def test_resume_mode_does_not_start_fresh(certificate,tmp_path):
    p=prepare_study('a_heat',certificate=certificate,root=tmp_path,capacity='verification',device='cpu',amp_mode='off',updates=2)
    with pytest.raises(ValidationError,match='no fresh training substituted'):run_study(p,run_mode='resume',notebook=False)


def test_staged_precision_and_memory_pilot_use_eventual_physics(unit_data,certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics.workflows import prepare_diagnostic,performance_preflight
    monkeypatch.setattr('pdeep.learning_diagnostics.workflows.prepare_data',lambda *a,**kw:unit_data)
    p=prepare_diagnostic(model='C_GEO',experiment='c_physics_ablation',certificate=certificate,root=tmp_path,capacity='verification',updates=2,device='cpu',
        query_mode='dense_resampled',query_count=32,query_chunk=17,query_revision='continuous-queries-v3',variant_settings={'STAGED_001':STAGED})
    called=[]
    def precision(model,batch,adapter,requested):
        assert not adapter.field_only and adapter.weak.weak_weight==.001
        called.append('precision');return {'resolved':'off'}
    def profile(model,batch,adapter,**kw):
        assert not adapter.field_only and adapter.weak.weak_weight==.001
        called.append('memory');return {'status':'fits','optimizer_steps':0}
    monkeypatch.setattr('pdeep.learning_diagnostics.performance.resolve_amp',precision)
    monkeypatch.setattr('pdeep.learning_diagnostics.performance.profile',profile)
    performance_preflight(p);assert called==['precision','memory']
