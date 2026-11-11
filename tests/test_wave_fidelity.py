"""No optimizer calls: independent numerical references and Run All launch safety."""
from dataclasses import replace
from pathlib import Path
import ast,json
import numpy as np
import pytest
from pdeep.core import ValidationError
from pdeep.geometric.domain import make_shape,mesh_shape
from pdeep.geometric.fem import FEM
from pdeep.wave_fidelity.reference import Space,ContinuousWave,SpectralControl,refine,lagrange,indices
from pdeep.wave_fidelity.fields import pair

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('h',[.04,.02,.01,.005])
def test_requested_area_realized(h):
    mesh=mesh_shape(make_shape('square'),h)
    assert mesh.cell_areas.max()<=h*h/2*(1+1e-10)
    assert len(mesh.vertices)>int(.7/(h*h))


@pytest.mark.parametrize('degree',[1,2,3,4])
def test_polynomial_space(degree):
    bary=np.asarray(indices(degree))/degree;p,g=lagrange(bary,degree)
    assert np.max(abs(p-np.eye(len(p))))<1e-13
    points=np.random.default_rng(1).dirichlet([1,1,1],17);v,d=lagrange(points,degree)
    assert np.max(abs(v.sum(1)-1))<1e-13 and np.max(abs(d.sum(1)))<1e-12
    delta=1e-6
    for k,dl in enumerate((np.array([-1,1,0]),np.array([-1,0,1]))):
        fd=(lagrange(points+delta*dl,degree)[0]-lagrange(points-delta*dl,degree)[0])/(2*delta)
        assert np.max(abs(fd-d[:,:,k]))<2e-8
    mesh=mesh_shape(make_shape('square'),.22);s=Space(mesh,degree)
    assert abs(s.M.sum()-1)<1e-12 and np.max(abs(s.K@np.ones(len(s.position))))<1e-12
    assert np.linalg.eigvalsh(s.Mi.toarray()).min()>0 and np.linalg.eigvalsh(s.Ki.toarray()).min()>0
    if degree==1:
        old=FEM(mesh,coefficient=.49)
        assert np.max(abs((old.M-s.M).toarray()))<1e-14
        assert np.max(abs((old.K-s.K).toarray()))<1e-13


@pytest.mark.parametrize('degree',[1,2,4])
def test_matrix_functions_against_full_generalized_eigen(degree):
    mesh=mesh_shape(make_shape('square'),.3);s=Space(mesh,degree)
    initial=lambda x:np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1]);velocity=lambda x:initial(x)*.4
    w=ContinuousWave(s,initial,velocity);e=SpectralControl(s,w.u0,w.v0);times=[0.,.0073,.237,1.]
    u,v=w.evaluate(s.position[s.free],times,return_nodal=True)
    assert e.diagnostics['relative_eigen_residual']<1e-12
    assert e.diagnostics['mass_orthogonality_max']<1e-12
    for i,t in enumerate(times):
        a,b=e.nodal(t)
        assert np.max(abs(u[i]-a))<2e-10 and np.max(abs(v[i]-b))<2e-9
        assert abs(s.energy(a,b)-s.energy(w.u0,w.v0))<1e-10
    assert np.max(abs(u[0]-w.u0))<1e-12 and np.max(abs(v[0]-w.v0))<1e-10
    assert w.last_diagnostics['scalar_relative_error']<1e-10


def test_nested_mesh_preserves_continuous_prior():
    from pdeep.geometric.data import sample_field
    base=mesh_shape(make_shape('hole'),.2);field=sample_field(base,np.random.default_rng(9),zero_trace=True);x=base.sample(41,np.random.default_rng(2));energy=[]
    for mesh in (base,refine(base)):
        for degree in (1,2):
            s=Space(mesh,degree);u=s.initial(field)
            assert np.max(abs(s.evaluate(u,x)-field(x)))<1e-12
            energy.append(float(u@s.Ki@u))
    assert max(energy)-min(energy)<1e-10


def test_smooth_prior_is_explicit_not_replacement():
    from pdeep.geometric.data import VERSION
    mesh=mesh_shape(make_shape('annulus'),.12);a,b=pair(mesh,'test',relative_radius=.18);again=pair(mesh,'test',relative_radius=.18)
    assert VERSION=='G1-canonical-P1-squared-elliptic-Gaussian-v1'
    assert a.identity==again[0].identity and a.identity!=b.identity
    boundary=mesh.vertices[np.unique(mesh.boundary_edges)]
    assert np.max(abs(a(boundary)))==0.
    # Global C3 at support cutoff: values decay as distance^4.
    c=a.centers[0];direction=np.array([1.,0.]);isolated=replace(a,centers=a.centers[:1],coefficients=np.ones(1))
    u1=isolated(np.array([c+(a.radius-1e-3)*direction]))[0];u2=isolated(np.array([c+(a.radius-5e-4)*direction]))[0]
    assert u1/u2>15


def test_launch_render_never_calls_callback(monkeypatch):
    import ipywidgets as widgets
    from pdeep.notebook_launch import offer_launch
    calls=[];monkeypatch.setattr('IPython.display.display',lambda *args:None)
    ui=offer_launch(lambda:calls.append('optimizer'))
    assert calls==[] and isinstance(ui.widget,widgets.Button)
    # Deliberately no synthetic button click: this task authorizes ZERO training.


def test_saved_preview_does_not_launch_and_enabled_mode_is_synchronous(monkeypatch):
    import nbformat
    from types import SimpleNamespace
    nb=nbformat.read(ROOT/'notebooks/02_train.ipynb',4)
    cell=next(c for c in nb.cells if c.cell_type=='code' and 'RESULTS = launch(PLAN' in c.source)
    import pdeep.smooth.workflows as smooth
    calls=[]
    monkeypatch.setattr(smooth,'launch',lambda *a,**kw:calls.append(kw['expected_identity']) or {'mocked':True})
    scope={'RUN_MODE':'preview'};exec(cell.source,scope);assert calls==[]
    for mode in ('train','resume'):
        scope=dict(RUN_MODE=mode,ACTION='fresh' if mode=='train' else 'resume',
            training_settings=lambda:{},SETTINGS={'certificate_path':'cert'},
            CERTIFICATION=SimpleNamespace(verified=True,location='cert',require=lambda:None),
            PLAN=object(),REVIEWED_IDENTITY='reviewed',Dashboard=lambda **kw:None)
        exec(cell.source,scope);assert scope['RESULTS']=={'mocked':True}
    assert calls==['reviewed','reviewed']
    evaluation=nbformat.read(ROOT/'notebooks/03_evaluate.ipynb',4)
    retained=next(c for c in evaluation.cells if c.cell_type=='code' and 'def _evaluate_retained_T1' in c.source)
    exec(retained.source,{'ENABLE_T1_EVALUATION':False})


def test_research_gate_cannot_be_bypassed_from_engine():
    from types import SimpleNamespace
    from pdeep.geometric.training import Engine
    for preset in ('bounded','research','large'):
        obj=object.__new__(Engine);obj.config=SimpleNamespace(preset=preset)
        with pytest.raises(ValidationError,match='G3'):obj.run(launch_full=True)


def test_migration_is_explicit_and_rejects_unknown_scientific_changes(tmp_path):
    from pdeep.wave_fidelity.compatibility import migrate_unaffected_checkpoint
    from pdeep.artifact_io import read_payload
    candidates=list((ROOT/'reports/geometry_phase/bounded_runs/A_GEO').rglob('geometry-update-00000001.json'))
    if not candidates:pytest.skip('historical geometry checkpoint unavailable in this checkout')
    old=candidates[0]
    before=old.read_bytes();new=migrate_unaffected_checkpoint(old,tmp_path/'migrated')
    a,_=read_payload(old);b,_=read_payload(new)
    assert a['payload_tree_identity']==b['payload_tree_identity'] and old.read_bytes()==before
    assert b['compatibility_migration']['weights_optimizer_RNG_changed'] is False


def test_fine_mesh_scalar_mapping_precision():
    from pdeep.wave_fidelity.reference import coefficients
    c,d=coefficients(10251878.42844125,[0.,.0073,.02,.05,.1,.237,.5,1.])
    assert d['scalar_relative_error']<2e-11
    assert abs(np.polynomial.chebyshev.chebval(-1.,c[-1,0])-1.)<2e-11


def test_gauss_integrator_against_semidiscrete_and_velocity_derivative():
    from pdeep.wave_fidelity.reference import GaussWave
    mesh=mesh_shape(make_shape('square'),.25);s=Space(mesh,2)
    u0=lambda x:np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1]);v0=lambda x:.3*u0(x);w=GaussWave(s,u0,v0);exact=SpectralControl(s,w.u0,w.v0)
    x=mesh.sample(16,np.random.default_rng(1));times=[0.,.0073,.137,1.];errors=[]
    for dt in (.008,.004,.002):
        u,v=w.evaluate(x,times,dt=dt);e=[]
        for j,t in enumerate(times):
            a,b=exact.nodal(t);e.append(max(abs(v[j]-s.evaluate(b,x))))
        errors.append(max(e));assert w.last_diagnostics['relative_grid_energy_drift']<1e-11
    assert errors[-1]<errors[0]/20
    d=1e-6;time=.137
    u,v=w.evaluate(x,[time-d,time,time+d],dt=.002)
    assert max(abs((u[2]-u[0])/(2*d)-v[1]))<2e-8
    boundary=mesh.vertices[np.unique(mesh.boundary_edges)];u,v=w.evaluate(boundary,[0.,.0073,1.])
    assert np.max(abs(u))<1e-13 and np.max(abs(v))<1e-12


def test_quartic_poisson_identity_independent_of_wave_propagator():
    from pdeep.geometric.quadrature import cells
    mesh=mesh_shape(make_shape('square'),.22);s=Space(mesh,4)
    u0=lambda x:x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1]);u=s.initial(u0)
    x=mesh.sample(37,np.random.default_rng(3));assert np.max(abs(s.evaluate(u,x)-u0(x)))<2e-14
    q=cells(mesh,6);basis=lagrange(q.barycentric,4)[0];f=2*(q.position[:,0]*(1-q.position[:,0])+q.position[:,1]*(1-q.position[:,1]));load=np.zeros(len(s.position))
    np.add.at(load,s.dofs[q.cell_id].ravel(),(q.weight[:,None]*basis*f[:,None]).ravel())
    residual=s.Ki@u/.49-load[s.free]
    assert np.linalg.norm(residual)/(1+np.linalg.norm(load[s.free]))<1e-12


def test_smooth_field_is_resolution_independent():
    a=mesh_shape(make_shape('square'),.16);b=mesh_shape(make_shape('square'),.08)
    x=pair(a,'frozen');y=pair(b,'frozen');assert x[0].identity==y[0].identity and x[1].identity==y[1].identity


def test_migration_rejects_changed_model_source_and_mesh(tmp_path):
    from pdeep.wave_fidelity.compatibility import migrate_unaffected_checkpoint
    from pdeep.artifact_io import read_payload,write_artifact
    candidates=list((ROOT/'reports/geometry_phase/bounded_runs/A_GEO').rglob('geometry-update-00000001.json'))
    if not candidates:pytest.skip('historical geometry checkpoint unavailable in this checkout')
    original,payload=read_payload(candidates[0])
    excluded={'artifact_id','schema_version','created_UTC','files','verification','content_identity','payload_tree_identity'}
    for reason in ('source','mesh'):
        import copy
        meta=copy.deepcopy({k:v for k,v in original.items() if k not in excluded})
        if reason=='source':meta['geometry_source']['models.py']='not-the-verified-model'
        else:meta['config']['reference_h']=.01
        changed=write_artifact(tmp_path/reason,None,meta,payload)
        with pytest.raises(ValidationError,match='scientific model/data|mesh/physical case'):
            migrate_unaffected_checkpoint(changed,tmp_path/(reason+'-migration'))


def test_fast_gauss_matches_full_spectral_rational_amplification():
    from pdeep.wave_fidelity.reference import GaussWave
    mesh=mesh_shape(make_shape('square'),.25);s=Space(mesh,1);stream=np.random.default_rng(99)
    a=np.zeros(len(mesh.vertices));b=a.copy();a[s.free]=stream.normal(size=len(s.free));b[s.free]=stream.normal(size=len(s.free))
    w=GaussWave(s,lambda x:mesh.evaluate(a,x),lambda x:mesh.evaluate(b,x));e=SpectralControl(s,w.u0,w.v0)
    dt=.2;nu=e.omega*dt;theta=2*np.arctan2(nu/2,1-nu*nu/12);phase=5*theta
    expected_u=e.phi@(e.a*np.cos(phase)+e.b/e.omega*np.sin(phase));expected_v=e.phi@(-e.omega*e.a*np.sin(phase)+e.b*np.cos(phase))
    u,v=w.evaluate(s.position[s.free],[1.],dt=dt)
    assert np.max(abs(u[0]-expected_u))<1e-11 and np.max(abs(v[0]-expected_v))<1e-10


def test_arbitrary_time_reference_query_chunk_consistency():
    from pdeep.wave_fidelity.reference import GaussWave
    mesh=mesh_shape(make_shape('square'),.25);s=Space(mesh,2);u0=lambda x:np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1]);v0=lambda x:.2*u0(x)
    x=mesh.sample(13,np.random.default_rng(4));times=[0.,.0073,.137,.237,1.]
    for reference in (GaussWave(s,u0,v0),ContinuousWave(s,u0,v0)):
        u,v=reference.evaluate(x,times)
        for j,t in enumerate(times):
            a,b=reference.evaluate(x,[t]);np.testing.assert_allclose(a[0],u[j],atol=2e-11,rtol=2e-11);np.testing.assert_allclose(b[0],v[j],atol=2e-10,rtol=2e-10)
