"""Bounded scientific checks; no automatic optimizer/training side effects."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from pdeep.core import ValidationError
from pdeep.geometric.domain import make_shape,mesh_shape,polygon_shape,GeometryDomain
from pdeep.geometric.quadrature import cells,boundary
from pdeep.geometric.data import physical_case,Normalizers,calibrate,batch,token_set,PDES,CATALOG,domain,rng,Problem
from pdeep.geometric.representation import layout,retrieval_weights,interpolate
from pdeep.geometric.models import create
from pdeep.geometric.physics import Adapter,WeakConfig,weak_loss


@pytest.fixture(scope='module')
def normals():return calibrate(5,64)


@pytest.mark.parametrize('family',['square','rectangle','lshape','disk','ellipse','hole','annulus','polygon'])
def test_domain(family,tmp_path):
    m=mesh_shape(make_shape(family),.16);m.validate();q=cells(m,4);b=boundary(m,4)
    assert abs(q.weight.sum()-m.shape.polygon.area)<1e-12
    assert abs(b.weight.sum()-m.shape.polygon.length)<1e-12
    x=m.sample(50,np.random.default_rng(9));assert np.max(abs(m.evaluate(m.vertices[:,0]+2*m.vertices[:,1],x)-x[:,0]-2*x[:,1]))<2e-14
    path=tmp_path/'mesh.npz';m.save(path);assert GeometryDomain.load(path).mesh_id==m.mesh_id
    assert mesh_shape(make_shape(family,seed=13),.12).geometry_id==m.geometry_id
    if family in {'hole','annulus'}:
        with pytest.raises(ValidationError):m.locate([[.5,.5]])


def test_physical_geometry_identity():
    a=polygon_shape(((0,0),(1,0),(1,1),(0,1)),tags=(1,2,3,4))
    b=polygon_shape(((1,1),(0,1),(0,0),(1,0)),tags=(9,8,7,6),family='custom')
    assert a.geometry_id==b.geometry_id
    with pytest.raises(ValidationError):polygon_shape(((0,0),(1,1),(0,1),(1,0)))
    with pytest.raises(ValidationError):polygon_shape(((0,0),(1,0),(1,1),(0,1)),(((2,2),(3,2),(3,3)),))


def test_data_identity_and_noise(normals):
    ids=set()
    for split in ('train','validation','test'):
        for p in PDES:
            c=physical_case(split,p,0,h=.16,dt=.02);assert c.case_id not in ids;ids.add(c.case_id)
            obs,meta=token_set(c,7,normals,observation=True);same,_=token_set(c,7,normals,observation=True)
            assert torch.equal(obs.value,same.value)
            assert obs.sigma.min()>0
            assert torch.equal(obs.time_present,torch.full_like(obs.valid,p!='poisson'))
            noise=(obs.value[0,:,0].double().numpy()-meta['clean'])/meta['sigma']
            from pdeep.geometric.data import rng
            np.testing.assert_allclose(noise,rng(c.case_id,'measurement_noise',0).standard_normal(7),atol=2e-5)
            refined=physical_case(split,p,0,h=.12,dt=.02);assert refined.case_id==c.case_id and refined.problem.mesh.mesh_id!=c.problem.mesh.mesh_id
            for k in c.fields:assert np.array_equal(c.fields[k].nodal,refined.fields[k].nodal)
    train={make_shape(f,aspect=a).geometry_id for f,a in CATALOG['train']};held={make_shape(f,aspect=a).geometry_id for f,a in CATALOG['geometry_held_out']}
    assert train.isdisjoint(held)
    assert normals.calibration['split']=='train' and len(normals.calibration['case_ids'])==15


def test_retrieval_derivatives():
    m=domain('hole',1.,.16);lay=layout(m,8,16);x=torch.tensor(m.sample(3,rng('derivative')),dtype=torch.float64,requires_grad=True)
    def f(z):return retrieval_weights(lay,z)
    assert torch.autograd.gradcheck(f,(x,),eps=1e-6,atol=2e-6,rtol=1e-4)
    assert torch.autograd.gradgradcheck(f,(x,),eps=1e-6,atol=2e-5,rtol=1e-4)
    a=f(x);assert torch.allclose(a[:,:8].sum(1),torch.ones(3,dtype=x.dtype));assert torch.allclose(a[:,8:].sum(1),torch.ones(3,dtype=x.dtype))
    np.testing.assert_allclose(a.detach(),torch.cat([f(z[None]) for z in x]).detach(),atol=1e-14)
    # Graph metric around hole is longer than straight crossing through void.
    distance=np.sqrt(lay.distance2);euclidean=np.linalg.norm(m.vertices[:,None]-m.vertices[lay.anchors][None],axis=-1)
    assert (distance-euclidean).max()>.1


def reindex(m):
    perm=np.random.default_rng(23).permutation(len(m.vertices));inv=np.argsort(perm)
    return GeometryDomain(m.shape,m.vertices[perm],inv[m.cells],inv[m.boundary_edges],m.boundary_tags.copy(),m.region_tags.copy(),dict(m.provenance)).validate()


@pytest.mark.parametrize('architecture',['A_GEO','C_GEO'])
def test_neural_invariances(architecture,normals):
    c=physical_case('geometry_held_out','wave',0,h=.16,dt=.02);b=batch([c],normals,sensors=9,queries=11)
    m=create(architecture,normals,'verification',seed=17);m.eval();ctx=m.prepare_context(b.specs,b.observations)
    y=m.predict(ctx,b.queries,chunk=11);assert torch.allclose(y,m.predict(ctx,b.queries,chunk=3),atol=2e-5,rtol=2e-5)
    obs=b.observations;perm=torch.tensor([8,1,4,5,3,2,7,0,6]);op=replace(obs,**{k:v[:,perm] for k,v in obs.__dict__.items() if isinstance(v,torch.Tensor)})
    yp=m(b.specs,op,b.queries);assert torch.allclose(y,yp,atol=3e-5,rtol=3e-5)
    padded={}
    for k,v in obs.__dict__.items():
        if v is None:padded[k]=None;continue
        shape=list(v.shape);shape[1]=3;pad=torch.zeros(shape,dtype=v.dtype)
        if v.is_floating_point():pad.fill_(float('nan'))
        padded[k]=torch.cat((v,pad),1)
    assert torch.allclose(y,m(b.specs,type(obs)(**padded),b.queries),atol=1e-6,rtol=1e-6)
    changed=replace(obs,value=obs.value+2*normals.scale('wave','u'));assert (y-m(b.specs,changed,b.queries)).abs().max()>1e-4
    x=obs.position.clone();x[:,0]=x[:,1];assert (y-m(b.specs,replace(obs,position=x),b.queries)).abs().max()>1e-5
    other=replace(c.problem,mesh=reindex(c.problem.mesh));assert torch.allclose(y,m((other,),obs,b.queries),atol=5e-5,rtol=5e-5)
    empty=batch([c],normals,sensors=0,queries=5);assert torch.isfinite(m(empty.specs,empty.observations,empty.queries)).all()
    loss=Adapter(architecture,WeakConfig(2,2,4,256)).loss(m,b);loss.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)>0


def test_numeric_coefficient_and_absent_time(normals):
    c=physical_case('validation','heat',0,h=.16,dt=.02);b=batch([c],normals,sensors=3,queries=3);m=create('A_GEO',normals,'verification')
    g1=m.equation((c.problem,),torch.device('cpu'))[1];g2=m.equation((replace(c.problem,coefficient=.04),),torch.device('cpu'))[1]
    assert (g1-g2).abs().max()>1e-5
    lay=layout(c.problem.mesh,8,16);x=b.queries.position[0];t=torch.zeros(len(x),1)
    a=m.coordinates(lay,x,t,False);z=m.coordinates(lay,x,t,True);assert not torch.equal(a,z)


def test_capacity_and_independence(normals):
    counts={}
    for a in ('A_GEO','C_GEO'):
        m=create(a,normals,'medium');counts[a]=m.parameter_count;assert 5_000_000<=m.parameter_count<9_000_000
        other=create(a,normals,'medium');assert next(m.parameters()).data_ptr()!=next(other.parameters()).data_ptr()
        large=create(a,normals,'large');assert 9_000_000<=large.parameter_count<12_000_000
    assert counts['A_GEO']!=counts['C_GEO']


@pytest.mark.parametrize('pde',PDES)
def test_independent_weak_manufactured_and_chunking(pde):
    from types import SimpleNamespace
    mesh=domain('hole',1.,.16);a=1. if pde=='poisson' else .02 if pde=='heat' else .49
    problem=Problem(pde,mesh,a);ctx=SimpleNamespace(problem=problem,equation=torch.zeros(1,dtype=torch.float64))
    class Manufactured:
        def evaluate(self,context,x,t,quantity='u'):
            phi=torch.sin(torch.pi*x[:,0:1])*torch.sin(torch.pi*x[:,1:2]);lam=2*torch.pi**2
            if quantity in {'u0','f'}:return phi*(lam if quantity=='f' else 1.)
            if quantity=='v0':return phi*0
            if pde=='poisson':return phi
            if pde=='heat':return phi*torch.exp(-a*lam*t)
            omega=np.sqrt(a*lam)
            return phi*torch.cos(omega*t) if quantity=='u' else -omega*phi*torch.sin(omega*t)
    norms=Normalizers({pde:{k:1. for k in ('u','v','f','u0','v0')}},{})
    full=weak_loss(Manufactured(),[ctx],norms,WeakConfig(6,3,12,10000))
    chunk=weak_loss(Manufactured(),[ctx],norms,WeakConfig(6,3,12,47))
    assert float(full['weak'])<1e-16
    assert torch.allclose(full['weak'],chunk['weak'],atol=1e-25,rtol=1e-6)


def test_assembly_and_failure_paths():
    from pdeep.geometric.fem import FEM,BoundaryData
    mesh=domain('square',1.,.16);f=FEM(mesh)
    assert abs(f.M.sum()-mesh.cell_areas.sum())<1e-13
    assert np.abs(f.K@np.ones(len(mesh.vertices))).max()<1e-12
    assert np.linalg.eigvalsh(f.K[f.free][:,f.free].toarray()).min()>0
    with pytest.raises(ValidationError):FEM(mesh,coefficient=-1.)
    with pytest.raises(ValidationError):FEM(mesh,conditions={1:BoundaryData()})
    with pytest.raises(ValidationError):f.dynamics('heat',np.ones(len(mesh.vertices)),dt=.1)


def test_mixed_neumann_weak_sign():
    from types import SimpleNamespace
    mesh=domain('square',1.,.16);spec=tuple((k,'DIRICHLET',0.) if k==4 else (k,'NEUMANN',2. if k==3 else -2. if k==1 else 1.) for k in range(1,5))
    p=Problem('poisson',mesh,1.,boundary_spec=spec);ctx=SimpleNamespace(problem=p,equation=torch.zeros(1,dtype=torch.float64))
    class Affine:
        def evaluate(self,c,x,t,quantity='u'):return (x[:,0:1]+2*x[:,1:2])*(0. if quantity=='f' else 1.)
    n=Normalizers({'poisson':{'u':1.,'f':1.}},{})
    r=weak_loss(Affine(),[ctx],n,WeakConfig(4,2,16,127));assert float(r['weak'])<1e-25


def test_variable_diffusion():
    from pdeep.geometric.fem import FEM,BoundaryData
    errors=[]
    for h in (.16,.08):
        m=domain('square',1.,h);u=lambda x,t:np.exp(x[:,0]+.3*x[:,1]);a=lambda x,t:1+x[:,0]
        f=FEM(m,coefficient=a,conditions={k:BoundaryData(value=u) for k in (1,2,3,4)},order=6)
        result=f.poisson(lambda x,t:-((1+x[:,0])*1.09+1)*u(x,t));q=cells(m,6);e=m.evaluate(result,q.position)-u(q.position,0.);errors.append(float(np.sqrt(q.weight@(e*e))))
    assert errors[1]<errors[0]/2
