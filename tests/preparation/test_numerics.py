from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from pdeep.smooth.definition import geometry
from pdeep.wave_fidelity.reference import Space
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.smooth.reference import Reference
from pdeep.smooth.data import Case
from pdeep.unified.config import DatasetConfig
from pdeep.unified.dataset import inventory
from pdeep.unified.streams import encounters as old_plan,prepare_labels as old_labels
from pdeep.preparation.planning import encounters
from pdeep.preparation.numerics import Sampler,FastReference,PoissonBasis
from pdeep.preparation.labels import Journal,prepare_labels

@pytest.mark.parametrize('family',('square','rectangle','lshape','disk','ellipse','hole','annulus','polygon'))
@pytest.mark.parametrize('degree',(1,2,4))
def test_full_basis_sampling_all_geometries(family,degree):
    # Pure full-space arithmetic control; does not claim reference certification.
    s=Space(geometry(family,.12),degree,speed=1.);rng=np.random.default_rng(72);x=s.mesh.sample(31,rng);boundary=s.mesh.vertices[np.unique(s.mesh.boundary_edges)][::7];x=np.vstack((x,boundary));t=np.resize([0.,.00013,.0173,.1,.999,1.],len(x));q=np.arange(len(x))%2
    obj=object.__new__(FixedReference);obj.reference=SimpleNamespace(space=s);obj.case=SimpleNamespace(pde='wave');obj.dt=.5;obj.u=rng.normal(size=(3,len(s.free)));obj.v=rng.normal(size=obj.u.shape);obj.evaluations=0
    expected=obj.evaluate(x,t,q);actual=Sampler(s).evaluate(obj,x,t,q,chunk=7)
    scale=max(1.,float(abs(expected).max()));assert abs(actual-expected).max()<2e-13*scale
    obj.case=SimpleNamespace(pde='poisson');obj.v=None;q[:]=0
    np.testing.assert_allclose(Sampler(s).evaluate(obj,x,t,q),obj.evaluate(x,t,q),rtol=2e-13,atol=2e-13)

def test_plan_exact_order_seeds_splits_and_prefix():
    a=DatasetConfig(preset='smoke',stream_updates=59,stream_queries=16);rows=inventory(a)
    assert encounters(a,rows)==old_plan(a,rows)
    b=replace(a,stream_updates=91);long=encounters(b,inventory(b));short=encounters(a,rows)
    for cid,events in short.items():assert events==[x for x in long[cid] if x[0]<59]

@pytest.mark.parametrize('pde',('poisson','heat','wave'))
def test_numerical_states_labels_and_partial_resume(pde,tmp_path):
    r=object.__new__(Reference);r.space=Space(geometry('hole',.12),2,speed=1.);r.family='hole';r.h=.12;r.degree=2;r.factors={}
    c=Case('hole',pde,'train',4);old=FixedReference(c,reference_object=r);basis=PoissonBasis(r,c.fields['f']) if pde=='poisson' else None;new=FastReference(c,reference_object=r,poisson_basis=basis)
    np.testing.assert_allclose(new.u,old.u,rtol=2e-12,atol=1e-14)
    if pde!='poisson':np.testing.assert_array_equal(new.v,old.v)
    config=DatasetConfig(preset='smoke',stream_updates=3,stream_queries=32,stream_grid=4);events=[(0,0,64),(1,0,64)]
    expected=old_labels(c,old,config,events);journal=Journal(tmp_path,c,config)
    partial,done=prepare_labels(c,new,config,events,journal=journal,max_batches=1,point_batch=64);assert not done and partial
    labels,done=prepare_labels(c,new,config,events,journal=Journal(tmp_path,c,config),point_batch=64);assert done and set(labels)==set(expected)
    for k,v in expected.items():np.testing.assert_allclose(labels[k],v,rtol=2e-11,atol=1e-11)
