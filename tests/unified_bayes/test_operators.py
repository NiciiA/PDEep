from functools import lru_cache
from pathlib import Path
import time
import numpy as np
import pytest
from pdeep.smooth.reference import Reference
from pdeep.smooth.definition import geometry,PDES,GEOMETRIES
from pdeep.wave_fidelity.reference import Space
from pdeep.unified_bayes.operators import ForwardOperators,Coordinates,prior_specification,_basis_fields
from pdeep.unified_bayes.interpolation import geodesic_distances,interpolate,UnsupportedInterpolation
from .conftest import EVIDENCE


@lru_cache(maxsize=1)
def control(family):
    r=object.__new__(Reference);r.space=Space(geometry(family,.16),2,speed=1.);r.family=family;r.h=.16;r.degree=2;r.factors={};return r

@pytest.mark.parametrize('pde',PDES)
@pytest.mark.parametrize('family',tuple(GEOMETRIES))
def test_all24_full_algorithm_small_space(tmp_path,certificate,pde,family):
    r=control(family);rng=np.random.default_rng(7);x=r.space.mesh.sample(12,rng);t=np.array([0,.0037,.0191,1.]*3);q=np.arange(12)%2 if pde=='wave' else np.zeros(12,int);c=Coordinates(x,None if pde=='poisson' else t,q)
    op=ForwardOperators(tmp_path,certificate,_reference_factory=control,_control_identity='NONPRODUCTION-P2-h.16-linear-algorithm-test-v1');start=time.perf_counter();report=op.prepare(pde,family,{'a':c});H,b,m=op.get(pde,family,c);p=prior_specification(pde,family)
    assert H.shape==(12,64 if pde=='wave' else 32) and H.dtype==np.float64 and not b.any();assert p['covariance']=='identity'
    assert m['superposition']['maximum_training_RMS_error']<1e-8
    H2,b2,_=ForwardOperators(tmp_path,certificate,_reference_factory=control,_control_identity=op.control).get(pde,family,c);np.testing.assert_array_equal(H,H2)
    assert op.prepare(pde,family,{'again':c})['cached']
    fields=_basis_fields(p)
    if pde!='poisson':
        zero=np.flatnonzero(t==0);S=r.space.sample_operator(x[zero]);u=S@r.initial(fields['u0']);v=S@r.initial(fields['v0']) if pde=='wave' else None
        expected=np.where(q[zero,None]==1,v,u) if pde=='wave' else u
        np.testing.assert_allclose(H[zero],expected,atol=2e-15)
    wrong=Coordinates(x,t if pde=='poisson' else None,q)
    with pytest.raises(ValueError,match='ABSENT'):wrong.validate(pde,family)
    EVIDENCE.setdefault('small_space_24_tasks',{})[pde+'/'+family]={'seconds':time.perf_counter()-start,**report['verification'],'dimension':p['dimension'],'scope':'same reference algorithm on P2 h.16 control; NOT production-reference fidelity certification'}


def test_full_production_poisson(tmp_path,certificate):
    x=np.array([[.27,.33],[.5,.5],[.72,.68]]);c=Coordinates(x,None,np.zeros(3,int));op=ForwardOperators(tmp_path,certificate);start=time.perf_counter();report=op.prepare('poisson','square',{'bounded':c});H,b,m=op.get('poisson','square',c)
    assert m['numerical_diagnostics']['spatial_basis']=='P4' and H.shape==(3,32)
    EVIDENCE['production_P4_square_poisson']={'seconds':time.perf_counter()-start,**report['verification'],'observations':3,'batch_RHS':32,'all24_full_production':'not run'}


def test_geodesic_hole_channels_and_absent_time():
    mesh=geometry('hole',.08);s=np.array([[.2,.5],[.8,.5]]);q=np.array([[.8,.5],[.2,.5]]);d=geodesic_distances(mesh,s,q);assert d[0,0]>.65 # straight shortcut would be .6
    a=Coordinates(s,None,np.zeros(2,int));b=Coordinates(q,None,np.zeros(2,int));v,cost=interpolate('poisson','hole',a,b,np.array([1.,2.]),np.array([.03,.04]));assert np.isfinite(v).all()
    with pytest.raises(UnsupportedInterpolation):interpolate('wave','hole',Coordinates(s,np.zeros(2),np.zeros(2,int)),Coordinates(q,np.zeros(2),np.ones(2,int)),[1.,2.],[.03,.04])
    assert a.identity!=Coordinates(s,np.zeros(2),np.zeros(2,int)).identity
    EVIDENCE['hole_distance']={'graph_geodesic':float(d[0,0]),'inadmissible_straight_distance':.6}
