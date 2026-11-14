from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from pdeep.unified.reference import compress_states,output_bounds,LinearCache
from pdeep.smooth.data import Case
from pdeep.smooth.reference import Reference
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.geometric.domain import mesh_shape,make_shape
from pdeep.wave_fidelity.reference import Space
from .conftest import EVIDENCE


def test_compression_measured_entrywise_and_output_bounds():
    rng=np.random.default_rng(7193);X=rng.normal(size=(160,5));T=rng.normal(size=(5,31));u=(X@T).T;v=(X@rng.normal(size=(5,31))).T
    A,B,r=compress_states(u,v,.002,tolerance=1e-11,block=16)
    eu=np.max(abs(A@B[:,:31]-u.T));ev=np.max(abs(A@B[:,31:]/.002-v.T))
    assert eu<=r['max_nodal_u_error']+1e-13 and ev<=r['max_nodal_v_error']+1e-13
    assert eu<1e-11 and ev<1e-11
    b=output_bounds(eu,ev,.002);assert b['v']>=1500*eu+2*ev
    EVIDENCE['factorization_max_nodal_u_error']=float(eu);EVIDENCE['factorization_max_nodal_v_error']=float(ev)

@pytest.mark.parametrize('pde',['poisson','heat','wave'])
def test_linear_cache_arbitrary_coordinates_time_chunk_linearity_reload(certificate,tmp_path,monkeypatch,pde):
    # Independent small mesh control, explicitly NOT production P4 qualification.
    # All 32/64 continuous basis fields are used; no truncation/field smoothing.
    from pdeep.unified import reference as mod
    small=object.__new__(Reference);small.space=Space(mesh_shape(make_shape('square'),.35),2,speed=1.);small.family='square';small.h=.35;small.degree=2;small.factors={}
    original=FixedReference
    monkeypatch.setattr(mod,'FixedReference',lambda c:original(c,reference_object=small))
    c=Case('square',pde,'validation',31);cache=LinearCache(tmp_path,certificate)
    cache.prepare(c)
    rng=np.random.default_rng(7193);x=rng.uniform(.1,.9,(17,2));t=np.r_[0.,.0073,.02,.237,1.,rng.uniform(0,1,12)];q=np.arange(17)%2 if pde=='wave' else np.zeros(17,int)
    full=original(c,reference_object=small).evaluate(x,t,q);y,b=cache.evaluate(c,x,t,q,return_bound=True)
    error=float(max(abs(y-full)));assert error<1e-6
    np.testing.assert_allclose(cache.evaluate(c,x,t,q,chunk=3),y,atol=1e-11,rtol=1e-11)
    reloaded=LinearCache(tmp_path,certificate);np.testing.assert_allclose(reloaded.evaluate(c,x,t,q),y,atol=0,rtol=0)
    assert max(abs(y-full))<=max(b)+1e-10
    from pdeep.unified.dataset import quality_check
    from pdeep.unified.config import DatasetConfig
    from pdeep.smooth.certification import normalizers
    monkeypatch.setattr('pdeep.unified.dataset.reference',lambda *a,**kw:small)
    qc=quality_check(c,cache,normalizers(certificate),DatasetConfig(preset='smoke',qc_points=8),finer=False)
    assert qc['passed'],qc
    if pde=='wave':
        tt=np.array([.0073,.237,.731]);xx=x[:3];h=1e-6;zero=np.zeros(3,int)
        fd=(cache.evaluate(c,xx,tt+h,zero)-cache.evaluate(c,xx,tt-h,zero))/(2*h)
        v=cache.evaluate(c,xx,tt,np.ones(3,int));difference=float(max(abs(fd-v)));assert difference<1e-7
        EVIDENCE['small_P2_cache_kinematic_FD_max_error']=difference
    EVIDENCE[pde+'_small_P2_linear_cache_max_error']=error
    assert len(list(tmp_path.glob('space-*.json')))==1
    # Preparation reuse never launches a second propagation.
    monkeypatch.setattr(mod,'FixedReference',lambda *a,**k:pytest.fail('verified basis regenerated'))
    cache.prepare(c)
    for path in tmp_path.glob('*.pt'):
        payload=torch.load(path,weights_only=True,map_location='cpu');assert isinstance(payload,dict)
