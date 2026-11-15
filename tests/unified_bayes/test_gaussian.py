import numpy as np
import pytest
from pdeep.unified_bayes.gaussian import condition
from .conftest import EVIDENCE


def test_affine_independent_forms_masks_and_dense():
    rng=np.random.default_rng(813);H=rng.normal(size=(9,5));y=rng.normal(size=9);sigma=rng.uniform(.03,.5,9);mu=rng.normal(size=5);A=rng.normal(size=(5,5));C=A@A.T+.2*np.eye(5);b=rng.normal(size=9)
    a=condition(H,y,sigma,prior_mean=mu,prior_covariance=C,offset=b);c=condition(H,y,sigma,prior_mean=mu,prior_covariance=C,offset=b,form='observation')
    S=H@C@H.T+np.diag(sigma*sigma);mean=mu+C@H.T@np.linalg.solve(S,y-b-H@mu);cov=C-C@H.T@np.linalg.solve(S,H@C)
    errors={'mean_forms':float(abs(a.mean-c.mean).max()),'covariance_forms':float(abs(a.covariance-c.covariance).max()),'mean_dense':float(abs(a.mean-mean).max()),'covariance_dense':float(abs(a.covariance-cov).max())}
    EVIDENCE['posterior_forms']=errors
    for v in errors.values():assert v<2e-11
    assert np.linalg.eigvalsh(a.covariance).min()>0 and np.isfinite(a.mean).all()
    pad=condition(np.vstack((H,np.full((2,5),np.nan))),np.r_[y,np.nan,np.nan],np.r_[sigma,0,np.nan],prior_mean=mu,prior_covariance=C,offset=np.r_[b,np.nan,np.nan],valid=[True]*9+[False]*2)
    np.testing.assert_array_equal(pad.mean,a.mean)
    prior=condition(np.zeros((0,5)),[],[],prior_mean=mu,prior_covariance=C);np.testing.assert_array_equal(prior.mean,mu);np.testing.assert_array_equal(prior.covariance,C)
    with pytest.raises(ValueError,match='strictly positive'):condition(H,y,np.zeros(9))


def test_limits_dependent_rows_contraction_and_chunks():
    rng=np.random.default_rng(62);H=rng.normal(size=(30,6));H[1]=H[0];H[2]=H[0]+1e-14;B=rng.normal(size=(91,6));a=rng.normal(size=6);y=H@a;s=np.full(30,.05)
    previous=np.full(91,np.inf)
    for n in (0,3,10,30):
        p=condition(H[:n],y[:n],s[:n]);v=p.predict(B,chunk=7)['latent_variance'];assert np.all(v<=previous+1e-13);previous=v
    p=condition(H,y,s);x=p.predict(B,full_covariance=True);c=p.predict(B,chunk=1)
    np.testing.assert_allclose(x['mean'],c['mean'],atol=2e-15);np.testing.assert_allclose(x['latent_variance'],np.diag(x['latent_covariance']),atol=1e-16)
    assert np.isfinite(p.mean).all() and (x['latent_variance']>=0).all()
    future=p.predict(B,noise_sigma=np.full(91,.2));np.testing.assert_allclose(future['measurement_variance']-future['latent_variance'],.04,rtol=1e-14)
    np.testing.assert_allclose(p.cross_covariance(B[:3],B[3:6]),x['latent_covariance'][:3,3:6],atol=1e-16)
    large=condition(H,y,np.full(30,1e8));np.testing.assert_allclose(large.covariance,np.eye(6),atol=1e-13)
    small=condition(H,y,np.full(30,1e-7));np.testing.assert_allclose(small.mean,a,atol=2e-13)
    # Extreme precision takes square-root QR, preserving the stated model/noise.
    extreme=condition(H,y,np.full(30,1e-12));np.testing.assert_allclose(extreme.mean,a,atol=5e-14)
    EVIDENCE['nested_contraction']={'last_maximum_variance':float(previous.max()),'independent_count_layouts':'not assumed nested','small_noise_max_error':float(abs(small.mean-a).max())}


def test_coherent_posterior_samples():
    rng=np.random.default_rng(6);H=rng.normal(size=(8,4));p=condition(H,rng.normal(size=8),np.full(8,.3));n=80000;draw=p.draw(n,seed=7193);np.testing.assert_array_equal(draw,p.draw(n,seed=7193))
    mean=draw.mean(0);cov=np.cov(draw,rowvar=False);sd=np.sqrt(np.diag(p.covariance));assert np.max(abs(mean-p.mean)/(sd/np.sqrt(n)))<5
    covariance_se=np.sqrt((np.diag(p.covariance)[:,None]*np.diag(p.covariance)[None,:]+p.covariance**2)/(n-1));assert np.max(abs(cov-p.covariance)/covariance_se)<5
    B=rng.normal(size=(12,4));field=draw[:100]@B.T;np.testing.assert_allclose(field[:,:4],draw[:100]@B[:4].T,rtol=0,atol=8*np.finfo(float).eps*max(1.,abs(field).max()))
    EVIDENCE['posterior_draws']={'count':n,'max_mean_standard_errors':float(np.max(abs(mean-p.mean)/(sd/np.sqrt(n)))),'max_covariance_standard_errors':float(np.max(abs(cov-p.covariance)/covariance_se))}
