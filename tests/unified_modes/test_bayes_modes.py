import numpy as np
from scipy.linalg import cho_solve,cho_factor
from pdeep.unified_bayes.gaussian import condition
from pdeep.unified_modes.bayes import conditional
from .conftest import EVIDENCE


def test_correlated_conditional_joint_independent_route():
    rng=np.random.default_rng(17);d=7;A=rng.normal(size=(d,d));C=A@A.T+.3*np.eye(d);mu=rng.normal(size=d);H=rng.normal(size=(11,d));b=rng.normal(size=11);y=rng.normal(size=11);sigma=rng.uniform(.1,.5,11);known=np.array([1,0,0,1,0,0,1],bool);values=rng.normal(size=d)
    p=conditional(H,y,sigma,values,known,prior_mean=mu,prior_covariance=C,offset=b)
    allpost=condition(H,y,sigma,prior_mean=mu,prior_covariance=C,offset=b);K=np.flatnonzero(known);gain=cho_solve(cho_factor(allpost.covariance[np.ix_(K,K)],lower=True),allpost.covariance[K]).T
    mean=allpost.mean+gain@(values[K]-allpost.mean[K]);cov=allpost.covariance-gain@allpost.covariance[K]
    em=float(abs(mean-p.mean).max());ec=float(abs(cov-p.covariance).max());assert em<1e-12 and ec<1e-12
    np.testing.assert_array_equal(p.mean[K],values[K]);assert not p.covariance[K].any()
    inv=conditional(H,y,sigma,np.full(d,np.nan),np.zeros(d,bool),prior_mean=mu,prior_covariance=C,offset=b);np.testing.assert_allclose(inv.mean,allpost.mean,atol=0)
    forward=conditional(np.empty((0,d)),[],[],values,np.ones(d,bool),prior_mean=mu,prior_covariance=C);B=rng.normal(size=(10,d));out=forward.predict(B);np.testing.assert_allclose(out['mean'],B@values,atol=0);assert not out['latent_variance'].any() and not forward.covariance.any();np.testing.assert_array_equal(forward.draw(5),np.tile(values,(5,1)))
    EVIDENCE['conditional_Gaussian']={'mean_discrepancy':em,'covariance_discrepancy':ec,'known_values_exact':True,'fully_known_variance':float(out['latent_variance'].max())}


def test_invalid_exact_knowledge_and_redundant_measurements():
    import pytest
    H=np.eye(2)
    with pytest.raises(ValueError,match='boolean'):conditional(H,[1,2],[.1,.2],[1,2],[.5,1])
    with pytest.raises(ValueError,match='finite'):conditional(H,[np.nan,2],[.1,.2],[1,2],[1,1])
    p=conditional(np.vstack((H,[np.nan,np.nan])),[1,2,np.nan],[.1,.2,0],[1,2],[1,1],valid=[1,1,0]);assert not p.covariance.any()
