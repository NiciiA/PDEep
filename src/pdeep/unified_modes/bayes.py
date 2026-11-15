"""Exact partial-knowledge conditioning; known values are never modified."""
import numpy as np
from scipy.linalg import cholesky,cho_solve
from ..core import require
from ..unified_bayes.gaussian import condition,GaussianPosterior


def conditional(H,y,sigma,known_values,known_mask,*,prior_mean=None,prior_covariance=None,offset=None,valid=None):
    H=np.asarray(H,dtype=np.float64);require(H.ndim==2,'observation matrix must be2D');d=H.shape[1];rawmask=np.asarray(known_mask);require(np.isin(rawmask,[0,1]).all(),'exact knowledge requires a boolean mask, not probabilities');mask=rawmask.astype(bool);values=np.asarray(known_values,dtype=np.float64)
    require(mask.shape==values.shape==(d,) and np.isfinite(values[mask]).all(),'invalid exact coefficient knowledge')
    mu=np.zeros(d) if prior_mean is None else np.asarray(prior_mean,dtype=np.float64);C=np.eye(d) if prior_covariance is None else np.asarray(prior_covariance,dtype=np.float64)
    require(mu.shape==(d,) and np.isfinite(mu).all(),'invalid prior mean')
    require(C.shape==(d,d) and np.allclose(C,C.T,rtol=1e-13,atol=0),'invalid prior covariance');cholesky(C,lower=True)
    K=np.flatnonzero(mask);U=np.flatnonzero(~mask);mean=np.zeros(d);mean[K]=values[K];cov=np.zeros((d,d));b=np.zeros(len(H)) if offset is None else np.asarray(offset,dtype=np.float64)
    yy=np.asarray(y,dtype=np.float64);ss=np.asarray(sigma,dtype=np.float64);valid_mask=np.ones(len(H),bool) if valid is None else np.asarray(valid,bool)
    require(yy.shape==ss.shape==b.shape==valid_mask.shape==(len(H),),'observation dimensions disagree')
    require(np.isfinite(H[valid_mask]).all() and np.isfinite(yy[valid_mask]).all() and np.isfinite(b[valid_mask]).all() and np.isfinite(ss[valid_mask]).all() and (ss[valid_mask]>0).all(),'valid measurements require finite values and strictly positive sigma')
    if len(U):
        if len(K):
            L=cholesky(C[np.ix_(K,K)],lower=True);gain=cho_solve((L,True),C[np.ix_(K,U)]).T
            cm=mu[U]+gain@(values[K]-mu[K]);cc=C[np.ix_(U,U)]-gain@C[np.ix_(K,U)];cc=(cc+cc.T)/2
        else:cm=mu.copy();cc=C.copy()
        p=condition(H[:,U],y,sigma,prior_mean=cm,prior_covariance=cc,offset=b+H[:,K]@values[K],valid=valid)
        mean[U]=p.mean;cov[np.ix_(U,U)]=p.covariance;factor=np.zeros((d,p.factor.shape[1]));factor[U]=p.factor;diagnostics=p.diagnostics
    else:
        # Validate actual measurements even though exact knowledge fixes the latent field.
        sig=np.asarray(sigma);v=np.ones(len(H),bool) if valid is None else np.asarray(valid,bool)
        require(np.isfinite(sig[v]).all() and (sig[v]>0).all(),'measurement sigma must be positive; noisy knowledge is a separate likelihood')
        factor=np.zeros((d,0));diagnostics={'route':'deterministic fully specified forward propagation','observations':int(v.sum())}
    return GaussianPosterior(mean,cov,factor,{**diagnostics,'known':len(K),'unknown':len(U),'exact_known':True,'model':'conditional public Gaussian prior; no artificial forward uncertainty'})
