"""CPU64 linear-Gaussian conditioning with physical offsets and no explicit inverse."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular
from ..core import require


def _array(x):return np.asarray(x,dtype=np.float64)

def _symmetric(x,name):
    x=_array(x);require(x.ndim==2 and x.shape[0]==x.shape[1] and np.isfinite(x).all(),name+' must be finite square')
    scale=max(float(np.max(abs(x),initial=0.)),np.finfo(float).tiny)
    require(np.max(abs(x-x.T),initial=0.)<=256*np.finfo(float).eps*scale*max(1,len(x)),name+' is materially asymmetric')
    return (x+x.T)/2


@dataclass
class GaussianPosterior:
    mean: np.ndarray
    covariance: np.ndarray
    factor: np.ndarray
    diagnostics: dict
    def predict(self,B,*,offset=None,noise_sigma=None,full_covariance=False,max_joint=256,chunk=2048):
        B=_array(B);require(B.ndim==2 and B.shape[1]==len(self.mean) and np.isfinite(B).all(),'invalid query operator')
        b=np.zeros(len(B)) if offset is None else _array(offset);require(b.shape==(len(B),),'query offset shape mismatch')
        means=[];variances=[]
        for a in range(0,len(B),chunk):
            block=B[a:a+chunk];F=block@self.factor;means.append(b[a:a+chunk]+block@self.mean);variances.append(np.einsum('ij,ij->i',F,F))
        out={'mean':np.concatenate(means) if means else np.empty(0),'latent_variance':np.concatenate(variances) if variances else np.empty(0)}
        if noise_sigma is not None:
            sigma=_array(noise_sigma);require(sigma.shape==(len(B),) and np.isfinite(sigma).all() and (sigma>0).all(),'future measurement sigma must be strictly positive')
            out['measurement_variance']=out['latent_variance']+sigma*sigma
        if full_covariance:
            require(len(B)<=max_joint,'full query covariance is limited; use marginal variances/chunks')
            F=B@self.factor;out['latent_covariance']=F@F.T
            if noise_sigma is not None:out['measurement_covariance']=out['latent_covariance']+np.diag(sigma*sigma)
        return out
    def cross_covariance(self,B_left,B_right,*,max_joint=256):
        A,C=_array(B_left),_array(B_right);require(len(A)<=max_joint and len(C)<=max_joint,'cross covariance exceeds explicit small-query limit')
        return (A@self.factor)@(C@self.factor).T
    def draw(self,count,*,seed=42):
        require(type(count) is int and count>0,'positive posterior draw count required')
        return self.mean[None]+np.random.default_rng(seed).standard_normal((count,self.factor.shape[1]))@self.factor.T


def condition(H,y,sigma,*,prior_mean=None,prior_covariance=None,offset=None,valid=None,form='coefficient'):
    """Public prior parameters only. Never accepts a truth/source/initial-field argument.

    Default: prior-whitened precision Cholesky; augmented QR is a stable square-root
    fallback for extreme conditioning. Independent observation-space conditioning
    uses Cholesky and Joseph covariance. NO jitter, noise substitution, or inverse.
    Exactly noiseless observations are explicitly unsupported by this route.
    """
    H,y,sigma=_array(H),_array(y),_array(sigma)
    require(H.ndim==2 and y.shape==sigma.shape==(len(H),),'observation dimensions disagree')
    mask=np.ones(len(H),bool) if valid is None else np.asarray(valid,dtype=bool);require(mask.shape==(len(H),),'valid mask shape mismatch')
    b=np.zeros(len(H)) if offset is None else _array(offset);require(b.shape==(len(H),),'observation offset shape mismatch')
    H,y,sigma,b=H[mask],y[mask],sigma[mask],b[mask]
    require(np.isfinite(H).all() and np.isfinite(y).all() and np.isfinite(b).all() and np.isfinite(sigma).all() and (sigma>0).all(),
        'valid measurements need finite values and strictly positive sigma; exact noiseless conditioning needs a separate constrained route')
    d=H.shape[1];mu=np.zeros(d) if prior_mean is None else _array(prior_mean);C=np.eye(d) if prior_covariance is None else _symmetric(prior_covariance,'prior covariance')
    require(mu.shape==(d,) and C.shape==(d,d) and np.isfinite(mu).all(),'invalid prior')
    L0=cholesky(C,lower=True,check_finite=True);require(form in ('coefficient','observation'),'unknown conditioning form')
    if not len(H):return GaussianPosterior(mu.copy(),C.copy(),L0,{'observations':0,'route':'prior only','rank':0,'condition':1.,'precision_relative_residual':0.})
    W=(H@L0)/sigma[:,None];r=(y-b-H@mu)/sigma;P=np.eye(d)+W.T@W;rhs=W.T@r
    singular=np.linalg.svd(W,compute_uv=False);cond=float(np.linalg.cond(P))
    if form=='coefficient':
        try:
            if cond>1e12:raise np.linalg.LinAlgError('precision requires square-root QR')
            L=cholesky(P,lower=True);z=cho_solve((L,True),rhs);F=L0@solve_triangular(L.T,np.eye(d),lower=False);route='whitened precision Cholesky'
        except np.linalg.LinAlgError:
            Q,R=np.linalg.qr(np.vstack((W,np.eye(d))),mode='reduced');z=solve_triangular(R,Q.T@np.r_[r,np.zeros(d)])
            F=L0@solve_triangular(R,np.eye(d));route='augmented square-root QR (no noise/diagonal change)'
    else:
        L=cholesky(np.eye(len(H))+W@W.T,lower=True);gain=cho_solve((L,True),W).T;z=gain@r
        A=np.eye(d)-gain@W
        # Joseph form is PSD even when subtractive C-KHC loses precision.
        F=L0@np.column_stack((A,gain));route='observation Cholesky + Joseph covariance'
    covariance=F@F.T;mean=mu+L0@z
    require(np.isfinite(mean).all() and np.isfinite(covariance).all(),'nonfinite posterior')
    residual=float(np.linalg.norm(P@z-rhs)/(1+np.linalg.norm(rhs)))
    return GaussianPosterior(mean,(covariance+covariance.T)/2,F,{'observations':len(H),'route':route,'rank':int(np.linalg.matrix_rank(W)),
        'singular_values':singular.tolist(),'condition':cond,'precision_relative_residual':residual,'prior':'explicit Gaussian; no fitted test moments',
        'minimum_covariance_eigenvalue':float(np.linalg.eigvalsh(covariance).min())})
