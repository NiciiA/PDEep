"""Separate classical T1 controls. These functions never update a neural model.

P is the physical coefficient posterior precision; no explicit matrix inverse.
Jacobi scaling is the positive vector 1/diag(P), NOT the inverse of P.
"""
from dataclasses import dataclass
import numpy as np
from .core import require

BUDGETS=(3,10,30,100,300)

@dataclass(frozen=True)
class Quadratic:
    H: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    P: np.ndarray
    b: np.ndarray

    @classmethod
    def build(cls,H,y,sigma,mean,scale):
        arrays=tuple(np.asarray(a,dtype=np.float64) for a in (H,y,sigma,mean,scale))
        H,y,sigma,mean,scale=arrays;d=len(mean)
        require(H.shape==(len(y),d) and sigma.shape==y.shape and scale.shape==mean.shape,'quadratic dimensions mismatch')
        require(all(np.isfinite(a).all() for a in arrays) and np.all(sigma>0) and np.all(scale>0),'finite inputs and positive noise/prior scales required')
        whitened=H/sigma[:,None]
        P=whitened.T@whitened+np.diag(1/scale**2)
        b=whitened.T@(y/sigma)+mean/scale**2
        return cls(H,y,sigma,mean,scale,P,b)

    def phi(self,x):
        r=(self.H@x-self.y)/self.sigma;z=(x-self.mean)/self.scale
        return float(.5*(r@r+z@z))

    def independent_gradient(self,x):
        return self.H.T@((self.H@x-self.y)/self.sigma**2)+(x-self.mean)/self.scale**2


def work_counts():
    return {'objective_evaluations':0,'gradient_evaluations':0,'precision_matvecs':0,
            'sensor_matvecs':0,'precision_builds':1,'rhs_builds':1,'eigensolves':0,
            'true_residual_recomputations':0,'prior_cholesky':0,'posterior_cholesky':0,
            'posterior_covariance_formations':0,'covariance_triangular_solves':0,'SVDs':0}


def gradient_descent(q,initial,steps,*,scaling='identity',trace=False):
    require(isinstance(steps,int) and steps>=0,'nonnegative integer iteration budget required')
    require(scaling in {'identity','jacobi','prior'},'unknown positive diagonal scaling')
    x=np.array(initial,dtype=np.float64,copy=True);require(x.shape==q.mean.shape,'initial dimensions mismatch')
    diagonal=np.ones_like(x) if scaling=='identity' else 1/np.diag(q.P) if scaling=='jacobi' else q.scale**2
    require(np.all(diagonal>0),'positive diagonal scaling required')
    root=np.sqrt(diagonal);L=float(np.linalg.eigvalsh(root[:,None]*q.P*root[None])[-1]);eta=1/L
    counts=work_counts();counts['eigensolves']=1
    history=[x.copy()] if trace else []
    for _ in range(steps):
        gradient=q.P@x-q.b;counts['gradient_evaluations']+=1;counts['precision_matvecs']+=1
        x-=eta*diagonal*gradient
        if trace:history.append(x.copy())
    return {'xi':x,'iterations':steps,'counts':counts,'eta':eta,'diagonal':diagonal,
            'scaling':scaling,'trajectory':history,'converged':None}


def conjugate_gradient(q,initial,*,rtol=1e-12,atol=1e-12,maxiter=None,trace=False):
    """CG on SPD P, verifying true residual before accepting convergence.

    Recurrence residual is not trusted at termination. A failed true-residual
    test restarts CG from that residual. All these matvecs are counted/timed.
    Independent sensor-space gradient checks belong to the external verifier.
    """
    require(rtol>0 and atol>=0,'invalid CG tolerances');maxiter=maxiter or 5*len(q.b)
    x=np.array(initial,dtype=np.float64,copy=True);counts=work_counts()
    require(x.shape==q.mean.shape,'CG initial dimensions mismatch')
    r=q.b-q.P@x;counts['precision_matvecs']=1;counts['gradient_evaluations']=1
    threshold=max(atol,rtol*float(np.linalg.norm(q.b)));rr=float(r@r)
    history=[x.copy()] if trace else [];direction=r.copy();iterations=0;converged=np.sqrt(rr)<=threshold
    while iterations<maxiter and not converged:
        Ap=q.P@direction;counts['precision_matvecs']+=1;den=float(direction@Ap)
        require(np.isfinite(den) and den>0,'CG encountered nonpositive curvature; no fallback or jitter')
        alpha=rr/den;x+=alpha*direction;r-=alpha*Ap;iterations+=1
        if trace:history.append(x.copy())
        new_rr=float(r@r)
        if np.sqrt(new_rr)<=threshold or iterations==maxiter:
            true_r=q.b-q.P@x;counts['precision_matvecs']+=1;counts['gradient_evaluations']+=1;counts['true_residual_recomputations']+=1
            new_rr=float(true_r@true_r)
            if np.sqrt(new_rr)<=threshold:converged=True;r=true_r;rr=new_rr;break
            r=true_r;direction=r.copy();rr=new_rr;continue
        direction=r+(new_rr/rr)*direction;rr=new_rr
    return {'xi':x,'iterations':iterations,'counts':counts,'converged':bool(converged),
            'residual_norm':float(np.sqrt(rr)),'residual_threshold':threshold,'rtol':rtol,'atol':atol,'trajectory':history}
