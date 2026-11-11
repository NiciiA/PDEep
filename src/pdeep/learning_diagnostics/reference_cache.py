"""CPU64 cached coefficient trajectories for the EXISTING certified P4 policy.

This is a derived execution cache, not a different solver/certificate. Construction
uses the same full operators and rational/CN steps; evaluation uses the SAME P4
sample_operator and C1 Hermite polynomial (wave v = its derivative). No modes
are truncated and no states are selected by nearest time. Independent regression
against Reference.poisson/dynamic is required; original sources remain untouched.
"""
import time
import numpy as np
from scipy.sparse.linalg import splu
from ..core import digest,require
from ..smooth.definition import POLICY,COEFFICIENTS
from ..smooth.reference import reference

class FixedReference:
    def __init__(self,case,*,reference_object=None):
        start=time.perf_counter();self.case=case
        self.reference=reference_object or reference(case.family,POLICY['reference_h'],POLICY['reference_degree'])
        r=self.reference;s=r.space;f=case.fields
        self.identity=digest(('fixed-reference-cache-P4-C1-v1',case.manifest,s.identity,POLICY))
        self.solves=0;self.evaluations=0;self.dt=1/int(np.ceil(1/POLICY['reference_dt']))
        if case.pde=='poisson':
            F=r.load([f['f']],POLICY['source_quadrature'])
            if 'poisson' not in r.factors:r.factors['poisson']=splu(s.Ki)
            u=r.factors['poisson'].solve(F);self.solves+=1
            require(np.linalg.norm(s.Ki@u-F)/(1+np.linalg.norm(F))<1e-10,'cached Poisson residual failed')
            self.u=u[:,0][None];self.v=None
        else:
            h=self.dt;n=round(1/h);u=r.initial([f['u0']]);v=r.initial([f['v0']]) if case.pde=='wave' else None
            self.u=np.empty((n+1,len(u)),dtype=np.float64);self.v=np.empty_like(self.u)
            K=COEFFICIENTS[case.pde]*s.Ki;key=(case.pde,h)
            if v is None:A=(s.Mi+h*K/2).tocsc();B=s.Mi-h*K/2
            else:gamma=(1+1j*np.sqrt(3))/24;beta=.25+1j/(4*np.sqrt(3));A=(s.Mi+gamma*h*h*K).tocsc()
            if key not in r.factors:r.factors[key]=splu(A)
            factor=r.factors[key];self.u[0]=u[:,0]
            self.v[0]=(v if v is not None else -s.factor.solve(K@u))[:,0]
            for k in range(n):
                if v is None:u=factor.solve(B@u);dv=-s.factor.solve(K@u)
                else:
                    z=factor.solve(beta*h*(s.Mi@v)-gamma*h*h*(K@u))
                    u=u+4*np.sqrt(3)*z.imag;v=v+12/h*(-z.real+np.sqrt(3)*z.imag);dv=v
                self.solves+=1;self.u[k+1]=u[:,0];self.v[k+1]=dv[:,0]
            require(np.isfinite(self.u).all() and np.isfinite(self.v).all(),'cached trajectory nonfinite')
        self.preparation_seconds=time.perf_counter()-start
        self.bytes=self.u.nbytes+(0 if self.v is None else self.v.nbytes)
    def evaluate(self,x,t,q=None,*,chunk=4096):
        self.evaluations+=1;x=np.asarray(x,dtype=np.float64);t=np.asarray(t,dtype=np.float64)
        require(t.shape==(len(x),) and np.isfinite(x).all() and np.isfinite(t).all() and ((t>=0)&(t<=1)).all(),'invalid cached reference queries')
        q=np.zeros(len(x),dtype=int) if q is None else np.asarray(q)
        require(q.shape==t.shape and np.isin(q,[0,1] if self.case.pde=='wave' else [0]).all(),'unsupported reference quantity')
        out=np.empty(len(x));s=self.reference.space
        for a in range(0,len(x),chunk):
            sl=slice(a,min(a+chunk,len(x)));S=s.sample_operator(x[sl])
            if self.v is None:out[sl]=S@self.u[0];continue
            # Evaluate sparse P4 rows against their OWN two adjacent full states.
            # Storage covers all500 intervals, not an approximate visualization grid.
            k=np.minimum(np.floor(t[sl]/self.dt).astype(int),len(self.u)-2)
            z=(t[sl]-k*self.dt)/self.dt
            def paired(A,step):return np.bincount(np.repeat(np.arange(S.shape[0]),np.diff(S.indptr)),weights=S.data*A[(k+step).repeat(np.diff(S.indptr)),S.indices],minlength=S.shape[0])
            ul,ur,vl,vr=paired(self.u,0),paired(self.u,1),paired(self.v,0),paired(self.v,1)
            value=(2*z**3-3*z*z+1)*ul+(z**3-2*z*z+z)*self.dt*vl+(-2*z**3+3*z*z)*ur+(z**3-z*z)*self.dt*vr
            velocity=((6*z*z-6*z)/self.dt*ul+(3*z*z-4*z+1)*vl+(-6*z*z+6*z)/self.dt*ur+(3*z*z-2*z)*vr)
            out[sl]=np.where(q[sl]==1,velocity,value)
        require(np.isfinite(out).all(),'nonfinite cached reference evaluation');return out
