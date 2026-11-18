"""Exact P4-value evaluation and linear Poisson reuse, CPU float64 only.

No mesh, quadrature, time step, field prior, eigen cutoff or interpolation law
changes. The original locator chooses the same element and barycentric values.
Only unused basis gradients and temporary sparse sampling matrices are removed.
"""
import time
import numpy as np
from scipy.sparse.linalg import splu
from ..core import digest,require
from ..learning_diagnostics.reference_cache import FixedReference
from ..wave_fidelity.reference import indices
from ..wave_fidelity.fields import SmoothField
from ..smooth.definition import POLICY

def values(bary,degree):
    result=[]
    for alpha in indices(degree):
        factors=[(degree*bary[:,i]-k)/(a-k) for i,a in enumerate(alpha) for k in range(a)]
        result.append(np.prod(factors,axis=0))
    return np.stack(result,axis=1)

class Sampler:
    def __init__(self,space):
        self.space=space;self.free=np.full(len(space.position),-1,dtype=np.int64);self.free[space.free]=np.arange(len(space.free))
    def evaluate(self,full,x,t,q,*,chunk=32768):
        x=np.asarray(x,dtype=np.float64);t=np.asarray(t,dtype=np.float64);q=np.zeros(len(x),int) if q is None else np.asarray(q)
        require(x.shape==(len(t),2) and q.shape==t.shape and np.isfinite(x).all() and np.isfinite(t).all() and ((0<=t)&(t<=1)).all(),'invalid continuous reference queries')
        require(np.isin(q,[0,1] if full.case.pde=='wave' else [0]).all(),'unsupported quantity')
        out=np.empty(len(x),np.float64)
        for a in range(0,len(x),chunk):
            sl=slice(a,min(a+chunk,len(x)));cell,bary=self.space.mesh.locate(x[sl]);w=values(bary,self.space.degree)
            dofs=self.free[self.space.dofs[cell]];w=w*(dofs>=0);dofs=np.maximum(dofs,0)
            if full.v is None:out[sl]=np.sum(w*full.u[0,dofs],axis=1);continue
            k=np.minimum(np.floor(t[sl]/full.dt).astype(int),len(full.u)-2);z=(t[sl]-k*full.dt)/full.dt
            paired=lambda states,step:np.sum(w*states[(k+step)[:,None],dofs],axis=1)
            ul,ur,vl,vr=paired(full.u,0),paired(full.u,1),paired(full.v,0),paired(full.v,1)
            u=(2*z**3-3*z*z+1)*ul+(z**3-2*z*z+z)*full.dt*vl+(-2*z**3+3*z*z)*ur+(z**3-z*z)*full.dt*vr
            v=(6*z*z-6*z)/full.dt*ul+(3*z*z-4*z+1)*vl+(-6*z*z+6*z)/full.dt*ur+(3*z*z-2*z)*vr
            out[sl]=np.where(q[sl]==1,v,u)
        require(np.isfinite(out).all(),'nonfinite continuous reference labels');return out

class PoissonBasis:
    """32 FULL FE source responses. Superposition commutes with the fixed solve.

    Independent per-case QC still assembles the original physical load and checks
    K*u-F. There is no reduced basis or approximation to the physical source law.
    """
    def __init__(self,reference,field):
        start=time.perf_counter();self.reference=reference;self.signature=(field.centers.copy(),field.radius,field.geometry_id,field.amplitude)
        unit=[SmoothField(field.centers,field.radius,np.eye(32)[j],field.geometry_id,field.amplitude) for j in range(32)]
        F=reference.load(unit,POLICY['source_quadrature']);s=reference.space
        if 'poisson' not in reference.factors:reference.factors['poisson']=splu(s.Ki)
        self.response=reference.factors['poisson'].solve(F)
        residual=np.linalg.norm(s.Ki@self.response-F,axis=0)/(1+np.linalg.norm(F,axis=0));require(residual.max()<1e-10,'unit source response residual failed')
        self.seconds=time.perf_counter()-start;self.bytes=self.response.nbytes
    def nodal(self,field):
        centers,radius,gid,amp=self.signature
        require(np.array_equal(field.centers,centers) and (field.radius,field.geometry_id,field.amplitude)==(radius,gid,amp),'source basis changed')
        return self.response@field.coefficients

class FastReference(FixedReference):
    def __init__(self,case,*,reference_object,poisson_basis=None):
        if poisson_basis is None:super().__init__(case,reference_object=reference_object)
        else:
            start=time.perf_counter();require(case.pde=='poisson','Poisson response reuse only')
            self.case=case;self.reference=reference_object;self.dt=1/int(np.ceil(1/POLICY['reference_dt']));self.u=poisson_basis.nodal(case.fields['f'])[None];self.v=None
            self.identity=digest(('fixed-reference-cache-P4-C1-v1',case.manifest,reference_object.space.identity,POLICY));self.solves=0;self.evaluations=0
            self.preparation_seconds=time.perf_counter()-start;self.bytes=self.u.nbytes
        self.sampler=Sampler(reference_object.space)
    def evaluate(self,x,t,q=None,*,chunk=32768):
        self.evaluations+=1;return self.sampler.evaluate(self,x,t,q,chunk=chunk)
