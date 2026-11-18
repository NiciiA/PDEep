"""Exact block application of the unchanged full FEM CN/Gauss4 evolution.

Columns are independent physical cases. No trajectory compression, spectral
cutoff, different quadrature/dt, temporal approximation or cross-case averaging.
Only initial-basis evaluation and the same sparse resolvent are reused.
"""
import time
import numpy as np
from scipy.sparse.linalg import splu
from ..core import require,digest
from ..smooth.definition import POLICY,COEFFICIENTS
from .numerics import FastReference,Sampler

class Evolution:
    def __init__(self,reference,pde):
        require(pde in ('heat','wave'),'dynamic evolution only')
        self.reference=reference;self.pde=pde;self.h=1/int(np.ceil(1/POLICY['reference_dt']));s=reference.space
        self.K=COEFFICIENTS[pde]*s.Ki;key=(pde,self.h)
        if pde=='heat':self.A=(s.Mi+self.h*self.K/2).tocsc();self.B=s.Mi-self.h*self.K/2
        else:
            self.gamma=(1+1j*np.sqrt(3))/24;self.beta=.25+1j/(4*np.sqrt(3));self.A=(s.Mi+self.gamma*self.h**2*self.K).tocsc()
        start=time.perf_counter();cpu=time.process_time()
        if key not in reference.factors:reference.factors[key]=splu(self.A)
        self.factorization_timing={'wall_seconds':time.perf_counter()-start,'CPU_seconds':time.process_time()-cpu}
        self.factor=reference.factors[key];self.initial_bases={}
    def initial(self,fields):
        f=fields[0];key=digest((f.centers.tolist(),f.radius,f.geometry_id,f.amplitude))
        require(all(digest((g.centers.tolist(),g.radius,g.geometry_id,g.amplitude))==key for g in fields),'initial basis differs across block')
        if key not in self.initial_bases:
            x=self.reference.space.position;B=np.maximum(0.,1-((x[:,None]-f.centers[None])**2).sum(-1)/f.radius**2)**4/np.sqrt(len(f.centers))*f.amplitude
            require(np.max(abs(B[self.reference.space.dirichlet]),initial=0.)<1e-10,'initial unit basis violates boundary')
            self.initial_bases[key]=B[self.reference.space.free]
        return np.asfortranarray(self.initial_bases[key]@np.column_stack([g.coefficients for g in fields]))
    def propagate(self,cases,*,progress=None,memory_bytes=2_000_000_000):
        require(cases and all(c.pde==self.pde and c.family==self.reference.family for c in cases),'incompatible propagation block')
        require(len({c.case_id for c in cases})==len(cases),'duplicate physical case in propagation block')
        start=time.perf_counter();cpu=time.process_time();s=self.reference.space;h=self.h;n=round(1/h);B=len(cases)
        state_bytes=2*B*(n+1)*len(s.free)*8
        require(state_bytes<=memory_bytes,'declared block trajectory memory budget exceeded; use fewer columns, never coarsen the reference')
        u=self.initial([c.fields['u0'] for c in cases]);v=self.initial([c.fields['v0'] for c in cases]) if self.pde=='wave' else None
        U=np.empty((B,n+1,len(s.free)),np.float64);V=np.empty_like(U);U[:,0]=u.T
        V[:,0]=(v if v is not None else -s.factor.solve(self.K@u)).T
        for k in range(n):
            if v is None:u=self.factor.solve(self.B@u);dv=-s.factor.solve(self.K@u)
            else:
                z=self.factor.solve(self.beta*h*(s.Mi@v)-self.gamma*h*h*(self.K@u))
                u=u+4*np.sqrt(3)*z.imag;v=v+12/h*(-z.real+np.sqrt(3)*z.imag);dv=v
            U[:,k+1]=u.T;V[:,k+1]=dv.T
            if progress and (k+1)%100==0:progress('Reference block propagation',pde=self.pde,geometry=self.reference.family,physical_cases=B,numerical_steps=k+1,total_steps=n,trajectory_bytes=state_bytes)
        require(np.isfinite(U).all() and np.isfinite(V).all(),'nonfinite block states')
        wall=time.perf_counter()-start;cpu=time.process_time()-cpu;result=[]
        for i,c in enumerate(cases):
            f=object.__new__(FastReference);f.case=c;f.reference=self.reference;f.dt=h;f.u=U[i];f.v=V[i];f.sampler=Sampler(s);f.solves=n;f.evaluations=0
            f.identity=digest(('fixed-reference-cache-P4-C1-v1',c.manifest,s.identity,POLICY));f.bytes=f.u.nbytes+f.v.nbytes;f.preparation_seconds=wall/B
            f.block_timing={'cases':B,'wall_seconds':wall,'CPU_seconds':cpu,'attributed_wall_seconds':wall/B,'attributed_CPU_seconds':cpu/B,'trajectory_bytes':state_bytes,'method':'unchanged full CN/Gauss4; independent multiple-RHS columns'}
            result.append(f)
        return result
