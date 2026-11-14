"""Shared, verified linear-response cache of the FULL certified P4 propagation.

No reduced PDE solve, Fourier/eigen cutoff or interpolation of plotting frames.
Offline each unit C3 basis response is propagated by FixedReference. Optional
float64 spatial factorization is checked entrywise against EVERY full nodal state.
An explicit conservative P4/Hermite bound propagates compression error to arbitrary
positions/times and actual Gaussian case coefficients. Cases additionally require
independent full-solver QC. Dataset preparation, never the trainer, creates assets.
"""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import math, time
from functools import lru_cache
from fractions import Fraction
import numpy as np
import torch
from scipy.sparse import coo_matrix
from ..core import digest, require
from ..artifact_io import read_manifest, read_payload, write_artifact
from ..artifact_tensors import tree_identity
from ..smooth.definition import POLICY, geometry, physics_identity
from ..smooth.reference import reference
from ..wave_fidelity.reference import Space, indices
from ..wave_fidelity.fields import SmoothField
from ..learning_diagnostics.reference_cache import FixedReference

CACHE_SCHEMA = 'G1-unified-full-P4-linear-responses-v2'


@lru_cache(maxsize=4)
def lagrange_l1_bound(degree):
    """Exact rational Bernstein-convex-hull bound, then round outward.

    In Bernstein coordinates, nonnegative basis functions sum to one. Lagrange
    basis L=B*T; max row sum |T| bounds sum_i |L_i(x)| at EVERY triangle point.
    Exact rational elimination gives P4 bound202/9 (no sampled-max assumption).
    """
    alpha=indices(degree);n=len(alpha);A=[]
    for node in alpha:
        x=[Fraction(v,degree) for v in node];row=[]
        for exponent in alpha:
            c=Fraction(math.factorial(degree),math.prod(math.factorial(v) for v in exponent))
            row.append(c*math.prod(v**k for v,k in zip(x,exponent)))
        A.append(row+[Fraction(int(len(A)==j)) for j in range(n)])
    for j in range(n):
        pivot=next(k for k in range(j,n) if A[k][j]);A[j],A[pivot]=A[pivot],A[j]
        factor=A[j][j];A[j]=[v/factor for v in A[j]]
        for k in range(n):
            if k!=j:
                factor=A[k][j];A[k]=[u-factor*v for u,v in zip(A[k],A[j])]
    bound=max(sum(abs(v) for v in row[n:]) for row in A)
    return float(bound)*(1+1e-12)


def output_bounds(eu, ev, dt, degree=4, stationary=False):
    L = lagrange_l1_bound(degree)
    return {'u': L*(eu if stationary else eu+.3*dt*ev),
            'v': 0. if stationary else L*(3*eu/dt+2*ev)}


def _residual_report(A,Q,B,n,dt,stationary,degree):
    """All nodal states AND continuous-time Bernstein control coefficients.

    Cubic displacement controls: ul,ul+h*vl/3,ur-h*vr/3,ur.
    Its derivative's quadratic controls: vl,3*(ur-ul)/h-vl-vr,vr.
    Compute signed residuals BEFORE forming differences: correlated temporal
    projection errors need no independent 1/h worst-case amplification assumption.
    """
    eu=ev=cu=cv=0.
    for a in range(0,len(A),2048):
        d=A[a:a+2048]-Q[a:a+2048]@B;u=d[:,:n]
        eu=max(eu,float(abs(u).max(initial=0.)));cu=max(cu,eu)
        if not stationary:
            v=d[:,n:]/dt;ev=max(ev,float(abs(v).max(initial=0.)))
            cu=max(cu,float(abs(u[:,:-1]+dt*v[:,:-1]/3).max(initial=0.)),float(abs(u[:,1:]-dt*v[:,1:]/3).max(initial=0.)))
            cv=max(cv,ev,float(abs(3*(u[:,1:]-u[:,:-1])/dt-v[:,:-1]-v[:,1:]).max(initial=0.)))
    L=lagrange_l1_bound(degree)
    return {'max_nodal_u_error':eu,'max_nodal_v_error':ev,'max_time_Bernstein_u_error':cu,'max_time_Bernstein_v_error':cv,
            'spatial_L1_bound':L,'error_bound':{'u':L*cu,'v':L*cv},
            'bound_scope':'all original nodal snapshots, all fixed integration intervals and all triangle points; algebraic compression, not continuum error'}


def compress_states(u, v, dt, *, tolerance=1e-12, block=32, seed=7193,output_tolerances=None,degree=4):
    """Deterministic range finder, with measured FULL response error bounds.

    Production uses quantity-specific continuous-output tolerances derived from
    frozen TRAIN RMS. Strict scaled-state tolerance remains a small-unit control.
    Fallback is lossless; a case cannot pass merely because compression was tried.
    """
    u=np.asarray(u,dtype=np.float64);n=len(u);stationary=v is None
    A=u.T.copy() if stationary else np.concatenate((u.T,dt*np.asarray(v).T),axis=1)
    Q=np.empty((len(A),0));rng=np.random.default_rng(seed);limit=min(A.shape)
    while Q.shape[1] < limit:
        take=min(block,limit-Q.shape[1]);Y=A@rng.standard_normal((A.shape[1],take))
        if Q.shape[1]:
            for _ in range(2):Y-=Q@(Q.T@Y)
        new=np.linalg.qr(Y,mode='reduced')[0];Q=np.column_stack((Q,new));B=Q.T@A
        report=_residual_report(A,Q,B,n,dt,stationary,degree)
        passed=all(report['error_bound'][k]<=value for k,value in output_tolerances.items()) if output_tolerances is not None else max(report['max_nodal_u_error'],dt*report['max_nodal_v_error'])<=tolerance
        if passed:return Q,B,{**report,'rank':Q.shape[1],'snapshot_columns':A.shape[1],'lossless_fallback':False,'output_tolerances':output_tolerances}
    # Rare poorly compressible responses retain all states; never sacrifice fidelity.
    ev=0. if stationary else float(np.max(abs(A[:,n:]/dt-np.asarray(v).T)))
    return A,np.eye(A.shape[1]),{'rank':A.shape[1],'snapshot_columns':A.shape[1],'max_nodal_u_error':0.,'max_nodal_v_error':ev,
        'error_bound':output_bounds(0.,ev,dt,degree=degree,stationary=stationary),'lossless_fallback':True,'output_tolerances':output_tolerances}


def _space_payload(space):
    m=space.mesh
    return {'degree':space.degree,'position':torch.tensor(space.position),'free':torch.tensor(space.free),
            'dofs':torch.tensor(space.dofs),'vertices':torch.tensor(m.vertices),'cells':torch.tensor(m.cells),
            'boundary_edges':torch.tensor(m.boundary_edges),'boundary_tags':torch.tensor(m.boundary_tags),
            'region_tags':torch.tensor(m.region_tags),'provenance':m.provenance,'mesh_id':m.mesh_id}


def _sampling_space(family,payload):
    from ..geometric.domain import GeometryDomain
    m=GeometryDomain(geometry(family).shape, *(payload[k].numpy() for k in ('vertices','cells','boundary_edges','boundary_tags','region_tags')), payload['provenance']).validate()
    require(m.mesh_id==payload['mesh_id'],'reference sampling mesh changed')
    s=object.__new__(Space)  # ONLY sample_operator; no assembly/factorization/solver in training.
    s.mesh=m;s.degree=payload['degree']
    for k in ('position','free','dofs'):setattr(s,k,payload[k].numpy())
    return s


def _mapped_payload(path):
    r=read_manifest(path,kind='diagnostic_bundle')
    p=torch.load(Path(path).parent/next(iter(r['files'])),map_location='cpu',weights_only=True,mmap=True)
    require(digest(tree_identity(p))==r['payload_tree_identity'],'mapped reference tensor identity mismatch')
    return r,p


class LinearCache:
    def __init__(self,root,certificate,*,response_relative_tolerance=1e-6,progress=None):
        self.root=Path(root);self.certificate=certificate;self.tolerance=response_relative_tolerance;self.progress=progress or (lambda *a,**k:None)
        self.loaded={};self.spaces={};self.assets={};self.created=[];self.sampling_assets={}
        self.physics=physics_identity()
        from .provenance import reference_execution_identity
        from ..smooth.certification import normalizers
        self.execution_source=reference_execution_identity();self.norms=normalizers(certificate)
    def asset_key(self,c,role,j):
        f=c.fields[role]
        return digest((CACHE_SCHEMA,c.family,c.pde,role,j,f.centers.tolist(),f.radius,f.amplitude,POLICY,
                       self.physics,self.execution_source,self.certificate['certification_id'],self.tolerance))
    def asset_path(self,c,role,j):return self.root/(self.asset_key(c,role,j)+'.json')
    def prepare(self,c,*,max_new_responses=None):
        """Explicit offline operation. Complete immutable basis assets survive disconnects."""
        prepared=0
        for role,field in c.fields.items():
            for j in range(len(field.coefficients)):
                path=self.asset_path(c,role,j)
                self.progress('Reference preparation',pde=c.pde,geometry=c.family,role=role,basis=f'{j+1}/32',cache='verified reuse' if path.exists() else 'preparing')
                if path.exists():
                    r=read_manifest(path);require(r.get('response_identity')==path.stem,'basis asset identity mismatch');continue
                if max_new_responses is not None and prepared>=max_new_responses:return [str(p) for p in self.created]
                z=np.zeros(32);z[j]=1.
                fields={k:SmoothField(f.centers,f.radius,z.copy() if k==role else np.zeros(32),f.geometry_id,f.amplitude) for k,f in c.fields.items()}
                proxy=SimpleNamespace(family=c.family,pde=c.pde,fields=fields,manifest={'basis_response':path.stem})
                full=FixedReference(proxy)
                limits={k:self.tolerance*self.norms.scale(c.pde,k) for k in (('u','v') if c.pde=='wave' else ('u',))}
                X,T,report=compress_states(full.u,full.v,full.dt,output_tolerances=limits,degree=full.reference.space.degree)
                space_key='space-'+digest((CACHE_SCHEMA,full.reference.space.identity,self.physics,self.execution_source))
                space_path=self.root/(space_key+'.json')
                if not space_path.exists():
                    write_artifact(self.root,space_key,{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY',
                        'run_id':'G1-unified-spaces','scientific_schema':CACHE_SCHEMA,'family':c.family,'pde':'shared'},
                        {'space':_space_payload(full.reference.space)},required=('space',),idempotent=True)
                    self.created.append(space_path)
                self.sampling_assets[c.family]=space_path
                payload={'spatial':torch.from_numpy(X),'temporal':torch.from_numpy(T)}
                meta={'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-unified-responses',
                      'scientific_schema':CACHE_SCHEMA,'response_identity':path.stem,'family':c.family,'pde':c.pde,'role':role,'basis':j,
                      'certificate_id':self.certificate['certification_id'],'physics':self.physics,'execution_source':self.execution_source,'reference_policy':POLICY,
                      'compression':report,'sampling_space':space_path.name,'dt':full.dt,'snapshots':len(full.u),'preparation_seconds':full.preparation_seconds,
                      'full_snapshot_bytes':full.bytes,'stored_tensor_bytes':X.nbytes+T.nbytes,
                      'scope':'derived execution cache; not a new continuum certificate; case QC required'}
                write_artifact(self.root,path.stem,meta,payload,required=tuple(payload),idempotent=True);self.created.append(path)
                prepared+=1
                del full,X,T,payload
        return [str(self.asset_path(c,k,j)) for k in c.fields for j in range(32)]
    def _get(self,c,role,j):
        path=self.asset_path(c,role,j);key=path.stem
        require(path.is_file(),'Reference asset missing: prepare dataset before training: '+str(path))
        if key not in self.loaded:
            r,p=_mapped_payload(path)
            require(r.get('response_identity')==key and r.get('physics')==self.physics and r.get('execution_source')==self.execution_source,'reference source/policy mismatch')
            require(r['reference_policy']==POLICY,'reference policy changed')
            if c.family not in self.spaces:
                space_path=self.root/r['sampling_space'];_,sp=_mapped_payload(space_path)
                self.spaces[c.family]=_sampling_space(c.family,sp['space']);self.sampling_assets[c.family]=space_path
            self.loaded[key]=(r,p);self.assets[key]=path
        return self.loaded[key]
    def sensitivities(self,c,x,t,q):
        x=np.asarray(x,dtype=np.float64);t=np.asarray(t,dtype=np.float64);q=np.asarray(q)
        require(x.shape==(len(t),2) and q.shape==t.shape and np.isfinite(x).all() and np.isfinite(t).all() and ((0<=t)&(t<=1)).all(), 'invalid reference coordinates')
        require(np.isin(q,[0,1] if c.pde=='wave' else [0]).all(),'invalid reference quantity')
        output=[];bounds=[];S=None
        for role in c.fields:
            for j in range(32):
                r,p=self._get(c,role,j)
                if S is None:S=self.spaces[c.family].sample_operator(x)
                X=p['spatial'].numpy();T=p['temporal'].numpy();Y=np.asarray(S@X);n=r['snapshots'];dt=r['dt']
                if c.pde=='poisson':values=Y@T[:,0]
                else:
                    k=np.minimum((t/dt).astype(int),n-2);z=(t-k*dt)/dt
                    # Same full integration intervals and C1 polynomial as certified solver.
                    ul=np.einsum('ij,ji->i',Y,T[:,k]);ur=np.einsum('ij,ji->i',Y,T[:,k+1])
                    vl=np.einsum('ij,ji->i',Y,T[:,n+k])/dt;vr=np.einsum('ij,ji->i',Y,T[:,n+k+1])/dt
                    u=(2*z**3-3*z*z+1)*ul+(z**3-2*z*z+z)*dt*vl+(-2*z**3+3*z*z)*ur+(z**3-z*z)*dt*vr
                    v=(6*z*z-6*z)/dt*ul+(3*z*z-4*z+1)*vl+(-6*z*z+6*z)/dt*ur+(3*z*z-2*z)*vr
                    values=np.where(q==1,v,u)
                output.append(values);bounds.append([r['compression']['error_bound']['v' if k else 'u'] for k in q])
        return np.column_stack(output),np.column_stack(bounds)
    def evaluate(self,c,x,t,q,*,return_bound=False,chunk=2048):
        z=np.concatenate([f.coefficients for f in c.fields.values()]);out=[];bound=[]
        for a in range(0,len(x),chunk):
            H,E=self.sensitivities(c,np.asarray(x)[a:a+chunk],np.asarray(t)[a:a+chunk],np.asarray(q)[a:a+chunk])
            out.append(H@z);bound.append(E@np.abs(z))
        y=np.concatenate(out);e=np.concatenate(bound);require(np.isfinite(y).all(),'nonfinite reference cache output')
        return (y,e) if return_bound else y
    def export_index(self,cases):
        out={self.asset_key(c,k,j):str(self.asset_path(c,k,j)) for c in cases for k in c.fields for j in range(32)}
        out.update({p.stem:str(p) for p in self.sampling_assets.values()});return out
