"""Multi-case CPU64 reference. Existing C3 functions/architectures are untouched.

Reuse the independently checked P1--P4 Space and Gauss4 wave rational formula.
Heat is consistent-mass CN with C1 Hermite output. Wave v is the analytical
Hermite derivative, NEVER a finite difference of saved displacement frames.
Only homogeneous tagged Dirichlet and known linear coefficients are supported.
"""
from functools import lru_cache
import numpy as np
from scipy.sparse.linalg import splu
from ..core import require
from ..wave_fidelity.reference import Space,lagrange
from ..geometric.quadrature import cells
from .definition import geometry,COEFFICIENTS


def field_matrix(fields,x):
    if not fields:return np.empty((len(x),0))
    first=fields[0]
    require(all(np.array_equal(f.centers,first.centers) and f.radius==first.radius for f in fields),'same geometry/field basis required for grouped evaluation')
    out=[]
    for a in range(0,len(x),4096):
        r2=np.sum((np.asarray(x[a:a+4096])[:,None]-first.centers[None])**2,-1)/first.radius**2
        B=np.maximum(0.,1-r2)**4/np.sqrt(len(first.centers))
        out.append(B@np.column_stack([f.coefficients*f.amplitude for f in fields]))
    return np.concatenate(out,0) if out else np.zeros((0,len(fields)))


class Reference:
    def __init__(self,family,h=None,degree=None):
        from .resolution import resolve_reference
        mesh,spec,_=resolve_reference(family,h=h,degree=degree)
        self.space=Space(mesh,spec.degree,speed=1.);self.family=family;self.h=spec.h;self.degree=spec.degree;self.factors={}
    def initial(self,fields):
        s=self.space;u=field_matrix(fields,s.position)
        require(np.max(np.abs(u[s.dirichlet]),initial=0.)<1e-10,'C3 initial trace incompatible with Dirichlet policy')
        return u[s.free]
    def load(self,fields,order=8):
        s=self.space;q=cells(s.mesh,order);F=np.zeros((len(s.position),len(fields)))
        for a in range(0,len(q.weight),4096):
            sl=slice(a,min(a+4096,len(q.weight)));basis=lagrange(q.barycentric[sl],s.degree)[0];value=field_matrix(fields,q.position[sl]);local=q.weight[sl,None,None]*basis[:,:,None]*value[:,None,:]
            np.add.at(F,s.dofs[q.cell_id[sl]].ravel(),local.reshape(-1,len(fields)))
        return F[s.free]
    def poisson(self,fields,position,order=8):
        s=self.space;F=self.load(fields,order)
        if 'poisson' not in self.factors:self.factors['poisson']=splu(s.Ki)
        u=self.factors['poisson'].solve(F);residual=np.linalg.norm(s.Ki@u-F,axis=0)/(1+np.linalg.norm(F,axis=0))
        require(np.isfinite(u).all() and residual.max(initial=0.)<1e-10,'elliptic reference residual failed')
        return np.asarray(s.sample_operator(position)@u).T,{'max_relative_residual':float(residual.max(initial=0.)),'precision':'CPU float64','spatial_basis':'P'+str(self.degree)}
    def dynamic(self,pde,initial,velocity,position,time,dt=.002):
        require(pde in {'heat','wave'},'dynamic family must be heat/wave');s=self.space
        t=np.asarray(time,dtype=float);require(t.shape==(len(position),) and np.isfinite(t).all() and (t>=0).all() and (t<=1).all(),'invalid continuous physical times')
        require(len(t)>0,'at least one physical query required')
        u=self.initial(initial);v=self.initial(velocity) if pde=='wave' else None;ncase=u.shape[1]
        S=s.sample_operator(position);U=np.zeros((ncase,len(t)));V=np.zeros_like(U) if pde=='wave' else None
        n=int(np.ceil(1/dt));h=1/n;steps=max(1,min(n,int(np.ceil(t.max(initial=0.)/h))))
        K=COEFFICIENTS[pde]*s.Ki
        key=(pde,h)
        if pde=='wave':gamma=(1+1j*np.sqrt(3))/24;beta=.25+1j/(4*np.sqrt(3));A=(s.Mi+gamma*h*h*K).tocsc()
        else:A=(s.Mi+h*K/2).tocsc();B=s.Mi-h*K/2
        if key not in self.factors:self.factors[key]=splu(A)
        factor=self.factors[key];energy=[];maxres=0.;increases=0.
        def norm_energy(a,b=None):
            return .5*np.sum(a*(K@a),axis=0)+.5*np.sum(b*(s.Mi@b),axis=0) if b is not None else np.sum(a*(s.Mi@a),axis=0)
        e0=norm_energy(u,v);last=e0;energy.append(e0)
        for k in range(steps):
            old_u=u;old_v=v
            if pde=='wave':
                rhs=beta*h*(s.Mi@v)-gamma*h*h*(K@u);solve_value=factor.solve(rhs)
                u=old_u+4*np.sqrt(3)*solve_value.imag;v=old_v+12/h*(-solve_value.real+np.sqrt(3)*solve_value.imag)
            else:rhs=B@u;u=factor.solve(rhs);solve_value=u
            take=np.flatnonzero((t>=k*h-1e-14)&(t<=(k+1)*h+1e-14))
            if len(take):
                if pde=='heat':old_v=-s.factor.solve(K@old_u);new_v=-s.factor.solve(K@u)
                else:new_v=v
                z=np.clip((t[take]-k*h)/h,0,1)[:,None];P=S[take];ul,ur,vl,vr=(np.asarray(P@a) for a in (old_u,u,old_v,new_v))
                values=(2*z**3-3*z*z+1)*ul+(z**3-2*z*z+z)*h*vl+(-2*z**3+3*z*z)*ur+(z**3-z*z)*h*vr
                U[:,take]=values.T
                if V is not None:V[:,take]=((6*z*z-6*z)/h*ul+(3*z*z-4*z+1)*vl+(-6*z*z+6*z)/h*ur+(3*z*z-2*z)*vr).T
            if k%max(1,steps//50)==0 or k==steps-1:
                e=norm_energy(u,v);energy.append(e);increases=max(increases,float(np.max(e-last)));last=e
                rr=np.linalg.norm(A@solve_value-rhs,axis=0)/(np.linalg.norm(rhs,axis=0)+1e-30);maxres=max(maxres,float(rr.max(initial=0.)))
        require(np.isfinite(U).all() and (V is None or np.isfinite(V).all()),'nonfinite dynamic reference')
        energy=np.asarray(energy);drift=float(np.max(np.abs(energy-e0)/(np.maximum(e0,1e-30))))
        return U,V,{'precision':'CPU float64','dt':h,'steps':steps,'method':'Gauss4 + C1 Hermite, v=du/dt' if pde=='wave' else 'CN + C1 Hermite',
                    'wave_relative_energy_drift':drift if pde=='wave' else None,'heat_max_mass_norm_increase':increases if pde=='heat' else None,
                    'max_relative_solve_residual':maxres,'energy_monitor':'initial + up to51 evenly spaced grid states',
                    'spatial_basis':'P'+str(self.degree)}
    def mesh_report(self):
        m=self.space.mesh
        return {'mesh_id':m.mesh_id,'polygon_id':m.shape.polygon_id,'geometry_id':m.geometry_id,'h':self.h,'degree':self.degree,
                'vertices':len(m.vertices),'cells':len(m.cells),'max_element_area':float(m.cell_areas.max()),'quality':m.quality(),'DOFs':len(self.space.position),'free_DOFs':len(self.space.free)}


@lru_cache(maxsize=1)
def reference(family,h=.02,degree=4):return Reference(family,h,degree)
