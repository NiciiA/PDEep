"""CPU float64 sparse P1 reference solvers. New numerical G1; T1 is untouched.

Consistent mass, variable positive diffusion, strong Dirichlet elimination and
positive outward Neumann RHS. Heat CN and wave implicit midpoint are second order.
Arbitrary time uses cubic Hermite dense output; wave velocity is its derivative.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix,bmat,csc_matrix
from scipy.sparse.linalg import splu,spsolve
from ..core import require
from .quadrature import cells,boundary


def values(fn,x,t=0.):
    a=np.asarray(fn(x,t) if callable(fn) else fn,dtype=np.float64)
    return np.broadcast_to(a,(len(x),)).copy()

@dataclass(frozen=True)
class BoundaryData:
    kind: str='DIRICHLET'
    value: object=0.
    velocity: object=0.

    def __post_init__(self):require(self.kind in {'DIRICHLET','NEUMANN','ROBIN','PERIODIC'},'unknown boundary kind')


class FEM:
    def __init__(self,mesh,*,coefficient=1.,conditions=None,order=4):
        self.mesh=mesh.validate();self.coefficient=coefficient;self.order=order
        self.conditions={int(t):BoundaryData() for t in np.unique(mesh.boundary_tags)} if conditions is None else conditions
        require(set(self.conditions)==set(map(int,mesh.boundary_tags)),'boundary conditions must cover every tag exactly')
        require(all(b.kind in {'DIRICHLET','NEUMANN'} for b in self.conditions.values()),'Robin/periodic schema reserved, solver not implemented')
        tri=mesh.vertices[mesh.cells];J=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)
        self.grad=np.array([[-1.,-1.],[1.,0.],[0.,1.]])[None]@np.linalg.inv(J) # 2x2 affine-coordinate transform, not a physical system inverse.
        q=cells(mesh,order);self.volume_quadrature=q;a=values(coefficient,q.position);require(np.isfinite(a).all() and (a>0).all(),'diffusivity/c^2 must be positive and finite')
        integral=np.bincount(q.cell_id,weights=q.weight*a,minlength=len(mesh.cells))
        localK=np.einsum('tik,tjk,t->tij',self.grad,self.grad,integral)
        localM=mesh.cell_areas[:,None,None]*(np.ones((3,3))+np.eye(3))[None]/12
        i=np.repeat(mesh.cells,3,axis=1).ravel();j=np.tile(mesh.cells,(1,3)).ravel();n=len(mesh.vertices)
        self.K=coo_matrix((localK.ravel(),(i,j)),shape=(n,n)).tocsc();self.M=coo_matrix((localM.ravel(),(i,j)),shape=(n,n)).tocsc()
        self.dirichlet=np.unique(mesh.boundary_edges[np.isin(mesh.boundary_tags,[t for t,b in self.conditions.items() if b.kind=='DIRICHLET'])])
        self.free=np.setdiff1d(np.arange(n),self.dirichlet);require(len(self.free)>0,'mesh has no unconstrained degrees of freedom')
        self.mass_weights=np.asarray(self.M.sum(1)).ravel();self.last_residual=None
        self._load_matrix=coo_matrix(((q.weight[:,None]*q.barycentric).ravel(),(mesh.cells[q.cell_id].ravel(),np.repeat(np.arange(len(q.weight)),3))),shape=(n,len(q.weight))).tocsr()
        self._flux_cache={}
        for tag,bc in self.conditions.items():
            if bc.kind=='NEUMANN':
                z=boundary(mesh,order,tags={tag});B=coo_matrix(((z.weight[:,None]*z.barycentric).ravel(),(mesh.boundary_edges[z.cell_id].ravel(),np.repeat(np.arange(len(z.weight)),2))),shape=(n,len(z.weight))).tocsr()
                self._flux_cache[tag]=(z,B)

        self._mass_factor=splu(self.M[self.free][:,self.free]);self._elliptic_factor=None

    def prescribed(self,t,*,velocity=False):
        u=np.zeros(len(self.mesh.vertices));assigned=np.zeros(len(u),bool)
        for tag,bc in self.conditions.items():
            if bc.kind!='DIRICHLET':continue
            ids=np.unique(self.mesh.boundary_edges[self.mesh.boundary_tags==tag]);v=values(bc.velocity if velocity else bc.value,self.mesh.vertices[ids],t)
            overlap=assigned[ids];require(np.allclose(u[ids][overlap],v[overlap],rtol=1e-10,atol=1e-12),'incompatible Dirichlet traces at a boundary junction')
            u[ids]=v;assigned[ids]=True
        return u

    def load(self,source=0.,t=0.):
        q=self.volume_quadrature
        if not callable(source) and float(source)==0:F=np.zeros(len(self.mesh.vertices))
        else:
            f=values(source,q.position,t);require(np.isfinite(f).all(),'nonfinite PDE source');F=self._load_matrix@f
        for tag,(q,B) in self._flux_cache.items():
            bc=self.conditions[tag]
            flux=np.asarray(bc.value(q.position,t,q.normal) if callable(bc.value) else bc.value,dtype=float)
            F+=B@np.broadcast_to(flux,(len(q.weight),))
        return F

    def _residual(self,A,x,b):
        r=A@x-b;self.last_residual={'absolute':float(np.linalg.norm(r)),'relative':float(np.linalg.norm(r)/(1+np.linalg.norm(b)))}
        require(np.isfinite(x).all() and self.last_residual['relative']<1e-9,'reference linear solve failed residual/finite gate')
        return x

    def poisson(self,source=0.,*,t=0.):
        F=self.load(source,t);u=self.prescribed(t);I=self.free;D=self.dirichlet
        if len(D):
            if self._elliptic_factor is None:self._elliptic_factor=splu(self.K[I][:,I])
            rhs=F[I]-self.K[I][:,D]@u[D];u[I]=self._residual(self.K[I][:,I],self._elliptic_factor.solve(rhs),rhs)
        else:
            require(abs(F.sum())<=1e-10*(1+np.abs(F).sum()),'pure-Neumann source/flux compatibility fails; no gauge can repair it')
            m=self.mass_weights[:,None];A=bmat([[self.K,csc_matrix(m)],[csc_matrix(m.T),None]],format='csc');b=np.r_[F,0.]
            solution=self._residual(A,spsolve(A,b),b);u=solution[:-1]
        return u

    def dynamics(self,pde,initial,*,velocity=None,source=0.,dt=.005,final_time=1.):
        return Evolution(self,pde,initial,velocity,source,dt,final_time)


class Evolution:
    def __init__(self,fem,pde,initial,velocity,source,dt,final_time):
        require(pde in {'heat','wave'} and dt>0 and final_time>0,'invalid evolution specification')
        self.fem=fem;self.pde=pde;self.source=source;self.steps=int(np.ceil(final_time/dt));self.dt=final_time/self.steps;self.final_time=final_time
        self.times=np.linspace(0,final_time,self.steps+1);I=fem.free;D=fem.dirichlet;n=len(fem.mesh.vertices)
        u=values(initial,fem.mesh.vertices) if callable(initial) else np.asarray(initial,dtype=float).copy()
        require(u.shape==(n,) and np.allclose(u[D],fem.prescribed(0)[D],atol=1e-10,rtol=1e-10),'initial displacement violates prescribed trace')
        self.U=[u];self.V=[];self.residuals=[];h=self.dt;M,K=fem.M,fem.K
        if pde=='heat':
            A=M+h*K/2;B=M-h*K/2;factor=splu(A[I][:,I])
            for k,t in enumerate(self.times[:-1]):
                new=fem.prescribed(self.times[k+1]);rhs=(B@u+h*(fem.load(source,t)+fem.load(source,self.times[k+1]))/2)[I]-A[I][:,D]@new[D]
                new[I]=fem._residual(A[I][:,I],factor.solve(rhs),rhs);self.residuals.append(fem.last_residual);self.U.append(new);u=new
            self.U=np.stack(self.U)
            # Semidiscrete derivatives, including the known boundary trace rate.
            for u,t in zip(self.U,self.times):
                v=fem.prescribed(t,velocity=True);rhs=(fem.load(source,t)-K@u)[I]-M[I][:,D]@v[D]
                v[I]=fem._mass_factor.solve(rhs);self.V.append(v)
        else:
            v=np.zeros(n) if velocity is None else values(velocity,fem.mesh.vertices) if callable(velocity) else np.asarray(velocity,dtype=float).copy()
            require(v.shape==(n,) and np.allclose(v[D],fem.prescribed(0,velocity=True)[D],atol=1e-10,rtol=1e-10),'initial velocity violates trace compatibility')
            self.V=[v];A=bmat([[M,-h*M/2],[h*K/2,M]],format='csc');B=bmat([[M,h*M/2],[-h*K/2,M]],format='csc')
            free=np.r_[I,n+I];fixed=np.r_[D,n+D];factor=splu(A[free][:,free])
            state=np.r_[u,v]
            for k,t in enumerate(self.times[:-1]):
                new=np.r_[fem.prescribed(self.times[k+1]),fem.prescribed(self.times[k+1],velocity=True)]
                rhs=B@state+np.r_[np.zeros(n),h*(fem.load(source,t)+fem.load(source,self.times[k+1]))/2]
                reduced=rhs[free]-A[free][:,fixed]@new[fixed];new[free]=fem._residual(A[free][:,free],factor.solve(reduced),reduced)
                self.residuals.append(fem.last_residual);u,v=new[:n],new[n:];self.U.append(u);self.V.append(v);state=new
            self.U=np.stack(self.U)
        self.V=np.stack(self.V)
        require(np.isfinite(self.U).all() and np.isfinite(self.V).all(),'reference evolution produced nonfinite state')

    def nodal(self,t,quantity='u'):
        require(0<=t<=self.final_time and quantity in {'u','v'},'invalid physical time/quantity')
        require(quantity=='u' or self.pde=='wave','heat has no independent velocity observation')
        k=min(int(np.floor(t/self.dt)),self.steps-1);s=(t-self.times[k])/self.dt;h=self.dt
        if quantity=='u':
            return (2*s**3-3*s*s+1)*self.U[k]+(s**3-2*s*s+s)*h*self.V[k]+(-2*s**3+3*s*s)*self.U[k+1]+(s**3-s*s)*h*self.V[k+1]
        return (6*s*s-6*s)/h*self.U[k]+(3*s*s-4*s+1)*self.V[k]+(-6*s*s+6*s)/h*self.U[k+1]+(3*s*s-2*s)*self.V[k+1]
    def evaluate(self,position,time,quantity='u'):
        p=np.asarray(position).reshape(-1,2);t=np.broadcast_to(np.asarray(time),(len(p),));q=np.broadcast_to(np.asarray(quantity),(len(p),));out=np.zeros(len(p))
        self.fem.mesh.locate(p)
        for key in set(zip(t.tolist(),q.tolist())):
            select=(t==key[0])&(q==key[1]);out[select]=self.fem.mesh.evaluate(self.nodal(*key),p[select])
        return out
    def energy(self):
        return np.array([.5*(v@self.fem.M@v+u@self.fem.K@u) for u,v in zip(self.U,self.V)])
