"""Independent P1--P4 wave reference tools; physical initial fields never filtered.

Homogeneous Dirichlet, zero forcing only. Mesh-conforming red refinement preserves
canonical P1 initial functions exactly. P2 uses vertex/edge Lagrange DOFs, consistent
mass and stiffness, and direct continuous-time Chebyshev matrix functions.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.fft import dct
from scipy.linalg import eigh,solve_triangular
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from ..core import require,digest
from ..geometric.domain import GeometryDomain
from ..geometric.quadrature import cells

VERSION='G1-wave-Lagrange-Chebyshev-v1'


def refine(mesh):
    """Conforming four-way refinement, with stable physical tags and polygon."""
    edges=mesh.edge_index.T;n=len(mesh.vertices);mid={tuple(e):n+i for i,e in enumerate(edges)}
    coordinates=np.vstack((mesh.vertices,mesh.vertices[edges].mean(1)));triangles=[]
    for a,b,c in mesh.cells:
        ab,bc,ca=(mid[tuple(sorted(e))] for e in ((a,b),(b,c),(c,a)))
        triangles.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)))
    boundary=[];tags=[]
    for (a,b),tag in zip(mesh.boundary_edges,mesh.boundary_tags):
        k=mid[tuple(sorted((a,b)))];boundary.extend(((a,k),(k,b)));tags.extend((tag,tag))
    return GeometryDomain(mesh.shape,coordinates,np.asarray(triangles),np.asarray(boundary),np.asarray(tags),
                          np.repeat(mesh.region_tags,4),{'backend':'conforming red refinement','parent_mesh':mesh.mesh_id,
                          'target_h':mesh.provenance['target_h']/2,'coordinate_system':'physical Cartesian'}).validate()


def shapes(bary,degree):
    if degree==1:return bary
    l0,l1,l2=bary.T
    return np.column_stack((l0*(2*l0-1),l1*(2*l1-1),l2*(2*l2-1),4*l0*l1,4*l1*l2,4*l2*l0))


def indices(degree):
    p=degree
    return [(p,0,0),(0,p,0),(0,0,p)]+[(p-k,k,0) for k in range(1,p)]+[(0,p-k,k) for k in range(1,p)]+[(k,0,p-k) for k in range(1,p)]+[(a,b,p-a-b) for a in range(1,p) for b in range(1,p-a)]


def lagrange(bary,degree):
    """Complete nodal triangular Lagrange basis, no division by vanishing factors."""
    value=[];gradient=[];ref=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
    for alpha in indices(degree):
        factors=[];derivatives=[]
        for i,a in enumerate(alpha):
            factors.extend((degree*bary[:,i]-k)/(a-k) for k in range(a))
            derivatives.extend(np.broadcast_to(degree*ref[i]/(a-k),(len(bary),2)) for k in range(a))
        product=np.prod(factors,axis=0);grad=np.zeros((len(bary),2))
        for j,d in enumerate(derivatives):grad+=d*np.prod([v for k,v in enumerate(factors) if k!=j],axis=0)[:,None] if len(factors)>1 else d
        value.append(product);gradient.append(grad)
    return np.stack(value,1),np.stack(gradient,1)


class Space:
    """Independent P1--P4 assembly, including local spectral upper bound."""
    def __init__(self,mesh,degree=2,speed=.7):
        require(degree in (1,2,3,4) and speed>0,'supported wave space: P1..P4, positive speed')
        self.mesh=mesh;self.degree=degree;self.speed=speed;n=len(mesh.vertices)
        edges=mesh.edge_index.T;edge_dof={tuple(e):n+i for i,e in enumerate(edges)}
        if degree==1:self.position=mesh.vertices.copy();self.dofs=mesh.cells.copy();D=np.unique(mesh.boundary_edges)
        elif degree==2:
            self.position=np.vstack((mesh.vertices,mesh.vertices[edges].mean(1)))
            self.dofs=np.column_stack((mesh.cells,np.array([[edge_dof[tuple(sorted(e))] for e in ((a,b),(b,c),(c,a))] for a,b,c in mesh.cells])))
            D=np.r_[np.unique(mesh.boundary_edges),[edge_dof[tuple(sorted(e))] for e in mesh.boundary_edges]]
        else:
            p=degree;coordinates=list(mesh.vertices);edge_dofs={}
            for a,b in edges:
                edge_dofs[(a,b)]=tuple(range(len(coordinates),len(coordinates)+p-1))
                coordinates.extend(((p-k)*mesh.vertices[a]+k*mesh.vertices[b])/p for k in range(1,p))
            dofs=[]
            for a,b,c in mesh.cells:
                row=[a,b,c]
                for i,j in ((a,b),(b,c),(c,a)):
                    e=edge_dofs[tuple(sorted((i,j)))];row.extend(e if i<j else e[::-1])
                for alpha in indices(p)[3+3*(p-1):]:
                    row.append(len(coordinates));coordinates.append(np.array(alpha)@mesh.vertices[[a,b,c]]/p)
                dofs.append(row)
            self.position=np.array(coordinates);self.dofs=np.array(dofs)
            D=np.r_[np.unique(mesh.boundary_edges),[v for e in mesh.boundary_edges for v in edge_dofs[tuple(sorted(e))]]]
        self.dirichlet=np.sort(D);self.free=np.setdiff1d(np.arange(len(self.position)),D)
        q=cells(mesh,max(4,degree+1));Q=len(q.weight)//len(mesh.cells);l=q.barycentric[:Q];w=q.weight[:Q]/mesh.cell_areas[0]
        p,dg=lagrange(l,degree)
        vertices=mesh.vertices[mesh.cells];J=np.stack((vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),-1)
        grad=np.einsum('qik,tkj->tqij',dg,np.linalg.inv(J));Mref=np.einsum('qi,qj,q->ij',p,p,w)
        localM=mesh.cell_areas[:,None,None]*Mref;localK=speed**2*np.einsum('tqik,tqjk,q,t->tij',grad,grad,w,mesh.cell_areas)
        d=self.dofs.shape[1];i=np.repeat(self.dofs,d,axis=1).ravel();j=np.tile(self.dofs,(1,d)).ravel();size=len(self.position)
        self.M=coo_matrix((localM.ravel(),(i,j)),shape=(size,size)).tocsc();self.K=coo_matrix((localK.ravel(),(i,j)),shape=(size,size)).tocsc()
        self.Mi=self.M[self.free][:,self.free].tocsc();self.Ki=self.K[self.free][:,self.free].tocsc();self.factor=splu(self.Mi)
        # Sum-of-element Rayleigh bound: global quotient <= largest local
        # generalized eigenvalue. This includes ALL frequencies; no truncation.
        L=np.linalg.cholesky(Mref);Li=solve_triangular(L,np.eye(d),lower=True)
        transformed=np.einsum('ij,tjk,lk->til',Li,localK/mesh.cell_areas[:,None,None],Li)
        self.lambda_bound=float(np.linalg.eigvalsh(transformed)[:,-1].max())*(1+1e-12)
        self.identity=digest((VERSION,mesh.mesh_id,degree,speed))

    def initial(self,field):
        value=np.asarray(field(self.position),dtype=np.float64)
        require(value.shape==(len(self.position),) and np.isfinite(value).all(),'invalid continuous initial field')
        require(np.max(np.abs(value[self.dirichlet]),initial=0.)<1e-10,'initial field violates homogeneous Dirichlet data')
        value[self.dirichlet]=0.;return value[self.free]

    def sample_operator(self,position):
        ids,bary=self.mesh.locate(position);p=lagrange(bary,self.degree)[0];n=len(ids)
        return coo_matrix((p.ravel(),(np.repeat(np.arange(n),p.shape[1]),self.dofs[ids].ravel())),shape=(n,len(self.position))).tocsr()[:,self.free]

    def evaluate(self,coefficient,position):return np.asarray(self.sample_operator(position)@coefficient)
    def energy(self,u,v):return float(.5*(u@self.Ki@u+v@self.Mi@v))
    def norm(self,u):return float(np.sqrt(max(0.,u@self.Mi@u)))


class SpectralControl:
    """Full dense generalized eigenbasis, small problems only. No mode cutoff."""
    def __init__(self,space,u0,v0):
        require(len(space.free)<=2500,'dense eigen control limited to <=2500 free DOFs')
        self.space=space;self.lam,self.phi=eigh(space.Ki.toarray(),space.Mi.toarray());require(self.lam.min()>0,'non-SPD wave system')
        self.omega=np.sqrt(self.lam);self.a=self.phi.T@(space.Mi@u0);self.b=self.phi.T@(space.Mi@v0)
        residual=space.Ki@self.phi-(space.Mi@self.phi)*self.lam
        self.diagnostics={'relative_eigen_residual':float(np.linalg.norm(residual)/np.linalg.norm(space.Ki@self.phi)),
                          'mass_orthogonality_max':float(np.max(abs(self.phi.T@(space.Mi@self.phi)-np.eye(len(self.lam)))))}
    def nodal(self,t):
        c=np.cos(self.omega*t);s=np.sin(self.omega*t)
        return self.phi@(self.a*c+self.b/self.omega*s),self.phi@(-self.omega*self.a*s+self.b*c)


def coefficients(bound,times,tolerance=1e-12):
    """Uniform Chebyshev approximation of exact cosine/sine matrix functions.

    Functions are analytic in lambda at zero. Degree grows with t*sqrt(bound),
    NEVER reduces a physical spectrum. Tail and off-node approximation checked.
    """
    t=np.asarray(times,dtype=float);require(t.ndim==1 and len(t)>0 and np.isfinite(t).all() and (t>=0).all(),'invalid physical times')
    N=max(64,int(np.ceil(np.sqrt(bound)*max(t.max(),.001)))+48)
    for _ in range(4):
        theta=np.pi*(np.arange(N)+.5)/N
        # Half-angle mapping avoids cancellation in 1+cos(theta) near pi.
        # At fine-mesh degrees O(sqrt(bound)), that cancellation otherwise
        # contaminates the low-frequency function values by O(eps*bound).
        lam=bound*np.cos(theta/2)**2;omega=np.sqrt(lam)
        phase=t[:,None]*omega;cos=np.cos(phase);sinc=t[:,None]*np.sinc(phase/np.pi);ds=-omega*np.sin(phase)
        functions=np.stack((cos,sinc,ds),1);c=dct(functions,type=2,axis=-1)/N;c[...,0]/=2
        # Tiny roundoff coefficient floor scales with physical function units.
        units=np.maximum(1.,np.max(abs(functions),axis=-1))
        tail=np.max(np.sum(abs(c[...,-16:]),axis=-1)/units)
        if tail<20*tolerance:break
        N*=2
    require(tail<20*tolerance,'Chebyshev tail failed; no filtered fallback')
    # Keep degree conservative: the full expansion, including small coefficients.
    test=np.linspace(-1,1,257);w=np.sqrt(bound*(1+test)/2);phase=t[:,None]*w
    exact=np.stack((np.cos(phase),t[:,None]*np.sinc(phase/np.pi),-w*np.sin(phase)),1)
    approx=np.stack([[np.polynomial.chebyshev.chebval(test,cc) for cc in row] for row in c])
    error=float(np.max(np.abs(approx-exact)/np.maximum(1.,np.max(abs(exact),axis=-1))[...,None]))
    require(error<100*tolerance,'scalar matrix-function approximation error exceeds tolerance')
    return c,{'terms':N,'lambda_bound':bound,'scalar_relative_error':error,'normalized_last16_coefficient_sum':float(tail),'tolerance':tolerance}


class ContinuousWave:
    """Direct arbitrary-time matrix-function evaluation; no frame interpolation.

    Computes only requested spatial outputs, using two vector recurrences and
    sparse mass solves. Memory O(V+TQ), rather than O(V*number_of_time_frames).
    """
    def __init__(self,space,initial,velocity):
        self.space=space;self.u0=space.initial(initial);self.v0=space.initial(velocity);self.last_diagnostics=None
    def evaluate(self,position,times,*,return_nodal=False):
        c,diagnostics=coefficients(self.space.lambda_bound,times);S=self.space.sample_operator(position) if not return_nodal else None
        sample=lambda z:z if S is None else np.asarray(S@z)
        size=len(self.u0) if S is None else len(position);out_u=np.zeros((len(times),size));out_v=np.zeros_like(out_u)
        # Two starting vectors, independently propagated by exactly the same L.
        p0=np.column_stack((self.u0,self.v0));bound=self.space.lambda_bound
        action=lambda z:2*self.space.factor.solve(self.space.Ki@z)/bound-z
        previous=p0;current=None
        for j in range(c.shape[-1]):
            if j==0:z=previous
            elif j==1:current=action(previous);z=current
            else:new=2*action(current)-previous;previous,current=current,new;z=current
            values=sample(z)
            out_u+=c[:,0,j,None]*values[None,:,0]+c[:,1,j,None]*values[None,:,1]
            out_v+=c[:,2,j,None]*values[None,:,0]+c[:,0,j,None]*values[None,:,1]
        self.last_diagnostics=diagnostics;require(np.isfinite(out_u).all() and np.isfinite(out_v).all(),'nonfinite continuous wave reference')
        return out_u,out_v


class GaussWave:
    """Fourth-order Gauss--Legendre propagation, with coherent Hermite output.

    Rational amplification is [2/2] Pade, evaluated with one complex resolvent
    (algebraically the product of two conjugate Cayley factors). No spatial
    eigenmodes are removed. Hermite displacement
    uses endpoint state velocities; reported velocity is its exact derivative.
    This is a separately versioned numerical approximation, checked by dt halving.
    """
    def __init__(self,space,initial,velocity):
        self.space=space;self.u0=space.initial(initial);self.v0=space.initial(velocity);self.last_diagnostics=None
    def evaluate(self,position,times,*,dt=.002):
        times=np.asarray(times,dtype=float);require(times.ndim==1 and len(times)>0 and np.isfinite(times).all() and (times>=0).all() and (times<=1).all() and dt>0,'invalid physical times/dt')
        # The integration grid belongs to the physical case, not the query set.
        # Stop early when possible, but never rescale h to a chunk's last query.
        full_steps=int(np.ceil(1/dt));h=1/full_steps
        steps=max(1,min(full_steps,int(np.ceil(float(times.max())/h))));s=self.space
        S=s.sample_operator(position);u=self.u0.copy();v=self.v0.copy();gamma=(1+1j*np.sqrt(3))/24;beta=.25+1j/(4*np.sqrt(3))
        A=(s.Mi+gamma*h*h*s.Ki).tocsc();factor=splu(A);U=np.zeros((len(times),len(position)));V=np.zeros_like(U)
        energies=[s.energy(u,v)];max_residual=0.;sample=lambda y:np.asarray(S@y)
        for k in range(steps):
            old_u,old_v=u,v
            # Delta form avoids subtraction of two O(u) states before /h.
            # It is exactly the real [2/2] Pade action, not a modified equation.
            rhs=beta*h*(s.Mi@v)-gamma*h*h*(s.Ki@u);w=factor.solve(rhs)
            u=old_u+4*np.sqrt(3)*w.imag;v=old_v+12/h*(-w.real+np.sqrt(3)*w.imag)
            start=k*h;end=(k+1)*h;choose=np.flatnonzero((times>=start-1e-14)&(times<=end+1e-14))
            if len(choose):
                z=np.clip((times[choose]-start)/h,0,1)[:,None];ul,ur,vl,vr=map(sample,(old_u,u,old_v,v))
                U[choose]=(2*z**3-3*z*z+1)*ul+(z**3-2*z*z+z)*h*vl+(-2*z**3+3*z*z)*ur+(z**3-z*z)*h*vr
                V[choose]=(6*z*z-6*z)/h*ul+(3*z*z-4*z+1)*vl+(-6*z*z+6*z)/h*ur+(3*z*z-2*z)*vr
            if k%max(1,steps//50)==0 or k==steps-1:
                energies.append(s.energy(u,v));max_residual=max(max_residual,float(np.linalg.norm(A@w-rhs)/(1+np.linalg.norm(rhs))))
        drift=float(np.max(abs(np.array(energies)-energies[0]))/(energies[0]+np.finfo(float).tiny))
        require(max_residual<1e-10 and np.isfinite(U).all() and np.isfinite(V).all(),'Gauss wave solve/real/finite verification failed')
        self.last_diagnostics={'method':'two-stage Gauss-Legendre [2/2] Pade + C1 Hermite (v=du/dt)','implementation':'one shifted complex solve, real delta form','dt':h,'steps':steps,'physical_grid_interval':[0.,1.],
                               'relative_grid_energy_drift':drift,'max_relative_solve_residual':max_residual,
                               'timing_accuracy':'must verify by halving dt; energy conservation alone does not prove phase accuracy'}
        return U,V
