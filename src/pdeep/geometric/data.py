"""G1 only: canonical-mesh Gaussian fields and independent keyed data streams.

Numerical fields are P1 on a canonical mesh, NOT nine/eighteen modal slots.
A remesh changes the reference discretization but not the hidden physical field.
No unknown field is included in Problem/Inputs passed to a neural predictor.
"""
from dataclasses import dataclass,field
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from ..core import require,digest
from ..benchmarks import benchmark
from .domain import make_shape,mesh_shape
from .fem import FEM

VERSION='G1-canonical-P1-squared-elliptic-Gaussian-v1'
PDES=('poisson','heat','wave')


def rng(*keys):return np.random.default_rng(int(digest((VERSION,keys))[:16],16))


@dataclass(frozen=True)
class Problem:
    name: str
    mesh: object
    coefficient: float
    known: tuple=('homogeneous_Dirichlet','zero_dynamic_source')
    boundary_spec: tuple=()  # (tag, DIRICHLET|NEUMANN, known constant value)

    @property
    def conditions(self):
        from .fem import BoundaryData
        raw=self.boundary_spec or tuple((int(k),'DIRICHLET',0.) for k in np.unique(self.mesh.boundary_tags))
        require(len({r[0] for r in raw})==len(raw) and {r[0] for r in raw}==set(self.mesh.boundary_tags),'complete unique BC tag coverage required')
        require(all(k in {'DIRICHLET','NEUMANN'} and np.isfinite(v) for _,k,v in raw),'G1 neural boundary descriptor supports known constant Dirichlet/Neumann only')
        return {tag:BoundaryData(kind,float(value)) for tag,kind,value in raw}

    @property
    def physical_time(self):return None if self.name=='poisson' else (0.,1.)
    @property
    def symbolic(self):
        # Canonical typed PDE DAG, with field descriptors of arbitrary dimension.
        s=benchmark(self.name)
        from dataclasses import replace
        from ..graph import GraphBuilder
        blocks=tuple(SimpleNamespace(id=b.id,quantity_binding=b.quantity_binding,status=b.status,
                    component_ids=(),known_values=(),prior=None) for b in s.parameter_blocks)
        bindings=tuple(replace(b,value=float(self.coefficient if self.name=='heat' else np.sqrt(self.coefficient))) if b.role=='coefficient' else b for b in s.bindings)
        builder=GraphBuilder(bindings,physical_time=self.physical_time is not None);pending={n.id:n for n in s.graph.nodes};mapped={}
        while pending:
            for key in tuple(pending):
                n=pending[key]
                if all(k in mapped for k in n.operands):
                    from ..graph import Op
                    mapped[key]=builder.make(n.op,*(mapped[k] for k in n.operands),binding=n.binding,values=n.values,constant_type=n.type if n.op==Op.CONSTANT else None);del pending[key]
        graph=builder.finish(*(mapped[r] for r in s.graph.roots))
        from ..core import Status
        boundaries=tuple(SimpleNamespace(target=state,status=Status.KNOWN,value=bc.value,tag=tag,kind=bc.kind) for state in (('u','v') if self.name=='wave' else ('u',)) for tag,bc in self.conditions.items())
        return SimpleNamespace(name=s.name,graph=graph,bindings=bindings,
                               parameter_blocks=blocks,initials=s.initials,boundaries=boundaries,
                               physical_time=self.physical_time,validate=lambda:graph.validate(bindings,physical_time=self.physical_time is not None))
    @property
    def identity(self):return digest((VERSION,self.name,self.mesh.mesh_id,self.coefficient,self.known,self.boundary_spec))


# Explicit disjoint geometry categories. ID validation reuses train geometries
# with independent field seeds; held-out domains never enter calibration.
CATALOG={
 'train':(('square',1.),('rectangle',1.2),('rectangle',1.6),('lshape',1.),('disk',1.)),
 'validation':(('square',1.),('rectangle',1.2),('rectangle',1.6),('lshape',1.),('disk',1.)),
 'test':(('square',1.),('rectangle',1.2),('rectangle',1.6),('lshape',1.),('disk',1.)),
 'interpolation':(('rectangle',1.4),),
 'geometry_held_out':(('hole',1.),('annulus',1.),('polygon',1.),('ellipse',1.8),('rectangle',2.4))}
LIMITS={'train':12000,'validation':1024,'test':1024,'interpolation':1024,'geometry_held_out':1024}


@lru_cache(maxsize=64)
def domain(family,aspect=1.,h=.08):return mesh_shape(make_shape(family,aspect=aspect),h)


@dataclass
class Field:
    mesh: object
    nodal: np.ndarray
    prior: dict
    def __call__(self,x,t=0.):return self.mesh.evaluate(self.nodal,x)


def sample_field(mesh,stream,*,zero_trace=False,amplitude=1.,length=.2):
    """u=A^-1 M_lump A^-1 M_lump^1/2 z, z iid N(0,1).

    A=M_lump+ell²K, on free indices for zero trace. With
    S=A^-1 M_lump A^-1 M_lump^1/2, covariance = amplitude² S S^T.
    Squared elliptic smoothing gives smooth low-frequency physical fields;
    single smoothing of white noise retains excessive high-frequency wave energy.
    Natural smoothing for sources does not force their boundary values to zero.
    """
    f=FEM(mesh);mass=f.mass_weights;A=diags(mass)+length**2*f.K
    I=f.free if zero_trace else np.arange(len(mass));u=np.zeros(len(mass))
    factor=splu(A.tocsc()[I][:,I]);white=np.sqrt(mass[I])*stream.standard_normal(len(I))
    u[I]=amplitude*factor.solve(mass[I]*factor.solve(white))
    return Field(mesh,u,{'kind':VERSION,'length':length,'amplitude':amplitude,'zero_trace':zero_trace,
                        'dimension':len(I),'canonical_mesh_id':mesh.mesh_id})


@dataclass
class Case:
    problem: Problem
    case_id: str
    split: str
    index: int
    fields: dict
    solution: object
    reference: dict
    def evaluate(self,x,time=None,quantity='u'):
        require((time is None)==(self.problem.physical_time is None),'stationary physical time must be absent; dynamics need physical time')
        if quantity in self.fields:return self.fields[quantity](x)
        if self.problem.name=='poisson':
            require(quantity=='u','unsupported stationary quantity');return self.problem.mesh.evaluate(self.solution,x)
        return self.solution.evaluate(x,time,quantity)
    @property
    def manifest(self):
        return {'benchmark':'G1','schema':VERSION,'physical_case_id':self.case_id,'split':self.split,'case_index':self.index,
                'pde':self.problem.name,'geometry_id':self.problem.mesh.geometry_id,'polygon_id':self.problem.mesh.shape.polygon_id,
                'mesh_id':self.problem.mesh.mesh_id,'topology_id':self.problem.mesh.shape.topology_id,
                'prior':{k:v.prior for k,v in self.fields.items()},'known':self.problem.known,'reference':self.reference}


@lru_cache(maxsize=64)
def physical_case(split,pde,index,*,h=.04,dt=.002):
    require(split in CATALOG and pde in PDES and type(index) is int and 0<=index<LIMITS[split],'invalid G1 split/PDE/index')
    family,aspect=CATALOG[split][index%len(CATALOG[split])];mesh=domain(family,aspect,h);canonical=domain(family,aspect,.08)
    case_id=digest((VERSION,split,pde,index,mesh.geometry_id))
    fields={};names=('f',) if pde=='poisson' else ('u0','v0') if pde=='wave' else ('u0',)
    for name in names:fields[name]=sample_field(canonical,rng(case_id,'physical_field',name),zero_trace=name!='f')
    a=1. if pde=='poisson' else .02 if pde=='heat' else .49
    problem=Problem(pde,mesh,a);solver=FEM(mesh,coefficient=a)
    if pde=='poisson':sol=solver.poisson(fields['f'])
    else:sol=solver.dynamics(pde,fields['u0'],velocity=fields.get('v0'),dt=dt)
    return Case(problem,case_id,split,index,fields,sol,{'space':'P1 FEM','h':h,'dt':None if pde=='poisson' else sol.dt,
              'time_output':'cubic Hermite C1; wave velocity is derivative of displacement polynomial',
              'geometry_error':mesh.shape.geometry_error,'precision':'CPU float64'})


@dataclass(frozen=True)
class Normalizers:
    values: dict
    calibration: dict
    @property
    def identity(self):return digest({'benchmark':'G1',**self.calibration,'RMS':self.values})
    def scale(self,pde,quantity=0):
        key=('v' if quantity==1 else 'u') if isinstance(quantity,int) else quantity
        return self.values[pde][key]
    def asdict(self):return {'values':self.values,'calibration':self.calibration,'identity':self.identity}
    @classmethod
    def fromdict(cls,d):
        n=cls(d['values'],d['calibration']);require(n.identity==d['identity'],'G1 normalization identity mismatch');return n


def calibrate(count=10,queries=256):
    """Finite Monte Carlo train-only equal-case/area/time RMS, frozen per run.

    Must cover the five training geometry entries; never reads validation/test.
    """
    require(count>=5 and queries>=32,'calibration must cover train geometry catalog')
    values={};ids=[]
    for pde in PDES:
        accum={k:[] for k in (('u','f') if pde=='poisson' else ('u','u0','v','v0') if pde=='wave' else ('u','u0'))}
        for i in range(count):
            c=physical_case('train',pde,i);ids.append(c.case_id);s=rng(c.case_id,'normalization');x=c.problem.mesh.sample(queries,s);t=None if pde=='poisson' else s.random(queries)
            for k in accum:accum[k].append(float(np.mean(c.evaluate(x,t,k)**2)))
        values[pde]={k:float(np.sqrt(np.mean(v))) for k,v in accum.items()}
    require(all(np.isfinite(s) and s>0 for d in values.values() for s in d.values()),'invalid G1 training RMS')
    return Normalizers(values,{'split':'train','count_per_pde':count,'area_uniform_queries':queries,'case_ids':ids,'version':VERSION})


@dataclass
class Tokens:
    position: object
    physical_time: object
    time_present: object
    quantity_id: object
    valid: object
    value: object=None
    sigma: object=None
    def to(self,device):
        import torch
        return Tokens(**{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in self.__dict__.items()})
    def slice(self,a,b):return Tokens(**{k:v[a:b] if v is not None else None for k,v in self.__dict__.items()})


def times(n,stream,*,training=False):
    z=stream.random(n);t=stream.random(n)
    t[z<(.125 if training else .1)]=0
    early=(z>=(.125 if training else .1))&(z<(.25 if training else .4))
    t[early]=np.maximum(stream.random(early.sum())*.1,np.nextafter(0.,1.))
    if not training:t[z>=.4]=.1+.9*stream.random((z>=.4).sum())
    return t


def token_set(case,n,normalizers,*,observation=False,set_id=0,strategy='uniform'):
    import torch
    s=rng(case.case_id,'sensor_locations' if observation else 'queries',set_id)
    mesh=case.problem.mesh;x=mesh.sample(n,s,strategy=strategy);present=case.problem.physical_time is not None
    t=times(n,rng(case.case_id,'sensor_times' if observation else 'query_times',set_id),training=not observation) if present else np.zeros(n)
    q=np.zeros(n,dtype=np.int64)
    if case.problem.name=='wave':q=np.arange(n)%2;s.shuffle(q)
    clean=np.zeros(n)
    for k in (0,1):
        select=q==k
        if select.any():clean[select]=case.evaluate(x[select],t[select] if present else None,'v' if k else 'u')
    sigma=None;value=clean.copy()
    if observation:
        eta=np.exp(rng(case.case_id,'noise_levels',set_id).uniform(np.log(.02),np.log(.08),n))
        sigma=eta*np.array([normalizers.scale(case.problem.name,int(k)) for k in q])
        value+=sigma*rng(case.case_id,'measurement_noise',set_id).standard_normal(n)
    tensor=lambda a,dtype=torch.float32:torch.as_tensor(a,dtype=dtype).unsqueeze(0)
    tokens=Tokens(tensor(x),tensor(t[:,None]),tensor(np.full(n,present),torch.bool),tensor(q,torch.long),tensor(np.ones(n),torch.bool),tensor(value[:,None]),None if sigma is None else tensor(sigma[:,None]))
    identity=digest((case.case_id,'observations' if observation else 'queries',set_id,n,strategy,normalizers.identity))
    return tokens,{'identity':identity,'clean':clean,'sigma':sigma,'set_id':set_id,'case_id':case.case_id}


@dataclass
class Batch:
    cases: tuple
    observations: Tokens
    queries: Tokens
    normalizers: Normalizers
    @property
    def specs(self):return tuple(c.problem for c in self.cases)
    @property
    def case_ids(self):return tuple(c.case_id for c in self.cases)
    def slice(self,a,b):return Batch(self.cases[a:b],self.observations.slice(a,b),self.queries.slice(a,b),self.normalizers)
    def to(self,device):return Batch(self.cases,self.observations.to(device),self.queries.to(device),self.normalizers)


def batch(cases,normalizers,*,sensors=None,queries=128,set_id=0,device='cpu'):
    import torch
    cases=tuple(cases);obs=[];qry=[]
    for c in cases:
        n=({'poisson':32,'heat':48,'wave':64}[c.problem.name] if sensors is None else sensors)
        obs.append(token_set(c,n,normalizers,observation=True,set_id=set_id)[0]);qry.append(token_set(c,queries,normalizers,set_id=set_id)[0])
    def pack(items):
        output={};N=max(t.valid.shape[1] for t in items)
        for k in items[0].__dict__:
            if getattr(items[0],k) is None:output[k]=None;continue
            rows=[]
            for t in items:
                v=getattr(t,k);shape=list(v.shape);shape[1]=N-v.shape[1];rows.append(torch.cat((v,torch.zeros(shape,dtype=v.dtype)),1))
            output[k]=torch.cat(rows,0)
        return Tokens(**output)
    return Batch(cases,pack(obs),pack(qry),normalizers).to(device)
