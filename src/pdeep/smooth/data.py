"""Versioned smooth physical cases, independent observations and query banks.

Reference labels use the high-order policy; geometric conditioning/weak tests
use a separate declared mesh. No hidden field is present in Problem/observations.
Calibration is reserved TRAIN-only; validation and test are never calibrated.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from ..core import digest,require
from ..geometric.data import Problem,Tokens,Batch,times
from .definition import *
from .reference import reference

@dataclass(frozen=True)
class Normalizers:
    values: dict
    calibration: dict
    @property
    def identity(self):return digest((BENCHMARK_ID,NORMALIZATION_VERSION,self.values,self.calibration))
    def scale(self,pde,quantity=0):
        key=('v' if int(quantity)==1 else 'u') if isinstance(quantity,(int,np.integer)) else quantity
        return self.values[pde][key]
    def asdict(self):return {'values':self.values,'calibration':self.calibration,'identity':self.identity}
    @classmethod
    def fromdict(cls,d):
        n=cls(d['values'],d['calibration']);require(n.identity==d['identity'],'smooth RMS identity mismatch');return n

@dataclass
class Case:
    family: str
    pde: str
    split: str
    index: int
    conditioning_h: float=.12
    @property
    def problem(self):return Problem(self.pde,geometry(self.family,self.conditioning_h),COEFFICIENTS[self.pde])
    @property
    def case_id(self):return physical_id(self.family,self.pde,self.split,self.index)
    @property
    def fields(self):return physical_fields(self.family,self.pde,self.split,self.index)
    @property
    def manifest(self):
        return {'benchmark':BENCHMARK_ID,'physical_case_id':self.case_id,'split':self.split,'index':self.index,'family':self.family,
                'geometry_id':self.problem.mesh.geometry_id,'conditioning_mesh_id':self.problem.mesh.mesh_id,
                'reference_policy':POLICY,'fields':{k:v.identity for k,v in self.fields.items()}}
    def evaluate(self,x,time=None,quantity='u'):
        require((time is None)==(self.pde=='poisson'),'stationary time must be absent; dynamics require physical time')
        if quantity in self.fields:return self.fields[quantity](x)
        r=reference(self.family,POLICY['reference_h'],POLICY['reference_degree']);f=self.fields
        if self.pde=='poisson':
            require(quantity=='u','unsupported stationary quantity');return r.poisson([f['f']],x,POLICY['source_quadrature'])[0][0]
        require(quantity in ({'u','v'} if self.pde=='wave' else {'u'}),'unknown dynamic quantity')
        u,v,_=r.dynamic(self.pde,[f['u0']],[f['v0']] if self.pde=='wave' else None,x,np.asarray(time),POLICY['reference_dt'])
        return (v if quantity=='v' else u)[0]


def case(split,pde,index,*,family=None,conditioning_h=.12):
    allowed=HELD_OUT if split=='geometry_held_out' else SEEN
    family=allowed[index%len(allowed)] if family is None else family
    require(family in allowed,'geometry split leakage: selected instance is not in this split')
    physical_id(family,pde,split,index)
    return Case(family,pde,split,index,conditioning_h)


def design(c,n,*,observation=False,set_id=0,strategy='uniform'):
    require(type(n) is int and n>0,'positive point count required')
    role='sensor' if observation else 'query';x=c.problem.mesh.sample(n,random_stream(c.case_id,role,'position',set_id),strategy=strategy)
    present=c.pde!='poisson';t=times(n,random_stream(c.case_id,role,'time',set_id),training=not observation) if present else np.zeros(n)
    q=np.arange(n,dtype=np.int64)%2 if c.pde=='wave' else np.zeros(n,dtype=np.int64)
    random_stream(c.case_id,role,'quantity',set_id).shuffle(q)
    return x,t,q,present


def evaluate_pairs(c,x,t,q):
    # One trajectory per case, not one per quantity. Fixed time grid independent
    # of query partition; spatial element basis and v=du/dt are declared.
    f=c.fields;r=reference(c.family,POLICY['reference_h'],POLICY['reference_degree'])
    if c.pde=='poisson':return r.poisson([f['f']],x,POLICY['source_quadrature'])[0][0]
    u,v,_=r.dynamic(c.pde,[f['u0']],[f['v0']] if c.pde=='wave' else None,x,t,POLICY['reference_dt'])
    return np.where(q==1,v[0],u[0]) if v is not None else u[0]


def tokens(c,n,norms,*,observation=False,set_id=0,coordinates=None,clean=None):
    import torch
    x,t,q,present=design(c,n,observation=observation,set_id=set_id) if coordinates is None else coordinates
    clean=evaluate_pairs(c,x,t,q) if clean is None else np.asarray(clean)
    value=clean.copy();sigma=None
    if observation:
        eta=np.exp(random_stream(c.case_id,'noise_levels',set_id).uniform(np.log(.02),np.log(.08),len(x)))
        sigma=eta*np.array([norms.scale(c.pde,int(k)) for k in q]);value+=sigma*random_stream(c.case_id,'measurement_noise',set_id).standard_normal(len(x))
    tensor=lambda a,dtype=torch.float32:torch.as_tensor(a,dtype=dtype).unsqueeze(0)
    out=Tokens(tensor(x),tensor(t[:,None]),tensor(np.full(len(x),present),torch.bool),tensor(q,torch.long),tensor(np.ones(len(x)),torch.bool),tensor(value[:,None]),None if sigma is None else tensor(sigma[:,None]))
    identity=digest((c.case_id,'observation' if observation else 'query',set_id,x.tolist(),t.tolist(),q.tolist(),POLICY,norms.identity))
    return out,{'identity':identity,'clean':clean,'sigma':sigma,'set_id':set_id,'physical_case_id':c.case_id,'reference_policy':POLICY}


def pack(items):
    import torch
    output={};N=max(t.valid.shape[1] for t in items)
    for k in items[0].__dict__:
        if getattr(items[0],k) is None:output[k]=None;continue
        rows=[]
        for t in items:
            v=getattr(t,k);shape=list(v.shape);shape[1]=N-v.shape[1];rows.append(torch.cat((v,torch.zeros(shape,dtype=v.dtype,device=v.device)),1))
        output[k]=torch.cat(rows,0)
    return Tokens(**output)


def calibrate(progress=print):
    """Frozen 4 TRAIN-reserved fields/PDE, equal geometry/area/time RMS.

    Wave u/v retain the published C3 TRAIN calibration exactly. Source and heat
    are newly calibrated before ANY certification/validation errors are read.
    The finite calibration estimator is part of the normalization identity.
    """
    values={};rows=[];time,w=np.polynomial.legendre.leggauss(8);time=(time+1)/2;w=w/2
    for pde in PDES:
        accum={k:[] for k in (('u','f') if pde=='poisson' else ('u','u0','v','v0') if pde=='wave' else ('u','u0'))}
        for family in SEEN:
            progress(f'calibration {pde} {family}')
            # Old wave physical keys remain EXACT train-calibration,0.
            f=physical_fields(family,pde,'train-calibration' if pde=='wave' else 'calibration',0)
            mesh=geometry(family);x=mesh.sample(256,random_stream('normalization',pde,family,'position'))
            r=reference(family,.02,4);xx=np.tile(x,(8,1));tt=np.repeat(time,len(x))
            if pde=='poisson':u=r.poisson([f['f']],x)[0][0];accum['u'].append(float(np.mean(u*u)))
            elif pde=='heat':
                u,_,_=r.dynamic(pde,[f['u0']],None,xx,tt,.002);accum['u'].append(float(w@np.mean(u[0].reshape(8,-1)**2,axis=1)))
            # Published wave u/v are frozen, not recalibrated for new test errors.
            for q in f:accum[q].append(float(np.mean(f[q](x)**2)))
            rows.append({'pde':pde,'family':family,'split':'reserved TRAIN calibration','field_ids':{k:v.identity for k,v in f.items()}})
        values[pde]={q:float(np.sqrt(np.mean(a))) for q,a in accum.items() if a}
        if pde=='wave':values[pde].update(u=.07565609305304831,v=.9261384252564839)
    require(all(s>0 and np.isfinite(s) for p in values.values() for s in p.values()),'invalid training-only RMS')
    return Normalizers(values,{'version':NORMALIZATION_VERSION,'split':'TRAIN-only reserved calibration','cases_per_pde':4,'positions':256,'time_Gauss_order':8,'wave_u_v':'frozen published C3 calibration; unchanged','rows':rows})
