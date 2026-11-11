"""Frozen TRAIN observations, finite training queries and independent point labels.

Reference evaluations delegate to the certified generator at the exact requested
continuous physical times. No trajectory-frame selection, new prior or calibration.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from ..core import digest, require
from ..artifact_io import read_payload, write_artifact
from ..artifact_tensors import state_identity
from ..geometric.data import Tokens, Batch
from ..smooth.data import Case, tokens, evaluate_pairs
from ..smooth.bank import physical_design
from ..smooth.definition import BENCHMARK_ID, SEEN

EXPERIMENTS=('poisson_square','poisson_rectangle','heat_single','wave_single','c_physics_ablation','observation_sensitivity')
TIMES=(0.,.05,.25,.5,1.)


def token_id(t):
    return digest({k:None if v is None else v.detach().cpu().tolist() for k,v in vars(t).items()})


def select(t,indices):
    return Tokens(**{k:None if v is None else v[:,indices].clone() for k,v in vars(t).items()})


def diagnostic_grid(c,size):
    mesh=c.problem.mesh; lo,hi=mesh.vertices.min(0),mesh.vertices.max(0)
    x=np.stack(np.meshgrid(*[np.linspace(lo[k],hi[k],size) for k in range(2)],indexing='ij'),-1).reshape(-1,2)
    x=x[mesh.features(x)['inside']]; x=x[mesh.features(x)['distance']>1e-7]
    times=(0.,) if c.pde=='poisson' else TIMES; quantities=(0,1) if c.pde=='wave' else (0,)
    return np.tile(x,(len(times)*len(quantities),1)),np.tile(np.repeat(times,len(x)),len(quantities)),np.repeat(quantities,len(times)*len(x)),c.pde!='poisson'


@dataclass
class FixedData:
    case: object
    observations: Tokens
    training: Tokens
    diagnostic: Tokens
    clean_observations: torch.Tensor
    normalizers: object
    identities: dict
    path: str
    @property
    def identity(self): return digest(self.identities)
    def batch(self,device='cpu',diagnostic=False):
        return Batch((self.case,),self.observations,self.diagnostic if diagnostic else self.training,self.normalizers).to(device)
    def assert_frozen(self):
        for k,t in [('observation',self.observations),('training',self.training),('diagnostic',self.diagnostic)]:
            require(token_id(t)==self.identities[k],f'{k} changed in frozen experiment')


def prepare_data(root,record,*,experiment='poisson_square',seed=42,count=128,grid_size=64,conditioning_h=.12,pde=None,family=None,paired=False):
    require(experiment in EXPERIMENTS,'unknown diagnostic experiment')
    defaults={'poisson_rectangle':('poisson','rectangle'),'heat_single':('heat','square'),'wave_single':('wave','square')}
    dp,dg=defaults.get(experiment,('poisson','square')); pde=pde or dp; family=family or dg
    require(family in SEEN and pde in {'poisson','heat','wave'},'diagnostics require seen geometry and certified PDE')
    from ..smooth.certification import authorize,normalizers
    authorize(record,pdes=(pde,),families=(family,)); norms=normalizers(record)
    index=4*((seed+int(paired))%3000)+SEEN.index(family)
    c=Case(family,pde,'train',index,conditioning_h)
    description={'schema':'fixed-TRAIN-learning-data-v1','case':c.manifest,'seed':seed,'observation_set':seed,'training_queries':count,'independent_grid_size':grid_size,'times':TIMES,'normalizers':norms.identity,'certificate':record['certification_id'],'query_policy':'strict fixed main queries, independent continuous-time grid'}
    key=digest(description); root=Path(root); path=root/(key+'.json')
    if path.exists():
        meta,p=read_payload(path,kind='diagnostic_bundle',required=('observation','training','diagnostic','clean_observations','identities'))
        require(meta['data_description']==__import__('json').loads(__import__('json').dumps(description)),'cached diagnostic data mismatch')
    else:
        designs=[physical_design(c,{'poisson':32,'heat':48,'wave':64}[pde],observation=True,set_id=seed),physical_design(c,count,set_id=10000+seed),diagnostic_grid(c,grid_size)]
        sizes=[len(d[0]) for d in designs]; require(min(sizes)>0,'diagnostic grid contains no interior points')
        # All labels use the same actual continuous initial field and reference policy.
        clean=evaluate_pairs(c,*[np.concatenate([d[k] for d in designs]) for k in range(3)])
        chunks=np.split(clean,np.cumsum(sizes)[:-1]); ts=[]
        for j,(d,y) in enumerate(zip(designs,chunks)):
            ts.append(tokens(c,len(y),norms,observation=j==0,set_id=seed if j==0 else 10000+seed+j-1,coordinates=d,clean=y)[0])
        ids={'description':key,'physical_case':c.case_id,'geometry':c.problem.mesh.geometry_id,'observation_set':seed,**{k:token_id(t) for k,t in zip(('observation','training','diagnostic'),ts)}}
        p={**{k:vars(t) for k,t in zip(('observation','training','diagnostic'),ts)},'clean_observations':torch.tensor(chunks[0],dtype=torch.float32)[None,:,None],'identities':ids}
        write_artifact(root,key,{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'fixed-TRAIN-diagnostics','benchmark':BENCHMARK_ID,'data_description':description},p,required=tuple(p),idempotent=True)
    result=FixedData(c,*[Tokens(**p[k]) for k in ('observation','training','diagnostic')],p['clean_observations'],norms,p['identities'],str(path))
    result.assert_frozen(); return result


class FixedSampler:
    def __init__(self,data,config): self.data=data; self.config=config
    def batch(self,cursor):
        self.data.assert_frozen()
        return self.data.batch(self.config.device),{'fixed_data':self.data.identity,'query_policy':'strict memorization; no resampling','cursor':cursor}
    def state_dict(self,cursor):
        return {'schema':'fixed-TRAIN-sampler-v1','cursor':cursor,'data_identity':self.data.identity,'config':self.config.identity}
