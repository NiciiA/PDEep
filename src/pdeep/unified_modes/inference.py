"""Minimal checkpoint-only playground. No numerical solve unless explicitly asked."""
from pathlib import Path
import numpy as np
import torch
from ..core import require
from ..artifact_io import read_payload
from ..smooth.data import Case,Normalizers,tokens
from ..smooth.certification import authorize
from ..unified.evaluation import predict
from ..unified_bayes.evaluation import query_design
from .conditioning import FieldCondition,CaseView,SCHEMA
from .config import extension_identity
from .model import create
from .evaluation import empty_observations
from .architecture import inventory,verify_definition
from .compatibility import approved_extension


def load(path,certificate,*,device='cpu',selection='best',expected_capacity=None):
    record,p=read_payload(path,kind='training_checkpoint',required=('model','normalizers','best_validation_model'))
    cfg=record['config'];require(cfg.get('conditioning_schema')==SCHEMA,'inverse-only checkpoint cannot run the new field pathway; explicit fine-tuning required')
    require(expected_capacity is None or cfg['capacity']==expected_capacity,'selected capacity differs from checkpoint; choose an explicit compatible checkpoint')
    require(approved_extension(cfg['extension_source'],cfg['capacity']),'incompatible field encoder/source schema');norms=Normalizers.fromdict(p['normalizers']);authorize(certificate,pdes=tuple(cfg['pdes']),families=tuple(v[1] for v in p['sampler']['tasks']),normalizer_identity=norms.identity)
    require(selection in ('best','latest'),'select validation best/latest explicitly');weights=p['best_validation_model'] if selection=='best' else p['model'];require(weights is not None,'no best weights saved')
    model=create(cfg['architecture'],norms,cfg['capacity'],seed=cfg['root_seed'],device=device);info=verify_definition(record,model);model.load_state_dict(weights,strict=True);model.eval();model.training_successful_updates=p['training_state']['best_validation_step'] if selection=='best' else record['successful_update'];return model,{'config':cfg,'architecture_inventory':info,'manifest':str(path),'artifact':record['artifact_id'],'selection':selection,'successful_updates':model.training_successful_updates,'parents':record.get('parents',[]),'status':'approximate neural predictor; not a certified numerical solver'}


def infer(model,pde,family,known_values,available,*,observations=None,mode='HYBRID',coordinates=None,grid_size=64,query_chunk=2048):
    c=Case(family,pde,'validation',0);f=FieldCondition.from_coefficients(pde,family,known_values,available)
    require(mode in ('FORWARD','INVERSE','HYBRID'),'unsupported mode label');require(mode!='FORWARD' or f.complete,'forward requires all source/initial coefficients, not only scalar parameters')
    obs=empty_observations(c,model.normalizers) if observations is None else observations;view=CaseView(c,f,mode);design=query_design(c,grid_size) if coordinates is None else coordinates
    require(bool(design[3])==(pde!='poisson'),'stationary time absent; dynamic physical time required')
    from ..unified_bayes.operators import Coordinates
    Coordinates.design(design).validate(pde,family)
    require(obs.valid.shape[0]==1,'single-case inference requires one observation row')
    take=obs.valid[0]
    if bool(take.any()):
        require(bool((obs.time_present[0,take]==(pde!='poisson')).all()),'observation physical-time presence differs')
        Coordinates(obs.position[0,take].detach().cpu().double().numpy(),None if pde=='poisson' else obs.physical_time[0,take,0].detach().cpu().double().numpy(),obs.quantity_id[0,take].detach().cpu().numpy()).validate(pde,family)
    qt=tokens(c,len(design[0]),model.normalizers,coordinates=design,clean=np.zeros(len(design[0])))[0]
    require(type(query_chunk) is int and query_chunk>0,'query_chunk must be positive')
    y,seconds=predict(model,view,obs,qt,chunk=query_chunk)
    return {'pde':pde,'family':family,'time_present':pde!='poisson','position':design[0],'time':design[1],'quantity':design[2],'prediction':y,'model_successful_updates':getattr(model,'training_successful_updates',None),'mode_label':mode,'known_fraction':f.known_fraction,'field_condition':f,'observations':obs,'seconds':seconds,'status':'fully specified' if f.complete else 'statistically underdetermined point prediction; not a posterior sample or certified solve'}


def plot(result,*,physical_time=.2,quantity=0,reference=None):
    import matplotlib.pyplot as plt
    t=result['time'];selected=float(np.unique(t)[np.argmin(abs(np.unique(t)-physical_time))]);take=(t==selected)&(result['quantity']==quantity);x=result['position'][take];y=result['prediction'][take]
    cols=3 if reference is not None else 1;fig,axes=plt.subplots(1,cols,figsize=(4*cols,3.5),squeeze=False);values=[y] if reference is None else [np.asarray(reference)[take],y,abs(y-np.asarray(reference)[take])]
    for ax,title,v in zip(axes[0],('Neural prediction',) if reference is None else ('Numerical reference','Neural prediction','Absolute error'),values):
        artist=ax.scatter(x[:,0],x[:,1],c=v,s=5);fig.colorbar(artist,ax=ax);obs=result['observations'];mask=(obs.quantity_id[0]==quantity)&obs.valid[0];sensor=obs.position[0,mask].cpu().numpy();ax.scatter(*sensor.T,c='red',s=12,marker='x');ax.set_aspect('equal');ax.set_title(f"{title}; "+(f't={selected:g}; ' if result['time_present'] else 'time absent; ')+('v' if quantity else 'u'))
    fig.tight_layout();return fig


def generate_example(pde,family,certificate,*,mode='INVERSE',known_fraction=.5,sensors=32,seed=42):
    """EXPLICIT expensive numerical observation generation, never a training call."""
    from ..smooth.data import tokens
    from .conditioning import mask_for
    authorize(certificate,pdes=(pde,),families=(family,),normalizer_identity=certificate['normalizers']['identity'])
    norms=Normalizers.fromdict(certificate['normalizers']);c=Case(family,pde,'validation',0);rng=np.random.default_rng(seed)
    mask,_=mask_for(pde,mode,rng,fraction=known_fraction);a=np.concatenate([c.fields[r].coefficients for r in __import__('pdeep.unified_bayes.operators',fromlist=['ROLES']).ROLES[pde]])
    condition=FieldCondition.from_coefficients(pde,family,a,mask)
    obs=empty_observations(c,norms) if sensors==0 or mode=='FORWARD' else tokens(c,sensors,norms,observation=True,set_id=('explicit-playground-example',seed))[0]
    known,available=condition.coefficients()
    return {'known_values':known,'available':available,'observations':obs,'reference_coefficients':a,'case_id':c.case_id,'reference_scope':'reference_coefficients are optional scoring targets, never neural inputs'}


def numerical_reference(pde,family,coefficients,coordinates,certificate):
    """Explicit original FULL CPU64 solve for a completely specified C3 field."""
    from ..unified_bayes.operators import prior_specification,ROLES
    from ..wave_fidelity.fields import SmoothField
    from ..smooth.reference import reference
    from ..smooth.definition import POLICY
    authorize(certificate,pdes=(pde,),families=(family,),normalizer_identity=certificate['normalizers']['identity'])
    spec=prior_specification(pde,family);a=np.asarray(coefficients,dtype=np.float64);require(a.shape==(spec['dimension'],) and np.isfinite(a).all(),'complete exact physical inputs required for numerical reference')
    fields={r:[SmoothField(np.asarray(spec['centers']),spec['radius'],a[i*32:(i+1)*32],spec['geometry_id'])] for i,r in enumerate(ROLES[pde])};r=reference(family,POLICY['reference_h'],POLICY['reference_degree']);x,t,q,present=coordinates
    require(bool(present)==(pde!='poisson'),'physical-time presence mismatch')
    if pde=='poisson':return r.poisson(fields['f'],x,POLICY['source_quadrature'])[0][0]
    u,v,_=r.dynamic(pde,fields['u0'],fields.get('v0'),x,t,POLICY['reference_dt']);return np.where(q==1,v[0],u[0]) if pde=='wave' else u[0]
