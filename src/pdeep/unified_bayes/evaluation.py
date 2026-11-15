"""Matched CPU64 classical evaluation, resumable per physical case/sensor set."""
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import time as _clock
import numpy as np
import torch
from ..core import digest,require
from ..artifact_io import read_payload,write_artifact,atomic_json,read_manifest,sha256
from ..smooth.bank import physical_design,grid
from ..unified.config import SCHEMA as DATA_SCHEMA
from ..unified.data import Provider,OBSERVATION_COUNTS
from ..unified.evaluation import _metrics
from ..unified.monitor import Dashboard
from .operators import Coordinates,ForwardOperators,prior_specification
from .gaussian import condition
from .interpolation import interpolate,UnsupportedInterpolation
from .calibration import marginal_diagnostics,aggregate_coverage,aggregate_probabilistic

SCHEMA='G1-matched-Bayesian-calibration-evaluation-v1'
CLASSICAL=('Prior mean','Interpolation','Bayesian inverse')
NEURAL=('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001')


def observation_design(c,n,seed):return physical_design(c,n,observation=True,set_id=(DATA_SCHEMA,'fixed-evaluation',seed))

def query_design(c,size):return grid(c,size)

def heldout_design(c,seed):return physical_design(c,128,observation=True,set_id=('unified-heldout-sensors',seed))


def selected_cases(dataset,split,max_cases_per_task=None):
    require(split in ('validation','test'),'unseen physical validation/test split required')
    cases=dataset.cases(split);out=[];counter=defaultdict(int)
    for c in cases:
        key=(c.pde,c.family)
        if max_cases_per_task is None or counter[key]<max_cases_per_task:out.append(c);counter[key]+=1
    return tuple(out)


def prepare_operators(dataset,operators,*,split='test',counts=OBSERVATION_COUNTS,size=64,max_cases_per_task=None,monitor=None):
    """No neural checkpoints/models/optimizer required. Explicit expensive action."""
    cases=selected_cases(dataset,split,max_cases_per_task);monitor=monitor or Dashboard(dataset.config)
    reports=[]
    with monitor.live():
        for p,g in dataset.config.tasks:
            group=[c for c in cases if (c.pde,c.family)==(p,g)]
            if not group:continue
            designs={}
            for c in group:
                for n in counts:designs[f'{c.case_id}/sensors/{n}']=Coordinates.design(observation_design(c,n,dataset.config.seed))
                designs[f'{c.case_id}/query']=Coordinates.design(query_design(c,size));designs[f'{c.case_id}/heldout']=Coordinates.design(heldout_design(c,dataset.config.seed))
            operators.progress=monitor.phase
            monitor.phase('Preparing references',pde=p,geometry=g,physical_cases=len(group),sensor_counts=counts,basis_dimension=prior_specification(p,g)['dimension'],operation='prepare exact matched full-reference sensitivities')
            reports.append(operators.prepare(p,g,designs))
        monitor.phase('Completed',operation='Bayesian operator preparation',tasks=len(reports))
    return reports


def _row(method,c,n,prediction,truth,q,t,norms,obs_id,query_id,cost):
    rows=[]
    for quantity in np.unique(q):
        take=q==quantity;scale=norms.scale(c.pde,int(quantity));r={'method':method,'model':method,'case_id':c.case_id,'split':c.split,
            'pde':c.pde,'family':c.family,'geometry_id':c.problem.mesh.geometry_id,'quantity':'v' if quantity else 'u','sensor_count':n,
            'observation_identity':obs_id,'query_identity':query_id,'cost':cost,**_metrics(prediction[take],truth[take],scale)}
        if c.pde!='poisson':
            r['time_standardized_MSE']={str(float(time)):float(np.mean(((prediction[take&(t==time)]-truth[take&(t==time)])/scale)**2)) for time in np.unique(t)}
        rows.append(r)
    return rows


def evaluate_case(c,dataset,operators,n,*,size=64,query_chunk=2048,save_example=False):
    """Posterior inputs contain public prior,H,y,sigma,offset ONLY. Truth is scored later."""
    dataset.require_case(c);provider=Provider(dataset);seed=dataset.config.seed
    od=observation_design(c,n,seed);qd=query_design(c,size);hd=heldout_design(c,seed)
    ot,oi=provider.token_set(c,n,observation=True,set_id=(DATA_SCHEMA,'fixed-evaluation',seed),coordinates=od)
    ht,hi=provider.token_set(c,128,observation=True,set_id=('unified-heldout-sensors',seed),coordinates=hd)
    oc,qc,hc=map(Coordinates.design,(od,qd,hd));lookup=_clock.perf_counter()
    H,b,om=operators.get(c.pde,c.family,oc);B,bq,qm=operators.get(c.pde,c.family,qc);Bh,bh,hm=operators.get(c.pde,c.family,hc)
    lookup_seconds=_clock.perf_counter()-lookup;y=ot.value[0,:,0].double().numpy();sigma=ot.sigma[0,:,0].double().numpy();prior=prior_specification(c.pde,c.family)
    start=_clock.perf_counter();post=condition(H,y,sigma,prior_mean=prior['mean'],prior_covariance=np.eye(prior['dimension']),offset=b,valid=ot.valid[0].numpy());condition_seconds=_clock.perf_counter()-start
    start=_clock.perf_counter();pred=post.predict(B,offset=bq,chunk=query_chunk);field_seconds=_clock.perf_counter()-start
    start=_clock.perf_counter();future=post.predict(Bh,offset=bh,noise_sigma=ht.sigma[0,:,0].double().numpy(),chunk=query_chunk);heldout_seconds=_clock.perf_counter()-start
    # Hidden reference values first enter here, AFTER posterior calculation.
    truth=provider.cache.evaluate(c,*qd[:3]);held_truth=provider.cache.evaluate(c,*hd[:3]);q,t=qd[2],qd[1]
    query_id=digest((c.case_id,qc.identity));cost={'operator_lookup_seconds':lookup_seconds,'conditioning_seconds':condition_seconds,'field_seconds':field_seconds,'heldout_seconds':heldout_seconds,
          'inference_seconds':condition_seconds+field_seconds,'basis_dimension':prior['dimension'],'operator_preprocessing_batches':list(dict.fromkeys(m['preprocessing_batch_identity'] for m in (om,qm,hm)))}
    rows=_row('Bayesian inverse',c,n,pred['mean'],truth,q,t,dataset.norms,oi['identity'],query_id,cost)
    start=_clock.perf_counter();prior_prediction=bq+B@np.asarray(prior['mean']);prior_seconds=_clock.perf_counter()-start
    rows+=_row('Prior mean',c,n,prior_prediction,truth,q,t,dataset.norms,oi['identity'],query_id,{'inference_seconds':prior_seconds,'scope':'public prior mean; forward matrix shared, no sensor values used'})
    try:
        ip,icost=interpolate(c.pde,c.family,oc,qc,y,sigma)
        rows+=_row('Interpolation',c,n,ip,truth,q,t,dataset.norms,oi['identity'],query_id,icost)
    except UnsupportedInterpolation as error:
        rows.extend({'method':'Interpolation','model':'Interpolation','case_id':c.case_id,'split':c.split,'pde':c.pde,'family':c.family,'quantity':'v' if k else 'u','sensor_count':n,'observation_identity':oi['identity'],'query_identity':query_id,'status':'unsupported','reason':str(error),'standardized_MSE':None} for k in np.unique(q))
    coverage=[]
    for k in np.unique(q):
        for time in [None,*(() if c.pde=='poisson' else np.unique(t))]:
            take=(q==k) if time is None else (q==k)&(t==time)
            coverage.append({'case_id':c.case_id,'pde':c.pde,'family':c.family,'quantity':'v' if k else 'u','sensor_count':n,'time':None if time is None else float(time),
                'uncertainty_target':'latent field','calibration':marginal_diagnostics(truth[take],pred['mean'][take],pred['latent_variance'][take],pit_seed=int(digest((c.case_id,n,int(k),str(time)))[:8],16))})
        take=hd[2]==k
        for name,target,var in (('latent heldout field',held_truth,future['latent_variance']),('future noisy measurement',ht.value[0,:,0].double().numpy(),future['measurement_variance'])):
            coverage.append({'case_id':c.case_id,'pde':c.pde,'family':c.family,'quantity':'v' if k else 'u','sensor_count':n,'time':None,'uncertainty_target':name,
                'calibration':marginal_diagnostics(target[take],future['mean'][take],var[take],pit_seed=int(digest((c.case_id,n,name,int(k)))[:8],16))})
    sensor=post.predict(H,offset=b)['mean'];held=future['mean'];sensor_rows=[]
    for k in np.unique(q):
        a=od[2]==k;v=hd[2]==k;scale=dataset.norms.scale(c.pde,int(k))
        sensor_rows.append({'quantity':'v' if k else 'u','conditioning_noisy_standardized_MSE':float(np.mean(((sensor[a]-y[a])/scale)**2)),
            'heldout_clean_standardized_MSE':float(np.mean(((held[v]-held_truth[v])/scale)**2)),
            'heldout_noisy_standardized_MSE':float(np.mean(((held[v]-ht.value[0,v,0].double().numpy())/scale)**2))})
    example=None
    if save_example:example={'position':torch.from_numpy(qd[0]),'time':torch.from_numpy(t),'quantity':torch.from_numpy(q),'reference':torch.from_numpy(truth.copy()),'posterior_mean':torch.from_numpy(pred['mean']),
        'latent_variance':torch.from_numpy(pred['latent_variance']),'sensor_position':ot.position[0].cpu(),'sensor_quantity':ot.quantity_id[0].cpu()}
    batches={m['preprocessing_batch_identity']:{'seconds':m['batch_preprocessing_seconds'],'designs':m['new_designs_in_batch'],'operator_identity':m['operator_spec']['identity'],**operators.preparation_cost(m['preprocessing_batch_identity'])} for m in (om,qm,hm)}
    return {'rows':rows,'coverage':coverage,'sensors':sensor_rows,'posterior_mean':torch.from_numpy(post.mean),'posterior_covariance':torch.from_numpy(post.covariance),
            'posterior_diagnostics':post.diagnostics,'prior':prior,'operator_batches':batches,'example':example,'query_identity':query_id,'observation_identity':oi['identity'],
            'truth_used_for_conditioning':False,'exactness':'exact linear-Gaussian algebra for FULL discretized G1 operator; numerical discretization uncertainty is not included'}


def evaluate_classical(dataset,operators,root,*,split='test',counts=OBSERVATION_COUNTS,size=64,max_cases_per_task=None,chunk=2048,monitor=None,persist=None):
    require(operators.control is None,'production evaluation refuses a numerical test-control operator')
    monitor=monitor or Dashboard(dataset.config);cases=selected_cases(dataset,split,max_cases_per_task);root=Path(root);paths=[];done=0;timings=[];start=_clock.perf_counter()
    recipe={'schema':SCHEMA,'dataset':dataset.config.identity,'split':split,'case_ids':[c.case_id for c in cases],'counts':list(counts),'size':size,'seed':dataset.config.seed,'operator_source':sha256(Path(__file__).with_name('operators.py')),'evaluation_source':sha256(Path(__file__)),'dependencies':{name:sha256(Path(__file__).with_name(name+'.py')) for name in ('gaussian','interpolation','calibration')}}
    identity=digest(recipe)
    with monitor.live():
        for c in cases:
            for n in counts:
                key=digest((identity,c.case_id,n));path=root/'cases'/(key+'.json')
                monitor.phase('Bayesian evaluation',pde=c.pde,geometry=c.family,cases_completed=done,total_case_sensor_sets=len(cases)*len(counts),sensor_count=n,basis_dimension=32 if c.pde!='wave' else 64,
                    cached_forward_operators=len(operators.loaded),operation='verified result reuse' if path.exists() else 'condition and predict',elapsed_seconds=_clock.perf_counter()-start,
                    estimated_remaining_seconds=(len(cases)*len(counts)-done)*float(np.median(timings)) if len(timings)>=5 else None)
                if path.exists():r,p=read_payload(path,kind='diagnostic_bundle');require(r['evaluation_identity']==identity,'evaluation recipe mismatch')
                else:
                    begin=_clock.perf_counter();p=evaluate_case(c,dataset,operators,n,size=size,query_chunk=chunk,save_example=(c.index==min(v.index for v in cases if (v.pde,v.family)==(c.pde,c.family)) and n==counts[-1]))
                    write_artifact(path.parent,path.stem,{'artifact_type':'diagnostic_bundle','architecture_id':'G1_BAYES_RESULTS','run_id':'G1-Bayesian-evaluation','scientific_schema':SCHEMA,'evaluation_identity':identity,'case_id':c.case_id,'split':split,'sensor_count':n,'recipe':recipe},p,required=('rows','coverage','posterior_mean','posterior_covariance'),idempotent=True)
                    timings.append(_clock.perf_counter()-begin)
                    if persist:persist(path)
                paths.append(str(path));done+=1
                atomic_json(root/('index-'+identity+'.json'),{'schema':SCHEMA,'evaluation_identity':identity,'recipe':recipe,'case_results':paths,'complete':done==len(cases)*len(counts)})
        monitor.phase('Completed',operation='classical/Bayesian evaluation',case_sensor_sets=done)
    return {'identity':identity,'index':str(root/('index-'+identity+'.json')),'cases':len(cases),'case_sensor_sets':done,'seconds':_clock.perf_counter()-start}


def load_classical(index):
    import json
    record=json.loads(Path(index).read_text());require(record['schema']==SCHEMA,'not a Bayesian evaluation index')
    rows=[];coverage=[];examples=[];batches={};paths=record['case_results']
    for path in paths:
        r,p=read_payload(path,kind='diagnostic_bundle');require(r['evaluation_identity']==record['evaluation_identity'],'case result index mismatch')
        rows+=p['rows'];coverage+=p['coverage'];batches.update(p['operator_batches'])
        if p['example'] is not None:examples.append({'case_id':r['case_id'],'sensor_count':r['sensor_count'],'data':p['example'],'row':next(v for v in p['rows'] if v['method']=='Bayesian inverse')})
    return {'rows':rows,'coverage':coverage,'coverage_summary':aggregate_coverage(coverage),'probabilistic_summary':aggregate_probabilistic(coverage),'coverage_by_geometry':{g:aggregate_coverage([r for r in coverage if r['family']==g]) for g in sorted({r['family'] for r in coverage})},'examples':examples,'operator_batches':batches,'index':record}
