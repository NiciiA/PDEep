"""Independent, streaming unseen-case evaluation; no optimizer or checkpoint selection on test."""
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from scipy.linalg import cho_factor,cho_solve
from ..core import digest,require
from ..artifact_io import atomic_json
from ..artifact_tensors import state_identity
from ..geometric.data import Tokens
from ..geometric.physics import weak_loss,WeakConfig
from ..smooth.bank import grid
from ..smooth.definition import random_stream,geometry
from ..representation_checks import sync
from ..learning_diagnostics.consistency import derivative_values,independent_design
from .config import TIMES
from .data import Provider,DEFAULT_COUNTS,OBSERVATION_COUNTS


def aggregate(rows):
    """Equal PDE, then active quantity, then geometry/case; no mixing physical units."""
    groups=defaultdict(list)
    for r in rows:groups[(r['pde'],r['quantity'])].append(r['standardized_MSE'])
    by_quantity={'/'.join(k):float(np.mean(v)) for k,v in sorted(groups.items())}
    by_pde={p:float(np.mean([v for k,v in by_quantity.items() if k.startswith(p+'/')])) for p in sorted({r['pde'] for r in rows})}
    return {'selection_standardized_MSE':float(np.mean(list(by_pde.values()))),'per_PDE':by_pde,'per_PDE_quantity':by_quantity,
            'weighting':'equal PDE; equal active quantity within PDE; equal geometry/case within quantity',
            'physical_RMSE_aggregate':'inapplicable across incompatible quantities'}


def statistics(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['pde'],r['family'],r['quantity'])].append(r['standardized_MSE'])
    out=[]
    for k,values in sorted(groups.items()):
        v=np.array(values);out.append(dict(zip(('model','pde','family','quantity'),k),cases=len(v),mean=float(v.mean()),median=float(np.median(v)),
            standard_deviation=float(v.std(ddof=1)) if len(v)>1 else None,IQR=float(np.quantile(v,.75)-np.quantile(v,.25)),q95=float(np.quantile(v,.95))))
    return out


def paired_intervals(rows,*,seed=42,replicates=1000):
    lookup=defaultdict(dict)
    for r in rows:lookup[r['model']][(r['case_id'],r['quantity'])]=r
    models=sorted(lookup);out=[];rng=np.random.default_rng(seed)
    for a in range(len(models)):
        for b in range(a+1,len(models)):
            A,B=lookup[models[a]],lookup[models[b]];keys=sorted(A.keys()&B.keys());tasks=defaultdict(list)
            for k in keys:tasks[(A[k]['pde'],A[k]['family'],A[k]['quantity'])].append(A[k]['standardized_MSE']-B[k]['standardized_MSE'])
            for task,d in sorted(tasks.items()):
                d=np.asarray(d);samples=np.mean(d[rng.integers(len(d),size=(replicates,len(d)))],axis=1)
                out.append({'models':[models[a],models[b]],'pde':task[0],'family':task[1],'quantity':task[2],'pairs':len(d),
                            'mean_difference':float(d.mean()),'paired_bootstrap_95_CI':np.quantile(samples,[.025,.975]).tolist() if len(d)>1 else None,
                            'scope':'fixed geometry instance; resampled independent physical cases, not query/frame replicates'})
    return out


def _metrics(pred,truth,scale):
    e=pred-truth;den=float(np.mean(truth**2));mse=float(np.mean(e*e))
    return {'standardized_MSE':mse/scale**2,'physical_RMSE':float(np.sqrt(mse)),
            'relative_L2':float(np.sqrt(mse/den)) if den>0 else None,'relative_denominator':'reference space-time RMS, physical units',
            'reference_RMS':float(np.sqrt(den)),'amplitude_ratio':float(np.sqrt(np.mean(pred**2)/den)) if den>0 else None}


def predict(model,c,ot,query,*,chunk=2048):
    device=next(model.parameters()).device;sync(device);start=time.perf_counter()
    with torch.no_grad():
        context=model.prepare_context((c.problem,),ot.to(device));y=model.predict(context,query.to(device),chunk=chunk)
    sync(device);return y[0,:,0].cpu().double().numpy(),time.perf_counter()-start


def _tokens_for_grid(c,provider,size):
    design=grid(c,size);y=provider.cache.evaluate(c,*design[:3])
    from ..smooth.data import tokens
    return tokens(c,len(y),provider.norms,set_id=('unified-grid',size),coordinates=design,clean=y)[0]


def field_rows(model,c,provider,*,seed=42,count=512,grid_size=None,observation_count=None,ablation='real',chunk=2048):
    n=observation_count or DEFAULT_COUNTS[c.pde];sid=('G1-unified-multicase-v1','fixed-evaluation',seed)
    ot,oi=provider.token_set(c,n,observation=True,set_id=sid)
    if ablation=='none':ot.valid[:]=False
    elif ablation=='shuffled':
        for q in (0,1) if c.pde=='wave' else (0,):
            ids=torch.where(ot.quantity_id[0]==q)[0];perm=random_stream('ablation-shuffle',c.case_id,seed,q).permutation(len(ids));ot.value[0,ids]=ot.value[0,ids[perm]].clone()
    elif ablation=='positions':ot,oi=provider.token_set(c,n,observation=True,set_id=(sid,'alternative-positions'))
    elif ablation in ('noise_half','noise_double','independent_noise'):
        coordinates=(ot.position[0].double().numpy(),ot.physical_time[0,:,0].double().numpy(),ot.quantity_id[0].numpy(),c.pde!='poisson')
        # Re-evaluate continuous truth at the serialized float32 positions, so the
        # independent-noise ablation never confuses coordinate rounding with noise.
        ot,oi=provider.token_set(c,n,observation=True,set_id=(sid,ablation),coordinates=coordinates,
                                 noise_multiplier=.5 if ablation=='noise_half' else 2. if ablation=='noise_double' else 1.)
    else:require(ablation=='real','unknown observation ablation')
    qt=_tokens_for_grid(c,provider,grid_size) if grid_size else provider.token_set(c,count,set_id=('unified-validation',seed))[0]
    prediction,seconds=predict(model,c,ot,qt,chunk=chunk)
    sensor,_=predict(model,c,ot,ot,chunk=chunk)
    held,hi=provider.token_set(c,128,observation=True,set_id=('unified-heldout-sensors',seed))
    hp,_=predict(model,c,ot,held,chunk=chunk)
    rows=[]
    for k in (0,1) if c.pde=='wave' else (0,):
        mask=qt.quantity_id[0].numpy()==k;truth=qt.value[0,:,0].double().numpy();scale=provider.norms.scale(c.pde,k)
        measured=(ot.quantity_id[0].numpy()==k)&ot.valid[0].numpy();hm=held.quantity_id[0].numpy()==k
        row={'case_id':c.case_id,'split':c.split,'pde':c.pde,'family':c.family,'geometry_id':c.problem.mesh.geometry_id,
             'quantity':'v' if k else 'u','observations':int(ot.valid.sum()),'ablation':ablation,'inference_seconds':seconds,
             **_metrics(prediction[mask],truth[mask],scale),'sensor_standardized_MSE':float(np.mean(((sensor[measured]-ot.value[0,:,0].double().numpy()[measured])/scale)**2)) if measured.any() else None,
             'sensor_clean_standardized_MSE':float(np.mean(((sensor[measured]-oi['clean'][measured])/scale)**2)) if measured.any() else None,
             'heldout_clean_standardized_MSE':float(np.mean(((hp[hm]-hi['clean'][hm])/scale)**2)),
             'heldout_noisy_standardized_MSE':float(np.mean(((hp[hm]-held.value[0,:,0].double().numpy()[hm])/scale)**2)),
             'observation_identity':oi['identity'],'query_identity':digest((qt.position.tolist(),qt.physical_time.tolist(),qt.quantity_id.tolist()))}
        if c.pde!='poisson':
            t=qt.physical_time[0,:,0].double().numpy();row['time_standardized_MSE']={str(time):float(np.mean(((prediction[mask&np.isclose(t,time,atol=1e-7)]-truth[mask&np.isclose(t,time,atol=1e-7)])/scale)**2)) for time in TIMES if np.any(mask&np.isclose(t,time,atol=1e-7))}
            late=mask&(t>=.75);row['late_standardized_MSE']=float(np.mean(((prediction[late]-truth[late])/scale)**2)) if late.any() else None
        rows.append(row)
    return rows,ot,qt,prediction


def physical_metrics(model,c,provider,ot,*,seed=42,derivative_points=16):
    device=next(model.parameters()).device;ot=ot.to(device);ctx=model.prepare_context((c.problem,),ot)
    w=weak_loss(model,ctx,provider.norms,WeakConfig());out={k:float(v.detach()) for k,v in w.items() if k!='diagnostics'}
    if c.pde=='poisson':out['initial']=None
    out['weak_definition']=w['diagnostics'];del w
    # Authorized auxiliary targets are evaluation targets ONLY.
    x=geometry(c.family,.08).sample(128,random_stream('unified-aux-evaluation',c.case_id,seed));xt=torch.as_tensor(x,device=device,dtype=torch.float32)
    out['auxiliary']={}
    for role,f in c.fields.items():
        with torch.no_grad():value=model.evaluate(ctx[0],xt,None if c.pde=='poisson' else xt.new_zeros((len(xt),1)),role)[:,0].cpu().double().numpy()
        out['auxiliary'][role]=_metrics(value,f(x),provider.norms.scale(c.pde,role))
    if c.pde=='wave':
        # Exclude both reference and neural element facets, reuse verified derivative path.
        provider.cache._get(c,'u0',0);space=provider.cache.spaces[c.family]
        xx,tt=independent_design(c,space,seed=seed,count=derivative_points);v=derivative_values(model,ctx[0],xx,tt)
        L=np.sqrt(c.problem.mesh.cell_areas.sum());scale=provider.norms.scale('wave','v')+c.problem.coefficient*provider.norms.scale('wave','u')/L**2
        out.update(kinematic_physical_RMS=float(np.sqrt(np.mean(v['kinematic']**2))),
                   kinematic_standardized_RMS=float(np.sqrt(np.mean(v['kinematic']**2)))/provider.norms.scale('wave','v'),
                   dynamic_interior_physical_RMS=float(np.sqrt(np.mean(v['dynamic']**2))),dynamic_interior_standardized_RMS=float(np.sqrt(np.mean(v['dynamic']**2)))/scale,
                   strong_scope='element-interior diagnostic only; C0/H1 interface distribution terms excluded; not global PDE certification')
    return out


def gaussian_posterior(H,y,sigma):
    """IID-standard C3 coefficients, strictly positive independent noise, CPU64."""
    H=np.asarray(H,dtype=np.float64);y=np.asarray(y,dtype=np.float64);sigma=np.asarray(sigma,dtype=np.float64)
    require(H.ndim==2 and y.shape==sigma.shape==(len(H),),'invalid Gaussian sensitivity/measurement shape')
    require(np.isfinite(H).all() and np.isfinite(y).all() and np.isfinite(sigma).all() and np.all(sigma>0),
            'Gaussian baseline needs finite values and strictly positive sigma; noiseless conditioning is not implemented')
    W=H/sigma[:,None];precision=np.eye(H.shape[1])+W.T@W;factor=cho_factor(precision,lower=True)
    rhs=W.T@(y/sigma);mean=cho_solve(factor,rhs);cov=cho_solve(factor,np.eye(H.shape[1]))
    residual=float(np.linalg.norm(precision@mean-rhs)/(1+np.linalg.norm(rhs)))
    return mean,cov,{'relative_precision_residual':residual,'precision_condition':float(np.linalg.cond(precision)),
                     'sensitivity_rank':int(np.linalg.matrix_rank(W)) if len(W) else 0,
                     'prior':'iid N(0,1) C3 coefficients, basis already includes physical amplitudes and 1/sqrt32'}


def baselines(c,provider,ot,qt,oi=None):
    """Proper iid C3-coefficient Gaussian estimator using geometry-specific sensitivities.

    Exact finite-dimensional Gaussian algebra for the numerically bounded cached
    P4 forward model; not T1 D and not an exact continuum posterior.
    """
    x=ot.position[0].double().numpy();t=ot.physical_time[0,:,0].double().numpy();q=ot.quantity_id[0].numpy();valid=ot.valid[0].numpy()
    xx=qt.position[0].double().numpy();tt=qt.physical_time[0,:,0].double().numpy();qq=qt.quantity_id[0].numpy();truth=qt.value[0,:,0].double().numpy()
    start=time.perf_counter();unavailable=None
    try:
        H,_=provider.cache.sensitivities(c,x[valid],t[valid],q[valid]);HQ,_=provider.cache.sensitivities(c,xx,tt,qq)
    except Exception as error:
        from .streams import BaselineUnavailable
        if not isinstance(error,BaselineUnavailable):raise
        unavailable=str(error)
    sigma=ot.sigma[0,:,0].double().numpy()[valid];y=ot.value[0,:,0].double().numpy()[valid]
    if unavailable is None:mean,cov,diagnostics=gaussian_posterior(H,y,sigma);gaussian=HQ@mean
    else:gaussian=None;diagnostics={'applicable':False,'reason':unavailable}
    cost=time.perf_counter()-start
    interp_start=time.perf_counter()
    interp=np.zeros(len(xx));z=np.column_stack((x,t));zz=np.column_stack((xx,tt))
    for k in np.unique(qq):
        take=valid&(q==k);target=qq==k
        if take.any():
            d=np.sum((zz[target,None]-z[None,take])**2,-1);weights=1/(d+1e-6);weights/=weights.sum(1,keepdims=True);interp[target]=weights@ot.value[0,take,0].double().numpy()
    interpolation_cost=time.perf_counter()-interp_start
    prior_start=time.perf_counter();prior_prediction=np.zeros_like(truth);prior_cost=time.perf_counter()-prior_start
    rows=[]
    for name,pred in (('zero-prior-mean',prior_prediction),('IDW-space-time',interp),('Gaussian-cached-P4',gaussian)):
        if pred is None:
            rows.append({'model':name,'case_id':c.case_id,'pde':c.pde,'family':c.family,'quantity':'not applicable','standardized_MSE':None,'inference_seconds':None,'applicability':unavailable});continue
        for k in np.unique(qq):
            take=qq==k;rows.append({'model':name,'case_id':c.case_id,'pde':c.pde,'family':c.family,'quantity':'v' if k else 'u',**_metrics(pred[take],truth[take],provider.norms.scale(c.pde,int(k))),
                'inference_seconds':cost if name=='Gaussian-cached-P4' else interpolation_cost if name=='IDW-space-time' else prior_cost,
                'Gaussian_diagnostics':diagnostics if name=='Gaussian-cached-P4' else None,
                'applicability':'geometry-specific C3 prior, known operator/noise, CPU64 bounded-cache linear model; includes H/Hq, mean and full covariance' if name=='Gaussian-cached-P4' else 'declared diagnostic baseline'})
    return rows


def evaluate_model(model,dataset,*,split='validation',model_key='',seed=42,grid_size=64,count=512,diagnostics=True,sensitivity=False,include_baselines=False,max_cases=None,progress=None,familiar_only=False):
    require(split in ('validation','test'),'generalization evaluation requires unseen physical split')
    provider=Provider(dataset);rows=[];physical=[];ablations=[];base=[];previous=model.training;before=state_identity(model)
    cases=dataset.cases(split)
    if familiar_only:cases=tuple(c for c in cases if c.family!=dataset.config.holdout_family)
    cases=cases if max_cases is None else cases[:max_cases];start=time.perf_counter();model.eval()
    try:
        for i,c in enumerate(cases):
            if progress:progress('Validation' if split=='validation' else 'Test evaluation',case=f'{i+1}/{len(cases)}',task=f'{c.pde}/{c.family}',model=model_key)
            r,ot,qt,_=field_rows(model,c,provider,seed=seed,count=count,grid_size=grid_size)
            rows.extend(dict(v,model=model_key,generalization='held-out family' if c.family==dataset.config.holdout_family else 'unseen physical case, familiar fixed instance') for v in r)
            if diagnostics:physical.append({'model':model_key,'case_id':c.case_id,'pde':c.pde,'family':c.family,**physical_metrics(model,c,provider,ot,seed=seed)})
            if sensitivity:
                for n in OBSERVATION_COUNTS:
                    rr,*_=field_rows(model,c,provider,seed=seed,count=count,grid_size=None,observation_count=n)
                    ablations.extend(dict(v,model=model_key) for v in rr)
                for a in ('none','shuffled','positions','noise_half','noise_double','independent_noise'):
                    rr,*_=field_rows(model,c,provider,seed=seed,count=count,grid_size=None,ablation=a)
                    ablations.extend(dict(v,model=model_key) for v in rr)
            if include_baselines:base.extend(baselines(c,provider,ot,qt))
        return {'model':model_key,'split':split,'rows':rows,'aggregate':aggregate(rows),'statistics':statistics(rows),
                'physical':physical,'sensitivity':ablations,'baselines':base,'seconds':time.perf_counter()-start,
                'dataset_identity':dataset.config.identity,'scope':dataset.config.preview()['scope'],
                'coverage':{'evaluated_cases':len(cases),'required_cases':len(dataset.cases(split)),'complete':len(cases)==len(dataset.cases(split))},'training_state_changed':False}
    finally:
        model.train(previous);require(state_identity(model)==before,'evaluation changed model weights')


def save_comparison(results,root):
    root=Path(root);rows=[r for v in results.values() for r in v['rows']]
    baseline_rows=[r for v in results.values() for r in v['baselines'] if r['standardized_MSE'] is not None]
    baseline_summary={name:aggregate([r for r in baseline_rows if r['model']==name]) for name in sorted({r['model'] for r in baseline_rows})}
    out={'models':results,'statistics':statistics(rows),'paired_intervals':paired_intervals(rows),'baseline_summaries':baseline_summary,
         'status':'measured evaluation; interpretation requires recorded training exposure/cost and >=3 independent neural seeds for final comparison'}
    atomic_json(root/'comparison.json',out);return out
