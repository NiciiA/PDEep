"""Unseen physical cases, matched information views; no optimization."""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from ..core import digest,require
from ..artifact_io import write_artifact,atomic_json,read_payload
from ..artifact_tensors import state_identity
from ..smooth.definition import random_stream
from ..smooth.data import tokens
from ..unified.data import Provider
from ..unified.evaluation import predict,physical_metrics,_metrics
from ..unified_bayes.evaluation import observation_design,query_design
from ..unified_bayes.operators import Coordinates
from ..unified_bayes.calibration import marginal_diagnostics
from .conditioning import FieldCondition,CaseView,MODES,SCHEMA,mask_for
from .bayes import conditional


def evaluation_views(c,seed=42,*,full=False):
    d=64 if c.pde=='wave' else 32
    yield 'FORWARD',np.ones(d,bool),0,'all'
    for n in ((8,16,32,64,128) if full else (64,)):yield 'INVERSE',np.zeros(d,bool),n,'none'
    patterns=('random','group','u0','v0') if full and c.pde=='wave' else ('random','group') if full else ('random',)
    for pattern in patterns:
        fractions=(.5,) if pattern in ('u0','v0') or not full else (.25,.5,.75)
        order=random_stream(SCHEMA,'evaluation-known',seed,c.case_id,pattern).permutation(d)
        for fraction in fractions:
            for n in ((0,8,16,32,64,128) if full else (64,)):
                mask=np.zeros(d,bool)
                if pattern in ('u0','v0'):mask[:32]=pattern=='u0';mask[32:]=pattern=='v0'
                elif pattern=='group':mask[:int(d*fraction)]=True
                else:mask[order[:int(d*fraction)]]=True
                yield 'HYBRID',mask,n,pattern


def empty_observations(c,norms):
    # Construct a genuine zero-token set, not hidden noisy values behind a mask.
    design=(np.empty((0,2)),np.empty(0),np.empty(0,dtype=np.int64),c.pde!='poisson')
    return tokens(c,0,norms,observation=True,coordinates=design,clean=np.empty(0))[0]


def information_view(c,dataset,mode,mask,n):
    condition=FieldCondition.from_case(c,mask);view=CaseView(c,condition,mode)
    if n:
        design=observation_design(c,n,dataset.config.seed)
        obs,meta=Provider(dataset).token_set(c,n,observation=True,set_id=('G1-unified-multicase-v1','fixed-evaluation',dataset.config.seed),coordinates=design)
    else:obs=empty_observations(c,dataset.norms);meta={'identity':digest((c.case_id,'empty observations'))}
    return view,obs,meta


def aggregate(rows):
    groups=defaultdict(list)
    # Macro tasks then quantities, equal modes for selection; never mix physical RMSE.
    for r in rows:groups[(r['mode'],r['pde'],r['family'],r['quantity'])].append(r['standardized_MSE'])
    bymode={};bypdequantity={}
    for mode in MODES:
        quantities={}
        for p,q in sorted({(k[1],k[3]) for k in groups if k[0]==mode}):quantities[p+'/'+q]=float(np.mean([np.mean(v) for k,v in groups.items() if k[0]==mode and k[1]==p and k[3]==q]))
        pdes={p:float(np.mean([v for k,v in quantities.items() if k.startswith(p+'/')])) for p in {k.split('/')[0] for k in quantities}}
        bymode[mode]=float(np.mean(list(pdes.values()))) if pdes else None;bypdequantity[mode]=quantities
    return {'selection_standardized_MSE':float(np.mean([v for v in bymode.values() if v is not None])),'per_mode':bymode,'per_mode_PDE_quantity':bypdequantity,
        'per_PDE':{p:float(np.mean([v for qs in bypdequantity.values() for k,v in qs.items() if k.startswith(p+'/')])) for p in sorted({r['pde'] for r in rows})},
        'per_PDE_quantity':{pq:float(np.mean([qs[pq] for qs in bypdequantity.values() if pq in qs])) for pq in sorted({k for qs in bypdequantity.values() for k in qs})},
        'weighting':'equal modes, equal PDE, equal active quantities, equal geometries/cases; hybrid design cells equal within case; physical units kept separate'}


def evaluate(model,dataset,*,split='validation',model_key='',full=False,grid_size=None,query_count=512,max_cases_per_task=2,diagnostics=False,progress=None,operators=None,root=None,persist=None):
    require(split in ('validation','test'),'unseen physical cases only');require(not full or split=='test','expanded sweeps reserved for explicit test evaluation')
    before=state_identity(model);was=model.training;model.eval();rows=[];examples=[];physics_rows=[];cases=[];counter=defaultdict(int);provider=Provider(dataset);paths=[];view_count=0;last_progress=-float('inf');last_task=None;start=time.perf_counter()
    for c in dataset.cases(split):
        key=(c.pde,c.family)
        if max_cases_per_task is None or counter[key]<max_cases_per_task:cases.append(c);counter[key]+=1
    try:
        for c in cases:
            qd=query_design(c,grid_size) if grid_size else None
            raw_truth=provider.cache.evaluate(c,*qd[:3]) if qd else None
            qt=tokens(c,len(qd[0]),dataset.norms,coordinates=qd,clean=raw_truth)[0] if qd else provider.token_set(c,query_count,set_id=('unified-validation',dataset.config.seed))[0]
            truth=raw_truth if qd else qt.value[0,:,0].double().numpy();qid=digest((c.case_id,Coordinates.design(qd).identity)) if qd else digest((qt.position.tolist(),qt.physical_time.tolist(),qt.quantity_id.tolist()))
            from .config import extension_identity
            operator_id=operators.spec(c.pde,c.family)['identity'] if operators is not None else None
            case_identity=digest((SCHEMA,extension_identity(),model_key,before,dataset.config.identity,c.case_id,qid,full,diagnostics,operator_id))
            case_path=Path(root)/'physical-cases'/(case_identity+'.json') if root else None
            if case_path and case_path.exists():
                record,cached=read_payload(case_path,kind='diagnostic_bundle',required=('rows','examples','physics','views'))
                require(record['case_identity']==case_identity,'cached physical information-view case mismatch')
                rows.extend(cached['rows']);examples.extend(cached['examples']);physics_rows.extend(cached['physics']);view_count+=cached['views'];paths.append(str(case_path))
                if progress and (len(paths)%10==0 or time.perf_counter()-last_progress>=2):
                    last_progress=time.perf_counter();progress('Evaluating information modes',model=model_key,pde=c.pde,geometry=c.family,completed_views=view_count,operation='verified physical-case reuse; no prediction')
                atomic_json(Path(root)/'index.json',{'schema':SCHEMA,'case_results':paths,'model':model_key,'split':split,'model_weights':before})
                continue
            row_start,example_start,physics_start,view_start=len(rows),len(examples),len(physics_rows),view_count
            for mode,mask,n,pattern in evaluation_views(c,dataset.config.seed,full=full):
                view,ot,meta=information_view(c,dataset,mode,mask,n)
                report_progress=progress is not None and (last_task!=(c.pde,c.family) or view_count%25==0 or time.perf_counter()-last_progress>=2)
                if report_progress:
                    last_progress=time.perf_counter();last_task=(c.pde,c.family)
                    progress('Evaluating information modes',model=model_key,pde=c.pde,geometry=c.family,mode=mode,known_fraction=view.condition.known_fraction,observations=n,completed_views=view_count,operation='prediction; no optimizer')
                prediction,seconds=predict(model,view,ot,qt,chunk=2048);rr=[]
                for q in (0,1) if c.pde=='wave' else (0,):
                    take=qt.quantity_id[0].numpy()==q;t=qt.physical_time[0,:,0].double().numpy();scale=dataset.norms.scale(c.pde,q)
                    row={'method':model_key,'model':model_key,'case_id':c.case_id,'split':split,'pde':c.pde,'family':c.family,'quantity':'v' if q else 'u','mode':mode,'known_fraction':view.condition.known_fraction,'known_pattern':pattern,'sensor_count':n,
                        'field_condition_identity':view.condition.identity,'observation_identity':meta['identity'],'query_identity':qid,'inference_seconds':seconds,'cost':{'inference_seconds':seconds,'device':str(next(model.parameters()).device),'dtype':'float32','includes_context_encoding':True},**_metrics(prediction[take],truth[take],scale)}
                    if c.pde!='poisson':
                        row['late_standardized_MSE']=float(np.mean(((prediction[take&(t>=.75)]-truth[take&(t>=.75)])/scale)**2)) if np.any(take&(t>=.75)) else None
                        row['time_standardized_MSE']={str(float(tt)):float(np.mean(((prediction[take&np.isclose(t,tt,atol=1e-7)]-truth[take&np.isclose(t,tt,atol=1e-7)])/scale)**2)) for tt in (0,.02,.05,.1,.2,.35,.5,.75,1) if np.any(take&np.isclose(t,tt,atol=1e-7))}
                    rr.append(row)
                physics=physical_metrics(model,view,provider,ot,seed=dataset.config.seed) if diagnostics and n in (0,64) and pattern in ('all','none','random') and view.condition.known_fraction in (0.,.5,1.) else None
                if operators is not None:
                    require(qd is not None,'Bayesian comparison requires explicit matched grid')
                    od=observation_design(c,n,dataset.config.seed) if n else None;B,bq,_=operators.get(c.pde,c.family,Coordinates.design(qd));H,b,_=operators.get(c.pde,c.family,Coordinates.design(od)) if n else (np.empty((0,B.shape[1])),np.empty(0),None)
                    a,known=view.condition.coefficients();post=conditional(H,ot.value[0,:,0].double().numpy(),ot.sigma[0,:,0].double().numpy(),a,known,offset=b);bp=post.predict(B,offset=bq)
                    for row in list(rr):
                        q=1 if row['quantity']=='v' else 0;take=qt.quantity_id[0].numpy()==q
                        rr.append({**row,'method':'Numerical forward' if mode=='FORWARD' else 'Conditional Bayesian' if mode=='HYBRID' else 'Bayesian inverse',**_metrics(bp['mean'][take],truth[take],dataset.norms.scale(c.pde,q)),
                            'calibration':marginal_diagnostics(truth[take],bp['mean'][take],bp['latent_variance'][take]),'inference_seconds':None,'timing_scope':'separately use exact baseline profiler; neural time not reassigned'})
                example=None
                representative=c.case_id==next(v.case_id for v in cases if v.pde==c.pde)
                if qd is not None and representative and (mode=='FORWARD' or n==64 and (mode=='INVERSE' or pattern=='random' and view.condition.known_fraction==.5)):
                    example={'mode':mode,'pde':c.pde,'family':c.family,'case_id':c.case_id,'position':torch.from_numpy(qd[0]),'time':torch.from_numpy(qd[1]),'quantity':torch.from_numpy(qd[2]),'reference':torch.from_numpy(np.asarray(truth).copy()),'prediction':torch.from_numpy(prediction),'sensor_position':ot.position[0].cpu(),'sensor_quantity':ot.quantity_id[0].cpu(),'sensor_valid':ot.valid[0].cpu()}
                payload={'example':example,'rows':rr,'physics_residuals_separate':physics,'optimizer_updates':0,'field_information_schema':SCHEMA}
                if payload.get('example') is not None:examples.append(payload['example'])
                if physics is not None:physics_rows.append({'case_id':c.case_id,'method':model_key,'mode':mode,'known_fraction':view.condition.known_fraction,'sensor_count':n,'known_pattern':pattern,'metrics':physics})
                rows.extend(rr);view_count+=1
            if case_path:
                case_payload={'rows':rows[row_start:],'examples':examples[example_start:],'physics':physics_rows[physics_start:],'views':view_count-view_start,'field_information_schema':SCHEMA,'optimizer_updates':0}
                write_artifact(case_path.parent,case_path.stem,{'artifact_type':'diagnostic_bundle','architecture_id':model.config.architecture,'run_id':'G1-three-mode-evaluation','case_identity':case_identity,'split':split,'source':extension_identity()},case_payload,required=('rows','examples','physics','views'),idempotent=True)
                if persist:persist(case_path)
                paths.append(str(case_path));atomic_json(Path(root)/'index.json',{'schema':SCHEMA,'case_results':paths,'model':model_key,'split':split,'model_weights':before})
    finally:model.train(was)
    require(state_identity(model)==before,'evaluation modified model');return {'rows':rows,'aggregate':aggregate([r for r in rows if r['method']==model_key]),'seconds':time.perf_counter()-start,'optimizer_updates':0,'paths':paths,'examples':examples,'physics_residuals_separate':physics_rows}
