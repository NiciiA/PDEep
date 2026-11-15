"""Standalone forward/conditional-prior/conditional-posterior views of exact G1 H."""
from pathlib import Path
import time
import numpy as np
from ..core import digest,require
from ..artifact_io import write_artifact,read_payload,atomic_json
from ..unified.data import Provider
from ..unified.evaluation import _metrics
from ..unified_bayes.operators import Coordinates
from ..unified_bayes.evaluation import selected_cases,query_design,observation_design
from ..unified_bayes.calibration import marginal_diagnostics
from .conditioning import SCHEMA
from .evaluation import evaluation_views,information_view
from .bayes import conditional
from .config import extension_identity
from .dashboard import Dashboard


def evaluate(dataset,operators,root,*,max_cases_per_task=None,grid_size=64,persist=None):
    require(operators.control is None,'production baseline refuses small-mesh test controls');dataset.require_ready('test');root=Path(root);paths=[];rows=[];batches={};view_count=0;started=time.perf_counter();last_progress=-float('inf');last_task=None;durations=[];monitor=Dashboard(dataset.config);provider=Provider(dataset)
    cases=selected_cases(dataset,'test',max_cases_per_task);total_views=sum(54 if c.pde=='wave' else 42 for c in cases)
    with monitor.live():
        for c in cases:
            qd=query_design(c,grid_size);qc=Coordinates.design(qd);B,bq,qmeta=operators.get(c.pde,c.family,qc);truth=provider.cache.evaluate(c,*qd[:3]);qid=digest((c.case_id,qc.identity))
            case_identity=digest((SCHEMA,extension_identity(),dataset.config.identity,c.case_id,qid,qmeta['operator_spec']['identity']))
            path=root/'physical-cases'/(case_identity+'.json')
            if path.exists():
                record,cached=read_payload(path,kind='diagnostic_bundle',required=('rows','operator_batches','views'))
                require(record['case_identity']==case_identity,'classical physical case identity mismatch')
                rows.extend(cached['rows']);batches.update(cached['operator_batches']);view_count+=cached['views'];paths.append(str(path))
                atomic_json(root/'index.json',{'schema':SCHEMA,'case_results':paths,'dataset':dataset.config.identity})
                continue
            case_rows=[];case_batches={};posteriors=[];case_views=0
            for mode,mask,n,pattern in evaluation_views(c,dataset.config.seed,full=True):
                view,obs,meta=information_view(c,dataset,mode,mask,n);known,available=view.condition.coefficients();view_started=time.perf_counter()
                if last_task!=(c.pde,c.family) or view_count%25==0 or time.perf_counter()-last_progress>=2:
                    last_progress=time.perf_counter();last_task=(c.pde,c.family)
                    monitor.phase('Conditional Gaussian evaluation',pde=c.pde,geometry=c.family,mode=mode,sensors=n,known_fraction=view.condition.known_fraction,completed=view_count,total_views=total_views,elapsed_seconds=time.perf_counter()-started,estimated_remaining_seconds=(total_views-view_count)*float(np.median(durations[-50:])) if len(durations)>=5 else None,ETA_scope='recent measured mixed-view wall time; excludes already prepared operators')
                lookup=time.perf_counter();H,b,hmeta=operators.get(c.pde,c.family,Coordinates.design(observation_design(c,n,dataset.config.seed))) if n else (np.empty((0,B.shape[1])),np.empty(0),None);lookup_seconds=time.perf_counter()-lookup
                start=time.perf_counter();post=conditional(H,obs.value[0,:,0].double().numpy(),obs.sigma[0,:,0].double().numpy(),known,available,offset=b);field=post.predict(B,offset=bq);seconds=time.perf_counter()-start
                preds=[('Numerical forward' if mode=='FORWARD' else 'Conditional Bayesian' if mode=='HYBRID' else 'Bayesian inverse',field,seconds)]
                if mode=='HYBRID':
                    start=time.perf_counter();prior=conditional(np.empty((0,B.shape[1])),[],[],known,available);preds.append(('Conditional prior',prior.predict(B,offset=bq),time.perf_counter()-start))
                rr=[]
                for method,pred,secs in preds:
                    for q in np.unique(qd[2]):
                        take=qd[2]==q;scale=dataset.norms.scale(c.pde,int(q));rr.append({'method':method,'model':method,'case_id':c.case_id,'split':'test','pde':c.pde,'family':c.family,'quantity':'v' if q else 'u','mode':mode,'known_fraction':view.condition.known_fraction,'known_pattern':pattern,'sensor_count':n,'field_condition_identity':view.condition.identity,'observation_identity':meta['identity'],'query_identity':qid,
                            **_metrics(pred['mean'][take],truth[take],scale),'calibration':None if mode=='FORWARD' else marginal_diagnostics(truth[take],pred['mean'][take],pred['latent_variance'][take]),'latent_variance_max':float(pred['latent_variance'][take].max()),'maximum_physical_reference_discrepancy':float(abs(pred['mean'][take]-truth[take]).max()),'inference_seconds':secs,'cost':{'inference_seconds':secs,'operator_lookup_seconds':lookup_seconds},'uncertainty_scope':'latent field, numerical discretization uncertainty not included','operator_preprocessing_batch':qmeta['preprocessing_batch_identity']})
                batch_costs={m['preprocessing_batch_identity']:operators.preparation_cost(m['preprocessing_batch_identity']) for m in (qmeta,hmeta) if m is not None}
                payload={'operator_batches':batch_costs,'rows':rr,'posterior_mean':__import__('torch').from_numpy(post.mean),'posterior_covariance':__import__('torch').from_numpy(post.covariance),'operator_spec':qmeta['operator_spec'],'optimizer_updates':0}
                case_rows.extend(payload['rows']);case_batches.update(payload['operator_batches']);case_views+=1;view_count+=1;durations.append(time.perf_counter()-view_started)
                posteriors.append({'mode':mode,'sensor_count':n,'pattern':pattern,'known_fraction':view.condition.known_fraction,'mean':payload['posterior_mean'],'covariance':payload['posterior_covariance']})
            bundle={'rows':case_rows,'operator_batches':case_batches,'views':case_views,'conditional_posteriors':posteriors,'operator_spec':qmeta['operator_spec'],'optimizer_updates':0}
            write_artifact(path.parent,path.stem,{'artifact_type':'diagnostic_bundle','architecture_id':'G1_CONDITIONAL_BAYES','run_id':'G1-modes-classical','case_identity':case_identity},bundle,required=('rows','operator_batches','views'),idempotent=True)
            if persist:persist(path)
            batches.update(case_batches);rows.extend(case_rows);paths.append(str(path));atomic_json(root/'index.json',{'schema':SCHEMA,'case_results':paths,'dataset':dataset.config.identity})
        monitor.phase('Completed',operation='conditional numerical/statistical baselines',case_views=view_count,evaluation_wall_seconds=time.perf_counter()-started)
    return {'rows':rows,'index':str(root/'index.json'),'operator_batches':batches,'unique_preprocessing_seconds':sum(v['end_to_end_preparation_seconds'] for v in batches.values()),'evaluation_wall_seconds':time.perf_counter()-started,'optimizer_updates':0}
