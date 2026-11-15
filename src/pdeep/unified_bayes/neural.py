"""Matched TEST evaluation of existing networks. No optimizer or weight selection."""
from pathlib import Path
import time
import torch
from ..core import digest,require
from ..artifact_io import read_payload,write_artifact,atomic_json,sha256
from ..artifact_tensors import state_identity
from ..geometric.models import create
from ..smooth.data import tokens
from ..unified.data import Provider
from ..unified.evaluation import predict,physical_metrics
from ..unified.training import inspect_checkpoint
from ..unified.monitor import Dashboard
from .evaluation import selected_cases,observation_design,query_design,heldout_design,_row,SCHEMA,DATA_SCHEMA
from .operators import Coordinates


def evaluate_neural(paths,configs,dataset,root,*,selection='best',counts=(8,16,32,64,128),size=64,max_cases_per_task=None,chunk=2048,diagnostics=True,monitor=None,persist=None):
    require(selection in ('best','latest'),'explicit validation-best/latest checkpoint selection required')
    monitor=monitor or Dashboard(dataset.config);provider=Provider(dataset);cases=selected_cases(dataset,'test',max_cases_per_task);root=Path(root);results=[];done=0;indices=[]
    with monitor.live():
        for config in configs:
            if config.model_key not in paths:continue # Main table still lists unavailable methods.
            record,payload=inspect_checkpoint(paths[config.model_key],config,dataset)
            weights=payload['best_validation_model'] if selection=='best' else payload['model'];require(weights is not None,'checkpoint has no validation-best weights')
            model=create(config.architecture,dataset.norms,config.capacity,seed=config.root_seed,device=config.device);model.load_state_dict(weights,strict=True);model.eval();before=state_identity(model)
            recipe={'schema':SCHEMA,'method':config.model_key,'checkpoint':record['artifact_id'],'selection':selection,'dataset':dataset.config.identity,'split':'test','counts':list(counts),'size':size,'seed':dataset.config.seed,'case_ids':[c.case_id for c in cases],'source':sha256(Path(__file__)),'diagnostics':diagnostics}
            identity=digest(recipe);case_paths=[]
            for c in cases:
                qd=query_design(c,size);qc=Coordinates.design(qd);truth=provider.cache.evaluate(c,*qd[:3])
                # Query values are targets only; prepare_context sees observations exclusively.
                qt=tokens(c,len(qd[0]),dataset.norms,set_id=('matched-query',size),coordinates=qd,clean=truth)[0]
                for n in counts:
                    path=root/'cases'/(digest((identity,c.case_id,n))+'.json')
                    monitor.phase('Neural evaluation',model=config.model_key,pde=c.pde,geometry=c.family,sensor_count=n,cases_completed=done,operation='verified result reuse' if path.exists() else 'predict; no optimizer')
                    if path.exists():r,p=read_payload(path,kind='diagnostic_bundle');require(r['evaluation_identity']==identity,'neural evaluation identity mismatch')
                    else:
                        od=observation_design(c,n,dataset.config.seed);ot,oi=provider.token_set(c,n,observation=True,set_id=(DATA_SCHEMA,'fixed-evaluation',dataset.config.seed),coordinates=od)
                        yp,seconds=predict(model,c,ot,qt,chunk=chunk);sensor,ss=predict(model,c,ot,ot,chunk=chunk)
                        hd=heldout_design(c,dataset.config.seed);ht,hi=provider.token_set(c,128,observation=True,set_id=('unified-heldout-sensors',dataset.config.seed),coordinates=hd);hp,hs=predict(model,c,ot,ht,chunk=chunk)
                        cost={'inference_seconds':seconds,'conditioning_sensor_seconds':ss,'heldout_prediction_seconds':hs,'device':str(config.device),'dtype':'float32','includes_context_encoding':True}
                        rows=_row(config.model_key,c,n,yp,truth,qd[2],qd[1],dataset.norms,oi['identity'],digest((c.case_id,qc.identity)),cost)
                        for row in rows:
                            row.update(checkpoint_artifact=record['artifact_id'],selection=selection,training_seconds=payload['training_state']['elapsed_seconds'])
                            k=1 if row['quantity']=='v' else 0;a=od[2]==k;h=hd[2]==k;scale=dataset.norms.scale(c.pde,k)
                            row.update(sensor_standardized_MSE=float((((sensor[a]-ot.value[0,a,0].double().numpy())/scale)**2).mean()),heldout_clean_standardized_MSE=float((((hp[h]-hi['clean'][h])/scale)**2).mean()))
                        physics=physical_metrics(model,c,provider,ot,seed=dataset.config.seed) if diagnostics and n==counts[-1] else None
                        p={'rows':rows,'physics_residuals_separate':physics,'prediction':torch.from_numpy(yp) if c.case_id==next(v.case_id for v in cases if (v.pde,v.family)==(c.pde,c.family)) and n==counts[-1] else None,
                           'case_id':c.case_id,'sensor_count':n,'method':config.model_key,'optimizer_updates':0}
                        write_artifact(path.parent,path.stem,{'artifact_type':'diagnostic_bundle','architecture_id':config.architecture,'run_id':config.run_id,'evaluation_identity':identity,'recipe':recipe},p,required=('rows',),idempotent=True)
                        if persist:persist(path)
                    case_paths.append(str(path));done+=1
                    atomic_json(root/('index-'+config.model_key.replace('/','_')+'-'+identity+'.json'),{'evaluation_identity':identity,'recipe':recipe,'case_results':case_paths})
                    results.append(p)
            indices.append(str(root/('index-'+config.model_key.replace('/','_')+'-'+identity+'.json')))
            require(state_identity(model)==before,'evaluation altered model parameters/buffers');del model,payload
        monitor.phase('Completed',operation='matched neural evaluation',case_sensor_sets=done,optimizer_updates=0)
    return {'rows':[r for p in results for r in p['rows']],'results':results,'indices':indices,'optimizer_updates':0}
