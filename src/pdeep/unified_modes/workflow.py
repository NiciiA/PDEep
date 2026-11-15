"""Additive facade over G1 data, shared optimizer engine and verified artifacts."""
from pathlib import Path
import uuid
from dataclasses import asdict
from collections import defaultdict
import numpy as np
from ..core import require,digest
from ..artifact_drive import PersistencePolicy
from ..artifact_io import atomic_json,read_manifest
from ..unified.workflow import prepare,explicit_experiment_paths
from ..unified_bayes.storage import ArtifactStore
from .config import TrainConfig
from .conditioning import MODES
from .evaluation import evaluate,aggregate
from .training import inspect_checkpoint
from .model import create
from .dashboard import Dashboard


def configurations(dataset_config,*,models=('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001'),experiment_id=None,device='auto',capacity='medium',updates=100000,main_target=50000,query_count=4096,query_chunk=2048,seed=42,learning_rate=3e-4,weight_decay=1e-4,grad_clip=1.,validate_every=500,checkpoint_every=500,amp='auto',mode='research',neural_seed=None,mode_mixture=(.3,.4,.3),initialization='fresh'):
    experiment_id=experiment_id or 'g1-modes-'+uuid.uuid4().hex[:16]
    return tuple(TrainConfig(architecture=k.split('/')[0],model_key=k,dataset_identity=dataset_config.identity,experiment_id=experiment_id,device=device,capacity=capacity,updates=updates,main_target=main_target,query_count=query_count,query_chunk=query_chunk,
        root_seed=seed if neural_seed is None else neural_seed,data_seed=dataset_config.seed,learning_rate=learning_rate,weight_decay=weight_decay,grad_clip_norm=grad_clip,validation_every=validate_every,checkpoint_every=checkpoint_every,amp=amp,preset=mode,mode_mixture=tuple(mode_mixture),initialization=initialization,
        pdes=dataset_config.pdes,probabilities=tuple(1/len(dataset_config.pdes) for _ in dataset_config.pdes),warmup_updates=0 if mode=='smoke' else 250,scheduler='constant' if mode=='smoke' else 'warmup_then_cosine') for k in models)


def preview(dataset_config,configs,root):
    from ..unified.workflow import preview as previous
    p=previous(dataset_config,configs,root);p.update(conditioning_modes=MODES,mode_mixture=configs[0].mode_mixture,main_target=configs[0].main_target,immutable_scheduler_horizon=configs[0].updates,learning_rate_policy='250-update warmup then cosine to0.1*LR at predeclared horizon; no stretch on resume',field_representation='exact public C3 basis coefficients + geometry metadata + known/unknown/N-A flags',optimizer_steps=0,scope='same physical cases/labels/splits, different information views; new checkpoint schema, not trained forward capability yet')
    return p


def evaluate_checkpoints(paths,configs,dataset,root,*,split='test',selection='best',max_cases_per_task=None,grid_size=64,diagnostics=True,operators=None,persistence=None):
    require(any(c.model_key in paths for c in configs),'supply at least one exact compatible checkpoint for evaluation; missing methods remain unavailable')
    dataset.require_ready('test')
    require(split=='test','final sweep uses unseen test cases; validation stays in trainer');require(selection in ('best','latest'),'explicit validation-best/latest')
    comparison=digest({'checkpoints':{k:read_manifest(v)['artifact_id'] for k,v in sorted(paths.items())},'configs':[c.identity for c in configs],'selection':selection,'grid':grid_size,'case_bound':max_cases_per_task,'diagnostics':diagnostics})[:24]
    root=Path(root)/'evaluations-modes'/configs[0].experiment_id/selection/comparison
    from .persistence import evaluation_store
    store=evaluation_store(root,persistence,experiment=configs[0].experiment_id,selection=selection,comparison=comparison);monitor=Dashboard(dataset.config);results={}
    with monitor.live():
        for c in configs:
            if c.model_key not in paths:continue
            record,p=inspect_checkpoint(paths[c.model_key],c,dataset);weights=p['best_validation_model'] if selection=='best' else p['model'];require(weights is not None,'no validation-best model')
            model=create(c.architecture,dataset.norms,c.capacity,seed=c.root_seed,device=c.device);model.load_state_dict(weights,strict=True)
            r=evaluate(model,dataset,split='test',model_key=c.model_key,full=True,grid_size=grid_size,max_cases_per_task=max_cases_per_task,diagnostics=diagnostics,operators=operators,root=root/c.model_key.replace('/','_'),progress=monitor.phase,persist=store)
            r.update(history_refs=p.get('history_refs',[]),checkpoint=str(paths[c.model_key]),checkpoint_artifact=record['artifact_id'],training_seconds=p['training_state']['elapsed_seconds'],selection=selection,initialization=c.initialization,parents=record.get('parents',[]),training_cost_scope='new experiment only; fine-tune parent cost is separate');results[c.model_key]=r
            store.index(root/c.model_key.replace('/','_')/'index.json');del model,p
        monitor.phase('Completed',models=len(results),operation='three-mode neural evaluation')
    rows=[v for r in results.values() for v in r['rows']];report=summaries(rows,configs);report['models']={k:{x:v[x] for x in ('checkpoint','checkpoint_artifact','training_seconds','selection','seconds','initialization','parents','training_cost_scope')} for k,v in results.items()}
    report['physics_residuals_separate']=[v for r in results.values() for v in r['physics_residuals_separate']]
    root.mkdir(parents=True,exist_ok=True);atomic_json(root/'comparison.json',report)
    from .plots import summaries as plot_summaries
    plots=plot_summaries(report,root)
    from .plots import histories,reconstructions
    plots+=histories(results,root)
    for key,result in results.items():plots+=reconstructions(result['examples'],root/key.replace('/','_'))
    from .persistence import archive_summary
    indices=[str(root/c.model_key.replace('/','_')/'index.json') for c in configs if c.model_key in results]
    manifest,status=archive_summary(store,report,plots,raw_indices=indices)
    return {'report':report,'rows':rows,'directory':str(root),'plots':plots,'comparison_identity':comparison,'comparison_manifest':manifest,'persistence':status,'indices':indices}


def summaries(rows,configs=()):
    methods=tuple(c.model_key for c in configs) or ('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001');tables=[];rng=np.random.default_rng(42)
    for method in methods:
        for mode in MODES:
            selected=[r for r in rows if r['method']==method and r['mode']==mode];a=aggregate(selected) if selected else None
            # Bootstrap independent physical cases while keeping wave quantities/design views clustered.
            strata=defaultdict(lambda:defaultdict(list))
            for r in selected:strata[(r['pde'],r['family'])][r['case_id']].append(r['standardized_MSE'])
            boot=[]
            if strata and all(len(v)>1 for v in strata.values()):
                for cluster in strata.values():
                    values=np.array([np.mean(v) for v in cluster.values()]);boot.append(values[rng.integers(len(values),size=(500,len(values)))].mean(1))
            interval=np.quantile(np.mean(boot,axis=0),[.025,.975]).tolist() if boot else None
            tables.append({'case_bootstrap_95_CI':interval,'bootstrap_unit':'physical case; includes correlated quantities and all information cells', 'method':method,'mode':mode,'standardized_MSE':a['per_mode'][mode] if a else None,'status':'available' if a else 'unavailable: no compatible checkpoint/evaluation','quantity_breakdown':a['per_mode_PDE_quantity'][mode] if a else {}})
    groups=defaultdict(list)
    for r in rows:groups[(r['method'],r['mode'],r['pde'],r['family'],r['quantity'],r['known_fraction'],r['sensor_count'],r['known_pattern'])].append(r)
    tasks=[]
    for key,values in sorted(groups.items()):
        require(len({r['case_id'] for r in values})==len(values),'duplicate physical view')
        a=np.array([v['standardized_MSE'] for v in values]);boot=a[rng.integers(len(a),size=(500,len(a)))].mean(1) if len(a)>1 else None
        tasks.append({**dict(zip(('method','mode','pde','family','quantity','known_fraction','sensor_count','known_pattern'),key)),'cases':len(a),'standardized_MSE':float(a.mean()),'physical_RMSE':float(np.mean([r['physical_RMSE'] for r in values])),'case_bootstrap_95_CI':np.quantile(boot,[.025,.975]).tolist() if boot is not None else None,'median_inference_seconds':float(np.median([r['inference_seconds'] for r in values if r.get('inference_seconds') is not None])) if any(r.get('inference_seconds') is not None for r in values) else None})
    # Strict matching for neural pair differences within each information cell.
    paired=[]
    from itertools import combinations
    cells=defaultdict(dict)
    for r in rows:cells[(r['case_id'],r['mode'],r['quantity'],r['known_fraction'],r['known_pattern'],r['sensor_count'])][r['method']]=r
    differences=defaultdict(list)
    for cell,available in cells.items():
        require(len({(r['observation_identity'],r['query_identity'],r['field_condition_identity']) for r in available.values()})==1,'unmatched information across methods')
        for a,b in combinations(sorted(available),2):
            x,y=available[a],available[b];differences[(a,b,x['mode'],x['pde'],x['family'],x['quantity'],x['known_fraction'],x['sensor_count'],x['known_pattern'])].append(x['standardized_MSE']-y['standardized_MSE'])
    for k,v in differences.items():
        a=np.array(v);boot=a[rng.integers(len(a),size=(500,len(a)))].mean(1) if len(a)>1 else None
        paired.append({'methods':list(k[:2]),'mode':k[2],'pde':k[3],'family':k[4],'quantity':k[5],'known_fraction':k[6],'sensor_count':k[7],'pattern':k[8],'cases':len(v),'mean_standardized_MSE_difference':float(a.mean()),'paired_case_bootstrap_95_CI':np.quantile(boot,[.025,.975]).tolist() if boot is not None else None})
    return {'main':tables,'task_information_cells':tasks,'paired_case_differences':paired,'rows':rows,'normalization':'frozen TRAIN quantity RMS squared; physical RMSE separate; equal modes/PDE/active quantities/geometries/cases; expanded hybrid cells averaged within case','uncertainty':'classical conditional Gaussian only; deterministic neural networks have no uncertainty intervals','optimizer_updates':0}


def combined(neural,conditional_results,configs,root,*,inverse_index=None,persistence=None):
    """One final, snapshot-scoped comparison; all raw evidence stays case-verified."""
    rows=list(neural['rows'])
    if conditional_results:rows+=conditional_results['rows']
    report=summaries(rows,configs);report['models']=neural['report']['models'];report['physics_residuals_separate']=neural['report'].get('physics_residuals_separate',[])
    if conditional_results:report['classical_preprocessing']={'unique_batches':len(conditional_results['operator_batches']),'seconds':conditional_results['unique_preprocessing_seconds'],'evaluation_wall_seconds':conditional_results['evaluation_wall_seconds'],'scope':'reusable full reference preprocessing charged once, not per view'}
    if inverse_index:
        from ..unified_bayes.evaluation import load_classical
        from ..unified_bayes.comparison import summarize
        report['inverse_six_method']=summarize(load_classical(inverse_index)['rows']+[r for r in neural['rows'] if r['mode']=='INVERSE'])
    quantities=('poisson/u','heat/u','wave/u','wave/v');analogues=[]
    for mode in ('FORWARD','HYBRID'):
        methods=('Numerical forward',) if mode=='FORWARD' else ('Conditional prior','Conditional Bayesian')
        for method in (*methods,*(c.model_key for c in configs)):
            selected=[r for r in rows if r['mode']==mode and r['method']==method];a=aggregate(selected) if selected else None
            analogues.append({'mode':mode,'method':method,'quantity_standardized_MSE':{q:a['per_mode_PDE_quantity'][mode].get(q) if a else None for q in quantities},'status':'available' if selected else 'unavailable','physical_RMSE':'see quantity/task cells; physical units never pooled'})
    report['forward_hybrid_baseline_comparison']=analogues
    identity=digest((neural['comparison_identity'],conditional_results.get('index') if conditional_results else None,inverse_index))[:24];root=Path(root)/'comparisons-modes'/configs[0].experiment_id/identity;root.mkdir(parents=True,exist_ok=True);atomic_json(root/'comparison.json',report)
    from .plots import summaries as plot_summaries
    plots=plot_summaries(report,root)+[p for p in neural['plots'] if Path(p).name in ('training-curves.png','matched-mode-reconstruction.png')]
    from .persistence import evaluation_store,archive_summary
    store=evaluation_store(root,persistence,experiment=configs[0].experiment_id,selection='comparison',comparison=identity)
    indices=list(neural['indices'])+([conditional_results['index']] if conditional_results else [])+([inverse_index] if inverse_index else [])
    manifest,status=archive_summary(store,report,plots,raw_indices=indices)
    return {'report':report,'directory':str(root),'plots':plots,'comparison_manifest':manifest,'persistence':status}
