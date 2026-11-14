"""Small notebook facade: preview, prepare, preflight, train, evaluate."""
from dataclasses import asdict
from pathlib import Path
import json,shutil,uuid
import numpy as np
import torch
from ..core import require
from ..artifact_io import atomic_json,read_payload
from ..artifact_drive import PersistencePolicy
from ..geometric.models import create
from ..smooth.definition import geometry
from .config import DatasetConfig,TrainConfig,MODELS
from .dataset import Dataset,prepare_dataset
from .monitor import Dashboard
from .persistence import DatasetPersistence,restore_dataset
from .training import train_models,inspect_checkpoint
from .evaluation import evaluate_model,save_comparison


def configurations(dataset_config,*,models=MODELS,experiment_id=None,device='auto',capacity='medium',updates=30000,
                   query_count=4096,query_chunk=2048,seed=42,learning_rate=3e-4,weight_decay=1e-4,grad_clip=1.,
                   validate_every=500,checkpoint_every=500,amp='auto',mode='research',neural_seed=None):
    experiment_id=experiment_id or ('g1-'+uuid.uuid4().hex[:16])
    return tuple(TrainConfig(architecture=k.split('/')[0],model_key=k,dataset_identity=dataset_config.identity,
        experiment_id=experiment_id,device=device,capacity=capacity,updates=updates,query_count=query_count,query_chunk=query_chunk,
        root_seed=seed if neural_seed is None else neural_seed,data_seed=dataset_config.seed,learning_rate=learning_rate,weight_decay=weight_decay,grad_clip_norm=grad_clip,
        validation_every=validate_every,checkpoint_every=checkpoint_every,amp=amp,preset=mode,
        pdes=dataset_config.pdes,probabilities=tuple(1/len(dataset_config.pdes) for _ in dataset_config.pdes),
        warmup_updates=0 if mode=='smoke' else 250,scheduler='constant' if mode=='smoke' else 'warmup_then_cosine') for k in models)


def preview(dataset_config,configs,root):
    # No mesh assembly/reference evaluation/model construction/optimizer/checkpoint lookup.
    free=shutil.disk_usage(Path(root) if Path(root).exists() else Path.cwd()).free
    return {'dataset':dataset_config.preview(),'models':[{'model':c.model_key,'capacity':c.capacity,'schedule_updates':c.updates,'config_identity':c.identity} for c in configs],
        'effective_batch':4,'queries_per_case':configs[0].query_count,'queries_per_update':4*configs[0].query_count,'query_chunk':configs[0].query_chunk,
        'local_storage_free_GB':free/1e9,'reference_storage':'exact independent scheduled-draw labels only; no full trajectories retained',
        'planned_training_label_GB_upper':configs[0].updates*4*(configs[0].query_count+128)*8/1e9,
        'reference_preparation':'one transient FULL CPU64 trajectory/physical case; CPU expensive, resumable; no solver in GPU updates',
        'memory':'microbatch one × four accumulations; actual 11.5 GB CUDA preflight required; no silent query/batch reduction',
        'AMP':'auto -> FP32; T4 BF16 unsupported, FP16/GradScaler with coordinate losses remains unverified',
        'checkpoint_lookup':'none unless exact explicit paths supplied','optimizer_steps':0,'reference_solves':0,
        'experiment_id':configs[0].experiment_id}


def prepare(config,root,certificate,*,action='preview',manifest=None,persistence=None,max_cases=None,monitor=None):
    require(action in ('preview','prepare','load','restore'),'invalid dataset action')
    if action=='preview':return None
    dashboard=monitor or Dashboard(config)
    with dashboard.live():
        if action=='restore':
            require(manifest,'restore requires exact persisted dataset index path')
            dashboard.phase('Dataset cache loading',index=manifest)
            manifest=restore_dataset(manifest,root)
        if action in ('load','restore'):
            require(manifest,'load requires exact dataset index path');return Dataset(manifest,certificate,progress=dashboard.phase)
        persistence=persistence or PersistencePolicy()
        dashboard.phase('Dataset inspection',dataset=config.identity,target=config.totals)
        return prepare_dataset(config,root,certificate,progress=dashboard.phase,
                               persist=DatasetPersistence(root,config.identity,persistence),max_cases=max_cases)


def explicit_experiment_paths(path,*,prefer_drive=False):
    """Known experiment.json only; never directory enumeration or old recovery."""
    value=json.loads(Path(path).read_text());out={}
    for key,r in value['models'].items():
        p=r.get('persistence',{}).get('last_sync',{}).get('Drive_manifest') if prefer_drive else r.get('latest')
        if prefer_drive and not p and r.get('successful',0)>0:raise ValueError('No verified Drive checkpoint pointer for '+key+'. Supply a known verified artifact explicitly; completed local work may not have synchronized.')
        if p:out[key]=p
    return out


def evaluate_checkpoints(paths,configs,dataset,root,*,split='validation',selection='best',sensitivity=True,diagnostics=True,include_baselines=True,grid_size=64,max_cases=None):
    require(selection in ('best','latest'),'choose validation best or latest explicitly')
    results={};dashboard=Dashboard(dataset.config)
    with dashboard.live():
        for c in configs:
            require(c.model_key in paths,'missing explicit checkpoint for '+c.model_key)
            record,payload=inspect_checkpoint(paths[c.model_key],c,dataset)
            weights=payload['best_validation_model'] if selection=='best' else payload['model'];require(weights is not None,'no validation-selected best model')
            model=create(c.architecture,dataset.norms,c.capacity,seed=c.root_seed,device=c.device);model.load_state_dict(weights,strict=True)
            result=evaluate_model(model,dataset,split=split,model_key=c.model_key,seed=c.data_seed,grid_size=grid_size,
                diagnostics=diagnostics,sensitivity=sensitivity,include_baselines=include_baselines,max_cases=max_cases,progress=dashboard.phase)
            from .plots import reconstruction_examples
            result['reconstruction_plot']=reconstruction_examples(model,dataset,Path(root)/'evaluations'/c.experiment_id/split/selection/(c.model_key.replace('/','_')+'-reconstruction.png'),split=split,seed=c.data_seed)
            result.update(neural_seed=c.root_seed,data_seed=c.data_seed,history_refs=payload.get('history_refs',[]),checkpoint=str(paths[c.model_key]),artifact_id=record['artifact_id'],selection=selection,
                          successful_update=payload['training_state']['best_validation_step'] if selection=='best' else record['successful_update'],
                          training_seconds=payload['training_state']['elapsed_seconds'],training_cases_processed=payload['training_state']['cases_processed'])
            results[c.model_key]=result;del model,payload
        destination=Path(root)/'evaluations'/configs[0].experiment_id/split/selection
        report=save_comparison(results,destination)
        from .plots import comparison_plots
        plots=comparison_plots(report,destination)
        plots.extend(r['reconstruction_plot'] for r in results.values())
        from .plots import dataset_plots,history_plots
        plots.extend(dataset_plots(dataset.config,destination,dataset))
        plots.extend(history_plots(results,destination))
        markdown='# G1 unified unseen-case evaluation\n\n'+f'Split: {split}. Selection: {selection} by fixed validation only.\n\n'
        markdown+='Scope: '+dataset.config.preview()['scope']+'.\n\n'
        markdown+='| Model | Updates | Equal-PDE standardized MSE | Training seconds | Evaluation seconds |\n|---|---:|---:|---:|---:|\n'
        for key,r in results.items():markdown+=f"| {key} | {r['successful_update']} | {r['aggregate']['selection_standardized_MSE']:.6g} | {r['training_seconds']:.4g} | {r['seconds']:.4g} |\n"
        markdown+='\nPhysical RMSE remains quantity-specific. Auxiliary inputs are supervised targets, never conditioning. Strong wave residuals are element-interior diagnostics. No accuracy conclusion from smoke runs. Final research comparisons require at least three separate neural seeds.\n'
        destination.mkdir(parents=True,exist_ok=True);(destination/'report.md').write_text(markdown)
        return {'report':report,'directory':str(destination),'plots':plots}
