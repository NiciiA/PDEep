"""Small additive notebook facade, three explicit actions; preview does no solves."""
from pathlib import Path
from ..core import require,digest
from ..artifact_drive import PersistencePolicy
from ..artifact_io import atomic_json,write_artifact,sha256
from .storage import ArtifactStore
from .operators import ForwardOperators
from .evaluation import prepare_operators,evaluate_classical,load_classical
from .comparison import summarize,save
from .plots import figures


def classical(dataset,root,*,action='preview',counts=(8,16,32,64,128),size=64,max_cases_per_task=None,persistence=None):
    require(action in ('preview','prepare','evaluate'),'baseline action: preview | prepare | evaluate')
    if action=='preview':return {'action':'preview','PDE_geometry_tasks':24,'counts':list(counts),'reference_solves':0,'posterior_evaluations':0,'neural_training_required':False,'label':'exact public-Gaussian posterior for full discretized G1 reference'}
    require(dataset is not None,'load prepared/qualified dataset first');dataset.require_ready('test');root=Path(root)/'baselines'/dataset.config.identity
    store=ArtifactStore(root,persistence or PersistencePolicy());operators=ForwardOperators(root/'operators',dataset.certificate,persist=store)
    if action=='prepare':return prepare_operators(dataset,operators,counts=counts,size=size,max_cases_per_task=max_cases_per_task)
    result=evaluate_classical(dataset,operators,root/'classical',counts=counts,size=size,max_cases_per_task=max_cases_per_task,persist=store);store.index(result['index']);return result


def neural(paths,configs,dataset,root,*,enabled=False,selection='best',counts=(8,16,32,64,128),size=64,max_cases_per_task=None,diagnostics=True,persistence=None):
    if not enabled:return None
    from .neural import evaluate_neural
    require(dataset is not None,'load prepared dataset');dataset.require_ready('test');root=Path(root)/'baselines'/dataset.config.identity
    store=ArtifactStore(root,persistence or PersistencePolicy())
    result=evaluate_neural(paths,configs,dataset,root/'neural',selection=selection,counts=counts,size=size,max_cases_per_task=max_cases_per_task,diagnostics=diagnostics,persist=store)
    # Explicit returned per-model indices only, never recursive discovery.
    for path in result['indices']:store.index(path)
    return result


def combined(classical_index,neural_result,root,*,physical_time=.2,persistence=None):
    classical_result=load_classical(classical_index);neural_result=neural_result or {'rows':[],'results':[]};report=summarize(classical_result['rows']+neural_result['rows'])
    batches=classical_result['operator_batches'];sets=len({(r['case_id'],r['sensor_count']) for r in classical_result['rows']});pre=sum(b['end_to_end_preparation_seconds'] for b in batches.values())
    report['Bayesian_preprocessing']={'unique_forward_batches':len(batches),'seconds':pre,'amortized_seconds_per_evaluated_case_sensor_set':pre/sets if sets else None,
         'scope':'count each shared full-reference batched solve once; split from verified lookup + conditioning + grid/heldout inference; end-to-end = shared preparation + each lookup/inference', 'batches':batches}
    for cost in report['cost']:
        mean=cost['mean_inference_seconds']
        if mean is not None:
            local=mean+(cost['mean_operator_lookup_seconds'] or 0)+(cost['mean_interpolation_preprocessing_seconds'] or 0)
            cost['mean_percase_compute_end_to_end_seconds']=local
            cost['amortized_compute_seconds_including_shared_preparation']=local+(pre/sets if cost['method'] in ('Bayesian inverse','Prior mean') and sets else 0)
        cost['cost_exclusions']='input dataset restore/label generation, result artifact IO and ledger write; costs shown here are numerical inference/preparation, not total Colab session time'
    report['physics_residuals_separate']=[{k:p[k] for k in ('case_id','method','sensor_count','physics_residuals_separate')} for p in neural_result.get('results',[]) if p.get('physics_residuals_separate') is not None]
    report['classical_physics_residual_scope']='full discrete solver residual/energy diagnostics are in verified operator artifacts; continuum weak/strong residuals not separately evaluated and not equated to reconstruction MSE'
    report['probabilistic_diagnostics']=classical_result['probabilistic_summary'];report['coverage']=classical_result['coverage_summary'];report['coverage_by_geometry']=classical_result['coverage_by_geometry']
    save(report,root);paths=figures(report,classical_result,neural_result,root,physical_time=physical_time)
    import torch
    import numpy as np
    key=digest((report,{Path(p).name:sha256(p) for p in paths}))
    bundle={'report':report,'figure_png':{Path(p).name:torch.from_numpy(np.frombuffer(Path(p).read_bytes(),dtype=np.uint8).copy()) for p in paths}}
    store=ArtifactStore(Path(root)/'verified',persistence or PersistencePolicy())
    manifest=write_artifact(store.root,key,{'artifact_type':'diagnostic_bundle','architecture_id':'G1_COMPARISON','run_id':'G1-six-method-comparison','scope':'derived matched TEST reports and compact figures; no optimizer'},bundle,required=('report','figure_png'),idempotent=True)
    status=store(manifest)
    return {'report':report,'plots':paths,'directory':str(root),'comparison_manifest':str(manifest),'persistence':status}
