"""Explicit matched arms; planning never launches the multi-run experiment."""
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import time
from ..core import require,digest
from ..artifact_io import write_artifact,read_payload,atomic_json
from ..artifact_tensors import state_identity
from ..geometric.models import create
from .workflows import prepare_diagnostic,DiagnosticPlan,launch_diagnostic,performance_preflight

ARMS={'fixed_1024':('fixed',1024),'resampled_1024':('resampled',1024),'dense_resampled_4096':('dense_resampled',4096),'dense_resampled_8192':('dense_resampled',8192)}

@dataclass
class Comparison:
    base: object
    arms: dict
    initial_manifest: str
    budget: str
    @property
    def identity(self):return digest((self.base.identity,{k:asdict(v) for k,v in self.arms.items()},self.initial_manifest,self.budget))
    def preview(self):
        return [{'arm':name,'mode':c.query_mode,'queries_per_update':c.query_count,'updates':c.updates,'field_query_evaluations':c.query_count*c.updates,'weak_evaluations':c.updates if c.architecture=='C_GEO' else 0,'initial_model':self.base.initial_model_ids[0],'observation':self.base.data.identities['observation'],'evaluation':self.base.data.identities['diagnostic'],'physical_case':self.base.data.case.case_id,'run_id':c.run_id,'budget':self.budget,'compute_caveat':'equal queries is not equal compute: encoded-context and weak-physics costs scale with update count'} for name,c in self.arms.items()]


def prepare_comparison(*,budget='equal_updates',total_queries=None,**kwargs):
    require(budget in {'equal_updates','equal_queries'},'unknown comparison budget')
    require(kwargs.get('experiment','poisson_square') not in {'c_physics_ablation','observation_sensitivity'},'query comparison selects one supervised objective, not nested ablation/sensitivity')
    require(kwargs.get('action','fresh')=='fresh','comparison initialization is fresh; resume individual arms explicitly')
    require(kwargs.get('amp_mode','off')!='auto','matched comparisons require one explicit precision for every arm: off (FP32) or GPU-verified bf16')
    base=prepare_diagnostic(query_mode='fixed',query_count=1024,**kwargs);c=base.configs[0]
    total_queries=c.updates*1024 if total_queries is None else total_queries
    configs={}
    for name,(mode,n) in ARMS.items():
        if budget=='equal_queries':require(type(total_queries) is int and total_queries%n==0 and 1<=total_queries//n<=1000,'equal-query budget must be divisible by8192 and yield1..1000 updates per arm')
        updates=c.updates if budget=='equal_updates' else total_queries//n
        configs[name]=replace(c,query_mode=mode,query_count=n,bank_queries=max(c.bank_queries,n),updates=updates,run_tag=c.run_tag+'-'+budget+'-'+name)
    key=digest((base.identity,budget,{k:asdict(v) for k,v in configs.items()}))
    path=write_artifact(Path(base.root)/'learning_diagnostics'/'query_comparisons'/key,'initial-model',{'artifact_type':'diagnostic_bundle','architecture_id':c.architecture,'run_id':'query-comparison-'+key[:16],'initial_model_state':state_identity(base.models[0]),'benchmark':c.benchmark,'data_identity':base.data.identity,'budget':budget},{'model':base.models[0].state_dict(),'model_config':asdict(base.models[0].config)},required=('model','model_config'),idempotent=True)
    result=Comparison(base,configs,str(path),budget);atomic_json(path.parent/'comparison-plan.json',{'identity':result.identity,'arms':result.preview()});return result


def launch_comparison(plan,*,expected_identity,run_mode='preview',selected_arms=None,monitor=None):
    require(expected_identity==plan.identity,'comparison changed; rerun/review preview')
    if run_mode=='preview':return {}
    require(run_mode=='train','resume individual saved arms through diagnostic resume controls')
    names=tuple(plan.arms) if selected_arms is None else tuple(selected_arms);require(names and set(names)<=set(plan.arms),'unknown comparison arm')
    from .dashboard import Dashboard
    monitor=monitor or Dashboard();meta,payload=read_payload(plan.initial_manifest,kind='diagnostic_bundle',required=('model',));result={};cache=None
    for name in names:
        c=plan.arms[name];m=create(c.architecture,plan.base.data.normalizers,c.capacity,seed=c.root_seed,device=c.device);m.load_state_dict(payload['model'],strict=True)
        require(state_identity(m)==meta['initial_model_state'],'comparison did not restore common initial state')
        arm=DiagnosticPlan((c,),plan.base.data,(m,),plan.base.certificate,'fresh',(None,),plan.base.root,plan.base.persistence,{'comparison':plan.identity,'arm':name},(state_identity(m),),reference_cache=cache)
        if c.device.startswith('cuda'):performance_preflight(arm)
        start=time.perf_counter();engines=launch_diagnostic(arm,expected_identity=arm.identity,run_mode='train',monitor=monitor);e=engines['FULL'];cache=e.sampler.reference
        result[name]=e
        profiles=arm.performance.get('FULL',{}).get('profiles',[])
        summary={'arm':name,'budget':plan.budget,'updates':e.state.successful,'actual_field_queries':e.state.queries_processed,'expected_field_queries':c.updates*c.query_count,'wall_training_cell_seconds':time.perf_counter()-start,'total_run_elapsed_seconds':e.state.elapsed_seconds,'metrics':e.latest_validation.architecture_metrics,'CUDA_event_spans_scope':'performance.jsonl per-step stream spans, not kernel-active compute time','profile':profiles,'checkpoint':e.state.last_checkpoint}
        atomic_json(Path(plan.initial_manifest).parent/(name+'-result.json'),summary)
    comparison_results(Path(plan.initial_manifest).parent)
    return result


def comparison_results(directory,*,make_plots=True):
    """Aggregate only ACTUALLY executed arms; never invent pending learning scores."""
    import json
    import numpy as np
    directory=Path(directory);rows=[]
    for path in sorted(directory.glob('*-result.json')):
        r=json.loads(path.read_text());m=r['metrics']
        for q,metric in m['field'].items():
            rows.append({'arm':r['arm'],'budget':r['budget'],'quantity':q,'updates':r['updates'],'field_queries':r['actual_field_queries'],'wall_seconds':r['wall_training_cell_seconds'],'fixed_control_MSE':m['training'][q]['standardized_MSE'],'independent_MSE':metric['standardized_MSE'],'sensor_MSE':m['sensors'][q]['standardized_MSE'],'physical_RMSE':metric['physical_RMSE'],'amplitude_ratio':metric['amplitude_ratio'],'field_queries_per_wall_second':r['actual_field_queries']/r['wall_training_cell_seconds']})
    atomic_json(directory/'comparison-results.json',{'rows':rows,'scope':'only executed runs; equal queries is not equal wall/physics compute; CUDA event spans are logged per-step; isolated kernel profiling is optional'})
    if rows and make_plots:
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
        for ax,key,label in zip(axs,['updates','field_queries','wall_seconds'],['Successful updates','Supervised field-query evaluations','Measured wall seconds']):
            for r in rows:
                ax.scatter(r[key],r['independent_MSE']);ax.annotate(r['arm']+'/'+r['quantity'],(r[key],r['independent_MSE']),fontsize=7)
            ax.set(xlabel=label,ylabel='Independent standardized MSE',yscale='log')
        fig.savefig(directory/'accuracy-vs-cost.png',dpi=140)
    return rows
