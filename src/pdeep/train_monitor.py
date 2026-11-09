"""One compact live display derived from structured reports, never parsed logs."""
from dataclasses import asdict
import html,json,math,resource,statistics,sys,time
import torch


def memory_report(device):
    d=torch.device(device)
    result={'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'scope':'process RSS includes libraries, calibration, validation and other models; not isolated model peak','current_RSS_bytes':None}
    try:
        import psutil
        result['current_RSS_bytes']=psutil.Process().memory_info().rss
        result['available_RAM_bytes']=psutil.virtual_memory().available
    except ImportError:pass
    if d.type=='cuda':result.update(device_current_allocated_bytes=torch.cuda.memory_allocated(d),device_peak_allocated_bytes=torch.cuda.max_memory_allocated(d),device_available_bytes=torch.cuda.mem_get_info(d)[0])
    elif d.type=='mps':result.update(device_current_allocated_bytes=torch.mps.current_allocated_memory(),device_driver_allocated_bytes=torch.mps.driver_allocated_memory(),device_peak_allocated_bytes=None)
    return result


def monitor_snapshot(config,state,step=None,validation=None):
    history=state.recent_steps[-config.rolling_window:]
    losses=[r['total'] for r in history if r['total'] is not None]
    durations=[sum(r['timings'].get(k,0) for k in ('data','forward','backward','optimizer')) for r in history]
    eta=None
    if len(durations)>=20:eta=max(0,config.updates-state.successful)*statistics.median(durations)
    return {'run_id':state.run_id,'source_id':config.source_id,'config_id':config.identity,'architecture':config.architecture,
            'PDE_mixture':dict(zip(config.pdes,config.probabilities)),'inverse_track':config.inverse_track,'stage':state.stage,'phase':state.phase,
            'successful':state.successful,'requested':config.updates,'attempted':state.attempted,'skipped':state.skipped,'failed':state.failed,
            'current_objective':None if step is None else step.total,'rolling_objective':sum(losses)/len(losses) if losses else None,
            'rolling_window':config.rolling_window,'unweighted':{} if step is None else step.unweighted,'weighted':{} if step is None else step.weighted,
            'objective_units':'dimensionless model-specific training objective; not comparable across architectures',
            'learning_rate':config.learning_rate if step is None else step.learning_rate,'gradient_before_clip':None if step is None else step.gradient_before_clip,
            'gradient_after_clip':None if step is None else step.gradient_after_clip,'cases':state.cases_processed,'queries':state.queries_processed,
            'microbatches':state.microbatches_processed,'actual_PDE_counts':dict(state.pde_counts),'elapsed_seconds':state.elapsed_seconds,
            'timings':{} if step is None else step.timings,'validation_seconds':state.validation_seconds,'saving_seconds':state.saving_seconds,
            'updates_per_second':state.successful/state.elapsed_seconds if state.elapsed_seconds>0 else None,
            'cases_per_second':state.cases_processed/state.elapsed_seconds if state.elapsed_seconds>0 else None,
            'ETA_seconds':eta,'ETA_status':'robust median of >=20 successful-update durations' if eta is not None else 'unavailable: fewer than20 successful observations',
            'device':config.device,'dtype':config.precision,'physics_policy':'explicit CPU64 bridge' if config.architecture.startswith('B') else 'neural device32; targets CPU64 before explicit copy',
            'memory':memory_report(config.device) if step is None else step.memory,'checkpoint':state.last_checkpoint,'Drive_sync':state.persistence.get('Drive_status','disabled'),
            'persistence':state.persistence,'best_validation':{'metric':state.validation_metric_name,'value':state.best_validation_metric,'step':state.best_validation_step,'checkpoint':state.best_checkpoint},
            'last_error':state.last_error,'latest_validation':None if validation is None else validation.metrics.get('common')}


class LiveMonitor:
    def __init__(self,*,enabled=True,notebook=False,stream=None,suite_budgets=None):
        self.enabled=enabled;self.notebook=notebook;self.stream=sys.stdout if stream is None else stream;self.suite_budgets=suite_budgets or {}
        self.handle=None;self.latest=None;self.timing_by_arch={};self.completed={};self.events=[]

    def update(self,config,state,step=None,validation=None):
        view=monitor_snapshot(config,state,step,validation)
        durations=[sum(r['timings'].get(k,0) for k in ('data','forward','backward','optimizer')) for r in state.recent_steps]
        if len(durations)>=20:self.timing_by_arch[config.architecture]=statistics.median(durations)
        self.completed[config.architecture]=state.successful
        remaining={k:max(0,v-self.completed.get(k,0)) for k,v in self.suite_budgets.items()}
        view['suite_ETA_seconds']=sum(n*self.timing_by_arch[k] for k,n in remaining.items() if n) if all(not n or k in self.timing_by_arch for k,n in remaining.items()) and self.suite_budgets else None
        self.latest=view;self.events.append((state.phase,state.successful))
        if not self.enabled or config.verbosity=='failure' and state.phase!='failed':return view
        def fmt(v):return '—' if v is None else f'{v:.5g}' if isinstance(v,float) else str(v)
        text=(f"{view['architecture']} {view['successful']}/{view['requested']} | {view['phase']} | attempts {view['attempted']} skip {view['skipped']} fail {view['failed']}\n"
              f"run {view['run_id']} · source {view['source_id'][:12]} · config {view['config_id'][:12]}\n"
              f"PDE {view['PDE_mixture']} · {view['inverse_track']} · {view['device']}/{view['dtype']}\n"
              f"objective {fmt(view['current_objective'])}, rolling{view['rolling_window']} {fmt(view['rolling_objective'])} (model-specific)\n"
              f"unweighted {view['unweighted']} · weighted {view['weighted']}\n"
              f"lr {fmt(view['learning_rate'])} · grad {fmt(view['gradient_before_clip'])}→{fmt(view['gradient_after_clip'])} · cases {view['cases']} queries {view['queries']}\n"
              f"elapsed {fmt(view['elapsed_seconds'])}s · timings {view['timings']} · validation {view['validation_seconds']:.3f}s save {view['saving_seconds']:.3f}s\n"
              f"updates/s {fmt(view['updates_per_second'])} · cases/s {fmt(view['cases_per_second'])} · ETA {fmt(view['ETA_seconds'])} · suite ETA {fmt(view['suite_ETA_seconds'])}\n"
              f"memory {view['memory']}\ncheckpoint {view['checkpoint']} · Drive {view['Drive_sync']} · error {view['last_error']}")
        if self.notebook:
            from IPython.display import display,HTML
            rendered=HTML('<pre style="white-space:pre-wrap;max-height:390px;overflow:auto">'+html.escape(text)+'</pre>')
            if self.handle is None:self.handle=display(rendered,display_id=True)
            else:self.handle.update(rendered)
        else:
            self.stream.write('\r'+text.replace('\n',' | ')+'\x1b[K');self.stream.flush()
        return view
