"""Bounded suite tables/figures; quantity and physical-time metrics stay separate."""
from dataclasses import asdict
from pathlib import Path
import json
import math
import numpy as np
from ..artifact_io import atomic_json


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def assess(before,after):
    # Existing diagnostic criterion, with exact quantity matching (no zip truncation).
    old={r['quantity']:r['standardized_MSE'] for r in before['rows']}
    new={r['quantity']:r['standardized_MSE'] for r in after['rows']}
    per_quantity={q:new[q]<=max(.25*value,1e-5) for q,value in old.items() if q in new}
    per_time={}
    for a in before['rows']:
        b=next((r for r in after['rows'] if r['quantity']==a['quantity']),{})
        for t,v in a.get('per_time',{}).items():
            other=b.get('per_time',{}).get(t)
            per_time[a['quantity']+'/'+t]=other is not None and other['standardized_MSE']<=max(.25*v['standardized_MSE'],1e-5)
    passed=bool(old) and set(old)==set(new) and all(per_quantity.values()) and all(per_time.values()) and after['value']<=.25*before['value']
    return {'status':'passed' if passed else 'partial' if any(per_quantity.values()) else 'failed',
            'per_quantity':per_quantity,'per_time':per_time,'criterion':'existing >=75% aggregate/every quantity reduction or per-quantity 1e-5 floor; additionally each evaluated physical time must improve by75% or reach1e-5; fixed-case only'}


def collect_run(engine,directory):
    from .plotting import plots
    import matplotlib.pyplot as plt
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    metrics=engine.latest_validation.architecture_metrics
    steps=_lines(engine.run_dir/'steps.jsonl');performance=_lines(engine.run_dir/'performance.jsonl')
    # Restore histories embedded in a checkpoint when a fresh Colab local disk
    # has only the final segment of JSONL. Do not invent timing records.
    recent={r['successful']:r for r in engine.state.recent_steps if r.get('status','successful')=='successful'}
    recent.update({r['successful']:r for r in steps if r.get('status')=='successful'})
    gradient_rows=[r for r in recent.values() if r.get('gradient_before_clip') is not None]
    durations=[r['step_wall_seconds'] for r in performance]
    peaks=[r['memory'].get('peak_allocated_bytes') for r in performance if r.get('memory',{}).get('peak_allocated_bytes') is not None]
    # memory() explicitly uses peak_allocated on CUDA; support its canonical key.
    peaks += [r['memory']['peak_allocated'] for r in performance if r.get('memory',{}).get('peak_allocated') is not None]
    stats={'elapsed_seconds':engine.state.elapsed_seconds,'optimizer_seconds':engine.state.optimizer_seconds if hasattr(engine.state,'optimizer_seconds') else None,
           'failed':engine.state.failed,'skipped':engine.state.skipped,'queries_processed':engine.state.queries_processed,
           'training_queries_per_second':engine.state.queries_processed/max(engine.state.elapsed_seconds,1e-30),
           'step_median_seconds':float(np.median(durations)) if durations else None,
           'step_p95_seconds':float(np.percentile(durations,95)) if durations else None,
           'peak_allocated_CUDA_bytes':max(peaks) if peaks else None,
           'clipping_fraction':sum(r['gradient_before_clip']>engine.config.grad_clip_norm for r in gradient_rows)/len(gradient_rows) if gradient_rows else None,
           'clipping_records':len(gradient_rows),'clipping_scope':'logged local segment plus checkpoint rolling window; complete only if all update records available'}
    def median(values):return float(np.median(values)) if values else None
    stats['median_phase_wall_seconds']={key:median([r['timings'][key] for r in recent.values() if key in r.get('timings',{})]) for key in ('data','forward','backward','optimizer')}
    stats['phase_timing_scope']='CPU wall/enqueue; CUDA stream spans and isolated preflight timings reported separately; not kernel-active utilization'
    stats['peak_reserved_CUDA_bytes']=max((r['memory']['reserved_bytes'] for r in performance if r.get('memory',{}).get('reserved_bytes') is not None),default=None)
    stats['median_reference_evaluation_seconds']=median([r['reference']['reference_evaluation_seconds'] for r in performance if r.get('reference',{}).get('reference_evaluation_seconds') is not None])
    stats['median_H2D_wall_seconds']=median([r['reference']['H2D_wall_seconds'] for r in performance if r.get('reference',{}).get('H2D_wall_seconds') is not None])
    report={'model':engine.config.architecture,'objective':engine.config.objective_variant,'updates':engine.state.successful,
            'case':engine.data.case.manifest,'identities':engine.data.identities,'metrics':metrics,'stability':stats,
            'assessment':assess(engine.state.validation_history[0],engine.state.validation_history[-1]),
            'checkpoint':engine.state.last_checkpoint,'persistence':engine.state.persistence,
            'loss_weights':{'field':1.,'auxiliary':.1,**({'weak':.1,'initial':.1,'boundary':.1} if engine.config.architecture=='C_GEO' and engine.config.objective_variant=='FULL' else {})},
            'initial_field':engine.state.validation_history[0],
            'scope':'fixed-case memorization; independent-coordinate interpolation; unseen physical cases NOT tested'}
    paths=plots(engine.model,engine.data,directory/'figures',label=engine.config.architecture+'/'+engine.config.objective_variant)
    for path in paths:plt.close('all')
    # Preserve full local histories in the suite artifact as well as existing
    # periodic per-run verified history bundles. These travel with Drive resume.
    for name in ('steps.jsonl','validation.jsonl','performance.jsonl','module_updates.jsonl','gradient_terms.json'):
        src=engine.run_dir/name
        if src.exists():(directory/name).write_bytes(src.read_bytes())
    atomic_json(directory/'report.json',report)
    return report


def summary_rows(state):
    rows=[]
    for name in state['order']:
        entry=state['entries'][name]
        for variant,run in entry.get('runs',{}).items():
            report=run.get('report')
            if not report:continue
            for quantity,field in report['metrics']['field'].items():
                m=report['metrics'];rows.append({'experiment':name,'model':report['model']+'/'+variant,'quantity':quantity,'updates':report['updates'],
                    'field_MSE':m['training'][quantity]['standardized_MSE'],
                    'field_MSE_scope':'fixed never-optimized control queries for resampled modes; see sampled batch losses in history',
                    'independent_MSE':field['standardized_MSE'],'physical_RMSE':field['physical_RMSE'],
                    'sensor_MSE':m['sensors'][quantity]['standardized_MSE'],'amplitude_ratio':field['amplitude_ratio'],
                    'status':report['assessment']['status'],'execution_status':run['status'],
                    **report['stability'],'auxiliary':m.get('auxiliary',{}),'boundary':m.get('boundary',{}),
                    'checkpoint':run.get('Drive_checkpoint') or run.get('checkpoint'),'checkpoint_verified':bool(run.get('checkpoint_id')),
                    'Drive_status':run.get('Drive_status')})
    return rows


def combined_report(state,directory):
    import matplotlib.pyplot as plt
    directory=Path(directory);rows=summary_rows(state);files=[]
    members=[(name,v,run['report']) for name in state['order'] for v,run in state['entries'][name].get('runs',{}).items() if run.get('report')]
    if members:
        fig,axs=plt.subplots(1,len(members),figsize=(5*len(members),3.2),squeeze=False,layout='constrained')
        for ax,(name,variant,report) in zip(axs[0],members):
            steps=_lines(directory/name/variant/'steps.jsonl')
            for key in sorted({k for r in steps for k in r.get('unweighted',{})}):
                ax.plot([r['successful'] for r in steps],[r['unweighted'].get(key,np.nan) for r in steps],label=key)
            ax.set(title=name+'/'+variant,xlabel='successful updates',ylabel='raw loss',yscale='log');
            if ax.lines:ax.legend()
        path=directory/'loss-histories.png';fig.savefig(path,dpi=130);plt.close(fig);files.append(str(path))
        fig,ax=plt.subplots(figsize=(8,3.3),layout='constrained')
        ax.bar([r['experiment']+'/'+r['model'].split('/')[-1]+'/'+r['quantity'] for r in rows],[r['independent_MSE'] for r in rows])
        ax.set(yscale='log',ylabel='independent standardized MSE',title='Fixed physical cases: independent coordinates');ax.tick_params(axis='x',rotation=25)
        path=directory/'independent-errors.png';fig.savefig(path,dpi=130);plt.close(fig);files.append(str(path))
        for name,variant,report in members:
            if report['metrics']['field']['u'].get('per_time'):
                fig,axs=plt.subplots(1,2,figsize=(9,3),layout='constrained')
                for q,r in report['metrics']['field'].items():
                    times=sorted((float(t),v) for t,v in r.get('per_time',{}).items())
                    axs[0].plot([t for t,_ in times],[v['standardized_MSE'] for _,v in times],'-o',label=q)
                    axs[1].plot([t for t,_ in times],[v['amplitude_ratio'] for _,v in times],'-o',label=q)
                axs[0].set(xlabel='physical time',ylabel='standardized MSE',yscale='log',title=name+'/'+variant)
                axs[1].set(xlabel='physical time',ylabel='prediction / reference RMS')
                for ax in axs:ax.legend()
                path=directory/(name+'-'+variant+'-time-errors.png');fig.savefig(path,dpi=130);plt.close(fig);files.append(str(path))
    result={'rows':rows,'figures':files,'scope':'fixed-case memorization and independent-coordinate interpolation; new-case generalization NOT tested',
            'execution_status':state['status'],'failed_experiments':[k for k,e in state['entries'].items() if e['status'] in {'failed','interrupted'}]}
    atomic_json(directory/'summary.json',result)
    columns=('experiment','model','quantity','updates','field_MSE','independent_MSE','physical_RMSE','sensor_MSE','status')
    def fmt(x):return f'{x:.5g}' if isinstance(x,float) else str(x)
    text='| '+' | '.join(columns)+' |\n|'+ '|'.join(['---']*len(columns))+'|\n'
    text+='\n'.join('| '+' | '.join(fmt(r[c]) for c in columns)+' |' for r in rows)
    text+='\n\n'+result['scope']+'\n\nField MSE column: fixed control set; sampled update losses remain in histories.\n'
    (directory/'report.md').write_text(text)
    return result


def display_suite_results(result):
    from IPython.display import display,Markdown,Image
    if result.get('status')=='preview':
        display(Markdown('Suite preview: **zero optimizer updates**. Set `SUITE_RUN_MODE="train"` and run the suite execution cell.'));return
    rows=result.get('summary',{}).get('rows',summary_rows(result))
    columns=('experiment','model','quantity','updates','field_MSE','independent_MSE','physical_RMSE','sensor_MSE','status')
    display(Markdown('| '+' | '.join(columns)+' |\n|'+ '|'.join(['---']*len(columns))+'|\n'+'\n'.join('| '+' | '.join(f'{r[c]:.5g}' if isinstance(r[c],float) else str(r[c]) for c in columns)+' |' for r in rows)))
    display(Markdown(f"Suite **{result['status']}**. Snapshot: `{result.get('snapshot')}` · Drive: **{result.get('suite_sync',{}).get('Drive_status')}**. Fixed cases and independent coordinates; new-case generalization **not tested**."))
    timing=('experiment','model','quantity','elapsed_seconds','peak_allocated_CUDA_bytes','training_queries_per_second','clipping_fraction','Drive_status','checkpoint')
    display(Markdown('| '+' | '.join(timing)+' |\n|'+ '|'.join(['---']*len(timing))+'|\n'+'\n'.join('| '+' | '.join(f'{r.get(c):.5g}' if isinstance(r.get(c),float) else str(r.get(c)) for c in timing)+' |' for r in rows)))
    auxiliary=[];time_rows=[]
    for name in result['order']:
        for variant,run in result['entries'][name].get('runs',{}).items():
            for q,r in run.get('report',{}).get('metrics',{}).get('auxiliary',{}).items():auxiliary.append(f"|{name}/{variant}|{q}|{r['standardized_MSE']:.5g}|{r['physical_RMSE']:.5g}|")
            for q,r in run.get('report',{}).get('metrics',{}).get('field',{}).items():
                for t,m in sorted(r.get('per_time',{}).items(),key=lambda item:float(item[0])):
                    time_rows.append(f"|{name}/{variant}|{q}|{float(t):g}|{m['standardized_MSE']:.5g}|{m['physical_RMSE']:.5g}|{m['amplitude_ratio']}|")
    if auxiliary:display(Markdown('|Experiment|Auxiliary|Standardized MSE|Physical RMSE|\n|---|---|---:|---:|\n'+'\n'.join(auxiliary)))
    if time_rows:display(Markdown('|Experiment|Quantity|Physical time|Standardized MSE|Physical RMSE|Amplitude ratio|\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(time_rows)))
    for path in result.get('summary',{}).get('figures',[]):display(Image(filename=path))
    # A bounded one figure/quantity trajectory panel, seven rows each; no
    # figure per update or per sensor. Auxiliary source/initial fields included.
    snapshot=Path(result['snapshot']);root=snapshot.parent.parent
    for name in result['order']:
        for variant in result['entries'][name].get('runs',{}):
            for path in sorted((root/name/variant/'figures').glob('*.png')):display(Image(filename=str(path)))
