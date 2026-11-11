"""Small aggregate tables and a few representative plots; no per-case print storm."""
from pathlib import Path
import json
import numpy as np


def validation_table(report):
    rows=report.architecture_metrics['per_case_quantity'];out=['| PDE | Quantity | Geometry | Cases | Standardized MSE | Physical RMSE | Sensor MSE |','|---|---|---|---:|---:|---:|---:|']
    groups=sorted({(r['pde'],r['quantity'],r['family']) for r in rows})
    for key in groups:
        selected=[r for r in rows if tuple(r[k] for k in ('pde','quantity','family'))==key]
        out.append('| '+' | '.join(key)+f" | {len(selected)} | {np.mean([r['standardized_MSE'] for r in selected]):.5g} | {np.sqrt(np.mean([r['physical_RMSE']**2 for r in selected])):.5g} | {np.mean([r['sensor_standardized_MSE'] for r in selected]):.5g} |")
    return '\n'.join(out)


def history_plots(run_dir):
    import matplotlib.pyplot as plt
    run_dir=Path(run_dir);steps=[json.loads(s) for s in (run_dir/'steps.jsonl').read_text().splitlines()] if (run_dir/'steps.jsonl').exists() else []
    if not steps:return []
    fig,axes=plt.subplots(1,3,figsize=(13,3.5));x=[s['successful'] for s in steps]
    for k in steps[0]['unweighted']:axes[0].plot(x,[s['unweighted'].get(k,np.nan) for s in steps],label=k)
    axes[0].set(title='Unweighted model-specific losses',yscale='log');axes[0].legend(fontsize=8)
    for k in steps[0]['weighted']:axes[1].plot(x,[s['weighted'].get(k,np.nan) for s in steps],label=k)
    axes[1].set(title='Weighted terms',yscale='log');axes[1].legend(fontsize=8)
    axes[2].plot(x,[s['gradient_before_clip'] for s in steps],label='before clip');axes[2].plot(x,[s['gradient_after_clip'] for s in steps],label='after clip');axes[2].set(title='Gradient norms',yscale='log');axes[2].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Successful updates');ax.grid(alpha=.2)
    fig.tight_layout();return [fig]


def time_plot(report):
    import matplotlib.pyplot as plt
    rows=report.architecture_metrics['per_case_quantity'];fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for pde,q in sorted({(r['pde'],r['quantity']) for r in rows}):
        selected=[r for r in rows if r['pde']==pde and r['quantity']==q];ts=sorted({float(t) for r in selected for t in r['time_standardized_MSE']})
        values=[np.mean([v for r in selected for t,v in r['time_standardized_MSE'].items() if abs(float(t)-time)<1e-7]) for time in ts]
        axes[1 if q=='v' else 0].plot(ts,np.sqrt(values),marker='.',label=f'{pde} {q}')
    for ax in axes:ax.set(xlabel='Physical time (stationary shown at0 for plotting only)',ylabel='Standardized field RMSE');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();return fig


def representative_field(engine,*,pde='wave'):
    """One deterministic case, two physical quantities for wave, no cherry-picking."""
    import matplotlib.pyplot as plt
    from .data import Case
    import torch
    raw=next(v for v in engine.manifest.cases if v[0]==pde);p,g,split,i=raw;c=Case(g,p,split,i,engine.config.conditioning_h)
    b,_=engine.bank.batch((c,),validation=True,device=engine.config.device)
    with torch.no_grad():prediction=engine.model(b.specs,b.observations,b.queries)
    quantities=(0,1) if pde=='wave' else (0,);fig,axes=plt.subplots(len(quantities),3,squeeze=False,figsize=(12,3.5*len(quantities)))
    for j,q in enumerate(quantities):
        valid=b.queries.valid[0]&(b.queries.quantity_id[0]==q);times=b.queries.physical_time[0,:,0]
        if pde!='poisson':valid=valid&(times==times[valid].unique().sort()[0][-1])
        x=b.queries.position[0,valid].cpu().numpy();truth=b.queries.value[0,valid,0].cpu().numpy();pred=prediction[0,valid,0].cpu().numpy();scale=engine.normalizers.scale(pde,q)
        for ax,values,title in zip(axes[j],(truth,pred,(pred-truth)/scale),('Reference','Prediction','Standardized error')):
            artist=ax.scatter(x[:,0],x[:,1],c=values,s=4,cmap='coolwarm');fig.colorbar(artist,ax=ax);ax.set_aspect('equal');ax.set_title(f'{pde} {"v" if q else "u"}: {title}')
            if title=='Reference':
                obs=b.observations.position[0,b.observations.valid[0]&(b.observations.quantity_id[0]==q)].cpu().numpy();ax.scatter(obs[:,0],obs[:,1],s=10,c='k',marker='x')
    fig.tight_layout();return fig
