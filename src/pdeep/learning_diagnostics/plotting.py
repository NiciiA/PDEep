"""Bounded diagnostic figures with common truth/prediction limits per quantity."""
from pathlib import Path
import json
import numpy as np
import torch
from .metrics import reconstruction


def plots(model,data,directory,*,label=None):
    import matplotlib.pyplot as plt
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    b=data.batch(next(model.parameters()).device,diagnostic=True)
    with torch.no_grad():cs=model.prepare_context(b.specs,b.observations);pred=model.predict(cs,b.queries,chunk=512)[0,:,0].cpu().numpy()
    q=b.queries;xy=q.position[0].cpu().numpy();time=q.physical_time[0,:,0].cpu().numpy();quantity=q.quantity_id[0].cpu().numpy();truth=q.value[0,:,0].cpu().numpy()
    files=[];label=label or model.config.architecture
    for k,name in ((0,'u'),(1,'v')) if data.case.pde=='wave' else ((0,'u'),):
        times=np.unique(time[quantity==k]);fig,axs=plt.subplots(len(times),3,figsize=(10,2.6*len(times)),squeeze=False,layout='constrained')
        takeq=quantity==k;lo=min(truth[takeq].min(),pred[takeq].min());hi=max(truth[takeq].max(),pred[takeq].max())
        for row,t in enumerate(times):
            take=takeq & (time==t)
            for col,(title,z) in enumerate((('truth',truth),('prediction',pred),('absolute error',abs(pred-truth)))):
                ax=axs[row,col];sc=ax.scatter(*xy[take].T,c=z[take],s=7,cmap='viridis' if col<2 else 'magma',vmin=lo if col<2 else 0,vmax=hi if col<2 else None)
                # Overlay genuine noisy sensors; times encoded separately in coverage plot.
                os=(data.observations.quantity_id[0].numpy()==k)&data.observations.valid[0].numpy();ox=data.observations.position[0].numpy()[os]
                ax.scatter(*ox.T,s=13,facecolors='none',edgecolors='cyan',linewidths=.6)
                for edge in data.case.problem.mesh.boundary_edges: ax.plot(*data.case.problem.mesh.vertices[edge].T,color='gray',lw=.5)
                ax.set(aspect='equal',title=f'{label} {name} · {title} · t={t:g}');fig.colorbar(sc,ax=ax,shrink=.7)
        path=directory/f'{name}-reconstruction.png';fig.savefig(path,dpi=130);files.append(str(path))
    # Unknown source / initial fields remain supervised targets, never context inputs.
    xs=q.position[0,quantity==0]; keep=torch.as_tensor(np.unique(xs.cpu().numpy(),axis=0,return_index=True)[1],device=xs.device);xs=xs[keep]
    for name,field in data.case.fields.items():
        with torch.no_grad():ap=model.evaluate(cs[0],xs,None if data.case.pde=='poisson' else xs.new_zeros((len(xs),1)),name)[:,0].cpu().numpy()
        at=field(xs.cpu().double().numpy());lo=min(at.min(),ap.min());hi=max(at.max(),ap.max())
        fig,axs=plt.subplots(1,3,figsize=(10,3),layout='constrained')
        for col,(title,z) in enumerate((('truth',at),('prediction',ap),('absolute error',abs(at-ap)))):
            sc=axs[col].scatter(*xs.cpu().numpy().T,c=z,s=7,cmap='viridis' if col<2 else 'magma',vmin=lo if col<2 else 0,vmax=hi if col<2 else None)
            axs[col].set(aspect='equal',title=f'{label} · {name} · {title}');fig.colorbar(sc,ax=axs[col],shrink=.7)
        path=directory/f'{name}-auxiliary.png';fig.savefig(path,dpi=130);files.append(str(path))
    fig,ax=plt.subplots(figsize=(7,2.4),layout='constrained')
    for k,name in ((0,'u'),(1,'v')) if data.case.pde=='wave' else ((0,'u'),):
        take=(data.observations.quantity_id[0].numpy()==k)&data.observations.valid[0].numpy()
        ax.scatter(data.observations.physical_time[0,take,0],np.full(take.sum(),k),s=18,label=name)
    ax.set(xlabel='physical time' if data.case.pde!='poisson' else 'stationary: time absent',title=f'{label} · fixed sensor time coverage');ax.legend()
    path=directory/'sensor-coverage.png';fig.savefig(path,dpi=130);files.append(str(path))
    (directory/'metadata.json').write_text(json.dumps({'label':label,'case':data.case.manifest,'identities':data.identities,'figures':files},indent=2))
    return files


def history(engine):
    import matplotlib.pyplot as plt
    path=engine.run_dir/'steps.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    fig,axs=plt.subplots(1,3,figsize=(12,3),layout='constrained')
    for key in sorted({k for r in rows for k in r.get('unweighted',{})}):axs[0].plot([r['successful'] for r in rows],[r['unweighted'].get(key,np.nan) for r in rows],label=key)
    axs[0].set(title='Raw objectives',yscale='log');axs[0].legend()
    for r in rows:
        if r.get('gradient_before_clip') is not None:axs[1].scatter(r['successful'],r['gradient_before_clip'],s=4,color='blue')
    axs[1].set(title='Gradient before clipping',yscale='log');axs[2].plot([r['successful'] for r in rows],[r['learning_rate'] for r in rows]);axs[2].set(title='Actual learning rate')
    clipped=sum(r.get('gradient_before_clip',0)>engine.config.grad_clip_norm for r in rows if r.get('gradient_before_clip') is not None)
    fig.savefig(engine.run_dir/'optimization.png',dpi=130)
    return {'clipping_frequency':clipped/len(rows) if rows else None,'updates_logged':len(rows),'scheduler':engine.scheduler.state_dict()}


def final_results(results):
    from IPython.display import display,Markdown
    rows=[]
    for variant,e in results.items():
        if isinstance(e,dict):
            r=e['result'];display(Markdown(f"Read-only sensitivity results: `{e['path']}`; optimizer updates=0"))
            sr=[f"|{k}|{v['mean_absolute_difference']:.4g}|{v['max_absolute_difference']:.4g}|{v['standardized_RMS_difference']:.4g}|{v['conditioning_token_difference']:.4g}|" for k,v in r['perturbations'].items()]
            display(Markdown('|Perturbation|Mean difference|Max difference|Standardized RMS change|Token change|\n|---|---:|---:|---:|---:|\n'+'\n'.join(sr)))
            if 'paired' in r:
                pr=[f"|{k}|{v['truth1_standardized_MSE']:.5g}|{v['truth2_standardized_MSE']:.5g}|" for k,v in r['paired'].items() if 'truth1_standardized_MSE' in v]
                display(Markdown('|Conditioning|Truth1 MSE|Truth2 MSE|\n|---|---:|---:|\n'+'\n'.join(pr)))
                display(Markdown('Correct-versus-wrong observation improvement (positive is better): `'+json.dumps(r['paired']['improvement_correct_vs_wrong'])+'`'))
            display(Markdown('Standardized observation derivative AD/FD: `'+json.dumps(r['value_derivative'])+'`. Sensitivity alone does not establish correct conditioning.'))
            continue
        m=e.latest_validation.architecture_metrics
        for q,r in m['field'].items():rows.append(f"|{e.config.architecture}/{variant}|{q}|{m['training'][q]['standardized_MSE']:.5g}|{r['standardized_MSE']:.5g}|{r['physical_RMSE']:.5g}|{m['sensors'][q]['standardized_MSE']:.5g}|{r['amplitude_ratio'] if r['amplitude_ratio'] is not None else 'undefined'}|")
        stats=history(e);plots(e.model,e.data,e.run_dir/'figures',label=e.config.architecture+'/'+variant)
        time_rows=[]
        for q,r in m['field'].items():
            for group,v in r.get('time_groups',{}).items():time_rows.append(f"|{q}|{group}|{v['standardized_MSE']:.5g}|{v['physical_RMSE']:.5g}|{v['amplitude_ratio']}|")
        if time_rows:display(Markdown('|Quantity|Physical time group|MSE|Physical RMSE|Amplitude ratio|\n|---|---|---:|---:|---:|\n'+'\n'.join(time_rows)))
        auxiliary=[f"|{q}|{v['standardized_MSE']:.5g}|{v['physical_RMSE']:.5g}|" for q,v in m['auxiliary'].items()]
        display(Markdown('|Supervised auxiliary|MSE|Physical RMSE|\n|---|---:|---:|\n'+'\n'.join(auxiliary)))
        physical=e.run_dir/'physical_diagnostics.json'
        if physical.exists():display(Markdown('Independent physical diagnostics (not reconstruction): `'+json.dumps(json.loads(physical.read_text())['unweighted'])+'`'))
        display(Markdown(f"Clipping fraction: {stats['clipping_frequency']} · final LR: {e.optimizer.param_groups[0]['lr']:.5g} · {stats['scheduler']['kind']} scheduler."))
        initial=e.state.validation_history[0]['value'];final=e.state.validation_history[-1]['value']
        # Reuse existing >=75%/per-quantity criterion without easier thresholds.
        before=e.state.validation_history[0]['rows'];after=e.state.validation_history[-1]['rows']
        passed=final<=.25*initial and all(b['standardized_MSE']<=max(.25*a['standardized_MSE'],1e-5) for a,b in zip(before,after)) and e.state.successful>0
        assessment={'criterion':'original >=75% aggregate and every quantity reduction or1e-5 floor; memorization only','passed':passed,'initial':initial,'final':final,'optimization':stats,'checkpoint':e.state.last_checkpoint,'Drive':e.state.persistence.get('Drive_status')}
        from ..artifact_io import atomic_json
        atomic_json(e.run_dir/'assessment.json',assessment)
        display(Markdown(f"**{e.config.architecture}/{variant}: {'passed' if passed else 'failed'}** (TRAIN memorization only). Checkpoint: `{e.state.last_checkpoint}` · Drive: **{e.state.persistence.get('Drive_status')}** · reports: `{e.run_dir}`"))
    if rows:display(Markdown('|Model/objective|Quantity|Fixed main MSE|Independent MSE|Physical RMSE|Sensor MSE|Amplitude ratio|\n|---|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)))
