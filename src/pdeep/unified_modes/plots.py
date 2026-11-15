"""Compact mode/task, information-efficiency and time summaries."""
from pathlib import Path
import numpy as np

def summaries(report,root):
    import matplotlib.pyplot as plt
    root=Path(root);paths=[];models=sorted({r['method'] for r in report['main']});modes=('FORWARD','INVERSE','HYBRID')
    fig,ax=plt.subplots(figsize=(8,3.5));x=np.arange(3)
    for i,m in enumerate(models):
        y=[next(r['standardized_MSE'] for r in report['main'] if r['method']==m and r['mode']==mode) for mode in modes];ax.bar(x+(i-1)*.23,[np.nan if v is None else v for v in y],width=.23,label=m)
    ax.set_xticks(x,modes);ax.set_ylabel('Macro standardized MSE');ax.legend(fontsize=7);fig.tight_layout();p=root/'mode-comparison.png';fig.savefig(p,dpi=150);plt.close(fig);paths.append(str(p))
    cells=report['task_information_cells'];tasks=sorted({r['pde']+'/'+r['family'] for r in cells});fig,axes=plt.subplots(1,3,figsize=(13,max(4,len(tasks)*.24)),squeeze=False)
    for ax,mode in zip(axes[0],modes):
        A=np.full((len(tasks),len(models)),np.nan)
        for i,t in enumerate(tasks):
            for j,m in enumerate(models):
                v=[r['standardized_MSE'] for r in cells if r['mode']==mode and r['method']==m and r['pde']+'/'+r['family']==t]
                if v:A[i,j]=np.mean(v)
        im=ax.imshow(np.ma.masked_invalid(A),aspect='auto');ax.set_title(mode);ax.set_yticks(range(len(tasks)),tasks,fontsize=6);ax.set_xticks(range(len(models)),models,rotation=40,ha='right',fontsize=6);fig.colorbar(im,ax=ax)
    fig.tight_layout();p=root/'24-task-modes.png';fig.savefig(p,dpi=150);plt.close(fig);paths.append(str(p))
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for m in models:
        for fraction in (.25,.5,.75):
            rows=[r for r in cells if r['method']==m and r['mode']=='HYBRID' and r['known_fraction']==fraction];counts=sorted({r['sensor_count'] for r in rows})
            if rows:axes[0].plot(counts,[np.mean([r['standardized_MSE'] for r in rows if r['sensor_count']==n]) for n in counts],label=f'{m} known={fraction:g}')
        for mode in modes:
            rows=[r for r in report['rows'] if r['method']==m and r['mode']==mode];times=sorted({float(t) for r in rows for t in r.get('time_standardized_MSE',{})})
            if times:axes[1].plot(times,[np.mean([v for r in rows for t,v in r.get('time_standardized_MSE',{}).items() if float(t)==tt]) for tt in times],label=m+'/'+mode)
    axes[0].set(xlabel='Observations',ylabel='Hybrid standardized MSE');axes[1].set(xlabel='Physical time (dynamics only)',ylabel='Standardized MSE');axes[0].legend(fontsize=5);
    if axes[1].get_legend_handles_labels()[0]:axes[1].legend(fontsize=5)
    fig.tight_layout();p=root/'information-time.png';fig.savefig(p,dpi=150);plt.close(fig);paths.append(str(p));return paths


def histories(results,root):
    import matplotlib.pyplot as plt
    from ..unified.history import read
    fig,axes=plt.subplots(1,3,figsize=(13,3.6));has=False
    for model,result in results.items():
        if not result.get('history_refs'):continue
        records=read(result['history_refs']);has=True;steps=records['steps.jsonl'];steps=steps[::max(1,len(steps)//2000)]
        axes[0].plot([r['successful'] for r in steps],[r['total'] for r in steps],label=model)
        vals=records['validation.jsonl']
        for mode in ('FORWARD','INVERSE','HYBRID'):
            take=[v for v in vals if v['aggregate']['per_mode'].get(mode) is not None]
            if take:axes[1].plot([v['update'] for v in take],[v['aggregate']['per_mode'][mode] for v in take],label=model+'/'+mode)
        if steps:axes[2].plot([r['successful'] for r in steps],[r['gradient_before_clip'] for r in steps],label=model)
    if not has:plt.close(fig);return []
    for ax,title in zip(axes,('Actual weighted training objective','Fixed unseen per-mode validation MSE','Gradient norm before clipping')):ax.set(xlabel='Successful updates',title=title);ax.legend(fontsize=5)
    fig.tight_layout();p=Path(root)/'training-curves.png';fig.savefig(p,dpi=150);plt.close(fig);return [str(p)]


def reconstructions(examples,root,physical_time=.2):
    import matplotlib.pyplot as plt
    if not examples:return []
    pde='wave' if any(e['pde']=='wave' for e in examples) else examples[0]['pde'];selected=[e for e in examples if e['pde']==pde];qset=(0,1) if pde=='wave' else (0,);fig,axes=plt.subplots(len(selected)*len(qset),3,figsize=(10,2.4*len(selected)*len(qset)),squeeze=False)
    row=0
    for e in selected:
        arr=lambda key:e[key].numpy();t=arr('time');time=float(np.unique(t)[np.argmin(abs(np.unique(t)-physical_time))])
        for q in qset:
            take=(arr('quantity')==q)&(t==time);x=arr('position')[take];truth=arr('reference')[take];pred=arr('prediction')[take];lo=min(truth.min(),pred.min());hi=max(truth.max(),pred.max())
            for ax,title,y in zip(axes[row],('Reference','Neural point prediction','Absolute error'),(truth,pred,abs(pred-truth))):
                im=ax.scatter(x[:,0],x[:,1],c=y,s=4,cmap='viridis',vmin=0 if title=='Absolute error' else lo,vmax=None if title=='Absolute error' else hi);sensor=arr('sensor_position')[(arr('sensor_quantity')==q)&arr('sensor_valid')];ax.scatter(*sensor.T,c='red',s=8,marker='x');ax.set_aspect('equal');ax.set_title(f"{e['mode']} {pde}/{'v' if q else 'u'} t={time if pde!='poisson' else 'absent'}: {title}",fontsize=8);fig.colorbar(im,ax=ax,shrink=.7)
            row+=1
    root=Path(root);root.mkdir(parents=True,exist_ok=True);fig.tight_layout();p=root/'matched-mode-reconstruction.png';fig.savefig(p,dpi=150);plt.close(fig);return [str(p)]
