"""Bounded figures; no plotting inside per-case evaluation loops."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from ..smooth.definition import geometry,random_stream
from .config import TIMES


def _save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig);return str(path)


def dataset_plots(config,root,dataset=None):
    root=Path(root);fig,axes=plt.subplots(2,4,figsize=(13,6))
    for g,ax in zip(config.families,axes.flat):
        m=geometry(g,.08);ax.triplot(m.vertices[:,0],m.vertices[:,1],m.cells,color='.75',lw=.35)
        for ring in (m.shape.outer,*m.shape.holes):
            a=np.array((*ring,ring[0]));ax.plot(a[:,0],a[:,1],color='black',lw=1)
        ax.set_title(g);ax.set_aspect('equal')
    out=[_save(fig,root/'geometries.png')]
    fig,ax=plt.subplots(figsize=(11,3));counts=np.full((len(config.pdes),len(config.families)),config.counts[0]);im=ax.imshow(counts,aspect='auto');ax.set_xticks(range(len(config.families)),config.families);ax.set_yticks(range(len(config.pdes)),config.pdes);ax.set_title('Target TRAIN cases/task (prepared counts reported in dataset index)');fig.colorbar(im,ax=ax)
    out.append(_save(fig,root/'coverage.png'))
    if dataset:
        from .data import Provider
        provider=Provider(dataset);fig,axes=plt.subplots(1,3,figsize=(12,3))
        for p,ax in zip(config.pdes,axes):
            c=next(c for c in dataset.cases('train') if c.pde==p);x=geometry(c.family).sample(512,random_stream('unified-field-example',c.case_id));role=next(iter(c.fields));val=c.fields[role](x)
            image=ax.scatter(x[:,0],x[:,1],c=val,s=8,cmap='RdBu_r');ax.set_title(p+' '+role+' (hidden target)');ax.set_aspect('equal');fig.colorbar(image,ax=ax)
            if c.case_id in dataset.index['records']:
                from ..smooth.bank import physical_design
                sensor_x=physical_design(c,32,observation=True,set_id=('dataset-sensor-example',42))[0];ax.scatter(*sensor_x.T,facecolors='none',edgecolors='k',s=18)
        out.append(_save(fig,root/'input_fields_sensors.png'))
    return out


def comparison_plots(report,root):
    root=Path(root);models=report['models'];out=[]
    fig,axes=plt.subplots(1,len(models),figsize=(5*len(models),3),squeeze=False)
    for (name,r),ax in zip(models.items(),axes.flat):
        pdes=sorted({v['pde'] for v in r['rows']});families=sorted({v['family'] for v in r['rows']});a=np.full((len(pdes),len(families)),np.nan)
        for i,p in enumerate(pdes):
            for j,g in enumerate(families):
                values=[v['standardized_MSE'] for v in r['rows'] if v['pde']==p and v['family']==g]
                if values:a[i,j]=np.mean(values)
        im=ax.imshow(a,aspect='auto');ax.set_xticks(range(len(families)),families,rotation=60);ax.set_yticks(range(len(pdes)),pdes);ax.set_title(name);fig.colorbar(im,ax=ax,label='standardized MSE, equal active quantities')
    out.append(_save(fig,root/'24_task_comparison.png'))
    fig,axes=plt.subplots(1,3,figsize=(13,3))
    for name,r in models.items():
        sensitivity=defaultdict(list)
        for v in r['sensitivity']:
            if v['ablation']=='real':sensitivity[v['observations']].append(v['standardized_MSE'])
        if sensitivity:axes[0].plot(sorted(sensitivity),[np.mean(sensitivity[n]) for n in sorted(sensitivity)],'o-',label=name)
        axes[1].scatter(r.get('training_seconds',0),r['aggregate']['selection_standardized_MSE'],label=name)
        seconds=[v['inference_seconds'] for v in r['rows']];axes[2].scatter(np.median(seconds),r['aggregate']['selection_standardized_MSE'],label=name)
    for ax,title,xlabel in zip(axes,('Observation sensitivity','Training cost','Inference cost'),('Sensors','Measured training wall seconds','Median field inference seconds')):
        ax.set_title(title);ax.set_xlabel(xlabel);ax.set_ylabel('Standardized MSE');ax.legend(fontsize=6)
    out.append(_save(fig,root/'information_and_cost.png'))
    fig,axes=plt.subplots(1,3,figsize=(13,3))
    for name,r in models.items():
        for ax,(p,q) in zip(axes,(('heat','u'),('wave','u'),('wave','v'))):
            bytime=defaultdict(list)
            for v in r['rows']:
                if (v['pde'],v['quantity'])==(p,q):
                    for t,e in v.get('time_standardized_MSE',{}).items():bytime[float(t)].append(e)
            if bytime:ax.plot(sorted(bytime),[np.mean(bytime[t]) for t in sorted(bytime)],label=name)
            ax.set_title(p+' '+q);ax.set_xlabel('Physical time');ax.set_ylabel('Standardized MSE');ax.legend(fontsize=6)
    out.append(_save(fig,root/'time_profiles.png'))
    fig,ax=plt.subplots(figsize=(5,3))
    for name,r in models.items():
        kin=[v['kinematic_standardized_RMS'] for v in r['physical'] if 'kinematic_standardized_RMS' in v]
        if kin:ax.scatter(r['aggregate']['selection_standardized_MSE'],np.mean(kin),label=name)
    ax.set_xlabel('Reconstruction standardized MSE');ax.set_ylabel('Wave kinematic RMS / training velocity RMS');ax.legend(fontsize=6)
    out.append(_save(fig,root/'reconstruction_consistency.png'));return out


def case_inspector(model,case,provider,path,*,time=.5,quantity='u',size=64,seed=42):
    """One requested case/time/quantity; continuous clipped coordinates, shared color range."""
    from .evaluation import predict
    from ..smooth.data import tokens
    from ..smooth.bank import grid
    x,_,_,present=grid(case,size);x=np.unique(x,axis=0);t=np.full(len(x),time if present else 0.);q=np.full(len(x),int(quantity=='v'))
    try:truth=provider.cache.evaluate(case,x,t,q)
    except ValueError:
        # Prepared full grid stores all quantities/times in one verified design.
        gx,gt,gq,gp=grid(case,size);values=provider.cache.evaluate(case,gx,gt,gq)
        selected=(gq==int(quantity=='v'))&np.isclose(gt,time if present else 0.,rtol=0,atol=1e-12)
        if selected.sum()!=len(x):raise ValueError('Inspector time/grid not prepared; explicitly evaluate an offline full reference for other continuous truth queries.')
        truth=values[selected]
    qt,_=tokens(case,len(x),provider.norms,coordinates=(x,t,q,present),clean=truth)
    ot,_=provider.token_set(case,{'poisson':32,'heat':48,'wave':64}[case.pde],observation=True,set_id=('G1-unified-multicase-v1','fixed-evaluation',seed))
    y,_=predict(model,case,ot,qt);lo=min(truth.min(),y.min());hi=max(truth.max(),y.max());fig,axes=plt.subplots(1,3,figsize=(12,3))
    for ax,val,label in zip(axes,(truth,y,abs(y-truth)),('Reference','Prediction','Absolute error')):
        im=ax.scatter(x[:,0],x[:,1],c=val,s=4,cmap='RdBu_r' if label!='Absolute error' else 'magma',vmin=lo if label!='Absolute error' else 0,vmax=hi if label!='Absolute error' else None)
        ax.scatter(*ot.position[0].numpy().T,facecolors='none',edgecolors='k',s=14);ax.set_title(label);ax.set_aspect('equal');fig.colorbar(im,ax=ax)
    fig.suptitle(f'{case.pde}/{case.family}/{case.split}/{case.index} {quantity}, t={time if present else "absent"}')
    return _save(fig,path)


def training_plots(results,root):
    fig,axes=plt.subplots(2,3,figsize=(13,6))
    for name,result in results.items():
        directory=Path(result['run_dir']);path=directory/'steps.jsonl'
        if not path.exists():continue
        rows=[]
        with path.open() as stream:
            for i,line in enumerate(stream):
                if i%10==0:rows.append(json.loads(line))
        x=[r['successful'] for r in rows];axes[0,0].plot(x,[r['total'] for r in rows],label=name)
        axes[0,1].plot(x,[float(r['gradient_before_clip']>1.) for r in rows],label=name)
        axes[0,2].plot(x,[r['memory'].get('device_peak_allocated_bytes') or np.nan for r in rows],label=name)
        axes[1,0].plot(x,[1/max(r.get('details',{}).get('measured_update_wall_seconds',sum(r['timings'].values())),1e-12) for r in rows],label=name)
        p=directory/'validation.jsonl'
        if p.exists():
            vals=[json.loads(s) for s in p.read_text().splitlines()]
            axes[1,1].plot([v['update'] for v in vals],[v['aggregate']['selection_standardized_MSE'] for v in vals],label=name)
            for pde in ('poisson','heat','wave'):
                take=[v for v in vals if pde in v['aggregate']['per_PDE']]
                if take:axes[1,2].plot([v['update'] for v in take],[v['aggregate']['per_PDE'][pde] for v in take],label=name+' '+pde)
    for ax,title in zip(axes.flat,('Training objective','Clipping indicator (sampled)','GPU peak bytes','Measured updates/s','Validation equal-PDE MSE','Per-PDE validation MSE')):
        ax.set_title(title);ax.set_xlabel('Successful updates');ax.legend(fontsize=5)
    return [_save(fig,Path(root)/'training.png')]


def history_plots(results,root):
    """Full persisted history, downsampled for display; no training/data RNG."""
    from .history import read
    fig,axes=plt.subplots(2,3,figsize=(13,6));counts=[];has=False
    for name,result in results.items():
        if not result.get('history_refs'):continue
        records=read(result['history_refs']);all_steps=records['steps.jsonl'];steps=all_steps[::max(1,len(all_steps)//2000)];has=True
        x=[r['successful'] for r in steps];axes[0,0].plot(x,[r['total'] for r in steps],label=name)
        for r in all_steps:
            counts.extend(v['observation_count'] for v in r.get('details',{}).get('identities',[]))
        window=50;clip=[float(r['gradient_before_clip']>1.) for r in all_steps]
        rolling=np.convolve(clip,np.ones(min(window,len(clip)))/min(window,len(clip)),mode='valid') if clip else []
        if len(rolling):axes[0,1].plot([r['successful'] for r in all_steps][-len(rolling):],rolling,label=name)
        axes[0,2].plot(x,[r['memory'].get('device_peak_allocated_bytes') or np.nan for r in steps],label=name)
        axes[1,0].plot(x,[1/max(r.get('details',{}).get('measured_update_wall_seconds',sum(r['timings'].values())),1e-12) for r in steps],label=name)
        vals=records['validation.jsonl']
        axes[1,1].plot([v['update'] for v in vals],[v['aggregate']['selection_standardized_MSE'] for v in vals],label=name)
        for p in ('poisson','heat','wave'):
            take=[v for v in vals if p in v['aggregate']['per_PDE']]
            if take:axes[1,2].plot([v['update'] for v in take],[v['aggregate']['per_PDE'][p] for v in take],label=name+' '+p)
    if not has:plt.close(fig);return []
    for ax,title in zip(axes.flat,('Training objective','Clipping fraction (window50)','GPU peak bytes','Measured updates/s','Validation equal-PDE MSE','Per-PDE validation MSE')):
        ax.set_title(title);ax.set_xlabel('Successful updates');ax.legend(fontsize=5)
    paths=[_save(fig,Path(root)/'training_history.png')]
    if counts:
        fig,ax=plt.subplots(figsize=(6,3));values,number=np.unique(counts,return_counts=True);ax.bar([str(n) for n in values],number);ax.set_xlabel('Training sensors per physical-case encounter');ax.set_ylabel('Observed encounters (all selected models)')
        paths.append(_save(fig,Path(root)/'observation_counts.png'))
    return paths


def reconstruction_examples(model,dataset,path,*,split='validation',seed=42):
    """One composite figure/model: simple/irregular, initial/late, wave u AND v."""
    from .data import Provider,DEFAULT_COUNTS
    from .evaluation import predict
    from ..smooth.bank import grid
    from ..smooth.data import tokens
    provider=Provider(dataset);selected=[]
    for p in dataset.config.pdes:
        cases=[c for c in dataset.cases(split) if c.pde==p]
        families=list(dict.fromkeys(c.family for c in cases));simple='square' if 'square' in families else families[0]
        irregular=next((g for g in ('annulus','hole','polygon','lshape') if g in families),families[-1])
        for family,time in ((simple,0.),(irregular,1.)):
            c=next(c for c in cases if c.family==family)
            for q in (0,1) if p=='wave' else (0,):selected.append((c,time,q))
    fig,axes=plt.subplots(len(selected),3,figsize=(11,2.3*len(selected)),squeeze=False);previous=model.training;model.eval()
    try:
        for (case,time,q),row in zip(selected,axes):
            dataset.require_case(case);design=grid(case,dataset.config.stream_grid);truth=provider.cache.evaluate(case,*design[:3])
            take=(design[2]==q)&np.isclose(design[1],time if case.pde!='poisson' else 0.,atol=1e-12,rtol=0)
            x,t,ids=design[0][take],design[1][take],design[2][take];y=truth[take]
            query,_=tokens(case,len(x),provider.norms,coordinates=(x,t,ids,case.pde!='poisson'),clean=y)
            obs,_=provider.token_set(case,DEFAULT_COUNTS[case.pde],observation=True,set_id=('G1-unified-multicase-v1','fixed-evaluation',seed))
            prediction,_=predict(model,case,obs,query);lo=min(y.min(),prediction.min());hi=max(y.max(),prediction.max())
            for ax,values,title in zip(row,(y,prediction,abs(y-prediction)),('Reference','Prediction','Absolute error')):
                image=ax.scatter(x[:,0],x[:,1],c=values,s=3,cmap='magma' if title=='Absolute error' else 'RdBu_r',vmin=0 if title=='Absolute error' else lo,vmax=None if title=='Absolute error' else hi)
                ax.scatter(*obs.position[0].numpy().T,s=10,facecolors='none',edgecolors='k');ax.set_aspect('equal');ax.set_title(f'{case.pde}/{case.family} {"v" if q else "u"} t={time if case.pde!="poisson" else "absent"}: '+title,fontsize=8);fig.colorbar(image,ax=ax)
    finally:model.train(previous)
    return _save(fig,path)
