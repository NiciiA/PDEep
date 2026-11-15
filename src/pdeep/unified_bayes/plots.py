"""Seven compact figure families. Missing measurements remain visibly unavailable."""
from pathlib import Path
import numpy as np
from .comparison import METHODS,QUANTITIES


def figures(report,classical,neural,root,*,physical_time=.2,ratio_method=None,ratio_floor=1e-10):
    import matplotlib.pyplot as plt
    root=Path(root);root.mkdir(parents=True,exist_ok=True);paths=[];tables=report['tables'];colors=plt.cm.tab10(np.arange(6))
    def finish(fig,name):
        fig.tight_layout();p=root/(name+'.png');fig.savefig(p,dpi=160,bbox_inches='tight');plt.close(fig);paths.append(str(p))
    fig,axes=plt.subplots(1,4,figsize=(13,3.3))
    for ax,pq in zip(axes,QUANTITIES):
        values=[next(r for r in tables if r['group']=='overall' and r['method']==m)[pq+'/standardized_MSE'] for m in METHODS]
        for i,v in enumerate(values):
            if v is not None:ax.bar(i,v,color=colors[i])
            else:ax.text(i,.02,'N/A',rotation=90,transform=ax.get_xaxis_transform(),fontsize=7)
        ax.set_title(pq);ax.set_xticks(range(6),METHODS,rotation=65,ha='right',fontsize=6);ax.set_ylabel('Standardized MSE')
    finish(fig,'A_reconstruction')
    fig,axes=plt.subplots(1,4,figsize=(13,3.3))
    for ax,pq in zip(axes,QUANTITIES):
        for i,m in enumerate(METHODS):
            rows=[r for r in tables if r['group']=='sensor_count' and r['method']==m and r[pq+'/standardized_MSE'] is not None]
            if rows:ax.plot([r['key'] for r in rows],[r[pq+'/standardized_MSE'] for r in rows],'o-',ms=3,color=colors[i],label=m)
        ax.set(title=pq,xlabel='Reproducible non-nested sensor count',ylabel='Standardized MSE');ax.set_xscale('log',base=2)
    handles,labels=axes[-1].get_legend_handles_labels()
    if handles:axes[-1].legend(fontsize=6)
    finish(fig,'B_sensor_efficiency')
    tasks=sorted({r['key'] for r in tables if r['group']=='task'});fig,ax=plt.subplots(figsize=(12,7));matrix=np.full((len(tasks),6),np.nan)
    for j,m in enumerate(METHODS):
        for i,task in enumerate(tasks):
            r=next(v for v in tables if v['group']=='task' and v['key']==task and v['method']==m);vals=[r[q+'/standardized_MSE'] for q in QUANTITIES if r[q+'/standardized_MSE'] is not None]
            if vals:matrix[i,j]=np.mean(vals)
    title='24 tasks: standardized MSE; wave averages u/v'
    if ratio_method:
        assert ratio_method in METHODS
        denominator=matrix[:,METHODS.index('Bayesian inverse')];matrix=np.divide(matrix,denominator[:,None],out=np.full_like(matrix,np.nan),where=denominator[:,None]>ratio_floor)
        title=f'Error / Bayesian error; masked if Bayesian MSE ≤ {ratio_floor:g}'
    im=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap='viridis');ax.set_xticks(range(6),METHODS,rotation=30,ha='right');ax.set_yticks(range(len(tasks)),tasks,fontsize=7);ax.set_title(title);fig.colorbar(im,ax=ax);finish(fig,'C_task_heatmap')
    fig,ax=plt.subplots(figsize=(5,4));ax.plot([0,1],[0,1],'k--',lw=1)
    for pq in QUANTITIES:
        rows=[r for r in classical['coverage_summary'] if r['pde']+'/'+r['quantity']==pq and r['sensor_count']==report['overall_sensor_count'] and r['time'] is None and r['uncertainty_target']=='latent field']
        if rows:ax.plot([r['nominal'] for r in rows],[r['empirical_coverage'] for r in rows],'o-',label=pq)
    ax.set(xlabel='Nominal marginal coverage',ylabel='Empirical case-macro coverage',xlim=(.45,1),ylim=(.4,1));ax.legend(fontsize=8);finish(fig,'D_calibration')
    examples=[]
    for pq in QUANTITIES:
        match=next((e for e in classical['examples'] if e['row']['pde']+'/'+e['row']['quantity']==pq),None)
        # Wave example bundle contains both quantities; use the u-labelled bundle for v too.
        if match is None and pq=='wave/v':match=next((e for e in classical['examples'] if e['row']['pde']=='wave'),None)
        if match:examples.append((pq,match))
    if examples:
        fig,axes=plt.subplots(len(examples),4,figsize=(12,2.5*len(examples)),squeeze=False)
        fig2,axes2=plt.subplots(len(examples),5,figsize=(14,2.5*len(examples)),squeeze=False)
        for row,(pq,e) in enumerate(examples):
            d=e['data'];arr=lambda name:d[name].numpy();q=1 if pq.endswith('/v') else 0;t=arr('time');target=0. if pq.startswith('poisson') else float(np.unique(t)[np.argmin(abs(np.unique(t)-physical_time))]);take=(arr('quantity')==q)&(t==target);x=arr('position')[take];truth=arr('reference')[take];mean=arr('posterior_mean')[take];sd=np.sqrt(arr('latent_variance')[take]);lo,hi=float(truth.min()),float(truth.max())
            for ax,title,v in zip(axes[row],('Reference','Posterior mean','Absolute error','Posterior SD'),(truth,mean,abs(mean-truth),sd)):
                artist=ax.scatter(x[:,0],x[:,1],c=v,s=3,cmap='viridis',vmin=lo if title in ('Reference','Posterior mean') else 0,vmax=hi if title in ('Reference','Posterior mean') else None)
                sensors=arr('sensor_position')[arr('sensor_quantity')==q];ax.scatter(*sensors.T,s=8,c='red',marker='x',linewidths=.4);ax.set_aspect('equal');ax.set_title(f'{pq} {title}\nt={target:g}; physical units',fontsize=8);fig.colorbar(artist,ax=ax,shrink=.7)
            candidates=[truth,mean]
            for m in METHODS[3:]:
                found=next((p for p in neural.get('results',[]) if p['method']==m and p['case_id']==e['case_id'] and p['sensor_count']==e['sensor_count'] and p['prediction'] is not None),None)
                candidates.append(found['prediction'].numpy()[take] if found else None)
            for ax,title,v in zip(axes2[row],('Reference','Bayesian',*METHODS[3:]),candidates):
                if v is not None:artist=ax.scatter(x[:,0],x[:,1],c=v,s=3,vmin=lo,vmax=hi,cmap='viridis');fig2.colorbar(artist,ax=ax,shrink=.7)
                else:ax.text(.5,.5,'Unavailable',ha='center',transform=ax.transAxes)
                ax.set_aspect('equal');ax.set_title(f'{pq} {title}',fontsize=8)
        fig.suptitle('Sensors: all conditioning times projected spatially',fontsize=9);finish(fig,'E_Bayesian_uncertainty');finish(fig2,'F_same_case_neural_Bayesian')
    fig,axes=plt.subplots(1,3,figsize=(13,3.5));batches=classical.get('operator_batches',{});pre=sum(b['end_to_end_preparation_seconds'] for b in batches.values());axes[0].bar(['Full-reference operator'],[pre]);axes[0].set_ylabel('Measured reusable preprocessing seconds')
    for i,c in enumerate(report['cost']):
        for pq in QUANTITIES:
            r=next(v for v in tables if v['group']=='overall' and v['method']==c['method']);v=r[pq+'/standardized_MSE']
            if c['median_inference_seconds'] is not None and v is not None:axes[1].scatter(c['median_inference_seconds'],v,c=[colors[i]],s=20,label=c['method'] if pq==QUANTITIES[0] else None)
        train=c['neural_one_time_training_seconds']
        if train:axes[2].bar(i,sum(train),color=colors[i])
    axes[1].set(xlabel='Measured median complete-grid inference seconds',ylabel='Standardized MSE (one point/quantity)');axes[1].legend(fontsize=6);axes[2].set(xticks=range(6),xticklabels=METHODS,ylabel='Recorded one-time neural training seconds');axes[2].tick_params(axis='x',rotation=65,labelsize=6);finish(fig,'G_cost_accuracy')
    return paths
