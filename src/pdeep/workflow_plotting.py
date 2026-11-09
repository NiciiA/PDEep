"""Small notebook views of existing reference, validation and evaluation results."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from .core import require
from .data import CaseDataset,observations,evaluation_grid,case_coefficients
from .reference import canonical_spec,evaluate_batch,evaluate
from .train_monitor import LiveMonitor


def _image(ax,values,title,*,limit=None,error=False):
    a=values.detach().cpu().numpy() if isinstance(values,torch.Tensor) else values
    a=np.asarray(a).reshape(64,64)
    im=ax.imshow(a.T,origin='lower',extent=(0,1,0,1),cmap='magma' if error else 'RdBu_r',
                 vmin=0 if error else -limit if limit is not None else None,vmax=None if error else limit)
    ax.set(xlabel='x',ylabel='y',title=title);return im


def plot_geometry():
    from .geometry import geometry_for
    geometry=geometry_for();fig,ax=plt.subplots(figsize=(4.5,4))
    ax.plot([0,1,1,0,0],[0,0,1,1,0],color='black',lw=2)
    ax.text(.5,.5,'Supported T1 domain\nΩ = (0,1)²\nHomogeneous Dirichlet boundary',ha='center',va='center')
    ax.set(xlim=(-.08,1.08),ylim=(-.08,1.08),aspect='equal',xlabel='x',ylabel='y',title='Square geometry; irregular domains are future work')
    fig.tight_layout();return fig


def plot_sensors(observation_set,pde):
    o=observation_set.inputs;fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for quantity in ((0,1) if pde=='wave' else (0,)):
        mask=(o.valid[0]&(o.quantity_id[0]==quantity));p=o.position[0,mask].numpy();t=o.physical_time[0,mask,0].numpy()
        name='v' if quantity else 'u';axes[0].scatter(p[:,0],p[:,1],label=name,marker='x' if quantity else 'o',s=22)
        axes[2].errorbar(np.arange(mask.sum()),o.value[0,mask,0],yerr=o.sigma[0,mask,0],fmt='.',label=name,alpha=.65)
        if pde!='poisson':axes[1].hist(t,bins=np.linspace(0,1,21),alpha=.5,label=name)
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='x',ylabel='y',title='Sparse observation positions');axes[0].legend()
    if pde=='poisson':axes[1].text(.5,.5,'Stationary: physical time absent',ha='center',va='center',transform=axes[1].transAxes);axes[1].axis('off')
    else:axes[1].set(xlabel='physical time',ylabel='sensors',title='Time coverage');axes[1].legend()
    axes[2].set(xlabel='index within quantity',ylabel='physical value ± declared σ',title='Noisy measurements');axes[2].legend()
    fig.suptitle(f'{pde} | T1 validation | {observation_set.metadata.identity[:12]}');fig.tight_layout();return fig


def plot_reference(case,*,times=(0.,.2,.75)):
    pde=case.key.pde;spec=canonical_spec(pde);quantities=(0,1) if pde=='wave' else (0,);times=(None,) if pde=='poisson' else times
    fig,axes=plt.subplots(len(quantities),len(times),figsize=(4*len(times),3.4*len(quantities)),squeeze=False)
    for i,qid in enumerate(quantities):
        for j,t in enumerate(times):
            q=evaluation_grid(pde,quantity_id=qid,**({} if t is None else {'times':(t,)}))
            value=evaluate_batch((spec,),case_coefficients((case,)),q)
            im=_image(axes[i,j],value,f"{'v' if qid else 'u'} | "+('time absent' if t is None else f't={t:g}'),limit=float(value.abs().max()) or 1.)
            fig.colorbar(im,ax=axes[i,j],shrink=.8)
    fig.suptitle(f'Analytical ground truth | {pde} T1 {case.key.physical_id} | CPU float64');fig.tight_layout();return fig


def plot_source_initial(case):
    pde=case.key.pde;q=evaluation_grid(pde,**({} if pde=='poisson' else {'times':(0.,)}))
    names=('f',) if pde=='poisson' else ('u0','v0') if pde=='wave' else ('u0',)
    fig,axes=plt.subplots(1,len(names),figsize=(4.5*len(names),3.4),squeeze=False)
    for ax,name in zip(axes[0],names):
        value=evaluate(pde,case.tensor(),q.position,None,quantity=name)
        im=_image(ax,value,name,limit=float(value.abs().max()) or 1.);fig.colorbar(im,ax=ax)
    fig.suptitle(f'{pde} T1 | unknown source / initial inputs (reference only)');fig.tight_layout();return fig


def plot_reconstruction(case,observation_set,method,normalizers,*,time=.2,quantity=0):
    from .evaluation_metrics import errors
    spec=canonical_spec(case.key.pde);q=evaluation_grid(spec.name,quantity_id=quantity,**({} if spec.name=='poisson' else {'times':(time,)}))
    state=method.prepare(spec,observation_set.inputs);pred=state.predict(q);truth=evaluate_batch((spec,),case_coefficients((case,)),q)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5));limit=max(float(truth.abs().max()),float(pred.abs().max()),1e-12)
    for ax,value,title,iserror in zip(axes,(truth,pred,(pred-truth).abs()),('reference','reconstruction','absolute error'),(False,False,True)):
        im=_image(ax,value,title,limit=limit,error=iserror);fig.colorbar(im,ax=ax,shrink=.8)
    o=observation_set.inputs;valid=o.valid[0]&(o.quantity_id[0]==quantity)
    axes[1].scatter(o.position[0,valid,0],o.position[0,valid,1],s=10,c='black',alpha=.55,label='sensor locations at their own times');axes[1].legend(fontsize=6)
    count=method.training.get('successful_updates',0);time_label='time absent' if spec.name=='poisson' else f't={time:g}'
    fig.suptitle(f'{method.name} | {method.status}, updates={count} | {spec.name} {"v" if quantity else "u"} T1 validation | {time_label}')
    fig.tight_layout();return fig,errors(pred,truth,normalizers.scale(spec.name,quantity),q.valid)


def persistence_warning(state):
    """Separate local commit and remote-copy status; never imply a failed copy saved."""
    p=state.persistence
    if p.get('Drive_status')!='failed':return None
    return ('DRIVE SYNC FAILED — Colab local storage is temporary. '
            f"Verified local checkpoint retained: {p.get('last_verified_local_checkpoint',state.last_checkpoint)}. "
            f"Last verified Drive copy: {p.get('last_verified_Drive_checkpoint','none')}. "
            f"Pending local checkpoint: {p.get('backlog',[])}. Error: {p.get('error','unknown')}")


class TrainingMonitor(LiveMonitor):
    """The working live monitor plus a separate, throttled validation plot view.

    No update loop. Plot failures are diagnostics and cannot interrupt training;
    the existing monitor's phase/loss/LR/gradient/speed/save/failure view stays live.
    """
    def __init__(self,*,validation_plots=True,**kwargs):
        super().__init__(**kwargs);self.engine=None;self.validation_plots=validation_plots;self.plot_handle=None;self.validation_seen=set();self.plot_errors=[]
        self.persistence_handle=None;self.persistence_warning_seen=None

    def bind(self,engine):self.engine=engine

    def update(self,config,state,step=None,validation=None):
        result=super().update(config,state,step,validation)
        warning=persistence_warning(state)
        if self.enabled and warning!=self.persistence_warning_seen and (warning or state.persistence.get('Drive_status')=='completed'):
            try:
                if self.notebook:
                    from IPython.display import display,HTML
                    import html
                    message=HTML('<div role="alert" style="color:#b00020;border:2px solid #b00020;padding:12px">'+html.escape(warning)+'</div>') if warning else HTML('Drive copy verified; previous sync warning cleared.')
                    if self.persistence_handle is None and warning:self.persistence_handle=display(message,display_id=True)
                    elif self.persistence_handle is not None:self.persistence_handle.update(message)
                elif warning:self.stream.write('\n'+warning+'\n');self.stream.flush()
                self.persistence_warning_seen=warning
            except Exception as error:self.plot_errors.append('Persistence display: '+str(error))
        if self.engine is None or self.engine.config.run_id!=config.run_id or validation is None:return result
        key=(config.run_id,validation.successful_update,validation.checkpoint_identity)
        if key in self.validation_seen:return result
        self.validation_seen.add(key)
        if self.notebook and self.validation_plots:
            try:
                from IPython.display import display
                fig=plot_training_validation(self.engine,validation)
                if self.plot_handle is None:self.plot_handle=display(fig,display_id=True)
                else:self.plot_handle.update(fig)
                path=self.engine.run_dir/'plots';path.mkdir(exist_ok=True)
                fig.savefig(path/f'validation-{state.successful:08d}.png',dpi=110,bbox_inches='tight');plt.close(fig)
            except Exception as error:self.plot_errors.append(str(error));print('Validation plot unavailable:',error)
        return result


def plot_training_validation(engine,report=None):
    from .evaluation_inference import Method
    from .notebook_workflows import training_status
    report=report or engine.latest_validation;require(report is not None,'no validation has run yet')
    split,pde,index=engine.manifest.keys[0];ds=CaseDataset(engine.normalizers.root_seed);case=ds.case(split,pde,index)
    obs=observations(ds,case,engine.normalizers,set_id=engine.manifest.observation_name)
    status=engine.config.preset if engine.config.preset in {'smoke','bounded'} else 'partially trained' if engine.state.successful<engine.config.updates else 'trained'
    method=Method(engine.config.architecture,engine.config.architecture,status,engine.model,training={'successful_updates':engine.state.successful})
    modes=[(m,m.training) for m in engine.model.modules()]
    try:
        engine.model.eval();spec=canonical_spec(pde);q=evaluation_grid(pde,**({} if pde=='poisson' else {'times':(.2,)}))
        session=method.prepare(spec,obs.inputs);pred=session.predict(q);truth=evaluate_batch((spec,),case_coefficients((case,)),q)
        fig,axes=plt.subplots(1,4,figsize=(14,3.2));lim=max(float(pred.abs().max()),float(truth.abs().max()),1e-12)
        for ax,value,title in zip(axes[:2],(truth,pred),('reference u','reconstructed u')):
            im=_image(ax,value,title,limit=lim);fig.colorbar(im,ax=ax,shrink=.7)
        path=engine.run_dir/'validation.jsonl';history=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        # Current report is emitted before validation() returns/logs on some hooks.
        points={h['successful_update']:h['metrics']['common']['field_normalized_MSE'] for h in history}
        points[report.successful_update]=report.metrics['common']['field_normalized_MSE']
        axes[2].plot(sorted(points),[points[s] for s in sorted(points)],marker='o',label='field standardized MSE')
        axes[2].set(xlabel='successful updates',ylabel='held-out field error');axes[2].legend(fontsize=7)
        if engine.config.architecture.startswith('B'):
            d=18 if pde=='wave' else 9;trajectory=session.context.trajectory[0,:,:d].detach().cpu()
            from .reference import prior_scales
            score=((trajectory-case.tensor())/prior_scales(pde)).square().mean(-1)
            axes[3].plot(score,marker='o',label='coefficient standardized MSE');axes[3].set(xlabel='refinement k',ylabel='coefficient error');axes[3].legend(fontsize=6)
            axes[3].text(.02,.02,'Phi: '+', '.join(f'{v:.3g}' for v in session.context.objectives[0]),transform=axes[3].transAxes,fontsize=6)
        elif engine.config.architecture=='C':
            specific=report.architecture_metrics['per_case'];rows=[v for v in specific if 'weak_objective' in v]
            axes[3].bar([v['pde'] for v in rows],[v['weak_objective'] for v in rows]);axes[3].set(ylabel='separate normalized weak residual',title='C physics diagnostic')
        else:axes[3].text(.5,.5,'Coefficient standardized MSE\n'+f"{report.metrics['common']['coefficient_normalized_MSE']:.4g}",ha='center',va='center',transform=axes[3].transAxes);axes[3].axis('off')
        fig.suptitle(f'{engine.config.architecture} {engine.state.run_id} | {status} | fixed validation {pde} u | '+('time absent' if pde=='poisson' else 't=.2'))
        fig.tight_layout();return fig
    finally:
        for module,mode in modes:module.training=mode


def plot_evaluation(result):
    """Plot existing metric/field arrays; actual checkpoint labels, no new scoring."""
    directory=Path(result['directory']);s=result['scientific'];paths=[]
    def save(fig,name):
        fig.tight_layout(rect=(0,0,1,.95));path=directory/name;fig.savefig(path,dpi=115,bbox_inches='tight');plt.close(fig);paths.append(path)
    for pde in sorted({r['pde'] for rows in s['cases'].values() for r in rows}):
        names=list(s['cases']);case=next(r for r in s['cases'][names[0]] if r['pde']==pde)
        for quantity in case['quantities']:
            ti=0 if pde=='poisson' else 4;fig,axes=plt.subplots(len(names),3,figsize=(10,3*len(names)),squeeze=False)
            datasets={name:np.load(directory/f"{name}__{case['case_id'].replace('/','_')}.npz") for name in names}
            def frame(a,key):return a[key][0,ti*4096:(ti+1)*4096,0]
            limit=max(max(np.abs(frame(a,'truth_'+quantity)).max(),np.abs(frame(a,'prediction_'+quantity)).max()) for a in datasets.values()) or 1.
            for i,name in enumerate(names):
                a=datasets[name];truth=frame(a,'truth_'+quantity);pred=frame(a,'prediction_'+quantity);meta=s['methods'][name]
                label=f"{name} | {meta['training_status']} | updates={meta['training_selection']['successful_updates'] or 0}\n{Path(meta['checkpoint']).stem}"
                for ax,value,title,err in zip(axes[i],(truth,pred,abs(pred-truth)),('truth','prediction','absolute error'),(False,False,True)):
                    im=_image(ax,value,label+' | '+title,limit=limit,error=err);fig.colorbar(im,ax=ax,shrink=.7)
                mask=a['conditioning_quantity'][0]==(1 if quantity=='v' else 0);p=a['conditioning_position'][0,mask]
                axes[i,1].scatter(p[:,0],p[:,1],c='black',s=7,alpha=.4)
            fig.suptitle(f"{pde} {quantity} T1 {case['split']} {case['case_id']} | "+('physical time absent' if pde=='poisson' else 't=.2; markers: sensor positions at their own times'))
            save(fig,f'fields_{pde}_{quantity}.png')
        if pde!='poisson':
            fig,axes=plt.subplots(1,len(case['quantities']),figsize=(6*len(case['quantities']),3.5),squeeze=False)
            for ax,q in zip(axes[0],case['quantities']):
                for name,rows in s['cases'].items():
                    row=next(r for r in rows if r['pde']==pde);series=row['time_errors'][q]
                    ax.plot([v['physical_time'] for v in series],[v['physical_RMSE'] for v in series],label=name+' '+row['training_status'])
                ax.set(title=q,xlabel='physical time',ylabel='physical RMSE');ax.legend(fontsize=7)
            fig.suptitle(pde+' T1 validation | actual selected checkpoints');save(fig,'time_'+pde+'.png')
    if 'D' in s['cases']:
        for row in s['cases']['D']:
            a=np.load(directory/f"D__{row['case_id'].replace('/','_')}.npz");ti=0 if row['pde']=='poisson' else 4
            fig,axes=plt.subplots(1,len(row['quantities']),figsize=(5*len(row['quantities']),3.6),squeeze=False)
            for ax,q in zip(axes[0],row['quantities']):
                std=np.sqrt(a['latent_variance_'+q][0,ti*4096:(ti+1)*4096,0]);im=_image(ax,std,q+' latent posterior std',error=True);fig.colorbar(im,ax=ax)
            fig.suptitle(f"D exact Gaussian, no training | {row['pde']} T1 validation | "+('time absent' if row['pde']=='poisson' else 't=.2')+' | no measurement noise added');save(fig,'D_uncertainty_'+row['pde']+'.png')
    selected=[name for name in ('B0','B1') if name in s['cases']]
    if selected:
        fig,axes=plt.subplots(1,len(s['aggregate'][selected[0]]['by_PDE']),figsize=(12,3.5),squeeze=False)
        for ax,pde in zip(axes[0],s['aggregate'][selected[0]]['by_PDE']):
            for name in selected:
                row=next(r for r in s['cases'][name] if r['pde']==pde);ax.plot(row['refinement']['Phi_by_iteration'],marker='o',label=name+' '+row['training_status'])
            ax.set(title=pde,xlabel='inverse refinement k (not time)',ylabel='Phi');ax.legend(fontsize=7)
        fig.suptitle('T1 validation | B0/B1 actual selected checkpoints, separate weights');save(fig,'B_refinement.png')
    return paths
