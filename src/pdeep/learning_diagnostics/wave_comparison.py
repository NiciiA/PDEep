"""Evaluation-only compact paired wave comparison. Never constructs an optimizer."""
from pathlib import Path
import numpy as np
from ..core import digest,require
from ..artifact_io import read_payload,atomic_json,write_artifact
from ..artifact_drive import sync_verified
from ..smooth.certification import normalizers
from .consistency import evaluate_wave_checkpoints,TIMES


def compare_wave(plan,verification,*,result=None,device='auto',count=128,chunk=128,display=True):
    root=Path(plan.root)/'learning_diagnostics'/'wave_kinematic'/plan.suite_id/'comparison'
    recovery={'status':'direct recorded paths only'}
    runs={}
    if plan.snapshot:
        runs.update(plan.snapshot['entries'][plan.experiments[0]].get('runs',{}))
    runs.update(getattr(plan,'reused_runs',{}))
    result=result or getattr(plan,'last_result',None)
    if result and result.get('entries'):runs.update(result['entries'][plan.experiments[0]]['runs'])
    from .suite import _checkpoint_path
    paths={label:_checkpoint_path(runs[label]) if label in runs else None for label in plan.settings.variant_settings}
    return _evaluate_comparison(plan,verification,paths,runs,recovery,root,device=device,count=count,chunk=chunk,display=display)


def _evaluate_comparison(plan,verification,paths,runs,recovery,root,*,device,count,chunk,display):
    report=evaluate_wave_checkpoints(paths,plan.certificate,root,device=device,count=count,chunk=chunk)
    require(report['comparison']=='paired','wave arms are not matched; do not rank them')
    require(len({r.get('coordinates_identity') for r in report['models'].values()})<=1,'evaluation query identities differ')
    sv=normalizers(plan.certificate).scale('wave','v');rows=[]
    for label,r in report['models'].items():
        require(r['physical_case']==verification['physical_case'] and r['observation_identity']==verification['observation_identity'],'comparison differs from verified matched case/sensors')
        arrays=np.load(root/(digest(label)[:12]+'.npz'));r['velocity_RMS']=sv;r['kinematic_normalization']='physical residual / frozen TRAIN s_v; distinct from legacy su+sv metric'
        for group,metric in r['pointwise']['groups'].items():
            metric['kinematic']['sv_normalized_MSE']=(metric['kinematic']['physical_RMS']/sv)**2
        meta,p=read_payload(paths[label],kind='training_checkpoint',required=('training_state',))
        state=p['training_state'];run=runs[label];stats=run.get('report',{}).get('stability',{})
        gradients=[row for row in state.get('recent_steps',[]) if row.get('gradient_before_clip') is not None]
        clip=meta.get('config',{}).get('grad_clip_norm',1.)
        clipped=sum(row['gradient_before_clip']>clip for row in gradients)/len(gradients) if gradients else None
        r['cost']={'cumulative_successful_updates':r['updates'],'session_optimizer_updates':run.get('session_optimizer_updates',0),
            'result_origin':run.get('result_origin','restored from completed checkpoint'),'training_elapsed_seconds':state.get('elapsed_seconds'),
            'optimizer_seconds':stats.get('optimizer_seconds',state.get('optimizer_seconds')),
            'clipping_fraction':stats.get('clipping_fraction',clipped),'clipping_records':stats.get('clipping_records',len(gradients)),
            'clipping_scope':stats.get('clipping_scope','checkpoint rolling window only; no complete-history claim'),
            'step_median_seconds':stats.get('step_median_seconds'),'step_p95_seconds':stats.get('step_p95_seconds'),
            'peak_allocated_CUDA_bytes':stats.get('peak_allocated_CUDA_bytes'),'scope':'checkpoint cumulative training clock; evaluation/recovery executes zero optimizer updates'}
        for quantity in ('u','v'):
            scale=normalizers(plan.certificate).scale('wave',quantity);pred=arrays[quantity];truth=arrays['truth_'+quantity]
            r.setdefault('independent_quantities',{})[quantity]={'standardized_MSE':float(np.mean(((pred-truth)/scale)**2)),
                'physical_RMSE':float(np.sqrt(np.mean((pred-truth)**2))),'amplitude_ratio':float(np.sqrt(np.mean(pred**2)/np.mean(truth**2))),
                'late_standardized_MSE':r['pointwise']['groups']['late'][quantity]['standardized_MSE'],
                'initial_physical_RMSE':float(np.sqrt(np.mean((pred[arrays['time']==0]-truth[arrays['time']==0])**2)))}
            allrow=r['pointwise']['groups']['all'];qm=r['independent_quantities'][quantity]
            rows.append({'variant':label,'quantity':quantity,**qm,'sensor_MSE':r['reconstruction']['sensors'][quantity]['standardized_MSE'],
                'kinematic_physical_RMS':allrow['kinematic']['physical_RMS'],'kinematic_sv_MSE':allrow['kinematic']['sv_normalized_MSE'],
                'dynamic_interior_physical_RMS':allrow['dynamic']['physical_RMS'],'weak_MSE':r['weak']['weak'],'reference_weak_floor':r['reference_weak']['weak'],
                'boundary_MSE':r['weak']['boundary'],'initial_consistency_MSE':r['weak']['initial'],**r['cost']})
    ratios={}
    def ratio(a,b):return float(a/b) if b>0 else None
    for label,r in report['models'].items():
        architecture=label.split('/')[0] if '/' in label else None
        baseline_label=architecture+('/BASELINE' if architecture=='A_GEO' else '/FIELD_ONLY') if architecture else 'FIELD_ONLY'
        if label==baseline_label or baseline_label not in report['models']:continue
        baseline=report['models'][baseline_label]
        allrow=r['pointwise']['groups']['all'];base=baseline['pointwise']['groups']['all']
        ratios[label]={'baseline':baseline_label,**{q+'_MSE_ratio':ratio(r['independent_quantities'][q]['standardized_MSE'],baseline['independent_quantities'][q]['standardized_MSE']) for q in ('u','v')},
            **{q+'_late_MSE_ratio':ratio(r['pointwise']['groups']['late'][q]['standardized_MSE'],baseline['pointwise']['groups']['late'][q]['standardized_MSE']) for q in ('u','v')},
            'kinematic_MSE_ratio':ratio(allrow['kinematic']['sv_normalized_MSE'],base['kinematic']['sv_normalized_MSE']),
            'dynamic_interior_RMS_ratio':ratio(allrow['dynamic']['physical_RMS'],base['dynamic']['physical_RMS']),
            'weak_MSE_ratio':ratio(r['weak']['weak'],baseline['weak']['weak'])}
    complete=set(report['models'])==set(paths)
    decision=('Matched trained checkpoints evaluated; compare measured ratios to each architecture baseline (below1 is lower error). '
              if complete else 'Pending missing/incompatible trained checkpoints; partial comparisons only. ')
    report.update(rows=rows,recovery=recovery,ratios_to_baseline=ratios,ratios_to_FIELD_ONLY=ratios if 'FIELD_ONLY' in paths else {},optimizer_updates=0,optimizer_constructed=False,
        decision=decision+'Kinematic improvement alone does not establish full wave dynamics. Interpret reconstruction/late-time/weak/dynamic/cost jointly; no automatic weight tuning or winner threshold.')
    report['figures']=_plots(report,root) if report['models'] else []
    atomic_json(root/'comparison.json',report)
    # Use existing verified artifact copier; preserve local result on Drive failure.
    files={p.name:p.read_bytes().hex() for p in root.glob('*.png')}
    bundle=write_artifact(root/'artifacts',None,{'artifact_type':'diagnostic_bundle','architecture_id':getattr(plan,'model','WAVE_COMPARISON'),'run_id':plan.suite_id,'benchmark':'G1_smooth_v1'},
        {'report':report,'figures':files},required=('report','figures'),idempotent=True)
    report['sync']=sync_verified(bundle,plan.persistence)
    if report['sync']['Drive_status']=='failed':
        import warnings
        warnings.warn('WAVE COMPARISON DRIVE SYNC FAILED; valid local results retained: '+str(report['sync']['error']),RuntimeWarning)
    if display:display_comparison(report)
    return report


def _plots(report,root):
    import matplotlib.pyplot as plt
    models=report['models'];labels=list(models);data={label:np.load(root/(digest(label)[:12]+'.npz')) for label in labels};first=data[labels[0]];paths=[]
    # One compact panel per quantity: same truth once, prediction/error for each
    # arm at each of three physical times. Physical color scale shared globally.
    for quantity in ('u','v'):
        maximum=max(np.max(abs(a[quantity])) for a in data.values());maximum=max(maximum,np.max(abs(first['truth_'+quantity])))
        error=max(np.max(abs(a[quantity]-a['truth_'+quantity])) for a in data.values());error=max(error,1e-15)
        cols=1+2*len(labels);fig,axes=plt.subplots(3,cols,figsize=(min(20,2.25*cols),6.8),squeeze=False,layout='constrained')
        for row,t in enumerate((0.,.5,1.)):
            take=first['time']==t;x=first['position'][take];truth=first['truth_'+quantity][take]
            axes[row,0].tricontourf(x[:,0],x[:,1],truth,levels=np.linspace(-maximum,maximum,33),cmap='RdBu_r')
            axes[row,0].set_title(f'{quantity} truth t={t:g}',fontsize=8)
            for j,label in enumerate(labels):
                a=data[label];mask=a['time']==t;xx=a['position'][mask];pred=a[quantity][mask]
                require(np.array_equal(xx,x),'plot coordinates differ across matched arms')
                for offset,values,limit,title in [(0,pred,maximum,'prediction'),(1,pred-truth,error,'error')]:
                    ax=axes[row,1+2*j+offset];ax.tricontourf(x[:,0],x[:,1],values,levels=np.linspace(-limit,limit,33),cmap='RdBu_r')
                    ax.set_title(label+' '+title,fontsize=8)
            for ax in axes[row]:ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'Physical {quantity}; shared prediction/truth ±{maximum:.3g}, shared error ±{error:.3g}. Independent interior queries.')
        path=root/(quantity+'-reconstruction.png');fig.savefig(path,dpi=120);plt.close(fig);paths.append(str(path))
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for label,r in models.items():
        points=[r['pointwise']['groups'][f't={t:g}'] for t in TIMES]
        for ax,key in zip(axes.ravel(),('u','v','kinematic','dynamic')):
            values=[p[key]['sv_normalized_MSE'] if key=='kinematic' else p[key]['standardized_MSE'] if key in {'u','v'} else p[key]['physical_RMS'] for p in points]
            ax.plot(TIMES,values,'-o',label=label);ax.set(xlabel='physical time',ylabel=key+(' MSE / s_v²' if key=='kinematic' else ' standardized MSE' if key in {'u','v'} else ' interior physical RMS'),yscale='log')
            if key in {'kinematic','dynamic'}:
                floor=r['reference_floor']['groups'];vv=[(floor[f't={t:g}'][key]['physical_RMS']/normalizers_scale(report,r,key))**(2 if key=='kinematic' else 1) for t in TIMES]
                ax.plot(TIMES,vv,'--',alpha=.4,label='reference floor' if label==labels[0] else None)
    for ax in axes.ravel():ax.legend(fontsize=7)
    path=root/'time-errors-consistency.png';fig.savefig(path,dpi=130);plt.close(fig);paths.append(str(path))
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for label,r in models.items():
        m=r['pointwise']['groups']['all'];field=np.mean([m[q]['standardized_MSE'] for q in ('u','v')]);kin=m['kinematic']['sv_normalized_MSE']
        axes[0].scatter(field,kin);axes[0].annotate(label,(field,kin),fontsize=8)
        seconds=r['cost']['training_elapsed_seconds']
        if seconds is not None:axes[1].scatter(seconds,field);axes[1].annotate(label,(seconds,field),fontsize=8)
    axes[0].set(xlabel='mean u/v reconstruction MSE',ylabel='kinematic MSE / s_v²',xscale='log',yscale='log')
    axes[1].set(xlabel='cumulative training wall seconds',ylabel='mean u/v reconstruction MSE',yscale='log')
    path=root/'quality-consistency-cost.png';fig.savefig(path,dpi=130);plt.close(fig);paths.append(str(path));return paths


def normalizers_scale(report,r,key):
    # Recover exactly s_v from the independently reported physical and normalized
    # metrics without relabeling old su+sv. Explicit value is stored below.
    return r['velocity_RMS'] if key=='kinematic' else 1.


def display_comparison(report):
    from IPython.display import display,Markdown,Image
    columns=('variant','quantity','standardized_MSE','physical_RMSE','sensor_MSE','late_standardized_MSE','amplitude_ratio','initial_physical_RMSE','kinematic_physical_RMS','kinematic_sv_MSE','dynamic_interior_physical_RMS','weak_MSE','clipping_fraction','training_elapsed_seconds','step_median_seconds','peak_allocated_CUDA_bytes','boundary_MSE','initial_consistency_MSE','session_optimizer_updates','result_origin')
    def fmt(v):return f'{v:.5g}' if isinstance(v,float) else str(v)
    display(Markdown('| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(row.get(k)) for k in columns)+' |' for row in report['rows'])))
    if report.get('ratios_to_baseline'):
        ratio_columns=tuple(next(iter(report['ratios_to_baseline'].values())))
        display(Markdown('|Variant / baseline|'+ '|'.join(ratio_columns)+'|\n|---|'+'|'.join(['---']*len(ratio_columns))+'|\n'+'\n'.join('|'+label+'|'+'|'.join(fmt(row[k]) for k in ratio_columns)+'|' for label,row in report['ratios_to_baseline'].items())))
    for label,message in report['missing'].items():print(label+': '+message)
    print(report['decision'])
    for path in report['figures']:display(Image(filename=path))


def compare_wave_experiments(plans,verification,*,result=None,device='auto',count=128,chunk=128,display=True):
    """One independent evaluation and compact report across both architectures."""
    from types import SimpleNamespace
    from .suite import _checkpoint_path
    recovery={'status':'direct recorded results / explicit suite paths only'};paths={};runs={}
    for plan in plans:
        local={}
        if plan.snapshot:local.update(plan.snapshot['entries'][plan.experiments[0]].get('runs',{}))
        local.update(getattr(plan,'reused_runs',{}))
        completed=(result or {}).get('architectures',{}).get(plan.model) or getattr(plan,'last_result',{})
        if completed.get('entries'):local.update(completed['entries'][plan.experiments[0]]['runs'])
        for variant in plan.settings.variant_settings:
            label=plan.model+'/'+variant
            paths[label]=_checkpoint_path(local[variant]) if variant in local else None
            if variant in local:runs[label]=local[variant]
    first=plans[0]
    joined=SimpleNamespace(root=first.root,suite_id='wave-comparison-'+digest((tuple((p.suite_id,p.identity) for p in plans),tuple((label,runs.get(label,{}).get('checkpoint_id')) for label in paths)))[:20],certificate=first.certificate,
        persistence=first.persistence,model='WAVE_COMPARISON')
    root=Path(joined.root)/'learning_diagnostics'/'wave_kinematic'/joined.suite_id/'comparison'
    return _evaluate_comparison(joined,verification[first.model],paths,runs,recovery,root,device=device,count=count,chunk=chunk,display=display)
