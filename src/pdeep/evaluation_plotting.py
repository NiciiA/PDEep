"""Labeled static scientific comparison figures, never synthetic table values."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def comparison_table(result):
    s=result['scientific']; lines=['| Method | Status | Updates | Macro standardized MSE | Median full inference, ms |','|---|---|---:|---:|---:|']
    for name,row in s['aggregate'].items():
        meta=s['methods'][name];train=result['performance']['training_metadata'][name]
        lines.append(f"| {name} | {meta['training_status']} | {train.get('successful_updates',0)} | {row['macro']['standardized_MSE']:.6g} | {1000*result['performance']['summary'][name]['median_seconds']:.3f} |")
    return '\n'.join(lines)


def create_plots(result, plan):
    directory=Path(result['directory']);s=result['scientific'];paths=[]
    def save(fig,name):
        fig.tight_layout(rect=(0,0,1,.95))
        path=directory/name;fig.savefig(path,dpi=125,bbox_inches='tight');plt.close(fig);paths.append(str(path));return path
    methods=list(s['aggregate']); names=[f"{n}\n{s['methods'][n]['training_status']}\n{Path(s['methods'][n]['checkpoint']).stem}" for n in methods]
    fig,axes=plt.subplots(1,2,figsize=(15,4))
    # Quantity-level bars preserve physical units and wave velocity visibility.
    for ax,key,title in zip(axes,('physical_RMSE','standardized_MSE'),('Physical RMSE by PDE/quantity','Training-RMS standardized MSE')):
        labels=[];vals=[]
        for name in methods:
            for pde,row in s['aggregate'][name]['by_PDE'].items():
                for q,metric in row['quantities'].items():
                    labels.append(f'{name} {pde}/{q}');vals.append(metric[key])
        ax.bar(np.arange(len(vals)),vals);ax.set_yscale('log');ax.set_xticks(np.arange(len(vals)),labels,rotation=90,fontsize=6);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'T1 {plan.split} | all declared times (Poisson time absent) | neural smoke update-00000003; controls exact | no ranking claim')
    save(fig,'quantity_errors.png')
    fig,ax=plt.subplots(figsize=(12,4))
    x=np.arange(len(methods));prep=[np.median([r['conditioning_refinement_or_posterior_seconds'] for r in result['performance']['per_case'][n]])*1000 for n in methods]
    query=[np.median([r['full_grid_queries_seconds'] for r in result['performance']['per_case'][n]])*1000 for n in methods]
    ax.bar(x,prep,label='conditioning/refinement/posterior');ax.bar(x,query,bottom=prep,label='full coordinate queries')
    ax.set_xticks(x,names,fontsize=7);ax.set_ylabel('ms; component medians across 3 mixed-PDE cases');ax.legend()
    ax.set_title(f'T1 {plan.split} | Poisson/heat/wave u/v |64x64, all declared times | CPU32 neural, CPU64 physics');save(fig,'cost.png')
    for entry in plan.cases:
        pde=entry.pde;time_index=0 if pde=='poisson' else 4; time_label='physical time absent' if pde=='poisson' else f't={plan.dynamic_times[time_index]}'
        for quantity in dict(plan.quantities)[pde]:
            q='v' if quantity else 'u'; fig,axes=plt.subplots(len(methods),3,figsize=(9,2.25*len(methods)),squeeze=False)
            for i,name in enumerate(methods):
                a=np.load(directory/f"{name}__{entry.physical_id.replace('/','_')}.npz")
                def frame(key):return a[key][0,time_index*4096:(time_index+1)*4096,0].reshape(64,64)
                truth=frame('truth_'+q);pred=frame('prediction_'+q);vmax=max(abs(truth).max(),abs(pred).max(),1e-12)
                for j,(value,title) in enumerate(((truth,'truth'),(pred,'prediction'),(abs(pred-truth),'absolute error'))):
                    im=axes[i,j].imshow(value.T,origin='lower',extent=(0,1,0,1),cmap='magma' if j==2 else 'RdBu_r',vmin=0 if j==2 else -vmax,vmax=None if j==2 else vmax)
                    axes[i,j].set_title(f"{name} {s['methods'][name]['training_status']} {Path(s['methods'][name]['checkpoint']).stem}\n{title}",fontsize=7);fig.colorbar(im,ax=axes[i,j],shrink=.65)
            fig.suptitle(f'{pde} {q} | T1 {plan.split} {entry.physical_id} | {time_label}',y=1.002)
            save(fig,f'fields_{pde}_{q}.png')
        # Display both independently seeded sets, quantity identities and actual time coverage.
        a=np.load(directory/f"{methods[0]}__{entry.physical_id.replace('/','_')}.npz")
        fig,axes=plt.subplots(1,2,figsize=(10,3.4))
        for group,marker in (('conditioning','o'),('heldout','x')):
            for quantity in dict(plan.quantities)[pde]:
                mask=a[group+'_quantity'][0]==quantity;p=a[group+'_position'][0,mask];t=a[group+'_time'][0,mask,0]
                label=f"{group} {'v' if quantity else 'u'}"
                axes[0].scatter(p[:,0],p[:,1],s=18,marker=marker,label=label);axes[1].scatter(t,np.arange(len(t)),s=14,marker=marker,label=label)
        axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='x',ylabel='y');axes[1].set(xlabel='physical time' if pde!='poisson' else 'time storage only; physically absent',ylabel='record index within quantity')
        axes[0].legend(fontsize=7);fig.suptitle(f'{pde} T1 {plan.split} {entry.physical_id} | shared all methods, all statuses/checkpoints');save(fig,f'sensors_{pde}.png')
    for entry in plan.cases:
        pde=entry.pde;main=[name for name in ('A','C','B0','B1','D') if name in s['cases']]
        quantities=dict(plan.quantities)[pde]
        fig,axes=plt.subplots(len(quantities),len(main),figsize=(3*len(main),3*len(quantities)),squeeze=False)
        for i,quantity in enumerate(quantities):
            q='v' if quantity else 'u'
            for j,name in enumerate(main):
                a=np.load(directory/f"{name}__{entry.physical_id.replace('/','_')}.npz")
                mask=a['heldout_quantity'][0]==quantity;y=a['heldout_y'][0,mask,0];pred=a['heldout_prediction'][0,mask,0]
                ax=axes[i,j];ax.scatter(y,pred,s=13)
                if name=='D':
                    row=next(r for r in s['cases']['D'] if r['case_id']==entry.physical_id)
                    std=np.sqrt(np.array(row['uncertainty']['heldout_variance']['measurement'])[mask])
                    ax.errorbar(y,pred,yerr=1.96*std,fmt='none',alpha=.4)
                lo=min(y.min(),pred.min());hi=max(y.max(),pred.max());ax.plot([lo,hi],[lo,hi],color='black',lw=.6)
                ax.set_title(f"{name} {s['methods'][name]['training_status']}\n{Path(s['methods'][name]['checkpoint']).stem} | {q}",fontsize=7)
                ax.set_xlabel('held-out noisy value');ax.set_ylabel('predicted mean; D 95% future interval')
        fig.suptitle(f"{pde} T1 {plan.split} {entry.physical_id} | independent sensors | "+('physical time absent' if pde=='poisson' else 'all sampled sensor times in [0,1]'))
        save(fig,f'heldout_predictions_{pde}.png')
    for entry in plan.cases:
        rows=[next(r for r in s['cases'][name] if r['case_id']==entry.physical_id) for name in methods]
        if not all('observation_diagnostics' in r for r in rows):continue
        variants=list(rows[0]['observation_diagnostics']['experiments'])
        values=np.array([[r['observation_diagnostics']['experiments'][v]['field_max_change'] for v in variants] for r in rows])
        fig,ax=plt.subplots(figsize=(11,4.5));im=ax.imshow(np.log10(np.maximum(values,1e-16)),aspect='auto',cmap='viridis')
        ax.set_xticks(np.arange(len(variants)),variants,rotation=30,ha='right',fontsize=7);ax.set_yticks(np.arange(len(methods)),names,fontsize=6)
        fig.colorbar(im,ax=ax,label='log10(max abs field response), display floor 1e-16')
        ax.set_title(f"{entry.pde} T1 {plan.split} {entry.physical_id} | u query probes, "+('time absent' if entry.pde=='poisson' else 'independent times [0,1]'))
        save(fig,f'observation_ablations_{entry.pde}.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for name,rows in s['cases'].items():
        for row in rows:
            for q,series in row['time_errors'].items():
                if not series:continue
                index=0 if row['pde']=='heat' else 1 if q=='u' else 2
                axes[index].plot([v['physical_time'] for v in series],[v['physical_RMSE'] for v in series],label=f"{name} {row['training_status']}")
    for ax,title in zip(axes,('heat u','wave u','wave v')):ax.set(title=title,xlabel='physical time',ylabel='physical RMSE',yscale='log');ax.legend(fontsize=5)
    fig.suptitle(f'T1 {plan.split} all cases | neural checkpoint update-00000003; exact controls');save(fig,'time_errors.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,pde in zip(axes,plan.pdes):
        for name,rows in s['cases'].items():
            for row in rows:
                if row['pde']==pde and 'refinement' in row:
                    ax.plot(row['refinement']['Phi_by_iteration'],marker='o',label=f"{name} {row['training_status']}")
        ax.set(title=pde,xlabel='inverse iteration k (not physical time)',ylabel='Phi',yscale='log');ax.legend(fontsize=6)
    fig.suptitle(f'T1 {plan.split} u/v observations, all physical times | checkpoint update-00000003; fixed control');save(fig,'refinement.png')
    if 'D' in s['cases']:
        for row in s['cases']['D']:
            pde=row['pde'];a=np.load(directory/f"D__{row['case_id'].replace('/','_')}.npz");ti=0 if pde=='poisson' else 4
            quantities=tuple(row['quantities']);fig,axes=plt.subplots(len(quantities),3,figsize=(10,3*len(quantities)),squeeze=False)
            for i,q in enumerate(quantities):
                for j,(key,title) in enumerate((('truth_'+q,'truth'),('prediction_'+q,'posterior mean'),('latent_variance_'+q,'latent posterior std; no noise'))):
                    data=a[key][0,ti*4096:(ti+1)*4096,0].reshape(64,64).T
                    if j==2:data=np.sqrt(data)
                    limit=max(abs(a['truth_'+q]).max(),abs(a['prediction_'+q]).max(),1e-12)
                    im=axes[i,j].imshow(data,origin='lower',extent=(0,1,0,1),cmap='viridis' if j==2 else 'RdBu_r',vmin=0 if j==2 else -limit,vmax=None if j==2 else limit);fig.colorbar(im,ax=axes[i,j]);axes[i,j].set_title(q+' '+title,fontsize=8)
            fig.suptitle(f"D exact; no training | exact-T1-Gaussian | {pde} T1 {plan.split} {row['case_id']} | "+('time absent' if pde=='poisson' else f't={plan.dynamic_times[ti]}'))
            save(fig,f'D_uncertainty_{pde}.png')
        fig,axes=plt.subplots(1,3,figsize=(13,3.6))
        for ax,row in zip(axes,s['cases']['D']):
            uq=row['uncertainty'];d=len(uq['coefficient_posterior_mean']);a=np.load(directory/f"D__{row['case_id'].replace('/','_')}.npz")
            # Truth = posterior mean - signed coefficient recovery error.
            mean=np.array(uq['coefficient_posterior_mean']);true=mean-np.array(row['coefficient_vs_hidden_truth']['per_mode_error']);std=np.sqrt(np.diag(uq['coefficient_posterior_covariance']))
            ax.errorbar(np.arange(d),mean,yerr=1.96*std,fmt='o',label='D mean + 95% marginal');ax.scatter(np.arange(d),true,marker='x',label='hidden truth');ax.set(title=row['pde'],xlabel='active coefficient (a then b)',ylabel='physical coefficient units');ax.legend(fontsize=6)
        fig.suptitle(f'D exact-T1-Gaussian; no training | T1 {plan.split} | coefficients, no physical time');save(fig,'D_coefficients.png')
    return paths


def write_report(result,plan,plots):
    directory=Path(result['directory']);s=result['scientific']
    chunk=max(r['chunk_max_discrepancy'] for rows in s['cases'].values() for r in rows)
    individual=max(r['individual_max_discrepancy'] for rows in s['cases'].values() for r in rows)
    permutation=max((r['observation_diagnostics']['experiments']['record_permutation']['field_max_change'] for rows in s['cases'].values() for r in rows if 'observation_diagnostics' in r),default=0.)
    report=f'''# Stage 7B matched T1 validation diagnostic

Plan `{plan.identity}`; scientific results `{result['identity']}`.
Source `{plan.source_identity}`; normalization `{plan.normalizer_identity}`.
Physical cases: {', '.join(c.physical_id for c in plan.cases)}. Observation IDs and independent held-out IDs are in the immutable plan and each JSON case row.

No training or test evaluation. Neural checkpoints have three smoke updates each, one training seed, nine processed training cases. No serious ranking is justified by one validation case/PDE. D uses its exact posterior mean; posterior samples are coherence diagnostics only.

{comparison_table(result)}

Reduction: valid query points -> equal u/v quantities within each physical case -> equal physical cases per PDE -> equal PDE macro. RMSE is sqrt of the corresponding reduced MSE. All physical quantity rows and time-resolved errors remain in JSON. Standardization uses frozen training RMS. Physical u/v aggregate units differ; inspect separate quantity metrics.

CPU64 direct references and D; float32 neural inputs and explicit CPU64 B physics. Identical raw measured records are copied to neural precision, never resampled. Conditioning noisy residuals, clean scoring errors and independent held-out scores are separate. No inference API accepts hidden coefficients or clean sensor metadata.

Maximum query-chunk discrepancy {chunk:.6g}; batched/individual discrepancy {individual:.6g}; whole-record permutation change {permutation:.6g}. Physical residuals use independent interior coordinates and coordinate AD, including modal B/D; actual maxima and AD/analytic cross-checks are in JSON. C weak diagnostics remain separate. Boundary and initial truth/consistency errors are separate. Near-zero solver residuals demonstrate representational consistency, not accurate inverse recovery.

D reports coefficient covariance, latent field variance, future measurement variance (latent + sigma²), joint/marginal held-out log density, 50/90/95% coverage and coherent samples. Coverage intervals resample the experimental unit correctly: bounded physical-case fractions with conservative Hoeffding intervals; the tiny run has broad intervals. Paired case-bootstrap CIs are unavailable with one case/PDE. Neural uncertainty is unavailable.

All selected cases receive all observation perturbations, including empty sets; each perturbation records exactly which metadata was held fixed. D covariance changes and neural pathway changes are reported without favorable-example selection. B1-disabled views reuse the B1 checkpoint with fresh contexts; they are not independently trained B0. B0/B1 primary trainable instances remain independent; A/C remain independent. Parameter counts include complete implemented models, not just representation utilities. Irregular-domain performance is untested.

Timing includes warmup, synchronized per-case preparation/refinement/posterior and all field queries. Dataset/sensor generation and field reference timing are separate. Diagnostics, including scoring-comparator D conditioning, are outside inference timing. Memory reports are process-wide RSS; they are not isolated per-model peaks. The mixed-PDE median over three cases is descriptive; p95 withheld below 20 cases. No training cost was incurred in 7B; existing successful-update and walltime records are retained separately. The selected C checkpoint has no isolated derivative timer; this is explicitly unavailable. Historical C derivative timings carry their own run identity and are excluded from matched cost comparisons. Physical work counters are explicit analytical operator/field evaluations, zero mesh solves.

Deterministic scientific JSON excludes timing/memory and is content-addressed. Performance records vary with execution. All model weights/buffers verified unchanged. Train/validation/test definitions, priors, noise, physics, model architectures and objectives remain unchanged.

Stage 7C still needs an explicitly authorized training protocol, matched successful-update or wall-clock budget, hardware disclosure, >=3 independent neural seeds, validation model selection and a separate explicit authorization for final untouched-test evaluation. Label-only capacity-matched equation ablations are configuration only. Structural encoder acceptance does not establish physical solver support or unseen-PDE generalization. Stop before Stage 7C.

Figures: {', '.join(Path(p).name for p in plots)}.
'''
    (directory/'diagnostic.md').write_text(report)
    return directory/'diagnostic.md'
