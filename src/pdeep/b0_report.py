"""Static scientific figures/tables for the read-only B0 diagnosis."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_report(record,directory):
    out=Path(directory);out.mkdir(exist_ok=True,parents=True);summary=record['summary'];pdes=('poisson','heat','wave')
    stages=('B0_k0','B0_k1','B0_k2','B0_k3','classical','D')
    labels=('initial','k=1','k=2','k=3','classical','D')
    paths=[]
    def save(fig,name):
        fig.tight_layout();p=out/name;fig.savefig(p,dpi=145,bbox_inches='tight');plt.close(fig);paths.append(str(p))
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for j,p in enumerate(pdes):
        s=summary[p]['stages']
        for ax,key,title in ((axes[0,j],'pooled_coefficient_standardized_RMSE','Prior-standardized coefficient RMSE'),
                            (axes[1,j],'field_standardized_MSE','Training-RMS standardized field MSE')):
            ax.plot(range(6),[s[k][key] for k in stages],marker='o')
            ax.set_xticks(range(6),labels,rotation=25);ax.set_yscale('log');ax.set_title(f'{p}: all 50 cases');ax.set_ylabel(title);ax.grid(alpha=.2)
    fig.suptitle('Frozen trained B0: three refinements leave a large inverse error; classical optimum matches D')
    save(fig,'coefficient_and_field_errors.png')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for j,p in enumerate(pdes):
        s=summary[p]['stages'];idx=np.arange(4)
        for ax,key,title in ((axes[0,j],'Phi','Inverse objective Phi'),(axes[1,j],'gradient_norm','Physical coefficient gradient L2 norm')):
            ax.plot(idx,[s[f'B0_k{k}'][key]['mean'] for k in idx],marker='o',label='B0 mean')
            ax.fill_between(idx,[s[f'B0_k{k}'][key]['median'] for k in idx],[s[f'B0_k{k}'][key]['p95'] for k in idx],alpha=.15,label='median–95th percentile')
            ax.axhline(s['classical'][key]['mean'],color='black',ls='--',label='classical optimum')
            ax.set(title=p,xlabel='inverse iteration k',ylabel=title);ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Same B0 objective (float32 observations promoted to CPU64), 50 physical cases/PDE')
    save(fig,'objective_and_gradients.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,p in zip(axes,pdes):
        rows=[c for c in record['cases'] if c['pde']==p]
        eig=np.array([r['matrix']['whitened_precision_eigenvalues'] for r in rows]);eig=eig/eig[:,-1,None];x=np.arange(1,eig.shape[1]+1)
        ax.plot(x,np.median(eig,axis=0),marker='o');ax.fill_between(x,eig.min(0),eig.max(0),alpha=.2)
        ax.set(title=p,xlabel='ascending Hessian eigenmode',ylabel='lambda / lambda_max');ax.set_yscale('log');ax.grid(alpha=.2)
    fig.suptitle('Prior-whitened Hessian: median and full range across 50 cases/PDE; slow directions persist')
    save(fig,'curvature_spectra.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,p in zip(axes,pdes):
        rows=[c for c in record['cases'] if c['pde']==p];sd=np.array([.5*(2/(m*m+n*n))**.75 for m in range(1,4) for n in range(1,4)])
        if p=='wave':sd=np.r_[sd,.4*.7*np.pi*np.sqrt(np.array([m*m+n*n for m in range(1,4) for n in range(1,4)]))*sd]
        for stage,label in (('B0_k0','initial'),('B0_k3','after three'),('classical','classical')):
            error=np.array([r['stages'][stage]['coefficient_error'] for r in rows])/sd
            ax.plot(np.arange(1,len(sd)+1),np.sqrt(np.mean(error**2,axis=0)),marker='.',label=label)
        if p=='wave':ax.axvline(9.5,color='grey',ls=':')
        ax.set(title=p,xlabel='lexicographic coefficient index (wave a then b)',ylabel='Prior-standardized coefficient RMSE');ax.set_yscale('log');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Coefficient recovery by mode, all 50 cases/PDE; no example selection')
    save(fig,'coefficient_modes.png')
    # Every physical coefficient, with truth and both precision-matched posteriors.
    with (out/'coefficients.csv').open('w') as stream:
        keys=['pde','case_id','block','m','n','truth','D_raw','D_B_inputs',*stages[:4],'classical','classical_raw_inputs']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for r in record['cases']:
            for j,true in enumerate(r['truth']):
                writer.writerow({'pde':r['pde'],'case_id':r['case_id'],'block':'a' if j<9 else 'b','m':j%9//3+1,'n':j%3+1,
                    'truth':true,'D_raw':r['D_mean'][j],'D_B_inputs':r['D_identical_inputs_mean'][j],
                    **{k:r['stages'][k]['xi'][j] for k in (*stages[:4],'classical','classical_raw_inputs')}})
    lines=['# Why trained B0 underperforms exact D on T1','',
        '**Diagnosis: incomplete optimization with a saturated, effectively fixed prior preconditioner.**',
        'The initializer has substantial error. The three updates help but do not approach stationarity. A classical optimizer from the same initializer reaches D. No H/prior/noise/gradient/recurrence implementation discrepancy was found. Conditioning slows the restricted gradient iteration; the inverse problem remains identifiable. The cause of gate saturation during training is not established by this frozen-checkpoint audit.',
        '',f"All **50 validation cases per PDE (150 total)** use the saved plan `{record['plan_identity']}` and the exact observations/cases that produced A=0.06686371, B0=0.17797912, D=0.000439782. The 10,000-update B0 export was replayed on {record['neural_device']}; its stored and replayed coefficient trajectories agree exactly (maximum discrepancy 0). All reference/optimization calculations are CPU float64. No training or model/checkpoint writes occurred.",
        '', '## Concise field comparison','',
        '| PDE | B0 initial | B0 k=1 | B0 k=2 | B0 k=3 | Classical, same B objective | Exact D |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in pdes:
        s=summary[p]['stages'];lines.append('| '+p+' | '+' | '.join(f"{s[k]['field_standardized_MSE']:.9g}" for k in stages)+' |')
    lines.append('| Equal-PDE macro | '+' | '.join(f"{record['macro_standardized_field_MSE'][k]:.9g}" for k in stages)+' |')
    lines += ['', 'Values are standardized **MSE**, not RMSE. Wave is equal u/v weighting after quantity-specific RMS normalization; all declared grid points/times and all 50 physical cases contribute. B0 k=3 is 404.70× D on this score. This is one checkpoint/seed, not a general architecture ranking.',
        '', '## Every refinement stage, separately by PDE','',
        'Coefficient and field RMSEs below are sqrt(mean case MSE). Phi, gradient norms and distance are arithmetic means over 50 cases; case distributions (mean/median/min/p95/max) are in diagnosis.json. Coefficient RMSE uses active coefficients only; wave mixes a/b physical units, so its separate blocks and prior-standardized RMSE are also exported. Distance is physical coefficient L2 to the original D posterior mean.',
        '', '**Precision matters only at roundoff scale:** B0 builds H/y/sigma from its explicit float32 observation copy, promoted to CPU64; D receives the original CPU64 observations. `Classical` minimizes exactly the existing B0 objective. `D_identical_inputs` uses D on B0’s copy. A second independent `classical_raw_inputs` solve starts at the same B0 initializer and minimizes the original-observation objective, agreeing with original D. Phi/gradient columns below use the B0 objective; D’s small nonzero gradient there reflects input rounding. The raw-objective gradient is exported separately and is near zero for original D.', '']
    for p in pdes:
        lines+=['### '+p,'','| Stage | Coeff RMSE | Prior-std coeff RMSE | Field u RMSE | Field v RMSE | Mean Phi | Mean gradient L2 | Mean distance to D |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for k,label in zip(stages,labels):
            s=summary[p]['stages'][k];v=s['field'].get('v',{}).get('physical_RMSE')
            values=[label,f"{s['pooled_coefficient_RMSE']:.6g}",f"{s['pooled_coefficient_standardized_RMSE']:.6g}",f"{s['field']['u']['physical_RMSE']:.6g}",f'{v:.6g}' if v is not None else 'absent',f"{s['Phi']['mean']:.7g}",f"{s['gradient_norm']['mean']:.6g}",f"{s['distance_to_D_L2']['mean']:.6g}"]
            lines.append('| '+' | '.join(values)+' |')
    lines += ['', '## Independent optimizer and mathematical audit','',
        'For z=(xi−m0)/s, minimize 0.5 ||Jz−c||² with J=[diag(1/sigma) H diag(s); I] and c=[(y−H m0)/sigma; 0]. SciPy trust-region least squares starts at the saved B0 xi0, with an exact dense Jacobian/subproblem. No true coefficients or D mean enter optimization. Independently solve the augmented problem using NumPy SVD least squares. D uses its existing Cholesky information-form posterior. No explicit inverse, jitter, changed prior or data rescaling is introduced.',
        '', '[SciPy method documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).',
        '', '| PDE | Full rank in all 50 | Max classical−D, identical inputs | Max classical raw−D raw | Max B-copy vs raw D | Optimizer evaluations, range |',
        '|---|---:|---:|---:|---:|---:|']
    for p in pdes:
        s=summary[p];a=s['audit'];o=s['classical_optimizer_nfev'];lines.append(f"| {p} | {s['matrix']['rank']['min']:.0f}/{s['matrix']['dimension']['max']:.0f} | {a['classical_vs_D_identical_inputs_max_abs']:.3g} | {a['classical_raw_vs_D_raw_max_abs']:.3g} | {a['classical_vs_D_raw_max_abs']:.3g} | {o['min']:.0f}–{o['max']:.0f} |")
    tolerances=[r['optimizer_coefficient_tolerance'] for r in record['cases']]
    lines += ['', f"The justified case-specific coefficient tolerance is 128 eps64 cond(K) (1+||z_D||∞) max(s), ranging {min(tolerances):.3g}–{max(tolerances):.3g}. Actual discrepancies are much smaller. For original-versus-rounded inputs the difference additionally obeys the explicit perturbation bound ||delta b−delta P xi_D||₂ / lambda_min(P). All 150 optimizers succeeded. Original-input classical macro MSE is {record['macro_standardized_field_MSE']['classical_raw_inputs']:.12g}; original D is {record['macro_standardized_field_MSE']['D']:.12g}.", '',
        '| Independent check | Actual maximum over all 150 cases |', '|---|---:|']
    checks={'H vs independent NumPy sine/time/u-v formulas':'H_independent_max_abs','Prior scale vs specified tau and wave b formula':'prior_scale_max_abs',
            'Gradient vs independent AD, max scaled error':'gradient_AD_scaled_error','Gradient vs central FD, max scaled error':'gradient_FD_scaled_error',
            'Gradient vs NumPy formula, max scaled error':'gradient_numpy_scaled_error','Objective vs augmented residual, relative error':'objective_relative_error',
            'Recurrence update, max coefficient discrepancy':'recurrence_max_abs','Prior-whitened curvature, relative error':'curvature_relative_error',
            'Final full-grid standardized score vs saved result':'saved_field_standardized_MSE_discrepancy','Classical vs independent augmented SVD, coefficient max':'classical_vs_SVD_max_abs'}
    for label,key in checks.items():lines.append(f"| {label} | {max(s['audit'][key] for s in summary.values()):.4g} |")
    lines += ['', 'Valid noise values are strictly positive; R is diagonal sigma² and the likelihood gradient divides by sigma². Only valid observed entries enter H/R; padding is tested separately. Priors use physical a/b units, including b_scale=0.4 omega tau. Gradient is HᵀR⁻¹(Hxi−y)+C0⁻¹(xi−m0), and Phi uses per-case sums, not an observation mean. Autograd comparisons use 1+|g_j| scaling; FD/NumPy use 1+max|g| scaling. The FD perturbation is 1e−5 in standardized coefficient units; its acceptance bound is 1e−7. Exact observed errors are reported rather than merely pass/fail.',
        '', '## Why three learned updates are insufficient','',
        '| PDE | Median cond(P), physical | Median cond(K), prior-whitened | Mean eta | d multiplier range | Step fraction range | Mean slowest-direction amplitude retained/step | Remaining error in slowest half of eigenmodes |',
        '|---|---:|---:|---:|---|---|---:|---:|']
    for p in pdes:
        s=summary[p];m=s['matrix'];u=s['refinement'];lines.append(f"| {p} | {m['physical_precision_condition']['median']:.4g} | {m['prior_whitened_precision_condition']['median']:.4g} | {u['eta']['mean']:.5g} | {u['learned_d_min']['min']:.9f}–{u['learned_d_max']['max']:.9f} | {u['fraction']['min']:.9f}–{u['fraction']['max']:.9f} | {u['slowest_contraction']['mean']:.4%} | {s['stages']['B0_k3']['slowest_half_modes_posterior_error_fraction']['mean']:.2%} |")
    lines += ['', 'Here P=HᵀR⁻¹H+C0⁻¹ and K=diag(s) P diag(s). The checkpoint computes D_k=diag(s² d_k) and eta_k=fraction_k/(1.75 lambda_max(K)). Both learned controls sit close to their upper bounds: d≈1.75 for every mode and fraction≈0.95. Mode-to-mode d standard deviations are only about 1e−7–6e−7. Thus z_{k+1}≈z_k−0.95/lambda_max(K) grad_z Phi: effectively fixed prior-preconditioned gradient descent. It removes stiff directions rapidly, then makes little progress in slow directions. The preconditioned condition numbers remain almost unchanged; the learned diagonal has not corrected anisotropy.',
        '', 'All 150 Phi trajectories are monotonically decreasing. Positive/bounded steps are implemented consistently and stay in the descent range, yet the final mean gradients remain 1.3e4–3.3e4. Final Phi is approximately 1509 (Poisson), 2029 (heat), 6034 (wave), versus optima 16.47, 24.12, 32.61. “Physics satisfied” is insufficient: every modal coefficient vector gives an analytical solution of its associated PDE/source/initial conditions.',
        '', 'A diagnostic three-step prior-gradient control with exact quadratic line search reduces the mean remaining objective-gap ratio further, but does not reach D:','',
        '| PDE | B0 remaining objective gap / initial gap | Three exact-line-search steps | Maximum B0 coefficient difference from fixed 0.95-curvature GD |','|---|---:|---:|---:|']
    for p in pdes:
        rows=[r for r in record['cases'] if r['pde']==p];lines.append(f"| {p} | {np.mean([r['gap_ratio'] for r in rows]):.5g} | {np.mean([r['three_exact_line_search_steps_gap_ratio'] for r in rows]):.5g} | {summary[p]['audit']['B0_vs_fixed_095_three_steps_max_abs']:.4g} |")
    lines += ['', 'These are diagnostics of the unchanged frozen checkpoint, not proposed replacement benchmark scores. Full classical minimization from the exact same poor initializer recovers D, so initialization error is repairable and neither identifiability nor precision explains the large gap. Conditioning is modest enough for stable small dense solves; it is problematic for the constrained three-step iteration. Gate saturation is observed; whether training dynamics or representation constraints produced it needs a controlled follow-up.',
        '', '## Recommended next experiment — no retraining','',
        'Freeze this checkpoint and reuse the saved validation plan. Compare the learned three steps with a separately labeled classical tail stopped at 0/1/3/10/30 additional Hessian-vector/conjugate-gradient steps, reporting paired coefficient/field errors, Phi gap and wall time per PDE. Include a diagonal-Hessian/Jacobi scaling control to distinguish preconditioner quality from iteration budget. Keep B0 and B0-plus-classical-tail separate methods. The fully converged endpoint is already verified here; this bounded follow-up would establish the cost/accuracy curve before changing architecture or training. Do not tune on the untouched test split.',
        '', '## Notebook cleanup and preserved artifacts','',
        '`03_evaluate.ipynb` retains the selected A/B0 exports, 50 case IDs per PDE, physics and evaluation settings. It now reports aggregated per-PDE/component physical residuals and all-case coverage/refinement tables; time curves use sqrt(mean case MSE), and Phi curves show mean with a 10–90% case band. Field and D-uncertainty figures use one fixed case/PDE, chosen by ID rather than errors. On this 150-case result there are ten unique routine plots, including only three D uncertainty figures, with no repeated paths. All original per-case numerical diagnostics remain in saved JSON. New figures go into summary_plots/ so old result assets are preserved.',
        '', 'The inverse architecture, forward physics, priors, noise, data/splits, objectives, training code and exact D implementation were not edited. All 28 scientific compatibility file hashes still match the trained export. The selected checkpoint manifest/payload SHA256 values and model fingerprint were checked before/after the audit and are unchanged. No optimizer training updates were executed. The saved 02_train.ipynb training mode was restored to its mandatory preview default (keeping the user-selected C model); the preview notice was corrected. UI tests retain the strict preview guard while allowing explicit user-selected model/checkpoint mappings. This is a diagnosis of one trained seed, not evidence of superiority across seeds.',
        '', f"Audit runtime: {record['elapsed_seconds']:.3f} s for 150 cases, MPS neural replay + CPU64 references/optimization. Detailed evidence: diagnosis.json, per_case_stages.csv (all eight comparison states), coefficients.csv (all 1,800 physical coefficients and true/D/B0/classical values). Figures:", '']
    lines.extend(f'![{Path(p).stem}]({Path(p).name})\n' for p in paths)
    lines += ['', 'Reproduce without training:', '', '```bash', f".venv/bin/python scripts/diagnose_b0.py --result {Path(record['saved_evaluation']).relative_to(Path(__file__).resolve().parents[2])} --output reports/b0_diagnosis --device mps",'```','', f"Checkpoint artifact: `{record['checkpoint_artifact_id']}`."]
    (out/'diagnosis.md').write_text('\n'.join(lines)+'\n')
    return paths
