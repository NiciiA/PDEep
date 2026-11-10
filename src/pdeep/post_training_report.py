"""Concise, case-aggregated views of a separate frozen B0+Classical benchmark."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import require
from .artifact_io import read_manifest
from .post_training_solvers import BUDGETS

FOCUS=('B0','B0+Classical GD 300','B0+Classical Jacobi GD 300','B0+Classical CG','Classical CG from prior','D matched B0 inputs')


def compact_table(record):
    lines=['| PDE | Method | Coeff std MSE | Field std MSE | Mean Phi gap | Iterations median | Inference median / p95 ms | Classical tail median ms |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for pde,methods in record['summary'].items():
        for name in FOCUS:
            r=methods[name];m=r['metrics'];t=r['timing'];lines.append(f"| {pde} | {name} | {m['coefficient_standardized_MSE']['mean']:.5g} | {m['field_standardized_MSE']['mean']:.5g} | {m['optimality_gap']['mean']:.3g} | {r['iterations']['median']:.0f} | {1e3*t['coefficient_total']['median']:.3f} / {1e3*t['coefficient_total']['p95']:.3f} | {1e3*t['classical_tail']['median']:.3f} |")
    return '\n'.join(lines)


def load_for_display(path,plan,checkpoint):
    """Read saved results only. Reject mismatched cases, observations or B0 weights."""
    record=json.loads(Path(path).read_text())
    require(record['version']=='post-training-optimization-v1' and record['neural_training_updates']==0,'unsupported/non-readonly benchmark')
    require(record['checkpoint_artifact_id']==read_manifest(checkpoint,architecture='B0')['artifact_id'],'post-training benchmark uses a different B0 checkpoint')
    require(record['normalizer_identity']==plan.normalizer_identity,'benchmark normalizers mismatch')
    expected={(c.case_identity,c.conditioning_identity) for c in plan.cases}
    actual={(c['case_identity'],c['conditioning_identity']) for c in record['cases']}
    require(expected==actual and len(record['cases'])==len(plan.cases),'post-training benchmark case/observation plan mismatch')
    return record


def make_plots(record,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);paths=[]
    pdes=('poisson','heat','wave');metric_keys=('field_standardized_MSE','coefficient_standardized_MSE')
    titles=('Standardized field MSE','Prior-standardized coefficient MSE')
    families=('GD','Jacobi GD','prior-scaled GD');colors=('tab:blue','tab:orange','tab:purple')
    fig,axes=plt.subplots(2,3,figsize=(13.5,8))
    for col,pde in enumerate(pdes):
        methods=record['summary'][pde]
        for row,(metric,title) in enumerate(zip(metric_keys,titles)):
            ax=axes[row,col]
            for family,color in zip(families,colors):
                names=[f'B0+Classical {family} {k}' for k in BUDGETS]
                vals=[methods['B0 initializer']['metrics'][metric]['mean']]+[methods[n]['metrics'][metric]['mean'] for n in names]
                ax.plot([0,*BUDGETS],vals,marker='.',color=color,label='B0+Classical '+family)
            group=[c['methods']['B0+Classical CG']['iteration_metrics'] for c in record['cases'] if c['pde']==pde]
            length=max(len(t) for t in group)
            vals=[np.mean([t[min(i,len(t)-1)][metric] for t in group]) for i in range(length)]
            ax.plot(range(length),vals,color='tab:green',label='B0+Classical CG')
            ax.scatter([3],[methods['B0']['metrics'][metric]['mean']],marker='x',s=65,color='red',label='B0, learned 3')
            ax.axhline(methods['D matched B0 inputs']['metrics'][metric]['mean'],color='black',ls='--',label='Exact D')
            ax.set(xscale='symlog',yscale='log',title=pde,xlabel='Classical iteration budget (B0 learned point at 3)',ylabel=title)
            ax.set_xticks([0,3,10,30,100,300],[0,3,10,30,100,300]);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Frozen B0: accuracy versus iteration budget | all 50 cases/PDE | converged CG cases held at solution')
    fig.tight_layout(rect=(0,.1,1,.96));path=directory/'accuracy_vs_iterations.png';fig.savefig(path,dpi=145);plt.close(fig);paths.append(path)
    fig,axes=plt.subplots(2,3,figsize=(13.5,8))
    for col,pde in enumerate(pdes):
        methods=record['summary'][pde]
        for row,(metric,title) in enumerate(zip(metric_keys,titles)):
            ax=axes[row,col]
            for family,color in zip(families,colors):
                names=[f'B0+Classical {family} {k}' for k in BUDGETS]
                ax.plot([methods[n]['timing']['coefficient_total']['median']*1e3 for n in names],
                        [methods[n]['metrics'][metric]['mean'] for n in names],marker='.',color=color,label='B0+Classical '+family)
            for name,marker,color in (('B0','x','red'),('B0+Classical CG','o','tab:green'),
                                     ('Classical CG from prior','s','tab:cyan'),('D matched B0 inputs','*','black')):
                t=methods[name]['timing']['coefficient_total'];x=t['median']*1e3;y=methods[name]['metrics'][metric]['mean']
                ax.errorbar(x,y,xerr=[[0.],[(t['p95']-t['median'])*1e3]],fmt=marker,color=color,markersize=7,capsize=2,label=name)
            ax.set(xscale='log',yscale='log',title=pde,xlabel='Coefficient/posterior inference, ms (includes initializer)',ylabel=title);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('All 50 cases/PDE | median case latency, whisker to p95 | D includes complete posterior fit')
    fig.tight_layout(rect=(0,.1,1,.96));path=directory/'accuracy_vs_wall_clock.png';fig.savefig(path,dpi=145);plt.close(fig);paths.append(path)
    return paths


def write_report(record,directory,*,cpu_control=None):
    directory=Path(directory);plots=make_plots(record,directory);summary=record['summary'];pdes=('poisson','heat','wave')
    lines=['# Frozen B0 + Classical post-training optimization benchmark','',
        f"**150 matched validation cases: 50 each of Poisson, heat and wave.** Frozen trained B0 `{record['checkpoint_artifact_id']}`. No weights, architectures, training data or physical definitions changed; zero neural training updates.",
        '', '## Protocol','',record['precision_policy'], '',record['GD_policy'],'',record['CG_policy'],'',
        'All main methods receive the same observations, strictly positive diagonal noise covariance, physical modal prior and physical case. The matched input copy is exactly the one already used by unmodified B0. Original-float64 D is retained separately; this does not change the original scientific benchmark or reference generator. Every classical run starts at B0 xi0, not after its three learned updates. The prior-start CG control uses m0 and no neural network. Metrics for the original-float64 D row are evaluated on the common matched-copy objective; its tiny nonzero distance/gap/gradient there reflects known input rounding, not an incomplete D solve.',
        '', '## Concise comparison','',compact_table(record),'',
        '“Std MSE” uses the saved training-only RMS/prior scales. All grid points, declared physical times and quantities are included; wave u/v are normalized separately then equally weighted. Equal physical cases are averaged per PDE. Incomplete methods can have slightly lower *realized* validation MSE than D through finite-sample variation; the posterior objective gap and residual are the convergence tests, not whether truth error crosses D on these 50 cases.',
        '', '## Accuracy versus iterations and cost','']
    lines += [f'![{p.stem}]({p.resolve()})\n' for p in plots]
    lines += ['', 'Each curve aggregates all 50 cases/PDE. CG trajectories retain converged cases at their final solution so later iteration points still average the same cases. All hybrid labels are B0+Classical. These are separate post-training controls, not modifications or new training of B0.',
        '', '## Timing scope and decomposition','',record['timing_policy'],'',record['field_timing_note'],'',
        f"Device: {record['device']} for B0 neural float32, CPU float64 for all physical/classical computations. {record['repeats']} warm timing repeats/case; torch CPU threads fixed at {record['CPU_threads_during_benchmark']}. Environment thread settings: {record['thread_environment']}. Case-median latencies are summarized across the 50 physical cases; a p95 across all repeated calls is also exported. A hybrid total is the paired sum of measured preparation + initializer head + classical tail. B0 total is preparation + native infer_unknowns (which includes its own head and exactly three refinements). These are measured component sums, not a claim of separately timed fused-pipeline execution. Scaling/precision/RHS formation and CG true-residual checks are inside tail timers. Loading, scoring and mathematical cross-checks are outside timers.",
        '', '| PDE | B0 initializer median ms | B0 total median / p95 ms | B0+Classical CG total median / p95 ms | CG classical tail median ms | D complete median / p95 ms | CG prior median / p95 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for pde in pdes:
        m=summary[pde];time=lambda n:m[n]['timing']['coefficient_total']
        fmt=lambda n:f"{time(n)['median']*1e3:.3f} / {time(n)['p95']*1e3:.3f}"
        lines.append(f"| {pde} | {time('B0 initializer')['median']*1e3:.3f} | {fmt('B0')} | {fmt('B0+Classical CG')} | {m['B0+Classical CG']['timing']['classical_tail']['median']*1e3:.3f} | {fmt('D matched B0 inputs')} | {fmt('Classical CG from prior')} |")
    lines += ['', 'D timing calls the unchanged ExactGaussianD.fit API: H/prior construction, prior/posterior Cholesky, posterior mean, covariance and covariance factor, plus existing eigenspectrum/SVD diagnostics. It is not just a precomputed mean solve. No posterior matrix or factorization is shared with classical solvers. Hybrids may reuse their own B0 physical H construction, whose cost is charged in preparation; the classical precision/RHS/scaling are built anew in the measured tail.',
        '', '| PDE | Shared full-grid mean decoder median ms | Estimated B0+CG + decoder median ms | Estimated complete D + decoder median ms |',
        '|---|---:|---:|---:|']
    for pde in pdes:
        a=summary[pde];lines.append(f"| {pde} | {np.median(record['shared_full_grid_mean_decoder_seconds'][pde])*1e3:.3f} | {a['B0+Classical CG']['timing']['estimated_with_shared_field_decode']['median']*1e3:.3f} | {a['D matched B0 inputs']['timing']['estimated_with_shared_field_decode']['median']*1e3:.3f} |")
    lines += ['', 'The decoder row is an explicitly labeled additive estimate using measured common CPU64 analytic mean decoding on the full canonical grid/times/u-v quantities. It is not independently timed end-to-end latency and does not include predictive variance field queries. Exact D’s coefficient covariance computation remains included in its posterior timing.',
        '', '## Every budget, separately by PDE','',
        'All counts and metrics below include every one of the 50 cases/PDE; full per-case values and raw timing repeats are in result.json and per_case.csv. Distance is physical coefficient L2 to matched D; standardized distance is additionally exported. Gradient is the physical coefficient Euclidean norm. Optimality gap uses 0.5(xi−m_post)^T P(xi−m_post), avoiding subtractive cancellation.', '']
    for pde in pdes:
        lines += ['### '+pde,'','| Method | Coeff std MSE | Field std MSE | Mean distance D | Mean Phi | Mean gap | Mean grad norm | Iter median/max | Obj / grad / precision-MV, median | Inference median/p95 ms |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name,r in summary[pde].items():
            m=r['metrics'];c=r['counts'];t=r['timing']['coefficient_total']
            lines.append(f"| {name} | {m['coefficient_standardized_MSE']['mean']:.6g} | {m['field_standardized_MSE']['mean']:.6g} | {m['distance_to_D_L2']['mean']:.3g} | {m['Phi']['mean']:.6g} | {m['optimality_gap']['mean']:.3g} | {m['gradient_norm']['mean']:.3g} | {r['iterations']['median']:.0f}/{r['iterations']['max']:.0f} | {c['objective_evaluations']['median']:.0f}/{c['gradient_evaluations']['median']:.0f}/{c['precision_matvecs']['median']:.0f} | {t['median']*1e3:.3f}/{t['p95']*1e3:.3f} |")
    lines += ['', '## Work counters and independent verification','',record['counts_policy'],'',
        'For N fixed GD steps there are exactly N gradient evaluations and N precision matvecs; no objective evaluations are needed by this fixed-step algorithm. Every evaluation endpoint is scored separately, outside optimization timing/counters. B0 itself evaluates four states (xi0…xi3): four objectives, four analytical gradients and eight H/H^T sensor products. An initializer requires its existing physical H/whitened-precision curvature preparation, but no objective/gradient call. The CSV and JSON separately count precision/RHS builds, eigensolves, true CG residual recomputations, two D Cholesky factorizations, covariance formation/triangular solve and two SVD diagnostics. A zero precision-matvec count for D does not mean zero work.',
        '', '| PDE | Max gradient vs production, scaled | Max gradient vs AD, scaled | Max independent CG relative residual | Max CG residual norm | Max CG coefficient distance to D | CG iterations: initializer / prior median (max) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for pde in pdes:
        group=[c for c in record['cases'] if c['pde']==pde];cg=summary[pde]['B0+Classical CG'];prior=summary[pde]['Classical CG from prior']
        lines.append(f"| {pde} | {max(c['audit']['gradient_vs_production_scaled'] for c in group):.3g} | {max(c['audit']['gradient_vs_AD_scaled'] for c in group):.3g} | {max(c['methods']['B0+Classical CG']['metrics']['independent_relative_residual'] for c in group):.3g} | {max(c['methods']['B0+Classical CG']['independent_residual_norm'] for c in group):.3g} | {cg['metrics']['distance_to_D_L2']['max']:.3g} | {cg['iterations']['median']:.0f} ({cg['iterations']['max']:.0f}) / {prior['iterations']['median']:.0f} ({prior['iterations']['max']:.0f}) |")
    lines += ['', 'All posterior precision matrices are checked symmetric positive definite. Gradients are checked against unchanged PhysicalSystem.state and CPU64 autograd; the independent residual is recomputed in sensor form H^T R^-1(Hxi−y)+C0^-1(xi−m0), rather than trusting CG’s recursive residual. CG must meet max(atol,rtol||b||) with a scale-aware float64 arithmetic allowance; coefficient distance is bounded independently by residual norm/lambda_min(P). Finite precision can require more than dimension iterations, particularly for wave. No jitter, explicit inverse or benchmark modifications are used. Unit tests additionally compare CG against SciPy’s independent implementation and D. [SciPy CG reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html).',
        '', '## Computational benefit','',
        '| PDE | B0+CG / complete D, median latency ratio | Mean paired B0+CG minus D, ms [95% case bootstrap CI] |',
        '|---|---:|---:|']
    for pde in pdes:
        r=record['paired_time_comparison_to_D'][pde]['B0+Classical CG'];ci=r['paired_case_bootstrap_CI95']
        lines.append(f"| {pde} | {r['median_time_ratio_to_D']:.2f}× | {r['mean_time_difference_seconds']*1e3:.3f} [{ci[0]*1e3:.3f}, {ci[1]*1e3:.3f}] |")
    if cpu_control:
        lines+=['','CPU-only sensitivity control (same checkpoint, cases, precision rules and seven-repeat protocol; independent model instance, no weight updates):','',
                '| PDE | CPU B0 median / p95 ms | CPU B0+CG median / p95 ms | CPU complete D median / p95 ms |','|---|---:|---:|---:|']
        for pde in pdes:
            s=cpu_control['summary'][pde];fmt=lambda n:f"{s[n]['timing']['coefficient_total']['median']*1e3:.3f} / {s[n]['timing']['coefficient_total']['p95']*1e3:.3f}"
            lines.append(f"| {pde} | {fmt('B0')} | {fmt('B0+Classical CG')} | {fmt('D matched B0 inputs')} |")
    lines += ['', '**Recommendation:** for this small 9/18-dimensional, correctly specified T1 problem, use the direct Gaussian solve when the model/operator is known. It returns the exact posterior uncertainty as well as the optimum mean. The learned initializer does not materially reduce converged CG iterations relative to a prior-mean start, and its preparation cost exceeds the solve cost. The existing learned three-step preconditioner does not offer a computational accuracy/cost advantage here; the nonlearned prior-scaling control and Jacobi results make that comparison explicit. These conclusions are specific to this frozen checkpoint, implementation and hardware, not a claim about learned inverse methods in general. Prior-start CG can be somewhat faster if only the mean is needed; it does not provide D’s posterior covariance. There is no evidence here to justify more neural training for computational advantage on this exact T1 inverse task.',
        '', 'No neural training was started. All 28 scientific compatibility hashes, the model fingerprint, selected checkpoint manifest/payload hashes and original saved validation-result hash are unchanged. Full direct solves and iterative controls are separate labeled benchmark utilities; B0 itself is unmodified. The evaluation notebook only reads compatible saved results and displays this concise table plus two figures; it does not automatically rerun the benchmark.',
        '', f"Runtime: {record['elapsed_seconds']:.2f} s for 150 cases and seven warm timing repeats. Source/versions/raw samples/work counters: result.json. Per-case metrics: per_case.csv.",
        '', 'Reproduce explicitly, without training:','', '```bash',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/benchmark_post_training.py --device mps --repeats 7',
        '```']
    path=directory/'report.md';path.write_text('\n'.join(lines)+'\n');return plots
