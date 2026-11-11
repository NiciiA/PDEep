"""Summarize measured G3 evidence only. No solver/model/optimizer execution."""
from pathlib import Path
import json,csv,hashlib,platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/wave_fidelity_g3'
def read(name):
 p=OUT/name;return json.loads(p.read_text()) if p.exists() else None

temporal=read('temporal.json');analytical=read('analytical-square.json');original=read('original-prior.json');correlation=read('correlation-length.json');regularity=read('initial-regularity.json');gauss=read('gauss-temporal.json');dispersion=read('dispersion.json');spectrum=read('unresolved-spectrum.json')
candidates=[]
for path in sorted(OUT.glob('candidate-*.json')):
 r=json.loads(path.read_text())
 if 'P4_h02_vs_P3_h01' in r:candidates.append(r)

def maxerr(row,key,quantity,metric='standardized_RMSE'):return max(e[metric] for e in row[key][quantity])

# Explicitly labeled exact-error vs mesh-difference export, per physical time.
flat=[]
def export(kind,family,method,h,q,items):
 for e in items:flat.append({'comparison':kind,'geometry':family,'method':method,'h':h,'quantity':q,**e})
for r in analytical['rows']:
 for q in ('u','v'):export('error_against_exact_T1','square','P'+str(r['degree'])+' exact semidiscrete',r['h'],q,r['errors_against_analytical'][q])
for family in original['families']:
 for r in family['rows']:
  for q in ('u','v'):export('difference_to_P1_h01_NOT_exact',family['family'],'P1 exact semidiscrete',r['h'],q,r['differences_to_h01'][q])
for r in candidates:
 for q in ('u','v'):export('difference_P4_h02_to_P3_h01',r['family'],'C3 candidate P4 h02 vs independent finer P3',.02,q,r['P4_h02_vs_P3_h01'][q])
with (OUT/'errors-by-geometry-time.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for q,ax in zip(('u','v'),axes):
 for d in (1,2,4):
  rows=[r for r in analytical['rows'] if r['degree']==d]
  ax.loglog([r['actual_h_max'] for r in rows],[max(z['standardized_RMSE'] for z in r['errors_against_analytical'][q]) for r in rows],'-o',label='P'+str(d))
 ax.axhline(.002,color='gray',ls=':',label='0.2% amplitude guide');ax.set_title('Exact T1 analytical '+q);ax.set_xlabel('actual maximum mesh edge');ax.set_ylabel('max-time RMS error / fixture RMS');ax.legend();ax.grid(alpha=.3)
fig.savefig(OUT/'analytical-spatial-convergence.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for q,ax in zip(('u','v'),axes):
 ax.loglog([r['dt'] for r in temporal['temporal']],[r['max_'+q+'_L2_error'] for r in temporal['temporal']],'-o',label='Midpoint + Hermite')
 if gauss:ax.loglog([r['dt'] for r in gauss['rows']],[r['max_'+q+'_L2_error'] for r in gauss['rows']],'-o',label='Gauss4 + Hermite')
 ax.set_title('Fixed spatial system: '+q);ax.set_xlabel('dt');ax.set_ylabel('max-time mass-norm error vs full eigen control');ax.legend();ax.grid(alpha=.3)
fig.savefig(OUT/'temporal-convergence.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True);x=np.arange(len(original['families']))
for q,ax in zip(('u','v'),axes):
 for j,h in enumerate((.04,.02)):
  values=[max(z['standardized_RMSE'] for z in next(r for r in f['rows'] if r['h']==h)['differences_to_h01'][q]) for f in original['families']]
  ax.bar(x+(j-.5)*.35,values,.35,label='h='+str(h)+' vs .01')
 ax.axhspan(.02,.08,color='orange',alpha=.2,label='sensor noise scale');ax.axhline(.002,color='black',ls=':',label='reference target');ax.set_xticks(x,[f['family'] for f in original['families']],rotation=35);ax.set_title('Original V1 '+q+' (finest is NOT exact truth)');ax.set_ylabel('max-time RMS difference / frozen training RMS');ax.legend(fontsize=7)
fig.savefig(OUT/'original-prior-fidelity.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for f in regularity['families']:axes[0].loglog([r['h'] for r in f['rows']],[r['initial_acceleration_L2'] for r in f['rows']],'-o',label=f['family'])
axes[0].set_xlabel('h');axes[0].set_ylabel('mass norm of discrete initial acceleration');axes[0].set_title('Same physical P1 field: norm grows about h^-1/2');axes[0].legend(fontsize=6);axes[0].grid(alpha=.3)
for r in correlation['rows']:
 om=np.array(r['eigen_omega']);E=np.array(r['energy_by_eigenmode']);tail=np.cumsum(E[::-1])[::-1]/E.sum();axes[1].semilogy(om,tail,label='ell='+str(r['length']))
axes[1].set_xlabel('angular frequency');axes[1].set_ylabel('fraction of initial wave energy above frequency');axes[1].set_title('Same white noise: correlation-length diagnostic');axes[1].legend();axes[1].grid(alpha=.3)
fig.savefig(OUT/'prior-regularity-spectrum.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for r in dispersion['rows']:axes[0].semilogy(np.arange(1,13),np.abs(r['relative_frequency_error']),'-o',ms=3,label=f'P{r["degree"]} h={r["h"]}')
axes[0].set_xlabel('square eigenmode index (including multiplicities)');axes[0].set_ylabel('relative frequency error');axes[0].set_title('Spatial dispersion only');axes[0].legend(fontsize=6);axes[0].grid(alpha=.3)
axes[1].loglog([r['dt'] for r in temporal['temporal']],[r['relative_grid_energy_drift'] for r in temporal['temporal']],'-o',label='Midpoint')
if gauss:axes[1].loglog([r['dt'] for r in gauss['rows']],[r['diagnostics']['relative_grid_energy_drift'] for r in gauss['rows']],'-o',label='Gauss4')
axes[1].set_xlabel('dt');axes[1].set_ylabel('max relative grid energy drift');axes[1].set_title('Energy conservation does not certify phase');axes[1].legend();axes[1].grid(alpha=.3)
fig.savefig(OUT/'dispersion-energy.png',dpi=160);plt.close(fig)
if candidates:
 fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
 for q,ax in zip(('u','v'),axes):
  ax.bar([r['family'] for r in candidates],[maxerr(r,'P4_h02_vs_P3_h01',q) for r in candidates]);ax.axhline(.002,color='black',ls=':',label='reference RMS target');ax.axhline(.0002,color='orange',ls=':',label='temporal sub-budget');ax.set_title('Versioned C3 candidate '+q);ax.set_ylabel('max-time RMS difference / NEW training RMS');ax.tick_params(axis='x',rotation=35);ax.legend(fontsize=7)
 fig.suptitle('P4 h=.02 vs finer P3 h=.01: different prior, NOT a repair of V1 cases');fig.savefig(OUT/'candidate-fidelity.png',dpi=160);plt.close(fig)

baseline=read('baseline.json');changed=[];modified=[];protected=[]
for name,sha in baseline['scientific_files'].items():
 if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:changed.append(name)
for name,sha in baseline['protected_source'].items():
 if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:protected.append(name)
for name,(size,mtime) in {**baseline['artifacts'],**baseline['old_geometry_checkpoints']}.items():
 p=Path(name);p=p if p.is_absolute() else ROOT/p
 if not p.exists() or [p.stat().st_size,p.stat().st_mtime_ns]!=[size,mtime]:modified.append(str(p))
assert not changed and not modified
assert set(protected)<={'src/pdeep/geometric/domain.py','src/pdeep/geometric/training.py'}
source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src/pdeep/wave_fidelity','scripts') for p in (ROOT/d).glob('*.py') if d!='scripts' or 'wave' in p.name}
preservation={'unchanged_T1_scientific_files':len(baseline['scientific_files']),'changed_T1_scientific_files':changed,'modified_original_artifacts':modified,
              'unchanged_original_artifact_count':len(baseline['artifacts']),'reviewed_geometry_utility_changes':protected,'neural_architecture_files_changed':False,'new_neural_optimizer_updates':0}
write=lambda name,value:(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
write('preservation.json',preservation)
complete=len(candidates)==8
candidate_pass=complete and all(r['necessary_RMS_gate'] and r['fine_temporal_gate'] and all(maxerr(r,'P4_h02_vs_P3_h01',q,'standardized_max_error')<.01 for q in ('u','v')) for r in candidates)
status={'original_G1_V1':'BLOCKED','candidate_V2':'bounded checks pass' if candidate_pass else 'incomplete or failed; see per-family flags',
        'candidate_families_completed':len(candidates),'benchmark_or_prior_automatically_changed':False,'full_training_enabled':False,'new_neural_optimizer_updates':0,
        'source':source,'CPU_precision':'float64, complex128 internally for Gauss','python':platform.python_version()}
import scipy,triangle,shapely,importlib.metadata
status['dependency_versions']={'numpy':np.__version__,'scipy':scipy.__version__,'triangle':triangle.__version__,'shapely':shapely.__version__,
                               'torch':importlib.metadata.version('torch'),'ipywidgets':importlib.metadata.version('ipywidgets')}
write('readiness.json',status)

lines=['# G3 wave reference fidelity investigation','','**Original G1 V1 remains blocked for research training. No neural optimizer updates occurred in this task.**','',
'## Findings and corrections','',
'1. **Fine-mesh implementation defect fixed.** Scientific notation in Triangle area switches was parsed as switches: h=.01 gave `a5e-05` and only four square vertices. Positional decimal formatting and a realized maximum-cell-area gate now enforce true refinement. The original h=.04/.02 comparisons were valid; the bug prevented meaningful finer follow-up.',
'2. **The canonical P1 prior is the primary structural bottleneck.** Discrete squared elliptic smoothing does not make its continuous P1 function globally smooth. Gradient jumps create broad physical frequencies. Wave velocity multiplies displacement frequencies by omega; wave evolution does not diffuse the tail away. This is valid finite-energy data, not a claim that the PDE is ill-posed. Pointwise velocity accuracy is much harder than initial L2 interpolation accuracy.',
'3. **Midpoint system/assembly are correct.** Consistent mass, c² stiffness, free/Dirichlet indexing and wave first-order signs agree with independently assembled matrices and exact semidiscrete propagation. Energy conservation coexists with phase error.',
'4. **Added accurate reference controls.** Independent P1–P4 assembly, full generalized eigenbasis, block exponential, direct continuous-time matrix functions, and a checked fourth-order Gauss implementation. No spatial eigenmodes are silently filtered. Dense-output velocity equals the derivative of its Hermite displacement.',
'5. **Run All cannot start research training from a persisted setting.** Routine/archival launch cells only register fresh-click button callbacks. The G1 engine itself rejects research/bounded runs while V1 is uncertified. Verification deliberately forced saved train mode with training engines and optimizers blocked.','',
'## Independent temporal controls','',
f'- Complete eigenbasis residual {temporal["eigen_diagnostics"]["relative_eigen_residual"]:.3e}, maximum mass-orthogonality error {temporal["eigen_diagnostics"]["mass_orthogonality_max"]:.3e}.',
f'- Chebyshev versus complete eigenbasis max u {temporal["Chebyshev_vs_eigen_max_u"]:.3e}, max v {temporal["Chebyshev_vs_eigen_max_v"]:.3e}; independent block exponential discrepancy {temporal["expm_vs_eigen_max"]:.3e}.',
'- Same canonical spatial system/fields throughout this dt study. Errors below are physical mass-norm maxima over the fixed arbitrary-time suite.','',
'| dt | midpoint u | midpoint v | midpoint v rate | Gauss4/Hermite u | Gauss4/Hermite v |','|---:|---:|---:|---:|---:|---:|']
for i,r in enumerate(temporal['temporal']):
 g=gauss['rows'][i];rate=r.get('v_rate');lines.append(f'| {r["dt"]:.4g} | {r["max_u_L2_error"]:.3e} | {r["max_v_L2_error"]:.3e} | {rate:.4f}'+' | '+f'{g["max_u_L2_error"]:.3e} | {g["max_v_L2_error"]:.3e} |' if rate is not None else f'| {r["dt"]:.4g} | {r["max_u_L2_error"]:.3e} | {r["max_v_L2_error"]:.3e} | — | {g["max_u_L2_error"]:.3e} | {g["max_v_L2_error"]:.3e} |')
lines+=['',
'Midpoint amplification is unit modulus with theta=2 atan(omega dt/2); Gauss4 uses the unit-modulus [2/2] Pade phase. Neither is exact at arbitrary omega*dt. Maximum midpoint grid energy drift: '+f'{max(r["relative_grid_energy_drift"] for r in temporal["temporal"]):.3e}'+'. Gauss4 grid order is4; coherent Hermite velocity can be order3 between nodes, so off-grid time halving is included. Direct matrix functions introduce no frame interpolation. A half-angle lambda mapping fixes cancellation at fine-mesh Chebyshev nodes without relaxing the numerical tolerance.','',
'## Analytical square comparison','',
'The same fixed nine-mode T1 coefficients and physical points are evaluated directly at t=0,.0073,.02,.05,.1,.237,.5,1. Errors here are against the **existing exact analytical evaluator**, not a finer FEM mesh. Normalization is explicitly this analytical fixture space/time RMS; it is not a recalibration of the T1 noise model.','',
'| Space | h | free DOFs | max-time u RMSE / fixture RMS | max-time v RMSE / fixture RMS |','|---|---:|---:|---:|---:|']
for r in analytical['rows']:lines.append(f'| P{r["degree"]} | {r["h"]} | {r["free_dofs"]} | {max(z["standardized_RMSE"] for z in r["errors_against_analytical"]["u"]):.6g} | {max(z["standardized_RMSE"] for z in r["errors_against_analytical"]["v"]):.6g} |')
lines+=['','For the first12 square eigenmodes, P1 h=.01 maximum t=1 phase discrepancy is '+f'{max(abs(v) for v in dispersion["rows"][2]["phase_error_at_t1"]):.6g} rad; P4 h=.04 is {max(abs(v) for v in dispersion["rows"][-1]["phase_error_at_t1"]):.6g} rad'+'. This isolates spatial dispersion from the time integrator.','',
'## Unchanged canonical-P1 prior, all geometry families','',
'Points and initial functions are identical across h=.04/.02/.01. The table reports **differences to h=.01**, not exact PDE errors. Each cell is the maximum RMS over the stated physical times, divided by the frozen original training RMS.','',
'| Geometry | u .04 vs .01 | v .04 vs .01 | u .02 vs .01 | v .02 vs .01 |','|---|---:|---:|---:|---:|']
for f in original['families']:
 a,b=f['rows'];lines.append('| '+f['family']+' | '+' | '.join(f'{max(z["standardized_RMSE"] for z in r["differences_to_h01"][q]):.6f}' for r in (a,b) for q in ('u','v'))+' |')
lines+=['','The finer spatial study still exceeds the2–8% noise scale. The old original-prior RMS is frozen at '+str(original['scales'])+'. It was not increased or recalibrated on validation data.','',
'| Geometry | initial acceleration norm h=.04 | h=.02 | h=.01 |','|---|---:|---:|---:|']
for f in regularity['families']:lines.append('| '+f['family']+' | '+' | '.join(f'{r["initial_acceleration_L2"]:.5f}' for r in f['rows'])+' |')
lines+=['','The approximately sqrt(2) growth per halving is consistent with a nonsmooth strong Laplacian of the fixed P1 function. Nested conforming P1/P2 controls preserve the original initial function and energy (~0.3319135664446) to roundoff, yet retain large velocity differences. This rules out re-sampling or initial interpolation as the sole cause.','',
'Correlation-length diagnostic uses **the same white-noise draws**, never substitutes them into the benchmark:','',
'| ell | energy fraction above omega15 | above omega30 |','|---:|---:|---:|']
for r in correlation['rows']:lines.append(f'| {r["length"]} | {r["above_omega15_fraction"]:.6f} | {r["above_omega30_fraction"]:.6f} |')
lines+=['','The particular validation field used in the temporal control has9.24% of its canonical semidiscrete energy above omega15. The canonical P1 maximum omega is59.56, but P2 on the SAME physical P1 initial field places3.665% of its energy above omega60 (a nested P1 control gives2.883%). Finite prior dimension does not make its continuous P1 function band-limited. See unresolved-spectrum.json. Changing ell is not sufficient to certify pointwise references and was not adopted.','',
'## Versioned C3 candidate, separate physical prior','',
'`G1-wave-C3-compact-Gaussian-candidate-v2` uses32 independent-Gaussian compact C3 bumps per initial quantity, radius .18 sqrt(area), with supports strictly inside the polygon. The covariance/center construction is explicit in the addendum. This is a declared new regularity restriction, **not silently smoothed V1 data**. No existing data generator or checkpoint has been switched to it.',
'Its four training-calibration fields use different keys from validation, on square/rectangle/L/disk; eight Gauss time nodes and256 area queries are used. The resulting scales are '+str(read('candidate-normalizers.json')['scales'])+'. This is a small declared finite training estimate, not a population theorem.',
'Each candidate has P2 h=.04/.02/.01, P4 h=.04/.02 and an independent finer P3 h=.01 comparator. The latter uses Gauss dt halving, .004/.002 and, if required, .002/.001. Values below are cross-discretization differences, not rigorous errors against exact irregular-domain solutions.','',
'| Geometry | max-time u RMS / new training RMS | v RMS / new training RMS | max point v / RMS | temporal RMS max | bounded checks |','|---|---:|---:|---:|---:|---|']
for r in candidates:
 e=r['P4_h02_vs_P3_h01'];u=max(z['standardized_RMSE'] for z in e['u']);v=max(z['standardized_RMSE'] for z in e['v']);point=max(z['standardized_max_error'] for z in e['v']);t=max(z['standardized_RMSE'] for q in ('u','v') for z in r['fine_temporal_halving'][q]);passed=r['necessary_RMS_gate'] and r['fine_temporal_gate'] and point<.01
 lines.append(f'| {r["family"]} | {u:.6g} | {v:.6g} | {point:.6g} | {t:.6g} | {"pass" if passed else "FAIL"} |')
lines+=['',f'Candidate families completed: {len(candidates)}/8. Candidate bounded checks: {"PASS" if candidate_pass else "NOT ACCEPTED (incomplete/failing checks)"}. The original V1 gate remains **BLOCKED**.',
'The target RMS .002*s_q is one tenth of the smallest sensor noise .02*s_q; temporal sub-budget is .0002*s_q, and sampled maximum error target .01*s_q. These are empirical finite-suite checks, not certification for every Gaussian draw. Fixed64-facet geometric approximation errors are outside this polygon-PDE validation. The candidate is not automatically adopted as the scientific training distribution.','',
'## Costs, reproducibility and regression','',
'All numerical calculations ran CPU float64, complex128 only in rational factorization. Reference tensors are [T,Q] u/v, with consistent sparse [V,V] matrices and variable degree/mesh DOFs. Fields, points, times, prior/RMS and mesh identities are recorded. Fine reference controls can take minutes per geometry; cost is not neural inference time. Configurations are individually cached with source/field/query identities. The original Cayley-product Gauss implementation and its equivalent faster single-resolvent form are independently checked; cached measurements retain their original provenance. Chebyshev caches are reused across that optimization only after AST equality checks of all used assembly/matrix-function code.','',
'No optimizer calls occurred. The regression runner patches Adam/AdamW/SGD step to fail and excludes legacy tests whose actual purpose is optimizer updates. See regression-final.log for the exact count; separate numerical tests and an actual saved-train Run All are also recorded. Notebook safety artifact: '+str(read('notebook-safety.json'))+'.',
'All'+str(len(baseline['artifacts']))+' pre-existing artifact file sizes/mtimes and old geometry checkpoints remain unchanged. All28 canonical T1 scientific source hashes remain unchanged. Neural architecture source hashes are unchanged. Changes inside the old geometry namespace are only the mesher correctness fix and the research run gate.',
'Geometry checkpoint source fingerprints intentionally detect those changes. An explicit reviewed migration verifies every numerical mesh/prior is identical, copies the full payload unchanged into a **new** checksummed artifact, and records the parent. An affected fine-mesh or unknown-source checkpoint is rejected; see checkpoint-compatibility-final.json.',
'Existing trained T1 A/B0 exports both pass strict stored-example replay on their recorded native MPS device with zero discrepancy. B0 also passes CPU replay. A CPU replay has maximum absolute discrepancy2.265e-6 and fails the archived tolerance at some entries; the production loader still rejects it. This is recorded as a device replay limitation, not hidden by relaxing a tolerance or rewriting a checkpoint. T1 architecture/scientific source hashes and tensor payloads are unchanged. See T1-exports-cpu.json and T1-exports-mps.json.',
'The prior49-update unintended notebook run belongs to the preceding Phase G task. There were **zero new unintended or deliberate neural updates here**. No old artifacts were overwritten. Persistent additions: numerical tools/tests/docs/reports, corrected formatter/gate, event-only notebook controls, explicit migrated copies, and the ipywidgets dependency. No flow matching, T2/T3 or neural architecture changes were made.','',
'## Readiness','',
'**Do not start original G1 V1 research training.** Fixing the mesh parser and time propagation does not make a nonsmooth prior reference mesh-converged. Retaining V1 needs a much finer/adaptive or independently certified rough-data reference. The versioned smooth candidate offers a practical, separately tested alternative; adopting it requires an explicit benchmark/prior selection and its own stored normalization/provenance, never overwriting V1. No research-training gate has been opened in this task.','',
'## Reproduction and sources','',
'Commands are in docs/scientific_contract_geometry_g3_reference_v2.md. Use `--output` for a separate experiment directory; preserve old outputs. The optional candidate study is substantially more expensive than the coarse controls.',
'Complete data: errors-by-geometry-time.csv, analytical-square.json, temporal.json, gauss-temporal.json, dispersion.json, original-prior.json, initial-regularity.json, unresolved-spectrum.json, correlation-length.json, candidate-*.json, readiness.json and preservation.json.',
'Generalized eigenproblem conventions follow [SciPy eigh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html); the independent block exponential uses [SciPy expm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html). Triangle exposes fixed/per-cell area constraints through its [official wrapper API](https://rufat.be/triangle/API.html). The formatter failure itself is a directly reproduced negative control, not an inference from those docs.']
log=(OUT/'regression-final.log').read_text()
summary=next((line for line in log.splitlines() if ' passed,' in line),'see regression-final.log')
lines+=['','## Final verification evidence','',summary+'. Optimizer calls:0. Legacy tests whose purpose is performing optimizer updates were intentionally excluded. The current numerical module also passes24 standalone tests.','',
'The optional Gauss reference uses a fixed physical[0,1] grid independent of query chunks; batched/individual arbitrary-time checks pass. All reported full-horizon numerical studies already use this same grid, so this consistency correction does not change their values.',
'Malformed fine-mesh pilot outputs are retained for audit and marked superseded in superseded-results.json; they are excluded from all acceptance calculations. mesher-negative-control.json reproduces the bad switches and records corrected realized areas/node counts.','',
'## Convergence plots','']
for title,name in [('Exact analytical square spatial convergence','analytical-spatial-convergence.png'),('Independent temporal convergence','temporal-convergence.png'),('Original prior fidelity by geometry','original-prior-fidelity.png'),('Initial regularity and frequency tails','prior-regularity-spectrum.png'),('Spatial dispersion and discrete energy','dispersion-energy.png'),('Separate C3 candidate fidelity','candidate-fidelity.png')]:
    lines.append(f'![{title}]({OUT/name})');lines.append('')
(OUT/'report.md').write_text('\n'.join(lines)+'\n');print('Report written. Original V1 BLOCKED; candidate complete:',complete,'bounded checks:',candidate_pass)
