"""Report/plot existing bounded evidence. No models, optimizers or new fields."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pdeep.geometric.domain import GeometryDomain
from pdeep.artifact_metadata import scientific_identity

root=Path(__file__).resolve().parents[1];out=root/'reports/geometry_phase'
f=json.loads((out/'foundation/foundation.json').read_text());r=json.loads((out/'data-and-retrieval.json').read_text());models=json.loads((out/'models-cpu.json').read_text());mps=json.loads((out/'models-mps.json').read_text())
wave=[row for row in r['reference_budget'] if row['pde']=='wave'];maxv=max(row['quantity_errors']['v'][-1]['standardized_RMS_difference_to_h02'] for row in wave)
r['dataset_release_gate']={'passed':False,'status':'BLOCKED','reason':'Wave velocity h=.04 vs h=.02 change exceeds even maximum noise RMS fraction .08; h=.02 is not asserted exact. Need independent space/time generated-field convergence before G1 release.','maximum_standardized_wave_velocity_difference':maxv}
(out/'data-and-retrieval.json').write_text(json.dumps(r,indent=2))
fig,axes=plt.subplots(2,4,figsize=(13,7),constrained_layout=True)
for ax,(family,rows) in zip(axes.flat,f['geometries'].items()):
 h=rows[0]['target_h'];mesh=GeometryDomain.load(out/'foundation'/f'{family}-{h}.npz');tri=mtri.Triangulation(*mesh.vertices.T,mesh.cells);ax.triplot(tri,lw=.35,color='#316889');ax.set_title(family);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Validated conforming polygon meshes (curves have 64 facets)');fig.savefig(out/'geometries.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
for ax,pde in zip(axes,('poisson','heat','wave')):
 for fam,rows in f['geometries'].items():ax.loglog([x['h_max'] for x in rows],[x[pde]['L2'] for x in rows],'-o',ms=3,label=fam)
 ax.set_title(pde+' smooth MMS');ax.set_xlabel('actual maximum edge');ax.set_ylabel('physical L2 error');ax.grid(alpha=.3)
axes[-1].legend(fontsize=6);fig.savefig(out/'reference-convergence.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for q,ax in zip(('u','v'),axes):
 vals=[a['quantity_errors'][q][-1]['standardized_RMS_difference_to_h02'] for a in wave];ax.bar([a['geometry'] for a in wave],vals,color='#377b9d');ax.axhspan(.02,.08,color='orange',alpha=.25,label='sensor sigma / training RMS');ax.set_title('Wave '+q+': h=.04 minus h=.02');ax.set_ylabel('RMS difference / training quantity RMS');ax.legend(fontsize=7)
fig.suptitle('Blocking generated-field fidelity audit: finer mesh is not exact truth');fig.savefig(out/'wave-fidelity-gate.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,3.5),constrained_layout=True);ax.axis('off')
for y,name,decoder in ((.75,'A_GEO','Direct residual coordinate decoder'),(.25,'C_GEO','FiLM / residual coordinate field')):
 for x,text in ((.07,'PDE DAG\n+ known physics'),(.29,'Mesh GNN\n+ observations'),(.53,'Multiscale local\n+ global context'),(.81,decoder+'\nu / v / f / u0 / v0')):
  ax.text(x,y,text,ha='center',va='center',fontsize=8,bbox={'boxstyle':'round,pad=.5','fc':'#e5f2f5','ec':'#22637a'})
 for a,b in ((.15,.21),(.38,.44),(.64,.70)):ax.annotate('',(b,y),(a,y),arrowprops={'arrowstyle':'->'})
 ax.text(.01,y+.19,name,fontsize=10,weight='bold')
ax.text(.5,0,'Independent weights. Deterministic coherent per-case latent. C adds weak/initial/boundary losses during training.',ha='center',fontsize=8)
fig.savefig(out/'model-diagram.png',dpi=150);plt.close(fig)
baseline=json.loads((out/'baseline.json').read_text());science=scientific_identity();changed=[k for k,v in science.items() if v!=baseline['scientific_files'][k]];modified=[]
for p,(size,mtime) in baseline['artifacts'].items():
 path=Path(p)
 if not path.is_absolute():path=root/path
 if not path.exists() or (path.stat().st_size,path.stat().st_mtime_ns)!=(size,mtime):modified.append(str(path))
assert not changed and not modified
preservation={'scientific_files_unchanged':len(science),'changed_scientific_files':changed,'modified_original_artifacts':modified,'original_artifact_count':len(baseline['artifacts'])};(out/'preservation.json').write_text(json.dumps(preservation,indent=2))
lines=['# Phase G bounded verification report','','**Scientific release status: STOPPED at G3 generated-field fidelity. Phase G acceptance is not complete.**','',
'The independent smooth-solution G1/G2 foundation passes. A later audit of generated G1 fields shows an unresolved wave-velocity reference error budget. The numerical reference at h=.02 is a refinement comparator, not exact ground truth. Later prototype engineering checks are reported transparently; they do not authorize research training on this dataset.','',
'## Implemented changes','',
'- Added `pdeep.geometric`: tagged validated geometry, binary Triangle/Shapely preprocessing, sparse CPU64 FEM, independent checks, canonical-mesh Gaussian hidden functions, keyed observations/queries and training-only G1 calibration. No fixed modal unknown slots.',
'- Added experimental independent A_GEO direct-memory and C_GEO FiLM-field implementations, flexible auxiliary field heads, H1 topology-aware retrieval, FE weak/BC/initial objectives and explicit geometry engine specialization of the existing update/run loop.',
'- Added opt-in branches to the three existing routine notebooks, geometry installation extra, strict geometry checkpoint metadata, full-state resume and existing verified Drive-sync reuse. Legacy public selectors, A/B/C/D definitions and mathematical contracts retain their original meaning.',
'- New scientific contract: `docs/scientific_contract_geometry_v1.md`. The mesh prior, numerical dense output, query regularity, known constant neural D/N boundary limitation and unreleased status are explicit.','',
'## Numerical reference foundation (CPU float64)','',f'- Eight geometry families, three refinements each; elapsed {f["seconds"]:.2f}s. Mesh area/edge quadrature, outward normals, tags, topology, interpolation and serialization checked.',
f'- Mixed-boundary affine maximum error {f["mixed_affine_max_error"]:.3e}; compatible pure-Neumann quadratic L2 error {f["Neumann_quadratic_L2_error"]:.3e}, with enforced mean-zero gauge; incompatible load rejected.',
'- Finest independent temporal rates: '+', '.join(f'{p} {rows[-1]["observed_time_rate"]:.6f}' for p,rows in f['temporal'].items())+'.',
f'- Wave midpoint relative energy drift max {max(x["energy_relative_drift"] for x in f["temporal"]["wave"]):.3e}; heat mass norm dissipates. Wave Hermite velocity checked against independent finite differences, using one-sided checks at C1 time knots.',
'- Existing T1 analytical comparison, finest h=.04 relative L2: '+', '.join(f'{p}/{q} {rows[-1][q]["relative_L2"]:.6f}' for p,rows in f['T1_analytic_comparison'].items() for q in ('u','v') if q in rows[-1])+'. These are numerical FEM discretization errors, not changes to T1.',
'- Annulus refinement uses h=.08/.04/.02 because boundary facets already over-refine the nominal h=.16 mesh. The error-reduction gate was retained. Curved-boundary polygon error remains distinct.','',
'## Blocking G3 audit','',
'Physical cases and canonical hidden functions are identical across h=.08/.04/.02. At t=.237, the following h=.04 versus h=.02 discrepancies occur:','',
'| Geometry | u RMS / training u RMS | v RMS / training v RMS |','|---|---:|---:|']
for a in wave:lines.append(f'| {a["geometry"]} | {a["quantity_errors"]["u"][-1]["standardized_RMS_difference_to_h02"]:.6f} | {a["quantity_errors"]["v"][-1]["standardized_RMS_difference_to_h02"]:.6f} |')
lines+=['','The v differences exceed the 0.02–0.08 measurement-noise standard-deviation range. The canonical P1 fields contain interface gradient jumps; wave velocity is sensitive to unresolved spatial frequencies. This is a hypothesis, not an established root cause. Separate spatial/time refinements of the same frozen physical fields and an energy-norm study are required. Do not silently smooth or resample these cases to pass the gate. h=.02 is not yet certified sufficient.','',
'Split hashes: '+str(r['split_counts_all_PDEs'])+' across all three PDEs; overlap0. ID, geometry interpolation, held-out geometry and resolution transfer are distinct. Calibration uses only5 training cases/PDE and64 area/time queries in this bounded report, not the production default10/256.','',
'## Coordinate retrieval regularity','',
'All coarse/fine anchors use separately normalized Gaussian weights over interpolated squared mesh-graph distances. No nearest-anchor top-k selection occurs. Graph paths respect holes.',
f'- Maximum independent AD versus central-FD first-derivative discrepancy: {r["retrieval"]["max_AD_FD_error"]:.3e} in float64 element interiors.',
'- At an internal edge, epsilon1e-3 to1e-7 reduces maximum weight change from '+f'{r["retrieval"]["edge_continuity"][0]["max_weight_jump"]:.3e} to {r["retrieval"]["edge_continuity"][-1]["max_weight_jump"]:.3e}'+'.',
'- First gradients are finite on both sides but have a persistent side difference (~9 for the deliberately weighted probe). This path is continuous H1, **not globally C1/C2**. Second derivatives are checked inside elements only. Weak FE integration is valid; a classical strong second derivative at an interface is not claimed.','',
'## Actual complete-model counts and bounded checks','',
'| Model | Medium | Large | CPU mixed-PDE forward+backward | Native MPS forward+backward |','|---|---:|---:|---:|---:|']
for name,row in models['architectures'].items():
 lines.append(f'| {name} | {row["medium_parameters"]:,} | {row["large_parameters"]:,} | {row["mixed_3PDE_adapter_forward_backward_seconds"]:.3f}s | {mps["architectures"][name]["mixed_3PDE_adapter_forward_backward_seconds"]:.3f}s |')
lines+=['','Counts cover the full useful trainable model, not representation utilities only. Medium width192/6 heads and large256/8 heads keep useful depth fixed; large counts are within15% of10M without dummy parameters. Separate instances own separate tensors. Main medium paths have nonzero gradients in every trainable module on the mixed-PDE loss. Large configurations were counted/instantiated, not trained.','',
'Float32 parameter+gradient+two-Adam-moment storage is16 bytes/parameter (~98.5MB A medium,103.3MB C medium), excluding activations, quadrature and references. Complexity: sparse mesh messages O(E d²), observation attention O(N² d), geometry-to-memory O(M V d), observation-to-memory O(M N d), memory mixing O(M² d), retrieval O(Q M), decoder O(Q L d²); no V² mesh attention. CPU geometry point-location/graph preprocessing is explicit. Full320×192 memory and arbitrary chunked Q were exercised separately from reduced-mesh weak smoke.','',
'Production medium smoke used three PDEs, eight sensors/eight queries per case, h=.16/dt=.02 and reduced weak order2/time2/four tests. This verifies computational paths, not quadrature/reference fidelity. Manufactured weak residual tests separately use higher quadrature and independently differentiated smooth exact functions.','',
'## Optimization and persistence evidence','']
for name,row in models['architectures'].items():
 b=row['bounded_updates'];d=row['short_descent_control'];lines += [f'- {name}: first fixed mixed-case losses at LR3e-4 were {b["losses"]}; the second loss increased. This is a warning about unvalidated optimization settings, not successful learning.',f'- {name}: two additional single-heat-case LR1e-6 updates gave losses {d["losses"]}. This demonstrates short descent only, not a converged overfit. Total actual new-model optimizer calls: **5**, including checkpoint replay. No further new-model updates were run.',f'- {name}: checkpoint at step1 restored model, optimizer, scheduler, RNG, sampler cursor and counters. Replayed step2 maximum parameter discrepancy {b["max_parameter_discrepancy"]}; scheduler and ordering agree. Local checkpoint: `{b["local_checkpoint"]}`.']
lines+=['','Native MPS checks used zero optimizer updates, no fallback. CUDA/actual Colab/Google Drive are **untested** on this Mac. Existing checksum-verified filesystem/disabled-Drive tests do not imply hosted Drive verification. G1 source/architecture/capacity/data/device checks reject incompatible resume. The old T1 loader is intentionally unchanged; geometry uses its explicit namespace/loader.','',
'## Verification incident and preservation','',
'A notebook execution attempt encountered a saved T1 `RUN_MODE="train"`. It unintentionally performed49 old-C updates before termination (about91s). No checkpoint or export was saved; only logs/configuration existed. Those logs were moved to `aborted-unintended-notebook-run/`, marked unusable for scientific evaluation. The kernel was stopped; no process remains. This violated the intended no-training notebook check and is not counted as scientific evidence. The saved launch default is now preview; subsequent notebook verification blocks optimizer methods in addition to inspecting the toggle.',
f'All {len(science)} canonical scientific source/contract fingerprints and all {len(baseline["artifacts"])} original artifact file sizes/mtimes remain unchanged. No original checkpoints were overwritten or removed.','',
'## Acceptance and next work','',
'G1/G2 foundations: pass. G3 identity/noise/split engineering: pass; **numerical reference fidelity: blocked**. G4–G7 prototypes have bounded engineering evidence but remain scientifically unaccepted pending G3. This is not a completed research-ready Phase G release. Full overfit/convergence, production quadrature convergence, calibrated references across the full catalog, long-run stability, trained generalization and hosted Colab persistence remain outstanding. No T2/T3/flow/nonlinear extension was started.','',
'Next experiment: freeze these physical IDs/prior fields, isolate wave time error at fixed fine h, refine h and dt jointly with independent energy-norm/velocity checks, and demonstrate a reference-error budget below the chosen sensor-noise scale. Investigate canonical-field regularity without changing cases during this comparison. Reopen later scientific gates only after that audit passes.','',
'## Practical Colab preview','',
'Push the source/pyproject/docs/notebooks to your own repository (reports/artifacts are gitignored; do not push checkpoints). Open `https://colab.research.google.com/github/OWNER/REPO/blob/REF/notebooks/02_train.ipynb`, using your actual owner/repository/commit. Select a GPU runtime. Configure `REPO_URL`, pin `REPO_REF`, set `ENABLE_GEOMETRY=True`, and optionally `USE_GOOGLE_DRIVE=True`. Run the first two setup cells, then skip the T1 training section and use the Phase G preview. Choose A_GEO or C_GEO and medium/large; medium and large both exceed5M. Local installs use `python -m pip install -e ".[geometry]"`. Do not launch research training while the G3 gate is blocked. An explicitly chosen smoke is an engineering check only. Future verified resumes select a matching geometry manifest; checkpoint state is local-first with periodic verified Drive copies, not dependent on the original large notebook.','',
'Artifacts: `foundation/foundation.json`, `data-and-retrieval.json`, `models-cpu.json`, `models-mps.json`, `normalizers.json`, `preservation.json`, pytest logs and PNGs in this directory.']
(out/'phase_G_report.md').write_text('\n'.join(lines)+'\n')
print('Report written; G3 release gate BLOCKED; preservation passed')
