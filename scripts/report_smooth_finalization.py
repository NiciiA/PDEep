"""Aggregate numerical evidence and static scientific plots; no model training."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pdeep.smooth.definition import *
from pdeep.smooth.certification import load_record
from pdeep.smooth.data import Normalizers

OUT=Path('reports/smooth_finalization')

def main():
 record=load_record();norms=Normalizers.fromdict(record['normalizers']);families=[json.loads((OUT/(g+'.json')).read_text()) for g in GEOMETRIES];rows=[r for f in families for r in f['rows']]
 with (OUT/'errors.csv').open('w') as stream:
  fields=('pde','quantity','family','index','case_id','comparison','field_normalized_RMSE','sensor_normalized_RMSE','maximum_normalized_absolute_error');w=csv.DictWriter(stream,fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
 table=['| Geometry | PDE/quantity | Cases | Median field % RMS |95th field % | Max field % | Max sensor % | Max at one time % |','|---|---|---:|---:|---:|---:|---:|---:|']
 for f in families:
  for p,q in [('poisson','u'),('heat','u'),('wave','u'),('wave','v')]:
   rr=[r for r in f['rows'] if (r['pde'],r['quantity'],r['comparison'])==(p,q,'production_vs_independent')];e=np.array([r['field_normalized_RMSE'] for r in rr]);sensor=max(r['sensor_normalized_RMSE'] for r in rr);t=max(v for r in rr for v in r['time_normalized_RMSE'].values())
   table.append(f"|{f['family']}|{p} {q}|4|{100*np.median(e):.5f}|{100*np.quantile(e,.95):.5f}|{100*max(e):.5f}|{100*sensor:.5f}|{100*t:.5f}|")
 (OUT/'error-table.md').write_text('\n'.join(table)+'\n')
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for ax,(p,q) in zip(axes.ravel(),[('poisson','u'),('heat','u'),('wave','u'),('wave','v')]):
  for family in GEOMETRIES:
   values=[]
   for label in ('mesh04_to02','production_vs_independent'):
    rr=[r for r in rows if (r['family'],r['pde'],r['quantity'],r['comparison'])==(family,p,q,label)];values.append(100*max(r['field_normalized_RMSE'] for r in rr))
   ax.plot([.04,.02],values,marker='.',label=family)
  ax.axhline(.2,color='k',ls='--',label='0.2% criterion');ax.set(xscale='log',yscale='log',xlabel='Coarser h in comparison',ylabel='Max case RMS difference / training RMS (%)',title=f'{p} {q}');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle('P4 .04 vs P4 .02; P4 .02 vs independent P3 .01\nMesh differences are not exact continuum errors');fig.tight_layout();fig.savefig(OUT/'spatial-convergence.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,3.8))
 for ax,(p,q) in zip(axes,[('heat','u'),('wave','u'),('wave','v')]):
  for family in GEOMETRIES:
   vals=[100*max(r['field_normalized_RMSE'] for r in rows if (r['family'],r['pde'],r['quantity'],r['comparison'])==(family,p,q,label)) for label in ('dt004_to002','dt002_to001')]
   ax.plot([.004,.002],vals,marker='.',label=family)
  ax.axhline(.2,color='k',ls='--');ax.set(xscale='log',yscale='log',title=f'{p} {q}; fixed P4 h=.02',xlabel='Coarser dt in comparison',ylabel='Max case difference (% training RMS)');ax.grid(alpha=.2)
 axes[0].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(OUT/'temporal-convergence.png',dpi=170);plt.close(fig)
 # Four time-resolved reference/comparator error examples, fixed case0 by ID.
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for f in families:
  for ax,q in zip(axes,('u','v')):
   r=next(r for r in f['rows'] if (r['pde'],r['quantity'],r['comparison'],r['index'])==('wave',q,'production_vs_independent',0));ts=sorted(map(float,r['time_normalized_RMSE']));vals=[r['time_normalized_RMSE'][str(t)] for t in ts];ax.plot(ts,np.array(vals)*100,label=f['family'])
 for ax,q in zip(axes,('u','v')):ax.axhline(.2,c='k',ls='--');ax.set(title=f'Wave {q}: fixed physical case0',xlabel='Physical time',ylabel='RMS difference (% training RMS)');ax.grid(alpha=.2)
 axes[0].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(OUT/'wave-time-errors.png',dpi=170);plt.close(fig)
 # Prior law visual comparison; separate random distributions, not paired truth.
 from pdeep.geometric.data import sample_field,rng
 m=geometry('square',.08);xx=m.sample(2000,random_stream('prior-plot'));rough=sample_field(m,rng('prior-law-illustration'),zero_trace=True);smooth=physical_fields('square','wave','train',0)['u0']
 fig,axes=plt.subplots(1,2,figsize=(9,4))
 for ax,g,title in zip(axes,(rough,smooth),('Original G1: canonical P1 random field','G1_smooth_v1: compact C3 Gaussian field')):
  im=ax.scatter(xx[:,0],xx[:,1],c=g(xx),s=5,cmap='coolwarm');fig.colorbar(im,ax=ax);ax.set_title(title,fontsize=10);ax.set_aspect('equal')
 fig.tight_layout();fig.savefig(OUT/'prior-comparison.png',dpi=170);plt.close(fig)
 controls=record['independent_controls'];maxerr=record['numerical_error_summary']['maximum'];energy=max(d['wave']['wave_relative_energy_drift'] for f in families for tag,d in f['diagnostics'].items());residual=max(d.get('max_relative_solve_residual',d.get('max_relative_residual',0.)) for f in families for ps in f['diagnostics'].values() for d in ps.values())
 models={device:json.loads((OUT/f'models-{device}.json').read_text()) for device in ('cpu','mps') if (OUT/f'models-{device}.json').exists()}
 model_table=['| Architecture | Medium parameters | Large count only | CPU real adapter s | Native MPS adapter s |','|---|---:|---:|---:|---:|']
 for architecture in ('A_GEO','C_GEO'):
  cpu=models['cpu']['architectures'][architecture];mps=models.get('mps',{}).get('architectures',{}).get(architecture,{})
  model_table.append(f"|{architecture}|{cpu['parameters']:,}|{cpu['large_parameters']:,}|{cpu['real_3PDE_adapter_forward_backward_seconds']:.3f}|{mps.get('real_3PDE_adapter_forward_backward_seconds',float('nan')):.3f}|")
 mesh_table=['| Geometry | P4/.02 free DOFs | P3/.01 free DOFs | .01 vertices/cells | Realized .01 max area | Numerical suite wall s |','|---|---:|---:|---|---:|---:|']
 for f in families:
  a,b=f['meshes']['production'],f['meshes']['independent'];mesh_table.append(f"|{f['family']}|{a['free_DOFs']}|{b['free_DOFs']}|{b['vertices']}/{b['cells']}|{b['max_element_area']:.5g}|{f['seconds']:.1f}|")
 spatial=json.loads((OUT/'fixed-time-spatial.json').read_text())
 report=f'''# G1_smooth_v1 finalization

Certificate `{record['certification_id']}`: **{record['pass_fail_status']}**.
96 fixed physical cases (4/PDE/geometry), eight existing represented polygon
instances. Maximum sampled error including field/sensor, each representative time,
initial compatibility and dt-halving: **{100*maxerr:.5f}% training RMS**, versus
0.2% reference criterion and2% minimum sensor noise. This is sampled numerical
certification, not a uniform Gaussian-tail bound or exact continuum solution.

## Scientific scope and differences

See ../../docs/scientific_contract_G1_smooth_v1.md for the precise equations,
C3 covariance/center/radius/RNG definitions, explicit authorized Poisson/heat
extension, old-wave identity preservation, normalization and limits. Rough G1
and all original T1 scientific files/checkpoints remain separate and unchanged.
Mixed/nonzero Neumann and arbitrary new instances are not research-certified;
24 independent affine D/N/mixed controls pass on all eight polygons. Curved-domain
polygon approximation remains separate from FEM mesh convergence.

## Numerical errors

Production P4 h=.02, dt=.002; independent P3 h=.01, dt=.001. Same continuous
physical fields, coordinates and times. The following table compares production
with the independent finer comparator; it never labels a mesh difference exact
continuum error. All values are percentages of frozen TRAIN-only quantity RMS.

{chr(10).join(table)}

## Independent space/time separation

P4 h=.04 versus .02 holds dt=.002 fixed. The finest fixed-time comparison
uses P4/.02/dt.001 versus P3/.01/dt.001. For all four cases its RMS discrepancy
is conservatively bounded using the triangle inequality from two immutable
measured errors; maximum bound **{100*spatial['maximum_conservative_fixed_time_spatial_RMSE_bound']:.5f}% training RMS**.
For case0 on each geometry/PDE/quantity, retained solution arrays also measure
that fixed-time difference directly. Bounds are labeled bounds, not measurements.
The engine checks them too. Details and actual case0 time curves are in
`fixed-time-spatial.json` and the portable immutable spatial supplement.
Temporal-halving measurements keep P4 h=.02 fixed throughout.

## Independent controls

Full eigen residual {controls['wave']['full_generalized_eigen_control']['relative_eigen_residual']:.3g};
Poisson quartic max error {controls['poisson_quartic_max_error']:.3g}; grouped wave versus
independently coded Gauss4 max error {controls['wave']['independent_Gauss4_max_discrepancy']:.3g};
v=du/dt centered-difference max error {controls['wave']['v_equals_du_dt_max_error']:.3g};
query-chunk discrepancy {controls['wave']['query_chunk_max_error']:.3g}.
Maximum wave grid-energy relative drift {energy:.3g}; solve residual {residual:.3g}.
Heat mass norms never increase at monitored grid states. Exact semidiscrete heat
and wave halving controls show CN second order and Gauss fourth-order grid
propagation (Hermite dense velocity has third-order local output accuracy).
CPU64 meshes, DOFs, realized area, quality, boundary traces, residuals and detailed
time errors are in each family JSON. `errors.csv` contains every case/quantity.
{chr(10).join(mesh_table)}

Wall time covers numerical reference verification, not model inference/training.
Workers ran concurrently; these values are not additive elapsed wall time.
Source quadrature8/12 controls and outward normals including inner boundaries
pass. Initial traces vanish by construction. All sampled points remain in-domain.

## Architecture and workflow verification

Real independent medium instances; layer definitions, original losses and weights
are unchanged. CPU/MPS mixed-three-PDE adapter backward plus full320-memory path
passed, with exact zero parameter changes and zero optimizer calls.

{chr(10).join(model_table)}

Adapter timing fixture uses three PDEs, P2 h=.16 labels, default sensors,8 queries
and unchanged loss weights/weak quadrature3. It measures interface forward/backward,
not training accuracy or certified-reference generation. Large variants were
instantiated/counted only. Neural inputs are float32; references remain CPU64.
Native CUDA
and real mounted Google Drive are **untested**. Persistence filesystem simulation
is labeled separately and does not certify Drive. The default conditioning mesh
h=.12 is separate from the high-order reference and caps actual anchor counts by
available interior vertices; preview shows counts explicitly. Full320-token
checks use a finer conditioning mesh.

Tiny100-update and bounded500/1000-update outcomes: **pending user execution**.
The shared successful-update AdamW loop is reused. Tiny cases and sensor noise
stay fixed; bounded runs sample the full lazy12000-case/PDE training split by
default (an optional finite diagnostic pool is explicit). Training queries vary
within a declared finite continuous-query bank per physical case. Many new
reference trajectories can make CPU64 warm-up expensive; no coarse substitute
is used.
Validation grids are independent. A and C compare common field errors, not raw
architecture-specific objectives. Original rough G1 and uncertified combinations
remain blocked by engine gates, not a UI-only flag.

## Artifacts and safety

New numerical reports, immutable certificates, prepared source/configuration,
notebooks/tests/docs are the intended persistent changes. No original checkpoints
or historical artifacts are overwritten. Numerical solves perform no neural
optimizer updates. Two redundant numerical workers were stopped after saving their completed
square and L-shape/hole results; duplicate unfinished rectangle/polygon work
was discarded. This did
not affect saved evidence or any model. A preliminary aggregate certificate
(eafbecc...) is retained as a superseded incomplete record; engine schema checks
reject it. The default pointer selects the complete certificate printed above. Preservation and fresh-kernel safety
checks are recorded separately alongside regression evidence.

## Manual guide

Open02 from GitHub at a reviewed commit, set REPO_URL/REPO_REF, enable Drive if
wanted, run both setup cells, review certification/configuration. A tiny test:
MODELS=("A_GEO",), EXPERIMENT="tiny_overfit", UPDATES=OVERFIT_UPDATES(100).
C tiny test: select only("C_GEO",) independently. For1000 updates use EXPERIMENT=
"bounded", UPDATES=1000 and the chosen single model. SetRUN_MODE="train", rerun
preview/launch cells, then click. Run All never launches an optimizer.

Checkpoints: local OUTPUT/G1_smooth_v1/runs/ARCH/RUN/checkpoints; verified Drive
copies under DRIVE_ROOT/checkpoints/RUN/ARTIFACT. Resume with same commit and
configuration, ACTION/RUN_MODE="resume", selected verified manifest, fresh click.
03's smooth section provides seen-geometry and explicit held-out-family evaluation;
no test data are used. See README for exact Colab URL and restart/version guidance.
'''
 (OUT/'report.md').write_text(report)
 (OUT/'benchmark-manifest.json').write_text(json.dumps(manifest(record),indent=2)+'\n')
 print('Report and four figures written; maximum sampled error',maxerr)
if __name__=='__main__':main()
