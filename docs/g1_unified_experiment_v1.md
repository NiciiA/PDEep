# G1 unified multicase experiment v1

This experiment is governed by `scientific_contract_v1.md`, the geometry contract,
the G3 reference addendum, and the later `scientific_contract_G1_smooth_v1.md`.
The latter authorizes the C3 Poisson/heat extension and the smooth benchmark;
original rough G1 stays blocked. This document versions the newly requested
all-eight **training experiment**, not a replacement numerical/scientific contract.
Schema: `G1-unified-multicase-v1`.

## Scientific scope and identities

Three PDEs, known coefficients 1/.02/.49, zero tagged Dirichlet conditions, dynamic
physical time [0,1], absent Poisson time. Source/initial inputs remain unknown.
C3 centers, support radius, iid Gaussian coefficient law and original wave streams
are unchanged. The existing frozen TRAIN-only normalizers are used, including on
the four formerly held-out geometries. No new calibration uses validation/test.

The 24 tasks use the qualified fixed instances square1, rectangle1.2, lshape1,
disk1, ellipse1.8, hole1, annulus1, polygon1. Curved domains retain the established
polygon representation. The previous bounded catalog trained on four families;
this explicitly authorized experiment trains on all eight using the already
available explicit geometry axis in physical IDs. It claims **unseen fields on
familiar instances**, not instance/family transfer. Historical split files and
held-out experiments are untouched.

Per task, subsets (train/validation/test) are smoke2/2/2, pilot32/16/16,
research128/32/64, larger512/32/64. The index is `4*ordinal+family_slot`, preserving
the old four-seen cyclic catalog and using the analogous four-other catalog on
its separate geometry axis. IDs use the original generator/PDE/geometry/split/key,
not dataset size or observations. Expansion preserves every earlier physical ID.
Duplicate IDs, latent digests, field digests and physical-case artifact aliases
are rejected. Shared **unit linear response operators** across splits are not
shared physical trajectories: every physical case has its own independent
coefficient draw, metadata/QC record and reconstruction.

## References and numerical qualification

The issued identity
`524b9aa89bb0baa3606cc47657d6ee31c95ef2d8a212e9c96b6a4ba43d33c711`
is finite evidence from96 cases. It is not a universal Gaussian-prior guarantee.
Only an existing runtime/source/mesh-compatible certificate can authorize a run.
No automatic recertification or certificate overwrite is implemented here.

Default execution schema: `G1-independent-continuous-encounters-v1`.

1. Declare the full successful-update schedule (30,000 by default), physical
   dataset, task order, data seed, sensor-count choices and query count BEFORE
   preparation. Every `(cursor, slot, physical_case)` has its own independent
   original counter-keyed continuous position/time/quantity draws. All selected
   networks use this identical stream.
2. Propagate ONE physical case at a time with the ORIGINAL full CPU64 P4/.02,
   dt=.002 reference and reuse compatible mesh/operator/factor assets. Compute
   labels at that case's independently drawn encounter coordinates, bounded
   preflight designs and fixed evaluation designs. Store only clean scalar labels,
   coordinate-recipe hashes and metadata; discard its full trajectory afterward.
3. During GPU training, regenerate the EXACT original continuous coordinates by
   cursor/slot, verify their coordinate hash, read their prepared CPU64 labels,
   and generate the original independent heteroscedastic noise. This is exactly
   the same counter-keyed stochastic sampling law as drawing online. No finite
   query/sensor bank is sampled with replacement, no earlier encounter is recycled,
   and different observations do not become different physical cases. There are
   no PDE solves or large response-matrix evaluations inside optimizer updates.
4. An unprepared coordinate request or mismatched budget/query/data-seed plan fails
   explicitly. No solver or empirical-bank fallback occurs during training.
   Arbitrary-coordinate **neural prediction** remains supported; additional
   arbitrary truth queries can be explicitly evaluated offline by the original
   reference. The optional inspector uses the prepared evaluation grid/times.
5. Each new case must pass finite/solve/boundary/initial/energy checks. Independent
   algebraic checks reconstruct Gauss4's complex delta from endpoint differences
   and verify M/K/RHS residuals; heat checks CN step residuals and mass-norm decay.
   Wave checks physical initial velocity and finite-difference kinematic agreement
   with the C1 evaluator. No extra reference approximation is introduced by the
   prepared labels; they are original CPU64 evaluations (then explicit float32
   neural inputs/targets, as previously specified).
6. First TRAIN case/task (configurable upward) additionally undergoes independent
   P3/.01/dt=.001 and fixed-P4 dt-halving comparisons. Spatial/full differences
   must be <=.002 RMS, temporal <=.0002 RMS. These separately identified,
   stratified finite dataset checks do not certify every random field uniformly.
   Failed cases are quarantined without replacement. Missing cases/qualification
   block training. No old96-case certificate is overwritten or rerun.

Coordinates need not be stored: their exact generator recipes/hashes reproduce
all draws. Training-label upper bound is
`updates*4*(queries_per_case+128)*8 bytes` (4.055GB at30,000/4096), plus fixed
validation/test labels, preflight designs, manifests and small shared sampling
meshes. Fixed64-grid/nine-time evaluation labels add at most .705GB for the
research validation+test subsets before geometry masking. Thus the main label
payload target is approximately5GB, not hundreds of GB of trajectories. Actual
bytes are recorded per case. Each temporary wave trajectory is about492MB for
the measured square; peak RAM includes sparse factors, not all cases at once.
Generation remains CPU expensive and is explicitly resumable per verified case.
Changing a declared data plan is a different dataset artifact; physical case IDs
remain stable. Interrupted preparation of the SAME plan reuses completed records.

### Optional explicit linear-response control

`G1-unified-full-P4-linear-responses-v2` remains available separately for classical
Gaussian controls and additional arbitrary reference queries. It propagates all32
unit C3 responses (64 for wave), then compresses the complete response in float64.
The original full PDE evolution and physical prior are unchanged. Compression is
an optional, versioned approximate execution representation, never continuum
certification. Every original nodal state/integration interval is checked.

Exact rational conversion to nonnegative Bernstein polynomials gives the P4
Lagrange absolute-sum bound202/9. Displacement time-Bernstein controls are
`ul, ul+h*vl/3, ur-h*vr/3, ur`; velocity controls are
`vl, 3*(ur-ul)/h-vl-vr, vr`. Applying these to signed reconstruction residuals
bounds continuous spatial/time output, including velocity, rather than amplifying
independent nodal maxima unnecessarily. Per-response targets are1e-6 of the
existing quantity TRAIN RMS. Actual case bounds sum unit bounds times absolute
coefficients; full-solver comparisons remain required. This approximation never
changes the prior or removes physical modes before the full reference solve.
Poorly compressible responses fall back losslessly.

The measured wave response compressed492MB to207MB at rank416; it remains too
large for the default online training path, so it is NOT required there. Classical
Gaussian sensitivity evaluation is explicitly N/A until its verified response
assets are deliberately prepared. It never substitutes T1's modal D. The main
training path uses the original full reference with zero added compression error.

## Inputs, objectives and fair exposure

Training sensors resample continuous area-uniform positions, original physical
time mixture, balanced wave quantity, log-uniform eta [.02,.08], independent
heteroscedastic Gaussian noise, and uniformly selected counts8/16/32/64/128 on
each encounter. Observation/query/count/noise streams are separately keyed.
Validation/test observations are fixed; official32/48/64 counts remain available.
Supervised queries are fresh continuous draws from the existing query policy.
Hidden fields appear only in auxiliary targets, never in conditioning Problems.

Exactly three independent unchanged medium networks:

| Configuration | Full network parameters | Objective |
|---|---:|---|
| A_GEO/KIN_001 | 6,158,789 | field+.1aux+.001 wave-only kinematic |
| C_GEO/FIELD_ONLY | 6,455,429 | field+.1aux |
| C_GEO/ALL_001 | 6,455,429 | field+.1aux+.001weak+.001initial+.001boundary |

These counts include complete architectures and their representation utilities.
They are not representation-only counts. C variants initialize with identical
values at a matched seed but have independent parameter objects/optimizers.
A/C never share weights. Every configuration uses ONE parameter set for all tasks.
Initial loss is absent for Poisson; wave weak loss includes both first-order
equations. All physics uses existing normalized weak quadrature/predicted-input
semantics; no strong residual replaces a weak objective. Kinematic uses512
separate derivative queries/wave case and frozen physical velocity RMS. ALL_001
weights are constant from the beginning. Auxiliary supervision is reported in all
comparisons. No loss-adaptive rescaling or architecture change is introduced.

Permutation of all task combinations each full cycle gives exactly equal task
exposure over24 case encounters. Four independently selected physical cases per
successful update share no observation/context data. Microbatch1 × accumulation4
averages cases; field and auxiliary terms average active quantities within case.
The three networks use identical counter-keyed cases/observations/query streams. The fixed data seed is separate from optional neural initialization seeds.
Per-case losses are logged, with per-PDE gradient-norm probes every500 updates.

## Training and persistence

The existing `TrainingEngine.update` bytecode and `SuccessfulScheduler` implement
AdamW, accumulation, finite checks, clipping, counters and transactions. The
existing opt-in asynchronous timing wrapper removes explicit timing barriers,
not optimizer logic. A private planned-boundary exception ends a session BEFORE
another attempted update; the scheduler always holds the original full budget.
Defaults30,000 updates, session stop3,000, warmup250 then cosine to .1 initial LR,
LR3e-4, decay1e-4, clipping1, validation/checkpoint500,4096 queries/case, chunk2048.
Explicit50,000 schedules are supported from the beginning; changing a saved
30,000 schedule is incompatible with exact resume and is rejected.

V1 deliberately resolves AMP auto to **FP32 for all three comparisons**. T4 BF16
is not assumed. An unverified FP16/GradScaler path through coordinate derivatives
is not enabled. Scaler state is explicitly None. This limitation is exposed in
preview and manifests. CPU/MPS/CUDA are explicit; Colab CUDA validation forbids
silent CPU fallback. The zero-update preflight measures each full per-case
objective, derivative paths, peak allocated/reserved CUDA memory and adds a
conservative Adam allowance against11.5GB (and80% of device capacity). It never
reduces batch, queries, weak quadrature or derivative coverage silently.

Use local fast storage for all training cache reads. Drive is persistence only.
No automatic checkpoint search, legacy warm-start or recursive scan is provided.
Every model's current verified path is written to a predictable experiment
registry after each checkpoint. Fresh run/experiment IDs cannot overwrite earlier
runs. Explicit compatible paths restore model, Adam moments, scheduler, supported
RNGs, keyed sampler/counters and best-validation weights. Incremental verified history shards preserve full training/validation curves across disconnects. Same source/config,
Torch and device are required. Known completed models can be retained while
other configurations resume or start fresh. Checkpoint and dataset copy failures
retain the verified local copy and warn about ephemeral Colab storage.

A single display handle shows stages immediately; a cached-state UI heartbeat
continues during CPU preparation/validation/save. It uses no model or RNG. ETA
is N/A until measured, particularly for a future model whose cost is unknown.

## Evaluation and future extensions

Best selection uses validation only: equal PDE, equal active quantity, then equal
geometry/case. Latest and validation-best weights remain available in immutable
checkpoints. Test is explicit and never calibrates, schedules or selects weights.
Reports are per physical case and quantity, with physical RMSE uncombined across
incompatible units, standardized MSE, relative RMS denominator, amplitude, sensors,
independent held-out sensors, weak/BC/IC/auxiliary errors, nine physical times,
late-time and wave element-interior derivative diagnostics. Strong residuals
exclude mesh interfaces and are not a global C0/H1 PDE certificate.

Baselines are zero/prior mean, declared space-time IDW, and geometry-specific
Gaussian conditioning with32/64 iid C3 coefficients and actual cached-P4
sensitivities/noise. The latter uses Cholesky solves for the mean AND full
covariance; it is an exact Gaussian algebra result for a numerically bounded
finite-dimensional forward operator, not an exact continuum posterior or T1 D.
Sensor count/noise/position/value/no-observation ablations are paired by case.
Mean/median/SD/IQR/q95 and paired bootstrap intervals use physical cases as
replicates, not sensors/frames. Final comparisons require >=3 separately launched
neural seeds; the initial job is not automatically tripled.

`holdout_family` excludes a declared family from training task order; validation
selection must also exclude it. Final evaluation marks held-out-family rows
explicitly. This is a separate experiment, not a claim from the all-eight run.
New parameterized geometry instances are still fail-closed by the fixed qualified
catalog until independently qualified references/generators are added; there is
no invented unseen-instance result. Eight leave-family-out jobs are not launched.

No full training or expanded numerical certification was executed during
implementation. See `reports/g1_unified/report.md` for measured bounded evidence,
real hardware coverage and remaining Colab/research gates.
