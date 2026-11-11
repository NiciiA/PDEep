# G1_smooth_v1 — authorized C3 extension and sampled numerical certification

This is a separate benchmark, not a correction or replacement of rough G1. The
T1 modal contracts, analytical references, Bayesian D, architectures and historical
checkpoints are unchanged. Original G1 remains blocked for research/bounded runs.
The legacy identifier `G1-wave-C3-compact-Gaussian-candidate-v2` maps **only its
wave prior** to this benchmark. It never defined Poisson or heat.

## Physical random fields

The existing wave implementation is `pdeep.wave_fidelity.fields`. At each fixed
represented polygon, sample8192 area-uniform candidates on the canonical h=.08
mesh, retain centers farther than r from the boundary, and deterministically
select32 centers by Euclidean farthest-point sampling. Here r=.18 sqrt(area).
The centers and radius depend on physical geometry and the frozen prior version,
not the propagation mesh. Every random field is

    b_j(x) = max(0, 1 - ||x-c_j||²/r²)^4
    g(x) = 1/sqrt(32) sum_j z_j b_j(x),  z_j iid N(0,1).

It is globally C3, compactly supported away from every boundary (including holes),
and exactly trace-compatible with homogeneous Dirichlet data. Its covariance is
`(1/32) sum_j b_j(x)b_j(y)`. The radius is a support radius, not a Matérn length;
there is no stationary covariance or unique exponential correlation length.
Overlapping centers induce spatial correlation. No Fourier filtering is applied.

Wave u0 and v0 call the old sampler **unchanged**, including seeds and independent
quantity streams. Poisson f and heat u0 use the same frozen basis/covariance but
independent, explicitly versioned RNG roles under
`G1-C3-authorized-source-and-heat-extension-1`. The user explicitly authorized this
extension. In particular, the new Poisson source vanishes near the boundary;
this restriction differs from the original rough G1 natural source prior.

Original rough G1 uses canonical P1 fields
`g = A^-1 M_lump A^-1 sqrt(M_lump) z`, `A=M_lump+.2² K`, with zero trace for initial
fields and natural smoothing for Poisson sources. These P1 physical functions
have interelement gradient jumps. Refining their propagation mesh preserves the
same nonsmooth physical field and does not turn it into the C3 distribution.

## PDEs, geometry and numerical policy

Known coefficients are1 (Poisson),.02 (heat), and c²=.49 (wave); interval[0,1] for
dynamics and physically absent time for Poisson. All new research policies use
known homogeneous tagged Dirichlet boundaries. Mixed/Neumann solver controls
remain tested separately; this certificate does **not** authorize research with
them. Exact eight existing instances: square1, rectangle1.2, lshape1, disk1,
ellipse1.8, hole1, annulus1, polygon1. Curved shapes are the existing fixed-facet
polygons; error against an ideal circle/ellipse is a separate geometry error.

Production reference: consistent-mass conforming P4, corrected Triangle h=.02,
source quadrature8, CPU float64; Poisson sparse solve, heat Crank–Nicolson,
wave fourth-order two-stage Gauss–Legendre, dt=.002. Continuous output is C1
Hermite on a fixed physical integration grid; wave v is its exact time derivative.
No saved-frame finite difference or mode truncation is used. Reference spatial
interpolation uses the full element Lagrange basis. Non-grid times are verified.

The geometric conditioning/weak-test mesh is independently declared (default
h=.12); it is **not** the reference mesh. The unchanged models accept a variable
number of mesh/anchor tokens, capped by their configured64 coarse/256 fine counts.
Their trainable parameters and layer definitions do not change with this mesh.
Full320-token paths are checked separately. This distinction is recorded in
case, bank, experiment and checkpoint identities.

## Error budget and evidence

Minimum observation sigma is .02 times the frozen training-quantity RMS. The
sampled reference threshold is .002 RMS, ten times smaller. Four fixed Gaussian
cases per PDE per geometry are tested (96 physical cases). Per case,128 fixed
area points at nine times plus128 independent arbitrary-time sensor coordinates
are shared by every discretization. Spatial comparisons use P4 h=.04/.02 and
independent P3 h=.01; temporal comparisons hold P4 h=.02 fixed and use
.004/.002/.001. Initial compatibility, each representative-time RMSE, full-field
RMSE and sensor RMSE are checked separately. A mesh difference is not called
exact continuum error. Full generalized-eigen/independently coded propagation,
manufactured Poisson, source-quadrature and tagged-boundary controls supplement
the larger convergence study.

A certificate is sampled numerical evidence, **not** a uniform error bound over
an unbounded Gaussian prior. It applies only to the declared fixed polygon
instances, numerical policy, prior, noise/normalization policy and source/dependency
fingerprints. Four cases are reported honestly; there is no favorable-case
selection or threshold adjustment. Failed combinations remain blocked.

## Data isolation and finite manual experiments

Physical identities include benchmark, PDE, geometry, split and integer case key;
they exclude the propagation/conditioning mesh. Each split has its own RNG key.
The index ranges are train0..11999, validation/test/geometry-held-out0..1023;
geometry is also an explicit identity axis. The default catalog maps indices
cyclically over four seen or four held-out instances; explicit tiny pools can
select one index on each named instance. These are indexed lazy namespaces,
not a claim that every cross-product field has been materialized.

Normalization is frozen before certification from reserved TRAIN-only keys on
square/rectangle/lshape/disk. Wave solution u/v retain the published C3 TRAIN RMS
exactly. New Poisson/heat and auxiliary quantities use a declared four-case,
256-area-point, eight-time-Gauss calibration. Validation, test and held-out
geometries never set scales. Training-bank position sampling uses the frozen canonical h=.08 triangulation,
so physical observations/queries survive changes in the neural conditioning mesh.
Position, time, quantity permutation, eta and noise
have independent reproducible streams. Default sensors32/48/64; balanced wave
u/v; eta log-uniform[.02,.08]; noise independently Gaussian per valid entry.

Manual tiny-overfit: six fixed train physical cases (three PDEs, two instances),
fixed observation set0, independently sampled finite training-query bank and fixed
64x64 clipped-domain validation grid at nine dynamic times. Default100 updates;
acceptance requires at least75% common and per-quantity MSE reduction, or reaching
1e-5 per-quantity MSE. This is an optimization/memorization check, not generalization.

Manual bounded experiment: default1000 updates (500 supported), lazy sampling from all12000 train
case indices per PDE over four seen instances, fixed observations per case,2048 continuous training
query labels/case with128 selected/update, and paired unseen validation cases.
The point bank per physical case is a **finite empirical-query approximation**;
physical cases sample the full declared train split by default. An explicit
`cases_per_geometry` setting can instead select a finite diagnostic pool. The
bank approximation and pool/split scope are explicit in previews and checkpoints;
changing them is a new experiment. Reference warm-up for many new trajectories
can be CPU-expensive; no inaccurate coarse references replace it. No claim of learning or
accuracy exists before user execution.

Geometry evaluation distinguishes familiar instances with unseen physical cases,
new meshes of familiar instances, unseen instances within a family, and unseen
families/topologies. Remeshing never creates a new physical case. The current
certificate does not include unseen rectangle1.4 or arbitrary new instances;
they are rejected until certified. Held-out-family evaluation is a separate manual
action. The test split remains untouched.

## Training and persistence

The smooth engine subclasses the existing successful-update engine and inherits
its AdamW accumulation, clipping, scheduler, monitoring and run loop. It calls the
unchanged geometric Adapter: A field+.1 auxiliary; C additionally .1 weak,
.1 boundary,.1 initial. No second optimizer loop or architecture is introduced.
C's coordinate path is C0/H1; first-derivative weak diagnostics are valid. Global
strong second-derivative residuals are not claimed for this representation.

The engine validates immutable certificate hashes, current source/dependencies,
exact reference mesh IDs, normalization and requested capabilities before
construction and every update/run. There is no certification-disable flag.
Checkpoint metadata includes the certificate, split/bank recipes, resolved Git
commit and runtime versions. Exact resume restores model, optimizer moments,
scheduler, all supported RNGs, ordering, counters and best-validation weights.
Same device type and implementation/Torch version are required; silent cross-device
resume is rejected. Verified old G1/T1 artifacts are never reinterpreted.

Checkpoints commit locally using the existing verified artifact system, then sync
through the existing mounted-Drive implementation every save cadence. Failure
retains the local checkpoint and reports separate local-save/Drive-sync states.
Real CUDA/Drive certification is not claimed by local filesystem simulations.
