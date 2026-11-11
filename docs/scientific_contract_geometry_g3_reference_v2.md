# G3 wave-reference audit and versioned numerical policy

This addendum preserves `scientific_contract_v1.md`, all analytical T1 definitions,
and the physical prior in `scientific_contract_geometry_v1.md`. It does not switch
any existing experiment to a different prior or observation operator. A_GEO and
C_GEO architecture/representation/loss definitions are unchanged.

**Canonical G1 V1 research training remains blocked.** Numerical validation of a
new smooth-field candidate is not retrospective certification of V1. The engine
rejects non-smoke G1 runs, independently of notebook settings.

## Fidelity criterion and scope

For each quantity q, freeze its independently computed training RMS s_q. Minimum
sensor standard deviation is .02*s_q. The proposed reference target is spatial/
space-time RMS error <=.002*s_q, one tenth of that minimum noise amplitude. This
would contribute at most1% of minimum noise mean-square error at matched averaging
measure. A temporal estimator budget of .0002*s_q leaves most of the allowance
for spatial discretization. An auxiliary maximum sampled point error of .01*s_q
is reported separately. These are numerical validation targets, not a change to
the observation noise and not a universal error theorem.

Mesh-to-mesh differences are explicitly labeled as differences, not exact errors.
A substantially converged comparator needs independent h/p and temporal checks.
Analytical T1 checks and complete finite-dimensional semidiscrete eigensolutions
provide independent exact references. Finite deterministic examples cannot
certify all realizations of an unbounded Gaussian prior. Read the actual acceptance
status in `reports/wave_fidelity_g3/report.md`.

All studies are CPU float64 (complex128 internally for one rational solve).
Fixed physical times are0,.0073,.02,.05,.1,.237,.5,1. Spatial points and continuous
initial functions are frozen before refinement. Geometry families are square,
rectangle1.2, L, disk, ellipse1.8, square-with-hole, annulus and polygon. Numerical
curved domains retain the same64-facet polygon across h. Errors against true
circles/ellipses are not certified by this study; their geometric approximation
budget remains separate. No test cases calibrate training noise or loss scales.

## Confirmed mesher correction

Triangle's numeric area switch must use positional decimal formatting. Formerly,
`h=.01` produced `a5e-05`, which was parsed as multiple switches and could return
a square with only four vertices. The corrected code uses a non-exponent decimal
string and verifies `max(cell_area)<=h²/2*(1+1e-10)`. h=.04/.02 meshes retain their
old identities. Results from the broken fine-mesh switch are not valid refinement
evidence. Existing artifacts are not overwritten or relabeled.

## Reference system and independent controls

For known c=.7 and homogeneous Dirichlet boundary, the intended system is

`M u_dot=M v, M v_dot+K u=0`,

with **consistent** mass and `K_ij=integral c² grad(phi_i).grad(phi_j)`.
No mass lumping is introduced into wave propagation. Canonical-prior smoothing
still uses its previously declared lumped mass; that is a separate operation.
The original midpoint implementation remains available and unchanged. Its
amplification is `(1+i*nu/2)/(1-i*nu/2)`, nu=omega*dt, unit modulus, phase
`2 atan(nu/2)`; conservation of energy does not eliminate dispersion.

The additive `pdeep.wave_fidelity` tools provide:

- Independent consistent P1 through P4 triangular Lagrange assembly, arbitrary
  coordinate basis evaluation, homogeneous Dirichlet elimination and positive
  mass/stiffness checks. Unsupported forcing or boundary semantics are not silently
  handled by this restricted wave reference.
- Complete generalized eigenproblem control `K Phi=M Phi Lambda`,
  `Phi.T M Phi=I`, on small meshes. No mode truncation.
- Independent dense block-matrix exponential control on small meshes.
- Direct arbitrary-time cosine/sine matrix-function evaluation using a Chebyshev
  recurrence over a proven elementwise upper bound of the full spectrum. Polynomial
  approximation is certified on scalar functions and cross-checked against the
  full eigenbasis. It does **not** remove physical eigenmodes or interpolate
  between frames. Half-angle evaluation of Chebyshev-node lambda avoids loss of
  precision near the small eigenvalues on fine meshes.
- Fourth-order two-stage Gauss–Legendre propagation as an optional cheaper
  numerical reference. Its unit-modulus rational amplification is
  `(1-nu²/12+i*nu/2)/(1-nu²/12-i*nu/2)`. A one-complex-solve delta formula is
  algebraically equivalent to the conjugate Cayley product; independent controls
  verify it. Grid states have fourth-order accuracy. C1 cubic Hermite displacement
  provides continuous-time output with **velocity equal to its exact derivative**;
  interior velocity generally has third-order dense-output accuracy. dt halving
  must verify its error separately. It is not described as exact time evolution.

No explicit mass/system inverse is formed. Sparse factorizations/solves are used;
2x2 affine-coordinate inverses are geometric transformations. Discrete energy is
`(v.T M v+u.T K u)/2`. Element conformity, source/initial consistency, boundary
values, eigen residuals, mass orthogonality, arbitrary times, phase and energy
are tested independently.

## Why V1 is difficult

The V1 physical function is **P1 on a canonical h=.08 mesh**. Squaring a discrete
elliptic smoother changes its nodal Gaussian distribution; it does not remove
jumps in the physical function's elementwise gradient. A nonzero generic P1 field
has a distributional Laplacian supported partly on interior edges, not an L2
strong Laplacian. It is valid finite-energy wave data, but it is not a globally
smooth, band-limited initial function. Wave evolution does not diffuse these
frequencies away, and velocity amplifies displacement modes by omega.

Initial L2 interpolation accuracy alone is insufficient. The study reports
initial H1 transfer error, growth of the discrete acceleration norm, frequency
energy, and nested-mesh controls that preserve the **exact same canonical P1
function and its energy**. Increasing correlation length is an explicit
same-white-noise diagnostic, not a replacement for the specified ell=.2 prior.
Neither a low solver residual nor exact discrete energy conservation is a
physical-reference accuracy certificate. Energy regularity alone also supplies
no uniform pointwise velocity-error guarantee.

## Explicit smooth-field candidate V2 (not adopted into the training benchmark)

Identifier: `G1-wave-C3-compact-Gaussian-candidate-v2`.

For K=32 deterministic geometry-dependent centers c_j, support radius
`r=.18*sqrt(area)`, define

`b_j(x)=max(0,1-||x-c_j||²/r²)^4`,
`u0(x)=sum_j z_u,j b_j(x)/sqrt(K)`,
`v0(x)=sum_j z_v,j b_j(x)/sqrt(K)`.

All z are independent standard Gaussians. Centers come from keyed area candidates
on a fixed canonical h=.08 mesh, are frozen across propagation meshes, and each
has boundary clearance strictly greater than r. Insufficient clearance raises;
the code never silently shrinks supports. Coefficients and center coordinates
are recorded by field identity. The covariance is exactly
`sum_j b_j(x)b_j(y)/K` for each independent quantity. The functions are globally
C3 and exactly zero near all boundary components. This is a **different physical
prior**, with a scientifically motivated regularity/bandwidth restriction, not a
smoothing of existing cases. No old fields, splits, normalizers or observations
are overwritten. It is not a fixed nine/eighteen-mode neural representation.

Candidate calibration uses four independent training-calibration fields on
square/rectangle/L/disk, area-uniform coordinates and eight Gauss time nodes.
It is a declared small finite estimate, not a population RMS theorem. Validation
fields use disjoint keys and all eight families; calibration is never recomputed
to make their errors smaller. The production candidate reference is compared
using P2 h=.04/.02/.01, P4 h=.04/.02 and an independent P3 h=.01 control with dt
halving. The numerical outputs and their acceptance flags are published even
when a criterion fails. No reference setting is silently promoted to truth.

## Checkpoints and notebook execution

All original checkpoints/experiments remain immutable. The mesher fix changes
the raw geometry source fingerprint, intentionally stopping an exact old resume.
`pdeep.wave_fidelity.compatibility.migrate_unaffected_checkpoint` performs a
**separate, explicit migration** only for the reviewed area-format and launch-gate
changes. It verifies every saved mesh and physical prior remains identical,
copies the complete model/optimizer/RNG payload unchanged, records its original
parent, and writes a new immutable checksummed artifact. An affected fine-mesh
checkpoint or any other scientific source change is rejected. T1 loaders are
unchanged.

`02_train.ipynb` and the archival research-launch cell only register button
callbacks. Saved train/resume strings or booleans cannot execute research training
through Run All. A fresh button event is required; the G1 engine gate still blocks
uncertified research even from a callback. Verification forbids optimizer calls
and tests saved train/resume modes without clicking any button. CLI training APIs
remain deliberate explicit APIs; no such call is made in this investigation.

## Reproduce (no neural training)

```
python scripts/check_wave_fidelity.py temporal
python scripts/check_wave_fidelity.py analytical
python scripts/check_wave_fidelity.py original
python scripts/check_wave_fidelity.py correlation
python scripts/check_wave_fidelity.py candidate
python scripts/check_wave_regressions.py
```

Candidate fine controls are expensive; configurations are checkpointed individually
with field/mesh/query/source identities. They can be resumed without rerunning
completed references. Reports distinguish numerical experiments that were
interrupted deliberately for implementation verification from completed evidence.
Actual CUDA/Colab/Drive verification is outside this CPU reference investigation.
No A_GEO/C_GEO training, flow model or T2/T3 extension is performed.
