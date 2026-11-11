# Scientific contract: geometry inverse research v1 (G1)

**Release status: G3 generated-field wave-velocity fidelity is BLOCKED.**
No research training is authorized by the engineering smoke checks.

This is an additive, experimental **G1** benchmark. `scientific_contract_v1.md`
continues to define the unchanged analytical T1 benchmark and A/B0/B1/C/D.
New architecture identifiers are **A_GEO** and **C_GEO**. A geometry checkpoint
cannot be interpreted as an old A/C artifact. No T2/T3, nonlinear PDE, flow,
unknown coefficient inverse track, or trained-quality claim is introduced.

## Physical and numerical definitions

Connected bounded planar polygonal domains may be nonconvex or contain holes.
Supported constructors: square, aspect-ratio rectangle, L, disk, ellipse, square
with a circular hole, annulus, and validated user polygons with polygon holes.
Invalid/self-intersecting/disconnected geometries are rejected, never repaired.
Curves have a separately recorded intended shape and fixed polygonal boundary;
FEM is performed on that polygon, not an exact curved finite element. The report
includes area/chord approximation errors, spatial errors and time errors separately.

A `GeometryDomain` owns placement, membership, outward boundary normals (including
hole boundaries), tags, connectivity, point location, quadrature and features.
The physical geometry ID excludes mesh resolution, seeds, node ordering and tag
names; the tagged polygon and full numerical mesh have separate IDs. Curved
geometry identity refers to its intended shape; changing boundary facet count
is a change of numerical domain and is not a certified transfer operation.
Vertex-index changes alter mesh IDs but preserve geometry IDs. Integer tags are
complete and positive; region labels are carried but the initial generator uses
one region. Reindexing must preserve physical interpolation and neural output to
floating point tolerance. Mesh quality gates check signed areas, coverage,
orientation, manifold boundary, connectedness, Euler characteristic and aspect.

Reference computation is CPU float64 P1 conforming FEM with consistent mass,
positive scalar diffusion (constant or quadrature-evaluable spatial function),
strong Dirichlet elimination, and **positive outward physical flux** on the load:

- Poisson: `-div(a grad u)=f`, stiffness `K_ij=integral a grad phi_i.grad phi_j`.
- Heat: `M u_dot + K u = F`, Crank–Nicolson, known positive diffusivity.
- Wave: `M u_dot=M v`, `M v_dot + K u=F`, implicit midpoint first-order system,
  with known speed squared in K and independent displacement/velocity fields.

For dynamic G1 data, a=.02 for heat and a=.49 for wave, known zero source;
Poisson has a=1 and an unknown source. Numerical reference solvers also support
mixed Dirichlet/Neumann, nonhomogeneous time-dependent Dirichlet data, variable
positive diffusion and compatible pure Neumann elliptic solves. Pure Neumann
Poisson requires the discrete load compatibility condition and an integral
zero-mean gauge. Incompatible loads and failed/nonfinite linear solves raise.
Robin and periodic enum values are reserved and **explicitly rejected**.
Boundary callbacks are `g_D(x,t)`, `d_t g_D(x,t)` and physical flux `g_N(x,t,n)`.
Users must supply trace-compatible initial data. Reference failures never become
neural targets or silent alternate PDEs.

Dynamic output uses cubic Hermite polynomials from stored FE states and their
semidiscrete derivatives. Wave v is **the derivative of the same polynomial**.
This is a documented numerical dense-output approximation, globally C1 in time,
not a new exact analytical formula. It is not nearest-frame sampling. T1 remains
direct analytical continuous-time evaluation without any interpolation.
Spatial reference evaluation is arbitrary-coordinate P1 interpolation.

## Unknown fields, observations and splits

`G1-canonical-P1-squared-elliptic-Gaussian-v1` defines hidden fields on a canonical
h=.08 mesh of the represented polygon. Let M_l be lumped mass, K the unit-diffusion
stiffness, A=M_l+ell² K, ell=.2. For independent z~N(0,I),

`u = amplitude A^{-1} M_l A^{-1} M_l^{1/2} z`, amplitude=1.

The covariance is amplitude² S S^T for the displayed linear map S. For u0/v0,
restrict both solves to homogeneous Dirichlet free nodes and extend by zero.
Source f uses the full natural-boundary smoothing operator and is **not forced
zero on the boundary**. Independent streams generate f/u0/v0. These are explicitly
finite-dimensional, canonical-mesh-dependent Gaussian functions, with arbitrary
coordinate P1 evaluation and hundreds/thousands of degrees of freedom. This is
not a claim of a mesh-independent white-noise prior or a fixed 9/18-slot state.
Squaring the elliptic smoother was chosen during G1 design to suppress unresolved
high-frequency wave content; the initial single-smoothing candidate is not a
released benchmark or a change to T1. Remeshing a case evaluates the **same**
canonical hidden function; it never draws another physical state.

Train/ID-validation/test have 12000/1024/1024 indices per PDE, lazily generated,
with independent physical-case hashes including split/PDE/index/geometry/prior.
Train includes square, rectangles1.2/1.6, L and disk; ID validation/test use new
fields on those geometries. Rectangle1.4 is parameter interpolation. Hole,
annulus, polygon, ellipse1.8 and rectangle2.4 form separate geometry-held-out
categories. Resolution transfer uses the same physical case at multiple h,
**within** a split; remeshing does not make a training case held out.
No held-out geometry/case contributes to normalizers or training.

Observations preserve position[B,N,2], physical_time[B,N,1], time_present[B,N],
quantity_id[B,N], value/sigma[B,N,1], valid[B,N]. Arbitrary N, including zero, is
supported; the training config allows0..256. Sampling is uniform in represented
area (or explicit coverage FPS, clustered, boundary-biased, partial controls).
Dynamic sensor time mixture is .1 exactly0, .3 early(0,.1), .6 uniform[.1,1]; wave
quantities are balanced. Independent eta~logUniform[.02,.08], independent Gaussian
measurement noise, sigma=eta*G1 training-quantity RMS. Queries use the declared
.125/.125/.75 zero/early/uniform mixture. Mixtures are IID category probabilities,
not exact rounded counts for small N. Physical time is absent for Poisson,
distinct from dynamic time zero. Position/time/noise/query/calibration streams
are independently keyed. Normalizers are a **finite Monte Carlo training-only**
equal-case, area/time uniform estimate (default10 cases/PDE,256 queries), saved
with identities. They do not change T1 normalizers. Numerical target fidelity is
bounded by polygon, spatial, and temporal errors; report these alongside model
errors. The default h=.04/dt=.002 reference is experimental; refine references
for precision claims, especially wave. No neural error is claimed below that
reference error floor. Full research training requires a resolution budget study.

## Typed neural input and geometry coordinate path

Reuse the validated symbolic PDE DAG and its ordered typed edges, physical field
identities, known/unknown/absent statuses and **actual numeric coefficient values**.
Geometry field descriptors have no modal coefficient slots. Known tagged BC type
and constant value reach geometry message passing; unknown fields never reach
context construction. Geometry node messages are degree-mean normalized; global
pooling uses FE lumped area weights. No full V² mesh attention is used.

Observation attention is permutation equivariant; context aggregation is invariant.
Only valid entries are selected before numerical operations, including Fourier
features and log sigma. Padded NaNs cannot influence predictions or gradients.
N=0 has an explicit zero observation context. Coordinates retain physical values,
normalized domain values, frequencies1,2,4,8,16,32, time and presence, and geometry
features. Missing stationary time Fourier features are zeroed.

Geodesic farthest-point placement targets64 coarse/256 fine interior anchors,
limited by available interior vertices, with effective counts and all-valid masks.
Coarse/fine sets may overlap as multiscale representations of the same location.
All graph edges belong to conforming triangles; no graph edge cuts through holes.
Graph distances are **edge-graph approximations**, not exact continuum geodesics.
At a query, nodal squared anchor graph distances are P1-interpolated, followed
by separate all-anchor Gaussian softmax with ell=.30/.15 times sqrt(area).
Geometry interpolation and anchor ownership are outside generic neural layers.
Coordinates remain connected to Torch barycentric weights after discrete point
location. Retrieval is C0/H1 across elements, smooth inside elements, and gives
valid first weak derivatives. **It is not globally C1/C2** at mesh edges, graph
cut loci or corners. Second AD derivatives are checked only inside elements;
strong-form pointwise PDE certification on interfaces is unsupported. C uses FE
weak integrals, so first weak derivatives suffice. Coordinate membership allows
only an explicit four-ulp input-precision boundary-roundoff envelope.

## Independent models and objectives

Medium width192/6 heads and large width256/8 heads keep useful depth fixed:
2 typed graph layers,2 mesh message layers,2 observation attention layers,
2 memory/conditioning blocks and4 decoder residual layers. Separate obs-to-memory,
area-weighted geometry-to-memory and cross-scale attention are active. Width
scaling yields real5M+ medium and approximately10M large models; measured counts
and per-module inventory, not a representation-only count, are authoritative.
Every instantiated A_GEO/C_GEO/ablation has independent trainable tensors.
No old A/B/C weights or geometry memory are silently shared.

A_GEO uses concatenated local coarse/fine memory, equation/global context, known
physics and a deterministic coherent latent in a residual coordinate decoder.
C_GEO uses a distinct coordinate FiLM/residual neural field with bounded
`.1*tanh` modulation from global/local conditioning. They are not the same decoder
with different losses. Both have arbitrary-coordinate auxiliary f/u0/v0 heads,
no18-slot xi head. Default wave u/v heads are independent. Explicit optional
`velocity='time_derivative'` uses du/dt; no silent coupling switch.
The deterministic latent is one vector per case reused across chunks. An external
per-case vector may replace it; no probabilistic sampler/flow is implemented.

A objective is equal-case/equal-present-quantity standardized field MSE plus .1
auxiliary hidden-field supervision. This explicit supervision uses hidden targets
only during loss computation. C adds .1 weak +.1 initial +.1 boundary losses.
FE hat tests vanish on Dirichlet boundaries and may be nonzero on Neumann ones.
Poisson integrates `a grad u.grad psi - f_hat psi - boundary g_N psi`.
Heat integrates `(u_t psi+a grad u.grad psi) - boundary g_N psi`.
Wave separately integrates `(u_t-v)psi` and `(v_t psi+c² grad u.grad psi)
minus the natural flux. Complete spatial integrals are accumulated across chunks
**before squaring**, then temporally Gauss weighted. Unknown f is predicted,
never read from the reference in weak physics. Initial consistency compares
u/v at0 with predicted u0/v0. Dirichlet and physical Neumann flux penalties use
boundary quadrature/normals, with a zero-mean pure-Neumann Poisson gauge.
Neural BC descriptors support known **constant** D/N; time-dependent reference
BC callbacks are not yet general neural known-field input APIs. Default generated
G1 data are homogeneous Dirichlet. Neural mixed-BC deployment is experimental.

Residual normalization uses fixed training RMS su/sv/sf, L=sqrt(area), T=1,
and integrated test-function magnitude/gradient, in matching physical term units.
It is not based on a model's current residual. Quadrature/test counts are explicit
configurations and require convergence checks before precision training.
A_NO_GEOMETRY (`geometry=False`), A_NO_MULTISCALE (`multiscale=False`),
C_NO_PHYSICS (`physics=False`), C_GEO_GLOBAL (`spatial=False`) are independent
configuration variants, not silently changed primary models. They are controls,
not trained experiments or claims of equal effective parameter counts.

## Runtime, persistence and status

Neural computation is explicit float32 CPU/MPS/CUDA; FEM preprocessing stays
CPU64. No automatic fallback after a device failure. No AMP is introduced.
The shared TrainingEngine.update/run, accumulation, clipping, scheduler,
reporting and live monitoring are inherited, with a G1 sampler/adapter/validation/
checkpoint specialization, not a second optimizer loop. Checkpoints use the
existing verified atomic artifact transaction and periodic verified Drive copy,
with explicit geometry schema, model capacity, mesh/prior/BC/normalizer identities,
full optimizer/scheduler/RNG/order/counters and source/dependency provenance.
Resume is strict about architecture/capacity/source/device/data compatibility.
Local and Drive sync states remain separate; remote failures retain local saves.
Actual Colab/Drive/CUDA verification cannot be claimed from local filesystem tests.

Implemented paths are experimental research infrastructure. Bounded tests are
not full convergence/training or geometry-generalization evidence. Consult
`reports/geometry_phase/phase_G_report.md` for actual gate outcomes, numerical
errors, costs, update counts, device coverage and remaining limitations. Do not
launch research training automatically or expand beyond G1 in this phase.
