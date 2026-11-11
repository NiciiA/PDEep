# PDE inverse reconstruction research

## Current workflow: G1_smooth_v1

`02_train.ipynb` now defaults to the separately versioned smooth geometry benchmark.
Original rough G1 remains blocked. T1 physics, exact D, A/B0/B1/C APIs, checkpoints
and historical results are preserved. No model is trained by notebook Run All.
See the [smooth scientific definition](docs/scientific_contract_G1_smooth_v1.md),
[numerical certificate](certifications/G1_smooth_v1/default.json), and generated
`reports/smooth_finalization/report.md` for evidence and limits.

- **Explore:** [01_explore.ipynb](notebooks/01_explore.ipynb) retains T1 and existing geometry diagnostics.
- **Train manually:** [02_train.ipynb](notebooks/02_train.ipynb), independently select A_GEO or C_GEO; preview first, then click the single launch button.
- **Evaluate:** [03_evaluate.ipynb](notebooks/03_evaluate.ipynb) retains selected T1 comparisons and adds a separate opt-in G1_smooth checkpoint section. D is not an exact geometry-field posterior.
- **Technical archive:** [pde_inverse_research.ipynb](notebooks/pde_inverse_research.ipynb); not required for training.

### Push and open directly in Colab

Review and push the complete project to a repository you control, including
`src/pdeep/geometric/`, `src/pdeep/wave_fidelity/`, `src/pdeep/smooth/`, **all files in certifications/G1_smooth_v1/portable/ and qualified/**, scientific
contracts, notebooks, scripts/tests and **certifications/**. Numerical reports are
local/gitignored; the small certificate and its independent-control summaries are
portable. Do not publish credentials or local experiment payloads. No push or
remote publication was performed for you.

For a new empty remote (replace OWNER/REPO; otherwise use your existing remote):

```bash
git add .gitignore pyproject.toml README.md src docs notebooks scripts tests certifications
git commit -m "Certify smooth geometry references and prepare manual training"
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
git rev-parse HEAD
```

Open directly, without uploading each time:

```
https://colab.research.google.com/github/OWNER/REPO/blob/REF/notebooks/02_train.ipynb
```

Or use Colab **File → Open notebook → GitHub**. Prefer the printed full commit SHA
for both REF and notebook `REPO_REF`; the default `main` is resolved and recorded.
Private notebook access and Git clone authentication are separate: configure an
external Git credential helper/SSH agent. Use token-free HTTPS or `git@HOST:OWNER/REPO.git`.
Never put tokens/passwords in repository URLs, cells, logs or checkpoints. Setup
suppresses raw Git/pip output and refuses credential-bearing URLs.

1. Select a **GPU** Colab runtime. Run **Setup configuration**, replacing
   `REPO_URL="SET_YOUR_GITHUB_REPOSITORY_URL"` and setting `REPO_REF` to the chosen
   commit/tag. Set `USE_GOOGLE_DRIVE=True` for persistence (saved preference isFalse).
2. Run **Automatic setup**, the collapsed next cell. It verifies/clones only an
   absent checkout and checks origin/revision/clean state. **Before project scientific
   imports**, it inspects installed AND loaded dependencies, keeps a consistent
   NumPy/SciPy/Torch stack, installs only missing numerical packages with `--no-deps`,
   then installs project code editable without upgrading notebook/kernel packages.
   It validates CUDA and mounts Drive if enabled. It never resets source files,
   reloads binary modules, restarts the kernel or changes a loaded NumPy installation.
   Python>=3.12 is required for this smooth workflow.
3. Run experiment configuration, then the separate **Compatible reference certification**
   cell, then plan review. A compatible immutable environment certificate is reused.
   A new platform/dependency profile performs an explicit bounded CPU64 qualification
   (96 fixed cases, roughly10–12 minutes locally), with stage progress and zero optimizer
   updates. Actual values must agree with frozen original references within1e-6 RMS;
   inherited spatial/time bounds plus measured discrepancy must remain below0.2% RMS.
   The exact original certified meshes and C3 bases are loaded from verified portable
   artifacts; identities are recomputed from their arrays, never copied onto new meshes.
   The engine checks certificate/source/dependencies, canonical reference mesh specification,
   prior, observations and normalization. Colab never
   silently falls back to CPU. Local CPU/MPS uses the installed project, with no
   clone/install/Google import or Drive mount.
4. **A tiny-overfit:** `MODELS=("A_GEO",)`, `EXPERIMENT="tiny_overfit"`,
   `OVERFIT_UPDATES=100` (selected automatically for tiny_overfit), `ACTION="fresh"`.
   **C tiny-overfit:** same settings with `MODELS=("C_GEO",)` and an independent run.
5. Review preview, set `RUN_MODE="train"`, rerun configuration/preview/launch cells,
   then **click**. Cell execution, saved modes and Run All cannot launch training.
   Inspect actual initial/final common field errors, wave u/v separately, real
   objective terms and gradient histories. Overfit success is pending execution;
   it requires>=75% field-error reduction or the declared numerical floor.
6. **1,000 A updates:** `MODELS=("A_GEO",)`, `EXPERIMENT="bounded"`, `UPDATES=1000`,
   `SAVE_EVERY=VALIDATE_EVERY=250`. For **1,000 C updates**, select only`("C_GEO",)`
   in a separate run.500 is also supported. Physical cases sample the full lazy12000-case/PDE train split. Each
   case uses a declared finite point-query bank. CPU64 reference warm-up across
   many new trajectories can take hours; previews never generate it. For an
   explicitly labeled finite diagnostic pool, set advanced`cases_per_geometry=2`
   (or another small count). This changes experiment scope, not the prior/reference.

### Checkpoints, resume and evaluation

Local: `SETUP["local_output_dir"]/G1_smooth_v1/runs/<architecture>/<run_id>/checkpoints/`.
Colab default root is`/content/pde-inverse-outputs`; local root is the installed
project's`artifacts/`. Every checkpoint commits and verifies locally first.
At each save (default every250 updates and best/final boundaries), the existing
persistence implementation optionally copies verified artifacts to
`DRIVE_ROOT/checkpoints/<run_id>/<artifact_id>/smooth-update-XXXXXXXX.json`.
Keep manifest and payload together. Separate local-save and Drive-sync states are
shown; failed sync retains the valid local artifact and warns prominently because
Colab storage is temporary. Real Drive copying is verified only when it actually
runs on a mounted Colab Drive, not by local simulations.

After disconnect: reopen the same commit, rerun setup/remount Drive, choose the
same architecture/capacity/PDE/device, set`ACTION="resume"`, select the verified
Drive manifest in`RESUME_CHECKPOINT`, preview, choose`RUN_MODE="resume"`, then click.
The saved optimizer moments, scheduler, RNG, data ordering and counters are restored;
the saved budget is not silently extended. Missing point banks regenerate from
saved deterministic recipes. Exact training resume requires the same device type,
implementation and Torch version. Every experiment records the resolved commit
and dependency versions. Portable CPU/MPS/CUDA **inference** is separate from exact resume.

In03's **G1_smooth_v1** section, set`SMOOTH_CHECKPOINT` and
`RUN_SMOOTH_EVALUATION=True`. Default scoring uses paired unseen cases on familiar
instances (tiny runs score their fixed train cases). Set
`SMOOTH_GEOMETRY_HELD_OUT=True` as a separate family/topology evaluation action.
Tables aggregate PDE, quantity and geometry; time curves and one representative
wave case limit plotting. Strong global second-derivative residuals are not
claimed for the C0/H1 geometry path; valid weak/boundary/initial diagnostics are
reported. Test cases remain untouched. Remeshing and new instances within a family
are distinct transfer categories; unlisted instances are not certified.

Non-optimizing verification for this phase: `python scripts/check_wave_regressions.py`
(forbids optimizer steps and excludes tests whose purpose is stepping optimizers).
The certified convergence evidence is already shipped;
`python scripts/certify_smooth.py` explicitly rebuilds the bounded CPU64 numerical
study and may take substantial time. Neither command launches neural training.

The unchanged T1 training API remains`pdeep.notebook_workflows` (e.g.
`prepare_training(ctx, models=("B0",), preset="smoke", updates=2)` followed by an
explicit`launch_training`). B1 is selected independently with`models=("B1",)`.
See [the retained T1 workflow notes](docs/adr_0012_routine_notebooks.md) and
[Colab setup details](docs/adr_0013_colab_bootstrap.md). No T1 or geometry training
was run during smooth finalization. CUDA and actual Google Drive remain unverified
unless the corresponding native verification report exists.

Analytical T1 reference physics and lazy data for Poisson, heat and wave, under the
unchanged [scientific contract](docs/scientific_contract_v1.md), plus exact T1
Gaussian Bayesian baseline D, independent representation utilities and the complete
direct Architecture A. independent A/C and both B0/B1 variants are available; the shared Stage7A engine is available; T2 remains unimplemented and no research training has been run.

## Installation and checks

Python >=3.10, with no upper bound; verified locally on Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest -q
python scripts/check_reference.py
python scripts/check_representations.py --device cpu
python scripts/check_notebook.py
python -m notebook notebooks/pde_inverse_research.ipynb
```

An existing dependency-complete environment may use
`python -m pip install --no-deps --no-build-isolation -e .` (setuptools >=68).
No external datasets or optional solver packages are required.

For fresh Colab training use `02_train.ipynb` and the automatic setup above. The
technical notebook retains its manual project setup and optional bounded checks;
its legacy optimizer preflights remain off.
The notebook has fifteen sections;01–13 execute. Section12 runs at most three shared-engine
updates per explicitly selected A/C/B0/B1 variant. Section13 is a disabled guarded research launch. Tiny overfit is disabled by default; sections 13–15 do not
launch research training. No automatic Drive access or external dataset download.

## Physics and scientific contracts

`benchmark("poisson" | "heat" | "wave")` in `pdeep.benchmarks` returns the
validated immutable `ProblemSpec`. The typed equation DAG, geometry, statuses,
priors, constraints and measurement declarations retain their foundation-stage
mathematics and serialization identities. Registry entries now identify the
implemented analytical T1 references. `require_supported(spec)` rejects changed
physics even when its graph is syntactically valid. See
[ADR 0002](docs/adr_0002_reference_data.md) for resolved data/normalizer choices.

`pdeep.reference.evaluate(name, xi, position, physical_time, quantity="u")`
directly evaluates at arbitrary coordinates and continuous physical time.
`position[...,2]` returns `[...,1]`; `xi[9]` or wave `xi[18]` broadcasts over
coordinates. For batched cases/queries use `xi[B,1,D]`, position `[B,Q,2]`,
time `[B,Q,1]`. `quantity="v"` queries wave velocity. Spatial input fields `f`,
`u0`, `v0` have no physical time. Poisson requires physical_time=None.
Spatial derivatives use `spatial_derivative=(dx,dy)` through order two per axis;
`time_order=0,1,2` gives physical-time derivatives. `basis`, `basis_gradient`,
`basis_laplacian`, `prior_scales`, `sample_prior`, and `prior_log_density` expose
basis/prior calculations. All reference tensors must be CPU float64, explicitly.
Autograd connections to coordinates, time and coefficients are preserved.
There are no stored simulation frames or temporal interpolation.

Existing observation/query contracts are unchanged: position `[B,N,2]`,
physical_time `[B,N,1]`, time_present/quantity_id/valid `[B,N]`, scalar value and
sigma `[B,N,1]`; queries omit value/sigma; predictions `[B,Q,1]`. Float physical
arrays, int64 quantity IDs and bool masks stay explicit. References require
float64 and CPU rather than silently casting. V1 coefficients are `[B,18]`,
active counts 9/9/18. `evaluate_batch(specs, coefficients, query)` applies the
declared u/v point operators and sanitizes padded/absent-time payloads. Symbolic
DAGs remain separate from any future spatial or learned attention graphs.

## Reproducible lazy data

```python
from pdeep.data import CaseDataset, calibrate, observations, training_queries

dataset = CaseDataset(root_seed=20261108)
case = dataset.case("train", "wave", 0)
normalizers = calibrate(dataset.root_seed)  # compute once, then save/load
sensors = observations(dataset, case, normalizers, set_id="default")
queries = training_queries(dataset, case)
model_inputs = sensors.inputs
clean_for_diagnostics_only = sensors.metadata.clean
```

`CaseDataset.cases(split,pde)` is a lazy iterator. Cases store coefficients only,
never fields. Each PDE has 12,000 train, 1,024 validation and 1,024 test cases,
with disjoint global ordinal ranges and separately split/PDE-keyed seeds.
Dataset/split/physical-case identities are stable and include generator/version
information. Case a and wave b use independent streams. Observation substreams
separate spatial positions, time, quantity labels, eta and noise. Sensors do not
consult hidden field features; modifying xi leaves placement/noise unchanged.
Named validation/test sets regenerate exactly. Additional held-out sensors use
another set_id and stay within the original physical case's split.

Default sensors: 32/48/64, spatial positions uniform [0.05,0.95]^2, dynamic times
10% zero/30% early/60% late. Queries default to 128, full spatial interior,
12.5% zero/12.5% early/75% uniform [0,1]. Deterministic largest-remainder rounding
with zero/early minimum coverage applies for N>=2. Wave u/v labels are balanced
to within one scalar query/sensor. The query uniform component may itself contain
early times. `evaluation_grid` uses 64x64 including the boundary, with physical
times 0,.02,.05,.1,.2,.35,.5,.75,1 (Poisson has no time axis).

Calibration is independent Monte Carlo from the training prior and the supervised
training query-risk measure: by default 512 cases x 128 points per PDE. It cannot
accept validation/test data. Quantity RMS values are finite estimates; exact
coefficient normalizers are the specified prior standard deviations. Positive
sigma=logUniform[.02,.08]*RMS and independent Gaussian noise stay in physical
units. Wave u/v use their own RMS. Clean/noise/eta are metadata, never fields in
ObservationBatch. No normalized model features are silently substituted.

`Normalizers.save/load` and `ObservationSet.save` use content-addressed immutable
JSON records. Observation recipes include case/set/calibration identities and
input/clean/noise checksums; `reload_observations` regenerates and verifies them.
The dataset recipe persists split definitions without storing fields. Changed
numerical dependencies can fail checksum verification; the diagnostic bundle
records environment versions. Exact cross-version numerical identity is not
promised. `collate_inputs` pads variable-size single-case batches with masked
payloads; `case_coefficients` creates padded coefficients/active masks.

## Diagnostics, notebook and outputs

`reference_checks.mathematical_checks` runs bounded scale-aware checks against
independent autograd derivatives and a separate NumPy scalar mode loop. Tolerance:
1e-12 + 1e-10 times the largest physical term; boundaries use absolute 1e-12.
Every check reports its actual maximum error, physical scale and tolerance.
Tests also check orthonormality, finite-difference coefficient sensitivities,
prior densities, noise statistics, RNG isolation, split disjointness and
fresh-process observation reconstruction.

The notebook displays all equations, reference/source/initial/u/v fields, all
nine evaluation times, prior coefficient histograms, split summaries, calibrated
RMS, sensor overlays and time coverage. Reports, six PNG figures, dataset recipes,
normalizers and observation identities live under `reports/reference/<run_id>/`.
`reports/latest_reference.json` points to the latest notebook bundle.

The verifier executes only the implemented prefix using the current interpreter
in a fresh kernel, with bounded per-cell timeout; it writes outputs into the same
notebook and `reports/notebook_prefix.json`. Kernel connection sockets may need
local execution permissions in a sandbox. Source identity hashes source/config/
tests/docs and notebook cell inputs, excluding execution outputs and generated
reports. Old source/config combinations get different bundle directories.
`reports/verification.md` documents the historical foundation stage; the current
stage's evidence is in `reports/reference_verification.md`.

Reference execution is CPU float64. Runtime inspection still reports CPU/MPS/CUDA
and refuses unavailable devices/dtypes. MPS float64 and silent MPS fallback are
unsupported. No accelerator reference implementation is claimed. Physical time t,
flow time tau, numerical step n, inverse iteration k and optimizer update s remain
separate; only physical time is used in this stage.

## Remaining limits

Canonical T1 unit-square, nine-mode, homogeneous Dirichlet problems only. T2 unknown
scalars, changed coefficients, irregular domains, general solvers, non-Gaussian
posterior calculations, A/B/C prediction architectures, optimization/training, learned
normalization, evaluation of reconstruction methods and resumable checkpoints
are unsupported. This stage stops before approach A.

## D: exact T1 Gaussian inverse baseline

`pdeep.bayes.select_architecture("D")` selects the implemented classical Bayesian
method; A/B/C remain unsupported. `ExactGaussianD.fit(spec, observation_batch,
row=0)` conditions one physical case using only valid entries, with the unchanged
modal prior. This operation has no learned weights or optimizer. H is constructed
analytically for Poisson u, heat u and wave u/v. All D tensors require CPU float64.

```python
from pdeep.bayes import select_architecture

baseline = select_architecture("D")
posterior = baseline.fit(spec, sensors.inputs)
mean_xi = posterior.mean                    # [D], physical units
covariance_xi = posterior.covariance         # [D,D], physical units squared
prediction = posterior.predict(queries.inputs)
mean_field = prediction.mean                # [1,Q,1], analytical posterior mean
latent_variance = prediction.latent_variance
bank = posterior.sample(8, generator=dataset.streams.torch(
    "posterior_samples", case.identity, "example"))
coherent_fields = bank.evaluate(queries.inputs)  # [8,Q,1], same xi per field
heldout_prediction = posterior.predict(heldout.inputs, sigma=heldout.inputs.sigma)
scores = posterior.log_predictive_density(heldout.inputs)
```

`spec` is `benchmark(case.key.pde)` and `sensors`, `queries`, `heldout` are records
for the same physical case; heldout uses a distinct observation set ID.
`GaussianPrediction.variance` is latent variance unless a future measurement sigma
is supplied, in which case it adds sigma². Scores include per-entry marginals and
the correlated joint log predictive density. Conditioning-set scores are replicate
predictive diagnostics, not held-out scores or marginal evidence. Pointwise
intervals are not simultaneous credible bands. Predictions never draw randomness.
Evaluate every chunk with the same CoefficientSamples bank for sample coherence.

D uses prior whitening, Cholesky and triangular solves; it never forms a numerical
matrix inverse or silently adds jitter. Empty/all-masked observations return the
prior exactly. Zero or negative valid sigma fails explicitly; exactly noiseless
conditioning has no implementation. Padding may contain NaNs and is excluded.
D supports mixed-case padded observation input by selecting each row explicitly;
unpadded posterior dimensions are 9/9/18. Generic information/covariance-form
functions support correlated SPD priors for independent verification, while the
architecture always uses the canonical T1 modal prior.

`regularized_least_squares` solves augmented prior-standardized WLS independently
using SVD. Ridge=1 equals Gaussian MAP/posterior mean and is labeled accordingly;
ridge=0 gives minimum-standardized-norm WLS. Coefficient and query-field recovery
are compared against prior-only, ridge-1/MAP and unregularized WLS. No test cases
choose ridge settings. Diagnostics retain raw/whitened singular values, rank,
nullity, column condition and posterior precision/covariance condition. Missing
column identifiability has condition_number=None and an explicit rank-deficient
status, rather than a misleading finite number.

Run `.venv/bin/python scripts/check_bayes.py` for bounded independent posterior
checks and 20,000 sample moment checks/PDE. Notebook sections 08–09 select D,
derive Gaussian conditioning, show coefficient/field/uncertainty/sample plots,
compare held-out sensor predictions and report identifiability. The prefix now
runs sections 01–09; sections 10–15 remain future work. Current D artifacts are
in `reports/bayes_D/<run_id>/`, with `reports/latest_bayes_D.json` as the latest
notebook pointer and `reports/bayes_verification.md` as the stage report.

The exact T1 posterior and untrained representation modules are implemented;
T2, full A/B/C predictors, training and resumable checkpoints remain unsupported. See
[ADR 0003](docs/adr_0003_exact_gaussian_D.md). This historical Stage 3 utility scope is superseded by complete A below.

## Independent typed representation modules

`pdeep.encoding` provides the graph packer/encoder, Fourier coordinates,
observation encoding and conditioned coordinate features. `pdeep.memory` provides
optional 80-anchor memory/retrieval and masked B-compatible coefficient tokens.
These return features, not scalar physical predictions. No A/B/C prediction
architecture, inverse-refinement step, objective, optimizer or training exists.

```python
from pdeep.encoding import EquationVocabulary, pack_graphs
from pdeep.representations import build_representations, float32_batch

vocabulary = EquationVocabulary.from_specs(specs)
modules = {owner: build_representations(owner, normalizers, vocabulary)
           for owner in ("A", "B", "C")}
a = modules["A"].to("cpu")
graph = pack_graphs(specs, vocabulary).to("cpu")
model_obs = float32_batch(packed_observations, device="cpu")
model_query = float32_batch(packed_queries, device="cpu")
equation = a.equation(graph)                         # nodes [B,V,96], context [B,96]
observed = a.observation(model_obs, specs)           # tokens [B,N,96], summary [B,96]
memory = a.memory(observed, equation.context, return_attention=True)
query_features = a.query(model_query, specs, equation.context, observed.summary)
retrieved = a.retrieval(memory, model_query, query_features, specs)  # [B,Q,96]
```

Each factory call constructs separate parameters; owners have separate seeded
initializations and no optimizer state. The factory preserves global RNG state.
B and C default to `memory=None` / `retrieval=None`; B coefficient tokens work
independently of spatial memory. `spatial_memory=False` disables memory for A as
well. Vocabulary and training-only normalization metadata are stateless inputs.
Encoder float32 device/dtype mismatches fail. `float32_batch` is an explicit
copy/cast/transfer request; it preserves the original physical reference batch.

The graph input includes ordered edges, operator/status/role/physical quantity
identities, shape/unit/derivative metadata, constraints, prior summaries, constants
and actual known coefficient values. Two bidirectional typed layers use width
96. Node hash IDs bind edges but are never used as learned label features.
Quantities use an explicit vocabulary; unknown identities fail. The graph encoder
supports rank <=2 and constants with <=4 values; further shapes require an
explicit vocabulary/feature extension. Altered valid graphs may be encoded even
when no physical solver supports them; that does not change the canonical data.

Observation values and sigma use fixed training-only quantity RMS normalization.
Features include normalized values, normalized sigma and its log; no clean truth
metadata enters. Raw coordinates and four side distances complement low/high
Fourier frequencies 1,2,4,8,16,32. An absent physical-time axis masks both raw and
Fourier time features and has a separate presence bit, so it differs from t=0.
Pointwise MLP/LayerNorm and masked means provide observation token equivariance
and summary invariance. Padded payloads can be NaN and yield zero output/gradient;
all-empty summaries and attention weights are zero and finite.

Memory has 16 coarse and 64 fine interior cell centers. Initialization combines
geometry, side/minimum boundary distances, scale, equation context and optional
known-field embeddings `[B,80,96]` plus `[B,80]` known mask. Two four-head layers
cross-attend to valid observations using position/time-presence/quantity/noise
bias. Cross-scale mixing exchanges pooled coarse/fine context without periodic
wraparound. Global context is computed once per observation set and reused for
query chunks. V1 retrieval uses all 16 coarse / 64 fine anchors, with separate
normalized Gaussian averages (lengthscales 0.30 / 0.15), plus global context.
Query chunks never pool information from other queries.

The revised readout has no hard neighbor selection or content logits in its
Gaussian weights. For fixed memory it is smooth in query coordinates, including
old nearest-neighbor transition regions. The memory-free Fourier/SiLU/FiLM
coordinate path remains unchanged and supports second spatial derivatives for
future C. Masked physical
time and padding have zero gradient. Gradients through encoders are feature-path
evidence, not PDE satisfaction or physical solver sensitivities.

For coefficient tokens, call `coefficient_inputs(specs, xi, gradient,
previous_update)` with `[B,18]` tensors, then the chosen owner's `.coefficient`.
The batch stores active masks, exact (m,n) and a/b identities and prior scales.
Features include physical inputs and xi/scale, gradient*scale, update/scale and
log(scale). Outputs are `[B,18,96]` and `[B,96]`; the first nine slots are active
for Poisson/heat, eighteen for wave. The gradient/update values in diagnostics
are explicit test fixtures; no inverse refinement algorithm is implied.

`parameter_inventory` and `computational_estimate` report representation counts,
float32 logical tensor sizes and attention MACs. These exclude real allocator,
workspace, autograd and optimizer overhead and are not complete A/B/C model
counts. `scripts/check_representations.py --device cpu|mps|cuda` runs bounded
invariance, dependency and derivative probes on the explicitly selected device.
Unavailable devices fail rather than falling back. Native available accelerators
are covered by parameterized encoder tests.

Notebook section 08 now includes a representation diagram, example embeddings,
compact statistics, anchor/attention plots and parameter/cost/gradient summaries.
They are untrained feature diagnostics, not learned reconstructions or physical
importance estimates. Current outputs are in `reports/representations/<run_id>/`,
with `reports/latest_representations.json` pointing to the notebook bundle and
`reports/representations_verification.md` recording stage evidence. See
[ADR 0004](docs/adr_0004_representation_modules.md). This historical Stage 3 utility scope is superseded by complete A below.


## Stage 3 retrieval review (before Stage 4)

The old hard top-k path can jump when a selected anchor is replaced. It has been
replaced by the explicitly versioned Gaussian retrieval in
[ADR 0005](docs/adr_0005_smooth_retrieval.md). `MemoryRetriever` accepts fixed
positive `coarse_lengthscale=0.30`, `fine_lengthscale=0.15`. It exposes
`coarse_context` and `fine_context` as the exact weighted averages. The main
feature output remains `[B,Q,96]`; diagnostic indices/weights now have 80 entries.
Each scale's weights sum to one (concatenated sums to two), invalid query rows
are zero, and the smooth output projection separately fuses query/coarse/fine/
global context. Prior top-k untrained retriever state dictionaries are incompatible
with the revised fusion; no checkpoint is silently migrated.

`pdeep.geometry` owns `GeometryProvider`, `AnchorLayout`, `BoxGeometry`, boundary
features and the unit-square physical adapter. `SpatialMemory(geometry_provider=...)`
gets layout/features from that interface; `CoordinateFeatures` also permits a
provider. Default unit-square anchor/boundary values are unchanged. Coarse/fine
readout and mixing use levels, not storage-order assumptions. Custom providers
do not imply new physical domain/solver support.

All A/B/C inventory counts refer only to representation utilities, with independent
trainable storage. A's revised inventory is 522,488 (retrieval 37,152); B/C remain
238,128 each and allocate no field memory by default. B's coefficient encoder
and C's memory-free query class, weights, default coordinate values and derivative
paths remain unchanged. Graph/observation encoder classes, physical/data contracts
and exact D are preserved.

Run `python scripts/check_retrieval.py --device cpu` or an explicitly available
`mps`/`cuda` device for bounded continuity/first-derivative/chunk/individual-query
checks. Float64 kernel checks use an independent NumPy Gaussian derivative and
autograd/finite differences; float32 tests include the fused feature utility.
The notebook adds a retrieval-review subsection in section 08 as the verified prerequisite for Stage 4. Evidence is in `reports/retrieval_review_verification.md`, with current
notebook artifacts pointed to by `reports/latest_retrieval_review.json`. No model
training is performed.


## Stage 4: complete direct A

```python
from pdeep.models import select_architecture
from pdeep.representations import float32_batch

a = select_architecture("A", normalizers=normalizers)
context = a.prepare_context((benchmark("wave"),), float32_batch(sensors.inputs))
field = a.predict_queries(context, float32_batch(queries.inputs)).predicted_value
xi = a.infer_unknowns(context)  # physical coefficients, inactive slots zero
```

A has 596,956 learned parameters and never calls a PDE solver for its field. Its
new independently initialized graph/observation modules feed 80 anchors, two
four-head sensor cross-attention blocks and geometry-supplied local/global mixing.
All-anchor Gaussian retrieval and a four-layer width-128 residual SiLU decoder
produce arbitrary-coordinate u/v values, multiplied by quantity RMS and an
explicit homogeneous Dirichlet treatment. An auxiliary standardized xi head is
supervised separately. The field need not satisfy the PDE or initial conditions.

The Stage 4 four-frequency observation/query option preserves six-frequency
Stage 3 defaults and B/C utility state. No full B/C model exists and no trainable
weights are shared. See [ADR 0006](docs/adr_0006_direct_architecture_A.md).

`python scripts/check_architecture_a.py --device cpu|mps|cuda` checks A1–A10
without optimizer updates. `--smoke` explicitly runs three AdamW updates using
the actual objective and default 16-case batch (microbatch8 × accumulation2).
`--tiny-overfit 50` explicitly enables a fixed training collection, at most 100
updates. This is not evidence of scientific accuracy or a research trainer.
The notebook's section 12 runs the authorized three-update smoke; optional tiny
overfit is off. All targets remain outside the model context, and no test cases
are used. Full research training remains disabled.

Reports include tensor/device checks, attention masks, observation sensitivities,
chunk/point independence, finite derivatives, per-module gradients, clipping,
objective components, rolling loss, update counters, timings and memory. Inference
exports include training normalizers, vocabulary and checksummed weights with
strict weights-only loading. They are not resumable optimizer checkpoints.
`reports/latest_architecture_A.json` identifies the current verification bundle.


## Stage 5: independently conditioned C with weak physical constraints

`select_architecture("C", normalizers=normalizers)` constructs C with 527,796
independent parameters, no spatial-memory anchors and no physical solution
operator. A two-layer masked observation self-attention processor supplies one
global96-vector with graph, geometry and known-physics conditioning. Four
width128 residual SiLU FiLM layers and shared-trunk u/v heads predict a continuous
neural field, with the explicit square Dirichlet boundary treatment.

`pdeep.constraints` implements16×16 Gauss-Legendre spatial integration and eight
weighted physical-time nodes, nine existing modal tests, differentiable physical
coordinate derivatives, family-specific weak residuals, inferred-source/initial
coupling and independent strong residual diagnostics. Neural derivatives stay
on the selected device. Initial-field RMS uses exact training-prior moments,
without modifying existing calibrations. A finite test space is not a PDE
certificate; neural16/32 quadrature discrepancies are measured and reported.

Run `python scripts/check_architecture_c.py --device cpu|mps|cuda` for bounded
mathematical/structural checks without optimizer updates. Explicit `--smoke`
runs three full-objective updates; `--tiny-overfit 50` is a separate opt-in capped
at100. Normal neural precision is float32; reference/quadrature diagnosis CPU64.
No partial neural-derivative fallback or AMP is enabled.

The single notebook adds C in08–12 and automatically runs only its three-update
preflight: microbatch1, three accumulated physical cases, full16×16×8 quadrature,
uniform stratified families. Optional C/A tiny overfit is off. Existing A
preflight can be explicitly enabled with RUN_A_SMOKE; otherwise its saved verified
Stage4 inference export is reloaded. If that artifact was omitted when copying
the project, recreate it explicitly using the Stage4 smoke command.

C inference exports save quantity/initial normalizers, vocabulary, configuration,
quadrature provenance and checksummed state, with strict weights-only reload.
They are not resumable optimizer checkpoints. Reports and inspected labeled
figures are linked by `reports/latest_architecture_C.json`. See
[ADR0007](docs/adr_0007_weak_neural_field_C.md) for mathematical/implementation
choices. Stop before B, flow matching, T2 or substantial training.


## Stage6: independent B0/B1 physical inverse refinement

Select `select_architecture("B0",normalizers=normalizers)` or `"B1"` explicitly.
Both initialize9/18 active coefficients, apply three learned positive-diagonal
physical posterior-gradient steps, then evaluate the final field through the
unchanged analytical reference. There is no learned field decoder in B. B0 has
399,428 parameters and no spatial anchors. B1 has682,924 parameters, including
283,496 extra spatial-path parameters, and the same coefficient-state mechanism.

B1 uses geometry-provided16/64 anchors, projected source/initial fields and
coefficient gradients, two sensor cross-attention/local-global blocks per step,
recurrent within-case state and spatial-to-coefficient exchange. Source projection
uses lambda*a and lambda*g_a: a gradient pushforward, not a functional gradient.
`BAblation` explicitly disables all spatial contribution, spatial attention or
physical projections. Fixed refinement and D are separate controls. These do not
launch trained ablation suites. Attention is not physical influence.

The physical bridge keeps H/Phi/g/xi/solutions on CPU float64 using the EXISTING
validated operators; neural computation is explicit CPU/MPS/CUDA float32. Casts
and transfers retain autograd. `BPrediction.physical_value` is CPU64;
`predicted_value` honors the query device/dtype. Bound steps use the largest
prior-whitened precision eigenvalue; every iteration's objective is recorded.

`python scripts/check_architecture_b.py --device cpu|mps|cuda` runs bounded
mathematical/gradient/mask checks, without optimizer updates. The single notebook
adds B in08–12 and explicitly selects `B_APPROACHES=("B0","B1")`, or either
singleton. Section12 runs three optimizer updates per selected variant, matched
training cases/sensors/queries, microbatch1/effective3/accumulation3. A/C previous
verified exports reload by default; their optimizer preflights are opt-in. All
tiny-overfit flags are off. Original A/C/reference/D implementations are unchanged.

B0/B1 COMPLETE checkpoints have separate IDs and paths, all model/spatial and Adam
state, constant-schedule declarations, counters, named-stream sampler root/next
step, normalizers, priors, vocabulary and source/environment. Strict checksummed
weights-only reload rejects incompatible variants. Resume supports the implemented
bounded constant-lr protocol; research scheduling/AMP/distributed resume is not
implemented. No substantial training or flow matching. See
[ADR0008](docs/adr_0008_B0_B1_inverse_refinement.md) and
`reports/latest_architecture_B.json` for the current verification bundle.


## Stage7A: shared training, fixed validation and one live monitor

```python
from pdeep.train_types import ResolvedTrainConfig
from pdeep.training import TrainingEngine
from pdeep.train_monitor import LiveMonitor

config = ResolvedTrainConfig("A", preset="smoke", run_tag="my-explicit-run",
    microbatch=2, accumulation=2, effective_batch=3, query_count=8,
    validation_cases_per_pde=1, validation_query_count=8)
engine = TrainingEngine(config, normalizers, "artifacts/training_runs",
                        monitor=LiveMonitor(notebook=True))
steps = engine.run()  # three successful updates; fixed tiny validation + local checkpoint
```

A/C/B0/B1 adapters call the existing objective functions unchanged. D is evaluation-
only; ambiguous B is rejected. Models/capacity/T1/precision/device are checked before
training, including actual C derivative and B bridge probes before optimizer creation.
No model weights are shared. Every microbatch contributes its case-count fraction
of the full update; clip once, then AdamW, successful scheduler tick and counters.
Nonfinite loss/gradients, OOM and unsupported operations stop with a reproducible
failure bundle; there is no automatic physics/device/batch fallback.

Defaults target16 cases/update: A/B0/B1 micro8×2; C micro1×16. AdamW lr3e-4,
weight decay1e-4,clip1,float32,no AMP. Smoke uses constant LR and <=3 updates.
Bounded permits <=100, explicit selected training IDs and fixed128-case/PDE
validation by default. Research is configured for10000 updates with250-step
warmup/cosine to.1 base LR and250-step validation/checkpoints; it requires BOTH
full_training_enabled=True and run(launch_full=True). Large requires an explicit
>=20000 update budget. None was launched during Stage7A.

Validation manifests record fixed validation-split case, observation and query
identities. Metrics distinguish physical errors, fixed-RMS normalized errors,
coefficients, diagnostic Gaussian NLL sums and architecture-specific objectives.
Wave u/v remain separate. C coordinate autograd stays enabled for weak/strong
validation; no parameter backward, .grad clearing or optimizer mutation occurs,
and original module modes restore even on failure. D supplies exact mean/latent
variance on the same tiny manifest through validate_D. This is not Stage7B full
64×64/all-times evaluation. Full1024 validation cases/PDE remain available by
explicit configuration.

LiveMonitor uses one replaceable notebook display. It reports raw/rolling50
model-specific losses, weighted terms, LR/gradients, counters, phases, timings,
throughput, memory and checkpoint/error states. ETA waits for20 successful
measurements; unmeasured suite ETA stays unavailable. normal/detailed/failure
verbosity changes diagnostics, never the scientific objective or RNG stream.
Legacy notebook stdout is saved under reports/notebook_legacy rather than dumped.

The single notebook explicitly selects TRAIN_APPROACHES, displays fully resolved
configs, separates tests/optional overfit/preflight/full training, and saves both
full-launch flags FALSE. Its Stage7A verification selection is A,C,B0,B1, three
updates each; change to a singleton to run only that model. Historical stage
optimizer loops are disabled. No input()/interactive prompt hangs RunAll.

Shared train-engine-v1 local checkpoints contain model/Adam/successful scheduler,
configuration/run/architecture, sampler cursor/RNG, normalizers/vocabulary and fixed
manifest identity. Save requires a successful update boundary; existing files are
never overwritten unless explicitly configured. Reload via
load_checkpoint(path,expected_architecture="A") resumes the exact configured budget
and stateless cursor. Earlier A/C inference exports and B checkpoints stay in their
own formats; no silent migration. Drive/expanded provenance are for Stage7C.
See [ADR0009](docs/adr_0009_reliable_training_engine.md).

## Stage 7B: matched scientific evaluation

Run `.venv/bin/python scripts/evaluate_matched.py` for **one fixed validation case
per PDE**, the full64×64/nine-time grid, wave u/v, available three-update smoke
checkpoints A/C/B0/B1, exact D and explicit B/fixed/prior controls. No training or
test evaluation is launched. The single notebook now implements sections01–14;
RunAll reloads Stage7A checkpoints (`RUN_SHARED_TRAINING_PREFLIGHT=False`). All
other optimizer/large-training guards remain false. Stage7C is still pending.

`EvaluationPlan` fixes the same physical cases, independent conditioning/held-out
sets, explicit grid/times/quantities, reference precision, normalization and source/
config identities across methods. Save/reload JSON plans independently of models.
`evaluate` uses one prepared context per case/method across chunks; inference sees
only measured inputs and known physics. Hidden truth is scoring-only. Test plans
require separate explicit final-evaluation authorization on creation/load/run.

Metrics include separate physical and training-RMS standardized quantity errors,
equal quantity/case/PDE reductions, independent AD strong physics and initial/
boundary checks, coefficient recovery versus truth and D mean, B trajectories,
controlled observation ablations, exact D uncertainty/coverage/log density and
coherent samples. Deterministic scientific JSON excludes variable timing/memory.
Full conditioning/refinement/posterior plus queries are timed after warmup. Reports
label training status and keep the tiny single-seed results descriptive. See
[ADR0010](docs/adr_0010_matched_evaluation.md) and `reports/latest_evaluation.json`.

## Stage 7C: complete persistence, resume and notebook integration

The single notebook now implements **all15 sections**, with explicit source/config
state guards. RunAll reloads existing smoke checkpoints and compatible persistence
verification evidence; research, overfit, legacy optimizer and persistence-preflight
training switches are saved false. D remains evaluation-only. Suite entries declare
**fresh**, **resume** or **inference** explicitly; preview resolved budgets, batch,
precision, checkpoint dependencies, destinations and Drive settings before launch.

New `pdeep-artifact-v2` JSON manifests distinguish complete resumable checkpoints,
standalone inference exports and rich diagnostic bundles. Every model component,
geometry/graph/prior/normalization/solver dependency, successful-step state and
checksum is recorded. Local payloads are fsynced, read back and verified before
manifest publication. `last`/`best_validation` pointers refer to immutable commits;
best uses the common fixed-validation metric. Drive is optional, verified and off
by default. Its failure never removes a valid local checkpoint. Retention never
silently deletes older runs.

```python
from pdeep.train_checkpoint import load_checkpoint
from pdeep.inference_artifacts import export_inference, load_inference
engine = load_checkpoint(checkpoint_path, expected_architecture="B1", restore_rng=True, probe=False)
# Same saved run/budget; full research resume still needs deliberate launch_full.
# Inference exports reload all required modules without importing a trainer.
loaded = load_inference(export_path, expected_architecture="B1", device="cpu")
```

`.venv/bin/python scripts/check_persistence.py` reuses compatible checked evidence.
`--fresh` explicitly launches the bounded actual-model **CPU8 versus4+fresh4** test
for A/C/B0/B1 (64 successful updates total), reusing step2 for each E2E demonstration.
It independently verifies inference exports and D in new Python processes. This is
not a research-training run. Legacy formats keep their explicit loaders and stated
limitations; v1 checkpoints are not silently relabeled full-RNG v2 artifacts.
See [ADR0011](docs/adr_0011_complete_persistence.md) and `reports/latest_persistence.json`.

## Historical Phase G rough-G1 implementation (its release gate remains blocked)

The additive geometry implementation and independent `A_GEO` / `C_GEO` prototypes
are in `pdeep.geometric`. T1, its original A/B0/B1/C/D identifiers, priors,
normalizers and saved checkpoints retain their existing definitions. See
[the geometry scientific contract](docs/scientific_contract_geometry_v1.md) and
[the implementation map](docs/phase_G_implementation_map.md).

**Do not launch research training yet.** Independent smooth manufactured-solution
FEM checks pass, but generated wave velocity changes by16–21% of its training RMS
between h=.04 and h=.02, exceeding the2–8% measurement-noise scale. A finer mesh
is a comparator, not exact ground truth. Phase G stopped at this scientific gate;
later prototype smoke checks are not full scientific acceptance. The local
`reports/geometry_phase/phase_G_report.md` contains numerical evidence, timings,
parameter inventories, plots, failed-gate details and the verification incident.
Reports/checkpoints are gitignored and must be regenerated or transferred separately.

Install locally using your active project environment:

```bash
python -m pip install -e ".[geometry]"
python scripts/check_geometry_foundation.py
python -m pytest tests/test_geometry_phase.py -q
python scripts/check_geometry_data.py  # writes diagnostics, exits2 while G3 is blocked
```

The binary Triangle/Shapely extra performs CPU preprocessing. Neural paths use
explicit float32 CPU/MPS/CUDA; references use CPU float64. Curved domains are
polygon approximations with separate geometric error budgets. The topology-aware
query path is continuous H1, not globally C1/C2; C-Geo uses element-wise weak
integration. Strong second-derivative claims at mesh interfaces are unsupported.

| Complete model | Medium (width192) | Large (width256) |
|---|---:|---:|
| A_GEO | 6,158,789 | 10,915,077 |
| C_GEO | 6,455,429 | 11,441,669 |

These are complete, independently instantiated models, not representation-only
counts. The old selector is deliberately unchanged; the explicit new API is
`pdeep.geometric.models.create`, with G1 normalizers. Both medium models received
bounded CPU and native MPS forward/backward checks. Each received exactly five
bounded CPU optimizer calls, including resume replay; no large model was trained.
CUDA, actual Colab and hosted Drive have not been verified locally. Full training,
reference/quadrature fidelity, convergence and geometry generalization remain unverified.

For a Colab **preview / explicit smoke check**:

1. Commit/push the source, notebooks, `pyproject.toml` and docs to your own GitHub
   repository. Keep checkpoints out of Git. Pin a commit/tag for reproducibility.
2. Open `https://colab.research.google.com/github/OWNER/REPO/blob/REF/notebooks/02_train.ipynb`
   with your actual repository/ref; select a GPU runtime.
3. In the first setup cell configure `REPO_URL`, `REPO_REF`, `ENABLE_GEOMETRY=True`,
   and optionally `USE_GOOGLE_DRIVE=True`. Run the first two setup cells. They
   verify checkout/revision, install `.[geometry]`, validate CUDA and optionally
   mount Drive. Use external credential helpers/SSH for private repos; never put
   credentials in URLs/notebooks. Local setup never clones or mounts Drive.
4. Skip the T1 training section and go directly to **Phase G**. Choose
   `GEO_ARCHITECTURE="A_GEO"` or `"C_GEO"`, `GEO_CAPACITY="medium"` or `"large"`.
   Preview prints resolved counts/configuration. Saved Run All is preview only.
5. Only an intentional `GEO_PRESET="smoke"`, `GEO_UPDATES<=3`, and
   `GEO_RUN_MODE="train"` launches bounded engineering updates. Research presets
   are blocked by the unresolved G3 gate. The initial LR3e-4 smoke was unstable;
   a two-update LR1e-6 control decreased loss but is not validated full training.
6. Checkpoints commit locally first under
   `OUTPUT/geometry/runs/ARCH/RUN/checkpoints`. With Drive enabled, periodic
   checksum-verified copies go under `DRIVE_ROOT/checkpoints/RUN/ARTIFACT`.
   A failed sync retains the local artifact and displays the failure.
7. After disconnect, rerun setup at the same commit, select a matching verified
   geometry manifest in `GEO_RESUME_CHECKPOINT`, and set `GEO_RUN_MODE="resume"`.
   Exact resume restores optimizer/scheduler/RNG/order/counters and the saved
   budget; it requires compatible architecture/capacity/data/source/device. A
   completed budget is not silently extended. No original large notebook is needed.

`01_explore.ipynb` has an opt-in geometry/reference display; `03_evaluate.ipynb`
has a separate concise G1 evaluation branch. G1 is not the nine-mode T1 problem,
so the exact T1 Bayesian D is not silently reused as a geometry baseline.

### Historical G3 follow-up: wave-reference fidelity

The separate smooth finalization and current manual workflow are described at the top of this README. The following retains the original rough-G1 diagnosis and preliminary C3 evidence.

See [the versioned G3 reference addendum](docs/scientific_contract_geometry_g3_reference_v2.md)
and the local `reports/wave_fidelity_g3/report.md` for convergence plots and
`errors-by-geometry-time.csv` for u/v errors by geometry and physical time.

The fine-mesh formatter is corrected: scientific notation in Triangle's area
switch could create a four-vertex square at h=.01. Actual maximum cell area is
now verified. Independently assembled P1–P4 spaces, exact semidiscrete eigen/
exponential controls, continuous-time matrix functions and a checked higher-order
wave integrator separate spatial dispersion, time error and initial-field transfer.

The original canonical-P1 field prior is unchanged and its research-training gate
remains closed. A **different**, explicitly versioned C3 compact-field prior is
being tested independently; it is not automatically installed as G1 data and must
not be used to claim V1 has passed. New numerical evidence is separate from the
historical Phase G report above. No neural architecture is changed or trained.

```bash
python scripts/check_wave_fidelity.py temporal
python scripts/check_wave_fidelity.py analytical
python scripts/check_wave_fidelity.py original
python scripts/check_wave_fidelity.py correlation
python scripts/check_wave_fidelity.py candidate  # expensive fine CPU controls; cached by configuration
python scripts/check_wave_regressions.py        # optimizer steps forbidden
python scripts/report_wave_fidelity.py
```

Use `--output reports/my-separate-wave-audit` to preserve a separate numerical
experiment. Source/field/query identities accompany cached candidate results.
Existing T1 checkpoint loaders are unchanged. For an unaffected old geometry
checkpoint whose source fingerprint predates the formatter fix/research gate,
`migrate_unaffected_checkpoint` in `pdeep.wave_fidelity.compatibility` creates an
explicit new, parent-linked artifact only after verifying all saved meshes/priors
are identical. It copies the full tensor payload unchanged. Affected fine meshes
and unreviewed source changes are rejected; originals are never rewritten.

Notebook **Run All only renders launch buttons**, even if train/resume settings
were saved. A fresh click is required for legacy training; G1 research is also
blocked at the engine entry point. The G3 verification executes a saved-train copy
with engines/optimizers blocked and records zero training calls. No new unintended
updates occurred during this follow-up.

## Colab setup/reference portability repair

See [the repair explanation](docs/colab_reference_repair.md) and local verification
outputs under`reports/colab_repair/`. The ordinary fresh-session path uses the
existing consistent scientific stack plus numerical qualification; **no restart**
is requested. Colab initializes NumPy itself, so pinning a different version after
startup was unsafe. A session already damaged by the OLD installer (loaded NumPy
differs from its on-disk installation) cannot be repaired safely in place: choose
**Runtime → Disconnect and delete runtime**, reconnect to GPU, and run the updated
notebook from the new commit. This limitation applies to that already contaminated
session, not to the new workflow. No automatic restart or hidden scientific worker
is used. Native CUDA/Drive still require actual Colab verification.
