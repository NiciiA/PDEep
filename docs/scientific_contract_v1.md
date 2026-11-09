# PDE inverse reconstruction: canonical scientific contract v1.0

Date: 2026-11-08. This document fixes scientific choices for a **clean repository**. It is a design specification, not an implementation. Codex must not silently change it to pass tests. Where an extension is unsupported, report it explicitly. Any scientific amendment requires a versioned change with rationale.

## 1. Purpose and architectures

Reconstruct PDE solutions and selected unknown physical inputs from sparse noisy observations. Compare independent approaches: A direct equation-conditioned neural operator; B a learned inverse proposal/refinement coupled to a specialist physical forward operator; C a direct conditional neural field with family-specific weak/variational constraints; D classical Bayesian inverse methods. One inspectable research notebook should run on CPU, Apple MPS and CUDA, with modules underneath. Never automatically launch expensive training; no existing code/repository is assumed.

Share mathematical specifications, datasets, interfaces, numerical reference definitions and evaluation. Reuse source-code patterns where useful, but do not share mutable trainable weights or optimizer states across A/B/C in the primary comparison. Within each architecture neural weights can be shared across PDE families. D has no neural weights in V1.

## 2. Mathematical problem

Problem specification P=(E,Omega,C) contains typed differential expression graph E, geometry Omega and boundary/initial/other constraints C. Known physical information is kappa (a *bundle*, not necessarily diffusivity). Unknown physical inputs xi may be scalar parameters, source fields, coefficients, initial fields or boundary functions. For well-posed forward problems u=S_P(xi;kappa). Noisy observations y_i=H_i(u,xi;kappa)+epsilon_i. H_i includes spatial position, optional physical time, quantity identity and measurement semantics. In V1 epsilon_i are independent Gaussian N(0,sigma_i^2), sigma_i>0 known. Prior p(xi|P,kappa) is explicit; posterior p(xi|O,P,kappa) proportional to likelihood times prior. If the forward problem has multiple physical branches, a branch variable or distribution over admissible solutions is needed. Posterior uncertainty is not the same as independent noise added to decoded pixels.

Five distinct axes: physical PDE time t; flow-matching time tau; inverse-refinement iteration k; numerical integrator step n; optimizer update s. Stationary equations have **no physical time**. Dummy tensor storage time must be paired with time_present=false and must never cause physical time conditioning.

Function spaces depend on PDE/solution concept. V1 represents unknown fields in a nine-mode basis. Later spatial fields can use basis coefficients, suitable mesh degrees of freedom or queryable neural fields. Positive scalar parameters require bounded transforms; SPD coefficient tensors require structure-preserving representations; interfaces/boundary data require explicit geometric associations.

## 3. V1 PDE benchmark: geometry, basis, known/unknown quantities

Omega=(0,1)^2, homogeneous Dirichlet u=0 on all sides. Dynamic equations use physical time in [0,1]. Modes indexed lexicographically j=(m,n), m,n in {1,2,3}: phi_mn(x,y)=2sin(m*pi*x)sin(n*pi*y); lambda_mn=pi^2(m^2+n^2); -Delta phi=lambda phi. Orthonormal in L2(Omega). Define tau_j=0.5*(lambda_11/lambda_j)^(3/4), independent z_j~N(0,1), a_j=tau_j*z_j.

P1 Poisson: -Delta u=f; unknown source f=sum_j lambda_j*a_j*phi_j; solution u=sum_j a_j phi_j; unknown xi=a (dimension 9). Domain/boundary/operator known. Point observations of u, no time.

P2 heat: u_t-alpha*Delta u=0, alpha=0.02 known. Unknown u0=sum_j a_j phi_j. Exact u(x,t)=sum_j a_j*exp(-alpha*lambda_j*t)*phi_j(x). Unknown xi=a (dimension 9). Point observations of u at continuous t.

P3 wave: u_tt-c^2*Delta u=0, c=0.7 known. Initial displacement u0=sum_j a_j phi_j. Independent z_v,j~N(0,1), omega_j=c*sqrt(lambda_j), b_j=0.4*omega_j*tau_j*z_v,j, initial velocity v0=sum_j b_j phi_j. Exact u=sum_j [a_j cos(omega_j*t)+(b_j/omega_j)sin(omega_j*t)]phi_j; v=u_t=sum_j[-omega_j*a_j*sin(omega_j*t)+b_j*cos(omega_j*t)]phi_j. Unknown xi=(a,b), dimension 18. Point observations of u or v.

T0: fully known forward-problem sanity, not an inverse experiment. T1: unknown field coefficients with other physics known (main V1). T2: one additional bounded scalar parameter: Poisson -beta*Delta u=f with beta~Uniform[0.5,1.5] and a **known, nonzero source** (no jointly unknown source); heat alpha~Uniform[0.01,0.05] and unknown initial a; wave c~Uniform[0.5,0.9] and unknown a,b. T2 after all T1 methods work. Avoid unrestricted joint coefficient/source/boundary inference.

## 4. Observations, noise, queries, datasets

Default sensor counts P1=32, P2=48, P3=64; sweep N={8,16,32,64,128}. Sample spatial positions iid Uniform[0.05,0.95]^2. Dynamic observation times: 10% exact t=0, 30% Uniform(0,0.1), 60% Uniform[0.1,1], with deterministic rounding guaranteeing early and t=0 coverage. Wave quantities balanced u/v. No observation placement may use hidden solution features in base V1.

Compute quantity-specific physical RMS scales s_q from *training distribution only* (separate deterministic calibration sampling). For each sensor eta_i ~ logUniform[0.02,0.08], sigma_i=eta_i*s_q, y_i=u_q(x_i,t_i)+N(0,sigma_i^2). Keep true noiseless value in dataset metadata only (not model input). All observation sets are independently seeded and persisted by identifiers.

Training supervised queries Q=128/case with spatial x uniform interior. Dynamic query times: 12.5% t=0, 12.5% Uniform(0,0.1), 75% Uniform[0,1]; this mixture defines the default training field-risk measure. Evaluation: spatial 64x64 grid, physical times {0,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1}, plus independent held-out sensors and time-resolved metrics. Reference evaluation is analytical at arbitrary t: zero time-interpolation error (up to floating-point). 64x64 is a visualization/evaluation resolution, not simulation resolution.

Case-level split per PDE: train=12000, validation=1024, test=1024. Key by (split,pde,case_id), with separate seed streams for case physics, observations, training queries, validation, tests and monitoring. All observation sets of one physical case remain in its split. Validation/test observation sets fixed. Never use test cases for model selection. Use matching cases, observations, queries, priors and noise across A/B/C/D.

Reference calculations CPU float64. Analytical PDE, initial, wave v=u_t and boundary checks target scale-aware discrepancies circa 1e-10 or better where representable. Boundary absolute target ~1e-12. D uses stable factorizations. Future non-analytical solvers require separate accuracy/tolerance declarations.

## 5. Structured contracts

ProblemSpec: versioned, validated, immutable, JSON-serializable physics structure (equation graph, geometry, known values, unknown descriptors/prior, initial/boundary constraints, measurement operators, solution concept, solver registry key). Do not serialize arbitrary executable strings.

Packed observation batch:
position [B,N,2], physical_time [B,N,1], time_present [B,N] bool, quantity_id [B,N] int, value [B,N,1], sigma [B,N,1], valid [B,N] bool. Padding masked everywhere; stationary time absent semantically. Preserve physical values and separately normalized features. Each observation is an explicit scalar measurement in V1. Query batch: position, time/time_present, quantity_id, valid; output prediction [B,Q,1].

Unknown coefficients: xi [B,18] padded with active mask [B,18]; first 9 active P1/P2, 18 P3. Maintain per-component identity, physical units, Gaussian prior mean/scale, and known/unknown status. Future typed parameter blocks are not restricted to 18.

Graph batch: node types, typed node attributes, ordered edges (source,destination,role/operand_index), masks, variable identity, constraint links. Neural graph encodings [B,V,96] plus graph summary [B,96], with independently trained copies per A/B/C.

Train normalization computed once from training-distribution calibration: s_u, s_v and coefficient scales s_xi,j. No test-derived normalization. Gaussian likelihood uses physical residuals and sigma, including log-sigma terms in probabilistic scores.

## 6. Equation graph and physical semantics

Typed DAG vocabulary initial: FIELD, PARAMETER, CONSTANT, GRAD, DIV, LAPLACIAN, TIME_D1, TIME_D2, ADD, NEG, MULTIPLY, DOT, POWER, EQUATION. Nodes store unique quantity bindings, tensor rank/dimension, derivative axes/orders, operand ordering, known/partially-known/unknown/absent status, coefficients, and references to boundary/initial constraints. Operator registry validates signatures; all equation terms must have compatible units/types. V1 is dimensionless but semantic/unit checking should be extensible.

P1 expression: NEG(LAPLACIAN(u)) + NEG(f) =0; f unknown modal source, Dirichlet constraint. P2: TIME_D1(u)+NEG(MULTIPLY(alpha,LAPLACIAN(u)))=0; u0 unknown modal initial, Dirichlet. P3 canonical internal first-order system: TIME_D1(u)-v=0 and TIME_D1(v)-c^2*LAPLACIAN(u)=0; initial u0/v0 unknown, Dirichlet. Display equivalent u_tt-c^2 Delta u=0.

Canonicalize stable operand order/hash/serialization and safe algebraic normalizations; recognize -Delta u and -div(I grad u) for the constant identity tensor under appropriate assumptions. Do not blindly expand divergence form across discontinuities. Distinguish symbolic expression graph, spatial discretization graph, and learned attention graph. A validated expression is not automatically solver-supported or trained.

## 7. Architecture contracts

A direct operator: independently trained graph encoder 2 typed message-passing layers width 96; observation encoder width 96; memory 16 coarse 4x4 and 64 fine 8x8 interior anchors (80 tokens), two cross-attention layers with four heads, local/global memory mixing, coordinate MLP 4 hidden layers width 128; wave separate u/v outputs, plus estimated xi head for selected unknown recovery. Prediction is a learned coordinate field, *never* a numerical solve. For V1 multiply outputs by boundary factor b(x,y)=x(1-x)y(1-y). Causal/time/quantity/noise masks explicit. Train deterministic with standardized field MSE + 0.1 standardized xi MSE. Observation likelihood is diagnostic initially. Global context fixed across query chunks.

B physics-constrained inverse: independently trained graph/observation initializer gives xi^0. At step k solve physical u^k=S_P(xi^k), predicted sensors yhat^k=H(xi^k), residual r^k=yhat^k-y. In linear-Gaussian T1, Phi(xi)=0.5(Hxi-y)^T R^-1(Hxi-y)+0.5 xi^T C^-1 xi; gradient g=H^T R^-1(Hxi-y)+C^-1 xi. For K=3 steps, xi^{k+1}=xi^k-eta_k D_{psi,k}g_k, positive diagonal D (learned bounded softplus) and positive step eta. Update network sees xi, prior scales, gradient, sensor residual summaries, equation context and previous step. Use 9/18 coefficient tokens; optional 16 coarse anchor features, not a huge mesh transformer. Monitor Phi after each step; positive preconditioner does not ensure descent for arbitrary eta. End-to-end supervised coefficient MSE at xi^K (optionally initializer auxiliary with separately documented weight), gradients through refinement; compare against unlearned stable optimization. At inference deterministic xi^K is *not* posterior sample. V1 forward operator analytical differentiable, no learned dynamics.

C neural field: separate independently trained equation/obs encoders; FiLM-conditioned smooth coordinate MLP, 4 hidden layers width 128; V1 boundary factor b(x,y), predicted xi head, independent wave u/v heads with weak equation u_t-v=0. Nine sine test functions. Poisson R_j=integral_Omega grad(u_hat).grad(phi_j)-f_hat*phi_j dx. Heat R_j(t)=integral_Omega [u_hat_t phi_j+alpha grad(u_hat).grad(phi_j)] dx. Wave R1_j(t)=integral_Omega (u_hat_t-v_hat)phi_j dx; R2_j(t)=integral_Omega [v_hat_t phi_j+c^2 grad(u_hat).grad(phi_j)]dx. Match u_hat(x,0) and v_hat(x,0) to inferred modal xi as appropriate. 16x16 tensor Gauss-Legendre spatial quadrature, verify 32x32; 8-point Gauss-Legendre physical time for weak residual. Normalize residuals by precomputed training-derived physical-term scales (not numerical loss magnitude). Default loss: standardized field MSE +0.1 xi MSE+0.1 normalized weak residual MSE+0.1 normalized initial residual MSE (initial term absent for P1). Boundaries hard in representation, weak PDE approximate. Compare weak weights 0,0.1,1; weak tests alone do not prove complete PDE satisfaction.

D classical baseline: T1 H maps modal unknown coefficients to selected observed u/v, exact Gaussian posterior under prior xi~N(m0,C0) and y~N(Hxi,R). Cpost=(C0^-1+H^T R^-1 H)^-1; mpost=Cpost(C0^-1 m0+H^T R^-1 y). Use Cholesky solves, posterior sampling and physical forward propagation, held-out predictive distributions. Compare against prior-only prediction and ridge/least-squares. In T2, integrate over one bounded scalar theta with numerical one-dimensional quadrature; conditional xi posterior Gaussian for fixed theta, marginal likelihood N(y;H_theta*m0,H_theta*C0*H_theta^T+R); check 129 vs 257 nodes. For P1 T2 f is known, so theta is the only unknown. D has no trainable neural encoder.

## 8. Spatial memory and attention

A memory anchor m_j updates by cross-attention over valid observation tokens: m'_j=m_j+sum_i softmax_i[(W_Q m_j)^T(W_K o_i)/sqrt(d_head)+bias(relative_position,physical_time_if_present,quantity,noise)] W_V o_i. Include residual/layer normalization, geometry/boundary distance, coarse-fine mixing, no periodic wraparound. Queries retrieve nearby fine/coarse anchors and global token; decoder receives coordinate/time/quantity and retrieved context. Complexity O(M*N*h) attention plus query retrieval. Padding masks must make extra padded observations inert. B's initial memory coefficient aligned: xi_j,g_j,Delta xi_prev,prior scale, mode identity; optional coarse field anchors. C may use simpler pooled observation conditioning to maintain independence.

## 9. Training and uncertainty

Deterministic A/C: Lfield=average over valid query scalar predictions of ((pred-target)/s_q)^2, queries drawn from the declared training risk distribution. Lxi=mean over active coefficients of ((estimated-truth)/s_xi,j)^2. Default A=Lfield+0.1Lxi; C=Lfield+0.1Lxi+0.1Lweak+0.1Linitial. B=Lxi of xi^K; physical posterior objective Phi supplies refinement gradient and is reported separately, not silently double counted. Known Gaussian observation negative loglikelihood is 0.5 sum_valid [((y-yhat)/sigma)^2+log(2pi sigma^2)]; per-case sum matters statistically. For SGD logging, per-observation normalized losses must be clearly identified as *surrogates*, not the true likelihood. Exactly noiseless observations require explicit constraints, not sigma=0 divisions. Missing/padded observations excluded. Gradient accumulation uses valid-item/quantity weighting so unequal microbatch sizes are not unintentionally overweighted. Simple field MSE is default; region-balanced and spectral MSE are explicit ablations only.

Probabilistic extension: first B conditional flow over standardized xi (dimension 9/18); optional A/C global stochastic xi-conditioned latent extension later, still direct coordinate decoders rather than solver calls. Training joint simulation pairs (xi,O) from prior and observation model. Generative tau∈[0,1], z0~N(0,I), z1=standardize(xi), z_tau=(1-tau)z0+tau*z1, target velocity v*=z1-z0, loss E||v_theta(z_tau,tau,encoder(O,P))-v*||² per active coordinate. ODE dz/dtau=v_theta from tau=0 to 1 (Euler/Heun configurable, convergence by step doubling). B samples xi and uses physical S_P on the *same sample for all queries/times/chunks*. Mask padded coefficient slots; transform bounded scalar params in unconstrained coordinates and account for Jacobians when evaluating densities. This is amortized posterior approximation, not exact sampling. Proposal correction/reweighting/MCMC must be separate named algorithms with demonstrated invariance/calibration. D exact Gaussian posterior is the V1 calibration reference. Do not apply samplewise reconstruction MSE to stochastic samples and then call them posterior samples.

## 10. Execution, validation and experiment policy

Neural default AdamW lr=3e-4, weight_decay=1e-4, grad clip 1, microbatch=8, accum=2 (effective single-device batch=16); research budget 10,000 successful optimizer updates, validation every 250 on fixed held-out examples, warmup+cosine with exact recorded schedule. Presets: smoke 3–5 updates, bounded ≤100, research 10k, large ≥20k explicitly enabled. Smoke runs same scientific model path, no fake simplified algorithm. Full training disabled by default even for Notebook Run All. Optimizer steps attempted/successful/skipped/failed counted separately.

Reference CPU float64. Neural CPU/MPS/CUDA float32 by default. MPS cannot allocate torch float64; no silent dtype/device fallbacks. For future CPU-only solvers, describe transfer, autograd/adjoint validity and timing. Provide capability gates. Check pointwise analytical identities, numerical weak residual quadrature, linear sensitivities against finite differences, padding and permutation invariance, nonfinite losses, gradients to intended parameters, query-chunk reproducibility, checkpoint reload, solver accuracy and observation ablations.

Compare all methods on same case-level split, observations, noise and queries; report physical-scale and normalized field error per PDE/quantity/time, observed and held-out sensor agreement, inferred xi, boundary/PDE residual, train time/inference time/memory, posterior coverage and predictive scoring where relevant. Distinguish posterior mean, MAP, independent posterior sample and ensemble mean. Use ≥3 seeds for real neural comparisons; test split never used to tune architecture.

## 11. Single notebook sections

One notebook `pde_inverse_research.ipynb` with independently inspectable sections: 01 environment, 02 device/precision, 03 resolved immutable configuration, 04 math/equation graph, 05 verified reference physics, 06 case generation/splits, 07 observation generation/plot, 08 architecture construction/parameter inventory, 09 forward checks, 10 objectives/gradient checks, 11 optional tiny fixed-case overfit, 12 bounded preflight, 13 explicitly enabled full training, 14 matched evaluation/comparison, 15 artifact save/reload/resume. Each section shows required state, settings, equations, inputs/outputs, diagnostics, plots, pass/fail, cost and a short copyable report. Source/version/config hashes detect stale state. Monitoring must not change training RNG or leave model in eval mode.

## 12. Storage and portability

Three artifact types: complete resumable checkpoint (model/optimizer/scheduler/scaler/RNG sampler and update state, graph vocabulary, prior/normalization, config, split IDs, dependency versions, source revision; save at optimizer-step boundary), inference export (all required encoders/decoders/normalizers/solver identifiers), and diagnostic bundle (intermediate modules, example cases, checks, logs). Verify checksums and reload in fresh process; optional verified Google Drive persistence during training; local checkpoints always remain authoritative. Do not silently overwrite older runs. Saving must be idempotent and atomic where practical.

## 13. Planned extension ladder

V2 non-Gaussian localized source/initial priors; V3 bounded unknown coefficients and piecewise diffusion/explicit interfaces; V4 irregular geometries and mixed BCs; V5 diffusion-advection-reaction composition; V6 monotone semilinear elliptic; V7 p-Laplacian weak forms; future shocks, Eikonal and Monge–Ampere requiring specialized conservation/entropy/viscosity/convexity solution mechanisms. Graph parsing and operator support are not claims of trained generalization. Hold out operator combinations and semantically equivalent graph forms to test graph-conditioning benefit; label-only baseline as ablation.

## 14. Required output and integrity of each Codex stage

Each stage must produce inspectable notebook cells, supporting modules and bounded tests; report file changes, equations implemented, shapes/dtypes/devices, numerical error, gradient evidence, runtime, known unsupported combinations and next stages. Fix failed deterministic acceptance checks without changing this contract, changing test thresholds, disabling tests or secretly making the benchmark easier. Expensive full training only by explicit user action. Use this file as canonical scientific authority in every later stage.
