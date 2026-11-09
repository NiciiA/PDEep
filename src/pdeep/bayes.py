"""D: exact finite-dimensional Gaussian conditioning, CPU float64 only.

No explicit matrix inverses, diagonal jitter, learned weights, or T2 physics.
Exactly noiseless valid measurements are explicitly unsupported. Padding never
enters a factorization; an empty likelihood returns the prior exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .batches import ObservationBatch, QueryBatch, masked_values
from .core import ValidationError, require
from .reference import basis, eigenvalues, reference_tensor, require_supported
from .spec import ProblemSpec


def single_query(query, row=0):
    require(type(row) is int and 0 <= row < query.position.shape[0], "batch row out of range")
    return QueryBatch(*(getattr(query, name)[row:row+1] for name in
                        ("position", "physical_time", "time_present", "quantity_id", "valid")))


def sensor_matrix(spec: ProblemSpec, query: QueryBatch, *, row=0):
    """Analytic H[N,D] (including zero padded rows), quantity u=0 / wave v=1."""
    require_supported(spec)
    q = single_query(query, row).validate((spec,)).masked()
    reference_tensor(q.position, "query position")
    reference_tensor(q.physical_time, "query physical_time")
    phi = basis(q.position[0])
    if spec.name == "poisson":
        matrix = phi
    elif spec.name == "heat":
        matrix = phi * (-.02*eigenvalues()*q.physical_time[0]).exp()
    else:
        omega = .7*eigenvalues().sqrt()
        phase = q.physical_time[0]*omega
        a = torch.where(q.quantity_id[0,:,None] == 1, -omega*phase.sin(), phase.cos())
        b = torch.where(q.quantity_id[0,:,None] == 1, phase.cos(), phase.sin()/omega)
        matrix = torch.cat((a*phi, b*phi), dim=-1)
    return masked_values(matrix, q.valid[0])


def cholesky(matrix, label):
    reference_tensor(matrix, label)
    try:
        return torch.linalg.cholesky(matrix)
    except torch.linalg.LinAlgError as error:
        raise ValidationError(f"{label} is not numerically positive definite; no jitter/fallback applied") from error


def validate_system(H, y, sigma, mean, covariance):
    for name, value in (("H",H),("y",y),("sigma",sigma),("prior mean",mean),("prior covariance",covariance)):
        reference_tensor(value,name)
    require(H.ndim == 2 and mean.ndim == 1 and len(mean) > 0 and H.shape[1] == len(mean), "H/prior dimensions mismatch")
    require(y.shape == sigma.shape == (H.shape[0],), "y and sigma must be vectors of valid observations")
    require(covariance.shape == (len(mean),len(mean)), "prior covariance shape mismatch")
    require(bool((sigma > 0).all()), "valid sigma must be strictly positive; exactly noiseless conditioning is unsupported")
    require(torch.allclose(covariance,covariance.T,rtol=1e-12,atol=1e-14), "prior covariance must be symmetric")
    return cholesky(covariance,"prior covariance")


@dataclass(frozen=True)
class GaussianState:
    mean: torch.Tensor  # [D], physical coefficient units
    covariance: torch.Tensor  # [D,D], physical coefficient units squared
    factor: torch.Tensor  # [D,D], factor @ factor.T == covariance; need not be triangular

    def sample(self, count: int, *, generator: torch.Generator):
        require(type(count) is int and count > 0, "positive posterior sample count required")
        require(str(generator.device) == "cpu", "posterior sampling requires explicit CPU generator")
        z = torch.randn(count,len(self.mean),dtype=torch.float64,generator=generator)
        return self.mean[None] + z @ self.factor.T


def condition_information(H, y, sigma, mean, covariance):
    """Whiten prior xi=m0+L0*z; K=I+A.T*A, A=R^-1/2 H L0.

    This is the information-form formula without forming C0^-1 or R^-1.
    Covariance is assembled from a triangular-solve factor F=L0 L_K^-T.
    """
    L0 = validate_system(H,y,sigma,mean,covariance)
    if H.shape[0] == 0:
        return GaussianState(mean.clone(),covariance.clone(),L0)
    A = (H @ L0) / sigma[:,None]
    residual = (y-H@mean)/sigma
    K = torch.eye(len(mean),dtype=torch.float64)+A.T@A
    L = cholesky(K,"prior-whitened posterior precision")
    z_mean = torch.cholesky_solve((A.T@residual)[:,None],L)[:,0]
    factor = torch.linalg.solve_triangular(L,L0.T,upper=False).T
    result = GaussianState(mean+L0@z_mean,factor@factor.T,factor)
    reference_tensor(result.mean,"posterior mean")
    reference_tensor(result.covariance,"posterior covariance")
    return result


def condition_covariance(H, y, sigma, mean, covariance):
    """Independent physical-space Gaussian conditioning for tiny cross-checks.

    Uses C0-C0 H.T (H C0 H.T+R)^-1 H C0 via solves. Subtractive
    cancellation limits this verification route near exact constraints.
    """
    L0 = validate_system(H,y,sigma,mean,covariance)
    if H.shape[0] == 0:
        return GaussianState(mean.clone(),covariance.clone(),L0)
    cross = covariance @ H.T
    L = cholesky(H@cross+torch.diag(sigma.square()),"marginal observation covariance")
    updated_mean = mean+cross@torch.cholesky_solve((y-H@mean)[:,None],L)[:,0]
    updated_cov = covariance-cross@torch.cholesky_solve(cross.T,L)
    updated_cov = .5*(updated_cov+updated_cov.T)
    return GaussianState(updated_mean,updated_cov,cholesky(updated_cov,"covariance-form posterior"))


def modal_prior(spec):
    require_supported(spec)
    means = tuple(m for block in spec.parameter_blocks for m in block.prior.mean)
    scales = tuple(s for block in spec.parameter_blocks for s in block.prior.scale)
    mean = torch.tensor(means,dtype=torch.float64)
    scale = torch.tensor(scales,dtype=torch.float64)
    return GaussianState(mean,torch.diag(scale.square()),torch.diag(scale))


def observed_system(spec, observations, *, row=0):
    q = single_query(observations,row)
    obs = ObservationBatch(q.position,q.physical_time,q.time_present,q.quantity_id,q.valid,
                           observations.value[row:row+1],observations.sigma[row:row+1])
    obs.validate((spec,))  # Explicit positive-sigma failure, including sigma==0.
    H = sensor_matrix(spec,obs)
    valid = obs.valid[0]
    y, sigma = obs.value[0,valid,0],obs.sigma[0,valid,0]
    reference_tensor(y,"valid y")
    reference_tensor(sigma,"valid sigma")
    return H[valid],y,sigma


def matrix_diagnostics(matrix):
    """Column identifiability, including unsensed columns when N<D."""
    singular = torch.linalg.svdvals(matrix)
    dimension = matrix.shape[1]
    values = torch.cat((singular,torch.zeros(max(0,dimension-len(singular)),dtype=torch.float64)))
    largest = float(values.max()) if dimension else 0.
    threshold = torch.finfo(torch.float64).eps*max(matrix.shape,default=0)*largest
    rank = int((values > threshold).sum())
    return {"rows":matrix.shape[0],"columns":dimension,"rank":rank,"nullity":dimension-rank,
            "singular_values":values.tolist(),"rank_tolerance":threshold,
            "column_condition_number":float(values[0]/values[-1]) if rank == dimension else None,
            "condition_status":"full_column_rank" if rank == dimension else "rank_deficient"}


def regularized_least_squares(H,y,sigma,prior: GaussianState, *, ridge=1.):
    """WLS + ridge*||z||², xi=m0+L0 z, using independent augmented SVD solve.

    ridge=1 is exactly the Gaussian MAP/posterior mean, not a distinct estimator.
    ridge=0 gives minimum-prior-standardized-norm WLS in underdetermined cases.
    """
    require(type(ridge) in (int,float) and math.isfinite(ridge) and ridge >= 0, "ridge must be finite and nonnegative")
    validate_system(H,y,sigma,prior.mean,prior.covariance)
    if H.shape[0] == 0:
        return prior.mean.clone()
    A = H@prior.factor/sigma[:,None]
    residual = (y-H@prior.mean)/sigma
    if ridge:
        A = torch.cat((A,math.sqrt(ridge)*torch.eye(len(prior.mean),dtype=torch.float64)))
        residual = torch.cat((residual,torch.zeros(len(prior.mean),dtype=torch.float64)))
    z = torch.linalg.lstsq(A,residual,driver="gelsd").solution
    return prior.mean+prior.factor@z


def coefficient_metrics(estimate, truth, scales):
    for name,tensor in (("estimate",estimate),("truth",truth),("coefficient scales",scales)):
        reference_tensor(tensor,name)
    require(estimate.shape == truth.shape == scales.shape and truth.ndim == 1 and bool((scales>0).all()), "coefficient metric shape/scales mismatch")
    residual = estimate-truth
    norm = float(torch.linalg.vector_norm(truth))
    return {"physical_rmse":float(residual.square().mean().sqrt()),
            "standardized_rmse":float((residual/scales).square().mean().sqrt()),
            "relative_l2":float(torch.linalg.vector_norm(residual))/norm if norm else None}


@dataclass(frozen=True)
class GaussianPrediction:
    mean: torch.Tensor  # [1,Q,1], posterior mean, never a random draw
    latent_variance: torch.Tensor  # [1,Q,1], diag(H Cpost H.T)
    variance: torch.Tensor  # adds supplied future measurement sigma², if any
    valid: torch.Tensor
    semantics: str


@dataclass(frozen=True)
class CoefficientSamples:
    spec: ProblemSpec
    xi: torch.Tensor  # [S,D], one shared coefficient vector per entire field sample

    def evaluate(self, query):
        require(query.position.shape[0] == 1,"sample evaluation expects one physical case")
        H = sensor_matrix(self.spec,query)
        return (self.xi @ H.T)[:,:,None]  # [S,Q,1]; no RNG during decoding/chunking


@dataclass(frozen=True)
class GaussianPosterior:
    spec: ProblemSpec
    state: GaussianState
    prior: GaussianState
    diagnostics: dict

    @property
    def mean(self):
        return self.state.mean

    @property
    def covariance(self):
        return self.state.covariance

    def sample(self,count,*,generator):
        return CoefficientSamples(self.spec,self.state.sample(count,generator=generator))

    def predict(self,query,*,sigma=None):
        require(query.position.shape[0] == 1,"prediction expects one physical case")
        H = sensor_matrix(self.spec,query)
        mean = (H@self.mean)[None,:,None]
        variance = (H@self.state.factor).square().sum(-1)[None,:,None]
        total = variance
        if sigma is not None:
            require(sigma.shape == mean.shape,"predictive sigma must have shape [1,Q,1]")
            reference_tensor(sigma[query.valid],"valid predictive sigma")
            require(bool((sigma[query.valid]>0).all()),"predictive sigma must be strictly positive; noiseless measurements unsupported")
            total = variance+masked_values(sigma,query.valid).square()
        return GaussianPrediction(masked_values(mean,query.valid),masked_values(variance,query.valid),
                                  masked_values(total,query.valid),query.valid,
                                  "future_noisy_measurement" if sigma is not None else "latent_field")

    def log_predictive_density(self,observations):
        """Marginal scores per entry and correlated joint score for ONE case.

        Held-out sets give out-of-sample scores. At conditioning locations this
        is a replicate predictive diagnostic, not marginal evidence p(y).
        """
        require(observations.position.shape[0] == 1,"scoring expects one physical case")
        H,y,sigma = observed_system(self.spec,observations)
        prediction = self.predict(observations,sigma=observations.sigma)
        valid = observations.valid
        marginal = torch.zeros_like(prediction.mean)
        variance = prediction.variance[valid]
        marginal[valid] = -.5*((observations.value[valid]-prediction.mean[valid]).square()/variance
                                + variance.log()+math.log(2*math.pi))
        if not len(y):
            joint = 0.
        else:
            projected = H@self.state.factor
            L = cholesky(projected@projected.T+torch.diag(sigma.square()),"joint predictive covariance")
            residual = (y-H@self.mean)[:,None]
            whitened = torch.linalg.solve_triangular(L,residual,upper=False)
            joint = float(-.5*(whitened.square().sum()+2*L.diag().log().sum()+len(y)*math.log(2*math.pi)))
        return {"marginal":marginal,"joint_log_density":joint,
                "sum_marginal_log_density":float(marginal.sum()),"valid_count":int(valid.sum())}


class ExactGaussianD:
    architecture = "D"
    trainable_parameters = 0

    def fit(self,spec,observations,*,row=0):
        H,y,sigma = observed_system(spec,observations,row=row)
        prior = modal_prior(spec)
        state = condition_information(H,y,sigma,prior.mean,prior.covariance)
        A = H@prior.factor/sigma[:,None]
        K = torch.eye(len(prior.mean),dtype=torch.float64)+A.T@A
        eig = torch.linalg.eigvalsh(K)
        post_eig = torch.linalg.eigvalsh(state.covariance)
        diagnostics = {"valid_observations":len(y),"dimension":len(prior.mean),
                       "H":matrix_diagnostics(H),"noise_and_prior_whitened_H":matrix_diagnostics(A),
                       "standardized_precision_condition":float(eig[-1]/eig[0]),
                       "physical_covariance_condition":float(post_eig[-1]/post_eig[0]),
                       "min_covariance_eigenvalue":float(post_eig[0]),
                       "symmetry_max_error":float((state.covariance-state.covariance.T).abs().max()),
                       "covariance_trace":float(state.covariance.trace()),"prior_trace":float(prior.covariance.trace()),
                       "device":"cpu","dtype":"float64","jitter":0.,"factorization":"prior-whitened information Cholesky"}
        return GaussianPosterior(spec,state,prior,diagnostics)


def select_architecture(name):
    require(name == "D",f"architecture {name!r} unsupported; available architectures: D (exact T1 Gaussian)")
    return ExactGaussianD()
