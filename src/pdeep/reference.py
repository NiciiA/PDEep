"""Direct nine-mode T1 physics on CPU float64; no temporal interpolation.

All tensor inputs retain autograd connectivity. Device/dtype conversion must be
explicit at the caller. Low-level evaluation takes unpadded xi[...,9 or 18].
"""
from __future__ import annotations

from functools import lru_cache
import math
import torch

from .batches import QueryBatch, UnknownCoefficients, masked_values, validate_prediction
from .benchmarks import benchmark
from .core import require
from .spec import ProblemSpec


def reference_tensor(t: torch.Tensor, name="tensor"):
    require(isinstance(t, torch.Tensor) and t.device.type == "cpu" and t.dtype == torch.float64,
            f"{name} must be CPU float64; no automatic conversion")
    require(bool(torch.isfinite(t).all()), f"nonfinite {name}")


def modes():
    return torch.tensor([(m, n) for m in range(1, 4) for n in range(1, 4)], dtype=torch.float64)


def eigenvalues():
    return math.pi**2 * modes().square().sum(-1)


def basis(position: torch.Tensor, derivative=(0, 0)) -> torch.Tensor:
    """phi or d_x^r d_y^s phi: position[...,2] -> [...,9]."""
    reference_tensor(position, "position")
    require(position.ndim >= 1 and position.shape[-1] == 2, "position must end in dimension 2")
    require(bool(((position >= 0) & (position <= 1)).all()), "position outside unit square")
    require(isinstance(derivative, tuple) and len(derivative) == 2
            and all(type(d) is int and d in (0, 1, 2) for d in derivative), "spatial derivative orders must be 0,1,2")
    wave_numbers = math.pi * modes()
    phase = position.unsqueeze(-2) * wave_numbers
    factors = []
    for axis, order in enumerate(derivative):
        p, k = phase[..., axis], wave_numbers[:, axis]
        factors.append(p.sin() if order == 0 else (k*p.cos() if order == 1 else -k.square()*p.sin()))
    return 2 * factors[0] * factors[1]


def basis_gradient(position):
    return torch.stack((basis(position, (1, 0)), basis(position, (0, 1))), dim=-1)


def basis_laplacian(position):
    return basis(position, (2, 0)) + basis(position, (0, 2))


@lru_cache(maxsize=3)
def canonical_spec(name):
    return benchmark(name)


def require_supported(spec: ProblemSpec):
    require(spec.name in {"poisson", "heat", "wave"} and spec == canonical_spec(spec.name),
            "analytical reference supports canonical T1 specs only; changed graphs/physics are unsupported")


def prior_scales(name):
    spec = canonical_spec(name)
    return torch.tensor(tuple(s for b in spec.parameter_blocks for s in b.prior.scale), dtype=torch.float64)


def sample_prior(name, rng, count=1):
    """Explicit generator; independent centered Gaussian coefficients [count,D]."""
    require(type(count) is int and count > 0, "positive prior sample count required")
    return torch.from_numpy(rng.standard_normal((count, len(prior_scales(name))))) * prior_scales(name)


def prior_log_density(name, xi):
    reference_tensor(xi, "xi")
    scales = prior_scales(name)
    require(xi.ndim >= 1 and xi.shape[-1] == len(scales), "prior xi component count mismatch")
    return (-.5*(xi/scales).square() - scales.log() - .5*math.log(2*math.pi)).sum(-1)


def evaluate(name: str, xi: torch.Tensor, position: torch.Tensor,
             physical_time: torch.Tensor | None = None, *, quantity="u",
             spatial_derivative=(0, 0), time_order=0) -> torch.Tensor:
    """Evaluate scalar field/derivative directly, output [...,1].

    xi[D] broadcasts over coordinates; xi[...,D] can also match position's
    leading dimensions. Add a singleton query dimension for batched cases.
    Inputs f/u0/v0 are spatial fields and take no physical-time argument.
    """
    spec = canonical_spec(name)
    reference_tensor(xi, "xi")
    require(xi.ndim >= 1 and xi.shape[-1] == (18 if name == "wave" else 9), "xi dimension mismatch")
    require(type(time_order) is int and time_order in (0, 1, 2), "time order must be 0,1,2")
    input_field = quantity in {"f", "u0", "v0"}
    require(quantity in ({"u", "f"} if name == "poisson" else ({"u", "u0"} if name == "heat" else {"u", "v", "u0", "v0"})), "unsupported physical quantity")
    if spec.physical_time is None or input_field:
        require(physical_time is None and time_order == 0, "stationary/input fields have no physical-time axis")
    else:
        require(physical_time is not None, "dynamic field requires continuous physical time")
        reference_tensor(physical_time, "physical_time")
        require(physical_time.shape == position.shape[:-1] + (1,), "physical_time must match position [...,1]")
        require(bool(((physical_time >= 0) & (physical_time <= 1)).all()), "physical time outside [0,1]")
    phi = basis(position, spatial_derivative)
    lam = eigenvalues()
    a = xi[..., :9]
    if quantity == "f":
        coefficients = a * lam
    elif quantity == "u0" or name == "poisson":
        coefficients = a
    elif quantity == "v0":
        coefficients = xi[..., 9:]
    elif name == "heat":
        rate = -.02 * lam
        coefficients = a * (physical_time*rate).exp() * rate.pow(time_order)
    else:
        omega = .7 * lam.sqrt()
        b = xi[..., 9:]
        phase = physical_time * omega
        # v is exactly the next physical derivative of displacement.
        order = time_order + (quantity == "v")
        if order == 0:
            coefficients = a*phase.cos() + (b/omega)*phase.sin()
        elif order == 1:
            coefficients = -omega*a*phase.sin() + b*phase.cos()
        elif order == 2:
            coefficients = -omega.square()*a*phase.cos() - omega*b*phase.sin()
        else:
            coefficients = omega.pow(3)*a*phase.sin() - omega.square()*b*phase.cos()
    return (phi * coefficients).sum(-1, keepdim=True)


def evaluate_batch(specs: tuple[ProblemSpec, ...], coefficients: UnknownCoefficients,
                   query: QueryBatch) -> torch.Tensor:
    """Apply point-observation declarations, ignoring padded payloads entirely."""
    coefficients.validate(specs)
    query.validate(specs)
    reference_tensor(coefficients.masked(), "xi")
    clean = query.masked()
    reference_tensor(clean.position, "position")
    reference_tensor(clean.physical_time, "physical_time")
    rows = []
    for i, spec in enumerate(specs):
        require_supported(spec)
        p = clean.position[i]
        t = None if spec.physical_time is None else clean.physical_time[i]
        xi = coefficients.masked()[i, :18 if spec.name == "wave" else 9]
        u = evaluate(spec.name, xi, p, t)
        if spec.name == "wave":
            v = evaluate(spec.name, xi, p, t, quantity="v")
            u = torch.where(clean.quantity_id[i, :, None] == 1, v, u)
        rows.append(u)
    result = masked_values(torch.stack(rows), clean.valid)
    return validate_prediction(result, query)
