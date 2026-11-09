"""Physical tensor contracts. These classes validate inputs, not generate data."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from .core import Status, require
from .spec import ProblemSpec


def masked_values(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Padding may contain NaNs. torch.where, rather than multiplication, removes it."""
    require(valid.dtype == torch.bool and values.device == valid.device, "mask must be bool on the value device")
    require(values.shape[:valid.ndim] == valid.shape, "mask dimensions mismatch")
    mask = valid.reshape(valid.shape + (1,) * (values.ndim - valid.ndim))
    return torch.where(mask, values, torch.zeros((), dtype=values.dtype, device=values.device))


def masked_sum(values: torch.Tensor, valid: torch.Tensor, dim=None):
    return masked_values(values, valid).sum(dim=dim)


def masked_mean(values: torch.Tensor, valid: torch.Tensor):
    """Mean over all valid scalar entries; empty selections return zero."""
    masked_values(values, valid)  # Validate broadcasting/device before selection.
    selected = values[valid]
    # Remove padding before reduction so inserted zeros cannot change reduction
    # grouping and roundoff. Empty differentiable sums are well-defined.
    return selected.sum() / max(selected.numel(), 1)


@dataclass(frozen=True)
class QueryBatch:
    position: torch.Tensor
    physical_time: torch.Tensor
    time_present: torch.Tensor
    quantity_id: torch.Tensor
    valid: torch.Tensor

    def validate(self, specs: tuple[ProblemSpec, ...]):
        require(self.position.ndim == 3 and self.position.shape[-1] == 2, "position must have shape [B,N,2]")
        b, n, _ = self.position.shape
        require(len(specs) == b, "one ProblemSpec per batch row required")
        require(self.position.dtype in (torch.float32, torch.float64), "physical tensors require explicit float32 or float64")
        expected = {"physical_time": (b, n, 1), "time_present": (b, n), "quantity_id": (b, n), "valid": (b, n)}
        for name, shape in expected.items():
            t = getattr(self, name)
            require(tuple(t.shape) == shape, f"{name} must have shape {shape}")
            require(t.device == self.position.device, f"{name} device mismatch; no implicit transfer")
        require(self.time_present.dtype == self.valid.dtype == torch.bool, "time_present and valid must be bool")
        require(self.quantity_id.dtype == torch.int64, "quantity_id must be int64")
        require(self.physical_time.dtype == self.position.dtype, "physical_time dtype mismatch; no implicit casts")
        require(bool(torch.isfinite(self.position[self.valid]).all()), "nonfinite valid position")
        for i, spec in enumerate(specs):
            mask = self.valid[i]
            p = self.position[i][mask]
            require(bool(((p >= 0) & (p <= 1)).all()), "position outside closed unit-square evaluation domain")
            present = self.time_present[i][mask]
            require(bool((present == (spec.physical_time is not None)).all()), "physical time presence conflicts with problem; stationary time is absent")
            if spec.physical_time is not None:
                t = self.physical_time[i, :, 0][mask]
                lo, hi = spec.physical_time.interval
                require(bool((torch.isfinite(t) & (t >= lo) & (t <= hi)).all()), "invalid physical time")
            q = self.quantity_id[i][mask]
            allowed = torch.tensor([o.quantity_id for o in spec.observations], device=q.device, dtype=q.dtype)
            require(bool(torch.isin(q, allowed).all()), "unknown observation quantity ID")
        return self

    def masked(self) -> QueryBatch:
        return QueryBatch(masked_values(self.position, self.valid),
                          masked_values(self.physical_time, self.valid & self.time_present),
                          self.time_present & self.valid, masked_values(self.quantity_id, self.valid), self.valid)


@dataclass(frozen=True)
class ObservationBatch(QueryBatch):
    value: torch.Tensor
    sigma: torch.Tensor

    def validate(self, specs):
        super().validate(specs)
        shape = self.position.shape[:2] + (1,)
        for name in ("value", "sigma"):
            t = getattr(self, name)
            require(t.shape == shape, f"{name} must have shape [B,N,1]")
            require(t.device == self.position.device and t.dtype == self.position.dtype, f"{name} device/dtype mismatch")
            require(bool(torch.isfinite(t[self.valid]).all()), f"nonfinite valid {name}")
        require(bool((self.sigma[self.valid] > 0).all()), "sigma must be strictly positive; noiseless constraints are unsupported")
        return self

    def masked(self) -> ObservationBatch:
        q = super().masked()
        return ObservationBatch(q.position, q.physical_time, q.time_present, q.quantity_id, q.valid,
                                masked_values(self.value, self.valid), masked_values(self.sigma, self.valid))


def validate_prediction(prediction: torch.Tensor, query: QueryBatch):
    require(prediction.shape == query.position.shape[:2] + (1,), "predictions must have shape [B,Q,1]")
    require(prediction.dtype == query.position.dtype and prediction.device == query.position.device, "prediction device/dtype mismatch")
    require(bool(torch.isfinite(prediction[query.valid]).all()), "nonfinite valid prediction")
    return prediction


def expected_active_mask(specs, device="cpu"):
    rows = []
    for spec in specs:
        require(spec.name in {"poisson", "heat", "wave"}, "18-slot mask supports V1 benchmark families only")
        expected = ("a", "b") if spec.name == "wave" else ("a",)
        blocks = {b.id: b for b in spec.parameter_blocks}
        require(set(blocks) == set(expected), "18-slot V1 contract excludes added scalar parameters/T2")
        row = []
        for key in expected:
            block = blocks[key]
            require(len(block.component_ids) == 9, "V1 blocks require nine modal coefficients")
            row.extend([block.status in {Status.UNKNOWN, Status.PARTIAL} and value is None for value in block.known_values])
        rows.append(row + [False] * (18 - len(row)))
    return torch.tensor(rows, dtype=torch.bool, device=device).reshape(len(rows), 18)


@dataclass(frozen=True)
class UnknownCoefficients:
    xi: torch.Tensor
    active: torch.Tensor

    def validate(self, specs):
        require(self.xi.shape == self.active.shape == (len(specs), 18), "xi and active must have shape [B,18]")
        require(self.xi.dtype in (torch.float32, torch.float64), "xi requires float32 or float64")
        require(self.active.dtype == torch.bool and self.active.device == self.xi.device, "active must be bool on xi device")
        require(torch.equal(self.active, expected_active_mask(specs, self.xi.device)), "active mask conflicts with component identities/status")
        require(bool(torch.isfinite(self.xi[self.active]).all()), "nonfinite active coefficient")
        return self

    def masked(self):
        return masked_values(self.xi, self.active)
