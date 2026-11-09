from dataclasses import replace
import pytest
import torch

from pdeep.batches import (ObservationBatch, QueryBatch, UnknownCoefficients,
                          expected_active_mask, masked_mean, masked_sum, validate_prediction)
from pdeep.benchmarks import benchmark
from pdeep.core import ValidationError


def observations():
    specs = tuple(benchmark(n) for n in ("poisson", "heat", "wave"))
    batch = ObservationBatch(
        position=torch.full((3, 2, 2), .5, dtype=torch.float64),
        physical_time=torch.tensor([[[float("nan")], [float("nan")]], [[0.], [.2]], [[.1], [.3]]], dtype=torch.float64),
        time_present=torch.tensor([[False, False], [True, True], [True, True]]),
        quantity_id=torch.tensor([[0, 0], [0, 0], [0, 1]]),
        valid=torch.ones(3, 2, dtype=torch.bool),
        value=torch.zeros(3, 2, 1, dtype=torch.float64),
        sigma=torch.ones(3, 2, 1, dtype=torch.float64) * .1)
    return specs, batch


def test_valid_observations_queries_predictions_and_absent_time():
    specs, obs = observations()
    assert obs.validate(specs) is obs
    clean = obs.masked()
    assert torch.equal(clean.physical_time[0], torch.zeros(2, 1, dtype=torch.float64))
    q = QueryBatch(obs.position, obs.physical_time, obs.time_present, obs.quantity_id, obs.valid)
    q.validate(specs)
    validate_prediction(torch.zeros(3, 2, 1, dtype=torch.float64), q)
    with pytest.raises(ValidationError, match="predictions"):
        validate_prediction(torch.zeros(3, 2), q)
    with pytest.raises(ValidationError, match="dtype"):
        validate_prediction(torch.zeros(3, 2, 1), q)


@pytest.mark.parametrize("field,change", [
    ("position", lambda t: t[:, :, :1]),
    ("position", lambda t: t + 2),
    ("position", lambda t: t.float()),
    ("physical_time", lambda t: t.squeeze(-1)),
    ("physical_time", lambda t: torch.full_like(t, 2.)),
    ("time_present", lambda t: torch.ones_like(t)),
    ("valid", lambda t: t.float()),
    ("quantity_id", lambda t: t.float()),
    ("quantity_id", lambda t: t + 9),
    ("sigma", lambda t: t * 0),
    ("sigma", lambda t: -t),
    ("value", lambda t: t + float("nan")),
])
def test_invalid_tensors(field, change):
    specs, obs = observations()
    with pytest.raises(ValidationError):
        replace(obs, **{field: change(getattr(obs, field))}).validate(specs)


def test_padding_nan_inert_and_permutation_invariant():
    specs, obs = observations()
    def pad(t, fill):
        return torch.cat((t, torch.full_like(t[:, :1], fill)), dim=1)
    padded = ObservationBatch(pad(obs.position, float("nan")), pad(obs.physical_time, float("nan")),
                              pad(obs.time_present, True), pad(obs.quantity_id, -99), pad(obs.valid, False),
                              pad(obs.value, float("nan")), pad(obs.sigma, float("nan")))
    padded.validate(specs)
    assert masked_mean(padded.sigma, padded.valid) == masked_mean(obs.sigma, obs.valid)
    assert torch.equal(masked_sum(padded.sigma, padded.valid, dim=1), masked_sum(obs.sigma, obs.valid, dim=1))
    clean = padded.masked()
    assert torch.isfinite(clean.value).all() and torch.isfinite(clean.physical_time).all()
    assert not clean.time_present[:, -1].any()
    permutation = torch.tensor([2, 0, 1])
    assert masked_mean(padded.sigma[:, permutation], padded.valid[:, permutation]) == masked_mean(obs.sigma, obs.valid)
    empty = torch.zeros_like(obs.valid)
    assert masked_mean(obs.sigma, empty) == 0


def test_masked_gradient_and_padding():
    x = torch.tensor([[[2.], [float("nan")]]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    masked_mean(x, mask).backward()
    assert torch.equal(x.grad, torch.tensor([[[1.], [0.]]]))


def test_v1_coefficient_slots_and_inactive_nan():
    specs, _ = observations()
    active = expected_active_mask(specs)
    assert active.sum(1).tolist() == [9, 9, 18]
    xi = torch.where(active, torch.ones(3, 18), torch.full((3, 18), float("nan")))
    coeffs = UnknownCoefficients(xi, active)
    coeffs.validate(specs)
    assert torch.isfinite(coeffs.masked()).all()
    assert torch.equal(coeffs.masked()[0, 9:], torch.zeros(9))
    wrong = active.clone()
    wrong[0, 9] = True
    with pytest.raises(ValidationError, match="identities"):
        UnknownCoefficients(xi, wrong).validate(specs)
    with pytest.raises(ValidationError, match="shape"):
        UnknownCoefficients(xi[:, :9], active[:, :9]).validate(specs)
