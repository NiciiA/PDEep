"""Lazy physical cases, independent measurement sets, and training-only calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import os
import tempfile

import numpy as np
import torch

from .batches import ObservationBatch, QueryBatch, UnknownCoefficients, expected_active_mask
from .core import canonical_json, decode, digest, require
from .diagnostics import RandomStreams
from .reference import (canonical_spec, evaluate, evaluate_batch, prior_scales,
                        reference_tensor, sample_prior)

VERSION = "reference-data-v1"
SPLITS = (("train", 12000, 0), ("validation", 1024, 12000), ("test", 1024, 13024))
SENSOR_COUNTS = {"poisson": 32, "heat": 48, "wave": 64}
EVALUATION_TIMES = (0., .02, .05, .1, .2, .35, .5, .75, 1.)


def array_hash(tensor):
    return digest({"shape": tuple(tensor.shape), "dtype": str(tensor.dtype),
                   "bytes_sha256": __import__("hashlib").sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()})


def persist_json(path: Path, value):
    """Content-addressed immutable records: idempotent, with atomic replacement."""
    path = Path(path)
    data = canonical_json(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text() == data, f"refusing to overwrite different artifact {path}")
        return path
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = stream.name
        stream.write(data)
    os.replace(temporary, path)
    return path


@dataclass(frozen=True)
class CaseKey:
    split: str
    pde: str
    case_id: int  # local index within a split

    def __post_init__(self):
        entries = {s: (n, start) for s, n, start in SPLITS}
        require(self.split in entries and self.pde in SENSOR_COUNTS, "unknown split/PDE")
        require(type(self.case_id) is int and 0 <= self.case_id < entries[self.split][0], "case ID outside split")

    @property
    def ordinal(self):
        return next(start for split, n, start in SPLITS if split == self.split) + self.case_id

    @property
    def physical_id(self):
        return f"{self.pde}/{self.ordinal:05d}"

    @property
    def rng_keys(self):
        return (self.split, self.pde, self.case_id)


@dataclass(frozen=True)
class PhysicalCase:
    key: CaseKey
    dataset_id: str
    xi: tuple[float, ...]

    def __post_init__(self):
        require(isinstance(self.xi, tuple) and len(self.xi) == (18 if self.key.pde == "wave" else 9)
                and all(math.isfinite(x) for x in self.xi), "invalid physical case coefficients")

    @property
    def identity(self):
        return digest(self)

    def tensor(self):
        return torch.tensor(self.xi, dtype=torch.float64)


@dataclass(frozen=True)
class CaseDataset:
    root_seed: int = 20261108

    def __post_init__(self):
        RandomStreams(self.root_seed)

    @property
    def identity(self):
        return digest({"version": VERSION, "root_seed": self.root_seed, "splits": SPLITS,
                       "problems": {p: canonical_spec(p).identity for p in SENSOR_COUNTS}})

    @property
    def streams(self):
        return RandomStreams(self.root_seed)

    def case(self, split, pde, case_id):
        key = CaseKey(split, pde, case_id)
        # a and wave b have separate substreams; neither shares state with sensors.
        a = self.streams.numpy("case_physics", *key.rng_keys, "a").standard_normal(9) * prior_scales(pde)[:9].numpy()
        xi = a
        if pde == "wave":
            b = self.streams.numpy("case_physics", *key.rng_keys, "b").standard_normal(9) * prior_scales(pde)[9:].numpy()
            xi = np.concatenate((a, b))
        return PhysicalCase(key, self.identity, tuple(float(x) for x in xi))

    def cases(self, split, pde):
        """Iterator, never materializes fields or all cases."""
        count = next((n for s, n, _ in SPLITS if s == split), None)
        require(count is not None and pde in SENSOR_COUNTS, "unknown split/PDE")
        return (self.case(split, pde, i) for i in range(count))

    def summary(self):
        return {"version": VERSION, "dataset_id": self.identity, "root_seed": self.root_seed,
                "per_pde": {p: {s: {"count": n, "global_ordinal_range": [start, start+n-1],
                                    "split_id": digest((self.identity, p, s, n, start))}
                                for s, n, start in SPLITS} for p in SENSOR_COUNTS},
                "storage": "lazy coefficients; fields evaluated only on requested coordinates"}


def mixture_counts(n, weights):
    require(type(n) is int and n >= 2, "at least two locations required for zero/early coverage")
    quotas = np.array(weights) * n
    counts = np.floor(quotas).astype(int)
    for index in sorted(range(3), key=lambda j: (-(quotas[j]-counts[j]), j))[:n-int(counts.sum())]:
        counts[index] += 1
    minimum = np.array((1, 1, 0))
    for index in (0, 1):
        if counts[index] < minimum[index]:
            donor = int(np.argmax(counts-minimum))
            counts[donor] -= 1
            counts[index] += 1
    return tuple(int(x) for x in counts)


def sample_times(n, rng, *, training_queries=False):
    counts = mixture_counts(n, (.125, .125, .75) if training_queries else (.1, .3, .6))
    zero, early, late = counts
    early_times = rng.uniform(0., .1, early).clip(np.nextafter(0., 1.), np.nextafter(.1, 0.))
    later = rng.uniform(0. if training_queries else .1, 1., late)
    times = np.concatenate((np.zeros(zero), early_times, later))
    groups = np.repeat(np.arange(3), counts)
    perm = rng.permutation(n)
    return torch.from_numpy(times[perm, None].copy()), tuple(int(g) for g in groups[perm]), counts


def quantity_labels(pde, n, rng):
    labels = np.zeros(n, dtype=np.int64)
    if pde == "wave":
        labels[n//2:] = 1  # balanced to within one scalar measurement
        rng.shuffle(labels)
    return torch.from_numpy(labels)


def query_layout(pde, n, position_rng, time_rng, quantity_rng, *, training=False):
    require(type(n) is int and n >= 2, "sample count must be >=2")
    low, high = (0., 1.) if training else (.05, .95)
    position = torch.from_numpy(position_rng.uniform(low, high, (n, 2)).clip(np.nextafter(low, high), np.nextafter(high, low)))
    time, groups, counts = (torch.zeros(n, 1, dtype=torch.float64), (), (0, 0, 0)) if pde == "poisson" else sample_times(n, time_rng, training_queries=training)
    query = QueryBatch(position[None], time[None], torch.full((1, n), pde != "poisson", dtype=torch.bool),
                       quantity_labels(pde, n, quantity_rng)[None], torch.ones(1, n, dtype=torch.bool))
    return query.validate((canonical_spec(pde),)), groups, counts


def case_coefficients(cases):
    cases = tuple(cases)
    specs = tuple(canonical_spec(c.key.pde) for c in cases)
    xi = torch.zeros(len(cases), 18, dtype=torch.float64)
    for i, case in enumerate(cases):
        xi[i, :len(case.xi)] = case.tensor()
    return UnknownCoefficients(xi, expected_active_mask(specs)).validate(specs)


@dataclass(frozen=True)
class QuantityScale:
    pde: str
    quantity_id: int
    rms: float


@dataclass(frozen=True)
class Normalizers:
    version: str
    root_seed: int
    calibration_cases: int
    points_per_case: int
    distribution: str
    problem_ids: tuple[tuple[str, str], ...]
    quantities: tuple[QuantityScale, ...]
    coefficient_scales: tuple[tuple[str, tuple[float, ...]], ...]

    def __post_init__(self):
        require(self.version == VERSION and self.distribution == "train_prior_and_training_query_risk", "unsupported normalizer provenance")
        require(self.calibration_cases > 0 and self.points_per_case >= 2, "invalid calibration budget")
        require(self.problem_ids == tuple((p, canonical_spec(p).identity) for p in SENSOR_COUNTS), "normalizers have stale problem identities")
        require(tuple((s.pde, s.quantity_id) for s in self.quantities) == (("poisson",0),("heat",0),("wave",0),("wave",1)), "normalizer quantity identities mismatch")
        require(all(math.isfinite(s.rms) and s.rms > 0 for s in self.quantities), "invalid RMS scales")
        require(self.coefficient_scales == tuple((p, tuple(prior_scales(p).tolist())) for p in SENSOR_COUNTS), "coefficient normalizers must match declared prior scales")

    @property
    def identity(self):
        return digest(self)

    def scale(self, pde, quantity_id):
        matches = [s.rms for s in self.quantities if s.pde == pde and s.quantity_id == quantity_id]
        require(len(matches) == 1, "unknown calibrated quantity")
        return matches[0]

    def save(self, directory):
        return persist_json(Path(directory) / f"{self.identity}.json", self)

    @classmethod
    def load(cls, path):
        restored = decode(cls, json.loads(Path(path).read_text()))
        require(Path(path).stem == restored.identity, "normalizer checksum/identity mismatch")
        return restored


def calibrate(root_seed=20261108, *, cases=512, points_per_case=128):
    """Training-only Monte Carlo calibration; no dataset/validation/test inputs.

    Each quantity uses the same independent prior/spacetime calibration draws.
    RMS averages all cases and the declared 12.5/12.5/75 training time mixture.
    Coefficient scales are the exact declared Gaussian prior standard deviations.
    """
    require(type(cases) is int and cases > 0 and type(points_per_case) is int and points_per_case >= 2, "invalid calibration budget")
    streams = RandomStreams(root_seed)
    scales = []
    for pde in SENSOR_COUNTS:
        xi = sample_prior(pde, streams.numpy("calibration", "train", pde, "xi"), cases)
        rng_x = streams.numpy("calibration", "train", pde, "position")
        rng_t = streams.numpy("calibration", "train", pde, "physical_time")
        sums = {q: 0. for q in ((0, 1) if pde == "wave" else (0,))}
        for start in range(0, cases, 32):
            count = min(32, cases-start)
            position = torch.from_numpy(rng_x.uniform(0., 1., (count, points_per_case, 2)).clip(np.nextafter(0.,1.), np.nextafter(1.,0.)))
            time = None if pde == "poisson" else torch.stack([sample_times(points_per_case, rng_t, training_queries=True)[0] for _ in range(count)])
            for q in sums:
                value = evaluate(pde, xi[start:start+count, None, :], position, time, quantity="v" if q else "u")
                sums[q] += float(value.square().sum())
        scales.extend(QuantityScale(pde, q, math.sqrt(total/(cases*points_per_case))) for q, total in sums.items())
    return Normalizers(VERSION, root_seed, cases, points_per_case, "train_prior_and_training_query_risk",
                       tuple((p, canonical_spec(p).identity) for p in SENSOR_COUNTS), tuple(scales),
                       tuple((p, tuple(prior_scales(p).tolist())) for p in SENSOR_COUNTS))


@dataclass(frozen=True)
class ObservationMetadata:
    identity: str
    case_identity: str
    set_id: str
    normalizer_identity: str
    clean: torch.Tensor
    noise: torch.Tensor
    eta: torch.Tensor
    time_groups: tuple[int, ...]
    time_counts: tuple[int, int, int]


@dataclass(frozen=True)
class ObservationSet:
    inputs: ObservationBatch  # Only this object may be passed to a future model.
    metadata: ObservationMetadata  # Hidden truth is never a field of inputs.
    recipe_json: str

    def save(self, directory):
        return persist_json(Path(directory) / f"{self.metadata.identity}.json", json.loads(self.recipe_json))


def observations(dataset: CaseDataset, case: PhysicalCase, normalizers: Normalizers,
                 *, set_id="default", count=None):
    require(case.dataset_id == dataset.identity, "case belongs to a different dataset")
    require(normalizers.root_seed == dataset.root_seed, "normalizers/dataset root seed mismatch")
    require(type(set_id) is str and bool(set_id), "observation set ID required")
    pde = case.key.pde
    n = SENSOR_COUNTS[pde] if count is None else count
    keys = (*case.key.rng_keys, set_id, n)
    def rng(part):
        return dataset.streams.numpy("observations", *keys, part)
    query, groups, counts = query_layout(pde, n, rng("position"), rng("time"), rng("quantity"))
    clean = evaluate_batch((canonical_spec(pde),), case_coefficients((case,)), query)
    eta = torch.from_numpy(np.exp(rng("eta").uniform(math.log(.02), math.log(.08), (1, n, 1))))
    scale = torch.tensor([normalizers.scale(pde, int(q)) for q in query.quantity_id[0]], dtype=torch.float64)[None, :, None]
    sigma = eta * scale
    noise = sigma * torch.from_numpy(rng("noise").standard_normal((1, n, 1)))
    inputs = ObservationBatch(query.position, query.physical_time, query.time_present, query.quantity_id,
                              query.valid, clean+noise, sigma).validate((canonical_spec(pde),))
    recipe = {"version": VERSION, "dataset_id": dataset.identity, "case_key": asdict(case.key),
              "case_identity": case.identity, "set_id": set_id, "count": n,
              "normalizer_identity": normalizers.identity,
              "input_hashes": {f.name: array_hash(getattr(inputs, f.name)) for f in fields(inputs)},
              "clean_hash": array_hash(clean), "noise_hash": array_hash(noise)}
    identity = digest(recipe)
    recipe["identity"] = identity
    return ObservationSet(inputs, ObservationMetadata(identity, case.identity, set_id, normalizers.identity,
                                                       clean, noise, eta, groups, counts), canonical_json(recipe))


def reload_observations(path, dataset, normalizers):
    recipe = json.loads(Path(path).read_text())
    require(recipe["version"] == VERSION and recipe["dataset_id"] == dataset.identity, "observation recipe dataset mismatch")
    require(recipe["normalizer_identity"] == normalizers.identity, "observation recipe normalizer mismatch")
    key = CaseKey(**recipe["case_key"])
    regenerated = observations(dataset, dataset.case(*key.rng_keys), normalizers,
                               set_id=recipe["set_id"], count=recipe["count"])
    require(json.loads(regenerated.recipe_json) == recipe and Path(path).stem == regenerated.metadata.identity,
            "observation recipe/content checksum mismatch")
    return regenerated


@dataclass(frozen=True)
class SupervisedQueries:
    inputs: QueryBatch
    target: torch.Tensor
    identity: str
    time_counts: tuple[int, int, int]
    time_groups: tuple[int, ...]


def training_queries(dataset, case, *, count=128, set_id="default"):
    require(case.dataset_id == dataset.identity, "case belongs to different dataset")
    require(type(set_id) is str and bool(set_id), "query set ID required")
    stream = {"train": "training_queries", "validation": "validation", "test": "tests"}[case.key.split]
    def rng(part):
        return dataset.streams.numpy(stream, *case.key.rng_keys, "queries", set_id, count, part)
    query, groups, counts = query_layout(case.key.pde, count, rng("position"), rng("time"), rng("quantity"), training=True)
    target = evaluate_batch((canonical_spec(case.key.pde),), case_coefficients((case,)), query)
    identity = digest({"version": VERSION, "case": case.identity, "set_id": set_id, "count": count,
                       "stream": stream, "input_hashes": {f.name: array_hash(getattr(query, f.name)) for f in fields(query)}})
    return SupervisedQueries(query, target, identity, counts, groups)


def evaluation_grid(pde, *, quantity_id=0, resolution=64, times=None):
    require(type(resolution) is int and resolution >= 2, "grid resolution must be >=2")
    spec = canonical_spec(pde)
    axis = torch.linspace(0., 1., resolution, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    position = torch.stack((xx, yy), -1).reshape(-1, 2)
    if pde == "poisson":
        require(times is None, "Poisson has no evaluation time axis")
        time = torch.zeros(len(position), 1, dtype=torch.float64)
    else:
        times = EVALUATION_TIMES if times is None else tuple(times)
        require(bool(times), "evaluation times cannot be empty")
        time = torch.tensor(times, dtype=torch.float64).repeat_interleave(len(position))[:, None]
        position = position.repeat(len(times), 1)
    n = len(position)
    return QueryBatch(position[None], time[None], torch.full((1,n), pde != "poisson", dtype=torch.bool),
                      torch.full((1,n), quantity_id, dtype=torch.int64), torch.ones(1,n,dtype=torch.bool)).validate((spec,))


def collate_inputs(inputs):
    """Pad single-row observation/query batches; padding carries no measurements."""
    require(bool(inputs) and all(x.position.shape[0] == 1 for x in inputs), "collation requires nonempty single-row inputs")
    cls = type(inputs[0])
    require(cls in {QueryBatch, ObservationBatch} and all(type(x) is cls for x in inputs), "cannot mix observation and query contracts")
    maximum = max(x.position.shape[1] for x in inputs)
    packed = {}
    for field in fields(inputs[0]):
        tensors = []
        for item in inputs:
            t = getattr(item, field.name)
            pad_shape = (1, maximum-t.shape[1], *t.shape[2:])
            filler = False if t.dtype == torch.bool else (-1 if t.dtype == torch.int64 else float("nan"))
            tensors.append(torch.cat((t, torch.full(pad_shape, filler, dtype=t.dtype, device=t.device)), dim=1))
        packed[field.name] = torch.cat(tensors)
    return cls(**packed)
