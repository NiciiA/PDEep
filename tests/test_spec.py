from dataclasses import FrozenInstanceError, replace
import json
import math
import pytest

from pdeep.benchmarks import benchmark, MODE_IDS, TAU, MODE_EIGENVALUES
from pdeep.core import Status, TensorType, ValidationError
from pdeep.spec import InitialCondition, ParameterBlock, Prior, ProblemSpec


@pytest.mark.parametrize("name", ["poisson", "heat", "wave"])
def test_stable_round_trip(name):
    spec = benchmark(name)
    restored = ProblemSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.to_json() == spec.to_json()
    assert len(spec.identity) == 64 and restored.identity == spec.identity
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"
    assert spec.physical_time is None if name == "poisson" else spec.physical_time.interval == (0, 1)


def test_prior_metadata_exact_contract():
    assert MODE_IDS == tuple(f"{m},{n}" for m in (1,2,3) for n in (1,2,3))
    assert TAU[0] == .5
    p = benchmark("poisson")
    assert p.parameter_blocks[0].coefficient_map == "poisson_source_lambda_times_a"
    assert p.parameter_blocks[0].prior.scale == TAU
    w = benchmark("wave")
    assert w.parameter_blocks[1].prior.scale == pytest.approx(tuple(.4*.7*math.sqrt(l)*t for l,t in zip(MODE_EIGENVALUES,TAU)))
    assert {b.id:b.value for b in benchmark("heat").bindings}["alpha"] == .02
    assert {b.id:b.value for b in w.bindings}["c"] == .7


def test_semantic_collection_order_is_canonical():
    spec = benchmark("wave")
    reordered = replace(spec, bindings=spec.bindings[::-1], parameter_blocks=spec.parameter_blocks[::-1],
                        boundaries=spec.boundaries[::-1], initials=spec.initials[::-1], observations=spec.observations[::-1])
    assert reordered.to_json() == spec.to_json()
    assert reordered.identity == spec.identity


@pytest.mark.parametrize("change", [
    {"schema_version":"9.0"}, {"solver_key":"execute.me"}, {"solver_key":"v1.analytical.wave"},
    {"solution_concepts":("entropy",)}, {"boundaries":()},
    {"initials":(InitialCondition("ic", "u", 0., 0, "a"),)},
])
def test_invalid_spec(change):
    with pytest.raises(ValidationError):
        replace(benchmark("poisson"), **change)


@pytest.mark.parametrize("name", ["heat", "wave"])
def test_dynamic_initials_required(name):
    spec = benchmark(name)
    with pytest.raises(ValidationError, match="initial"):
        replace(spec, initials=())
    with pytest.raises(ValidationError, match="initial time"):
        replace(spec, initials=(replace(spec.initials[0], physical_time=.1), *spec.initials[1:]))


def test_unknown_zero_partial_and_absent():
    b = benchmark("poisson").parameter_blocks[0]
    assert b.known_values == (None,)*9
    for bad in [replace(b, known_values=(0.,)*9), replace(b, prior=None), replace(b, known_values=(float("nan"),)*9)]:
        with pytest.raises(ValidationError):
            bad.validate()
    known = replace(b, status=Status.KNOWN, known_values=(0.,)*9, prior=None)
    known.validate()
    partial = replace(b, status=Status.PARTIAL, known_values=(0.,)+(None,)*8)
    partial.validate()
    replace(b, status=Status.ABSENT, prior=None).validate()
    with pytest.raises(ValidationError, match="partial"):
        replace(b, status=Status.PARTIAL).validate()
    with pytest.raises(ValidationError, match="scales"):
        replace(b, prior=Prior("independent_gaussian", (0.,)*9, (0.,)*9)).validate()


def test_scalar_unknown_block():
    block = ParameterBlock("theta", "alpha", "scalar", "scalar", ("theta",), TensorType(),
                           Status.UNKNOWN, (None,), Prior("bounded_uniform", bounds=(.01,.05)), transform="bounded_logistic")
    block.validate()
    with pytest.raises(ValidationError, match="bounded transform"):
        replace(block, transform="identity").validate()


def test_bc_conflicts_and_unknown_observations():
    spec = benchmark("poisson")
    bc = spec.boundaries[0]
    for boundaries in [(replace(bc, parts=("missing",)),), (replace(bc, value=1.),),
                       (bc, replace(bc, id="conflict")), (replace(bc, target="f"),)]:
        with pytest.raises(ValidationError):
            replace(spec, boundaries=boundaries)
    with pytest.raises(ValidationError, match="measurement"):
        replace(spec, observations=(replace(spec.observations[0], target="f"),))


def test_binding_dependency_validation():
    spec = benchmark("poisson")
    with pytest.raises(ValidationError, match="binding"):
        replace(spec, bindings=(*spec.bindings, spec.bindings[0]))
    bindings = tuple(replace(b, dependencies=("nonexistent",)) if b.id=="f" else b for b in spec.bindings)
    with pytest.raises(ValidationError, match="dependency"):
        replace(spec, bindings=bindings)
    with pytest.raises(ValidationError, match="cyclic"):
        replace(spec, parameter_blocks=(replace(spec.parameter_blocks[0], dependencies=("f",)),))


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(extra="not allowed"),
    lambda d: d.update(schema_version=1),
    lambda d: d["graph"]["nodes"][0].update(op="EVAL"),
    lambda d: d["bindings"][0].update(value="0.0"),
    lambda d: d["parameter_blocks"][0]["prior"].update(scale=[float("inf")]*9),
])
def test_strict_json_rejects_invalid_inputs(mutation):
    raw = json.loads(benchmark("poisson").to_json())
    mutation(raw)
    with pytest.raises((ValidationError, ValueError)):
        ProblemSpec.from_json(json.dumps(raw))


def test_duplicate_json_keys_and_deep_immutability():
    with pytest.raises(ValidationError, match="duplicate JSON"):
        ProblemSpec.from_json('{"schema_version":"1.0","schema_version":"2.0"}')
    spec = benchmark("poisson")
    with pytest.raises(ValidationError, match="immutable"):
        replace(spec, boundaries=list(spec.boundaries))
