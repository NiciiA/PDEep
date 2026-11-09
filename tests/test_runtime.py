from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from pdeep.benchmarks import benchmark
from pdeep.core import ValidationError
from pdeep.diagnostics import ExperimentConfig, RandomStreams, compact_report, foundation_checks, source_identity
from pdeep.runtime import capabilities, environment_report, resolve_device


def test_runtime_capabilities_without_fallback():
    caps = {c.device: c for c in capabilities()}
    assert set(caps) == {"cpu", "mps", "cuda"}
    assert caps["cpu"].available
    assert caps["mps"].supported_dtypes == ("float32",)
    assert resolve_device("cpu", "float64", reference=True) == (torch.device("cpu"), torch.float64)
    for name, cap in caps.items():
        if not cap.available:
            with pytest.raises(ValidationError, match="unavailable"):
                resolve_device(name, "float32")
    with pytest.raises(ValidationError, match="CPU float64"):
        resolve_device("cpu", "float32", reference=True)
    with pytest.raises(ValidationError):
        resolve_device("cpu", "float16")
    report = environment_report()
    assert report["dependencies"]["pdeep-research"] == "0.1.0"
    assert all(v != "NOT INSTALLED" for v in report["dependencies"].values())


def test_mps_precision_rejected_even_on_mock_available(monkeypatch):
    from pdeep import runtime
    original = capabilities()
    from dataclasses import replace
    monkeypatch.setattr(runtime, "capabilities", lambda: tuple(replace(c, available=True) for c in original))
    with pytest.raises(ValidationError, match="no silent dtype"):
        runtime.resolve_device("mps", "float64")


def test_mps_fallback_environment_rejected(monkeypatch):
    from pdeep import runtime
    from dataclasses import replace
    original = capabilities()
    monkeypatch.setattr(runtime, "capabilities", lambda: tuple(replace(c, available=True) for c in original))
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValidationError, match="FALLBACK"):
        runtime.resolve_device("mps", "float32")


def test_config_identity_and_validation():
    a = ExperimentConfig()
    assert a.identity == ExperimentConfig().identity
    assert a.identity != ExperimentConfig(root_seed=1).identity
    assert json.loads(a.to_json()) == asdict(a)
    with pytest.raises(ValidationError, match="training"):
        ExperimentConfig(training_enabled=True)


def test_independent_streams_and_global_rng_untouched():
    streams = RandomStreams(42)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    expected = streams.numpy("training_queries", "train", "heat", 17).normal(size=4)
    streams.numpy("monitoring", 17).normal(size=100)
    actual = streams.numpy("training_queries", "train", "heat", 17).normal(size=4)
    assert np.array_equal(expected, actual)
    assert streams.seed("observations", "train", "heat", 17) != streams.seed("case_physics", "train", "heat", 17)
    assert streams.seed("observations", 17) != streams.seed("observations", "17")
    a = torch.randn(4, generator=streams.torch("tests", 1))
    b = torch.randn(4, generator=streams.torch("tests", 1))
    assert torch.equal(a, b)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert np.array_equal(numpy_before[1], np.random.get_state()[1])
    assert numpy_before[2:] == np.random.get_state()[2:]
    with pytest.raises(ValidationError, match="unregistered"):
        streams.numpy("typo")


def test_source_changes_and_reports(tmp_path):
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/example.py"
    source.write_text("x = 1\n")
    first = source_identity(tmp_path)
    source.write_text("x = 2\n")
    assert source_identity(tmp_path)["sha256"] != first["sha256"]
    specs = tuple(benchmark(n) for n in ("poisson", "heat", "wave"))
    report = compact_report(ExperimentConfig(), specs, Path(__file__).resolve().parents[1])
    assert all(c["passed"] for c in report["validation"])
    assert report["source"]["files"]["docs/scientific_contract_v1.md"]
    json.dumps(report, allow_nan=False)
