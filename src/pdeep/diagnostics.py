"""Immutable config, source identity, independent random streams, compact checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import subprocess
import numpy as np
import torch
from .core import canonical_json, digest, require
from .runtime import resolve_device


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str = "1.0"
    stage: str = "foundations"
    device: str = "cpu"
    dtype: str = "float32"
    reference_device: str = "cpu"
    reference_dtype: str = "float64"
    root_seed: int = 20261108
    training_enabled: bool = False
    architecture: str = "D"

    def __post_init__(self):
        require(self.schema_version == "1.0" and self.stage in {"foundations", "reference_data", "bayesian_D", "representations", "architecture_A", "architecture_C", "architecture_B", "training_engine"}, "unsupported config version/stage")
        require(type(self.root_seed) is int and 0 <= self.root_seed < 2**63, "seed must be an integer in [0,2^63)")
        require(self.training_enabled is False, "full training remains disabled; bounded A/C diagnostics require their explicit training configs")
        require(self.architecture in {"A","B0","B1","C","D"}, "unsupported architecture; select A, B0, B1, C or D")
        resolve_device(self.device, self.dtype)
        resolve_device(self.reference_device, self.reference_dtype, reference=True)

    @property
    def identity(self):
        return digest(self)

    def to_json(self):
        return canonical_json(self)


STREAMS = ("case_physics", "observations", "training_queries", "calibration", "posterior_samples",
           "validation", "tests", "monitoring", "model_initialization", "optimization")


@dataclass(frozen=True)
class RandomStreams:
    root_seed: int

    def __post_init__(self):
        require(type(self.root_seed) is int and 0 <= self.root_seed < 2**63, "invalid root seed")

    def seed(self, stream: str, *keys: str | int) -> int:
        require(stream in STREAMS, f"unregistered random stream {stream!r}")
        require(all(type(k) in (str, int) for k in keys), "random keys must be strings/integers")
        raw = canonical_json((self.root_seed, stream, keys)).encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63)

    def numpy(self, stream, *keys):
        return np.random.default_rng(self.seed(stream, *keys))

    def torch(self, stream, *keys, device="cpu"):
        resolved, _ = resolve_device(device, "float32")
        require(resolved.type in {"cpu", "cuda"}, "independent torch.Generator on MPS unsupported; use explicit CPU generator and transfer")
        return torch.Generator(device=resolved).manual_seed(self.seed(stream, *keys))


def source_identity(root: Path) -> dict:
    """Hash scientific source/config/notebook inputs, excluding outputs and caches."""
    root = root.resolve()
    candidates = [root / "pyproject.toml", root / "README.md"]
    for directory in ("src", "tests", "scripts", "docs", "notebooks"):
        candidates.extend(p for p in (root / directory).rglob("*")
                          if p.is_file() and p.suffix in {".py", ".md", ".ipynb", ".json"}
                          and "__pycache__" not in p.parts)
    def input_bytes(path):
        if path.suffix == ".ipynb":
            import json
            notebook = json.loads(path.read_text())
            return canonical_json({"nbformat": notebook["nbformat"],
                                   "cells": [{"cell_type": c["cell_type"], "source": c["source"],
                                              "tags": c.get("metadata", {}).get("tags", [])}
                                             for c in notebook["cells"]]}).encode()
        return path.read_bytes()
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(input_bytes(p)).hexdigest()
              for p in sorted(candidates) if p.is_file()}
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"sha256": digest(hashes), "files": hashes, "git_revision": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"),
            "identity_method": "SHA256 manifest; notebook hashes cover cell sources/types/tags, excluding outputs"}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def foundation_checks(specs) -> tuple[CheckResult, ...]:
    from .spec import ProblemSpec
    results = []
    for spec in specs:
        try:
            spec.validate()
            restored = ProblemSpec.from_json(spec.to_json())
            require(restored == spec and restored.identity == spec.identity, "round-trip mismatch")
            results.append(CheckResult(spec.name, True, f"valid DAG; {len(spec.graph.nodes)} nodes; stable JSON/hash; reference execution gated to canonical T1"))
        except (ValueError, TypeError) as error:
            results.append(CheckResult(spec.name, False, str(error)))
    return tuple(results)


def compact_report(config, specs, root):
    from .runtime import capabilities, environment_report
    return {"config": asdict(config), "config_sha256": config.identity,
            "environment": environment_report(), "devices": [asdict(c) for c in capabilities()],
            "source": source_identity(Path(root)), "problem_sha256": {s.name: s.identity for s in specs},
            "validation": [asdict(c) for c in foundation_checks(specs)],
            "implemented": "T1 physics/data, A/C/B0/B1, exact D, shared successful-update engine, fixed validation, live reports and local resumable checkpoints",
            "not_implemented": ["T2", "flow matching", "Drive sync", "extensive Stage7B evaluation", "AMP/distributed training"]}
