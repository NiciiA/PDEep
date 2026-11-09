"""Explicit device and precision capabilities; no allocation-time fallbacks."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import platform
import sys
import torch
from .core import ValidationError, require


@dataclass(frozen=True)
class Capability:
    device: str
    available: bool
    supported_dtypes: tuple[str, ...]
    reference_allowed: bool
    detail: str


def capabilities() -> tuple[Capability, ...]:
    cuda = torch.cuda.is_available()
    mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return (
        Capability("cpu", True, ("float32", "float64"), True, platform.processor() or platform.machine()),
        Capability("mps", mps, ("float32",), False, "MPS float64 unsupported; explicit float32 neural execution only"),
        Capability("cuda", cuda, ("float32", "float64"), False,
                   torch.cuda.get_device_name(0) if cuda else "CUDA unavailable"),
    )


def resolve_device(device: str, dtype: str, *, reference=False) -> tuple[torch.device, torch.dtype]:
    require(dtype in {"float32", "float64"}, f"unsupported explicit precision {dtype!r}")
    try:
        parsed = torch.device(device)
    except (RuntimeError, ValueError, TypeError) as error:
        raise ValidationError(f"invalid requested device {device!r}: {error}") from error
    reports = {c.device: c for c in capabilities()}
    require(parsed.type in reports, f"unsupported device {device!r}")
    cap = reports[parsed.type]
    require(cap.available, f"requested {device} unavailable; select a device explicitly")
    require(dtype in cap.supported_dtypes, f"{device} does not support {dtype}; no silent dtype change")
    if parsed.type == "cuda" and parsed.index is not None:
        require(parsed.index < torch.cuda.device_count(), "CUDA device index out of range")
    if parsed.type in {"cpu", "mps"}:
        require(parsed.index is None, "CPU/MPS device indices are unsupported")
    if parsed.type == "mps":
        import os
        require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1", "disable PYTORCH_ENABLE_MPS_FALLBACK before explicit MPS execution")
    require(not reference or (parsed.type == "cpu" and dtype == "float64"), "reference calculations require CPU float64")
    return parsed, getattr(torch, dtype)


def environment_report() -> dict:
    packages = ("pdeep-research", "torch", "numpy", "scipy", "matplotlib", "pytest", "notebook", "nbformat", "nbclient", "ipykernel", "setuptools")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "dependencies": versions, "torch_cuda_build": torch.version.cuda,
            "mps_built": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built()),
            "mps_fallback_env": __import__("os").environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset")}
