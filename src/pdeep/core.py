"""Small immutable building blocks and strict JSON infrastructure."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import types
from typing import Union, get_args, get_origin, get_type_hints


class ValidationError(ValueError):
    """A scientific contract is invalid or unsupported."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


class Status(str, Enum):
    KNOWN = "known"
    PARTIAL = "partially_known"
    UNKNOWN = "unknown"
    ABSENT = "absent"


def canonical_json(value) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=lambda obj: asdict(obj) if is_dataclass(obj) else _unsupported(obj))


def _unsupported(obj):
    raise TypeError(f"not serializable: {type(obj).__name__}")


def require_immutable(value):
    """Frozen dataclasses must not conceal mutable child containers."""
    require(not isinstance(value, (list, dict, set)), "specification containers must be immutable tuples")
    if is_dataclass(value):
        for field in fields(value):
            require_immutable(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            require_immutable(item)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def decode(cls, value, path="spec"):
    """Decode declared dataclasses only; reject extra keys and coercive typing.

    Type hints come exclusively from this installed package, never from JSON.
    """
    origin, args = get_origin(cls), get_args(cls)
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return decode(option, value, path)
            except (ValidationError, TypeError, ValueError):
                pass
        raise ValidationError(f"{path}: incompatible value {value!r}")
    if cls is type(None):
        require(value is None, f"{path}: expected null")
        return None
    if origin is tuple:
        require(isinstance(value, (list, tuple)), f"{path}: expected array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(args[0], x, f"{path}[{i}]") for i, x in enumerate(value))
        require(len(value) == len(args), f"{path}: wrong tuple length")
        return tuple(decode(t, x, f"{path}[{i}]") for i, (t, x) in enumerate(zip(args, value)))
    if isinstance(cls, type) and issubclass(cls, Enum):
        try:
            return cls(value)
        except ValueError as error:
            raise ValidationError(f"{path}: unsupported {cls.__name__} {value!r}") from error
    if is_dataclass(cls):
        require(isinstance(value, dict), f"{path}: expected object")
        names = {f.name for f in fields(cls)}
        require(not (value.keys() - names), f"{path}: unknown keys {value.keys() - names}")
        hints = get_type_hints(cls)
        try:
            return cls(**{k: decode(hints[k], v, f"{path}.{k}") for k, v in value.items()})
        except TypeError as error:
            raise ValidationError(f"{path}: {error}") from error
    if cls is float:
        require(type(value) in (int, float) and math.isfinite(value), f"{path}: expected finite number")
        return float(value)
    require(type(value) is cls, f"{path}: expected {cls.__name__}")
    return value


@dataclass(frozen=True)
class TensorType:
    # Powers of length, physical time, and a common physical amplitude unit.
    shape: tuple[int, ...] = ()
    units: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        require(isinstance(self.shape, tuple) and all(type(x) is int and x > 0 for x in self.shape), "invalid tensor shape")
        require(isinstance(self.units, tuple) and len(self.units) == 3 and all(type(x) is int for x in self.units), "units must be integer (L,T,U) powers")

    @property
    def rank(self) -> int:
        return len(self.shape)


def unit_add(a, b, factor=1):
    return tuple(x + factor * y for x, y in zip(a, b))
