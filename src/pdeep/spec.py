"""Versioned immutable scientific specification, independent of execution."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from .core import Status, TensorType, canonical_json, decode, digest, require, require_immutable
from .graph import Binding, EquationGraph


@dataclass(frozen=True)
class Geometry:
    kind: str = "open_unit_square"
    bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
    boundary_parts: tuple[str, ...] = ("x0", "x1", "y0", "y1")
    coordinate_units: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class PhysicalTime:
    interval: tuple[float, float] = (0.0, 1.0)
    units: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class Prior:
    family: str
    mean: tuple[float, ...] = ()
    scale: tuple[float, ...] = ()
    bounds: tuple[float, float] | None = None

    def validate(self, size):
        require(self.family in {"independent_gaussian", "bounded_uniform"}, "unsupported prior")
        if self.family == "independent_gaussian":
            require(self.bounds is None, "Gaussian prior cannot have bounds")
            require(len(self.mean) == len(self.scale) == size, "prior component count mismatch")
            require(all(math.isfinite(x) for x in self.mean), "nonfinite prior mean")
            require(all(math.isfinite(x) and x > 0 for x in self.scale), "prior scales must be positive finite")
        else:
            require(size == 1 and self.bounds is not None and not self.mean and not self.scale, "bounded uniform is scalar with explicit bounds")
            lo, hi = self.bounds
            require(math.isfinite(lo) and math.isfinite(hi) and lo < hi, "invalid prior bounds")


@dataclass(frozen=True)
class ParameterBlock:
    id: str
    quantity_binding: str
    kind: str  # scalar or field
    representation: str  # named declaration, never executable code
    component_ids: tuple[str, ...]
    component_type: TensorType
    status: Status
    known_values: tuple[float | None, ...]
    prior: Prior | None
    dependencies: tuple[str, ...] = ()
    coefficient_map: str = "identity"
    transform: str = "identity"

    def validate(self):
        require_immutable(self)
        require(bool(self.id) and len(set(self.component_ids)) == len(self.component_ids) and bool(self.component_ids), "block needs unique component identities")
        require(self.kind in {"scalar", "field"}, "unsupported parameter block kind")
        require(self.representation in {"scalar", "sine_basis_3x3"}, "unsupported parameter representation")
        require((self.kind == "scalar" and self.representation == "scalar" and len(self.component_ids) == 1)
                or (self.kind == "field" and self.representation == "sine_basis_3x3" and len(self.component_ids) == 9), "block representation/size mismatch")
        require(self.component_type.rank == 0, "V1 coefficients must be scalars")
        require(len(self.known_values) == len(self.component_ids), "known-value component count mismatch")
        require(all(x is None or math.isfinite(x) for x in self.known_values), "nonfinite known value")
        known = sum(x is not None for x in self.known_values)
        require(isinstance(self.status, Status), "invalid block status")
        if self.status == Status.KNOWN:
            require(known == len(self.component_ids) and self.prior is None, "known block needs all values and no inference prior")
        elif self.status == Status.PARTIAL:
            require(0 < known < len(self.component_ids) and self.prior is not None, "partial block needs both known and unknown components and a prior")
        elif self.status == Status.UNKNOWN:
            require(known == 0 and self.prior is not None, "unknown is not zero: use null values and an explicit prior")
        else:
            require(known == 0 and self.prior is None, "absent block has no values or prior")
        if self.prior:
            self.prior.validate(len(self.component_ids))
        require(self.coefficient_map in {"identity", "poisson_source_lambda_times_a"}, "unsupported coefficient dependency map")
        require(self.transform in {"identity", "bounded_logistic"}, "unsupported parameter transform")
        if self.prior and self.prior.family == "bounded_uniform":
            require(self.transform == "bounded_logistic", "bounded unknown scalar requires a bounded transform declaration")


@dataclass(frozen=True)
class BoundaryCondition:
    id: str
    target: str
    kind: str
    parts: tuple[str, ...]
    status: Status
    value: float | None


@dataclass(frozen=True)
class InitialCondition:
    id: str
    target: str
    physical_time: float
    derivative_order: int
    block_id: str


@dataclass(frozen=True)
class ObservationOperator:
    id: str
    quantity_id: int
    target: str
    kind: str = "scalar_point_evaluation"
    noise: str = "independent_gaussian_known_positive_sigma"


@dataclass(frozen=True)
class SolverDeclaration:
    key: str
    version: str
    implemented: bool
    solution_concepts: tuple[str, ...]


# Reference implementations live in reference.py, gated to canonical T1 specs.
SOLVER_REGISTRY = (
    SolverDeclaration("v1.analytical.poisson", "1", True, ("classical", "weak")),
    SolverDeclaration("v1.analytical.heat", "1", True, ("classical", "weak")),
    SolverDeclaration("v1.analytical.wave", "1", True, ("classical", "weak")),
)


@dataclass(frozen=True)
class ProblemSpec:
    schema_version: str
    name: str
    geometry: Geometry
    physical_time: PhysicalTime | None
    bindings: tuple[Binding, ...]
    graph: EquationGraph
    parameter_blocks: tuple[ParameterBlock, ...]
    boundaries: tuple[BoundaryCondition, ...]
    initials: tuple[InitialCondition, ...]
    observations: tuple[ObservationOperator, ...]
    solution_concepts: tuple[str, ...]
    solver_key: str

    def __post_init__(self):
        require_immutable(self)
        # These collections are semantic sets. Component/operand/coordinate
        # orders remain untouched because they carry mathematical meaning.
        for attr in ("bindings", "parameter_blocks", "boundaries", "initials", "observations"):
            object.__setattr__(self, attr, tuple(sorted(getattr(self, attr), key=lambda x: x.id)))
        object.__setattr__(self, "solution_concepts", tuple(sorted(self.solution_concepts)))
        self.validate()

    def validate(self):
        require_immutable(self)
        require(self.schema_version == "1.0", "unsupported ProblemSpec schema version")
        require(bool(self.name), "problem name required")
        require(self.geometry.kind == "open_unit_square" and self.geometry.bounds == ((0., 1.), (0., 1.)), "only Omega=(0,1)^2 supported")
        require(self.geometry.boundary_parts == ("x0", "x1", "y0", "y1"), "invalid square boundary parts")
        TensorType((), self.geometry.coordinate_units)
        require(isinstance(self.bindings, tuple) and isinstance(self.parameter_blocks, tuple), "spec collections must be immutable tuples")
        bmap = {b.id: b for b in self.bindings}
        require(len(bmap) == len(self.bindings), "duplicate/conflicting field bindings")
        require(len({b.quantity_id for b in self.bindings}) == len(self.bindings), "duplicate/conflicting quantity bindings")
        pmap = {b.id: b for b in self.parameter_blocks}
        require(len(pmap) == len(self.parameter_blocks), "duplicate parameter block identity")
        require(not (bmap.keys() & pmap.keys()), "binding and block identities must be distinct")
        dependencies = {b.id: b.dependencies for b in (*self.bindings, *self.parameter_blocks)}
        for key, deps in dependencies.items():
            require(len(set(deps)) == len(deps) and all(d in dependencies for d in deps), f"{key}: missing/duplicate coefficient dependency")
        visited, visiting = set(), set()
        def visit(key):
            require(key not in visiting, "cyclic coefficient dependencies")
            if key not in visited:
                visiting.add(key)
                for dep in dependencies[key]:
                    visit(dep)
                visiting.remove(key)
                visited.add(key)
        for key in dependencies:
            visit(key)
        block_targets = set()
        for block in self.parameter_blocks:
            block.validate()
            require(block.quantity_binding in bmap, "parameter block refers to unbound quantity")
            require(block.quantity_binding not in block_targets, "multiple parameter blocks conflict on one quantity")
            block_targets.add(block.quantity_binding)
            target = bmap[block.quantity_binding]
            require(target.type == block.component_type and target.status == block.status, "block/quantity type or status mismatch")
            require(target.role != "state", "solution state is derived; parameter blocks describe physical inputs")
        for b in self.bindings:
            if b.role in {"source", "initial"} and b.status != Status.ABSENT:
                require(b.id in block_targets, f"{b.id}: input field requires a parameter block")
            if b.role == "coefficient" and b.status in {Status.UNKNOWN, Status.PARTIAL}:
                require(b.id in block_targets, "unknown coefficient needs a parameter block/prior")
        time_units = (0, 0, 0)
        if self.physical_time is not None:
            lo, hi = self.physical_time.interval
            require(math.isfinite(lo) and math.isfinite(hi) and lo == 0 and hi == 1, "V1 physical time must be [0,1]")
            TensorType((), self.physical_time.units)
            time_units = self.physical_time.units
        else:
            require(not self.initials, "stationary equation has no initial-time axis or initial conditions")
            require(not any(b.role == "initial" and b.status != Status.ABSENT for b in self.bindings), "stationary equation cannot declare initial fields")
        self.graph.validate(self.bindings, physical_time=self.physical_time is not None,
                            space_units=self.geometry.coordinate_units, time_units=time_units)
        states = {b.id for b in self.bindings if b.role == "state" and b.status != Status.ABSENT}
        require(bool(states), "need a solution state binding")
        for state in states:
            require(any(n.binding == state for n in self.graph.nodes), "state not bound in equation graph")
        require(len({b.id for b in self.boundaries}) == len(self.boundaries), "duplicate boundary identity")
        covered = set()
        for bc in self.boundaries:
            require(bc.target in states, "boundary target must be a declared state")
            require(bc.kind == "dirichlet" and bc.status == Status.KNOWN and bc.value == 0., "V1 supports known homogeneous Dirichlet boundaries only")
            require(bool(bc.parts) and len(set(bc.parts)) == len(bc.parts) and set(bc.parts) <= set(self.geometry.boundary_parts), "invalid boundary geometry association")
            for part in bc.parts:
                require((bc.target, part) not in covered, "duplicate/conflicting boundary constraints")
                covered.add((bc.target, part))
        require(covered == {(s, p) for s in states for p in self.geometry.boundary_parts}, "all state fields require complete Dirichlet coverage")
        require(len({x.id for x in self.initials}) == len(self.initials), "duplicate initial identity")
        targets = set()
        for ic in self.initials:
            require(self.physical_time is not None and ic.physical_time == self.physical_time.interval[0], "initial condition needs the declared physical initial time")
            require(ic.target in states and ic.derivative_order in (0, 1), "invalid initial target/order")
            require((ic.target, ic.derivative_order) not in targets, "duplicate/conflicting initial constraint")
            targets.add((ic.target, ic.derivative_order))
            require(ic.block_id in pmap and bmap[pmap[ic.block_id].quantity_binding].role == "initial", "initial condition must reference initial field parameter block")
            require(pmap[ic.block_id].component_type == bmap[ic.target].type, "initial field type mismatch")
        if self.physical_time is not None:
            # Match the order actually declared by the graph, not a PDE label.
            for node in self.graph.nodes:
                if node.op.value in {"TIME_D1", "TIME_D2"}:
                    operand = next(n for n in self.graph.nodes if n.id == node.operands[0])
                    require(operand.binding in states, "V1 time derivatives must act on state terminals")
                    for order in range(node.derivative[1]):
                        require((operand.binding, order) in targets, "missing initial condition for time derivative order")
        require(self.observations and len({o.id for o in self.observations}) == len(self.observations), "unique observation declarations required")
        require(len({o.quantity_id for o in self.observations}) == len(self.observations), "duplicate observation quantity ID")
        for obs in self.observations:
            require(type(obs.quantity_id) is int and obs.quantity_id >= 0 and obs.target in states and bmap[obs.target].type.rank == 0, "invalid scalar measurement quantity")
            require(obs.kind == "scalar_point_evaluation" and obs.noise == "independent_gaussian_known_positive_sigma", "unsupported observation semantics")
        registry = {x.key: x for x in SOLVER_REGISTRY}
        require(self.solver_key in registry, "unregistered physical solver identity")
        require(self.solution_concepts and len(set(self.solution_concepts)) == len(self.solution_concepts)
                and set(self.solution_concepts) <= set(registry[self.solver_key].solution_concepts), "unsupported solution concept")
        expected_key = {"poisson": "v1.analytical.poisson", "heat": "v1.analytical.heat", "wave": "v1.analytical.wave"}.get(self.name)
        if expected_key:
            require(self.solver_key == expected_key, "solver identity conflicts with benchmark family")

    def to_json(self) -> str:
        return canonical_json(self)

    @classmethod
    def from_json(cls, text: str) -> ProblemSpec:
        try:
            def unique_object(pairs):
                result = {}
                for key, value in pairs:
                    require(key not in result, f"duplicate JSON key {key}")
                    result[key] = value
                return result
            raw = json.loads(text, object_pairs_hook=unique_object)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid ProblemSpec JSON: {error}") from error
        return decode(cls, raw)

    @property
    def identity(self) -> str:
        return digest(self)

    def information(self) -> dict:
        return {"kappa": [{"id": b.id, "quantity": b.quantity_id, "value": b.value,
                            "dependencies": b.dependencies} for b in self.bindings if b.status == Status.KNOWN],
                "known_components": [{"block": b.id, "components": tuple((key, value) for key, value in zip(b.component_ids, b.known_values) if value is not None)}
                                     for b in self.parameter_blocks if any(x is not None for x in b.known_values)],
                "xi": [{"block": b.id, "quantity": b.quantity_binding, "status": b.status.value,
                         "components": len(b.component_ids), "coefficient_map": b.coefficient_map,
                         "active_components": tuple(key for key, value in zip(b.component_ids, b.known_values) if value is None),
                         "known_values": b.known_values, "prior": b.prior} for b in self.parameter_blocks
                       if b.status in {Status.UNKNOWN, Status.PARTIAL}],
                "absent_blocks": tuple(b.id for b in self.parameter_blocks if b.status == Status.ABSENT),
                "physical_time": self.physical_time,
                "solver": next(s for s in SOLVER_REGISTRY if s.key == self.solver_key)}
