"""Typed symbolic expression DAG, deliberately unrelated to spatial graphs."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from .core import Status, TensorType, ValidationError, digest, require, unit_add


class Op(str, Enum):
    FIELD = "FIELD"
    PARAMETER = "PARAMETER"
    CONSTANT = "CONSTANT"
    GRAD = "GRAD"
    DIV = "DIV"
    LAPLACIAN = "LAPLACIAN"
    TIME_D1 = "TIME_D1"
    TIME_D2 = "TIME_D2"
    ADD = "ADD"
    NEG = "NEG"
    MULTIPLY = "MULTIPLY"
    DOT = "DOT"
    POWER = "POWER"
    EQUATION = "EQUATION"


@dataclass(frozen=True)
class Binding:
    id: str
    quantity_id: str
    type: TensorType
    status: Status
    role: str  # state, source, initial, coefficient
    value: float | None = None
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        require(bool(self.id) and bool(self.quantity_id), "bindings need unique identities")
        require(isinstance(self.status, Status), "invalid information status")
        require(self.role in {"state", "source", "initial", "coefficient"}, "unsupported binding role")
        if self.value is not None:
            import math
            require(type(self.value) in (int, float) and math.isfinite(self.value), "binding value must be finite")
        require(self.status == Status.KNOWN or self.value is None, "unknown/partial/absent binding cannot masquerade as a known scalar")
        if self.role == "coefficient" and self.status == Status.KNOWN:
            require(self.value is not None, "known scalar coefficient needs an explicit value (zero is valid)")


@dataclass(frozen=True)
class Node:
    id: str
    op: Op
    operands: tuple[str, ...]
    type: TensorType
    quantity_id: str | None = None
    binding: str | None = None
    status: Status | None = None
    values: tuple[float, ...] = ()
    # Local derivative annotation; full derivative ancestry remains in operands.
    derivative: tuple[str, int] | None = None

    def identity(self):
        return {k: v for k, v in self.__dict__.items() if k != "id"}


@dataclass(frozen=True)
class EquationGraph:
    nodes: tuple[Node, ...]
    roots: tuple[str, ...]

    def edges(self):
        return tuple((src, node.id, i) for node in self.nodes for i, src in enumerate(node.operands))

    def validate(self, bindings: tuple[Binding, ...], *, physical_time: bool,
                 space_units=(0, 0, 0), time_units=(0, 0, 0)):
        builder = GraphBuilder(bindings, physical_time=physical_time,
                               space_units=space_units, time_units=time_units)
        require(len({n.id for n in self.nodes}) == len(self.nodes), "duplicate node identity")
        pending = {n.id: n for n in self.nodes}
        while pending:
            ready = sorted(k for k, n in pending.items() if all(x in builder.nodes for x in n.operands))
            require(bool(ready), "cycle or missing operand in equation DAG")
            for key in ready:
                node = pending.pop(key)
                expected = builder.make(node.op, *node.operands, binding=node.binding,
                                        values=node.values, constant_type=node.type if node.op == Op.CONSTANT else None)
                require(builder.nodes[expected] == node, f"node {key}: type, derivative, status, identity or canonicalization mismatch")
        require(self.roots and len(set(self.roots)) == len(self.roots), "need unique equation roots")
        for root in self.roots:
            require(root in builder.nodes and builder.nodes[root].op == Op.EQUATION, "roots must bind EQUATION nodes")
        require(self == builder.finish(*self.roots), "graph must be canonical, reachable and sorted")


class GraphBuilder:
    """Safe constructor; no parsing or evaluation of user-provided code."""
    def __init__(self, bindings=(), *, physical_time=False,
                 space_units=(0, 0, 0), time_units=(0, 0, 0)):
        require(len({x.id for x in bindings}) == len(bindings), "duplicate/conflicting field binding")
        require(len({x.quantity_id for x in bindings}) == len(bindings), "duplicate/conflicting quantity binding")
        self.bindings = {x.id: x for x in bindings}
        self.nodes: dict[str, Node] = {}
        self.physical_time = physical_time
        self.space_units, self.time_units = space_units, time_units

    def make(self, op, *operands, binding=None, values=(), constant_type=None):
        try:
            op = Op(op)
        except ValueError as error:
            raise ValidationError(f"unsupported operator {op!r}") from error
        require(all(x in self.nodes for x in operands), "operand is not bound in this graph")
        arity = {Op.FIELD: 0, Op.PARAMETER: 0, Op.CONSTANT: 0, Op.GRAD: 1,
                 Op.DIV: 1, Op.LAPLACIAN: 1, Op.TIME_D1: 1, Op.TIME_D2: 1,
                 Op.NEG: 1, Op.ADD: 2, Op.MULTIPLY: 2, Op.DOT: 2, Op.POWER: 2, Op.EQUATION: 1}
        require(len(operands) == arity[op], f"{op.value}: expected {arity[op]} operands")
        require(binding is None or op in {Op.FIELD, Op.PARAMETER}, "only terminals have bindings")
        require(not values or op == Op.CONSTANT, "only CONSTANT has literal values")
        require(constant_type is None or op == Op.CONSTANT, "only CONSTANT accepts a literal type")
        if op in {Op.ADD, Op.MULTIPLY, Op.DOT}:
            operands = tuple(sorted(operands))
        args = [self.nodes[x] for x in operands]
        quantity, status, derivative = None, None, None
        if op in {Op.FIELD, Op.PARAMETER}:
            require(binding in self.bindings, f"unbound physical quantity {binding!r}")
            b = self.bindings[binding]
            require(b.status != Status.ABSENT, "absent quantity cannot enter equation")
            require((op == Op.PARAMETER) == (b.role == "coefficient"), "FIELD/PARAMETER binding role mismatch")
            typ, quantity, status = b.type, b.quantity_id, b.status
        elif op == Op.CONSTANT:
            import math
            from math import prod
            typ = constant_type or TensorType()
            require(len(values) == prod(typ.shape), "CONSTANT literal count disagrees with shape")
            require(all(type(x) in (int, float) and math.isfinite(x) for x in values), "nonfinite CONSTANT")
            values = tuple(float(x) for x in values)
        elif op in {Op.ADD, Op.EQUATION, Op.NEG, Op.LAPLACIAN, Op.GRAD, Op.DIV, Op.TIME_D1, Op.TIME_D2}:
            typ = args[0].type
            quantity = args[0].quantity_id if op not in {Op.EQUATION, Op.ADD} else None
            if op == Op.ADD:
                require(typ == args[1].type, "ADD: incompatible tensor types or dimensions")
            elif op == Op.EQUATION:
                require(typ.rank == 0, "EQUATION residual must be scalar")
                require(args[0].op != Op.EQUATION, "nested EQUATION is invalid")
            elif op == Op.NEG and args[0].op == Op.NEG:
                return args[0].operands[0]
            elif op in {Op.GRAD, Op.DIV, Op.LAPLACIAN}:
                order = 2 if op == Op.LAPLACIAN else 1
                shape = typ.shape
                if op == Op.GRAD:
                    require(typ.rank <= 1, "GRAD supports scalar/vector only")
                    shape = shape + (2,)
                if op == Op.DIV:
                    require(typ.rank in (1, 2) and typ.shape[-1] == 2, "DIV requires a spatial vector/tensor")
                    shape = shape[:-1]
                    # Exact identity only; never expand variable coefficients.
                    if args[0].op == Op.GRAD:
                        return self.make(Op.LAPLACIAN, args[0].operands[0])
                    if args[0].op == Op.MULTIPLY:
                        factors = [self.nodes[x] for x in args[0].operands]
                        for identity, gradient in (factors, factors[::-1]):
                            if (identity.op == Op.CONSTANT
                                    and ((identity.type == TensorType() and identity.values == (1.0,))
                                         or (identity.type == TensorType((2, 2)) and identity.values == (1., 0., 0., 1.)))
                                    and gradient.op == Op.GRAD):
                                return self.make(Op.LAPLACIAN, gradient.operands[0])
                typ = TensorType(shape, unit_add(typ.units, self.space_units, -order))
                derivative = ("space", order)
            elif op in {Op.TIME_D1, Op.TIME_D2}:
                require(self.physical_time, "physical-time derivative illegal in stationary problem")
                order = 1 if op == Op.TIME_D1 else 2
                typ = TensorType(typ.shape, unit_add(typ.units, self.time_units, -order))
                derivative = ("physical_time", order)
        elif op == Op.MULTIPLY:
            a, b = (n.type for n in args)
            identity_vector = any(i.op == Op.CONSTANT and i.type == TensorType((2, 2))
                                  and i.values == (1., 0., 0., 1.) and v.type.shape == (2,)
                                  for i, v in (args, args[::-1]))
            require(a.rank == 0 or b.rank == 0 or identity_vector,
                    "MULTIPLY requires a scalar or explicit constant identity acting on a vector; use DOT for contraction")
            typ = TensorType((2,) if identity_vector else (a.shape or b.shape), unit_add(a.units, b.units))
        elif op == Op.DOT:
            a, b = (n.type for n in args)
            require(a.shape == b.shape and a.rank == 1, "DOT requires equal vectors")
            typ = TensorType((), unit_add(a.units, b.units))
        elif op == Op.POWER:
            a, exponent = args
            require(a.type.rank == 0 and exponent.op == Op.CONSTANT and exponent.type == TensorType(), "POWER requires scalar base and literal dimensionless exponent")
            p = exponent.values[0]
            require(p.is_integer(), "only integer POWER exponents supported in V1")
            typ = TensorType((), tuple(int(p) * x for x in a.type.units))
        require(not any(n.op == Op.EQUATION for n in args), "EQUATION cannot be an expression operand")
        node = Node("", op, tuple(operands), typ, quantity, binding, status, tuple(values), derivative)
        node = Node(digest(node.identity()), op, tuple(operands), typ, quantity, binding, status, tuple(values), derivative)
        self.nodes[node.id] = node
        return node.id

    def finish(self, *roots):
        reachable = set()
        def visit(key):
            require(key in self.nodes, "unknown graph root")
            if key not in reachable:
                reachable.add(key)
                for child in self.nodes[key].operands:
                    visit(child)
        for root in roots:
            visit(root)
        # IDs give stable storage order; operand_index retains noncommutative order.
        return EquationGraph(tuple(self.nodes[k] for k in sorted(reachable)), tuple(sorted(roots)))


def graph_table(graph: EquationGraph) -> list[dict]:
    return [{"id": n.id[:10], "op": n.op.value, "quantity": n.quantity_id,
             "shape": n.type.shape, "units": n.type.units,
             "status": n.status.value if n.status else None, "derivative": n.derivative,
             "operands": [x[:10] for x in n.operands]} for n in graph.nodes]
