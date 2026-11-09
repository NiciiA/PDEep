from dataclasses import replace
import pytest

from pdeep.benchmarks import benchmark
from pdeep.core import Status, TensorType, ValidationError
from pdeep.graph import Binding, EquationGraph, GraphBuilder, Op


def builder(time=False, units=(0, 0, 0)):
    u = Binding("u", "u", TensorType((), units), Status.UNKNOWN, "state")
    g = GraphBuilder((u,), physical_time=time, space_units=(1, 0, 0), time_units=(0, 1, 0))
    return g, g.make(Op.FIELD, binding="u")


@pytest.mark.parametrize("name,nodes,equations", [("poisson", 7, 1), ("heat", 8, 1), ("wave", 15, 2)])
def test_graphs(name, nodes, equations):
    spec = benchmark(name)
    assert len(spec.graph.nodes) == nodes
    assert len(spec.graph.roots) == equations
    spec.validate()
    assert len(spec.graph.edges()) == sum(len(n.operands) for n in spec.graph.nodes)
    if name == "poisson":
        assert not any(n.derivative and n.derivative[0] == "physical_time" for n in spec.graph.nodes)
    if name == "wave":
        assert sum(n.op == Op.TIME_D1 for n in spec.graph.nodes) == 2
        assert {next(x.binding for x in spec.graph.nodes if x.id == n.operands[0])
                for n in spec.graph.nodes if n.op == Op.TIME_D1} == {"u", "v"}


def test_operator_signatures_and_units():
    g, u = builder(True, (0, 0, 1))
    grad = g.make(Op.GRAD, u)
    assert g.nodes[grad].type == TensorType((2,), (-1, 0, 1))
    assert g.nodes[g.make(Op.DOT, grad, grad)].type == TensorType((), (-2, 0, 2))
    assert g.nodes[g.make(Op.TIME_D2, u)].type.units == (0, -2, 1)
    assert g.nodes[g.make(Op.TIME_D1, u)].derivative == ("physical_time", 1)
    assert g.nodes[g.make(Op.LAPLACIAN, u)].type.units == (-2, 0, 1)
    exponent = g.make(Op.CONSTANT, values=(2.,))
    assert g.nodes[g.make(Op.POWER, u, exponent)].type.units == (0, 0, 2)


@pytest.mark.parametrize("op", ["EXEC", "SIN", "__import__('os')"])
def test_unsupported_operator(op):
    g, _ = builder()
    with pytest.raises(ValidationError, match="unsupported operator"):
        g.make(op)


@pytest.mark.parametrize("op", [Op.TIME_D1, Op.TIME_D2])
def test_stationary_time_illegal(op):
    g, u = builder()
    with pytest.raises(ValidationError, match="stationary"):
        g.make(op, u)


@pytest.mark.parametrize("op", [Op.DIV, Op.DOT, Op.ADD, Op.NEG, Op.MULTIPLY, Op.POWER])
def test_wrong_arity(op):
    g, _ = builder()
    with pytest.raises(ValidationError, match="operands"):
        g.make(op)


def test_invalid_shapes_units_and_values():
    g, u = builder()
    grad = g.make(Op.GRAD, u)
    for op, operands in [(Op.ADD, (u, grad)), (Op.ADD, (u, g.make(Op.LAPLACIAN, u))),
                         (Op.DOT, (u, u)), (Op.MULTIPLY, (grad, grad)),
                         (Op.DIV, (u,)), (Op.EQUATION, (grad,))]:
        with pytest.raises(ValidationError):
            g.make(op, *operands)
    with pytest.raises(ValidationError, match="nonfinite"):
        g.make(Op.CONSTANT, values=(float("nan"),))
    with pytest.raises(ValidationError, match="integer"):
        g.make(Op.POWER, u, g.make(Op.CONSTANT, values=(.5,)))
    with pytest.raises(ValidationError, match="bound"):
        g.make(Op.FIELD, binding="missing")
    with pytest.raises(ValidationError, match="role"):
        g.make(Op.PARAMETER, binding="u")


def test_conflicting_binding_and_absent_field():
    b = Binding("u", "u", TensorType(), Status.UNKNOWN, "state")
    for bindings in [(b, b), (b, replace(b, id="different"))]:
        with pytest.raises(ValidationError, match="binding"):
            GraphBuilder(bindings)
    g = GraphBuilder((replace(b, status=Status.ABSENT),))
    with pytest.raises(ValidationError, match="absent"):
        g.make(Op.FIELD, binding="u")


def test_canonicalization_and_ordered_power():
    g, u = builder()
    one = g.make(Op.CONSTANT, values=(1.,))
    two = g.make(Op.CONSTANT, values=(2.,))
    assert g.make(Op.ADD, one, two) == g.make(Op.ADD, two, one)
    assert g.make(Op.MULTIPLY, one, two) == g.make(Op.MULTIPLY, two, one)
    assert g.make(Op.NEG, g.make(Op.NEG, u)) == u
    assert g.make(Op.POWER, one, two) != g.make(Op.POWER, two, one)
    lap = g.make(Op.LAPLACIAN, u)
    grad = g.make(Op.GRAD, u)
    identity = g.make(Op.CONSTANT, values=(1., 0., 0., 1.), constant_type=TensorType((2, 2)))
    assert g.make(Op.DIV, grad) == lap
    assert g.make(Op.DIV, g.make(Op.MULTIPLY, identity, grad)) == lap
    variable = g.make(Op.MULTIPLY, u, grad)
    assert g.nodes[g.make(Op.DIV, variable)].op == Op.DIV


def test_forged_graph_and_dangling_edge():
    spec = benchmark("poisson")
    n = spec.graph.nodes[0]
    for node in [replace(n, type=TensorType((2,))), replace(n, id="forged"), replace(n, operands=("missing",))]:
        graph = replace(spec.graph, nodes=(node, *spec.graph.nodes[1:]))
        with pytest.raises(ValidationError):
            replace(spec, graph=graph)


def test_equations_are_terminal_roots():
    g, u = builder()
    eq = g.make(Op.EQUATION, u)
    with pytest.raises(ValidationError, match="EQUATION"):
        g.make(Op.NEG, eq)
