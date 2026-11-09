"""Three T1 graph definitions and prior declarations, with no physics evaluation."""
from __future__ import annotations

import math
from .core import Status, TensorType
from .graph import Binding, GraphBuilder, Op
from .spec import (BoundaryCondition, Geometry, InitialCondition, ObservationOperator,
                   ParameterBlock, PhysicalTime, Prior, ProblemSpec)

MODE_IDS = tuple(f"{m},{n}" for m in range(1, 4) for n in range(1, 4))
MODE_EIGENVALUES = tuple(math.pi**2 * (m*m + n*n) for m in range(1, 4) for n in range(1, 4))
TAU = tuple(.5 * (MODE_EIGENVALUES[0] / x)**.75 for x in MODE_EIGENVALUES)


def benchmark(name: str) -> ProblemSpec:
    if name not in {"poisson", "heat", "wave"}:
        raise ValueError(f"unsupported benchmark {name!r}")
    state = Binding("u", "solution.u", TensorType(), Status.UNKNOWN, "state")
    bindings = [state]
    field = "f" if name == "poisson" else "u0"
    bindings.append(Binding(field, f"input.{field}", TensorType(), Status.UNKNOWN, "source" if name == "poisson" else "initial", dependencies=("a",)))
    blocks = [ParameterBlock("a", field, "field", "sine_basis_3x3", tuple(f"a[{j}]" for j in MODE_IDS),
                             TensorType(), Status.UNKNOWN, (None,)*9,
                             Prior("independent_gaussian", (0.,)*9, TAU),
                             coefficient_map="poisson_source_lambda_times_a" if name == "poisson" else "identity")]
    if name == "heat":
        bindings.append(Binding("alpha", "physics.diffusivity", TensorType(), Status.KNOWN, "coefficient", .02))
    if name == "wave":
        bindings.extend((Binding("v", "solution.v", TensorType(), Status.UNKNOWN, "state"),
                         Binding("v0", "input.v0", TensorType(), Status.UNKNOWN, "initial", dependencies=("b",)),
                         Binding("c", "physics.speed", TensorType(), Status.KNOWN, "coefficient", .7)))
        b_scales = tuple(.4 * .7 * math.sqrt(lam) * tau for lam, tau in zip(MODE_EIGENVALUES, TAU))
        blocks.append(ParameterBlock("b", "v0", "field", "sine_basis_3x3", tuple(f"b[{j}]" for j in MODE_IDS),
                                     TensorType(), Status.UNKNOWN, (None,)*9,
                                     Prior("independent_gaussian", (0.,)*9, b_scales), dependencies=("c",)))
    g = GraphBuilder(tuple(bindings), physical_time=name != "poisson")
    u = g.make(Op.FIELD, binding="u")
    lap = g.make(Op.LAPLACIAN, u)
    if name == "poisson":
        residual = g.make(Op.ADD, g.make(Op.NEG, lap), g.make(Op.NEG, g.make(Op.FIELD, binding="f")))
        roots = [g.make(Op.EQUATION, residual)]
    elif name == "heat":
        alpha = g.make(Op.PARAMETER, binding="alpha")
        residual = g.make(Op.ADD, g.make(Op.TIME_D1, u), g.make(Op.NEG, g.make(Op.MULTIPLY, alpha, lap)))
        roots = [g.make(Op.EQUATION, residual)]
    else:
        v = g.make(Op.FIELD, binding="v")
        c = g.make(Op.PARAMETER, binding="c")
        c2 = g.make(Op.POWER, c, g.make(Op.CONSTANT, values=(2.,)))
        r1 = g.make(Op.ADD, g.make(Op.TIME_D1, u), g.make(Op.NEG, v))
        r2 = g.make(Op.ADD, g.make(Op.TIME_D1, v), g.make(Op.NEG, g.make(Op.MULTIPLY, c2, lap)))
        roots = [g.make(Op.EQUATION, r1), g.make(Op.EQUATION, r2)]
    states = ("u", "v") if name == "wave" else ("u",)
    boundaries = tuple(BoundaryCondition(f"bc.{s}", s, "dirichlet", ("x0", "x1", "y0", "y1"), Status.KNOWN, 0.) for s in states)
    initials = () if name == "poisson" else (InitialCondition("ic.u", "u", 0., 0, "a"),)
    if name == "wave":
        initials += (InitialCondition("ic.v", "v", 0., 0, "b"),)
    return ProblemSpec("1.0", name, Geometry(), None if name == "poisson" else PhysicalTime(),
                       tuple(sorted(bindings, key=lambda b: b.id)), g.finish(*roots), tuple(blocks), boundaries, initials,
                       tuple(ObservationOperator(f"point.{s}", i, s) for i, s in enumerate(states)),
                       ("classical", "weak"), f"v1.analytical.{name}")


EQUATIONS = {
    "poisson": r"-\Delta u-f=0,\quad u|_{\partial\Omega}=0",
    "heat": r"\partial_tu-\alpha\Delta u=0,\quad\alpha=0.02,\quad u(0)=u_0,\quad u|_{\partial\Omega}=0",
    "wave": r"\partial_tu-v=0,\quad\partial_tv-c^2\Delta u=0,\quad c=0.7\quad(\Leftrightarrow\partial_{tt}u-c^2\Delta u=0)",
}
