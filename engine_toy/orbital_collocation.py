"""Orbital craft, build step 4: the collocation planner.

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``
(decision 4, ON PLAN: the planner holds the whole remaining trip from the
present state; decision 7, the planner's cost).  Differentiation:
``turing/docs/DIFFERENTIATION_FEASIBILITY_2026-10-02.md`` item 1 (the slice
Jacobian, graph-native) and item 4 (its block-bidiagonal sparsity).

Transcription (decision variables).  The trip from the present state x_0 at
t_0 to the target circle is N slices.  Slice k holds its throttles u_k (one
per declared thruster, inside the declared box) and its duration dt_k; the
states x_k = (position_k, momentum_k) at nodes 1..N are free.  x_0 is the
present state and is not a variable, so every re-plan is the total trip
from the current moment over a shrinking horizon (sum dt_k is free).

Residuals (dynamics defects).  Slice k's defect is
``x_{k+1} - step(x_k, u_k, dt_k)`` where ``step`` is the jumper's own
symplectic-Euler round (``orbital_jumper_dt_pieces``): actuation eq_TS1_2
(``orbital_actuation``) -> gravity eq_N4_1 (``orbital_jumper``) -> momentum
eq_N1_2 -> position eq_N1_1, composed by substitution exactly as
``tools/compiler_probes/probe_collocation_jacobian.py slice_laws`` composes
them, with the design and centers as input columns (one compile serves every
design with the same thruster and center counts).  Arrival at node N:
radius = r_target, vis-viva eq_KE1_3 (``orbital_plan``, from the original
``Orbit.vis_viva``) at a = r_target, zero radial velocity, and the present
orbit plane (two rows).

Cost (decision 7; plan deviation is NOT here -- it is zero by construction):

    I = sum_k dt_k * sum_j thrust_j(u_k)        thrust impulse (eq_TS1_2)
    T = sum_k dt_k                              time to arrive
    J = alpha * I / T + beta * T - kappa * log(1 - I / fuel_budget)

the last term being the run-out-of-fuel barrier, infinite at an empty tank.

Jacobian.  Every row (6 slice defects, 5 arrival rows, the cost) is a sympy
law ingested by ``symbolic_process_graph.ingest_sympy_expressions`` and
differentiated by ``process_graph_autograd`` (graph-native reverse), lowered
and compiled to native code.  :func:`compile_reverse_rows` is the ONE place
that decides how: today one unit-seed motion per row (the explicit-seed fused
motion is not yet green); one slice row artifact is reused for every slice.
Nothing here differentiates numerically or symbolically.

Provenance (decision 5): the cost's impulse is the original set's
``force_cost_integral`` with arc length as time, as the jumper's thrust-cost
piece spells it per thruster (``src/transmogrifier/orbital_transfer.py``).

Public surface:

    CollocationProblem      frozen: craft, centers, target, cost weights, N
    CollocationPlan         the planned trip (nodes, throttles, cost receipts)
    plan_transfer(problem, t0, position, velocity) -> CollocationPlan
    hohmann_warm_start(problem, t0, position, velocity) -> (plan, z0)
    reference(plan, t)      (r, v) the plan prescribes at t
    compile_reverse_rows(name, expressions, wrt) -> evaluators (the Jacobian)
"""
from __future__ import annotations

import functools
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import sympy as sp

import honorary_engine_equation_catalogue as honorary
import orbital_jumper as oj
from orbital_actuation import (AXES, CraftDesign, actuation_force_rhs,
                               actuation_matrix, thrust_magnitude,
                               thruster_columns, thruster_symbols)
from orbital_jumper import GravityCenter
from orbital_plan import KEPLER_LAWS, hohmann_plan
from orbital_plan import reference as hohmann_reference
from orbital_tracker import allocate_throttles

# ---------------------------------------------------------------- the laws
_DT, _MASS = sp.symbols("dt mass")
_POSITION = {a: sp.Symbol(f"position_{a}") for a in AXES}
_MOMENTUM = {a: sp.Symbol(f"momentum_{a}") for a in AXES}
_POSITION_NEXT = {a: sp.Symbol(f"position_{a}_next") for a in AXES}
_MOMENTUM_NEXT = {a: sp.Symbol(f"momentum_{a}_next") for a in AXES}
_TARGET_RADIUS = sp.Symbol("target_radius")
_NORMAL = {a: sp.Symbol(f"plane_normal_{a}") for a in AXES}
_ALPHA, _BETA, _KAPPA, _BUDGET = sp.symbols(
    "cost_alpha cost_beta cost_kappa fuel_budget")


def slice_defect_laws(center_count: int, thruster_count: int):
    """The six defects ``x_next - step(x, u, dt)`` of one slice, spelled on
    the jumper's columns (``*_next`` for node k+1); momentum rows first."""
    force = {a: oj.gravity_force_rhs(a, center_count)
             + actuation_force_rhs(a, thruster_count) for a in AXES}
    p_next = {a: _MOMENTUM[a] + _DT * honorary.eq_N1_2.rhs.xreplace(
        {honorary.F_i(honorary.t): force[a]}) for a in AXES}
    r_next = {a: _POSITION[a] + _DT * honorary.eq_N1_1.rhs.xreplace(
        {honorary.m_i(honorary.t): _MASS, honorary.p_i(honorary.t): p_next[a]})
        for a in AXES}
    return ([_MOMENTUM_NEXT[a] - p_next[a] for a in AXES]
            + [_POSITION_NEXT[a] - r_next[a] for a in AXES])


def arrival_laws():
    """Five rows that vanish on the target circle about center 0, in the
    plane whose unit normal is ``plane_normal_*``."""
    center, mu = oj._center_symbols(0)
    rel = {a: _POSITION[a] - center[a] for a in AXES}
    radius = sp.sqrt(sum(rel[a]**2 for a in AXES))
    vis_viva = KEPLER_LAWS["eq_KE1_3"]
    speed = vis_viva.rhs.xreplace({sp.Symbol("mu", positive=True): mu,
                                   sp.Symbol("r", positive=True): radius,
                                   sp.Symbol("a", positive=True): _TARGET_RADIUS})
    if speed.has(sp.Symbol("mu", positive=True)):
        raise RuntimeError("eq_KE1_3 no longer spells mu, r, a as expected")
    speed_sq = sum(_MOMENTUM[a]**2 for a in AXES) / _MASS**2
    return [radius - _TARGET_RADIUS,
            speed_sq - speed**2,
            sum(rel[a] * _MOMENTUM[a] for a in AXES) / (_MASS * radius),
            sum(_NORMAL[a] * rel[a] for a in AXES),
            sum(_NORMAL[a] * _MOMENTUM[a] for a in AXES)]


def _slice_symbol(k: int, symbol: sp.Symbol) -> sp.Symbol:
    return sp.Symbol(f"slice{k}_{symbol.name}")


def trip_cost_law(slices: int, thruster_count: int):
    """Decision 7's cost over ``slices`` slices; per-slice throttles and
    durations are ``slice{k}_thruster{j}_throttle`` / ``slice{k}_dt``."""
    rate = sum((thrust_magnitude(j) for j in range(thruster_count)),
               sp.Integer(0))
    impulse, duration = sp.Integer(0), sp.Integer(0)
    for k in range(slices):
        rename = {thruster_symbols(j)["throttle"]:
                  _slice_symbol(k, thruster_symbols(j)["throttle"])
                  for j in range(thruster_count)}
        dt_k = _slice_symbol(k, _DT)
        impulse += dt_k * rate.xreplace(rename)
        duration += dt_k
    return (_ALPHA * impulse / duration + _BETA * duration
            - _KAPPA * sp.log(1 - impulse / _BUDGET))


# ------------------------------------------------------- the Jacobian source
@dataclass
class ReverseRow:
    """One compiled row: ``row(feeds) -> (value, {wrt name: d value})``."""

    name: str
    artifact: object
    input_ids: dict          # symbol name -> motion input value id
    loss_id: int
    gradient_ids: dict       # wrt symbol name -> output value id

    def __call__(self, feeds: dict):
        from src.compiler.ssa_llvm_backend import prepare_artifact_execution
        execution = prepare_artifact_execution(self.artifact, {
            vid: np.asarray(float(feeds[name]), dtype=np.float64)
            for name, vid in self.input_ids.items()}).run()
        buffers = execution.buffers
        value = float(np.asarray(buffers[self.loss_id]).reshape(-1)[0])
        return value, {name: float(np.asarray(buffers[vid]).reshape(-1)[0])
                       for name, vid in self.gradient_ids.items()}


class CompiledReverseUnavailable(RuntimeError):
    """The compiled graph reverse did not publish a requested gradient."""


def compile_reverse_rows(name: str, expressions: Sequence[sp.Expr],
                         wrt: Sequence[sp.Symbol]) -> tuple[ReverseRow, ...]:
    """THE Jacobian source: graph-native reverse of each expression, compiled.

    Route: ``ingest_sympy_expressions(strict)`` ->
    ``compile_process_graph_backward(packaging="combined",
    unit_loss_seed=True)`` (one motion per row, unit seed) ->
    ``lower_training_motion_to_repository_ssa`` -> LLVM -> native.  When the
    explicit-seed fused motion is green this becomes one motion over all
    rows with ``unit_loss_seed=False`` and a one-hot seed per row.
    """
    from src.compiler.identity_concordance import (begin_identity_book,
                                                   end_identity_book)

    directory = Path(tempfile.mkdtemp(prefix=f"colloc_{name}_"))
    rows = []
    for index, expression in enumerate(expressions):
        # one compile, one identity book: without it every row of the
        # process posts into the same detached book and the second compile
        # is refused as a concordance disagreement (observed)
        _book, token = begin_identity_book()
        try:
            rows.append(_compile_reverse_row(f"{name}_row{index}", expression,
                                             wrt, directory))
        finally:
            end_identity_book(token)
    return tuple(rows)


def _compile_reverse_row(row_name: str, expression: sp.Expr,
                         wrt: Sequence[sp.Symbol], directory: Path):
    from src.compiler.process_graph_autograd import (
        compile_process_graph_backward, lower_training_motion_to_repository_ssa)
    from src.compiler.ssa_llvm_backend import (compile_artifact,
                                               emit_ssa_function_to_llvm)
    from src.compiler.symbolic_process_graph import ingest_sympy_expressions
    from src.transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(materialize_memory=False)
    (root,) = ingest_sympy_expressions(graph, [expression],
                                       output_names=[row_name], strict=True)
    inputs = {}
    for node, data in graph.G.nodes(data=True):
        symbol = data.get("expr_obj")
        if isinstance(symbol, sp.Symbol) and data.get("op") == "input":
            inputs[symbol.name] = int(node)
    free = {s.name for s in expression.free_symbols}
    if set(inputs) != free:
        raise RuntimeError(f"{row_name}: input leaves {sorted(inputs)} "
                           f"!= free symbols {sorted(free)}")
    targets = [s.name for s in wrt if s.name in inputs]
    product = compile_process_graph_backward(
        graph, outputs=[root], wrt=[inputs[n] for n in targets],
        packaging="combined", unit_loss_seed=True)
    lowering = lower_training_motion_to_repository_ssa(
        product.motion, function_name=row_name)
    if lowering.shortfalls:
        raise CompiledReverseUnavailable(
            f"{row_name}: SSA shortfalls {lowering.shortfalls!r}")
    emitted = emit_ssa_function_to_llvm(lowering.module,
                                        lowering.function_name,
                                        entry_name=lowering.function_name)
    if emitted.shortfalls:
        raise CompiledReverseUnavailable(
            f"{row_name}: LLVM shortfalls " + "; ".join(
                f"{s.function}: {s.operation}: {s.reason}"
                for s in emitted.shortfalls[:4]))
    artifact = compile_artifact(emitted, directory=directory)
    published = set(map(int, artifact.buffer_order))
    gradient_ids = {n: int(lowering.outputs[f"grad_{inputs[n]}"])
                    for n in targets}
    missing = sorted(n for n, vid in gradient_ids.items()
                     if vid not in published)
    loss_id = int(lowering.outputs["loss_0"])
    if missing or loss_id not in published:
        raise CompiledReverseUnavailable(
            f"{row_name}: the compiled reverse does not publish the "
            f"gradients of {missing} (loss published: {loss_id in published}); "
            "see turing/docs/concordance_census/"
            "CONTINUATION_orbital_step4_collocation.md")
    return ReverseRow(row_name, artifact, inputs, loss_id, gradient_ids)


@functools.lru_cache(maxsize=None)
def _slice_rows(center_count: int, thruster_count: int):
    wrt = (list(_POSITION.values()) + list(_MOMENTUM.values())
           + [thruster_symbols(j)["throttle"] for j in range(thruster_count)]
           + [_DT] + list(_POSITION_NEXT.values())
           + list(_MOMENTUM_NEXT.values()))
    return compile_reverse_rows(f"slice_c{center_count}_t{thruster_count}",
                                slice_defect_laws(center_count, thruster_count),
                                wrt)


@functools.lru_cache(maxsize=None)
def _arrival_rows():
    wrt = list(_POSITION.values()) + list(_MOMENTUM.values())
    return compile_reverse_rows("arrival", arrival_laws(), wrt)


@functools.lru_cache(maxsize=None)
def _cost_row(slices: int, thruster_count: int):
    wrt = [_slice_symbol(k, s) for k in range(slices)
           for s in [thruster_symbols(j)["throttle"]
                     for j in range(thruster_count)] + [_DT]]
    (row,) = compile_reverse_rows(f"cost_n{slices}_t{thruster_count}",
                                  [trip_cost_law(slices, thruster_count)], wrt)
    return row


# ------------------------------------------------------------- the problem
@dataclass(frozen=True)
class CollocationProblem:
    """One planning request's fixed data.

    ``alpha``/``beta`` weigh the average thrust rate (N) and the time to
    arrive (s); ``None`` balances them at the Hohmann warm start (beta =
    alpha * I_h / T_h**2, so dJ/dT = 0 there).  ``kappa`` scales the fuel
    barrier; ``fuel_budget_n_s`` is the remaining thrust impulse.
    """

    design: CraftDesign
    centers: tuple[GravityCenter, ...]
    target_radius_m: float
    fuel_budget_n_s: float
    slices: int = 40
    alpha: float | None = None
    beta: float | None = None
    kappa: float = 1.0
    dt_bounds_s: tuple[float, float] = (0.05, 240.0)

    def __post_init__(self):
        object.__setattr__(self, "centers", tuple(self.centers))
        if self.slices < 3:
            raise ValueError("a transfer needs at least 3 slices")
        if not (self.target_radius_m > 0.0 and self.fuel_budget_n_s > 0.0):
            raise ValueError("target radius and fuel budget must be positive")


@dataclass(frozen=True)
class CollocationPlan:
    """A planned trip: node times (N+1), node states, per-slice throttles.

    ``ideal_delta_v`` is the plan's own thrust impulse over the craft mass,
    so a tracker's ``mass * ideal_delta_v`` is the impulse it planned."""

    mu: float
    mass_kg: float
    target_radius_m: float
    times: np.ndarray            # (N+1,)
    positions: np.ndarray        # (N+1, 3)
    velocities: np.ndarray       # (N+1, 3)
    throttles: np.ndarray        # (N, n_u)
    impulse_n_s: float
    cost: float
    max_defect: float
    converged: bool
    message: str
    columns: dict = field(repr=False, compare=False, default_factory=dict)
    center_count: int = 1

    @property
    def t_start(self) -> float:
        return float(self.times[0])

    @property
    def t_arrive(self) -> float:
        return float(self.times[-1])

    @property
    def transfer_time(self) -> float:
        return self.t_arrive - self.t_start

    @property
    def ideal_delta_v(self) -> float:
        return self.impulse_n_s / self.mass_kg


def _design_columns(problem: CollocationProblem) -> dict:
    columns = {name: float(value[0]) for name, value in
               thruster_columns(problem.design).items()}
    columns["mass"] = float(problem.design.mass_kg)
    for index, center in enumerate(problem.centers):
        csym, musym = oj._center_symbols(index)
        for slot, a in enumerate(AXES):
            columns[csym[a].name] = float(center.position_m[slot])
        columns[musym.name] = float(center.mu_m3_s2)
    return columns


class _Transcription:
    """z = [u_0, dt_0, ..., u_{N-1}, dt_{N-1}, x_1, ..., x_N], scaled."""

    def __init__(self, problem, t0, position, velocity):
        self.problem = problem
        self.n_u = problem.design.thruster_count
        self.N = problem.slices
        mu = problem.centers[0].mu_m3_s2
        mass = problem.design.mass_kg
        self.t0 = float(t0)
        self.x0 = np.concatenate([np.asarray(position, float),
                                  mass * np.asarray(velocity, float)])
        normal = np.cross(self.x0[:3] - problem.centers[0].position_m,
                          self.x0[3:])
        self.normal = normal / np.linalg.norm(normal)
        self.columns = _design_columns(problem)
        self.columns.update({_TARGET_RADIUS.name: problem.target_radius_m,
                             **{_NORMAL[a].name: self.normal[i]
                                for i, a in enumerate(AXES)}})
        # scales: position, momentum, slice duration
        self.s_r = problem.target_radius_m
        self.s_p = mass * math.sqrt(mu / problem.target_radius_m)
        self.s_dt = 60.0
        self.per_slice = self.n_u + 1
        self.n_z = self.N * self.per_slice + 6 * self.N
        self.u_names = [thruster_symbols(j)["throttle"].name
                        for j in range(self.n_u)]
        self.x_names = ([_POSITION[a].name for a in AXES]
                        + [_MOMENTUM[a].name for a in AXES])
        self.x_next_names = ([_POSITION_NEXT[a].name for a in AXES]
                             + [_MOMENTUM_NEXT[a].name for a in AXES])
        self.x_scale = np.array([self.s_r] * 3 + [self.s_p] * 3)
        # defect rows are momentum (0..2) then position (3..5)
        self.row_scale = np.array([self.s_p] * 3 + [self.s_r] * 3)
        self.arrival_scale = np.array([self.s_r, mu / self.s_r,
                                       math.sqrt(mu / self.s_r), self.s_r,
                                       self.s_p])
        self._cache_key = None

    # -- the compiled rows (built on first use, cached per shape)
    @functools.cached_property
    def slice_rows(self):
        return _slice_rows(len(self.problem.centers), self.n_u)

    @functools.cached_property
    def arrival_rows(self):
        return _arrival_rows()

    @functools.cached_property
    def cost_row(self):
        return _cost_row(self.N, self.n_u)

    # -- layout
    def u_slot(self, k):
        return slice(k * self.per_slice, k * self.per_slice + self.n_u)

    def dt_slot(self, k):
        return k * self.per_slice + self.n_u

    def x_slot(self, k):                  # node k >= 1
        base = self.N * self.per_slice + 6 * (k - 1)
        return slice(base, base + 6)

    def unpack(self, z):
        u = np.stack([z[self.u_slot(k)] for k in range(self.N)])
        dt = np.array([z[self.dt_slot(k)] for k in range(self.N)]) * self.s_dt
        x = np.vstack([self.x0] + [z[self.x_slot(k)] * self.x_scale
                                   for k in range(1, self.N + 1)])
        return u, dt, x

    def pack(self, u, dt, x):
        z = np.zeros(self.n_z)
        for k in range(self.N):
            z[self.u_slot(k)] = u[k]
            z[self.dt_slot(k)] = dt[k] / self.s_dt
        for k in range(1, self.N + 1):
            z[self.x_slot(k)] = x[k] / self.x_scale
        return z

    def bounds(self):
        low = np.full(self.n_z, -np.inf)
        high = np.full(self.n_z, np.inf)
        u_low = [t.throttle_min for t in self.problem.design.thrusters]
        u_high = [t.throttle_max for t in self.problem.design.thrusters]
        dt_low, dt_high = self.problem.dt_bounds_s
        for k in range(self.N):
            low[self.u_slot(k)], high[self.u_slot(k)] = u_low, u_high
            low[self.dt_slot(k)] = dt_low / self.s_dt
            high[self.dt_slot(k)] = dt_high / self.s_dt
        return list(zip(low, high))

    # -- evaluation through the compiled rows
    def slice_feeds(self, x_k, u_k, dt_k, x_next):
        feeds = dict(self.columns)
        feeds.update(zip(self.x_names, x_k))
        feeds.update(zip(self.u_names, u_k))
        feeds[_DT.name] = dt_k
        feeds.update(zip(self.x_next_names, x_next))
        return feeds

    def evaluate(self, z):
        """(cost, d cost/dz, constraints c(z), dc/dz) in scaled units."""
        key = z.tobytes()
        if key == self._cache_key:
            return self._cache
        u, dt, x = self.unpack(z)
        N, n_u = self.N, self.n_u
        n_c = 6 * N + 5
        c = np.zeros(n_c)
        J = np.zeros((n_c, self.n_z))
        for k in range(N):
            feeds = self.slice_feeds(x[k], u[k], dt[k], x[k + 1])
            for i, row in enumerate(self.slice_rows):
                value, grad = row(feeds)
                r = 6 * k + i
                c[r] = value / self.row_scale[i]
                for j, name in enumerate(self.u_names):
                    J[r, self.u_slot(k).start + j] = grad.get(name, 0.0)
                J[r, self.dt_slot(k)] = grad.get(_DT.name, 0.0) * self.s_dt
                if k >= 1:
                    s = self.x_slot(k)
                    J[r, s] = [grad.get(n, 0.0) for n in self.x_names] * self.x_scale
                s = self.x_slot(k + 1)
                J[r, s] = [grad.get(n, 0.0) for n in self.x_next_names] * self.x_scale
                J[r] /= self.row_scale[i]
        feeds = dict(self.columns)
        feeds.update(zip(self.x_names, x[N]))
        s = self.x_slot(N)
        for i, row in enumerate(self.arrival_rows):
            value, grad = row(feeds)
            r = 6 * N + i
            c[r] = value / self.arrival_scale[i]
            J[r, s] = ([grad.get(n, 0.0) for n in self.x_names] * self.x_scale
                       / self.arrival_scale[i])
        cost_feeds = dict(self.columns)
        cost_feeds.update(self.weights)
        for k in range(N):
            for j, name in enumerate(self.u_names):
                cost_feeds[_slice_symbol(k, sp.Symbol(name)).name] = u[k, j]
            cost_feeds[_slice_symbol(k, _DT).name] = dt[k]
        cost, grad = self.cost_row(cost_feeds)
        g = np.zeros(self.n_z)
        for k in range(N):
            for j, name in enumerate(self.u_names):
                g[self.u_slot(k).start + j] = grad.get(
                    _slice_symbol(k, sp.Symbol(name)).name, 0.0)
            g[self.dt_slot(k)] = grad.get(
                _slice_symbol(k, _DT).name, 0.0) * self.s_dt
        self._cache_key, self._cache = key, (cost, g, c, J)
        return self._cache

    def step(self, x_k, u_k, tau):
        """``step(x_k, u_k, tau)`` through the compiled defect rows: with
        ``x_next = 0`` each defect is minus the stepped state."""
        feeds = self.slice_feeds(x_k, u_k, tau, np.zeros(6))
        values = np.array([-row(feeds)[0] for row in self.slice_rows])
        return np.concatenate([values[3:], values[:3]])     # (r, p)


# --------------------------------------------------------- the warm start
def _burn(design: CraftDesign, mass: float, delta_v: np.ndarray):
    """(duration, throttles) of a finite burn delivering ``delta_v`` at
    half the summed thrust of the thrusters that push along it."""
    B = actuation_matrix(design)
    speed = float(np.linalg.norm(delta_v))
    direction = delta_v / speed
    authority = 0.5 * float(np.sum(np.maximum(B.T @ direction, 0.0)))
    duration = mass * speed / authority
    throttles = allocate_throttles(design, mass * delta_v / duration, 0.0)
    return duration, throttles


def hohmann_warm_start(problem: CollocationProblem, t0: float, position,
                       velocity):
    """The Hohmann plan (``orbital_plan.hohmann_plan``) transcribed onto the
    N slices: burn 1, N-2 equal coast slices, burn 2.  Returns
    (HohmannPlan, transcription, z0)."""
    mu = problem.centers[0].mu_m3_s2
    mass = problem.design.mass_kg
    tr = _Transcription(problem, t0, position, velocity)
    r0 = np.asarray(position, float) - problem.centers[0].position_m
    hp = hohmann_plan(mu, float(np.linalg.norm(r0)), problem.target_radius_m,
                      t_burn1=float(t0), phase=math.atan2(r0[1], r0[0]))
    _, v_pre1 = hohmann_reference(hp, t0 - 1.0e-9)
    _, v_post1 = hohmann_reference(hp, t0)
    _, v_pre2 = hohmann_reference(hp, hp.t_burn2 - 1.0e-9)
    _, v_post2 = hohmann_reference(hp, hp.t_burn2)
    dt1, u1 = _burn(problem.design, mass, v_post1 - v_pre1)
    dt2, u2 = _burn(problem.design, mass, v_post2 - v_pre2)
    N = problem.slices
    times = np.concatenate([[t0], np.linspace(t0 + dt1, hp.t_burn2, N - 1),
                            [hp.t_burn2 + dt2]])
    r_nodes, v_nodes = hohmann_reference(hp, times)
    v_nodes[N - 1] = v_pre2              # node N-1 is burn 2's start
    x = np.hstack([r_nodes, mass * v_nodes])
    x[0] = tr.x0
    u = np.zeros((N, tr.n_u))
    u[0], u[-1] = u1, u2
    # balance the cost weights at the warm start (see CollocationProblem)
    impulse = sum(float(np.sum(uk * [t.max_thrust_n for t in
                                     problem.design.thrusters])) * d
                  for uk, d in zip(u, np.diff(times)))
    duration = float(times[-1] - times[0])
    alpha = (problem.alpha if problem.alpha is not None
             else 1.0 / float(np.sum([t.max_thrust_n for t in
                                      problem.design.thrusters])))
    beta = (problem.beta if problem.beta is not None
            else alpha * impulse / duration**2)
    tr.weights = {_ALPHA.name: alpha, _BETA.name: beta,
                  _KAPPA.name: problem.kappa,
                  _BUDGET.name: problem.fuel_budget_n_s}
    return hp, tr, tr.pack(u, np.diff(times), x)


# ------------------------------------------------------------- the planner
def plan_transfer(problem: CollocationProblem, t0: float, position, velocity,
                  *, max_iterations: int = 200, tolerance: float = 1.0e-10):
    """Solve the transcription from the present state (SLSQP fed the
    compiled Jacobian).  Returns (CollocationPlan, HohmannPlan)."""
    from scipy.optimize import minimize

    hp, tr, z0 = hohmann_warm_start(problem, t0, position, velocity)
    result = minimize(
        lambda z: tr.evaluate(z)[0], z0, jac=lambda z: tr.evaluate(z)[1],
        method="SLSQP", bounds=tr.bounds(),
        constraints=[{"type": "eq", "fun": lambda z: tr.evaluate(z)[2],
                      "jac": lambda z: tr.evaluate(z)[3]}],
        options={"maxiter": max_iterations, "ftol": tolerance})
    return _plan_from(tr, result.x, result.success, str(result.message)), hp


def _plan_from(tr: _Transcription, z, success: bool, message: str):
    cost, _, c, _ = tr.evaluate(z)
    u, dt, x = tr.unpack(z)
    mass = tr.problem.design.mass_kg
    thrust = np.asarray([t.max_thrust_n for t in tr.problem.design.thrusters])
    plan = CollocationPlan(
        mu=float(tr.problem.centers[0].mu_m3_s2), mass_kg=mass,
        target_radius_m=tr.problem.target_radius_m,
        times=tr.t0 + np.concatenate([[0.0], np.cumsum(dt)]),
        positions=x[:, :3].copy(), velocities=x[:, 3:] / mass,
        throttles=u.copy(), impulse_n_s=float(np.sum((u @ thrust) * dt)),
        cost=float(cost), max_defect=float(np.max(np.abs(c))),
        converged=bool(success), message=message, center_count=len(
            tr.problem.centers))
    plan.columns["transcription"] = tr
    return plan


# ----------------------------------------------------------- the reference
def reference(plan: CollocationPlan, t):
    """(r, v) the plan prescribes at ``t``: inside slice k the slice's own
    step law from node k over ``t - t_k`` with u_k held (the transcription's
    dynamics); after arrival, the target circle (``orbital_plan``'s Kepler
    legs, xy-plane prograde geometry) through the arrival node."""
    tr = plan.columns["transcription"]
    times = np.atleast_1d(np.asarray(t, dtype=float))
    out_r, out_v = np.zeros((times.size, 3)), np.zeros((times.size, 3))
    arrival = None
    for i, ti in enumerate(times):
        if ti >= plan.t_arrive:
            if arrival is None:
                rN = plan.positions[-1] - tr.problem.centers[0].position_m
                arrival = hohmann_plan(plan.mu, plan.target_radius_m,
                                       plan.target_radius_m,
                                       t_burn1=plan.t_arrive,
                                       phase=math.atan2(rN[1], rN[0]))
            out_r[i], out_v[i] = hohmann_reference(arrival, float(ti))
            continue
        k = max(0, int(np.searchsorted(plan.times, ti, side="right")) - 1)
        x_k = np.concatenate([plan.positions[k],
                              plan.mass_kg * plan.velocities[k]])
        state = tr.step(x_k, plan.throttles[k], float(ti - plan.times[k]))
        out_r[i], out_v[i] = state[:3], state[3:] / plan.mass_kg
    if np.ndim(t) == 0:
        return out_r[0], out_v[0]
    return out_r, out_v
