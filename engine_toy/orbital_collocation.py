"""Orbital craft, build step 4: the collocation planner.

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``
(decision 4: the planner holds the WHOLE remaining trip from the present
state; decision 7: its cost).  Differentiation:
``turing/docs/DIFFERENTIATION_FEASIBILITY_2026-10-02.md`` (graph-native
reverse of the slice; block-bidiagonal sparsity).

Transcription.  The trip from the present state x_0 at t_0 to the target
circle is N slices.  Slice k holds the throttles u_k of a planning design
(one per thruster, inside its declared box, held over the slice) and its
duration dt_k; the node states x_k = (position, momentum, propellant mass),
k = 1..N, are free.  x_0 is the present state and is not a variable, so
every plan -- and every re-plan -- is the total trip from the current moment
over a horizon that shrinks as the trip proceeds (sum dt_k is free).

Dynamics defects ``x_{k+1} - flow(x_k, u_k, dt_k)``; the flow is
``substeps`` steps of ``dt_k / substeps`` (the existing default is 16).
``step`` is constructed by the dt library's actual ``RK4Integrator.step``
from the jumper's seven continuous laws
(``orbital_jumper`` / ``orbital_actuation``): actuation
eq_TS1_2 at the planning attitude, propellant flow eq_TS1_2/TS1_4, gravity
eq_N4_1, the variable-mass momentum law eq_N7_2, position eq_N1_1 -- stepped
with the mass ``dry + P`` and gravity evaluated at each library stage.

The propellant supply is 1 (the planner keeps the tank above empty:
the barrier below).  Arrival at node N: radius = r_target, vis-viva eq_KE1_3
(``orbital_plan``, from the original ``Orbit.vis_viva``) at a = r_target,
no radial velocity, and the present orbit's plane (two rows).

Cost (decision 7; plan deviation is not here -- zero by construction):

    I = sum_k dt_k sum_j w_j thrust_j(u_k)      fuel consumed
    T = sum_k dt_k                              time to arrive
    J = alpha I / T + beta T - kappa log(1 - I / fuel_budget)

compiled as two motions: the slice's fuel (one per slice, shared) and J on
the totals (I, T); the totals' adjoint is dJ/dI, dJ/dT broadcast to every
slice (the adjoint of a sum), so nothing compiles per N.

``w_j`` prices thruster j as the tracker does (``orbital_tracker.
fuel_price``'s rule): its propellant per impulse 1/c (kg/(N s)) when the
design burns propellant, else 1 (impulse, the reactionless kind).  The thrust
integrand is the original set's ``force_cost_integral`` with arc length as
time, as the jumper's thrust-cost piece spells it per thruster
(``src/transmogrifier/orbital_transfer.py``, decision 5).

Jacobian.  :func:`compile_reverse_rows` is the one seam: the rows are
ingested by ``symbolic_process_graph.ingest_sympy_expressions``,
differentiated by ``process_graph_autograd`` (graph-native reverse), fused
into ONE forward/backward motion with an explicit upstream seed per row,
lowered and compiled once; a Jacobian row is one native run with that row's
seed one-hot.  One slice motion serves every slice and sub-step: a slice's
Jacobian is its sub-steps' compiled Jacobians chained in order.  Compiled
rows persist across processes (:data:`CACHE_DIRECTORY`).  Nothing here
differentiates numerically or symbolically.

Solve (2026-10-03 rework, measurements in
``turing/docs/concordance_census/CONTINUATION_orbital_step4_collocation.md``).
The full transcription above stays the plan's form and the reference
method (``plan_transfer(method="SLSQP")``), but the planner solves it
CONDENSED over a burn structure (:class:`_Condensed`): burn slices own a
duration and per-thruster impulses, coast slices form arcs of equal slices,
and the nodes follow by forward substitution of the block-bidiagonal
defects (each defect zero by construction).  :func:`solve_structured`
promotes coast slices to burns by the switching function (primer vector:
the adjoint of the same compiled Jacobians).  Every plan starts from
Hohmann or, for a re-plan, from the previous plan's remainder
(:func:`remainder_warm_start`).  The compiled rows are read inside the
throttle box (:meth:`_Transcription.inside`): the laws' clamp halves the
derivative on the box edge.

Public surface:

    CollocationProblem   frozen: craft, centers, target, budget, weights, N
    CollocationPlan      the trip: nodes, throttles; ``reference(t)``,
                         ``impulses()``, ``mu``, ``ideal_delta_v`` (the
                         tracker's plan protocol)
    plan_transfer(problem, t0, position, velocity, previous=, ...)
                         -> (plan, hohmann)
    remainder_warm_start(previous, problem, t0, position, velocity, ...)
    solve_structured(transcription, u, dt, burn, ...)
    collocation_replanner(problem, craft=) -> the tracker's re-planner
    pointing_proxy(design) -> a planning design for a craft that points
    compile_reverse_rows(name, expressions, wrt) -> ReverseRows
"""
from __future__ import annotations

import dataclasses
import functools
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import sympy as sp

import honorary_engine_equation_catalogue as honorary
import orbital_jumper as oj
from orbital_actuation import (AXES, PROPELLANT_FLOW, PROPELLANT_MASS,
                               PROPELLANT_SUPPLY, CraftDesign,
                               actuation_force_rhs, actuation_matrix,
                               attitude_symbols, clamp_throttles,
                               propellant_flow_rhs, thrust_magnitude,
                               thruster_columns, thruster_symbols)
from orbital_jumper import GravityCenter
from orbital_plan import KEPLER_LAWS, hohmann_plan
from orbital_plan import reference as hohmann_reference
from orbital_tracker import PlannedImpulse, allocate_throttles
from src.common.dt_system.integrator.integrator import RK4Integrator

# ---------------------------------------------------------------- the laws
_DT, _MASS, _DRY = sp.Symbol("dt"), sp.Symbol("mass"), sp.Symbol("dry_mass")
_POSITION = {a: sp.Symbol(f"position_{a}") for a in AXES}
_MOMENTUM = {a: sp.Symbol(f"momentum_{a}") for a in AXES}
_FORCE = {a: sp.Symbol(f"force_{a}") for a in AXES}
_NEXT = {s: sp.Symbol(f"{s.name}_next") for s in
         (*_MOMENTUM.values(), *_POSITION.values(), PROPELLANT_MASS)}
#: node state order: momentum (3), position (3), propellant (1)
STATE = (*_MOMENTUM.values(), *_POSITION.values(), PROPELLANT_MASS)
_TARGET_RADIUS = sp.Symbol("target_radius")
_NORMAL = {a: sp.Symbol(f"plane_normal_{a}") for a in AXES}
_ALPHA, _BETA, _KAPPA, _BUDGET = sp.symbols(
    "cost_alpha cost_beta cost_kappa fuel_budget")


def _fuel_price_symbol(j: int) -> sp.Symbol:
    return sp.Symbol(f"thruster{j}_fuel_price")


def _cancel_affine_physical_law(expression, variables):
    """Cancel rational coefficients without expanding the physical forces.

    The library RK4 position/momentum relations are affine in the initial
    momenta and the forces from its actual callbacks. Preserve every existing
    Float's exact binary value before polynomial reconstruction; otherwise
    SymPy can round coefficients while cancelling the stage mass factors.
    This is rational algebra on the authored law, not numerical precision.
    Its domain retains the original nonzero stage masses.
    """
    exact = expression.xreplace({atom: sp.Rational(atom)
                                 for atom in expression.atoms(sp.Float)})
    numerator, denominator = exact.as_numer_denom()
    polynomial = sp.Poly(numerator, *variables)
    return sp.Add(*(
        sp.factor(sp.cancel(coefficient / denominator))
        * sp.prod(variable**power for variable, power in zip(variables, powers))
        for powers, coefficient in polynomial.terms()))


def slice_defect_laws(center_count: int, thruster_count: int):
    """The seven defects ``x_next - step(x, u, dt)`` of one slice (module
    docstring), spelled on the jumper's columns; ``*_next`` is node k+1."""
    supply = {PROPELLANT_SUPPLY: sp.Integer(1)}
    flow = propellant_flow_rhs(thruster_count).xreplace(supply)
    applied = {a: actuation_force_rhs(a, thruster_count).xreplace(supply)
               for a in AXES}
    gravity = {a: oj.gravity_force_rhs(a, center_count) for a in AXES}
    n72 = {a: oj.variable_mass_momentum_rate(a) for a in AXES}
    outflow = sp.Dummy("physical_outflow")
    forces = {}

    def derivative(_time, state):
        at = dict(zip(STATE, state))
        mass = _DRY + at[PROPELLANT_MASS]
        position = {a: at[_POSITION[a]] for a in AXES}
        momentum = {a: at[_MOMENTUM[a]] for a in AXES}
        stage_forces = {}
        for a in AXES:
            force = sp.Dummy(f"physical_force_{a}")
            forces[force] = gravity[a].xreplace({
                **{_POSITION[b]: position[b] for b in AXES}, _MASS: mass}) + applied[a]
            stage_forces[a] = force
        return sp.Matrix(
            [n72[a].xreplace({_FORCE[a]: stage_forces[a], PROPELLANT_FLOW: outflow,
                             _MOMENTUM[a]: momentum[a], _MASS: mass}) for a in AXES]
            + [honorary.eq_N1_1.rhs.xreplace({
                honorary.m_i(honorary.t): mass,
                honorary.p_i(honorary.t): momentum[a]}) for a in AXES]
            + [-outflow])

    advanced = RK4Integrator().step(derivative, sp.Integer(0),
                                   sp.Matrix(STATE), _DT)
    # The native c0/t1 gate's transverse throttle derivatives cancelled five
    # nonzero adjoints only after rounding. Collect these exact relations
    # before AD instead: ballistic velocity then contains no outflow factor.
    # Force definitions come from the callbacks above, including their actual
    # gravity positions. They are substituted in reverse dependency order;
    # no stage schedule or RK weight is authored here.
    variables = (*_MOMENTUM.values(), *forces)
    reduced = []
    for expression in advanced:
        expression = _cancel_affine_physical_law(expression, variables)
        for force, definition in reversed(tuple(forces.items())):
            expression = expression.xreplace({force: definition})
        reduced.append(expression.xreplace({outflow: flow}))
    return [_NEXT[s] - reduced[i] for i, s in enumerate(STATE)]


def arrival_laws():
    """Five rows that vanish on the target circle about center 0 in the
    plane of unit normal ``plane_normal_*``."""
    center, mu = oj._center_symbols(0)
    rel = {a: _POSITION[a] - center[a] for a in AXES}
    radius = sp.sqrt(sum(rel[a]**2 for a in AXES))
    mass = _DRY + PROPELLANT_MASS
    vis_viva = KEPLER_LAWS["eq_KE1_3"]
    speed = vis_viva.rhs.xreplace({sp.Symbol("mu", positive=True): mu,
                                   sp.Symbol("r", positive=True): radius,
                                   sp.Symbol("a", positive=True): _TARGET_RADIUS})
    if speed.has(sp.Symbol("mu", positive=True)):
        raise RuntimeError("eq_KE1_3 no longer spells mu, r, a as expected")
    return [radius - _TARGET_RADIUS,
            sum(_MOMENTUM[a]**2 for a in AXES) / mass**2 - speed**2,
            sum(rel[a] * _MOMENTUM[a] for a in AXES) / (mass * radius),
            sum(_NORMAL[a] * rel[a] for a in AXES),
            sum(_NORMAL[a] * _MOMENTUM[a] for a in AXES)]


_FUEL, _DURATION = sp.symbols("trip_fuel trip_duration")


def slice_fuel_law(thruster_count: int):
    """One slice's fuel: ``dt * sum_j w_j thrust_j(u)`` (eq_TS1_2 priced)."""
    return _DT * sum((_fuel_price_symbol(j) * thrust_magnitude(j)
                      for j in range(thruster_count)), sp.Integer(0))


def trip_cost_law():
    """Decision 7 on the trip's totals I (``trip_fuel``) and T
    (``trip_duration``): ``alpha I / T + beta T - kappa log(1 - I / B)``.
    The totals are sums of the slices' fuel and durations; their adjoint is
    the broadcast of dJ/dI and dJ/dT to every slice (:meth:`evaluate`)."""
    return (_ALPHA * _FUEL / _DURATION + _BETA * _DURATION
            - _KAPPA * sp.log(1 - _FUEL / _BUDGET))


# ------------------------------------------------------- the Jacobian source
class CompiledReverseUnavailable(RuntimeError):
    """The compiled graph reverse did not publish a requested output."""


@dataclass
class ReverseRows:
    """Compiled rows: ``rows(feeds) -> (values, gradients)`` with
    ``gradients[i] = {wrt name: d row_i}``; ``rows.values(feeds)`` is the
    forward alone (one run, every seed zero).

    The native ABI is allocated once (``prepare_artifact_execution``) and
    re-run: every scalar lives in the execution's scalar arena, so a feed is
    one indexed write and a row one indexed read."""

    name: str
    artifact: object
    input_ids: dict              # symbol name -> motion input value id
    loss_ids: tuple              # per row
    seed_ids: tuple              # per row
    gradient_ids: dict           # wrt symbol name -> output value id
    compile_s: float = 0.0
    compiler: object = None

    @functools.cached_property
    def _execution(self):
        from src.compiler.ssa_llvm_backend import prepare_artifact_execution
        execution = prepare_artifact_execution(self.artifact, {})
        (arena,) = [a for a in execution.scalar_arena.values()
                    if a.dtype == np.float64]
        slot = execution.scalar_index
        everyone = (*self.input_ids.values(), *self.loss_ids, *self.seed_ids,
                    *self.gradient_ids.values())
        if any(int(v) not in slot for v in everyone):
            raise CompiledReverseUnavailable(
                f"{self.name}: a published value is not a float64 scalar")
        self._names = tuple(self.input_ids)
        self._in = np.array([slot[v] for v in self.input_ids.values()])
        self._loss = np.array([slot[v] for v in self.loss_ids])
        self._seed = np.array([slot[v] for v in self.seed_ids])
        self._grad_names = tuple(self.gradient_ids)
        self._grad = np.array([slot[v] for v in self.gradient_ids.values()])
        return execution, arena

    def _load(self, feeds):
        execution, arena = self._execution
        arena[self._in] = [float(feeds[n]) for n in self._names]
        return execution, arena

    def values(self, feeds) -> np.ndarray:
        execution, arena = self._load(feeds)
        arena[self._seed] = 0.0
        execution.run()
        return arena[self._loss].copy()

    def jacobian(self, feeds):
        """(values (rows,), J (rows, wrt) in ``gradient_ids`` order)."""
        execution, arena = self._load(feeds)
        rows = len(self.seed_ids)
        J = np.empty((rows, len(self._grad)))
        values = None
        for row in range(rows):
            arena[self._seed] = 0.0
            arena[self._seed[row]] = 1.0
            execution.run()
            if values is None:
                values = arena[self._loss].copy()
            J[row] = arena[self._grad]
        return values, J

    def __call__(self, feeds):
        values, J = self.jacobian(feeds)
        return values, [dict(zip(self._grad_names, row)) for row in J]


def compile_reverse_rows(name: str, expressions: Sequence[sp.Expr],
                         wrt: Sequence[sp.Symbol]) -> ReverseRows:
    """THE Jacobian source: graph-native reverse of the rows, compiled once.

    ``ingest_sympy_expressions(strict)`` -> ``differentiate_process_graph``
    over every row -> ``fuse_forward_loss_backward(unit_loss_seed=False)``
    (one motion, an explicit upstream seed per row) ->
    ``lower_training_motion_to_repository_ssa`` -> LLVM -> native, inside
    one ``reverse_compile_book``.
    """
    from src.compiler.process_graph_autograd import (
        differentiate_process_graph, fuse_forward_loss_backward,
        lower_training_motion_to_repository_ssa, reverse_compile_book)
    from src.compiler.ssa_llvm_backend import (compile_artifact,
                                               emit_ssa_function_to_llvm)
    from src.compiler.symbolic_process_graph import ingest_sympy_expressions
    from src.transmogrifier.graph.graph_express2 import ProcessGraph

    started = time.perf_counter()
    key = _cache_key(name, expressions, wrt)
    cached = _load_cached_rows(name, key)
    if cached is not None:
        return cached
    with reverse_compile_book():
        graph = ProcessGraph(materialize_memory=False)
        roots = ingest_sympy_expressions(
            graph, list(expressions),
            output_names=[f"{name}_row{i}" for i in range(len(expressions))],
            strict=True)
        inputs = {}
        for node, data in graph.G.nodes(data=True):
            symbol = data.get("expr_obj")
            if isinstance(symbol, sp.Symbol) and data.get("op") == "input":
                inputs[symbol.name] = int(node)
        free = set().union(*(e.free_symbols for e in expressions))
        if set(inputs) != {s.name for s in free}:
            raise RuntimeError(f"{name}: input leaves {sorted(inputs)} != free "
                               f"symbols {sorted(s.name for s in free)}")
        targets = [s.name for s in wrt if s.name in inputs]
        adjoint = differentiate_process_graph(
            graph, outputs=list(roots), wrt=[inputs[n] for n in targets])
        motion = fuse_forward_loss_backward(adjoint, unit_loss_seed=False)
        lowering = lower_training_motion_to_repository_ssa(
            motion, function_name=name)
    if lowering.shortfalls:
        raise CompiledReverseUnavailable(
            f"{name}: SSA shortfalls {lowering.shortfalls!r}")
    emitted = emit_ssa_function_to_llvm(lowering.module,
                                        lowering.function_name,
                                        entry_name=lowering.function_name)
    if emitted.shortfalls:
        raise CompiledReverseUnavailable(
            f"{name}: LLVM shortfalls " + "; ".join(
                f"{s.function}: {s.operation}: {s.reason}"
                for s in emitted.shortfalls[:4]))
    artifact = compile_artifact(
        emitted, directory=Path(tempfile.mkdtemp(prefix=f"colloc_{name}_")))
    published = set(map(int, artifact.buffer_order))
    seeds = tuple(int(motion.seed_value_ids[int(root)]) for root in roots)
    losses = tuple(int(lowering.outputs[f"loss_{i}"])
                   for i in range(len(roots)))
    gradient_ids = {n: int(lowering.outputs[f"grad_{inputs[n]}"])
                    for n in targets}
    missing = sorted({*(n for n, v in gradient_ids.items()
                        if v not in published),
                      *(f"seed{i}" for i, v in enumerate(seeds)
                        if v not in published),
                      *(f"loss{i}" for i, v in enumerate(losses)
                        if v not in published)})
    if missing:
        raise CompiledReverseUnavailable(
            f"{name}: the compiled reverse does not publish {missing}")
    from src.compiler.native_law_kernels import route_compiler_record
    rows = ReverseRows(name, artifact, inputs, losses, seeds, gradient_ids,
                       time.perf_counter() - started, route_compiler_record())
    _store_rows(rows, key)
    return rows


#: Compiled rows persist across processes, as ``perforated_network_llvm``
#: persists its compiled reverse: a manifest per artifact, keyed by the laws
#: and the wrt order.  The compiler-source fingerprint is RECORDED, not part
#: of the key. Each load checks the same PieceCompilerRecord as equation_piece:
#: the old law-only manifest silently served rows built before compiler fixes.
CACHE_DIRECTORY = Path(tempfile.gettempdir()) / "orbital_collocation_rows"


def _cache_key(name, expressions, wrt) -> str:
    import hashlib
    digest = hashlib.sha256()
    for text in (name, *(sp.srepr(e) for e in expressions),
                 *(s.name for s in wrt)):
        digest.update(text.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()[:32]


def _store_rows(rows: ReverseRows, key: str) -> None:
    import json
    import shutil
    from src.compiler.native_law_kernels import (
        PieceCompilerRecord, piece_staleness, post_piece_book)
    from types import SimpleNamespace
    a = rows.artifact
    directory = CACHE_DIRECTORY / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "contract.json"
    stale = stale_record = None
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        saved = previous.get("compiler")
        stale_record = (None if saved is None else PieceCompilerRecord(
            saved["digest"], tuple(tuple(row) for row in saved["modules"])))
        stale = piece_staleness(SimpleNamespace(compiler=stale_record)) or None
    library = directory / Path(a.library_path).name
    shutil.copy2(a.library_path, library)
    manifest.write_text(json.dumps({
        "name": rows.name, "entry": a.name, "library": library.name,
        "buffer_order": list(map(int, a.buffer_order)),
        "buffer_shapes": [list(shape) for shape in a.buffer_shapes],
        "buffer_dtypes": list(a.buffer_dtypes),
        "extent_order": [list(item) for item in a.extent_order],
        "input_ids": rows.input_ids, "loss_ids": list(rows.loss_ids),
        "seed_ids": list(rows.seed_ids), "gradient_ids": rows.gradient_ids,
        "compile_s": rows.compile_s,
        "compiler": {"digest": rows.compiler.digest,
                     "modules": rows.compiler.modules}}), encoding="utf-8")
    post_piece_book(directory, rows.name, 1, key, built=rows,
                    stale=stale, stale_record=stale_record)


def _load_cached_rows(name: str, key: str):
    import json
    from src.compiler.native_law_kernels import (
        PieceCompilerRecord, piece_staleness, post_piece_book)
    from types import SimpleNamespace
    from src.compiler.ssa_llvm_backend import LLVMFunctionArtifact
    manifest = CACHE_DIRECTORY / key / "contract.json"
    if not manifest.is_file():
        return None
    record = json.loads(manifest.read_text(encoding="utf-8"))
    saved_compiler = record.get("compiler")
    compiler = (None if saved_compiler is None else PieceCompilerRecord(
        saved_compiler["digest"],
        tuple(tuple(row) for row in saved_compiler["modules"])))
    changed = piece_staleness(SimpleNamespace(compiler=compiler))
    if changed:
        post_piece_book(manifest.parent, name, 1, key, stale=changed,
                        stale_record=compiler, decision="rebuild_required")
        print(f"[collocation_rows] {name}: stale, rebuilding; compiler "
              f"changed in {list(changed[:8])}", flush=True)
        return None
    library = manifest.parent / record["library"]
    if not library.is_file():
        return None
    artifact = LLVMFunctionArtifact(
        name=record["entry"], llvm_ir="",
        buffer_order=tuple(map(int, record["buffer_order"])),
        buffer_shapes=tuple(tuple(s) for s in record["buffer_shapes"]),
        extent_order=tuple(tuple(i) for i in record["extent_order"]),
        shortfalls=(), buffer_dtypes=tuple(record["buffer_dtypes"]),
        library_path=library.resolve())
    return ReverseRows(name, artifact,
                       {k: int(v) for k, v in record["input_ids"].items()},
                       tuple(record["loss_ids"]), tuple(record["seed_ids"]),
                       {k: int(v) for k, v in record["gradient_ids"].items()},
                       float(record["compile_s"]), compiler)


def _slice_wrt(thruster_count: int):
    return (list(STATE) + [thruster_symbols(j)["throttle"]
                           for j in range(thruster_count)]
            + [_DT] + [_NEXT[s] for s in STATE])


@functools.lru_cache(maxsize=None)
def _slice_rows(center_count: int, thruster_count: int) -> ReverseRows:
    return compile_reverse_rows(
        f"colloc_slice_c{center_count}_t{thruster_count}",
        slice_defect_laws(center_count, thruster_count),
        _slice_wrt(thruster_count))


@functools.lru_cache(maxsize=None)
def _arrival_rows() -> ReverseRows:
    return compile_reverse_rows("colloc_arrival", arrival_laws(), list(STATE))


@functools.lru_cache(maxsize=None)
def _fuel_rows(thruster_count: int) -> ReverseRows:
    wrt = [thruster_symbols(j)["throttle"] for j in range(thruster_count)]
    return compile_reverse_rows(f"colloc_fuel_t{thruster_count}",
                                [slice_fuel_law(thruster_count)], wrt + [_DT])


@functools.lru_cache(maxsize=None)
def _cost_rows() -> ReverseRows:
    return compile_reverse_rows("colloc_trip_cost", [trip_cost_law()],
                                [_FUEL, _DURATION])


def prepare_rows(problem):
    """Load/build all four row families during setup, before flight workers."""
    return (_slice_rows(len(problem.centers), problem.design.thruster_count),
            _arrival_rows(), _fuel_rows(problem.design.thruster_count), _cost_rows())


# ------------------------------------------------------------- the problem
@dataclass(frozen=True)
class CollocationProblem:
    """One planning request's fixed data.

    ``design`` is the PLANNING design: its thrusters at ``attitude`` (world
    frame) are the slice's actuation.  ``fuel_budget`` is in the cost's
    fuel units (kg of propellant for a design that burns it, N s of impulse
    for the reactionless kind); ``None`` = the design's propellant.
    ``alpha``/``beta`` weigh the average fuel rate and the time to arrive;
    ``None`` = alpha 1 per (budget / Hohmann time) and beta balanced at the
    Hohmann warm start (dJ/dT = 0 there).  ``kappa`` scales the barrier.
    """

    design: CraftDesign
    centers: tuple
    target_radius_m: float
    fuel_budget: float | None = None
    slices: int = 40
    alpha: float | None = None
    beta: float | None = None
    kappa: float = 1.0e-3
    attitude: tuple | None = None
    dt_bounds_s: tuple = (0.05, 600.0)
    substeps: int = 16

    def __post_init__(self):
        object.__setattr__(self, "centers", tuple(self.centers))
        if self.slices < 3:
            raise ValueError("a transfer needs at least 3 slices")
        if not self.target_radius_m > 0.0:
            raise ValueError("target radius must be positive")

    @property
    def burns_propellant(self) -> bool:
        return any(t.thruster_kind.propellant_per_impulse_kg_n_s > 0.0
                   for t in self.design.thrusters)

    @property
    def budget(self) -> float:
        if self.fuel_budget is not None:
            return float(self.fuel_budget)
        if not self.burns_propellant:
            raise ValueError("a reactionless design needs fuel_budget (N s)")
        return float(self.design.propellant_kg)


def _columns(problem: CollocationProblem) -> dict:
    """The constant input columns: design, centers, attitude, prices."""
    columns = {n: float(v[0]) for n, v in
               thruster_columns(problem.design).items()}
    columns[_DRY.name] = float(problem.design.dry_mass_kg)
    R = (np.eye(3) if problem.attitude is None
         else np.asarray(problem.attitude, float).reshape(3, 3))
    for (row, col), symbol in attitude_symbols().items():
        columns[symbol.name] = float(R[row, col])
    burns = problem.burns_propellant
    for j, t in enumerate(problem.design.thrusters):
        columns[_fuel_price_symbol(j).name] = (
            t.thruster_kind.propellant_per_impulse_kg_n_s if burns else 1.0)
    for index, center in enumerate(problem.centers):
        csym, musym = oj._center_symbols(index)
        for slot, a in enumerate(AXES):
            columns[csym[a].name] = float(center.position_m[slot])
        columns[musym.name] = float(center.mu_m3_s2)
    columns[_TARGET_RADIUS.name] = float(problem.target_radius_m)
    columns[_BUDGET.name] = problem.budget
    columns[_KAPPA.name] = float(problem.kappa)
    return columns


class _Transcription:
    """``z = [u_0, dt_0, ..., u_{N-1}, dt_{N-1}, x_1, ..., x_N]``, scaled."""

    def __init__(self, problem: CollocationProblem, t0, position, velocity,
                 propellant_kg):
        self.problem = problem
        self.n_u = problem.design.thruster_count
        self.N = problem.slices
        mu = problem.centers[0].mu_m3_s2
        self.dry = problem.design.dry_mass_kg
        mass0 = self.dry + float(propellant_kg)
        self.t0 = float(t0)
        self.x0 = np.concatenate([mass0 * np.asarray(velocity, float),
                                  np.asarray(position, float),
                                  [float(propellant_kg)]])
        rel = self.x0[3:6] - np.asarray(problem.centers[0].position_m)
        normal = np.cross(rel, self.x0[:3])
        self.normal = normal / np.linalg.norm(normal)
        self.columns = _columns(problem)
        self.columns.update({_NORMAL[a].name: self.normal[i]
                             for i, a in enumerate(AXES)})
        s_r = problem.target_radius_m
        s_p = mass0 * math.sqrt(mu / s_r)
        self.x_scale = np.array([s_p] * 3 + [s_r] * 3
                                + [max(float(problem.design.propellant_kg),
                                       1.0)])
        self.arrival_scale = np.array([s_r, mu / s_r, math.sqrt(mu / s_r),
                                       s_r, s_p])
        self.s_dt = 60.0
        self.per_slice = self.n_u + 1
        self.n_z = self.N * self.per_slice + 7 * self.N
        self.u_names = [thruster_symbols(j)["throttle"].name
                        for j in range(self.n_u)]
        self.x_names = [s.name for s in STATE]
        self.x_next_names = [_NEXT[s].name for s in STATE]
        self.weights = {}
        self._cache_key = None
        self.evaluations = 0
        self.jacobian_s = 0.0

    @functools.cached_property
    def slice_rows(self):
        return dataclasses.replace(_slice_rows(len(self.problem.centers), self.n_u))

    @functools.cached_property
    def arrival_rows(self):
        return dataclasses.replace(_arrival_rows())

    @functools.cached_property
    def fuel_rows(self):
        return dataclasses.replace(_fuel_rows(self.n_u))

    @functools.cached_property
    def cost_rows(self):
        return dataclasses.replace(_cost_rows())

    # -- layout
    def u_slot(self, k):
        return slice(k * self.per_slice, k * self.per_slice + self.n_u)

    def dt_slot(self, k):
        return k * self.per_slice + self.n_u

    def x_slot(self, k):                  # node k >= 1
        base = self.N * self.per_slice + 7 * (k - 1)
        return slice(base, base + 7)

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
            low[self.x_slot(k + 1).stop - 1] = 0.0          # propellant
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
        """(cost, d cost/dz, constraints c(z), dc/dz), scaled units."""
        key = z.tobytes()
        if key == self._cache_key:
            return self._cache
        started = time.perf_counter()
        u, dt, x = self.unpack(z)
        N = self.N
        n_c = 7 * N + 5
        c = np.zeros(n_c)
        J = np.zeros((n_c, self.n_z))
        eye = np.eye(7)
        for k in range(N):
            end, Jx, Ju, Jdt = self.flow(x[k], u[k], dt[k])
            rows = slice(7 * k, 7 * k + 7)
            scale = self.x_scale[:, None]
            c[rows] = (x[k + 1] - end) / self.x_scale
            J[rows, self.u_slot(k)] = -Ju / scale
            J[rows, self.dt_slot(k)] = -Jdt * self.s_dt / self.x_scale
            if k >= 1:
                J[rows, self.x_slot(k)] = -Jx * self.x_scale / scale
            J[rows, self.x_slot(k + 1)] = eye * self.x_scale / scale
        feeds = dict(self.columns)
        feeds.update(zip(self.x_names, x[N]))
        values, grads = self.arrival_rows(feeds)
        for i, grad in enumerate(grads):
            r = 7 * N + i
            c[r] = values[i] / self.arrival_scale[i]
            J[r, self.x_slot(N)] = np.array(
                [grad.get(n, 0.0) for n in self.x_names]
            ) * self.x_scale / self.arrival_scale[i]
        fuel = np.zeros(N)
        d_fuel = np.zeros((N, self.n_u + 1))
        feeds = dict(self.columns)
        for k in range(N):
            feeds.update(zip(self.u_names, u[k]))
            feeds[_DT.name] = dt[k]
            (fuel[k],), (grad,) = self.fuel_rows(feeds)
            inside = self.inside(u[k])
            if not np.array_equal(inside, u[k]):    # see inside()
                _, (grad_u,) = self.fuel_rows({**feeds, **dict(
                    zip(self.u_names, inside))})
                grad = {**grad, **{n: grad_u[n] for n in self.u_names}}
            d_fuel[k] = [grad.get(n, 0.0) for n in self.u_names] + [
                grad.get(_DT.name, 0.0)]
        (cost,), (grad,) = self.cost_rows({
            **self.columns, **self.weights,
            _FUEL.name: float(np.sum(fuel)), _DURATION.name: float(np.sum(dt))})
        dJ_dI, dJ_dT = grad[_FUEL.name], grad[_DURATION.name]
        g = np.zeros(self.n_z)
        for k in range(N):
            g[self.u_slot(k)] = dJ_dI * d_fuel[k, :-1]
            g[self.dt_slot(k)] = (dJ_dI * d_fuel[k, -1] + dJ_dT) * self.s_dt
        if not (np.all(np.isfinite(J)) and np.all(np.isfinite(g))):
            raise FloatingPointError("non-finite compiled gradient")
        self.evaluations += 1
        self.jacobian_s += time.perf_counter() - started
        self._cache_key, self._cache = key, (float(cost), g, c, J)
        return self._cache

    @functools.cached_property
    def _columns_of(self):
        """Positions of x, u, dt among the slice rows' gradient columns."""
        where = {n: i for i, n in enumerate(self.slice_rows.gradient_ids)}
        return (np.array([where[n] for n in self.x_names]),
                np.array([where[n] for n in self.u_names]),
                where[_DT.name])

    def step(self, x_k, u_k, tau):
        """One step of the slice law through the compiled slice motion: with
        ``x_next = 0`` each defect is minus the stepped state."""
        feeds = self.slice_feeds(x_k, u_k, tau, np.zeros(7))
        return -self.slice_rows.values(feeds)

    def advance(self, x_k, u_k, tau):
        """The slice's flow over ``tau``: ``substeps`` equal steps."""
        x = np.asarray(x_k, float)
        for _ in range(self.problem.substeps):
            x = self.step(x, u_k, tau / self.problem.substeps)
        return x

    def flow(self, x_k, u_k, dt):
        """The slice's flow over ``dt`` in ``substeps`` steps of the compiled
        law, and its Jacobians d end / d (x_k, u_k, dt): each step's compiled
        Jacobian (the motion's rows at ``x_next = 0`` are minus the step)
        chained in order -- forward accumulation of compiled derivatives."""
        S = self.problem.substeps
        ix, iu, idt = self._columns_of
        x = np.asarray(x_k, float)
        inside = self.inside(u_k)
        on_edge = not np.array_equal(inside, u_k)
        Jx, Ju, Jdt = np.eye(7), np.zeros((7, self.n_u)), np.zeros(7)
        for _ in range(S):
            values, D = self.slice_rows.jacobian(
                self.slice_feeds(x, inside, dt / S, np.zeros(7)))
            if on_edge:          # the value at the throttle itself
                values = self.slice_rows.values(
                    self.slice_feeds(x, u_k, dt / S, np.zeros(7)))
            A = -D[:, ix]
            Jx = A @ Jx
            Ju = A @ Ju - D[:, iu]
            Jdt = A @ Jdt - D[:, idt] / S
            x = -values
        return x, Jx, Ju, Jdt

    #: How far inside its box a throttle on the box's edge is read for
    #: DERIVATIVES (values are always read at the throttle itself).
    EDGE_READ = 1.0e-9

    @functools.cached_property
    def _throttle_box(self):
        return (np.array([t.throttle_min for t in self.problem.design.thrusters]),
                np.array([t.throttle_max for t in self.problem.design.thrusters]))

    def inside(self, u_k) -> np.ndarray:
        """``u_k`` with every throttle on its box's edge moved
        :data:`EDGE_READ` inside.

        The laws clamp the throttle (``thrust_magnitude``: Min(Max(u, lo),
        hi)); the compiled reverse reads a tie of Max/Min as the average of
        its two sides, so on the edge itself every throttle column of the
        slice and fuel Jacobians is HALF its value inside (measured: d
        momentum/d throttle -10043 at u = 0 and at u = 1 against -20086 at
        1e-12 and 0.5).  The planner's iterates live in the box, where the
        clamp is the identity: its derivative there is the inside one.  The
        law is linear in the throttle inside the box, so the inside reading
        is exact."""
        lo, hi = self._throttle_box
        u = np.asarray(u_k, float)
        eps = self.EDGE_READ * (hi - lo)
        return np.where(u <= lo + eps, lo + eps,
                        np.where(u >= hi - eps, hi - eps, u))

    def slice_fuel(self, u, dt, *, jac=True):
        """Per-slice fuel through the compiled fuel rows: (N,), and
        (N, n_u + 1) d fuel/d (u, dt [s]) read inside the box (see
        :meth:`inside`) when ``jac``."""
        fw = self.fuel_rows
        where = {n: i for i, n in enumerate(fw.gradient_ids)}
        cols = [where[n] for n in self.u_names] + [where[_DT.name]]
        fuel = np.zeros(self.N)
        d_fuel = np.zeros((self.N, self.n_u + 1)) if jac else None
        feeds = dict(self.columns)
        for k in range(self.N):
            feeds.update(zip(self.u_names, u[k]))
            feeds[_DT.name] = dt[k]
            if not jac:
                fuel[k] = fw.values(feeds)[0]
                continue
            values, D = fw.jacobian(feeds)
            fuel[k] = values[0]
            d_fuel[k] = D[0, cols]
            inside = self.inside(u[k])
            if not np.array_equal(inside, u[k]):
                _, D = fw.jacobian({**feeds, **dict(zip(self.u_names,
                                                        inside))})
                d_fuel[k, :-1] = D[0, cols[:-1]]
        return fuel, d_fuel

    def fuel(self, u, dt):
        price =np.array([self.columns[_fuel_price_symbol(j).name]
                          * t.max_thrust_n for j, t in
                          enumerate(self.problem.design.thrusters)])
        return float(sum((clamp_throttles(self.problem.design, uk) @ price)
                         * d for uk, d in zip(u, dt)))


# --------------------------------------------- the condensed transcription
class _Condensed:
    """The transcription with its nodes eliminated by their defects, over a
    declared burn STRUCTURE.

    Why (measured 2026-10-03, CONTINUATION_orbital_step4_collocation.md):
    on the full transcription (560 variables, 285 defects) SLSQP never
    terminates on the kick re-plan -- 300 iterations, defects swinging to
    0.3, and restarted from its own polished result it creeps 0.94910 ->
    0.94841 over 300 more.  The full transcription carries ~35 exactly flat
    directions (redistributing one coast arc's duration among its slices
    changes nothing but discretization) plus the bilinear throttle x
    duration valley of every slice, and SLSQP's dense BFGS has to learn
    them; a structured SQP (per-slice damped BFGS, sparse elastic QP) crept
    the same way.  This class removes both:

    - burn slices (``burn``) own a duration and an IMPULSE per thruster
      ``w = u dt`` (``u = w / dt``, box ``u_min dt <= w <= u_max dt`` as
      linear rows); every other slice coasts (u = 0) and a run of coast
      slices -- an arc -- shares one duration, split equally;
    - the nodes follow from the present state by the slice flow (the
      block-bidiagonal defect system solved by forward substitution: every
      defect is zero by construction) and derivatives by forward
      accumulation of the compiled slice Jacobians through the chain.

    What remains is a dense problem of ``(n_u + 1) * burns + arcs``
    variables and the five arrival rows -- SLSQP's own size.  With
    ``fixed_durations`` only the impulses are free (the continuation's
    first stage)."""

    def __init__(self, tr, burn, u, dt, *, fixed_durations=False):
        self.tr = tr
        N, n_u = tr.N, tr.n_u
        self.burn = np.asarray(burn, bool)
        self.B = np.flatnonzero(self.burn)
        arcs, k = [], 0
        while k < N:
            if self.burn[k]:
                k += 1
                continue
            arc = []
            while k < N and not self.burn[k]:
                arc.append(k)
                k += 1
            arcs.append(arc)
        self.arcs = arcs
        nb = len(self.B)
        n_w = nb * n_u
        self.n_v = n_w if fixed_durations else n_w + nb + len(arcs)
        dt = np.asarray(dt, float)
        u = np.asarray(u, float)
        self.dt_fixed = np.zeros(N)
        self.W = np.zeros((N, n_u, self.n_v))       # d w_k / d v
        self.D = np.zeros((N, self.n_v))            # d dt_k / d v (seconds)
        for i, k in enumerate(self.B):
            self.W[k, :, i * n_u:(i + 1) * n_u] = np.eye(n_u)
        if fixed_durations:
            self.dt_fixed = dt.copy()
        else:
            for i, k in enumerate(self.B):
                self.D[k, n_w + i] = tr.s_dt
            for a, arc in enumerate(arcs):
                self.D[arc, n_w + nb + a] = tr.s_dt / len(arc)
        self.u_lo, self.u_hi = tr._throttle_box
        dlo, dhi = tr.problem.dt_bounds_s
        lo = [0.0] * n_w
        hi = [np.inf] * n_w
        if not fixed_durations:
            lo += [dlo / tr.s_dt] * nb + [len(a) * dlo / tr.s_dt for a in arcs]
            hi += [dhi / tr.s_dt] * nb + [len(a) * dhi / tr.s_dt for a in arcs]
        self.lo, self.hi = np.array(lo), np.array(hi)
        v0 = list(np.ravel(u[self.B] * dt[self.B, None] / tr.s_dt))
        if not fixed_durations:
            v0 += list(dt[self.B] / tr.s_dt)
            v0 += [float(np.sum(dt[a])) / tr.s_dt for a in arcs]
        self.v0 = np.clip(np.array(v0), self.lo, self.hi)
        # the throttle box as linear rows on (w, dt): w - u_min dt >= 0,
        # u_max dt - w >= 0.  With u_min = 0 the first row IS the bound
        # w >= 0 and is left out: the duplicate makes the active set
        # degenerate at w = 0, and SLSQP then stops at its first iteration
        # (measured: a burn promoted at zero impulse never moved)
        rows, offsets = [], []
        for i, k in enumerate(self.B):
            for j in range(n_u):
                pairs = [(-1.0, self.u_hi[j])]
                if self.u_lo[j] > 0.0:
                    pairs.append((1.0, self.u_lo[j]))
                for sign, bound in pairs:
                    row = np.zeros(self.n_v)
                    row[i * n_u + j] = sign
                    row -= sign * bound * self.D[k] / tr.s_dt
                    rows.append(row)
                    offsets.append(-sign * bound * self.dt_fixed[k] / tr.s_dt)
        self.box_rows = np.array(rows).reshape(-1, self.n_v)
        self.box_offsets = np.array(offsets)
        self._key = None
        self.jacobians = self.forwards = 0

    def slices(self, v):
        """u (N, n_u), dt (N,) and du/dv (N, n_u, n_v)."""
        w = np.einsum("kjv,v->kj", self.W, v)
        dt = self.dt_fixed + self.D @ v
        u = np.zeros_like(w)
        U = np.zeros_like(self.W)
        s = self.tr.s_dt
        for k in self.B:
            u[k] = w[k] * s / dt[k]
            U[k] = (self.W[k] * s / dt[k]
                    - np.outer(w[k] * s / dt[k]**2, self.D[k]))
        return u, dt, U

    def evaluate(self, v, jac=True):
        """(cost, d cost/dv, arrival rows c (scaled), dc/dv, nodes, u, dt)."""
        key = v.tobytes()
        if self._key == key and (self._jac or not jac):
            return self._value
        tr = self.tr
        N = tr.N
        started = time.perf_counter()
        u, dt, U = self.slices(v)
        x = np.empty((N + 1, 7))
        x[0] = tr.x0
        X = np.zeros((7, self.n_v))
        flows = []
        for k in range(N):
            if jac:
                x[k + 1], Jx, Ju, Jdt = tr.flow(x[k], u[k], dt[k])
                X = Jx @ X + Ju @ U[k] + np.outer(Jdt, self.D[k])
                flows.append((Jx, Ju))
            else:
                x[k + 1] = tr.advance(x[k], u[k], dt[k])
        feeds = dict(tr.columns)
        feeds.update(zip(tr.x_names, x[N]))
        if jac:
            values, Ja = tr.arrival_rows.jacobian(feeds)
            where = {n: i for i, n in enumerate(tr.arrival_rows.gradient_ids)}
            Ja = Ja[:, [where[n] for n in tr.x_names]] / tr.arrival_scale[:, None]
            dc = Ja @ X
        else:
            values, dc = tr.arrival_rows.values(feeds), None
        c = values / tr.arrival_scale
        fuel, d_fuel = tr.slice_fuel(u, dt, jac=jac)
        I, T = float(np.sum(fuel)), float(np.sum(dt))
        inputs = {**tr.columns, **tr.weights, _FUEL.name: I, _DURATION.name: T}
        if jac:
            (cost,), (grad,) = tr.cost_rows(inputs)
            dJ_dI = grad[_FUEL.name]
            dI = (np.einsum("kj,kjv->v", d_fuel[:, :-1], U)
                  + d_fuel[:, -1] @ self.D)
            g = dJ_dI * dI + grad[_DURATION.name] * self.D.sum(axis=0)
            self._adjoint = (flows, Ja, dJ_dI * d_fuel[:, :-1])
            self.jacobians += 1
            tr.evaluations += 1
            tr.jacobian_s += time.perf_counter() - started
        else:
            (cost,) = tr.cost_rows.values(inputs)
            g = None
            self.forwards += 1
        self._key, self._jac = key, jac
        self._value = (float(cost), g, c, dc, x, u, dt)
        return self._value

    def solve(self, *, max_iterations, tolerance):
        from scipy.optimize import minimize
        value = lambda v: self.evaluate(v, jac=False)[0]
        result = minimize(
            value, self.v0, jac=lambda v: self.evaluate(v)[1],
            method="SLSQP", bounds=list(zip(self.lo, self.hi)),
            constraints=[
                {"type": "eq", "fun": lambda v: self.evaluate(v, jac=False)[2],
                 "jac": lambda v: self.evaluate(v)[3]},
                {"type": "ineq",
                 "fun": lambda v: self.box_rows @ v + self.box_offsets,
                 "jac": lambda v: self.box_rows}],
            options={"maxiter": max_iterations, "ftol": tolerance})
        self.result = result
        return result

    def multipliers(self, v) -> np.ndarray:
        """The arrival rows' multipliers at ``v`` (``L = J - lam . c``):
        minimum-norm least squares of stationarity on the free variables,
        with the active throttle-box rows as their own unknowns.

        Not SLSQP's ``multipliers``: on an in-plane transfer the two plane
        rows have no gradient on any free variable, so stationarity leaves
        their multipliers undetermined and SLSQP's QP returns arbitrary
        ones (measured: -9.98, -6.68 where the free set determines
        nothing), which then read as a spurious negative switching value on
        the out-of-plane thrusters of every coast slice."""
        _f, g, _c, dc, *_ = self.evaluate(v)
        free = (v > self.lo + 1e-9) & (v < self.hi - 1e-9)
        active = self.box_rows @ v + self.box_offsets < 1e-9
        A = np.vstack([dc, self.box_rows[active]])
        if not np.any(free):
            return np.zeros(dc.shape[0])
        solution, *_ = np.linalg.lstsq(A[:, free].T, g[free], rcond=None)
        return solution[:dc.shape[0]]

    def switching(self, v, multipliers) -> np.ndarray:
        """dL/du on every slice, ``L = J - multipliers . c`` (SLSQP's sign),
        by one adjoint sweep back through the stored slice Jacobians: the
        primer-vector test -- a coast slice whose entry is negative lowers
        the cost by thrusting there."""
        self.evaluate(v)
        flows, Ja, dJ_du = self._adjoint
        p = Ja.T @ np.asarray(multipliers, float)
        out = np.zeros((self.tr.N, self.tr.n_u))
        for k in range(self.tr.N - 1, -1, -1):
            Jx, Ju = flows[k]
            out[k] = dJ_du[k] - Ju.T @ p
            p = Jx.T @ p
        return out


#: A coast slice is promoted to a burn when its switching value per
#: throttle-second (cost units) is below minus this.
SWITCH_TOLERANCE = 1.0e-4


def solve_structured(tr, u, dt, burn, *, max_iterations=200,
                     tolerance=1.0e-10, fixed_first=False, rounds=4):
    """The planner's solve: the condensed transcription over a burn
    structure, by continuation --

    1. (``fixed_first``, off by default) the structure fixed, durations
       included: only the burns' impulses move.  Measured on the 300 m/s
       kick re-plans from a collocation plan's remainder (kicks at 602,
       900, 1800 s): this stage never reached a feasible point (SLSQP
       "positive directional derivative" at defects 0.02-0.04 -- the
       remainder's timing cannot absorb the kick with its burns) and added
       0.5-1.4 s; stage 2 from the same start converged to the same plans;
    2. the durations freed (burn durations and coast arcs);
    3. up to ``rounds`` structure changes: the coast slice with the most
       negative switching value (:meth:`_Condensed.switching`) becomes a
       burn and 2 is re-solved from there; a promotion that does not lower
       the cost is undone and ends the rounds.

    Returns (u, dt, nodes, receipts)."""
    burn = np.asarray(burn, bool).copy()
    receipts = {"iterations": 0, "jacobians": 0, "forwards": 0,
                "stages": [], "message": ""}

    def run(fixed):
        stage = _Condensed(tr, burn, u, dt, fixed_durations=fixed)
        result = stage.solve(max_iterations=max_iterations,
                             tolerance=tolerance)
        receipts["iterations"] += int(result.nit)
        receipts["jacobians"] += stage.jacobians
        receipts["forwards"] += stage.forwards
        cost, _g, c, _dc, x, uu, dd = stage.evaluate(result.x)
        receipts["stages"].append((
            "fixed" if fixed else "free", tuple(map(int, stage.B)),
            int(result.nit), str(result.message), cost,
            float(np.max(np.abs(c)))))
        return stage, result, cost, x, uu, dd

    if fixed_first:
        _s, _r, _c, _x, u, dt = run(True)
    stage, result, cost, x, u, dt = run(False)
    for _ in range(rounds):
        sw = stage.switching(result.x, stage.multipliers(result.x))
        per_second = sw / dt[:, None]
        per_second[burn] = np.inf
        k = int(np.argmin(per_second.min(axis=1)))
        if not per_second[k].min() < -SWITCH_TOLERANCE:
            break
        burn[k] = True
        trial = run(False)
        feasible = trial[1].x is not None and float(np.max(np.abs(
            trial[0].evaluate(trial[1].x, jac=False)[2]))) <= 1.0e-8
        if not (feasible and trial[2] < cost - 1e-12 * abs(cost)):
            burn[k] = False
            receipts["stages"].append(("undone", k))
            break
        stage, result, cost, x, u, dt = trial
    receipts["result"] = result
    receipts["structure"] = tuple(map(int, np.flatnonzero(burn)))
    return u, dt, x, receipts


# --------------------------------------------------------- the warm start
def _burn(design, attitude, mass, delta_v):
    """(duration, throttles) of a finite burn delivering ``delta_v`` at half
    the summed thrust of the thrusters that push along it."""
    B = actuation_matrix(design, attitude)
    speed = float(np.linalg.norm(delta_v))
    direction = delta_v / speed
    authority = 0.5 * float(np.sum(np.maximum(B.T @ direction, 0.0)))
    duration = mass * speed / authority
    return duration, allocate_throttles(design, mass * delta_v / duration,
                                        0.0, attitude=attitude)


def hohmann_warm_start(problem: CollocationProblem, t0: float, position,
                       velocity, propellant_kg=None):
    """The Hohmann plan from the present radius and polar angle
    (``orbital_plan.hohmann_plan``, burn 1 now) on the N slices: burn 1,
    N-2 equal coast slices, burn 2.  Returns (HohmannPlan, transcription,
    z0)."""
    propellant = (problem.design.propellant_kg if propellant_kg is None
                  else float(propellant_kg))
    tr = _Transcription(problem, t0, position, velocity, propellant)
    mu = problem.centers[0].mu_m3_s2
    rel = np.asarray(position, float) - problem.centers[0].position_m
    hp = hohmann_plan(mu, float(np.linalg.norm(rel)), problem.target_radius_m,
                      t_burn1=float(t0), phase=math.atan2(rel[1], rel[0]))
    _, v_post1 = hohmann_reference(hp, t0)
    _, v_pre2 = hohmann_reference(hp, hp.t_burn2 - 1.0e-9)
    _, v_post2 = hohmann_reference(hp, hp.t_burn2)
    mass = tr.dry + propellant
    R = problem.attitude
    dt1, u1 = _burn(problem.design, R, mass,
                    v_post1 - np.asarray(velocity, float))
    dt2, u2 = _burn(problem.design, R, mass, v_post2 - v_pre2)
    N = problem.slices
    times = np.concatenate([[t0], np.linspace(t0 + dt1, hp.t_burn2, N - 1),
                            [hp.t_burn2 + dt2]])
    r_nodes, v_nodes = hohmann_reference(hp, times)
    v_nodes[N - 1] = v_pre2                  # node N-1 is burn 2's start
    u = np.zeros((N, tr.n_u))
    u[0], u[-1] = u1, u2
    dt = np.diff(times)
    # propellant along the warm start, by the slices' own flow law
    flow = np.array([tr.fuel(uk[None], [1.0]) if problem.burns_propellant
                     else 0.0 for uk in u])
    P = propellant - np.concatenate([[0.0], np.cumsum(flow * dt)])
    masses = tr.dry + P
    x = np.hstack([masses[:, None] * v_nodes, r_nodes, P[:, None]])
    x[0] = tr.x0
    fuel, duration = tr.fuel(u, dt), float(np.sum(dt))
    budget = problem.budget
    alpha = (problem.alpha if problem.alpha is not None
             else hp.transfer_time / budget)
    beta = (problem.beta if problem.beta is not None
            else alpha * fuel / duration**2)
    tr.weights = {_ALPHA.name: alpha, _BETA.name: beta}
    return hp, tr, tr.pack(u, dt, x)


# ----------------------------------------------------------------- the plan
@dataclass
class CollocationPlan:
    """A planned trip: node times (N+1), node states, per-slice throttles,
    with the receipts of its solve.  Reads as the tracker's plan protocol:
    ``reference(t)``, ``impulses()``, ``mu``, ``ideal_delta_v``."""

    mu: float
    target_radius_m: float
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    propellant_kg: np.ndarray
    throttles: np.ndarray
    fuel: float
    cost: float
    max_defect: float
    converged: bool
    message: str
    iterations: int
    evaluations: int
    jacobian_s: float
    solve_s: float
    transcription: object = field(repr=False, default=None)
    polish_evaluations: int = 0
    #: condensed solve receipts: burn slices, stages, forward-only runs
    structure: tuple = ()
    stages: list = field(default_factory=list)
    forwards: int = 0
    _thrusting_cache: object = field(repr=False, default=None)
    _coast: list = field(repr=False, default_factory=list)

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
    def r2(self) -> float:
        return self.target_radius_m

    def slice_delta_v(self) -> np.ndarray:
        """(N, 3) world thrust delta-v of each slice: ``R B clamp(u) dt /
        m_h`` (the slice's applied force over its mid mass)."""
        tr = self.transcription
        B = actuation_matrix(tr.problem.design, tr.problem.attitude)
        dt = np.diff(self.times)
        m_h = tr.dry + 0.5 * (self.propellant_kg[:-1] + self.propellant_kg[1:])
        return np.stack([B @ clamp_throttles(tr.problem.design, u) * d / m
                         for u, d, m in zip(self.throttles, dt, m_h)])

    @property
    def ideal_delta_v(self) -> float:
        return float(np.sum(np.linalg.norm(self.slice_delta_v(), axis=1)))

    #: A slice whose thrust delta-v is at most this is a coast slice (its
    #: reference is its flow and the tracker's trims take it up).  Below
    #: 1 m/s the slices of an unconverged (polished) plan carry optimizer
    #: residue -- 0.0-0.9 m/s on the kick re-plan -- and each became a burn
    #: the craft slewed to.
    THRUST_THRESHOLD_M_S = 1.0

    def _thrusting(self) -> np.ndarray:
        return (np.linalg.norm(self.slice_delta_v(), axis=1)
                > self.THRUST_THRESHOLD_M_S)

    def impulses(self) -> tuple:
        """One impulse per thrusting slice, at its midpoint: the step the
        reference takes there (see :meth:`reference`)."""
        out = []
        for k in np.flatnonzero(self._thrusting()):
            mid = 0.5 * float(self.times[k] + self.times[k + 1])
            before = self._side(k, mid, after=False)
            after = self._side(k, mid, after=True)
            out.append(PlannedImpulse(mid, (after[:3] / (self.transcription.dry
                                                         + after[6]))
                                      - before[:3] / (self.transcription.dry
                                                      + before[6])))
        return tuple(out)

    def _side(self, k, t, *, after: bool):
        """Coast (u = 0) from node k forward, or from node k+1 backward."""
        tr = self.transcription
        coast = np.zeros(tr.n_u)
        if after:
            return tr.advance(self._state(k + 1), coast,
                              t - float(self.times[k + 1]))
        return tr.advance(self._state(k), coast, t - float(self.times[k]))

    def _state(self, k):
        return np.concatenate([
            (self.transcription.dry + self.propellant_kg[k])
            * self.velocities[k], self.positions[k], [self.propellant_kg[k]]])

    def reference(self, t):
        """``(r, v)`` the plan prescribes at ``t``.

        Coast slice k: the slice's flow (``substeps`` steps of its law) from
        node k over ``t - t_k``.  Thrusting slice k: its impulsive
        equivalent -- coast from node k up to the midpoint, coast BACK from
        node k+1 after it -- so the reference steps exactly by
        :meth:`impulses` at the time the tracker centres its burn (the
        tracker reads a burn against a pre-burn reference; a reference that
        ramps through a long partial-throttle slice was counted twice:
        measured, the kick re-plan's 243 m/s burn was never flown).  For
        constant thrust the midpoint impulse has no position step to first
        order.  After arrival: a coast by the same flow in legs of at most
        60 s from the arrival node; before the start, the start node."""
        tr, t = self.transcription, float(t)
        if t <= self.t_start:
            x = self._state(0)
        elif t < self.t_arrive:
            k = int(np.searchsorted(self.times, t, side="right")) - 1
            if self._thrusting_cache is None:
                self._thrusting_cache = self._thrusting()
            if self._thrusting_cache[k]:
                mid = 0.5 * float(self.times[k] + self.times[k + 1])
                x = self._side(k, t, after=t >= mid)
            else:
                x = tr.advance(self._state(k), self.throttles[k],
                               t - float(self.times[k]))
        else:
            if not self._coast:
                self._coast.append((self.t_arrive, self._state(-1)))
            coast = np.zeros(tr.n_u)
            while self._coast[-1][0] + 60.0 < t:
                t_c, x_c = self._coast[-1]
                self._coast.append((t_c + 60.0, tr.advance(x_c, coast, 60.0)))
            index = int(np.searchsorted([c[0] for c in self._coast], t,
                                        side="right")) - 1
            t_c, x_c = self._coast[index]
            x = tr.advance(x_c, coast, t - t_c) if t > t_c else x_c
        mass = tr.dry + x[6]
        return x[3:6].copy(), x[:3] / mass


def _plan_from(tr, z, result, solve_s):
    cost, _, c, _ = tr.evaluate(z)
    u, dt, x = tr.unpack(z)
    mass = tr.dry + x[:, 6]
    return CollocationPlan(
        mu=float(tr.problem.centers[0].mu_m3_s2),
        target_radius_m=tr.problem.target_radius_m,
        times=tr.t0 + np.concatenate([[0.0], np.cumsum(dt)]),
        positions=x[:, 3:6].copy(), velocities=x[:, :3] / mass[:, None],
        propellant_kg=x[:, 6].copy(), throttles=u.copy(),
        fuel=tr.fuel(u, dt), cost=cost, max_defect=float(np.max(np.abs(c))),
        converged=bool(result.success), message=str(result.message),
        iterations=int(getattr(result, "nit", -1)),
        evaluations=tr.evaluations, jacobian_s=tr.jacobian_s,
        solve_s=solve_s, transcription=tr)


def remainder_warm_start(previous: "CollocationPlan", problem,
                         t0: float, position, velocity, propellant_kg=None):
    """The re-plan's start (decision 4): the previous plan's REMAINDER from
    ``t0`` on the problem's N slices -- its burns as they are (the one in
    progress cut to what is left), its coasts merged into arcs between
    them, plus a burn slice NOW of zero impulse (the correction the present
    state may need; the solve gives it impulse or leaves it empty).  Coast
    slices are spread over the arcs by duration (at least one each).
    Weights as :func:`hohmann_warm_start` sets them (the same cost).
    Returns (HohmannPlan, transcription, u, dt, burn)."""
    hp, tr, z_h = hohmann_warm_start(problem, t0, position, velocity,
                                     propellant_kg)
    N, n_u = tr.N, tr.n_u
    if previous.throttles.shape[1] != n_u:
        raise ValueError("the previous plan was made for another design")
    thrusting = previous._thrusting()
    # (duration, throttles, burn?) in order, from t0 to the old arrival
    pieces = [(problem.dt_bounds_s[0] * 20.0, np.zeros(n_u), True)]
    for k in range(len(previous.throttles)):
        start = max(float(previous.times[k]), float(t0))
        end = float(previous.times[k + 1])
        if end - start <= 0.0:
            continue
        if thrusting[k]:
            pieces.append((end - start, previous.throttles[k], True))
        elif pieces[-1][2]:
            pieces.append((end - start, np.zeros(n_u), False))
        else:
            pieces[-1] = (pieces[-1][0] + end - start, np.zeros(n_u), False)
    burns = sum(1 for piece in pieces if piece[2])
    arcs = np.array([piece[0] for piece in pieces if not piece[2]])
    if burns + arcs.size > N or arcs.size == 0:
        u, dt, _x = tr.unpack(z_h)
        burn = np.zeros(N, bool)
        burn[[0, N - 1]] = True
        return hp, tr, u, dt, burn
    share = np.maximum(1, np.floor((N - burns) * arcs / arcs.sum())
                       ).astype(int)
    while share.sum() > N - burns:
        share[np.argmax(share)] -= 1
    while share.sum() < N - burns:
        share[np.argmax(arcs / share)] += 1
    u, dt, burn = [], [], []
    arc = iter(share)
    for duration, throttles, is_burn in pieces:
        if is_burn:
            u.append(np.asarray(throttles, float))
            dt.append(duration)
            burn.append(True)
            continue
        n = int(next(arc))
        u += [np.zeros(n_u)] * n
        dt += [duration / n] * n
        burn += [False] * n
    return hp, tr, np.array(u), np.array(dt), np.array(burn)


def capture_remainder(previous):
    """Copy the NumPy warm-start data; no native call or live craft read."""
    if not isinstance(previous, CollocationPlan):
        return previous
    arrays = {}
    for name in ("times", "positions", "velocities", "propellant_kg", "throttles"):
        value = np.array(getattr(previous, name), copy=True)
        value.setflags(write=False)
        arrays[name] = value
    tr = _Transcription(previous.transcription.problem, previous.t_start,
                        arrays["positions"][0], arrays["velocities"][0],
                        arrays["propellant_kg"][0])
    return dataclasses.replace(previous, **arrays, transcription=tr,
                               stages=[], _thrusting_cache=None, _coast=[])


def plan_transfer(problem: CollocationProblem, t0: float, position, velocity,
                  *, propellant_kg=None, previous=None,
                  method: str = "condensed", max_iterations: int = 200,
                  tolerance: float | None = None, fixed_first: bool = False,
                  rounds: int = 4, feasibility: float = 1.0e-8):
    """Plan the whole trip from the present state.

    ``method="condensed"`` (the planner): :func:`solve_structured` over the
    burn structure of the warm start -- the Hohmann transfer from here, or
    with ``previous`` (a :class:`CollocationPlan`) that plan's remainder
    (:func:`remainder_warm_start`).

    ``method="SLSQP"`` / ``"trust-constr"``: the full transcription (every
    node, throttle and duration free) from Hohmann by
    ``scipy.optimize.minimize``; kept as the measured reference (it does
    not terminate on the kick re-plan, see :class:`_Condensed`).  A result
    whose scaled defects exceed ``feasibility`` is polished by least
    squares on the defects alone (``plan.polish_evaluations``).

    Returns (plan, hohmann)."""
    from types import SimpleNamespace
    if method == "condensed":
        started = time.perf_counter()
        if isinstance(previous, CollocationPlan):
            hp, tr, u, dt, burn = remainder_warm_start(
                previous, problem, t0, position, velocity, propellant_kg)
            start = "remainder"
        else:
            hp, tr, z0 = hohmann_warm_start(problem, t0, position, velocity,
                                            propellant_kg)
            u, dt, _x = tr.unpack(z0)
            burn = np.zeros(tr.N, bool)
            burn[[0, tr.N - 1]] = True
            start = "hohmann"
        u, dt, x, receipts = solve_structured(
            tr, u, dt, burn, max_iterations=max_iterations,
            tolerance=1.0e-10 if tolerance is None else tolerance,
            fixed_first=fixed_first, rounds=rounds)
        result = receipts["result"]
        z = tr.pack(u, dt, x)
        _cost, _g, c, _J = tr.evaluate(z)
        feasible = float(np.max(np.abs(c))) <= feasibility
        # SLSQP's "positive directional derivative" (status 8) at a
        # feasible point is its line search finding no decrease left: the
        # noise floor of the compiled cost, not a failure
        done = SimpleNamespace(
            success=bool(feasible and (result.success or result.status == 8)),
            message=f"{start}: {result.message}", nit=receipts["iterations"])
        plan = _plan_from(tr, z, done, time.perf_counter() - started)
        plan.structure = receipts["structure"]
        plan.stages = receipts["stages"]
        plan.forwards = receipts["forwards"]
        return plan, hp
    from scipy.optimize import Bounds, NonlinearConstraint, minimize

    hp, tr, z0 = hohmann_warm_start(problem, t0, position, velocity,
                                    propellant_kg)
    tolerance = 1.0e-12 if tolerance is None else tolerance
    started = time.perf_counter()
    cost = lambda z: tr.evaluate(z)[0]
    gradient = lambda z: tr.evaluate(z)[1]
    defects = lambda z: tr.evaluate(z)[2]
    jacobian = lambda z: tr.evaluate(z)[3]
    if method == "SLSQP":
        result = minimize(
            cost, z0, jac=gradient, method="SLSQP", bounds=tr.bounds(),
            constraints=[{"type": "eq", "fun": defects, "jac": jacobian}],
            options={"maxiter": max_iterations, "ftol": tolerance})
    elif method == "trust-constr":
        low, high = np.array(tr.bounds()).T
        result = minimize(
            cost, z0, jac=gradient, method="trust-constr",
            hess=lambda z: np.zeros((z.size, z.size)),
            bounds=Bounds(low, high),
            constraints=[NonlinearConstraint(defects, 0.0, 0.0, jac=jacobian)],
            options={"maxiter": max_iterations, "gtol": tolerance,
                     "xtol": tolerance})
        result.nit = getattr(result, "nit", result.get("niter", -1))
    else:
        raise ValueError(f"unknown method {method!r}")
    z = result.x
    polished = 0
    if float(np.max(np.abs(defects(z)))) > feasibility:
        # the optimizer stopped short of feasible (iteration limit on a flat
        # optimum): restore the dynamics by least squares on the defects
        # alone from where it stopped, same compiled Jacobian, same bounds
        from scipy.optimize import least_squares
        low, high = np.array(tr.bounds()).T
        polish = least_squares(defects, np.clip(z, low, high), jac=jacobian,
                               bounds=(low, high), method="trf",
                               xtol=1e-15, ftol=1e-15, gtol=1e-15,
                               max_nfev=1000)
        z, polished = polish.x, int(polish.nfev)
    plan = _plan_from(tr, z, result, time.perf_counter() - started)
    plan.polish_evaluations = polished
    return plan, hp


def collocation_replanner(problem: CollocationProblem, craft=None, **solve):
    """The tracker's re-planner (``fly(..., replanner=)``): the whole
    remaining trip from the present state to the problem's target circle,
    warm-started from the previous plan's remainder when the previous plan
    is a :class:`CollocationPlan` (decision 4), else from Hohmann.
    ``craft``, when given, supplies the propellant left
    (``craft.propellant_kg``; the tracker's re-planner signature carries no
    mass), which is then also the fuel budget.  ``replanner.plans`` keeps
    every plan made."""
    def replanner(previous, time_s, position_m, velocity_m_s):
        now = problem
        propellant = None
        if craft is not None:
            propellant = float(craft.propellant_kg)
            if problem.burns_propellant:      # the tank left is the budget
                now = dataclasses.replace(problem, fuel_budget=propellant)
        plan, _hohmann = plan_transfer(now, time_s, position_m, velocity_m_s,
                                       propellant_kg=propellant,
                                       previous=previous, **solve)
        replanner.plans.append(plan)
        return plan
    replanner.plans = []
    return replanner


def pointing_proxy(design: CraftDesign) -> CraftDesign:
    """A planning design for a craft that POINTS its strongest thruster
    (the tracker slews it onto each burn): one thruster per world +/- axis
    with that thruster's thrust and kind, the craft's mass and propellant.
    The planner's fuel is then the L1 norm of the thrust (up to sqrt(3) x
    the pointed burn's for a diagonal burn) -- a declared simplification."""
    from orbital_actuation import six_axis_jumper
    strongest = max(design.thrusters, key=lambda t: t.max_thrust_n)
    return six_axis_jumper(strongest.max_thrust_n, design.mass_kg,
                           kind=strongest.kind,
                           propellant_kg=design.propellant_kg,
                           body_size_m=design.body_size_m)
