"""Orbital craft, build step 1: the prototype jumper behind the r()/F() seam.

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``.  The
jumper is a point craft (decision 3) run in the lockstep dt system
(decision 1), manifested exactly the Woodshop way
(``woodshop.newton_world_dt_pieces``): catalogue laws, column spelling,
explicit symplectic-Euler time discretization, ``equation_piece``.  Whatever
the craft later becomes, it meets the trajectory through two functions
(decision 8):

    r() -> (position, velocity)     what the dt state holds now
    F(force)                        the raw applied force (an override)
    throttle(u)                     the control signal: per-thruster
                                    throttles (build step 2)

Pieces, in causal order (one ``RoundNode``, ``sequential`` schedule):

    actuation      applied_force_* <- B @ clamp(u) + raw_force_*
                   (``orbital_actuation``: eq_TS1_2 per declared thruster)
    N4.1 gravity   force_* <- sum over centers of N4.1 + applied_force_*
    N1.2 momentum  momentum_* <- momentum_* + dt * force_*
    N1.1 position  position_* <- position_* + dt * momentum_* / mass;
                   publishes ``max_vel`` (the controller's CFL input, as the
                   Woodshop Newton pieces do; they publish no energy channel)
    thrust cost    thruster{k}_impulse += dt * clamp(u_k) * max_thrust_k;
                   fuel_impulse += dt * (sum_k clamp(u_k) * max_thrust_k
                                         + |raw_force|)

Which one wins: neither -- they superpose.  The throttles drive the
actuation piece; ``F()`` writes ``raw_force_*``, which the same piece adds
on top, unchanged from step 1's meaning (a raw force, kept for tests and
for forces that are not thrusters).  With every throttle at zero ``F()`` is
exactly step 1's seam; with ``F`` at zero the force is the thrusters' alone.

Provenance (decision 5).  The catalogue law ``eq_N4_1`` is the truth for
gravity; it is the same form as the original set's
``F_grav = -mu * (r - c) / |r - c|**3``
(``src/transmogrifier/orbital_transfer.py``,
``OrbitalTransfer.symbolic_transfer_spline``) times the craft mass, with
``G * m_j = mu``.  The thrust-cost integrand is imported from that original
set (its ``force_cost_integral``) unchanged; arc length becomes time
(decision 1), so ``ds`` is discretized as the step's ``dt``.

Mass is a column, constant for now (fuel-burn mass loss is build step 7).
``fuel_impulse`` (name kept from step 1) accumulates the total thrust
impulse in N*s, now per thruster (``thruster{k}_impulse``, decision 7): two
opposed thrusters at full throttle cost fuel though their net force is
zero.  With a specific impulse per thruster kind each becomes propellant
mass (step 7); the total is the consumption term of decision 7's cost.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import sympy as sp

import honorary_engine_equation_catalogue as honorary
from honorary_engine_equation_catalogue import equation_piece

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))
_TURING_EXAMPLES = _TURING_ROOT / "examples"
if str(_TURING_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_TURING_EXAMPLES))

from llvm_dt_system import advance_round, instantiate_system, piece_leaf
from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_graph import ControllerNode, RoundNode
from src.transmogrifier.orbital_transfer import OrbitalTransfer

from orbital_actuation import (
    AXES,
    CraftDesign,
    actuation_force_rhs,
    actuation_matrix,
    clamp_throttles,
    thrust_magnitude,
    thruster_columns,
    thruster_symbols,
)


PIECE_LABELS = ("actuation", "N4.1 gravity", "N1.2 momentum",
                "N1.1 position", "thrust cost")
#: The original set keys its gravitational parameter by the Greek letter mu.
_ORIGINAL_MU_KEY = chr(0x3BC)


@dataclass(frozen=True)
class GravityCenter:
    """A point mass: position (m) and gravitational parameter mu = G*M
    (m^3/s^2), the original set's ``mu`` and ``c``."""

    position_m: tuple[float, float, float]
    mu_m3_s2: float


def _center_symbols(index: int):
    return ({axis: sp.Symbol(f"center{index}_{axis}") for axis in AXES},
            sp.Symbol(f"center{index}_mu"))


def original_transfer_set():
    """The original two-center set, imported as written (provenance)."""
    mu_1, mu_2 = sp.symbols("mu_1 mu_2", positive=True)
    return OrbitalTransfer.symbolic_transfer_spline(
        {_ORIGINAL_MU_KEY: mu_1}, {_ORIGINAL_MU_KEY: mu_2})


def gravity_force_rhs(axis: str, center_count: int):
    """Total N4.1 force on the craft along ``axis`` from every center.

    ``eq_N4_1`` is written on one coordinate; the vector form reads its
    ``Abs(x_j - x_i)`` as the Euclidean distance and its numerator per axis.
    ``m_j`` enters as ``mu / G`` so ``G * m_j`` is the center's mu, as in
    the original set; ``G`` cancels and is not a column.
    """
    law = honorary.eq_N4_1.rhs
    (distance_term,) = law.atoms(sp.Abs)
    position = {name: sp.Symbol(f"position_{name}") for name in AXES}
    total = sp.Integer(0)
    for index in range(center_count):
        center, mu = _center_symbols(index)
        distance = sp.sqrt(sum((center[name] - position[name])**2
                               for name in AXES))
        term = law.xreplace({
            distance_term: distance,
            honorary.m_j: mu / honorary.G,
            honorary.m_i(honorary.t): sp.Symbol("mass"),
            honorary.x_i(honorary.t): position[axis],
            honorary.x_j: center[axis],
        })
        if term.has(honorary.G):
            raise RuntimeError("N4.1: G did not cancel against mu / G")
        total += term
    return total


def thrust_cost_integrand(prefix: str = "applied_force"):
    """The original ``force_cost_integral`` integrand, |F_extra|, spelled
    on the ``{prefix}_*`` columns (the cost piece spells it on
    ``raw_force_*``)."""
    spline = original_transfer_set()
    integral = spline["force_cost_integral"]
    extra = spline["force_components"]["F_extra"]
    return integral.function.xreplace({
        extra[index]: sp.Symbol(f"{prefix}_{axis}")
        for index, axis in enumerate(AXES)
    })


def orbital_jumper_dt_pieces(center_count: int, thruster_count: int = 0,
                             batch: int = 1):
    """Manifest actuation -> N4.1 -> N1.2 -> N1.1 (+ thrust cost) as
    batched LLVM pieces.

    Substitutions into the catalogue equations followed only by
    symplectic-Euler time discretization, as ``newton_world_dt_pieces``.
    """
    if center_count < 0:
        raise ValueError("center_count must be non-negative")
    if thruster_count < 0:
        raise ValueError("thruster_count must be non-negative")
    dt, mass = sp.symbols("dt mass")
    applied = {axis: sp.Symbol(f"applied_force_{axis}") for axis in AXES}
    raw = {axis: sp.Symbol(f"raw_force_{axis}") for axis in AXES}

    actuation = equation_piece(
        f"orbital_jumper_actuation_t{thruster_count}", tuple(
            sp.Eq(sp.Symbol(f"applied_force_{axis}_next"),
                  actuation_force_rhs(axis, thruster_count) + raw[axis],
                  evaluate=False)
            for axis in AXES), batch=batch)
    force = {axis: sp.Symbol(f"force_{axis}") for axis in AXES}
    momentum = {axis: sp.Symbol(f"momentum_{axis}") for axis in AXES}
    position = {axis: sp.Symbol(f"position_{axis}") for axis in AXES}

    gravity = equation_piece(f"orbital_jumper_gravity_c{center_count}", tuple(
        sp.Eq(sp.Symbol(f"force_{axis}_next"),
              gravity_force_rhs(axis, center_count) + applied[axis],
              evaluate=False)
        for axis in AXES), batch=batch)

    def momentum_next(axis):
        derivative = honorary.eq_N1_2.rhs.xreplace({
            honorary.F_i(honorary.t): force[axis],
        })
        return momentum[axis] + dt * derivative

    momentum_piece = equation_piece("orbital_jumper_momentum", tuple(
        sp.Eq(sp.Symbol(f"momentum_{axis}_next"), momentum_next(axis),
              evaluate=False)
        for axis in AXES), batch=batch)

    def position_next(axis):
        velocity = honorary.eq_N1_1.rhs.xreplace({
            honorary.m_i(honorary.t): mass,
            honorary.p_i(honorary.t): momentum[axis],
        })
        return position[axis] + dt * velocity

    speed = sp.sqrt(sum(momentum[axis]**2 for axis in AXES)) / mass
    position_piece = equation_piece("orbital_jumper_position", (
        *(sp.Eq(sp.Symbol(f"position_{axis}_next"), position_next(axis),
                evaluate=False) for axis in AXES),
        sp.Eq(sp.Symbol("max_vel"), speed, evaluate=False),
    ), batch=batch)

    # per-thruster fuel (decision 7): each thruster's TS1.2 thrust is its
    # impulse rate, whatever the others do; the raw override costs |F| as
    # in the original set's integrand
    fuel = sp.Symbol("fuel_impulse")
    per_thruster = []
    for index in range(thruster_count):
        impulse = thruster_symbols(index)["impulse"]
        per_thruster.append(sp.Eq(
            sp.Symbol(f"{impulse.name}_next"),
            impulse + dt * thrust_magnitude(index), evaluate=False))
    total_rate = (sum((thrust_magnitude(index)
                       for index in range(thruster_count)), sp.Integer(0))
                  + thrust_cost_integrand("raw_force"))
    cost_piece = equation_piece(
        f"orbital_jumper_thrust_cost_t{thruster_count}", (
            *per_thruster,
            sp.Eq(sp.Symbol("fuel_impulse_next"), fuel + dt * total_rate,
                  evaluate=False),
        ), batch=batch)
    return actuation, gravity, momentum_piece, position_piece, cost_piece


class OrbitalJumper:
    """One point craft in one persistent lockstep dt state.

    The dt system owns the state, the round and the controller's
    continuation; the graph is built and instantiated once, here, and every
    round only writes the seam's force columns and calls ``advance_round``.
    ``length_scale_m`` is the controller's ``dx``: the CFL bound is
    ``dt <= cfl * dx / max_vel``.

    The craft is a :class:`orbital_actuation.CraftDesign` (thrusters +
    mass).  ``mass_kg`` alone is step 1's thrusterless jumper.
    """

    def __init__(self, centers: Sequence[GravityCenter], *,
                 position_m, velocity_m_s, length_scale_m: float,
                 window_s: float, mass_kg: float | None = None,
                 design: CraftDesign | None = None, cfl: float = 0.5):
        if (mass_kg is None) == (design is None):
            raise ValueError("give exactly one of mass_kg (no thrusters) "
                             "or design")
        self.design = (CraftDesign((), float(mass_kg), identity="jumper")
                       if design is None else design)
        self.centers = tuple(centers)
        self.length_scale_m = float(length_scale_m)
        self.window_s = float(window_s)
        self.pieces = orbital_jumper_dt_pieces(
            len(self.centers), self.design.thruster_count)
        columns = self._initial_columns(
            self.design.mass_kg, position_m, velocity_m_s)
        # Woodshop's targets.  The Newton pieces publish no energy channel, so
        # every participant reads HOLD ("do not grow"); dt is the CFL bound on
        # ``max_vel``.  The window's clipped landing substep no longer becomes
        # the continuation (``run_superstep``; see
        # ``turing/docs/DT_LAST_SUBSTEP_RATCHET_CONTINUATION.md``).
        targets = Targets(cfl=float(cfl), div_max=1.0e9, mass_max=1.0e-3,
                          energy_exchange_fraction=0.2)
        # The first attempt is the controller's own CFL proposal at the
        # initial state; every later attempt is the controller's continuation.
        speed = float(np.linalg.norm(np.asarray(velocity_m_s, dtype=float)))
        dt_init = self.window_s if speed <= 0.0 else min(
            self.window_s, targets.cfl * self.length_scale_m / speed)
        self.dt_graph = RoundNode(
            plan=SuperstepPlan(round_max=self.window_s, dt_init=dt_init),
            controller=ControllerNode(
                ctrl=STController(dt_min=self.window_s * 1.0e-6),
                targets=targets,
                dx=self.length_scale_m,
            ),
            children=[piece_leaf(piece, label=label)
                      for piece, label in zip(self.pieces, PIECE_LABELS)],
            schedule="sequential",
            label="orbital-jumper",
        )
        self.dt_state = instantiate_system(self.dt_graph, columns)
        self.time_s = 0.0

    def _initial_columns(self, mass_kg, position_m, velocity_m_s) -> dict:
        mass = float(mass_kg)
        if mass <= 0.0:
            raise ValueError("jumper mass must be positive")
        position = np.asarray(position_m, dtype=float).reshape(3)
        velocity = np.asarray(velocity_m_s, dtype=float).reshape(3)
        columns = {"mass": np.full(1, mass),
                   "fuel_impulse": np.zeros(1)}
        for index, axis in enumerate(AXES):
            columns[f"position_{axis}"] = np.full(1, position[index])
            columns[f"momentum_{axis}"] = np.full(1, mass * velocity[index])
            columns[f"force_{axis}"] = np.zeros(1)
            columns[f"applied_force_{axis}"] = np.zeros(1)
            columns[f"raw_force_{axis}"] = np.zeros(1)
        columns.update(thruster_columns(self.design))
        for index, center in enumerate(self.centers):
            for slot, axis in enumerate(AXES):
                columns[f"center{index}_{axis}"] = np.full(
                    1, float(center.position_m[slot]))
            columns[f"center{index}_mu"] = np.full(1, float(center.mu_m3_s2))
        return columns

    def _span(self, name: str) -> np.ndarray:
        return getattr(self.dt_state, name)

    # ------------------------------------------------------------- the seam
    def r(self) -> tuple[np.ndarray, np.ndarray]:
        """Current position (m) and velocity (m/s, N1.1: p / m)."""
        position = np.asarray([float(self._span(f"position_{axis}")[0])
                               for axis in AXES])
        momentum = np.asarray([float(self._span(f"momentum_{axis}")[0])
                               for axis in AXES])
        return position, momentum / float(self._span("mass")[0])

    def F(self, force_n) -> None:
        """Set the raw applied force (N) the next round integrates.

        It superposes on the thrusters' ``B @ clamp(u)`` in the actuation
        piece; with every throttle at zero it is the whole applied force."""
        force = np.asarray(force_n, dtype=float).reshape(3)
        for index, axis in enumerate(AXES):
            self._span(f"raw_force_{axis}")[...] = force[index]

    def throttle(self, throttles) -> None:
        """Set the per-thruster throttles (design order) the next round
        applies; the piece clamps each to its declared range."""
        u = np.asarray(throttles, dtype=float).reshape(
            self.design.thruster_count)
        for index, value in enumerate(u):
            self._span(f"thruster{index}_throttle")[...] = value

    def throttles(self) -> np.ndarray:
        """The commanded (unclamped) throttles."""
        return np.asarray([float(self._span(f"thruster{k}_throttle")[0])
                           for k in range(self.design.thruster_count)])

    def applied_force(self) -> np.ndarray:
        """The applied force (N) the last substep integrated."""
        return np.asarray([float(self._span(f"applied_force_{axis}")[0])
                           for axis in AXES])

    def commanded_force(self) -> np.ndarray:
        """``B @ clamp(u) + raw`` evaluated on the host from the design --
        what the next round's actuation piece will write."""
        raw = np.asarray([float(self._span(f"raw_force_{axis}")[0])
                          for axis in AXES])
        return (actuation_matrix(self.design)
                @ clamp_throttles(self.design, self.throttles()) + raw)

    # ------------------------------------------------------------ the round
    def advance(self, window_s: float | None = None):
        """One lockstep round over the persistent state; must land."""
        requested = self.window_s if window_s is None else float(window_s)
        advanced, dt_next, telemetry = advance_round(self.dt_state, requested)
        if not math.isclose(float(advanced), requested, rel_tol=0.0,
                            abs_tol=1.0e-12 * max(1.0, requested)):
            raise RuntimeError(
                f"orbital jumper advanced {float(advanced)} of {requested}")
        self.time_s += float(advanced)
        return advanced, dt_next, telemetry

    @property
    def mass_kg(self) -> float:
        return float(self._span("mass")[0])

    @property
    def fuel_impulse_n_s(self) -> float:
        return float(self._span("fuel_impulse")[0])

    @property
    def thruster_impulses_n_s(self) -> np.ndarray:
        """Per-thruster accumulated thrust impulse (N*s), design order."""
        return np.asarray([float(self._span(f"thruster{k}_impulse")[0])
                           for k in range(self.design.thruster_count)])

    @property
    def substeps(self) -> int:
        """Accepted substeps so far: the first piece's call count (the plan
        rolls back only on error channels, which no piece publishes)."""
        return int(self.dt_state.wall_cost_ledger.calls[0])
