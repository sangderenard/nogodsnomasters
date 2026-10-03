"""Orbital craft, build steps 1 and 7: the jumper behind the r()/F() seam,
with propellant mass and attitude.

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``.  The
jumper is a craft (decision 3) run in the lockstep dt system (decision 1),
manifested exactly the Woodshop way (``woodshop.newton_world_dt_pieces``):
catalogue laws, column spelling, explicit symplectic-Euler time
discretization, ``equation_piece``.  Whatever the craft later becomes, it
meets the trajectory through two functions (decision 8):

    r() -> (position, velocity)     what the dt state holds now
    F(force)                        the raw applied force (an override)
    throttle(u)                     the control signal: per-thruster
                                    throttles (build step 2)

Pieces, in causal order (one ``RoundNode``, ``sequential`` schedule):

    supply         propellant_supply <- step average of H(propellant)
    actuation      applied_force_* <- R @ B_craft @ (supply clamp(u)) + raw
                   torque_*        <- sum_k r_k x F_k (craft frame)
                   propellant_flow <- supply * sum_k F_k / (I_sp_k g_0)
                   (``orbital_actuation``: TS1.2 + TS1.4 per thruster)
    N4.1 gravity   force_* <- sum over centers of N4.1 + applied_force_*
    momentum       N7.2  momentum_* <- momentum_* + dt * (force_*
                                        - propellant_flow * momentum_*/mass)
                   mass balance: propellant_mass -= min(propellant_mass,
                                               dt * propellant_flow),
                                 mass = dry_mass + propellant_mass
                   N1.6  angular_velocity_* (craft frame, principal axes)
                         += dt * (torque - w x I w) / I
    position       N1.1  position_* += dt * momentum_* / mass (new mass);
                   N1.3  attitude R <- R @ cayley(dt * skew(w_new));
                   publishes ``max_vel`` (the controller's CFL input) and
                   ``dt_limit`` = attitude_step_max / |w| (the attitude's
                   own bound on the next step; +inf when not rotating)
    thrust cost    thruster{k}_impulse += dt * delivered F_k;
                   fuel_impulse += dt * (sum_k delivered F_k + |raw_force|)

Which one wins: neither -- they superpose.  The throttles drive the
actuation piece; ``F()`` writes ``raw_force_*`` (world frame), which the same
piece adds on top, unchanged from step 1's meaning (a raw force, kept for
tests and for forces that are not thrusters; it burns no propellant).

Variable mass (decision 9).  ``eq_N7_2``'s outflow momentum term,
``-sum m_out v_out`` with the exhaust's absolute velocity
``v_out = v - c d_world``, splits into the TS1.2 thrust ``+F d_world``
(already in ``force_*`` through the actuation piece) and the share that
leaves moving with the craft, ``-m_dot v``; ``N7.2`` here spells that second
part.  With the mass updated in the same piece from the same old state, the
discrete velocity obeys ``v_new = v + dt * force / mass_new`` exactly, so
the burn integrates Tsiolkovsky (TS2.1) to first order in dt.

Attitude kinematics.  ``eq_N1_3``, ``dR/dt = R skew(w)``, with ``w`` frozen
at the step's new rate (symplectic Euler's order: rate first, then the
coordinate), is integrated by the Cayley map
``(I + K)(I - K)^-1 = I + 2 (K + K^2) / (1 + |k|^2)``, ``K = dt/2 skew(w)``:
the closed form of the implicit midpoint rule for this linear law.  It keeps
``eq_N1_4``/``eq_N1_5`` (R orthonormal, det 1) to rounding, where the
explicit Euler step stretches R (and the thrust it carries) by
``sqrt(1 + dt^2 |w|^2)`` per step.

Provenance (decision 5).  The catalogue law ``eq_N4_1`` is the truth for
gravity; it is the same form as the original set's
``F_grav = -mu * (r - c) / |r - c|**3``
(``src/transmogrifier/orbital_transfer.py``,
``OrbitalTransfer.symbolic_transfer_spline``) times the craft mass, with
``G * m_j = mu``.  The thrust-cost integrand is imported from that original
set (its ``force_cost_integral``) unchanged; arc length becomes time
(decision 1), so ``ds`` is discretized as the step's ``dt``.
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
    DT,
    PROPELLANT_FLOW,
    PROPELLANT_MASS,
    PROPELLANT_SUPPLY,
    CraftDesign,
    actuation_force_rhs,
    actuation_matrix,
    actuation_torque_rhs,
    attitude_symbols,
    clamp_throttles,
    delivered_thrust,
    principal_inertia,
    propellant_flow_rhs,
    propellant_supply_rhs,
    thruster_columns,
    thruster_symbols,
)


PIECE_LABELS = ("propellant supply", "actuation", "N4.1 gravity",
                "N7.2/N1.6 momentum", "N1.1/N1.3 position", "thrust cost")
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




def _skew(vector):
    """The cross-product matrix ``[w]_x`` that ``eq_N1_3``'s ``skew``
    placeholder names."""
    x, y, z = vector
    return sp.Matrix(((0, -z, y), (z, 0, -x), (-y, x, 0)))


def variable_mass_momentum_rate(axis: str):
    """``eq_N7_2`` along ``axis``: no inflow; the outflow term's share that
    leaves with the craft's velocity, ``propellant_flow * p / m`` (the
    TS1.2 part of the exhaust momentum flux is the thrust in ``force_*``)."""
    law = honorary.eq_N7_2
    inflow, outflow = sorted(law.rhs.atoms(sp.Sum),
                             key=lambda term: term.has(honorary.m_out))
    if not (outflow.has(honorary.m_out) and inflow.has(honorary.m_in)):
        raise RuntimeError("eq_N7_2 no longer has inflow and outflow sums")
    momentum = sp.Symbol(f"momentum_{axis}")
    return law.rhs.xreplace({
        honorary.F_ext: sp.Symbol(f"force_{axis}"),
        inflow: sp.Integer(0),
        outflow: PROPELLANT_FLOW * momentum / sp.Symbol("mass"),
    })


def euler_rate_rhs(axis: str):
    """``eq_N1_6`` solved for ``d omega_B / dt`` along a principal axis:
    ``(tau - (w x I w)) / I`` with ``I`` diagonal (``inertia_*``)."""
    t = honorary.t
    (rate,) = sp.solve(honorary.eq_N1_6,
                       sp.Derivative(honorary.omega_B(t), t))
    crosses = [term for term in rate.atoms(sp.Function)
               if term.func == honorary.cross]
    if len(crosses) != 1:
        raise RuntimeError("eq_N1_6 no longer has one w x I w term")
    omega = [sp.Symbol(f"angular_velocity_{a}") for a in AXES]
    inertia = [sp.Symbol(f"inertia_{a}") for a in AXES]
    momentum = [inertia[i] * omega[i] for i in range(3)]
    i = AXES.index(axis)
    j, k = (i + 1) % 3, (i + 2) % 3
    gyroscopic = omega[j] * momentum[k] - omega[k] * momentum[j]
    spelled = rate.xreplace({
        crosses[0]: gyroscopic,
        honorary.tau_B(t): sp.Symbol(f"torque_{axis}"),
        honorary.I_B(t): inertia[i],
    })
    if spelled.has(honorary.omega_B, honorary.I_B, honorary.tau_B):
        raise RuntimeError("N1.6 left an unspelled body function")
    return spelled


def attitude_next_rhs():
    """``eq_N1_3`` stepped by the Cayley map at the new rate: ``R_next =
    R @ (I + 2 (K + K^2) / (1 + |k|^2))``, ``K = skew(k)``, ``k = dt/2 w``.
    Returns ``{(row, col): expression}``."""
    t = honorary.t
    law = honorary.eq_N1_3.rhs
    factors = law.as_ordered_factors()
    if (len(factors) != 2 or factors[0] != honorary.R(t)
            or getattr(factors[1].func, "__name__", "") != "skew"):
        raise RuntimeError("eq_N1_3 is no longer R * skew(omega_B)")
    half = [DT / 2 * sp.Symbol(f"angular_velocity_{a}") for a in AXES]
    K = _skew(half)
    norm2 = sum(value**2 for value in half)
    cayley = sp.eye(3) + 2 * (K + K * K) / (1 + norm2)
    rotation = sp.Matrix(3, 3, lambda r, c: attitude_symbols()[(r, c)])
    stepped = rotation * cayley
    return {(r, c): stepped[r, c] for r in range(3) for c in range(3)}


#: The attitude's declared step bound: the largest rotation one substep may
#: take, ``|omega| dt <= ATTITUDE_STEP_RAD``.  The Cayley step turns
#: ``2 atan(|omega| dt / 2)`` where the rate turns ``|omega| dt``; at 1/8 rad
#: the shortfall is ``(|omega| dt)^2 / 12`` = 0.13 %.
ATTITUDE_STEP_RAD = 0.125
ATTITUDE_STEP_MAX = sp.Symbol("attitude_step_max")


def attitude_dt_limit_rhs():
    """The attitude piece's own stability limit, published as ``dt_limit``
    (``llvm_dt_system``'s per-piece floor, folded by min into
    ``Metrics.dt_limit``, which bounds the controller's next step):
    ``attitude_step_max / |omega|`` at the new rate.  A craft that does not
    rotate publishes ``+inf``, the dt system's own "no bound"."""
    rate = sp.sqrt(sum(sp.Symbol(f"angular_velocity_{a}")**2 for a in AXES))
    return ATTITUDE_STEP_MAX / rate


def orbital_jumper_dt_pieces(center_count: int, thruster_count: int = 0,
                             batch: int = 1):
    """Manifest supply -> actuation -> N4.1 -> N7.2/N1.6 -> N1.1/N1.3
    (+ thrust cost) as batched LLVM pieces.

    Substitutions into the catalogue equations followed only by time
    discretization: symplectic Euler, as ``newton_world_dt_pieces``, and the
    Cayley step for the attitude coordinate (module docstring).
    """
    if center_count < 0:
        raise ValueError("center_count must be non-negative")
    if thruster_count < 0:
        raise ValueError("thruster_count must be non-negative")
    dt, mass = DT, sp.Symbol("mass")
    applied = {axis: sp.Symbol(f"applied_force_{axis}") for axis in AXES}
    raw = {axis: sp.Symbol(f"raw_force_{axis}") for axis in AXES}

    # the supply is its own piece: a piece's rhs reads its inputs, so the
    # actuation piece must find this substep's supply already in the column
    supply = equation_piece(
        f"orbital_craft_propellant_supply_t{thruster_count}", (
            sp.Eq(sp.Symbol("propellant_supply_next"),
                  propellant_supply_rhs(thruster_count), evaluate=False),
        ), batch=batch)
    actuation = equation_piece(
        f"orbital_craft_actuation_t{thruster_count}", (
            *(sp.Eq(sp.Symbol(f"applied_force_{axis}_next"),
                    actuation_force_rhs(axis, thruster_count) + raw[axis],
                    evaluate=False)
              for axis in AXES),
            *(sp.Eq(sp.Symbol(f"torque_{axis}_next"),
                    actuation_torque_rhs(axis, thruster_count),
                    evaluate=False)
              for axis in AXES),
            sp.Eq(sp.Symbol("propellant_flow_next"),
                  propellant_flow_rhs(thruster_count), evaluate=False),
        ), batch=batch)
    momentum = {axis: sp.Symbol(f"momentum_{axis}") for axis in AXES}
    position = {axis: sp.Symbol(f"position_{axis}") for axis in AXES}
    omega = {axis: sp.Symbol(f"angular_velocity_{axis}") for axis in AXES}

    gravity = equation_piece(f"orbital_jumper_gravity_c{center_count}", tuple(
        sp.Eq(sp.Symbol(f"force_{axis}_next"),
              gravity_force_rhs(axis, center_count) + applied[axis],
              evaluate=False)
        for axis in AXES), batch=batch)

    # mass balance: the step draws dt * propellant_flow, which the supply
    # law already holds to at most the tank; spelling the draw as
    # min(tank, dt * flow) is the same quantity with the empty step's
    # rounding (P - dt * (P / (dt D)) * D = -1e-16) removed
    propellant_next = PROPELLANT_MASS - sp.Min(PROPELLANT_MASS,
                                               dt * PROPELLANT_FLOW)
    momentum_piece = equation_piece("orbital_craft_momentum", (
        *(sp.Eq(sp.Symbol(f"momentum_{axis}_next"),
                momentum[axis] + dt * variable_mass_momentum_rate(axis),
                evaluate=False) for axis in AXES),
        sp.Eq(sp.Symbol("propellant_mass_next"), propellant_next,
              evaluate=False),
        sp.Eq(sp.Symbol("mass_next"), sp.Symbol("dry_mass") + propellant_next,
              evaluate=False),
        *(sp.Eq(sp.Symbol(f"angular_velocity_{axis}_next"),
                omega[axis] + dt * euler_rate_rhs(axis), evaluate=False)
          for axis in AXES),
    ), batch=batch)

    def position_next(axis):
        velocity = honorary.eq_N1_1.rhs.xreplace({
            honorary.m_i(honorary.t): mass,
            honorary.p_i(honorary.t): momentum[axis],
        })
        return position[axis] + dt * velocity

    speed = sp.sqrt(sum(momentum[axis]**2 for axis in AXES)) / mass
    attitude = attitude_next_rhs()
    names = attitude_symbols()
    position_piece = equation_piece("orbital_craft_position", (
        *(sp.Eq(sp.Symbol(f"position_{axis}_next"), position_next(axis),
                evaluate=False) for axis in AXES),
        *(sp.Eq(sp.Symbol(f"{names[slot].name}_next"), attitude[slot],
                evaluate=False) for slot in sorted(attitude)),
        sp.Eq(sp.Symbol("max_vel"), speed, evaluate=False),
        sp.Eq(sp.Symbol("dt_limit"), attitude_dt_limit_rhs(),
              evaluate=False),
    ), batch=batch)

    # per-thruster fuel (decision 7): each thruster's delivered TS1.2
    # thrust is its impulse rate, whatever the others do; the raw override
    # costs |F| as in the original set's integrand
    fuel = sp.Symbol("fuel_impulse")
    per_thruster = []
    for index in range(thruster_count):
        impulse = thruster_symbols(index)["impulse"]
        per_thruster.append(sp.Eq(
            sp.Symbol(f"{impulse.name}_next"),
            impulse + dt * delivered_thrust(index), evaluate=False))
    total_rate = (sum((delivered_thrust(index)
                       for index in range(thruster_count)), sp.Integer(0))
                  + thrust_cost_integrand("raw_force"))
    cost_piece = equation_piece(
        f"orbital_craft_thrust_cost_t{thruster_count}", (
            *per_thruster,
            sp.Eq(sp.Symbol("fuel_impulse_next"), fuel + dt * total_rate,
                  evaluate=False),
        ), batch=batch)
    return (supply, actuation, gravity, momentum_piece, position_piece,
            cost_piece)


def _attitude_array(attitude) -> np.ndarray:
    rotation = (np.eye(3) if attitude is None
                else np.asarray(attitude, dtype=float).reshape(3, 3))
    if (np.abs(rotation.T @ rotation - np.eye(3)).max() > 1.0e-12
            or not math.isclose(float(np.linalg.det(rotation)), 1.0,
                                rel_tol=0.0, abs_tol=1.0e-12)):
        raise ValueError("attitude must be a rotation matrix (N1.4, N1.5)")
    return rotation


class OrbitalJumper:
    """One craft -- or ``batch`` craft -- in one persistent lockstep dt state.

    The dt system owns the state, the round and the controller's
    continuation; the graph is built and instantiated once, here, and every
    round only writes the seam's force and throttle columns and calls
    ``advance_round``.  ``length_scale_m`` is the controller's ``dx``: the
    CFL bound is ``dt <= cfl * dx / max_vel``.

    The craft is a :class:`orbital_actuation.CraftDesign` (thrusters, wet
    mass, propellant, body box).  ``mass_kg`` alone is step 1's thrusterless
    jumper.  ``attitude`` is the initial ``R`` (world = R @ craft; identity
    by default) and ``angular_velocity_rad_s`` the initial craft-frame rate.

    ``batch``: the number of lanes.  Every column is one cell per lane and
    the pieces are built at that batch, so ``batch`` bodies with the same
    design and centers run in ONE dt state at one shared dt (the dt system
    folds ``max_vel`` by max and ``dt_limit`` by min over the lanes).  Per-
    lane initial values carry a leading lane axis (``position_m`` /
    ``velocity_m_s`` (batch, 3), ``mass_kg`` (batch,) for a thrusterless
    jumper, ``attitude`` (batch, 3, 3), ...); a single value is every lane's.
    With ``batch == 1`` every seam reading has the shape it always had; with
    ``batch > 1`` each carries the leading lane axis.
    """

    def __init__(self, centers: Sequence[GravityCenter], *,
                 position_m, velocity_m_s, length_scale_m: float,
                 window_s: float, mass_kg=None,
                 design: CraftDesign | None = None, cfl: float = 0.5,
                 attitude=None, angular_velocity_rad_s=(0.0, 0.0, 0.0),
                 attitude_step_rad: float = ATTITUDE_STEP_RAD,
                 batch: int = 1):
        if (mass_kg is None) == (design is None):
            raise ValueError("give exactly one of mass_kg (no thrusters) "
                             "or design")
        if int(batch) != batch or int(batch) < 1:
            raise ValueError("batch must be a positive integer")
        self.batch = int(batch)
        self._lane_mass_kg = None
        if design is None:
            lane_mass = self._lanes(mass_kg)
            if not np.all(lane_mass > 0.0):
                raise ValueError("every lane's mass must be positive")
            # the shared design names the craft; the lanes' own masses
            # are the columns
            design = CraftDesign((), float(lane_mass[0]), identity="jumper")
            self._lane_mass_kg = lane_mass
        self.design = design
        self.centers = tuple(centers)
        self.length_scale_m = float(length_scale_m)
        self.window_s = float(window_s)
        self.inertia_kg_m2 = self._inertia_kg_m2()
        if not float(attitude_step_rad) > 0.0:
            raise ValueError("attitude_step_rad must be positive (math.inf "
                             "declares no attitude bound)")
        self.attitude_step_rad = float(attitude_step_rad)
        self.pieces, labels = self._dt_pieces()
        attitudes = np.broadcast_to(
            np.eye(3) if attitude is None
            else np.asarray(attitude, dtype=float),
            (self.batch, 3, 3))
        for rotation in attitudes:
            _attitude_array(rotation)
        columns = self._initial_columns(
            position_m, velocity_m_s, attitudes, angular_velocity_rad_s)
        # Woodshop's targets.  The Newton pieces publish no energy channel, so
        # every participant reads HOLD ("do not grow"); dt is the CFL bound on
        # ``max_vel``.  The window's clipped landing substep no longer becomes
        # the continuation (``run_superstep``; see
        # ``turing/docs/DT_LAST_SUBSTEP_RATCHET_CONTINUATION.md``).
        targets = Targets(cfl=float(cfl), div_max=1.0e9, mass_max=1.0e-3,
                          energy_exchange_fraction=0.2)
        # The first attempt is the controller's own CFL proposal at the
        # initial state (the fastest lane's); every later attempt is the
        # controller's continuation.
        speed = float(np.max(np.linalg.norm(
            self._lanes(velocity_m_s, 3), axis=1)))
        dt_init = self.window_s if speed <= 0.0 else min(
            self.window_s, targets.cfl * self.length_scale_m / speed)
        # ... and the attitude piece's own bound at the initial rate (the
        # first attempt has no publication behind it; every later one is
        # bounded by the piece's published ``dt_limit``)
        rate = float(np.max(np.linalg.norm(
            self._lanes(angular_velocity_rad_s, 3), axis=1)))
        if rate > 0.0:
            dt_init = min(dt_init, self.attitude_step_rad / rate)
        # Symplectic Euler as leapfrog: the momentum column is the HALF-STEP
        # momentum (kick by dt * F(x_n), then drift by the new velocity),
        # so it starts half a first substep behind the initial state (see
        # _half_step_kick and r()).
        self.piece_labels = tuple(labels)
        self._dt_first = dt_init
        kick = self._half_step_kick(columns, 0.5 * dt_init)
        for index, axis in enumerate(AXES):
            columns[f"momentum_{axis}"] = (columns[f"momentum_{axis}"]
                                           - kick[:, index])
        self.dt_graph = RoundNode(
            plan=SuperstepPlan(round_max=self.window_s, dt_init=dt_init),
            controller=ControllerNode(
                ctrl=STController(dt_min=self.window_s * 1.0e-6),
                targets=targets,
                dx=self.length_scale_m,
            ),
            children=[piece_leaf(piece, label=label)
                      for piece, label in zip(self.pieces, labels)],
            schedule="sequential",
            label="orbital-jumper",
        )
        self.dt_state = instantiate_system(self.dt_graph, columns)
        self.time_s = 0.0

    # ----------------------------------------------------------- the hooks
    def _inertia_kg_m2(self) -> np.ndarray:
        """The inertia the ``inertia_*`` columns start from (a craft whose
        mass properties are laws of its own state overrides this)."""
        return principal_inertia(self.design)

    def _dt_pieces(self):
        """``(pieces, labels)`` in causal order (a craft with its own
        actuation and mass-property laws overrides this)."""
        return (orbital_jumper_dt_pieces(len(self.centers),
                                         self.design.thruster_count,
                                         batch=self.batch),
                PIECE_LABELS)

    def _lanes(self, value, width: int | None = None) -> np.ndarray:
        """``value`` as one row per lane, ``(batch,)`` or ``(batch, width)``;
        a single value is broadcast to every lane."""
        shape = (self.batch,) if width is None else (self.batch, width)
        return np.array(np.broadcast_to(np.asarray(value, dtype=float),
                                        shape))

    def _initial_columns(self, position_m, velocity_m_s, attitudes,
                         angular_velocity_rad_s) -> dict:
        design = self.design
        lanes = self.batch
        mass = (self._lane_mass_kg if self._lane_mass_kg is not None
                else self._lanes(float(design.mass_kg)))
        position = self._lanes(position_m, 3)
        velocity = self._lanes(velocity_m_s, 3)
        omega = self._lanes(angular_velocity_rad_s, 3)
        columns = {"mass": mass.copy(),
                   "dry_mass": mass - float(design.propellant_kg),
                   "propellant_mass": np.full(lanes, design.propellant_kg),
                   "propellant_flow": np.zeros(lanes),
                   "propellant_supply": np.ones(lanes),
                   "attitude_step_max": np.full(lanes,
                                                self.attitude_step_rad),
                   "fuel_impulse": np.zeros(lanes)}
        if self._lane_mass_kg is None:
            # the design's own dry mass, not the difference recomputed
            columns["dry_mass"] = np.full(lanes, design.dry_mass_kg)
        for index, axis in enumerate(AXES):
            columns[f"position_{axis}"] = position[:, index].copy()
            columns[f"momentum_{axis}"] = mass * velocity[:, index]
            columns[f"force_{axis}"] = np.zeros(lanes)
            columns[f"applied_force_{axis}"] = np.zeros(lanes)
            columns[f"raw_force_{axis}"] = np.zeros(lanes)
            columns[f"torque_{axis}"] = np.zeros(lanes)
            columns[f"angular_velocity_{axis}"] = omega[:, index].copy()
            columns[f"inertia_{axis}"] = np.full(
                lanes, float(self.inertia_kg_m2[index]))
        for (row, col), symbol in attitude_symbols().items():
            columns[symbol.name] = np.array(attitudes[:, row, col],
                                            dtype=float)
        for name, value in thruster_columns(design).items():
            columns[name] = np.full(lanes, float(value[0]))
        for index, center in enumerate(self.centers):
            for slot, axis in enumerate(AXES):
                columns[f"center{index}_{axis}"] = np.full(
                    lanes, float(center.position_m[slot]))
            columns[f"center{index}_mu"] = np.full(lanes,
                                                   float(center.mu_m3_s2))
        return columns

    def _span(self, name: str) -> np.ndarray:
        return getattr(self.dt_state, name)

    def _scalar(self, name: str):
        """A column: a float for one lane, ``(batch,)`` for several."""
        span = self._span(name)
        if self.batch == 1:
            return float(span[0])
        return np.array(np.asarray(span, dtype=float)[:self.batch])

    def _vector(self, prefix: str) -> np.ndarray:
        if self.batch == 1:
            return np.asarray([float(self._span(f"{prefix}_{axis}")[0])
                               for axis in AXES])
        return np.stack([np.asarray(self._span(f"{prefix}_{axis}"),
                                    dtype=float)[:self.batch]
                         for axis in AXES], axis=1)

    def _per_thruster(self, suffix: str) -> np.ndarray:
        """``thruster{k}_{suffix}``: ``(k,)`` for one lane, else
        ``(batch, k)``."""
        count = self.design.thruster_count
        if self.batch == 1:
            return np.asarray([float(self._span(f"thruster{k}_{suffix}")[0])
                               for k in range(count)])
        out = np.empty((self.batch, count))
        for k in range(count):
            out[:, k] = np.asarray(self._span(f"thruster{k}_{suffix}"),
                                   dtype=float)[:self.batch]
        return out

    def _write_lanes(self, name: str, values) -> None:
        span = self._span(name)
        for lane, value in enumerate(self._lanes(values)):
            span[lane] = value

    # ------------------------------------------------------------- the seam
    def _gravity_force(self, columns) -> np.ndarray:
        """``(batch, 3)``: the compiled N4.1 gravity piece's force at the
        positions in ``columns``, with no applied force (the piece itself,
        called on the host)."""
        piece = self.pieces[self.piece_labels.index("N4.1 gravity")]
        arguments = []
        for name in piece.argument_names:
            if name.startswith("applied_force_"):
                arguments.append(np.zeros(self.batch))
            else:
                arguments.append(np.ascontiguousarray(
                    np.asarray(columns[name], dtype=np.float64)))
        outputs = dict(zip(piece.output_names, piece(*arguments)))
        return np.stack([np.asarray(outputs[f"force_{axis}_next"],
                                    dtype=float)[:self.batch]
                         for axis in AXES], axis=1)

    def _half_step_kick(self, columns, half_dt) -> np.ndarray:
        """``(batch, 3)``: ``half_dt * F_grav(x)``, the momentum the
        integrator's own kick adds over half a substep at the positions in
        ``columns``.  Gravity only: an applied force is constant over every
        substep, and the left-sum kick of a constant force is exact, so it
        has no half-step lag (with no centers this is zero)."""
        if not self.centers:
            return np.zeros((self.batch, 3))
        return np.asarray(half_dt, dtype=float).reshape(-1, 1)             * self._gravity_force(columns)

    def r(self) -> tuple[np.ndarray, np.ndarray]:
        """Current position (m) and velocity (m/s), at the SAME instant.

        The integrator is symplectic Euler read as leapfrog: the step
        kicks the momentum by ``dt * F(x_n)`` and drifts the position with
        the new velocity, so the stored momentum is the half-step value
        ``p_{n+1/2}`` while the position is ``x_{n+1}`` (and the initial
        momentum is seeded half a first substep back).  The velocity at
        ``x``'s instant is the second half kick, ``(p_{n+1/2} + dt/2
        F_grav(x_{n+1})) / m`` (N1.1), with the last substep's ``dt`` and the
        compiled N4.1 piece's force at the current position."""
        mass = self._scalar("mass")
        names = {name: self._span(name) for name in
                 self.pieces[self.piece_labels.index("N4.1 gravity")]
                 .argument_names if not name.startswith("applied_force_")}
        half_dt = 0.5 * (self._dt_first if self.time_s == 0.0 else
                         np.asarray(self._span("dt"), float)[:self.batch])
        momentum = self._lanes(self._vector("momentum"), 3)             + self._half_step_kick(names, half_dt)
        velocity = momentum / np.asarray(self._lanes(mass))[:, None]
        if self.batch == 1:
            return self._vector("position"), velocity[0]
        return self._vector("position"), velocity

    def F(self, force_n) -> None:
        """Set the raw applied force (N, world frame) the next round
        integrates.

        It superposes on the thrusters' ``R @ B_craft @ clamp(u)`` in the
        actuation piece; with every throttle at zero it is the whole applied
        force.  It burns no propellant."""
        force = self._lanes(force_n, 3)
        for index, axis in enumerate(AXES):
            self._write_lanes(f"raw_force_{axis}", force[:, index])

    def throttle(self, throttles) -> None:
        """Set the per-thruster throttles (design order) the next round
        applies; the piece clamps each to its declared range."""
        u = self._lanes(throttles, self.design.thruster_count)
        for index in range(self.design.thruster_count):
            self._write_lanes(f"thruster{index}_throttle", u[:, index])

    def throttles(self) -> np.ndarray:
        """The commanded (unclamped) throttles."""
        return self._per_thruster("throttle")

    def applied_force(self) -> np.ndarray:
        """The applied force (N, world) the last substep integrated."""
        return self._vector("applied_force")

    def torque(self) -> np.ndarray:
        """The craft-frame torque (N m) the last substep integrated."""
        return self._vector("torque")

    def attitude(self) -> np.ndarray:
        """The attitude ``R`` now (world = R @ craft)."""
        rotation = np.empty((self.batch, 3, 3))
        for (row, col), symbol in attitude_symbols().items():
            rotation[:, row, col] = np.asarray(self._span(symbol.name),
                                               dtype=float)[:self.batch]
        return rotation[0] if self.batch == 1 else rotation

    def angular_velocity(self) -> np.ndarray:
        """The craft-frame angular velocity ``omega_B`` (rad/s) now."""
        return self._vector("angular_velocity")

    def actuation_matrix(self) -> np.ndarray:
        """``R @ B_craft`` at the attitude now (world frame)."""
        if self.batch == 1:
            return actuation_matrix(self.design, self.attitude())
        return np.stack([actuation_matrix(self.design, rotation)
                         for rotation in self.attitude()])

    def commanded_force(self) -> np.ndarray:
        """``R @ B_craft @ clamp(u) + raw`` evaluated on the host at the
        attitude now -- what the next substep's actuation piece writes while
        propellant lasts (the supply limit is the step's, not the host's)."""
        raw = self._vector("raw_force")
        if self.batch == 1:
            return (self.actuation_matrix()
                    @ clamp_throttles(self.design, self.throttles()) + raw)
        return np.stack([B @ clamp_throttles(self.design, u) for B, u in
                         zip(self.actuation_matrix(), self.throttles())]
                        ) + raw

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
    def mass_kg(self):
        return self._scalar("mass")

    @property
    def propellant_kg(self):
        return self._scalar("propellant_mass")

    @property
    def propellant_flow_kg_s(self):
        """The propellant flow the last substep drew."""
        return self._scalar("propellant_flow")

    @property
    def propellant_supply(self):
        """The fraction of the demand the last substep delivered."""
        return self._scalar("propellant_supply")

    @property
    def fuel_impulse_n_s(self):
        return self._scalar("fuel_impulse")

    @property
    def thruster_impulses_n_s(self) -> np.ndarray:
        """Per-thruster accumulated delivered impulse (N*s), design order."""
        return self._per_thruster("impulse")

    @property
    def substeps(self) -> int:
        """Accepted substeps so far: the first piece's call count (the plan
        rolls back only on error channels, which no piece publishes)."""
        return int(self.dt_state.wall_cost_ledger.calls[0])
