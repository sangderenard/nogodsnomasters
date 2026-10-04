"""Orbital craft on the real managed dt system and its RK4 integrator.

Catalogue N4.1/N7.2/N1.6/N1.3 laws define the coupled physical state.
The actual RK4Integrator.step callback constructs its stages as ordinary
LLVM equation pieces.  PieceState owns all canonical and stage columns,
rollback, the adaptive substeps and each requested outer window.

The r()/F()/throttle() seam remains the trajectory interface.  Position,
momentum, attitude and wheel state share one accepted endpoint.  Work and
external angular impulse follow the same integrator stages; their endpoint
conservation defects are declared to the dt controller in physical units.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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
from src.common.dt_system.time_contracts import BIND
from src.common.dt_system.integrator.integrator import RK4Integrator
from src.common.dt_system.error_channels import DT_CHANNEL_NAMES, channel_fields
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
    wheel_symbols,
    wheel_relative_speed,
)


# Declared accuracy defaults for the coupled integration.  These are absolute
# local defects, with independent translation and rotation budgets; orbital
# kinetic energy cannot dilute a rotor's conservation error.
ORBITAL_ERROR_LIMITS = {
    "orbital_green_collocation_residual_m": 1.0e-3,
    "orbital_guidance_commitment": 1.0,
    "orbital_translation_energy_residual_j": 1.0e-2,
    "orbital_rotation_energy_residual_j": 1.0e-6,
    "orbital_angular_momentum_residual_n_m_s": 1.0e-6,
    # This is a local incremental Gram defect, not accumulated drift.
    # The existing 10 s spin gate requires global drift <1e-14: a 1e-9
    # local budget measured 1.18e-8 drift; 1e-17 measured 2.33e-15.
    "orbital_attitude_orthogonality": 1.0e-17,
    "orbital_wheel_speed_excess_rad_s": 1.0e-9,
}
ORBITAL_CHANNEL_NAMES = (*DT_CHANNEL_NAMES, *ORBITAL_ERROR_LIMITS)
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


def attitude_rate_rhs():
    """The catalogue's continuous ``dR/dt = R skew(omega_B)`` columns."""
    law = honorary.eq_N1_3.rhs
    factors = law.as_ordered_factors()
    if (len(factors) != 2 or factors[0] != honorary.R(honorary.t)
            or getattr(factors[1].func, "__name__", "") != "skew"):
        raise RuntimeError("eq_N1_3 is no longer R * skew(omega_B)")
    rotation = sp.Matrix(3, 3, lambda r, c: attitude_symbols()[(r, c)])
    rate = rotation * _skew([sp.Symbol(f"angular_velocity_{a}") for a in AXES])
    return {symbol.name: rate[r, c]
            for (r, c), symbol in attitude_symbols().items()}


def integrated_state_equations(names, dynamics):
    """Manifest the existing RK4 call as a bounded graph of equation pieces.

    Each callback of ``RK4Integrator.step`` declares one derivative piece
    and returns that piece's symbols.  The returned update is the library's
    expression, verbatim.  There are no authored RK coefficients or stage
    schedules here.  Intermediate columns are derivative values, and the
    canonical physical columns are committed only after every stage.

    Inlining all stages into SymPy expressions made a one-wheel, 13-column
    law spend more than 90 CPU seconds constructing before compilation.
    The existing piece graph retains the same dependencies without expanding
    them; PieceState owns these columns and its normal transaction restores
    them along with the physical state on rejection.
    """
    names = tuple(names)
    symbols = tuple(sp.Symbol(name) for name in names)
    initial = sp.Matrix(symbols)
    stages = []
    calls = 0

    def declare_derivative(stage_time, stage_state):
        nonlocal calls
        evaluated = dynamics(stage_time, dict(zip(names, stage_state)))
        if isinstance(evaluated, tuple):
            preparations, rates = evaluated
            stages.extend(preparations)
        else:
            rates = evaluated
        derivative = tuple(sp.Symbol(f"rk_stage{calls}_{name}_rate")
                           for name in names)
        calls += 1
        stages.append(tuple(
            sp.Eq(sp.Symbol(f"{symbol.name}_next"), rates[name], evaluate=False)
            for name, symbol in zip(names, derivative)))
        return sp.Matrix(derivative)

    result = RK4Integrator().step(declare_derivative, sp.Integer(0), initial, DT)
    return tuple(stages), dict(zip(names, result))


def mechanical_endpoint_metrics(next_state, *, mass_before, mass_after,
                                inertia_before, inertia_change,
                                center_count, wheel_count=0):
    """Open-system conservation defects from the same RK4 endpoints/work.

    Differences of squares and reciprocal radii are expressed as products
    of their sums and changes.  A small burn is not recovered by subtracting
    two orbital kinetic energies of order 1e10 J.
    """
    old = {name: sp.Symbol(name) for name in next_state}
    delta = {name: value - old[name] for name, value in next_state.items()}
    vector = lambda values, prefix: sp.Matrix([values[f"{prefix}_{a}"] for a in AXES])
    p, p1, dp = (vector(values, "momentum") for values in (old, next_state, delta))
    x, x1, dx = (vector(values, "position") for values in (old, next_state, delta))
    omega, omega1, domega = (vector(values, "angular_velocity")
                            for values in (old, next_state, delta))
    dm = mass_after - mass_before
    dtranslation = ((p1 + p).dot(dp) / (2 * mass_after)
                    - p.dot(p) * dm / (2 * mass_before * mass_after))
    potential0 = sp.Integer(0)
    potential1 = sp.Integer(0)
    dpotential = sp.Integer(0)
    for index in range(center_count):
        center, mu = _center_symbols(index)
        c = sp.Matrix([center[a] for a in AXES])
        r0, r1 = x - c, x1 - c
        norm0, norm1 = sp.sqrt(r0.dot(r0)), sp.sqrt(r1.dot(r1))
        potential0 -= mu / norm0
        potential1 -= mu / norm1
        dpotential += mu * (r1 + r0).dot(dx) / ((norm1 + norm0) * norm1 * norm0)
    dtranslation += dm * potential1 + mass_before * dpotential
    inertia1 = inertia_before + inertia_change
    drotational = ((omega1 + omega).dot(inertia1 * domega)
                   + omega.dot(inertia_change * omega)) / 2
    rotor_h, rotor_dh = sp.zeros(3, 1), sp.zeros(3, 1)
    wheel_stored = sp.Integer(0)
    excess = []
    for w in range(wheel_count):
        s = wheel_symbols(w)
        h1, dh = next_state[f"wheel{w}_momentum"], delta[f"wheel{w}_momentum"]
        axis = sp.Matrix([s["axis"][a] for a in AXES])
        rotor_h += axis * s["momentum"]
        rotor_dh += axis * dh
        drotational += dh * (h1 + s["momentum"]) / (2 * s["inertia"])
        wheel_stored += h1**2 / (2 * s["inertia"])
        excess.append(sp.Max(sp.Integer(0), sp.Abs(h1 / s["inertia"] - axis.dot(omega1))
                             - s["max_speed"]))
    rotation = sp.Matrix(3, 3, lambda r, c: old[attitude_symbols()[r, c].name])
    rotation1 = sp.Matrix(3, 3, lambda r, c: next_state[attitude_symbols()[r, c].name])
    dr = sp.Matrix(3, 3, lambda r, c: delta[attitude_symbols()[r, c].name])
    dh_world = (dr * (inertia_before * omega + rotor_h)
                + rotation1 * (inertia1 * domega + inertia_change * omega + rotor_dh))
    angular_error = dh_world - vector(delta, "angular_impulse_world")
    gram_error = rotation.T * dr + dr.T * rotation + dr.T * dr
    return {
        "orbital_translation_energy_residual_j": sp.Abs(dtranslation - delta["translation_work"]),
        "orbital_rotation_energy_residual_j": sp.Abs(drotational - delta["rotation_work"]),
        "orbital_angular_momentum_residual_n_m_s": sp.sqrt(angular_error.dot(angular_error)),
        "orbital_attitude_orthogonality": sp.sqrt(sum(value**2 for value in gram_error)),
        "orbital_wheel_speed_excess_rad_s": sp.Max(*excess) if excess else sp.Integer(0),
        "translation_stored_next": p1.dot(p1) / (2 * mass_after),
        "gravity_stored_next": -mass_after * potential1,
        "rotation_stored_next": omega1.dot(inertia1 * omega1) / 2,
        "wheel_stored_next": wheel_stored,
        "max_vel": sp.sqrt(p1.dot(p1)) / mass_after,
    }


def energy_metric_equations(tag, center_count):
    """Separate physical stores, each with its own declared exchange power.

    A torque or thrust source has no finite equilibrium: its exchangeable
    energy is unbounded.  Gravity has the finite binding store m*mu/r.
    Conservation defects, rather than E measured from absolute zero, test
    the accuracy of a source's integration through rest and reversal.

    This is the pure-source convention of
    ``examples.symbolic_chamber_solvers.exchangeable_energy``: zero
    restoring slope has no finite equilibrium or energy bound, so the
    conservation metrics govern.  The wheel participant publishes rotor
    kinetic energy and absolute electrical exchange power; signed input
    work and copper dissipation have separate physical ledger columns.
    """
    eq = lambda name, rhs: sp.Eq(sp.Symbol(name), rhs, evaluate=False)
    pieces = []
    for label in ("translation", "rotation", "wheel"):
        exchange = (sp.Symbol("gravity_stored") if label == "translation" and center_count
                    else sp.oo)
        pieces.append((f"{tag}_{label}_energy", (
            eq("energy_j", sp.Symbol(f"{label}_stored")),
            eq("exchangeable_energy_j", exchange),
            eq("power_w", sp.Symbol(f"{label}_power")),
        )))
    return tuple(pieces)


def declare_binding(pieces):
    """Every orbital piece declares ``BIND`` (``llvm_dt_system.RoundPiece``'s
    precedent): its exchange time bounds the step when it publishes one, and
    a piece that exchanges nothing this step binds nobody instead of reading
    as ``HOLD``'s "do not grow".  Explicit participant contracts retain that
    authority when every participant is quiet."""
    for piece in pieces:
        piece.contract = BIND
    return pieces


def orbital_jumper_equations(center_count: int, thruster_count: int = 0):
    """Catalogue continuous craft laws advanced by the dt library's RK4."""
    if center_count < 0 or thruster_count < 0:
        raise ValueError("center and thruster counts must be non-negative")
    eq = lambda name, rhs: sp.Eq(sp.Symbol(name), rhs, evaluate=False)
    tag = f"orbital_rk4_c{center_count}_t{thruster_count}"
    supply = (f"{tag}_supply", (
        eq("propellant_supply_next", propellant_supply_rhs(thruster_count)),
    ))
    mass = sp.Symbol("dry_mass") + PROPELLANT_MASS
    p = sp.Matrix([sp.Symbol(f"momentum_{a}") for a in AXES])
    omega = sp.Matrix([sp.Symbol(f"angular_velocity_{a}") for a in AXES])
    inertia = sp.diag(*[sp.Symbol(f"inertia_{a}") for a in AXES])
    rotation = sp.Matrix(3, 3, lambda r, c: attitude_symbols()[r, c])
    applied = sp.Matrix([actuation_force_rhs(a, thruster_count)
                         + sp.Symbol(f"raw_force_{a}") for a in AXES])
    torque = sp.Matrix([actuation_torque_rhs(a, thruster_count) for a in AXES])
    flow = propellant_flow_rhs(thruster_count)
    gravity = sp.Matrix([gravity_force_rhs(a, center_count).xreplace(
                         {sp.Symbol("mass"): mass}) for a in AXES])
    replacements = {sp.Symbol("mass"): mass, PROPELLANT_FLOW: flow}
    replacements.update({sp.Symbol(f"force_{a}"): gravity[i] + applied[i]
                         for i, a in enumerate(AXES)})
    replacements.update({sp.Symbol(f"torque_{a}"): torque[i] for i, a in enumerate(AXES)})
    rates = {f"momentum_{a}": variable_mass_momentum_rate(a).xreplace(replacements)
             for a in AXES}
    rates.update({f"position_{a}": p[i] / mass for i, a in enumerate(AXES)})
    rates.update({f"angular_velocity_{a}": euler_rate_rhs(a).xreplace(replacements)
                  for a in AXES})
    rates.update(attitude_rate_rhs())
    rates["propellant_mass"] = -flow
    rates["fuel_impulse"] = (sum((delivered_thrust(k) for k in range(thruster_count)),
                                  sp.Integer(0)) + thrust_cost_integrand("raw_force"))
    for k in range(thruster_count):
        rates[f"thruster{k}_impulse"] = delivered_thrust(k)
    potential = sp.Integer(0)
    for k in range(center_count):
        center, mu = _center_symbols(k)
        radius = sp.sqrt(sum((sp.Symbol(f"position_{a}")-center[a])**2 for a in AXES))
        potential -= mu / radius
    velocity = p / mass
    translation_flux = -flow * (velocity.dot(velocity)/2 + potential)
    rates["translation_work"] = applied.dot(velocity) + translation_flux
    rates["rotation_work"] = torque.dot(omega)
    rates["translation_exchange"] = (sp.Abs(applied.dot(velocity))
                                      + sp.Abs(gravity.dot(velocity)) + sp.Abs(translation_flux))
    rates["rotation_exchange"] = sp.Abs(torque.dot(omega))
    rates["wheel_exchange"] = sp.Integer(0)
    world_torque = rotation * torque
    for i, a in enumerate(AXES):
        rates[f"angular_impulse_world_{a}"] = world_torque[i]
        rates[f"applied_impulse_world_{a}"] = applied[i]
        rates[f"applied_delta_v_world_{a}"] = applied[i] / mass
        rates[f"torque_impulse_body_{a}"] = torque[i]

    def dynamics(_elapsed, state):
        substitutions = {sp.Symbol(name): value for name, value in state.items()}
        return {name: rhs.xreplace(substitutions) for name, rhs in rates.items()}

    stages, next_state = integrated_state_equations(rates, dynamics)
    # The original supply law bounds the whole-step draw.  Preserve its exact
    # nonnegative store boundary, with the known increment exposed so that
    # the energy law cancels the old charge before native evaluation.
    old_propellant = sp.Symbol("propellant_mass")
    next_state["propellant_mass"] = old_propellant + sp.Max(
        -old_propellant, next_state["propellant_mass"] - old_propellant)
    mass_next = sp.Symbol("dry_mass") + next_state["propellant_mass"]
    metrics = mechanical_endpoint_metrics(
        next_state, mass_before=mass, mass_after=mass_next,
        inertia_before=inertia, inertia_change=sp.zeros(3), center_count=center_count)
    commit = [eq(f"{name}_next", value) for name, value in next_state.items()]
    commit.extend(eq(name, value) for name, value in metrics.items())
    # The ordinary RK4 participant has no Green candidate.  Publishing zero
    # keeps the shared orbital channel catalogue complete; MachineCraft
    # replaces it with the coast law's defect when that law owns translation.
    commit.append(eq("orbital_green_collocation_residual_m", sp.Integer(0)))
    commit.extend((eq("mass_next", mass_next), eq("propellant_flow_next", flow)))
    for i, a in enumerate(AXES):
        mean_force = (next_state[f"applied_impulse_world_{a}"] - sp.Symbol(f"applied_impulse_world_{a}")) / DT
        mean_torque = (next_state[f"torque_impulse_body_{a}"] - sp.Symbol(f"torque_impulse_body_{a}")) / DT
        commit.extend((eq(f"applied_force_{a}_next", mean_force),
                       eq(f"torque_{a}_next", mean_torque)))
    for store in ("translation", "rotation", "wheel"):
        commit.append(eq(f"{store}_power_next",
                         (next_state[f"{store}_exchange"] - sp.Symbol(f"{store}_exchange")) / DT))
    pieces = [supply]
    pieces.extend((f"{tag}_stage{k}", equations)
                  for k, equations in enumerate(stages))
    pieces.append((f"{tag}_commit", tuple(commit)))
    pieces.extend(energy_metric_equations(tag, center_count))
    return tuple(pieces)


def orbital_jumper_dt_pieces(center_count: int, thruster_count: int = 0,
                             batch: int = 1, *, retain_compilation: bool = False):
    return declare_binding(tuple(
        equation_piece(name, equations, batch=batch,
                       retain_compilation=retain_compilation)
        for name, equations in orbital_jumper_equations(center_count, thruster_count)))


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
    folds the configured metrics over the lanes).  Per-
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
                 error_limits: Mapping[str, float] | None = None,
                 best_effort_metrics: Sequence[str] = (),
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
        self.error_limits = dict(ORBITAL_ERROR_LIMITS)
        if error_limits is not None:
            undeclared = set(error_limits) - self.error_limits.keys()
            if undeclared:
                raise ValueError(f"undeclared orbital error channels: {sorted(undeclared)}")
            for name, value in error_limits.items():
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"{name}: error limit must be finite and positive")
                self.error_limits[name] = float(value)
        self.best_effort_metrics = frozenset(str(name)
                                             for name in best_effort_metrics)
        unknown_best_effort = self.best_effort_metrics - self.error_limits.keys()
        if unknown_best_effort:
            raise ValueError(
                f"undeclared best-effort metrics: {sorted(unknown_best_effort)}")
        self.pieces, labels = self._dt_pieces()
        attitudes = np.broadcast_to(
            np.eye(3) if attitude is None
            else np.asarray(attitude, dtype=float),
            (self.batch, 3, 3))
        for rotation in attitudes:
            _attitude_array(rotation)
        columns = self._initial_columns(
            position_m, velocity_m_s, attitudes, angular_velocity_rad_s)
        # Physical stores and local conservation defects are independent
        # declarations; the dt controller owns every acceptance and retry.
        # Publication and judgment are separate dt-system declarations.  A
        # best-effort metric remains in every participant's publication row
        # for diagnostics, but has no target-presence bit, so it neither
        # rejects a state nor steers dt toward an unattainable numerical
        # floor.
        judged_limits = {
            name: value for name, value in self.error_limits.items()
            if name not in self.best_effort_metrics
        }
        targets = Targets(cfl=float(cfl), div_max=1.0e9, mass_max=1.0e-3,
                          energy_exchange_fraction=0.2,
                          **channel_fields(judged_limits,
                                           names=ORBITAL_CHANNEL_NAMES, limits=True))
        # The first attempt is the controller's own CFL proposal at the
        # initial state (the fastest lane's); every later attempt is the
        # controller's continuation.
        speed = float(np.max(np.linalg.norm(
            self._lanes(velocity_m_s, 3), axis=1)))
        # This is the controller's opening proposal, not the requested outer
        # window.  ``advance_round`` deliberately carries proposals into a
        # shorter window unclipped, so clipping the CFL proposal here made a
        # caller's frame span masquerade as dt policy before the first round.
        stability_dt_max = (None if speed <= 0.0
                            else self.length_scale_m / speed)
        dt_init = (self.window_s if stability_dt_max is None
                   else targets.cfl * stability_dt_max)
        self.dt_controller = STController(dt_min=None,
                                          dt_max=stability_dt_max)
        self.piece_labels = tuple(labels)
        inner_graph = RoundNode(
            plan=SuperstepPlan(round_max=self.window_s, dt_init=dt_init,
                               allow_increase_mid_round=True),
            controller=ControllerNode(
                ctrl=self.dt_controller,
                targets=targets,
                dx=self.length_scale_m,
            ),
            children=[piece_leaf(piece, label=label)
                      for piece, label in zip(self.pieces, labels)],
            schedule="sequential",
            label="orbital-jumper",
        )
        self.dt_graph = self._compose_dt_graph(inner_graph, targets, dt_init)
        self.dt_state = instantiate_system(self.dt_graph, columns,
                                            channel_names=ORBITAL_CHANNEL_NAMES)
        self.time_s = 0.0

    # ----------------------------------------------------------- the hooks
    def _initial_mass_kg(self):
        """The declared initial mass used by every mass-dependent column."""
        return (self._lane_mass_kg if self._lane_mass_kg is not None
                else self.design.mass_kg)

    def _inertia_kg_m2(self) -> np.ndarray:
        """The inertia the ``inertia_*`` columns start from (a craft whose
        mass properties are laws of its own state overrides this)."""
        return principal_inertia(self.design)

    def _dt_pieces(self):
        """``(pieces, labels)`` in causal order (a craft with its own
        actuation and mass-property laws overrides this)."""
        pieces = orbital_jumper_dt_pieces(len(self.centers),
                                          self.design.thruster_count,
                                          batch=self.batch)
        return pieces, tuple(piece.entry for piece in pieces)

    def _compose_dt_graph(self, inner_graph, targets, dt_init):
        """Compose a containing dt system before the sole instantiation."""
        return inner_graph

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
        mass = self._lanes(self._initial_mass_kg())
        position = self._lanes(position_m, 3)
        velocity = self._lanes(velocity_m_s, 3)
        omega = self._lanes(angular_velocity_rad_s, 3)
        columns = {"mass": mass.copy(),
                   "dry_mass": mass - float(design.propellant_kg),
                   "propellant_mass": np.full(lanes, design.propellant_kg),
                   "propellant_flow": np.zeros(lanes),
                     "propellant_supply": np.ones(lanes),
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
        columns["translation_stored"] = mass * np.sum(velocity**2, axis=1) / 2
        columns["gravity_stored"] = sum(
            (mass * center.mu_m3_s2
             / np.linalg.norm(position - np.asarray(center.position_m), axis=1)
             for center in self.centers), np.zeros(lanes))
        columns["rotation_stored"] = np.sum(
            omega**2 * self.inertia_kg_m2[None, :], axis=1) / 2
        columns["wheel_stored"] = np.zeros(lanes)
        for store in ("translation", "rotation", "wheel"):
            columns[f"{store}_exchange"] = np.zeros(lanes)
            columns[f"{store}_power"] = np.zeros(lanes)
        for name in ("translation_work", "rotation_work", "wheel_energy",
                     "wheel_copper_loss"):
            columns[name] = np.zeros(lanes)
        for axis in AXES:
            for prefix in ("applied_impulse_world", "applied_delta_v_world",
                           "angular_impulse_world", "torque_impulse_body"):
                columns[f"{prefix}_{axis}"] = np.zeros(lanes)
        # Only the library-call manifestation's declared scratch values
        # get automatic zero initialization.  Physical columns above (and
        # the machine's override) have explicit initial values; missing
        # physical parameters are rejected by PieceState construction.
        for piece in self.pieces:
            for output in piece.output_names:
                if ((output.startswith("rk_stage")
                     or output.startswith("green_coast_"))
                        and output.endswith("_next")
                        and output[:-5] not in columns):
                    columns[output[:-5]] = np.zeros(lanes)
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
    def r(self) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity at the same accepted library-integrator endpoint."""
        mass = self._lanes(self._scalar("mass"))
        velocity = self._lanes(self._vector("momentum"), 3) / mass[:, None]
        return self._vector("position"), velocity[0] if self.batch == 1 else velocity

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

    @property
    def applied_delta_v_m_s(self) -> np.ndarray:
        """Accumulated world applied acceleration, using the physics stages.

        Subtract accepted endpoint readings to measure a burn's delivered
        delta-v.  The integrand uses that stage's attitude, thrust, supply
        and mass, so the tracker does not reconstruct a direction or mass
        quadrature on its caller's frame boundaries.
        """
        return self._vector("applied_delta_v_world")

    @property
    def applied_impulse_n_s(self) -> np.ndarray:
        """Accumulated applied world force, on the same library quadrature."""
        return self._vector("applied_impulse_world")

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

    #: Declared: this craft has no allocation of its own; a driver commands
    #: it with ``throttle(u)`` (``MachineCraft`` declares True).
    applies_allocation = False

    def inertia_tensor(self) -> np.ndarray:
        """The inertia tensor (craft axes, about the centre of mass): the
        principal moments the ``inertia_*`` columns hold, as a matrix."""
        return np.diag(self._inertia_kg_m2())

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
    def last_managed_dt_s(self) -> float:
        """The dt system's last executed internal step, for observation."""
        return float(np.asarray(self.dt_state.dt, dtype=float).reshape(-1)[0])

    @property
    def managed_dt_proposal_s(self) -> float:
        """The dt system's next window: continuation, or opening stability cap."""
        value = self.dt_state.dt_next
        if value is None:
            value = (self.dt_controller.dt_max
                     if self.dt_controller.dt_max is not None
                     else self.dt_state.dt_init)
        return float(value)

    @property
    def managed_dt_blocker(self) -> str:
        """Human-readable identity of the publication limiting ``dt_next``.

        Reporting only: this reads the accepted attempt's existing dense
        publication spans and controller state.  It does not feed any value
        back into dt selection.
        """
        if self.dt_state.dt_next is None:
            return "opening dt_max"

        def array(value):
            if hasattr(value, "numpy"):
                value = value.numpy()
            return np.asarray(value, dtype=float)

        state = self.dt_state
        proposal = self.managed_dt_proposal_s
        labels = tuple(self.piece_labels)
        ceilings = []

        telemetry = array(state.telemetry).reshape(-1)
        max_vel = float(telemetry[2]) if telemetry.size > 2 else 0.0
        if max_vel > 0.0:
            ceilings.append((float(state.targets.cfl) * float(state.dx)
                             / max_vel, "max_vel (CFL)"))
        if self.dt_controller.dt_max is not None:
            ceilings.append((float(self.dt_controller.dt_max),
                             "controller dt_max"))

        participant_dt = array(state.pub_dt_limit).reshape(-1)
        participant_dt_present = array(
            state.pub_dt_limit_present).reshape(-1) > 0.5
        for index, (value, present) in enumerate(
                zip(participant_dt, participant_dt_present)):
            if present and value > 0.0 and math.isfinite(float(value)):
                who = labels[index] if index < len(labels) else str(index)
                ceilings.append((float(value), f"{who} / dt_limit"))

        fraction = getattr(state.targets, "energy_exchange_fraction", None)
        if fraction is not None:
            exchange = array(state.pub_exchange_time).reshape(-1)
            exchange_present = array(
                state.pub_exchange_time_present).reshape(-1) > 0.5
            contracts = array(state.pub_contract).reshape(-1)
            for index, (value, present, contract) in enumerate(
                    zip(exchange, exchange_present, contracts)):
                if (present and int(contract) == BIND and value >= 0.0
                        and math.isfinite(float(value))):
                    who = labels[index] if index < len(labels) else str(index)
                    ceilings.append((float(fraction) * float(value),
                                     f"{who} / exchange_time"))

        finite = [(limit, identity) for limit, identity in ceilings
                  if limit >= 0.0 and math.isfinite(limit)]
        if finite:
            limit, identity = min(finite, key=lambda item: item[0])
            tolerance = 1.0e-6 * max(1.0, abs(proposal), abs(limit))
            if limit <= proposal + tolerance:
                return identity

        channels = tuple(state.channel_names)
        count = len(channels)
        values = array(state.pub_values).reshape((-1, count))
        present = array(state.pub_present).reshape((-1, count)) > 0.5
        limits = array(state.pub_limits).reshape((-1, count))
        limited = array(state.pub_limits_present).reshape((-1, count)) > 0.5
        judged = present & limited
        ratios = np.where(judged, values / np.maximum(limits, 1.0e-30), 0.0)
        if ratios.size:
            participant, channel = np.unravel_index(
                int(np.argmax(ratios)), ratios.shape)
            ratio = float(ratios[participant, channel])
            if ratio > 0.0:
                who = (labels[participant]
                       if participant < len(labels) else str(participant))
                name = channels[channel] if channel < len(channels) else str(channel)
                return f"{who} / {name} ({ratio:.3g}x)"
        return "PI growth history"

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
        """Attempted substeps: the first law's measured native call count."""
        return int(self.dt_state.wall_cost_ledger.calls[0])
