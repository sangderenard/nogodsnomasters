"""Orbital craft, build steps 3b and 5: the live tracker -- planned burns
flown open loop, PD trims, a desired wrench per round through the craft's
allocation seam, off-plan switching with hysteresis, full-trip re-plans.

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``
(decision 4, the ON PLAN / OFF PLAN pipeline; decision 7, fuel and
deviation belong in the live controller; decision 9, thrust direction
comes from the attitude).  The tracker meets the craft only through the
seam (decision 8): ``r()``, ``throttle(u)``, ``advance(window)``,
``attitude()``, ``angular_velocity()``, ``allocate(wrench)``, and the
public ``design``, ``mass_kg``, ``time_s``, ``fuel_impulse_n_s``.  Public
surface:

    TrackingGains           frozen: PD gains, deadbands, switch band
    TrackingMode            mutable: plan flown now, ON/OFF, phase, burns
    tracker_mode(craft)     the mode fly keeps per craft (for display)
    Wrench, Allocation      the allocation seam's request and answer
    PlannedImpulse          a plan's impulsive burn (time, world delta-v)
    LeastSquaresAllocator   stand-in for the craft's own ``allocate``
    plan_impulses(plan)     the plan's burns
    allocate_throttles(design, force, fuel_weight, ...) -> u
    attitude_torque_demand(design, gains, R, w, force) -> (tau, angle)
    tracking_command(plan, design, gains, t, r, v, mass, ...) -> command
    tracking_error(r, v, r_ref, v_ref, shortfall=) -> scale-free error
    hohmann_replanner(previous_plan, t, r, v) -> HohmannPlan
    fly(craft, plan, gains, until_s=, round_s=, replanner=, mode=,
        record=) -> report

A plan is anything :func:`plan_reference` can evaluate: an object with a
``reference(t) -> (r, v)`` method, or a :class:`HohmannPlan` (through
``orbital_plan.reference``); it carries ``mu``.  Its impulsive burns come
from its own ``impulses()`` when it declares one, from the velocity steps
of a HohmannPlan's legs otherwise (a continuous-thrust plan has none and is
flown by the PD alone).

The allocation seam.  The tracker does not choose throttles: it asks for a
DESIRED WRENCH -- a world-frame force and a craft-frame torque -- and the
craft answers ``allocate(wrench) -> Allocation(achieved, throttles)``, the
wrench its thrusters actually produce and the throttles that produce it.
The tracker commands those throttles and treats the ACHIEVED wrench as what
happens.  Until the craft layer provides ``allocate``,
:class:`LeastSquaresAllocator` stands in (the one line is in :func:`fly`).

Method, once per round of length ``h`` (throttles held for the round):

1. Plan error.  ``(r_ref, v_ref) = plan_reference(plan, t)``;
   ``e_r = r - r_ref``, ``e_v = v - v_ref``.
2. Switch (build step 5, only when a re-planner is injected).  The
   scale-free tracking error, with last round's force shortfall ``s``
   (requested minus achieved) counted as the velocity error it leaves,

       eps = sqrt((|e_r| / |r_ref|)^2 + ((|e_v| + |s| h / m) / |v_ref|)^2)

   drives a hysteresis switch: ON PLAN -> OFF PLAN when ``eps`` exceeds
   ``off_plan_threshold``, and on that transition the WHOLE remaining trip
   is re-planned from the present state to the final orbit by the injected
   ``replanner(previous_plan, t, r, v)`` (the previous plan is its starting
   guess); OFF PLAN -> ON PLAN when ``eps`` falls below
   ``on_plan_threshold``.  While OFF the tracker re-homes on the fresh plan
   and does not re-plan again, so one excursion is one re-plan.
3. Burns (fuel-honest: the plan's impulses are flown, not chased).  Each
   impulse ``dv`` at ``t_b`` is flown as a finite burn at full capacity
   along a fixed direction, centred on ``t_b``.  Its velocity change,
   fixed when the burn is armed, carries the craft onto the post-burn
   reference: ``v_ref + dv - v`` while the reference is still pre-burn,
   ``v_ref - v`` after ``t_b`` (a re-plan's burn 1 is due now); its duration from the rocket
   equation (TS2.1 at the thrusters' effective exhaust velocity:
   ``m c (1 - exp(-|dv|/c)) / F``) sets the start ``t_b - duration/2``.
   A craft with a pointing axis is armed ``slew_lead`` early and turns the
   axis onto the burn direction; it fires only when within
   ``burn_alignment_rad``.  The burn is closed on DELIVERED velocity
   (achieved force over mass, every round), not on time, so mass loss and
   alignment are paid for exactly.  The PD is off during a burn.
4. Trims (the PD only trims).  Between burns the computed-acceleration PD

       F_pd = m ([g(r_ref) - g(r)] - w^2 e_r - 2 zeta w e_v)

   fires only outside a coast deadband: trimming starts when
   ``|e_r| > p_band`` or ``|e_v| > v_band`` and stops when both are below
   half of theirs.  ``v_band = max(coast_velocity_deadband * |v_ref|,
   |g(r)| dt)`` is sized against the integrator's known reading stagger:
   ``r()``'s symplectic-Euler velocity is half a substep off its position,
   ``|v_read - v(t)| ~ g dt / 2`` (13 m/s at LEO with 3.3 s substeps) --
   not a real error, and fighting it costs fuel.  ``dt`` is the dt
   system's continuation step that ``advance`` returns (the round ``h``
   until one is seen; ``dt <= h``).  The PD answers that stagger with a
   position offset up to ``2 (g dt / 2) / w``, so ``p_band = max(
   coast_position_deadband * |r_ref|, |g| dt / w)``.  (Measured: wider
   bands turn coasting into a trim limit cycle -- see the step-5
   continuation.)  A trim request carries last round's shortfall forward,
   ``F_des = F_pd + c``, ``c = F_des' - F_achieved'`` clipped to
   ``|c| <= |F_pd|`` (no windup), and the request is clipped to what the
   craft can push that way once pointed.  Inside the deadband the craft coasts
   (no force; the attitude is held).
5. Torque (decision 9).  The pointing axis is the craft-frame direction of
   the declared strongest thruster (none when no thruster is strictly
   strongest: a six-axis craft has no preferred axis and only damps its
   rate).  A trim first asks for ``F_des`` with the attitude HELD
   (desired = current: rate damping only); when the achieved force falls
   short by more than ``pointing_tolerance * |F_des|`` plus the fuel
   deadband (``fuel_weight * T_max``) it asks again with the attitude
   turned by the shortest rotation carrying the axis onto ``F_des``.  The
   PD on the rotation-matrix error (``e_R = vee(R_d^T R - R^T R_d) / 2``,
   the SO(3) attitude error) is

       tau_des = I (-w_a^2 e_R - 2 zeta_a w_a omega) + omega x I omega

   (craft frame, principal ``I`` from ``orbital_actuation``).
6. Allocation through the seam: ``(F_des, tau_des)`` -> achieved wrench +
   throttles.  The stand-in minimises

       J(u) = |B u - F_des|^2 / (2 T_max^2) + |T u - tau_des|^2 / (2 tau_max^2)
            + fuel_weight * sum_k T_k u_k / T_max

   over the declared throttle box by L-BFGS-B (``scipy.optimize``).
"""
from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import linprog, minimize

from orbital_actuation import (
    CraftDesign,
    actuation_matrix,
    clamp_throttles,
    principal_inertia,
    propellant_flow_per_throttle,
    torque_matrix,
)
from orbital_plan import (
    HohmannPlan,
    hohmann_plan,
    reference,
    two_body_acceleration,
)


@dataclass(frozen=True)
class TrackingGains:
    """Gains, deadbands and the off-plan switch band.

    ``natural_frequency_rad_s``/``damping_ratio``: the trim PD.
    ``fuel_weight``: dimensionless force deadband as a fraction of the
    largest thruster (the stand-in allocator's fuel price).
    ``attitude_frequency_rad_s``/``attitude_damping_ratio``: the attitude PD
    (keep ``w_a * round_s`` well under 1: throttles are held for a round).
    ``pointing_tolerance``: the force shortfall, as a fraction of the
    request, the held attitude may leave before the tracker turns the
    pointing axis onto the request.  ``coast_position_deadband`` /
    ``coast_velocity_deadband``: errors (over orbit radius / orbital speed)
    the PD leaves alone (the velocity band is never below ``|g| dt``, twice
    the integrator's reading stagger).  ``burn_alignment_rad``: how close
    the pointing axis must be to a burn's direction before it fires.
    ``slew_lead_s``: how long before a burn's start the craft turns onto
    it (``None``: six attitude time constants, ``6 / w_a``).
    ``off_plan_threshold`` / ``on_plan_threshold``: the hysteresis band on
    the scale-free tracking error (upper, lower)."""

    natural_frequency_rad_s: float = 0.02
    damping_ratio: float = 1.0
    fuel_weight: float = 1.0e-4
    attitude_frequency_rad_s: float = 0.05
    attitude_damping_ratio: float = 1.0
    pointing_tolerance: float = 0.1
    coast_position_deadband: float = 4.0e-6
    coast_velocity_deadband: float = 0.0
    burn_alignment_rad: float = 0.05
    slew_lead_s: float | None = None
    off_plan_threshold: float = 0.05
    on_plan_threshold: float = 0.01

    def __post_init__(self):
        if not (self.natural_frequency_rad_s > 0.0
                and self.damping_ratio > 0.0 and self.fuel_weight >= 0.0
                and self.attitude_frequency_rad_s > 0.0
                and self.attitude_damping_ratio > 0.0
                and self.pointing_tolerance >= 0.0
                and self.coast_position_deadband >= 0.0
                and self.coast_velocity_deadband >= 0.0
                and self.burn_alignment_rad > 0.0
                and (self.slew_lead_s is None or self.slew_lead_s >= 0.0)
                and 0.0 < self.on_plan_threshold < self.off_plan_threshold):
            raise ValueError(f"invalid tracking gains {self}")

    @property
    def slew_lead(self) -> float:
        return (6.0 / self.attitude_frequency_rad_s if self.slew_lead_s is None
                else self.slew_lead_s)


@dataclass(frozen=True)
class Wrench:
    """A force (N, world frame) and a torque (N m, craft frame)."""

    force_n: np.ndarray
    torque_n_m: np.ndarray


@dataclass(frozen=True)
class Allocation:
    """The craft's answer to a requested wrench: what it achieves and the
    throttles (design order) that achieve it."""

    achieved: Wrench
    throttles: np.ndarray


@dataclass(frozen=True)
class PlannedImpulse:
    """An impulsive burn of the plan: time (s) and world delta-v (m/s)."""

    time_s: float
    delta_v_m_s: np.ndarray


@dataclass(frozen=True)
class TrackingCommand:
    """One round's decision: the request, the craft's answer, the errors.
    ``phase`` is ``"trim"``, ``"coast"`` or ``"burn"``."""

    throttles: np.ndarray
    position_error_m: np.ndarray
    velocity_error_m_s: np.ndarray
    force_demand_n: np.ndarray
    torque_demand_n_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    pd_force_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    achieved_force_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    achieved_torque_n_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    attitude_error_rad: float = 0.0
    phase: str = "trim"

    @property
    def force_shortfall_n(self) -> np.ndarray:
        """What the craft did not deliver of the request."""
        return self.force_demand_n - self.achieved_force_n


@dataclass
class Burn:
    """An impulse being flown: fixed world direction, velocity still to
    deliver, start time; ``fired`` once it has delivered anything."""

    impulse: PlannedImpulse
    direction: np.ndarray
    remaining_m_s: float
    start_s: float
    duration_s: float
    thrust_n: float = math.inf
    fired: bool = False


@dataclass
class TrackingMode:
    """The tracker's state, carried across :func:`fly` calls (a game calls
    ``fly`` once per frame): the plan flown now, ON/OFF PLAN, the number of
    re-plans, ``events`` -- ``(t, "off" | "on", eps)`` per switch -- the
    phase (``"coast"``, ``"trim"``, ``"burn"``), the burn being flown, the
    plan impulses already flown (by time), and the last round's command (its
    shortfall perturbs the next)."""

    plan: object
    on_plan: bool = True
    replans: int = 0
    events: list = field(default_factory=list)
    phase: str = "coast"
    burn: Burn | None = None
    flown: set = field(default_factory=set)
    burns: list = field(default_factory=list)
    last: TrackingCommand | None = None
    last_round_s: float = 0.0
    origin: object = None
    substep_s: float | None = None


#: The mode ``fly`` keeps per craft when the caller passes none (a game that
#: calls ``fly`` once per frame keeps its burns and switch state this way);
#: renewed when a different plan is handed in.
_MODES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def tracker_mode(craft) -> TrackingMode | None:
    """The mode ``fly`` keeps for ``craft`` (``None`` before any flight):
    the plan flown, ON/OFF PLAN, the phase -- for display."""
    return _MODES.get(craft)


@dataclass(frozen=True)
class FlightReport:
    """The flight against the plan, read through the seam.  ``throttles``
    is the last round's command (``fly`` zeroes the craft's afterwards);
    ``throttle_history`` is ``(t, window, throttles)`` per round."""

    time_s: float
    rounds: int
    final_position_error_m: float
    final_velocity_error_m_s: float
    max_position_error_m: float
    fuel_impulse_n_s: float
    ideal_impulse_n_s: float
    plan: object = None
    on_plan: bool = True
    replans: int = 0
    phase: str = "coast"
    max_tracking_error: float = 0.0
    max_force_shortfall_n: float = 0.0
    throttles: np.ndarray = field(default_factory=lambda: np.zeros(0))
    throttle_history: tuple = ()

    @property
    def fuel_ratio(self) -> float:
        """Fuel used over the impulsive ideal ``mass * (|dv1| + |dv2|)``."""
        return self.fuel_impulse_n_s / self.ideal_impulse_n_s


Replanner = Callable[[object, float, np.ndarray, np.ndarray], object]


# ------------------------------------------------------------------ plans
def plan_reference(plan, t):
    """``(r, v)`` the plan prescribes at ``t``: the plan's own
    ``reference(t)`` method (HohmannPlan's delegates to
    ``orbital_plan.reference``; a collocation plan brings its own), with
    ``orbital_plan.reference`` kept for a plan object without one."""
    own = getattr(plan, "reference", None)
    if callable(own):
        position, velocity = own(t)
        return (np.asarray(position, dtype=float).reshape(3),
                np.asarray(velocity, dtype=float).reshape(3))
    return reference(plan, float(t))


def plan_impulses(plan) -> tuple:
    """The plan's impulsive burns: its own ``impulses()`` when declared;
    for a HohmannPlan the velocity steps of its legs at ``t_burn1`` and
    ``t_burn2`` (the leg before evaluated at ``t_b - 1e-6 s``); else none."""
    own = getattr(plan, "impulses", None)
    if callable(own):
        return tuple(own())
    if not isinstance(plan, HohmannPlan):
        return ()
    out = []
    for t_b in (plan.t_burn1, plan.t_burn2):
        _r, before = reference(plan, t_b - 1.0e-6)
        _r, after = reference(plan, t_b)
        out.append(PlannedImpulse(float(t_b), after - before))
    return tuple(out)


def tracking_error(position_m, velocity_m_s, r_ref, v_ref, *,
                   shortfall_m_s: float = 0.0) -> float:
    """``sqrt((|e_r|/|r_ref|)^2 + ((|e_v| + shortfall)/|v_ref|)^2)``:
    scale-free; ``shortfall_m_s`` is the velocity the last round's
    undelivered force leaves behind (``|s| h / m``)."""
    e_r = np.linalg.norm(np.asarray(position_m, float) - r_ref)
    e_v = np.linalg.norm(np.asarray(velocity_m_s, float) - v_ref)
    return float(math.hypot(e_r / np.linalg.norm(r_ref),
                            (e_v + float(shortfall_m_s))
                            / np.linalg.norm(v_ref)))


def hohmann_replanner(previous: HohmannPlan, time_s: float, position_m,
                      velocity_m_s) -> HohmannPlan:
    """The whole remaining trip from the present state, by the existing
    planner: a Hohmann transfer from the present radius and polar angle to
    the previous plan's final orbit, burn 1 now.  The previous plan is the
    starting guess; Hohmann takes only its ``mu`` and final radius from it
    (a collocation planner with this signature takes its remainder)."""
    position = np.asarray(position_m, dtype=float).reshape(3)
    return hohmann_plan(previous.mu, float(np.linalg.norm(position)),
                        previous.r2, t_burn1=float(time_s),
                        phase=float(math.atan2(position[1], position[0])))


# --------------------------------------------------------------- attitude
def pointing_axis(design: CraftDesign):
    """The craft-frame direction of the declared strongest thruster, or
    ``None`` when no thruster is strictly stronger than every thruster
    pointing elsewhere (no preferred axis)."""
    if design.thruster_count == 0:
        return None
    strongest = max(design.thrusters, key=lambda t: t.max_thrust_n)
    axis = np.asarray(strongest.direction, dtype=float)
    for thruster in design.thrusters:
        same = np.allclose(thruster.direction, axis, rtol=0.0, atol=1e-12)
        if not same and thruster.max_thrust_n >= strongest.max_thrust_n:
            return None
    return axis


def _vee(matrix) -> np.ndarray:
    return np.asarray((matrix[2, 1], matrix[0, 2], matrix[1, 0]))


def _skew(vector) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def rotation_between(a, d) -> np.ndarray:
    """The shortest rotation carrying unit ``a`` onto unit ``d``."""
    a = np.asarray(a, dtype=float)
    d = np.asarray(d, dtype=float)
    c = float(a @ d)
    if c < -1.0 + 1.0e-12:                     # half turn: any normal axis
        other = np.eye(3)[int(np.argmin(np.abs(a)))]
        n = np.cross(a, other)
        n /= np.linalg.norm(n)
        return 2.0 * np.outer(n, n) - np.eye(3)
    K = _skew(np.cross(a, d))
    return np.eye(3) + K + K @ K / (1.0 + c)


def attitude_torque_demand(design: CraftDesign, gains: TrackingGains,
                           attitude, angular_velocity_rad_s, force_demand_n,
                           *, point: bool = True):
    """Step 4 of the method: ``(tau_des (craft frame), angle error)``;
    ``point=False`` holds the attitude (rate damping only)."""
    R = np.asarray(attitude, dtype=float).reshape(3, 3)
    w = np.asarray(angular_velocity_rad_s, dtype=float).reshape(3)
    inertia = principal_inertia(design)
    desired = R
    axis = pointing_axis(design)
    force = np.asarray(force_demand_n, dtype=float).reshape(3)
    magnitude = float(np.linalg.norm(force))
    if point and axis is not None:
        deadband = gains.fuel_weight * max(t.max_thrust_n
                                           for t in design.thrusters)
        if magnitude > 0.0 and magnitude > deadband:
            desired = rotation_between(R @ axis, force / magnitude) @ R
    error = 0.5 * _vee(desired.T @ R - R.T @ desired)
    angle = float(math.acos(max(-1.0, min(1.0, 0.5 * (np.trace(
        desired.T @ R) - 1.0)))))
    wa, za = gains.attitude_frequency_rad_s, gains.attitude_damping_ratio
    tau = (inertia * (-wa**2 * error - 2.0 * za * wa * w)
           + np.cross(w, inertia * w))
    return tau, angle


# ------------------------------------------------------------- allocation
def allocate_throttles(design: CraftDesign, force_demand_n,
                       fuel_weight: float, *, attitude=None,
                       torque_demand_n_m=None) -> np.ndarray:
    """The stand-in allocation (step 5 of the method): argmin over the
    throttle box of the deviation + torque + fuel cost.  Without a torque
    demand, or for a craft whose thrusters make no torque, the torque rows
    are absent (step 3's allocation)."""
    B = actuation_matrix(design, attitude)
    if B.shape[1] == 0:
        return np.zeros(0)
    demand = np.asarray(force_demand_n, dtype=float).reshape(3)
    thrust = np.asarray([t.max_thrust_n for t in design.thrusters])
    low = np.asarray([t.throttle_min for t in design.thrusters])
    high = np.asarray([t.throttle_max for t in design.thrusters])
    scale = float(np.max(thrust))
    rows, targets = [B / scale], [demand / scale]
    T = torque_matrix(design)
    torque_scale = float(np.max(np.linalg.norm(T, axis=0)))
    if torque_demand_n_m is not None and torque_scale > 0.0:
        rows.append(T / torque_scale)
        targets.append(np.asarray(torque_demand_n_m, dtype=float).reshape(3)
                       / torque_scale)
    A, b = np.vstack(rows), np.concatenate(targets)
    linear = fuel_weight * thrust / scale

    def cost(u):
        miss = A @ u - b
        return 0.5 * (miss @ miss) + linear @ u, A.T @ miss + linear

    start = np.clip(np.linalg.lstsq(A, b, rcond=None)[0], low, high)
    result = minimize(cost, start, jac=True, method="L-BFGS-B",
                      bounds=list(zip(low, high)),
                      options={"ftol": 0.0, "gtol": 1.0e-12, "maxiter": 500})
    return np.clip(result.x, low, high)


def least_squares_allocation(design: CraftDesign, attitude,
                             fuel_weight: float, wrench: Wrench) -> Allocation:
    """:func:`allocate_throttles` answered as the seam answers: the wrench
    ``R @ B_craft @ clamp(u)``, ``torque_matrix @ clamp(u)`` the throttles
    produce at the attitude now (while the tank lasts)."""
    u = allocate_throttles(design, wrench.force_n, fuel_weight,
                           attitude=attitude,
                           torque_demand_n_m=wrench.torque_n_m)
    applied = clamp_throttles(design, u) if u.size else u
    return Allocation(
        achieved=Wrench(actuation_matrix(design, attitude) @ applied,
                        torque_matrix(design) @ applied),
        throttles=u)


class LeastSquaresAllocator:
    """Stand-in for the craft's own ``allocate(wrench)`` until the machine
    layer provides it: the least squares above, at the craft's attitude
    now."""

    def __init__(self, craft, fuel_weight: float):
        self.craft = craft
        self.fuel_weight = float(fuel_weight)

    def allocate(self, wrench: Wrench) -> Allocation:
        design = self.craft.design
        allocation = least_squares_allocation(
            design, _seam(self.craft, "attitude"), self.fuel_weight, wrench)
        if _seam(self.craft, "propellant_kg", math.inf) > 0.0:
            return allocation
        # an empty tank: thrusters that burn propellant achieve nothing
        burning = np.asarray([t.thruster_kind.propellant_per_impulse_kg_n_s
                              > 0.0 for t in design.thrusters])
        applied = np.where(burning, 0.0,
                           clamp_throttles(design, allocation.throttles))
        attitude = _seam(self.craft, "attitude")
        return Allocation(
            achieved=Wrench(actuation_matrix(design, attitude) @ applied,
                            torque_matrix(design) @ applied),
            throttles=allocation.throttles)


# ---------------------------------------------------------------- command
def _request(design, gains, attitude, angular_velocity_rad_s, force,
             allocate, *, point: str):
    """Ask the seam for ``force`` with the torque that ``point`` calls for:
    ``"hold"`` (rate damping), ``"auto"`` (hold, then point when the held
    attitude falls short) or ``"always"``.  Returns
    ``(allocation, tau, angle)``."""
    force = np.asarray(force, dtype=float).reshape(3)
    if attitude is None or not np.any(torque_matrix(design) != 0.0):
        return allocate(Wrench(force, np.zeros(3))), np.zeros(3), 0.0
    rate = (np.zeros(3) if angular_velocity_rad_s is None
            else angular_velocity_rad_s)
    tau, angle = attitude_torque_demand(design, gains, attitude, rate, force,
                                        point=(point == "always"))
    allocation = allocate(Wrench(force, tau))
    if point == "auto":
        short = np.linalg.norm(force - np.asarray(allocation.achieved.force_n,
                                                  dtype=float))
        deadband = gains.fuel_weight * max(t.max_thrust_n
                                           for t in design.thrusters)
        if short > gains.pointing_tolerance * np.linalg.norm(force) + deadband:
            tau, angle = attitude_torque_demand(design, gains, attitude, rate,
                                                force)
            allocation = allocate(Wrench(force, tau))
    return allocation, tau, angle


def _default_allocate(design, attitude, gains):
    def allocate(wrench):
        return least_squares_allocation(design, attitude, gains.fuel_weight,
                                        wrench)
    return allocate


def _command(allocation, tau, angle, e_r, e_v, demand, pd, phase):
    return TrackingCommand(
        throttles=np.asarray(allocation.throttles, dtype=float),
        position_error_m=e_r, velocity_error_m_s=e_v, force_demand_n=demand,
        torque_demand_n_m=tau, pd_force_n=pd,
        achieved_force_n=np.asarray(allocation.achieved.force_n, float),
        achieved_torque_n_m=np.asarray(allocation.achieved.torque_n_m, float),
        attitude_error_rad=angle, phase=phase)


def tracking_command(plan, design: CraftDesign, gains: TrackingGains,
                     t: float, position_m, velocity_m_s, mass_kg: float, *,
                     attitude=None, angular_velocity_rad_s=None,
                     carry_n=None, allocate=None) -> TrackingCommand:
    """A trim (steps 1, 4-6 of the method) at time ``t`` for the craft
    state given.  ``carry_n`` is last round's shortfall; ``allocate`` the
    seam (default: the stand-in least squares at ``attitude``).  Without
    ``attitude`` the craft frame is the world frame and no torque is
    demanded (step 3's craft)."""
    position = np.asarray(position_m, dtype=float).reshape(3)
    velocity = np.asarray(velocity_m_s, dtype=float).reshape(3)
    r_ref, v_ref = plan_reference(plan, float(t))
    e_r, e_v = position - r_ref, velocity - v_ref
    g_ref, g_craft = two_body_acceleration(plan.mu, np.stack([r_ref,
                                                               position]))
    w, zeta = gains.natural_frequency_rad_s, gains.damping_ratio
    pd = float(mass_kg) * ((g_ref - g_craft) - w**2 * e_r
                           - 2.0 * zeta * w * e_v)
    demand = pd
    if carry_n is not None:
        carry = np.asarray(carry_n, dtype=float).reshape(3)
        size, bound = np.linalg.norm(carry), np.linalg.norm(pd)
        demand = pd + (carry if size <= bound else carry * (bound / size))
    # never ask for more than the craft can push that way once pointed: a
    # saturated request otherwise drowns the torque rows (the stand-in
    # spends its RCS on sideways force instead of turning the engine)
    size = float(np.linalg.norm(demand))
    if size > 0.0:
        capacity, _c = burn_capacity(design, attitude, demand / size)
        if capacity > 0.0 and size > capacity:
            demand = demand * (capacity / size)
    if allocate is None:
        allocate = _default_allocate(design, attitude, gains)
    allocation, tau, angle = _request(design, gains, attitude,
                                      angular_velocity_rad_s, demand,
                                      allocate, point="auto")
    return _command(allocation, tau, angle, e_r, e_v, demand, pd, "trim")


# ------------------------------------------------------------------ burns
def burn_capacity(design: CraftDesign, attitude, direction):
    """``(thrust N, effective exhaust velocity m/s)``: the largest force
    the thrusters deliver EXACTLY along the world ``direction`` (no
    sideways part) once the craft is pointed for it -- along the pointing
    axis when there is one, else at the attitude now -- by the linear
    program ``max s : B u = s d, u in the throttle box``
    (``scipy.optimize.linprog``).  The exhaust velocity is that throttle
    set's thrust over its propellant flow (TS1.2, TS1.4; inf if
    reactionless)."""
    axis = pointing_axis(design)
    if axis is None:
        R = np.eye(3) if attitude is None else np.asarray(attitude, float)
        axis = R.T @ np.asarray(direction, dtype=float)
    B = actuation_matrix(design)                  # craft frame
    n = design.thruster_count
    if n == 0:
        return 0.0, math.inf
    low = [t.throttle_min for t in design.thrusters]
    high = [t.throttle_max for t in design.thrusters]
    cost = np.zeros(n + 1)
    cost[-1] = -1.0
    equality = np.hstack([B, -np.asarray(axis, dtype=float).reshape(3, 1)])
    result = linprog(cost, A_eq=equality, b_eq=np.zeros(3),
                     bounds=list(zip(low, high)) + [(0.0, None)],
                     method="highs")
    if not result.success or result.x[-1] <= 0.0:
        return 0.0, math.inf
    u, thrust = result.x[:n], float(result.x[-1])
    flow = float(propellant_flow_per_throttle(design) @ u)
    return thrust, (math.inf if flow <= 0.0 else thrust / flow)


def _arm_burn(mode: TrackingMode, design, gains, t, attitude, velocity,
              mass) -> Burn | None:
    """The next unflown impulse as a burn, once it is due to be armed."""
    lead = gains.slew_lead if pointing_axis(design) is not None else 0.0
    for impulse in sorted(plan_impulses(mode.plan), key=lambda i: i.time_s):
        if impulse.time_s in mode.flown:
            continue
        r_ref, v_ref = plan_reference(mode.plan, t)
        # the velocity the burn must add: to the post-burn reference; while
        # the reference is still pre-burn that is v_ref + the plan's step
        dv = v_ref - velocity
        if t < impulse.time_s:
            dv = dv + np.asarray(impulse.delta_v_m_s, float)
        size = float(np.linalg.norm(dv))
        direction = dv / size if size > 0.0 else dv
        thrust, exhaust = burn_capacity(design, attitude, direction)
        if size == 0.0 or thrust <= 0.0:
            mode.flown.add(impulse.time_s)
            continue
        duration = (mass * size / thrust if math.isinf(exhaust) else
                    mass * exhaust * (1.0 - math.exp(-size / exhaust))
                    / thrust)
        start = impulse.time_s - 0.5 * duration
        if t < start - lead:
            return None
        return Burn(impulse, direction, size, start, duration, thrust)
    return None


def _burn_command(burn: Burn, design, gains, t, h, position, velocity, mass,
                  attitude, angular_velocity, plan, allocate):
    r_ref, v_ref = plan_reference(plan, t)
    e_r, e_v = position - r_ref, velocity - v_ref
    if allocate is None:
        allocate = _default_allocate(design, attitude, gains)
    aim = burn.direction * max(t.max_thrust_n for t in design.thrusters)
    # the attitude turns onto the burn; fire once there and due
    _alloc, tau, angle = _request(design, gains, attitude, angular_velocity,
                                  aim, lambda w: Allocation(w, np.zeros(
                                      design.thruster_count)),
                                  point="always")
    aligned = (pointing_axis(design) is None or attitude is None
               or angle <= gains.burn_alignment_rad)
    force = np.zeros(3)
    if t >= burn.start_s - 1.0e-9 and aligned:
        # exactly along the burn (a saturated request would leave the
        # least squares a sideways part to spend propellant on)
        force = burn.direction * min(burn.thrust_n,
                                     mass * burn.remaining_m_s / h)
    allocation = allocate(Wrench(force, tau))
    return _command(allocation, tau, angle, e_r, e_v, force, np.zeros(3),
                    "burn")


# --------------------------------------------------------------- the loop
def _seam(craft, name, default=None):
    """An optional seam reading (a step-3 craft has no attitude)."""
    member = getattr(craft, name, None)
    if member is None:
        return default
    return member() if callable(member) else member


def fly(craft, plan, gains: TrackingGains, *, until_s: float,
        round_s: float, replanner: Replanner | None = None,
        mode: TrackingMode | None = None,
        record: list | None = None) -> FlightReport:
    """Fly the plan from the craft's present time to ``until_s``: each
    round read the seam, switch, burn / trim / coast, command the
    allocated throttles, ``advance``.  Leaves the craft's throttles at zero
    (the report keeps the last command and the per-round history).

    ``replanner`` (e.g. :func:`hohmann_replanner`) enables the off-plan
    switch; without it the plan is only flown.  ``mode`` carries the
    switch, the current plan, the burns and the last shortfall across calls
    (its ``plan`` is flown); when omitted, the mode kept for this craft
    (:func:`tracker_mode`) is used while ``plan`` is the plan it was made
    from, else a fresh one; the report's ``plan``, ``on_plan`` and ``phase`` are the
    state at the end.  ``record``, when given, receives ``(t, command)``
    for every round."""
    if not round_s > 0.0:
        raise ValueError("round_s must be positive")
    if mode is None:
        mode = _MODES.get(craft)
        if mode is None or mode.origin is not plan:
            mode = TrackingMode(plan, origin=plan)
            _MODES[craft] = mode
    # THE seam: the craft's own allocation once it has one
    allocate = getattr(craft, "allocate", None) or LeastSquaresAllocator(
        craft, gains.fuel_weight).allocate
    rounds, worst, worst_eps, worst_short = 0, 0.0, 0.0, 0.0
    history = []
    last_throttles = np.zeros(craft.design.thruster_count)
    while craft.time_s < until_s - 1.0e-9 * max(1.0, abs(until_s)):
        t = craft.time_s
        position, velocity = craft.r()
        mass = craft.mass_kg
        attitude = _seam(craft, "attitude")
        rate = _seam(craft, "angular_velocity")
        lag = 0.0
        if mode.last is not None and mode.last.phase == "trim":
            lag = (float(np.linalg.norm(mode.last.force_shortfall_n))
                   * mode.last_round_s / mass)
        r_ref, v_ref = plan_reference(mode.plan, t)
        eps = tracking_error(position, velocity, r_ref, v_ref,
                             shortfall_m_s=lag)
        if replanner is not None:
            if mode.on_plan and eps > gains.off_plan_threshold:
                mode.plan = replanner(mode.plan, t, position, velocity)
                mode.on_plan = False
                mode.replans += 1
                mode.events.append((t, "off", eps))
                mode.burn, mode.flown, mode.last = None, set(), None
                r_ref, v_ref = plan_reference(mode.plan, t)
            elif not mode.on_plan and eps < gains.on_plan_threshold:
                mode.on_plan = True
                mode.events.append((t, "on", eps))
        window = min(round_s, until_s - t)
        if mode.burn is None:
            mode.burn = _arm_burn(mode, craft.design, gains, t, attitude,
                                  velocity, mass)
        if mode.burn is not None:
            command = _burn_command(mode.burn, craft.design, gains, t, window,
                                    position, velocity, mass, attitude, rate,
                                    mode.plan, allocate)
        else:
            e_r = np.linalg.norm(position - r_ref) / np.linalg.norm(r_ref)
            e_v = np.linalg.norm(velocity - v_ref)
            substep = min(round_s, mode.substep_s or round_s)
            g = np.linalg.norm(two_body_acceleration(mode.plan.mu,
                                                     position[None, :])[0])
            # r()'s velocity is about g dt / 2 off its position: not an
            # error to fight
            v_band = max(gains.coast_velocity_deadband
                         * np.linalg.norm(v_ref), g * substep)
            # ... and the PD answers that stagger with a position offset of
            # up to 2 (g dt / 2) / w
            p_band = max(gains.coast_position_deadband,
                         g * substep / gains.natural_frequency_rad_s
                         / np.linalg.norm(r_ref))
            if mode.phase != "trim" and (e_r > p_band or e_v > v_band):
                mode.phase = "trim"
            elif mode.phase == "trim" and (e_r < 0.5 * p_band
                                           and e_v < 0.5 * v_band):
                mode.phase = "coast"
            if mode.phase == "trim":
                carry = (mode.last.force_shortfall_n
                         if mode.last is not None and mode.last.phase == "trim"
                         else None)
                command = tracking_command(
                    mode.plan, craft.design, gains, t, position, velocity,
                    mass, attitude=attitude, angular_velocity_rad_s=rate,
                    carry_n=carry, allocate=allocate)
            else:
                allocation, tau, angle = _request(
                    craft.design, gains, attitude, rate, np.zeros(3),
                    allocate, point="hold")
                command = _command(allocation, tau, angle, position - r_ref,
                                   velocity - v_ref, np.zeros(3),
                                   np.zeros(3), "coast")
        worst = max(worst, float(np.linalg.norm(command.position_error_m)))
        worst_eps = max(worst_eps, eps)
        if command.phase == "trim":
            worst_short = max(worst_short,
                              float(np.linalg.norm(command.force_shortfall_n)))
        if record is not None:
            record.append((t, command))
        craft.throttle(command.throttles)
        advanced = craft.advance(window)
        # the dt system's continuation step, when the craft reports it
        # (``OrbitalJumper.advance`` returns (advanced, dt_next, telemetry))
        if isinstance(advanced, tuple) and len(advanced) >= 2:
            mode.substep_s = float(advanced[1])
        last_throttles = command.throttles
        history.append((t, window, command.throttles))
        if command.phase == "burn":
            burn = mode.burn
            mean_mass = 0.5 * (mass + craft.mass_kg)
            delivered = (float(command.achieved_force_n @ burn.direction)
                         * window / mean_mass)
            if delivered > 0.0:
                burn.fired = True
            burn.remaining_m_s -= delivered
            # done once what is left is inside the allocation's fuel
            # deadband (the trims take it from there)
            deadband = gains.fuel_weight * max(
                t.max_thrust_n for t in craft.design.thrusters)
            if burn.fired and (burn.remaining_m_s * craft.mass_kg / window
                               <= 1.5 * deadband):
                mode.flown.add(burn.impulse.time_s)
                mode.burns.append((burn.impulse.time_s, burn.start_s,
                                   t + window, burn.remaining_m_s))
                mode.burn = None
                mode.phase = "trim"
        else:
            mode.phase = command.phase
        mode.last, mode.last_round_s = command, window
        rounds += 1
    if mode.burn is not None:
        mode.phase = "burn"
    craft.throttle(np.zeros(craft.design.thruster_count))
    position, velocity = craft.r()
    r_ref, v_ref = plan_reference(mode.plan, craft.time_s)
    error_r = float(np.linalg.norm(position - r_ref))
    ideal = getattr(mode.plan, "ideal_delta_v", math.nan)
    return FlightReport(
        time_s=craft.time_s, rounds=rounds,
        final_position_error_m=error_r,
        final_velocity_error_m_s=float(np.linalg.norm(velocity - v_ref)),
        max_position_error_m=max(worst, error_r),
        fuel_impulse_n_s=craft.fuel_impulse_n_s,
        ideal_impulse_n_s=craft.mass_kg * ideal,
        plan=mode.plan, on_plan=mode.on_plan, replans=mode.replans,
        phase=mode.phase, max_tracking_error=worst_eps,
        max_force_shortfall_n=worst_short,
        throttles=np.asarray(last_throttles, dtype=float),
        throttle_history=tuple(history))
