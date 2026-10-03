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
   A craft with a pointing axis is armed early by the attitude PD's own
   settling time from its present angle off the burn and turns the
   axis onto the burn direction; it lights only within
   ``burn_alignment_rad`` and, once lit, burns on within
   ``burn_release_rad``.  The pointed axis is the line the burn thrust
   actually takes: a craft with its own allocation is asked for thrust
   along its strongest thruster's axis with no torque, and the achieved
   force's direction is pointed (a gimballed engine whose line misses the
   centre of mass leans off the axis to cancel the moment).  The burn is closed on DELIVERED velocity
   (achieved force over mass, every round), not on time, so mass loss and
   alignment are paid for exactly.  The PD is off during a burn.
4. Trims (the PD only trims).  Between burns the computed-acceleration PD

       F_pd = m ([g(r_ref) - g(r)] - w^2 e_r - 2 zeta w e_v)

   fires only outside a coast deadband: trimming starts when
   ``|e_r| > p_band`` or ``|e_v| > v_band`` and stops when both are below
   half of theirs, with ``p_band = coast_position_deadband * |r_ref|`` and
   ``v_band = coast_velocity_deadband * |v_ref|``.  ``r()`` reads
   position and velocity at the same instant, so every error it shows is
   real; the defaults (1e-6, 1e-5: 7 m and 0.075 m/s at 7000 km) sit at
   the integrator's own coast error (6.9 m per orbit on the jumper).
   Measured on the main+RCS transfer: wider bands cost MORE (1e-5/1e-4:
   1.126 x ideal against 1.044 x) -- a late trim is a large one, and each
   costs a slew; see the step-5 continuation.  A trim request carries last round's shortfall forward,
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
            + fuel_weight * sum_k p_k u_k

   over the declared throttle box by L-BFGS-B (``scipy.optimize``), with
   ``p_k`` the propellant mass flow ``T_k / (I_sp,k g_0)`` scaled so the
   most efficient kind's strongest thruster costs 1 (:func:`fuel_price`;
   the craft's allocator prices fuel the same way).
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
    the PD leaves alone.  ``burn_alignment_rad``: how close
    the pointing axis must be to a burn's direction before it fires;
    ``burn_release_rad``: how far it may stray once lit before the burn
    pauses (1 - cos 0.2 = 2 % cosine loss).
    ``slew_lead_s``: how long before a burn's start the craft turns onto
    it (``None``: the attitude PD's settling time from the present angle,
    :meth:`slew_lead`).
    ``off_plan_threshold`` / ``on_plan_threshold``: the hysteresis band on
    the scale-free tracking error (upper, lower)."""

    natural_frequency_rad_s: float = 0.02
    damping_ratio: float = 1.0
    fuel_weight: float = 1.0e-4
    attitude_frequency_rad_s: float = 0.05
    attitude_damping_ratio: float = 1.0
    pointing_tolerance: float = 0.1
    coast_position_deadband: float = 1.0e-6
    coast_velocity_deadband: float = 1.0e-5
    burn_alignment_rad: float = 0.05
    burn_release_rad: float = 0.2
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
                and self.burn_release_rad >= self.burn_alignment_rad
                and (self.slew_lead_s is None or self.slew_lead_s >= 0.0)
                and 0.0 < self.on_plan_threshold < self.off_plan_threshold):
            raise ValueError(f"invalid tracking gains {self}")

    def slew_lead(self, angle_rad: float, round_s: float) -> float:
        """How long before a burn's start to turn onto it from
        ``angle_rad`` off: ``slew_lead_s`` when declared, else the attitude
        PD's own settling time to ``burn_alignment_rad`` -- the envelope
        ``(1 + z w t) exp(-z w t) = alignment / angle`` of its
        critically-damped response, solved for ``t`` -- plus two rounds
        (the throttles are held a round; the alignment is read a round
        late)."""
        if self.slew_lead_s is not None:
            return self.slew_lead_s
        ratio = self.burn_alignment_rad / max(angle_rad, 1.0e-12)
        x = 0.0
        if ratio < 1.0:                    # Newton on (1+x)e^-x = ratio
            x = 1.0
            for _ in range(60):
                f = (1.0 + x) * math.exp(-x) - ratio
                x -= f / (-x * math.exp(-x))
                x = max(x, 1.0e-9)
        rate = self.attitude_frequency_rad_s * self.attitude_damping_ratio
        return x / rate + 2.0 * round_s


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
    ``phase`` is ``"trim"``, ``"coast"`` or ``"burn"``; ``gimbal_rad`` the
    gimbal commands when the craft's own allocation answers with them."""

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
    gimbal_rad: np.ndarray | None = None
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
    #: ``"main"`` (the pointing thruster's full thrust, cut at the round
    #: that ends on the predicted cutoff) -> ``"residual"`` (what is left
    #: below the main engine's minimum throttle, by the small thrusters)
    stage: str = "main"
    residual_rounds: int = 0


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
    #: The round grid: rounds end at ``anchor_s + k * round_s`` (the time of
    #: the mode's first flight) or at a plan event (a burn's start, its
    #: predicted cutoff) -- never at a caller's frame boundary.  A round a
    #: caller's ``until_s`` cuts is finished by the next ``fly`` with the
    #: SAME command (``round_end_s``, ``round_command``).
    anchor_s: float | None = None
    round_start_s: float | None = None
    round_end_s: float | None = None
    round_command: TrackingCommand | None = None
    round_cut: bool = False
    #: The dt system's continuation step (``advance``'s ``dt_next``): the
    #: substeps the slew-law prediction of a cutoff assumes.
    dt_next_s: float | None = None
    last_eps: float = 0.0


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
                           *, point: bool = True, inertia=None,
                           axis=None):
    """Step 5 of the method: ``(tau_des (craft frame), angle error)``;
    ``point=False`` holds the attitude (rate damping only).  ``inertia``
    is the craft's full 3x3 tensor about its centre of mass (the machine
    craft has products of inertia); default the design's principal
    moments.  ``axis`` is the craft-frame thrust line to point (default
    :func:`pointing_axis`)."""
    R = np.asarray(attitude, dtype=float).reshape(3, 3)
    w = np.asarray(angular_velocity_rad_s, dtype=float).reshape(3)
    inertia = (np.diag(principal_inertia(design)) if inertia is None
               else np.asarray(inertia, dtype=float).reshape(3, 3))
    desired = R
    if axis is None:
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
    tau = (inertia @ (-wa**2 * error - 2.0 * za * wa * w)
           + np.cross(w, inertia @ w))
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
    linear = fuel_weight * fuel_price(design)

    def cost(u):
        miss = A @ u - b
        return 0.5 * (miss @ miss) + linear @ u, A.T @ miss + linear

    start = np.clip(np.linalg.lstsq(A, b, rcond=None)[0], low, high)
    result = minimize(cost, start, jac=True, method="L-BFGS-B",
                      bounds=list(zip(low, high)),
                      options={"ftol": 0.0, "gtol": 1.0e-12, "maxiter": 500})
    return np.clip(result.x, low, high)


def fuel_price(design: CraftDesign) -> np.ndarray:
    """Per-thruster fuel price per unit throttle, as the craft prices it:
    propellant MASS flow, ``T_k / (I_sp,k g_0)`` (TS1.2, TS1.4), scaled so
    the most efficient kind's strongest thruster costs 1 -- the allocation's
    deadband stays ``fuel_weight * T_max`` for it.  A design that burns no
    propellant (the reactionless ``ideal`` kind) is priced by impulse,
    ``T_k / T_max``."""
    thrust = np.asarray([t.max_thrust_n for t in design.thrusters], float)
    per_impulse = np.asarray([t.thruster_kind.propellant_per_impulse_kg_n_s
                              for t in design.thrusters], float)
    if not np.any(per_impulse > 0.0):
        return thrust / float(np.max(thrust))
    best = float(np.min(per_impulse[per_impulse > 0.0]))
    return thrust * per_impulse / (float(np.max(thrust)) * best)


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


class Actuation:
    """Everything the tracker asks of the craft beyond ``r()`` and
    ``advance``, in one place.

    ``allocate(wrench)``: the seam -- the craft's own ``allocate`` when it
    has one (the machine craft: gimbals, cones, feeds, slew limits; it
    APPLIES its commands, declared by ``applies_allocation``), else the
    least-squares stand-in at the attitude now (or an explicit callable).
    ``probe(wrench)``: the same question without commanding anything (the
    machine's ``apply=False``).  ``inertia()``: the craft's full tensor
    when it reports one.  ``can_torque``: whether any thruster makes torque
    -- from the fixed design's ``torque_matrix``, or, for a craft with its
    own allocation, by probing it with a pure torque.
    ``capacity(attitude, d)``: the largest force along world ``d`` once
    pointed, and its exhaust velocity -- by the LP over a fixed design, or
    by probing the craft's own allocation with the pointing thruster's
    thrust along it and reading the achieved force (cached per craft-frame direction; the
    pointing axis makes it one direction)."""

    def __init__(self, design, gains, *, craft=None, allocate=None,
                 attitude=None, round_s=None):
        self.design, self.gains, self.craft = design, gains, craft
        self.round_s = round_s
        own = None if craft is None else getattr(craft, "allocate", None)
        self.own = allocate is None and own is not None
        self.applies = bool(self.own and craft.applies_allocation)
        if allocate is not None:
            self._allocate = self._probe = allocate
        elif self.own:
            self._allocate = self._own_allocate
            self._probe = (self._own_probe if self.applies
                           else self._own_allocate)
        else:
            stand_in = (LeastSquaresAllocator(craft, gains.fuel_weight)
                        .allocate if craft is not None else
                        _default_allocate(design, attitude, gains))
            self._allocate = self._probe = stand_in
        self._capacity = {}
        self._line = {}
        self._torque = None

    def _own_allocate(self, wrench):
        if self.applies and self.round_s is not None:
            return self.craft.allocate(wrench, round_s=self.round_s)
        return self.craft.allocate(wrench)

    def _own_probe(self, wrench):
        return self.craft.allocate(wrench, apply=False)

    def allocate(self, wrench):
        return self._allocate(wrench)

    def probe(self, wrench):
        return self._probe(wrench)

    def inertia(self):
        tensor = (None if self.craft is None
                  else _seam(self.craft, "inertia_tensor"))
        return (np.diag(principal_inertia(self.design)) if tensor is None
                else np.asarray(tensor, dtype=float).reshape(3, 3))

    @property
    def can_torque(self) -> bool:
        if self._torque is None:
            if self.own:
                arm = max(float(np.linalg.norm(t.position_m))
                          for t in self.design.thrusters)
                least = min(t.max_thrust_n for t in self.design.thrusters)
                probe = self.probe(Wrench(np.zeros(3), np.asarray(
                    (0.0, 0.0, arm * least))))
                self._torque = bool(np.linalg.norm(
                    probe.achieved.torque_n_m) > 0.0)
            else:
                self._torque = bool(np.any(torque_matrix(self.design)
                                           != 0.0))
        return self._torque

    def capacity(self, attitude, direction):
        if not self.own:
            return burn_capacity(self.design, attitude, direction)
        axis = pointing_axis(self.design)
        R = np.eye(3) if attitude is None else np.asarray(attitude, float)
        craft_dir = (np.asarray(axis, float) if axis is not None
                     else R.T @ np.asarray(direction, dtype=float))
        key = tuple(np.round(craft_dir, 9))
        if key not in self._capacity:
            world = R @ craft_dir
            # the pointing thruster's own thrust, not the sum: a request
            # beyond it would also light the attitude thrusters, leaving no
            # torque authority to hold the burn (measured: the machine
            # craft's forward RCS joined the main engine and it tumbled)
            own = max(t.max_thrust_n for t in self.design.thrusters)
            probe = self.probe(Wrench(own * world, np.zeros(3)))
            thrust = max(0.0, float(np.asarray(probe.achieved.force_n,
                                               float) @ world))
            achieved = np.asarray(probe.achieved.force_n, float)
            size = float(np.linalg.norm(achieved))
            self._line[key] = (R.T @ achieved / size if size > 0.0
                               else craft_dir)
            u = np.asarray(probe.throttles, dtype=float)
            flow = float(sum(
                uk * t.max_thrust_n
                * t.thruster_kind.propellant_per_impulse_kg_n_s
                for uk, t in zip(u, self.design.thrusters)))
            self._capacity[key] = (thrust, math.inf if flow <= 0.0
                                   else thrust / flow)
        return self._capacity[key]

    def pointing(self, attitude):
        """The craft-frame line the burn thrust actually takes: the declared
        pointing axis for a fixed design; for a craft with its own
        allocation, the direction of the force it achieves when asked for
        thrust along that axis with no torque (a gimballed engine whose
        line misses the centre of mass is turned to cancel the moment, so
        the thrust leans off the axis) -- pointing THAT line keeps the burn
        on its direction."""
        axis = pointing_axis(self.design)
        if axis is None or not self.own:
            return axis
        R = np.eye(3) if attitude is None else np.asarray(attitude, float)
        self.capacity(attitude, R @ axis)
        return self._line[tuple(np.round(np.asarray(axis, float), 9))]


# ---------------------------------------------------------------- command
def _request(design, gains, attitude, angular_velocity_rad_s, force,
             actuation, *, point: str):
    """Ask the seam for ``force`` with the torque that ``point`` calls for:
    ``"hold"`` (rate damping), ``"auto"`` (hold, then point when the held
    attitude falls short -- decided on a probe, commanded once),
    ``"always"``, or ``"never"`` (the pointing torque only; nothing is
    allocated).  Returns ``(allocation, tau, angle)``."""
    force = np.asarray(force, dtype=float).reshape(3)
    allocate = actuation.allocate
    if attitude is None or not actuation.can_torque:
        if point == "never":
            return None, np.zeros(3), 0.0
        return allocate(Wrench(force, np.zeros(3))), np.zeros(3), 0.0
    rate = (np.zeros(3) if angular_velocity_rad_s is None
            else angular_velocity_rad_s)
    inertia = actuation.inertia()
    axis = actuation.pointing(attitude)
    tau, angle = attitude_torque_demand(design, gains, attitude, rate, force,
                                        point=(point != "hold"),
                                        inertia=inertia, axis=axis)
    if point == "never":                   # the torque alone, no command
        return None, tau, angle
    if point == "auto":                    # held first
        tau, angle = attitude_torque_demand(design, gains, attitude, rate,
                                            force, point=False,
                                            inertia=inertia, axis=axis)
    # "auto" decides on a probe, then commands once
    allocation = (actuation.probe if point == "auto" else allocate)(
        Wrench(force, tau))
    if point == "auto":
        short = np.linalg.norm(force - np.asarray(allocation.achieved.force_n,
                                                  dtype=float))
        deadband = gains.fuel_weight * max(t.max_thrust_n
                                           for t in design.thrusters)
        if short > gains.pointing_tolerance * np.linalg.norm(force) + deadband:
            tau, angle = attitude_torque_demand(design, gains, attitude, rate,
                                                force, inertia=inertia,
                                                axis=axis)
            if angle > gains.burn_release_rad:
                # too far off to push usefully: turn first, as a burn does
                # (a force asked for meanwhile competes with the torque and
                # is spent sideways); the request stays the trim's, so its
                # shortfall is carried and counted
                return allocate(Wrench(np.zeros(3), tau)), tau, angle
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
        attitude_error_rad=angle, phase=phase,
        gimbal_rad=(None if getattr(allocation, "gimbal_rad", None) is None
                    else np.asarray(allocation.gimbal_rad, dtype=float)))


def tracking_command(plan, design: CraftDesign, gains: TrackingGains,
                     t: float, position_m, velocity_m_s, mass_kg: float, *,
                     attitude=None, angular_velocity_rad_s=None,
                     carry_n=None, allocate=None,
                     actuation=None) -> TrackingCommand:
    """A trim (steps 1, 4-6 of the method) at time ``t`` for the craft
    state given.  ``carry_n`` is last round's shortfall; ``allocate`` the
    seam (default: the stand-in least squares at ``attitude``), or
    ``actuation`` the whole craft-facing seam (:class:`Actuation`).  Without
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
    if actuation is None:
        actuation = Actuation(design, gains, allocate=allocate,
                              attitude=attitude)
    size = float(np.linalg.norm(demand))
    if size > 0.0:
        capacity, _c = actuation.capacity(attitude, demand / size)
        if capacity > 0.0 and size > capacity:
            demand = demand * (capacity / size)
    allocation, tau, angle = _request(design, gains, attitude,
                                      angular_velocity_rad_s, demand,
                                      actuation, point="auto")
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
              mass, round_s, actuation) -> Burn | None:
    """The next unflown impulse as a burn, once it is due to be armed: its
    start minus the time the attitude needs to turn onto it from where it
    points now."""
    axis = actuation.pointing(attitude)
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
        thrust, exhaust = actuation.capacity(attitude, direction)
        if size == 0.0 or thrust <= 0.0:
            mode.flown.add(impulse.time_s)
            continue
        duration = (mass * size / thrust if math.isinf(exhaust) else
                    mass * exhaust * (1.0 - math.exp(-size / exhaust))
                    / thrust)
        start = impulse.time_s - 0.5 * duration
        lead = 0.0
        if axis is not None and attitude is not None:
            pointing = np.asarray(attitude, float) @ axis
            angle = math.acos(max(-1.0, min(1.0, float(pointing
                                                       @ direction))))
            lead = gains.slew_lead(angle, round_s)
        if t < start - lead:
            return None
        return Burn(impulse, direction, size, start, duration, thrust)
    return None


# ------------------------------------------------------- rounds and cutoff
def _time_tol(t: float) -> float:
    return 1.0e-9 * max(1.0, abs(float(t)))


def _grid_end(anchor_s: float, t: float, round_s: float) -> float:
    """The first round-grid point (``anchor_s + k round_s``) after ``t``."""
    end = anchor_s + (math.floor((t - anchor_s) / round_s) + 1) * round_s
    while end - t <= _time_tol(t):
        end += round_s
    return end


def _substeps(length_s: float, dt_next_s: float | None) -> list:
    """The substeps the dt system takes over a round of ``length_s``: its
    continuation step, the last one clipped to land."""
    if dt_next_s is None or not dt_next_s > 0.0 or dt_next_s >= length_s:
        return [length_s]
    count = int(math.floor(length_s / dt_next_s))
    out = [dt_next_s] * count
    rest = length_s - count * dt_next_s
    if rest > _time_tol(length_s):
        out.append(rest)
    return out


def _slew_impulses(thrusters, state, segments) -> tuple:
    """The slew and delivered-throttle laws (``orbital_actuation.
    throttle_state_rhs``, ``delivered_throttle_rhs``) on the host: each
    substep the state moves toward the clamped command by at most
    ``throttle_slew_per_s * dt`` and the thruster delivers its NEW state,
    or nothing below its deadband.  ``segments`` is ``((command,
    substeps), ...)``.  Returns ``(impulse per thruster N s, final
    state)``."""
    thrust = np.asarray([t.max_thrust_n for t in thrusters], float)
    low = np.asarray([t.throttle_min for t in thrusters], float)
    high = np.asarray([t.throttle_max for t in thrusters], float)
    rate = np.asarray([t.throttle_slew_per_s for t in thrusters], float)
    band = np.asarray([t.deadband for t in thrusters], float)
    s = np.asarray(state, dtype=float).copy()
    impulse = np.zeros(len(thrusters))
    for command, steps in segments:
        target = np.clip(np.asarray(command, float), low, high)
        for dt in steps:
            reach = rate * dt
            s = s + np.clip(target - s, -reach, reach)
            impulse += thrust * np.where(s >= band, s, 0.0) * dt
    return impulse, s


def _thrust_lines(craft, gimbal_rad=None) -> np.ndarray:
    """(n, 3) world unit thrust directions now (or at ``gimbal_rad``)."""
    thrusters = craft.design.thrusters
    attitude = _seam(craft, "attitude")
    R = np.eye(3) if attitude is None else np.asarray(attitude, float)
    if gimbal_rad is None:
        states = _seam(craft, "gimbal_states")
        gimbal_rad = (np.zeros((len(thrusters), 2)) if states is None
                      else np.asarray(states, float))
    gimbal_rad = np.asarray(gimbal_rad, float).reshape(len(thrusters), 2)
    return np.stack([R @ t.direction_at(*g)
                     for t, g in zip(thrusters, gimbal_rad)]) if len(
        thrusters) else np.zeros((0, 3))


def _throttle_states(craft, fallback) -> np.ndarray:
    """The throttle STATE columns (a slewing craft), else the command
    (a fixed design's throttle is its state)."""
    states = _seam(craft, "throttle_states")
    return np.asarray(fallback if states is None else states, dtype=float)


def _per_impulse(thrusters) -> np.ndarray:
    return np.asarray([t.thruster_kind.propellant_per_impulse_kg_n_s
                       for t in thrusters], float)


def _delta_v_along(craft, allocation, direction, mass, length_s, mode,
                   round_s) -> Callable[[float], float]:
    """``dv(tau)``: the velocity along ``direction`` the craft gains when
    ``allocation`` is commanded for ``tau`` s and every thruster is then
    commanded off -- the cut round by the slew law, then its spool-down
    TAIL over the rounds the grid places after the cut, at the dt system's
    continuation substeps."""
    thrusters = craft.design.thrusters
    command = np.asarray(allocation.throttles, float)
    state0 = _throttle_states(craft, command)
    gimbal = getattr(allocation, "gimbal_rad", None)
    project = _thrust_lines(craft, gimbal) @ np.asarray(direction, float)
    per_impulse = _per_impulse(thrusters)
    off = np.zeros(len(thrusters))
    t0 = float(craft.time_s)

    def dv(tau: float) -> float:
        if tau <= 0.0:
            segments = []
        else:
            segments = [(command, _substeps(tau, mode.dt_next_s))]
        cut = t0 + tau
        first = _grid_end(mode.anchor_s, cut, round_s) - cut
        segments += [(off, _substeps(first, mode.dt_next_s))]
        segments += [(off, _substeps(round_s, mode.dt_next_s))] * 2
        impulse, _s = _slew_impulses(thrusters, state0, segments)
        mean_mass = mass - 0.5 * float(impulse @ per_impulse)
        return float(impulse @ project) / mean_mass
    return dv


def _cutoff(dv, remaining: float, length_s: float, tol: float):
    """``(tau, overshoot)``: the shortest ``tau <= length_s`` whose
    delivery (tail included) reaches ``remaining`` (bisection: the
    delivery never falls as the cut moves later), or ``None`` when the
    whole round does not reach it."""
    if dv(length_s) < remaining - tol:
        return None, 0.0
    low, high = 0.0, length_s
    if dv(0.0) >= remaining - tol:
        return 0.0, dv(0.0) - remaining
    for _ in range(60):
        mid = 0.5 * (low + high)
        if dv(mid) >= remaining:
            high = mid
        else:
            low = mid
        if high - low <= 1.0e-9 * length_s:
            break
    return high, dv(high) - remaining


def _residual_force(design, burn: Burn) -> float:
    """The force the residual stage asks for: under half the smallest force
    a deadbanded thruster can deliver once lit (``deadband * T``), so the
    allocation leaves every such thruster dark and the small thrusters
    (RCS) take it; a design without deadbands asks for the burn's own
    thrust."""
    least = min((t.deadband * t.max_thrust_n for t in design.thrusters
                 if t.deadband > 0.0), default=math.inf)
    return min(burn.thrust_n, 0.45 * least)


def _burn_round(burn: Burn, craft, gains, mode, t, length_s, round_s,
                position, velocity, mass, attitude, angular_velocity,
                actuation):
    """One burn round from ``t``: ``(command, round length, cut)``.

    Lit, the round asks for the burn's full thrust along its direction and
    ENDS at the predicted cutoff when that falls inside it: the cut time
    solves ``dv(tau) = remaining`` with ``dv`` the slew law's delivery,
    spool-down tail included (:func:`_delta_v_along`).  When the main
    engine cannot deliver that little (its deadband: the delivery jumps
    past the remaining), or once it is cut, the residual stage asks the
    small thrusters (:func:`_residual_force`) for what is left, again in a
    round that ends when it is delivered."""
    design = craft.design
    r_ref, v_ref = plan_reference(mode.plan, t)
    e_r, e_v = position - r_ref, velocity - v_ref
    aim = burn.direction * max(th.max_thrust_n for th in design.thrusters)
    _none, tau, angle = _request(design, gains, attitude, angular_velocity,
                                 aim, actuation, point="never")
    gate = gains.burn_release_rad if burn.fired else gains.burn_alignment_rad
    aligned = (actuation.pointing(attitude) is None or attitude is None
               or angle <= gate)
    tol = _burn_tolerance(design, gains, mass, round_s)
    actuation.round_s = length_s
    if not (t >= burn.start_s - _time_tol(t) and aligned):
        allocation = actuation.allocate(Wrench(np.zeros(3), tau))
        return (_command(allocation, tau, angle, e_r, e_v, np.zeros(3),
                         np.zeros(3), "burn"), length_s, False)

    def attempt(force, remaining, direction):
        actuation.round_s = length_s
        allocation = actuation.allocate(Wrench(force, tau))
        dv = _delta_v_along(craft, allocation, direction, mass, length_s,
                            mode, round_s)
        cut, over = _cutoff(dv, remaining, length_s, tol)
        return allocation, cut, over

    force = np.zeros(3)
    cut = None
    if burn.stage == "main":
        force = burn.direction * burn.thrust_n
        allocation, cut, over = attempt(force, burn.remaining_m_s,
                                        burn.direction)
        if cut is not None and over > tol:
            # the main engine cannot deliver that little (its deadband)
            burn.stage = "residual"
    if burn.stage == "residual":
        sign = 1.0 if burn.remaining_m_s >= 0.0 else -1.0
        direction = sign * burn.direction
        force = direction * _residual_force(design, burn)
        allocation, cut, over = attempt(force, abs(burn.remaining_m_s),
                                        direction)
    if cut is None:
        return (_command(allocation, tau, angle, e_r, e_v, force,
                         np.zeros(3), "burn"), length_s, False)
    if cut < length_s - _time_tol(t):
        if cut <= _time_tol(t):
            # already delivered (the tail does it): nothing more to ask
            allocation = actuation.allocate(Wrench(np.zeros(3), tau))
            force = np.zeros(3)
            return (_command(allocation, tau, angle, e_r, e_v, force,
                             np.zeros(3), "burn"), length_s, True)
        actuation.round_s = cut
        allocation = actuation.allocate(Wrench(force, tau))
        length_s = cut
    return (_command(allocation, tau, angle, e_r, e_v, force, np.zeros(3),
                     "burn"), length_s, True)


def _burn_tolerance(design, gains, mass, round_s) -> float:
    """The velocity a burn may leave: the allocation's fuel deadband force
    (``fuel_weight * T_max``) over a round."""
    return (gains.fuel_weight * max(t.max_thrust_n for t in design.thrusters)
            * round_s / mass)


def _outstanding_tail(craft, burn: Burn, mass, mode, round_s) -> float:
    """The spool-down delivery still to come along the burn from the
    present throttle states (every thruster commanded off)."""
    thrusters = craft.design.thrusters
    off = np.zeros(len(thrusters))

    class _Off:
        throttles = off
        gimbal_rad = None
    return _delta_v_along(craft, _Off, burn.direction, mass, 0.0, mode,
                          round_s)(0.0)


# --------------------------------------------------------------- the loop
def _coast(craft, gains, attitude, rate, actuation):
    """A coast round: no force, the attitude held (rate damping).  A hold
    torque inside the torque deadband -- the fuel deadband force
    ``fuel_weight * T_max`` at the longest thruster arm -- is not asked for:
    the round commands every thruster off without consulting the
    allocation (what it would answer)."""
    design = craft.design
    allocation, tau, angle = _request(design, gains, attitude, rate,
                                      np.zeros(3), actuation, point="never")
    arm = max((float(np.linalg.norm(t.position_m)) for t in design.thrusters),
              default=0.0)
    deadband = gains.fuel_weight * max((t.max_thrust_n
                                        for t in design.thrusters),
                                       default=0.0) * arm
    if float(np.linalg.norm(tau)) > deadband:
        allocation, tau, angle = _request(design, gains, attitude, rate,
                                          np.zeros(3), actuation,
                                          point="hold")
        return allocation, tau, angle
    off = np.zeros(design.thruster_count)
    if craft.applies_allocation:
        craft.throttle(off)
    return Allocation(Wrench(np.zeros(3), np.zeros(3)), off), tau, angle


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
    for every round.

    Rounds are the controller's: they end on the grid ``anchor + k
    round_s`` (the mode's first flight) or at a plan event -- a burn's
    start, a burn's predicted cutoff -- never because of ``until_s``.  The
    flight still stops at ``until_s``; a round it cuts keeps its command
    and is finished by the next call, so ``fly(t0 -> t2)`` and ``fly(t0 ->
    t1); fly(t1 -> t2)`` decide the same rounds (the dt system's substep
    partition alone differs).  ``rounds`` and ``throttle_history`` count
    the ``advance`` calls."""
    if not round_s > 0.0:
        raise ValueError("round_s must be positive")
    if mode is None:
        mode = _MODES.get(craft)
        if mode is None or mode.origin is not plan:
            mode = TrackingMode(plan, origin=plan)
            _MODES[craft] = mode
    # THE seam: the craft's own allocation once it has one
    actuation = Actuation(craft.design, gains, craft=craft, round_s=round_s)
    rounds, worst, worst_eps, worst_short = 0, 0.0, 0.0, 0.0
    history = []
    last_throttles = np.zeros(craft.design.thruster_count)
    if mode.anchor_s is None:
        mode.anchor_s = float(craft.time_s)
    impulses = _seam(craft, "thruster_impulses_n_s")
    while craft.time_s < until_s - _time_tol(until_s):
        t = craft.time_s
        if (mode.round_end_s is not None and mode.round_command is not None
                and mode.round_start_s - _time_tol(t) <= t
                < mode.round_end_s - _time_tol(t)):
            # a round the caller's frame cut: finish it with its command
            command = mode.round_command
            craft.throttle(command.throttles)
            if command.gimbal_rad is not None and callable(
                    getattr(craft, "gimbal", None)):
                craft.gimbal(command.gimbal_rad)
        else:
            command = _decide_round(craft, mode, gains, actuation, t,
                                    round_s, replanner)
            if not craft.applies_allocation:   # the machine applied its own
                craft.throttle(command.throttles)
            worst = max(worst, float(np.linalg.norm(
                command.position_error_m)))
            worst_eps = max(worst_eps, mode.last_eps)
            if command.phase == "trim":
                worst_short = max(worst_short, float(np.linalg.norm(
                    command.force_shortfall_n)))
            if record is not None:
                record.append((t, command))
        end = mode.round_end_s
        window = (end - t if end <= until_s + _time_tol(until_s)
                  else until_s - t)
        lines0 = _thrust_lines(craft) if impulses is not None else None
        mass0 = craft.mass_kg
        _advanced, dt_next, _telemetry = craft.advance(window)
        mode.dt_next_s = float(dt_next)
        last_throttles = command.throttles
        history.append((t, window, command.throttles))
        rounds += 1
        burn = mode.burn
        if command.phase == "burn" and burn is not None:
            # the velocity delivered along the burn, read from the thrust
            # the craft integrated (per-thruster impulse on its thrust line)
            mean_mass = 0.5 * (mass0 + craft.mass_kg)
            if impulses is not None:
                now = _seam(craft, "thruster_impulses_n_s")
                lines = lines0 + _thrust_lines(craft)
                lines /= np.maximum(np.linalg.norm(lines, axis=1,
                                                   keepdims=True), 1e-300)
                delivered = float((now - impulses) @ (lines
                                                      @ burn.direction)
                                  ) / mean_mass
            else:
                delivered = (float(command.achieved_force_n @ burn.direction)
                             * window / mean_mass)
            if delivered > 0.0 and burn.stage == "main":
                burn.fired = True
            burn.remaining_m_s -= delivered
        impulses = _seam(craft, "thruster_impulses_n_s")
        if craft.time_s >= mode.round_end_s - _time_tol(craft.time_s):
            _finish_round(craft, mode, command)
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


def _finish_round(craft, mode: TrackingMode, command) -> None:
    """A scheduled round has landed: a burn whose main stage was cut moves
    to its residual stage; the phase and the carried command update."""
    burn = mode.burn
    if command.phase == "burn" and burn is not None:
        if mode.round_cut:
            if burn.stage == "residual":
                burn.residual_rounds += 1
            burn.stage = "residual"
    else:
        mode.phase = command.phase
    mode.last = command
    mode.last_round_s = mode.round_end_s - mode.round_start_s
    mode.round_command = None


def _close_burn(mode: TrackingMode, t, remaining) -> None:
    burn = mode.burn
    mode.flown.add(burn.impulse.time_s)
    mode.burns.append((burn.impulse.time_s, burn.start_s, t, remaining))
    mode.burn = None
    mode.phase = "trim"


def _decide_round(craft, mode: TrackingMode, gains, actuation, t, round_s,
                  replanner):
    """Read the seam at a round's start, switch, and decide the round:
    burn / trim / coast, and its end -- the next grid point, a burn's
    start, or a burn's predicted cutoff (``mode.round_end_s``)."""
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
    mode.last_eps = eps
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
    length = _grid_end(mode.anchor_s, t, round_s) - t
    actuation.round_s = length
    burn = mode.burn
    if burn is not None and burn.stage == "residual":
        # close once what is left is what the spool-down still delivers
        tail = _outstanding_tail(craft, burn, mass, mode, round_s)
        left = burn.remaining_m_s - tail
        tol = _burn_tolerance(craft.design, gains, mass, round_s)
        if abs(left) <= tol or burn.residual_rounds >= 8:
            _close_burn(mode, t, left)
    if mode.burn is None:
        mode.burn = _arm_burn(mode, craft.design, gains, t, attitude,
                              velocity, mass, round_s, actuation)
    cut = False
    if mode.burn is not None:
        burn = mode.burn
        if t < burn.start_s - _time_tol(t):
            length = min(length, burn.start_s - t)
        command, length, cut = _burn_round(
            burn, craft, gains, mode, t, length, round_s, position,
            velocity, mass, attitude, rate, actuation)
    else:
        e_r = np.linalg.norm(position - r_ref) / np.linalg.norm(r_ref)
        e_v = np.linalg.norm(velocity - v_ref)
        v_band = gains.coast_velocity_deadband * np.linalg.norm(v_ref)
        p_band = gains.coast_position_deadband
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
                carry_n=carry, actuation=actuation)
        else:
            allocation, tau, angle = _coast(craft, gains, attitude,
                                            rate, actuation)
            command = _command(allocation, tau, angle, position - r_ref,
                               velocity - v_ref, np.zeros(3),
                               np.zeros(3), "coast")
    mode.round_start_s, mode.round_end_s = t, t + length
    mode.round_command, mode.round_cut = command, cut
    return command
