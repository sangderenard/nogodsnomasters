"""Orbital craft, build step 3b: the live on-plan tracker.

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``
(decision 4, ON PLAN state only; decision 7, deviation belongs in the live
controller's cost).  Self-contained: it reads a :class:`HohmannPlan`
(``orbital_plan``) and a :class:`CraftDesign` (``orbital_actuation``) and
meets the craft only through the seam (decision 8): ``r()``,
``throttle(u)``, ``advance(window)``, and the public ``design``,
``mass_kg``, ``time_s``, ``fuel_impulse_n_s``.  Public surface:

    TrackingGains           frozen: natural frequency, damping, fuel weight
    allocate_throttles(design, force, fuel_weight) -> u
    tracking_command(plan, design, gains, t, r, v, mass) -> TrackingCommand
    fly(craft, plan, gains, until_s=, round_s=) -> FlightReport

Method, once per round (throttles held for the round):

1. Plan error.  ``(r_ref, v_ref) = reference(plan, t)``;
   ``e_r = r - r_ref``, ``e_v = v - v_ref``.
2. Demand (computed-acceleration PD).  The plan's feedforward acceleration
   is the reference's own two-body acceleration; gravity already gives the
   craft the two-body acceleration at its own position, so the thrusters
   owe the difference, plus a critically-damped-by-default PD on the error:

       a_des = [g(r_ref) - g(r)] - w^2 e_r - 2 zeta w e_v,  F_des = m a_des

   (``g`` = ``orbital_plan.two_body_acceleration``, eq_KE2_11..13; the
   plan's impulsive burns enter as steps in ``v_ref``, which the bounded
   allocation turns into a saturated finite burn.)
3. Allocation (the controller's cost).  With ``B = actuation_matrix``
   (the control Jacobian dF/du) and ``T_k`` the thrusters' maximum thrust,

       J(u) = |B u - F_des|^2 / (2 T_max^2) + fuel_weight * sum_k T_k u_k / T_max

   minimised over the declared throttle box ``[u_min, u_max]`` -- a convex
   box-constrained QP, solved by L-BFGS-B (``scipy.optimize``) from the
   clipped least-squares start.  The first term is plan deviation (the
   force the PD asks for and does not get); the second is decision 7's
   fuel rate (``sum_k u_k max_thrust_k``).  A thruster fires only once its
   share of the demand exceeds ``fuel_weight * T_max``, and opposed
   thrusters never fire together.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from orbital_actuation import CraftDesign, actuation_matrix
from orbital_plan import HohmannPlan, reference, two_body_acceleration


@dataclass(frozen=True)
class TrackingGains:
    """PD natural frequency (rad/s) and damping ratio on the plan error,
    and the fuel weight of the allocation cost (dimensionless: the force
    deadband as a fraction of the largest thruster)."""

    natural_frequency_rad_s: float = 0.02
    damping_ratio: float = 1.0
    fuel_weight: float = 1.0e-4

    def __post_init__(self):
        if not (self.natural_frequency_rad_s > 0.0
                and self.damping_ratio > 0.0 and self.fuel_weight >= 0.0):
            raise ValueError(f"invalid tracking gains {self}")


@dataclass(frozen=True)
class TrackingCommand:
    """One round's decision: throttles and what they answer."""

    throttles: np.ndarray
    position_error_m: np.ndarray
    velocity_error_m_s: np.ndarray
    force_demand_n: np.ndarray


@dataclass(frozen=True)
class FlightReport:
    """The flight against the plan, read through the seam."""

    time_s: float
    rounds: int
    final_position_error_m: float
    final_velocity_error_m_s: float
    max_position_error_m: float
    fuel_impulse_n_s: float
    ideal_impulse_n_s: float

    @property
    def fuel_ratio(self) -> float:
        """Fuel used over the impulsive ideal ``mass * (|dv1| + |dv2|)``."""
        return self.fuel_impulse_n_s / self.ideal_impulse_n_s


def allocate_throttles(design: CraftDesign, force_demand_n,
                       fuel_weight: float) -> np.ndarray:
    """argmin over the throttle box of the deviation + fuel cost above."""
    B = actuation_matrix(design)
    if B.shape[1] == 0:
        return np.zeros(0)
    demand = np.asarray(force_demand_n, dtype=float).reshape(3)
    thrust = np.asarray([t.max_thrust_n for t in design.thrusters])
    low = np.asarray([t.throttle_min for t in design.thrusters])
    high = np.asarray([t.throttle_max for t in design.thrusters])
    scale = float(np.max(thrust))

    def cost(u):
        miss = B @ u - demand
        value = 0.5 * (miss @ miss) / scale**2 + fuel_weight * (thrust @ u) / scale
        gradient = B.T @ miss / scale**2 + fuel_weight * thrust / scale
        return value, gradient

    start = np.clip(np.linalg.lstsq(B, demand, rcond=None)[0], low, high)
    result = minimize(cost, start, jac=True, method="L-BFGS-B",
                      bounds=list(zip(low, high)),
                      options={"ftol": 0.0, "gtol": 1.0e-12, "maxiter": 500})
    return np.clip(result.x, low, high)


def tracking_command(plan: HohmannPlan, design: CraftDesign,
                     gains: TrackingGains, t: float, position_m,
                     velocity_m_s, mass_kg: float) -> TrackingCommand:
    """Steps 1-3 of the method at time ``t`` for the craft state given."""
    position = np.asarray(position_m, dtype=float).reshape(3)
    velocity = np.asarray(velocity_m_s, dtype=float).reshape(3)
    r_ref, v_ref = reference(plan, float(t))
    e_r, e_v = position - r_ref, velocity - v_ref
    g_ref, g_craft = two_body_acceleration(plan.mu, np.stack([r_ref,
                                                               position]))
    w, zeta = gains.natural_frequency_rad_s, gains.damping_ratio
    demand = float(mass_kg) * ((g_ref - g_craft) - w**2 * e_r
                               - 2.0 * zeta * w * e_v)
    return TrackingCommand(
        throttles=allocate_throttles(design, demand, gains.fuel_weight),
        position_error_m=e_r, velocity_error_m_s=e_v, force_demand_n=demand)


def fly(craft, plan: HohmannPlan, gains: TrackingGains, *, until_s: float,
        round_s: float) -> FlightReport:
    """Track ``plan`` from the craft's present time to ``until_s``: each
    round read ``r()``, decide, ``throttle(u)``, ``advance(round)``.
    Leaves the throttles at zero."""
    if not round_s > 0.0:
        raise ValueError("round_s must be positive")
    rounds, worst = 0, 0.0
    while craft.time_s < until_s - 1.0e-9 * max(1.0, abs(until_s)):
        position, velocity = craft.r()
        command = tracking_command(plan, craft.design, gains, craft.time_s,
                                   position, velocity, craft.mass_kg)
        worst = max(worst, float(np.linalg.norm(command.position_error_m)))
        craft.throttle(command.throttles)
        craft.advance(min(round_s, until_s - craft.time_s))
        rounds += 1
    craft.throttle(np.zeros(craft.design.thruster_count))
    position, velocity = craft.r()
    r_ref, v_ref = reference(plan, craft.time_s)
    error_r = float(np.linalg.norm(position - r_ref))
    return FlightReport(
        time_s=craft.time_s, rounds=rounds,
        final_position_error_m=error_r,
        final_velocity_error_m_s=float(np.linalg.norm(velocity - v_ref)),
        max_position_error_m=max(worst, error_r),
        fuel_impulse_n_s=craft.fuel_impulse_n_s,
        ideal_impulse_n_s=craft.mass_kg * plan.ideal_delta_v)
