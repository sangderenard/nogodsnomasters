"""Orbital craft build step 3b: the live on-plan tracker.

    python -m pytest tests/test_orbital_tracker.py -q     (from engine_toy/)
"""
import math

import numpy as np
import pytest

from orbital_actuation import actuation_matrix, six_axis_jumper
from orbital_jumper import GravityCenter, OrbitalJumper
from orbital_plan import hohmann_plan, reference
from orbital_tracker import (
    TrackingGains,
    allocate_throttles,
    fly,
    tracking_command,
)

MU_EARTH = 3.986004418e14
R_LEO = 7.0e6
R_GEO = 42164.0e3
MASS = 1000.0
THRUST = 1.0e5          # per thruster: 10 g, near-impulsive burns (~23 s)


def test_allocation_fires_only_the_aligned_thruster():
    design = six_axis_jumper(THRUST, MASS)
    gains = TrackingGains()
    u = allocate_throttles(design, (0.0, -0.25 * THRUST, 0.0),
                           gains.fuel_weight)
    expected = np.zeros(6)
    expected[3] = 0.25 - gains.fuel_weight     # -y; deadband = weight * T
    assert u == pytest.approx(expected, abs=1e-9)
    # saturates at the box, never fires opposed pairs
    u = allocate_throttles(design, (3.0 * THRUST, 0.5 * THRUST, 0.0), 0.0)
    assert u == pytest.approx([1.0, 0.0, 0.5, 0.0, 0.0, 0.0], abs=1e-9)
    assert np.all(allocate_throttles(design, np.zeros(3), 1e-4) == 0.0)


def test_on_reference_state_commands_nothing():
    design = six_axis_jumper(THRUST, MASS)
    plan = hohmann_plan(MU_EARTH, R_LEO, R_GEO, t_burn1=300.0)
    r, v = reference(plan, 5000.0)
    command = tracking_command(plan, design, TrackingGains(), 5000.0, r, v,
                               MASS)
    assert np.all(command.throttles == 0.0)
    assert np.linalg.norm(command.force_demand_n) < 1e-9


def test_tracker_flies_leo_to_geo():
    plan = hohmann_plan(MU_EARTH, R_LEO, R_GEO, t_burn1=300.0)
    design = six_axis_jumper(THRUST, MASS)
    r0, v0 = reference(plan, 0.0)
    craft = OrbitalJumper([GravityCenter((0.0, 0.0, 0.0), MU_EARTH)],
                          design=design, position_m=r0, velocity_m_s=v0,
                          length_scale_m=5.0e4, window_s=10.0)
    report = fly(craft, plan, TrackingGains(),
                 until_s=plan.t_burn2 + 3000.0, round_s=10.0)
    position, velocity = craft.r()
    radius = np.linalg.norm(position)
    radial_speed = position @ velocity / radius
    circular = math.sqrt(MU_EARTH / R_GEO)
    print(f"\nrounds {report.rounds}, substeps {craft.substeps}; final "
          f"|r - r_ref| {report.final_position_error_m:.1f} m, "
          f"|v - v_ref| {report.final_velocity_error_m_s:.3f} m/s; "
          f"|r| - r2 {radius - R_GEO:.1f} m, |v| - v_circ "
          f"{np.linalg.norm(velocity) - circular:.4f} m/s, v_r "
          f"{radial_speed:.3f} m/s; fuel {report.fuel_impulse_n_s:.4e} N*s "
          f"= {report.fuel_ratio:.4f} x ideal {report.ideal_impulse_n_s:.4e}")
    # on the target circle (step 5: trims leave velocity errors inside the
    # coast band |g| h alone -- the size of r()'s own reading stagger,
    # 0.224 m/s^2 * 10 s = 2.2 m/s at GEO; measured 1.04 m/s)
    assert abs(radius - R_GEO) < 200.0
    assert abs(np.linalg.norm(velocity) - circular) < (
        MU_EARTH / R_GEO**2 * 10.0)
    assert abs(radial_speed) < 1.0
    # on the plan
    assert report.final_position_error_m < 200.0
    assert report.final_velocity_error_m_s < MU_EARTH / R_GEO**2 * 10.0
    # fuel against the impulsive ideal (a finite-thrust six-axis craft
    # cannot beat it); step 3's PD-only tracker measured 1.2416
    assert 1.0 <= report.fuel_ratio < 1.2
    assert report.ideal_impulse_n_s == pytest.approx(
        MASS * (abs(plan.dv1) + abs(plan.dv2)), rel=1e-15)


# ------------------------------------------------- step 5 + step-7 craft
# The tracker flies a craft that must turn its main engine onto the force
# it needs (decision 9), within its propellant (decision 7), and re-plans
# the whole remaining trip once when pushed off plan (decision 4).
from orbital_actuation import CraftDesign, Thruster
from orbital_tracker import (
    Allocation,
    TrackingMode,
    Wrench,
    hohmann_replanner,
    least_squares_allocation,
    plan_impulses,
    plan_reference,
)

R_HIGH = 8.0e6
ROUND = 2.0
STEER = TrackingGains(attitude_frequency_rad_s=0.2)


def _main_and_rcs(propellant_kg=300.0):
    """A 1 g bipropellant main engine along craft +x through the centre of
    mass, and two monopropellant RCS couples about z (+z and -z)."""
    rcs = 50.0
    return CraftDesign((
        Thruster("main", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0e4,
                 kind="bipropellant"),
        Thruster("rcs+z fore", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), rcs,
                 kind="monopropellant"),
        Thruster("rcs+z aft", (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), rcs,
                 kind="monopropellant"),
        Thruster("rcs-z fore", (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), rcs,
                 kind="monopropellant"),
        Thruster("rcs-z aft", (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), rcs,
                 kind="monopropellant"),
    ), mass_kg=MASS, propellant_kg=propellant_kg, body_size_m=(2.0, 1.0, 1.0),
        identity="main + RCS test craft")


def _spinning_craft(plan, propellant_kg=300.0):
    # main engine radial (90 degrees off the prograde burn), spinning
    r0, v0 = reference(plan, 0.0)
    return OrbitalJumper([GravityCenter((0.0, 0.0, 0.0), MU_EARTH)],
                         design=_main_and_rcs(propellant_kg), position_m=r0,
                         velocity_m_s=v0, length_scale_m=5.0e4,
                         window_s=ROUND,
                         angular_velocity_rad_s=(0.0, 0.0, 0.05))


def _on_circle(craft, radius):
    position, velocity = craft.r()
    r = np.linalg.norm(position)
    return (r - radius, np.linalg.norm(velocity) - math.sqrt(MU_EARTH / radius),
            position @ velocity / r)


def _arrived(craft, report):
    """On the target circle and on the plan, up to the seam's stagger.

    ``r()`` reads symplectic Euler's state: the velocity is half a substep
    ahead of the position, ``|v_read - v(t)| ~ g dt / 2`` (3.3 m/s at
    8000 km with the ~1 s substeps the attitude bound leaves), and the PD
    settles where ``w^2 e_r = 2 w e_v``: ``|e_r| = 2 (g dt / 2) / w`` (330 m
    at w = 0.02).  The bounds are that stagger's, not the tracker's."""
    dr, dv, vr = _on_circle(craft, R_HIGH)
    assert abs(dr) < 500.0 and abs(dv) < 1.0 and abs(vr) < 5.0
    assert report.final_position_error_m < 500.0
    assert report.final_velocity_error_m_s < 5.0


def _propellant_ideal(design, plan):
    """TS2.1: the bipropellant main engine's propellant for the plan's
    impulsive delta-v (the rocket-equation ideal)."""
    c = design.thrusters[0].thruster_kind.exhaust_velocity_m_s
    return design.mass_kg * (1.0 - math.exp(-plan.ideal_delta_v / c))


def test_rotating_craft_slews_its_main_engine_and_arrives():
    plan = hohmann_plan(MU_EARTH, R_LEO, R_HIGH, t_burn1=300.0)
    craft = _spinning_craft(plan)
    mode, history = TrackingMode(plan), []
    report = fly(craft, plan, STEER, until_s=plan.t_burn2 + 1500.0,
                 round_s=ROUND, mode=mode, record=history)
    burn1 = [(t, c) for t, c in history
             if c.phase == "burn" and t < plan.t_burn1 + 100.0]
    firing = [(t, c) for t, c in burn1 if c.throttles[0] > 0.5]
    used = craft.design.propellant_kg - craft.propellant_kg
    ideal = _propellant_ideal(craft.design, plan)
    dr, dv, vr = _on_circle(craft, R_HIGH)
    print(f"\nburn 1 armed at t_burn1 {burn1[0][0] - plan.t_burn1:+.1f} s "
          f"with the engine {burn1[0][1].attitude_error_rad:.3f} rad off; "
          f"fires t_burn1 {firing[0][0] - plan.t_burn1:+.1f} .. "
          f"{firing[-1][0] - plan.t_burn1:+.1f} s, worst pointing while "
          f"firing {max(c.attitude_error_rad for _t, c in firing):.4f} rad; "
          f"burns {mode.burns}; propellant {used:.2f} kg = "
          f"{used / ideal:.4f} x TS2.1 ideal {ideal:.2f} kg; rounds "
          f"{report.rounds}, substeps {craft.substeps}; final |r - r_ref| "
          f"{report.final_position_error_m:.1f} m, |v - v_ref| "
          f"{report.final_velocity_error_m_s:.4f} m/s; |r| - r2 {dr:.1f} m, "
          f"|v| - v_circ {dv:.4f} m/s, v_r {vr:.4f}; max eps "
          f"{report.max_tracking_error:.4f}")
    # armed with the engine a quarter turn off (and spinning); the slew
    # carries it onto the burn before it fires, and it fires centred
    assert burn1[0][1].attitude_error_rad > 0.25
    assert (max(c.attitude_error_rad for _t, c in firing)
            <= STEER.burn_alignment_rad)
    assert firing[0][0] < plan.t_burn1 < firing[-1][0]
    assert [b[0] for b in mode.burns] == [plan.t_burn1, plan.t_burn2]
    assert used / ideal < 1.15
    _arrived(craft, report)
    assert report.replans == 0 and report.on_plan
    assert report.throttles.shape == (5,)
    assert len(report.throttle_history) == report.rounds


def test_kick_off_plan_replans_once_and_arrives():
    # the band sits above the nominal flight's worst eps (0.0188 on the
    # plan, 0.0250 on the re-plan's burn 2: a centred finite burn against
    # the impulsive reference) and below the kick's
    gains = TrackingGains(attitude_frequency_rad_s=0.2,
                          off_plan_threshold=0.03, on_plan_threshold=0.005)
    plan = hohmann_plan(MU_EARTH, R_LEO, R_HIGH, t_burn1=300.0)
    craft = _spinning_craft(plan, propellant_kg=450.0)
    mode = TrackingMode(plan)
    # on the transfer leg, 300 s after burn 1 (Hohmann-from-present, the
    # stand-in re-planner, departs horizontally: far from periapsis it
    # throws the transfer ellipse away -- see the step-5 continuation)
    kick_at = plan.t_burn1 + 300.0
    nominal = fly(craft, plan, gains, until_s=kick_at, round_s=ROUND,
                  replanner=hohmann_replanner, mode=mode)
    assert mode.replans == 0 and mode.on_plan    # nominal burns stay ON
    position, _velocity = craft.r()
    kick = 300.0 * position / np.linalg.norm(position)    # radial, m/s
    craft.F(craft.mass_kg * kick / ROUND)
    craft.advance(ROUND)
    craft.F((0.0, 0.0, 0.0))
    first = fly(craft, plan, gains, until_s=kick_at + 600.0, round_s=ROUND,
                replanner=hohmann_replanner, mode=mode)
    replanned = mode.plan
    report = fly(craft, plan, gains, until_s=replanned.t_burn2 + 1500.0,
                 round_s=ROUND, replanner=hohmann_replanner, mode=mode)
    dr, dv, vr = _on_circle(craft, R_HIGH)
    print(f"\nnominal max eps {nominal.max_tracking_error:.4f}; events "
          f"{mode.events}; re-plan r1 {replanned.r1:.1f} m, t_burn1 "
          f"{replanned.t_burn1:.1f} s, t_burn2 {replanned.t_burn2:.1f} s; "
          f"burns {mode.burns}; after-kick max eps "
          f"{first.max_tracking_error:.4f}; final |r - r_ref| "
          f"{report.final_position_error_m:.1f} m, |v - v_ref| "
          f"{report.final_velocity_error_m_s:.4f} m/s; |r| - r2 {dr:.1f} m, "
          f"|v| - v_circ {dv:.4f}, v_r {vr:.4f}; propellant left "
          f"{craft.propellant_kg:.2f} kg")
    assert nominal.max_tracking_error < gains.off_plan_threshold
    assert mode.replans == 1 and report.replans == 1
    assert [kind for _t, kind, _eps in mode.events] == ["off", "on"]
    assert report.on_plan and report.plan is replanned
    assert replanned.t_burn1 == pytest.approx(kick_at + ROUND)
    assert replanned.r2 == plan.r2               # the same final orbit
    _arrived(craft, report)


# ------------------------------------------------- the seam, no compile
class _HalfCraft:
    """An allocation seam that achieves half of any requested force."""

    def __init__(self, design):
        self.design = design

    def allocate(self, wrench):
        return Allocation(Wrench(0.5 * np.asarray(wrench.force_n),
                                 np.zeros(3)),
                          np.zeros(self.design.thruster_count))


def test_the_seam_answers_and_the_shortfall_is_carried():
    design = six_axis_jumper(THRUST, MASS)
    plan = hohmann_plan(MU_EARTH, R_LEO, R_GEO, t_burn1=300.0)
    r, v = reference(plan, 5000.0)
    v = v + np.asarray((0.0, 1.0, 0.0))
    seam = _HalfCraft(design)
    first = tracking_command(plan, design, TrackingGains(), 5000.0, r, v,
                             MASS, allocate=seam.allocate)
    assert first.achieved_force_n == pytest.approx(0.5 * first.pd_force_n)
    assert first.force_shortfall_n == pytest.approx(0.5 * first.pd_force_n)
    second = tracking_command(plan, design, TrackingGains(), 5000.0, r, v,
                              MASS, carry_n=first.force_shortfall_n,
                              allocate=seam.allocate)
    assert second.force_demand_n == pytest.approx(1.5 * second.pd_force_n)
    # the carry never exceeds the PD force (no windup)
    third = tracking_command(plan, design, TrackingGains(), 5000.0, r, v,
                             MASS, carry_n=100.0 * first.pd_force_n,
                             allocate=seam.allocate)
    assert third.force_demand_n == pytest.approx(2.0 * third.pd_force_n)
    # the stand-in answers like the seam: achieved = R B clamp(u)
    allocation = least_squares_allocation(
        design, None, 0.0,
        Wrench(np.asarray((0.0, 5.0e4, 0.0)), np.zeros(3)))
    assert allocation.achieved.force_n == pytest.approx((0.0, 5.0e4, 0.0))


def test_plans_are_read_through_their_own_reference():
    plan = hohmann_plan(MU_EARTH, R_LEO, R_GEO, t_burn1=300.0)
    assert np.array_equal(plan.reference(1234.0)[0],
                          reference(plan, 1234.0)[0])

    class Straight:
        mu = MU_EARTH

        def reference(self, t):
            return (np.asarray((R_LEO, 10.0 * t, 0.0)),
                    np.asarray((0.0, 10.0, 0.0)))

    r, v = plan_reference(Straight(), 3.0)
    assert r == pytest.approx((R_LEO, 30.0, 0.0)) and v[1] == 10.0
    assert plan_impulses(Straight()) == ()
    impulses = plan_impulses(plan)
    assert [i.time_s for i in impulses] == [plan.t_burn1, plan.t_burn2]
    assert np.linalg.norm(impulses[0].delta_v_m_s) == pytest.approx(
        abs(plan.dv1), rel=1e-9)


# ------------------------------------------- the attitude bounds the step
def test_attitude_dt_limit_keeps_a_spin_per_step_small():
    # step 7's open case: CFL substeps of ~22 s in LEO with a 0.1 rad/s
    # spin turned 2 atan(1.1) = 1.67 rad per step where the spin turns 2.2
    speed = math.sqrt(MU_EARTH / R_LEO)
    turned = {}
    for bound in (math.inf, None):
        extra = {} if bound is None else {"attitude_step_rad": bound}
        craft = OrbitalJumper([GravityCenter((0.0, 0.0, 0.0), MU_EARTH)],
                              design=six_axis_jumper(100.0, MASS),
                              position_m=(R_LEO, 0.0, 0.0),
                              velocity_m_s=(0.0, speed, 0.0),
                              length_scale_m=22.0 * speed / 0.5,
                              window_s=22.0,
                              angular_velocity_rad_s=(0.0, 0.0, 0.1),
                              **extra)
        before = craft.attitude()
        craft.advance()
        step = before.T @ craft.attitude()
        turned[bound] = (math.atan2(step[1, 0], step[0, 0]), craft.substeps)
    print(f"\nper 22 s window: unbounded {turned[math.inf]}, bounded "
          f"{turned[None]} (omega t = 2.2 rad)")
    assert turned[math.inf][0] == pytest.approx(2.0 * math.atan(1.1),
                                                abs=1e-3)
    # the published dt_limit = 0.125 rad / |omega| = 1.25 s: 18 substeps,
    # each turning 2 atan(dt w / 2) -> 2.1972 rad, 0.13 % short of 2.2
    assert turned[None][1] == 18
    assert turned[None][0] == pytest.approx(2.2, rel=2e-3)
