"""Orbital craft build step 4: the collocation planner.

    python -m pytest tests/test_orbital_collocation.py -q     (from engine_toy/)

The first two tests need no compile.  The rest compile the slice, arrival
and cost motions once per process (graph-native reverse, explicit seeds).
"""
import math

import numpy as np
import pytest

import orbital_collocation as oc
from orbital_actuation import six_axis_jumper
from orbital_jumper import GravityCenter, OrbitalJumper
from orbital_plan import hohmann_plan
from orbital_plan import reference as hohmann_reference
from orbital_tracker import TrackingGains, TrackingMode, fly

MU_EARTH = 3.986004418e14
R_LEO = 7.0e6
R_HIGH = 8.0e6
MASS = 1000.0
THRUST = 1.0e5
BUDGET = 2.0e6          # N*s: four Hohmann impulses
EARTH = (GravityCenter((0.0, 0.0, 0.0), MU_EARTH),)


def _problem(**overrides):
    return oc.CollocationProblem(six_axis_jumper(THRUST, MASS), EARTH,
                                 R_HIGH, fuel_budget=BUDGET, **overrides)


def _leo_state(t):
    plan = hohmann_plan(MU_EARTH, R_LEO, R_HIGH, t_burn1=t)
    return hohmann_reference(plan, t - 1.0e-9)


def test_arrival_rows_vanish_on_the_target_circle():
    problem = _problem()
    columns = oc._columns(problem)
    speed = math.sqrt(MU_EARTH / R_HIGH)
    angle = 0.7
    point = {**columns, "plane_normal_x": 0.0, "plane_normal_y": 0.0,
             "plane_normal_z": 1.0, "propellant_mass": 0.0,
             "position_x": R_HIGH * math.cos(angle),
             "position_y": R_HIGH * math.sin(angle), "position_z": 0.0,
             "momentum_x": -MASS * speed * math.sin(angle),
             "momentum_y": MASS * speed * math.cos(angle), "momentum_z": 0.0}
    rows = oc.arrival_laws()
    values = [float(row.evalf(subs={s: point[s.name]
                                    for s in row.free_symbols}))
              for row in rows]
    scale = [R_HIGH, MU_EARTH / R_HIGH, speed, R_HIGH, MASS * speed]
    assert np.max(np.abs(np.divide(values, scale))) < 1e-14


def test_warm_start_is_the_hohmann_transfer():
    problem = _problem()
    r0, v0 = _leo_state(0.0)
    hp, tr, z0 = oc.hohmann_warm_start(problem, 0.0, r0, v0)
    u, dt, x = tr.unpack(z0)
    assert dt.size == problem.slices and np.all(dt > 0.0)
    assert float(np.sum(dt)) == pytest.approx(hp.transfer_time + dt[-1],
                                              rel=1e-12)
    assert np.all(u[1:-1] == 0.0)
    B = oc.actuation_matrix(problem.design)
    assert np.linalg.norm(B @ u[0]) * dt[0] / MASS == pytest.approx(
        abs(hp.dv1), rel=1e-6)
    assert np.linalg.norm(B @ u[-1]) * dt[-1] / MASS == pytest.approx(
        abs(hp.dv2), rel=1e-6)
    assert np.linalg.norm(x[-1, 3:6]) == pytest.approx(R_HIGH, rel=1e-12)
    bounds = np.array(tr.bounds())
    assert np.all(bounds[:, 0] <= z0) and np.all(z0 <= bounds[:, 1])


@pytest.fixture(scope="module")
def planned():
    problem = _problem()
    r0, v0 = _leo_state(0.0)
    plan, hp = oc.plan_transfer(problem, 0.0, r0, v0)
    return problem, plan, hp


def test_planner_converges_from_hohmann(planned):
    problem, plan, hp = planned
    tr = plan.transcription
    hohmann_fuel = MASS * hp.ideal_delta_v
    print(f"\n{plan.message}: {plan.iterations} iterations, "
          f"{plan.evaluations} Jacobians in {plan.jacobian_s:.1f} s "
          f"({1e3 * plan.jacobian_s / plan.evaluations:.0f} ms each), solve "
          f"{plan.solve_s:.1f} s; compile slice "
          f"{tr.slice_rows.compile_s:.1f} s, arrival "
          f"{tr.arrival_rows.compile_s:.1f} s, cost "
          f"{tr.cost_rows.compile_s:.1f} s; max scaled defect "
          f"{plan.max_defect:.2e}; impulse {plan.fuel:.5e} N*s = "
          f"{plan.fuel / hohmann_fuel:.4f} x Hohmann; time "
          f"{plan.transfer_time:.1f} s = "
          f"{plan.transfer_time / hp.transfer_time:.4f} x Hohmann; burns "
          f"{len(plan.impulses())}")
    assert plan.converged
    assert plan.max_defect < 1e-8
    r_end, v_end = plan.positions[-1], plan.velocities[-1]
    assert np.linalg.norm(r_end) == pytest.approx(R_HIGH, rel=1e-8)
    assert np.linalg.norm(v_end) == pytest.approx(
        math.sqrt(MU_EARTH / R_HIGH), rel=1e-8)
    assert plan.fuel < BUDGET
    # the plan's reference meets its own nodes
    r_mid, _ = plan.reference(float(plan.times[7]))
    assert r_mid == pytest.approx(plan.positions[7], rel=1e-12)


def test_tracker_flies_the_collocation_plan(planned):
    problem, plan, _ = planned
    r0, v0 = plan.reference(plan.t_start)
    craft = OrbitalJumper(EARTH, design=problem.design, position_m=r0,
                          velocity_m_s=v0, length_scale_m=5.0e4,
                          window_s=10.0)
    mode = TrackingMode(plan)
    report = fly(craft, plan, TrackingGains(), until_s=plan.t_arrive + 600.0,
                 round_s=10.0, mode=mode)
    position, velocity = craft.r()
    radius = np.linalg.norm(position)
    print(f"\nburns {len(mode.burns)}; final |r - r_ref| "
          f"{report.final_position_error_m:.1f} m, |v - v_ref| "
          f"{report.final_velocity_error_m_s:.3f} m/s; |r| - r2 "
          f"{radius - R_HIGH:.1f} m, |v| - v_circ "
          f"{np.linalg.norm(velocity) - math.sqrt(MU_EARTH / R_HIGH):.4f}; "
          f"fuel {report.fuel_impulse_n_s:.4e} N*s = "
          f"{report.fuel_impulse_n_s / plan.fuel:.4f} x plan")
    assert abs(radius - R_HIGH) < 500.0
    assert abs(np.linalg.norm(velocity) - math.sqrt(MU_EARTH / R_HIGH)) < 1.0
    assert report.final_position_error_m < 500.0
    assert report.fuel_impulse_n_s < 1.2 * plan.fuel


# ------------------------------------------------ off plan: the re-planner
def _main_and_rcs(propellant_kg):
    from orbital_actuation import CraftDesign, Thruster
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


def test_kick_replans_with_collocation():
    # the step-5 tracker test's scenario (test_orbital_tracker.
    # test_kick_off_plan_replans_once_and_arrives) with the collocation
    # re-planner in place of hohmann_replanner
    from orbital_tracker import hohmann_replanner
    round_s = 2.0
    gains = TrackingGains(attitude_frequency_rad_s=0.2,
                          off_plan_threshold=0.03, on_plan_threshold=0.005)
    plan = hohmann_plan(MU_EARTH, R_LEO, R_HIGH, t_burn1=300.0)
    design = _main_and_rcs(450.0)
    r0, v0 = hohmann_reference(plan, 0.0)
    craft = OrbitalJumper(EARTH, design=design, position_m=r0,
                          velocity_m_s=v0, length_scale_m=5.0e4,
                          window_s=round_s,
                          angular_velocity_rad_s=(0.0, 0.0, 0.05))
    problem = oc.CollocationProblem(oc.pointing_proxy(design), EARTH, R_HIGH)
    replanner = oc.collocation_replanner(problem, craft=craft)
    seen = {}

    def compared(previous, t, r, v):
        seen["hohmann"] = hohmann_replanner(previous, t, r, v)
        seen["propellant"] = craft.propellant_kg
        return replanner(previous, t, r, v)

    mode = TrackingMode(plan)
    kick_at = plan.t_burn1 + 300.0
    fly(craft, plan, gains, until_s=kick_at, round_s=round_s,
        replanner=compared, mode=mode)
    position, _velocity = craft.r()
    kick = 300.0 * position / np.linalg.norm(position)        # radial, m/s
    craft.F(craft.mass_kg * kick / round_s)
    craft.advance(round_s)
    craft.F((0.0, 0.0, 0.0))
    fly(craft, plan, gains, until_s=kick_at + 600.0, round_s=round_s,
        replanner=compared, mode=mode)
    replanned = mode.plan
    report = fly(craft, plan, gains, until_s=replanned.t_arrive + 1500.0,
                 round_s=round_s, replanner=compared, mode=mode)
    position, velocity = craft.r()
    radius = np.linalg.norm(position)
    used = seen["propellant"] - craft.propellant_kg
    print(f"\nre-plan: {replanned.message}, {replanned.iterations} "
          f"iterations, {replanned.solve_s:.1f} s; delta-v "
          f"{replanned.ideal_delta_v:.1f} m/s (Hohmann-from-present "
          f"{seen['hohmann'].ideal_delta_v:.1f} m/s); trip "
          f"{replanned.transfer_time:.0f} s (Hohmann "
          f"{seen['hohmann'].t_burn2 - seen['hohmann'].t_burn1:.0f} s); "
          f"propellant after the kick {used:.2f} kg; events {mode.events}; "
          f"|r| - r2 {radius - R_HIGH:.1f} m, |v| - v_circ "
          f"{np.linalg.norm(velocity) - math.sqrt(MU_EARTH / R_HIGH):.4f}")
    assert replanned is replanner.plans[0] and replanned.converged
    assert mode.replans == 1
    assert replanned.ideal_delta_v < seen["hohmann"].ideal_delta_v
    assert abs(radius - R_HIGH) < 500.0
    assert abs(np.linalg.norm(velocity) - math.sqrt(MU_EARTH / R_HIGH)) < 1.0
