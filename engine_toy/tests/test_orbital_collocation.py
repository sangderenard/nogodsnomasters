"""Orbital craft build step 4: the collocation planner.

    python -m pytest tests/test_orbital_collocation.py -q     (from engine_toy/)

The first two tests need no compile.  The last two need the compiled
graph-native reverse (``orbital_collocation.compile_reverse_rows``) and,
for the flight, a tracker that reads a non-Hohmann plan's reference (see
turing/docs/concordance_census/CONTINUATION_orbital_step4_collocation.md).
"""
import math

import numpy as np
import pytest
import sympy as sp

import orbital_collocation as oc
from orbital_actuation import six_axis_jumper
from orbital_jumper import GravityCenter, OrbitalJumper
from orbital_plan import reference as hohmann_reference
from orbital_tracker import TrackingGains, fly

MU_EARTH = 3.986004418e14
R_LEO = 7.0e6
R_HIGH = 8.0e6
MASS = 1000.0
THRUST = 1.0e5
BUDGET = 2.0e6          # N*s: four Hohmann impulses


def _problem(slices=40):
    return oc.CollocationProblem(
        six_axis_jumper(THRUST, MASS),
        (GravityCenter((0.0, 0.0, 0.0), MU_EARTH),), R_HIGH,
        fuel_budget_n_s=BUDGET, slices=slices)


def _leo_state(t):
    plan = oc.hohmann_plan(MU_EARTH, R_LEO, R_HIGH, t_burn1=t)
    return hohmann_reference(plan, t - 1.0e-9)


def test_arrival_rows_vanish_on_the_target_circle():
    problem = _problem()
    columns = oc._design_columns(problem)
    speed = math.sqrt(MU_EARTH / R_HIGH)
    angle = 0.7
    point = {**columns, "target_radius": R_HIGH, "plane_normal_x": 0.0,
             "plane_normal_y": 0.0, "plane_normal_z": 1.0,
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
    assert float(np.sum(dt)) == pytest.approx(
        hp.transfer_time + dt[-1], rel=1e-12)
    # burns only in the first and last slices, each delivering the
    # Hohmann delta-v through the actuation matrix
    assert np.all(u[1:-1] == 0.0)
    B = oc.actuation_matrix(problem.design)
    assert np.linalg.norm(B @ u[0]) * dt[0] / MASS == pytest.approx(
        abs(hp.dv1), rel=1e-6)
    assert np.linalg.norm(B @ u[-1]) * dt[-1] / MASS == pytest.approx(
        abs(hp.dv2), rel=1e-6)
    assert np.linalg.norm(x[-1, :3]) == pytest.approx(R_HIGH, rel=1e-12)
    assert np.all(np.array(tr.bounds())[:, 0] <= z0)
    assert np.all(z0 <= np.array(tr.bounds())[:, 1])


@pytest.fixture(scope="module")
def planned():
    problem = _problem()
    r0, v0 = _leo_state(0.0)
    plan, hp = oc.plan_transfer(problem, 0.0, r0, v0)
    return problem, plan, hp


def test_planner_converges_from_hohmann(planned):
    problem, plan, hp = planned
    hohmann_impulse = MASS * hp.ideal_delta_v
    print(f"\n{plan.message}; max scaled defect {plan.max_defect:.2e}; "
          f"impulse {plan.impulse_n_s:.5e} N*s = "
          f"{plan.impulse_n_s / hohmann_impulse:.4f} x Hohmann; "
          f"time {plan.transfer_time:.1f} s = "
          f"{plan.transfer_time / hp.transfer_time:.4f} x Hohmann")
    assert plan.converged
    assert plan.max_defect < 1e-8
    r_end, v_end = plan.positions[-1], plan.velocities[-1]
    assert np.linalg.norm(r_end) == pytest.approx(R_HIGH, rel=1e-8)
    assert np.linalg.norm(v_end) == pytest.approx(
        math.sqrt(MU_EARTH / R_HIGH), rel=1e-8)
    assert plan.impulse_n_s < BUDGET


def test_tracker_flies_the_collocation_plan(planned):
    problem, plan, _ = planned
    r0, v0 = oc.reference(plan, plan.t_start)
    craft = OrbitalJumper(problem.centers, design=problem.design,
                          position_m=r0, velocity_m_s=v0,
                          length_scale_m=5.0e4, window_s=10.0)
    report = fly(craft, plan, TrackingGains(),
                 until_s=plan.t_arrive + 600.0, round_s=10.0)
    position, velocity = craft.r()
    print(f"\nfinal |r - r_ref| {report.final_position_error_m:.1f} m, "
          f"|r| - r2 {np.linalg.norm(position) - R_HIGH:.1f} m; fuel "
          f"{report.fuel_impulse_n_s:.4e} N*s = {report.fuel_ratio:.4f} x plan")
    assert abs(np.linalg.norm(position) - R_HIGH) < 500.0
    assert abs(np.linalg.norm(velocity) - math.sqrt(MU_EARTH / R_HIGH)) < 0.5
    assert report.final_position_error_m < 500.0
