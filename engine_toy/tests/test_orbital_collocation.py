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
          f"{len(plan.impulses())}; structure {plan.structure}; stages "
          f"{plan.stages}")
    assert plan.converged
    assert plan.max_defect < 1e-8
    assert plan.solve_s < 5.0
    assert plan.fuel == pytest.approx(hohmann_fuel, rel=1e-3)
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
def _main_and_rcs_plan():
    """The kick scenario's craft on a collocation plan from LEO at t = 0."""
    problem = oc.CollocationProblem(oc.pointing_proxy(_main_and_rcs(450.0)),
                                    EARTH, R_HIGH)
    r0, v0 = _leo_state(0.0)
    plan, _hp = oc.plan_transfer(problem, 0.0, r0, v0, propellant_kg=450.0)
    return problem, plan


@pytest.mark.parametrize("kick_at", [302.0, 600.0, 1500.0])
def test_kick_replan_from_the_remainder_is_live(kick_at):
    # a 300 m/s radial kick off the plan's own reference; the re-plan is
    # warm-started from the plan's remainder and must be live (< 5 s),
    # reach the same plan as a cold start from Hohmann, and beat the
    # Hohmann transfer from the kicked state
    from orbital_tracker import hohmann_replanner
    problem, plan = _main_and_rcs_plan()
    r, v = plan.reference(kick_at)
    v = v + 300.0 * r / np.linalg.norm(r)
    k = int(np.searchsorted(plan.times, kick_at)) - 1
    propellant = float(plan.propellant_kg[k + 1])
    now = oc.dataclasses.replace(problem, fuel_budget=propellant)
    warm, _ = oc.plan_transfer(now, kick_at, r, v, propellant_kg=propellant,
                               previous=plan)
    cold, _ = oc.plan_transfer(now, kick_at, r, v, propellant_kg=propellant)
    hohmann = hohmann_replanner(plan, kick_at, r, v)
    _r, after = hohmann_reference(hohmann, kick_at)
    hohmann_dv = float(np.linalg.norm(after - v) + abs(hohmann.dv2))
    print(f"\nkick at {kick_at:.0f} s: remainder {warm.message}, "
          f"{warm.iterations} it, {warm.evaluations} Jacobians, "
          f"{warm.solve_s:.2f} s, dv {warm.ideal_delta_v:.1f} m/s, fuel "
          f"{warm.fuel:.2f} kg, trip {warm.transfer_time:.0f} s, structure "
          f"{warm.structure}; cold {cold.solve_s:.2f} s, dv "
          f"{cold.ideal_delta_v:.1f}, cost {cold.cost:.8f} vs "
          f"{warm.cost:.8f}; Hohmann from present {hohmann_dv:.1f} m/s")
    assert warm.converged and warm.max_defect < 1e-8
    assert warm.solve_s < 5.0
    assert warm.cost <= cold.cost * (1.0 + 1e-6)
    assert warm.ideal_delta_v < hohmann_dv


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
    # (the craft now flies a COLLOCATION plan, so the re-plan is warm-
    # started from that plan's remainder -- decision 4)
    round_s = 2.0
    gains = TrackingGains(attitude_frequency_rad_s=0.2,
                          off_plan_threshold=0.03, on_plan_threshold=0.005)
    design = _main_and_rcs(450.0)
    problem, plan = _main_and_rcs_plan()
    r0, v0 = plan.reference(0.0)
    craft = OrbitalJumper(EARTH, design=design, position_m=r0,
                          velocity_m_s=v0, length_scale_m=5.0e4,
                          window_s=round_s,
                          angular_velocity_rad_s=(0.0, 0.0, 0.05))
    replanner = oc.collocation_replanner(problem, craft=craft)
    seen = {}

    def compared(previous, t, r, v):
        # what the Hohmann re-planner asks from here: its burn 1 from the
        # present velocity onto its transfer, then its burn 2
        hohmann = hohmann_replanner(previous, t, r, v)
        _r, after = hohmann_reference(hohmann, t)
        if t >= kick_at and "hohmann_dv" not in seen:   # the kick's re-plan
            seen["hohmann_dv"] = float(
                np.linalg.norm(after - np.asarray(v)) + abs(hohmann.dv2))
            seen["propellant"] = craft.propellant_kg
            seen["index"] = len(replanner.plans)
        return replanner(previous, t, r, v)

    mode = TrackingMode(plan)
    kick_at = 600.0
    fly(craft, plan, gains, until_s=kick_at, round_s=round_s,
        replanner=compared, mode=mode)
    position, _velocity = craft.r()
    kick = 300.0 * position / np.linalg.norm(position)        # radial, m/s
    craft.F(craft.mass_kg * kick / round_s)
    craft.advance(round_s)
    craft.F((0.0, 0.0, 0.0))
    fly(craft, plan, gains, until_s=kick_at + 600.0, round_s=round_s,
        replanner=compared, mode=mode)
    replanned = replanner.plans[seen["index"]]
    history = []
    while True:                  # fly to the arrival of the plan flown now
        flown = mode.plan
        report = fly(craft, plan, gains, until_s=flown.t_arrive + 1500.0,
                     round_s=round_s, replanner=compared, mode=mode,
                     record=history)
        if mode.plan is flown:
            break
    position, velocity = craft.r()
    radius = np.linalg.norm(position)
    used = seen["propellant"] - craft.propellant_kg
    burns = [(round(i.time_s), round(float(np.linalg.norm(i.delta_v_m_s)), 1))
             for i in replanned.impulses()
             if np.linalg.norm(i.delta_v_m_s) > 1.0]
    print(f"\nre-plan: {replanned.message}, {replanned.iterations} "
          f"iterations + {replanned.polish_evaluations} polish, "
          f"{replanned.solve_s:.1f} s, defect {replanned.max_defect:.1e}; "
          f"delta-v {replanned.ideal_delta_v:.1f} m/s (Hohmann-from-present "
          f"{seen['hohmann_dv']:.1f} m/s); trip "
          f"{replanned.transfer_time:.0f} s; burns {burns}; flown burns "
          f"{[tuple(round(v, 1) for v in b) for b in mode.burns]}; re-plans "
          f"{mode.replans}, events {mode.events}; propellant after the "
          f"kick {used:.2f} kg; final |r - r_ref| "
          f"{report.final_position_error_m:.1f} m; |r| - r2 "
          f"{radius - R_HIGH:.1f} m, |v| - v_circ "
          f"{np.linalg.norm(velocity) - math.sqrt(MU_EARTH / R_HIGH):.4f}")
    final = max(replanned.impulses(), key=lambda i: i.time_s)
    around = [(t, c.phase, round(c.attitude_error_rad, 3),
               round(float(np.linalg.norm(c.achieved_force_n))))
              for t, c in history if abs(t - final.time_s) < 60.0]
    print(f"  around the last planned burn ({final.time_s:.1f} s): {around}")
    for p in replanner.plans:
        r_end = np.linalg.norm(p.positions[-1])
        print(f"  plan from t={p.t_start:.0f}: {p.message}, it "
              f"{p.iterations}, {p.solve_s:.2f} s, defect "
              f"{p.max_defect:.1e}, dv {p.ideal_delta_v:.1f}, arrive "
              f"{p.t_arrive:.0f} s at |r| - r2 {r_end - R_HIGH:.3f} m")
    assert mode.replans >= 1 and replanned.max_defect < 1e-8
    assert replanned.message.startswith("remainder")
    assert replanned.solve_s < 5.0
    assert replanned.ideal_delta_v < seen["hohmann_dv"]
    assert abs(radius - R_HIGH) < 500.0
    assert abs(np.linalg.norm(velocity) - math.sqrt(MU_EARTH / R_HIGH)) < 1.0


# ------------------------------------------------ the machine craft (step 8)
def test_machine_craft_flies_the_collocation_plan():
    from orbital_craft_machine import MachineCraft, orbital_craft
    round_s = 2.0
    gains = TrackingGains(attitude_frequency_rad_s=0.2)
    r0, v0 = _leo_state(0.0)
    craft = MachineCraft(EARTH, orbital_craft(), position_m=r0,
                         velocity_m_s=v0, length_scale_m=5.0e4,
                         window_s=round_s)
    problem = oc.CollocationProblem(oc.pointing_proxy(craft.design), EARTH,
                                    R_HIGH)
    # the machine's propellant is the sum of its tank charges
    propellant = craft.propellant_kg
    assert propellant == pytest.approx(
        sum(craft.tank_propellant_kg().values()), rel=1e-15)
    plan, hp = oc.plan_transfer(problem, craft.time_s, *craft.r(),
                                propellant_kg=propellant)
    tanks0, mass0 = craft.tank_propellant_kg(), craft.mass_kg
    mode = TrackingMode(plan)
    report = fly(craft, plan, gains, until_s=plan.t_arrive + 1500.0,
                 round_s=round_s, mode=mode)
    tanks = craft.tank_propellant_kg()
    used = sum(tanks0[k] - tanks[k] for k in tanks0)
    position, velocity = craft.r()
    radius = np.linalg.norm(position)
    print(f"\nmachine craft: plan {plan.message}, {plan.iterations} it, "
          f"defect {plan.max_defect:.1e}, dv {plan.ideal_delta_v:.1f} m/s "
          f"(Hohmann {hp.ideal_delta_v:.1f}), planned propellant "
          f"{plan.propellant_kg[0] - plan.propellant_kg[-1]:.2f} kg; flown "
          f"burns {[tuple(round(v, 1) for v in b) for b in mode.burns]}; "
          f"propellant used {used:.2f} kg (all tanks); re-plans "
          f"{report.replans}; final |r - r_ref| "
          f"{report.final_position_error_m:.1f} m; |r| - r2 "
          f"{radius - R_HIGH:.1f} m, |v| - v_circ "
          f"{np.linalg.norm(velocity) - math.sqrt(MU_EARTH / R_HIGH):.4f}")
    assert plan.max_defect < 1e-8 and plan.converged
    assert plan.solve_s < 5.0
    assert craft.propellant_kg == pytest.approx(propellant - used, rel=1e-12)
    assert abs(radius - R_HIGH) < 500.0
    assert abs(np.linalg.norm(velocity) - math.sqrt(MU_EARTH / R_HIGH)) < 1.0
