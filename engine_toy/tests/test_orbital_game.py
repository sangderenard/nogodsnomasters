"""Orbital craft build step 6: the live game's non-graphical logic.

    python -m pytest tests/test_orbital_game.py -q -s    (from engine_toy/)
"""
import math

import numpy as np
import pytest

from orbital_game import (
    MU_EARTH,
    OrbitalGame,
    TargetSpec,
    _collocation_planner,
    _plan_sweep,
    _plan_times,
    craft_drawing,
    phasing,
)
from orbital_plan import hohmann_plan

R1 = 7.0e6


def _mean_motion(radius):
    return math.sqrt(MU_EARTH / radius**3)


def _wrapped(angle):
    return math.remainder(angle, 2.0 * math.pi)


@pytest.mark.parametrize("r2, phi_0, theta_0", [
    (9.0e6, 2.2, 0.0),          # raising, target ahead
    (2.0e7, -2.9, 1.1),         # raising, target behind
    (6.8e6, 0.4, -0.7),         # lowering: the target gains on the craft
    (4.2164e7, 3.0, 2.0),
])
def test_phasing_puts_the_target_at_the_arrival_point(r2, phi_0, theta_0):
    plan = hohmann_plan(MU_EARTH, R1, r2)
    timing = phasing(MU_EARTH, R1, r2, plan.transfer_time, math.pi,
                     phi_0, theta_0)
    n1, n2 = _mean_motion(R1), _mean_motion(r2)
    synodic = 2.0 * math.pi / abs(n1 - n2)
    assert 0.0 <= timing.t_wait < synodic
    # the craft is at ``phase`` at burn 1 and arrives at phase + pi
    assert _wrapped(timing.phase - (theta_0 + n1 * timing.t_wait)) \
        == pytest.approx(0.0, abs=1e-12)
    target_at_arrival = theta_0 + phi_0 + n2 * (timing.t_wait
                                                + plan.transfer_time)
    assert _wrapped(target_at_arrival - (timing.phase + math.pi)) \
        == pytest.approx(0.0, abs=1e-9)


def test_phasing_refuses_a_co_orbital_target():
    with pytest.raises(ValueError, match="co-orbital"):
        phasing(MU_EARTH, R1, R1, 1000.0, math.pi, 0.5, 0.0)


def test_collocation_adapter_uses_problem_transfer_api(monkeypatch):
    import orbital_collocation

    expected = object()
    seen = {}

    def plan_transfer(problem, t0, position, velocity, **kwargs):
        seen.update(problem=problem, t0=t0, position=position,
                    velocity=velocity, kwargs=kwargs)
        return expected, object()

    monkeypatch.setattr(orbital_collocation, "plan_transfer", plan_transfer)
    problem, position, velocity, previous = object(), (1, 2, 3), (4, 5, 6), object()
    plan = _collocation_planner()(problem, 12.0, position, velocity,
                                  propellant_kg=34.0, previous=previous)
    assert plan is expected
    assert seen == {
        "problem": problem, "t0": 12.0, "position": position,
        "velocity": velocity,
        "kwargs": {"propellant_kg": 34.0, "previous": previous}}


def test_collocation_plan_sweep_uses_its_nodes_and_impulse_times():
    from types import SimpleNamespace

    angles = np.array((2.9, -2.9, -2.1))
    plan = SimpleNamespace(
        t_start=10.0, t_arrive=30.0,
        positions=np.column_stack((np.cos(angles), np.sin(angles),
                                   np.zeros(angles.size))),
        impulses=lambda: (SimpleNamespace(time_s=12.0),
                          SimpleNamespace(time_s=28.0)))
    assert _plan_times(plan) == (10.0, 30.0, 12.0, 28.0)
    assert _plan_sweep(plan) == pytest.approx(1.2831853071795862)


def test_collocation_order_phases_from_probe_and_replans_with_problem(
        monkeypatch):
    from types import SimpleNamespace

    import orbital_collocation
    import orbital_game

    angles = np.array((0.0, 0.4, 1.1))

    def plan(t0):
        return SimpleNamespace(
            t_start=t0, t_arrive=t0 + 100.0, transfer_time=100.0,
            positions=np.column_stack((R1 * np.cos(angles),
                                       R1 * np.sin(angles),
                                       np.zeros(angles.size))),
            impulses=lambda: (SimpleNamespace(time_s=t0 + 10.0),
                              SimpleNamespace(time_s=t0 + 90.0)))

    planned = [plan(10.0), plan(30.0)]
    calls = []
    monkeypatch.setattr(orbital_collocation, "CollocationProblem",
                        lambda design, centers, radius, **kwargs:
                        (design, centers, radius, kwargs))
    monkeypatch.setattr(orbital_collocation, "pointing_proxy",
                        lambda design: ("pointing-proxy", design))
    monkeypatch.setattr(orbital_collocation, "plan_transfer",
                        lambda problem, t0, position, velocity, **kwargs:
                        (calls.append((problem, t0, position, velocity,
                                       kwargs)) or planned[len(calls) - 1],
                         object()))
    timing = SimpleNamespace(t_wait=20.0, phase=0.2, lead_angle=0.0)
    phasing_calls = []

    def fake_phasing(mu, r1, r2, transfer_time, sweep, phi_0, theta_0):
        phasing_calls.append((transfer_time, sweep))
        return timing

    monkeypatch.setattr(orbital_game, "phasing", fake_phasing)

    class Craft:
        time_s = 10.0
        propellant_kg = 12.0
        design = object()

        def r(self):
            return np.array((R1, 0.0, 0.0)), np.array((0.0, 7500.0, 0.0))

    game = object.__new__(OrbitalGame)
    game.mu = MU_EARTH
    game.planner_name = "collocation"
    game.planner = _collocation_planner()
    game.craft = Craft()
    game.center = object()
    game.specs = (TargetSpec("probe", 9.0e6, 0.5),)
    game.order = None
    game.stations = SimpleNamespace(time_s=10.0)
    game.target_state = lambda _index: (
        np.array((9.0e6 * math.cos(0.5), 9.0e6 * math.sin(0.5), 0.0)),
        np.zeros(3))

    game._init_planning()
    request = game.select(0)
    # Adapter gate uses a captured request directly; worker lifecycle is
    # separately tested. It is not native game acceptance.
    order = orbital_game._solve_planning(request.payload)
    assert order.plan is planned[1]
    assert game.order is None
    assert phasing_calls == [(100.0, pytest.approx(1.1))]
    assert len(calls) == 2
    assert calls[0][1] == 10.0
    assert calls[0][4] == {"propellant_kg": 12.0, "previous": None}
    assert calls[1][1] == 30.0
    assert calls[1][4] == {"propellant_kg": 12.0, "previous": None}
    expected_r = R1 * np.array((math.cos(0.2), math.sin(0.2), 0.0))
    expected_v = math.sqrt(MU_EARTH / R1) * np.array(
        (-math.sin(0.2), math.cos(0.2), 0.0))
    assert calls[1][2] == pytest.approx(expected_r)
    assert calls[1][3] == pytest.approx(expected_v)


def _on_circle_error(game, index):
    """The station lane against its analytic circle at the game's time."""
    spec = game.specs[index]
    angle = spec.angle_rad + _mean_motion(spec.radius_m) * game.time_s
    truth = spec.radius_m * np.array([math.cos(angle), math.sin(angle), 0.0])
    position, _velocity = game.target_state(index)
    return float(np.linalg.norm(position - truth))


def test_delayed_immediate_hohmann_replan_uses_captured_assumptions(monkeypatch):
    """Actual native plan/reference laws; no craft-flight acceptance claim."""
    from types import SimpleNamespace
    from orbital_game import Phasing, TransferOrder
    from orbital_tracker import hohmann_replanner

    previous = hohmann_plan(MU_EARTH, R1, 8e6, t_burn1=100.0)

    class Craft:
        time_s = 0.0
        propellant_kg = 12.0
        offset = np.zeros(3)

        def r(self):
            r, v = previous.reference(self.time_s)
            return r + self.offset, v

    game = object.__new__(OrbitalGame)
    game.mu, game.planner_name = MU_EARTH, "hohmann"
    game.planner = hohmann_plan
    game.craft = Craft()
    game.gains = __import__("orbital_game").MACHINE_GAINS
    game.specs = (TargetSpec("probe", 8e6, 0.5),)
    game.stations = SimpleNamespace(time_s=0.0)
    target_r, target_v = __import__("orbital_game")._circular_state(
        MU_EARTH, 8e6, 0.5)
    game.target_state = lambda index: (np.asarray(target_r), np.asarray(target_v))
    game._init_planning()
    game.order = TransferOrder(0, 0.0, previous, Phasing(0, 100, 0), 0.5)
    captured = game._capture_plan(0, previous)
    candidate = hohmann_replanner(previous, 0.0, captured.position, captured.velocity)
    game.craft.time_s = 2.0
    assert candidate.t_burn1 < game.time_s
    assert game._fresh_plan(captured, candidate) is None
    game.craft.offset = np.array((2e6, 0, 0))
    assert "ON PLAN band" in game._fresh_plan(captured, candidate)


def test_click_flies_the_machine_craft_to_the_target():
    r2, wait = 8.0e6, 600.0
    probe = hohmann_plan(MU_EARTH, R1, r2)
    lead = phasing(MU_EARTH, R1, r2, probe.transfer_time, math.pi,
                   0.0, 0.0).lead_angle
    # place the target so the phasing wait is ``wait``: the lead closes at
    # n1 - n2 (raising); a second station rides along in the same batch
    angle = lead + (_mean_motion(R1) - _mean_motion(r2)) * wait
    game = OrbitalGame(targets=(TargetSpec("probe", r2, angle),
                                TargetSpec("far", 1.3e7, -1.0)))
    try:
        assert game.stations.batch == 2          # ONE batched dt state
        game.select(0)
        # Initial solving is independent; the real craft/stations keep advancing.
        while game.order is None:
            game.step(2.0)
            if game.planning_error is not None:
                raise RuntimeError(game.planning_error)
        order = game.order
        assert order.phasing.t_wait == pytest.approx(wait, rel=1e-9)
        assert order.plan.t_burn1 == pytest.approx(wait, rel=1e-9)
        with pytest.raises(RuntimeError, match="in progress"):
            game.select(0)

        loaded = dict(game.loaded_tank_kg)
        peak = {}
        tilt = 0.0
        while not game.rendezvous:
            game.step(game.frame_window(120.0))
            for label, _count, _lit, live, fired in game.thruster_groups():
                peak[label] = max(peak.get(label, 0.0), live, fired)
            tilt = max(tilt, game.main_gimbal()[0])
        meet = game.rendezvous[0]
        left = game.craft.tank_propellant_kg()
        biprop = sum(loaded[k] - left[k] for k in ("tank.mmh", "tank.nto"))
        main = game.craft.craft.thrusters[
            game.craft.craft.thrusters_by_role("main")[0]]
        ideal = game.craft.craft.mass_kg * (1.0 - math.exp(
            -order.plan.ideal_delta_v / main.thruster_kind.exhaust_velocity_m_s))
        shift = np.linalg.norm(game.craft.centre_of_mass()
                               - game.loaded_centre_of_mass)
        stations = [_on_circle_error(game, i) for i in range(2)]
        print(f"\nrendezvous at t={meet.time_s:.1f} s: distance "
              f"{meet.distance_m:.2f} m, relative speed "
              f"{meet.relative_speed_m_s:.4f} m/s; plan deviation "
              f"{game.plan_deviation_m:.2f} m; on plan {game.on_plan()}, "
              f"re-plans {game.replans()}; bipropellant {biprop:.2f} kg = "
              f"{biprop / ideal:.4f} x TS2.1 ideal; peak groups {peak}; main "
              f"gimbal up to {math.degrees(tilt):.3f} deg; CoM moved "
              f"{1000.0 * shift:.1f} mm; stations off their circles {stations} m")
        assert meet.time_s == pytest.approx(game.arrival_s(), abs=1e-6)
        assert game.on_plan() and game.replans() == 0
        # measured 2026-10-03: 450.1 m, 4.01 m/s, bipropellant 1.113 x ideal.
        # The 4 kN machine finishes its arrival burn ~40 s after t_burn2 and is
        # still trimming at t_burn2 + ARRIVAL_SETTLE_S (step-6 continuation)
        assert meet.distance_m < 1.0e3
        assert meet.relative_speed_m_s < 10.0
        assert 1.0 <= biprop / ideal < 1.2
        # the main engine fires and steers; RCS hold the attitude; the retros
        # are not asked for on a raising transfer
        assert peak["main"] > 0.9 and peak["RCS"] > 0.0
        assert tilt > 0.0
        # the centre of mass moves as the tanks drain
        assert shift > 1.0e-3
        # the batched stations stay on their analytic circles (no lane garbage)
        assert max(stations) < 10.0
        assert not game.busy()
    finally:
        game.close()


def test_drawing_reads_the_machine_parts_gimbal_and_centre_of_mass():
    game = OrbitalGame(targets=(TargetSpec("probe", 8.0e6, 1.0),))
    try:
        craft = game.craft
        machine = craft.craft
        k_main = machine.thrusters_by_role("main")[0]
        # tilt the main engine and light it for a few rounds
        angles = np.zeros((machine.thruster_count, 2))
        angles[k_main] = (0.05, -0.03)
        craft.gimbal(angles)
        throttles = np.zeros(machine.thruster_count)
        throttles[k_main] = 1.0
        for _ in range(10):
            craft.throttle(throttles)
            craft.advance(2.0)
        drawing = craft_drawing(craft, game.loaded_centre_of_mass)
        rotation = craft.attitude()
        prism_centre = 0.5 * (drawing.body.max(axis=0) + drawing.body.min(axis=0))
        assert drawing.body.shape == (8, 3)
        gimbal = craft.gimbal_states()
        for part, thruster, geometry in zip(drawing.parts, machine.thrusters,
                                            craft.thruster_geometry()):
            assert (part.identity, part.role, part.throttle) == (
                geometry.identity, geometry.role, geometry.throttle)
            assert part.exhaust == pytest.approx(geometry.exhaust, abs=0.0)
            assert part.mount_m == pytest.approx(geometry.mount_m, abs=0.0)
            # mount + centre of mass = the declared mount about the fixed point
            fixed = rotation @ (np.asarray(thruster.position_m)
                                - np.asarray(craft.centre_of_mass()))
            assert part.mount_m == pytest.approx(fixed, abs=1e-12)
        main = drawing.parts[k_main]
        # the plume leaves along the CURRENT gimbal direction (tilted)
        assert main.exhaust == pytest.approx(
            -(rotation @ machine.thrusters[k_main].direction_at(*gimbal[k_main])),
            abs=1e-12)
        tilt, command = game.main_gimbal()
        assert tilt > 1.0e-3 and math.isnan(command)   # no tracker command yet
        assert main.throttle > 0.4                      # lit, past its deadband
        groups = {label: lit for label, _n, lit, _live, _fired
                  in game.thruster_groups()}
        assert groups["main"] == 1
        # the centre of mass has moved from where it was at the declared fill
        moved = drawing.centre_of_mass_m - drawing.loaded_centre_of_mass_m
        assert np.linalg.norm(moved) > 1.0e-6
        assert moved == pytest.approx(
            rotation @ (craft.centre_of_mass() - game.loaded_centre_of_mass),
            abs=1e-12)
        fills = {identity: fill for identity, _p, fill in drawing.tanks}
        for tank in machine.tanks:
            assert fills[tank.identity] == pytest.approx(
                craft.tank_propellant_kg()[tank.identity] / tank.capacity_kg)
        assert np.allclose(prism_centre, 0.0, atol=1e-12)
    finally:
        game.close()
