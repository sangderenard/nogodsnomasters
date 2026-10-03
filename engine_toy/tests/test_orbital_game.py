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


def _on_circle_error(game, index):
    """The station lane against its analytic circle at the game's time."""
    spec = game.specs[index]
    angle = spec.angle_rad + _mean_motion(spec.radius_m) * game.time_s
    truth = spec.radius_m * np.array([math.cos(angle), math.sin(angle), 0.0])
    position, _velocity = game.target_state(index)
    return float(np.linalg.norm(position - truth))


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
    assert game.stations.batch == 2          # ONE batched dt state
    order = game.select(0)
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


def test_drawing_reads_the_machine_parts_gimbal_and_centre_of_mass():
    game = OrbitalGame(targets=(TargetSpec("probe", 8.0e6, 1.0),))
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
