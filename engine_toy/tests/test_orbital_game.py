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
    craft_body_corners,
    craft_part_geometry,
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


def test_click_flies_a_phased_transfer_to_the_target():
    r2, wait = 8.0e6, 600.0
    probe = hohmann_plan(MU_EARTH, R1, r2)
    lead = phasing(MU_EARTH, R1, r2, probe.transfer_time, math.pi,
                   0.0, 0.0).lead_angle
    # place the target so the phasing wait is ``wait``: the lead closes at
    # n1 - n2 (raising)
    angle = lead + (_mean_motion(R1) - _mean_motion(r2)) * wait
    game = OrbitalGame(targets=(TargetSpec("probe", r2, angle),))
    order = game.select(0)
    assert order.phasing.t_wait == pytest.approx(wait, rel=1e-9)
    assert order.plan.t_burn1 == pytest.approx(wait, rel=1e-9)
    with pytest.raises(RuntimeError, match="in progress"):
        game.select(0)

    fired = np.zeros(game.craft.design.thruster_count)
    while not game.rendezvous:
        game.step(game.frame_window(120.0))
        fired = np.maximum(fired, game.plume_throttles)
    meet = game.rendezvous[0]
    loaded = game.craft.design.propellant_kg
    used = loaded - game.propellant_kg
    # Tsiolkovsky (eq_TS1_4's c = I_sp g0) for the impulsive ideal
    kind = game.craft.design.thrusters[0].thruster_kind
    ideal = game.craft.design.mass_kg * (1.0 - math.exp(
        -order.plan.ideal_delta_v / kind.exhaust_velocity_m_s))
    print(f"\nrendezvous at t={meet.time_s:.1f} s: distance "
          f"{meet.distance_m:.1f} m, relative speed "
          f"{meet.relative_speed_m_s:.4f} m/s; plan deviation "
          f"{game.plan_deviation_m:.1f} m; propellant {used:.2f} kg = "
          f"{used / ideal:.4f} x ideal; peak plume {fired.round(3)}")
    assert meet.time_s == pytest.approx(game.arrival_s(), abs=1e-6)
    # measured 2026-10-03: 4.93 km, 9.5 m/s.  The distance is dominated by
    # the station's own integration error (4.8 km off its analytic circle
    # at dx = 5 km over the flight); the speed is the tracker still
    # settling the finite arrival burn (TrackingGains defaults)
    assert meet.distance_m < 1.0e4
    assert meet.relative_speed_m_s < 15.0
    # measured 1.71: the default PD spreads each burn over ~1/w = 50 s and
    # its critically-damped velocity response reverses sign (the tracker's
    # cost, not the game's); a loose sanity bound only
    assert 1.0 <= used / ideal < 2.0
    # the plumes animate: both burns light thrusters (measured peak frame
    # throttle 0.39: the PD demand 2 w dv m is ~0.5 of the 100 kN)
    assert fired.max() > 0.2
    assert np.count_nonzero(fired > 0.1) >= 2
    assert not game.busy()


def test_plumes_come_from_the_declared_parts():
    game = OrbitalGame(targets=(TargetSpec("probe", 8.0e6, 1.0),))
    throttles = np.linspace(0.0, 1.0, game.craft.design.thruster_count)
    parts = craft_part_geometry(game.craft, throttles)
    attitude = game.craft.attitude()
    for part, thruster, u in zip(parts, game.craft.design.thrusters,
                                 throttles):
        assert part.identity == thruster.identity
        assert part.mount_m == pytest.approx(
            attitude @ np.asarray(thruster.position_m), abs=1e-12)
        assert part.exhaust == pytest.approx(
            -(attitude @ np.asarray(thruster.direction)), abs=1e-12)
        assert part.throttle == u
    corners = craft_body_corners(game.craft)
    assert corners.shape == (8, 3)
    assert np.ptp(corners @ attitude, axis=0) == pytest.approx(
        game.craft.design.body_size_m, abs=1e-12)
