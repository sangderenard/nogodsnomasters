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
    # on the target circle
    assert abs(radius - R_GEO) < 200.0
    assert abs(np.linalg.norm(velocity) - circular) < 0.05
    assert abs(radial_speed) < 1.0
    # on the plan
    assert report.final_position_error_m < 200.0
    assert report.final_velocity_error_m_s < 1.0
    # fuel against the impulsive ideal (a finite-thrust six-axis craft
    # cannot beat it)
    assert 1.0 <= report.fuel_ratio < 1.3
    assert report.ideal_impulse_n_s == pytest.approx(
        MASS * (abs(plan.dv1) + abs(plan.dv2)), rel=1e-15)
