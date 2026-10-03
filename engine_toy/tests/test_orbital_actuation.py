"""Orbital craft build step 2: the actuation matrix.

    python -m pytest tests/test_orbital_actuation.py -q     (from engine_toy/)
"""
import math

import numpy as np
import pytest
import sympy as sp

import honorary_engine_equation_catalogue as honorary
from orbital_actuation import (
    CraftDesign,
    Thruster,
    actuation_matrix,
    six_axis_jumper,
    thrust_magnitude,
)
from orbital_jumper import OrbitalJumper

THRUST = 250.0
MASS = 1000.0


def _resting(design, window=1.0):
    # at rest, no centers: the first attempt is the whole window
    return OrbitalJumper([], design=design, position_m=(0.0, 0.0, 0.0),
                         velocity_m_s=(0.0, 0.0, 0.0), length_scale_m=1.0e6,
                         window_s=window)


def test_thrust_law_is_catalogue_ts1_2_with_c_cancelled():
    magnitude = thrust_magnitude(0)
    assert not magnitude.has(honorary.c_eff)
    u, lo, hi, fmax = sp.symbols("thruster0_throttle thruster0_throttle_min "
                                 "thruster0_throttle_max thruster0_max_thrust")
    assert magnitude == sp.Min(sp.Max(u, lo), hi) * fmax


def test_six_axis_jumper_matrix():
    design = six_axis_jumper(THRUST, MASS)
    B = actuation_matrix(design)
    expected = THRUST * np.asarray([[1, -1, 0, 0, 0, 0],
                                    [0, 0, 1, -1, 0, 0],
                                    [0, 0, 0, 0, 1, -1]], dtype=float)
    assert B.shape == (3, 6) and np.array_equal(B, expected)
    # every mount is on the line of action through the centre of mass
    for thruster in design.thrusters:
        assert np.linalg.norm(np.cross(thruster.position_m,
                                       thruster.direction)) == 0.0


def test_each_single_thruster_produces_its_column_of_B():
    design = six_axis_jumper(THRUST, MASS)
    B = actuation_matrix(design)
    jumper = _resting(design)
    for k in range(design.thruster_count):
        u = np.zeros(design.thruster_count)
        u[k] = 1.0
        jumper.throttle(u)
        jumper.advance()
        assert np.array_equal(jumper.applied_force(), B[:, k])


def test_mixed_throttles_produce_B_at_u_for_any_design():
    tilt = 1.0 / math.sqrt(3.0)
    design = CraftDesign((
        Thruster("aft", (-2.0, 0.0, 0.0), (1.0, 0.0, 0.0), 900.0),
        Thruster("canted", (0.0, 1.0, 0.0), (tilt, -tilt, tilt), 120.0,
                 kind="cold-gas"),
        Thruster("ventral", (0.0, 0.0, 1.0), (0.0, 0.6, -0.8), 45.0),
    ), mass_kg=750.0, identity="three-thruster test craft",
        # step 7: the cold-gas thruster burns propellant; declare some
        propellant_kg=50.0)
    jumper = _resting(design)
    u = np.asarray((0.3, 0.75, 1.0))
    jumper.throttle(u)
    jumper.advance()
    B = actuation_matrix(design)
    assert jumper.applied_force() == pytest.approx(B @ u, rel=1e-15,
                                                   abs=1e-12)

    six = six_axis_jumper(THRUST, MASS)
    jumper = _resting(six)
    u = np.asarray((0.2, 0.5, 1.0, 0.0, 0.125, 0.9))
    jumper.throttle(u)
    jumper.advance()
    assert jumper.applied_force() == pytest.approx(
        actuation_matrix(six) @ u, rel=1e-15, abs=1e-12)


def test_throttles_are_clamped_to_their_declared_range():
    design = six_axis_jumper(THRUST, MASS)
    B = actuation_matrix(design)
    jumper = _resting(design)
    commanded = np.asarray((1.7, -0.4, 0.5, 3.0, -2.0, 0.0))
    jumper.throttle(commanded)
    jumper.advance()
    applied = np.clip(commanded, 0.0, 1.0)
    assert np.array_equal(jumper.applied_force(), B @ applied)
    assert np.array_equal(jumper.commanded_force(), B @ applied)
    # fuel is charged on the clamped throttle, per thruster
    assert jumper.thruster_impulses_n_s == pytest.approx(
        THRUST * applied * 1.0, rel=1e-15)

    # a declared narrower range clamps there
    limited = CraftDesign((Thruster("vernier", (0.0, 0.0, 0.0),
                                    (0.0, 0.0, 1.0), 10.0,
                                    throttle_min=0.2, throttle_max=0.6),),
                          mass_kg=50.0)
    jumper = _resting(limited)
    for commanded, applied in ((0.9, 0.6), (0.0, 0.2), (0.4, 0.4)):
        jumper.throttle((commanded,))
        jumper.advance()
        assert jumper.applied_force() == pytest.approx(
            (0.0, 0.0, 10.0 * applied), rel=1e-15)


def test_plus_x_burn_without_gravity_matches_step1_constant_force():
    # step 1's constant-force test, driven by the +x thruster's throttle
    design = six_axis_jumper(100.0, 1000.0)
    x0, v0 = np.asarray((10.0, -4.0, 2.0)), np.asarray((10.0, 5.0, 0.0))
    dx, window, rounds = 10.0, 10.0, 10
    jumper = OrbitalJumper([], design=design, position_m=x0, velocity_m_s=v0,
                           length_scale_m=dx, window_s=window)
    jumper.throttle((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    for _ in range(rounds):
        jumper.advance()
    t = window * rounds
    force = np.asarray((100.0, 0.0, 0.0))
    position, velocity = jumper.r()
    expected = x0 + v0 * t + force * t**2 / (2.0 * design.mass_kg)
    bound = 0.5 * (force[0] / design.mass_kg) * t * dx / np.linalg.norm(v0)
    error = position - expected
    print(f"\nx error {error[0]:.4e} m (bound {bound:.4e}), substeps "
          f"{jumper.substeps}")
    assert 0.0 <= error[0] <= bound
    assert error[1:] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert velocity == pytest.approx(v0 + force * t / design.mass_kg,
                                     rel=1e-12)
    assert jumper.fuel_impulse_n_s == pytest.approx(100.0 * t, rel=1e-12)
    assert jumper.thruster_impulses_n_s == pytest.approx(
        (100.0 * t, 0, 0, 0, 0, 0), rel=1e-12, abs=0.0)


def test_opposed_thrusters_cancel_force_but_both_burn_fuel():
    design = six_axis_jumper(THRUST, MASS)
    jumper = _resting(design, window=5.0)
    jumper.throttle((1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    jumper.advance()
    assert np.array_equal(jumper.applied_force(), np.zeros(3))
    assert jumper.r()[1] == pytest.approx((0.0, 0.0, 0.0), abs=0.0)
    assert jumper.fuel_impulse_n_s == pytest.approx(2.0 * THRUST * 5.0,
                                                    rel=1e-15)


def test_raw_F_superposes_on_the_throttled_force():
    design = six_axis_jumper(THRUST, MASS)
    jumper = _resting(design, window=2.0)
    raw = np.asarray((0.0, 30.0, -40.0))
    jumper.throttle((0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    jumper.F(raw)
    jumper.advance()
    assert jumper.applied_force() == pytest.approx(
        raw + (0.0, 0.0, THRUST), rel=1e-15)
    # thruster impulse plus the original set's |F| for the raw part
    assert jumper.fuel_impulse_n_s == pytest.approx(
        (THRUST + 50.0) * 2.0, rel=1e-15)
