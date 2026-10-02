"""Orbital craft build step 3a: the Hohmann plan and its reference.

    python -m pytest tests/test_orbital_plan.py -q     (from engine_toy/)
"""
import math

import numpy as np
import pytest
import sympy as sp

import orbital_plan
from orbital_plan import (
    KEPLER_LAWS,
    KEPLER_RESIDUAL_BOUND,
    hohmann_plan,
    kepler_residual,
    reference,
    two_body_acceleration,
)

MU_EARTH = 3.986004418e14
R_LEO = 7.0e6
R_GEO = 42164.0e3


def _textbook(mu, r1, r2):
    dv1 = math.sqrt(mu / r1) * (math.sqrt(2.0 * r2 / (r1 + r2)) - 1.0)
    dv2 = math.sqrt(mu / r2) * (1.0 - math.sqrt(2.0 * r1 / (r1 + r2)))
    time = math.pi * math.sqrt(((r1 + r2) / 2.0)**3 / mu)
    return dv1, dv2, time


@pytest.fixture(scope="module")
def plan():
    return hohmann_plan(MU_EARTH, R_LEO, R_GEO, t_burn1=300.0, phase=0.4)


def test_plan_matches_textbook_hohmann(plan):
    dv1, dv2, time = _textbook(MU_EARTH, R_LEO, R_GEO)
    assert plan.dv1 == pytest.approx(dv1, rel=1e-12)
    assert plan.dv2 == pytest.approx(dv2, rel=1e-12)
    assert plan.transfer_time == pytest.approx(time, rel=1e-12)
    assert plan.a_transfer == pytest.approx((R_LEO + R_GEO) / 2, rel=1e-15)
    assert plan.t_burn2 == pytest.approx(300.0 + time, rel=1e-15)


def test_laws_are_the_originals_composed():
    a, e = sp.symbols("a e", positive=True)
    E = sp.Symbol("E", real=True)
    # orbital_radius_theta at the true anomaly of eq_KE2_4 is a(1 - e cos E)
    radius = KEPLER_LAWS["eq_KE2_6"].rhs.xreplace({
        sp.Symbol("cos_nu", real=True): KEPLER_LAWS["eq_KE2_4"].rhs})
    assert sp.simplify(radius - a * (1 - e * sp.cos(E))) == 0
    # the stage is Newton on Kepler's equation: a root is a fixed point
    stage = KEPLER_LAWS["eq_KE2_2"].rhs
    M, E_k = sp.symbols("M E_k", real=True)
    assert sp.simplify(stage.xreplace({M: E_k - e * sp.sin(E_k)}) - E_k) == 0
    assert set(orbital_plan.KEPLER_SCALE.laws) == set(KEPLER_LAWS)


def test_reference_stays_on_the_transfer_ellipse(plan):
    t = np.linspace(plan.t_burn1, plan.t_burn2, 401)[:-1]
    r, v = reference(plan, t)
    radius = np.linalg.norm(r, axis=1)
    vis_viva = MU_EARTH * (2.0 / radius - 1.0 / plan.a_transfer)
    speed2 = np.sum(v * v, axis=1)
    assert np.max(np.abs(speed2 - vis_viva) / vis_viva) < 1e-9
    assert np.all(r[:, 2] == 0.0) and np.all(v[:, 2] == 0.0)
    # burn 1 at periapsis r1, polar angle = phase
    r_start, _ = reference(plan, plan.t_burn1)
    assert np.linalg.norm(r_start) == pytest.approx(R_LEO, rel=1e-14)
    assert math.atan2(r_start[1], r_start[0]) == pytest.approx(0.4, abs=1e-14)


def test_reference_reaches_r2_at_the_transfer_time(plan):
    r_end, v_end = reference(plan, plan.t_burn2 * (1.0 - 1e-16))
    assert np.linalg.norm(r_end) == pytest.approx(R_GEO, rel=1e-12)
    dv1, dv2, _ = _textbook(MU_EARTH, R_LEO, R_GEO)
    arrival = math.sqrt(MU_EARTH / R_GEO) - dv2
    assert np.linalg.norm(v_end) == pytest.approx(arrival, rel=1e-10)
    # after burn 2 the reference is the target circle
    t = np.linspace(plan.t_burn2, plan.t_burn2 + 86164.0, 97)
    r, v = reference(plan, t)
    assert np.max(np.abs(np.linalg.norm(r, axis=1) / R_GEO - 1)) < 1e-14
    assert np.max(np.abs(np.linalg.norm(v, axis=1)
                         / math.sqrt(MU_EARTH / R_GEO) - 1)) < 1e-14


def test_reference_velocity_is_the_time_derivative(plan):
    t = np.linspace(plan.t_burn1 + 1.0, plan.t_burn2 - 1.0, 64)
    h = 0.01
    ahead, _ = reference(plan, t + h)
    behind, _ = reference(plan, t - h)
    _, v = reference(plan, t)
    error = np.max(np.abs((ahead - behind) / (2 * h) - v))
    assert error / np.max(np.linalg.norm(v, axis=1)) < 1e-8


def test_kepler_stage_residual_below_bound(plan):
    t = np.linspace(-1000.0, plan.t_burn2 + 5000.0, 2048)
    residual = kepler_residual(plan, t)
    assert np.max(residual) <= KEPLER_RESIDUAL_BOUND
    assert np.max(residual) < 1e-14        # measured 2.4e-16


def test_lowering_transfer_is_the_same_laws():
    plan = hohmann_plan(MU_EARTH, R_GEO, R_LEO, t_burn1=0.0)
    dv1, dv2, time = _textbook(MU_EARTH, R_GEO, R_LEO)
    assert plan.dv1 == pytest.approx(dv1, rel=1e-12) and plan.dv1 < 0
    assert plan.dv2 == pytest.approx(dv2, rel=1e-12) and plan.dv2 < 0
    assert plan.transfer_time == pytest.approx(time, rel=1e-12)
    r_end, _ = reference(plan, plan.t_burn2 * (1.0 - 1e-16))
    assert np.linalg.norm(r_end) == pytest.approx(R_LEO, rel=1e-12)


def test_plan_refuses_outside_validated_eccentricity():
    with pytest.raises(ValueError, match="eccentricity"):
        hohmann_plan(MU_EARTH, R_LEO, 13.0 * R_LEO)


def test_two_body_acceleration_is_inverse_square():
    points = np.asarray([[R_LEO, 0.0, 0.0], [0.0, -R_GEO, 0.0],
                         [3.0e6, 4.0e6, 12.0e6]])
    g = two_body_acceleration(MU_EARTH, points)
    radius = np.linalg.norm(points, axis=1, keepdims=True)
    assert g == pytest.approx(-MU_EARTH * points / radius**3, rel=1e-14)
