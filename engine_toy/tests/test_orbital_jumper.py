"""Orbital craft build step 1: the jumper behind the r()/F() seam.

    python -m pytest tests/test_orbital_jumper.py -q     (from engine_toy/)
"""
import math

import numpy as np
import pytest
import sympy as sp
from sympy.core.function import AppliedUndef

from orbital_jumper import (
    AXES,
    GravityCenter,
    OrbitalJumper,
    gravity_force_rhs,
    original_transfer_set,
    thrust_cost_integrand,
)

MU_EARTH = 3.986004418e14
R_ORBIT = 7.0e6


def _original_specific_energy():
    """The original set's ``total_energy_expression`` as a function of
    (position, velocity, mu_1, mu_2) -- the provenance energy."""
    energy = original_transfer_set()["total_energy_expression"]
    derivatives = sorted(energy.atoms(sp.Derivative), key=str)
    functions = sorted(energy.atoms(AppliedUndef), key=str)
    velocity = sp.symbols("vx vy vz")
    position = sp.symbols("px py pz")
    mu_1, mu_2 = sorted((sym for sym in energy.free_symbols
                         if sym.name.startswith("mu")), key=str)
    spelled = energy.xreplace(dict(zip(derivatives, velocity)))
    spelled = spelled.xreplace(dict(zip(functions, position)))
    return sp.lambdify((position, velocity, mu_1, mu_2), spelled, "math")


def test_circular_orbit_one_period_holds_radius_and_energy():
    # provenance: the catalogue N4.1 vector form per unit mass IS the
    # original set's F_grav (center at the origin)
    original = original_transfer_set()["force_components"]["F_grav1"]
    point = {"position_x": 6.1e6, "position_y": -2.3e6, "position_z": 1.7e6}
    mu_1 = next(sym for sym in original.free_symbols if sym.name == "mu_1")
    functions = sorted(original.atoms(AppliedUndef), key=str)
    for index, axis in enumerate(AXES):
        mine = gravity_force_rhs(axis, 1).subs({
            **{sp.Symbol(k): v for k, v in point.items()},
            sp.Symbol("center0_x"): 0.0, sp.Symbol("center0_y"): 0.0,
            sp.Symbol("center0_z"): 0.0, sp.Symbol("center0_mu"): MU_EARTH,
            sp.Symbol("mass"): 1.0})
        theirs = original[index].subs(dict(zip(
            functions, (point["position_x"], point["position_y"],
                        point["position_z"])))).subs(mu_1, MU_EARTH)
        assert float(mine) == pytest.approx(float(theirs), rel=1e-13)

    speed = math.sqrt(MU_EARTH / R_ORBIT)
    period = 2.0 * math.pi * math.sqrt(R_ORBIT**3 / MU_EARTH)
    rounds = 20
    jumper = OrbitalJumper(
        [GravityCenter((0.0, 0.0, 0.0), MU_EARTH)], mass_kg=1000.0,
        position_m=(R_ORBIT, 0.0, 0.0), velocity_m_s=(0.0, speed, 0.0),
        length_scale_m=5.0e4, window_s=period / rounds)
    state = jumper.dt_state
    energy = _original_specific_energy()
    energy_0 = energy((R_ORBIT, 0.0, 0.0), (0.0, speed, 0.0), MU_EARTH, 0.0)
    assert energy_0 == pytest.approx(-MU_EARTH / (2.0 * R_ORBIT), rel=1e-15)

    radius_error = energy_drift = 0.0
    for _ in range(rounds):
        jumper.advance()
        position, velocity = jumper.r()
        radius_error = max(radius_error,
                           abs(np.linalg.norm(position) - R_ORBIT) / R_ORBIT)
        energy_drift = max(energy_drift, abs(
            energy(tuple(position), tuple(velocity), MU_EARTH, 0.0)
            - energy_0) / abs(energy_0))
    assert jumper.dt_state is state          # one persistent state
    assert jumper.time_s == pytest.approx(period, rel=1e-12)
    mean_dt = period / jumper.substeps
    print(f"\nradius error {radius_error:.3e}, specific-energy drift "
          f"{energy_drift:.3e}, substeps {jumper.substeps}, mean dt "
          f"{mean_dt:.3f} s (CFL bound 0.5*dx/v = "
          f"{0.5 * 5.0e4 / speed:.3f} s)")
    # measured (2026-10-02): radius 2.06e-3, energy 1.38e-5, 2224 substeps
    assert radius_error < 5.0e-3
    assert energy_drift < 1.0e-4
    assert jumper.fuel_impulse_n_s == 0.0


def test_constant_force_without_gravity_is_uniform_acceleration():
    mass, force = 1000.0, np.asarray((100.0, 0.0, 0.0))
    x0, v0 = np.asarray((10.0, -4.0, 2.0)), np.asarray((10.0, 5.0, 0.0))
    dx, window, rounds = 10.0, 10.0, 10
    jumper = OrbitalJumper([], mass_kg=mass, position_m=x0, velocity_m_s=v0,
                           length_scale_m=dx, window_s=window)
    jumper.F(force)
    for _ in range(rounds):
        jumper.advance()
    t = window * rounds
    position, velocity = jumper.r()
    expected = x0 + v0 * t + force * t**2 / (2.0 * mass)
    # symplectic Euler's first-order term is (a/2) * sum(dt_i^2), at most
    # (a/2) * t * dt_max, and dt_max = dx / max_vel_ever <= dx / |v0|
    bound = 0.5 * (force[0] / mass) * t * dx / np.linalg.norm(v0)
    error = position - expected
    print(f"\nx error {error[0]:.4e} m (bound {bound:.4e}), substeps "
          f"{jumper.substeps}")
    assert 0.0 <= error[0] <= bound
    assert error[1:] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert velocity == pytest.approx(v0 + force * t / mass, rel=1e-12)
    assert jumper.fuel_impulse_n_s == pytest.approx(100.0 * t, rel=1e-12)
    assert thrust_cost_integrand() == sp.sqrt(sum(
        sp.Symbol(f"applied_force_{axis}")**2 for axis in AXES))


def test_seam_r_reflects_state_and_F_sets_next_round_force():
    mass, window = 500.0, 4.0
    x0, v0 = np.asarray((1.0, 2.0, 3.0)), np.asarray((0.5, 0.0, -0.25))
    jumper = OrbitalJumper([], mass_kg=mass, position_m=x0, velocity_m_s=v0,
                           length_scale_m=1.0, window_s=window)
    position, velocity = jumper.r()
    assert position == pytest.approx(x0) and velocity == pytest.approx(v0)
    position[...] = 0.0                       # r() hands out copies
    assert jumper.r()[0] == pytest.approx(x0)

    jumper.F((0.0, 0.0, 0.0))
    jumper.advance()
    position, velocity = jumper.r()
    assert position == pytest.approx(x0 + v0 * window, rel=1e-13)
    assert velocity == pytest.approx(v0, rel=1e-13)
    assert float(jumper.dt_state.position_x[0]) == position[0]

    kick = np.asarray((0.0, 2.0 * mass, -mass))   # 2 m/s^2 along y, -1 along z
    jumper.F(kick)
    assert velocity == pytest.approx(jumper.r()[1])  # no effect until a round
    jumper.advance()
    _position, after = jumper.r()
    assert after == pytest.approx(v0 + kick / mass * window, rel=1e-12)
    assert jumper.fuel_impulse_n_s == pytest.approx(
        np.linalg.norm(kick) * window, rel=1e-12)

    jumper.F((0.0, 0.0, 0.0))
    jumper.advance()
    assert jumper.r()[1] == pytest.approx(after, rel=1e-12)
