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
    orbital_jumper_equations,
    original_transfer_set,
    thrust_cost_integrand,
)

MU_EARTH = 3.986004418e14
R_ORBIT = 7.0e6


def test_orbital_laws_publish_metrics_without_dictating_dt():
    pieces = orbital_jumper_equations(center_count=1, thruster_count=0)
    equations = tuple(equation for _name, group in pieces for equation in group)
    assert not any(equation.has(sp.zoo) for equation in equations)
    assert not any(equation.lhs == sp.Symbol("dt_limit")
                   for equation in equations)
    metric_names = {equation.lhs.name for equation in equations}
    assert set((
        "orbital_translation_energy_residual_j",
        "orbital_rotation_energy_residual_j",
        "orbital_angular_momentum_residual_n_m_s",
        "orbital_attitude_orthogonality",
    )).issubset(metric_names)


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
    # measured (2026-10-02): radius 2.06e-3, energy 1.38e-5, 2224 substeps;
    # (2026-10-03, r() synchronised, momentum seeded half a step back:
    # the leapfrog reading of the same integrator) radius 6.6e-6,
    # energy 4.3e-9, 1760 substeps
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


# ---------------------------------------------------------------- step 7
# Decision 9: propellant mass falls at F / (I_sp g_0); thrust stops at
# empty by the law; a torque spins the craft at tau / I; thrust direction
# follows the attitude.
import honorary_engine_equation_catalogue as honorary
from orbital_actuation import (
    STANDARD_GRAVITY_M_S2,
    THRUSTER_KINDS,
    CraftDesign,
    Thruster,
    actuation_matrix,
    principal_inertia,
    six_axis_jumper,
    torque_matrix,
)


def _tsiolkovsky_delta_v(exhaust_velocity, m_0, m_f):
    """The catalogue's TS2.1 evaluated."""
    return float(honorary.eq_TS2_1.rhs.subs({
        honorary.c_eff: exhaust_velocity, honorary.m_0: m_0,
        honorary.m_f: m_f}))


def _aft_engine(kind, thrust, mass, propellant):
    # on the axis through the centre of mass: force, no torque
    return CraftDesign((Thruster("main", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                 thrust, kind=kind),),
                       mass_kg=mass, propellant_kg=propellant,
                       identity="aft-engine test craft")


def _resting(design, window, **kwargs):
    return OrbitalJumper([], design=design, position_m=(0.0, 0.0, 0.0),
                         velocity_m_s=(0.0, 0.0, 0.0), length_scale_m=1.0e6,
                         window_s=window, **kwargs)


def test_mass_falls_at_thrust_over_isp_g0_through_a_burn():
    thrust, mass, propellant = 400.0, 1000.0, 200.0
    design = _aft_engine("bipropellant", thrust, mass, propellant)
    c = (THRUSTER_KINDS["bipropellant"].specific_impulse_s
         * STANDARD_GRAVITY_M_S2)
    window, rounds = 10.0, 5
    jumper = _resting(design, window)
    jumper.throttle((1.0,))
    for _ in range(rounds):
        jumper.advance()
    t = window * rounds
    burned = thrust * t / c                 # m_dot = F / (I_sp g_0), constant
    print(f"\nburned {mass - jumper.mass_kg:.12f} kg, F t/(Isp g0) "
          f"{burned:.12f} kg, substeps {jumper.substeps}")
    assert jumper.mass_kg == pytest.approx(mass - burned, rel=1e-13)
    assert jumper.propellant_kg == pytest.approx(propellant - burned,
                                                 rel=1e-13)
    assert jumper.propellant_flow_kg_s == pytest.approx(thrust / c,
                                                        rel=1e-14)
    assert jumper.thruster_impulses_n_s == pytest.approx((thrust * t,),
                                                         rel=1e-13)
    # the burn integrates Tsiolkovsky: v_new = v + dt F / m_new is the right
    # Riemann sum of F / m(t), above TS2.1 by at most dt_max (F/m_f - F/m_0)
    _position, velocity = jumper.r()
    ideal = _tsiolkovsky_delta_v(c, mass, mass - burned)
    bound = window * (thrust / (mass - burned) - thrust / mass)
    print(f"delta-v {velocity[0]:.9f} m/s, TS2.1 {ideal:.9f} m/s, "
          f"excess {velocity[0] - ideal:.3e} (bound {bound:.3e})")
    assert 0.0 <= velocity[0] - ideal <= bound
    assert velocity[1:] == pytest.approx((0.0, 0.0), abs=0.0)


def test_thrust_stops_at_empty_by_the_supply_law():
    thrust, mass, propellant = 100.0, 300.0, 2.0
    design = _aft_engine("cold-gas", thrust, mass, propellant)
    c = THRUSTER_KINDS["cold-gas"].specific_impulse_s * STANDARD_GRAVITY_M_S2
    burn_time = propellant * c / thrust     # 13.7 s
    window = 10.0
    jumper = _resting(design, window)
    jumper.throttle((1.0,))
    jumper.advance()                        # t = 10: still burning
    assert jumper.propellant_supply == 1.0
    jumper.advance()                        # t = 20: empties inside
    _p, v_empty = jumper.r()
    for _ in range(2):                      # t = 40: throttle still open
        jumper.advance()
    _p, v_after = jumper.r()
    print(f"\nburn time {burn_time:.4f} s; propellant left "
          f"{jumper.propellant_kg:.3e} kg; supply {jumper.propellant_supply};"
          f" delivered impulse {jumper.thruster_impulses_n_s[0]:.9f} N*s "
          f"(c P0 = {c * propellant:.9f}); substeps {jumper.substeps}")
    assert jumper.propellant_kg == pytest.approx(0.0, abs=1e-12)
    assert jumper.propellant_kg >= 0.0
    assert jumper.mass_kg == pytest.approx(mass - propellant, rel=1e-14)
    assert jumper.propellant_supply == pytest.approx(0.0, abs=1e-12)
    assert jumper.applied_force() == pytest.approx((0.0, 0.0, 0.0),
                                                   abs=1e-9)
    assert v_after == pytest.approx(v_empty, abs=1e-12)
    # exactly the tank's worth of impulse was delivered: c * P0
    assert jumper.thruster_impulses_n_s[0] == pytest.approx(c * propellant,
                                                            rel=1e-12)
    ideal = _tsiolkovsky_delta_v(c, mass, mass - propellant)
    bound = window * (thrust / (mass - propellant) - thrust / mass)
    print(f"delta-v {v_after[0]:.9f}, TS2.1 {ideal:.9f}, excess "
          f"{v_after[0] - ideal:.3e} (bound {bound:.3e})")
    assert abs(v_after[0] - ideal) <= bound


def _spin_craft():
    # an RCS couple about z (equal and opposite, offset 1 m along y: zero
    # net force) and a main engine along craft +x through the centre of mass
    return CraftDesign((
        Thruster("rcs+y", (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), 1.0),
        Thruster("rcs-y", (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), 1.0),
        Thruster("main", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 50.0),
    ), mass_kg=100.0, body_size_m=(2.0, 1.0, 0.5), identity="spin craft")


def _rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def test_pure_torque_spins_at_the_rate_the_inertia_predicts():
    design = _spin_craft()
    inertia = principal_inertia(design)
    assert inertia == pytest.approx(100.0 / 12.0 * np.asarray(
        (1.0 + 0.25, 4.0 + 0.25, 4.0 + 1.0)), rel=1e-15)
    u = np.asarray((1.0, 1.0, 0.0))
    tau = torque_matrix(design) @ u
    assert tau == pytest.approx((0.0, 0.0, 2.0), abs=0.0)
    assert actuation_matrix(design) @ u == pytest.approx((0.0, 0.0, 0.0),
                                                         abs=0.0)
    window, rounds = 0.25, 40
    jumper = _resting(design, window)
    from orbital_jumper import ORBITAL_CHANNEL_NAMES, ORBITAL_ERROR_LIMITS
    attempt_errors = []
    native_advance = jumper.dt_state.program["advance_pieces"]

    def observe(actual, dt):
        result = native_advance(actual, dt)
        values = np.asarray(actual.pub_values).reshape(
            len(jumper.pieces), len(ORBITAL_CHANNEL_NAMES))
        attempt_errors.append(values[:, -len(ORBITAL_ERROR_LIMITS):].max(axis=0).copy())
        return result

    jumper.dt_state.program["advance_pieces"] = observe
    jumper.throttle(u)
    for _ in range(rounds):
        jumper.advance()
    t = window * rounds
    alpha = tau[2] / inertia[2]
    omega = jumper.angular_velocity()
    R = jumper.attitude()
    limits = np.asarray(tuple(ORBITAL_ERROR_LIMITS.values()))
    rejected = sum(bool(np.any(errors > limits)) for errors in attempt_errors)
    print(f"\nspin attempts={len(attempt_errors)}, metric-rejected={rejected}, "
          f"within-error-limits={len(attempt_errors)-rejected}", flush=True)
    print(f"\nphysical spin endpoint: substeps={jumper.substeps}, "
          f"angle={math.atan2(R[1, 0], R[0, 0]):.15g}, "
          f"continuum={alpha * t * t / 2:.15g}, "
          f"max|R^T R-I|={np.abs(R.T @ R - np.eye(3)).max():.15g}, "
          f"det={np.linalg.det(R):.15g}", flush=True)
    print(f"\nomega_z {omega[2]:.15f} rad/s, tau t / I_z "
          f"{alpha * t:.15f} rad/s")
    assert jumper.torque() == pytest.approx(tau, abs=0.0)
    assert omega[2] == pytest.approx(alpha * t, rel=1e-13)
    assert omega[:2] == pytest.approx((0.0, 0.0), abs=0.0)
    position, velocity = jumper.r()
    assert position == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert velocity == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    # The physical endpoint remains a rotation and follows the continuum
    # angle. Internal step counts and the former Cayley formula are not
    # properties of the library integrator. Keep the established physical
    # phase-error bound and orthogonality requirement below.
    turned = math.atan2(R[1, 0], R[0, 0])
    print(f"turned {turned:.12f} rad (continuum alpha t^2/2 = "
          f"{alpha * t * t / 2:.12f}); |R^T R - I| "
          f"{np.abs(R.T @ R - np.eye(3)).max():.2e}, det {np.linalg.det(R)}")
    assert R == pytest.approx(_rz(turned), abs=1e-13)
    assert np.abs(R.T @ R - np.eye(3)).max() < 1e-14
    assert abs(turned - alpha * t * t / 2) <= alpha * t * window / 2

    # Thrust follows the integrated attitude during the next outer window.
    # For this constant z spin, d(R e_y)/dt = -omega_z R e_x, so its actual
    # endpoint difference determines the force integral without predicting
    # internal steps or treating the direction as fixed during the window.
    before = jumper.attitude()
    impulse_before = jumper.applied_impulse_n_s
    jumper.throttle((0.0, 0.0, 1.0))
    jumper.advance()
    mean_force = (jumper.applied_impulse_n_s - impulse_before) / window
    expected_mean = (50.0 / (omega[2] * window)
                     * (before[:, 1] - jumper.attitude()[:, 1]))
    assert mean_force == pytest.approx(expected_mean, rel=1e-14, abs=1e-12)
    assert jumper.torque() == pytest.approx((0.0, 0.0, 0.0), abs=0.0)
    assert jumper.angular_velocity()[2] == pytest.approx(alpha * t,
                                                         rel=1e-13)


def test_thrust_direction_follows_a_declared_attitude():
    design = six_axis_jumper(250.0, 1000.0)
    quarter = _rz(math.pi / 2.0)
    jumper = _resting(design, 1.0, attitude=quarter)
    jumper.throttle((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))     # craft +x
    jumper.advance()
    assert jumper.applied_force() == pytest.approx((0.0, 250.0, 0.0),
                                                   abs=1e-12)
    _position, velocity = jumper.r()
    assert velocity == pytest.approx((0.0, 0.25, 0.0), abs=1e-15)

    # an arbitrary attitude: the world force is R @ B_craft @ u
    axis = np.asarray((1.0, -2.0, 0.5)) / np.linalg.norm((1.0, -2.0, 0.5))
    angle = 0.83
    K = np.asarray(((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]),
                    (-axis[1], axis[0], 0.0)))
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    jumper = _resting(design, 1.0, attitude=R)
    u = np.asarray((0.2, 0.5, 1.0, 0.0, 0.125, 0.9))
    jumper.throttle(u)
    assert jumper.actuation_matrix() == pytest.approx(
        R @ actuation_matrix(design), rel=1e-15, abs=1e-15)
    jumper.advance()
    assert jumper.applied_force() == pytest.approx(
        actuation_matrix(design, R) @ u, rel=1e-13, abs=1e-12)
    # no rate, no torque: the attitude is held exactly
    assert np.array_equal(jumper.attitude(), R)


def _stations():
    """The game's stations: thrusterless bodies at one radius, different
    phases (one speed, so a shared dt is each single lane's own dt) and
    different masses."""
    speed = math.sqrt(MU_EARTH / R_ORBIT)
    phases = np.asarray((0.0, 0.9, 2.1, 4.0))
    masses = np.asarray((1000.0, 2.0e4, 5.0, 3.3e5))
    positions = R_ORBIT * np.stack([np.cos(phases), np.sin(phases),
                                    np.zeros(4)], axis=1)
    velocities = speed * np.stack([-np.sin(phases), np.cos(phases),
                                   np.zeros(4)], axis=1)
    return masses, positions, velocities


def test_batched_lanes_share_one_dt_state_and_match_single_lanes():
    # Four stations as ONE dt state at batch 4 against four batch-1 states.
    # Each persistent state runs its own program (see
    # test_interleaved_states_each_run_their_own_program).
    import time
    masses, positions, velocities = _stations()
    center = [GravityCenter((0.0, 0.0, 0.0), MU_EARTH)]
    common = dict(length_scale_m=5.0e4, window_s=300.0)
    rounds = 5
    singles = []
    start = time.perf_counter()
    for m, p, v in zip(masses, positions, velocities):
        single = OrbitalJumper(center, mass_kg=m, position_m=p,
                               velocity_m_s=v, **common)
        for _ in range(rounds):
            single.advance()
        singles.append((single.r(), single.substeps))
    singles_s = time.perf_counter() - start
    start = time.perf_counter()
    batched = OrbitalJumper(center, mass_kg=masses, position_m=positions,
                            velocity_m_s=velocities, batch=4, **common)
    for _ in range(rounds):
        batched.advance()
    batched_s = time.perf_counter() - start
    position, velocity = batched.r()
    assert position.shape == velocity.shape == (4, 3)
    assert batched.mass_kg == pytest.approx(masses, rel=0.0)
    worst = 0.0
    for lane, ((p, v), substeps) in enumerate(singles):
        worst = max(worst, float(np.linalg.norm(position[lane] - p)))
        assert position[lane] == pytest.approx(p, rel=1e-12, abs=1e-6)
        assert velocity[lane] == pytest.approx(v, rel=1e-12, abs=1e-9)
        assert substeps == batched.substeps
    print(f"\nbatch 4: {batched.substeps} substeps, {batched_s:.3f} s "
          f"(build + {rounds} rounds); 4 singles: "
          f"{sum(n for _, n in singles)} substeps, {singles_s:.3f} s; "
          f"worst lane difference {worst:.3e} m")


def test_interleaved_states_each_run_their_own_program():
    masses, positions, velocities = _stations()
    center = [GravityCenter((0.0, 0.0, 0.0), MU_EARTH)]
    common = dict(length_scale_m=5.0e4, window_s=300.0)
    batched = OrbitalJumper(center, mass_kg=masses, position_m=positions,
                            velocity_m_s=velocities, batch=4, **common)
    OrbitalJumper(center, mass_kg=1000.0, position_m=positions[0],
                  velocity_m_s=velocities[0], **common)
    batched.advance()
    assert batched.mass_kg == pytest.approx(masses, rel=0.0)


def test_dt_grows_back_once_a_spin_stops():
    # every orbital piece declares BIND; the configured error metrics refine
    # the spin and the controller grows back after the despin
    design = CraftDesign((
        Thruster("rcs+y", (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), 1.0),
        Thruster("rcs-y", (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), 1.0),
        Thruster("anti+y", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), 1.0),
        Thruster("anti-y", (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), 1.0),
    ), mass_kg=100.0, body_size_m=(2.0, 1.0, 0.5))
    jumper = _resting(design, 10.0)
    counts = []
    for throttles, rounds in (((1, 1, 0, 0), 3), ((0, 0, 1, 1), 3),
                              ((0, 0, 0, 0), 3)):
        jumper.throttle(throttles)
        for _ in range(rounds):
            before = jumper.substeps
            jumper.advance()
            counts.append(jumper.substeps - before)
            print(f"spin recovery t={jumper.time_s:.12g}: "
                  f"attempts={counts[-1]}, omega_z={jumper.angular_velocity()[2]:.12g}",
                  flush=True)
    print(f"\nsubsteps per 10 s round, spin/despin/coast: {counts}; "
          f"w_z {jumper.angular_velocity()[2]:.3e}")
    assert jumper.angular_velocity()[2] == pytest.approx(0.0, abs=1e-12)
    assert max(counts[1:6]) > 50          # the spin bounded dt
    assert counts[-2:] == [1, 1]          # and it is gone


def test_clipped_rounds_coast_one_orbit_with_the_mean_step_kick():
    # 5 s rounds against a 3.3 s CFL step: every round ends on a clipped
    # substep.  Measured: before the kick weight (dt alone) |r - R| 18965 m,
    # |v_r| 20.3 m/s, energy 7.3e-6; with (dt_prev + dt) / 2: 101.7 m,
    # 0.055 m/s, 2.1e-10.
    speed = math.sqrt(MU_EARTH / R_ORBIT)
    period = 2.0 * math.pi * math.sqrt(R_ORBIT**3 / MU_EARTH)
    jumper = OrbitalJumper(
        [GravityCenter((0.0, 0.0, 0.0), MU_EARTH)], mass_kg=1000.0,
        position_m=(R_ORBIT, 0.0, 0.0), velocity_m_s=(0.0, speed, 0.0),
        length_scale_m=5.0e4, window_s=5.0)
    energy_0 = 0.5 * speed**2 - MU_EARTH / R_ORBIT
    worst_r = worst_vr = worst_e = 0.0
    while jumper.time_s < period:
        jumper.advance()
        position, velocity = jumper.r()
        r = float(np.linalg.norm(position))
        worst_r = max(worst_r, abs(r - R_ORBIT))
        worst_vr = max(worst_vr, abs(float(position @ velocity)) / r)
        worst_e = max(worst_e, abs(0.5 * float(velocity @ velocity)
                                   - MU_EARTH / r - energy_0) / abs(energy_0))
    print(f"\n|r - R| {worst_r:.1f} m, |v_r| {worst_vr:.4f} m/s, energy "
          f"{worst_e:.2e}, substeps {jumper.substeps}")
    assert worst_r < 200.0 and worst_vr < 0.1 and worst_e < 1.0e-9
