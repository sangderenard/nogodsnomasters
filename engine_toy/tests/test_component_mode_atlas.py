import numpy as np
import pytest

from component_mode_atlas import (AdaptiveGestaltPolicy,
                                  AdaptiveGestaltRuntime,
                                  AdaptiveGestaltState,
                                  GestaltStressEnvelope,
                                  assemble_component_mode_atlases,
                                  build_component_mode_atlas,
                                  build_rigid_gestalt_atlas)


def _three_dof_component():
    # A short serial structure with its first coordinate as the shared mount.
    k = np.array([[20.0, -10.0, 0.0],
                  [-10.0, 20.0, -10.0],
                  [0.0, -10.0, 10.0]])
    m = np.diag([2.0, 1.0, 0.5])
    return k, m


def test_full_component_atlas_is_an_exact_change_of_basis():
    k, m = _three_dof_component()
    atlas = build_component_mode_atlas(
        "witness", k, m, np.array([10, 11, 12]), np.array([0]))

    assert atlas.interface_dofs.tolist() == [10]
    assert atlas.retained_internal_modes == 2
    assert atlas.reduced_size == 3
    assert atlas.discarded_flexibility_bound == 0.0
    np.testing.assert_allclose(atlas.transformation.T @ k @ atlas.transformation,
                               atlas.reduced_stiffness)
    np.testing.assert_allclose(atlas.transformation.T @ m @ atlas.transformation,
                               atlas.reduced_mass)
    # A full atlas spans the original component exactly.
    physical = np.array([0.2, -0.1, 0.35])
    reduced = np.linalg.solve(atlas.transformation, physical)
    np.testing.assert_allclose(atlas.recover(reduced), physical)


def test_truncation_keeps_interface_and_static_constraint_shape_exact():
    k, m = _three_dof_component()
    atlas = build_component_mode_atlas(
        "witness", k, m, np.array([10, 11, 12]), np.array([0]),
        internal_mode_count=0)

    assert atlas.reduced_size == 1
    assert atlas.discarded_flexibility_bound > 0.0
    displacement = atlas.recover(np.array([0.125]))
    assert displacement[0] == pytest.approx(0.125)
    # Internal equilibrium under a prescribed interface displacement is the
    # defining Craig--Bampton constraint mode: K_ii u_i + K_ib u_b = 0.
    np.testing.assert_allclose(k[1:, 1:] @ displacement[1:]
                               + k[1:, :1] @ displacement[:1], 0.0,
                               atol=1.0e-12)


def test_internal_mechanism_cannot_be_hidden_inside_component_atlas():
    k = np.diag([10.0, 0.0])
    m = np.eye(2)
    with pytest.raises(ValueError, match="explicit interface freedom"):
        build_component_mode_atlas(
            "bad-piece", k, m, np.array([0, 1]), np.array([0]))


def test_very_soft_positive_internal_mode_is_not_classified_as_a_mechanism():
    k = np.diag([10.0, 1.0e-10])
    atlas = build_component_mode_atlas(
        "soft-piece", k, np.eye(2), np.array([0, 1]), np.array([0]))
    assert atlas.retained_internal_modes == 1
    assert atlas.fixed_interface_eigenvalues[0] == pytest.approx(1.0e-10)


def test_two_component_atlases_assemble_exactly_on_one_shared_interface():
    spring = np.array([[10.0, -10.0], [-10.0, 10.0]])
    left = build_component_mode_atlas(
        "left", spring, np.diag([2.0, 1.5]), np.array([0, 1]),
        np.array([1]))
    right = build_component_mode_atlas(
        "right", 2.0 * spring, np.diag([1.5, 3.0]), np.array([1, 2]),
        np.array([0]))
    assembled = assemble_component_mode_atlases([left, right])

    assert assembled.interface_dofs.tolist() == [1]
    assert assembled.reduced_size == 3
    full_k = np.array([[10.0, -10.0, 0.0],
                       [-10.0, 30.0, -20.0],
                       [0.0, -20.0, 20.0]])
    # Shared interface mass is contributed by each component's element/body
    # partition. This test deliberately assigns half to each piece.
    full_m = np.diag([2.0, 3.0, 3.0])
    t = assembled.transformation
    np.testing.assert_allclose(assembled.reduced_stiffness, t.T @ full_k @ t,
                               atol=1.0e-12)
    np.testing.assert_allclose(assembled.reduced_mass, t.T @ full_m @ t,
                               atol=1.0e-12)
    physical = np.array([0.3, -0.2, 0.4])
    reduced = np.linalg.solve(t, physical)
    np.testing.assert_allclose(assembled.recover(reduced), physical)


def test_rigid_gestalt_preserves_points_mass_and_resultant_wrench():
    positions = np.array([[0.0, 0.0, 0.0],
                          [2.0, 0.0, 0.0]])
    # Translational point masses plus small positive rotary inertias.
    mass = np.diag([2.0, 2.0, 2.0, .1, .1, .1,
                    3.0, 3.0, 3.0, .2, .2, .2])
    atlas = build_rigid_gestalt_atlas(
        "rigid-witness", np.zeros((12, 12)), mass, np.arange(12),
        positions, np.array([1.0, 0.0, 0.0]))

    recovered = atlas.recover(np.array([.1, .2, .3, 0.0, 0.0, .01]))
    # Rotation about z gives equal and opposite y motion around the centre.
    np.testing.assert_allclose(recovered[[1, 7]], [.19, .21])
    np.testing.assert_allclose(np.diag(atlas.reduced_mass)[:3], [5., 5., 5.])
    load = np.zeros(12)
    load[7] = 10.0  # +y at x=+1 relative to the gestalt centre.
    np.testing.assert_allclose(atlas.project_load(load),
                               [0., 10., 0., 0., 0., 10.])


def test_adaptive_gestalt_requires_solved_quiet_dwell_and_wakes_from_bound():
    envelope = GestaltStressEnvelope(
        influence_pa_per_wrench=np.array([[2.0, 0.0], [0.0, 3.0]]),
        residual_bound_pa=5.0)
    policy = AdaptiveGestaltPolicy(
        sleep_stress_bound_pa=20.0, wake_stress_bound_pa=50.0,
        sleep_internal_kinetic_bound_j=.1, sleep_dwell_s=0.5,
        sleep_wrench_rate_limit_per_s=4.0)
    state = AdaptiveGestaltState("adaptive-witness", policy, envelope)

    # A dormant estimate alone may never justify turning the elastic solve off.
    assert state.observe(.5, np.zeros(2)) == "active"
    assert state.elastic
    # Actual local stress, its error bound, and a quiet load history do.
    assert state.observe(.25, np.zeros(2), solved_peak_stress_pa=10.0,
                         solved_stress_error_bound_pa=2.0,
                         solved_internal_kinetic_j=.01,
                         interface_wrench_rate_per_s=1.0) == "active"
    assert state.observe(.25, np.zeros(2), solved_peak_stress_pa=10.0,
                         solved_stress_error_bound_pa=2.0,
                         solved_internal_kinetic_j=.01,
                         interface_wrench_rate_per_s=1.0) == "slept"
    assert not state.elastic
    assert state.observe(.01, np.zeros(2)) == "dormant"
    # 3 * 16 + 5 = 53 Pa: conservative envelope crosses wake hysteresis.
    assert state.observe(.01, np.array([0.0, 16.0])) == "woke"
    assert state.elastic


def test_adaptive_gestalt_policy_has_no_implicit_gameplay_threshold():
    with pytest.raises(TypeError):
        AdaptiveGestaltPolicy()  # every physical threshold is caller-owned
    with pytest.raises(ValueError, match="hysteresis"):
        AdaptiveGestaltPolicy(10.0, 10.0, .1, 1.0)


def test_adaptive_gestalt_cannot_sleep_at_zero_strain_high_speed_crossing():
    state = AdaptiveGestaltState(
        "ringing-witness",
        AdaptiveGestaltPolicy(20.0, 50.0, .1, .25),
        GestaltStressEnvelope(np.zeros((1, 1)), 0.0))
    assert state.observe(.25, np.zeros(1), solved_peak_stress_pa=0.0,
                         solved_internal_kinetic_j=10.0) == "active"
    assert state.elastic


def _adaptive_runtime_witness():
    positions = np.array([[-1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0]])
    mass = np.diag([2.0, 2.0, 2.0, .1, .1, .1,
                    3.0, 3.0, 3.0, .2, .2, .2])
    stiffness = np.zeros((12, 12))
    elastic = build_component_mode_atlas(
        "adaptive-runtime", stiffness, mass, np.arange(12), np.arange(12))
    rigid = build_rigid_gestalt_atlas(
        "adaptive-runtime", stiffness, mass, np.arange(12), positions,
        np.zeros(3))
    state = AdaptiveGestaltState(
        "adaptive-runtime",
        AdaptiveGestaltPolicy(20.0, 50.0, 1.0, .1),
        GestaltStressEnvelope(np.array([[10.0]]), 0.0))
    return AdaptiveGestaltRuntime(elastic, rigid, state, mass)


def test_adaptive_runtime_transition_preserves_rigid_pose_and_momentum():
    runtime = _adaptive_runtime_witness()
    rigid_q = np.array([.1, .2, .3, .01, -.02, .03])
    rigid_v = np.array([1., 2., 3., .1, .2, .3])
    physical_u = runtime.rigid_atlas.transformation @ rigid_q
    physical_v = runtime.rigid_atlas.transformation @ rigid_v
    runtime.set_physical_state(physical_u, physical_v)
    momentum_before = (runtime.rigid_atlas.transformation.T
                       @ runtime.physical_mass @ physical_v)

    assert runtime.observe(.1, np.zeros(1), solved_peak_stress_pa=1.0,
                           solved_internal_kinetic_j=0.0) == "slept"
    slept_u, slept_v, _ = runtime.physical_state()
    np.testing.assert_allclose(slept_u, physical_u, atol=1e-12)
    np.testing.assert_allclose(slept_v, physical_v, atol=1e-12)
    np.testing.assert_allclose(
        runtime.rigid_atlas.transformation.T @ runtime.physical_mass @ slept_v,
        momentum_before, atol=1e-12)

    assert runtime.observe(.01, np.array([6.0])) == "woke"
    woke_u, woke_v, _ = runtime.physical_state()
    np.testing.assert_allclose(woke_u, slept_u, atol=1e-12)
    np.testing.assert_allclose(woke_v, slept_v, atol=1e-12)


def test_adaptive_runtime_measures_internal_motion_before_sleeping():
    runtime = _adaptive_runtime_witness()
    velocity = np.zeros(12)
    velocity[0], velocity[6] = -1.0, 1.0
    runtime.set_physical_state(np.zeros(12), velocity)

    assert runtime.observe(.1, np.zeros(1), solved_peak_stress_pa=0.0,
                           solved_internal_kinetic_j=0.0) == "active"
    assert runtime.residency.elastic
