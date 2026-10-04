"""Real library integration and native physics publications, one coupled rotor."""

import numpy as np
import pytest

from honorary_engine_equation_catalogue import equation_piece
from orbital_actuation import AXES, attitude_symbols
from orbital_craft_machine import rotational_step_equations
from orbital_jumper import declare_binding
from llvm_dt_system import advance_round, column_names_of, instantiate_system, piece_leaf
from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_graph import ControllerNode, RoundNode
from src.common.dt_system.error_channels import DT_CHANNEL_NAMES, channel_fields


@pytest.mark.parametrize("part", ("stage0", "stage3", "metrics", "diagnostics", "commit",
                                  "properties_after", "inverse_after", "translation_energy",
                                  "rotation_energy", "wheel_energy"))
def test_production_piece_endpoint_and_exchange(part):
    """One production artifact with analytic inputs and a runtime-only handle.

    The compiler archives stay on disk; the handle uses the game's existing
    optional retention mode. This verifies endpoint laws, not a complete flight.
    The controlled endpoint is constant-velocity mass outflow: dp = v dm
    and dK = v^2 dm/2, so its conservation defect is exactly zero.
    """
    import os
    from orbital_actuation import thruster_columns
    from orbital_craft_machine import craft_machine_columns, craft_machine_equations, orbital_craft

    print(f"Native production part={part} pid={os.getpid()}", flush=True)
    design = orbital_craft()
    name, equations = next((name, equations) for name, equations in craft_machine_equations(design, 0)
                           if name.endswith("_c0_" + part))
    piece = equation_piece(name, equations, retain_compilation=False)
    # Scratch values are explicit test inputs. Physical parameters come
    # from the same two declarations used by the real constructor; a
    # missing physical input must fail instead of acquiring an implicit 0.
    values = {name: 0.0 for name in piece.argument_names if name.startswith("rk_stage")}
    values.update({name: float(value[0]) for name, value in thruster_columns(design).items()})
    values.update({name: float(value[0]) for name, value in craft_machine_columns(design).items()})
    dt, flow = 0.25, 0.1
    velocity = np.array((3.0, -4.0, 2.0))
    mass = design.mass_kg
    values.update(dt=dt, mass=mass, dry_mass=design.dry_mass_kg,
                  propellant_mass=design.propellant_kg,
                  translation_stored=mass * (velocity @ velocity) / 2,
                  rotation_stored=0.8, wheel_stored=1.2,
                  translation_power=1.45, rotation_power=0.02, wheel_power=0.04)
    values.update({name: 0.0 for name in (
        "fuel_impulse", "translation_work", "rotation_work", "wheel_energy", "wheel_copper_loss",
        "translation_exchange", "rotation_exchange", "wheel_exchange", "propellant_flow")})
    values["propellant_supply"] = 1.0
    values.update({symbol.name: float(row == col)
                   for (row, col), symbol in attitude_symbols().items()})
    for index, axis in enumerate(AXES):
        for prefix in ("position", "angular_velocity", "raw_force", "applied_force", "torque",
                       "angular_impulse_world", "applied_impulse_world", "applied_delta_v_world",
                       "torque_impulse_body"):
            values[f"{prefix}_{axis}"] = 0.0
        values[f"momentum_{axis}"] = mass * velocity[index]
    for stage in range(4):
        prefix = f"rk_stage{stage}_"
        values[prefix + "tank0_propellant_rate"] = -flow
        values[prefix + "tank0_demand_rate"] = flow
        values[prefix + "translation_work_rate"] = -flow * (velocity @ velocity) / 2
        values[prefix + "translation_exchange_rate"] = flow * (velocity @ velocity) / 2
        for index, axis in enumerate(AXES):
            values[prefix + f"momentum_{axis}_rate"] = -flow * velocity[index]
            values[prefix + f"position_{axis}_rate"] = velocity[index]

    def run():
        assert not (set(piece.argument_names) - values.keys())
        return {name: float(np.asarray(value).reshape(-1)[0])
                for name, value in zip(piece.output_names, piece(*(
                    np.array([values[name]]) for name in piece.argument_names)))}

    if part == "stage0":
        main = design.thrusters_by_role("main")[0]
        key = f"thruster{main}"
        values.update({key + "_throttle": 1.0, key + "_throttle_state": 0.0,
                       key + "_throttle_slew": 2.0})
        assert run()[f"rk_stage0_{key}_delivered_value_next"] == 0.0
        values[key + "_throttle_slew"] = float("inf")
        assert run()[f"rk_stage0_{key}_delivered_value_next"] == 1.0
        values.update({key + "_throttle": 0.0, key + "_throttle_state": 1.0})
        assert run()[f"rk_stage0_{key}_delivered_value_next"] == 0.0
        gimbal = next(k for k, thruster in enumerate(design.thrusters) if thruster.gimballed)
        key = f"thruster{gimbal}"
        values.update({key + "_gimbal_a": 0.0, key + "_gimbal_a_command": 0.05,
                       key + "_gimbal_slew": float("inf")})
        assert run()[f"rk_stage0_{key}_gimbal_a_value_next"] == 0.05
    elif part == "stage3":
        for name, value in tuple(values.items()):
            if not name.startswith("rk_stage"):
                values[f"rk_stage0_{name}_value"] = value
        values["rk_stage0_propellant_flow_value"] = flow
        values["rk_stage0_tank0_flow_value"] = flow
        omega = np.array((0.1, 0.2, -0.3))
        for index, axis in enumerate(AXES):
            values[f"angular_velocity_{axis}"] = omega[index]
        result = run()
        assert result["rk_stage0_translation_exchange_rate_next"] == pytest.approx(1.45)
        assert result["rk_stage0_translation_work_rate_next"] == pytest.approx(-1.45)
        rotation_flux = result["rk_stage0_rotation_work_rate_next"]
        assert rotation_flux < 0.0
        assert result["rk_stage0_rotation_exchange_rate_next"] == pytest.approx(abs(rotation_flux))
        for w, command in enumerate((0.6, -0.4, 0.0)):
            wheel = design.wheels[w]
            values.update({f"wheel{w}_rotor_inertia": 1.0,
                           f"wheel{w}_momentum": 10.0 + np.asarray(wheel.axis) @ omega,
                           f"wheel{w}_torque_command": command, f"wheel{w}_max_torque": 1.0,
                           f"wheel{w}_torque_constant": 1.0, f"wheel{w}_winding_resistance": 0.2})
        result = run()
        assert result["rk_stage0_wheel_energy_rate_next"] == pytest.approx(2.104)
        assert result["rk_stage0_wheel_copper_loss_rate_next"] == pytest.approx(0.104)
        assert result["rk_stage0_wheel_exchange_rate_next"] == pytest.approx(10.04)
    elif part == "metrics":
        result = run()
        for name, value in result.items():
            if name.startswith("orbital_"):
                assert abs(value) < 1e-8, (name, value)
        assert result["translation_stored_next"] == pytest.approx((mass - flow * dt) * 29 / 2)
        assert result["max_vel"] == pytest.approx(np.sqrt(29))
    elif part == "diagnostics":
        result = run()
        assert result["tank0_flow_next"] == pytest.approx(flow)
        assert result["tank0_supply_next"] == pytest.approx(1.0)
        assert result["propellant_flow_next"] == pytest.approx(flow)
        assert result["translation_power_next"] == pytest.approx(1.45)
        assert result["applied_force_x_next"] == 0.0
    elif part == "commit":
        result = run()
        assert result["mass_next"] == pytest.approx(mass - flow * dt)
        assert result["tank0_propellant_next"] == pytest.approx(values["tank0_propellant"] - flow * dt)
        for index, axis in enumerate(AXES):
            assert result[f"position_{axis}_next"] == pytest.approx(velocity[index] * dt)
            assert result[f"momentum_{axis}_next"] == pytest.approx((mass - flow * dt) * velocity[index])
        assert result["translation_work_next"] == pytest.approx(-1.45 * dt)
    elif part in ("properties_after", "inverse_after"):
        charges = {tank.identity: tank.fill_kg for tank in design.tanks}
        charges[design.tanks[0].identity] -= flow * dt
        properties = design.mass_properties(charges)
        expected_inertia = np.asarray(properties.inertia_tensor_kg_m2)
        values["tank0_propellant"] -= flow * dt
        for i, axis in enumerate(AXES):
            for j, other in enumerate(AXES[i:], i):
                values[f"inertia_{axis}{other}"] = expected_inertia[i, j]
        result = run()
        expected = (expected_inertia if part == "properties_after"
                    else np.linalg.inv(design.spin_free_inertia(charges)))
        prefix = "inertia" if part == "properties_after" else "inverse_inertia"
        for i, axis in enumerate(AXES):
            for j, other in enumerate(AXES[i:], i):
                assert result[f"{prefix}_{axis}{other}_next"] == pytest.approx(
                    expected[i, j], rel=1e-11, abs=1e-10)
        if part == "properties_after":
            np.testing.assert_allclose(
                [result[f"centre_of_mass_{axis}_next"] for axis in AXES],
                properties.center_of_gravity, rtol=0, atol=1e-12)
        else:
            inverse = np.array([[result[f"inverse_inertia_{''.join(sorted((a, b)))}_next"]
                                 for b in AXES] for a in AXES])
            np.testing.assert_allclose(inverse @ design.spin_free_inertia(charges),
                                       np.eye(3), rtol=0, atol=1e-12)
    else:
        store = part.removesuffix("_energy")
        result = run()
        assert result["energy_j"] == values[f"{store}_stored"]
        assert result["power_w"] == values[f"{store}_power"]
        assert result["exchangeable_energy_j"] == float("inf")
    print(f"Verified native production part={part}", flush=True)


def test_library_rk4_couples_rotor_body_and_attitude_with_native_conservation(tmp_path):
    stages, commit = rotational_step_equations(1)
    pieces = []
    for index, equations in enumerate((*stages, commit)):
        print(f"Compiling coupled rotor piece {index + 1}/{len(stages) + 1}", flush=True)
        pieces.append(equation_piece(f"orbital_library_rotor_gate_{index}",
                                    equations, cache_dir=tmp_path, retain_compilation=False))
    declare_binding(pieces)
    names = (*DT_CHANNEL_NAMES, "orbital_rotation_energy_residual_j",
             "orbital_angular_momentum_residual_n_m_s",
             "orbital_attitude_orthogonality", "orbital_wheel_speed_excess_rad_s")
    limits = dict(zip(names[len(DT_CHANNEL_NAMES):], (1e-8, 1e-8, 1e-9, 1e-8)))
    targets = Targets(1.0, 1.0, 1.0, **channel_fields(limits, names=names, limits=True))
    columns = {name: np.zeros(1) for name in column_names_of(pieces)}
    values = {"inertia_xx": 2.0, "inertia_yy": 3.0, "inertia_zz": 4.2,
              "inverse_inertia_xx": 0.5, "inverse_inertia_yy": 1/3,
              "inverse_inertia_zz": 0.25, "wheel0_rotor_inertia": 0.2,
              "wheel0_axis_z": 1.0, "wheel0_max_speed": 600.0,
              "wheel0_max_torque": 1.0, "wheel0_torque_command": 0.06,
              "wheel0_torque_constant": 0.4, "wheel0_winding_resistance": 0.2,
              "angular_velocity_x": 0.1, "angular_velocity_y": -0.2,
              "angular_velocity_z": 0.15}
    values.update({symbol.name: float(r == c)
                   for (r, c), symbol in attitude_symbols().items()})
    for name, value in values.items():
        columns[name][...] = value
    graph = RoundNode(
        plan=SuperstepPlan(round_max=2.0, dt_init=2.0),
        controller=ControllerNode(ctrl=STController(dt_min=None), targets=targets, dx=1.0),
        children=[piece_leaf(piece) for piece in pieces], schedule="sequential")
    state = instantiate_system(graph, columns, channel_names=names)
    owned = {name: getattr(state, name) for name in columns}
    first = {name: value.copy() for name, value in owned.items()}
    attempts = []
    native_advance = state.program["advance_pieces"]

    def observe(actual, dt):
        before = actual.wheel0_momentum.copy()
        result = native_advance(actual, dt)
        errors = np.asarray(actual.pub_values).reshape(len(pieces), len(names))[-1, -4:].copy()
        attempts.append((float(dt), before, errors))
        return result

    state.program["advance_pieces"] = observe
    initial_h = np.diag((2.0, 3.0, 4.0)) @ np.array((0.1, -0.2, 0.15))
    for command in (0.06, -0.04):
        state.wheel0_torque_command[...] = command
        advanced, _, _ = advance_round(state, 2.0)
        assert advanced == 2.0
    rotation = np.array([[float(getattr(state, attitude_symbols()[r, c].name)[0])
                          for c in range(3)] for r in range(3)])
    omega = np.array([float(getattr(state, f"angular_velocity_{a}")[0]) for a in AXES])
    final_h = rotation @ (np.diag((2.0, 3.0, 4.0)) @ omega
                          + np.array((0.0, 0.0, float(state.wheel0_momentum[0]))))
    print(f"Coupled rotor: {len(attempts)} attempts, first dt={attempts[0][0]}, "
          f"first errors={attempts[0][2]}, |H-H0|={np.linalg.norm(final_h-initial_h):.12g}",
          flush=True)
    assert attempts[0][0] == 2.0
    assert np.any(attempts[0][2] > np.array(tuple(limits.values())))
    np.testing.assert_array_equal(attempts[1][1], first["wheel0_momentum"])
    assert np.linalg.norm(final_h - initial_h) < 1e-6
    assert float(state.wheel_energy[0]) > 0
    assert float(state.wheel_copper_loss[0]) == pytest.approx(0.013, abs=1e-12)
    assert abs(float(state.wheel_energy[0] - state.wheel_copper_loss[0]
                     - state.rotation_work[0])) < 1e-12
    assert abs(float(state.wheel0_momentum[0]) - 0.04) < 1e-12
    assert all(getattr(state, name) is span for name, span in owned.items())


def test_production_craft_declares_energy_and_refines_command_changes():
    """Actual machine stages, commands, registered error meters and ledgers."""
    from orbital_craft_machine import MachineCraft, orbital_craft
    from orbital_jumper import ORBITAL_CHANNEL_NAMES, ORBITAL_ERROR_LIMITS
    from time import perf_counter

    started = perf_counter()
    print("production craft: construction begin", flush=True)
    design = orbital_craft()
    craft = MachineCraft([], design, position_m=(0.0, 0.0, 0.0),
                         velocity_m_s=(0.0, 0.0, 0.0), length_scale_m=1e6,
                         window_s=2.0, angular_velocity_rad_s=(0.01, -0.02, 0.015))
    print(f"production craft: construction complete in {perf_counter()-started:.3f}s", flush=True)
    state = craft.dt_state
    owned = {name: getattr(state, name) for name in column_names_of(craft.pieces)}
    native = state.program["advance_pieces"]
    attempts = []
    channel_count = len(ORBITAL_CHANNEL_NAMES)
    limits = np.array(tuple(ORBITAL_ERROR_LIMITS.values()))
    phase = "wheel ignition"

    def observe(actual, dt):
        result = native(actual, dt)
        errors = np.asarray(actual.pub_values).reshape(len(craft.pieces), channel_count)
        declared = errors[:, -len(limits):].max(axis=0).copy()
        attempts.append((phase, float(dt), declared))
        if len(attempts) % 500 == 0:
            ratios = declared / limits
            governing = int(np.argmax(ratios))
            print(f"production attempt={len(attempts)} phase={phase} dt={dt:.12g} "
                  f"governing={tuple(ORBITAL_ERROR_LIMITS)[governing]} "
                  f"ratio={ratios[governing]:.12g} ratios={ratios} "
                  f"elapsed={perf_counter()-started:.3f}s", flush=True)
        return result

    state.program["advance_pieces"] = observe

    def advance_phase(label):
        nonlocal phase
        phase = label
        print(f"production {phase}: begin t={craft.time_s:.12g}", flush=True)
        craft.advance()
        print(f"production {phase}: end t={craft.time_s:.12g}, "
              f"attempts={len(attempts)}, elapsed={perf_counter()-started:.3f}s", flush=True)

    h0 = craft.angular_momentum().copy()
    tanks0 = craft.tank_propellant_kg()
    craft.wheel_torque((0.6, -0.4, 0.9))
    advance_phase("wheel ignition")
    craft.wheel_torque((-0.9, 0.7, -0.5))
    advance_phase("wheel reversal")
    assert np.linalg.norm(craft.angular_momentum() - h0) < 1e-5
    assert craft.tank_propellant_kg() == tanks0
    assert craft.wheel_copper_loss_j > 0
    assert abs(craft.wheel_energy_j - craft.wheel_copper_loss_j
               - craft._scalar("rotation_work")) < 1e-10

    main = design.thrusters_by_role("main")[0]
    throttle = np.zeros(design.thruster_count)
    throttle[main] = 1.0
    craft.throttle(throttle)
    craft.wheel_torque((0.0, 0.0, 0.0))
    advance_phase("main ignition")
    craft.throttle(np.zeros(design.thruster_count))
    advance_phase("main cutoff")
    after_cutoff = craft.tank_propellant_kg()
    impulse = craft.thruster_impulses_n_s.copy()
    advance_phase("cold coast")
    np.testing.assert_array_equal(craft.thruster_impulses_n_s, impulse)
    assert craft.tank_propellant_kg() == after_cutoff
    burned = sum(tanks0[name] - after_cutoff[name] for name in tanks0)
    assert abs(craft.mass_kg - (design.mass_kg - burned)) < 1e-10
    assert abs(burned - impulse[main]
               / design.thrusters[main].thruster_kind.exhaust_velocity_m_s) < 1e-10
    np.testing.assert_allclose(craft.applied_delta_v_m_s, craft.r()[1], rtol=0, atol=1e-10)
    assert all(getattr(state, name) is span for name, span in owned.items())
    for label in dict.fromkeys(row[0] for row in attempts):
        rows = [(dt, errors) for phase_name, dt, errors in attempts if phase_name == label]
        rejected = sum(bool(np.any(errors > limits)) for _, errors in rows)
        print(f"{label}: attempts={len(rows)}, error-rejected={rejected}, "
              f"dt=[{min(dt for dt, _ in rows):.9g},{max(dt for dt, _ in rows):.9g}], "
              f"first={rows[0][1]}, last={rows[-1][1]}", flush=True)
        assert np.all(rows[-1][1] <= limits)
    print(f"production craft: mass={craft.mass_kg:.12g}, burned={burned:.12g}, "
          f"electrical={craft.wheel_energy_j:.12g}, copper={craft.wheel_copper_loss_j:.12g}, "
          f"orthogonality={np.max(np.abs(craft.attitude().T @ craft.attitude()-np.eye(3))):.12g}",
          flush=True)
