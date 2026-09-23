import math

import numpy as np
import pytest

from thermal_domains import ThermalEngine, ThermalSystem, build_thermal_assembly


def _helix(turns: float, *, phase: float, reverse: bool = False,
           samples: int = 49) -> list[list[float]]:
    u = np.linspace(0.0, 1.0, samples)
    angle = 2.0 * math.pi * turns * u + phase
    points = np.column_stack((
        0.050 * np.cos(angle),
        0.30 * u,
        0.050 * np.sin(angle),
    ))
    if reverse:
        points = points[::-1]
    return points.tolist()


def _graph(turns=5.0, coefficient=90.0):
    radius = 0.004
    area = math.pi * radius * radius
    return {
        "nodes": [
            {
                "identity": "recuperator.high", "kind": "coiled-passage",
                "thermal_domain": "path", "material": "copper",
                "thermal_path_points_m": _helix(turns, phase=0.0),
                "thermal_cross_section_area_m2": area,
                "thermal_group": "hot",
            },
            {
                "identity": "recuperator.return", "kind": "coiled-passage",
                "thermal_domain": "path", "material": "copper",
                "thermal_path_points_m": _helix(
                    turns, phase=math.pi / 8.0, reverse=True),
                "thermal_cross_section_area_m2": area,
                "thermal_group": "cold",
            },
        ],
        "edges": [{
            "identity": "recuperator.exchange",
            "a": "recuperator.high", "b": "recuperator.return",
            "constraint": "thermal-interface",
            "thermal_exchange": "distributed-path",
            "heat_transfer_coefficient_w_m2_k": coefficient,
            "thermal_exchange_perimeter_m": 2.0 * math.pi * radius,
        }],
    }


def test_coiled_counterflow_exchange_is_geometry_matched_and_conservative():
    assembly = build_thermal_assembly(
        _graph(), temperatures_by_group={"hot": 360.0, "cold": 260.0})
    interface = assembly.interfaces[0]

    # The return path was authored in counterflow order.  Geometry must map
    # its last sample to the high-side first sample, not match array indices.
    assert np.argmax(interface.conductance_w_k[0]) == 48

    before = assembly.energy_j
    hot_before = assembly.temperature_k("recuperator.high")
    cold_before = assembly.temperature_k("recuperator.return")
    assembly.step(2.0)
    assert assembly.energy_j == pytest.approx(before, rel=1e-12)
    assert assembly.temperature_k("recuperator.high") < hot_before
    assert assembly.temperature_k("recuperator.return") > cold_before


def test_exchange_coefficient_is_a_calibration_knob_with_linear_initial_power():
    low = build_thermal_assembly(
        _graph(coefficient=50.0),
        temperatures_by_group={"hot": 350.0, "cold": 250.0})
    high = build_thermal_assembly(
        _graph(coefficient=100.0),
        temperatures_by_group={"hot": 350.0, "cold": 250.0})
    low_before = low.states["recuperator.high"].energy_j
    high_before = high.states["recuperator.high"].energy_j
    low.step(1.0e-3)
    high.step(1.0e-3)
    low_loss = low_before - low.states["recuperator.high"].energy_j
    high_loss = high_before - high.states["recuperator.high"].energy_j
    assert high_loss == pytest.approx(2.0 * low_loss, rel=2e-4)


def test_more_coil_length_produces_more_declared_exchange_area():
    short = build_thermal_assembly(_graph(turns=2.0))
    long = build_thermal_assembly(_graph(turns=8.0))
    short_g = float(short.interfaces[0].conductance_w_k.sum())
    long_g = float(long.interfaces[0].conductance_w_k.sum())
    assert long_g > short_g


def test_thermal_engine_is_one_dt_participant_for_the_whole_batch():
    first = build_thermal_assembly(
        _graph(), temperatures_by_group={"hot": 350.0, "cold": 250.0})
    second = build_thermal_assembly(
        _graph(), temperatures_by_group={"hot": 330.0, "cold": 270.0})
    engine = ThermalEngine(
        (first, second),
        source_w=({"recuperator.high": 25.0},
                  {"recuperator.return": -10.0}),
    )
    dt = min(1.0e-3, engine.preferred_dt())

    ok, metrics, state = engine.step(dt)

    assert ok
    assert state is None
    assert metrics.pub_exchange_time.shape == (1,)
    assert metrics.pub_dt_limit.shape == (1,)
    assert metrics.pub_values.shape == metrics.error_channels.shape
    assert metrics.pub_exchange_time_present.tolist() == [1.0]
    assert metrics.advanced_dt == pytest.approx(dt)
    assert engine.last_conservation_error_j == pytest.approx(0.0, abs=1e-6)


def test_empty_thermal_system_has_zero_participant_extent_until_registered():
    system = ThermalSystem()
    ok, metrics, _state = system.step(0.01)
    assert ok
    assert system.active_count == 0
    assert metrics.pub_exchange_time.shape == (0,)

    system.register_assembly(build_thermal_assembly(_graph()))
    assert system.active_count == 1


def test_thermal_engine_rollback_restores_every_temperature_column():
    assembly = build_thermal_assembly(
        _graph(), temperatures_by_group={"hot": 350.0, "cold": 250.0})
    engine = ThermalEngine(
        assembly, source_w={"recuperator.high": 100.0})
    before = {identity: state.temperature_k.copy()
              for identity, state in assembly.states.items()}
    snapshot = engine.snapshot()

    engine.step(min(0.01, engine.preferred_dt()))
    assert any(not np.array_equal(state.temperature_k, before[identity])
               for identity, state in assembly.states.items())

    engine.restore(snapshot)
    assert assembly.elapsed_s == 0.0
    for identity, state in assembly.states.items():
        np.testing.assert_array_equal(state.temperature_k, before[identity])
