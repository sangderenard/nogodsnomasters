"""The authored ramp's integral is independent of a dt substep partition."""
import math

import numpy as np
import pytest

from orbital_actuation import Thruster, throttle_delivery


def _thruster(index, rate):
    return Thruster(str(index), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 4000.0,
                    throttle_slew_per_s=rate, deadband=0.4)


def test_native_ramp_area_crosses_deadband_and_clamps_commands():
    # All lanes ask for one second. These integrals are triangle/trapezoid
    # areas above the authored 0.4 deadband, plus any steady plateau.
    thrusters = tuple(_thruster(i, rate) for i, rate in enumerate(
        (2.0, 2.0, 0.2, 0.4, 2.0, 2.0, math.inf, 2.0, 2.0)))
    areas, endpoints = throttle_delivery(
        thrusters, [0.0, 1.0, 0.8, 0.2, 0.2, 0.8, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.2, 1.0, 0.3, 0.8, 1.0, 2.0, -2.0], 1.0)
    assert areas == pytest.approx(
        [0.71, 0.21, 0.7, 0.25, 0.0, 0.8, 1.0, 0.71, 0.21], abs=1e-14)
    assert endpoints == pytest.approx(
        [1.0, 0.0, 0.6, 0.6, 0.3, 0.8, 1.0, 1.0, 0.0], abs=1e-14)


def test_native_area_is_additive_across_frames_and_command_reversal():
    thrusters = (_thruster(0, 2.0),)
    whole, end = throttle_delivery(thrusters, [0.0], [1.0], 1.0)
    total = np.zeros(1)
    state = np.zeros(1)
    # Splits include both sides of the deadband crossing and ramp endpoint.
    for duration in (0.13, 0.17, 0.23, 0.47):
        area, state = throttle_delivery(thrusters, state, [1.0], duration)
        total += area
    assert total == pytest.approx(whole, abs=1e-14)
    assert state == pytest.approx(end, abs=1e-14)

    down, mid = throttle_delivery(thrusters, [1.0], [0.0], 0.1)
    up, end = throttle_delivery(thrusters, mid, [1.0], 1.0)
    assert down == pytest.approx([0.09], abs=1e-14)
    assert up == pytest.approx([0.99], abs=1e-14)
    assert end == pytest.approx([1.0], abs=1e-14)


def test_zero_duration_keeps_state_even_for_instant_actuator():
    areas, endpoints = throttle_delivery((_thruster(0, math.inf),),
                                          [0.3], [1.0], 0.0)
    assert areas.tolist() == [0.0]
    assert endpoints.tolist() == [0.3]
