import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from automatic_transmission import ATF_OXIDATION_K, AutomaticTransmission


@pytest.mark.parametrize("over", [10.0, 100.0])
def test_oxidation_preserves_temperature_doubling_law(over):
    transmission = AutomaticTransmission(cooler_ua_w_per_k=0.0)
    transmission.temp_k = ATF_OXIDATION_K + over
    transmission.life_frac = 0.75
    transmission.step(0.005, 0.0, 0.0, 0.0)
    assert transmission.life_frac == pytest.approx(
        0.75 - 2.0 ** (over / 10.0) * 0.005 / 3.0e5)


def test_extreme_heat_exhausts_fluid_life_without_overflow_or_recovery():
    transmission = AutomaticTransmission(cooler_ua_w_per_k=0.0)
    transmission.temp_k = 100_000.0
    for _ in range(2):
        transmission.step(0.005, 0.0, 0.0, 0.0)
        assert transmission.life_frac == 0.0
        assert math.isfinite(transmission.temp_k)
    transmission.temp_k = 293.15
    transmission.step(0.005, 0.0, 0.0, 0.0)
    assert transmission.life_frac == 0.0


def test_tiny_timestep_can_keep_damage_finite_when_rate_would_overflow():
    transmission = AutomaticTransmission(cooler_ua_w_per_k=0.0)
    transmission.temp_k = ATF_OXIDATION_K + 10_240.0
    transmission.step(1e-310, 0.0, 0.0, 0.0)
    expected_damage = math.ldexp(1e-310, 1023) * 2.0 / 3.0e5
    assert transmission.life_frac == pytest.approx(1.0 - expected_damage)
