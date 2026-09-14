from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine_sound


def test_turbine_whine_compression_is_default_on_and_frequency_keyed(monkeypatch):
    knee = engine_sound.TURBINE_WHINE_COMPRESSION_START_HZ
    assert engine_sound.COMPRESS_HIGH_FREQUENCY_TURBINE_WHINE is True
    assert engine_sound.turbine_whine_loudness_gain(knee) == 1.0
    assert 0.0 < engine_sound.turbine_whine_loudness_gain(knee * 2.0) < 1.0
    assert (engine_sound.turbine_whine_loudness_gain(knee * 4.0)
            < engine_sound.turbine_whine_loudness_gain(knee * 2.0))

    monkeypatch.setattr(
        engine_sound, "COMPRESS_HIGH_FREQUENCY_TURBINE_WHINE", False)
    assert engine_sound.turbine_whine_loudness_gain(knee * 100.0) == 1.0


def test_both_turbine_whine_synthesis_paths_use_the_compressor(monkeypatch):
    frequencies = []

    def record(frequency):
        frequencies.append(float(frequency))
        return 1.0

    monkeypatch.setattr(engine_sound, "turbine_whine_loudness_gain", record)
    synth = engine_sound.EngineSoundSynth(sample_rate=44_100)
    turbine = SimpleNamespace(
        kind="turbine", forced_induction=SimpleNamespace(kind="none"))

    synth._render_header_turbine(
        turbine, n1_frac=1.0, throttle=1.0, load_frac=1.0, n_frames=16)
    synth._render_induction(
        turbine, n_frames=16, turbo_spool_frac=1.0,
        wastegate_flutter=False, surge_active=False)

    assert frequencies[:3] == pytest.approx((3420.0, 6840.0, 10260.0))
    assert frequencies[3] == pytest.approx(6500.0)
