import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gun_rig import GunRig


def test_compiled_breech_force_profile_preserves_recoil_momentum():
    rig = GunRig(bore_mm=20.0, document={"nodes": [
        {"identity": "turret.outer_barrel", "drum_length_m": 8.4,
         "mass_kg": 725.0},
        {"identity": "turret.weapon", "mass_kg": 26.0},
    ]})

    profile = rig.recoil_force_profile(structural_interval_s=2.0e-4)

    assert len(profile["segments"]) > 10
    assert profile["duration_s"] > 0.0
    assert profile["peak_force_n"] > 0.0
    np.testing.assert_allclose(
        profile["impulse_n_s"],
        profile["recoiling_mass_kg"] * profile["recoil_velocity_m_s"],
        rtol=1.0e-12, atol=1.0e-10)
