"""Bake a handful of example acoustic/vibration signatures to disk so
they can actually be listened to, plus a waveform comparison plot."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import engines
import engine_baker

OUT = os.path.dirname(os.path.abspath(__file__))

CASES = [
    dict(identity="supercharged-drag-v8-8200", rpm=6200, throttle=0.95, load=0.3,
         tag="crossplane-v8-wot", note="crossplane V8, supercharger boosted, WOT"),
    dict(identity="monster-632-twin-turbo", rpm=5200, throttle=0.85, load=0.4,
         tag="twin-turbo-bigblock", note="twin-turbo big-block under load (hear the compressor whistle in intake)"),
    dict(identity="pw-r1340-wasp", rpm=2100, throttle=0.8, load=0.3,
         tag="radial-wasp-cruise", note="9-cylinder radial, polar firing order, supercharged"),
    dict(identity="superbike-i4-1340", rpm=9500, throttle=1.0, load=0.2,
         tag="superbike-i4-scream", note="1340cc superbike I4 near redline"),
    dict(identity="aircooled-flat-four-1584", rpm=3200, throttle=0.5, load=0.2,
         tag="aircooled-flat-four", note="air-cooled flat-four, no water pump"),
]

TARGET_CLIP_S = 2.5


def tile_to_duration(samples: np.ndarray, sample_rate: int, target_s: float) -> np.ndarray:
    reps = max(1, int(np.ceil(target_s * sample_rate / len(samples))))
    return np.tile(samples, reps)


fig, axes = plt.subplots(len(CASES), 2, figsize=(11, 2.1 * len(CASES)))

for row, case in enumerate(CASES):
    eng = engines.get(case["identity"])
    acoustic, loop_s = engine_baker.bake_acoustic(eng, case["rpm"], case["throttle"], case["load"], cycles=4)
    for name, samples in acoustic.items():
        clip = tile_to_duration(samples, engine_baker.AUDIO_SAMPLE_RATE, TARGET_CLIP_S)
        path = os.path.join(OUT, f"{case['tag']}_{name}.wav")
        engine_baker.write_wav(path, clip, engine_baker.AUDIO_SAMPLE_RATE)

    vibration, vloop_s = engine_baker.bake_vibration(eng, case["rpm"], case["throttle"], case["load"], cycles=4)
    mount_name = next(iter(vibration))
    mount_clip = tile_to_duration(vibration[mount_name], engine_baker.VIBE_SAMPLE_RATE, TARGET_CLIP_S)
    engine_baker.write_wav(os.path.join(OUT, f"{case['tag']}_mount_{mount_name}.wav"),
                            mount_clip, engine_baker.VIBE_SAMPLE_RATE)

    header = acoustic["header"]
    sr = engine_baker.AUDIO_SAMPLE_RATE
    t = np.arange(len(header)) / sr * 1000.0

    ax_wave, ax_spec = axes[row]
    ax_wave.plot(t, header, linewidth=0.6, color="#c0392b")
    ax_wave.set_xlim(0, min(t[-1], 60))
    ax_wave.set_title(f"{eng.label}  [{case['note']}]", fontsize=9)
    ax_wave.set_ylabel("header\nwaveform", fontsize=8)
    ax_wave.tick_params(labelsize=7)

    spec = np.abs(np.fft.rfft(header * np.hanning(len(header))))
    freqs = np.fft.rfftfreq(len(header), 1 / sr)
    mask = freqs <= 4000
    ax_spec.plot(freqs[mask], spec[mask], linewidth=0.7, color="#2c3e50")
    ax_spec.set_xlim(0, 4000)
    ax_spec.set_ylabel("spectrum", fontsize=8)
    ax_spec.tick_params(labelsize=7)

axes[-1][0].set_xlabel("ms")
axes[-1][1].set_xlabel("Hz")
fig.suptitle("Engine header-note signatures: waveform (60ms window) and frequency spectrum", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "signatures.png"), dpi=140)
print("done")
