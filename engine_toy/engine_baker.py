"""Builds an engine block as a mech_parts.Assembly and bakes it.

This is the "engine" specialization of the general parts/assembly
system in mech_parts.py: CombustionCrankPart + ValvetrainPart + a
per-loadout set of RotaryTonePart accessories (oil pump always,
water pump/fan/alternator only if the engine's Accessories say so),
hung off the cylinder/mount/acoustic geometry from engine_geometry.py.

Public API (unchanged from before the mech_parts refactor):
    bake_acoustic()     -- the two acoustic points (header, engine bay)
    bake_vibration()    -- the structural mount points
    bake_physics_rate() -- mount points decimated to physics rate,
                            for feeding a chassis/suspension solver
"""
from __future__ import annotations

import numpy as np

from engines import Engine
from engine_geometry import cylinder_sites, mount_points, acoustic_points, accessory_front_point
from engine_sim import torque_fraction
from mech_parts import Assembly, CombustionCrankPart, ValvetrainPart, RotaryTonePart, bake_assembly, write_wav

AUDIO_SAMPLE_RATE = 44100
VIBE_SAMPLE_RATE = 22050
PHYSICS_SAMPLE_RATE = 400

__all__ = [
    "AUDIO_SAMPLE_RATE", "VIBE_SAMPLE_RATE", "PHYSICS_SAMPLE_RATE",
    "build_assembly", "bake_acoustic", "bake_vibration", "bake_physics_rate", "write_wav",
]


def build_assembly(engine: Engine) -> Assembly:
    arch = engine.architecture
    sites = cylinder_sites(engine)
    front = accessory_front_point(engine)
    positions = {s.number: s.position for s in sites}
    valve_positions = {s.number: (s.position[0], s.position[1] * 1.05, s.position[2] * 1.15) for s in sites}

    def rpm_frac(rpm):
        return min(1.2, max(0.0, rpm / max(engine.redline_rpm, 1.0)))

    parts: list = []

    if arch.cylinders:
        parts.append(CombustionCrankPart(
            cylinders=arch.cylinders, firing_order=tuple(arch.firing_order),
            cycle_degrees=arch.cycle_degrees, cylinder_positions=positions,
            slot_angles_deg=tuple(arch.slot_angles_deg()),
            strength_fn=lambda rpm, thr, load: torque_fraction(engine, rpm) * (0.2 + 0.8 * thr),
            primary_amp_fn=lambda rpm, thr, load: 0.35 * (0.3 + 0.7 * load + 0.3 * thr),
            secondary_amp_fn=lambda rpm, thr, load: 0.18 * (0.3 + 0.7 * load) * (arch.cylinders / 4.0),
        ))
        if not arch.rotary:
            # port-timed rotary has no poppet valves/lifter springs
            spring = engine.lifter_spring
            spring_ratio = spring.spring_rate_n_per_mm / 45.0   # relative to the "stock" preset
            parts.append(ValvetrainPart(
                cylinder_positions=valve_positions, cycle_degrees=arch.cycle_degrees,
                amp_fn=lambda rpm, thr, load: 0.22 * (0.4 + 0.6 * rpm_frac(rpm)) * min(2.0, spring_ratio ** 0.5),
                ring_freq_hz=520.0 * min(1.8, spring_ratio ** 0.5),
                ring_tau_s=0.008 / min(1.6, spring_ratio ** 0.4),
            ))
        parts.append(RotaryTonePart(
            name="oil-pump", domain="both", position=(front[0], front[1], front[2] - 0.05), order=1.0,
            amp_fn=lambda rpm, thr, load: 0.08 + 0.10 * rpm_frac(rpm),
        ))
        acc = engine.accessories
        if acc.water_pump:
            parts.append(RotaryTonePart(
                name="water-pump", domain="both", position=front, order=1.2,
                amp_fn=lambda rpm, thr, load: 0.06 + 0.08 * rpm_frac(rpm),
            ))
        if acc.mechanical_fan or acc.alternator:
            parts.append(RotaryTonePart(
                name="belt-flutter", domain="both", position=front, order=2.6, phase_rad=0.7,
                amp_fn=lambda rpm, thr, load: 0.04 + 0.03 * load,
            ))
        if acc.mechanical_fan:
            fan_pos = (front[0] - 0.03, front[1], front[2] + 0.02)

            def fan_amp(rpm, thr, load):
                engaged = min(1.0, max(0.0, rpm_frac(rpm) * 1.4 - 0.15))
                return 0.05 * engaged * engaged
            parts.append(RotaryTonePart(name="fan-blade-pass", domain="airborne", position=fan_pos,
                                         order=1.05 * 6.0, amp_fn=fan_amp))
        if acc.alternator:
            parts.append(RotaryTonePart(
                name="alternator-whine", domain="both", position=(front[0], front[1] + 0.02, front[2]),
                order=2.9, amp_fn=lambda rpm, thr, load: 0.05 + 0.10 * load,
            ))
    else:
        centroid = (0.0, 0.0, 0.0)
        parts.append(RotaryTonePart(
            name="rotor-order", domain="both", position=centroid, order=6.0,
            amp_fn=lambda rpm, thr, load: 0.25 * (0.3 + 0.7 * load),
        ))
        parts.append(RotaryTonePart(
            name="inverter-switching", domain="airborne", position=front, fixed_freq_hz=8000.0,
            amp_fn=lambda rpm, thr, load: 0.06 + 0.10 * thr,
        ))
        if engine.accessories.coolant_pump:
            parts.append(RotaryTonePart(
                name="coolant-pump", domain="both", position=front, order=1.3,
                amp_fn=lambda rpm, thr, load: 0.05 + 0.05 * rpm_frac(rpm),
            ))

    acoustic = acoustic_points(engine)
    fi = engine.forced_induction
    if fi.kind == "turbo":
        # steady-state spool estimate -- the same target engine_cycle_sim's
        # lagged integrator converges to, without the lag itself (there's
        # no time axis to lag across in a single-instant snapshot bake)
        def turbo_spool(rpm, thr, load):
            return min(1.0, rpm_frac(rpm) * (0.25 + 0.75 * thr))
        parts.append(RotaryTonePart(
            name="turbo-compressor-whistle", domain="airborne", position=acoustic["intake"],
            freq_fn=lambda rpm, thr, load: 600.0 + 7000.0 * turbo_spool(rpm, thr, load),
            amp_fn=lambda rpm, thr, load: 0.35 * turbo_spool(rpm, thr, load),
        ))
    elif fi.kind == "supercharger":
        order = fi.belt_ratio * fi.lobe_count
        parts.append(RotaryTonePart(
            name="supercharger-whine", domain="airborne", position=acoustic["intake"], order=order,
            amp_fn=lambda rpm, thr, load: 0.30 * (0.3 + 0.7 * fi.max_boost_frac * thr * min(1.0, rpm_frac(rpm) * 1.3)),
        ))
    if fi.kind != "none":
        parts.append(RotaryTonePart(
            name="intake-roar", domain="airborne", position=acoustic["intake"], fixed_freq_hz=180.0,
            amp_fn=lambda rpm, thr, load: 0.10 + 0.30 * thr * rpm_frac(rpm),
        ))

    return Assembly(name=f"engine:{engine.identity}", parts=parts,
                     acoustic_points=acoustic, structural_points=mount_points(engine))


def bake_vibration(engine: Engine, rpm: float, throttle: float, load_frac: float,
                    cycles: int = 4, sample_rate: int = VIBE_SAMPLE_RATE) -> tuple[dict[str, np.ndarray], float]:
    return bake_assembly(build_assembly(engine), rpm, throttle, load_frac, "structural", cycles, sample_rate)


def bake_acoustic(engine: Engine, rpm: float, throttle: float, load_frac: float,
                   cycles: int = 4, sample_rate: int = AUDIO_SAMPLE_RATE) -> tuple[dict[str, np.ndarray], float]:
    return bake_assembly(build_assembly(engine), rpm, throttle, load_frac, "acoustic", cycles, sample_rate)


def bake_physics_rate(engine: Engine, rpm: float, throttle: float, load_frac: float,
                       cycles: int = 1, sample_rate: int = PHYSICS_SAMPLE_RATE) -> tuple[dict[str, np.ndarray], float]:
    """Same structural source model, decimated to a rate a chassis/
    suspension solver can consume as a per-tick force trace. Left
    un-normalized -- physics wants real relative magnitude, not a
    loudness-matched waveform."""
    return bake_assembly(build_assembly(engine), rpm, throttle, load_frac, "structural", cycles, sample_rate,
                          normalize_output=False)
