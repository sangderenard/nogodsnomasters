"""Scripted dyno-test throttle programs: standardized, repeatable event
sequences for testing engines head-to-head.

There's no transmission in this toy -- no gear ratios, no clutch, no
shift logic -- so a real "six-speed pull" can't be driven by a human
hand on a pedal the way an interactive throttle key is. The only way to
get a genuinely repeatable, comparable run across every engine in the
catalogue is a scripted, automated event sequence: this module is that.

Two real, different ways an event can hold the engine at a target are
both here on purpose, matching two real dyno setups:
  - a plain throttle command (a genuine free pull -- nothing is
    controlling rpm at all, the engine just revs on its own combustion
    torque against friction/pumping loss, a real inertia-dyno run)
  - a brake_target_rpm command (the real PI dyno controller already in
    engine_cycle_sim.py, holding rpm by adjusting LOAD -- the load-
    absorber side of a real dyno, or standing in for "something external
    is holding this engine at a speed" the way a throttle servo would
    from the engine's own point of view, without this toy needing a
    second, separate throttle-actuator controller to get the same
    real effect)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import wave

import numpy as np

from engine_cycle_sim import EngineCycleSim
from engine_sound import EngineSoundSynth, SAMPLE_RATE, BLOCK_SIZE


@dataclass
class ThrottleEvent:
    """One scripted segment: hold for up to duration_s (a real safety
    cap), ending early the moment until_rpm_frac (if set) is reached --
    a real WOT pull doesn't run a fixed clock, it runs until it hits the
    limiter, same here."""
    duration_s: float
    throttle: float | None = None
    brake_target_rpm: float | None = None
    until_rpm_frac: float | None = None
    label: str = ""


@dataclass
class ThrottleProgram:
    """An ordered, real dyno-test event sequence."""
    events: tuple[ThrottleEvent, ...]
    name: str = ""

    def run(self, sim: EngineCycleSim, dt: float = 0.001,
            on_sample: Callable[[EngineCycleSim, ThrottleEvent, float], None] | None = None,
            sample_every_s: float = 0.05) -> None:
        redline = sim.engine.redline_rpm
        for event in self.events:
            if event.throttle is not None:
                sim.throttle = event.throttle
            # releasing brake_target_rpm now genuinely releases the dyno
            # absorber too (engine_cycle_sim.py's own fix), not just the
            # target number -- a real free pull actually is free
            sim.brake_target_rpm = event.brake_target_rpm

            elapsed = 0.0
            next_sample = 0.0
            n_ticks = max(1, int(event.duration_s / dt))
            for _ in range(n_ticks):
                sim.step(dt)
                elapsed += dt
                if on_sample is not None and elapsed >= next_sample:
                    on_sample(sim, event, elapsed)
                    next_sample += sample_every_s
                if event.until_rpm_frac is not None and sim.rpm >= event.until_rpm_frac * redline:
                    break


def six_speed_wot_pull(engine, gears: int = 6, shift_rpm_frac: float = 0.62,
                        shift_duration_s: float = 0.35, pull_max_duration_s: float = 8.0,
                        redline_frac: float = 0.94) -> ThrottleProgram:
    """A real, standardized full-throttle dyno pull, gears simulated
    without a transmission: each 'gear' is a genuine free WOT rev from
    the shift-down rpm to redline (no brake load at all -- a real
    inertia-dyno pull), then a brief load-controlled drop back to the
    shift rpm (the real PI dyno controller holding a target through
    LOAD, standing in for the clutch-and-gear-change moment a real
    transmission would otherwise provide) before the next gear's pull
    resumes. Ends the moment each gear reaches redline, not on a fixed
    clock -- same as a real WOT pull actually would."""
    events: list[ThrottleEvent] = []
    idle_rpm = engine.idle_rpm
    redline = engine.redline_rpm
    shift_rpm = max(idle_rpm * 1.2, redline * shift_rpm_frac)
    events.append(ThrottleEvent(duration_s=1.0, throttle=0.0, brake_target_rpm=idle_rpm, label="idle settle"))
    for gear in range(1, gears + 1):
        events.append(ThrottleEvent(duration_s=pull_max_duration_s, throttle=1.0, brake_target_rpm=None,
                                     until_rpm_frac=redline_frac, label=f"gear {gear} pull"))
        if gear < gears:
            events.append(ThrottleEvent(duration_s=shift_duration_s, throttle=0.15, brake_target_rpm=shift_rpm,
                                         label=f"shift {gear}->{gear + 1}"))
    return ThrottleProgram(events=tuple(events), name=f"{gears}-speed WOT pull")


def run_catalogue_queue(catalogue, program_builder=six_speed_wot_pull,
                        on_engine_start: Callable[[object], None] | None = None,
                        on_sample: Callable[[EngineCycleSim, ThrottleEvent, float], None] | None = None,
                        dt: float = 0.001) -> list[tuple[str, ThrottleProgram]]:
    """Run the same standardized program against every engine in the
    catalogue, one after another -- a real dyno-day queue. Returns the
    (identity, program) pairs actually run, in order, for a caller that
    wants to know what just happened."""
    ran: list[tuple[str, ThrottleProgram]] = []
    for engine in catalogue:
        fuel_choices = list(engine.fuel_compatibility.keys())
        if not fuel_choices:
            continue
        sim = EngineCycleSim(engine=engine, fuel_choice=fuel_choices[0])
        sim.start()
        if on_engine_start is not None:
            on_engine_start(engine)
        program = program_builder(engine)
        program.run(sim, dt=dt, on_sample=on_sample)
        ran.append((engine.identity, program))
    return ran


def _write_stereo_wav(path: str, left: np.ndarray, right: np.ndarray, sample_rate: int) -> None:
    """The same header/bay stereo image the live audio callback mixes
    (make_audio_callback in toy_shared.py), baked start to finish instead
    of played live."""
    n = min(len(left), len(right))
    interleaved = np.empty(n * 2, dtype=np.float32)
    interleaved[0::2] = left[:n]
    interleaved[1::2] = right[:n]
    clipped = np.clip(interleaved, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def bake_audio(program: ThrottleProgram, engine, fuel_choice: str, out_path: str,
                sample_rate: int = SAMPLE_RATE, physics_dt: float = 0.001,
                on_progress=None) -> float:
    """Run a ThrottleProgram start to finish and bake the whole run as one
    continuous, non-looping stereo WAV -- the script itself becomes the
    finished audio product, not a steady-state loop sample."""
    sim = EngineCycleSim(engine=engine, fuel_choice=fuel_choice)
    sim.start()
    synth = EngineSoundSynth(sample_rate)
    block_dt = BLOCK_SIZE / sample_rate
    left_blocks: list[np.ndarray] = []
    right_blocks: list[np.ndarray] = []
    block_accum_s = 0.0

    def render_one_block() -> None:
        st = sim.state
        load_frac = min(1.0, sim.brake_load_nm / max(engine.peak_torque_nm, 1.0))
        header, bay = synth.render_stereo(
            engine, sim.rpm, sim.throttle, load_frac, BLOCK_SIZE,
            knock_active=st.knock_flag, knock_intensity=st.knock_intensity,
            misfire_active=st.misfire_flag,
            boost_frac=st.boost_frac, turbo_spool_frac=st.turbo_spool_frac,
            wastegate_flutter=st.wastegate_flutter, surge_active=st.surge_flag,
            backfire_active=st.backfire_flag, backfire_kind=st.backfire_kind,
            backfire_strength=1.0,
        )
        left_blocks.append(header)
        right_blocks.append(bay)

    for event in program.events:
        if event.throttle is not None:
            sim.throttle = event.throttle
        sim.brake_target_rpm = event.brake_target_rpm
        elapsed = 0.0
        n_ticks = max(1, int(event.duration_s / physics_dt))
        for _ in range(n_ticks):
            sim.step(physics_dt)
            elapsed += physics_dt
            block_accum_s += physics_dt
            while block_accum_s >= block_dt:
                render_one_block()
                block_accum_s -= block_dt
            if event.until_rpm_frac is not None and sim.rpm >= event.until_rpm_frac * engine.redline_rpm:
                break
        if on_progress is not None:
            on_progress(event.label, elapsed)

    left = np.concatenate(left_blocks) if left_blocks else np.zeros(0, dtype=np.float32)
    right = np.concatenate(right_blocks) if right_blocks else np.zeros(0, dtype=np.float32)
    _write_stereo_wav(out_path, left, right, sample_rate)
    return len(left) / sample_rate
