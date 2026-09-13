"""Continuous background PCM generation, decoupled from both the realtime
audio callback and the physics tick.

Previously synth.render_stereo() ran synchronously inside the PortAudio
callback (toy_shared.make_audio_callback) -- doing real synthesis work on a
hard realtime deadline, with per-sample noise-coloring loops as a genuine
underrun risk (those loops are now native-compiled, see native_audio.py,
but the callback still had to run all of render_stereo live).

This module removes synthesis from the callback path entirely:
  - LiveAudioState is a small thread-safe snapshot of the control values
    render_stereo needs (rpm, throttle, load, knock/misfire/boost/etc).
    The physics tick writes it every step via set_from_sim(sim) -- it never
    reads the sim directly itself.
  - AudioStreamer owns an EngineSoundSynth and runs it on a background
    thread, continuously, staying a fixed lookahead ahead of playback by
    reading the latest LiveAudioState snapshot each pass and appending
    rendered blocks to a ring buffer.
  - pull(n_frames), called from the realtime PortAudio callback, only ever
    copies already-rendered samples out of that buffer -- it never runs
    synthesis itself, so it can't miss its deadline because of it. Frames
    not yet ready come back as honest silence (a real underrun reported as
    zeros) rather than blocking the audio thread.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from engine_sound import EngineSoundSynth, SAMPLE_RATE, BLOCK_SIZE
from damage_sound import DamageSoundSynth

# Control-value smoothing time constants (seconds). The physics tick
# writes a new snapshot at its own rate (60 Hz nominal, with real jitter
# when a tick runs long); the synth thread renders 11.6 ms blocks. Read
# raw, every block took whatever the LAST tick happened to leave --
# rpm stepping in 60 Hz stairs, and a late tick followed by a jump: the
# audible "digital catch-up". These slew each continuous control toward
# the latest snapshot every block, so what the synth hears is a smooth
# trajectory through the sim's samples, and a stale snapshot simply
# HOLDS (state continuance): a paused or slow sim keeps the engine
# sounding exactly as it last was, never a drop-out or a lurch.
RPM_SLEW_TAU_S = 0.045        # a crank's own inertia is the real smoother; this just spans tick jitter
THROTTLE_SLEW_TAU_S = 0.060
LOAD_SLEW_TAU_S = 0.080
BOOST_SLEW_TAU_S = 0.090
FIRE_HZ_SLEW_TAU_S = 0.045


class LiveAudioState:
    """Thread-safe snapshot of the control values render_stereo needs. The
    physics tick is the only writer (set_from_sim); the synthesis thread is
    the only reader (snapshot) -- one lock, no partial reads/writes torn
    across the two threads."""

    def __init__(self, engine) -> None:
        self._lock = threading.Lock()
        self.engine = engine
        self.rpm = 0.0
        self.throttle = 0.0
        self.load_frac = 0.0
        self.knock_active = False
        self.knock_intensity = 0.0
        self.misfire_active = False
        self.boost_frac = 0.0
        self.turbo_spool_frac = 0.0
        self.wastegate_flutter = False
        self.surge_active = False
        self.backfire_active = False
        self.backfire_kind = None
        self.backfire_strength = 1.0
        self.real_fire_hz = 0.0
        self.intake_flow_demand_frac = 0.0
        self.knock_ring_hz = ()      # the chamber's own bore modes this tick (engine_sound.knock_ring_modes_hz)
        self.exhaust_temp_k = 293.15
        self.preignition_active = False
        self.preignition_intensity = 0.0
        self.compression_brake_active = False
        # the event-side facts (engine_sound.py's own crank clock schedules
        # its kernels off these): per-slot firing records, held-open
        # exhaust, ignition count, manifold pressure, the parts graph
        self.extras = {"slot_records": (), "slot_angles_deg": (), "exhaust_valve_held_open": False,
                       "fire_event_count": 0, "manifold_pressure_frac": 1.0, "graph": None}
        self.damage_events: list[dict] = []     # queued for the damage synth, drained by snapshot()
        self.emitters: tuple = ()
        self.stamp = time.monotonic()

    def set_from_sim(self, sim) -> None:
        engine = sim.engine
        st = sim.state
        load_frac = min(1.0, sim.brake_load_nm / max(engine.peak_torque_nm, 1.0))
        with self._lock:
            self.engine = engine
            self.rpm = sim.rpm
            self.throttle = sim.throttle
            self.load_frac = load_frac
            self.knock_active = st.knock_flag
            self.knock_intensity = st.knock_intensity
            self.misfire_active = st.misfire_flag
            self.boost_frac = st.boost_frac
            self.turbo_spool_frac = st.turbo_spool_frac
            self.wastegate_flutter = st.wastegate_flutter
            self.surge_active = st.surge_flag
            self.backfire_active = st.backfire_flag
            self.backfire_kind = st.backfire_kind
            self.backfire_strength = 1.0
            # the real, measured firing rate (EngineCycleState.
            # real_fire_hz) -- see engine_sound.py's own comment on why
            # this beats an rpm-derived formula for every engine kind,
            # not just the free-piston one it was found chasing
            self.real_fire_hz = st.real_fire_hz
            # the real port restriction (drivetrain_graph.py's own
            # flow_capacity_kg_s ceiling) -- what engine_sound.py's
            # induction roar now scales off instead of a throttle/rpm
            # proxy
            self.intake_flow_demand_frac = st.intake_flow_demand_frac
            # the knock ring: the combustion chamber's OWN acoustic modes
            # (bore diameter, hot-gas sound speed), not a fixed chord
            from engine_sound import knock_ring_modes_hz
            self.knock_ring_hz = knock_ring_modes_hz(engine, getattr(st, "exhaust_temp_k", 900.0), load_frac)
            self.exhaust_temp_k = float(getattr(st, "exhaust_temp_k", 293.15))
            self.preignition_active = bool(getattr(st, "preignition_flag", False))
            self.preignition_intensity = float(getattr(st, "preignition_intensity", 0.0))
            self.compression_brake_active = bool(getattr(st, "compression_brake_active", False))
            self.extras = {
                "slot_records": tuple(tuple(r) for r in getattr(st, "slot_records", ())),
                "slot_angles_deg": tuple(getattr(st, "slot_angles_deg", ())),
                "exhaust_valve_held_open": bool(getattr(st, "exhaust_valve_held_open", False)),
                "fire_event_count": int(getattr(st, "fire_event_count", 0)),
                "manifold_pressure_frac": float(getattr(st, "manifold_pressure_frac", 1.0)),
                "graph": getattr(getattr(sim, "_drivetrain", None), "graph", None),
                "exhaust_open_frac": float(getattr(st, "exhaust_open_frac", 0.0)),
            }
            if getattr(sim, "damage_events", None):
                self.damage_events.extend(sim.drain_damage_events())
            field = getattr(sim, "hole_emitters", None)
            self.emitters = field.summary() if field is not None and field.emitters else ()
            self.stamp = time.monotonic()

    def take_damage(self) -> tuple[list[dict], tuple]:
        with self._lock:
            ev, self.damage_events = self.damage_events, []
            return ev, self.emitters

    def snapshot(self):
        with self._lock:
            return (
                self.engine, self.rpm, self.throttle, self.load_frac,
                self.knock_active, self.knock_intensity, self.misfire_active,
                self.boost_frac, self.turbo_spool_frac, self.wastegate_flutter,
                self.surge_active, self.backfire_active, self.backfire_kind,
                self.backfire_strength, self.real_fire_hz, self.intake_flow_demand_frac,
                self.knock_ring_hz, self.exhaust_temp_k, self.preignition_active, self.preignition_intensity,
                self.compression_brake_active, self.extras,
            )


class AudioStreamer:
    """Runs EngineSoundSynth.render_stereo continuously on a background
    thread, a fixed lookahead ahead of playback, into a ring buffer the
    realtime callback drains from -- live PortAudio output never itself
    does synthesis work."""

    def __init__(self, state: LiveAudioState, sample_rate: int = SAMPLE_RATE,
                 lookahead_s: float = 0.35) -> None:
        self.state = state
        self.synth = EngineSoundSynth(sample_rate)
        self.damage_synth = DamageSoundSynth(sample_rate)
        self.sample_rate = sample_rate
        self._lookahead_frames = int(lookahead_s * sample_rate)
        self._lock = threading.Lock()
        self._left = np.zeros(0, dtype=np.float32)
        self._right = np.zeros(0, dtype=np.float32)
        self._read_pos = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # the smoothed controls the synth actually hears (see the module
        # constants above); None until the first snapshot seeds them
        self._smooth: dict | None = None
        self._last_stamp = None

    def _smoothed(self, snap: tuple, block_s: float) -> tuple:
        (engine, rpm, throttle, load_frac, knock_active, knock_intensity, misfire_active,
         boost_frac, turbo_spool_frac, wastegate_flutter, surge_active, backfire_active,
         backfire_kind, backfire_strength, real_fire_hz, intake_flow_demand_frac, knock_ring_hz,
         exhaust_temp_k, preignition_active, preignition_intensity, compression_brake_active, extras) = snap
        targets = {"rpm": (rpm, RPM_SLEW_TAU_S), "throttle": (throttle, THROTTLE_SLEW_TAU_S),
                   "load_frac": (load_frac, LOAD_SLEW_TAU_S), "boost_frac": (boost_frac, BOOST_SLEW_TAU_S),
                   "turbo_spool_frac": (turbo_spool_frac, BOOST_SLEW_TAU_S),
                   "real_fire_hz": (real_fire_hz, FIRE_HZ_SLEW_TAU_S),
                   "intake_flow_demand_frac": (intake_flow_demand_frac, LOAD_SLEW_TAU_S)}
        if self._smooth is None or self._smooth.get("engine") is not engine:
            # a new engine (or first block): seed at the real values, no glide from an unrelated engine
            self._smooth = {k: v for k, (v, _tau) in targets.items()}
            self._smooth["engine"] = engine
        else:
            for k, (v, tau) in targets.items():
                a = 1.0 - np.exp(-block_s / max(tau, 1e-4))
                self._smooth[k] += (v - self._smooth[k]) * a
        s = self._smooth
        return (engine, s["rpm"], s["throttle"], s["load_frac"], knock_active, knock_intensity, misfire_active,
                s["boost_frac"], s["turbo_spool_frac"], wastegate_flutter, surge_active, backfire_active,
                backfire_kind, backfire_strength, s["real_fire_hz"], s["intake_flow_demand_frac"], knock_ring_hz,
                exhaust_temp_k, preignition_active, preignition_intensity, compression_brake_active, extras)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="audio-synth", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _buffered_frames(self) -> int:
        with self._lock:
            return len(self._left) - self._read_pos

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._buffered_frames() >= self._lookahead_frames:
                time.sleep(0.002)
                continue
            (engine, rpm, throttle, load_frac, knock_active, knock_intensity,
             misfire_active, boost_frac, turbo_spool_frac, wastegate_flutter,
             surge_active, backfire_active, backfire_kind, backfire_strength,
             real_fire_hz, intake_flow_demand_frac, knock_ring_hz,
             exhaust_temp_k, preignition_active, preignition_intensity, compression_brake_active, extras) = self._smoothed(
                self.state.snapshot(), BLOCK_SIZE / self.sample_rate)
            header, bay = self.synth.render_stereo(
                engine, rpm, throttle, load_frac, BLOCK_SIZE,
                knock_active=knock_active, knock_intensity=knock_intensity,
                misfire_active=misfire_active,
                boost_frac=boost_frac, turbo_spool_frac=turbo_spool_frac,
                wastegate_flutter=wastegate_flutter, surge_active=surge_active,
                backfire_active=backfire_active, backfire_kind=backfire_kind,
                backfire_strength=backfire_strength,
                real_fire_hz=real_fire_hz,
                intake_flow_demand_frac=intake_flow_demand_frac,
                knock_ring_hz=knock_ring_hz, exhaust_temp_k=exhaust_temp_k,
                preignition_active=preignition_active, preignition_intensity=preignition_intensity,
                compression_brake_active=compression_brake_active,
                **extras,
            )
            # the damage system's own sounds, in the bay (and a little in
            # the header -- a hit on the block carries into the pipe)
            events, emitters = self.state.take_damage()
            for ev in events:
                if ev.get("kind") == "blast":
                    self.damage_synth.blast(ev["energy_j"], ev.get("volume_m3", 0.01))
                else:
                    self.damage_synth.impact(ev["mode"], ev["material"], ev["wall_m"], ev["energy_spent_j"],
                                             ev.get("pressure_pa", 101_325.0), ev.get("speed_after_m_s", 0.0),
                                             ev.get("calibre_m", 0.0076))
            dmg = self.damage_synth.render(BLOCK_SIZE, emitters)
            if np.any(dmg):
                bay = np.tanh(bay + dmg * 1.0).astype(np.float32)
                header = np.tanh(header + dmg * 0.25).astype(np.float32)
            with self._lock:
                if self._read_pos > 0:
                    self._left = self._left[self._read_pos:]
                    self._right = self._right[self._read_pos:]
                    self._read_pos = 0
                self._left = np.concatenate([self._left, header.astype(np.float32)])
                self._right = np.concatenate([self._right, bay.astype(np.float32)])

    def pull(self, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Realtime-safe: only ever slices/copies already-rendered samples.
        Returns silence for any frames not yet ready -- a genuine underrun,
        reported honestly as zeros rather than blocking the audio thread."""
        with self._lock:
            available = len(self._left) - self._read_pos
            take = min(n_frames, max(0, available))
            left = self._left[self._read_pos:self._read_pos + take].copy()
            right = self._right[self._read_pos:self._read_pos + take].copy()
            self._read_pos += take
        if take < n_frames:
            pad = n_frames - take
            left = np.concatenate([left, np.zeros(pad, dtype=np.float32)])
            right = np.concatenate([right, np.zeros(pad, dtype=np.float32)])
        return left, right
