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

    def snapshot(self):
        with self._lock:
            return (
                self.engine, self.rpm, self.throttle, self.load_frac,
                self.knock_active, self.knock_intensity, self.misfire_active,
                self.boost_frac, self.turbo_spool_frac, self.wastegate_flutter,
                self.surge_active, self.backfire_active, self.backfire_kind,
                self.backfire_strength, self.real_fire_hz, self.intake_flow_demand_frac,
            )


class AudioStreamer:
    """Runs EngineSoundSynth.render_stereo continuously on a background
    thread, a fixed lookahead ahead of playback, into a ring buffer the
    realtime callback drains from -- live PortAudio output never itself
    does synthesis work."""

    def __init__(self, state: LiveAudioState, sample_rate: int = SAMPLE_RATE,
                 lookahead_s: float = 0.25) -> None:
        self.state = state
        self.synth = EngineSoundSynth(sample_rate)
        self.sample_rate = sample_rate
        self._lookahead_frames = int(lookahead_s * sample_rate)
        self._lock = threading.Lock()
        self._left = np.zeros(0, dtype=np.float32)
        self._right = np.zeros(0, dtype=np.float32)
        self._read_pos = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
             real_fire_hz, intake_flow_demand_frac) = self.state.snapshot()
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
            )
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
