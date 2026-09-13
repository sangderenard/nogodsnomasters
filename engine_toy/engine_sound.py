"""Real-time synthesis at the two acoustic emission points, from parts.

`render_stereo` is the live (streaming, phase-continuous) counterpart to
`engine_baker.bake_acoustic` -- same idea, two listening points, built
from persistent state block-to-block so it can run under a live audio
callback without clicking.

  header      -- inside the header/primary-drive housing: the exhaust
                 events (each exhaust port cracking open) through the
                 pipe's own fixed response.
  engine_bay  -- standing next to the block: the combustion ring in the
                 casting, valve seats, the pitched accessories, plus a
                 heavily muffled bleed of the header.

The organising rule (sound_parts.py): PITCH belongs only to parts that
spin -- each pitched part is a real graph node with its own pass count
and drive ratio, all advanced together as one sine bank. EVENTS
(combustion, blowdown, seats, knock, a Jake crack, an injector plunger)
have a FIXED signature set by the geometry they excite; rpm only sets
when they occur. The synth keeps its own crank-angle clock, integrated
per sample from the smoothed rpm, and fires each cylinder's events when
that clock crosses the slot's angle -- with the amplitude the sim
recorded for what that cylinder actually did on its last pass
(EngineCycleState.slot_records). A cut cylinder makes no combustion
sound; its exhaust port still passes whatever pressure it really holds
(a motored compression: a little; a held-open hit-and-miss valve:
nothing; a compression-release crack at TDC: a great deal).

Every event class is rendered as an impulse train convolved with its
kernel (cached FFTs), and every noise part as one noise stream shaped by
(train * envelope) -- statistically identical to per-event noise and one
vector op per class per block.
"""
from __future__ import annotations

import math

import numpy as np

from engines import Engine
import native_audio
import sound_parts

SAMPLE_RATE = 44100
BLOCK_SIZE = 512

# Detonation rings the COMBUSTION CHAMBER, not the pipes: the autoignition
# pressure wave bounces across the bore, so the ring sits at the chamber's
# own acoustic modes (Draper): f = c * rho / (pi * B) with the Bessel
# roots rho of the circumferential/radial modes -- 1.841 for the first
# circumferential (the 5-7 kHz a knock sensor is tuned to on a car
# bore), then 3.054, 3.832, 4.201. c is the hot gas's own sound speed
# (sqrt(gamma R T)), so the ring climbs with charge temperature/load.
KNOCK_MODE_ROOTS = (1.841, 3.054, 3.832, 4.201)
KNOCK_GAS_GAMMA = 1.30
KNOCK_GAS_R = 287.0

# Exhaust port pressure at opening, relative to a full-strength burn:
MOTORED_BLOWDOWN_FRAC = 0.12      # a compressed-and-expanded charge that never burned, per unit MAP
JAKE_RELEASE_FRAC = 1.1           # the compression-release crack at TDC: full compression pressure


def knock_ring_modes_hz(engine, exhaust_temp_k: float, load_frac: float) -> tuple[float, ...]:
    """The chamber ring frequencies for this engine's own bore at the
    current gas state. Bore from the architecture (or, for a catalogue
    entry that never declared one, from displacement over cylinders at
    a square bore/stroke)."""
    arch = engine.architecture
    bore = float(getattr(arch, "bore_m", 0.0) or 0.0)
    if bore <= 0.0:
        n = max(1, int(getattr(arch, "cylinders", 1) or 1))
        vol = max(engine.displacement_l, 0.05) / 1000.0 / n
        bore = (4.0 * vol / math.pi) ** (1.0 / 3.0)
    # end-of-compression gas is far hotter than the exhaust the sim
    # reports; a knocking charge sits ~900-1500 K depending on load
    gas_k = max(700.0, 900.0 + 500.0 * max(0.0, min(1.0, load_frac)) + 0.15 * max(0.0, exhaust_temp_k - 600.0))
    c = math.sqrt(KNOCK_GAS_GAMMA * KNOCK_GAS_R * gas_k)
    shape = getattr(getattr(engine, "chamber", None), "ring_mode_factor", 1.0)
    return tuple(shape * c * rho / (math.pi * bore) for rho in KNOCK_MODE_ROOTS)


# ---------------------------------------------------------------------------
# vectorised machinery: a tail mixer, cached-FFT convolution, a sine bank
# ---------------------------------------------------------------------------

class _Tail:
    """Overlap-add accumulator: kernels longer than a block spill into
    the next blocks."""

    def __init__(self) -> None:
        self.buf = np.zeros(0)

    def add(self, sig: np.ndarray) -> None:
        if len(sig) > len(self.buf):
            self.buf = np.concatenate([self.buf, np.zeros(len(sig) - len(self.buf))])
        self.buf[:len(sig)] += sig

    def take(self, n: int) -> np.ndarray:
        if len(self.buf) < n:
            self.buf = np.concatenate([self.buf, np.zeros(n - len(self.buf))])
        out = self.buf[:n].copy()
        self.buf = self.buf[n:]
        return out


class _Convolver:
    """train (n) * kernel (L) -> n+L-1, through cached kernel spectra."""

    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[int, np.ndarray]] = {}

    def __call__(self, train: np.ndarray, kernel: np.ndarray, key) -> np.ndarray:
        n, L = len(train), len(kernel)
        full = n + L - 1
        nfft = 1 << (full - 1).bit_length()
        ent = self._cache.get(key)
        if ent is None or ent[0] != nfft:
            ent = (nfft, np.fft.rfft(kernel, nfft))
            self._cache[key] = ent
        return np.fft.irfft(np.fft.rfft(train, nfft) * ent[1], nfft)[:full]


class _SineBank:
    """N phase-continuous sines advanced together: per-sample frequency
    ramps from each voice's previous frequency, one matrix op."""

    def __init__(self) -> None:
        self.keys: tuple = ()
        self.phase = np.zeros(0)
        self.prev_f = np.zeros(0)

    def render(self, keys: tuple, freqs: np.ndarray, amps: np.ndarray, n: int, sr: int) -> np.ndarray:
        if keys != self.keys:
            phase = np.zeros(len(keys)); prev = np.array(freqs, dtype=np.float64)
            old = dict(zip(self.keys, zip(self.phase, self.prev_f)))
            for i, k in enumerate(keys):
                if k in old:
                    phase[i], prev[i] = old[k]
            self.keys, self.phase, self.prev_f = keys, phase, prev
        if not len(keys):
            return np.zeros(n)
        ramp = np.arange(1, n + 1) / n
        f = self.prev_f[:, None] + (freqs - self.prev_f)[:, None] * ramp[None, :]
        ph = self.phase[:, None] + np.cumsum(2 * np.pi * f / sr, axis=1)
        out = (amps[:, None] * np.sin(ph)).sum(axis=0)
        self.phase = ph[:, -1] % (2 * np.pi)
        self.prev_f = np.array(freqs, dtype=np.float64)
        return out


class EngineSoundSynth:
    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sr = sample_rate
        self.rng = np.random.default_rng(1234)
        self.motor_phase = np.zeros(3)
        self.turbine_whine_phase = np.zeros(3)
        self.turbine_noise_state = 0.0
        self.bleed_state = 0.0
        self.muffle_state = 0.0
        self.muffle_state_2 = 0.0
        self.hiss_state = 0.0
        self.turbo_phase = 0.0
        self.wastegate_phase = 0.0
        self.intake_noise_state = 0.0
        self._prev_throttle_for_intake = 0.0
        self.backfire_boom_env = 0.0
        self.backfire_crack_env = 0.0
        self.backfire_noise_lp_state = 0.0
        self.backfire_pipe_phase = np.zeros(3)
        self.backfire_kind_live = None
        self.intake_pop_env = 0.0
        self.intake_pop_phase = 0.0
        self.intake_pop_noise_state = 0.0
        self.jake_noise_state = 0.0
        self.misfire_noise_state = 0.0
        # the event machinery
        self._crank_deg: float | None = None          # the synth's own crank clock (unwrapped degrees)
        self._prev_rpm: float | None = None
        self._header_tail = _Tail()
        self._bay_tail = _Tail()
        self._conv = _Convolver()
        self._kernels: dict = {}
        self._kernel_key = None
        self._bank = _SineBank()
        self._catalogue = None
        self._catalogue_key = None
        self._fire_count_seen: int | None = None
        self._free_piston_phase = 0.0
        self._prev_knock = False
        self._prev_misfire = False
        self._prev_preignition = False
        self._exhaust_open = 0.0

    # ------------------------------------------------------------------
    def render_stereo(self, engine: Engine, rpm: float, throttle: float,
                       load_frac: float, n_frames: int,
                       knock_active: bool = False, knock_intensity: float = 0.0,
                       misfire_active: bool = False,
                       boost_frac: float = 0.0, turbo_spool_frac: float = 0.0,
                       wastegate_flutter: bool = False, surge_active: bool = False,
                       backfire_active: bool = False, backfire_kind: str | None = None,
                       backfire_strength: float = 0.0,
                       real_fire_hz: float = 0.0,
                       intake_flow_demand_frac: float = 0.0,
                       knock_ring_hz: tuple = (), exhaust_temp_k: float = 293.15,
                       preignition_active: bool = False, preignition_intensity: float = 0.0,
                       compression_brake_active: bool = False,
                       slot_records: tuple = (), slot_angles_deg: tuple = (),
                       exhaust_valve_held_open: bool = False, fire_event_count: int = 0,
                       manifold_pressure_frac: float = 1.0, graph: dict | None = None,
                       exhaust_open_frac: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        # a holed/open exhaust (node_effects): the pipe is that much
        # brighter and the bay hears that much of the raw header
        self._exhaust_open = max(0.0, min(1.0, exhaust_open_frac))
        cylinders = engine.architecture.cylinders
        sr = self.sr
        header = np.zeros(n_frames)
        bay = np.zeros(n_frames)

        if cylinders > 0:
            # ---- the crank clock, per sample, from the smoothed rpm ----
            prev_rpm = rpm if self._prev_rpm is None else self._prev_rpm
            rpm_ramp = prev_rpm + (rpm - prev_rpm) * (np.arange(1, n_frames + 1) / n_frames)
            self._prev_rpm = rpm
            if self._crank_deg is None:
                self._crank_deg = 0.0
            theta = self._crank_deg + np.cumsum(np.maximum(rpm_ramp, 0.0) * 6.0 / sr)
            theta_prev = self._crank_deg
            self._crank_deg = float(theta[-1]) % float(engine.architecture.cycle_degrees) + \
                float(engine.architecture.cycle_degrees) * 4.0   # keep it bounded and away from zero
            deg_per_s = max(rpm, 0.0) * 6.0
            h, b = self._render_events(engine, theta, theta_prev, deg_per_s, rpm, throttle, load_frac, n_frames,
                                       slot_records, slot_angles_deg, exhaust_valve_held_open, manifold_pressure_frac,
                                       compression_brake_active, knock_ring_hz, exhaust_temp_k, intake_flow_demand_frac,
                                       graph)
            header += h
            bay += b
        elif engine.kind == "atmospheric":
            h, b = self._render_free_piston_events(engine, n_frames, real_fire_hz, fire_event_count, exhaust_temp_k,
                                                   knock_ring_hz, load_frac)
            header += h
            bay += b
        elif engine.kind == "turbine":
            header += self._render_header_turbine(engine, turbo_spool_frac, throttle, load_frac, n_frames)
        else:
            header += self._render_header_electric(engine, rpm, throttle, load_frac, n_frames)

        # tick-flag events for kinds without slots (a free piston's own knock/misfire flags)
        if cylinders == 0:
            knock_edge = knock_active and not self._prev_knock
            preignition_edge = preignition_active and not self._prev_preignition
            if knock_edge or preignition_edge:
                modes = knock_ring_hz or knock_ring_modes_hz(engine, 900.0, load_frac)
                k = self._chamber_kernels(modes)
                amp = (0.5 + 0.5 * knock_intensity) if knock_edge else (0.6 + 0.4 * min(1.0, preignition_intensity))
                self._bay_tail.add(amp * (k["preignition"] if preignition_edge else k["knock"]))
        self._prev_knock = knock_active
        self._prev_misfire = misfire_active
        self._prev_preignition = preignition_active

        header += self._render_backfire(n_frames, backfire_active, backfire_strength,
                                        engine=engine, kind=backfire_kind, exhaust_temp_k=exhaust_temp_k)
        header += self._header_tail.take(n_frames)
        bay += self._bay_tail.take(n_frames)

        # the muffler/cat is a fixed low-pass on the pipe (open header
        # ~6 kHz corner, boxed stocker ~1.2 kHz): geometry, not rpm
        if cylinders > 0:
            brightness = max(engine.exhaust_system.brightness_frac, self._exhaust_open)
            corner_hz = 1200.0 + 4800.0 * brightness
            # two poles (12 dB/oct): a chambered box is steeper than a
            # single RC, and one pole still let the click's white top
            # through
            alpha = 1.0 - math.exp(-2 * math.pi * corner_hz / sr)
            header, self.muffle_state = native_audio.leaky_integrator(header, alpha, self.muffle_state)
            header, self.muffle_state_2 = native_audio.leaky_integrator(header, alpha, self.muffle_state_2)

        bay += self._render_engine_bay(engine, rpm, throttle, load_frac, boost_frac, turbo_spool_frac, n_frames, header, graph)
        bay += self._render_intake_pop(engine, n_frames, backfire_active and backfire_kind == "intake-flashback",
                                       backfire_strength)
        bay += self._render_induction(engine, n_frames, turbo_spool_frac, wastegate_flutter, surge_active)

        # A stopped crank has no physical excitation to make any of this
        # noise from. A smooth ramp over the first 40 rpm (starter-cranking
        # speed) so cranking fades in instead of clicking. For the free-
        # piston kind rpm is the flywheel's speed, which can lag real
        # piston activity; real_fire_hz is the piston's own signal.
        activity = max(rpm, 0.0) if engine.kind != "atmospheric" else max(rpm, real_fire_hz * 40.0)
        run_gate = np.clip(activity / 40.0, 0.0, 1.0)
        header = header * run_gate
        bay = bay * run_gate
        return np.tanh(header * 1.3).astype(np.float32), np.tanh(bay * 1.3).astype(np.float32)

    # Backward-compatible single-channel accessor (header only).
    def render(self, engine: Engine, rpm: float, throttle: float,
               load_frac: float, n_frames: int) -> np.ndarray:
        header, _bay = self.render_stereo(engine, rpm, throttle, load_frac, n_frames)
        return header

    # ------------------------------------------------------------------
    # fixed kernels (geometry + gas temperature), rebuilt only when those move
    # ------------------------------------------------------------------
    def _event_kernels(self, engine, exhaust_temp_k: float) -> dict:
        exhaust = engine.exhaust_system
        f0 = max(25.0, min(600.0, exhaust.tuned_frequency_hz(exhaust_temp_k)))
        brightness = exhaust.brightness_frac
        spring_ratio = engine.lifter_spring.spring_rate_n_per_mm / 45.0
        key = (id(engine), round(f0 / 4.0), round(brightness, 2))
        if key == self._kernel_key:
            return self._kernels
        sr = self.sr
        block_hz = sound_parts.block_ring_hz(engine)
        k = {
            "pipe": sound_parts.pipe_response_kernel(sr, f0, brightness),
            "plosive": sound_parts.plosive_kernel(sr, 0.0014),
            # the casting rings at its own mode and a stiffer second one
            "block": sound_parts.ring_kernel(sr, (block_hz, block_hz * 2.7), (0.022, 0.010), (1.0, 0.35)),
            # a valve seating: a short bright clack, sharper with a stiffer spring
            "seat": sound_parts.ring_kernel(sr, (520.0, 1900.0), (0.008 / max(0.6, spring_ratio ** 0.5), 0.003), (1.0, 0.5), max_s=0.03),
            # an injector plunger / pump cam: a hard little tick
            "injector": sound_parts.ring_kernel(sr, (1800.0, 4200.0), (0.0025, 0.0012), (1.0, 0.4), max_s=0.015),
            "jake_crack": sound_parts.plosive_kernel(sr, 0.0020),
            "f0": f0, "brightness": brightness,
        }
        self._kernels, self._kernel_key = k, key
        return k

    def _chamber_kernels(self, modes) -> dict:
        key = ("chamber", tuple(round(m) for m in modes))
        k = self._kernels.get(key)
        if k is None:
            sr = self.sr
            n_m = len(modes)
            k = {
                # knock: bore modes, ~3 ms, in sine and cosine phase so each
                # hit can carry its own random phase as a two-weight mix
                "knock": sound_parts.ring_kernel(sr, modes, (0.003,) * n_m, tuple(1.0 / (i + 1) for i in range(n_m)), max_s=0.02),
                "knock_q": sound_parts.ring_kernel(sr, modes, (0.003,) * n_m, tuple(1.0 / (i + 1) for i in range(n_m)),
                                                   max_s=0.02, phase_rad=math.pi / 2),
                # pre-ignition: the whole early burn drives the modes, the
                # lowest dominates, ~18 ms, plus a low thud through the block
                "preignition": sound_parts.ring_kernel(sr, tuple(modes) + (95.0,), (0.018,) * n_m + (0.03,),
                                                       tuple(0.7 / (i + 1) ** 1.5 for i in range(n_m)) + (0.5,), max_s=0.09),
            }
            self._kernels[key] = k
        return k

    # ------------------------------------------------------------------
    # crank-clocked events
    # ------------------------------------------------------------------
    @staticmethod
    def _crossings(theta: np.ndarray, theta_prev: float, angles: np.ndarray, cycle: float) -> tuple[np.ndarray, np.ndarray]:
        """(slot index, sample index) of every crossing of each slot
        angle by the per-sample crank clock in this block."""
        if not len(angles):
            return np.zeros(0, int), np.zeros(0, int)
        k = np.floor((theta[None, :] - angles[:, None]) / cycle)
        k_prev = np.floor((theta_prev - angles) / cycle)
        d = np.diff(np.concatenate([k_prev[:, None], k], axis=1), axis=1)
        slots, samples = np.nonzero(d > 0)
        return slots, samples

    def _render_events(self, engine, theta, theta_prev, deg_per_s, rpm, throttle, load_frac, n,
                       slot_records, slot_angles_deg, held_open, map_frac, jake, knock_ring_hz, exhaust_temp_k,
                       intake_flow_demand_frac, graph) -> tuple[np.ndarray, np.ndarray]:
        sr = self.sr
        arch = engine.architecture
        cycle = float(arch.cycle_degrees)
        n_slots = len(arch.firing_order)
        if not slot_records or len(slot_records) != n_slots:
            # a caller without the sim's records (an offline program):
            # every slot burning at the throttle's charge
            slot_records = tuple((0.35 + 0.65 * throttle, 0.0, 0.0, 0.0) for _ in range(n_slots))
        if not slot_angles_deg or len(slot_angles_deg) != n_slots:
            slot_angles_deg = tuple(i * cycle / n_slots for i in range(n_slots))
        angles = np.asarray(slot_angles_deg, dtype=np.float64)
        rec = np.asarray(slot_records, dtype=np.float64).reshape(n_slots, 4)
        strength, misfire, knock_i, preign = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
        k = self._event_kernels(engine, exhaust_temp_k)
        four_stroke = cycle >= 700.0
        evo_deg = 180.0 if four_stroke else 100.0
        open_deg = sound_parts.VALVE_OPEN_FRAC_OF_CYCLE * cycle
        rpm_frac = np.clip(rpm / max(engine.redline_rpm, 1.0), 0.0, 1.2)
        header = np.zeros(n)
        bay = np.zeros(n)

        # ---------------- the exhaust port opening: blowdown ----------------
        # amplitude per slot from what that cylinder really holds at EVO
        fired = strength > 0.0
        p_evo = np.where(fired, 0.3 + 0.7 * strength, MOTORED_BLOWDOWN_FRAC * max(0.0, map_frac))
        if held_open:
            p_evo = np.zeros(n_slots)          # the governor holds the valve open: no compression, no blowdown
        p_evo = p_evo * (0.4 + 0.6 * k["brightness"])
        s_idx, t_idx = self._crossings(theta, theta_prev, angles + evo_deg, cycle)
        if len(t_idx):
            train = np.zeros(n)
            np.add.at(train, t_idx, p_evo[s_idx])
            self._header_tail.add(self._conv(train, k["plosive"], ("plosive", id(k["plosive"]))) * 0.45)
            self._header_tail.add(self._conv(train, k["pipe"], ("pipe", id(k["pipe"]))) * 1.3)
            # the turbulent jet past the seat: one noise stream shaped by
            # the sum of every event's real valve-lift envelope
            env = sound_parts.blowdown_envelope_kernel(sr, deg_per_s, cycle)
            shaped = self._conv(train, env, ("bd_env", round(deg_per_s / 50.0)))
            raw = self.rng.standard_normal(len(shaped))
            hp = raw - np.convolve(raw, np.ones(8) / 8.0, mode="same")
            self._header_tail.add(hp * shaped * 0.28)
            # a misfire's unburnt charge going out the port: a soft chuff
            mis = misfire[s_idx] > 0.5
            if mis.any():
                ctrain = np.zeros(n)
                np.add.at(ctrain, t_idx[mis], 0.35 * max(0.0, map_frac))
                cenv = np.exp(-np.arange(int(sr * 0.15)) / (sr * 0.03))
                cshaped = self._conv(ctrain, cenv, ("chuff_env", 0))
                lp, self.misfire_noise_state = native_audio.leaky_integrator(self.rng.standard_normal(len(cshaped)), 0.06, self.misfire_noise_state)
                self._header_tail.add(lp * cshaped)

        # ---------------- the compression-release brake: a crack at TDC ----------------
        if jake:
            s_idx, t_idx = self._crossings(theta, theta_prev, angles, cycle)
            if len(t_idx):
                train = np.zeros(n)
                np.add.at(train, t_idx, JAKE_RELEASE_FRAC * max(0.2, map_frac))
                self._header_tail.add(self._conv(train, k["jake_crack"], ("jake", id(k["jake_crack"]))) * 1.3)
                self._header_tail.add(self._conv(train, k["pipe"], ("pipe", id(k["pipe"]))) * 0.8)
                cenv = np.exp(-np.arange(int(sr * 0.03)) / (sr * 0.006))
                shaped = self._conv(train, cenv, ("jake_env", 0))
                raw = self.rng.standard_normal(len(shaped))
                self._header_tail.add((raw - np.convolve(raw, np.ones(4) / 4.0, mode="same")) * shaped * 1.2)

        # ---------------- combustion: the casting rings, only where it burned ----------------
        s_idx, t_idx = self._crossings(theta, theta_prev, angles, cycle)
        if len(t_idx):
            burn = strength[s_idx]
            if (burn > 0.0).any():
                train = np.zeros(n)
                np.add.at(train, t_idx, burn * 1.0)
                self._bay_tail.add(self._conv(train, k["block"], ("block", id(k["block"]))))
            # detonation / pre-ignition on the slots the sim says knocked
            ki = knock_i[s_idx]
            hit = (ki > 0.0) & (self.rng.random(len(ki)) < 0.35 + 0.65 * ki)
            if hit.any():
                modes = knock_ring_hz or knock_ring_modes_hz(engine, exhaust_temp_k, load_frac)
                ck = self._chamber_kernels(modes)
                pre = preign[s_idx] > 0.5
                amp = 0.5 + 0.5 * ki
                phase = self.rng.uniform(0.0, 2 * np.pi, len(ki))
                ks = np.zeros(n); kc = np.zeros(n); kp = np.zeros(n)
                np.add.at(ks, t_idx[hit & ~pre], (amp * np.cos(phase))[hit & ~pre])
                np.add.at(kc, t_idx[hit & ~pre], (amp * np.sin(phase))[hit & ~pre])
                np.add.at(kp, t_idx[hit & pre], amp[hit & pre])
                ring = (self._conv(ks, ck["knock"], ("knock", id(ck["knock"]))) +
                        self._conv(kc, ck["knock_q"], ("knock_q", id(ck["knock_q"]))))
                # a marble on iron: the ring half-mixed with a matching
                # metallic burst, and the front's own click at the wall
                menv = np.exp(-np.arange(int(sr * 0.02)) / (sr * 0.003))
                mshaped = self._conv(np.abs(ks) + np.abs(kc), menv, ("knock_menv", 0))
                raw = self.rng.standard_normal(len(mshaped))
                metal = (raw - np.convolve(raw, np.ones(4) / 4.0, mode="same")) * mshaped
                click = self._conv(np.abs(ks) + np.abs(kc), k["plosive"], ("plosive", id(k["plosive"])))
                for comp, w in ((ring, 0.35), (metal, 0.25), (click, 0.6)):
                    self._bay_tail.add(comp * w)
                    self._header_tail.add(comp * (w * 0.5))
                if (hit & pre).any():
                    pk = self._conv(kp, ck["preignition"], ("preignition", id(ck["preignition"])))
                    self._bay_tail.add(pk)
                    self._header_tail.add(pk * 0.4)

        # ---------------- valve seats: every poppet valve closing ----------------
        if not arch.rotary and not arch.two_stroke:
            spring_ratio = engine.lifter_spring.spring_rate_n_per_mm / 45.0
            seat_angles = np.concatenate([angles - 140.0, angles + evo_deg + open_deg])
            s_idx, t_idx = self._crossings(theta, theta_prev, seat_angles, cycle)
            if len(t_idx):
                train = np.zeros(n)
                np.add.at(train, t_idx, 0.22 * (0.3 + 0.7 * rpm_frac) * min(2.0, spring_ratio ** 0.5))
                self._bay_tail.add(self._conv(train, k["seat"], ("seat", id(k["seat"]))))

        # ---------------- an injection pump's plunger, when the engine carries one ----------------
        if graph is not None and self._has_node(graph, "powertrain.injection_pump"):
            s_idx, t_idx = self._crossings(theta, theta_prev, angles - 15.0, cycle)
            if len(t_idx):
                train = np.zeros(n)
                np.add.at(train, t_idx, 0.35 * np.where(strength[s_idx] > 0.0, 1.0, 0.15))
                self._bay_tail.add(self._conv(train, k["injector"], ("injector", id(k["injector"]))))

        # ---------------- the intake ports: real gated flow roar ----------------
        # each slot's intake opens half a cycle before its firing angle
        # (a four-stroke's induction stroke; a two-stroke's transfer
        # window) for a real fraction of the cycle; the roar is the sum
        # of every open port's raised-cosine window, loud with the real
        # port restriction (intake_flow_demand_frac), not the pedal
        open_frac = 0.35 if four_stroke else 0.22
        offs = angles - (cycle / 2.0 if four_stroke else 120.0)
        frac = ((theta[None, :] - offs[:, None]) % cycle) / cycle
        gate = np.where(frac < open_frac, 0.5 - 0.5 * np.cos(2 * np.pi * frac / open_frac), 0.0).sum(axis=0)
        dthrottle = throttle - self._prev_throttle_for_intake
        self._prev_throttle_for_intake = throttle
        raw = self.rng.standard_normal(n)
        filtered, self.intake_noise_state = native_audio.leaky_integrator(raw, 0.25, self.intake_noise_state)
        roar_level = 0.05 + 0.55 * min(1.3, max(0.0, intake_flow_demand_frac)) + 2.0 * min(0.3, abs(dthrottle))
        bay += filtered * roar_level * gate

        # ---------------- exhaust flow hiss: continuous gas flow, small ----------------
        raw = self.rng.standard_normal(n)
        hiss, self.hiss_state = native_audio.leaky_integrator(raw, 0.15, self.hiss_state)
        header += hiss * (0.02 + 0.10 * throttle * rpm_frac) * float(np.mean(p_evo))
        return header, bay

    @staticmethod
    def _has_node(graph, identity) -> bool:
        cache = graph.setdefault("_sound_node_ids", None)
        if cache is None:
            cache = {n["identity"] for n in graph.get("nodes", ())}
            graph["_sound_node_ids"] = cache
        return identity in cache

    def _render_free_piston_events(self, engine, n, real_fire_hz, fire_event_count, exhaust_temp_k, knock_ring_hz, load_frac):
        """The Otto-Langen kind: no crank slots; each real ignition the
        sim counted is one event (the free piston's charge lighting: a
        big low thud in the frame) and, 0.3 s later on the piston's
        return, the exhaust valve dumping the spent charge."""
        k = self._event_kernels(engine, exhaust_temp_k)
        if self._fire_count_seen is None:
            self._fire_count_seen = fire_event_count
        new = fire_event_count - self._fire_count_seen
        self._fire_count_seen = fire_event_count
        if new <= 0 and fire_event_count == 0 and real_fire_hz > 0.0:
            # a caller with no counter: fall back to the measured rate
            self._free_piston_phase += real_fire_hz * n / self.sr
            new = int(self._free_piston_phase)
            self._free_piston_phase -= new
        if new > 0:
            offs = (np.arange(new) * n / new).astype(int)
            train = np.zeros(n)
            train[offs] = 1.0
            self._bay_tail.add(self._conv(train, k["block"], ("block", id(k["block"]))) * 0.9)
            delay = np.zeros(int(self.sr * 0.3))
            dtrain = np.concatenate([delay, train])
            self._header_tail.add(self._conv(dtrain, k["plosive"], ("plosive", id(k["plosive"]))) * 0.8)
            self._header_tail.add(self._conv(dtrain, k["pipe"], ("pipe", id(k["pipe"]))) * 0.8)
            cenv = np.exp(-np.arange(int(self.sr * 0.2)) / (self.sr * 0.05))
            shaped = self._conv(dtrain, cenv, ("fp_env", 0))
            raw = self.rng.standard_normal(len(shaped))
            self._header_tail.add((raw - np.convolve(raw, np.ones(8) / 8.0, mode="same")) * shaped * 0.5)
        return np.zeros(n), np.zeros(n)

    # ------------------------------------------------------------------
    def _render_header_electric(self, engine, rpm, throttle, load_frac, n_frames) -> np.ndarray:
        sr = self.sr
        base = max(rpm, 0.0) / 60.0
        mults = (6, 12, 18)
        sig = np.zeros(n_frames)
        steps = np.arange(1, n_frames + 1)
        for i, mult in enumerate(mults):
            freq = base * mult
            phase_inc = 2 * np.pi * freq / sr
            phases = self.motor_phase[i] + phase_inc * steps
            amp = 0.6 if i == 0 else 0.25 / i
            sig += amp * np.sin(phases)
            self.motor_phase[i] = phases[-1] % (2 * np.pi)

        whine = 0.05 + 0.3 * throttle + 0.2 * load_frac
        rpm_frac = np.clip(rpm / engine.redline_rpm, 0.0, 1.2)
        loudness = 0.2 + 0.8 * (0.3 + 0.7 * rpm_frac)
        return sig * whine * loudness * 1.5

    def _render_header_turbine(self, engine, n1_frac, throttle, load_frac, n_frames) -> np.ndarray:
        """A real gas turbine's own compressor/turbine spool whine --
        pitched off N1 (turbo_spool_frac, the gas-generator's own real
        shaft-speed fraction), not the output shaft rpm: the dominant
        real tone is the blade-pass note, set by how fast that spool
        actually turns, independent of the real reduction gearbox
        (engines.TurbineSpec.reduction_ratio) to the output shaft."""
        sr = self.sr
        n1 = max(n1_frac, 0.0)
        base_freq = 220.0 + 3200.0 * n1
        mults = (1.0, 2.0, 3.0)
        sig = np.zeros(n_frames)
        steps = np.arange(1, n_frames + 1)
        for i, mult in enumerate(mults):
            freq = base_freq * mult
            phase_inc = 2 * np.pi * freq / sr
            phases = self.turbine_whine_phase[i] + phase_inc * steps
            amp = 0.6 if i == 0 else 0.3 / i
            sig += amp * np.sin(phases)
            self.turbine_whine_phase[i] = phases[-1] % (2 * np.pi)

        noise = self.rng.standard_normal(n_frames)
        filtered, self.turbine_noise_state = native_audio.leaky_integrator(noise, 0.2, self.turbine_noise_state)
        roar_level = 0.05 + 0.30 * n1 + 0.15 * throttle
        sig += filtered * roar_level
        loudness = 0.25 + 0.75 * n1
        return sig * loudness

    # ------------------------------------------------------------------
    # the pitched parts: one sine bank over the whole catalogue
    # ------------------------------------------------------------------
    def _catalogue_for(self, engine, graph):
        key = (id(engine), id(graph) if graph is not None else None)
        if key != self._catalogue_key:
            self._catalogue = sound_parts.build_pitched_catalogue(engine, graph) if graph is not None else self._fallback_catalogue(engine)
            self._catalogue_key = key
        return self._catalogue

    @staticmethod
    def _fallback_catalogue(engine):
        """No graph in hand (an offline caller): the engine's declared
        accessories at the disclosed typical ratios and pass counts."""
        acc = engine.accessories
        nodes, edges = [], []

        def add(ident, ratio, kind="accessory-drive-belt"):
            nodes.append({"identity": ident, "kind": "rotating-mass", "reference_position": [0.0, 0.0, 0.0]})
            edges.append({"identity": f"drv.{ident}", "constraint": kind, "a": "powertrain.engine", "b": ident, "ratio": ratio})
        if engine.architecture.cylinders > 0:
            add("powertrain.oil_pump", 1.0, "geared-timing-drive")
            if acc.water_pump:
                add("powertrain.water_pump", 1.1)
            if acc.mechanical_fan:
                add("powertrain.cooling_fan", 1.0)
            if acc.alternator:
                add("electrical.alternator", 2.6)
            fi = engine.forced_induction
            if fi.kind == "supercharger":
                add("supercharger_rotor", fi.belt_ratio)
            elif fi.kind == "turbo":
                for i in range(max(1, int(fi.turbo_count))):
                    nodes.append({"identity": "powertrain.turbocharger" + ("" if i == 0 else f"_{i + 1}"), "kind": "rotating-mass",
                                  "inertia_kg_m2": 1e-4 * max(0.3, engine.displacement_l / 2.0), "reference_position": [0.0, 0.0, 0.0]})
        elif acc.coolant_pump:
            add("powertrain.coolant_pump", 1.3)
        nodes.append({"identity": "powertrain.engine", "kind": "rotating-mass", "reference_position": [0.0, 0.0, 0.0]})
        return sound_parts.build_pitched_catalogue(engine, {"nodes": nodes, "edges": edges})

    def _render_engine_bay(self, engine, rpm, throttle, load_frac, boost_frac, turbo_spool_frac, n_frames, header, graph) -> np.ndarray:
        sr = self.sr
        crank_hz = max(rpm, 0.0) / 60.0
        rpm_frac = float(np.clip(rpm / max(engine.redline_rpm, 1.0), 0.0, 1.2))
        cat = self._catalogue_for(engine, graph)
        keys, freqs, amps = [], [], []
        nyq = 0.45 * sr
        for part in cat.parts:
            a = part.amp_law(rpm_frac, throttle, load_frac, boost_frac)
            for vi, voice in enumerate(part.voices):
                f = crank_hz * part.order * (2.0 ** ((voice.detune_cents + part.wear) / 1200.0))
                if f >= nyq:
                    continue
                keys.append((part.name, vi)); freqs.append(f); amps.append(a * voice.amp_scale)
        for t in cat.turbos:
            shaft = max(0.0, turbo_spool_frac) * t.rated_shaft_hz
            if 0.0 < shaft < nyq:
                keys.append((t.identity, 0)); freqs.append(shaft); amps.append(0.35 * turbo_spool_frac ** 2)
            bp = shaft * t.blades
            if 0.0 < bp < nyq:
                keys.append((t.identity, 1)); freqs.append(bp); amps.append(0.12 * turbo_spool_frac ** 2)
        sig = self._bank.render(tuple(keys), np.asarray(freqs, dtype=np.float64), np.asarray(amps, dtype=np.float64), n_frames, sr)

        # muffled bleed of the header note through the block and firewall
        bleed, self.bleed_state = native_audio.leaky_integrator(header, 0.06, self.bleed_state)
        sig = sig + bleed * 0.35 + header * 0.6 * getattr(self, "_exhaust_open", 0.0)
        loudness = 0.3 + 0.5 * throttle + 0.2 * load_frac
        return sig * loudness

    def _render_induction(self, engine, n_frames, turbo_spool_frac, wastegate_flutter, surge_active) -> np.ndarray:
        """The non-tonal induction-side events a turbo/blower adds (its
        whistle itself is a pitched part in the bay bank): wastegate
        flutter and compressor surge."""
        sr = self.sr
        steps = np.arange(1, n_frames + 1)
        sig = np.zeros(n_frames)
        fi = engine.forced_induction
        if fi.kind == "turbo" and wastegate_flutter:
            noise = self.rng.standard_normal(n_frames)
            gate_phase = self.wastegate_phase + 2 * np.pi * 40.0 * steps / sr
            gate = (np.sin(gate_phase) > 0.3).astype(np.float64)
            sig += 0.15 * noise * gate
            self.wastegate_phase = gate_phase[-1] % (2 * np.pi)
        if surge_active and fi.kind != "none":
            lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 9.0 * steps / sr)
            sig += 0.3 * lfo * self.rng.standard_normal(n_frames) * 0.2
        if engine.kind == "turbine":
            # the machine's own compressor: continuous flow, no port events
            raw = self.rng.standard_normal(n_frames)
            filtered, self.intake_noise_state = native_audio.leaky_integrator(raw, 0.25, self.intake_noise_state)
            sig += filtered * (0.05 + 0.35 * turbo_spool_frac)
            whine_freq = 500.0 + 6000.0 * turbo_spool_frac
            phases = self.turbo_phase + 2 * np.pi * whine_freq / sr * steps
            sig += 0.30 * turbo_spool_frac * np.sin(phases)
            self.turbo_phase = phases[-1] % (2 * np.pi)
        return sig

    def _render_intake_pop(self, engine, n_frames, active, strength) -> np.ndarray:
        """An intake flashback: the flame back through an open intake
        valve into the plenum -- it rings the intake's own Helmholtz
        resonator (engines.IntakeSystem.tuned_frequency_hz, the plenum
        cavity on its runner neck) as a low 'whoof'/cough with a
        low-passed gust of noise, no sharp crack (nothing is open to the
        air but the filter). Heard in the bay, not the header."""
        if active:
            self.intake_pop_env = max(self.intake_pop_env, 0.5 + 0.5 * min(1.0, strength))
        if self.intake_pop_env <= 1e-4:
            return np.zeros(n_frames)
        sr = self.sr
        f = max(30.0, min(400.0, engine.intake_system.tuned_frequency_hz()))
        steps = np.arange(1, n_frames + 1)
        phases = self.intake_pop_phase + 2 * np.pi * f / sr * steps
        self.intake_pop_phase = phases[-1] % (2 * np.pi)
        raw = self.rng.standard_normal(n_frames)
        gust, self.intake_pop_noise_state = native_audio.leaky_integrator(raw, 0.08, self.intake_pop_noise_state)
        idx = np.arange(n_frames)
        ring_decay = np.exp(-idx / (sr * 0.09))
        gust_decay = np.exp(-idx / (sr * 0.04))
        sig = self.intake_pop_env * (0.7 * np.sin(phases) * ring_decay + 0.9 * gust * gust_decay)
        self.intake_pop_env *= math.exp(-n_frames / (sr * 0.09))
        return sig

    def _render_backfire(self, n_frames, active, strength, engine=None, kind=None,
                         exhaust_temp_k: float = 293.15) -> np.ndarray:
        """A gunshot-like bang, not a soft whoosh -- three components on
        three different decay times: a single-sample hard click at the
        trigger instant, a fast highpassed broadband crack (~8ms), and
        the pipe's own quarter-wave boom ringing on (~50ms). Triggered by
        engine_cycle_sim's planned (anti-lag) or unplanned events."""
        sr = self.sr
        if active and kind == "intake-flashback":
            self.backfire_kind_live = kind
            active = False
        was_idle = self.backfire_boom_env <= 1e-4 and self.backfire_crack_env <= 1e-4
        if active:
            level = 0.6 + 0.4 * min(1.0, strength)
            self.backfire_boom_env = max(self.backfire_boom_env, level)
            self.backfire_crack_env = max(self.backfire_crack_env, level)
        is_new_trigger = active and was_idle
        if self.backfire_boom_env <= 1e-4 and self.backfire_crack_env <= 1e-4:
            return np.zeros(n_frames)

        steps = np.arange(1, n_frames + 1)
        idx = np.arange(n_frames)
        # the bang rings the EXHAUST PIPE at its own quarter-wave
        # resonance at the live gas temperature and its odd harmonics; a
        # muffler swallows the crack and leaves the boom
        if engine is not None:
            f0 = max(25.0, min(600.0, engine.exhaust_system.tuned_frequency_hz(exhaust_temp_k)))
            brightness = engine.exhaust_system.brightness_frac
        else:
            f0, brightness = 130.0, 1.0
        thump = np.zeros(n_frames)
        for i, mult in enumerate((1.0, 3.0, 5.0)):
            ph = self.backfire_pipe_phase[i] + 2 * np.pi * f0 * mult / sr * steps
            thump += np.sin(ph) / (i + 1) ** 1.3
            self.backfire_pipe_phase[i] = ph[-1] % (2 * np.pi)

        raw_noise = self.rng.standard_normal(n_frames)
        lowpassed, self.backfire_noise_lp_state = native_audio.leaky_integrator(
            raw_noise, 0.5, self.backfire_noise_lp_state)
        crack_noise = raw_noise - lowpassed

        boom_tau_s = 0.05
        crack_tau_s = 0.008
        boom_decay = np.exp(-idx / (sr * boom_tau_s))
        crack_decay = np.exp(-idx / (sr * crack_tau_s))

        crack_gain = 0.8 * (0.25 + 0.75 * brightness)
        boom_gain = 0.45 * (1.3 - 0.3 * brightness)
        sig = thump * boom_gain * self.backfire_boom_env * boom_decay
        sig = sig + crack_noise * crack_gain * self.backfire_crack_env * crack_decay
        if is_new_trigger:
            sig[0] += 1.5 * (0.6 + 0.4 * min(1.0, strength)) * (0.3 + 0.7 * brightness)

        self.backfire_boom_env *= math.exp(-n_frames / (sr * boom_tau_s))
        self.backfire_crack_env *= math.exp(-n_frames / (sr * crack_tau_s))
        return sig
