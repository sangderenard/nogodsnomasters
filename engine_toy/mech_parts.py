"""Composable mechanical noise-kernel parts.

The primitive is a `Source` (a positioned impulse or sine emitter --
unchanged from the original engine-only baker). What's new is that
sources are no longer hand-assembled inside one big engine function:
each mechanical mechanism is its own `Part`, and an `Assembly` is just
a list of parts plus a set of named listening points.

An engine block is one Assembly built from CombustionCrankPart +
ValvetrainPart + a handful of RotaryTonePart accessories (see
engine_baker.py). A transfer case is a *different* Assembly built from
nothing but RotaryTonePart/GearMeshPart gear stages -- no combustion,
no crank, no cylinders at all (see transfer_case.py). Both bake through
the exact same `bake_points` renderer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import math
import wave

import numpy as np

Vec3 = tuple[float, float, float]


# The exhaust blowdown's real shape: the valve's own lift curve (the same
# half-sine over VALVE_OPEN_FRAC_OF_CYCLE the valvetrain is animated
# with -- valvetrain_parts.py) opens a curtain area, and the cylinder
# pressure behind it collapses over BLOWDOWN_DECAY_DEG; the jet noise
# follows their product. Used by the bake AND the live synth.
VALVE_OPEN_FRAC_OF_CYCLE = 0.32
BLOWDOWN_DECAY_DEG = 55.0


def blowdown_envelope(theta_deg, cycle_degrees: float = 720.0,
                      open_frac: float = VALVE_OPEN_FRAC_OF_CYCLE, decay_deg: float = BLOWDOWN_DECAY_DEG):
    """0..1 turbulent-flow envelope vs crank degrees after the exhaust
    valve starts to open (array or scalar). Zero before EVO and after
    the valve seats again."""
    th = np.asarray(theta_deg, dtype=np.float64)
    open_deg = open_frac * cycle_degrees
    lift = np.where((th >= 0.0) & (th < open_deg), np.sin(np.pi * np.clip(th, 0.0, open_deg) / open_deg), 0.0)
    pressure = np.exp(-np.maximum(th, 0.0) / decay_deg)
    env = lift * pressure
    peak = float(np.max(blowdown_envelope_peak(open_deg, decay_deg)))
    return env / max(peak, 1e-9)


def blowdown_envelope_peak(open_deg: float, decay_deg: float) -> np.ndarray:
    th = np.linspace(0.0, open_deg, 64)
    return np.sin(np.pi * th / open_deg) * np.exp(-th / decay_deg)


@dataclass
class Source:
    name: str
    kind: str               # "impulse" (per-event, damped ring) or "sine" (continuous order)
    domain: str              # "structural" | "airborne" | "both"
    position: Vec3
    amplitude: float
    ring_freq_hz: float = 260.0
    ring_tau_s: float = 0.03
    frequency_hz: float = 0.0     # for "sine" kind
    phase_rad: float = 0.0
    event_times_s: tuple[float, ...] = ()    # for "impulse" kind, within one cycle
    # an impulse event's two real non-tonal parts -- the PLOSIVE (the
    # one-sided pressure step of the exhaust valve cracking open on a
    # still-hot, still-pressurised cylinder, or of a detonation front
    # hitting the wall: a single sharp click, plosive_tau_s ~1-2 ms) and
    # the TURBULENT blowdown that follows it (the charge jetting past
    # the valve seat: seeded broadband noise dying over the blowdown,
    # noise_tau_s, tens of ms). 0 = neither (a plain damped ring).
    plosive_tau_s: float = 0.0
    plosive_amp: float = 0.0
    noise_tau_s: float = 0.0
    noise_amp: float = 0.0
    noise_seed: int = 0
    # when > 0, the noise follows the real valve-lift blowdown envelope
    # (blowdown_envelope) over this many crank degrees of opening with
    # its own pressure-decay, instead of a bare exponential
    valve_open_s: float = 0.0
    blowdown_decay_s: float = 0.0


@dataclass
class Voice:
    """One instance of a multi-voice kernel: a detuned/offset copy of
    its part's base emission. A belt with two strands, a head with two
    resonant ring modes, a gear mesh heard through two different
    contact-tooth phases -- all just N voices of one Part instead of N
    separate parts."""
    detune_cents: float = 0.0
    phase_offset_rad: float = 0.0
    amp_scale: float = 1.0
    position_offset: Vec3 = (0.0, 0.0, 0.0)


def _detune_ratio(cents: float) -> float:
    return 2.0 ** (cents / 1200.0)


class Part:
    """Base class for a mechanical noise kernel. `enabled=False` removes
    it from an assembly's output entirely -- this is how "cylinders and
    crank and detonation" gets turned off for a non-engine assembly:
    the CombustionCrankPart just isn't in the parts list.

    A Part is allowed real persistent state (running integrators, wear,
    phase history, misfire/knock memory): `step()` advances that state
    forward in time; `sources()` only ever reads current state to emit
    this instant's Source list. Baking a loop with `Assembly.advance()`
    ticks every part's `step()` across the loop and records what
    `sources()` says at each tick, rather than evaluating one closed-
    form formula for the whole buffer -- so a kernel's own policy
    (how it integrates, what it remembers) is what shapes the bake.
    """
    name: str = "part"
    enabled: bool = True

    def __post_init_state__(self) -> None:
        self.state: dict = {}

    def step(self, dt: float, input_rpm: float, throttle: float, load_frac: float) -> None:
        """Advance persistent internal state. Default no-op -- most
        accessory tones are stateless functions of the instant input."""
        return None

    def sources(self, input_rpm: float, throttle: float, load_frac: float) -> list[Source]:
        raise NotImplementedError

    def period_s(self, input_rpm: float) -> float | None:
        """Natural repeat period at this input speed, or None if the
        part has no event structure that needs one (a plain continuous
        tone loops cleanly at any period once frequency-snapped)."""
        return None


@dataclass
class CombustionCrankPart(Part):
    """Cylinders + crank + detonation, bundled as one togglable unit --
    an idealized, perfectly-even-firing model. For the live interactive
    toy this gets superseded by engine_cycle_sim.py's real per-event
    crank-domain integration; this idealized version is what the
    offline snapshot baker (engine_baker.bake_*) uses."""
    name: str = "combustion-crank"
    enabled: bool = True
    cylinders: int = 0
    firing_order: tuple[int, ...] = ()
    cycle_degrees: float = 720.0
    cylinder_positions: dict[int, Vec3] = field(default_factory=dict)
    strength_fn: Callable[[float, float, float], float] = lambda rpm, thr, load: 0.0
    imbalance_position: Vec3 = (0.0, 0.0, 0.0)
    primary_amp_fn: Callable[[float, float, float], float] = lambda rpm, thr, load: 0.0
    secondary_amp_fn: Callable[[float, float, float], float] = lambda rpm, thr, load: 0.0
    ring_freq_hz: float = 320.0
    ring_tau_s: float = 0.022
    # multiple resonant "ring" modes per combustion event (block/head
    # structural modes), each an independently detuned/offset voice
    ring_voices: tuple[Voice, ...] = (Voice(),)

    def period_s(self, input_rpm: float) -> float:
        return self.cycle_degrees / 360.0 * 60.0 / max(input_rpm, 1.0)

    def sources(self, input_rpm, throttle, load_frac) -> list[Source]:
        cycle_time_s = self.period_s(input_rpm)
        sec_per_deg = cycle_time_s / self.cycle_degrees
        crank_freq_hz = max(input_rpm, 1.0) / 60.0
        strength = self.strength_fn(input_rpm, throttle, load_frac)
        step_deg = self.cycle_degrees / max(len(self.firing_order), 1)

        out = []
        for slot, cyl in enumerate(self.firing_order):
            pos = self.cylinder_positions.get(cyl, (0.0, 0.0, 0.0))
            t_event = (slot * step_deg * sec_per_deg) % cycle_time_s
            for voice in self.ring_voices:
                vpos = (pos[0] + voice.position_offset[0], pos[1] + voice.position_offset[1],
                        pos[2] + voice.position_offset[2])
                time_offset = voice.phase_offset_rad / (2 * math.pi) * cycle_time_s
                out.append(Source(
                    name=f"cyl{cyl}-combustion", kind="impulse", domain="both", position=vpos,
                    amplitude=strength * voice.amp_scale,
                    ring_freq_hz=self.ring_freq_hz * _detune_ratio(voice.detune_cents),
                    ring_tau_s=self.ring_tau_s, event_times_s=((t_event + time_offset) % cycle_time_s,),
                ))
            # the exhaust blowdown: the exhaust valve cracks open ~180
            # crank degrees after ignition on a still-pressurised
            # cylinder (the layout's own EVO), and THAT plosive plus the
            # turbulent jet past the seat is the real airborne exhaust
            # event -- not the combustion ring, which is what the block
            # carries. The blowdown lasts ~55 crank degrees.
            evo_deg = 180.0 if self.cycle_degrees >= 700.0 else 100.0
            blowdown_deg = 55.0 if self.cycle_degrees >= 700.0 else 45.0
            out.append(Source(
                name=f"cyl{cyl}-blowdown", kind="impulse", domain="airborne", position=pos,
                amplitude=0.0, ring_freq_hz=self.ring_freq_hz, ring_tau_s=self.ring_tau_s,
                event_times_s=((t_event + evo_deg * sec_per_deg) % cycle_time_s,),
                plosive_tau_s=0.0014, plosive_amp=0.9 * strength,
                noise_tau_s=blowdown_deg * sec_per_deg, noise_amp=0.55 * strength * (0.4 + 0.6 * throttle),
                noise_seed=1000 + cyl,
                valve_open_s=VALVE_OPEN_FRAC_OF_CYCLE * self.cycle_degrees * sec_per_deg,
                blowdown_decay_s=BLOWDOWN_DECAY_DEG * sec_per_deg,
            ))
        out.append(Source(
            name="primary-imbalance", kind="sine", domain="structural", position=self.imbalance_position,
            amplitude=self.primary_amp_fn(input_rpm, throttle, load_frac), frequency_hz=crank_freq_hz,
        ))
        out.append(Source(
            name="secondary-imbalance", kind="sine", domain="structural", position=self.imbalance_position,
            amplitude=self.secondary_amp_fn(input_rpm, throttle, load_frac), frequency_hz=crank_freq_hz * 2.0,
        ))
        return out


@dataclass
class ValvetrainPart(Part):
    name: str = "valvetrain"
    enabled: bool = True
    cylinder_positions: dict[int, Vec3] = field(default_factory=lambda: {})
    cam_order: float = 0.5     # camshaft turns at half crank speed, four-stroke
    amp_fn: Callable[[float, float, float], float] = lambda rpm, thr, load: 0.0
    ring_freq_hz: float = 520.0
    ring_tau_s: float = 0.008
    cycle_degrees: float = 720.0

    def period_s(self, input_rpm: float) -> float:
        return self.cycle_degrees / 360.0 * 60.0 / max(input_rpm, 1.0)

    def sources(self, input_rpm, throttle, load_frac) -> list[Source]:
        cycle_time_s = self.period_s(input_rpm)
        amp = self.amp_fn(input_rpm, throttle, load_frac)
        out = []
        for cyl, pos in self.cylinder_positions.items():
            out.append(Source(
                name=f"cyl{cyl}-valvetrain", kind="impulse", domain="both", position=pos,
                amplitude=amp, ring_freq_hz=self.ring_freq_hz, ring_tau_s=self.ring_tau_s,
                event_times_s=(0.0, cycle_time_s / 2.0),
            ))
        return out


@dataclass
class RotaryTonePart(Part):
    """Any continuously-spinning noisemaker driven at some order (or
    ratio) of the assembly's input rpm: an oil/water pump, a fan, an
    alternator, an EV inverter's switching tone, or a gearbox/transfer-
    case mesh (tooth count folded into `order` as
    local_rpm/60 * tooth_count -- with typical tooth counts a gear mesh
    is well approximated as a single tone, not per-tooth impulses).

    `voices` lets one kernel stand in for several coupled emitters at
    once -- e.g. a belt's two strands beating slightly out of tune, or
    a gear mesh heard through two contact phases -- without hand-adding
    separate Part instances. `wear` is small persistent state: it
    drifts under sustained load and slowly detunes the tone, so two
    bakes of the same part at the same instantaneous input can differ
    once the part has been run for different lengths of time.
    """
    name: str = "rotary"
    enabled: bool = True
    position: Vec3 = (0.0, 0.0, 0.0)
    domain: str = "both"
    order: float = 1.0          # multiple of input_rpm; ignored if fixed_freq_hz > 0 or freq_fn set
    fixed_freq_hz: float = 0.0
    freq_fn: Callable[[float, float, float], float] | None = None   # overrides order/fixed_freq_hz when set
    amp_fn: Callable[[float, float, float], float] = lambda rpm, thr, load: 0.0
    phase_rad: float = 0.0
    voices: tuple[Voice, ...] = (Voice(),)
    wear_rate_per_s: float = 0.0     # cents of detune drift per second under full load
    wear: float = 0.0                 # persistent state: accumulated detune, cents

    def step(self, dt, input_rpm, throttle, load_frac) -> None:
        if self.wear_rate_per_s:
            self.wear += self.wear_rate_per_s * load_frac * dt

    def sources(self, input_rpm, throttle, load_frac) -> list[Source]:
        if self.freq_fn is not None:
            base_freq = self.freq_fn(input_rpm, throttle, load_frac)
        else:
            base_freq = self.fixed_freq_hz if self.fixed_freq_hz > 0 else max(input_rpm, 0.0) / 60.0 * self.order
        amp = self.amp_fn(input_rpm, throttle, load_frac)
        out = []
        for voice in self.voices:
            pos = (self.position[0] + voice.position_offset[0], self.position[1] + voice.position_offset[1],
                   self.position[2] + voice.position_offset[2])
            freq = base_freq * _detune_ratio(voice.detune_cents + self.wear)
            out.append(Source(
                name=self.name, kind="sine", domain=self.domain, position=pos,
                amplitude=amp * voice.amp_scale, frequency_hz=freq,
                phase_rad=self.phase_rad + voice.phase_offset_rad,
            ))
        return out


@dataclass
class Assembly:
    """A named collection of parts plus the named 3D points that listen
    to them. `input_rpm` is whatever drives the assembly -- crankshaft
    rpm for an engine block, transfer-case input-shaft rpm for a
    gearbox, etc; each part interprets it in its own terms (an order,
    a ratio, a fixed frequency)."""
    name: str
    parts: list[Part]
    acoustic_points: dict[str, Vec3] = field(default_factory=dict)
    structural_points: dict[str, Vec3] = field(default_factory=dict)

    def advance(self, dt: float, input_rpm: float, throttle: float, load_frac: float) -> None:
        """Tick every enabled part's persistent state forward by dt."""
        for part in self.parts:
            if part.enabled:
                part.step(dt, input_rpm, throttle, load_frac)

    def build_sources(self, input_rpm: float, throttle: float, load_frac: float) -> tuple[list[Source], float]:
        """Snapshot: read each part's current (possibly stateful) output
        without advancing anything. Call `advance()` first if the parts'
        persistent state (wear, integrators, memory) should evolve."""
        sources: list[Source] = []
        periods = []
        for part in self.parts:
            if not part.enabled:
                continue
            sources.extend(part.sources(input_rpm, throttle, load_frac))
            p = part.period_s(input_rpm)
            if p:
                periods.append(p)
        cycle_time_s = max(periods) if periods else 60.0 / max(input_rpm, 1.0)
        return sources, max(cycle_time_s, 1e-4)


def _wrapped_delta(t_grid: np.ndarray, t_event: float, period: float) -> np.ndarray:
    d = (t_grid - t_event) % period
    d = np.where(d > period / 2.0, d - period, d)
    return d


def bake_points(sources: list[Source], cycle_time_s: float, points: dict[str, Vec3],
                 domains: tuple[str, ...], atten_k: float,
                 cycles: int, sample_rate: int) -> tuple[dict[str, np.ndarray], float]:
    """Generic spatial renderer shared by every receiver set and every
    assembly kind. `domains` restricts which source domains contribute.
    Impulse events wrap modulo `cycle_time_s` (the assembly's chosen
    reference period); continuous sine sources have their frequency
    snapped to complete a whole number of cycles across the rendered
    loop, so every source -- regardless of its own natural order --
    closes with no seam. Returns (point -> raw samples, loop_duration_s).
    """
    loop_time_s = cycle_time_s * max(1, cycles)
    n_frames = max(int(round(loop_time_s * sample_rate)), 8)
    t_grid = np.arange(n_frames) / sample_rate

    out: dict[str, np.ndarray] = {name: np.zeros(n_frames) for name in points}
    relevant = [s for s in sources if s.domain in domains]
    for name, pos in points.items():
        acc = out[name]
        for src in relevant:
            dx = src.position[0] - pos[0]
            dy = src.position[1] - pos[1]
            dz = src.position[2] - pos[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            atten = 1.0 / (1.0 + dist * atten_k)
            if src.kind == "impulse":
                for t_event in src.event_times_s:
                    delta = _wrapped_delta(t_grid, t_event, cycle_time_s)
                    causal = delta >= 0.0
                    ring = np.zeros(n_frames)
                    d = delta[causal]
                    if src.amplitude != 0.0:
                        ring[causal] = np.exp(-d / src.ring_tau_s) * np.sin(2 * math.pi * src.ring_freq_hz * d)
                        acc += src.amplitude * atten * ring
                    if src.plosive_amp != 0.0 and src.plosive_tau_s > 0.0:
                        # a single one-sided pressure step: a quarter-wave
                        # click at 1/(4 tau), decaying on tau
                        click = np.zeros(n_frames)
                        click[causal] = np.exp(-d / src.plosive_tau_s) * np.cos(2 * math.pi * d / (4.0 * src.plosive_tau_s))
                        acc += src.plosive_amp * atten * click
                    if src.noise_amp != 0.0 and src.noise_tau_s > 0.0:
                        rng = np.random.default_rng(src.noise_seed)
                        noise = rng.standard_normal(n_frames)
                        # high-passed (a jet past a seat is hiss, not rumble)
                        noise = noise - np.convolve(noise, np.ones(8) / 8.0, mode="same")
                        env = np.zeros(n_frames)
                        if src.valve_open_s > 0.0:
                            # the real valve-lift-shaped blowdown
                            open_deg = 360.0
                            th = d / max(src.valve_open_s, 1e-9) * open_deg
                            env[causal] = blowdown_envelope(th, cycle_degrees=open_deg / VALVE_OPEN_FRAC_OF_CYCLE,
                                                            decay_deg=src.blowdown_decay_s / max(src.valve_open_s, 1e-9) * open_deg)
                        else:
                            env[causal] = np.exp(-d / src.noise_tau_s)
                        acc += src.noise_amp * atten * noise * env
            else:
                n_cycles = max(1, round(src.frequency_hz * loop_time_s))
                snapped_freq = n_cycles / loop_time_s if src.frequency_hz > 0 else 0.0
                acc += src.amplitude * atten * np.sin(2 * math.pi * snapped_freq * t_grid + src.phase_rad)
        out[name] = acc
    return out, loop_time_s


def normalize(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    peak = max((float(np.max(np.abs(s))) for s in signals.values() if s.size), default=1.0)
    peak = max(peak, 1e-6)
    return {name: np.tanh(sig / peak * 1.4).astype(np.float32) for name, sig in signals.items()}


def bake_assembly(assembly: Assembly, input_rpm: float, throttle: float, load_frac: float,
                   receiver: str, cycles: int, sample_rate: int,
                   normalize_output: bool = True) -> tuple[dict[str, np.ndarray], float]:
    """receiver: 'acoustic' (airborne+both @ assembly.acoustic_points),
    'structural' (structural+both @ assembly.structural_points)."""
    sources, cycle_time_s = assembly.build_sources(input_rpm, throttle, load_frac)
    if receiver == "acoustic":
        points, domains, atten_k = assembly.acoustic_points, ("airborne", "both"), 3.2
    else:
        points, domains, atten_k = assembly.structural_points, ("structural", "both"), 6.0
    raw, loop_s = bake_points(sources, cycle_time_s, points, domains, atten_k, cycles, sample_rate)
    return (normalize(raw) if normalize_output else raw), loop_s


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
