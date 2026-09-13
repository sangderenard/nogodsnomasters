"""The damage system's own synth: what a hit sounds like, what a hole
sounds like afterwards.

Two sources, both from real records, never from a keypress:

  IMPACTS -- every ballistics.ImpactResult the ray applied to the mesh
  (damage_state.record_penetration keeps them per part). The struck
  part's material and wall pick the kernel it rings with (a thin
  aluminium cover clangs high and dies fast; a cast-iron case is a low
  long bell; a plastic intake thuds), the damage mode picks the event
  on top of the ring:
    dent                a dull, low-passed thud, the wall belling in
    crater / spalling   a hard crack (the surface breaking) then the ring
    puncture            the click of the wall failing, the ring, and the
                        hole itself venting if the part was pressurised
                        (a "pop": a hiss burst decaying as the local
                        pressure equalises)
    ricochet            a click and the zing: the deflected projectile's
                        descending whistle
  and energy_spent_j sets how hard all of it is.

  BLASTS -- an explosive failure (a fuel volume igniting, a pressure
  vessel letting go): a low boom the size of the released energy, a
  crack, and a spray of debris rattle after it. blast() is the API;
  the sim raises it from its own failure events.

  EMITTERS -- hole_emitters.HoleEmitterField.summary(): every hole with
  something passing through it, continuously:
    spray   a high-passed hiss, level with jet speed x mass flow
    pour    low-passed rush with a slow glug (air displacing back in)
    drip    one plink per real drop (the emitter counts them by Tate's
            law), a damped two-mode ring, low-passed -- a drop on a pan
    gas     a leak whistle: band-passed noise pitched by the hole (jet
            screech, Strouhal ~0.2 at the gas sound speed over the
            hole diameter), level with mass flow
    ingest  the negative emitter's suction: the same orifice noise
            inward, lower and duller (the jet forms inside the volume,
            heard through its wall), level with the mass drawn in
    splash  the crank's dipper flinging oil at the bores: a patter of
            drops on iron at crank rate, through the case wall
  Every liquid sound is shaped by the emitter's flow CHARACTER
  (hole_emitters.flow_character: Reynolds and Weber numbers for that
  fluid at that speed and size): a laminar stream is smooth, a
  turbulent one rushes, droplets patter, a mist hisses.

Same machinery as engine_sound.py: trains convolved with fixed kernels
through cached FFTs, one noise stream shaped per class per block.
"""
from __future__ import annotations

import math

import numpy as np

import native_audio
import sound_parts
from engine_sound import _Convolver, _Tail

# material -> (ring frequency Hz at the reference wall, decay s, brightness 0..1)
MATERIAL_RING = {
    "cover": (1800.0, 0.040, 0.9), "intake": (600.0, 0.015, 0.4), "fuel": (2600.0, 0.070, 0.8),
    "lube": (2400.0, 0.060, 0.8), "ignition": (900.0, 0.010, 0.5), "exhaust": (2600.0, 0.090, 0.9),
    "head": (1400.0, 0.060, 0.7), "piston": (2100.0, 0.030, 0.7), "case": (900.0, 0.140, 0.6),
    "cylinder": (950.0, 0.150, 0.6), "flame": (0.0, 0.0, 0.0), "smoke": (0.0, 0.0, 0.0),
}
REFERENCE_WALL_M = 0.004


def _ring_for(material: str, wall_m: float) -> tuple[float, float, float]:
    f, tau, bright = MATERIAL_RING.get(material, (1200.0, 0.05, 0.6))
    if f <= 0.0:
        return 0.0, 0.0, 0.0
    # a thicker wall rings lower and longer (plate frequency ~ h / L^2 at
    # fixed span for the mode, but the struck spot is stiffer: net a mild
    # drop with thickness), disclosed as sqrt
    scale = math.sqrt(REFERENCE_WALL_M / max(wall_m, 0.0005))
    return f * min(2.0, max(0.5, scale)), tau / min(2.0, max(0.5, scale)), bright


class DamageSoundSynth:
    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate
        self.rng = np.random.default_rng(4242)
        self._tail = _Tail()
        self._conv = _Convolver()
        self._kernels: dict = {}
        self.pour_lp_state = 0.0
        self.spray_hp_state = 0.0
        self.gas_state = 0.0
        self.glug_phase = 0.0
        self._plink = None

    # ------------------------------------------------------------------
    def _kernel(self, key, build):
        k = self._kernels.get(key)
        if k is None:
            k = build()
            self._kernels[key] = k
        return k

    def impact(self, mode: str, material: str, wall_m: float, energy_spent_j: float, pressure_pa: float = 101_325.0,
               speed_after_m_s: float = 0.0, calibre_m: float = 0.0076) -> None:
        """Schedule one hit; successive hits of one shot land ~1 ms apart
        (the projectile's own travel between the parts it crossed)."""
        sr = self.sr
        self._stagger = (getattr(self, "_stagger", -1) + 1) % 8
        n_frames = 1 + int(sr * 0.001) * self._stagger
        level = min(1.5, 0.15 + 0.3 * math.log10(max(energy_spent_j, 1.0)))
        f, tau, bright = _ring_for(material, wall_m)
        train = np.zeros(n_frames); train[-1] = level
        if f > 0.0:
            ring = self._kernel(("ring", round(f), round(tau, 3)),
                                lambda: sound_parts.ring_kernel(sr, (f, f * 2.4, f * 4.1), (tau, tau * 0.5, tau * 0.25),
                                                                (1.0, 0.45 * bright, 0.2 * bright), max_s=min(0.5, tau * 5.0)))
            self._tail.add(self._conv(train, ring, ("ring", id(ring))) * 0.35)
        if mode == "dent":
            thud = self._kernel(("thud",), lambda: self._thud_kernel())
            self._tail.add(self._conv(train, thud, ("thud", id(thud))) * 0.6)
        elif mode in ("crater", "spalling-puncture", "puncture"):
            self._tail.add(self._burst(level * (0.6 + 0.4 * bright) * 0.4, 0.006, hp=True))
            click = self._kernel(("click",), lambda: sound_parts.plosive_kernel(sr, 0.0009))
            self._tail.add(self._conv(train, click, ("click", id(click))) * 0.5)
            if mode == "puncture" and pressure_pa > 120_000.0:
                # the pop: the wall opens on a pressurised volume
                self._tail.add(self._burst(min(1.2, 0.3 * math.log10(pressure_pa / 101_325.0) + 0.3), 0.08, hp=True))
        elif mode == "ricochet":
            click = self._kernel(("click",), lambda: sound_parts.plosive_kernel(sr, 0.0009))
            self._tail.add(self._conv(train, click, ("click", id(click))) * 0.4)
            self._tail.add(self._zing(level, speed_after_m_s, calibre_m))

    def blast(self, energy_j: float, volume_m3: float = 0.01) -> None:
        """An explosive failure: boom sized by the energy, crack, debris."""
        sr = self.sr
        level = min(2.0, 0.3 + 0.35 * math.log10(max(energy_j, 10.0) / 10.0))
        # the boom: a Helmholtz-ish low mode of the bay volume, long decay
        f_boom = max(18.0, min(70.0, 55.0 * (0.05 / max(volume_m3, 0.001)) ** 0.33))
        boom = self._kernel(("boom", round(f_boom)), lambda: sound_parts.ring_kernel(sr, (f_boom, f_boom * 2.1), (0.35, 0.12), (1.0, 0.3), max_s=1.2))
        train = np.zeros(1); train[0] = level
        self._tail.add(self._conv(train, boom, ("boom", id(boom))) * 1.4)
        self._tail.add(self._burst(level, 0.012, hp=True))
        # debris: sparse random clicks over the next half second
        n = int(sr * 0.6)
        deb = np.zeros(n)
        hits = self.rng.integers(int(sr * 0.03), n, size=int(12 + 20 * level))
        deb[hits] = self.rng.uniform(0.2, 1.0, len(hits)) * level * 0.5
        click = self._kernel(("click",), lambda: sound_parts.plosive_kernel(sr, 0.0009))
        self._tail.add(self._conv(deb, click, ("click", id(click))))

    # ------------------------------------------------------------------
    def _thud_kernel(self) -> np.ndarray:
        n = int(self.sr * 0.05)
        t = np.arange(n) / self.sr
        return np.exp(-t / 0.012) * np.sin(2 * np.pi * 140.0 * t)

    def _burst(self, level: float, tau_s: float, hp: bool) -> np.ndarray:
        n = max(8, int(self.sr * tau_s * 5.0))
        raw = self.rng.standard_normal(n)
        sig = raw - np.convolve(raw, np.ones(6) / 6.0, mode="same") if hp else np.convolve(raw, np.ones(12) / 12.0, mode="same")
        return sig * np.exp(-np.arange(n) / (self.sr * tau_s)) * level

    def _zing(self, level: float, speed_m_s: float, calibre_m: float) -> np.ndarray:
        """The deflected projectile's whistle: a descending chirp -- its
        tumbling, slowing body shedding vortices (Strouhal ~0.2 over the
        calibre) as it flies away, fading over ~150 ms."""
        n = int(self.sr * 0.18)
        t = np.arange(n) / self.sr
        v0 = max(80.0, speed_m_s)
        v = v0 * np.exp(-t / 0.25)
        f = np.minimum(0.2 * v / max(calibre_m, 0.003), 0.4 * self.sr)
        f = np.minimum(f, 6000.0)
        ph = np.cumsum(2 * np.pi * f / self.sr)
        return np.sin(ph) * np.exp(-t / 0.06) * level * 0.5

    # ------------------------------------------------------------------
    def render(self, n_frames: int, emitters: tuple = ()) -> np.ndarray:
        """One block: the scheduled tails plus the continuous emitter
        sounds for every hole currently passing something."""
        sr = self.sr
        sig = self._tail.take(n_frames)
        if not emitters:
            return sig
        spray_gain = 0.0; pour_gain = 0.0; gas_gain = 0.0; gas_f = 0.0; drips = []; ingest_gain = 0.0
        splash_gain = 0.0; patter = 0
        for ident, fluid, regime, mflow, vjet, ndrips, radius, *rest in emitters:
            character = rest[0] if rest else ""
            hiss_w = 1.0 if "mist" in character else 0.6 if "turbulent" in character else 0.25   # a laminar rope is nearly silent
            if regime == "spray":
                spray_gain += hiss_w * min(1.0, math.sqrt(max(vjet, 0.0) / 20.0) * min(1.0, mflow / 0.05))
            elif regime == "pour":
                pour_gain += (0.35 if "laminar" in character else 1.0) * min(1.0, mflow / 0.2)
            elif regime == "splash":
                splash_gain += min(1.0, mflow / 0.05)
                patter += 1
            elif regime == "drip":
                drips.extend([1.0] * int(ndrips))
            elif regime == "gas":
                g = min(1.0, mflow / 0.02)
                gas_gain += g
                gas_f += g * min(0.4 * sr, 0.2 * max(vjet, 50.0) / max(2.0 * radius, 0.002))
            elif regime == "ingest":
                ingest_gain += min(1.0, mflow / 0.01)
        raw = self.rng.standard_normal(n_frames)
        if spray_gain > 0.0:
            lp, self.spray_hp_state = native_audio.leaky_integrator(raw, 0.35, self.spray_hp_state)
            sig += (raw - lp) * 0.10 * min(1.5, spray_gain)
        if pour_gain > 0.0:
            lp, self.pour_lp_state = native_audio.leaky_integrator(raw, 0.08, self.pour_lp_state)
            steps = np.arange(1, n_frames + 1)
            glug = self.glug_phase + 2 * np.pi * 5.5 / sr * steps
            self.glug_phase = glug[-1] % (2 * np.pi)
            sig += lp * (0.6 + 0.4 * np.sin(glug)) * 0.3 * min(1.5, pour_gain)
        if gas_gain > 0.0:
            f = gas_f / gas_gain
            # a resonant band around the screech: a two-pole on the noise
            bp = self._bandpass(raw, f, 0.15)
            sig += bp * 0.18 * min(1.5, gas_gain)
        if ingest_gain > 0.0:
            lp, self.gas_state = native_audio.leaky_integrator(raw, 0.12, self.gas_state)
            sig += lp * 0.22 * min(1.5, ingest_gain)
        if splash_gain > 0.0:
            # drops on the bore/case at random within the block, dull through the wall
            n_hits = int(self.rng.poisson(2.0 * patter * n_frames / sr * 60.0))
            if n_hits:
                plink = self._kernel(("splash_drop",), lambda: sound_parts.ring_kernel(sr, (600.0, 1400.0), (0.006, 0.003), (1.0, 0.4), max_s=0.03))
                train = np.zeros(n_frames)
                np.add.at(train, self.rng.integers(0, n_frames, size=n_hits), self.rng.uniform(0.02, 0.06, n_hits) * min(1.5, splash_gain))
                self._tail.add(self._conv(train, plink, ("splash_drop", id(plink))))
        if drips:
            plink = self._kernel(("plink",), lambda: sound_parts.ring_kernel(sr, (2200.0, 3100.0, 700.0), (0.025, 0.012, 0.04),
                                                                             (1.0, 0.5, 0.35), max_s=0.12))
            train = np.zeros(n_frames)
            idx = self.rng.integers(0, n_frames, size=len(drips))
            np.add.at(train, idx, self.rng.uniform(0.25, 0.5, len(drips)))
            self._tail.add(self._conv(train, plink, ("plink", id(plink))))
        return sig

    def _bandpass(self, x: np.ndarray, f_hz: float, bw_frac: float) -> np.ndarray:
        """A band around f: the difference of two one-pole low-passes
        (native, vectorised), corners at f*(1 +/- bw)."""
        f_hi = min(0.45 * self.sr, f_hz * (1.0 + bw_frac))
        f_lo = max(20.0, f_hz * (1.0 - bw_frac))
        a_hi = 1.0 - math.exp(-2 * math.pi * f_hi / self.sr)
        a_lo = 1.0 - math.exp(-2 * math.pi * f_lo / self.sr)
        st = getattr(self, "_bp_state", [0.0, 0.0])
        hi, st[0] = native_audio.leaky_integrator(x, a_hi, st[0])
        lo, st[1] = native_audio.leaky_integrator(x, a_lo, st[1])
        self._bp_state = st
        return (hi - lo) * 2.0
