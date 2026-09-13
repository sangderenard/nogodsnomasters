"""Every sound the engine makes, pinned to the part that makes it.

Two families, and the split is physical, not stylistic:

  PITCHED parts -- anything that spins and passes a countable feature
  past a fixed point: fan blades, pump vanes, gerotor lobes, alternator
  claw poles, blower lobes/impeller blades, sprocket teeth, reduction
  pinion teeth, a turbo shaft. Their PITCH is shaft speed x feature
  count, and the shaft speed is the crank's speed through the real
  drive ratio the drivetrain graph carries on the edge that drives them
  (accessory-drive-belt / geared-timing-drive ratio, rigid-keyed-hub =
  1:1). These genuinely climb with rpm. They are mech_parts.
  RotaryTonePart instances -- the same catalogue the offline baker uses
  -- built from the graph, one per real part, never a hand-typed list.

  EVENT parts -- combustion, the exhaust valve cracking open, a valve
  seating, a detonation front, an injector plunger. Each is ONE
  occurrence with a fixed acoustic signature set by the geometry it
  excites: the block's own structural ring, the exhaust pipe's odd
  quarter-wave modes, the chamber's Draper bore modes, a seat impact.
  rpm sets only how OFTEN they occur. If a cylinder does not fire there
  is no combustion event; whether its exhaust port still passes gas is
  a question about that port (a motored compression, a held-open
  hit-and-miss exhaust valve, a compression-release crack at TDC), so
  the blowdown amplitude is decided per port, per event, from what the
  sim says that cylinder actually did. EventKernel below is the fixed
  signature; the live synth (engine_sound.py) schedules copies of it
  off its own crank-angle clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from mech_parts import RotaryTonePart, Voice, blowdown_envelope, VALVE_OPEN_FRAC_OF_CYCLE, BLOWDOWN_DECAY_DEG


# Edges that carry rotation from one shaft to another. Everything else
# (a bolted mount, a pipe, a wire) does not turn the part on its far end.
ROTATION_EDGE_KINDS = ("accessory-drive-belt", "geared-timing-drive", "rigid-keyed-hub",
                       "gear-mesh", "chain-drive", "timing-chain", "timing-belt")
CRANK_IDENTITIES = ("powertrain.engine", "powertrain.crank_shaft.front", "powertrain.crank_shaft.rear",
                    "powertrain.harmonic_balancer", "powertrain.flywheel")


def drive_ratios_to_crank(graph: dict) -> dict[str, float]:
    """Shaft speed / crank speed for every node the crank turns, walking
    the graph's own rotation-carrying edges outward from the crank and
    multiplying each edge's real `ratio` (1.0 for a keyed hub)."""
    edges = graph.get("edges", ())
    out_edges: dict[str, list[tuple[str, float]]] = {}
    for e in edges:
        kind = e.get("constraint") or e.get("kind")
        if kind not in ROTATION_EDGE_KINDS:
            continue
        r = float(e.get("ratio", 1.0) or 1.0)
        out_edges.setdefault(e["a"], []).append((e["b"], r))
        out_edges.setdefault(e["b"], []).append((e["a"], 1.0 / r if r else 1.0))
    ratios: dict[str, float] = {}
    frontier = []
    for cid in CRANK_IDENTITIES:
        ratios[cid] = 1.0
        frontier.append(cid)
    while frontier:
        cur = frontier.pop()
        for nxt, r in out_edges.get(cur, ()):
            if nxt in ratios:
                continue
            ratios[nxt] = ratios[cur] * r
            frontier.append(nxt)
    return ratios


# ---------------------------------------------------------------------------
# what each spinning part passes per revolution, and how loud that is
# ---------------------------------------------------------------------------

def _pass_count(engine, identity: str, node: dict) -> tuple[str, int] | None:
    """(feature kind, count per shaft revolution) for a rotating-mass
    node, from the node's own declaration when it carries one, else the
    disclosed typical count for that part class."""
    if node.get("pass_count"):
        return (str(node.get("pass_kind", "features")), int(node["pass_count"]))
    disp = float(getattr(engine, "displacement_l", 2.0))
    fi = getattr(engine, "forced_induction", None)
    tail = identity.split(".")[-1]
    if identity == "supercharger_rotor" or tail == "supercharger_rotor":
        if fi is not None and getattr(fi, "blower_type", "roots") == "centrifugal":
            return ("blades", max(1, int(fi.impeller_blades)))
        # a two-rotor positive-displacement blower discharges lobes x 2
        # pulses per rotor turn (each rotor's lobes sweep the outlet)
        return ("lobes", 2 * max(1, int(getattr(fi, "lobe_count", 3))))
    if identity.startswith("supercharger_impeller_stage"):
        return ("blades", max(1, int(getattr(fi, "impeller_blades", 12))))
    if tail == "oil_pump":
        return ("lobes", 6 if disp > 10.0 else 4)          # gerotor inner-rotor lobes
    if tail in ("water_pump", "seawater_pump", "coolant_pump"):
        return ("vanes", 8 if disp > 10.0 else 6)
    if tail == "scavenge_pump":
        return ("teeth", 9)                                 # a gear-type scavenge stage
    if tail == "cooling_fan":
        return ("blades", int(node.get("blade_count", 7 if disp > 6.0 else 5)))
    if tail == "alternator":
        return ("pole-pairs", 6)                            # a 12-pole claw rotor
    if tail == "cam_sprocket":
        return ("teeth", 40)                                # cam sprocket = 2 x crank sprocket
    if tail == "prop_reduction_gearbox":
        return ("teeth", 30)                                # the crank-side pinion
    if tail in ("transaxle", "final_drive"):
        return ("teeth", 17)                                # a final-drive pinion
    return None


def _amp_law(kind: str, tail: str):
    """Loudness vs (rpm_frac, throttle, load, boost) for a part class --
    a pump grows with speed, an alternator with electrical load, a
    blower with the boost it is actually making, a fan with speed^2."""
    if tail == "alternator":
        return lambda rf, thr, load, boost: 0.04 + 0.12 * load
    if tail == "cooling_fan":
        return lambda rf, thr, load, boost: 0.09 * min(1.0, max(0.0, rf * 1.4 - 0.15)) ** 2
    if tail in ("supercharger_rotor",) or tail.startswith("supercharger_impeller"):
        if kind == "blades":
            return lambda rf, thr, load, boost: 0.22 * (0.15 + 0.85 * min(1.0, boost))
        return lambda rf, thr, load, boost: 0.30 * (0.3 + 0.7 * min(1.0, boost))
    if tail in ("cam_sprocket",):
        return lambda rf, thr, load, boost: 0.025 + 0.03 * rf
    if tail in ("prop_reduction_gearbox", "transaxle", "final_drive"):
        return lambda rf, thr, load, boost: 0.04 + 0.10 * load
    if tail == "scavenge_pump":
        return lambda rf, thr, load, boost: 0.05 + 0.06 * rf
    return lambda rf, thr, load, boost: 0.08 + 0.12 * rf     # pumps


@dataclass
class TurboWhine:
    """A turbocharger's own shaft: pitched off ITS speed (spool fraction
    of a rated shaft speed set by the wheel's inertia -- a small car
    turbo at 120k rpm, a marine unit at a tenth of that), not the
    crank's. The whistle is the shaft's first order (rotor imbalance)
    plus the compressor blade-pass where it is still below Nyquist."""
    identity: str
    rated_shaft_hz: float
    blades: int = 11

    @staticmethod
    def from_node(node: dict) -> "TurboWhine":
        inertia = max(float(node.get("inertia_kg_m2", 1e-4) or 1e-4), 1e-6)
        rated_rpm = max(8000.0, min(150000.0, 120000.0 * (1e-4 / inertia) ** 0.25))
        return TurboWhine(node["identity"], rated_rpm / 60.0, int(node.get("blade_count", 11)))


@dataclass
class PitchedCatalogue:
    """Everything on this engine that whines: the crank-driven parts as
    RotaryTonePart (order = drive ratio x pass count), and any turbo
    shafts. `describe()` is the audit -- which part, what it passes,
    at what order of the crank."""
    parts: list[RotaryTonePart] = field(default_factory=list)
    turbos: list[TurboWhine] = field(default_factory=list)
    orders: dict[str, tuple[str, int, float]] = field(default_factory=dict)   # identity -> (kind, count, ratio)

    def describe(self) -> list[str]:
        lines = [f"{ident}: {count} {kind} x ratio {ratio:.3g} = order {count * ratio:.3g}"
                 for ident, (kind, count, ratio) in self.orders.items()]
        lines += [f"{t.identity}: turbo shaft, rated {t.rated_shaft_hz * 60:.0f} rpm, {t.blades} blades" for t in self.turbos]
        return lines


def build_pitched_catalogue(engine, graph: dict) -> PitchedCatalogue:
    cat = PitchedCatalogue()
    if graph is None:
        return cat
    ratios = drive_ratios_to_crank(graph)
    for n in graph.get("nodes", ()):
        if n.get("kind") != "rotating-mass":
            continue
        ident = n["identity"]
        if ident.startswith("powertrain.turbocharger"):
            cat.turbos.append(TurboWhine.from_node(n))
            continue
        if ident in CRANK_IDENTITIES or ident.endswith(".pulley") or ident == "dyno_absorber":
            continue
        ratio = ratios.get(ident)
        if ratio is None or ratio <= 0.0:
            continue
        pc = _pass_count(engine, ident, n)
        if pc is None:
            continue
        kind, count = pc
        tail = ident.split(".")[-1]
        amp = _amp_law(kind, tail)
        pos = tuple(float(v) for v in n.get("reference_position", (0.0, 0.0, 0.0)))
        # a belt-driven part is heard through its belt's two strands, a
        # hair apart in pitch (the classic accessory-belt shimmer); a
        # gear/keyed drive is one clean voice
        driven_by_belt = any((e.get("constraint") == "accessory-drive-belt" and e["b"] == ident)
                             for e in graph.get("edges", ()))
        voices = (Voice(detune_cents=-2.0, amp_scale=0.55), Voice(detune_cents=2.0, amp_scale=0.55)) if driven_by_belt else (Voice(),)
        part = RotaryTonePart(name=ident, position=pos, domain="both", order=ratio * count, voices=voices,
                              amp_fn=lambda rpm, thr, load, _a=amp: _a(0.5, thr, load, 0.0))
        # the live synth's richer law: (rpm_frac, throttle, load, boost) -> amplitude
        part.amp_law = amp          # type: ignore[attr-defined]
        cat.parts.append(part)
        cat.orders[ident] = (kind, count, ratio)
    return cat


# ---------------------------------------------------------------------------
# fixed event signatures
# ---------------------------------------------------------------------------

def pipe_response_kernel(sr: int, f0_hz: float, brightness: float, modes=(1.0, 3.0, 5.0, 7.0),
                         max_s: float = 0.16) -> np.ndarray:
    """The exhaust pipe's own impulse response: a closed-open pipe's
    odd quarter-wave modes at f0 x (1,3,5,7), each ringing with the
    pipe's Q (an open header rings ~6, a muffled pipe ~2.5 -- the box
    damps it), falling as 1/m^1.3. Pure geometry + gas temperature."""
    q = 2.5 + 3.5 * brightness
    n = int(sr * max_s)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for m in modes:
        f = f0_hz * m
        if f >= 0.45 * sr:
            continue
        tau = q / (math.pi * f)
        out += np.exp(-t / tau) * np.sin(2 * math.pi * f * t) / (m ** 1.3)
    peak = float(np.max(np.abs(out))) or 1.0
    return out / peak


def ring_kernel(sr: int, freqs_hz, taus_s, weights, max_s: float = 0.08, phase_rad: float = 0.0) -> np.ndarray:
    """A damped multi-mode ring (a block, a chamber, a valve seat).
    phase_rad: a quadrature copy (pi/2) lets a caller mix two kernels
    into any per-event phase."""
    n = int(sr * max_s)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for f, tau, w in zip(freqs_hz, taus_s, weights):
        if f >= 0.45 * sr:
            continue
        out += w * np.exp(-t / tau) * np.sin(2 * math.pi * f * t + phase_rad)
    peak = float(np.max(np.abs(out))) or 1.0
    return out / peak


def plosive_kernel(sr: int, tau_s: float = 0.0014) -> np.ndarray:
    """A single one-sided pressure step: a quarter-wave click at
    1/(4 tau) decaying on tau -- the same shape mech_parts.bake_points
    uses; ~1.4 ms of real time whatever the rpm."""
    n = max(4, int(sr * tau_s * 5.0))
    t = np.arange(n) / sr
    return np.exp(-t / tau_s) * np.cos(2 * math.pi * t / (4.0 * tau_s))


def block_ring_hz(engine) -> float:
    """The block/head structural ring a combustion event excites: a
    bigger bore is a bigger, lower-ringing casting. Anchored at the
    catalogue's 320 Hz for a 96.5 mm bore (mech_parts.
    CombustionCrankPart.ring_freq_hz) and scaled 1/sqrt(bore)."""
    bore = float(getattr(engine.architecture, "bore_m", 0.0) or 0.0965)
    return max(60.0, min(800.0, 320.0 * math.sqrt(0.0965 / max(bore, 0.02))))


def blowdown_envelope_kernel(sr: int, deg_per_s: float, cycle_deg: float,
                             open_frac: float = VALVE_OPEN_FRAC_OF_CYCLE, decay_deg: float = BLOWDOWN_DECAY_DEG) -> np.ndarray:
    """The turbulent jet's envelope for one exhaust event: the real
    valve-lift x pressure-decay shape (mech_parts.blowdown_envelope) in
    crank degrees at the CURRENT crank speed -- the one place rpm
    legitimately enters an event: how many seconds its crank degrees
    take. Convolved with the event train and multiplied by one noise
    stream, it shapes every event in a block at once."""
    open_deg = open_frac * cycle_deg
    n = max(8, min(int(sr * 0.25), int(sr * open_deg / max(deg_per_s, 1.0))))
    theta = np.arange(n) / sr * deg_per_s
    return blowdown_envelope(theta, cycle_deg, open_frac, decay_deg)
