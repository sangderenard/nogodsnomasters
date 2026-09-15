"""SIZING A DYNO TO AN ENGINE, across the whole operating range.

A dynamometer has THREE independent limits and an engine can exceed any
of them without touching the other two:

  TORQUE   what the shaft twists with, worst at the torque peak. Sets
           the coupling, the roller normal force and the drum radius.
  POWER    what the absorber must turn into heat and carry away, worst
           at the POWER peak -- a different speed. This is the limit a
           torque-only sizing silently misses, and it is the one that
           boils a water brake.
  SPEED    what the drum must survive, worst at redline. A drum sized
           for torque alone can be past its own surface-speed limit
           before the engine is anywhere near its.

Sizing on peak torque alone gives a rig that holds the engine at low
speed and runs out of absorber at high speed, which reads as "the load
stops increasing" rather than as an obvious failure.

WHY THE ENVELOPE AND NOT A FORMULA. Peak torque is bmep times swept
volume, which is one number at one speed. Peak POWER is not derivable
from it, because where the torque curve falls away is a property of this
engine's own breathing -- `engine_sim.torque_fraction` already composes
that from the declared intake resonance and valvetrain rather than from a
drawn curve. So this sweeps the real curve rather than assuming a shape,
and a peaky engine and a flat one of the same peak torque correctly get
different absorbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RPM_TO_RAD_S = 2.0 * math.pi / 60.0

#: Real steel drum surface-speed limit. Past this a roller is a hazard
#: rather than an instrument, so a fast engine gets a SMALLER drum, not
#: a bigger one -- the opposite of what torque alone would say.
MAX_DRUM_SURFACE_M_S = 95.0

#: A real absorber is specified above what it will actually meet, because
#: the rating is continuous and an engine is not.
POWER_MARGIN = 1.25
TORQUE_MARGIN = 1.30
SPEED_MARGIN = 1.10

#: Disclosed: roller contact is sized so friction capacity comfortably
#: exceeds shaft torque even at a low coefficient, since a roller that
#: slips is not absorbing.
CONTACT_FRICTION = 0.9
CONTACT_CAPACITY_FACTOR = 3.0


@dataclass(frozen=True)
class DynoSizing:
    """One absorber, sized to one engine's whole envelope."""

    identity: str
    peak_torque_nm: float
    peak_power_w: float
    peak_power_rpm: float
    max_rpm: float
    drum_radius_m: float
    drum_inertia_kg_m2: float
    drum_mass_kg: float
    absorber_power_rating_w: float
    coupling_torque_rating_nm: float
    contact_normal_force_n: float
    speed_rating_rpm: float

    @property
    def surface_speed_m_s(self) -> float:
        return self.drum_radius_m * self.max_rpm * RPM_TO_RAD_S

    def consumes_range(self) -> tuple[bool, tuple[str, ...]]:
        """Whether this absorber fully consumes the engine's range, and
        what it fails on if not. Stated rather than assumed, because a
        rig that runs out quietly reads as an engine that stopped
        making power."""
        faults: list[str] = []
        if self.absorber_power_rating_w < self.peak_power_w:
            faults.append(
                f"absorber rated {self.absorber_power_rating_w/1000:.0f} kW "
                f"against {self.peak_power_w/1000:.0f} kW at the power peak")
        if self.coupling_torque_rating_nm < self.peak_torque_nm:
            faults.append(
                f"coupling rated {self.coupling_torque_rating_nm:.0f} Nm "
                f"against {self.peak_torque_nm:.0f} Nm")
        if self.speed_rating_rpm < self.max_rpm:
            faults.append(
                f"drum rated {self.speed_rating_rpm:.0f} rpm against "
                f"{self.max_rpm:.0f} rpm")
        if self.surface_speed_m_s > MAX_DRUM_SURFACE_M_S + 1e-6:
            faults.append(
                f"drum surface {self.surface_speed_m_s:.0f} m/s over the "
                f"{MAX_DRUM_SURFACE_M_S:.0f} m/s limit")
        return (not faults), tuple(faults)


def engine_power_envelope(engine) -> tuple[float, float]:
    """Sweep the engine's OWN torque curve for its real power peak.

    Returns (peak_power_w, rpm). Uses engine_sim.torque_fraction, which
    composes the curve from this engine's declared breathing rather than
    from an assumed shape, so a peaky engine and a flat one of equal peak
    torque get correctly different answers.
    """
    from engine_sim import torque_fraction

    peak_nm = max(float(engine.peak_torque_nm), 0.0)
    redline = max(float(engine.redline_rpm), 1.0)
    best_w = 0.0
    best_rpm = redline
    steps = 160
    low = max(float(engine.idle_rpm), redline / steps)
    for index in range(steps + 1):
        rpm = low + (redline - low) * index / steps
        try:
            fraction = float(torque_fraction(engine, rpm))
        except Exception:
            fraction = 1.0
        power = peak_nm * fraction * rpm * RPM_TO_RAD_S
        if power > best_w:
            best_w, best_rpm = power, rpm
    if best_w <= 0.0:
        best_w = peak_nm * redline * RPM_TO_RAD_S
    return best_w, best_rpm


def size_dyno(engine) -> DynoSizing:
    """Size an absorber that fully consumes this engine's range, safely."""
    peak_nm = max(float(engine.peak_torque_nm), 1e-6)
    peak_w, peak_w_rpm = engine_power_envelope(engine)
    max_rpm = max(float(engine.redline_rpm), 1.0)

    # DRUM RADIUS, from torque but CAPPED BY SURFACE SPEED. Torque wants
    # a big drum (more leverage at the contact); speed forbids one. The
    # previous rule saturated at 400 Nm, so a 255 Nm road V6 and a
    # 7851 Nm aero radial got nearly the same drum across a 30x span.
    # Scaling on the cube root keeps it sane over the catalogue's nine
    # orders of magnitude instead of running away.
    wanted = 0.10 + 0.085 * (peak_nm / 300.0) ** (1.0 / 3.0)
    speed_limited = MAX_DRUM_SURFACE_M_S / max(max_rpm * RPM_TO_RAD_S, 1e-9)
    drum_radius = max(0.02, min(wanted, speed_limited))

    # INERTIA: the existing rule, kept deliberately. A roller a disclosed
    # 3x the engine's own crank inertia stays light enough that the
    # engine dominates, and scales with the engine rather than sitting on
    # a fixed floor -- the fixed floor gave a 25cc trimmer a roller 2600x
    # heavier than its crank and pinned it at idle.
    inertia = max(1e-8, float(engine.inertia_kg_m2) * 3.0)
    mass = inertia / max(0.75 * drum_radius * drum_radius, 1e-9)

    return DynoSizing(
        identity=str(getattr(engine, "identity", "?")),
        peak_torque_nm=peak_nm,
        peak_power_w=peak_w,
        peak_power_rpm=peak_w_rpm,
        max_rpm=max_rpm,
        drum_radius_m=drum_radius,
        drum_inertia_kg_m2=inertia,
        drum_mass_kg=mass,
        absorber_power_rating_w=peak_w * POWER_MARGIN,
        coupling_torque_rating_nm=peak_nm * TORQUE_MARGIN,
        contact_normal_force_n=(peak_nm * CONTACT_CAPACITY_FACTOR
                                / max(CONTACT_FRICTION * drum_radius, 1e-9)),
        speed_rating_rpm=max_rpm * SPEED_MARGIN,
    )


if __name__ == "__main__":
    import engines

    show = ("25cc-two-stroke-trimmer", "alfa-busso-v6-3000-12v",
            "superbike-i4-1340", "pw-r4360-wasp-major",
            "agt1500-abrams-turbine", "wartsila-rta96c-14cyl-marine-diesel")
    print(f"{'engine':<38}{'peak Nm':>9}{'peak kW':>9}{'@rpm':>7}"
          f"{'drum m':>8}{'surf m/s':>9}{'absorber kW':>12}  range")
    print("-" * 100)
    for identity in show:
        s = size_dyno(engines.get(identity))
        ok, faults = s.consumes_range()
        print(f"{identity:<38}{s.peak_torque_nm:>9.0f}{s.peak_power_w/1000:>9.0f}"
              f"{s.peak_power_rpm:>7.0f}{s.drum_radius_m:>8.3f}"
              f"{s.surface_speed_m_s:>9.0f}{s.absorber_power_rating_w/1000:>12.0f}"
              f"  {'consumed' if ok else faults[0]}")
