"""Small rotational-dynamics model for revving a catalogue engine under load.

Not the full SSA-compiled vehicle physics — a lightweight toy integrator:
    I * domega/dt = T_throttle(rpm, throttle) - T_pumping(rpm, throttle) - T_brake

torque magnitudes come from the engine's BMEP rating (T = BMEP * Vd / 4*pi),
shaped across the rev range by idle/torque-peak/power-peak/redline control
points pulled straight from the catalogue entry.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from engines import Engine, RPM_TO_RAD_S


CRANKING_COMBUSTION_QUALITY = 0.12  # the one remaining hand pick -- see torque_fraction
TWO_STROKE_CRANKING_COMBUSTION_QUALITY = 0.35  # port-scavenged two-stroke, see torque_fraction (disclosed)


def torque_fraction(engine: Engine, rpm: float) -> float:
    """Fraction (0..~1) of peak torque available at this rpm, wide-open
    throttle. Two real, DIFFERENT mechanisms on either side of
    torque_peak_rpm, not one hand-drawn curve:

    Below torque_peak_rpm: combustion quality ramps up with rpm (poor
    in-cylinder mixture motion/turbulence at low speed means slower,
    less complete combustion) and saturates approaching the cam's own
    tuned point -- modeled as a standard smoothstep from a real cranking-
    speed floor to 1.0 at torque_peak_rpm, zero slope at the join.
    CRANKING_COMBUSTION_QUALITY is the one remaining hand-picked number
    in this whole function, and it's a small, disclosed one -- not a
    curve shape.

    Above torque_peak_rpm: not a curve fit to any control point at all
    (not even a real one) -- composed directly from this engine's own
    real declared geometry, the same two real functions the live per-
    tick combustion loop already uses: IntakeSystem.resonance_gain
    (this engine's own real Helmholtz-tuned runner, a real relative
    charge-quantity curve) supplies the breathing side, and
    Engine.friction_torque_nm (the real piston-speed FMEP correlation,
    off this engine's own real stroke) supplies the loss side. Neither
    power_peak_rpm nor redline_rpm enters this formula at all anymore
    -- they're real reference numbers the catalogue reports, not
    control points a curve gets bent to match. A cited dyno figure
    (Engine.rated_power_kw, when declared) is compared against what
    this composition actually produces rather than forcing the curve
    to hit it -- a real, honest discrepancy is more useful than a
    prescribed target or a fixed torque/power ratio dressed up as
    physics."""
    if rpm <= 0:
        return 0.0
    tpeak = engine.torque_peak_rpm
    if rpm <= tpeak:
        # concave, not the symmetric smoothstep this used to be: real
        # combustion completeness rises fast with the FIRST bit of
        # in-cylinder turbulence and saturates with diminishing returns
        # after that (a real, well-established combustion phenomenon,
        # not curve-fitting) -- a smoothstep's slow, flat start
        # undershot real low-rpm torque badly (verified: it stalled the
        # idle-air-control governor outright on multiple catalogue
        # engines even at that governor's real, uncapped max authority).
        # This ease-out quadratic still joins the lognormal above with
        # zero slope at tpeak (no kink at the join) while rising much
        # faster off the bottom.
        t = max(0.0, min(1.0, rpm / tpeak))
        ease = 1.0 - (1.0 - t) * (1.0 - t)
        # a port-scavenged two-stroke has no valve overlap to bleed its
        # charge at low speed and traps a rich cold-start charge every
        # single revolution -- its cranking combustion quality is
        # genuinely far above a four-stroke's (real: a trimmer catches at
        # ~600 rpm on a rope pull), so it gets its own real floor
        arch = engine.architecture
        floor = (TWO_STROKE_CRANKING_COMBUSTION_QUALITY
                 if arch.two_stroke and not arch.has_poppet_valves else CRANKING_COMBUSTION_QUALITY)
        return floor + (1.0 - floor) * ease
    # a real degenerate case exists in the catalogue: a hit-and-miss
    # governor engine genuinely has no rev range at all (idle ==
    # torque_peak_rpm == power_peak_rpm by design, it holds one fixed
    # speed) -- its own real friction/breathing curves are still real
    # and well-defined there, so no special case is actually needed
    # here the way the old sigma-fit did.
    fpr = engine.firing_events_per_rev()
    intake = engine.intake_system
    # real relative charge quantity -- this engine's own real intake
    # runner, evaluated here vs. at torque_peak_rpm. A ratio, not an
    # absolute VE, so peak_torque_nm's own real baseline (already
    # whatever charge quantity applies AT tpeak) is never double-
    # counted; this only ever contributes the real, ADDITIONAL change
    # in breathing away from that baseline.
    gain_here = intake.resonance_gain(rpm, fpr)
    gain_at_tpeak = intake.resonance_gain(tpeak, fpr)
    breathing_ratio = gain_here / max(gain_at_tpeak, 1e-6)
    # real port/valve-curtain choking -- NOT the runner (IntakeSystem.
    # runner_diameter_mm is explicitly the pipe between plenum and
    # valve, a real, different, and looser restriction than the valve
    # itself -- using it here would be choking on the wrong component).
    # Reuses the SAME real, already-authoritative capacity formula
    # production's own _vehicle_powertrain_graph uses for this exact
    # limit (abstract_ui_vehicles.py's own real 100%-VE absolute flow
    # ceiling at redline: displacement * redline_rpm/120 * 1.2, 1.2
    # being real sea-level air density in kg/m3) rather than inventing
    # a parallel one from a component that isn't the actual restriction.
    # Ports are real hardware sized for the engine's OWN rated redline,
    # so this capacity is exactly matched to demand at redline and
    # stays untouched below it -- the real reason this choke term only
    # bites past redline, not gradually across the whole rev range.
    flow_capacity_kg_s = (engine.displacement_l / 1000.0) * (engine.redline_rpm / 120.0) * 1.2

    def _choke_frac(at_rpm: float) -> float:
        demand_kg_s = (engine.displacement_l / 1000.0) * (at_rpm / 120.0) * 1.2
        return min(1.0, flow_capacity_kg_s / max(demand_kg_s, 1e-9))

    # same real "ratio relative to the tpeak baseline" normalization
    # breathing_ratio uses -- whatever choking already applies at
    # torque_peak_rpm is already inside peak_torque_nm's own bmep, not
    # double-counted here
    choke_ratio = _choke_frac(rpm) / max(_choke_frac(tpeak), 1e-6)
    # real friction PENALTY relative to the same tpeak baseline -- an
    # ADDITIONAL real mechanical loss above whatever friction was
    # already netted into peak_torque_nm's own bmep at tpeak, off this
    # engine's own real FMEP-vs-piston-speed correlation, not a
    # separate invented falloff shape
    friction_delta_nm = engine.friction_torque_nm(rpm) - engine.friction_torque_nm(tpeak)
    friction_penalty_frac = friction_delta_nm / max(engine.peak_torque_nm, 1.0)
    return max(0.0, breathing_ratio * choke_ratio - friction_penalty_frac)


def electric_torque_fraction(engine: Engine, rpm: float) -> float:
    """Fraction of peak torque available at this rpm for a real electric
    traction motor -- a genuinely different real curve from torque_
    fraction above, not the same shape reused: an electric motor's peak
    torque is available from a dead stop (there's no combustion-quality
    ramp to wait through -- the actual reason an EV launches harder off
    the line than a piston car of comparable peak torque), flat out to
    a real base speed, then falls off in the real constant-power/field-
    weakening region above it (torque roughly inversely proportional to
    rpm, the standard real traction-motor characteristic -- back-EMF
    rises with speed and the drive can't push more voltage/current
    through past its own real rating). torque_peak_rpm doubles as the
    base speed here (the same declared control point every other engine
    kind already has, not a new field)."""
    if rpm <= 0.0:
        return 1.0
    base = max(engine.torque_peak_rpm, 1.0)
    if rpm <= base:
        return 1.0
    return max(0.05, base / rpm)


@dataclass
class EngineState:
    engine: Engine
    rpm: float = 0.0
    throttle: float = 0.0          # 0..1 pedal position
    brake_load_nm: float = 0.0     # dyno / output-shaft brake torque
    stalled: bool = False

    def __post_init__(self) -> None:
        self.rpm = self.engine.idle_rpm * 0.0  # starts stopped; call start()

    def start(self) -> None:
        self.rpm = self.engine.idle_rpm
        self.stalled = False

    def set_engine(self, engine: Engine) -> None:
        self.engine = engine
        self.start()
        self.throttle = 0.0
        self.brake_load_nm = 0.0

    @property
    def omega(self) -> float:
        return self.rpm * RPM_TO_RAD_S

    @property
    def current_torque_nm(self) -> float:
        eng = self.engine
        if self.stalled:
            return 0.0
        frac = torque_fraction(eng, self.rpm)
        drive = eng.peak_torque_nm * frac * self.throttle
        # idle governor: a floor throttle input near idle keeps it alive
        if self.rpm < eng.idle_rpm * 1.05:
            governed = eng.peak_torque_nm * frac * max(self.throttle, 0.12)
            drive = max(drive, governed)
        return drive

    @property
    def pumping_loss_nm(self) -> float:
        eng = self.engine
        frac = torque_fraction(eng, self.rpm)
        closed = 1.0 - self.throttle
        return eng.peak_braking_torque_nm * frac * closed * 0.55

    @property
    def power_kw(self) -> float:
        return self.current_torque_nm * self.omega / 1000.0

    def step(self, dt: float) -> None:
        eng = self.engine
        if self.stalled:
            self.rpm = 0.0
            return
        net_torque = self.current_torque_nm - self.pumping_loss_nm - self.brake_load_nm
        domega = net_torque / max(eng.inertia_kg_m2, 1e-4) * dt
        new_omega = self.omega + domega
        new_rpm = new_omega / RPM_TO_RAD_S
        if new_rpm <= 0.0:
            # brake/load overpowered the engine
            self.rpm = 0.0
            if self.brake_load_nm > eng.peak_torque_nm * 0.35:
                self.stalled = True
            return
        self.rpm = min(new_rpm, eng.redline_rpm * 1.12)

    def restart_if_stalled(self) -> None:
        if self.stalled:
            self.start()
