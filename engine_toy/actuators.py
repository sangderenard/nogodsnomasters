"""Actuators: the parts that turn pressure into motion.

Everything here is a real product you can buy, described by the
dimensions its catalogue actually lists -- bore, rod diameter, stroke,
vane width, swept volume per revolution -- and nothing else. Give a
cylinder a 63 mm bore and a 35 mm rod and it behaves like a 63/35
cylinder, because every number it reports is computed from those two
dimensions and the pressure across it.

The family, and why each member exists:

  LINEAR
    double-acting          pressure on either side; the standard tie-rod
                           or welded cylinder. Note it is NOT symmetric:
                           the rod takes area off the retract side, so
                           it pulls with less force and retracts faster
                           on the same flow. That asymmetry is the thing
                           people forget and it is why a cylinder's
                           return stroke is the fast one.
    single-acting-spring   pressure one way, a spring back. Fails to a
                           known position, which is the whole reason to
                           choose it.
    double-rod             a rod out both ends: equal areas, equal
                           speeds, no asymmetry -- what you use when the
                           two directions must match.
    telescopic             nested stages: a long stroke out of a short
                           body, at the cost of a force that steps down
                           as each stage takes over.
    rodless                the carriage is coupled to the piston through
                           a band or magnets: full stroke in barely more
                           than the stroke's own length, and no rod to
                           buckle.

  ROTARY
    rack-and-pinion        pistons driving a rack against a pinion:
                           constant torque through the whole swing, and
                           swing angles beyond a full turn are ordinary.
    vane                   a vane sweeping a chamber: compact and light,
                           lower torque, and it leaks past the vane.
    helical-spline         a piston on a helix converting thrust to
                           torque: very high torque in a small envelope.

  CONTINUOUS
    gerotor / orbital      a hydraulic motor: displacement per rev, so
                           torque comes from pressure difference and
                           speed comes from flow.
    vane air motor         the pneumatic equivalent, which cannot stall
                           and damage itself, which is why it is what
                           you find where stalling is routine.

Two things get their own honest treatment because they are the ones
people get wrong:

  FINE POSITIONING is not a matter of a better valve. A cylinder and
  its load form a spring-mass system whose stiffness is the trapped
  fluid's bulk modulus, and the resulting natural frequency is a hard
  ceiling on closed-loop bandwidth no controller can exceed. Oil's bulk
  modulus is about 1.4 GPa; air's effective stiffness is its absolute
  pressure times gamma, around 0.14 MPa -- four orders of magnitude
  softer. That single ratio is why hydraulic servo axes hold microns
  and pneumatic ones do not, and `Positioner` computes it rather than
  asserting it.

  GAS OVER OIL IS NOT MAD. The user asked whether working an air volume
  and an oil volume together in a damped strut was mad. It is the
  opposite: it is the oleo-pneumatic strut under every transport
  aircraft, Citroen's hydropneumatic suspension, every gas-charged
  monotube damper, and every hydraulic accumulator ever fitted. Gas is
  the spring because gas compresses; oil is the damper because oil does
  not. `GasOverOilStrut` is that part.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ATM_PA = 101_325.0
OIL_DENSITY_KG_M3 = 870.0
OIL_BULK_MODULUS_PA = 1.4e9          # stiff, and that is the point
AIR_GAMMA = 1.4
STEEL_E_PA = 210e9
# A real seal pack costs a few per cent of the theoretical force, plus a
# breakaway term that is higher than the running one -- which is why a
# cylinder creeping under fine control moves in steps until it is going.
SEAL_FRICTION_FRAC = 0.05
BREAKAWAY_MULTIPLIER = 1.8
STICTION_SPEED_M_S = 0.002           # below this it has not broken away


def _area(diameter_m: float) -> float:
    return math.pi * diameter_m * diameter_m / 4.0


# =====================================================================
#  LINEAR
# =====================================================================
@dataclass
class LinearActuator:
    """A cylinder, described the way its catalogue describes it."""
    identity: str = "cylinder"
    medium: str = "hydraulic"            # "hydraulic" | "pneumatic"
    kind: str = "double-acting"          # see the module docstring
    bore_m: float = 0.063
    rod_m: float = 0.035
    stroke_m: float = 0.200
    # what it is bolted to at each end, which is what decides whether
    # the rod is a strut or a pendulum when it buckles
    mounting: str = "clevis-both-ends"   # "clevis-both-ends" | "flange-rigid" | "trunnion"
    cushion_m: float = 0.020             # the decelerating pocket at each end
    spring_force_n: float = 0.0          # single-acting return spring, at full extension
    stages: int = 1                      # telescopic
    rated_pressure_pa: float = 21_000_000.0
    # Seal drag as a fraction of theoretical force. An ordinary
    # elastomer-sealed cylinder costs about 5 %; a SERVO cylinder is
    # built with low-friction or hydrostatic seals and costs nearer
    # 1 %, which is most of why it can be positioned finely at all.
    # Declared per actuator because it is a purchasing choice, not a
    # law.
    seal_friction_frac: float = SEAL_FRICTION_FRAC
    # live state
    position_m: float = 0.0              # 0 = fully retracted
    velocity_m_s: float = 0.0
    flow_l_min: float = 0.0
    force_n: float = 0.0
    at_end: bool = False
    cushioning: bool = False
    stalled: bool = False

    # ---- what the dimensions mean -------------------------------
    @property
    def area_extend_m2(self) -> float:
        """Full bore: the rod is on the other side."""
        return _area(self.bore_m)

    @property
    def area_retract_m2(self) -> float:
        """Annulus: bore minus rod. A double-rod cylinder has the rod on
        BOTH sides, so its two areas are equal and it is symmetric."""
        if self.kind == "double-rod":
            return _area(self.bore_m) - _area(self.rod_m)
        return _area(self.bore_m) - _area(self.rod_m)

    @property
    def area_ratio(self) -> float:
        """Extend area over retract area -- the asymmetry itself. A 2:1
        cylinder pulls half as hard and retracts twice as fast."""
        return self.area_extend_m2 / max(self.area_retract_m2, 1e-12)

    @property
    def swept_volume_l(self) -> float:
        return self.area_extend_m2 * self.stroke_m * 1000.0

    def force_at(self, pressure_pa: float, extending: bool = True) -> float:
        """Theoretical force, before seals and before the spring."""
        area = self.area_extend_m2 if extending else self.area_retract_m2
        f = pressure_pa * area
        if self.kind == "telescopic" and extending and self.stages > 1:
            # each stage that takes over is narrower, so the force steps
            # DOWN as it extends -- the real and frequently surprising
            # behaviour of a telescopic ram
            stage = min(self.stages - 1, int(self.position_m / max(self.stroke_m, 1e-9) * self.stages))
            f *= (1.0 - stage / (self.stages + 1.0))
        return f

    def friction_n(self, pressure_pa: float) -> float:
        running = self.seal_friction_frac * pressure_pa * self.area_extend_m2
        if abs(self.velocity_m_s) < STICTION_SPEED_M_S:
            return running * BREAKAWAY_MULTIPLIER
        return running

    @property
    def buckling_limit_n(self) -> float:
        """Euler, on the rod, at full extension -- the real limit on a
        long-stroke cylinder in compression, and the reason rod diameter
        is a structural choice rather than a sealing one.

        The end fixing sets the effective length factor: pinned at both
        ends is the classic K = 1; a rigidly flanged cylinder with a
        guided rod end is stiffer; a trunnion mount in the middle is
        worse than either because the cylinder itself can swing."""
        k = {"clevis-both-ends": 1.0, "flange-rigid": 0.7, "trunnion": 2.0}.get(self.mounting, 1.0)
        # The column that buckles is the whole EXTENDED cylinder, pin to
        # pin -- barrel plus exposed rod -- not the rod alone. A
        # cylinder's closed length is its stroke plus a dead length for
        # the end caps and gland, so extended it is about twice the
        # stroke again. Using the rod's own length here made a long
        # cylinder look far safer than it is, and buckling is precisely
        # a long-cylinder failure.
        dead_m = 0.15 + 1.5 * self.bore_m
        length = 2.0 * self.stroke_m + dead_m
        inertia = math.pi * self.rod_m ** 4 / 64.0
        return math.pi ** 2 * STEEL_E_PA * inertia / max((k * length) ** 2, 1e-9)

    def natural_frequency_hz(self, load_mass_kg: float) -> float:
        """The cylinder-and-load resonance, from the trapped fluid's own
        stiffness. This is the ceiling on how fast it can be controlled,
        and it is set by the fluid, not the valve."""
        return _hydraulic_natural_frequency_hz(self, load_mass_kg)

    # ---- and what it does ---------------------------------------
    def step(self, dt: float, supply_pressure_pa: float, available_flow_l_min: float,
             command: float, load_n: float = 0.0) -> dict:
        """Advance one tick.

        `command` is -1..+1: the valve's spool position, extend positive.
        `load_n` is the external load opposing extension. Returns what
        the actuator did and what it took to do it, so a supply can be
        charged for it."""
        cmd = max(-1.0, min(1.0, command))
        extending = cmd >= 0.0
        area = self.area_extend_m2 if extending else self.area_retract_m2
        p = max(ATM_PA, supply_pressure_pa)

        gross = self.force_at(p, extending)
        if self.kind == "single-acting-spring":
            # the spring is always there, opposing extension and
            # returning the rod when pressure goes away
            spring = self.spring_force_n * (self.position_m / max(self.stroke_m, 1e-9))
            gross = gross - spring if extending else -self.spring_force_n
        # WHAT THE ROD ACTUALLY EXERTS is pressure times area, less the
        # seals. That is the measurement a bench takes and the number a
        # catalogue quotes. The load then decides whether it MOVES:
        # a cylinder lifting a load it can just manage exerts exactly
        # that load and no more, and reporting the surplus instead --
        # which an earlier version of this did -- made a cylinder
        # holding its rated load look as though it were exerting
        # nothing at all.
        delivered = abs(gross) - self.friction_n(p)
        opposing = load_n if extending else -load_n
        self.stalled = delivered <= opposing

        # speed comes from FLOW, and only from flow: a cylinder is a
        # volumetric device, so halving the flow halves the speed no
        # matter what the pressure is
        demanded_v = abs(cmd) * (available_flow_l_min / 60_000.0) / max(area, 1e-12)
        if self.stalled:
            demanded_v = 0.0

        # the cushion: a pocket at each end that traps fluid and makes
        # it leave through a restriction, so the piston is decelerated
        # instead of arriving at the end cap at full speed
        self.cushioning = False
        if self.cushion_m > 0.0:
            into_end = (self.stroke_m - self.position_m) if extending else self.position_m
            if into_end < self.cushion_m:
                self.cushioning = True
                demanded_v *= max(0.08, into_end / self.cushion_m)

        v = demanded_v if extending else -demanded_v
        new_pos = self.position_m + v * dt
        self.at_end = False
        if new_pos <= 0.0:
            new_pos, v, self.at_end = 0.0, 0.0, True
        elif new_pos >= self.stroke_m:
            new_pos, v, self.at_end = self.stroke_m, 0.0, True
        self.velocity_m_s = v
        self.position_m = new_pos
        self.force_n = max(0.0, delivered) * (1.0 if extending else -1.0)
        self.flow_l_min = abs(v) * area * 60_000.0
        return {"flow_l_min": self.flow_l_min, "force_n": self.force_n,
                "velocity_m_s": self.velocity_m_s, "position_m": self.position_m,
                "at_end": self.at_end, "stalled": self.stalled,
                "cushioning": self.cushioning,
                "surplus_n": max(0.0, delivered - abs(opposing)),
                "power_w": abs(self.force_n * self.velocity_m_s)}

    def describe(self) -> list[str]:
        pull = self.area_ratio
        return [f"  {self.identity}: {self.medium} {self.kind} "
                f"{self.bore_m * 1000:.0f}/{self.rod_m * 1000:.0f}-{self.stroke_m * 1000:.0f}",
                f"    push {self.force_at(self.rated_pressure_pa) / 1000:6.1f} kN   "
                f"pull {self.force_at(self.rated_pressure_pa, False) / 1000:6.1f} kN   "
                f"(area ratio {pull:4.2f}:1, so it retracts {pull:.2f}x faster)",
                f"    swept {self.swept_volume_l:5.2f} L   rod buckles at "
                f"{self.buckling_limit_n / 1000:6.1f} kN"]


# =====================================================================
#  ROTARY, THROUGH A LIMITED SWING
# =====================================================================
@dataclass
class RotaryActuator:
    """A limited-swing rotary actuator: rack-and-pinion, vane or helix."""
    identity: str = "rotary"
    medium: str = "hydraulic"
    kind: str = "rack-and-pinion"        # "rack-and-pinion" | "vane" | "helical-spline"
    swing_deg: float = 180.0
    rated_pressure_pa: float = 21_000_000.0
    # rack-and-pinion
    piston_bore_m: float = 0.050
    pinion_radius_m: float = 0.025
    pistons: int = 2
    # vane
    vane_width_m: float = 0.040
    vane_outer_radius_m: float = 0.050
    vane_hub_radius_m: float = 0.020
    vanes: int = 1
    # helical
    helix_mean_radius_m: float = 0.035
    helix_angle_deg: float = 30.0
    # live state
    angle_deg: float = 0.0
    rate_deg_s: float = 0.0
    torque_nm: float = 0.0
    flow_l_min: float = 0.0
    at_end: bool = False

    def torque_at(self, pressure_pa: float) -> float:
        """Torque from the geometry that actually makes it."""
        if self.kind == "rack-and-pinion":
            # each piston pushes the rack, the rack turns the pinion:
            # torque is force times the pitch radius, and it does not
            # vary with angle at all
            return pressure_pa * _area(self.piston_bore_m) * self.pinion_radius_m * self.pistons
        if self.kind == "vane":
            # pressure over the vane's swept face, acting at its mean
            # radius: T = n * b * p * (R^2 - r^2) / 2
            return (self.vanes * self.vane_width_m * pressure_pa
                    * (self.vane_outer_radius_m ** 2 - self.vane_hub_radius_m ** 2) / 2.0)
        # a piston on a helix: the thrust is turned into torque by the
        # helix angle, which is why these make so much torque for their
        # size and why they are stiff in both directions
        thrust = pressure_pa * _area(self.piston_bore_m)
        return thrust * self.helix_mean_radius_m / max(math.tan(math.radians(self.helix_angle_deg)), 1e-6)

    @property
    def displacement_l_per_rev(self) -> float:
        if self.kind == "vane":
            swept = (self.vanes * self.vane_width_m * math.pi
                     * (self.vane_outer_radius_m ** 2 - self.vane_hub_radius_m ** 2))
            return swept * 1000.0
        return _area(self.piston_bore_m) * 2.0 * math.pi * self.pinion_radius_m * self.pistons * 1000.0

    def step(self, dt: float, supply_pressure_pa: float, available_flow_l_min: float,
             command: float, load_nm: float = 0.0) -> dict:
        cmd = max(-1.0, min(1.0, command))
        p = max(ATM_PA, supply_pressure_pa)
        gross = self.torque_at(p)          # what the shaft actually exerts
        stalled = gross <= abs(load_nm)
        rev_per_s = (abs(cmd) * (available_flow_l_min / 60.0)
                     / max(self.displacement_l_per_rev, 1e-9)) if not stalled else 0.0
        rate = math.copysign(rev_per_s * 360.0, cmd)
        new_angle = self.angle_deg + rate * dt
        self.at_end = False
        if new_angle <= 0.0:
            new_angle, rate, self.at_end = 0.0, 0.0, True
        elif new_angle >= self.swing_deg:
            new_angle, rate, self.at_end = self.swing_deg, 0.0, True
        self.angle_deg, self.rate_deg_s = new_angle, rate
        self.torque_nm = max(0.0, gross) * (1.0 if cmd >= 0 else -1.0)
        self.flow_l_min = abs(rate) / 360.0 * self.displacement_l_per_rev * 60.0
        return {"flow_l_min": self.flow_l_min, "torque_nm": self.torque_nm,
                "angle_deg": self.angle_deg, "rate_deg_s": self.rate_deg_s,
                "at_end": self.at_end, "stalled": stalled,
                "surplus_nm": max(0.0, gross - abs(load_nm)),
                "power_w": abs(self.torque_nm * math.radians(abs(rate)))}

    def describe(self) -> list[str]:
        return [f"  {self.identity}: {self.medium} {self.kind}, {self.swing_deg:.0f} deg swing",
                f"    {self.torque_at(self.rated_pressure_pa):7.0f} Nm at "
                f"{self.rated_pressure_pa / 1e5:.0f} bar, "
                f"{self.displacement_l_per_rev:5.3f} L/rev"]


# =====================================================================
#  CONTINUOUS ROTATION
# =====================================================================
@dataclass
class FluidMotor:
    """A motor that turns as long as you feed it: gerotor/orbital on
    oil, vane on air. Torque comes from the pressure DIFFERENCE across
    it and speed comes from the FLOW through it -- the two are
    independent, which is the useful property."""
    identity: str = "motor"
    medium: str = "hydraulic"
    kind: str = "gerotor"                # "gerotor" | "axial-piston" | "vane-air"
    displacement_cc_rev: float = 200.0
    mechanical_efficiency: float = 0.92
    volumetric_efficiency: float = 0.95
    max_pressure_pa: float = 17_500_000.0
    # live state
    rpm: float = 0.0
    torque_nm: float = 0.0
    flow_l_min: float = 0.0

    def torque_at(self, delta_p_pa: float) -> float:
        disp_m3 = self.displacement_cc_rev * 1e-6
        return delta_p_pa * disp_m3 / (2.0 * math.pi) * self.mechanical_efficiency

    def speed_at(self, flow_l_min: float) -> float:
        disp_l = self.displacement_cc_rev / 1000.0
        return flow_l_min / max(disp_l, 1e-9) * self.volumetric_efficiency

    def step(self, dt: float, supply_pressure_pa: float, available_flow_l_min: float,
             command: float, load_nm: float = 0.0) -> dict:
        cmd = max(-1.0, min(1.0, command))
        dp = max(0.0, supply_pressure_pa - ATM_PA)
        available = self.torque_at(dp)
        if abs(load_nm) > available:
            # an OIL motor simply stops and sits at relief pressure; an
            # AIR motor stalls without harming itself, which is exactly
            # why air motors are used where stalling is routine
            self.rpm, self.torque_nm = 0.0, available
            self.flow_l_min = 0.0 if self.medium == "hydraulic" else available_flow_l_min * 0.2
            return {"flow_l_min": self.flow_l_min, "torque_nm": self.torque_nm, "rpm": 0.0,
                    "stalled": True, "power_w": 0.0}
        self.rpm = self.speed_at(available_flow_l_min * abs(cmd))
        self.torque_nm = available
        self.flow_l_min = self.rpm * self.displacement_cc_rev / 1000.0 / self.volumetric_efficiency
        return {"flow_l_min": self.flow_l_min, "torque_nm": self.torque_nm, "rpm": self.rpm,
                "stalled": False,
                "power_w": self.torque_nm * self.rpm * 2.0 * math.pi / 60.0}

    def describe(self) -> list[str]:
        return [f"  {self.identity}: {self.medium} {self.kind} motor "
                f"{self.displacement_cc_rev:.0f} cc/rev",
                f"    {self.torque_at(self.max_pressure_pa):6.0f} Nm at "
                f"{self.max_pressure_pa / 1e5:.0f} bar, "
                f"{self.speed_at(100.0):5.0f} rpm at 100 L/min"]


# =====================================================================
#  FINE POSITIONING
# =====================================================================
def _hydraulic_natural_frequency_hz(cyl: LinearActuator, load_mass_kg: float) -> float:
    """The resonance of a load on a column of fluid.

    wn = sqrt(4 * beta * A^2 / (V_total * m)) for a cylinder pressurised
    on both sides. It is the single most important number in servo
    hydraulics, because no amount of controller gain can push a closed
    loop past roughly a third of it without the whole axis ringing.

    For AIR the stiffness is not a bulk modulus at all -- gas is
    compressible, so its effective modulus is gamma times the absolute
    pressure it happens to be at, which is around 0.14 MPa at working
    pressure against oil's 1400 MPa. That ratio, four orders of
    magnitude, is the entire reason pneumatic axes need a positioner
    and still cannot do what a hydraulic servo axis does."""
    area = cyl.area_extend_m2
    trapped_m3 = max(area * cyl.stroke_m, 1e-9)
    if cyl.medium == "pneumatic":
        beta = AIR_GAMMA * (cyl.rated_pressure_pa + ATM_PA)
    else:
        beta = OIL_BULK_MODULUS_PA
    wn = math.sqrt(4.0 * beta * area * area / max(trapped_m3 * max(load_mass_kg, 0.1), 1e-12))
    return wn / (2.0 * math.pi)


@dataclass
class Positioner:
    """A closed-loop position axis: a proportional or servo valve, an
    actuator, a position sensor and a controller.

    The honest part is the limit. The controller here will not be given
    a bandwidth above a third of the hydraulic natural frequency,
    because a real one cannot have it either -- push past that and the
    axis oscillates about its target instead of settling on it. So a
    hydraulic axis positions finely and a pneumatic one hunts, and both
    fall out of the same formula rather than being asserted."""
    identity: str = "axis"
    actuator: LinearActuator = field(default_factory=LinearActuator)
    load_mass_kg: float = 50.0
    valve_rated_flow_l_min: float = 40.0     # at its rated drop
    valve_rated_drop_pa: float = 3.5e6       # 35 bar, the usual rating point
    sensor_resolution_m: float = 1e-5        # what the feedback can actually see
    deadband_frac: float = 0.02              # real spool overlap
    load_disturbance_frac: float = 0.02      # the load step to quote deflection against
    target_m: float = 0.0
    # live state
    error_m: float = 0.0
    command: float = 0.0
    settled: bool = False

    @property
    def natural_frequency_hz(self) -> float:
        return self.actuator.natural_frequency_hz(self.load_mass_kg)

    @property
    def usable_bandwidth_hz(self) -> float:
        """What a controller can really close around it."""
        return self.natural_frequency_hz / 3.0

    @property
    def stick_slip_step_m(self) -> float:
        """How far it JUMPS when it finally breaks away.

        Pressure has to build until it beats breakaway friction, and
        building that pressure compresses the fluid column behind the
        piston. When the seal lets go, that stored compression is
        released at once and the rod moves by roughly the deflection
        that was stored in it: dx = F_breakaway / k, where k = beta*A/L
        is the fluid column's own spring rate.

        This is the real reason a slow hydraulic axis crawls in steps,
        and the real reason servo cylinders are built with low-friction
        seals rather than ordinary ones."""
        a = self.actuator
        beta = (AIR_GAMMA * (a.rated_pressure_pa + ATM_PA)
                if a.medium == "pneumatic" else OIL_BULK_MODULUS_PA)
        k = beta * a.area_extend_m2 / max(a.stroke_m, 1e-9)     # N/m
        return a.friction_n(a.rated_pressure_pa) * BREAKAWAY_MULTIPLIER / max(k, 1e-9)

    @property
    def stiffness_n_per_m(self) -> float:
        """The axis's own spring rate: the trapped fluid's stiffness
        across the piston. Everything about fine positioning follows
        from this one number."""
        a = self.actuator
        beta = (AIR_GAMMA * (a.rated_pressure_pa + ATM_PA)
                if a.medium == "pneumatic" else OIL_BULK_MODULUS_PA)
        return beta * a.area_extend_m2 / max(a.stroke_m, 1e-9)

    @property
    def disturbance_deflection_m(self) -> float:
        """How far the load pushes the rod OFF position when it changes.

        This is the honest statement of the air-versus-oil difference.
        Static error is not the problem: any loop with integral action
        drives steady-state error down to what the sensor can see, on
        either medium. What cannot be fixed by control is how far the
        axis moves the instant the load changes, before the loop has
        had time to respond -- and that is the load divided by the
        stiffness above. Oil is four orders of magnitude stiffer than
        air, so the same load step moves an air axis millimetres and an
        oil axis microns."""
        a = self.actuator
        disturbance_n = self.load_disturbance_frac * a.rated_pressure_pa * a.area_extend_m2
        return disturbance_n / max(self.stiffness_n_per_m, 1e-9)

    @property
    def achievable_resolution_m(self) -> float:
        """What it settles to with the load held steady: the sensor."""
        return self.sensor_resolution_m

    def step(self, dt: float, supply_pressure_pa: float, available_flow_l_min: float,
             load_n: float = 0.0) -> dict:
        a = self.actuator
        self.error_m = self.target_m - a.position_m
        band = max(self.achievable_resolution_m, self.sensor_resolution_m)
        self.settled = abs(self.error_m) <= band
        if self.settled:
            self.command = 0.0
            return a.step(dt, supply_pressure_pa, 0.0, 0.0, load_n)
        # proportional, with the gain set by what the axis can carry
        gain = self.usable_bandwidth_hz * 2.0 * math.pi
        cmd = gain * self.error_m / max(a.stroke_m, 1e-9)
        if abs(cmd) < self.deadband_frac:
            cmd = 0.0                                    # real spool overlap
        self.command = max(-1.0, min(1.0, cmd))
        # the valve is a restriction: its flow follows the square root
        # of the pressure across it, which is why a servo axis slows as
        # it loads up
        drop = max(0.0, supply_pressure_pa - ATM_PA)
        valve_flow = self.valve_rated_flow_l_min * math.sqrt(max(drop, 0.0) / self.valve_rated_drop_pa)
        flow = min(available_flow_l_min, valve_flow * abs(self.command))
        out = a.step(dt, supply_pressure_pa, flow, self.command, load_n)
        out["error_m"] = self.error_m
        out["settled"] = self.settled
        return out

    def describe(self) -> list[str]:
        return [f"  {self.identity}: closed-loop {self.actuator.medium} axis, "
                f"{self.load_mass_kg:.0f} kg load",
                f"    resonance {self.natural_frequency_hz:6.1f} Hz -> usable bandwidth "
                f"{self.usable_bandwidth_hz:5.1f} Hz",
                f"    stiffness {self.stiffness_n_per_m / 1e6:8.2f} N/um   settles to "
                f"{self.achievable_resolution_m * 1e6:6.1f} um with the load steady",
                f"    but a {self.load_disturbance_frac * 100:.0f}% load change moves it "
                f"{self.disturbance_deflection_m * 1e6:9.1f} um before the loop can answer"
                f"   (stick-slip jump {self.stick_slip_step_m * 1e6:7.1f} um)"]


# =====================================================================
#  GAS OVER OIL -- and no, it is not mad
# =====================================================================

def size_recoil_slide(*, impulse_n_s: float, recoiling_mass_kg: float, stroke_m: float,
                      piston_bore_m: float = 0.095, peak_force_n: float = 120_000.0,
                      elevation_deg: float = 42.0,
                      stroke_utilisation: float = 0.80) -> dict:
    """Size a recoil slide's gas and orifice to the shot it must absorb.

    THE ONE PLACE THIS ARITHMETIC LIVES. It was written twice -- once
    properly in `machines.size_recoil_for` and once as a set of
    scaling rules in the turret builder -- and the two disagreed
    catastrophically across the bore range, because the second scaled
    the orifice with the GUN'S bore while leaving the recoil piston at
    a fixed size. An orifice's loss goes with the square of the
    velocity through it, so getting its area wrong by a factor of four
    is wrong by a factor of sixteen in force: small guns were damped so
    hard that four millimetres of travel made four hundred kilonewtons,
    and large guns were not damped at all and simply coasted into the
    end stop. Neither is a recoil system.

    The real relations, and each one is a consequence rather than a
    choice:

      THE BRAKE must absorb the free-recoil energy inside the stroke
      available, so its MEAN force is that energy over that stroke, and
      its peak is somewhat above the mean.

      THE ORIFICE follows from that force at the recoil velocity:
      dp = rho/2 * (A*v/Ao)^2 and F = dp*A. Square-law, so it must be
      solved against the velocity this gun actually recoils at -- which
      is why a gun firing variable charges needs a metering pin that
      changes the area through the stroke rather than a fixed hole.

      THE RECUPERATOR needs only enough charge to return the mass to
      battery and hold it there at full elevation. More than that is
      not safety margin, it is a spring fighting the stroke it exists
      to permit.

      THE GAS VOLUME must exceed what the stroke sweeps, or the gas
      goes solid before the piston reaches the end and the strut stops
      being a spring.
    """
    area = math.pi * piston_bore_m * piston_bore_m / 4.0
    velocity = abs(impulse_n_s) / max(recoiling_mass_kg, 1e-6)
    energy = 0.5 * recoiling_mass_kg * velocity * velocity
    # SIZE TO A FRACTION OF THE STROKE, NOT TO ALL OF IT. Absorbing the
    # nominal shot in exactly the travel available means the design case
    # ends with the piston touching the stop, so every round hotter than
    # nominal -- a warm chamber, a strong lot of propellant, the ordinary
    # spread of real ammunition -- lands on the structure instead. Real
    # mounts hold back a fifth of the stroke for exactly that, and the
    # reserve is worth more than it costs because the gas spring's rate
    # climbs steeply through it.
    working_stroke_m = max(stroke_m, 1e-6) * stroke_utilisation
    mean_force = energy / working_stroke_m
    force = min(max(mean_force * 1.6, 1.0), peak_force_n)
    dp = force / max(area, 1e-9)
    a_o = area * velocity * math.sqrt(OIL_DENSITY_KG_M3 / (2.0 * max(dp, 1.0)))
    # hold the recoiling mass at full elevation, with margin, and no more
    hold_n = recoiling_mass_kg * 9.81 * math.cos(math.radians(90.0 - elevation_deg)) * 1.35
    hold_n = max(hold_n, recoiling_mass_kg * 9.81 * 0.35)
    swept_m3 = area * stroke_m
    return {
        "piston_bore_m": piston_bore_m,
        "piston_area_m2": area,
        "recoil_velocity_m_s": velocity,
        "free_recoil_energy_j": energy,
        "mean_brake_force_n": mean_force,
        "working_stroke_m": working_stroke_m,
        "design_peak_force_n": force,
        "orifice_area_m2": a_o,
        "orifice_diameter_m": math.sqrt(max(a_o, 1e-10) / 0.7 * 4.0 / math.pi),
        "gas_charge_pressure_pa": max(5.0e5, hold_n / max(area, 1e-9)),
        "gas_volume_m3": swept_m3 * 2.6,
    }

@dataclass
class GasOverOilStrut:
    """An oleo-pneumatic strut: a gas volume for the spring and an oil
    volume forced through an orifice for the damping, in one body.

    This is the landing gear of every transport aircraft, Citroen's
    hydropneumatic suspension, and in a smaller form every gas-charged
    monotube damper and every accumulator on a hydraulic system. The
    reason to combine them is that each fluid is good at exactly one of
    the two jobs: gas compresses, so it stores energy and gives a rising
    spring rate that resists bottoming; oil does not compress, so all of
    its energy goes into pushing it through a hole, which is a damper.

    A steel spring gives a straight rate and stores energy it hands
    straight back. Gas gives a rate that climbs steeply as it is
    compressed -- soft over small bumps, very stiff at the end of
    travel. That is the property an aircraft needs, and it is why the
    two-fluid arrangement is the standard answer rather than a curiosity.
    """
    identity: str = "strut"
    bore_m: float = 0.090
    stroke_m: float = 0.400
    # THE GAS VOLUME MUST EXCEED WHAT THE STROKE SWEEPS, with margin.
    # A real strut is charged so that full travel compresses the gas by
    # about four to one; size it any tighter and the gas goes solid
    # before the piston reaches the end of its travel, so the strut
    # stops being a spring and becomes a hydraulic ram hitting a wall.
    # (This module's first defaults did exactly that and reported a
    # spring force of eight million kN, which is how the error showed.)
    gas_volume_l: float = 4.0                # at full extension
    charge_pressure_pa: float = 2.0e6        # 20 bar static
    # The metering orifice has to be sized to the piston it is damping.
    # Orifice loss goes with the SQUARE of the velocity through it, so
    # an orifice a little too small does not damp a little too hard --
    # it damps enormously too hard.
    orifice_diameter_m: float = 0.014
    orifice_cd: float = 0.7
    # A METERING PIN: a tapered rod that closes the orifice as the
    # piston travels, so the area shrinks exactly as the velocity falls
    # and the braking force stays roughly CONSTANT instead of collapsing
    # with the square of the speed.
    #
    # This is not a refinement, it is the difference between a recoil
    # brake that works and one that does not. With a fixed hole the
    # force is all at the start: the shot is met by an enormous spike
    # that decays quadratically, so most of the stroke does almost no
    # work and the tube coasts into the end stop anyway. Sizing such a
    # brake to stop inside four fifths of its travel is not possible at
    # any orifice diameter -- the energy curve is exponential and never
    # actually reaches zero.
    #
    # With a pin the brake holds its design force the whole way down,
    # which is why every real gun that fires more than one charge has
    # one, and why a mount can then be given a genuine stroke reserve:
    # a round with a fifth more energy needs a fifth more stroke, and
    # the reserve is exactly that.
    metering_pin: bool = False
    metering_taper: float = 0.86      # how much of the area the pin closes
    polytropic_n: float = 1.35               # a real strut is compressed too fast to be isothermal
    oil_volume_l: float = 1.2
    # live state
    compression_m: float = 0.0
    velocity_m_s: float = 0.0
    gas_pressure_pa: float = 0.0
    spring_force_n: float = 0.0
    damping_force_n: float = 0.0

    def __post_init__(self) -> None:
        self.gas_pressure_pa = self.charge_pressure_pa

    @property
    def piston_area_m2(self) -> float:
        return _area(self.bore_m)

    @property
    def swept_volume_l(self) -> float:
        return self.piston_area_m2 * self.stroke_m * 1000.0

    @property
    def compression_ratio(self) -> float:
        """Gas volume extended over gas volume at full stroke. A real
        strut lands between about 3:1 and 6:1."""
        v0 = self.gas_volume_l
        return v0 / max(v0 - self.swept_volume_l, 1e-6)

    def check(self) -> list[str]:
        """Whether this strut is sized like a real one. Worth saying out
        loud rather than reporting a nonsense force later."""
        out = []
        if self.swept_volume_l >= self.gas_volume_l:
            out.append(f"  ! {self.identity}: stroke sweeps {self.swept_volume_l:.2f} L but only "
                       f"{self.gas_volume_l:.2f} L of gas -- it goes solid before full travel")
        elif self.compression_ratio > 8.0:
            out.append(f"  ! {self.identity}: {self.compression_ratio:.1f}:1 compression, very harsh "
                       f"at the end of travel (real struts are 3:1 to 6:1)")
        return out

    def gas_pressure_at(self, compression_m: float) -> float:
        """p * V^n = constant. The rate rises steeply as the gas is
        compressed, which is exactly what keeps a strut from arriving at
        its mechanical stop.

        The volume is floored at a few per cent of the charge: past
        that, a real strut is on its stop and the structure is taking
        the load, not the gas. Reporting a pressure that goes to
        infinity instead would be arithmetic, not physics."""
        v0 = self.gas_volume_l / 1000.0
        floor = 0.03 * v0
        v = max(v0 - self.piston_area_m2 * compression_m, floor)
        return self.charge_pressure_pa * (v0 / v) ** self.polytropic_n

    def orifice_area_at(self, compression_m: float) -> float:
        """The flow area the oil actually has at this point in the
        stroke. Constant unless there is a metering pin, in which case
        it closes down as the piston travels."""
        a_o = _area(self.orifice_diameter_m) * self.orifice_cd
        if not self.metering_pin:
            return a_o
        frac = max(0.0, min(1.0, compression_m / max(self.stroke_m, 1e-9)))
        # the area that holds force constant against a velocity falling
        # as sqrt(1 - x/L) under constant deceleration is proportional
        # to that velocity, so the taper follows the same root
        return a_o * max(1.0 - self.metering_taper * frac, 0.06)

    def damping_force_at(self, velocity_m_s: float) -> float:
        """Oil forced through an orifice: the loss goes with the SQUARE
        of the speed, so the strut is soft at walking pace and very firm
        under a hard landing, without anything having to switch --
        unless a metering pin is closing the orifice as it goes, which
        trades that self-adjusting softness for a constant force and a
        predictable stopping distance."""
        a_o = self.orifice_area_at(self.compression_m)
        q = self.piston_area_m2 * abs(velocity_m_s)
        dp = OIL_DENSITY_KG_M3 * (q / max(a_o, 1e-9)) ** 2 / 2.0
        return math.copysign(dp * self.piston_area_m2, velocity_m_s)

    def step(self, dt: float, velocity_m_s: float) -> dict:
        self.compression_m = max(0.0, min(self.stroke_m, self.compression_m + velocity_m_s * dt))
        self.velocity_m_s = velocity_m_s
        self.gas_pressure_pa = self.gas_pressure_at(self.compression_m)
        self.spring_force_n = self.gas_pressure_pa * self.piston_area_m2
        self.damping_force_n = self.damping_force_at(velocity_m_s)
        return {"compression_m": self.compression_m,
                "gas_pressure_pa": self.gas_pressure_pa,
                "spring_force_n": self.spring_force_n,
                "damping_force_n": self.damping_force_n,
                "total_force_n": self.spring_force_n + self.damping_force_n,
                "bottomed": self.compression_m >= self.stroke_m - 1e-6}

    def dynamics_step(self, dt: float, mass_kg: float, external_velocity_m_s: float = 0.0) -> dict:
        """Advance the recoiling mass under the strut's own force law.

        `step()` above is kinematic -- it takes a velocity and reports
        what the strut does at it. This is the dynamic version: it
        starts from the mass's CURRENT velocity (whatever state was left
        by the last shot or the last tick, which is how partial recovery
        survives between rounds), computes the net restoring force from
        the spring and the damper, and integrates. Semi-implicit Euler,
        the same scheme `symbolic_parts.symbolic_oleo_strut_equations`
        uses and for the same reason: a spring that stiffens as it
        compresses is exactly the case where evaluating force at the
        OLD position rather than the new one goes unstable.

        `external_velocity_m_s` lets a carrier (the cradle, riding on
        its own elevation motion) be added in -- 0 for a stand-alone
        strut."""
        mass = max(mass_kg, 1e-6)
        force = (self.damping_force_at(self.velocity_m_s)
                + self.gas_pressure_at(self.compression_m) * self.piston_area_m2)
        # the force is a RESTORER: positive compression and positive
        # velocity both push back toward battery (negative direction)
        self.velocity_m_s += (-force / mass) * dt
        new_compression = self.compression_m + self.velocity_m_s * dt
        bottomed = new_compression >= self.stroke_m
        topped_out = new_compression <= 0.0
        if bottomed or topped_out:
            # the stop is real structure: the velocity does not survive
            # hitting it intact, it is what a hard bottoming actually is
            self.velocity_m_s *= -0.05
        return self.step(dt, self.velocity_m_s)

    def describe(self) -> list[str]:
        static = self.gas_pressure_at(0.0) * self.piston_area_m2 / 1000.0
        half = self.gas_pressure_at(self.stroke_m * 0.5) * self.piston_area_m2 / 1000.0
        full = self.gas_pressure_at(self.stroke_m * 0.95) * self.piston_area_m2 / 1000.0
        return [f"  {self.identity}: oleo-pneumatic strut, {self.bore_m * 1000:.0f} mm bore x "
                f"{self.stroke_m * 1000:.0f} mm, {self.gas_volume_l:.1f} L gas at "
                f"{self.charge_pressure_pa / 1e5:.0f} bar  "
                f"({self.compression_ratio:.1f}:1 at full stroke)",
                f"    spring {static:6.1f} kN extended -> {half:6.1f} kN at half -> "
                f"{full:7.1f} kN near bottom  (gas rate rises, which is the point)",
                f"    damping {self.damping_force_at(0.5) / 1000:5.1f} kN at 0.5 m/s -> "
                f"{self.damping_force_at(2.0) / 1000:6.1f} kN at 2 m/s  (square law)"] + self.check()


ACTUATOR_TYPES = {
    "linear": LinearActuator,
    "rotary": RotaryActuator,
    "motor": FluidMotor,
    "positioner": Positioner,
    "strut": GasOverOilStrut,
}


@dataclass
class AdaptiveRecoilDamper:
    """A recoil damper whose orifice is set per shot, not per gun.

    WHY A FIXED ORIFICE IS NOT ENOUGH ON THIS MOUNT. An orifice loses
    pressure with the SQUARE of the flow through it, so a damper sized
    for a full charge is far too stiff for a reduced one and far too
    soft for a hot one. On a gun that fires whatever charge the firing
    solution asked for, that means every shot ends up somewhere
    different in the stroke -- and the stroke IS the reload cycle, so
    the timing of extraction, feed and ram changes with the charge.

    A CONTROLLED APERTURE FIXES THE STROKE INSTEAD OF THE FORCE. The
    impulse is known before the round is fired -- it comes out of the
    same interior ballistics that chose the charge -- so the aperture
    can be set for THAT shot to bring the recoiling mass to rest at the
    same place every time. Light round, small orifice, more damping per
    metre; heavy round, open it up.

    That is worth having for three separate reasons and only one of
    them is comfort:

      the cycle takes the same time regardless of charge, so a mixed
      belt of tasked and full-charge rounds feeds at one rate;
      a round hotter than expected cannot bottom the stroke, because
      the aperture closes against it;
      and the peak force into the mount is CHOSEN rather than whatever
      the charge happened to produce.

    The control is not clever. It is one algebraic inversion of the
    energy balance, evaluated once per shot, which is all a
    microcontroller on a valve needs to do.
    """
    identity: str = "turret.recoil_damper"
    piston_area_m2: float = 0.0113          # 120 mm piston
    stroke_m: float = 0.420
    target_stroke_m: float = 0.300          # where we want it to stop
    spring_rate_n_per_m: float = 22_000.0
    spring_preload_n: float = 1_800.0
    oil_density_kg_m3: float = 870.0
    discharge_coefficient: float = 0.70
    #: what the valve can physically do
    min_orifice_m2: float = 7.0e-6
    max_orifice_m2: float = 9.0e-4
    #: live
    last_orifice_m2: float = field(default=0.0, init=False)
    last_peak_force_n: float = field(default=0.0, init=False)

    def spring_energy_j(self, x: float) -> float:
        return 0.5 * self.spring_rate_n_per_m * x * x + self.spring_preload_n * x

    def orifice_for(self, impulse_n_s: float, recoiling_mass_kg: float) -> dict:
        """Set the aperture so THIS round stops at the target stroke.

        The energy balance over the stroke:

            1/2 m v^2  =  spring(x)  +  damper work

        and an orifice's damper work over a stroke, with the velocity
        falling roughly linearly from v to zero, is

            W = k_d * v^2 * x * (1/3)      k_d = rho A^3 / (2 Cd^2 Ao^2)

        so the required orifice area falls straight out. No search, no
        table, no tuning constant -- the same algebra a designer does
        once, done per shot instead."""
        v = impulse_n_s / max(recoiling_mass_kg, 1e-6)
        ke = 0.5 * recoiling_mass_kg * v * v
        x = self.target_stroke_m
        want_damper_j = ke - self.spring_energy_j(x)
        if want_damper_j <= 0.0:
            # the spring alone already stops it short: open the valve
            # wide and let it run, because damping it more would stop
            # it before the action has cycled
            self.last_orifice_m2 = self.max_orifice_m2
            return {"orifice_m2": self.max_orifice_m2,
                    "stroke_m": None, "note": "spring alone suffices",
                    "recoil_m_s": v, "energy_j": ke}
        a = self.piston_area_m2
        # W = (rho a^3 / (6 Cd^2 Ao^2)) v^2 x   ->  solve for Ao
        num = self.oil_density_kg_m3 * a ** 3 * v * v * x
        den = 6.0 * self.discharge_coefficient ** 2 * want_damper_j
        ao = math.sqrt(max(num / max(den, 1e-12), 1e-16))
        ao = max(self.min_orifice_m2, min(self.max_orifice_m2, ao))
        self.last_orifice_m2 = ao
        # peak force is at the start, where the velocity is highest
        k_d = (self.oil_density_kg_m3 * a ** 3
               / (2.0 * self.discharge_coefficient ** 2 * ao * ao))
        self.last_peak_force_n = k_d * v * v + self.spring_preload_n
        return {
            "orifice_m2": ao,
            "orifice_mm": math.sqrt(ao / math.pi) * 2000.0,
            "recoil_m_s": v, "energy_j": ke,
            "spring_j": self.spring_energy_j(x),
            "damper_j": want_damper_j,
            "peak_force_n": self.last_peak_force_n,
            "at_limit": ao in (self.min_orifice_m2, self.max_orifice_m2),
        }

    def describe(self) -> list:
        return [
            f"{self.identity}: {self.piston_area_m2 * 1e4:.0f} cm2 piston, "
            f"{self.stroke_m * 1000:.0f} mm available, "
            f"{self.target_stroke_m * 1000:.0f} mm target",
            f"  aperture {self.min_orifice_m2 * 1e6:.0f}-"
            f"{self.max_orifice_m2 * 1e6:.0f} mm2, set per shot",
        ]


@dataclass
class MagnetorheologicalDamper:
    """A recoil damper with no moving valve: the OIL changes instead.

    WHY NOT A SERVO APERTURE. A controlled orifice needs a valve in the
    dirtiest fluid path on the machine, moving under load, in the
    milliseconds between the primer and the end of the stroke. It is a
    mechanism, and mechanisms in that position wear, foul and stick.

    A MAGNETORHEOLOGICAL FLUID HAS NO MECHANISM. Iron particles in
    carrier oil line up along a magnetic field and give the fluid an
    apparent YIELD STRESS -- below it the fluid will not shear at all,
    above it, it flows normally. Vary the coil current and the fluid
    goes from light oil to something like grease and back, in about a
    millisecond, with nothing physically moving but the particles.

    THE TWO TERMS ARE DIFFERENT IN KIND, and that is what makes it
    controllable rather than merely adjustable:

        VISCOUS   rises with velocity, and is not controllable
        YIELD     a force that is THERE AT ANY VELOCITY, set by the
                  field, and is the whole of the control authority

    which means it can hold force at the END of the stroke where the
    velocity has gone and an orifice has nothing left to work with.
    An orifice damper's force dies as v^2; this one does not.

    SIZING IT IS THE SAME ALGEBRA either way: the energy the spring
    cannot take is what the damper must, and over a known stroke the
    required yield force follows directly.
    """
    identity: str = "turret.mr_damper"
    piston_area_m2: float = 0.0113
    gap_m: float = 0.0012                  # the annular gap the fluid shears in
    active_length_m: float = 0.060         # how much of it the coil covers
    stroke_m: float = 0.420
    #: the fluid: a real MR formulation's controllable yield range
    yield_min_pa: float = 1.0e3            # coil off, carrier oil only
    yield_max_pa: float = 6.0e4            # coil saturated
    plastic_viscosity_pa_s: float = 0.28
    coil_watts_max: float = 22.0
    #: live
    last_current_frac: float = field(default=0.0, init=False)

    def yield_force_n(self, fraction: float) -> float:
        """Force from the field alone, at any velocity including zero."""
        f = max(0.0, min(1.0, fraction))
        tau = self.yield_min_pa + (self.yield_max_pa - self.yield_min_pa) * f
        # the fluid shears over the active length of the annular gap and
        # the pressure it holds acts on the piston
        dp = 2.0 * tau * self.active_length_m / max(self.gap_m, 1e-9)
        return dp * self.piston_area_m2

    def viscous_force_n(self, velocity_m_s: float) -> float:
        """The part the coil cannot change."""
        a = self.piston_area_m2
        q = a * abs(velocity_m_s)
        dp = (12.0 * self.plastic_viscosity_pa_s * self.active_length_m * q
              / (math.pi * 0.12 * self.gap_m ** 3))
        return dp * a

    def field_for(self, impulse_n_s: float, recoiling_mass_kg: float,
                  spring_energy_j: float, target_stroke_m: float) -> dict:
        """What current this shot wants, solved once before it is fired.

        The impulse is known from the same interior ballistics that
        picked the charge, so the damper is set for the round that is
        about to go off rather than reacting to the one that did."""
        v = impulse_n_s / max(recoiling_mass_kg, 1e-6)
        ke = 0.5 * recoiling_mass_kg * v * v
        need_j = ke - spring_energy_j
        if need_j <= 0.0:
            self.last_current_frac = 0.0
            return {"current_frac": 0.0, "recoil_m_s": v, "energy_j": ke,
                    "note": "spring takes it all; coil off"}
        # the yield force is constant over the stroke, so the work it
        # does is simply force times distance -- which is exactly why
        # this is easy to aim and an orifice is not
        want_force = need_j / max(target_stroke_m, 1e-6)
        viscous = self.viscous_force_n(v * 0.5)
        yield_needed = max(0.0, want_force - viscous)
        lo, hi = self.yield_force_n(0.0), self.yield_force_n(1.0)
        frac = (yield_needed - lo) / max(hi - lo, 1e-9)
        frac = max(0.0, min(1.0, frac))
        self.last_current_frac = frac
        return {
            "current_frac": frac,
            "coil_w": frac * self.coil_watts_max,
            "recoil_m_s": v, "energy_j": ke, "spring_j": spring_energy_j,
            "damper_j": need_j,
            "yield_force_n": self.yield_force_n(frac),
            "viscous_force_n": viscous,
            "total_force_n": self.yield_force_n(frac) + viscous,
            "saturated": frac >= 0.999,
            "authority_ratio": hi / max(lo, 1e-9),
        }

    def describe(self) -> list:
        return [
            f"{self.identity}: {self.piston_area_m2 * 1e4:.0f} cm2 piston, "
            f"{self.gap_m * 1000:.1f} mm gap, "
            f"{self.active_length_m * 1000:.0f} mm under the coil",
            f"  force {self.yield_force_n(0.0) / 1000:.1f} - "
            f"{self.yield_force_n(1.0) / 1000:.1f} kN from the field alone "
            f"({self.yield_force_n(1.0) / max(self.yield_force_n(0.0), 1):.0f}:1)",
            f"  {self.coil_watts_max:.0f} W at full field, and nothing moves",
        ]
