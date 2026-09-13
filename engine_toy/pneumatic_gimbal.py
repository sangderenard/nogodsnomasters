"""A pneumatic omnidirectional gimbal, and what aiming actually costs.

The question that started this was whether aiming consumes the right
energy for the work it does, with the efficiency of whatever technology
is doing it. The answer for compressed air is uncomfortable and worth
stating plainly: PNEUMATIC AIMING IS THERMODYNAMICALLY EXPENSIVE. You
spend roughly seven to ten joules at the compressor for every joule of
mechanical work at the mount. Nobody chooses it for efficiency.

They choose it for what it buys instead, and those reasons are real:

  NO MAGNETIC SIGNATURE and no commutation, because there is no motor.
  NO HEAT at the mount -- expanding air is COLD, so the moving part
  runs cooler than ambient rather than hotter.
  INTRINSICALLY SAFE: no arcs, which matters anywhere with fuel vapour.
  QUIET AT THE MOUNT. This is the one people get wrong. Compressed air
  is not quiet -- the compressor is loud and unsilenced exhaust is very
  loud -- but the NOISE IS WHERE THE COMPRESSOR IS, not where the gimbal
  is, and exhaust can be muffled to a hiss. A hydraulic mount carries
  its own whining pump; an electric one carries an audible commutation
  whine. A pneumatic mount fed from a distant receiver makes almost
  nothing at the mount itself.

WHY THIS PARTICULAR MACHINE

Asked for an omnidirectional pneumatic gimbal, the obvious answers are
bad. Stacked yaw-then-pitch bearings are not omnidirectional, they are
two hinges with a gimbal lock between them. A pneumatic rotary vane
actuator per axis has backlash and seal drag and needs gears.

The arrangement here is neither, and every piece of it is real hardware:

  AN AEROSTATIC SPHERICAL BEARING. A ball floating on a film of air a
  few microns thick, fed through orifices in a cup. These exist and are
  ordinary -- spacecraft attitude simulators float entire satellites on
  them precisely because the friction is effectively zero in EVERY
  direction at once. That is what makes the mount omnidirectional
  rather than two-axis: there are no axes.

  THREE PNEUMATIC MUSCLES at a hundred and twenty degrees. A McKibben
  muscle is a bladder in a braided sleeve: pressurise it and the braid
  forces it to shorten while pulling very hard. Three of them pulling
  on a ball can point it anywhere within their cone, the way three
  tendons aim an eye, and they are compliant by construction so a knock
  moves the mount instead of breaking it.

  ONE SUPPLY does both: the same air floats the bearing and works the
  muscles. The bearing's consumption is continuous and the muscles' is
  only when moving, which is the opposite of a hydraulic mount, where
  holding position costs nothing and the pump runs anyway.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ATM_PA = 101_325.0
AIR_GAMMA = 1.4
AIR_DENSITY_KG_M3 = 1.2
AIR_R_J_PER_KG_K = 287.0
# A real industrial compressor delivers about 40 % of the isothermal
# ideal at the receiver, after motor, heat of compression and leaks.
COMPRESSOR_EFFICIENCY = 0.40


def free_air_compression_energy_j(free_volume_m3: float, pressure_pa: float) -> float:
    """What it cost to make this air.

    Isothermal compression of `free_volume_m3` of atmosphere up to
    `pressure_pa`, divided by what a real compressor achieves. This is
    the number that makes pneumatics expensive, and it is why the
    efficiency of an air system has to be measured from the WALL, not
    from the receiver."""
    if free_volume_m3 <= 0.0 or pressure_pa <= ATM_PA:
        return 0.0
    ideal = ATM_PA * free_volume_m3 * math.log(pressure_pa / ATM_PA)
    return ideal / COMPRESSOR_EFFICIENCY


# =====================================================================
#  THE BEARING
# =====================================================================
@dataclass
class AerostaticSphericalBearing:
    """A ball floating on an air film. No axes, no friction worth the
    name, and a constant appetite for air."""
    ball_diameter_m: float = 0.220
    film_gap_m: float = 12e-6            # a few microns: this is the whole trick
    supply_pressure_pa: float = 6.0e5    # 6 bar gauge-ish
    orifices: int = 12
    orifice_diameter_m: float = 0.20e-3
    discharge_coefficient: float = 0.8

    @property
    def projected_area_m2(self) -> float:
        return math.pi * (self.ball_diameter_m / 2.0) ** 2

    @property
    def load_capacity_n(self) -> float:
        """What it can hold up.

        The film never develops the full supply pressure across the
        whole face -- it falls from the orifice outward -- so a real
        aerostatic bearing carries roughly a third of supply times
        projected area. Designing to the full figure is the classic way
        to build one that touches down under load."""
        return 0.35 * self.supply_pressure_pa * self.projected_area_m2

    @property
    def stiffness_n_per_m(self) -> float:
        """Squeeze the film and the pressure climbs steeply, which is
        why these are stiff despite being made of nothing."""
        return self.load_capacity_n / max(self.film_gap_m, 1e-9) * 0.6

    @property
    def free_air_l_min(self) -> float:
        """It leaks continuously, by design, and that is the running
        cost of having no friction. Choked flow through the orifices,
        which is the right regime at any normal supply pressure."""
        area = math.pi * self.orifice_diameter_m ** 2 / 4.0 * self.discharge_coefficient
        # choked mass flow: 0.0404 * A * P / sqrt(T) for air, SI
        mass_flow = 0.0404 * area * self.orifices * self.supply_pressure_pa / math.sqrt(293.15)
        return mass_flow / AIR_DENSITY_KG_M3 * 60_000.0

    def friction_torque_nm(self, rate_rad_s: float) -> float:
        """Shear in the film, and it is tiny: this is the entire reason
        for the bearing. Newtonian shear of air across a 12 micron gap
        over the ball's area."""
        mu = 1.81e-5
        radius = self.ball_diameter_m / 2.0
        return (mu * abs(rate_rad_s) * radius * self.projected_area_m2 * 2.0
                / max(self.film_gap_m, 1e-9) * radius)

    def describe(self) -> list[str]:
        return [f"  aerostatic sphere {self.ball_diameter_m * 1000:.0f} mm on a "
                f"{self.film_gap_m * 1e6:.0f} um film at {self.supply_pressure_pa / 1e5:.1f} bar",
                f"    holds {self.load_capacity_n / 1000:5.2f} kN, stiffness "
                f"{self.stiffness_n_per_m / 1e6:6.1f} MN/m, "
                f"drag {self.friction_torque_nm(1.0) * 1000:.3f} mNm at 1 rad/s",
                f"    leaks {self.free_air_l_min:5.1f} L/min of free air, continuously"]


# =====================================================================
#  THE MUSCLES
# =====================================================================
@dataclass
class PneumaticMuscle:
    """A McKibben muscle: a bladder in a braided sleeve.

    Pressurise it and the braid geometry forces it to shorten while
    pulling hard -- very hard, for its weight. The force falls as it
    contracts and reaches zero at the braid's locking angle, which is
    why these are quoted with a maximum contraction and why you never
    size one at its limit."""
    resting_length_m: float = 0.320
    diameter_m: float = 0.040
    braid_angle_deg: float = 20.0        # at rest; 54.7 degrees is the locking angle
    max_contraction: float = 0.25

    def force_n(self, pressure_pa: float, contraction: float) -> float:
        """Gaylord's relation: force from the braid geometry alone.

        F = (pi D0^2 P / 4) * [ 3(1-eps)^2 / tan^2(theta0) - 1/sin^2(theta0) ]
        """
        eps = max(0.0, min(self.max_contraction, contraction))
        t0 = math.radians(self.braid_angle_deg)
        a = 3.0 * (1.0 - eps) ** 2 / (math.tan(t0) ** 2)
        b = 1.0 / (math.sin(t0) ** 2)
        return max(0.0, math.pi * self.diameter_m ** 2 * pressure_pa / 4.0 * (a - b))

    def swept_volume_m3(self, contraction: float) -> float:
        """What it swallows to get there. The bladder fattens as it
        shortens, so the volume grows even though the length falls."""
        eps = max(0.0, min(self.max_contraction, contraction))
        length = self.resting_length_m * (1.0 - eps)
        # constant braid length: the diameter grows as the square root
        # of the length ratio's complement
        diameter = self.diameter_m * math.sqrt(max(1.0, 1.0 / max(1.0 - eps, 1e-3)))
        return math.pi * diameter ** 2 / 4.0 * length

    def describe(self, pressure_pa: float) -> list[str]:
        return [f"  muscle {self.diameter_m * 1000:.0f} x {self.resting_length_m * 1000:.0f} mm, "
                f"braid {self.braid_angle_deg:.0f} deg",
                f"    pulls {self.force_n(pressure_pa, 0.0) / 1000:6.2f} kN at rest -> "
                f"{self.force_n(pressure_pa, 0.15) / 1000:6.2f} kN at 15 % -> "
                f"{self.force_n(pressure_pa, self.max_contraction) / 1000:6.2f} kN at its limit"]


# =====================================================================
#  THE GIMBAL
# =====================================================================
@dataclass
class PneumaticGimbal:
    """Three muscles pulling on a floating ball."""
    bearing: AerostaticSphericalBearing = field(default_factory=AerostaticSphericalBearing)
    muscle: PneumaticMuscle = field(default_factory=PneumaticMuscle)
    muscles: int = 3
    moment_arm_m: float = 0.150          # where the tendons attach, off the centre
    supply_pressure_pa: float = 6.0e5
    payload_inertia_kg_m2: float = 0.9
    payload_mass_kg: float = 42.0
    payload_offset_m: float = 0.06       # centre of mass off the pivot: the unbalance
    # totals
    free_air_used_l: float = 0.0
    work_done_j: float = 0.0
    compression_energy_j: float = 0.0

    @property
    def peak_torque_nm(self) -> float:
        """Two muscles pull while the third pays out, so the useful
        torque is the differential across the pair."""
        pull = self.muscle.force_n(self.supply_pressure_pa, 0.10)
        return pull * self.moment_arm_m * 2.0 * math.sin(math.radians(60.0))

    @property
    def unbalance_torque_nm(self) -> float:
        """What gravity asks for when the payload's centre of mass is
        off the pivot. A gimbal that is balanced needs none of this,
        which is why real mounts are counterweighted before they are
        powered."""
        return self.payload_mass_kg * 9.80665 * self.payload_offset_m

    def hold_cost_l_min(self) -> float:
        """Holding still is not free: the bearing leaks the whole time.
        The muscles do not, which is the useful half of the bargain."""
        return self.bearing.free_air_l_min

    def slew(self, angle_rad: float, rate_rad_s: float) -> dict:
        """Move through an angle, and account for every joule.

        The work is the honest integral: inertia through the
        acceleration, the unbalance through the whole sweep, and the
        film's own drag. The air is what the muscles actually swallow to
        do it, expressed as free air, because free air is what the
        compressor had to make."""
        angle = abs(float(angle_rad))
        rate = max(abs(float(rate_rad_s)), 1e-6)
        # a trapezoid: accelerate, run, decelerate
        accel_time = rate / max(self.peak_torque_nm / max(self.payload_inertia_kg_m2, 1e-9), 1e-9)
        alpha = rate / max(accel_time, 1e-9)
        inertial_work = self.payload_inertia_kg_m2 * alpha * rate * accel_time  # spin up and down
        unbalance_work = self.unbalance_torque_nm * angle
        film_work = self.bearing.friction_torque_nm(rate) * angle
        work = inertial_work + unbalance_work + film_work

        # the air: each working muscle sweeps its volume at supply
        # pressure, and that volume expressed at atmosphere is what the
        # compressor had to deliver
        contraction = min(self.muscle.max_contraction,
                          angle * self.moment_arm_m / max(self.muscle.resting_length_m, 1e-9))
        swept = self.muscle.swept_volume_m3(contraction) * 2.0        # two muscles work
        free_air_m3 = swept * self.supply_pressure_pa / ATM_PA
        duration = angle / rate + accel_time
        free_air_m3 += self.bearing.free_air_l_min / 60_000.0 * duration

        energy_in = free_air_compression_energy_j(free_air_m3, self.supply_pressure_pa)
        self.free_air_used_l += free_air_m3 * 1000.0
        self.work_done_j += work
        self.compression_energy_j += energy_in
        return {
            "angle_deg": math.degrees(angle), "duration_s": duration,
            "work_j": work, "free_air_l": free_air_m3 * 1000.0,
            "compressor_energy_j": energy_in,
            "efficiency": work / max(energy_in, 1e-9),
        }

    def describe(self) -> list[str]:
        out = ["  pneumatic omnidirectional gimbal"]
        out.extend(self.bearing.describe())
        out.extend(self.muscle.describe(self.supply_pressure_pa))
        out.append(f"  {self.muscles} muscles on a {self.moment_arm_m * 1000:.0f} mm arm -> "
                   f"{self.peak_torque_nm:6.0f} Nm peak")
        out.append(f"    unbalance asks {self.unbalance_torque_nm:5.1f} Nm continuously "
                   f"({self.payload_mass_kg:.0f} kg at {self.payload_offset_m * 1000:.0f} mm)")
        out.append(f"    holding still costs {self.hold_cost_l_min():.1f} L/min of free air "
                   f"(the bearing), and nothing from the muscles")
        return out


# =====================================================================
#  WHAT EACH TECHNOLOGY COSTS TO AIM WITH
# =====================================================================
@dataclass(frozen=True)
class AimDrive:
    """One way of moving a mount, and what it really costs.

    `wire_to_work` is measured from the WALL, which is the only fair
    place: it includes making the working fluid, not just spending it.
    """
    key: str
    label: str
    wire_to_work: float
    noise_at_mount_dba: float
    holds_without_power: bool
    note: str = ""


AIM_DRIVES = (
    AimDrive("electric", "electric servo + harmonic drive", 0.72, 52.0, False,
             "the efficient answer, and the one that whines at the mount"),
    AimDrive("hydraulic", "hydraulic vane or rack", 0.55, 68.0, True,
             "huge torque density; the pump runs whether you are moving or not"),
    AimDrive("pneumatic", "pneumatic muscles on an air bearing", 0.12, 38.0, False,
             "expensive in energy, cheap in noise, heat and signature -- and the "
             "noise it does make is at the compressor, not at the mount"),
    AimDrive("manual", "handwheels", 0.95, 30.0, True,
             "the most efficient drive ever built, and the slowest"),
)
AIM_DRIVE_BY_KEY = {d.key: d for d in AIM_DRIVES}


def compare_aim_drives(work_j: float, slews_per_hour: float = 120.0) -> list[str]:
    """What a duty cycle of aiming costs on each technology."""
    out = [f"  aiming work {work_j:,.0f} J per slew, {slews_per_hour:.0f} slews per hour:",
           f"    {'drive':32s} {'input':>10s} {'per hour':>11s} {'noise':>8s}  holds unpowered"]
    for d in AIM_DRIVES:
        per_slew = work_j / max(d.wire_to_work, 1e-9)
        out.append(f"    {d.label:32s} {per_slew:9.0f}J {per_slew * slews_per_hour / 3.6e6:9.3f}kWh "
                   f"{d.noise_at_mount_dba:6.0f}dBA  {'yes' if d.holds_without_power else 'no'}")
    return out


# =====================================================================
#  SIZING IT TO THE JOB
# =====================================================================
def sized_gimbal(*, payload_mass_kg: float, payload_inertia_kg_m2: float,
                 payload_offset_m: float = 0.02, slew_rate_rad_s: float = math.radians(45.0),
                 accel_time_s: float = 0.35, supply_pressure_pa: float = 6.0e5,
                 moment_arm_m: float = 0.150, torque_margin: float = 2.5) -> PneumaticGimbal:
    """Build a gimbal sized for the load rather than for a round number.

    THIS MATTERS MORE FOR AIR THAN FOR ANYTHING ELSE. An oversized
    electric motor wastes a little copper and some standby current. An
    oversized pneumatic muscle swallows its whole swept volume on every
    stroke whether the load needed it or not, so the air bill scales
    with the SIZE OF THE ACTUATOR, not with the work done. The first
    version of this module was built with a 40 mm muscle against a 25 Nm
    unbalance -- seventy-eight times the torque required -- and turned
    in an efficiency of one tenth of one per cent. The physics was
    right; the hardware was ridiculous.

    Sizing rule, in the order a designer would actually apply it:
      1. the torque the job needs -- unbalance, plus inertia through the
         acceleration you want,
      2. times a margin, because a mount that can only just move cannot
         hold against wind or recoil,
      3. the muscle diameter that makes that torque at a sensible
         working contraction, not at its limit,
      4. and a bearing just big enough to carry the payload, because its
         leak is continuous and scales with its orifices.
    """
    unbalance = payload_mass_kg * 9.80665 * payload_offset_m
    inertial = payload_inertia_kg_m2 * slew_rate_rad_s / max(accel_time_s, 1e-3)
    required = (unbalance + inertial) * torque_margin

    # two muscles work the pair, at 120 degrees
    pair = 2.0 * math.sin(math.radians(60.0))
    force_needed = required / (moment_arm_m * pair)

    # invert Gaylord at a 15 % working contraction
    probe = PneumaticMuscle()
    t0 = math.radians(probe.braid_angle_deg)
    shape = (3.0 * (1.0 - 0.15) ** 2 / math.tan(t0) ** 2) - (1.0 / math.sin(t0) ** 2)
    diameter = math.sqrt(force_needed * 4.0 / (math.pi * supply_pressure_pa * max(shape, 1e-9)))
    diameter = max(0.008, math.ceil(diameter * 1000.0) / 1000.0)

    # the bearing carries the payload plus the muscles pulling down on it
    load = payload_mass_kg * 9.80665 + force_needed * 3.0
    ball = max(0.060, math.sqrt(load / (0.35 * supply_pressure_pa) * 4.0 / math.pi))
    ball = math.ceil(ball * 1000.0) / 1000.0

    bearing = AerostaticSphericalBearing(
        ball_diameter_m=ball, supply_pressure_pa=supply_pressure_pa,
        orifices=max(6, int(round(ball * 40.0))), orifice_diameter_m=0.15e-3)
    muscle = PneumaticMuscle(resting_length_m=max(0.12, moment_arm_m * 2.0),
                             diameter_m=diameter)
    return PneumaticGimbal(
        bearing=bearing, muscle=muscle, moment_arm_m=moment_arm_m,
        supply_pressure_pa=supply_pressure_pa,
        payload_inertia_kg_m2=payload_inertia_kg_m2, payload_mass_kg=payload_mass_kg,
        payload_offset_m=payload_offset_m)


def measured_efficiency(gimbal: PneumaticGimbal, *, angle_rad: float = math.radians(90.0),
                        rate_rad_s: float = math.radians(45.0)) -> float:
    """What this particular build actually achieves, rather than a
    figure quoted from a table."""
    probe = PneumaticGimbal(
        bearing=gimbal.bearing, muscle=gimbal.muscle, moment_arm_m=gimbal.moment_arm_m,
        supply_pressure_pa=gimbal.supply_pressure_pa,
        payload_inertia_kg_m2=gimbal.payload_inertia_kg_m2,
        payload_mass_kg=gimbal.payload_mass_kg, payload_offset_m=gimbal.payload_offset_m)
    return probe.slew(angle_rad, rate_rad_s)["efficiency"]
