"""Taking work from propellant gas without letting it near anything clean.

A fired charge throws away most of itself. Of roughly 34 MJ in a 120 mm
charge only about four go into a sub-calibre projectile; the rest leaves
as hot gas and muzzle blast. Tapping some of that is obviously
attractive, and the obvious way to do it is obviously wrong.

DO NOT PIPE PROPELLANT GAS INTO THE AIR SYSTEM. It carries unburnt
propellant, primer residue and copper stripped off the driving band; its
nitrogen oxides make nitric acid with the water it is also carrying; and
it arrives at something like two thousand kelvin. The plant's desiccant
beds are already poisoned by ordinary compressor oil carryover -- this
would finish them on the first shot, and take the refrigerated dryer
behind them at the same time.

SO THE TWO FLUIDS NEVER MEET. The gas drives an EXPANDER, the expander
drives a COMPRESSOR drawing clean ambient air, and the dirty side vents
to atmosphere once its work is done. What crosses between them is a
shaft. The fouling lands on an expander built to be fouled, instead of
on a bed that has to be replaced.

A RECIPROCATING EXPANDER, NOT A TURBINE, and the reason is the gun. A
turbine wants continuous flow and hates particulates -- solids at those
velocities erode blading quickly. A gun delivers one discrete pulse per
shot, which is exactly what a piston expander is: one shot, one power
stroke, one compression stroke of clean air on the other end of the rod.
The awkward pulsed nature of the source is the thing that makes the
reciprocating machine the right one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Propellant combustion gas, not air. Lower molecular weight products
#: give a higher gas constant and a softer ratio of specific heats.
GAS_CONSTANT = 350.0        # J/kg.K
GAMMA = 1.25
ATMOSPHERE_PA = 101_325.0


@dataclass
class ExpansionEngine:
    """A reciprocating expander on a gas port, driving a clean-air
    compressor."""
    identity: str = "gun.gas_harvester"
    #: mass of gas bled per shot. A fraction of a percent of the charge:
    #: take much more and the projectile notices.
    tapped_kg: float = 0.050
    port_pressure_pa: float = 50.0e6
    port_temperature_k: float = 2000.0
    exhaust_pressure_pa: float = ATMOSPHERE_PA
    #: a real machine, doing a fast stroke on a dirty pulse
    expander_efficiency: float = 0.50
    compressor_efficiency: float = 0.65
    #: how much of the charge this represents, for the velocity penalty
    charge_kg: float = 7.9

    # ------------------------------------------------------------------
    @property
    def pressure_ratio(self) -> float:
        return self.port_pressure_pa / max(self.exhaust_pressure_pa, 1.0)

    def ideal_work_j(self) -> float:
        """Polytropic expansion work from port to exhaust.

        W = m R T1 / (n-1) * [1 - (P2/P1)^((n-1)/n)] -- the standard
        expression, with the propellant's own gamma rather than air's."""
        exponent = (GAMMA - 1.0) / GAMMA
        drop = 1.0 - (1.0 / self.pressure_ratio) ** exponent
        return (self.tapped_kg * GAS_CONSTANT * self.port_temperature_k
                / (GAMMA - 1.0) * drop)

    def shaft_work_j(self) -> float:
        return self.ideal_work_j() * self.expander_efficiency

    def exhaust_temperature_k(self) -> float:
        """What the gas has cooled to by the time it is let go.

        Expansion IS cooling -- the work comes out of the internal
        energy. A gas that leaves hot is a gas whose energy was not
        taken."""
        exponent = (GAMMA - 1.0) / GAMMA
        return self.port_temperature_k * (1.0 / self.pressure_ratio) ** exponent

    # ------------------------------------------------------------------
    def air_stored_j(self) -> float:
        """Work actually landing in the clean-air reservoir."""
        return self.shaft_work_j() * self.compressor_efficiency

    def air_charged_kg(self, to_pressure_pa: float = 1.0e6,
                       ambient_k: float = 293.15) -> float:
        """How much ambient air this shot puts away, isothermally.

        w = R T ln(P/P0) per kilogram, with AIR's gas constant because
        this is the clean side."""
        specific = 287.0 * ambient_k * math.log(
            max(to_pressure_pa, ATMOSPHERE_PA * 1.01) / ATMOSPHERE_PA)
        return self.air_stored_j() / max(specific, 1.0)

    def velocity_penalty_fraction(self) -> float:
        """What bleeding this gas costs the projectile.

        Small, because the tapped mass is small -- but it is not zero and
        it is the reason the port is sized in grams rather than
        kilograms."""
        return 0.5 * self.tapped_kg / max(self.charge_kg, 1e-9)

    def shots_to_fill(self, reservoir_l: float = 80.0,
                      to_pressure_pa: float = 1.0e6,
                      ambient_k: float = 293.15) -> float:
        volume = reservoir_l / 1000.0
        energy = (to_pressure_pa * volume
                  * math.log(to_pressure_pa / ATMOSPHERE_PA))
        return energy / max(self.air_stored_j(), 1.0)

    # ------------------------------------------------------------------
    def fouling_per_shot_g(self) -> float:
        """Solids deposited in the expander, which is where they belong.

        Roughly a per cent of the tapped mass is condensed-phase residue
        -- carbon, primer compounds, driving-band metal. It lands here
        instead of on a desiccant bed, and it is why the expander is a
        serviceable item with a wear rate rather than a sealed one."""
        return self.tapped_kg * 0.01 * 1000.0

    def describe(self) -> list:
        return [
            f"  {self.identity}: {self.tapped_kg * 1000:.0f} g tapped at "
            f"{self.port_pressure_pa / 1e6:.0f} MPa, {self.port_temperature_k:.0f} K",
            f"    expansion ratio {self.pressure_ratio:7.0f}:1   "
            f"ideal work {self.ideal_work_j() / 1000:7.1f} kJ   "
            f"at the shaft {self.shaft_work_j() / 1000:7.1f} kJ",
            f"    vents at {self.exhaust_temperature_k():.0f} K "
            f"(from {self.port_temperature_k:.0f} -- the drop IS the work)",
            f"    into clean air: {self.air_stored_j() / 1000:7.1f} kJ per shot, "
            f"{self.air_charged_kg() * 1000:6.1f} g of air at 10 bar",
            f"    costs the projectile {self.velocity_penalty_fraction() * 100:.2f}% "
            f"of muzzle velocity",
            f"    leaves {self.fouling_per_shot_g():.2f} g of residue per shot "
            f"IN THE EXPANDER, not in the dryer",
        ]


# =====================================================================
#  THE HIT-OR-MISS EXPANDER ON A HEAVY FREEWHEEL
# =====================================================================
@dataclass
class HitOrMissExpander:
    """The expander as a MACHINE, with a flywheel and a governor.

    `ExpansionEngine` above answers "how much work is in one shot's
    worth of gas". It is an energy sum and it has no idea what time is.
    That is not enough to build with, because the defining feature of
    this source is that it is PULSED and its rate is decided by
    something else entirely -- the gun.

    WHY HIT-OR-MISS IS THE RIGHT GOVERNOR, and it is not nostalgia. A
    throttled engine meters its charge to match the load. This one
    cannot: the charge arrives when a round is fired and is whatever
    that round had left over. So the only control available is whether
    to ADMIT the pulse or vent it -- fire on this one, skip the next
    two. That is exactly a hit-or-miss engine, and a gun is an ideal
    prime mover for one because its supply is already discrete.

    THE FLYWHEEL IS THE BATTERY. Between shots there is no gas at all,
    so everything drawn in the gaps comes out of stored angular
    momentum. Size it by the longest silence it must run through, not
    by the average power -- an expander that averages fine and stalls
    between bursts has not worked.

    AND EXTREME FIRING IS AN OVERSPEED PROBLEM, not a shortage. Fire
    fast enough and pulses arrive quicker than the load can take them
    away. The governor starts missing, and if every miss still leaves a
    surplus the wheel runs away. The answer is to LOAD IT HARDER --
    dump into the compressor, the alternator, the coolant pumps -- which
    is why heavy firing and heavy electrical demand are the same event
    on this machine, and want to be wired together rather than fought.
    """
    identity: str = "gun.hit_or_miss"
    #: the wheel is a real rim, not a number: J = m r^2
    rim_mass_kg: float = 240.0
    rim_radius_m: float = 0.62
    #: the governor's window
    cut_in_rad_s: float = 88.0
    cut_out_rad_s: float = 104.0
    overspeed_rad_s: float = 126.0
    port: "ExpansionEngine" = None
    bearing_drag_nm: float = 2.4
    #: live
    speed_rad_s: float = 92.0
    hits: int = 0
    misses: int = 0
    vented_kg: float = 0.0
    heat_rejected_j: float = 0.0
    overspeed_events: int = 0

    def __post_init__(self) -> None:
        if self.port is None:
            self.port = ExpansionEngine()

    @property
    def inertia_kg_m2(self) -> float:
        return self.rim_mass_kg * self.rim_radius_m ** 2

    @property
    def stored_j(self) -> float:
        return 0.5 * self.inertia_kg_m2 * self.speed_rad_s ** 2

    def usable_j(self, floor_rad_s: float = None) -> float:
        """What you may actually take out. Below the governor's cut-in
        the wheel is not a store any more, it is a thing that stopped."""
        floor = self.cut_in_rad_s if floor_rad_s is None else floor_rad_s
        return max(0.0, self.stored_j - 0.5 * self.inertia_kg_m2 * floor ** 2)

    def coast_seconds(self, load_w: float) -> float:
        """How long it carries that load with no shots at all."""
        return self.usable_j() / max(load_w, 1e-9)

    def load_for_rate(self, shots_per_s: float) -> float:
        """The load this rate of fire has to be matched with.

        Under-loading a hit-or-miss engine on a rich supply is how it
        runs away, so the right response to heavy firing is to switch
        heavy consumers ON."""
        return self.port.shaft_work_j() * shots_per_s

    # ------------------------------------------------------------------
    def step(self, dt: float, shots: int = 0, load_w: float = 0.0) -> dict:
        """One tick. `shots` is how many rounds went off in it -- the gun
        decides that, not this machine."""
        admitted = 0
        for _ in range(max(0, int(shots))):
            if self.speed_rad_s < self.cut_out_rad_s:
                work = self.port.shaft_work_j()
                self.speed_rad_s = math.sqrt(max(
                    0.0, self.speed_rad_s ** 2 + 2.0 * work / self.inertia_kg_m2))
                self.hits += 1
                admitted += 1
                # expansion IS cooling: it leaves cooler because the work
                # came out of its own internal energy
                self.heat_rejected_j += (
                    self.port.tapped_kg * 1150.0
                    * max(0.0, self.port.exhaust_temperature_k() - 293.15))
            else:
                # MISS: the inlet stays shut, the pulse goes out of the
                # port. Nothing gained, nothing broken -- and the heat
                # that would have been taken leaves at full temperature.
                self.misses += 1
                self.vented_kg += self.port.tapped_kg
                self.heat_rejected_j += (
                    self.port.tapped_kg * 1150.0
                    * max(0.0, self.port.port_temperature_k - 293.15))
        drawn = load_w * dt + self.bearing_drag_nm * self.speed_rad_s * dt
        self.speed_rad_s = math.sqrt(max(
            0.0, self.speed_rad_s ** 2 - 2.0 * drawn / self.inertia_kg_m2))
        if self.speed_rad_s > self.overspeed_rad_s:
            self.overspeed_events += 1
        return {
            "speed_rad_s": self.speed_rad_s,
            "rpm": self.speed_rad_s * 60.0 / (2.0 * math.pi),
            "stored_j": self.stored_j,
            "admitted": admitted,
            "hits": self.hits,
            "misses": self.misses,
            "duty": self.hits / max(self.hits + self.misses, 1),
            "overspeed": self.speed_rad_s > self.overspeed_rad_s,
            "heat_rejected_j": self.heat_rejected_j,
        }

    def describe(self) -> list:
        return [
            f"{self.identity}: {self.rim_mass_kg:.0f} kg rim at "
            f"{self.rim_radius_m * 1000:.0f} mm = {self.inertia_kg_m2:.1f} kg.m2",
            f"  governor {self.cut_in_rad_s * 9.549:.0f}-"
            f"{self.cut_out_rad_s * 9.549:.0f} rpm, overspeed at "
            f"{self.overspeed_rad_s * 9.549:.0f} rpm",
            f"  stored {self.stored_j / 1000:.1f} kJ, usable "
            f"{self.usable_j() / 1000:.1f} kJ",
            f"  one admitted shot gives it "
            f"{self.port.shaft_work_j() / 1000:.2f} kJ",
        ]


# =====================================================================
#  REAL FLOW, FROM REAL LOCAL PRESSURE
# =====================================================================
@dataclass
class CollectorPort:
    """A port into the collector, flowing what the pressure actually drives.

    WHAT THIS REPLACES. The interior ballistics kernel had a
    `gas_port_fraction` -- "take this fraction of the chamber gas per
    second" -- which is not a physical quantity at all. A hole does not
    remove a fraction of anything; it passes the mass flow that the
    pressure across it and its own area allow, and at gun pressures
    that flow is CHOKED: the gas leaves at the local speed of sound and
    the downstream pressure stops mattering entirely.

    Choked flow through an orifice is one equation and every term in it
    is something the kernel already knows or the hardware declares:

        mdot = Cd A p0 sqrt(gamma / (R T0)) * ((gamma+1)/2)^-((gamma+1)/(2(gamma-1)))

    so the collector's take stops being a tuning knob and becomes a
    consequence of how big the hole is.
    """
    identity: str = "gun.collector_port"
    area_m2: float = 3.0e-5              # a 6 mm hole
    discharge_coefficient: float = 0.80
    gas_constant: float = 350.0
    gamma: float = 1.25
    #: the collector behind it. Choked flow ignores this until the
    #: pressure ratio falls below critical, and then it does not.
    collector_pressure_pa: float = 2.0e6

    @property
    def critical_ratio(self) -> float:
        g = self.gamma
        return (2.0 / (g + 1.0)) ** (g / (g - 1.0))

    def mass_flow_kg_s(self, upstream_pa: float, upstream_k: float) -> float:
        """What this hole passes, right now."""
        if upstream_pa <= self.collector_pressure_pa:
            return 0.0
        g = self.gamma
        choked = (self.collector_pressure_pa / upstream_pa) <= self.critical_ratio
        flux = (upstream_pa * math.sqrt(g / (self.gas_constant * max(upstream_k, 1.0)))
                * ((g + 1.0) / 2.0) ** (-(g + 1.0) / (2.0 * (g - 1.0))))
        if not choked:
            # subsonic: the downstream pressure is now in charge
            r = self.collector_pressure_pa / upstream_pa
            flux *= math.sqrt(max(0.0, (r ** (2.0 / g) - r ** ((g + 1.0) / g))
                                  * 2.0 * g / (g - 1.0)))
        return self.discharge_coefficient * self.area_m2 * flux

    def describe(self) -> list:
        return [
            f"{self.identity}: {math.sqrt(self.area_m2 / math.pi) * 2000:.1f} mm "
            f"hole, Cd {self.discharge_coefficient}",
            f"  chokes until the pressure ratio falls below "
            f"{self.critical_ratio:.3f}",
            f"  collector held at "
            f"{self.collector_pressure_pa / 1e6:.1f} MPa",
        ]
