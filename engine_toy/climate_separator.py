"""The held atmosphere inside a climate-controlled separator.

climate_separator_production.py builds the real machine -- the sealed
cabinet, the drum, the coil, the heater, the compressor that feeds it.
This module is the one number that machine exists to hold: the
temperature and the composition of the gas inside it, and what it
takes to keep both where they are set.

WHY A GAS LOOP AND NOT JUST "THE CHAMBER IS COLD". A centrifugal cut
depends on the oil's viscosity (centrifuges.oil_viscosity_pa_s), and
viscosity is exponential in temperature -- so a separator whose bowl
runs at whatever the shop happens to be that day is not a sized
machine, it is a guess with a nameplate. Holding the CHAMBER at a
declared setpoint, rather than the product, is what a real climate
cell does: heat and cold act on the gas around the bowl, and the gas
carries it to everything inside, uniformly, because the recirculation
blower makes that true rather than assumed. `centrifuge_for` is the
other half of the same connection: read `ClimateChamber.temp_k`
straight into `centrifuges.Centrifuge.cut_size_m(flow, oil_temp_k=...)`
and the cut this machine achieves is genuinely a function of what the
chamber is actually holding, not a number picked beside it.

THIS DOES NOT REINVENT THE ATMOSPHERE OR THE WALL. Both problems --
"a real gas mixture sealed in a vessel, with a pressure and a way to
vent it" and "conduction through an insulating wrap, hot side to cold
side" -- are already solved, correctly, in cryogenics.py, for a dewar.
Nothing about either solution is cryogenic:

    DewarAtmosphere    mass PER SPECIES, an ideal-gas pressure off a
                        declared volume, `add` (take gas in, at its own
                        temperature) and `remove_proportional` (vent a
                        fraction -- everything leaves in the proportion
                        it is present, which is exactly what an open
                        purge valve does)
    wrap_heat_leak_w    plain Fourier conduction through a declared
                        insulation medium and thickness, hot side to
                        cold side -- and it is already bidirectional
                        (abs(outer_k - inner_k)), which is the one
                        property this module actually needs that a
                        dewar never does, because THIS chamber runs hot
                        as often as it runs cold

So `ClimateChamber` composes a `DewarAtmosphere` for what it holds and
calls `wrap_heat_leak_w` for what it loses, rather than carrying a
second gas-mass field and a flat W/K guess beside them. One inert-gas
enclosure law set, used by a dewar and by this machine, is the whole
point of putting it in cryogenics.py rather than autoclave.py in the
first place.

GAS INLET AND PURGE ARE PORTS, NOT A PUMPED CYCLE -- BUT A PORT STILL
HAS CONSEQUENCES. autoclave.py's PurgeCycle models a real pump pulling
a real vacuum against a porous load, pulse by pulse, and this chamber
does not need that machinery to state the plain fact underneath it:
a sealed box does not evacuate itself by gravity. Crack `purge_open`
on a chamber that is above ambient and its own charge blows itself
down, same as before -- but once the pressure inside reaches ambient
there is nothing left pushing gas OUT, and an open valve past that
point is a hole ambient air flows IN through. That is not a modelling
choice, it is what the pressure differential is, and it is the whole
reason `if you don't vacuum or fully purge you get a mix` is true of a
real chamber and has to be true of this one: `DewarAtmosphere.add` and
`.remove_proportional` already make a genuine mixture out of whatever
is now in the box, so backflowing air through the SAME port the charge
just left through costs nothing extra to get right.

Getting BELOW ambient -- the only way to hold the mix down near zero
rather than merely near what one atmosphere's worth of dilution
allows -- needs an actual vacuum pump on the line, and this module
does not invent a second one: attach a real `compressors.VacuumPump`
as `vacuum_pump` and its own `ultimate_pa()` becomes the floor the
purge can reach, exactly the way autoclave.PurgeCycle.pump_floor_frac
already refuses to pretend a pump can pull past its own clearance.
Leave it unattached and the chamber is honestly limited to blowing
itself down to atmospheric and no further -- which is the case this
module has to get right by default, because most enclosures do not
have a vacuum pump hung off their purge line and are not thereby
exempt from what opening one does."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import fluids
from cryogenics import DewarAtmosphere, wrap_heat_leak_w, insulation, ATM_PA

#: The chamber's working gas is fluids.py's "gas" row -- the same
#: numbers a nitrogen blanket circuit anywhere else in the project
#: would read, so a change to one is a change to both.
_GAS = fluids.BY_KEY["gas"]
GAS_CP_J_PER_KG_K = _GAS.specific_heat_j_per_kg_k

AMBIENT_K = 293.15
CHARGE_SPECIES = "nitrogen"
#: What backflows through an open purge once there is nothing left
#: pushing the charge out through it -- see the module docstring.
AIR_SPECIES = "air"
#: A backstop, not an operating limit -- this machine has no business
#: running anywhere near either bound. Below it nitrogen has left the
#: gas phase and "gas temperature" stops meaning anything; above it is
#: past anything a purifier-duty chamber like this is built for.
TEMP_FLOOR_K = 90.0
TEMP_CEILING_K = 500.0


def box_surface_area_m2(inside_m: tuple) -> float:
    """A sealed box's own skin, from the inside dimensions
    climate_separator_production.CabinetSpec already declares -- so the
    insulation's area is read off the real cabinet rather than guessed
    beside it, the same discipline centrifuge_for applies to the bowl."""
    w, h, d = inside_m
    return 2.0 * (w * h + w * d + h * d)


#: Nitrogen's own specific gas constant (R / M), the same figure
#: DewarAtmosphere.pressure_pa already uses for every non-hydrogen fill.
NITROGEN_R_J_PER_KGK = 296.8


def charge_mass_for_pressure_kg(pressure_pa: float, volume_m3: float,
                                temp_k: float = AMBIENT_K,
                                r_specific: float = NITROGEN_R_J_PER_KGK) -> float:
    """How much gas a declared ABSOLUTE pressure actually is, in this
    chamber's own volume -- so a build asking for "a slight positive
    hold" gets a fill sized to that hold rather than a fixed mass that
    happens to be wildly wrong for whatever box it ends up in."""
    return max(0.0, float(pressure_pa)) * max(1e-6, float(volume_m3)) / (
        r_specific * max(1.0, float(temp_k)))


@dataclass
class ClimateChamber:
    """The gas actually inside the sealed cabinet: one lumped mass and
    one real mixture, because the recirculation blower's entire job is
    to make "one lumped mass" a true statement rather than an
    approximation. Small compared to a duct or a room -- it is the
    free volume of one cabinet, not a building -- which is exactly why
    it answers fast and why a chamber like this can hold a setpoint a
    warehouse never could."""
    volume_m3: float = 0.35
    surface_area_m2: float = 3.0
    insulation_media: str = "aerogel-blanket"     # no vacuum needed, hot or cold
    insulation_thickness_m: float = 0.05
    target_k: float = AMBIENT_K
    #: rated electric trim heat; live draw is bounded by how far under
    #: setpoint the chamber actually is, the same shape HydraulicTank
    #: uses for its own chiller demand
    heater_kw: float = 1.5
    #: heat the spinning bowl itself puts back in -- windage and
    #: bearing drag, the same return_heat_w role HydraulicTank's
    #: reservoir carries for the pump it cools
    return_heat_w: float = 0.0
    cp_j_kg_k: float = GAS_CP_J_PER_KG_K
    #: the two valves, read from climate_separator_production's own
    #: `gas_inlet` / `purge` ports -- open or shut, nothing between
    inlet_open: bool = False
    purge_open: bool = False
    inlet_flow_kg_s: float = 0.02
    purge_flow_kg_s: float = 0.02
    #: a real compressors.VacuumPump on the purge line, or None -- see
    #: the module docstring for what having one changes and what not
    #: having one costs
    vacuum_pump: object | None = None
    atmosphere: DewarAtmosphere = field(
        default_factory=lambda: DewarAtmosphere(kg={CHARGE_SPECIES: 1.4}, temp_k=AMBIENT_K))

    @property
    def temp_k(self) -> float:
        return self.atmosphere.temp_k

    @property
    def pressure_pa(self) -> float:
        return self.atmosphere.pressure_pa(self.volume_m3)

    @property
    def inert_frac(self) -> float:
        return self.atmosphere.fractions().get(CHARGE_SPECIES, 0.0)

    @property
    def demand_w(self) -> float:
        """What the coil is being asked for: only above setpoint,
        proportional to how far over -- refrigeration.HydraulicTank's
        own demand_w, read for a gas mass instead of an oil charge."""
        over = self.temp_k - self.target_k
        return max(0.0, min(4000.0, over * 250.0))

    @property
    def heater_calling(self) -> bool:
        return self.temp_k < self.target_k

    def step(self, dt: float, cooling_w: float, ambient_k: float = AMBIENT_K) -> dict:
        """One tick. `cooling_w` is whatever refrigeration.RefrigerantLoop
        actually delivered this step to this chamber's CoolingLoad --
        never assumed, always read off the loop the way
        RefrigerantLoop.from_compressor_node insists the compressor's own
        rating is read off the graph."""
        dt = max(0.0, float(dt))
        if self.inlet_open:
            self.atmosphere.add(CHARGE_SPECIES, self.inlet_flow_kg_s * dt, at_k=ambient_k)
        if self.purge_open:
            floor_pa = self.vacuum_pump.ultimate_pa() if self.vacuum_pump is not None else ATM_PA
            p = self.pressure_pa
            if p > floor_pa:
                # something is still pushing the charge OUT -- either
                # the chamber's own overpressure against atmosphere, or
                # a pump holding suction below it. Capped to what is
                # actually IN EXCESS of the floor: pressure is linear in
                # mass at fixed volume and temperature, so removing more
                # than that overshoots past the floor in one tick rather
                # than asymptoting to it the way a real valve does.
                excess_kg = self.atmosphere.total_kg * (1.0 - floor_pa / p)
                self.atmosphere.remove_proportional(
                    min(self.purge_flow_kg_s * dt, max(0.0, excess_kg)))
            elif self.vacuum_pump is None:
                # nothing left to blow itself out with, and no pump
                # pulling: the open valve is now a hole ambient air
                # flows IN through, which is the mix a crack-the-valve
                # purge with no vacuum actually leaves you with
                self.atmosphere.add(AIR_SPECIES, self.purge_flow_kg_s * dt, at_k=ambient_k)
            # a pump holding the chamber at its own ultimate is doing
            # its job; nothing more to do this tick
        m = max(1e-6, self.atmosphere.total_kg)
        heater_w = self.heater_kw * 1000.0 if self.heater_calling else 0.0
        # THE STABLE FORM, not forward Euler. dT/dt = (Q - UA(T-Tamb)) /
        # (m.cp) is a linear relaxation toward a steady state T_ss with
        # time constant tau = m.cp/UA, and that has an EXACT closed
        # form -- the same trick cryogenics.VacuumJacketedVessel.
        # _warm_ullage already uses for its own ullage, and for the same
        # reason: a fixed per-tick nudge makes the answer depend on the
        # step size, and can overshoot or ring for a big one. The
        # exponential form is exact at any dt and cannot.
        #
        # UA is the wrap's own conductance -- exactly wrap_heat_leak_w
        # divided by its gap, since that function is already linear in
        # the temperature difference, so a 1 K probe reads UA directly
        # without a second, disagreeing formula for the same wrap.
        ua = wrap_heat_leak_w(self.insulation_media, self.surface_area_m2,
                              self.insulation_thickness_m, ambient_k + 1.0, ambient_k)
        q_const = self.return_heat_w + heater_w - cooling_w
        if ua <= 0.0:
            self.atmosphere.temp_k += q_const * dt / (m * self.cp_j_kg_k)
        else:
            tau = m * self.cp_j_kg_k / ua
            t_ss = ambient_k + q_const / ua
            self.atmosphere.temp_k = t_ss + (self.temp_k - t_ss) * math.exp(
                -dt / max(1e-9, tau))
        # THE EXACT SOLUTION IS EXACT FOR THE Q IT WAS GIVEN, not for
        # what Q ought to have been partway through a long step -- a
        # `cooling_w` held fixed for an hour by a caller that never
        # re-reads `demand_w` mid-step can still walk this toward a
        # steady state past where the gas is gas at all. That is a
        # caller-pacing problem, not an integration one, and it is
        # exactly what `dt_limit_s` below exists to let a scheduler
        # avoid; this floor is the backstop for a caller that does not
        # use one, the same role cryogenics.VacuumJacketedVessel's own
        # `max(self.boil_k, min(ambient_k, ...))` plays for its ullage.
        self.atmosphere.temp_k = max(TEMP_FLOOR_K, min(TEMP_CEILING_K, self.atmosphere.temp_k))
        return {"temp_k": self.temp_k, "pressure_pa": self.pressure_pa,
               "heater_w": heater_w, "cooling_w": cooling_w}

    def dt_limit_s(self, ambient_k: float = AMBIENT_K, *, safety: float = 0.5) -> float:
        """This chamber's own proposed stability/control-resolution
        hint, in the shape `time_allocator.derive_dt_limit_s` already
        establishes for a drivetrain: an engine PROPOSES a limit into
        `Metrics.dt_limit` (via `time_allocator.simple_metrics`) and the
        shared dt_system's central controller is what enforces
        subdivision -- an engine does not self-clamp.

        Nothing about `step`'s own thermal integration NEEDS this: the
        closed form above is exact and unconditionally stable for
        whatever dt it is handed, the way every other lumped-parameter
        `.step(dt)` object in this project (refrigeration.HydraulicTank,
        cryogenics.VacuumJacketedVessel, centrifuges.Centrifuge) already
        is, and none of those register with dt_system either -- it is
        scoped to the vehicle/engine realtime scheduling layer
        (time_allocator.py, engine_abi.py), not to a single thermal
        mass. What DOES depend on dt is the heater's own bang-bang
        decision, evaluated once per call: a caller stepping this
        chamber coarser than its own thermal time constant will see a
        less responsive thermostat, exactly as a real one would if
        sampled that rarely, not a wrong answer -- so this is a
        real-usefulness hint for a caller wanting tight setpoint
        behaviour, not a correctness requirement."""
        ua = wrap_heat_leak_w(self.insulation_media, self.surface_area_m2,
                              self.insulation_thickness_m, ambient_k + 1.0, ambient_k)
        if ua <= 0.0:
            return 0.0
        tau = self.atmosphere.total_kg * self.cp_j_kg_k / ua
        return safety * tau

    def describe(self) -> str:
        media = insulation(self.insulation_media)
        return (f"  CHAMBER {self.temp_k - 273.15:.1f} C (target {self.target_k - 273.15:.0f} C)  "
               f"{self.pressure_pa / 1000.0:.1f} kPa  calling {self.demand_w / 1000:.2f} kW\n"
               f"    {self.atmosphere.total_kg * 1000:.0f} g charge, "
               f"{self.inert_frac * 100:.0f}% {CHARGE_SPECIES}, "
               f"{media.label} wrap {self.insulation_thickness_m * 1000:.0f} mm")


def centrifuge_for(drum, drive, *, kind: str = "solid-bowl-batch"):
    """A centrifuges.Centrifuge sized off this machine's own Drum and
    DrumDrive, rather than a second, disagreeing set of numbers.

    cabinet.Drum already declares the bowl's radius and length and its
    rated speed; DrumDrive already declares the motor that turns it.
    Every separation number -- sigma, cut size, g-force -- has to read
    those, not guess a bowl beside them, the same discipline
    RefrigerantLoop.from_compressor_node applies to the compressor.

    THE LOOP CLOSES ON THE CHAMBER'S OWN TEMPERATURE: call
    `unit.cut_size_m(flow_m3_s, oil_temp_k=chamber.temp_k)` on the
    Centrifuge this returns, not a fixed 363.15 K default -- that is
    what makes the climate control worth building at all."""
    from centrifuges import Centrifuge
    inner = drum.radius_m * 0.35
    return Centrifuge(kind=kind, bowl_radius_m=drum.radius_m, inner_radius_m=inner,
                      bowl_height_m=drum.length_m, speed_rpm=drum.rated_rpm,
                      drive="electric-direct" if (drive is not None and drive.kind == "direct")
                      else "electric-belt",
                      motor_kw=drive.motor_kw if drive is not None else 0.0,
                      bowl_mass_kg=drum.mass_kg)
