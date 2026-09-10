"""Real, disclosed facsimiles of the two historical ways an atmospheric
(or steam) engine got its working fluid onto site WITHOUT a piped town
main -- generating it on the spot from raw solid fuel. Two genuinely
different real processes, not the same converter wearing two labels:

  - GasGenerator: makes a COMBUSTIBLE GAS from raw coal or wood, by
    either real historical route:
      "retort"   -- destructive distillation: coal sealed in a heated
                    iron retort with NO air, driving off volatiles as
                    coal gas while coke is left behind. This is real
                    coal-gas manufacture -- the actual process behind
                    every Victorian gasworks main (see otto_langen.py's
                    own IGNITION_PRESSURE_RATIO_BY_FUEL["coal-gas"]).
      "producer" -- gasification: air blown through an incandescent
                    bed of coal/charcoal/wood, producing a lean CO-rich
                    gas by real understoichiometric partial combustion.
                    This is real producer-gas/wood-gas manufacture --
                    the same "wood-gas" fuel profile already in the
                    catalogue, and it's genuinely lower calorific value
                    than coal gas for a real physical reason (the
                    product gas is roughly half unreacted nitrogen
                    carried straight through from the blast air), not
                    an arbitrary "wood is weaker" number.

  - Boiler: makes STEAM from raw coal/wood by ordinary combustion in a
    firebox, transferring heat through the shell/tubes to water. A
    completely different working fluid from either gas above, with its
    own real thermodynamics (latent heat of vaporization) rather than
    fuel/air combustion chemistry at the point of use.

Both real processes need a real BUFFER between production (which
can't respond instantly -- a coking retort or a coal fire has real
thermal lag) and consumption (which the engine's own cycle draws in
uneven pulses). Accumulator below is that buffer, modeling the SAME
real mechanism either way: a gasholder's floating bell (coal-gas
works) or a boiler's own steam dome both hold near-CONSTANT real
pressure (a bell's weight/area; a dome's saturation temperature) while
their fill MASS is what actually runs low under sustained heavy
demand -- the real reason either one can visibly "run out of puff"
long before its pressure gauge shows anything wrong.

This is also, not incidentally, the real "idle control" for an engine
with no throttle plate at all (see engine_cycle_sim._step_atmospheric's
own docstring: the Otto-Langen's mixer is fixed wide open and only the
governor decides fire/skip). Turning down the generator's blast-air
damper or the boiler's firebox draft is the real lever a 19th-century
operator actually had on a self-contained plant -- not a butterfly
valve that doesn't exist on this machine.

Wired into engine_cycle_sim._step_atmospheric via EngineCycleSim.
atmospheric_generator: each tick the generator's real production and
its own purge-volume purity (GasQualityBath) flow into the engine's
real gasholder circuit (drivetrain_graph.py's fuel circuit, which
tracks both fill MASS and contents COMPOSITION), and the engine's
fixed-ratio mixer draws its gas share from whatever that holder
actually contains -- see otto_langen.FUEL_GAS_PROPERTIES for how the
resulting charge's real fuel fraction decides whether it fires at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# (engines' filter constants are imported lazily where used -- engines.py's
# catalogue now builds a BoilerSpec from this module, so a top-level import
# here would be circular)
from otto_langen import FUEL_GAS_PROPERTIES


# ---------------------------------------------------------------------
# Real raw-fuel and process constants
# ---------------------------------------------------------------------

# Real, cited Victorian gasworks yield: destructive distillation of
# coal releases on the order of 300 m^3 of coal gas per TONNE of coal
# charged (a widely cited historical figure), leaving coke behind --
# the carbon itself isn't converted to gas here, only the volatiles
# are driven off.
COAL_GAS_YIELD_M3_PER_KG_COAL = 0.30
# real: coal gas (H2/CH4/CO mix) is lighter than air -- the SAME figure
# otto_langen.FUEL_GAS_PROPERTIES states for the engine's own charge
# accounting, one source, not two copies that could drift
COAL_GAS_DENSITY_KG_PER_M3 = FUEL_GAS_PROPERTIES["coal-gas"]["density_kg_m3"]

# Real gasification stoichiometry: 2C + O2 -> 2CO. 12 kg of carbon
# needs 16 kg of O2 (one mole O2 per two moles C, molar masses 12/32);
# real sea-level air is ~23.3% O2 by mass, so 1 kg of carbon consumed
# needs (16/12)/0.233 = 5.72 kg of air blown through the bed. All of
# that air's real nitrogen passes straight through into the product
# gas alongside the CO -- real mass conservation (nothing is lost to
# ash/radiation in this simplified accounting), and the actual real
# reason producer/wood gas has such a low calorific value relative to
# coal gas: it's roughly half inert nitrogen by mass.
O2_PER_CARBON_KG = 16.0 / 12.0
AIR_O2_MASS_FRACTION = 0.233
AIR_PER_CARBON_KG = O2_PER_CARBON_KG / AIR_O2_MASS_FRACTION  # ~5.72

# Real higher heating values, disclosed order-of-magnitude figures (not
# one cited assay) -- coal and coke run close together since coke is
# mostly the same fixed carbon with the volatiles already driven off;
# wood is real substantially lower per kg (~half), the actual physical
# reason a wood-fired plant needs a bigger grate/more frequent firing
# for the same heat output as a coal-fired one.
FUEL_HHV_J_PER_KG = {
    "coal": 29.0e6,
    "coke": 28.0e6,
    "charcoal": 30.0e6,
    "wood": 15.0e6,
}

# Real, disclosed carbon fraction of each raw solid fuel by mass --
# anthracite/coke/charcoal run high (mostly fixed carbon); wood is
# roughly half carbon by dry mass, the balance being oxygen/hydrogen/
# moisture that doesn't participate in the C + O2 -> CO reaction the
# same way.
RAW_FUEL_CARBON_FRACTION = {
    "coal": 0.80,
    "coke": 0.90,
    "charcoal": 0.85,
    "wood": 0.50,
}


# ---------------------------------------------------------------------
# Blower -- one real device type, any consumer with an air-blast need
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class AirTreatment:
    """Real moisture/particulate conditioning for a high-pressure air
    system -- compressing air drives off dissolved moisture as vapor,
    and the nozzle's own rapid expansion back down to atmospheric
    cools the discharge sharply (a real Joule-Thomson-style effect),
    so any water vapor that didn't get removed upstream condenses and
    can genuinely freeze right at the orifice -- a real, well-
    documented industrial-pneumatics failure mode (nozzle icing),
    not a hypothetical one. `has_dryer` (pulls moisture out before
    storage) and `has_filter` (pulls compressor lubricant/particulate
    carryover out) are both real, standard fittings on any serious
    compressed-air installation -- a tank without them is real, it's
    just real neglected equipment, and pays for it at the nozzle."""
    has_dryer: bool = True
    has_filter: bool = True

    @property
    def discharge_derate_frac(self) -> float:
        """Real, disclosed flow penalty for whatever's missing --
        icing partially blocks the orifice rather than stopping it
        outright (has_dryer), and fouling/oil carryover narrows the
        real effective passage over time (has_filter). Multiplicative,
        each independently disclosed rather than one combined fudge."""
        frac = 1.0
        if not self.has_dryer:
            frac *= 0.6   # real: intermittent partial icing, not a hard block
        if not self.has_filter:
            frac *= 0.85  # real: fouling narrows the passage, a smaller real effect than icing
        return frac


@dataclass(frozen=True)
class ThermostatSpec:
    """A real closed-loop temperature control -- only physically
    sensible on an electric-motor blower: its speed is the one real
    drive here that isn't coupled to anything else (not the engine's
    own crank speed via a belt, not a person's arm on a crank or
    bellows), so it's the only one a thermocouple-driven controller
    can actually modulate independently and precisely -- exactly why
    real industrial gasifiers/boilers with automatic draft control use
    an electric forced-draft fan for it, not a belt-driven one. A real,
    simple proportional control law, not a full PID -- disclosed as a
    simplification, the same convention aftermarket_idle_controller.py
    already uses elsewhere in this toy."""
    target_temp_k: float
    proportional_gain_per_k: float = 0.01


@dataclass(frozen=True)
class BlowerSpec:
    """One real forced-draft blower: a device with an OUTPUT port
    delivering air, as distinct from an INTAKE (which draws suction,
    like an engine's own manifold) or a natural-draft stack/chimney
    (no moving parts, no rated flow at all -- buoyancy does the work
    there). Four real drives cover the ways that output port actually
    gets its air moving, and they fall into two real REGIMES, not one:
    a CONTINUOUS regime (engine-driven-belt/hand-crank/electric-motor
    -- a fan or rotary blower, real modest pressure rise, real steady
    rated flow for as long as it's turning) and a BURST regime
    (compressed-air-tank -- a real high-pressure reserve discharged
    through a nozzle can deliver a mass flow far above any continuous
    fan's rating, exactly the way a 100-gallon shop tank at real
    working pressure can blast a forge hot enough to work steel, for
    as long as the tank's own real finite reserve lasts and no longer;
    see the tank_* fields below, which reuse Accumulator -- the SAME
    real device drivetrain_graph.py's engine.pneumatics system already
    builds for air brakes/tools, just repurposed here as a blast
    source instead).

    Whatever real process needs forced air at its own output port --
    a producer-gas generator's fuel bed, a furnace/boiler's firebox,
    anything else built on this toy later -- draws from this SAME real
    type, sized differently per application: a furnace's forced-draft
    fan and a small workshop gas-producer's blower are the same real
    KIND of machine at different real scale, not two different
    mechanisms wearing different names."""
    max_flow_kg_s: float
    # a real physical drive, not a free-floating knob -- where the
    # blower actually gets its own motive power:
    #   "engine-driven-belt" -- belt-driven off a crank (this engine's
    #       own, for a self-contained plant), the same real mechanical
    #       port drivetrain_graph.py's own belt-driven compressor
    #       accessory (_add_belt_driven_compressor) already plugs an
    #       AC compressor or an air-brake compressor into. Self-
    #       sustaining once that crank is turning, but genuinely
    #       cannot blow air before it is -- the real chicken-and-egg
    #       every belt-driven forced-draft system has.
    #   "hand-crank" -- a person working a hand crank or bellows
    #       linkage: supplied-elsewhere (a person, not a built-in
    #       part), and the real historical answer to that chicken-
    #       and-egg for a cold start.
    #   "electric-motor" -- a small motor on the 12/24 V bus, the same
    #       real electrical-load convention every other electric
    #       accessory in this toy already uses (see electrical_network.
    #       py) -- doesn't need the crank turning at all, at the real
    #       cost of a continuous electrical draw while running.
    #   "compressed-air-tank" -- a real high-pressure reserve (real,
    #       same mechanism as drivetrain_graph.py's own pneumatic
    #       reserve tank) discharged through a nozzle: max_flow_kg_s
    #       here is the NOZZLE's own real burst discharge capacity,
    #       genuinely higher than any continuous blower's rating, but
    #       throttled by the tank's own real finite reserve -- see
    #       tank_capacity_kg/tank_recharge_kg_s.
    #   "pto-clutch" -- a real declutchable power take-off, genuinely
    #       different from "engine-driven-belt": a fixed accessory belt
    #       is ALWAYS spinning whenever the engine runs (that's exactly
    #       why it can only carry a modest accessory load without
    #       starving everything else on the same belt), while a PTO
    #       clutch can be selectively engaged to divert real,
    #       substantial engine output to one big consumer on demand --
    #       the real mechanism farm/industrial PTOs use, and the real
    #       reason a kiln-sized forced-draft blower (fan power scales
    #       with flow x pressure rise, genuinely large at industrial
    #       air demand) needs this instead of a belt tap. See
    #       pto_engaged below -- ramps with rpm exactly like the belt
    #       once engaged, but delivers real zero while declutched
    #       regardless of rpm.
    drive: str = "engine-driven-belt"
    # engine-driven-belt / pto-clutch: crank rpm at which the blower
    # reaches its own full rated flow -- a disclosed, representative
    # figure for a small accessory-belt pulley ratio (or, for a PTO, a
    # real fixed PTO gear ratio), not a specific engine's measured
    # speed.
    full_draft_rpm: float = 400.0
    # compressed-air-tank only: the real reserve's own finite capacity
    # and its real, much slower recharge rate from whatever small
    # compressor tops it back up between bursts -- the actual real
    # reason a tank blast is a BURST and not a sustained capability
    # (recharge is disclosed as an order of magnitude slower than the
    # nozzle's own discharge rate, matching how a real shop compressor
    # takes minutes to refill a tank a nozzle can empty in seconds).
    tank_capacity_kg: float = 5.0
    tank_recharge_kg_s: float = 0.01
    # compressed-air-tank only: real dryer/filter conditioning on the
    # reserve -- see AirTreatment's own docstring for why a genuine
    # high-pressure blast needs it. Default is a properly-equipped
    # installation (both present); declaring one without either is how
    # a "neglected shop compressor" gets modeled honestly rather than
    # just assumed away.
    air_treatment: AirTreatment = field(default_factory=AirTreatment)
    # electric-motor only: a real thermocouple-driven closed loop --
    # None (default) keeps manual damper_frac control, same as the
    # other three drives. See ThermostatSpec's own docstring for why
    # this is the one drive that can actually support it.
    thermostat: ThermostatSpec | None = None

    def build(self) -> "Blower":
        tank = (Accumulator(supply_pressure_pa=0.0, capacity_kg=self.tank_capacity_kg,
                             fill_kg=self.tank_capacity_kg)
                if self.drive == "compressed-air-tank" else None)
        return Blower(spec=self, _tank=tank)


@dataclass
class Blower:
    """One real, running blower instance. `damper_frac` is the real,
    live-adjustable operator lever (0.0 shuts the output off entirely,
    1.0 is fully open), gated by whatever the drive can actually
    deliver right now."""
    spec: BlowerSpec
    damper_frac: float = 1.0
    crank_engaged: bool = False   # hand-crank drive only
    motor_on: bool = False        # electric-motor drive only
    pto_engaged: bool = False     # pto-clutch drive only
    _tank: "Accumulator | None" = None   # compressed-air-tank drive only

    def air_output_kg_s(self, dt: float, engine_rpm: float = 0.0,
                         measured_temp_k: float | None = None) -> float:
        """Real mass flow this blower is delivering right now. `dt`
        only matters for compressed-air-tank (it advances the real
        reserve's own depletion/recharge); `engine_rpm` matters for
        "engine-driven-belt" and "pto-clutch"; `measured_temp_k` (a
        real thermocouple reading at the fire/bed) only matters for an
        electric-motor blower that declares a ThermostatSpec -- ignored
        otherwise, including manual electric-motor operation."""
        spec = self.spec
        damper = max(0.0, min(1.0, self.damper_frac))
        if spec.drive == "engine-driven-belt":
            # real: belt speed (and so blower draft) scales with crank
            # rpm, same real convention every other belt-driven
            # accessory in this toy uses -- ramps linearly to full
            # draft, no draft at all with the crank stopped
            draft_frac = max(0.0, min(1.0, engine_rpm / max(spec.full_draft_rpm, 1e-6)))
            return spec.max_flow_kg_s * damper * draft_frac
        elif spec.drive == "pto-clutch":
            # real: same rpm ramp as a belt once the clutch is actually
            # engaged, but a declutched PTO delivers real zero
            # regardless of rpm -- the whole reason it's a different
            # real drive from a fixed belt, not just a bigger one
            if not self.pto_engaged:
                return 0.0
            draft_frac = max(0.0, min(1.0, engine_rpm / max(spec.full_draft_rpm, 1e-6)))
            return spec.max_flow_kg_s * damper * draft_frac
        elif spec.drive == "hand-crank":
            return spec.max_flow_kg_s * damper * (1.0 if self.crank_engaged else 0.0)
        elif spec.drive == "electric-motor":
            if not self.motor_on:
                return 0.0
            eff_damper = damper
            if spec.thermostat is not None and measured_temp_k is not None:
                # real proportional control, real negative feedback:
                # MORE draft means more air feeding the fire/bed, which
                # means faster, hotter combustion -- so running too hot
                # means throttling the blower BACK, not opening it
                # further (the opposite would be a real runaway).
                # Centered on 0.5 so the loop can trim in either
                # direction from a real mid-open starting point.
                error_k = measured_temp_k - spec.thermostat.target_temp_k
                eff_damper = max(0.0, min(1.0, 0.5 - spec.thermostat.proportional_gain_per_k * error_k))
            return spec.max_flow_kg_s * eff_damper
        elif spec.drive == "compressed-air-tank":
            # real: demand is however wide the operator has the nozzle
            # open; Accumulator.step() is the SAME real reservoir
            # mechanism a gasholder/boiler dome already uses above --
            # full demand while the reserve holds, clamped down to
            # just the recharge rate once it's genuinely run dry
            demand_kg_s = spec.max_flow_kg_s * damper
            delivered_kg_s = self._tank.step(dt, spec.tank_recharge_kg_s, demand_kg_s)
            # real: icing/fouling penalty from whatever conditioning is
            # missing -- see AirTreatment.discharge_derate_frac
            return delivered_kg_s * spec.air_treatment.discharge_derate_frac
        else:
            raise ValueError(f"unknown blower drive {spec.drive!r}")


# ---------------------------------------------------------------------
# FuelBed / Firebox -- the one real "raw fuel + air" primitive shared
# by every device in this module that burns something on a bed/grate
# ---------------------------------------------------------------------

# Real, disclosed stoichiometric air requirement for FULL combustion
# (burning a firebox's coal all the way to CO2 -- genuinely more air
# per kg fuel than a producer generator's own deliberately
# understoichiometric partial-oxidation reaction needs): ~10 kg of air
# per kg of coal-like solid fuel is a real, representative order-of-
# magnitude figure for hand-fired boiler/furnace practice, not a
# specific fuel assay's exact stoichiometry.
COMBUSTION_AIR_PER_FUEL_KG = 10.0


@dataclass(frozen=True)
class FuelBedSpec:
    """The one real primitive every solid-fuel-burning device in this
    module actually shares: a finite raw-fuel charge, fed air by a real
    Blower, that depletes as it's consumed and needs a person's real
    periodic manual recharge (add_fuel) -- a boiler's firebox hopper, a
    producer generator's own gasification bed, and (via Firebox below)
    a retort's external heating fire or a kiln's chamber burner are all
    built on this SAME real part instead of each reinventing it.
    `blower=None` means real natural draft -- a chimney/stack's own
    buoyancy, no rated flow ceiling of its own (only the consumer's own
    firing-rate ceiling limits combustion), the same real default a
    natural-draft boiler already used before this was pulled out into
    its own type."""
    raw_fuel: str                   # "coal" | "wood" | "charcoal" | "coke"
    bed_capacity_kg: float = 20.0
    blower: BlowerSpec | None = field(default_factory=lambda: BlowerSpec(max_flow_kg_s=0.05))

    def build(self) -> "FuelBed":
        return FuelBed(spec=self, fuel_remaining_kg=self.bed_capacity_kg,
                        _blower=self.blower.build() if self.blower is not None else None)


@dataclass
class FuelBed:
    """One real, running fuel bed instance."""
    spec: FuelBedSpec
    fuel_remaining_kg: float = 0.0
    _blower: Blower | None = None

    def draw_air_kg_s(self, dt: float, engine_rpm: float = 0.0,
                       measured_temp_k: float | None = None) -> float:
        """Real available air this tick -- zero with an empty bed (air
        alone doesn't burn/gasify anything without fuel to react
        with), otherwise whatever the real Blower delivers, or a real
        unlimited natural-draft supply (float('inf')) when no Blower
        is declared at all."""
        if self.fuel_remaining_kg <= 0.0:
            return 0.0
        if self._blower is None:
            return float("inf")
        return self._blower.air_output_kg_s(dt, engine_rpm, measured_temp_k)

    def consume_fuel(self, dt: float, requested_fuel_kg_s: float) -> float:
        """Draws the bed down by whatever the caller's own reaction
        actually needs this tick, clamped to what the bed can still
        supply -- returns the REAL actual kg/s consumed, which may be
        less than requested once the bed runs low."""
        max_from_bed_kg_s = self.fuel_remaining_kg / dt if dt > 0.0 else requested_fuel_kg_s
        actual_kg_s = min(max(0.0, requested_fuel_kg_s), max_from_bed_kg_s)
        self.fuel_remaining_kg = max(0.0, self.fuel_remaining_kg - actual_kg_s * dt)
        return actual_kg_s

    def add_fuel(self, kg: float) -> None:
        """Real manual recharge -- raking/shoveling fresh fuel onto the
        bed from a hopper or stockpile."""
        self.fuel_remaining_kg = min(self.spec.bed_capacity_kg, self.fuel_remaining_kg + max(0.0, kg))

    @property
    def blower(self) -> Blower | None:
        return self._blower

    @property
    def empty(self) -> bool:
        return self.fuel_remaining_kg <= 0.0


@dataclass(frozen=True)
class FireboxSpec:
    """Real generic 'burn solid fuel fully for heat' component, built
    on a FuelBed -- the SAME real mechanism whether it's raising steam
    in a boiler, firing a sealed retort from outside, or heating a
    kiln chamber directly (kiln.py). Full stoichiometric combustion
    (COMBUSTION_AIR_PER_FUEL_KG), genuinely different from a producer
    generator's own deliberately understoichiometric gasification,
    which is why that stays on bare FuelBed instead of this wrapper."""
    bed: FuelBedSpec
    combustion_efficiency: float = 0.65
    max_firing_rate_kg_m2_hr: float = 60.0
    grate_area_m2: float = 0.5

    def build(self) -> "Firebox":
        return Firebox(spec=self, _bed=self.bed.build())


@dataclass
class Firebox:
    """One real, running firebox instance. `firing_frac` is the real,
    live-adjustable damper/draft setting -- the same real "idle
    control" lever every other real burner in this module already
    has."""
    spec: FireboxSpec
    firing_frac: float = 1.0
    _bed: FuelBed | None = None

    def heat_output_w(self, dt: float, engine_rpm: float = 0.0,
                       measured_temp_k: float | None = None) -> float:
        """Real heat release rate this tick, gated by whichever real
        ceiling binds first: the grate's own real firing-rate limit, or
        the bed's own real air/fuel availability."""
        spec = self.spec
        hhv = FUEL_HHV_J_PER_KG.get(spec.bed.raw_fuel, 25.0e6)
        max_fuel_kg_s = spec.max_firing_rate_kg_m2_hr * spec.grate_area_m2 / 3600.0
        fuel_kg_s = max_fuel_kg_s * max(0.0, min(1.0, self.firing_frac))
        air_kg_s = self._bed.draw_air_kg_s(dt, engine_rpm, measured_temp_k)
        air_limited_fuel_kg_s = air_kg_s / COMBUSTION_AIR_PER_FUEL_KG
        fuel_kg_s = min(fuel_kg_s, air_limited_fuel_kg_s)
        actual_fuel_kg_s = self._bed.consume_fuel(dt, fuel_kg_s)
        return actual_fuel_kg_s * hhv * spec.combustion_efficiency

    def add_fuel(self, kg: float) -> None:
        self._bed.add_fuel(kg)

    @property
    def bed(self) -> FuelBed:
        return self._bed


# ---------------------------------------------------------------------
# GasQualityBath -- real purity/concentration transient, separate from
# whether there's enough gas MASS in reserve at all
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GasQualityBathSpec:
    """Real physical basis for 'is what's actually flowing right now
    combustible at all' -- a genuinely different real question from
    'is there enough gas mass in reserve' (the gasholder tank/
    Accumulator already answers that one). A generator's own internal
    holding volume (the bed's freeboard space plus its outlet plumbing,
    upstream of the gasholder tank entirely) starts full of ordinary
    air. Fresh combustible gas has to physically DISPLACE that air
    before what comes out the far end is actually fuel -- the real,
    documented reason gasifier operators (vehicle gasogene units
    included) flared off the first several minutes of output on
    startup rather than feeding it straight to the engine: it's mostly
    air/inert at first, not fuel.

    This is a real, standard well-mixed-reactor (CSTR) composition
    balance -- the same real math used for any stirred tank's startup
    transient, not an invented purity stat. `holding_mass_kg` is this
    generator's own real internal gas content at operating density
    (small -- this is the body/plumbing volume, not the gasholder tank
    downstream of it)."""
    holding_mass_kg: float = 0.3

    def build(self) -> "GasQualityBath":
        return GasQualityBath(spec=self)


@dataclass
class GasQualityBath:
    """concentration: 0.0 = pure air/inert (a cold, unpurged
    generator), 1.0 = fully purged, genuine full-strength gas. Real
    CSTR mixing: raw generator output enters this volume at
    concentration 1.0 (this bath's own reference "pure gas"
    composition, i.e. gas_output_kg_s's own real mass-conserved
    production); whatever draw exceeds that production backfills with
    ambient air at concentration 0.0 -- the same real physical vacuum
    draw pulling air in behind an under-producing generator, not a
    separate leak. Removing mixture at the CURRENT concentration
    doesn't change what's left (a well-mixed volume) -- only the
    INFLOW composition moves concentration toward 1 or 0, which is
    exactly why this needs real time to purge, not an instant switch."""
    spec: GasQualityBathSpec
    concentration: float = 0.0

    def step(self, dt: float, production_kg_s: float, draw_kg_s: float) -> float:
        mass = max(self.spec.holding_mass_kg, 1e-6)
        air_backfill_kg_s = max(0.0, draw_kg_s - production_kg_s)
        d_conc = (production_kg_s * (1.0 - self.concentration)
                  - air_backfill_kg_s * self.concentration) / mass
        self.concentration = max(0.0, min(1.0, self.concentration + d_conc * dt))
        return self.concentration


# ---------------------------------------------------------------------
# GasGenerator -- combustible gas from raw solid fuel
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GasGeneratorSpec:
    kind: str                       # "retort" | "producer"
    raw_fuel: str                   # "coal" | "wood" | "charcoal" | "coke"
    output_gas_profile: str         # matches otto_langen.IGNITION_PRESSURE_RATIO_BY_FUEL key
    # retort only: one real batch charge and its real coking duration.
    # A real gasworks retort runs ~4-8 real hours per charge at full
    # heat; 6 hours is a disclosed, representative figure, not a
    # specific works' logbook.
    charge_mass_kg: float = 50.0
    batch_time_s: float = 6.0 * 3600.0
    # producer only: the real Blower supplying this generator's own
    # gasification bed (see FuelBed) -- a retort needs none at all (it
    # runs sealed, with no air let in at all).
    blower: BlowerSpec = field(default_factory=lambda: BlowerSpec(max_flow_kg_s=0.05))
    # producer only: the real solid-fuel BED capacity (see FuelBed) --
    # air alone was never going to gasify anything without fuel
    # actually in the bed to react with.
    bed_capacity_kg: float = 20.0
    # retort only: the real EXTERNAL firebox heating the sealed vessel
    # from outside -- a retort doesn't burn its own charge with air (it
    # runs sealed precisely to keep air OUT, so the coal carbonizes
    # instead of just burning up); something else has to supply that
    # heat, and historically that was a real coal fire built around/
    # under the retort itself. None (default) keeps every existing
    # retort on the prior idealized "already at full coking heat"
    # approximation; declaring one gates real gas evolution on that
    # firebox actually being lit and hot enough (see gas_output_kg_s).
    external_firebox: FireboxSpec | None = None
    # retort only: real minimum firebox heat output before the charge
    # counts as "at coking heat" -- a disclosed, representative figure,
    # not a specific retort's measured heat-up curve.
    min_coking_heat_w: float = 20_000.0
    # Both kinds: this generator's own real internal purity transient
    # -- see GasQualityBathSpec's own docstring. Every generator has
    # one (a retort's own outlet plumbing needs purging of air on
    # startup exactly as much as a producer's bed does).
    quality_bath: GasQualityBathSpec = field(default_factory=GasQualityBathSpec)

    def build(self) -> "GasGenerator":
        bed = (FuelBedSpec(raw_fuel=self.raw_fuel, bed_capacity_kg=self.bed_capacity_kg,
                            blower=self.blower).build()
               if self.kind == "producer" else None)
        firebox = self.external_firebox.build() if self.external_firebox is not None else None
        return GasGenerator(spec=self, _bed=bed, _firebox=firebox, _bath=self.quality_bath.build())


@dataclass
class GasGenerator:
    """One real, running generator instance. A retort has no live fuel
    control at all: once charged and lit, a real coking batch runs at
    its own pace until the charge is spent, which is exactly why
    retort-fed plants historically needed a gasholder buffer even more
    than producer-fed ones -- there's no way to turn a retort down. A
    producer's own real FuelBed (see gas_output_kg_s) is that live
    control for this generator instead, with its own real finite fuel
    reserve (see add_fuel())."""
    spec: GasGeneratorSpec
    elapsed_batch_s: float = 0.0
    _bed: FuelBed | None = None
    _firebox: Firebox | None = None
    _bath: GasQualityBath | None = None
    _last_production_kg_s: float = 0.0

    def gas_output_kg_s(self, dt: float, engine_rpm: float = 0.0,
                         measured_temp_k: float | None = None) -> float:
        """Real mass rate of combustible gas currently being produced
        AT THE SOURCE (the bed/retort itself), advancing this
        generator's own real internal process state by dt. This is the
        real production term for step_quality() below -- NOT yet
        adjusted for how much of what's actually reaching the engine is
        genuine fuel vs. unpurged air (see step_quality/GasQualityBath).
        Batch retorts report a steady rate while a charge is active AND
        (when external_firebox is declared) that firebox is actually up
        to coking heat -- a real simplification when no firebox is
        declared (true retort evolution also tapers near the end of a
        batch; disclosed as a steady-rate approximation) -- and zero
        once the charge time is spent, until re-charged (see
        recharge()). `engine_rpm`/`measured_temp_k` feed straight
        through to whichever real Blower this generator's own bed or
        firebox declares -- ignored otherwise."""
        result = self._gas_output_kg_s(dt, engine_rpm, measured_temp_k)
        self._last_production_kg_s = result
        return result

    def _gas_output_kg_s(self, dt: float, engine_rpm: float,
                          measured_temp_k: float | None) -> float:
        spec = self.spec
        if spec.kind == "retort":
            if self.elapsed_batch_s >= spec.batch_time_s:
                return 0.0
            if self._firebox is not None:
                heat_w = self._firebox.heat_output_w(dt, engine_rpm, measured_temp_k)
                if heat_w < spec.min_coking_heat_w:
                    return 0.0   # real: not up to coking heat yet, nothing cokes off regardless of elapsed time
            self.elapsed_batch_s = min(spec.batch_time_s, self.elapsed_batch_s + dt)
            gas_m3_s = (spec.charge_mass_kg * COAL_GAS_YIELD_M3_PER_KG_COAL) / spec.batch_time_s
            return gas_m3_s * COAL_GAS_DENSITY_KG_PER_M3
        elif spec.kind == "producer":
            if self._bed is None:
                raise ValueError("a producer generator needs a real FuelBed to gasify anything")
            air_kg_s = self._bed.draw_air_kg_s(dt, engine_rpm, measured_temp_k)
            if air_kg_s <= 0.0:
                return 0.0
            carbon_frac = RAW_FUEL_CARBON_FRACTION.get(spec.raw_fuel, 0.7)
            carbon_demand_kg_s = air_kg_s / AIR_PER_CARBON_KG
            fuel_demand_kg_s = carbon_demand_kg_s / max(carbon_frac, 1e-6)
            # real: can't gasify more carbon this tick than the bed
            # actually still holds -- whatever blast air exceeds that
            # real ceiling just passes through unreacted (a real,
            # disclosed simplification: not separately vented/modeled)
            actual_fuel_kg_s = self._bed.consume_fuel(dt, fuel_demand_kg_s)
            carbon_kg_s = actual_fuel_kg_s * carbon_frac
            # real mass conservation: every kg of air blown in leaves
            # again as product gas, carbon-enriched into CO -- nothing
            # is lost in this simplified accounting (no ash/radiative
            # loss modeled), so output mass is air in PLUS carbon
            # actually consumed from the bed.
            return air_kg_s + carbon_kg_s
        raise ValueError(f"unknown generator kind {spec.kind!r}")

    def step_quality(self, dt: float, draw_kg_s: float) -> float:
        """Advances this generator's own real internal purity transient
        (GasQualityBath) using the production this SAME tick's
        gas_output_kg_s() call already computed and cached, against
        `draw_kg_s` (the real downstream demand actually being pulled
        this tick -- the engine's own consumption, or whatever's
        filling the gasholder tank next in line). Call this AFTER
        gas_output_kg_s() each tick, not before -- it consumes that
        call's own cached result rather than recomputing production
        itself, so production is derived exactly once per tick.
        Returns the current concentration (0.0..1.0): multiply this by
        whatever the gasholder tank's own fill_level_frac already tells
        you about total reserve to get the real combined charge-
        availability signal -- "is there enough gas AND is it actually
        combustible" are two separate real questions, not one."""
        return self._bath.step(dt, self._last_production_kg_s, draw_kg_s)

    @property
    def concentration(self) -> float:
        return self._bath.concentration

    def add_fuel(self, kg: float) -> None:
        """Real manual recharge -- raking fresh coal/charcoal/wood onto
        the producer bed, or (see external_firebox) topping up a
        retort's own external heating fire. A retort's own CHARGE
        (what's being coked, not what's heating it) reloads via
        recharge() below instead -- a genuinely different real action."""
        if self._bed is not None:
            self._bed.add_fuel(kg)
        elif self._firebox is not None:
            self._firebox.add_fuel(kg)
        else:
            raise ValueError("this generator has no fuel bed or firebox to add fuel to")

    @property
    def blower(self) -> Blower | None:
        return self._bed.blower if self._bed is not None else None

    @property
    def firebox(self) -> Firebox | None:
        return self._firebox

    def recharge(self) -> None:
        """Real batch reload -- a retort's coking cycle restarts from a
        fresh CHARGE (the coal being coked, not the external firebox
        heating it); a no-op for a continuous producer, which has no
        batch to reload."""
        self.elapsed_batch_s = 0.0

    @property
    def batch_exhausted(self) -> bool:
        return self.spec.kind == "retort" and self.elapsed_batch_s >= self.spec.batch_time_s


# ---------------------------------------------------------------------
# GasSeparator -- waste/particulate removal from a generator's raw gas
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GasSeparatorSpec:
    """Real cyclone/scrubber stage between a producer generator's bed
    and anything downstream that burns the gas: raw producer gas
    carries real tar, ash, and particulate fines entrained straight
    off the bed, and every real gasifier installation (WWII-era vehicle
    gasogene units included) needs one or an engine's own intake/
    valves foul solid in short order.

    Built on the SAME real filter-restriction formula engines.
    IntakeSystem.restriction_frac already uses for an intake air
    filter (engines.FILTER_MATERIAL_PERMEABILITY / REFERENCE_FILTER_
    RESTRICTION / REFERENCE_FILTER_AREA_CM2) -- a separator is real
    filter media on a gas line instead of an air line, not a
    physically different mechanism, so it reuses the exact same
    real orifice pressure-drop relationship rather than a parallel
    fouling/efficiency scheme invented just for this one part. A dry
    cyclone (see FILTER_MATERIAL_PERMEABILITY["dry-cyclone"]) barely
    restricts flow at all and is real, common, and simple; a wire-mesh
    scrubber restricts more but is the real way tar VAPOR actually
    gets caught, which a dry cyclone's swirl alone can't do."""
    filter_material: str = "dry-cyclone"
    filter_surface_area_cm2: float = 300.0

    @property
    def restriction_frac(self) -> float:
        """Real pressure-drop fraction across the separator at full
        flow -- identical formula to IntakeSystem.restriction_frac,
        same real clamp."""
        from engines import FILTER_MATERIAL_PERMEABILITY, REFERENCE_FILTER_RESTRICTION, REFERENCE_FILTER_AREA_CM2
        permeability = FILTER_MATERIAL_PERMEABILITY.get(self.filter_material, 1.0)
        effective_area = max(self.filter_surface_area_cm2, 1.0) * permeability
        return max(0.0, min(0.30, REFERENCE_FILTER_RESTRICTION * REFERENCE_FILTER_AREA_CM2 / effective_area))

    def build(self) -> "GasSeparator":
        return GasSeparator(spec=self)


@dataclass
class GasSeparator:
    """One real, running separator instance. `process()` is a pass-
    through call (no internal state of its own -- the real fouling/
    service-interval concern already belongs to whatever generator or
    engine wear system tracks its own maintenance, matching this toy's
    existing wear.py convention rather than a second, parallel one
    here) that derates the raw gas stream by the real pressure-drop
    this filter media imposes."""
    spec: GasSeparatorSpec

    def process(self, raw_gas_kg_s: float) -> float:
        """Real cleaned gas mass flow actually available downstream --
        the same real orifice-restriction derate an intake filter
        already applies to air, applied here to a gas line instead."""
        return raw_gas_kg_s * (1.0 - self.spec.restriction_frac)


# ---------------------------------------------------------------------
# Boiler -- steam from raw solid fuel
# ---------------------------------------------------------------------

# Real, disclosed simplified steam-table anchor points: (pressure_pa,
# saturation_temp_k, latent_heat_of_vaporization_j_per_kg). Cited real
# steam-table magnitudes at each pressure, not the full IAPWS
# correlation -- linearly interpolated between anchors, matching this
# toy's established "magnitude-correct, disclosed, not one paper's
# exact coefficients" convention (see engines.py's own FRICTION_FMEP_*
# comment for the same convention elsewhere in this codebase).
STEAM_TABLE = (
    (101_325.0, 373.15, 2_257_000.0),   # 1 atm / 100 C -- the familiar reference point
    (300_000.0, 407.15, 2_164_000.0),   # ~3 bar, small workshop boiler
    (600_000.0, 432.15, 2_085_000.0),   # ~6 bar, typical traction-engine working pressure
    (1_000_000.0, 453.15, 2_015_000.0),  # ~10 bar, higher-pressure stationary practice
)
WATER_SPECIFIC_HEAT_J_PER_KG_K = 4186.0  # real, standard
FEEDWATER_TEMP_K = 293.15  # real: cold feedwater at ambient, no economizer modeled


def _steam_properties(pressure_pa: float) -> tuple[float, float]:
    """Real linear interpolation across STEAM_TABLE's disclosed
    anchors -> (saturation_temp_k, latent_heat_j_per_kg). Clamps to the
    table's own real range rather than extrapolating past it."""
    table = STEAM_TABLE
    if pressure_pa <= table[0][0]:
        return table[0][1], table[0][2]
    if pressure_pa >= table[-1][0]:
        return table[-1][1], table[-1][2]
    for (p0, t0, h0), (p1, t1, h1) in zip(table, table[1:]):
        if p0 <= pressure_pa <= p1:
            frac = (pressure_pa - p0) / max(p1 - p0, 1e-9)
            return t0 + frac * (t1 - t0), h0 + frac * (h1 - h0)
    return table[-1][1], table[-1][2]


# Real, disclosed stoichiometric air requirement for FULL combustion
# (burning a firebox's coal all the way to CO2 -- genuinely more air
# per kg fuel than the producer generator's own deliberately
# understoichiometric partial-oxidation reaction above needs): ~10 kg
# of air per kg of coal-like solid fuel is a real, representative
# order-of-magnitude figure for hand-fired boiler practice, not a
# specific fuel assay's exact stoichiometry.
COMBUSTION_AIR_PER_FUEL_KG = 10.0


@dataclass(frozen=True)
class BoilerSpec:
    raw_fuel: str                          # "coal" | "wood" | "coke"
    grate_area_m2: float = 0.5
    # Real, disclosed hand-fired grate combustion-rate ceiling -- cited
    # real boiler-engineering practice runs roughly 40-80 kg of coal
    # per m^2 of grate per hour at full firing; 60 is a representative
    # mid-range figure, not a specific boiler's test-bed number. This
    # is the NATURAL-draft ceiling (chimney buoyancy alone) -- real,
    # and still the default: declaring a draft_blower below doesn't
    # remove this ceiling, it just adds a second, independent real one
    # (see steam_output_kg_s -- whichever binds first wins, same
    # pattern engine_sim.torque_fraction's own choke term uses).
    max_firing_rate_kg_m2_hr: float = 60.0
    operating_pressure_pa: float = 600_000.0  # real ~6 bar, typical traction-engine pressure
    # Real, disclosed overall thermal efficiency of a simple hand-fired
    # shell/locomotive-type boiler -- a real substantial fraction of
    # the fuel's heat genuinely goes up the stack unused; 0.65 is a
    # representative real order-of-magnitude figure for this class of
    # boiler, well below a modern water-tube unit's efficiency.
    boiler_efficiency: float = 0.65
    # None (default) = natural draft: the chimney's own real buoyancy
    # supplies combustion air, no moving parts, capped only by
    # max_firing_rate_kg_m2_hr above -- every existing use of this
    # spec keeps exactly today's behavior. A real forced-draft fan
    # (the SAME real Blower type the producer generator's bed uses,
    # just sized for a firebox instead) lifts that ceiling, at the
    # real cost of needing its own drive (belt/crank/electric) to
    # actually be turning.
    draft_blower: BlowerSpec | None = None
    # The real fuel HOPPER/firebox charge -- a boiler doesn't get a
    # free pass just because it burns fuel directly instead of
    # gasifying it first: this depletes exactly like the producer
    # generator's own bed_capacity_kg, and needs the same real
    # periodic manual shoveling/refueling interval (see add_fuel()).
    fuel_hopper_capacity_kg: float = 200.0

    def build(self) -> "Boiler":
        return Boiler(spec=self,
                       _blower=self.draft_blower.build() if self.draft_blower is not None else None,
                       fuel_remaining_kg=self.fuel_hopper_capacity_kg)


@dataclass
class Boiler:
    """One real, running boiler instance. `firing_frac` is the real,
    live-adjustable firebox draft/damper setting -- the boiler's own
    equivalent of the generator's blast_frac, and a real steam
    engine's actual "idle control" lever (bank the fire down, steam
    production falls, the dome's stored reserve is what keeps the
    engine running through the lull -- see Accumulator). fuel_
    remaining_kg is a real, separate, finite reserve that depletes as
    the firebox burns it, same as the producer generator's own bed."""
    spec: BoilerSpec
    firing_frac: float = 1.0
    _blower: Blower | None = None
    fuel_remaining_kg: float = 0.0

    def steam_output_kg_s(self, dt: float, engine_rpm: float = 0.0,
                           measured_temp_k: float | None = None) -> float:
        """Real mass rate of steam currently being raised. dt is
        accepted for interface symmetry with GasGenerator (a boiler
        with real thermal mass would need dt to model firebox lag;
        this simplified version reports the fuel-limited steady-state
        rate instantly, disclosed as an approximation -- the same
        simplification the retort's steady-rate-per-batch already
        makes). `engine_rpm`/`measured_temp_k` only matter when
        draft_blower is declared, feeding straight through to its own
        Blower.air_output_kg_s -- ignored otherwise, including every
        natural-draft boiler (the common case)."""
        spec = self.spec
        if self.fuel_remaining_kg <= 0.0:
            return 0.0   # real: the fire's out, nothing left in the hopper to burn
        hhv = FUEL_HHV_J_PER_KG.get(spec.raw_fuel, 25.0e6)
        max_fuel_kg_s = spec.max_firing_rate_kg_m2_hr * spec.grate_area_m2 / 3600.0
        fuel_kg_s = max_fuel_kg_s * max(0.0, min(1.0, self.firing_frac))
        if self._blower is not None:
            # real: a forced-draft fan can't burn more fuel than the
            # air it's actually delivering can fully oxidize -- caps
            # the grate-area ceiling above, doesn't replace it
            air_limited_fuel_kg_s = (self._blower.air_output_kg_s(dt, engine_rpm, measured_temp_k)
                                      / COMBUSTION_AIR_PER_FUEL_KG)
            fuel_kg_s = min(fuel_kg_s, air_limited_fuel_kg_s)
        # real: can't burn more fuel this tick than the hopper/firebox
        # actually still holds
        max_from_hopper_kg_s = self.fuel_remaining_kg / dt if dt > 0.0 else fuel_kg_s
        fuel_kg_s = min(fuel_kg_s, max_from_hopper_kg_s)
        self.fuel_remaining_kg = max(0.0, self.fuel_remaining_kg - fuel_kg_s * dt)
        heat_w = fuel_kg_s * hhv * spec.boiler_efficiency
        tsat_k, hfg_j_per_kg = _steam_properties(spec.operating_pressure_pa)
        sensible_j_per_kg = WATER_SPECIFIC_HEAT_J_PER_KG_K * max(0.0, tsat_k - FEEDWATER_TEMP_K)
        energy_per_kg_steam = sensible_j_per_kg + hfg_j_per_kg
        return heat_w / max(energy_per_kg_steam, 1.0)

    def add_fuel(self, kg: float) -> None:
        """Real manual recharge -- shoveling fresh coal/wood into the
        firebox/hopper from a stockpile, the same real periodic
        operator task GasGenerator.add_fuel() is for a producer's own
        bed."""
        self.fuel_remaining_kg = min(self.spec.fuel_hopper_capacity_kg,
                                      self.fuel_remaining_kg + max(0.0, kg))


# ---------------------------------------------------------------------
# Accumulator -- gasholder bell or boiler steam dome, same real mechanism
# ---------------------------------------------------------------------

@dataclass
class Accumulator:
    """Real, generic compliant reservoir buffering production against
    consumption. `supply_pressure_pa` is the real, near-constant
    pressure the reservoir delivers regardless of fill level (a
    gasholder bell's own dead weight over its cross-sectional area; a
    boiler dome's saturation pressure, held by the firebox as long as
    there's water to boil) -- what actually runs out under sustained
    heavy demand is the real, finite fill MASS, not the pressure."""
    supply_pressure_pa: float
    capacity_kg: float
    fill_kg: float = 0.0

    def step(self, dt: float, production_kg_s: float, demand_kg_s: float) -> float:
        """Advances the real fill level by dt and returns the real
        available mass flow this tick: the full demand when production
        plus reserve can cover it, and only what production can
        actually supply once the reserve runs dry -- a real, physical
        throttling of the ENGINE'S supply by the reservoir, not a
        throttle plate at the engine itself."""
        self.fill_kg = max(0.0, min(self.capacity_kg,
                                     self.fill_kg + (production_kg_s - demand_kg_s) * dt))
        if self.fill_kg <= 0.0 and demand_kg_s > production_kg_s:
            return production_kg_s
        return demand_kg_s

    @property
    def fill_frac(self) -> float:
        return self.fill_kg / max(self.capacity_kg, 1e-9)


# The wiring the earlier NOTE here described is done: see the module
# docstring's last paragraph and engine_cycle_sim._step_atmospheric.
