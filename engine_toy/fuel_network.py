"""The fuel NETWORK: the real hardware between where a working fluid
is stored and the point it's admitted to a cylinder -- declared once,
as an ordered chain of real stages, emitted into the SAME drivetrain
graph / FluidCircuit system every existing tank, bottle and gasholder
already lives in (drivetrain_graph.py), and stepped by one runtime
that hands the circuit its real production, composition, flow ceiling
and supply pressure each tick.

This is the whole point of working_fluids.py: converting a cylinder to
another fuel is a different NETWORK on the fluid system, not a second
engine model. The cylinder itself stays what it is (a spark-ignition
piston loop, an Otto-Langen free piston, a cutoff expander -- see
expander.py) and only reads the fluid's properties and what the
network actually delivered.

Stages, source -> admission (every one a real, installable device):

  sources    LiquidTank          vented tank + pump (the existing
                                  FuelDeliverySystem hardware)
             PressurizedBottle   CNG / H2 / LPG bottle: pressure follows
                                  fill for a gas, sits at vapour pressure
                                  for a liquefied gas
             Gasholder           low-pressure holder, optionally fed by
                                  a real on-site GasGenerator
             UtilityMain         unmetered piped supply (never depletes)
             BoilerSource        firebox + shell + steam dome + FEEDWATER
                                  (a boiler run dry is the fatal failure)
             Receiver            compressed-air receiver charged by an
                                  external (shop) compressor
  in-line    Regulator           staged pressure reduction, rated flow
             Vaporizer           LPG/LNG evaporator: delivery is capped by
                                  the heat it can pull from the coolant
             CoolerFilter        producer-gas cooler / cyclone / filter
                                  (a real restriction, tar drops out)
             FlameArrestor       stops a flame propagating back up the
                                  line; a real small restriction
             PurgeValve          vents production to atmosphere instead
                                  of into the holder (startup purge)
             LockoffSolenoid     closes the line unless the engine runs
  admission  Mixer               venturi/slide-valve, fixed gas fraction
             GasInjector         metered gaseous port injection
             LiquidCarburetor    the engine's own existing carburetor
             LiquidInjector      the engine's own existing injectors
             CutoffValve         expander admission with cutoff (steam,
                                  compressed air)

`required_stages_missing()` lists what a real installation of that
fluid MUST have and this network doesn't (hydrogen without an arrestor,
LPG without a vaporizer, steam without feedwater) -- a build check, not
a silent default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from working_fluids import (WorkingFluid, working_fluid, GASEOUS_FUEL, EXPANDER_FLUID, LIQUID_FUEL,
                            PRESSURIZED_LIQUID, HIGH_PRESSURE_GAS, CRYOGENIC, GASHOLDER, BOILER,
                            RECEIVER, ATMOSPHERIC_TANK)

P_ATM_PA = 101_325.0
T_AMBIENT_K = 293.15


# ---------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class LiquidTank:
    capacity_l: float = 60.0
    pump_kind: str = "electric"            # "electric" | "mechanical"
    pump_flow_capacity_kg_s: float = 0.02
    line_diameter_mm: float = 8.0
    kind: str = "liquid-tank"


@dataclass(frozen=True)
class PressurizedBottle:
    """CNG/H2 (HIGH_PRESSURE_GAS): contents pressure is proportional to
    fill (ideal gas at fixed volume). LPG (PRESSURIZED_LIQUID): the
    liquid sits at its own vapour pressure until the last of it boils
    off -- constant pressure, then nothing. Either way the bottle
    itself is the same depletable reservoir every other tank is."""
    capacity_kg: float = 10.0
    kind: str = "pressurized-bottle"


@dataclass(frozen=True)
class Gasholder:
    capacity_kg: float = 2.0
    generator: object | None = None        # gas_works.GasGeneratorSpec (built by the runtime), or None = main-fed
    kind: str = "gasholder"


@dataclass(frozen=True)
class UtilityMain:
    kind: str = "utility-main"


@dataclass(frozen=True)
class BoilerSource:
    """A real boiler is TWO reservoirs: the steam dome (what the engine
    draws) and the water in the shell (what the firebox boils). Steam
    raised comes out of the water; a feed pump/injector puts it back
    from a feedwater tank. Let the water run out and the firebox keeps
    heating an empty shell -- crown-sheet failure, the classic boiler
    explosion. That is the fatal failure this source models, not a
    soft 'out of steam'."""
    boiler: object                          # gas_works.BoilerSpec
    water_capacity_kg: float = 200.0
    dome_capacity_kg: float = 5.0           # steam mass the dome holds at working pressure
    feedwater_tank_kg: float = 400.0
    feed_pump_kg_s: float = 0.05            # what the injector/pump can put back when running
    low_water_frac: float = 0.15            # below this the crown sheet is exposed
    kind: str = "boiler"


@dataclass(frozen=True)
class Receiver:
    """A compressed-air receiver charged by an external compressor with
    its own rated delivery and a real unloader: charge until cut-out,
    idle until cut-in."""
    capacity_kg: float = 3.0
    compressor_rated_kg_s: float = 0.02
    cut_in_frac: float = 0.84
    cut_out_frac: float = 1.0
    kind: str = "receiver"


# ---------------------------------------------------------------------
# In-line stages
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Regulator:
    outlet_pressure_pa: float
    stages: int = 2
    flow_capacity_kg_s: float = 0.05
    kind: str = "regulator"


@dataclass(frozen=True)
class Vaporizer:
    """Coolant-heated evaporator. Delivery is capped by heat: at most
    max_heat_w / latent_heat kg/s can be vaporized, and only once the
    coolant is warm -- a cold LPG conversion genuinely starves until
    the engine warms, the real reason most start on petrol."""
    max_heat_w: float = 8_000.0
    min_coolant_k: float = 310.0
    cold_heat_frac: float = 0.25            # ambient-conduction share of rating with a stone-cold engine
    kind: str = "vaporizer"


@dataclass(frozen=True)
class FuelHeater:
    """Coolant-heated fuel heater for a fuel too viscous to pump and
    atomize cold (straight vegetable oil, crude). Delivery is capped by
    viscosity until the fuel is up to temperature, and the heater can
    only give what the coolant has -- a real SVO conversion starts and
    warms up on diesel from a second tank, then switches over."""
    target_k: float = 343.0
    min_coolant_k: float = 320.0
    cold_flow_frac: float = 0.15            # what a cold, viscous fuel still pushes through the pump
    kind: str = "fuel-heater"


@dataclass(frozen=True)
class ElectricFuelHeater:
    """Inline electric fuel heater. Works from cold, and costs watts.

    The coolant-heated FuelHeater above cannot start a cold plant on a
    viscous fuel, because there is no coolant heat until something has
    already been running -- which is the whole reason an SVO conversion
    starts on diesel from a second tank. A fixed installation does not
    have that problem and does not need that workaround: it has mains
    power, so it can heat the fuel before anything is running at all.
    That is why a station burning waste oil or crude uses one of these
    and a vehicle usually does not.

    It is POWER LIMITED, which is the honest constraint: raising a fuel
    stream by dT costs mdot * cp * dT, so the achievable temperature
    falls as demand rises. A heater sized for idle will not keep up at
    full flow, and the fuel arrives half-heated rather than the heater
    politely refusing.
    """

    rated_w: float = 6_000.0
    target_k: float = 353.0
    #: real specific heat of a liquid hydrocarbon, the same figure
    #: engines.py already uses for fuel charge cooling
    fuel_cp_j_per_kgk: float = 2100.0
    #: what a cold, viscous fuel still pushes through the pump before
    #: the heater has caught up -- same meaning as FuelHeater's
    cold_flow_frac: float = 0.15
    kind: str = "electric-fuel-heater"

    def delivered_temp_k(self, mass_flow_kg_s: float,
                         inlet_k: float = 288.15) -> float:
        """How warm this heater can actually get that much fuel."""
        flow = max(float(mass_flow_kg_s), 1e-9)
        rise = self.rated_w / (flow * max(self.fuel_cp_j_per_kgk, 1.0))
        return min(self.target_k, float(inlet_k) + rise)

    def draw_w(self, mass_flow_kg_s: float, inlet_k: float = 288.15) -> float:
        """Electrical load, which a station has to actually supply."""
        flow = max(float(mass_flow_kg_s), 0.0)
        needed = flow * self.fuel_cp_j_per_kgk * max(self.target_k - float(inlet_k), 0.0)
        return min(self.rated_w, needed)

    def flow_factor(self, mass_flow_kg_s: float, required_k: float,
                    inlet_k: float = 288.15) -> float:
        """0..1 -- how much of the demanded flow this fuel will actually
        pass at the temperature the heater managed to reach."""
        if required_k <= inlet_k:
            return 1.0
        reached = self.delivered_temp_k(mass_flow_kg_s, inlet_k)
        span = max(required_k - inlet_k, 1e-6)
        warmed = max(0.0, min(1.0, (reached - inlet_k) / span))
        return self.cold_flow_frac + (1.0 - self.cold_flow_frac) * warmed


@dataclass(frozen=True)
class CoolerFilter:
    restriction_frac: float = 0.15          # fraction of the upstream flow ceiling it takes away
    kind: str = "cooler-filter"


@dataclass(frozen=True)
class FlameArrestor:
    flow_capacity_kg_s: float = 0.05
    kind: str = "flame-arrestor"


@dataclass(frozen=True)
class PurgeValve:
    kind: str = "purge-valve"


@dataclass(frozen=True)
class LockoffSolenoid:
    kind: str = "lockoff-solenoid"


# ---------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Mixer:
    gas_volume_fraction: float | None = None   # None = the fluid's own design lean setting
    kind: str = "mixer"


@dataclass(frozen=True)
class GasInjector:
    rail_pressure_pa: float = 300_000.0
    kind: str = "gas-injector"


@dataclass(frozen=True)
class LiquidCarburetor:
    kind: str = "liquid-carburetor"


@dataclass(frozen=True)
class LiquidInjector:
    kind: str = "liquid-injector"


@dataclass(frozen=True)
class CutoffValve:
    cutoff_frac: float = 0.35
    kind: str = "cutoff-valve"


SOURCE_KINDS = (LiquidTank, PressurizedBottle, Gasholder, UtilityMain, BoilerSource, Receiver)
INLINE_KINDS = (Regulator, Vaporizer, FuelHeater, ElectricFuelHeater, CoolerFilter,
                FlameArrestor, PurgeValve, LockoffSolenoid)
ADMISSION_KINDS = (Mixer, GasInjector, LiquidCarburetor, LiquidInjector, CutoffValve)


@dataclass(frozen=True)
class FuelNetworkSpec:
    fluid: str
    source: object
    admission: object
    stages: tuple = ()

    def __post_init__(self) -> None:
        working_fluid(self.fluid)
        if not isinstance(self.source, SOURCE_KINDS):
            raise TypeError(f"source must be one of {[k.__name__ for k in SOURCE_KINDS]}")
        if not isinstance(self.admission, ADMISSION_KINDS):
            raise TypeError(f"admission must be one of {[k.__name__ for k in ADMISSION_KINDS]}")
        for st in self.stages:
            if not isinstance(st, INLINE_KINDS):
                raise TypeError(f"stage {st!r} is not an in-line stage")

    @property
    def working_fluid(self) -> WorkingFluid:
        return working_fluid(self.fluid)

    def has(self, kind_cls) -> bool:
        return any(isinstance(st, kind_cls) for st in self.stages)

    def required_stages_missing(self) -> list[str]:
        """What a real installation of this fluid must have that this
        network lacks. Not auto-added: a missing arrestor on a hydrogen
        line is a real, declared hazard, not a default to paper over."""
        f = self.working_fluid
        missing: list[str] = []
        if f.phase == EXPANDER_FLUID and not isinstance(self.admission, CutoffValve):
            missing.append("CutoffValve admission (an expander fluid has no mixture to meter)")
        if f.phase != EXPANDER_FLUID and isinstance(self.admission, CutoffValve):
            missing.append("a combustion admission (Mixer/GasInjector/...) -- CutoffValve is for expander fluids")
        if f.phase == GASEOUS_FUEL and isinstance(self.admission, (LiquidCarburetor, LiquidInjector)):
            missing.append("a gaseous admission (Mixer or GasInjector)")
        if f.phase == LIQUID_FUEL and isinstance(self.admission, (Mixer, GasInjector)):
            missing.append("a liquid admission (LiquidCarburetor or LiquidInjector)")
        if f.storage in (HIGH_PRESSURE_GAS, PRESSURIZED_LIQUID, CRYOGENIC) and not self.has(Regulator):
            missing.append("Regulator (bottle pressure is far above any admission pressure)")
        if f.storage in (PRESSURIZED_LIQUID, CRYOGENIC) and not self.has(Vaporizer):
            missing.append("Vaporizer (the store holds a liquid; the cylinder needs a gas)")
        if f.viscous_at_ambient and not (self.has(FuelHeater) or self.has(ElectricFuelHeater)):
            missing.append("FuelHeater or ElectricFuelHeater "
                           "(this fuel is too viscous to pump and atomize cold)")
        if f.flame_arrestor_required and not self.has(FlameArrestor):
            missing.append("FlameArrestor (min ignition energy / flame speed make line flashback real)")
        if f.storage == GASHOLDER and isinstance(self.source, Gasholder) and self.source.generator is not None \
                and not self.has(PurgeValve):
            missing.append("PurgeValve (a generator's startup output is air, not fuel -- it has to go somewhere)")
        if f.storage == BOILER and not isinstance(self.source, BoilerSource):
            missing.append("BoilerSource (steam comes from a boiler, with feedwater)")
        if f.storage == RECEIVER and not isinstance(self.source, Receiver):
            missing.append("Receiver (compressed air comes from a charged receiver)")
        if f.storage == GASHOLDER and not isinstance(self.source, (Gasholder, UtilityMain)):
            missing.append("Gasholder or UtilityMain source")
        if f.storage in (HIGH_PRESSURE_GAS, PRESSURIZED_LIQUID, CRYOGENIC) and not isinstance(self.source, PressurizedBottle):
            missing.append("PressurizedBottle source")
        if f.storage == ATMOSPHERIC_TANK and f.phase == LIQUID_FUEL and not isinstance(self.source, LiquidTank):
            missing.append("LiquidTank source")
        return missing

    def validate(self) -> None:
        missing = self.required_stages_missing()
        if missing:
            raise ValueError(f"fuel network for {self.fluid!r} is missing: " + "; ".join(missing))


# ---------------------------------------------------------------------
# Graph emission -- into the existing drivetrain graph builder
# ---------------------------------------------------------------------

def emit_fuel_network_graph(spec: FuelNetworkSpec, node, edge, anchor_pos: tuple[float, float, float],
                            terminal_node: str, is_carbureted: bool) -> None:
    """Emit this network as nodes/edges the existing _discover_fluid_
    circuits will find as ONE 'fuel' circuit: capacity_kg on the
    source node becomes the circuit's depletable bottle_capacity_kg;
    the tightest flow_capacity_kg_s across the in-line stages becomes
    the circuit's flow ceiling (a real series line is limited by its
    tightest stage); a UtilityMain emits no capacity at all (the
    zero-capacity 'never depletes' case the fuel branch already
    handles). `terminal_node` is where the admission device feeds
    (the engine's fuel rail / float bowl / engine body)."""
    f = spec.working_fluid
    x, y, z = anchor_pos
    src = spec.source
    src_id = "fuel.source"
    if isinstance(src, LiquidTank):
        node(src_id, (x - 0.30, y, z - 0.05), "high-pressure-canister",
             capacity_kg=src.capacity_l * f.density_kg_m3 / 1000.0, supply_kind=src.kind, fluid=f.name)
    elif isinstance(src, PressurizedBottle):
        node(src_id, (x - 0.30, y, z - 0.05), "high-pressure-canister", capacity_kg=src.capacity_kg,
             working_pressure_pa=f.storage_pressure_pa, supply_kind=src.kind, fluid=f.name)
    elif isinstance(src, Gasholder):
        node(src_id, (x - 0.30, y, z - 0.05), "high-pressure-canister", capacity_kg=src.capacity_kg,
             working_pressure_pa=f.storage_pressure_pa, supply_kind=src.kind, fluid=f.name)
    elif isinstance(src, BoilerSource):
        node(src_id, (x - 0.40, y, z - 0.10), "high-pressure-canister", capacity_kg=src.dome_capacity_kg,
             working_pressure_pa=src.boiler.operating_pressure_pa, supply_kind=src.kind, fluid=f.name)
    elif isinstance(src, Receiver):
        node(src_id, (x - 0.30, y, z - 0.05), "high-pressure-canister", capacity_kg=src.capacity_kg,
             working_pressure_pa=f.storage_pressure_pa, supply_kind=src.kind, fluid=f.name)
    else:   # UtilityMain -- a bare fitting, no vessel
        node(src_id, (x - 0.20, y, z), "gas-supply-fitting", supply_kind=src.kind, fluid=f.name)

    prev = src_id
    flow_caps: list[float] = []
    if isinstance(src, LiquidTank):
        node("fuel.pump", (x - 0.20, y, z - 0.05),
             "electro-mechanical-pump" if src.pump_kind == "electric" else "mechanical-diaphragm-pump")
        edge("fuel.source_to_pump", src_id, "fuel.pump", "fuel-supply-line",
             radius=max(0.002, src.line_diameter_mm / 2000.0), circuit_identity="fuel",
             medium_rate_state="fuel-flow-and-pressure")
        flow_caps.append(src.pump_flow_capacity_kg_s)
        prev = "fuel.pump"
    for i, st in enumerate(spec.stages):
        nid = f"fuel.stage_{i}_{st.kind}"
        node(nid, (x - 0.15 + 0.02 * i, y, z - 0.05), "fuel-line-device", device_kind=st.kind)
        edge(f"fuel.{prev.split('.')[-1]}_to_{nid.split('.')[-1]}", prev, nid, "fuel-supply-line",
             radius=0.004, circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
        if isinstance(st, (Regulator, FlameArrestor)):
            flow_caps.append(st.flow_capacity_kg_s)
        prev = nid
    adm = spec.admission
    adm_id = f"fuel.admission_{adm.kind}"
    node(adm_id, (x - 0.05, y, z - 0.02), "fuel-admission-device", device_kind=adm.kind)
    edge(f"fuel.{prev.split('.')[-1]}_to_admission", prev, adm_id, "fuel-supply-line", radius=0.004,
         circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure",
         **({"flow_capacity_kg_s": min(flow_caps)} if flow_caps else {}))
    edge("fuel.admission_to_engine", adm_id, terminal_node, "fuel-supply-line", radius=0.004,
         circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")


# ---------------------------------------------------------------------
# Runtime -- what the network actually does each tick
# ---------------------------------------------------------------------

@dataclass
class SupplyTick:
    production_kg_s: float = 0.0             # into the reservoir this tick
    production_composition_frac: float = 1.0
    flow_capacity_kg_s: float = 0.0          # 0 = uncapped
    supply_pressure_pa: float = P_ATM_PA     # what the admission device sees
    valve_open: bool = True
    coolant_heat_draw_w: float = 0.0         # a vaporizer's real draw on the coolant
    fuel_conditioning_target_k: float | None = None   # a fuel heater's target for the line's fuel temperature
    purge_open: bool = False
    boiler_water_frac: float = 1.0
    boiler_failed: bool = False
    warnings: tuple[str, ...] = ()


@dataclass
class FuelNetworkRuntime:
    spec: FuelNetworkSpec
    generator: object | None = None
    boiler: object | None = None
    boiler_water_kg: float = 0.0
    feedwater_tank_kg: float = 0.0
    feed_pump_on: bool = True
    boiler_failed: bool = False
    compressor_loaded: bool = True
    purge_open: bool = False
    _last_fill_frac: float = 1.0
    _vaporizer_capped_kg_s: float = 0.0

    @classmethod
    def build(cls, spec: FuelNetworkSpec) -> "FuelNetworkRuntime":
        spec.validate()
        rt = cls(spec=spec)
        src = spec.source
        if isinstance(src, Gasholder) and src.generator is not None:
            rt.generator = src.generator.build()
            rt.purge_open = spec.has(PurgeValve)   # a real plant starts purging, the operator closes it
        if isinstance(src, BoilerSource):
            rt.boiler = src.boiler.build()
            rt.boiler_water_kg = src.water_capacity_kg
            rt.feedwater_tank_kg = src.feedwater_tank_kg
        return rt

    @property
    def fluid(self) -> WorkingFluid:
        return self.spec.working_fluid

    @property
    def is_expander(self) -> bool:
        return self.fluid.phase == EXPANDER_FLUID

    def mixer_gas_volume_fraction(self) -> float | None:
        adm = self.spec.admission
        if isinstance(adm, Mixer):
            if adm.gas_volume_fraction is not None:
                return adm.gas_volume_fraction
            from otto_langen import design_mixer_gas_volume_fraction
            return design_mixer_gas_volume_fraction(self.fluid.name)
        return None

    def step(self, dt: float, engine_rpm: float, demand_kg_s: float, fill_frac: float,
             engine_running: bool, coolant_temp_k: float = T_AMBIENT_K,
             fuel_temp_k: float = T_AMBIENT_K) -> SupplyTick:
        """Advance the source and every in-line stage; return what the
        reservoir step and the cylinder need. `fill_frac` is the
        circuit's fill from the previous tick (the reservoir is stepped
        by drivetrain_graph after this, with these numbers)."""
        f = self.fluid
        src = self.spec.source
        tick = SupplyTick()
        warnings: list[str] = []
        self._last_fill_frac = fill_frac

        # --- source: production into the reservoir and its pressure ---
        if isinstance(src, Gasholder):
            tick.supply_pressure_pa = f.storage_pressure_pa
            if self.generator is not None:
                prod = self.generator.gas_output_kg_s(dt, engine_rpm=engine_rpm)
                comp = self.generator.step_quality(dt, prod) if hasattr(self.generator, "step_quality") else 1.0
                if self.purge_open:
                    # vented to atmosphere: the holder gets nothing, but
                    # the generator's own plumbing still purges
                    tick.purge_open = True
                    prod = 0.0
                tick.production_kg_s = prod
                tick.production_composition_frac = comp
        elif isinstance(src, UtilityMain):
            tick.supply_pressure_pa = f.storage_pressure_pa
        elif isinstance(src, PressurizedBottle):
            if f.storage == PRESSURIZED_LIQUID:
                tick.supply_pressure_pa = f.storage_pressure_pa if fill_frac > 0.0 else P_ATM_PA
            else:
                tick.supply_pressure_pa = P_ATM_PA + (f.storage_pressure_pa - P_ATM_PA) * fill_frac
        elif isinstance(src, BoilerSource):
            # the dome is a fixed volume: its pressure follows the steam
            # mass it holds (ideal gas), so an engine outrunning the fire
            # "loses its head" continuously and settles where the fire's
            # own steaming rate meets the draw -- the real, gauge-visible
            # behaviour, and what keeps chest pressure honest with an
            # emptying dome instead of a separate availability fudge
            tick.supply_pressure_pa = P_ATM_PA + (src.boiler.operating_pressure_pa - P_ATM_PA) * max(0.0, min(1.0, fill_frac))
            if self.boiler_failed:
                tick.boiler_failed = True
                tick.production_kg_s = 0.0
            else:
                steam = self.boiler.steam_output_kg_s(dt, engine_rpm=engine_rpm)
                if self.boiler_water_kg <= 0.0:
                    steam = 0.0
                # steam raised leaves the water; the feed pump puts it back
                self.boiler_water_kg = max(0.0, self.boiler_water_kg - steam * dt)
                if self.feed_pump_on and self.feedwater_tank_kg > 0.0 and self.boiler_water_kg < src.water_capacity_kg:
                    fed = min(src.feed_pump_kg_s * dt, self.feedwater_tank_kg,
                              src.water_capacity_kg - self.boiler_water_kg)
                    self.boiler_water_kg += fed
                    self.feedwater_tank_kg -= fed
                water_frac = self.boiler_water_kg / max(src.water_capacity_kg, 1e-9)
                tick.boiler_water_frac = water_frac
                if water_frac < src.low_water_frac:
                    warnings.append("boiler low water: crown sheet exposed")
                    # the firebox is still lit with nothing over the crown
                    # sheet -- the real fatal failure; steam stops and the
                    # boiler is finished
                    if self.boiler_water_kg <= 0.0 and self.boiler.fuel_remaining_kg > 0.0:
                        self.boiler_failed = True
                        tick.boiler_failed = True
                        steam = 0.0
                tick.production_kg_s = steam
        elif isinstance(src, Receiver):
            # unloader hysteresis: charge until cut-out, rest until cut-in
            if fill_frac >= src.cut_out_frac:
                self.compressor_loaded = False
            elif fill_frac <= src.cut_in_frac:
                self.compressor_loaded = True
            tick.production_kg_s = src.compressor_rated_kg_s if self.compressor_loaded else 0.0
            tick.supply_pressure_pa = P_ATM_PA + (f.storage_pressure_pa - P_ATM_PA) * fill_frac
        elif isinstance(src, LiquidTank):
            tick.supply_pressure_pa = P_ATM_PA

        # --- in-line stages, in order ---
        caps: list[float] = []
        if isinstance(src, LiquidTank):
            caps.append(src.pump_flow_capacity_kg_s)
        valve_open = True
        for st in self.spec.stages:
            if isinstance(st, Regulator):
                tick.supply_pressure_pa = min(tick.supply_pressure_pa, st.outlet_pressure_pa)
                caps.append(st.flow_capacity_kg_s)
            elif isinstance(st, Vaporizer):
                latent = max(f.latent_heat_j_per_kg, 1.0)
                # cold: the vaporizer body still sits ~50 K above LPG's own
                # boiling point (231 K), so ambient conduction alone
                # vaporizes a real trickle (cold_heat_frac of rating) --
                # the engine starts and runs weak, then comes good as the
                # coolant warms, which is exactly a real LPG cold start
                if coolant_temp_k >= st.min_coolant_k:
                    heat_avail_w = st.max_heat_w
                else:
                    warm_frac = max(0.0, (coolant_temp_k - T_AMBIENT_K) / max(st.min_coolant_k - T_AMBIENT_K, 1.0))
                    heat_avail_w = st.max_heat_w * (st.cold_heat_frac + (1.0 - st.cold_heat_frac) * warm_frac)
                cap = heat_avail_w / latent
                self._vaporizer_capped_kg_s = cap
                caps.append(max(cap, 1e-6))
                tick.coolant_heat_draw_w = min(demand_kg_s, cap) * latent
                if demand_kg_s > cap:
                    warnings.append("vaporizer heat-limited: cold coolant starves the engine")
            elif isinstance(st, FuelHeater):
                # the heater can only hand over what the coolant has
                target = st.target_k if coolant_temp_k >= st.min_coolant_k else max(T_AMBIENT_K, coolant_temp_k)
                tick.fuel_conditioning_target_k = target
                warm_frac = max(0.0, min(1.0, (fuel_temp_k - (T_AMBIENT_K - 10.0)) / max(st.target_k - (T_AMBIENT_K - 10.0), 1.0)))
                visc_frac = st.cold_flow_frac + (1.0 - st.cold_flow_frac) * warm_frac
                base_cap = caps[-1] if caps else 0.05
                caps.append(base_cap * visc_frac)
                if visc_frac < 0.95 and demand_kg_s > base_cap * visc_frac:
                    warnings.append("fuel heater: viscous cold fuel limits delivery")
            elif isinstance(st, CoolerFilter):
                if caps:
                    caps[-1] = caps[-1] * (1.0 - st.restriction_frac)
            elif isinstance(st, FlameArrestor):
                caps.append(st.flow_capacity_kg_s)
            elif isinstance(st, LockoffSolenoid):
                valve_open = valve_open and engine_running
            elif isinstance(st, PurgeValve):
                pass   # its state lives on the runtime (purge_open)
        tick.flow_capacity_kg_s = min(caps) if caps else 0.0
        tick.valve_open = valve_open
        tick.warnings = tuple(warnings)
        return tick

    def set_purge(self, open_: bool) -> None:
        if not self.spec.has(PurgeValve):
            raise ValueError("this network has no PurgeValve to operate")
        self.purge_open = open_

    def add_fuel(self, kg: float) -> None:
        """Rake fuel onto whichever fire this network has (a generator's
        bed or a boiler's firebox)."""
        if self.generator is not None:
            self.generator.add_fuel(kg)
        elif self.boiler is not None:
            self.boiler.add_fuel(kg)
        else:
            raise ValueError("no fire on this network to feed")


# ---------------------------------------------------------------------
# Conversion physics shared by every combustion cylinder
# ---------------------------------------------------------------------

REFERENCE_LIQUID_FUEL = "pump-gasoline-91"
AIR_DENSITY_KG_M3 = P_ATM_PA / (287.0 * T_AMBIENT_K)


def stoichiometric_charge_energy_j_per_m3(fluid_name: str) -> float:
    """Real energy in one cubic metre of stoichiometric CHARGE at
    ambient -- the number that actually sets a cylinder's power on a
    given fuel, because a cylinder is a fixed volume. A liquid fuel's
    vapour displaces almost no air (all the air fits, energy = air /
    AFR * LHV); a gaseous fuel displaces its own volume fraction of the
    air (hydrogen ~30%, producer gas ~47%), which is the real, physical
    reason a natural-gas conversion loses ~10% and a wood-gas one
    30-50% even before slower burning is counted."""
    f = working_fluid(fluid_name)
    if f.phase == LIQUID_FUEL:
        return AIR_DENSITY_KG_M3 / max(f.stoich_afr, 1e-6) * f.energy_density_j_per_kg
    if f.phase == GASEOUS_FUEL:
        return f.stoich_vol_frac * f.density_kg_m3 * f.energy_density_j_per_kg
    return 0.0


def charge_energy_factor(fluid_name: str, reference: str = REFERENCE_LIQUID_FUEL) -> float:
    """This fluid's stoichiometric charge energy relative to the
    reference gasoline the catalogue's bmep figures were rated on --
    the real, derived power multiplier for a fuel conversion."""
    ref = stoichiometric_charge_energy_j_per_m3(reference)
    return stoichiometric_charge_energy_j_per_m3(fluid_name) / max(ref, 1e-9)


def knock_compatibility_factor(fluid_name: str, required_octane: float) -> float:
    """Derived, not hand-set: a fuel at or above the engine's own
    required octane runs at full rating; below it, a real derate of
    2% per octane point short (the same order as the ~1-2% per point
    timing pullback a knock-limited engine actually takes), floored so
    a wildly wrong fuel still runs, badly."""
    f = working_fluid(fluid_name)
    if not f.combustible:
        return 0.0
    if f.effective_octane >= required_octane:
        return 1.0
    return max(0.6, 1.0 - 0.02 * (required_octane - f.effective_octane))


def intake_flashback_risk(fluid_name: str, intake_charge_temp_k: float, admission) -> float:
    """0..1 real risk that a hot intake lights the mixture upstream of
    the valve this cycle (an intake backfire / flashback) -- a real,
    documented hydrogen-port-injection failure. Scales with how far
    below hydrocarbon-normal the fuel's minimum ignition energy sits
    and how hot the tract already is; zero for direct/liquid admission
    (no premixed charge sits in the tract at all) and for expander
    fluids."""
    f = working_fluid(fluid_name)
    if not f.combustible or f.min_ignition_energy_mj <= 0.0:
        return 0.0
    if not isinstance(admission, (Mixer, GasInjector)):
        return 0.0
    sensitivity = max(0.0, 0.25 / f.min_ignition_energy_mj - 1.0)   # 0 for hydrocarbons, ~14 for hydrogen
    hot = max(0.0, (intake_charge_temp_k - 330.0) / 150.0)
    return max(0.0, min(1.0, sensitivity * hot * 0.1))


def hardware_notes(spec: FuelNetworkSpec) -> list[str]:
    """Real engine-side consequences the network implies (advice for
    the build, not physics the sim enforces)."""
    f = spec.working_fluid
    notes: list[str] = []
    if f.hardened_valve_seats_required:
        notes.append("hardened valve seats: a dry gaseous fuel gives no seat lubrication")
    if f.phase == GASEOUS_FUEL and f.laminar_flame_speed_m_s > 1.5:
        notes.append("retard timing / run lean: very fast flame, pre-ignition prone")
    if f.phase == GASEOUS_FUEL and f.laminar_flame_speed_m_s < 0.35:
        notes.append("advance timing: slow flame")
    if f.effective_octane >= 120.0:
        notes.append("compression ratio can be raised: high knock resistance")
    if f.phase == EXPANDER_FLUID:
        notes.append("double-acting cylinder, crosshead, gland, lubricator, drain cocks")
    return notes


# ---------------------------------------------------------------------
# The conversion itself
# ---------------------------------------------------------------------

def convert_engine(engine, spec: FuelNetworkSpec):
    """Return a copy of `engine` running on `spec`'s fluid through
    `spec`'s network -- the same cylinder, a different fuel network.
    The knock derate is DERIVED against the octane the engine was
    catalogued on (knock_compatibility_factor), the charge-energy
    factor is applied live by engine_cycle_sim, and an Otto-Langen's
    own ignition table follows the fluid. Raises if the network is
    missing anything a real installation of that fluid requires."""
    import dataclasses
    spec.validate()
    required_octane = working_fluid(engine.preferred_fuel_profile).effective_octane \
        if engine.preferred_fuel_profile in __import__("working_fluids").WORKING_FLUIDS else 91.0
    compat = dict(engine.fuel_compatibility)
    compat[spec.fluid] = knock_compatibility_factor(spec.fluid, required_octane)
    changes = dict(preferred_fuel_profile=spec.fluid, fuel_compatibility=compat, fuel_network=spec)
    atmo = getattr(engine, "atmospheric", None)
    if atmo is not None:
        changes["atmospheric"] = dataclasses.replace(atmo, fuel_profile=spec.fluid)
    return dataclasses.replace(engine, **changes)
