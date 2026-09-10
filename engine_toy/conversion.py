"""Cylinder CONVERSION contracts: given this engine's actual combustion
chamber (bore, stroke, static compression ratio, how it ignites) and
a target working fluid, what does a faithful conversion have to
change, and what does it become? The contract is the engineering
answer written down -- ignition mode, compression-ratio target and
the two real routes to it (a piston/head clearance change, or a
stroke change), the timing shift the fluid's own flame speed calls
for, the fuel conditioning and hardware the fluid's network needs --
and apply_conversion() then produces the converted Engine from it:
the same cylinder, re-rated from first principles with the fuel's own
chemistry (engines.derive_gross_bmep_pa), on the fluid's own network
(fuel_network.py). This is what lets an antique run a modern fuel and
a commuter engine run old fry oil, with the consequences shown rather
than hidden.

Everything here is derived, not tuned:
  - Spark fuels: the knock-limited compression ratio is the inverse of
    the SAME relation derive_gross_bmep_pa already rates knock with
    (required octane = 80 + 3*(CR - 8)), so CR_max = 8 + (octane-80)/3,
    held under a practical ceiling (hydrogen pre-ignites well before
    its octane says it would knock; gaseous fuels top out ~14:1 in
    real conversions).
  - Compression-ignition fuels: 17.5:1 (the real range for indirect/
    direct-injection light diesels), never below 15:1.
  - Timing: laminar flame speed relative to gasoline's ~0.4 m/s --
    slower burns want advance, hydrogen's ~3 m/s wants retard.
  - Clearance route: Vc = Vd/(CR-1); a head skim / domed piston takes
    the difference out (or a dished piston / thicker gasket puts it
    in); the stroke route keeps Vc and lengthens the crank throw.
  - Expander fluids: the cylinder becomes a cutoff expander
    (expander.py) -- single-acting, because a trunk-piston engine has
    no crosshead to seal a rod through the bottom cover.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from working_fluids import (working_fluid, WorkingFluid, LIQUID_FUEL, GASEOUS_FUEL, EXPANDER_FLUID,
                            PRESSURIZED_LIQUID, HIGH_PRESSURE_GAS, CRYOGENIC, GASHOLDER, BOILER, RECEIVER)
from fuel_network import (FuelNetworkSpec, LiquidTank, PressurizedBottle, Gasholder, UtilityMain, BoilerSource,
                          Receiver, Regulator, Vaporizer, FlameArrestor, LockoffSolenoid, FuelHeater, CoolerFilter,
                          Mixer, GasInjector, LiquidCarburetor, LiquidInjector, CutoffValve,
                          knock_compatibility_factor, hardware_notes)
from expander import ExpanderCylinderSpec, mean_effective_pressure_pa

GASOLINE_FLAME_SPEED_M_S = 0.40
CI_TARGET_CR = 17.5
CI_MIN_CR = 15.0
SPARK_PRACTICAL_CR_CAP = {"liquid": 12.5, "gas": 14.0}
HYDROGEN_PREIGNITION_CR_CAP = 11.0
CI_REDLINE_CAP_RPM = 4_800.0


def knock_limited_cr(octane: float) -> float:
    return 8.0 + (octane - 80.0) / 3.0


@dataclass(frozen=True)
class ConversionContract:
    base_identity: str
    fluid: str
    ignition_mode: str                  # "spark" | "compression" | "expander"
    cr_current: float
    cr_target: float
    cr_route: str                       # "unchanged" | "piston/head" | "stroke"
    clearance_change_cc: float          # per cylinder; + = clearance removed (skim/dome), - = added (dish/gasket)
    head_skim_mm: float                 # the piston/head route expressed as a skim depth (negative = add)
    stroke_delta_mm: float              # the stroke route
    timing_shift_deg: float             # + advance, - retard (spark only)
    network: FuelNetworkSpec
    fuel_conditioning: tuple[str, ...]
    hardware: tuple[str, ...]
    predicted_torque_factor: float
    predicted_power_factor: float
    notes: tuple[str, ...]

    def describe(self) -> list[str]:
        f = working_fluid(self.fluid)
        out = [f"CONVERT {self.base_identity} -> {self.fluid} [{self.ignition_mode}]"]
        if self.ignition_mode == "expander":
            out.append(f"  cylinder becomes a single-acting cutoff expander on {self.fluid}")
        else:
            route = {"unchanged": "chamber unchanged",
                     "piston/head": f"clearance {self.clearance_change_cc:+.1f} cc/cyl ({self.head_skim_mm:+.2f} mm skim; or stroke {self.stroke_delta_mm:+.1f} mm)",
                     "stroke": f"stroke {self.stroke_delta_mm:+.1f} mm"}[self.cr_route]
            out.append(f"  CR {self.cr_current:.1f} -> {self.cr_target:.1f}: {route}")
            if self.ignition_mode == "spark":
                out.append(f"  timing {self.timing_shift_deg:+.0f} deg (flame {f.laminar_flame_speed_m_s:.2f} m/s vs gasoline {GASOLINE_FLAME_SPEED_M_S:.2f})")
        stages = " > ".join([self.network.source.kind] + [s.kind for s in self.network.stages] + [self.network.admission.kind])
        out.append(f"  network: {stages}")
        for c in self.fuel_conditioning:
            out.append(f"  conditioning: {c}")
        for h in self.hardware:
            out.append(f"  hardware: {h}")
        out.append(f"  predicted torque x{self.predicted_torque_factor:.2f}, power x{self.predicted_power_factor:.2f}")
        for n in self.notes:
            out.append(f"  note: {n}")
        return out


# ---------------------------------------------------------------------
# A real default network for each fluid
# ---------------------------------------------------------------------

def default_network(fluid_name: str, engine) -> FuelNetworkSpec:
    f = working_fluid(fluid_name)
    fd = engine.fuel_delivery
    if f.phase == LIQUID_FUEL:
        tank = LiquidTank(capacity_l=fd.tank_capacity_l, pump_kind=fd.pump_kind,
                          pump_flow_capacity_kg_s=fd.pump_flow_capacity_kg_s, line_diameter_mm=fd.line_diameter_mm)
        stages = (FuelHeater(),) if f.viscous_at_ambient else ()
        if f.ignition == "compression":
            return FuelNetworkSpec(f.name, tank, LiquidInjector(), stages)
        carb = getattr(engine, "carburetor", None)
        adm = LiquidCarburetor() if (carb is not None and carb.is_carbureted) else LiquidInjector()
        return FuelNetworkSpec(f.name, tank, adm, stages)
    if f.phase == GASEOUS_FUEL:
        if f.storage == HIGH_PRESSURE_GAS:
            stages = [Regulator(outlet_pressure_pa=3.0e5, flow_capacity_kg_s=max(0.02, fd.pump_flow_capacity_kg_s))]
            if f.flame_arrestor_required:
                stages.append(FlameArrestor(flow_capacity_kg_s=max(0.02, fd.pump_flow_capacity_kg_s)))
            stages.append(LockoffSolenoid())
            adm = GasInjector() if f.name == "hydrogen" else Mixer()
            return FuelNetworkSpec(f.name, PressurizedBottle(capacity_kg=8.0), adm, tuple(stages))
        if f.storage in (PRESSURIZED_LIQUID, CRYOGENIC):
            return FuelNetworkSpec(f.name, PressurizedBottle(capacity_kg=20.0), Mixer(),
                                   (Regulator(1.2e5, flow_capacity_kg_s=max(0.02, fd.pump_flow_capacity_kg_s)),
                                    Vaporizer(), LockoffSolenoid()))
        if f.storage == GASHOLDER:
            stages = [CoolerFilter()] if f.name == "wood-gas" else []
            if f.flame_arrestor_required:
                stages.append(FlameArrestor(flow_capacity_kg_s=max(0.02, fd.pump_flow_capacity_kg_s)))
            return FuelNetworkSpec(f.name, UtilityMain(), Mixer(), tuple(stages))
        return FuelNetworkSpec(f.name, LiquidTank(capacity_l=fd.tank_capacity_l), Mixer(), ())
    # expander fluids
    if f.storage == BOILER:
        from gas_works import BoilerSpec
        return FuelNetworkSpec(f.name, BoilerSource(boiler=BoilerSpec(raw_fuel="coal", grate_area_m2=0.5,
                                                                      operating_pressure_pa=f.storage_pressure_pa)),
                               CutoffValve(cutoff_frac=0.40), ())
    return FuelNetworkSpec(f.name, Receiver(capacity_kg=60.0, compressor_rated_kg_s=0.05), CutoffValve(cutoff_frac=0.40),
                           (Regulator(outlet_pressure_pa=f.storage_pressure_pa * 0.85, flow_capacity_kg_s=1.0),
                            CoolerFilter(restriction_frac=0.05)))


# ---------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------

def _chamber(engine) -> tuple[float, float, float, int]:
    arch = engine.architecture
    cyl = max(1, arch.cylinders)
    bore = arch.bore_m
    stroke = arch.stroke_m
    cr = arch.compression_ratio
    if getattr(engine, "compression_ignition", False) and cr < CI_MIN_CR:
        # a catalogue diesel left on the dataclass's generic 10:1 default
        # never had a declared chamber; a real light/industrial diesel
        # sits ~17.5:1, and that is the chamber a conversion starts from
        cr = CI_TARGET_CR
    if bore <= 0.0 or stroke <= 0.0:
        # no declared slider-crank (rotary/electric): back one out of
        # displacement with a square bore/stroke so the clearance
        # arithmetic still has a real volume to work on
        vd_cyl = engine.displacement_l / 1000.0 / cyl
        bore = stroke = (4.0 * vd_cyl / math.pi) ** (1.0 / 3.0)
    return bore, stroke, cr, cyl


def plan_conversion(engine, fluid_name: str, network: FuelNetworkSpec | None = None,
                    cr_route: str = "piston/head") -> ConversionContract:
    f = working_fluid(fluid_name)
    net = network if network is not None else default_network(fluid_name, engine)
    net.validate()
    bore, stroke, cr_now, cyl = _chamber(engine)
    vd_cyl = math.pi / 4.0 * bore * bore * stroke
    notes: list[str] = []
    conditioning: list[str] = []
    hardware = list(hardware_notes(net))
    was_ci = bool(getattr(engine, "compression_ignition", False))

    if f.phase == EXPANDER_FLUID:
        mode = "expander"
        cr_target = cr_now
        route = "unchanged"
        timing = 0.0
        hardware.append("crosshead-less trunk piston: single-acting only; valve gear replaces the camshaft")
        hardware.append("drain cocks and a lubricator on the cylinder" if f.name == "steam"
                        else "exhaust dryer/separator (icing) and an air-line lubricator")
    elif f.ignition == "compression":
        mode = "compression"
        cr_target = max(CI_MIN_CR, CI_TARGET_CR) if not was_ci else max(cr_now, CI_MIN_CR)
        route = cr_route if abs(cr_target - cr_now) > 0.05 else "unchanged"
        timing = 0.0
        if not was_ci:
            hardware.append("high-pressure injection pump + injectors, glow plugs; spark system removed")
            hardware.append("stronger rods/bearings for the higher peak pressure")
        if f.viscous_at_ambient:
            conditioning.append("fuel heater to ~70 C (viscosity) with a two-tank start on diesel")
    else:
        mode = "spark"
        cap = SPARK_PRACTICAL_CR_CAP["gas" if f.phase == GASEOUS_FUEL else "liquid"]
        if f.name == "hydrogen":
            cap = HYDROGEN_PREIGNITION_CR_CAP
            notes.append("hydrogen pre-ignites on hot spots long before it knocks: CR capped for that, not for octane")
        limit = min(knock_limited_cr(f.effective_octane), cap)
        if was_ci:
            # a diesel chamber is far too tight for a spark fuel: new
            # pistons bring it down, and a spark system goes in
            cr_target = min(limit, 12.0)
            hardware.append("spark ignition system fitted; lower-crown pistons")
        elif cr_now > limit + 0.05:
            cr_target = limit
            notes.append("current CR exceeds this fuel's knock limit: chamber must be opened up, not left")
        else:
            cr_target = min(limit, max(cr_now, cr_now + 0.0))
            # raise toward the fuel's own knock limit only when it's a
            # real gain worth new pistons: a full point or more
            if limit - cr_now >= 1.0:
                cr_target = limit
        route = cr_route if abs(cr_target - cr_now) > 0.05 else "unchanged"
        v = max(f.laminar_flame_speed_m_s, 0.05)
        timing = max(-15.0, min(20.0, 15.0 * (GASOLINE_FLAME_SPEED_M_S / v - 1.0)))
        if f.phase == GASEOUS_FUEL and f.storage in (PRESSURIZED_LIQUID, CRYOGENIC):
            conditioning.append("coolant-heated vaporizer: weak until the engine is warm")

    # the two real routes to the target clearance
    vc_now = vd_cyl / max(cr_now - 1.0, 0.01)
    vc_target = vd_cyl / max(cr_target - 1.0, 0.01)
    clearance_change_cc = (vc_now - vc_target) * 1e6
    head_skim_mm = clearance_change_cc * 1e-6 / (math.pi / 4.0 * bore * bore) * 1000.0
    stroke_delta_mm = (stroke * (cr_target - 1.0) / max(cr_now - 1.0, 0.01) - stroke) * 1000.0

    # predicted rating change from the same first-principles rating the
    # catalogue's own builder uses
    from engines import derive_gross_bmep_pa
    if mode == "expander":
        spec = _expander_spec(engine, bore, stroke, cyl)
        supply = _network_supply_pressure(net, f)
        mep = max(0.0, mean_effective_pressure_pa(supply, net.admission.cutoff_frac, f.gamma, spec.back_pressure_pa))
        bmep_new = mep * spec.mechanical_efficiency
        torque_factor = (bmep_new * spec.swept_volume_m3 * spec.strokes_per_rev * cyl) / max(
            engine.bmep_pa * engine.displacement_l / 1000.0 / engine._cycle_radians * 2.0 * math.pi, 1e-9)
        power_factor = torque_factor * (_expander_rated_rpm(engine) / max(engine.power_peak_rpm, 1.0))
    else:
        bmep_old = engine.bmep_pa
        bmep_new = derive_gross_bmep_pa(
            compression_ratio=cr_target, fuel_profile=f.name, intake_system=engine.intake_system,
            exhaust_system=engine.exhaust_system, forced_induction=engine.forced_induction,
            combustion_efficiency=engine.combustion_efficiency, compression_ignition=(mode == "compression"))
        if f.phase == GASEOUS_FUEL:
            # a gas displaces its own share of the air charge -- the
            # fixed-volume correction derive_gross_bmep_pa's per-kg-air
            # rating doesn't see
            bmep_new *= (1.0 - f.stoich_vol_frac)
        # the catalogue's rating for the base engine also came off that
        # derivation for ITS fuel/CR (or was hand-rated); scale relative
        # so a hand-rated antique keeps its own baseline
        bmep_base_derived = derive_gross_bmep_pa(
            compression_ratio=cr_now, fuel_profile=engine.preferred_fuel_profile, intake_system=engine.intake_system,
            exhaust_system=engine.exhaust_system, forced_induction=engine.forced_induction,
            combustion_efficiency=engine.combustion_efficiency, compression_ignition=was_ci)
        if engine.preferred_fuel_profile in _GAS_NAMES:
            bmep_base_derived *= (1.0 - working_fluid(engine.preferred_fuel_profile).stoich_vol_frac)
        torque_factor = bmep_new / max(bmep_base_derived, 1e-9)
        rpm_factor = (min(engine.power_peak_rpm, CI_REDLINE_CAP_RPM) / max(engine.power_peak_rpm, 1.0)
                      if mode == "compression" else 1.0)
        power_factor = torque_factor * rpm_factor
        if mode == "compression" and not was_ci:
            notes.append(f"redline capped at {CI_REDLINE_CAP_RPM:.0f} rpm: injection/combustion duration limits a diesel")

    return ConversionContract(
        base_identity=engine.identity, fluid=f.name, ignition_mode=mode, cr_current=cr_now, cr_target=cr_target,
        cr_route=route, clearance_change_cc=clearance_change_cc, head_skim_mm=head_skim_mm,
        stroke_delta_mm=stroke_delta_mm, timing_shift_deg=timing, network=net,
        fuel_conditioning=tuple(conditioning), hardware=tuple(hardware),
        predicted_torque_factor=torque_factor, predicted_power_factor=power_factor, notes=tuple(notes))


_GAS_NAMES = {n for n, wf in __import__("working_fluids").WORKING_FLUIDS.items() if wf.phase == GASEOUS_FUEL}


def _expander_spec(engine, bore, stroke, cyl) -> ExpanderCylinderSpec:
    return ExpanderCylinderSpec(bore_m=bore, stroke_m=stroke, cylinders=cyl, double_acting=False,
                                default_cutoff_frac=0.40, mechanical_efficiency=0.85)


def _expander_rated_rpm(engine) -> float:
    return min(engine.redline_rpm * 0.4, 1_500.0)


def _network_supply_pressure(net: FuelNetworkSpec, f: WorkingFluid) -> float:
    src = net.source
    supply = src.boiler.operating_pressure_pa if hasattr(src, "boiler") else f.storage_pressure_pa
    for st in net.stages:
        if isinstance(st, Regulator):
            supply = min(supply, st.outlet_pressure_pa)
    return supply


# ---------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------

def apply_conversion(engine, contract: ConversionContract):
    """The converted Engine: chamber changed per the contract's route,
    ignition mode switched, re-rated from first principles on the new
    fluid, on the contract's network. bmep_rated_fuel records that
    the rating already carries the fluid's charge energy, so the live
    sim doesn't apply it a second time (see engine_cycle_sim)."""
    from engines import derive_gross_bmep_pa
    f = working_fluid(contract.fluid)
    bore, stroke, cr_now, cyl = _chamber(engine)
    arch = engine.architecture
    label = f"{engine.label} on {f.name}"
    identity = f"{engine.identity}+{f.name}"
    braking_frac = engine.braking_bmep_pa / max(engine.bmep_pa, 1e-9)
    compat = dict(engine.fuel_compatibility)

    if contract.ignition_mode == "expander":
        spec = _expander_spec(engine, bore, stroke, cyl)
        supply = _network_supply_pressure(contract.network, f)
        mep = max(0.0, mean_effective_pressure_pa(supply, contract.network.admission.cutoff_frac, f.gamma,
                                                  spec.back_pressure_pa))
        rated = _expander_rated_rpm(engine)
        new_arch = replace(arch, cylinders=0, layout=f"{cyl}-cylinder-single-acting-expander (converted {arch.layout})",
                           firing_order=[])
        compat[f.name] = 1.0
        return replace(
            engine, identity=identity, label=label, kind="expander",
            displacement_l=spec.swept_volume_m3 * spec.strokes_per_rev * cyl * 1000.0,
            bmep_pa=mep * spec.mechanical_efficiency, braking_bmep_pa=0.0,
            idle_rpm=rated * 0.15, torque_peak_rpm=rated * 0.15, power_peak_rpm=rated, redline_rpm=rated * 1.4,
            combustion_efficiency=1.0, architecture=new_arch, preferred_fuel_profile=f.name,
            fuel_compatibility=compat, compression_ignition=False, expander=spec, fuel_network=contract.network,
            bmep_rated_fuel=f.name, ignition_timing_offset_deg=0.0, atmospheric=None, turbine=None)

    ci = contract.ignition_mode == "compression"
    new_stroke = stroke
    if contract.cr_route == "stroke":
        new_stroke = stroke + contract.stroke_delta_mm / 1000.0
    new_arch = replace(arch, compression_ratio=contract.cr_target,
                       bore_m=bore if arch.bore_m > 0.0 else arch.bore_m,
                       stroke_m=new_stroke if arch.stroke_m > 0.0 else arch.stroke_m)
    displacement_l = engine.displacement_l * (new_stroke / stroke)
    bmep = derive_gross_bmep_pa(
        compression_ratio=contract.cr_target, fuel_profile=f.name, intake_system=engine.intake_system,
        exhaust_system=engine.exhaust_system, forced_induction=engine.forced_induction,
        combustion_efficiency=engine.combustion_efficiency, compression_ignition=ci)
    if f.phase == GASEOUS_FUEL:
        bmep *= (1.0 - f.stoich_vol_frac)
    # keep a hand-rated base's own baseline: scale the base rating by
    # the derived ratio rather than replacing it outright
    bmep = engine.bmep_pa * contract.predicted_torque_factor
    required_octane = working_fluid(engine.preferred_fuel_profile).effective_octane \
        if engine.preferred_fuel_profile in __import__("working_fluids").WORKING_FLUIDS else 91.0
    compat[f.name] = 1.0 if ci else knock_compatibility_factor(f.name, min(required_octane, f.effective_octane))
    redline = engine.redline_rpm
    power_peak = engine.power_peak_rpm
    torque_peak = engine.torque_peak_rpm
    if ci and not getattr(engine, "compression_ignition", False):
        scale = min(1.0, CI_REDLINE_CAP_RPM / max(redline, 1.0))
        redline *= scale; power_peak *= scale; torque_peak *= scale
    return replace(
        engine, identity=identity, label=label, architecture=new_arch, displacement_l=displacement_l,
        bmep_pa=bmep, braking_bmep_pa=bmep * braking_frac, redline_rpm=redline, power_peak_rpm=power_peak,
        torque_peak_rpm=torque_peak, compression_ignition=ci, preferred_fuel_profile=f.name,
        fuel_compatibility=compat, fuel_network=contract.network, bmep_rated_fuel=f.name,
        ignition_timing_offset_deg=contract.timing_shift_deg if not ci else 0.0)


def convert(engine, fluid_name: str, **kw):
    """plan + apply in one call; returns (contract, converted_engine)."""
    contract = plan_conversion(engine, fluid_name, **kw)
    return contract, apply_conversion(engine, contract)


def conversion_targets(engine) -> list[str]:
    """Every registered working fluid other than the one this engine
    already runs on -- what the demo cycles through."""
    from working_fluids import WORKING_FLUIDS
    return [n for n in WORKING_FLUIDS if n != engine.preferred_fuel_profile]
