"""A reusable engine-bay powerplant package.

The engine is replaceable.  The package around it is not an engine and is
therefore authored with :mod:`machines`: PTO coupling, hydraulic pump, air
compressor, permanent-magnet generator and battery, and refrigerant
compressor.  The beam model sees their bodies and their mounts.  Shaft,
fluid and electrical routes remain routed graph edges and cannot stiffen the
platform accidentally.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fuel_network import ElectricFuelHeater
from working_fluids import WORKING_FLUIDS
import math

import numpy as np

from assembly_ports import PartPort
from electrical_distribution import (
    BuildingCable, CircuitBreaker, DistributionPanel, OutletBox,
    standard_services,
)
from dc_power import Conductor
from machines import Machine, MachineLine, MachinePart


PMG_RATED_W = 30_000.0
PMG_RAW_AC_RATED_W = 35_000.0
PMG_REFERENCE_RPM = 3_000.0
PMG_EFFICIENCY = 0.91
PMG_SOURCE = "https://www.meccalte.com/en/products/alternators/zanardi-products/pmg"
UTILITY_PTO_STEP_UP = 4.5
FIFTY_US_GALLONS_L = 50.0 * 3.785411784
FUEL_DRUM_RADIUS_M = 0.286
FUEL_DRUM_HEIGHT_M = 0.851
FUEL_DRUM_TARE_KG = 19.0


@dataclass(frozen=True)
class DropInBarrelFuelPump:
    """A cap, pickup and electric lift pump that makes any drum a source."""
    identity: str
    barrel_identity: str
    fuel: str
    rated_flow_kg_s: float = 0.18
    rated_pressure_pa: float = 550_000.0
    dry_run_cutoff_frac: float = 0.005
    check_valve_cracking_pa: float = 12_000.0

    def available(self, fill_frac: float, enabled: bool = True) -> bool:
        return bool(enabled and float(fill_frac) > self.dry_run_cutoff_frac)


@dataclass
class FuelControlUnit:
    """Stage compatible barrel pumps onto one fuel-typed manifold.

    Carries an INLINE ELECTRIC HEATER, because a fixed installation is
    exactly the case the coolant-heated fuel heater cannot serve: that
    one needs an engine already warm, which is why a vehicle SVO
    conversion starts on diesel from a second tank and switches over. A
    station has mains power and can heat the fuel before anything is
    running, so it can start cold on a drum of waste oil or crude.

    The heater is power limited rather than magic: a stream costs
    mdot * cp * dT to raise, so the temperature it reaches falls as
    demand rises and the fuel arrives half-heated instead of the heater
    refusing. What that costs the manifold is delivery -- a fuel below
    its working temperature does not pump.
    """
    manifold_fuel: str = "multifuel"
    pumps: tuple[DropInBarrelFuelPump, ...] = ()
    enabled: dict[str, bool] = field(default_factory=dict)
    last_command: dict[str, float] = field(default_factory=dict)
    heater: ElectricFuelHeater | None = field(default_factory=ElectricFuelHeater)
    #: Fuel temperature arriving at the FCU. NOT a constant 15 C: this is
    #: a deployed installation, and a drum standing in the sun is nothing
    #: like a drum in a northern European yard. It matters more than it
    #: looks -- waste oil wants 80 C, which is a 65 K lift from 15 C and
    #: only a 35 K lift from 45 C, so the same heater that falls short in
    #: the cold has margin to spare in the desert.
    inlet_k: float = 318.15            # 45 C, a drum in desert sun
    #: what the heater actually managed last command, for the panel
    heater_draw_w: float = 0.0
    heater_outlet_k: float = 318.15
    #: DOWNSTREAM REHEAT, once something is running. The FCU's electric
    #: heater only has to get the plant STARTED; a turbine exhausting
    #: north of 700 K has heat to spare and can raise its own fuel the
    #: rest of the way. Set by the plant when an engine is up, so a cold
    #: start is electric and a running station is nearly free -- the same
    #: arrangement as an SVO conversion starting on diesel and switching
    #: over, done with waste heat instead of a second tank.
    reheat_k: float = 0.0

    def attach(self, pump: DropInBarrelFuelPump) -> None:
        """Register a cap-pump dropped into a newly rolled-up barrel."""
        if any(existing.identity == pump.identity for existing in self.pumps):
            raise ValueError(f"fuel pump {pump.identity!r} already attached")
        self.pumps = (*self.pumps, pump)
        self.enabled[pump.identity] = True

    def detach(self, pump_identity: str) -> None:
        self.pumps = tuple(p for p in self.pumps
                           if p.identity != pump_identity)
        self.enabled.pop(pump_identity, None)
        self.last_command.pop(pump_identity, None)

    def command(self, demand_kg_s: float, fill_fraction: dict[str, float],
                declared_fuel: dict[str, str] | None = None) -> dict[str, float]:
        remaining = max(0.0, float(demand_kg_s))
        fuel_of = declared_fuel or {}
        # THE HEATER, BEFORE ANY OF IT IS PUMPED. A viscous fuel that
        # has not reached temperature simply does not deliver, so this
        # scales the whole manifold's demand rather than being a note on
        # a panel somewhere.
        required_k = 288.15
        fluid = WORKING_FLUIDS.get(self.manifold_fuel)
        if fluid is not None and getattr(fluid, "preheat_c", 0.0) > 0.0:
            required_k = 273.15 + float(fluid.preheat_c)
        elif fluid is not None and fluid.viscous_at_ambient:
            required_k = 323.15
        inlet_k = max(float(self.inlet_k), 0.0)
        if self.heater is not None and required_k > inlet_k:
            self.heater_outlet_k = max(
                self.heater.delivered_temp_k(remaining, inlet_k),
                float(self.reheat_k))
            # the electric heater only pays for what the reheat did not
            # already provide -- a running plant barely loads it
            self.heater_draw_w = (
                0.0 if self.reheat_k >= required_k
                else self.heater.draw_w(remaining, inlet_k))
            span = max(required_k - inlet_k, 1e-6)
            warmed = max(0.0, min(1.0, (self.heater_outlet_k - inlet_k) / span))
            remaining *= (self.heater.cold_flow_frac
                          + (1.0 - self.heater.cold_flow_frac) * warmed)
        else:
            self.heater_outlet_k = inlet_k
            self.heater_draw_w = 0.0

        result = {}
        for pump in self.pumps:
            compatible = fuel_of.get(pump.barrel_identity, pump.fuel) == self.manifold_fuel
            on = self.enabled.get(pump.identity, True)
            available = compatible and pump.available(
                fill_fraction.get(pump.barrel_identity, 0.0), on)
            flow = min(remaining, pump.rated_flow_kg_s) if available else 0.0
            result[pump.identity] = flow
            remaining -= flow
        self.last_command = result
        return result


def _port(identity, part, kind, position, direction, radius, *,
          fluid="mechanical", connected_to=None):
    return PartPort(identity, part, kind, np.asarray(position, float),
                    np.asarray(direction, float), radius, False,
                    connected_to=connected_to, fluid=fluid,
                    joint="bolted-flange")


@dataclass(frozen=True)
class PowerplantBasic:
    """One standard bay package into which a compatible engine fits."""
    identity: str
    engine_identity: str
    centre: tuple[float, float, float]
    engine_half_extent_m: tuple[float, float, float]
    bay_half_extent_m: tuple[float, float, float] = (1.15, 0.62, 1.10)

    def accepts(self, half_extent_m) -> bool:
        return all(float(have) <= float(room) for have, room in zip(
            half_extent_m, self.engine_half_extent_m))

    def machine(self) -> Machine:
        p = np.asarray(self.centre, float)
        side = -1.0 if p[0] < 0.0 else 1.0
        # Put services along the inboard edge and fore/aft shoulders of
        # the prime mover.  Mirroring X gives both bays the same package.
        dx = np.array([-side * 0.67, -0.02, 0.0])
        locations = {
            "pto": p + dx,
            "hydraulic": p + np.array([-side * 0.68, 0.02, 0.48]),
            "air": p + np.array([-side * 0.68, 0.02, -0.48]),
            "generator": p + np.array([side * 0.66, 0.04, 0.46]),
            "power_converter": p + np.array([side * 0.66, 0.30, 0.18]),
            "switchboard": p + np.array([side * 0.72, 0.30, -0.16]),
            "outlet_dc": p + np.array([side * 0.76, 0.33, -0.34]),
            "outlet_230": p + np.array([side * 0.76, 0.33, -0.46]),
            "outlet_400_3ph": p + np.array([side * 0.76, 0.33, -0.58]),
            "outlet_split": p + np.array([side * 0.76, 0.33, -0.70]),
            "battery": p + np.array([side * 0.68, -0.08, -0.46]),
            "refrigerant": p + np.array([0.0, 0.38, -0.60]),
            "coolant_reservoir": p + np.array([side * 0.58, 0.34, -0.08]),
            "coolant_pump": p + np.array([side * 0.58, 0.34, 0.26]),
        }
        prefix = self.identity
        def ident(name): return f"{prefix}.{name}"
        services = standard_services(prefix)

        def outlet(name, service_key, awg, rating_a, connector):
            service = services[service_key]
            cable = BuildingCable(
                ident(f"cable.{name}"), service, awg=awg,
                length_m=float(np.linalg.norm(
                    locations[name] - locations["switchboard"])),
                solid_copper=True, in_conduit=True,
                allowable_ampacity_a=rating_a,
            )
            breaker = CircuitBreaker(
                ident(f"breaker.{name}"), service, rating_a, cable)
            return cable, OutletBox(
                ident(name), service, breaker, connector_standard=connector)

        outlet_specs = {
            "outlet_dc": outlet(
                "outlet_dc", "48v-dc", 0, 150.0, "high-current-2p+pe"),
            "outlet_230": outlet(
                "outlet_230", "230v-1ph-50hz", 6, 63.0,
                "iec-60309-2p+e"),
            "outlet_400_3ph": outlet(
                "outlet_400_3ph", "230-400v-3ph-50hz", 8, 32.0,
                "iec-60309-3p+n+e"),
            "outlet_split": outlet(
                "outlet_split", "120-240v-split-60hz", 8, 50.0,
                "split-phase-2p+n+e"),
        }
        distribution_sections = tuple(
            DistributionPanel(
                ident(f"switchboard.{name}"), spec[1].service,
                spec[1].breaker.rating_a, (spec[1],),
                neutral_to_frame_bond=(
                    spec[1].service.arrangement != "dc-two-wire"),
            )
            for name, spec in outlet_specs.items()
        )
        parts = []
        specifications = (
            ("pto", "power-takeoff-coupler", (0.12, .11, .11), 18.0,
             "power-input", {"rated_torque_nm": 1800.0,
                              "step_up_ratio": UTILITY_PTO_STEP_UP}),
            ("hydraulic", "shaft-hydraulic-gear-pump", (.18, .16, .18), 42.0,
             "hydraulic-pump", {"displacement_cc_rev": 48.0,
                                "relief_pressure_pa": 21_000_000.0}),
            ("air", "shaft-pneumatic-compressor", (.22, .19, .20), 58.0,
             "pneumatic-compressor", {"rated_free_air_l_min": 900.0,
                                      "rated_pressure_pa": 1_000_000.0}),
            ("generator", "permanent-magnet-generator", (.22, .22, .24), 78.0,
             "shaft-electrical-generator", {
                 "manufacturer_basis": "Mecc Alte Zanardi PM7G-80 Medio",
                 "rated_w": PMG_RATED_W,
                 "raw_ac_rated_w": PMG_RAW_AC_RATED_W,
                 "raw_winding": "three-phase-variable-frequency",
                 "installed_output": "external-power-converter",
                 "electrical_port_model": "pmg-three-phase-source",
                 "thermal_domain": "volume",
                 "rated_speed_range_rpm": [2500.0, 3600.0],
                 "full_load_efficiency": PMG_EFFICIENCY,
                 "specification_source": PMG_SOURCE,
                 "mass_basis": "modeled installed-package estimate; manufacturer mass not claimed"}),
            ("power_converter", "generator-power-converter", (.24, .12, .20), 34.0,
             "electrical-power-converter", {
                 "input": "three-phase-variable-frequency",
                 "continuous_output_w": PMG_RATED_W,
                 "output_services": [
                     spec[1].service.graph_attributes()
                     for spec in outlet_specs.values()],
                 "circuit_realization": "spectral-graph-multiport",
                 "electrical_port_model": "power-electronic-converter",
                 "thermal_domain": "volume"}),
            ("switchboard", "electrical-distribution-switchboard",
             (.22, .12, .18), 28.0, "electrical-distribution-panel", {
                 "distribution_sections": [
                     section.graph_attributes()
                     for section in distribution_sections],
                 "aggregate_source_limit_w": PMG_RATED_W,
                 "electrical_port_model": "breaker-switchboard",
                 "thermal_domain": "volume"}),
            ("outlet_dc", "electrical-receptacle-box", (.07, .06, .06), 3.0,
             "electrical-receptacle", {
                 **outlet_specs["outlet_dc"][1].graph_attributes(),
                 "electrical_port_model": "receptacle-contact-bank",
                 "thermal_domain": "volume"}),
            ("outlet_230", "electrical-receptacle-box", (.07, .06, .06), 3.0,
             "electrical-receptacle", {
                 **outlet_specs["outlet_230"][1].graph_attributes(),
                 "electrical_port_model": "receptacle-contact-bank",
                 "thermal_domain": "volume"}),
            ("outlet_400_3ph", "electrical-receptacle-box", (.07, .06, .06), 4.0,
             "electrical-receptacle", {
                 **outlet_specs["outlet_400_3ph"][1].graph_attributes(),
                 "electrical_port_model": "receptacle-contact-bank",
                 "thermal_domain": "volume"}),
            ("outlet_split", "electrical-receptacle-box", (.07, .06, .06), 3.0,
             "electrical-receptacle", {
                 **outlet_specs["outlet_split"][1].graph_attributes(),
                 "electrical_port_model": "receptacle-contact-bank",
                 "thermal_domain": "volume"}),
            ("battery", "battery-pack-48v", (.30, .16, .24), 94.0,
             "electrical-storage", {"nominal_voltage_v": 48.0,
                                    "capacity_ah": 200.0,
                                    "stored_energy_kwh": 9.6,
                                    "electrical_port_model": "battery-storage",
                                    "thermal_domain": "volume"}),
            ("refrigerant", "shaft-refrigerant-compressor", (.17, .16, .18), 31.0,
             "refrigerant-compressor", {"rated_shaft_w": 7500.0,
                                        "refrigerant": "r134a"}),
            ("coolant_reservoir", "engine-coolant-header-tank", (.20, .20, .18), 49.0,
             "platform-engine-coolant-reservoir", {
                 "capacity_l": 36.0, "working_fill_l": 30.0,
                 "fluid": "propylene-glycol-coolant", "potable": False,
                 "maximum_pressure_pa": 350_000.0}),
            ("coolant_pump", "variable-speed-electric-coolant-pump", (.16, .14, .15), 19.0,
             "platform-engine-coolant-assist-pump", {
                 "rated_flow_l_min": 160.0, "rated_head_pa": 220_000.0,
                 "rated_w": 1800.0, "supply_voltage_v": 48.0,
                 "electrical_port_model": "electric-motor-drive",
                 "thermal_domain": "volume",
                 "control": "HCU-coolant-manifold",
                 "role": "primary when engine has no pump; parallel assist otherwise"}),
        )
        for name, kind, ext, mass, role, attributes in specifications:
            q = locations[name]
            ports = []
            if name not in (
                    "battery", "coolant_reservoir", "coolant_pump",
                    "power_converter", "switchboard", "outlet_dc",
                    "outlet_230", "outlet_400_3ph", "outlet_split"):
                ports.append(_port(f"{ident(name)}.shaft_in", ident(name),
                                   "shaft-power-in", q, (-side, 0, 0), .025,
                                   connected_to=ident("pto")))
            if name == "hydraulic":
                ports += [_port(f"{ident(name)}.pressure", ident(name),
                                "hydraulic-pressure-out", q, (0, 0, 1), .012,
                                fluid="hydraulic-oil"),
                          _port(f"{ident(name)}.return", ident(name),
                                "hydraulic-return-in", q, (0, 0, -1), .016,
                                fluid="hydraulic-oil")]
            elif name == "air":
                ports.append(_port(f"{ident(name)}.delivery", ident(name),
                                   "compressed-air-out", q, (0, 0, -1), .012,
                                   fluid="air"))
            elif name == "generator":
                ports.append(_port(f"{ident(name)}.raw_ac_out", ident(name),
                                   "three-phase-winding-out", q, (0, 1, 0), .010,
                                   fluid="electricity",
                                   connected_to=f"{ident('power_converter')}.raw_ac_in"))
            elif name == "power_converter":
                ports += [
                    _port(f"{ident(name)}.raw_ac_in", ident(name),
                          "three-phase-winding-in", q, (0, -1, 0), .010,
                          fluid="electricity",
                          connected_to=f"{ident('generator')}.raw_ac_out"),
                    _port(f"{ident(name)}.distribution_out", ident(name),
                          "multi-service-electrical-out", q, (0, 1, 0), .012,
                          fluid="electricity",
                          connected_to=f"{ident('switchboard')}.supply"),
                ]
            elif name == "switchboard":
                ports.append(_port(
                    f"{ident(name)}.supply", ident(name),
                    "multi-service-electrical-in", q, (0, -1, 0), .012,
                    fluid="electricity",
                    connected_to=f"{ident('power_converter')}.distribution_out"))
            elif name.startswith("outlet_"):
                ports.append(_port(
                    f"{ident(name)}.receptacle", ident(name),
                    "electrical-receptacle", q, (side, 0, 0), .012,
                    fluid="electricity"))
            elif name == "battery":
                ports.append(_port(f"{ident(name)}.dc_bus", ident(name),
                                   "48v-dc-bidirectional", q, (0, 1, 0), .010,
                                   fluid="electricity",
                                   connected_to=f"{ident('power_converter')}.distribution_out"))
            elif name == "refrigerant":
                ports += [_port(f"{ident(name)}.suction", ident(name),
                                "refrigerant-suction-in", q, (0, 0, -1), .010,
                                fluid="r134a"),
                          _port(f"{ident(name)}.discharge", ident(name),
                                "refrigerant-discharge-out", q, (0, 0, 1), .008,
                                fluid="r134a")]
            elif name == "coolant_reservoir":
                ports += [_port(f"{ident(name)}.supply", ident(name),
                                "engine-coolant-supply-out", q, (0, 0, 1), .019,
                                fluid="propylene-glycol-coolant"),
                          _port(f"{ident(name)}.return", ident(name),
                                "engine-coolant-return-in", q, (0, 0, -1), .019,
                                fluid="propylene-glycol-coolant")]
            elif name == "coolant_pump":
                ports += [_port(f"{ident(name)}.suction", ident(name),
                                "engine-coolant-suction-in", q, (0, 0, -1), .019,
                                fluid="propylene-glycol-coolant"),
                          _port(f"{ident(name)}.delivery", ident(name),
                                "engine-coolant-pressure-out", q, (0, 0, 1), .019,
                                fluid="propylene-glycol-coolant"),
                          _port(f"{ident(name)}.dc_in", ident(name),
                                "48v-dc-in", q, (0, 1, 0), .008,
                                fluid="electricity",
                                connected_to=f"{ident('battery')}.dc_bus")]
            parts.append(MachinePart(ident(name), kind, tuple(q), ext,
                                     mass_kg=mass, material="steel-plate",
                                     part_role=role, ports=ports,
                                     thermal_capacity_j_k=mass * 500.0,
                                     attributes=attributes))
        lines = []
        for name in ("hydraulic", "air", "generator", "refrigerant"):
            lines.append(MachineLine(f"{prefix}.shaft.{name}", ident("pto"),
                                     ident(name), "shaft-service-drive", .018,
                                     "shaft", "hardened-steel"))
        raw_length = float(np.linalg.norm(
            locations["generator"] - locations["power_converter"]))
        raw_conductors = []
        for role in ("line-1", "line-2", "line-3", "protective-earth"):
            conductor = Conductor(
                f"{prefix}.generator_raw.{role}", awg=4,
                length_m=raw_length, both_directions=False)
            raw_conductors.append({
                "role": role, "awg": 4,
                "area_mm2": conductor.area_mm2,
                "resistance_ohm": conductor.resistance_ohm,
                "ampacity_a": conductor.ampacity_a,
                "material": "stranded-copper",
            })
        lines.append(MachineLine(
            f"{prefix}.generator_raw_to_converter",
            ident("generator"), ident("power_converter"),
            "insulated-copper-wire", .010, "generator-raw-ac", "copper",
            {"transport_domain": "electrical",
             "phase_arrangement": "three-phase-variable-frequency",
             "conductor_roles": [row["role"] for row in raw_conductors],
             "conductors": raw_conductors}))

        def cable_line(name, a, b, cable, circuit, breaker=None):
            attributes = cable.graph_attributes()
            if breaker is not None:
                attributes.update(breaker.graph_attributes())
            return MachineLine(
                f"{prefix}.{name}", ident(a), ident(b),
                "insulated-copper-wire", .010, circuit, "copper",
                attributes)

        # Each converted service remains a distinct conductor bundle even
        # though the runs share one structural conduit and enclosure.
        for name, (branch_cable, _outlet) in outlet_specs.items():
            panel_cable = BuildingCable(
                ident(f"cable.converter.{name}"), branch_cable.service,
                awg=branch_cable.awg,
                length_m=float(np.linalg.norm(
                    locations["power_converter"] - locations["switchboard"])),
                solid_copper=False, in_conduit=True,
                allowable_ampacity_a=branch_cable.ampacity_a)
            lines.append(cable_line(
                f"converter_to_switchboard.{name}", "power_converter",
                "switchboard", panel_cable, branch_cable.service.identity,
                _outlet.breaker))
            lines.append(cable_line(
                f"switchboard_to_{name}", "switchboard", name,
                branch_cable, branch_cable.service.identity,
                _outlet.breaker))
            lines.append(MachineLine(
                f"{prefix}.conduit.switchboard_to_{name}",
                ident("switchboard"), ident(name), "electrical-conduit",
                .018, "conduit-structure", "steel-pipe",
                {"load_bearing": True,
                 "contains_circuits": [branch_cable.service.identity]}))

        lines.append(MachineLine(
            f"{prefix}.conduit.converter_to_switchboard",
            ident("power_converter"), ident("switchboard"),
            "electrical-conduit", .025, "conduit-structure", "steel-pipe",
            {"load_bearing": True,
             "contains_circuits": [
                 cable.service.identity for cable, _ in outlet_specs.values()]}))

        battery_cable = BuildingCable(
            ident("cable.converter_battery"), services["48v-dc"],
            awg=0, length_m=float(np.linalg.norm(
                locations["power_converter"] - locations["battery"])),
            solid_copper=False, allowable_ampacity_a=150.0)
        coolant_cable = BuildingCable(
            ident("cable.battery_coolant_pump"), services["48v-dc"],
            awg=10, length_m=float(np.linalg.norm(
                locations["battery"] - locations["coolant_pump"])),
            solid_copper=False, allowable_ampacity_a=30.0)
        lines.append(cable_line(
            "converter_to_battery", "power_converter", "battery",
            battery_cable, services["48v-dc"].identity))
        lines.append(cable_line(
            "battery_to_coolant_pump", "battery", "coolant_pump",
            coolant_cable, services["48v-dc"].identity))
        lines.append(MachineLine(f"{prefix}.coolant_reservoir_to_pump",
                                 ident("coolant_reservoir"), ident("coolant_pump"),
                                 "coolant-line", .019, "engine-coolant",
                                 "reinforced-rubber"))
        return Machine(prefix, f"{self.engine_identity} powerplant basic",
                       power="engine-driven", medium="multi-service",
                       parts=parts, lines=lines,
                       note="replaceable engine envelope plus standard shaft utilities")


@dataclass
class PowerplantRuntime:
    """Simple operations of the non-engine shaft machines."""
    package: PowerplantBasic
    battery_soc: float = 0.80
    battery_capacity_ah: float = 200.0
    temperatures_k: dict[str, float] = field(default_factory=dict)
    refrigerant_loop: object = None

    def __post_init__(self) -> None:
        from refrigeration import CoolingLoad, RefrigerantLoop
        rated_omega = PMG_REFERENCE_RPM * 2.0 * math.pi / 60.0
        self.refrigerant_loop = RefrigerantLoop(
            displacement_w_per_rad_s=7_500.0 / rated_omega,
            max_shaft_w=7_500.0)
        self.refrigerant_loop.loads = [CoolingLoad(
            "site-potable-water-coil", evaporating_k=274.15,
            demand_w=12_000.0, priority=1.0)]

    def step(self, dt: float, shaft_rpm: float, *, electrical_load_w=0.0,
             hydraulic_command=0.0, pneumatic_demand_l_min=0.0,
             refrigeration_command=0.0, ambient_k=293.15,
             refrigeration_demand_w=12_000.0,
             hydraulic_demand_l_min=None,
             hydraulic_required_pressure_pa=0.0) -> dict:
        rpm = max(0.0, float(shaft_rpm))
        speed = min(1.0, rpm / PMG_REFERENCE_RPM)
        hydraulic_capacity_l_min = 48.0 * rpm / 1000.0
        requested_l_min = (hydraulic_capacity_l_min
                           * max(0.0, min(1.0, hydraulic_command))
                           if hydraulic_demand_l_min is None else
                           max(0.0, float(hydraulic_demand_l_min)))
        hydraulic_l_min = min(hydraulic_capacity_l_min, requested_l_min)
        hydraulic_pressure_pa = min(21_000_000.0,
                                    max(101_325.0, float(hydraulic_required_pressure_pa)))
        hydraulic_shaft_w = (hydraulic_pressure_pa * hydraulic_l_min
                             / 60_000.0 / 0.85)
        pneumatic_l_min = 900.0 * speed
        generator_capacity_w = PMG_RATED_W * speed * PMG_EFFICIENCY
        charge_request_w = (48.0 * self.battery_capacity_ah * 0.25
                            if self.battery_soc < 0.98 else 0.0)
        generator_w = min(generator_capacity_w,
                          max(0.0, electrical_load_w) + charge_request_w)
        refrigeration_command = max(0.0, min(1.0, refrigeration_command))
        refrigerant_load = self.refrigerant_loop.load("site-potable-water-coil")
        refrigerant_load.calling = refrigeration_command > 0.0
        refrigerant_load.demand_w = max(0.0, float(refrigeration_demand_w))
        self.refrigerant_loop.main_switch = refrigeration_command > 0.0
        self.refrigerant_loop.step(
            dt, rpm * 2.0 * math.pi / 60.0 * refrigeration_command,
            ambient_k)
        refrigerant_shaft_w = self.refrigerant_loop.shaft_w
        refrigerant_cooling_w = self.refrigerant_loop.delivered(
            "site-potable-water-coil")
        net_w = generator_w - max(0.0, electrical_load_w)
        self.battery_soc = max(0.0, min(1.0, self.battery_soc
            + net_w * dt / (48.0 * self.battery_capacity_ah * 3600.0)))
        losses = {
            "hydraulic": hydraulic_l_min / 48.0 * 1800.0,
            "air": pneumatic_l_min / 900.0 * 2600.0,
            "generator": generator_w * (1.0 / PMG_EFFICIENCY - 1.0),
            "battery": abs(net_w) * 0.015,
            "refrigerant": refrigerant_shaft_w * 0.18,
        }
        masses = {p.identity.rsplit(".", 1)[-1]: p.mass_kg
                  for p in self.package.machine().parts}
        for name, heat_w in losses.items():
            t = self.temperatures_k.get(name, ambient_k)
            t += (heat_w - (t - ambient_k) * 8.0) * dt / max(masses[name] * 500.0, 1.0)
            self.temperatures_k[name] = t
        total_shaft_load_w = (hydraulic_shaft_w + refrigerant_shaft_w
                              + generator_w / PMG_EFFICIENCY)
        return {"shaft_rpm": rpm, "hydraulic_flow_l_min": hydraulic_l_min,
                "hydraulic_capacity_l_min": hydraulic_capacity_l_min,
                "hydraulic_pressure_pa": hydraulic_pressure_pa,
                "hydraulic_shaft_load_w": hydraulic_shaft_w,
                "pneumatic_capacity_l_min": pneumatic_l_min,
                "pneumatic_served_l_min": min(pneumatic_l_min, max(0.0, pneumatic_demand_l_min)),
                "generator_w": generator_w,
                "generator_capacity_w": generator_capacity_w,
                "battery_soc": self.battery_soc,
                "refrigerant_shaft_w": refrigerant_shaft_w,
                "refrigerant_cooling_w": refrigerant_cooling_w,
                "refrigerant_suction_pa": self.refrigerant_loop.suction_pa,
                "refrigerant_discharge_pa": self.refrigerant_loop.discharge_pa,
                "total_shaft_load_w": total_shaft_load_w,
                "temperatures_k": dict(self.temperatures_k)}


def run_multifuel_pillar_lift(stand_spec, carried_kg: float, *,
                              dt: float = 0.10, timeout_s: float = 180.0,
                              throttle: float = 0.55) -> dict:
    """Run the four upright pillars from the real multifuel PTO pump.

    Corner braces are unloaded for this first phase; their fold actuators are
    the next configuration exchange.  The engine sees last tick's actual
    shaft power through its clutch/transmission load, preserving one outer dt.
    """
    from engines import get
    from engine_cycle_sim import EngineCycleSim
    import stand as stand_module

    engine = EngineCycleSim(get("ldt465-multifuel-deuce"))
    engine.start()
    engine.throttle = float(throttle)
    # Let the commanded diesel rack actually reach working speed before
    # connecting the service shaft.  Three seconds only brought this engine
    # to about 750 rpm; engaging there stalled a perfectly adequate engine
    # and looked like a pump-capacity failure.
    elapsed = 0.0
    spin_target_rpm = max(1_500.0, engine.engine.idle_rpm * 1.8)
    while elapsed < 12.0 and engine.state.rpm < spin_target_rpm:
        engine.step(dt)
        elapsed += dt
    engine.gear_index = len(engine.engine.transmission.gear_ratios)
    engine.clutch_frac = 0.0
    # Synchronise the existing gearbox/load shaft through its real clutch
    # before asking the pump for pressure and flow.
    clutch_elapsed = 0.0
    while clutch_elapsed < 1.5:
        engine.clutch_frac = min(1.0, clutch_elapsed / 1.0)
        engine.brake_load_nm = 0.0
        engine.step(dt)
        clutch_elapsed += dt
    engine.clutch_frac = 1.0

    package = PowerplantBasic("powerplant.port", engine.engine.identity,
                              (0.0, 0.0, 0.0), (.62, .48, .58))
    utility = PowerplantRuntime(package)
    legs = stand_module.outrigger_set(stand_spec, carried_kg)
    # The HCU's plant schedule.  Full spool would demand 222 L/min from
    # these four 140 mm cylinders and turn line loss into the dominant load;
    # quarter spool stays inside both the pump and prime mover envelope.
    commands = {name: (0.25 if name.startswith("inner.") else 0.0)
                for name in legs.legs}
    loads = {name: (carried_kg * 9.80665 / 4.0
                    if name.startswith("inner.") else 0.0)
             for name in legs.legs}
    previous_shaft_w = 0.0
    peak_shaft_w = 0.0
    minimum_rpm = float(engine.state.rpm)
    log = None
    lift_elapsed = 0.0
    while lift_elapsed < timeout_s:
        ratio = max(engine._current_gear_ratio(), 1e-9)
        transfer_omega = max(float(engine.state.rpm) / ratio
                             * 2.0 * math.pi / 60.0, 1e-6)
        engine.brake_load_nm = previous_shaft_w / transfer_omega
        engine.known_accessory_shaft_load_w = previous_shaft_w
        engine.step(dt)
        pto_rpm = max(0.0, float(engine.state.rpm) / ratio
                      * UTILITY_PTO_STEP_UP)
        reading = {}

        def supply(demand_l_min, required_pressure_pa):
            nonlocal reading
            reading = utility.step(
                dt, pto_rpm, electrical_load_w=2_000.0,
                hydraulic_demand_l_min=demand_l_min,
                hydraulic_required_pressure_pa=required_pressure_pa)
            return {"pressure_pa": reading["hydraulic_pressure_pa"],
                    "delivered_l_min": reading["hydraulic_flow_l_min"],
                    "shaft_load_w": reading["hydraulic_shaft_load_w"]}

        log = legs.step(dt, commands, supply=supply, loads_n=loads)
        previous_shaft_w = float(reading.get("total_shaft_load_w", 0.0))
        peak_shaft_w = max(peak_shaft_w, previous_shaft_w)
        minimum_rpm = min(minimum_rpm, float(engine.state.rpm))
        lift_elapsed += dt
        inner = [leg for name, leg in legs.legs.items()
                 if name.startswith("inner.")]
        if min(leg.extension_m / leg.cylinder.stroke_m for leg in inner) >= 1.0 - 1e-6:
            break
    inner_extension = min(leg.extension_m for name, leg in legs.legs.items()
                          if name.startswith("inner."))
    return {"completed": inner_extension >= stand_spec.trailer_clearance_m - 1e-4,
            "lift_elapsed_s": lift_elapsed,
            "inner_extension_m": inner_extension,
            "deck_height_above_ground_m": (stand_spec.lower_room_clear_height_m
                                             + inner_extension),
            "engine_rpm": float(engine.state.rpm),
            "minimum_engine_rpm": minimum_rpm,
            "peak_shaft_load_w": peak_shaft_w,
            "last_leg_state": log, "utility": utility,
            "engine": engine}


def emit_powerplant_basic(g, package: PowerplantBasic, bay_nodes: dict[str, str]) -> dict:
    """Put the package bodies on a beam-mounted skid and route its services."""
    machine = package.machine()
    document = machine.build_graph()
    centre = np.asarray(package.centre, float)
    skid = f"{package.identity}.skid"
    g.motion_group, g.assembly = "frame", package.identity
    g.node(skid, tuple(centre + np.array([0.0, -0.40, 0.0])),
           "load-bearing-structure", half_extent_m=(0.98, .055, .92),
           material="4130n", mass_kg=112.0, mass_in_total=True,
           part_role="powerplant-utility-skid", engine_envelope={
               "half_extent_m": list(package.engine_half_extent_m),
               "engine_identity": package.engine_identity})
    for tag, target in bay_nodes.items():
        g.edge(f"{skid}.mount.{tag}", skid, target,
               "engine-mount-isolator", radius=.040, alloy="4130n",
               beam_solvable=True, part_role="utility-skid-mount",
               port_role="structural-mount",
               load_path="utility-machines-through-skid-into-bay-frame")
    for node in document["nodes"]:
        attrs = {k: v for k, v in node.items()
                 if k not in ("identity", "kind", "reference_position",
                              "body_half_extent_m", "motion_group", "assembly")}
        g.node(node["identity"], node["reference_position"], node["kind"],
               half_extent_m=node["body_half_extent_m"],
               thermal_group=node["identity"], **attrs)
        # Each housing is bolted to the skid.  This is its structural
        # load path; the lines below deliberately carry no beam load.
        g.edge(f"{node['identity']}.seat", node["identity"], skid,
               "rigid-distance", radius=.030, alloy="4130n",
               beam_solvable=True, fastening="bolted",
               part_role="machine-foot",
               load_path="machine-housing-through-foot-into-utility-skid")
        if node["identity"].endswith(".battery"):
            g.edge(f"{node['identity']}.hold_down", node["identity"],
                   f"{package.identity}.pto", "rigid-distance", radius=.018,
                   alloy="4130n", beam_solvable=True, fastening="bolted",
                   part_role="battery-hold-down",
                   load_path="battery-tray-held-against-shock")
    for edge in document["edges"]:
        g.edge(edge["identity"], edge["a"], edge["b"], edge["constraint"],
               radius=edge["radius"], palette="service-line",
               beam_solvable=False, routed=True,
               circuit_identity=edge["circuit_identity"],
               material=edge["material"], part_role="routed-service")
    g.edge(f"{package.identity}.engine_pto", f"engine.{package.identity.rsplit('.', 1)[-1]}",
           f"{package.identity}.pto", "shaft-service-drive", radius=.024,
           palette="service-line", beam_solvable=False, routed=True,
           source_port="powertrain.transfer_case.output",
           sink_port=f"{package.identity}.pto.shaft_in",
           circuit_identity=f"{package.identity}.shaft")
    side = package.identity.rsplit(".", 1)[-1]
    engine_node = f"engine.{side}"
    g.edge(f"{package.identity}.coolant_pump_to_engine",
           f"{package.identity}.coolant_pump", engine_node,
           "coolant-line", radius=.019, palette="service-line",
           beam_solvable=False, routed=True,
           fluid="propylene-glycol-coolant",
           circuit_identity=f"{package.identity}.engine-coolant",
           source_port=f"{package.identity}.coolant_pump.delivery",
           sink_port=f"{engine_node}.external_coolant_in",
           isolation="engine coolant only; never potable water")
    g.edge(f"{package.identity}.engine_to_coolant_reservoir",
           engine_node, f"{package.identity}.coolant_reservoir",
           "coolant-line", radius=.019, palette="service-line",
           beam_solvable=False, routed=True,
           fluid="propylene-glycol-coolant",
           circuit_identity=f"{package.identity}.engine-coolant",
           source_port=f"{engine_node}.external_coolant_out",
           sink_port=f"{package.identity}.coolant_reservoir.return",
           isolation="engine coolant only; never potable water")
    return {"package": package, "machine": machine, "skid": skid,
            "parts": tuple(n["identity"] for n in document["nodes"])}


def shared_storage_ballast(centre_y: float = -0.70, radius_m: float = 1.12) -> Machine:
    """Paired central reservoirs serving both packages without moving CG.

    Every medium is split into diametrically opposed vessels.  Thus the
    contents remain ballast around the access trunk instead of producing a
    lateral trim change merely because oil is denser than compressed air.
    The default vertical station leaves a service gap below the drum rather
    than occupying the drum's swept/service envelope.
    """
    services = (
        ("hydraulic", "hydraulic-storage-tank", 90.0, 92.0, "hydraulic-oil", .27),
        ("air", "pneumatic-receiver", 180.0, 58.0, "compressed-air", .25),
        ("refrigerant", "refrigerant-receiver", 24.0, 27.0, "r134a", .20),
    )
    parts = []
    angles = (22.5, 67.5, 112.5, 157.5)
    for (name, kind, volume_l, pair_mass, fluid, extent), angle in zip(services, angles):
        for suffix, degrees in (("a", angle), ("b", angle + 180.0)):
            a = math.radians(degrees)
            q = np.array([radius_m * math.cos(a), centre_y,
                          radius_m * math.sin(a)])
            ident = f"station.storage.{name}.{suffix}"
            ports = []
            for side in ("port", "starboard"):
                ports.append(_port(f"{ident}.to_{side}", ident,
                                   f"{name}-supply-out", q,
                                   (math.cos(a), 0, math.sin(a)), .012,
                                   fluid=fluid,
                                   connected_to=f"powerplant.{side}.{name}"))
            parts.append(MachinePart(
                ident, kind, tuple(q), (extent, .28, extent),
                mass_kg=pair_mass / 2.0, material="steel-plate",
                fluid=fluid, fluid_volume_l=volume_l / 2.0,
                part_role=f"shared-{name}-storage-ballast", ports=ports,
                thermal_capacity_j_k=pair_mass / 2.0 * 700.0))
    return Machine("station.storage", "paired central service storage",
                   power="engine-driven", medium="multi-service", parts=parts,
                   note="diametrically paired around the access trunk; both bays share every medium")


def emit_shared_storage(g, structural_targets: list[str]) -> dict:
    """Seat shared tanks on one ring cradle around the access opening."""
    machine = shared_storage_ballast()
    document = machine.build_graph()
    position = {n["identity"]: np.asarray(n["reference_position"], float)
                for n in g.nodes}
    g.motion_group, g.assembly = "frame", "central-storage-ballast"
    cradles = []
    for i, node in enumerate(document["nodes"]):
        vessel_at = np.asarray(node["reference_position"], float)
        cradle = f"station.storage.cradle.{i}"
        cradle_at = vessel_at + np.array([0.0, -0.36, 0.0])
        g.node(cradle, cradle_at, "chassis-load-node",
               half_extent_m=(.08, .06, .08), material="4130n",
               mass_kg=9.0, mass_in_total=True,
               part_role="central-storage-cradle-node")
        cradles.append(cradle)
    for i, cradle in enumerate(cradles):
        g.edge(f"station.storage.cradle.ring.{i}", cradle,
               cradles[(i + 1) % len(cradles)], "rigid-distance",
               radius=.038, alloy="4130n", beam_solvable=True,
               part_role="storage-cradle-ring",
               load_path="closed-ballast-cradle-around-access-opening")
        here = np.asarray(next(n["reference_position"] for n in g.nodes
                               if n["identity"] == cradle), float)
        target = min(structural_targets,
                     key=lambda k: float(np.linalg.norm(position[k] - here)))
        g.edge(f"{cradle}.spoke", cradle, target, "rigid-distance",
               radius=.032, alloy="4130n", beam_solvable=True,
               part_role="storage-cradle-spoke",
               load_path="ballast-cradle-into-access-trunk-and-work-frame")
    for i, node in enumerate(document["nodes"]):
        attrs = {k: v for k, v in node.items()
                 if k not in ("identity", "kind", "reference_position",
                              "body_half_extent_m", "motion_group", "assembly")}
        g.node(node["identity"], node["reference_position"], node["kind"],
               half_extent_m=node["body_half_extent_m"],
               thermal_group=node["identity"], **attrs)
        g.edge(f"{node['identity']}.seat", node["identity"], cradles[i],
               "engine-mount-isolator", radius=.032, alloy="4130n",
               beam_solvable=True, part_role="storage-vessel-mount",
               load_path="vessel-down-short-seat-into-ballast-cradle")
    return {"machine": machine,
            "parts": tuple(n["identity"] for n in document["nodes"]),
            "cradle": tuple(cradles)}


def emit_post_fuel_barrels(g, site: dict) -> dict:
    """Four 50-gallon drums, drop-in pumps, typed manifold and FCU."""
    from fabrication import WrapProfile, emit_shaped_iron_strap

    node_by_id = {n["identity"]: n for n in g.nodes}
    fuel_by_side = {"port": "ultra-low-sulfur-diesel",
                    "starboard": "jet-a-kerosene"}
    manifolds = {fuel: f"station.fuel.manifold.{fuel}"
                 for fuel in fuel_by_side.values()}
    fcu_node = "station.fuel.fcu"
    g.motion_group, g.assembly = "frame", "barrel-fuel-system"
    for x, (fuel, manifold) in zip((-.32, .32), manifolds.items()):
        g.node(manifold, (x, site["floor_surface_y_m"] + 0.48, -1.10),
               "fuel-type-manifold", material="hardened-steel", mass_kg=12.0,
               mass_in_total=True, part_role="common-fuel-type-manifold",
               accepted_fuel=fuel, crossfeed=False,
               half_extent_m=(0.22, 0.12, 0.10))
    g.node(fcu_node, (0.0, site["floor_surface_y_m"] + 0.76, -1.10),
           "fuel-control-unit", material="steel-plate", mass_kg=14.0,
           mass_in_total=True, part_role="fuel-control-unit",
           controls="all-compatible-drop-in-barrel-pumps",
           half_extent_m=(0.22, 0.10, 0.18))
    floor_nodes = list(site["floor_plate"]["nodes"].values())
    all_nodes = {n["identity"]: n for n in g.nodes}
    for ident in (*manifolds.values(), fcu_node):
        at = np.asarray(all_nodes[ident]["reference_position"], float)
        targets = sorted(floor_nodes, key=lambda name: float(np.linalg.norm(
            np.asarray(all_nodes[name]["reference_position"], float) - at)))[:3]
        for i, target in enumerate(targets):
            g.edge(f"{ident}.mount.{i}", ident, target,
                   "engine-mount-isolator", radius=.026, alloy="4130n",
                   part_role="fuel-system-floor-mount")

    by_side = {}
    pumps = []
    from engines import FUEL_DENSITY_KG_M3
    for side in ("port", "starboard"):
        fuel = fuel_by_side[side]
        fuel_density = FUEL_DENSITY_KG_M3[fuel]
        manifold = manifolds[fuel]
        rear_posts = (site["lower_bays"][side]["ir"],
                      site["lower_bays"][side]["or"])
        barrels = []
        side_pumps = []
        for i, post in enumerate(rear_posts):
            post_node = next(n for n in g.nodes if n["identity"] == post)
            pp = np.asarray(post_node["reference_position"], float)
            post_half = np.asarray(post_node["body_half_extent_m"], float)
            # ``ir/or`` are the negative-Z rear row.  The old positive-Z
            # offset put the drums on the room side of those posts -- inside
            # the occupied work area.  A back-post rack goes farther negative
            # in Z, entirely outside the post's rear face.
            centre = pp + np.asarray((0.0, FUEL_DRUM_HEIGHT_M / 2.0 + .08,
                                      -(FUEL_DRUM_RADIUS_M + post_half[2] + .08)))
            barrel = f"station.fuel.{side}.barrel.{i}"
            capacity_kg = FIFTY_US_GALLONS_L / 1000.0 * fuel_density
            g.node(barrel, centre, "fuel-storage-vessel",
                   material="steel-plate", mass_in_total=True,
                   mass_kg=FUEL_DRUM_TARE_KG + capacity_kg,
                   tare_mass_kg=FUEL_DRUM_TARE_KG, capacity_kg=capacity_kg,
                   fluid_volume_l=FIFTY_US_GALLONS_L, fluid=fuel,
                   initial_fill_frac=1.0, depletable=True,
                   accepts_drop_in_pump=True,
                   part_role="post-mounted-50-gallon-fuel-drum",
                   location_role="exterior-back-post-rack",
                   shape="drum",
                   drum_axis=(0.0, 1.0, 0.0),
                   drum_radius_m=FUEL_DRUM_RADIUS_M,
                   drum_length_m=FUEL_DRUM_HEIGHT_M,
                   half_extent_m=(FUEL_DRUM_RADIUS_M,
                                  FUEL_DRUM_HEIGHT_M / 2.0,
                                  FUEL_DRUM_RADIUS_M))
            barrels.append(barrel)
            for strap_i, y_offset in enumerate((-.24, .24)):
                emit_shaped_iron_strap(
                    g, f"{barrel}.strap.{strap_i}",
                    (WrapProfile(post, "box", (pp[0], pp[2]),
                                 half_extent_m=(post_half[0], post_half[2])),
                     WrapProfile(barrel, "circle", (centre[0], centre[2]),
                                 radius_m=FUEL_DRUM_RADIUS_M)),
                    station_m=float(centre[1] + y_offset), plane_axes=(0, 2),
                    width_m=.050, thickness_m=.006, clearance_m=.002,
                    material="4130n", bolt="M12", bolt_class="10.9",
                    set_torque_nm=95.0)
            pump_id = f"{barrel}.cap_pump"
            pump = DropInBarrelFuelPump(pump_id, barrel, fuel)
            pumps.append(pump); side_pumps.append(pump_id)
            cap = centre + np.asarray((0.0, FUEL_DRUM_HEIGHT_M / 2.0 + .07, 0.0))
            g.node(pump_id, cap, "drop-in-barrel-cap-fuel-pump",
                   material="hardened-steel", mass_kg=7.5, mass_in_total=True,
                   solver_condensed_into=barrel, solver_condensed_mass=True,
                   part_role="auto-off-drop-in-barrel-fuel-pump",
                   fuel=fuel, rated_flow_kg_s=pump.rated_flow_kg_s,
                   rated_pressure_pa=pump.rated_pressure_pa,
                   dry_run_cutoff_frac=pump.dry_run_cutoff_frac,
                   check_valve_cracking_pa=pump.check_valve_cracking_pa,
                   auto_off=("empty", "dry-run", "FCU-disabled"),
                   quick_disconnect=True, half_extent_m=(.11, .07, .11))
            g.edge(f"{pump_id}.pickup", barrel, pump_id, "fuel-line",
                   radius=.009, routed=True, beam_solvable=False,
                   fluid=fuel, circuit_identity=f"station.fuel.{fuel}",
                   fluid_route_class="rigid-orthogonal",
                   obstacle_aware_route=True,
                   pickup="drop-in-to-bottom-with-water-separator")
            g.edge(f"{pump_id}.to_manifold", pump_id, manifold, "fuel-line",
                   radius=.009, routed=True, beam_solvable=False,
                   fluid=fuel, circuit_identity=f"station.fuel.{fuel}",
                   fluid_route_class="flexible-hose",
                   hose_slack_fraction=.12,
                   flow_capacity_kg_s=pump.rated_flow_kg_s,
                   check_valve=True, controlled_by=fcu_node)
        by_side[side] = {"barrels": tuple(barrels), "pumps": tuple(side_pumps),
                         "fuel": fuel, "manifold": manifold}
        g.edge(f"{manifold}.to_{side}", manifold,
               f"engine.{side}.external_fuel_in",
               "fuel-line", radius=.010, routed=True, beam_solvable=False,
               fluid=fuel, circuit_identity=f"station.fuel.{fuel}",
               fluid_route_class="flexible-hose", hose_slack_fraction=.10,
               source_port=f"{manifold}.{side}_out",
               sink_port=f"engine.{side}.external_fuel_in",
               controlled_by=fcu_node)
    runtimes = {fuel: FuelControlUnit(
        fuel, tuple(p for p in pumps if p.fuel == fuel))
        for fuel in manifolds}
    return {"manifolds": manifolds, "fcu_node": fcu_node,
            "runtimes": runtimes,
            "by_side": by_side}


def route_shared_storage(g, packages: dict, storage: dict, *,
                         metaconduits: dict | None = None) -> None:
    """Connect stores through each bay's removable multi-circuit trunk."""
    constraints = {"hydraulic": "pressure-rated-hydraulic-line",
                   "air": "pressure-rated-air-line",
                   "refrigerant": "refrigerant-line"}
    sinks = {"hydraulic": "hydraulic",
             "air": "air", "refrigerant": "refrigerant"}
    channels = {"hydraulic": "hydraulic-pressure",
                "air": "compressed-air", "refrigerant": "refrigerant-suction"}
    for side in ("port", "starboard"):
        prefix = packages[side]["package"].identity
        for medium, constraint in constraints.items():
            sink = f"{prefix}.{sinks[medium]}"
            radius = .010 if medium != "air" else .014
            conduit = None if metaconduits is None else metaconduits[side]
            station_end = conduit["station_end"] if conduit else sink
            for half in ("a", "b"):
                source = f"station.storage.{medium}.{half}"
                g.edge(f"station.service.{medium}.{half}.to_{side}", source,
                       station_end, constraint, radius=radius,
                       palette="service-line", beam_solvable=False, routed=True,
                       circuit_identity=f"station.shared.{medium}",
                       source_port=f"{source}.to_{side}",
                       sink_port=(f"{sink}.{medium}_in" if conduit is None else
                                  f"{station_end}.{channels[medium]}"),
                       routed_through_metaconduit=(conduit["jacket"]
                                                   if conduit else None))
            if conduit is not None:
                channel_identity = (f"station.metaconduit.{side}.channel."
                                    f"{channels[medium]}")
                channel = next(edge for edge in g.edges
                               if edge["identity"] == channel_identity)
                channel.update(constraint=constraint,
                               circuit_identity=f"station.shared.{medium}")
                g.edge(f"station.service.{medium}.metaconduit_to_{side}",
                       conduit["engine_end"], sink, constraint, radius=radius,
                       palette="service-line", beam_solvable=False, routed=True,
                       circuit_identity=f"station.shared.{medium}",
                       source_port=f"{conduit['engine_end']}.{channels[medium]}",
                       sink_port=f"{sink}.{medium}_in",
                       routed_through_metaconduit=conduit["jacket"])


# ---------------------------------------------------------------------
# ON-SITE FLUID RECYCLING
# ---------------------------------------------------------------------
#
# A station that runs engines makes dirty fluid, and hauling it away is
# the expensive answer. The cheap one is a separator bank: drain the
# sump into a dirty tank, run it through a purifier, and put it back.
# That is what every real installation with more than one engine does,
# and it is why a ship can run one oil charge for months.
#
# THREE SEPARATE MACHINES, NOT ONE WITH A VALVE. Oil, fuel and coolant
# each get their own unit, because cross-contaminating any two of them
# is worse than never cleaning either: a litre of coolant in the oil
# emulsifies the whole charge, and a litre of oil in the fuel fouls
# every injector downstream. Real installations keep them physically
# apart for exactly that reason, and so does this.
#
# All three are ELECTRIC. That is not incidental -- a separator bowl
# holds megajoules at speed and takes minutes to run up (see
# centrifuges.spin_up_time_s), so it wants a motor it can leave running,
# not something started and stopped with a shift. The station's own
# generation has to carry that: the bank's standing load is small, but
# starting three of them together is not.

#: The bank's units, each sized for what it cleans. Bowl dimensions and
#: motor ratings are the real range for a small industrial separator.
RECYCLING_UNITS = (
    # (service, fluid, bowl radius, bowl height, rpm, motor kW, tank litres)
    ("oil", "engine-oil", 0.150, 0.180, 8000.0, 5.5, 400.0),
    ("fuel", "diesel", 0.130, 0.160, 9000.0, 4.0, 600.0),
    ("coolant", "coolant-water-glycol", 0.120, 0.140, 7000.0, 3.0, 300.0),
)


def fluid_recycling_bank(centre=(0.0, -0.35, 0.0), spacing_m: float = 0.95) -> Machine:
    """The station's separator bank: three electric purifiers, each with
    a dirty tank in front of it and a clean tank behind, sharing one
    sludge receiver.

    The sludge tank is the part worth having: it is where everything
    taken out of every fluid ends up, it has a real capacity, and when
    it is full the bowls keep ejecting into it anyway (see
    centrifuges.SludgeReceiver) -- so a station that never empties it
    ends up wearing what it was trying to recycle."""
    import centrifuges as cf
    cx, cy, cz = centre
    parts: list = []
    lines: list = []
    for i, (service, fluid, r_bowl, h_bowl, rpm, kw, tank_l) in enumerate(RECYCLING_UNITS):
        x = cx + (i - 1) * spacing_m
        unit = cf.Centrifuge(kind="disc-stack-self-cleaning", bowl_radius_m=r_bowl,
                             inner_radius_m=r_bowl * 0.40, bowl_height_m=h_bowl,
                             speed_rpm=rpm, drive="electric-belt", motor_kw=kw)
        sep_id = f"station.recycle.{service}.separator"
        # the separator itself: a real rotating mass, declared as one
        parts.append(MachinePart(
            sep_id, "centrifugal-separator", (x, cy, cz),
            (r_bowl * 1.5, h_bowl * 1.6, r_bowl * 1.5),
            mass_kg=cf._bowl_inertia(unit) / max(1e-6, 0.6 * r_bowl ** 2) + 40.0,
            material="steel-plate", fluid=fluid, fluid_volume_l=r_bowl * 40.0,
            part_role=f"{service}-separator",
            ports=[_port(f"{sep_id}.feed", sep_id, f"{service}-dirty-in",
                         (x - 0.25, cy, cz), (-1, 0, 0), 0.012, fluid=fluid),
                   _port(f"{sep_id}.clean", sep_id, f"{service}-clean-out",
                         (x + 0.25, cy, cz), (1, 0, 0), 0.012, fluid=fluid),
                   _port(f"{sep_id}.eject", sep_id, "sludge-discharge",
                         (x, cy - 0.20, cz), (0, -1, 0), 0.018, fluid="sludge",
                         connected_to="station.recycle.sludge_tank")],
            thermal_capacity_j_k=120.0 * 500.0))
        for side, sign, role in (("dirty", -1.0, "waste"), ("clean", 1.0, "service")):
            t_id = f"station.recycle.{service}.{side}_tank"
            parts.append(MachinePart(
                t_id, "storage-tank", (x, cy - 0.55, cz + sign * 0.55),
                (0.32, 0.30, 0.32), mass_kg=tank_l * 0.12, material="steel-plate",
                fluid=fluid, fluid_volume_l=tank_l,
                part_role=f"{service}-{role}-tank",
                thermal_capacity_j_k=tank_l * 0.12 * 500.0))
            lines.append(MachineLine(
                f"station.recycle.{service}.{side}_line",
                t_id if side == "dirty" else sep_id,
                sep_id if side == "dirty" else t_id,
                constraint="oil-line" if service == "oil" else (
                    "fuel-supply-line" if service == "fuel" else "coolant-line"),
                radius=0.012, circuit_identity=f"recycle-{service}"))
    # one sludge receiver for the whole bank: everything taken out of
    # every fluid ends up in here, and it is a real vessel with a limit
    parts.append(MachinePart(
        "station.recycle.sludge_tank", "sludge-receiver",
        (cx, cy - 0.95, cz), (0.45, 0.35, 0.45), mass_kg=90.0,
        material="steel-plate", fluid="sludge", fluid_volume_l=500.0,
        capacity_kg=550.0, part_role="sludge-receiver",
        thermal_capacity_j_k=90.0 * 500.0))
    return Machine(
        "station.recycle", "on-site fluid recycling bank",
        power="shore-supplied", medium="multi-service", parts=parts, lines=lines,
        note="three electric separators kept physically apart -- oil, fuel and coolant "
             "never share a bowl -- discharging to one sludge receiver")


# ---------------------------------------------------------------------
# THE PLANT SHELTER, AND HOW FUEL GETS TO IT
# ---------------------------------------------------------------------
#
# The separators and the dehydrator want to be out of the weather and
# near each other, and they do not want a building. What they get is
# what a real field installation gets: a frame with a canvas over it.
#
# CANVAS IS NOT A WALL and that is the whole point of choosing it. It
# keeps rain off a bowl that is open for a strip-down and keeps the
# light off at night; it stops nothing that is shot at it, it tears
# rather than holes, and it burns. A steel room would be a different
# decision with different consequences, and pretending a tent is one is
# how an installation ends up feeling safer than it is.
#
# FUEL: PIPED OR CARRIED, and real practice is both. Bulk goes down a
# hose line -- that is what a tactical petroleum system IS, six-inch
# hose over ground with pump stations, because hand-moving bulk fuel
# destroys throughput. The last hundred metres is drums, because a line
# is a fixed target that needs pumps and a crew and cannot follow
# anything around. The deciding factors are duration, throughput and
# vulnerability, so a semi-permanent plant like this one is piped, with
# drum handling at the margins and as the fallback when the line is cut.
#
# Which makes the line worth cutting, and makes the fallback a real
# labour problem: a 205 litre drum is about 180 kg full, and somebody
# has to move it.

DRUM_CAPACITY_L = 205.0
DRUM_FULL_MASS_KG = 180.0
#: Minutes for one crew member to move and decant one drum. Portering is
#: the expensive answer and the numbers should say so.
DRUM_PORTER_MINUTES = 12.0


def plant_shelter(centre=(0.0, -0.35, 0.0), span_m: float = 3.4,
                  depth_m: float = 2.6, height_m: float = 2.3) -> Machine:
    """A frame-and-canvas enclosure over the fluid plant.

    Steel tube frame, canvas panels, open at one end for access. The
    frame is structure and the canvas is weather -- declared as
    different materials because they behave completely differently when
    anything happens to them."""
    cx, cy, cz = centre
    parts: list = []
    # the frame: four legs and a ridge, in tube
    for i, (dx, dz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        parts.append(MachinePart(
            f"station.shelter.leg_{i + 1}", "frame-tube",
            (cx + dx * span_m / 2.0, cy + height_m / 2.0, cz + dz * depth_m / 2.0),
            (0.03, height_m / 2.0, 0.03), mass_kg=11.0, material="steel-tube",
            part_role="shelter-frame", thermal_capacity_j_k=11.0 * 500.0))
    parts.append(MachinePart(
        "station.shelter.ridge", "frame-tube", (cx, cy + height_m, cz),
        (span_m / 2.0, 0.035, 0.035), mass_kg=16.0, material="steel-tube",
        part_role="shelter-frame", thermal_capacity_j_k=16.0 * 500.0))
    # the canvas: a roof and three sides, the fourth left open to work
    panels = (("roof", (cx, cy + height_m, cz), (span_m / 2.0, 0.002, depth_m / 2.0)),
              ("side_a", (cx - span_m / 2.0, cy + height_m / 2.0, cz),
               (0.002, height_m / 2.0, depth_m / 2.0)),
              ("side_b", (cx + span_m / 2.0, cy + height_m / 2.0, cz),
               (0.002, height_m / 2.0, depth_m / 2.0)),
              ("back", (cx, cy + height_m / 2.0, cz - depth_m / 2.0),
               (span_m / 2.0, height_m / 2.0, 0.002)))
    for name, pos, half in panels:
        area = 4.0 * half[0] * half[1] if name != "roof" else 4.0 * half[0] * half[2]
        parts.append(MachinePart(
            f"station.shelter.{name}", "canvas-panel", pos, half,
            mass_kg=max(2.0, area * 0.6), material="canvas",
            part_role="shelter-canvas", thermal_capacity_j_k=area * 0.6 * 1400.0))
    return Machine(
        "station.shelter", "frame-and-canvas plant shelter",
        power="none", medium="structure", parts=parts,
        note="weather cover over the fluid plant; canvas stops rain and nothing else, "
             "and the open end is the working face")


def fuel_supply_route(piped: bool = True, distance_m: float = 400.0,
                      demand_l_per_day: float = 1800.0) -> dict:
    """How fuel actually reaches this plant, and what it costs to do it.

    Returns both answers so a caller can compare them, because the real
    decision is a comparison: a hose line is laid once and then costs
    almost nothing per litre, and a drum party costs the same labour
    every single day. The line is also one cut from nothing, which is
    why the drum figure is worth keeping in front of you."""
    drums_per_day = demand_l_per_day / DRUM_CAPACITY_L
    porter_minutes = drums_per_day * DRUM_PORTER_MINUTES * max(1.0, distance_m / 100.0)
    return {
        "mode": "piped" if piped else "portered",
        "distance_m": distance_m,
        "demand_l_per_day": demand_l_per_day,
        "drums_per_day": drums_per_day,
        # WHAT IT COSTS PER DAY, and what it would cost if the line went.
        # A piped route costs a pump and nothing else per litre; the
        # portering figure is what you fall back TO, not what you pay
        # while the line is up. Reporting the same number for both made
        # the comparison say the line was worthless.
        "porter_crew_hours_per_day": 0.0 if piped else porter_minutes / 60.0,
        "fallback_crew_hours_per_day": porter_minutes / 60.0,
        # laying it is a one-off; after that a pump does the work
        "line_setup_crew_hours": distance_m / 100.0 * 1.5 if piped else 0.0,
        "pump_kw": 4.0 if piped else 0.0,
        "single_point_of_failure": piped,
        "note": ("a hose line: cheap per litre once laid, and one cut from nothing"
                 if piped else
                 "drums by hand: survivable, dispersed, and expensive every single day"),
    }


# ---------------------------------------------------------------------
# HOSE, SPIGOT AND DRUMS
# ---------------------------------------------------------------------
#
# Real military bulk fuel is collapsible hose in sections, coupled with
# cam-lock or grooved fittings and laid over the ground -- that is what
# a tactical petroleum system is. The sizes below are the real ones: a
# six-inch main for bulk, four-inch for distribution around a site, and
# two-inch at the dispensing end where somebody is holding it.
#
# THE SECTIONS ARE THE POINT. A hose run is not one object, it is a
# string of sections that each have to be carried out and coupled, and
# a six-inch section full of nothing is already a two-man lift. That
# makes laying a line a real job of work with a real duration -- which
# is exactly the sort of thing a crew that has got around to it can be
# sent to do, and exactly what has to be redone when somebody cuts it.
#
# A run that is not fully laid still carries fuel through the sections
# that ARE coupled, up to wherever the string stops. It just does not
# reach the far end yet.

@dataclass(frozen=True)
class HoseSpec:
    key: str
    label: str
    bore_mm: float
    section_m: float
    section_mass_kg: float
    working_bar: float
    crew: int
    minutes_per_section: float
    why: str


MILITARY_HOSE: dict[str, HoseSpec] = {
    "bulk-6in": HoseSpec(
        "bulk-6in", "6 inch collapsible bulk hose", 152.0, 15.0, 62.0, 10.0, 2, 10.0,
        why="the size a tactical petroleum line is actually run in; a section is a "
            "two-man lift empty and immovable full, which is why it is laid once"),
    "distribution-4in": HoseSpec(
        "distribution-4in", "4 inch distribution hose", 102.0, 15.0, 28.0, 10.0, 1, 6.0,
        why="around a site, between a main and the equipment that uses it"),
    "dispensing-2in": HoseSpec(
        "dispensing-2in", "2 inch dispensing hose", 51.0, 10.0, 9.0, 14.0, 1, 3.0,
        why="the end somebody holds: short, light, and the only one moved often"),
}


@dataclass
class HoseRun:
    """A string of hose sections between two points, part-laid or whole."""
    identity: str
    spec: str
    sections_required: int
    sections_laid: int = 0
    from_part: str = ""
    to_part: str = ""
    position: tuple = (0.0, 0.0, 0.0)
    cut_sections: int = 0

    @property
    def hose(self) -> HoseSpec:
        return MILITARY_HOSE[self.spec]

    @property
    def length_m(self) -> float:
        return self.sections_required * self.hose.section_m

    @property
    def complete(self) -> bool:
        return self.sections_laid >= self.sections_required and self.cut_sections == 0

    @property
    def reach_m(self) -> float:
        """How far the fuel actually gets. A part-laid run delivers to
        wherever the string stops, and a cut one stops at the cut."""
        good = self.sections_laid - self.cut_sections
        return max(0, good) * self.hose.section_m

    @property
    def outstanding_sections(self) -> int:
        return max(0, self.sections_required - self.sections_laid) + self.cut_sections

    def need(self):
        """What laying or repairing this run would take, as a crew need."""
        import servicing as sv
        n = self.outstanding_sections
        if n <= 0:
            return None
        h = self.hose
        return sv.Need(
            identity=self.identity, want="hose-section", position=tuple(self.position),
            quantity=float(n), unit="sections", matches=h.key,
            urgency=1.0 + n / max(1, self.sections_required),
            minutes=n * h.minutes_per_section, skill="operator",
            label=f"{self.identity} ({h.label})",
            why=("a cut line is a line that stops at the cut" if self.cut_sections
                 else "an unlaid run delivers only as far as it has been coupled"))


def fuel_hose_set(centre=(0.0, -0.35, 0.0), main_distance_m: float = 400.0) -> tuple:
    """The station's fuel hose, spigot and drum stock.

    Returns (machine, runs): the hardware, and the hose runs as live
    objects a crew can work on. The main is deliberately NOT laid to
    begin with -- somebody has to go and do it, and until they have,
    the plant is on drums."""
    cx, cy, cz = centre
    parts: list = []
    # the spigot: where a hose is coupled and a drum is filled
    parts.append(MachinePart(
        "station.fuel.spigot", "fuel-manifold", (cx + 1.9, cy + 0.4, cz),
        (0.18, 0.30, 0.18), mass_kg=26.0, material="steel-plate",
        fluid="diesel", fluid_volume_l=8.0, part_role="dispensing-spigot",
        ports=[_port("station.fuel.spigot.inlet", "station.fuel.spigot",
                     "fuel-bulk-in", (cx + 1.9, cy + 0.2, cz - 0.2), (0, 0, -1),
                     0.076, fluid="diesel"),
               _port("station.fuel.spigot.outlet", "station.fuel.spigot",
                     "fuel-dispense-out", (cx + 1.9, cy + 0.55, cz), (1, 0, 0),
                     0.025, fluid="diesel")],
        thermal_capacity_j_k=26.0 * 500.0))
    # drum stock on a rack beside it
    for i in range(6):
        row, col = divmod(i, 3)
        parts.append(MachinePart(
            f"station.fuel.drum_{i + 1}", "fuel-drum",
            (cx + 2.5 + col * 0.62, cy + 0.44, cz - 0.7 + row * 0.62),
            (0.29, 0.44, 0.29),
            mass_kg=DRUM_FULL_MASS_KG, material="steel-plate",
            fluid="diesel", fluid_volume_l=DRUM_CAPACITY_L,
            capacity_kg=DRUM_FULL_MASS_KG, part_role="fuel-drum",
            thermal_capacity_j_k=22.0 * 500.0))
    machine = Machine(
        "station.fuel", "fuel spigot, drum stock and hose", power="none",
        medium="fuel", parts=parts,
        note="the main is coupled at the spigot; drums are the fallback and the margin")
    spec = MILITARY_HOSE["bulk-6in"]
    runs = [
        HoseRun("station.fuel.main_line", "bulk-6in",
                sections_required=max(1, int(main_distance_m / spec.section_m)),
                sections_laid=0, from_part="depot", to_part="station.fuel.spigot",
                position=(cx + 1.9, cy, cz - 0.4)),
        HoseRun("station.fuel.plant_feed", "distribution-4in", sections_required=3,
                sections_laid=3, from_part="station.fuel.spigot",
                to_part="station.recycle.fuel.dirty_tank",
                position=(cx + 0.8, cy, cz)),
    ]
    return machine, runs


def bunk_shelter(centre=(-4.2, -0.35, 0.0), span_m: float = 3.0,
                 depth_m: float = 4.2, height_m: float = 2.0) -> Machine:
    """Crew quarters: the same frame-and-canvas as the plant shelter,
    closed on all four sides.

    SEALED FOR NOW, DELIBERATELY. There is nobody to put in it yet, so
    it is built as real geometry with real materials and no way in --
    `sealed=True` on the machine and no open face in the panels. When
    there are agents to occupy it, the door is a panel that opens, not a
    rebuild: the hardware is already here and already the right size for
    two people to live in.

    A CANVAS TENT IN A DESERT IS AN OVEN, which is why real practice is
    a double skin: a fly sheet over the tent with an air gap between,
    so the sun heats the fly and the gap carries it away rather than the
    inner wall. That is declared here as a separate panel set rather
    than thicker canvas, because it is a different mechanism -- shade
    and a ventilated gap, not insulation."""
    cx, cy, cz = centre
    parts: list = []
    for i, (dx, dz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        parts.append(MachinePart(
            f"station.bunk.leg_{i + 1}", "frame-tube",
            (cx + dx * span_m / 2.0, cy + height_m / 2.0, cz + dz * depth_m / 2.0),
            (0.028, height_m / 2.0, 0.028), mass_kg=9.0, material="steel-tube",
            part_role="bunk-frame", thermal_capacity_j_k=9.0 * 500.0))
    parts.append(MachinePart(
        "station.bunk.ridge", "frame-tube", (cx, cy + height_m, cz),
        (0.03, 0.03, depth_m / 2.0), mass_kg=13.0, material="steel-tube",
        part_role="bunk-frame", thermal_capacity_j_k=13.0 * 500.0))
    # four walls and a roof: no working face, because there is nobody to
    # work in it yet
    walls = (("roof", (cx, cy + height_m, cz), (span_m / 2.0, 0.002, depth_m / 2.0)),
             ("side_a", (cx - span_m / 2.0, cy + height_m / 2.0, cz),
              (0.002, height_m / 2.0, depth_m / 2.0)),
             ("side_b", (cx + span_m / 2.0, cy + height_m / 2.0, cz),
              (0.002, height_m / 2.0, depth_m / 2.0)),
             ("front", (cx, cy + height_m / 2.0, cz + depth_m / 2.0),
              (span_m / 2.0, height_m / 2.0, 0.002)),
             ("back", (cx, cy + height_m / 2.0, cz - depth_m / 2.0),
              (span_m / 2.0, height_m / 2.0, 0.002)))
    for name, pos, half in walls:
        area = 4.0 * half[0] * (half[2] if name == "roof" else half[1])
        parts.append(MachinePart(
            f"station.bunk.{name}", "canvas-panel", pos, half,
            mass_kg=max(2.0, area * 0.6), material="canvas",
            part_role="bunk-canvas", thermal_capacity_j_k=area * 0.6 * 1400.0))
    # the fly sheet: shade with an air gap, which is what makes a tent
    # survivable in a desert rather than thicker canvas
    parts.append(MachinePart(
        "station.bunk.fly", "canvas-panel", (cx, cy + height_m + 0.35, cz),
        (span_m / 2.0 + 0.4, 0.002, depth_m / 2.0 + 0.4),
        mass_kg=7.0, material="canvas", part_role="bunk-fly-sheet",
        thermal_capacity_j_k=7.0 * 1400.0))
    # what is actually in it, so the volume is not empty when it opens
    for i, dz in enumerate((-1, 1)):
        parts.append(MachinePart(
            f"station.bunk.cot_{i + 1}", "cot",
            (cx + dz * 0.75, cy + 0.25, cz), (0.35, 0.22, 0.95),
            mass_kg=11.0, material="steel-tube", part_role="crew-bunk",
            thermal_capacity_j_k=11.0 * 500.0))
        parts.append(MachinePart(
            f"station.bunk.locker_{i + 1}", "footlocker",
            (cx + dz * 0.75, cy + 0.20, cz - depth_m / 2.0 + 0.45),
            (0.30, 0.20, 0.22), mass_kg=18.0, material="steel-plate",
            capacity_kg=40.0, part_role="crew-stowage",
            thermal_capacity_j_k=18.0 * 500.0))
    machine = Machine(
        "station.bunk", "crew bunk shelter", power="none", medium="structure",
        parts=parts,
        note="sealed until there are agents to occupy it; double-skinned, because a "
             "single-skin canvas tent in a desert is an oven")
    # not a field on Machine, and deliberately explicit: anything that
    # offers entry has to look for this and find it True
    setattr(machine, "sealed", True)
    setattr(machine, "berths", 2)
    return machine
