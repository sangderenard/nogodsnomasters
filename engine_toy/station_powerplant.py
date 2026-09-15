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
from machines import Machine, MachineLine, MachinePart


PMG_RATED_W = 30_000.0
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
            "battery": p + np.array([side * 0.68, -0.08, -0.46]),
            "refrigerant": p + np.array([0.0, 0.38, -0.60]),
            "coolant_reservoir": p + np.array([side * 0.58, 0.34, -0.08]),
            "coolant_pump": p + np.array([side * 0.58, 0.34, 0.26]),
        }
        prefix = self.identity
        def ident(name): return f"{prefix}.{name}"
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
                 "rated_w": PMG_RATED_W, "dc_voltage_range_v": [48.0, 56.0],
                 "rated_speed_range_rpm": [2500.0, 3600.0],
                 "full_load_efficiency": PMG_EFFICIENCY,
                 "specification_source": PMG_SOURCE,
                 "mass_basis": "modeled installed-package estimate; manufacturer mass not claimed"}),
            ("battery", "battery-pack-48v", (.30, .16, .24), 94.0,
             "electrical-storage", {"nominal_voltage_v": 48.0,
                                    "capacity_ah": 200.0,
                                    "stored_energy_kwh": 9.6}),
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
                 "control": "HCU-coolant-manifold",
                 "role": "primary when engine has no pump; parallel assist otherwise"}),
        )
        for name, kind, ext, mass, role, attributes in specifications:
            q = locations[name]
            ports = []
            if name not in ("battery", "coolant_reservoir", "coolant_pump"):
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
                ports.append(_port(f"{ident(name)}.dc_out", ident(name),
                                   "48v-dc-out", q, (0, 1, 0), .010,
                                   fluid="electricity",
                                   connected_to=f"{ident('battery')}.dc_bus"))
            elif name == "battery":
                ports.append(_port(f"{ident(name)}.dc_bus", ident(name),
                                   "48v-dc-bidirectional", q, (0, 1, 0), .010,
                                   fluid="electricity",
                                   connected_to=f"{ident('generator')}.dc_out"))
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
        lines.append(MachineLine(f"{prefix}.generator_to_battery",
                                 ident("generator"), ident("battery"),
                                 "insulated-copper-wire", .010, "48v-dc",
                                 "copper"))
        lines.append(MachineLine(f"{prefix}.battery_to_coolant_pump",
                                 ident("battery"), ident("coolant_pump"),
                                 "insulated-copper-wire", .008, "48v-dc",
                                 "copper"))
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
