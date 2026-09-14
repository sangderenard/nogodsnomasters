"""Station cooling without ever circulating potable water through machinery.

There are three physically isolated fluids:

* R134a runs from either powerplant compressor to a sealed evaporator coil.
* potable water surrounds that coil and stores cold for the site.
* propylene-glycol engine coolant exchanges with the water through a
  double-wall, leak-detecting heat exchanger.

The flexible metaconduit is a routing object, not a magic common pipe.  It
contains separately ported channels and its runtime exchanges heat through
their walls while conserving each channel's fluid and total thermal energy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from machines import Machine, MachineLine, MachinePart


@dataclass(frozen=True)
class ConduitChannel:
    name: str
    fluid: str
    nominal_id_m: float
    direction: str = "bidirectional"


DEFAULT_CHANNELS = (
    ConduitChannel("fuel", "multifuel", .016, "station-to-engine"),
    ConduitChannel("hydraulic-pressure", "hydraulic-oil", .050, "station-to-engine"),
    ConduitChannel("hydraulic-return", "hydraulic-oil", .065, "engine-to-station"),
    ConduitChannel("compressed-air", "air", .025, "engine-to-station"),
    ConduitChannel("refrigerant-discharge", "r134a", .012, "engine-to-station"),
    ConduitChannel("refrigerant-suction", "r134a", .016, "station-to-engine"),
    ConduitChannel("exhaust", "exhaust-gas", .36, "engine-to-station"),
    ConduitChannel("coolant-supply", "propylene-glycol-coolant", .038, "station-to-engine"),
    ConduitChannel("coolant-return", "propylene-glycol-coolant", .044, "engine-to-station"),
    ConduitChannel("48v-dc", "electricity", .012, "bidirectional"),
    ConduitChannel("control", "signal", .006, "bidirectional"),
)


@dataclass(frozen=True)
class CylindricalServiceTank:
    """Parametric low-pressure service tank, sized from useful volume."""
    identity: str
    fluid: str
    capacity_l: float
    working_fill_l: float
    inner_radius_m: float = 0.38
    shell_thickness_m: float = 0.0016
    insulation_m: float = 0.040
    fluid_density_kg_m3: float = 998.0
    shell_density_kg_m3: float = 8000.0

    @property
    def inner_length_m(self) -> float:
        return self.capacity_l / 1000.0 / (math.pi * self.inner_radius_m ** 2)

    @property
    def outer_radius_m(self) -> float:
        return self.inner_radius_m + self.shell_thickness_m + self.insulation_m

    @property
    def outer_length_m(self) -> float:
        return self.inner_length_m + 2.0 * (
            self.shell_thickness_m + self.insulation_m)

    @property
    def mass_kg(self) -> float:
        shell_area = (2.0 * math.pi * self.inner_radius_m
                      * self.inner_length_m
                      + 2.0 * math.pi * self.inner_radius_m ** 2)
        shell = shell_area * self.shell_thickness_m * self.shell_density_kg_m3
        fluid = self.working_fill_l / 1000.0 * self.fluid_density_kg_m3
        return shell + fluid

    def graph_attributes(self) -> dict:
        return {
            "material": "stainless-steel",
            "mass_in_total": True,
            "mass_kg": round(self.mass_kg, 2),
            "half_extent_m": (self.outer_radius_m, self.outer_radius_m,
                              self.outer_length_m / 2.0),
            "shape": "drum", "drum_axis": (0.0, 0.0, 1.0),
            "drum_radius_m": self.outer_radius_m,
            "drum_length_m": self.outer_length_m,
            "fluid": self.fluid, "capacity_l": self.capacity_l,
            "working_fill_l": self.working_fill_l,
            "shell_thickness_m": self.shell_thickness_m,
            "insulation_m": self.insulation_m,
        }


@dataclass
class MetaConduitRuntime:
    """Conservative wall exchange among isolated flexible channels."""
    channels: tuple[ConduitChannel, ...] = DEFAULT_CHANNELS
    wall_conductance_w_per_k: float = 3.5

    def step(self, dt: float, temperatures_k: dict[str, float],
             heat_capacities_j_per_k: dict[str, float]) -> dict[str, float]:
        names = [c.name for c in self.channels
                 if c.name in temperatures_k and c.name in heat_capacities_j_per_k]
        temperature = {name: float(temperatures_k[name]) for name in names}
        energy_delta = {name: 0.0 for name in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ca = max(float(heat_capacities_j_per_k[a]), 1.0)
                cb = max(float(heat_capacities_j_per_k[b]), 1.0)
                delta = temperature[a] - temperature[b]
                transfer_j = self.wall_conductance_w_per_k * delta * max(dt, 0.0)
                # Explicit exchange must not cross the pair's equilibrium.
                equilibrium_j = delta / (1.0 / ca + 1.0 / cb)
                transfer_j = math.copysign(
                    min(abs(transfer_j), abs(equilibrium_j)), transfer_j)
                energy_delta[a] -= transfer_j
                energy_delta[b] += transfer_j
        return {name: temperature[name] + energy_delta[name]
                / max(float(heat_capacities_j_per_k[name]), 1.0)
                for name in names}


@dataclass
class PlatformCoolantRuntime:
    """The coolant tank and assist pump supplied with every engine bay.

    An engine's own discovered coolant pump remains authoritative when it
    exists.  The platform pump becomes primary when it does not, and the HCU
    may parallel it as an assist.  A turbine with no water jacket rejects its
    bearing/reduction-gear oil heat through an oil-to-coolant exchanger; the
    fluids remain separate.
    """
    reservoir_l: float = 30.0
    reservoir_temp_k: float = 293.15
    pump_rated_flow_l_min: float = 160.0
    engine_exchanger_ua_w_per_k: float = 1600.0
    station_exchanger_ua_w_per_k: float = 2200.0
    coolant_density_kg_m3: float = 1040.0
    coolant_cp_j_kg_k: float = 3500.0

    @property
    def thermal_capacity_j_per_k(self) -> float:
        return self.reservoir_l / 1000.0 * self.coolant_density_kg_m3 * self.coolant_cp_j_kg_k

    @staticmethod
    def _engine_circuit(sim):
        circuits = getattr(getattr(sim, "_drivetrain", None), "fluid_circuits", ())
        coolant = next((c for c in circuits if c.kind_class == "thermal-liquid"
                        and any("coolant" in n or "radiator" in n for n in c.nodes)), None)
        if coolant is not None:
            return coolant, "engine-coolant"
        oil = next((c for c in circuits if c.kind_class == "thermal-liquid"
                    and any("oil" in n for n in c.nodes)), None)
        return oil, "turbine-oil-to-coolant" if oil is not None else "no-engine-thermal-circuit"

    def step(self, dt: float, sim, station_supply_temp_k: float,
             hcu_command: float = 1.0) -> dict:
        circuit, interface = self._engine_circuit(sim)
        command = max(0.0, min(1.0, float(hcu_command)))
        engine_has_pump = bool(circuit is not None and circuit.pump_node)
        engine_flow = float(getattr(circuit, "flow_lpm", 0.0)) if circuit is not None else 0.0
        assist_flow = self.pump_rated_flow_l_min * command
        flow_factor = min(1.0, (engine_flow + assist_flow)
                          / max(self.pump_rated_flow_l_min, 1.0))
        engine_heat_w = 0.0
        if circuit is not None:
            delta = float(circuit.temp_k) - self.reservoir_temp_k
            engine_heat_w = self.engine_exchanger_ua_w_per_k * flow_factor * delta
            capacity = max(float(circuit.thermal_mass_kj_per_k) * 1000.0, 1.0)
            circuit.temp_k -= engine_heat_w * dt / capacity
            self.reservoir_temp_k += engine_heat_w * dt / self.thermal_capacity_j_per_k
        station_heat_w = (self.station_exchanger_ua_w_per_k * command
                          * (self.reservoir_temp_k - float(station_supply_temp_k)))
        self.reservoir_temp_k -= station_heat_w * dt / self.thermal_capacity_j_per_k
        return {"interface": interface, "engine_has_own_pump": engine_has_pump,
                "platform_pump_flow_l_min": assist_flow,
                "engine_heat_removed_w": engine_heat_w,
                "station_heat_return_w": station_heat_w,
                "reservoir_temp_k": self.reservoir_temp_k}


@dataclass
class SiteCoolingRuntime:
    """Cold potable store and the separate station coolant inventory."""
    potable_water_l: float = 1000.0
    potable_water_temp_k: float = 277.15
    coolant_l: float = 260.0
    coolant_temp_k: float = 285.15
    water_to_coolant_ua_w_per_k: float = 6500.0
    ambient_k: float = 293.15
    water_skin_ua_w_per_k: float = 24.0

    def step(self, dt: float, *, refrigerant_cooling_w: float,
             platform_heat_w: float, hcu_valve: float = 1.0,
             potable_draw_l: float = 0.0) -> dict:
        water_capacity = max(self.potable_water_l * 4180.0, 1.0)
        coolant_capacity = max(self.coolant_l * 1.04 * 3500.0, 1.0)
        valve = max(0.0, min(1.0, float(hcu_valve)))
        exchange_w = (self.water_to_coolant_ua_w_per_k * valve
                      * (self.coolant_temp_k - self.potable_water_temp_k))
        water_skin_w = self.water_skin_ua_w_per_k * (self.ambient_k - self.potable_water_temp_k)
        self.coolant_temp_k += (float(platform_heat_w) - exchange_w) * dt / coolant_capacity
        self.potable_water_temp_k += (exchange_w + water_skin_w
                                      - max(0.0, float(refrigerant_cooling_w))) * dt / water_capacity
        self.potable_water_temp_k = max(273.65, self.potable_water_temp_k)
        self.potable_water_l = max(0.0, self.potable_water_l - max(0.0, potable_draw_l))
        return {"potable_water_l": self.potable_water_l,
                "potable_water_temp_k": self.potable_water_temp_k,
                "coolant_temp_k": self.coolant_temp_k,
                "water_to_coolant_heat_w": exchange_w,
                "refrigerant_cooling_w": max(0.0, float(refrigerant_cooling_w)),
                "fluids_mixed": False}


@dataclass
class DesertHeatRejectorRuntime:
    """Remote multi-core heat and exhaust diffuser.

    It cannot erase heat.  It trades a compact hot point for a large,
    lower-temperature surface and a high-mass, low-velocity mixed plume.
    """
    core_count: int = 8
    core_ua_w_per_k: float = 5000.0
    blower_flow_m3_s_each: float = 12.0
    blower_rated_w_each: float = 7500.0
    exhaust_ejector_entrainment_ratio: float = 4.0
    air_density_kg_m3: float = 1.12
    air_cp_j_kg_k: float = 1006.0

    def step(self, dt: float, *, ambient_k: float, coolant_in_k: float,
             coolant_heat_capacity_j_per_k: float,
             coolant_flow_command: float, exhaust_temp_k: float,
             exhaust_mass_flow_kg_s: float, blower_command: float) -> dict:
        blower = max(0.0, min(1.0, float(blower_command)))
        coolant = max(0.0, min(1.0, float(coolant_flow_command)))
        airflow_m3_s = self.core_count * self.blower_flow_m3_s_each * blower
        airflow_factor = min(1.0, airflow_m3_s
                             / max(self.core_count * self.blower_flow_m3_s_each, 1e-9))
        rejected_w = (self.core_count * self.core_ua_w_per_k * coolant
                      * (0.18 + 0.82 * airflow_factor)
                      * max(0.0, float(coolant_in_k) - float(ambient_k)))
        capacity = max(float(coolant_heat_capacity_j_per_k), 1.0)
        coolant_out_k = max(float(ambient_k),
                            float(coolant_in_k) - rejected_w * max(dt, 0.0) / capacity)
        exhaust_m = max(0.0, float(exhaust_mass_flow_kg_s))
        # The ejector entrains ambient from the core/blower stream.  It is
        # bounded by actual blower mass flow, not a requested ratio.
        available_air_m = airflow_m3_s * self.air_density_kg_m3
        entrained_m = min(available_air_m,
                          exhaust_m * self.exhaust_ejector_entrainment_ratio)
        mixed_m = exhaust_m + entrained_m
        plume_k = (exhaust_m * float(exhaust_temp_k)
                   + entrained_m * float(ambient_k)) / max(mixed_m, 1e-9)
        return {"rejected_w": rejected_w, "coolant_out_k": coolant_out_k,
                "airflow_m3_s": airflow_m3_s,
                "blower_electrical_w": self.core_count * self.blower_rated_w_each * blower ** 3,
                "entrained_air_kg_s": entrained_m,
                "diffuser_mass_flow_kg_s": mixed_m,
                "diffuser_outlet_temp_k": plume_k,
                "temperature_contrast_k": max(0.0, plume_k - float(ambient_k))}


def desert_heat_rejector(identity: str = "station.heat_rejector",
                         at=(0.0, -1.75, -5.2), core_count: int = 8) -> Machine:
    """A tow/drop-in, non-engine machine with N parallel exchange cores."""
    p = np.asarray(at, float)
    parts = []
    inlet = f"{identity}.inlet_plenum"
    outlet = f"{identity}.diffuser"
    parts.append(MachinePart(
        inlet, "insulated-multi-fluid-and-exhaust-plenum", tuple(p + (0, .35, .95)),
        (.72, .34, .28), mass_kg=185.0, material="stainless-steel",
        part_role="remote-hot-service-inlet", thermal_capacity_j_k=90_000.0,
        attributes={"insulated": True, "high_flow": True,
                    "separate_internal_passages": True}))
    parts.append(MachinePart(
        outlet, "high-entrainment-low-velocity-exhaust-diffuser", tuple(p + (0, .30, -.95)),
        (1.65, .30, .42), mass_kg=270.0, material="stainless-steel",
        part_role="distributed-low-contrast-exhaust-outlet",
        thermal_capacity_j_k=135_000.0,
        attributes={"ejector_entrainment_ratio": 4.0,
                    "multiport_outlet_count": 32,
                    "goal": "spread heat and momentum; never claim zero IR signature"}))
    lines = []
    for i in range(core_count):
        row, col = divmod(i, 4)
        x = (col - 1.5) * .72
        z = (row - .5) * .78
        core = f"{identity}.core.{i}"
        blower = f"{identity}.blower.{i}"
        parts.append(MachinePart(
            core, "crossflow-air-exchange-core", tuple(p + (x, .55, z)),
            (.30, .48, .32), mass_kg=94.0, material="aluminium-cast",
            part_role="parallel-air-exchange-core", thermal_capacity_j_k=80_000.0,
            attributes={"heat_exchange_ua_w_per_k": 5000.0,
                        "coolant_passage_isolated": True,
                        "hydraulic_oil_passage_isolated": True,
                        "refrigerant_condenser_passage_isolated": True}))
        parts.append(MachinePart(
            blower, "variable-speed-electric-axial-blower", tuple(p + (x, .55, z - .38)),
            (.27, .27, .14), mass_kg=38.0, material="aluminium-cast",
            part_role="quiet-fine-control-core-blower", thermal_capacity_j_k=18_000.0,
            attributes={"rated_flow_m3_s": 12.0, "rated_w": 7500.0,
                        "supply_voltage_v": 48.0, "variable_speed": True,
                        "low_tip_speed_noise_control": True}))
        # The blower is flange-mounted locally to its core casing.  Four
        # separated surface ports form a real moment couple; the old model
        # ran one 1.7 m tube from the common inlet centroid to each blower
        # and called that a mount.
        for sx, sy in ((-1.0, -1.0), (-1.0, 1.0),
                       (1.0, -1.0), (1.0, 1.0)):
            tag = f"{'l' if sx < 0 else 'r'}{'b' if sy < 0 else 't'}"
            blower_port = f"{blower}.mount_port.{tag}"
            core_port = f"{core}.blower_port.{tag}"
            parts.append(MachinePart(
                blower_port, "structural-mount-port",
                tuple(p + (x + sx * .20, .55 + sy * .18, z - .24)),
                (.018, .018, .018), mass_kg=0.0, material="4130n",
                part_role="blower-flange-port",
                attributes={"solver_condensed_into": blower,
                            "solver_condensed_mass": False,
                            "wrench_point": True, "surface_of": blower,
                            "in_view": False}))
            parts.append(MachinePart(
                core_port, "structural-mount-port",
                tuple(p + (x + sx * .20, .55 + sy * .18, z - .32)),
                (.018, .018, .018), mass_kg=0.0, material="4130n",
                part_role="core-blower-flange-port",
                attributes={"solver_condensed_into": core,
                            "solver_condensed_mass": False,
                            "wrench_point": True, "surface_of": core,
                            "in_view": False}))
            lines.append(MachineLine(
                f"{identity}.blower_flange_standoff.{i}.{tag}",
                blower_port, core_port, "rigid-distance", .012,
                "rejector-frame", "4130n"))
        lines.append(MachineLine(f"{identity}.hot_to_core.{i}", inlet, core,
                                 "insulated-flexible-thermal-duct", .10,
                                 f"thermal-bank-{i}", "aerogel-jacket"))
        lines.append(MachineLine(f"{identity}.core_to_diffuser.{i}", core, outlet,
                                 "insulated-flexible-thermal-duct", .10,
                                 f"thermal-bank-{i}", "aerogel-jacket"))
        lines.append(MachineLine(f"{identity}.blower_feed.{i}", inlet, blower,
                                 "insulated-copper-wire", .008,
                                 "48v-dc", "copper"))
        # Service lines do not locate machinery.  Each core is carried
        # between the two structural plenums and each blower is located by
        # the inlet frame and its own core casing.  Keeping these as actual
        # members matters to the station graph: a hose must never be allowed
        # to masquerade as the third leg of a physical mount.
        lines.append(MachineLine(f"{identity}.core_inlet_mount.{i}", inlet, core,
                                 "rigid-distance", .024,
                                 "rejector-frame", "4130n"))
        lines.append(MachineLine(f"{identity}.core_outlet_mount.{i}", core, outlet,
                                 "rigid-distance", .024,
                                 "rejector-frame", "4130n"))
    return Machine(identity, "sand-deployable multi-core heat and exhaust rejector",
                   power="48v-dc-electric-blowers", medium="isolated-fluids-plus-exhaust",
                   parts=parts, lines=lines,
                   note="large-area heat spreading and plume dilution; no heat destruction")


def emit_desert_heat_rejector(g, stand, cooling: dict,
                              metaconduits: dict) -> dict:
    """Place the remote rejector on sand and connect both removable bays."""
    machine = desert_heat_rejector(at=(0.0, -stand.lower_room_clear_height_m + .65, -5.2))
    document = machine.build_graph()
    g.motion_group, g.assembly = "world", "remote-heat-rejection"
    for node in document["nodes"]:
        attrs = {k: v for k, v in node.items()
                 if k not in ("identity", "kind", "reference_position",
                              "body_half_extent_m", "motion_group", "assembly",
                              "in_view", "thermal_group")}
        g.node(node["identity"], node["reference_position"], node["kind"],
               half_extent_m=node["body_half_extent_m"], in_view=True,
               thermal_group=node["identity"], **attrs)
    for edge in document["edges"]:
        structural = edge["constraint"] == "rigid-distance"
        g.edge(edge["identity"], edge["a"], edge["b"], edge["constraint"],
               radius=edge["radius"],
               palette="chassis-grey" if structural else "service-line",
               beam_solvable=structural, routed=not structural,
               circuit_identity=edge["circuit_identity"],
               material=edge["material"],
               part_role=("rejector-equipment-mount" if structural
                          else "rejector-internal-route"))
    inlet = f"{machine.identity}.inlet_plenum"
    diffuser = f"{machine.identity}.diffuser"
    # Broad sand shoes are the remote machine's boundary, not the station's.
    for i, x in enumerate((-1.35, 1.35)):
        foot = f"{machine.identity}.sand_foot.{i}"
        g.node(foot, (x, -stand.lower_room_clear_height_m, -5.2),
               "load-bearing-structure", material="steel-plate",
               mass_in_total=False, mass_kg=75.0, in_view=True,
               fixed_to="world", half_extent_m=(.34, .06, .34),
               part_role="sand-spreading-rejector-foot")
        g.edge(f"{foot}.mount", foot, inlet, "rigid-distance", radius=.045,
               alloy="4130n", palette="chassis-grey", beam_solvable=True,
               part_role="rejector-drop-frame")
    for side in ("port", "starboard"):
        station_head = metaconduits[side]["station_end"]
        engine_head = metaconduits[side]["engine_end"]
        exhaust_channel = next(edge for edge in g.edges if edge["identity"] ==
                               f"station.metaconduit.{side}.channel.exhaust")
        exhaust_channel.update(
            constraint="insulated-flexible-exhaust-duct",
            circuit_identity=f"remote-exhaust-{side}", fluid="exhaust-gas")
        g.edge(f"station.heat_rejector.exhaust_engine_pigtail.{side}",
               f"engine.{side}", engine_head,
               "insulated-flexible-exhaust-duct", radius=.18,
               palette="service-line", beam_solvable=False, routed=True,
               fluid="exhaust-gas", circuit_identity=f"remote-exhaust-{side}",
               high_flow=True, expansion_joint=True)
        g.edge(f"station.heat_rejector.exhaust_station_pigtail.{side}",
               station_head, inlet, "insulated-flexible-exhaust-duct",
               radius=.18, palette="service-line", beam_solvable=False,
               routed=True, fluid="exhaust-gas",
               circuit_identity=f"remote-exhaust-{side}", high_flow=True,
               expansion_joint=True)
        g.edge(f"station.heat_rejector.control.{side}", station_head, inlet,
               "insulated-copper-wire", radius=.006, palette="service-line",
               beam_solvable=False, routed=True,
               circuit_identity="heat-rejector-48v-control")
    g.edge("station.heat_rejector.coolant_supply",
           cooling["coolant_manifold"], inlet, "coolant-line", radius=.032,
           palette="service-line", beam_solvable=False, routed=True,
           fluid="propylene-glycol-coolant",
           circuit_identity="station-engine-coolant")
    g.edge("station.heat_rejector.coolant_return", diffuser,
           cooling["coolant_reservoir"], "coolant-line", radius=.038,
           palette="service-line", beam_solvable=False, routed=True,
           fluid="propylene-glycol-coolant",
           circuit_identity="station-engine-coolant")
    return {"machine": machine, "parts": tuple(n["identity"] for n in document["nodes"]),
            "inlet": inlet, "diffuser": diffuser}


def emit_site_cooling(g, stand, site: dict) -> dict:
    """Install the two isolated reservoirs and both real exchangers."""
    y = stand.deck_y - 1.20
    floor_y = float(site.get(
        "floor_surface_y_m",
        stand.deck_y - stand.lower_room_clear_height_m))
    made = {}
    tank_spec = CylindricalServiceTank(
        identity="station.cooling.potable_tank", fluid="potable-water",
        capacity_l=500.0, working_fill_l=500.0)
    wall_x = stand.half
    tank_x = stand.half - tank_spec.outer_radius_m - 0.10
    aisle_m = 2.0 * (tank_x - tank_spec.outer_radius_m)
    rail_z = min(stand.half - 0.20, tank_spec.outer_length_m * 0.40)
    for side, sign in (("port", -1.0), ("starboard", 1.0)):
        ident = f"{tank_spec.identity}.{side}"
        x = sign * tank_x
        g.node(ident, (x, floor_y + tank_spec.outer_radius_m, 0.0),
               "insulated-potable-water-tank", in_view=True,
               sanitary_liner=True,
               part_role="chilled-potable-thermal-store",
               installed_on="lower-room-floor-cradle",
               installed_under_counter=True, service_face="inboard",
               clear_central_aisle_m=round(aisle_m, 3),
               **tank_spec.graph_attributes())
        frame_tags = (("fl", "rl") if side == "port" else ("fr", "rr"))
        plate_i = 0 if side == "port" else 4
        rail_nodes = []
        for end, z, frame_tag, plate_j in (
                ("front", rail_z, frame_tags[0], 3),
                ("rear", -rail_z, frame_tags[1], 1)):
            rail = f"station.cooling.floor_rail.{side}.{end}"
            foot = f"{ident}.foot.{end}"
            g.node(rail, (sign * wall_x, floor_y, z), "chassis-load-node",
                   material="4130n", mass_in_total=False, mass_kg=12.0,
                   part_role="tank-service-floor-rail-node", in_view=False,
                   half_extent_m=(.06, .06, .06))
            g.node(foot, (x, floor_y, z), "structural-mount-port",
                   material="stainless-steel", mass_in_total=False,
                   mass_kg=0.0, wrench_point=True, surface_of=ident,
                   solver_condensed_into=ident, solver_condensed_mass=False,
                   part_role="tank-cradle-foot", in_view=False,
                   half_extent_m=(.05, .04, .05))
            g.edge(f"{rail}.tie", rail,
                   site["floor_plate"]["nodes"][(plate_i, plate_j)],
                   "rigid-distance", radius=.034, alloy="4130n",
                   palette="chassis-grey", beam_solvable=True,
                   part_role="tank-service-floor-plate-mount",
                   adjacent_frame_corner=site["lower_room"][frame_tag])
            g.edge(f"{ident}.mount.{end}", foot, rail, "rigid-distance",
                   radius=.030, alloy="4130n", palette="chassis-grey",
                   beam_solvable=True, part_role="water-tank-cradle-mount",
                   load_share=0.5,
                   load_path="floor-cradle-carrying-the-tank-into-the-lower-"
                             "room-side-frame")
            rail_nodes.append(rail)
        g.edge(f"station.cooling.floor_rail.{side}.run", *rail_nodes,
               "rigid-distance", radius=.034, alloy="4130n",
               palette="chassis-grey", beam_solvable=True,
               part_role="tank-service-floor-side-rail")

        # A physical counter rail above the tank, leaving its inboard face and
        # both end fittings accessible from the central service aisle.
        counter = []
        counter_y = floor_y + 0.92
        for end, z, rail in (("front", rail_z, rail_nodes[0]),
                             ("rear", -rail_z, rail_nodes[1])):
            node = f"station.cooling.counter.{side}.{end}"
            g.node(node, (x, counter_y, z), "chassis-load-node",
                   material="a36", mass_in_total=False, mass_kg=8.0,
                   part_role="tank-service-counter", in_view=True,
                   half_extent_m=(.05, .05, .05))
            g.edge(f"{node}.post", rail, node, "rigid-distance",
                   radius=.022, alloy="a36", palette="chassis-grey",
                   beam_solvable=True, part_role="tank-service-counter-post")
            counter.append(node)
        g.edge(f"station.cooling.counter.{side}.worktop", *counter,
               "rigid-distance", radius=.035, alloy="a36",
               palette="chassis-grey", beam_solvable=True,
               part_role="tank-service-counter-worktop",
               worktop_height_above_floor_m=0.92)
        made[side] = ident
    coil = "station.cooling.potable_refrigerant_coil"
    exchanger = "station.cooling.double_wall_water_to_coolant_hx"
    reservoir = "station.cooling.coolant_reservoir"
    pump = "station.cooling.coolant_pump"
    coolant_manifold = site["hcu"]["coolant_manifold"]
    g.node(coil, (0.0, y + .18, .55), "sanitary-refrigerant-evaporator-coil",
           material="stainless-steel", mass_in_total=True, mass_kg=48.0,
           in_view=True, half_extent_m=(.34, .18, .16),
           part_role="sealed-refrigerant-to-potable-water-exchanger",
           fluids_isolated=True, leak_detection=True, refrigerant="r134a")
    g.node(exchanger, (0.0, y + .18, -.55), "double-wall-plate-heat-exchanger",
           material="stainless-steel", mass_in_total=True, mass_kg=71.0,
           in_view=True, half_extent_m=(.34, .22, .18),
           part_role="potable-water-to-engine-coolant-interchange",
           fluids_isolated=True, interstitial_leak_detection=True,
           potable_side="potable-water", machine_side="propylene-glycol-coolant",
           heat_exchange_ua_w_per_k=6500.0)
    g.node(reservoir, (0.0, y - .36, -.30), "station-coolant-reservoir",
           material="stainless-steel", mass_in_total=True, mass_kg=316.0,
           in_view=True, half_extent_m=(.42, .30, .38),
           capacity_l=300.0, working_fill_l=260.0,
           fluid="propylene-glycol-coolant", potable=False,
           part_role="station-engine-coolant-buffer")
    g.node(pump, (0.0, y - .36, .38), "variable-speed-electric-coolant-pump",
           material="aluminium-cast", mass_in_total=True, mass_kg=34.0,
           in_view=True, half_extent_m=(.23, .18, .20),
           rated_flow_l_min=320.0, rated_head_pa=280_000.0,
           rated_w=4200.0, supply_voltage_v=48.0,
           part_role="station-coolant-circulation-pump")
    rack_targets = tuple(site["lower_room"].values())
    for ident in (coil, exchanger, reservoir, pump):
        for i, target in enumerate(rack_targets):
            g.edge(f"{ident}.mount.{i}", ident, target, "rigid-distance",
                   radius=.024, alloy="4130n", palette="chassis-grey",
                   beam_solvable=True, part_role="cooling-plant-mount",
                   load_share=0.25,
                   load_path="four-point-cooling-rack-into-the-lower-room-"
                             "end-crossmembers")
    # Potable water touches only sanitary sides of the two exchangers.
    for i, (a, b) in enumerate(((made["port"], coil), (coil, made["starboard"]),
                                (made["starboard"], exchanger),
                                (exchanger, made["port"]))):
        g.edge(f"station.cooling.potable_loop.{i}", a, b, "coolant-line",
               radius=.026, palette="service-line", beam_solvable=False,
               routed=True, fluid="potable-water",
               circuit_identity="station-potable-water",
               sanitary=True, never_engine_coolant=True)
    for ident, a, b in (("reservoir_to_pump", reservoir, pump),
                        ("pump_to_hcu", pump, coolant_manifold),
                        ("hcu_to_hx", coolant_manifold, exchanger),
                        ("hx_to_reservoir", exchanger, reservoir)):
        g.edge(f"station.cooling.coolant.{ident}", a, b, "coolant-line",
               radius=.024, palette="service-line", beam_solvable=False,
               routed=True, fluid="propylene-glycol-coolant",
               circuit_identity="station-engine-coolant",
               potable=False)
    return {**made, "refrigerant_coil": coil, "water_coolant_exchanger": exchanger,
            "coolant_reservoir": reservoir, "coolant_pump": pump,
            "coolant_manifold": coolant_manifold}


def emit_metaconduit(g, side: str, package: dict, cooling: dict,
                     channels: tuple[ConduitChannel, ...] = DEFAULT_CHANNELS) -> dict:
    """One removable flexible bundle with individually ported channels."""
    station_end = f"station.metaconduit.{side}.station_end"
    engine_end = f"station.metaconduit.{side}.engine_end"
    skid = package["skid"]
    positions = {n["identity"]: np.asarray(n["reference_position"], float)
                 for n in g.nodes}
    a = positions[cooling["coolant_manifold"]] + np.array([
        -.34 if side == "port" else .34, 0.0, 0.0])
    b = positions[skid] + np.array([0.0, .14, 0.0])
    for ident, at, role in ((station_end, a, "station-side-dry-break-head"),
                            (engine_end, b, "lowerable-engine-side-dry-break-head")):
        gestalt = cooling["coolant_manifold"] if ident == station_end else skid
        g.node(ident, tuple(at), "multi-circuit-dry-break-coupler",
               material="stainless-steel", mass_in_total=True, mass_kg=18.0,
               in_view=True, half_extent_m=(.18, .15, .18), part_role=role,
               structural_participation=False,
               solver_condensed_into=gestalt,
               solver_condensed_mass=True,
               channel_count=len(channels), simultaneous_disconnect=True,
               spill_minimising_dry_breaks=True)
    g.edge(f"station.metaconduit.{side}.jacket", station_end, engine_end,
           "flexible-multi-circuit-conduit", radius=.24,
           palette="service-line", beam_solvable=False, routed=True,
           in_view=True, part_role="thermally-coupled-isolated-service-bundle",
           contained_channels=tuple(c.name for c in channels),
           channel_fluids=tuple(c.fluid for c in channels),
           channel_thermal_conductance_w_per_k=3.5,
           carries_structural_load=False,
           service_motion="engine-platform-operates-at-any-lowering-height")
    channel_edges = []
    for channel in channels:
        ident = f"station.metaconduit.{side}.channel.{channel.name}"
        g.edge(ident, station_end, engine_end, "metaconduit-isolated-channel",
               radius=channel.nominal_id_m / 2.0, palette="service-line",
               beam_solvable=False, routed=True, in_view=False,
               circuit_identity=f"metaconduit.{side}.{channel.name}",
               fluid=channel.fluid, flow_direction=channel.direction,
               thermal_exchange_group=f"metaconduit.{side}",
               fluid_isolated=True)
        channel_edges.append(ident)
    # The channel edges above are the long flexible spans.  These short
    # pigtails bind them to actual machine ports at either end.
    prefix = package["package"].identity
    bindings = (
        ("coolant-supply", cooling["coolant_manifold"], station_end,
         engine_end, f"{prefix}.coolant_pump", "coolant-line",
         "station-engine-coolant", "propylene-glycol-coolant", .019),
        ("coolant-return", f"{prefix}.coolant_reservoir", engine_end,
         station_end, cooling["coolant_manifold"], "coolant-line",
         "station-engine-coolant", "propylene-glycol-coolant", .022),
        ("refrigerant-discharge", f"{prefix}.refrigerant", engine_end,
         station_end, cooling["refrigerant_coil"], "refrigerant-line",
         f"station-refrigerant-{side}", "r134a", .010),
        ("refrigerant-suction", cooling["refrigerant_coil"], station_end,
         engine_end, f"{prefix}.refrigerant", "refrigerant-line",
         f"station-refrigerant-{side}", "r134a", .012),
    )
    for (name, upstream, first, second, downstream, constraint,
         circuit, fluid, radius) in bindings:
        channel = next(edge for edge in g.edges
                       if edge["identity"] == f"station.metaconduit.{side}.channel.{name}")
        channel.update(constraint=constraint, circuit_identity=circuit,
                       fluid=fluid)
        g.edge(f"station.metaconduit.{side}.{name}.upstream", upstream, first,
               constraint, radius=radius, palette="service-line",
               beam_solvable=False, routed=True, circuit_identity=circuit,
               fluid=fluid, dry_break=True)
        g.edge(f"station.metaconduit.{side}.{name}.downstream", second, downstream,
               constraint, radius=radius, palette="service-line",
               beam_solvable=False, routed=True, circuit_identity=circuit,
               fluid=fluid, dry_break=True)
    return {"station_end": station_end, "engine_end": engine_end,
            "jacket": f"station.metaconduit.{side}.jacket",
            "channels": tuple(channel_edges)}
