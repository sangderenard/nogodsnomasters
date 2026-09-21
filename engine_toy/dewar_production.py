"""Parametric production graph for an open atmospheric dewar.

This module declares parts, ports and geometry. Runtime
advancement belongs to engine-toy's existing systems; the dewar is not an
engine and owns no step method.
"""
from __future__ import annotations

import math
import numpy as np

from assembly_ports import PartPort
from compressors import DiaphragmCompressorSet
from cryogenics import compression_power_w
from cryogenic_cold_head_production import build as build_cold_head
from dc_power import DCBattery
from electrical_distribution import BuildingCable, CircuitBreaker, standard_services
from machines import Machine


def build(identity: str = "lab.dewar", *,
          chamber_shape: tuple[int, int, int] = (5, 5, 5),
          cell_size_m: float = 0.25,
          open_port_radius_m: float = 0.003,
          annulus_pressure_pa: float = 1.0,
          vacuum_intact: bool = True,
          insulation_media: str = "mli",
          insulation_thickness_m: float = 0.035,
          mass_flow_kg_s: float = 0.05,
          inlet_pressure_pa: float = 4.0e6,
          return_pressure_pa: float = 1.5e5,
          recuperator_effectiveness: float = 0.97,
          working_gas_capacity_kg: float = 5.0,
          working_gas_initial_kg: float | None = None,
          drain_capacity_kg: float = 20.0):
    """Build the dewar from parametric parts in the production vocabulary."""
    inside_size_m = tuple(float(n) * float(cell_size_m) for n in chamber_shape)
    open_area_m2 = math.pi * float(open_port_radius_m) ** 2
    if working_gas_initial_kg is None:
        working_gas_initial_kg = working_gas_capacity_kg
    if not 0.0 <= working_gas_initial_kg <= working_gas_capacity_kg:
        raise ValueError("working gas mass must be within bottle capacity")
    cold_head = f"{identity}.cold_head"
    compressor_spec = DiaphragmCompressorSet(
        identity=f"{cold_head}.compressor",
        suction_pressure_pa=return_pressure_pa,
        discharge_pressure_pa=inlet_pressure_pa,
        mass_flow_kg_s=mass_flow_kg_s)
    battery_spec = DCBattery(
        identity=f"{identity}.battery", chemistry="lifepo4",
        capacity_ah=400.0, series_packs=4, state_of_charge=0.95)
    electrical_service = standard_services(identity)["48v-dc"]
    compressor_shaft_w = compression_power_w(
        "air-diaphragm", mass_flow_kg_s, 293.15,
        inlet_pressure_pa / max(return_pressure_pa, 1.0))
    compressor_electrical_w = (
        compressor_shaft_w / compressor_spec.motor_efficiency)
    g = build_cold_head(
        cold_head, mass_flow_kg_s=mass_flow_kg_s,
        inlet_pressure_pa=inlet_pressure_pa,
        return_pressure_pa=return_pressure_pa,
        recuperator_effectiveness=recuperator_effectiveness)
    g.identity = f"{identity}/production"

    sx, sy, sz = inside_size_m
    hx, hz = sx / 2.0, sz / 2.0
    wall = 0.025
    jacket_vacuum_volume_m3 = (
        (sx + 2.0 * wall) * (sy + 2.0 * wall) * (sz + 2.0 * wall)
        - sx * sy * sz
    )
    cold_shift_y = sy + 0.055
    cold_prefix = cold_head
    for node in g.nodes:
        node["reference_position"][1] += cold_shift_y

    # The gas volume and vacuum annulus are state records, not rendered or
    # voxelized bodies.  Their bounds are declarations for the world coupler.
    g.node(f"{identity}.chamber_volume", (0.0, sy / 2.0, 0.0),
           "gas-volume", half_extent_m=(hx, sy / 2.0, hz), mass_kg=0.0,
           render_primitive=False, structural_participation=False,
           thermal_domain="volume",
           thermal_state_owner="chamber",
           fluid_state_owner="atmosphere", fluid="gas",
           fluid_volume_l=sx * sy * sz * 1000.0,
           fluid_spatial_model="voxel",
           voxel_shape=chamber_shape, voxel_size_m=cell_size_m,
           open_face="+y")
    g.node(f"{identity}.jacket_vacuum", (0.0, sy / 2.0, 0.0),
           "vacuum-volume", half_extent_m=(hx + wall, sy / 2.0 + wall, hz + wall),
           mass_kg=0.0, render_primitive=False, structural_participation=False,
           thermal_domain="none",
           fluid="gas", fluid_volume_l=jacket_vacuum_volume_m3 * 1000.0,
           pressure_pa=annulus_pressure_pa,
           vacuum_intact=vacuum_intact,
           insulation_media=insulation_media,
           insulation_thickness_m=insulation_thickness_m)
    jacket_vacuum = f"{identity}.jacket-vacuum"
    for name, y, direction in (
        ("bottom", -wall, (0.0, -1.0, 0.0)),
        ("top", sy + wall, (0.0, 1.0, 0.0)),
    ):
        port = f"{identity}.jacket_vacuum_{name}_port"
        g.node(port, (0.0, y, hz + wall), "valved-vacuum-service-port",
               half_extent_m=(0.012, 0.012, 0.012), material="stainless-steel",
               thermal_domain="none", fluid="gas", pressure_pa=annulus_pressure_pa,
               valve_open_fraction=0.0, normally_closed=True,
               port_role=f"jacket-vacuum-{name}-service",
               direction=direction)
        g.edge(f"{port}.line", port, f"{identity}.jacket_vacuum",
               "vacuum-line", radius=0.005,
               circuit_identity=jacket_vacuum,
               valve_node=port, medium_rate_state="mass-flow-pressure-temperature")

    panels = {
        "floor": ((0.0, -wall, 0.0), (hx + wall, wall, hz + wall)),
        "left": ((-hx - wall, sy / 2.0, 0.0), (wall, sy / 2.0, hz + wall)),
        "right": ((hx + wall, sy / 2.0, 0.0), (wall, sy / 2.0, hz + wall)),
        "front": ((0.0, sy / 2.0, -hz - wall), (hx, sy / 2.0, wall)),
        "back": ((0.0, sy / 2.0, hz + wall), (hx, sy / 2.0, wall)),
    }
    for name, (position, extent) in panels.items():
        g.node(f"{identity}.jacket.{name}", position,
               "vacuum-jacket-panel", half_extent_m=extent,
               material="stainless-steel",
               thermal_domain="shell",
               thermal_shell_thickness_m=0.003,
               thermal_group=f"{identity}.warm_jacket",
               interaction_surface="chamber-wall",
               # A through puncture in this composite wall opens a real
               # chamber boundary.  Damage owns the aperture geometry;
               # atmosphere and the jacket-vacuum circuit consume it.
               damage_opens_to=f"{identity}.chamber_volume",
               damage_crosses_volume=f"{identity}.jacket_vacuum")

    # The complete cold-head source lives on one service skid: battery,
    # closed-loop diaphragm compressor (including its radiator and fan),
    # working-gas makeup bottle, and condensate receiver.  This follows the
    # station powerplant pattern: each live source owns its own production
    # graph and the platform composes those graphs through declared ports.
    skid_x = hx + 0.72
    skid = f"{identity}.service_skid"
    g.node(skid, (skid_x, 0.10, 0.0), "load-bearing-structure",
           half_extent_m=(0.58, 0.06, 0.48), material="steel-plate",
           thermal_domain="volume",
           part_role="dewar-composite-service-platform")

    compressor_graph = compressor_spec.build_graph()
    compressor_shift = np.asarray((skid_x, 0.34, 0.0), float)
    for source in compressor_graph.nodes:
        node = dict(source)
        node["reference_position"] = list(
            np.asarray(node["reference_position"], float) + compressor_shift)
        node["assembly"] = identity
        if node["identity"] == f"{compressor_spec.identity}.motor":
            node.update({
                "electrical_port_model": "dc-motor-drive",
                "electrical_service": electrical_service.identity,
                "rated_shaft_w": compressor_shaft_w,
                "rated_electrical_power_w": compressor_electrical_w,
                "supply_voltage_v": battery_spec.nominal_v,
            })
        elif node["identity"] == f"{compressor_spec.identity}.fan":
            node.update({
                "electrical_port_model": "dc-fan-drive",
                "electrical_service": electrical_service.identity,
                "supply_voltage_v": battery_spec.nominal_v,
            })
        g.nodes.append(node)
    for source in compressor_graph.edges:
        edge = dict(source)
        edge["assembly"] = identity
        g.edges.append(edge)

    battery = f"{identity}.battery"
    gas = f"{identity}.working_gas_bottle"
    regulator = f"{identity}.makeup_regulator"
    tank = f"{identity}.drain_tank"
    drain = f"{identity}.floor_drain"
    g.node(battery, (skid_x - 0.34, 0.23, 0.31), "battery-pack",
           half_extent_m=(0.18, 0.14, 0.13), material="battery-module",
           thermal_domain="volume",
           chemistry=battery_spec.chemistry,
           nominal_voltage_v=battery_spec.nominal_v,
           capacity_ah=battery_spec.capacity_ah,
           state_of_charge=battery_spec.state_of_charge,
           series_packs=battery_spec.series_packs,
           electrical_port_model="battery-storage",
           electrical_service=electrical_service.identity,
           grounded_conductor_roles=("return", "protective-earth"),
           grounding_bond_resistance_ohm=0.0005)
    g.node(gas, (skid_x + 0.35, 0.32, 0.28), "high-pressure-gas-bottle",
           half_extent_m=(0.10, 0.28, 0.10), shape="drum",
           drum_axis=(0.0, 1.0, 0.0), drum_radius_m=0.10,
           drum_length_m=0.56, material="steel-plate",
           thermal_domain="shell",
           thermal_shell_thickness_m=0.004,
           fluid="working-gas", capacity_kg=working_gas_capacity_kg,
           held_kg=working_gas_initial_kg,
           pressure_pa=inlet_pressure_pa,
           part_role="working-gas-makeup-source")
    g.node(regulator, (skid_x + 0.22, 0.30, 0.08), "gas-pressure-regulator",
           half_extent_m=(0.045, 0.045, 0.045), material="brass",
           thermal_domain="volume",
           outlet_pressure_pa=return_pressure_pa)
    g.node(tank, (skid_x + 0.34, 0.22, -0.27), "condensate-receiver",
           half_extent_m=(0.16, 0.17, 0.16), material="polyethylene",
           thermal_domain="shell",
           thermal_shell_thickness_m=0.006,
           fluid="water", capacity_kg=drain_capacity_kg,
           held_kg=0.0,
           part_role="dewar-drain-receiver")
    g.node(drain, (0.0, 0.01, 0.0), "floor-drain",
           half_extent_m=(0.035, 0.015, 0.035), material="stainless-steel",
           thermal_domain="shell",
           thermal_shell_thickness_m=0.002,
           fluid="water", blocked_frac=0.0,
           part_role="chamber-condensate-drain")

    compressor = compressor_spec.identity
    return_port = f"{cold_head}.low_pressure_return"
    supply_port = f"{cold_head}.high_pressure_inlet"
    compressor_suction = f"{compressor}.suction"
    compressor_discharge = f"{compressor}.discharge"
    working_gas_loop = f"{identity}.working-gas-loop"
    # Composition turns the cold head and compressor's standalone circuit
    # names into one closed-loop identity.  Fluid discovery partitions by
    # both connectivity and circuit identity because unrelated media can meet
    # at one machine node; merely touching the same port cannot join them.
    cold_head_loop_edges = {
        f"{cold_head}.supply_line",
        f"{cold_head}.high_to_expander",
        f"{cold_head}.expander_to_cold_tip",
        f"{cold_head}.cold_tip_to_return",
        f"{cold_head}.return_line",
    }
    for edge in g.edges:
        if (edge["identity"] in cold_head_loop_edges
                or edge.get("circuit_identity") == f"{compressor}.working-gas"):
            edge["circuit_identity"] = working_gas_loop
    for port_identity in (return_port, supply_port):
        node = next(n for n in g.nodes if n["identity"] == port_identity)
        node["supplied_elsewhere"] = False
    # These are through-ports on a standalone cold head.  Once the source is
    # composed onto this platform they are internal loop connections.
    for edge_identity in (f"{cold_head}.supply_line",
                          f"{cold_head}.return_line"):
        edge = next(e for e in g.edges if e["identity"] == edge_identity)
        edge.pop("part_role", None)
    g.edge(f"{identity}.working_gas_return", return_port,
           compressor_suction, "pressure-rated-air-line", radius=0.012,
           circuit_identity=working_gas_loop,
           medium_rate_state="mass-flow-pressure-temperature")
    g.edge(f"{identity}.working_gas_supply", compressor_discharge,
           supply_port, "pressure-rated-air-line", radius=0.009,
           circuit_identity=working_gas_loop,
           medium_rate_state="mass-flow-pressure-temperature")
    g.edge(f"{identity}.makeup_bottle_line", gas, regulator,
           "pressure-rated-air-line", radius=0.005,
           circuit_identity=f"{identity}.working-gas-makeup")
    g.edge(f"{identity}.makeup_to_suction", regulator,
           compressor_suction, "pressure-rated-air-line", radius=0.004,
           circuit_identity=f"{identity}.working-gas-makeup")
    node_positions = {
        node["identity"]: np.asarray(node["reference_position"], dtype=float)
        for node in g.nodes
    }
    motor = f"{compressor}.motor"
    fan = f"{compressor}.fan"
    motor_cable = BuildingCable(
        f"{identity}.cable.battery_to_compressor", electrical_service,
        awg=0, length_m=float(np.linalg.norm(
            node_positions[battery] - node_positions[motor])),
        solid_copper=False, in_conduit=True,
        allowable_ampacity_a=150.0, parallel_sets=6)
    motor_breaker = CircuitBreaker(
        f"{identity}.breaker.compressor", electrical_service,
        800.0, motor_cable)
    g.edge(f"{identity}.battery_to_compressor", battery, motor,
           "insulated-copper-wire", radius=0.009,
           circuit_identity=electrical_service.identity,
           **motor_cable.graph_attributes(),
           **motor_breaker.graph_attributes())
    fan_cable = BuildingCable(
        f"{identity}.cable.compressor_fan", electrical_service,
        awg=14, length_m=float(np.linalg.norm(
            node_positions[motor] - node_positions[fan])),
        solid_copper=False, in_conduit=True,
        allowable_ampacity_a=15.0)
    fan_breaker = CircuitBreaker(
        f"{identity}.breaker.compressor_fan", electrical_service,
        15.0, fan_cable)
    fan_edge = next(
        edge for edge in g.edges
        if edge["identity"] == f"{compressor}.fan_power")
    fan_edge.update({
        "circuit_identity": electrical_service.identity,
        **fan_cable.graph_attributes(),
        **fan_breaker.graph_attributes(),
    })
    g.edge(f"{identity}.drain_line", drain, tank,
           "condensate-line", radius=0.012,
           circuit_identity=f"{identity}.condensate-drain")
    for index, body in enumerate((battery, gas, tank, f"{compressor}.motor"), 1):
        g.edge(f"{identity}.skid_mount_{index}", skid, body,
               "rigid-distance", radius=0.014,
               part_role="service-skid-equipment-mount")

    # The top is physically open.  This is the identity the chamber's
    # pressure-driven PartPort/HoleEmitter boundary binds to.
    g.node(f"{identity}.open_top", (0.0, sy, 0.0), "gas-service-port",
           half_extent_m=(0.01, 0.01, 0.01), mass_kg=0.0,
           render_primitive=False, structural_participation=False,
           thermal_domain="none",
           port_role="open-atmosphere", fluid="gas", closure="",
           radius_m=math.sqrt(open_area_m2 / math.pi),
           open_area_m2=open_area_m2)

    # The cold tip now spans y=sy-0.135 .. sy+0.035, so 13.5 cm of it is
    # inside the top-centre 25 cm voxel and the rest remains in the head.
    tip = next(n for n in g.nodes if n["identity"] == f"{cold_prefix}.cold_tip")
    tip["protrudes_into"] = f"{identity}.chamber_volume"
    tip["interaction_surface"] = "cold-tip"
    return g



def machine(identity: str = "lab.dewar", **parameters) -> Machine:
    graph = build(identity, **parameters).as_document()
    return Machine(
        identity=identity, label="open atmospheric dewar",
        power="battery-electric", medium="multi-service",
        production_graph=graph,
        note="battery, diaphragm compressor, cold head, chamber and drain")


def atmospheric_port(identity: str = "lab.dewar", *,
                     chamber_shape=(5, 5, 5), cell_size_m=0.25,
                     open_port_radius_m=0.003) -> PartPort:
    sy = float(chamber_shape[1]) * float(cell_size_m)
    return PartPort(
        identity=f"{identity}.open_top", part=identity, kind="breather",
        position=np.array([0.0, sy, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
        radius_m=float(open_port_radius_m), mating=False,
        fluid="gas", closure="")
