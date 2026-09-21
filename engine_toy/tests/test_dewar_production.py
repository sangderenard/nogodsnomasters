import numpy as np
import pytest

from dewar_production import atmospheric_port, build, machine
from drivetrain_graph import (
    DrivetrainSolver, FluidCircuitInputs, _discover_fluid_circuits,
)
from engine_mesh import build_engine_mesh, merge_engine_meshes
from machines import MachineSim
from thermal_domains import (
    ShellThermalDomain,
    PathThermalDomain,
    ThermalInterface,
    VolumeThermalDomain,
    build_shell_domain,
    build_thermal_assembly,
    build_thermal_domains,
    build_volume_domain,
)
from vehicle_mesh import build_drivetrain_solid_parts
from voxel_ports import project_circular_port


def test_dewar_is_one_production_graph_with_closed_working_gas_loop():
    graph = build()
    node_ids = {node["identity"] for node in graph.nodes}
    edge_ids = {edge["identity"] for edge in graph.edges}

    assert "lab.dewar.cold_head.compressor.radiator" in node_ids
    assert "lab.dewar.cold_head.compressor.fan" in node_ids
    assert "lab.dewar.battery" in node_ids
    assert "lab.dewar.working_gas_bottle" in node_ids
    assert "lab.dewar.drain_tank" in node_ids
    assert "lab.dewar.working_gas_return" in edge_ids
    assert "lab.dewar.working_gas_supply" in edge_ids
    assert "lab.dewar.makeup_to_suction" in edge_ids

    circuits = _discover_fluid_circuits(graph.as_document())
    loop = [circuit for circuit in circuits
            if circuit.circuit_identity == "lab.dewar.working-gas-loop"]
    assert len(loop) == 1
    assert "lab.dewar.cold_head.expander" in loop[0].nodes
    assert "lab.dewar.cold_head.compressor.stage_1" in loop[0].nodes
    assert len(loop[0].edges) == 11
    assert "lab.dewar.cold_head.cold_tip" in loop[0].nodes
    assert {edge["identity"] for edge in loop[0].edges} >= {
        "lab.dewar.cold_head.expander_to_cold_tip",
        "lab.dewar.cold_head.cold_tip_to_return",
    }
    electrical = [edge for edge in graph.edges
                  if edge.get("transport_domain") == "electrical"]
    assert {edge["identity"] for edge in electrical} == {
        "lab.dewar.battery_to_compressor",
        "lab.dewar.cold_head.compressor.fan_power",
    }
    main = next(edge for edge in electrical
                if edge["identity"] == "lab.dewar.battery_to_compressor")
    assert main["parallel_sets"] == 6
    assert len(main["conductors"]) == 18
    assert main["breaker_rating_a"] == 800.0


def test_dewar_drain_and_vacuum_service_are_fluid_circuits():
    graph = build().as_document()
    circuits = _discover_fluid_circuits(graph)
    by_identity = {circuit.circuit_identity: circuit for circuit in circuits}
    assert by_identity["lab.dewar.condensate-drain"].kind_class == "thermal-liquid"
    vacuum = by_identity["lab.dewar.cold_head.vacuum-service"]
    assert vacuum.kind_class == "compressible-gas"
    jacket = by_identity["lab.dewar.jacket-vacuum"]
    assert jacket.kind_class == "compressible-gas"
    assert {"lab.dewar.jacket_vacuum_top_port",
            "lab.dewar.jacket_vacuum_bottom_port"} <= jacket.nodes
    nodes = {node["identity"]: node for node in graph["nodes"]}
    assert nodes["lab.dewar.jacket_vacuum"]["fluid_volume_l"] > 0.0
    assert nodes["lab.dewar.jacket_vacuum_top_port"]["valve_open_fraction"] == 0.0
    assert nodes["lab.dewar.jacket_vacuum_bottom_port"]["valve_open_fraction"] == 0.0
    assert [volume.identity for volume in jacket.volumes] == [
        "lab.dewar.jacket_vacuum"]
    assert jacket.pipe_volume_l > 0.0


def test_powered_working_gas_moves_through_pipes_and_cools_the_real_tip():
    graph = build().as_document()
    solver = DrivetrainSolver(graph)
    fluid = solver.fluid_system
    motor = "lab.dewar.cold_head.compressor.motor"
    fan = "lab.dewar.cold_head.compressor.fan"
    tip = "lab.dewar.cold_head.cold_tip"
    radiator = "lab.dewar.cold_head.compressor.radiator"
    fluid.stage(FluidCircuitInputs(
        electrical_power_w={motor: 28_500.0, fan: 420.0},
        thermal_temperature_k={tip: 293.15, radiator: 293.15},
    ))

    ok, _metrics, _state = fluid.advance(0.02)
    loop = next(circuit for circuit in fluid.circuits
                if circuit.process_resolved)

    assert ok
    assert loop.delivered_flow_kg_s == pytest.approx(0.05)
    assert all(value == pytest.approx(0.05)
               for value in loop.edge_mass_flow_kg_s.values())
    assert loop.node_pressure_pa["lab.dewar.cold_head.recuperator_high"] > (
        loop.node_pressure_pa["lab.dewar.cold_head.recuperator_return"])
    assert loop.node_temperature_k["lab.dewar.cold_head.expander"] < 293.15
    assert fluid.thermal_source_w()[tip] < 0.0
    assert fluid.component_power_w["compressor_shaft_input_w"] > 0.0
    assert fluid.component_power_w["expander_shaft_output_w"] > 0.0


def test_unpowered_graph_does_not_manufacture_a_cold_source():
    graph = build().as_document()
    fluid = DrivetrainSolver(graph).fluid_system
    tip = "lab.dewar.cold_head.cold_tip"
    radiator = "lab.dewar.cold_head.compressor.radiator"
    fluid.stage(FluidCircuitInputs(
        thermal_temperature_k={tip: 293.15, radiator: 293.15},
    ))

    fluid.advance(0.02)
    loop = next(circuit for circuit in fluid.circuits
                if circuit.process_resolved)

    assert loop.delivered_flow_kg_s == 0.0
    assert fluid.component_power_w["compressor_shaft_input_w"] == 0.0
    assert fluid.component_power_w["cold_tip_heat_removed_w"] == 0.0
    assert fluid.thermal_source_w()[tip] == 0.0


def test_dewar_is_a_machine_description_advanced_by_machine_sim():
    authored = machine()
    assert not hasattr(authored, "step")
    sim = MachineSim(machine=authored)
    assert sim.graph["identity"] == "lab.dewar/machine"
    checkpoint = sim.snapshot()
    assert sim.step(0.125) == {}
    assert sim.elapsed_s == pytest.approx(0.125)
    sim.restore(checkpoint)
    assert sim.elapsed_s == 0.0


def test_open_top_is_the_same_declared_port_in_graph_and_part_api():
    graph = build().as_document()
    node = next(node for node in graph["nodes"]
                if node["identity"] == "lab.dewar.open_top")
    port = atmospheric_port()
    assert node["port_role"] == "open-atmosphere"
    assert node["open_area_m2"] == pytest.approx(np.pi * port.radius_m ** 2)
    assert port.identity == node["identity"]
    assert port.fluid == "gas"


def test_one_port_can_span_voxel_faces_without_multiplying_its_area():
    radius = 0.08
    shares = project_circular_port(
        (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), radius,
        shape=(4, 4, 4), bounds_min_m=(-0.5, 0.0, -0.5),
        bounds_max_m=(0.5, 1.0, 0.5))

    assert len(shares) == 4
    assert sum(share.area_fraction for share in shares) == pytest.approx(1.0)
    assert sum(share.area_m2 for share in shares) == pytest.approx(
        np.pi * radius ** 2)
    assert len({share.voxel for share in shares}) == 4


def test_machine_damage_keeps_the_fluid_system_circuit_identity_on_restore():
    authored = machine()
    sim = MachineSim(machine=authored)
    circuits = _discover_fluid_circuits(sim.graph)
    sim.bind_fluid_circuits(circuits)
    checkpoint = sim.snapshot()

    sim.elapsed_s = 2.0
    sim.restore(checkpoint)

    assert sim.elapsed_s == 0.0
    assert sim._fluid_circuits[0] is circuits[0]


def test_dewar_mesh_preserves_declared_machine_thermal_groups():
    graph = build().as_document()
    static, moving = build_engine_mesh(graph)
    mesh = merge_engine_meshes([static, moving])
    groups = dict(zip(mesh.part_names, mesh.part_groups))

    assert groups["node_lab_dewar_cold_head_cold_tip"] == "lab.dewar.cold_head.cold_end"
    assert groups["node_lab_dewar_cold_head_compressor_stage_1"] == "lab.dewar.cold_head.compressor.compressor"
    assert groups["node_lab_dewar_cold_head_compressor_radiator"] == "lab.dewar.cold_head.compressor.radiator"


def test_every_dewar_object_declares_its_thermal_domain():
    graph = build().as_document()
    missing = [node["identity"] for node in graph["nodes"]
               if node.get("thermal_domain") not in {"none", "shell", "volume"}]
    assert missing == []
    domains = {node["identity"]: node["thermal_domain"] for node in graph["nodes"]}
    assert domains["lab.dewar.chamber_volume"] == "volume"
    assert domains["lab.dewar.jacket_vacuum"] == "none"
    assert domains["lab.dewar.jacket.floor"] == "shell"
    assert domains["lab.dewar.cold_head.cold_tip"] == "volume"
    assert domains["lab.dewar.cold_head.compressor.radiator"] == "shell"


def test_shell_and_volume_domains_annihilate_uniform_temperature():
    graph = build().as_document()
    nodes = {node["identity"]: node for node in graph["nodes"]}
    parts = {part.name: part for part in build_drivetrain_solid_parts(graph)}

    panel_id = "lab.dewar.jacket.floor"
    panel = build_shell_domain(nodes[panel_id], parts["node_lab_dewar_jacket_floor"])
    assert not panel.geometry.invalid_vertex_mask.any()
    np.testing.assert_allclose(panel.laplace(np.full(len(panel.vertices), 280.0)), 0.0,
                               atol=1e-10)

    tip_id = "lab.dewar.cold_head.cold_tip"
    tip = build_volume_domain(nodes[tip_id], default_resolution=3)
    assert tip.laplacian.shape == (np.prod(tip.shape),) * 2
    np.testing.assert_allclose(tip.laplace(np.full(tip.shape, 80.0)), 0.0, atol=1e-10)


def test_complete_dewar_builds_declared_thermal_domains():
    domains = build_thermal_domains(build().as_document(), default_volume_resolution=3)
    assert len(domains) == 48
    assert sum(isinstance(value, ShellThermalDomain) for value in domains.values()) == 13
    assert sum(isinstance(value, VolumeThermalDomain) for value in domains.values()) == 14
    assert sum(isinstance(value, PathThermalDomain) for value in domains.values()) == 21
    assert domains["lab.dewar.chamber_volume"].shape == (5, 5, 5)


def test_shell_volume_interface_exchange_conserves_energy():
    panel = "lab.dewar.jacket.floor"
    tip = "lab.dewar.cold_head.cold_tip"
    assembly = build_thermal_assembly(
        build().as_document(),
        temperatures_by_group={
            "lab.dewar.warm_jacket": 293.15,
            "lab.dewar.cold_head.cold_end": 80.0,
        },
        interfaces=[ThermalInterface(panel, tip, 12.0)],
        default_volume_resolution=3,
    )
    before = assembly.energy_j
    hot_before = assembly.temperature_k(panel)
    cold_before = assembly.temperature_k(tip)
    result = assembly.step(1.0)

    assert result["energy_j"] == pytest.approx(before, rel=1e-12)
    assert assembly.temperature_k(panel) < hot_before
    assert assembly.temperature_k(tip) > cold_before
