import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from station_cooling import (DEFAULT_CHANNELS, DesertHeatRejectorRuntime,
                             MetaConduitRuntime, SiteCoolingRuntime,
                             desert_heat_rejector)


def test_metaconduit_exchanges_heat_but_conserves_channel_energy():
    runtime = MetaConduitRuntime(channels=DEFAULT_CHANNELS[:3])
    temperatures = {"fuel": 330.0, "hydraulic-pressure": 300.0,
                    "hydraulic-return": 280.0}
    capacities = {name: 10_000.0 for name in temperatures}
    before = sum(temperatures[name] * capacities[name] for name in temperatures)
    after_t = runtime.step(10.0, temperatures, capacities)
    after = sum(after_t[name] * capacities[name] for name in temperatures)

    np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-8)
    assert after_t["fuel"] < temperatures["fuel"]
    assert after_t["hydraulic-return"] > temperatures["hydraulic-return"]


def test_potable_water_and_machine_coolant_exchange_heat_without_mixing():
    runtime = SiteCoolingRuntime(potable_water_temp_k=277.15,
                                 coolant_temp_k=315.15)
    result = runtime.step(1.0, refrigerant_cooling_w=12_000.0,
                          platform_heat_w=25_000.0, hcu_valve=1.0)

    assert result["water_to_coolant_heat_w"] > 0.0
    assert result["fluids_mixed"] is False
    assert runtime.coolant_temp_k < 315.15
    assert runtime.potable_water_l == 1000.0


def test_remote_rejector_spreads_heat_and_dilutes_exhaust():
    runtime = DesertHeatRejectorRuntime(core_count=8)
    result = runtime.step(
        0.1, ambient_k=320.0, coolant_in_k=355.0,
        coolant_heat_capacity_j_per_k=900_000.0,
        coolant_flow_command=1.0, exhaust_temp_k=780.0,
        exhaust_mass_flow_kg_s=5.6, blower_command=1.0)

    assert result["rejected_w"] > 0.0
    assert 320.0 < result["diffuser_outlet_temp_k"] < 780.0
    assert result["diffuser_mass_flow_kg_s"] > 5.6
    assert result["blower_electrical_w"] == 60_000.0


def test_remote_rejector_is_a_non_engine_machine_with_n_cores_and_blowers():
    graph = desert_heat_rejector(core_count=8).build_graph()
    roles = [node["part_role"] for node in graph["nodes"]]
    nodes = {node["identity"]: node for node in graph["nodes"]}
    degree = {node["identity"]: 0 for node in graph["nodes"]}
    for edge in graph["edges"]:
        a = nodes[edge["a"]].get("solver_condensed_into", edge["a"])
        b = nodes[edge["b"]].get("solver_condensed_into", edge["b"])
        degree[a] += 1
        degree[b] += 1

    assert roles.count("parallel-air-exchange-core") == 8
    assert roles.count("quiet-fine-control-core-blower") == 8
    assert not any(node["kind"] == "engine" for node in graph["nodes"])
    assert all(edge["constraint"] != "shaft-service-drive"
               for edge in graph["edges"])
    assert all(degree[node["identity"]] >= 3 for node in graph["nodes"]
               if node["part_role"] in {
                   "parallel-air-exchange-core",
                   "quiet-fine-control-core-blower",
               })
    long_fake_mounts = [edge for edge in graph["edges"]
                        if "blower_inlet_mount" in edge["identity"]]
    standoffs = [edge for edge in graph["edges"]
                 if "blower_flange_standoff" in edge["identity"]]
    positions = {node["identity"]: np.asarray(node["reference_position"])
                 for node in graph["nodes"]}
    assert not long_fake_mounts
    assert len(standoffs) == 8 * 4
    assert all(np.linalg.norm(positions[edge["b"]] - positions[edge["a"]])
               <= 0.081 for edge in standoffs)
    assert all(graph_node.get("solver_condensed_into")
               for graph_node in graph["nodes"]
               if graph_node.get("part_role") in {
                   "blower-flange-port", "core-blower-flange-port"})
