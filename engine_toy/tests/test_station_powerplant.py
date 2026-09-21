import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from station_powerplant import (PMG_RATED_W, DropInBarrelFuelPump,
                                FuelControlUnit, PowerplantBasic,
                                PowerplantRuntime, shared_storage_ballast)


def _package():
    return PowerplantBasic("powerplant.test", "test-engine", (3.0, .52, 0.0),
                           (.62, .48, .58))


def test_powerplant_basic_is_a_ported_thermal_machine_with_replaceable_envelope():
    package = _package()
    graph = package.machine().build_graph()
    roles = {node["part_role"] for node in graph["nodes"]}

    assert package.accepts((.55, .45, .40))
    assert not package.accepts((.70, .45, .40))
    assert roles == {"power-input", "hydraulic-pump", "pneumatic-compressor",
                     "shaft-electrical-generator", "electrical-storage",
                     "electrical-power-converter",
                     "electrical-distribution-panel", "electrical-receptacle",
                     "refrigerant-compressor",
                     "platform-engine-coolant-reservoir",
                     "platform-engine-coolant-assist-pump"}
    assert all(node["mass_kg"] > 0.0 for node in graph["nodes"])
    assert all(node["thermal_capacity_j_k"] > 0.0 for node in graph["nodes"])
    assert all(node["ports"] for node in graph["nodes"])
    assert {edge["constraint"] for edge in graph["edges"]} == {
        "shaft-service-drive", "insulated-copper-wire", "electrical-conduit",
        "coolant-line"}
    generator = next(node for node in graph["nodes"]
                     if node["part_role"] == "shaft-electrical-generator")
    assert generator["rated_w"] == 30_000.0
    assert generator["raw_ac_rated_w"] == 35_000.0
    assert generator["raw_winding"] == "three-phase-variable-frequency"
    assert generator["rated_speed_range_rpm"] == [2500.0, 3600.0]
    assert generator["full_load_efficiency"] == .91
    assert generator["specification_source"].startswith("https://www.meccalte.com/")

    outlets = [node for node in graph["nodes"]
               if node["part_role"] == "electrical-receptacle"]
    assert {node["phase_arrangement"] for node in outlets} == {
        "dc-two-wire", "single-phase", "split-phase", "three-phase-wye"}
    assert all(node["breaker_rating_a"] > 0.0 for node in outlets)
    electrical = [edge for edge in graph["edges"]
                  if edge.get("transport_domain") == "electrical"]
    assert electrical
    assert all(edge.get("conductors") for edge in electrical)
    conduits = [edge for edge in graph["edges"]
                if edge["constraint"] == "electrical-conduit"]
    assert conduits and all(edge["load_bearing"] for edge in conduits)


def test_powerplant_runtime_makes_no_power_stopped_and_runs_all_shaft_services():
    runtime = PowerplantRuntime(_package())
    stopped = runtime.step(.1, 0.0, hydraulic_command=1.0,
                           refrigeration_command=1.0)
    running = runtime.step(.1, 3000.0, electrical_load_w=10_000.0,
                           hydraulic_command=1.0,
                           pneumatic_demand_l_min=500.0,
                           refrigeration_command=1.0)

    assert stopped["generator_w"] == 0.0
    assert stopped["hydraulic_flow_l_min"] == 0.0
    assert running["generator_capacity_w"] == PMG_RATED_W * .91
    assert 10_000.0 < running["generator_w"] < running["generator_capacity_w"]
    assert running["hydraulic_flow_l_min"] > 0.0
    assert running["pneumatic_served_l_min"] == 500.0
    assert running["refrigerant_shaft_w"] > 0.0
    assert running["battery_soc"] > stopped["battery_soc"]


def test_shared_storage_is_mass_balanced_and_each_port_names_a_bay_consumer():
    graph = shared_storage_ballast().build_graph()
    nodes = graph["nodes"]
    mass = sum(node["mass_kg"] for node in nodes)
    cg = sum(np.asarray(node["reference_position"])[[0, 2]] * node["mass_kg"]
             for node in nodes) / mass

    np.testing.assert_allclose(cg, (0.0, 0.0), atol=1e-12)
    assert len(nodes) == 6
    assert all("fuel" not in node["identity"] for node in nodes)
    assert all(len(node["ports"]) == 2 for node in nodes)
    assert {port["connected_to"].split(".")[1]
            for node in nodes for port in node["ports"]} == {"port", "starboard"}
    # Vessel tops stay below y=-0.42, leaving more than 250 mm below the
    # lowest drum feature in the reference assembly (y=-0.1377 m).
    assert max(node["reference_position"][1] + node["body_half_extent_m"][1]
               for node in nodes) <= -0.42 + 1e-12


def test_fcu_stages_only_compatible_nonempty_enabled_barrel_pumps():
    pumps = (DropInBarrelFuelPump("p0", "b0", "diesel", rated_flow_kg_s=.2),
             DropInBarrelFuelPump("p1", "b1", "diesel", rated_flow_kg_s=.2),
             DropInBarrelFuelPump("pj", "bj", "jet-a", rated_flow_kg_s=.2))
    fcu = FuelControlUnit("diesel", pumps)
    fcu.enabled["p1"] = False
    command = fcu.command(.3, {"b0": .8, "b1": .8, "bj": .8})
    assert command == {"p0": .2, "p1": 0.0, "pj": 0.0}
    # Empty/dry-run cutoff is automatic even if the FCU enable remains on.
    assert fcu.command(.1, {"b0": 0.0, "b1": .8, "bj": .8}) == {
        "p0": 0.0, "p1": 0.0, "pj": 0.0}
    rolled_up = DropInBarrelFuelPump("p2", "b2", "diesel",
                                     rated_flow_kg_s=.2)
    fcu.attach(rolled_up)
    assert fcu.command(.1, {"b0": 0.0, "b1": .8, "bj": .8, "b2": .7})["p2"] == .1
    fcu.detach("p2")
    assert all(p.identity != "p2" for p in fcu.pumps)


def test_station_engine_graph_uses_external_barrels_not_synthetic_tank():
    import engines
    from drivetrain_graph import build_drivetrain_graph
    engine = engines.get("ldt465-multifuel-deuce")
    supply = {
        "reservoirs": (
            {"identity": "site.barrel.0", "capacity_l": 189.2705892},
            {"identity": "site.barrel.1", "capacity_l": 189.2705892}),
        "pumps": (
            {"identity": "site.barrel.0.cap_pump",
             "reservoir_identity": "site.barrel.0",
             "pump_kind": "electric", "flow_capacity_kg_s": .18},
            {"identity": "site.barrel.1.cap_pump",
             "reservoir_identity": "site.barrel.1",
             "pump_kind": "electric", "flow_capacity_kg_s": .18}),
        "inlet_identity": "installed.engine.external_fuel_in",
    }
    graph = build_drivetrain_graph(engine, external_fuel_supply=supply)
    nodes = {n["identity"]: n for n in graph["nodes"]}
    assert "fuel.tank" not in nodes
    assert {"site.barrel.0", "site.barrel.1", "site.barrel.0.cap_pump",
            "site.barrel.1.cap_pump",
            "installed.engine.external_fuel_in"} <= set(nodes)
    assert not any("pump.bank" in name for name in nodes)
    assert all(nodes[name]["externally_authored"]
               for name in ("site.barrel.0", "site.barrel.1"))
    fuel_edges = [e for e in graph["edges"]
                  if e.get("circuit_identity") == "fuel"]
    assert sum(e.get("flow_capacity_kg_s", 0.0) for e in fuel_edges) == .36
    pairs = {(e["a"], e["b"]) for e in fuel_edges}
    assert ("site.barrel.0.cap_pump",
            "installed.engine.external_fuel_in") in pairs
    assert ("site.barrel.1.cap_pump",
            "installed.engine.external_fuel_in") in pairs
    assert ("installed.engine.external_fuel_in",
            "powertrain.fuel_rail") in pairs
