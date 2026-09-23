"""Run against the real workspace, never a fake replacement for machines.py.

Skipped when the complete upstream imports are unavailable. A skip is an
unverified integration boundary, NOT a passing runtime test.
"""
from pathlib import Path
import sys
import json
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def real_workspace():
    try:
        import machines
        import assembly_ports
        import dc_power
    except ModuleNotFoundError as error:
        pytest.skip(f"Real engine-toy/Turing workspace not available: {error}")
    return machines


def checked(machine):
    graph = machine.build_graph()
    json.dumps(graph,allow_nan=False)
    nodes = {n["identity"] for n in graph["nodes"]}
    assert len(nodes) == len(graph["nodes"])
    assert len({e["identity"] for e in graph["edges"]}) == len(graph["edges"])
    assert all(e["a"] in nodes and e["b"] in nodes for e in graph["edges"])
    return graph


def test_all_connectors_build_real_machine_graphs():
    machines=real_workspace()
    from hardware_catalogue import connector_catalogue
    from electrical_hardware import build_connector
    for spec in connector_catalogue().values():
        machine=build_connector(spec)
        assert isinstance(machine,machines.Machine)
        graph=checked(machine)
        assert len(graph["nodes"]) == len(spec.contacts)+1
        body = next(node for node in graph["nodes"]
                    if node["part_role"] == "electrical-connector-body")
        assert body["electrical_service"] == spec.details["electrical_service"].value["identity"]
        for node in graph["nodes"]:
            if node["part_role"] == "electrical-contact":
                assert node["electrical_bus_identity"].startswith(
                    f"{body['identity']}::{body['electrical_service']}::")


def test_all_cables_build_real_machine_graphs():
    real_workspace()
    from hardware_catalogue import cable_catalogue
    from electrical_hardware import build_cable
    for spec in cable_catalogue().values():
        graph=checked(build_cable(spec,points=((0,0,0),(1,0,0),(1,2,0))))
        edge=graph["edges"][0]
        assert edge["cable_length_m"] == 3
        assert edge["load_bearing"] is False
        if spec.identity == "home.data.1583a":
            assert edge["transport_domain"] == "signal-electrical"
            assert len(edge["unresolved_electrical_cores"]) == 8


def test_power_cables_feed_existing_electrical_and_thermal_abis():
    real_workspace()
    from hardware_catalogue import cable_catalogue
    from electrical_hardware import build_cable
    from thermal_domains import electrical_conductor_thermal_nodes
    for spec in cable_catalogue().values():
        graph = checked(build_cable(spec))
        edge = graph["edges"][0]
        if edge["transport_domain"] == "signal-electrical":
            assert edge["conductors"] == []
            assert electrical_conductor_thermal_nodes(graph) == []
            continue
        assert edge["electrical_service"] == spec.details["electrical_service"].value["identity"]
        assert len(edge["conductors"]) == len(spec.cores)
        assert all(row["resistance_ohm_at_20c"] > 0 for row in edge["conductors"])
        assert len(electrical_conductor_thermal_nodes(graph)) == len(spec.cores)


def test_duplex_has_independent_tabs_and_unswitched_ground_strap():
    real_workspace()
    from electrical_hardware import build_duplex
    graph=checked(build_duplex())
    links=[e for e in graph["edges"] if e.get("part_role") in {"breakoff-tab","ground-strap"}]
    assert len(links)==3
    assert sum(e["removable"] for e in links)==2


def test_panel_schedule_points_to_real_handles_and_terminals():
    real_workspace()
    from electrical_hardware import build_breaker_panel
    graph=checked(build_breaker_panel(neutral_to_frame_bond=True))
    ids={n["identity"] for n in graph["nodes"]}
    schedule=graph["nodes"][0]["panel_schedule"]
    assert all(row["handle_identity"] in ids for row in schedule)
    assert all(t in ids for row in schedule for t in row["output_terminals"])
    assert any(e.get("bond_purpose")=="protective-earth-to-enclosure" for e in graph["edges"])
    assert not any("neutral" in e["identity"] and e.get("contact_state_owner") for e in graph["edges"])


def test_patch_panel_has_192_separate_links():
    real_workspace()
    from electrical_hardware import build_patch_panel
    graph=checked(build_patch_panel())
    assert len(graph["edges"])==192
    endpoints=[(e["terminal_a_identity"],e["terminal_b_identity"]) for e in graph["edges"]]
    assert len(set(endpoints))==192


def test_cord_binds_by_role_and_preserves_all_layers():
    real_workspace()
    from dataclasses import replace
    from hardware_catalogue import cable_catalogue, class_l_32_12
    from electrical_hardware import build_cord
    a,b=class_l_32_12("plug"),class_l_32_12()
    b=replace(b,contacts=tuple(reversed(b.contacts)))
    graph=checked(build_cord(cable_catalogue()["military.flex.6-5"],a,b,
                            points=(p for p in ((0,0,0),(1,0,0)))))
    run=next(e for e in graph["edges"] if e.get("cable_length_m"))
    assert run["terminal_map_a"]["neutral"].endswith("::neutral")
    assert run["terminal_map_b"]["neutral"].endswith("::neutral")
    assert run["contact_identity_map_a"]["neutral"].endswith(".N")
    assert run["contact_identity_map_b"]["neutral"].endswith(".N")
    assert run["a"] == "service-cord.a.body"
    assert run["b"] == "service-cord.b.body"
    assert len(run["physical_construction"]["layers"])==3


def test_graphs_respect_the_existing_electrical_registration_boundary():
    real_workspace()
    spectral = ROOT.parent / "spectral-analyzer"
    sys.path.insert(0, str(spectral))
    from electrical_tensor_network import ElectricalDeviceRegistry
    from machine_electrical_graph import (
        audit_machine_electrical_graph, register_machine_electrical_graph)
    from hardware_catalogue import cable_catalogue, class_l_32_12
    from electrical_hardware import (
        build_breaker_panel, build_cord, build_duplex, build_patch_panel)

    cable = cable_catalogue()["military.flex.6-5"]
    cord = build_cord(cable, class_l_32_12("plug"), class_l_32_12())
    cord_graph = cord.build_graph()
    for machine in (cord, build_breaker_panel(neutral_to_frame_bond=True),
                    build_duplex(), build_patch_panel()):
        assert audit_machine_electrical_graph(machine.build_graph()).complete

    registry = ElectricalDeviceRegistry([0.0, 60.0])
    register_machine_electrical_graph(registry, cord_graph, device_laws={})
    run = next(edge for edge in cord_graph["edges"] if edge.get("cable_length_m"))
    neutral = next(branch for branch in registry.branches
                   if branch.identity.endswith("::neutral"))
    assert neutral.a == run["terminal_map_a"]["neutral"]
    assert neutral.b == run["terminal_map_b"]["neutral"]
