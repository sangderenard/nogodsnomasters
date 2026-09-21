import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from electrical_distribution import (
    BuildingCable,
    CircuitBreaker,
    DistributionPanel,
    OutletBox,
    standard_services,
)
from machines import Machine, MachineLine, MachinePart


def test_three_phase_cable_keeps_phase_neutral_and_earth_separate():
    service = standard_services("plant")["230-400v-3ph-50hz"]
    cable = BuildingCable("pump-feeder", service, awg=10, length_m=12.0,
                          allowable_ampacity_a=32.0)
    attributes = cable.graph_attributes()

    assert service.conductor_roles == (
        "line-1", "line-2", "line-3", "neutral", "protective-earth")
    assert [row["role"] for row in attributes["conductors"]] \
        == list(service.conductor_roles)
    assert all(row["material"] == "solid-copper"
               for row in attributes["conductors"])
    assert all(row["resistance_ohm"] > 0.0
               for row in attributes["conductors"])
    assert {row["endpoint"] for row in attributes["junctions"]} == {"a", "b"}
    assert all(row["torque_verified"] for row in attributes["junctions"])


def test_breaker_cannot_claim_more_current_than_installed_cable():
    service = standard_services()["230v-1ph-50hz"]
    cable = BuildingCable("branch", service, awg=14, length_m=8.0,
                          allowable_ampacity_a=15.0)

    with pytest.raises(ValueError, match="exceeds"):
        CircuitBreaker("oversized", service, 20.0, cable)


def test_panel_matches_outlet_by_complete_service_not_voltage_alone():
    services = standard_services("deck")
    ac = services["230-400v-3ph-50hz"]
    cable = BuildingCable("pump-cable", ac, awg=8, length_m=4.0,
                          allowable_ampacity_a=32.0)
    outlet = OutletBox(
        "pump-outlet", ac, CircuitBreaker("pump-breaker", ac, 32.0, cable),
        connector_standard="iec-60309-3p+n+e",
    )
    panel = DistributionPanel(
        "main-panel", services["48v-dc"], 400.0, (outlet,),
        neutral_to_frame_bond=True,
    )

    assert panel.compatible_outlets(ac) == (outlet,)
    assert not panel.compatible_outlets(services["277-480v-3ph-60hz"])
    assert outlet.graph_attributes()["breaker_poles"] == 3
    assert outlet.graph_attributes()["continuous_load_limit_a"] == 25.6
    assert outlet.graph_attributes()["terminal_roles"][-1] == "protective-earth"


def test_machine_line_preserves_electrical_cable_declaration():
    service = standard_services("machine")["48v-dc"]
    cable = BuildingCable("dc-feed", service, awg=4, length_m=2.0,
                          solid_copper=False, allowable_ampacity_a=80.0)
    machine = Machine(
        "m", "machine", parts=[
            MachinePart("source", "source", (0, 0, 0), (.1, .1, .1)),
            MachinePart("load", "load", (1, 0, 0), (.1, .1, .1)),
        ], lines=[
            MachineLine("feed", "source", "load",
                        "insulated-copper-wire", .01, "dc", "copper",
                        cable.graph_attributes()),
        ])

    edge = machine.build_graph()["edges"][0]
    assert edge["transport_domain"] == "electrical"
    assert edge["phase_arrangement"] == "dc-two-wire"
    assert len(edge["conductors"]) == 3


def test_parallel_sets_remain_distinct_conductors_on_common_service_buses():
    service = standard_services("machine")["48v-dc"]
    cable = BuildingCable(
        "parallel-feed", service, awg=0, length_m=1.5,
        solid_copper=False, allowable_ampacity_a=150.0,
        parallel_sets=6)
    breaker = CircuitBreaker("main", service, 800.0, cable)
    attributes = {**cable.graph_attributes(), **breaker.graph_attributes()}

    assert len(attributes["conductors"]) == 18
    assert {row["service_role"] for row in attributes["conductors"]} == {
        "positive", "return", "protective-earth"}
    assert cable.aggregate_ampacity_a == 900.0
    assert attributes["continuous_load_limit_per_conductor_a"] \
        == pytest.approx(800.0 / 1.25 / 6.0)
