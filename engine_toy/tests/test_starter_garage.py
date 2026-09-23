"""Starter garage content must use the game's objects, not a mini-engine."""
import inspect

from air_volumes import AirVolume
from machines import Machine, MachineSim
import starter_garage as garage
import starter_garage_catalogue as catalogue


def test_catalogue_contract_is_retained():
    assert len(catalogue.INVENTORY) == 162
    assert len(catalogue.CONSTITUENTS) == 40
    assert sum(entry.quantity for entry in catalogue.INVENTORY) == 314


def test_garage_build_returns_only_existing_engine_object_types():
    built = garage.build_starter_garage("objects")
    assert built["machines"]
    assert all(isinstance(machine, Machine) for machine in built["machines"])
    assert len(built["air_volumes"]) == 1
    assert isinstance(built["air_volumes"][0], AirVolume)


def test_every_catalogue_entry_is_an_actual_machine_or_machine_part():
    built = garage.build_starter_garage("coverage")
    represented = set()
    for machine in built["machines"]:
        graph = machine.build_graph()
        represented.update(node.get("catalogue_entry") for node in graph["nodes"]
                           if node.get("catalogue_entry"))
        assert not any(node.get("kind") == "inventory-object" for node in graph["nodes"])
        assert not any(node.get("fixed_to") == "starting-placement" for node in graph["nodes"])
    represented.add("garage.room-air")
    assert represented == {entry.identity for entry in catalogue.INVENTORY}


def test_every_loose_tool_and_stock_instance_is_a_machine():
    built = garage.build_starter_garage("tools")
    machine_entries = []
    for machine in built["machines"]:
        for node in machine.build_graph()["nodes"]:
            entry = node.get("catalogue_entry")
            if entry:
                machine_entries.append((machine.identity, entry))
    for category in ("hand-tools", "measuring", "power-and-cleanup",
                     "dryer-spares", "stock-and-clutter", "safety-and-reference"):
        for entry in (row for row in catalogue.INVENTORY if row.category == category):
            owners = {identity for identity, key in machine_entries if key == entry.identity}
            assert len(owners) == entry.quantity


def test_module_has_no_parallel_runtime_or_physics_classes():
    source = inspect.getsource(garage)
    forbidden = ("class StarterGarage", "class ExhaustPassage", "class LintDeposit",
                 "class MaterialReceiver", "class TextileLoad", "class GarageDoorState",
                 "def snapshot", "def restore", "def pressure_loss_pa",
                 "def delivered_flow_m3_s")
    assert not [token for token in forbidden if token in source]


def test_fault_space_is_exact_and_faults_land_on_real_dryer_components():
    assert len(garage.PERMITTED_DRYER_FAULT_SETS) == 47
    first = garage.build_dryer_machine("same")
    second = garage.build_dryer_machine("same")
    assert garage.dryer_faults_for_seed("same") == garage.dryer_faults_for_seed("same")
    for machine in (first, second):
        graph = machine.build_graph()
        faulted = [part for part in (*graph["nodes"], *graph["edges"])
                   if part.get("generated_fault")]
        assert {part["starting_condition"] for part in faulted} == set(
            garage.dryer_faults_for_seed("same"))
        assert all(part["identity"].startswith("laundry.dryer.") for part in faulted)


def test_dryer_uses_existing_cabinet_and_real_process_air_circuit():
    graph = garage.build_dryer_machine(8).build_graph()
    assert any(node["identity"] == "laundry.dryer.drum" for node in graph["nodes"])
    assert any(edge.get("part_role") == "drum-roller" for edge in graph["edges"])
    air = [edge for edge in graph["edges"]
           if edge.get("circuit_identity") == "laundry.dryer.process-air"]
    assert len(air) == 3
    assert all(edge["constraint"] == "air-line" for edge in air)
    assert any(node["identity"] == "laundry.dryer.door_switch" for node in graph["nodes"])
    assert any(node["identity"] == "laundry.dryer.bottom.thermal_fuse" for node in graph["nodes"])


def test_building_vent_is_a_separate_machine_using_existing_fouling_and_fluid_vocabularies():
    machine = garage.build_exhaust_machine()
    graph = machine.build_graph()
    assert machine.identity == "garage.dryer-exhaust"
    assert len({edge["circuit_identity"] for edge in graph["edges"]}) == 1
    assert all(edge["constraint"] == "air-line" for edge in graph["edges"])
    deposit = next(node for node in graph["nodes"]
                   if node.get("catalogue_entry") == "vent.deposit")
    assert deposit["fouling_mode"] == "core-debris"
    assert deposit["blocked_frac"] is None
    assert deposit["fouling_calibration_required"]
    assert deposit["contained_by"] == machine.identity
    from drivetrain_graph import _discover_fluid_circuits
    circuits = _discover_fluid_circuits(graph)
    assert len(circuits) == 1
    assert circuits[0].circuit_identity == "garage.dryer-exhaust-air"


def test_machine_sim_owns_dryer_snapshot_and_damage_state():
    sim = MachineSim(garage.build_dryer_machine(9))
    target = "laundry.dryer.door_switch"
    snapshot = sim.snapshot()
    sim.state.absent_parts.add(target)
    sim.restore(snapshot)
    assert target not in sim.state.absent_parts
    assert target in sim.state.part_damage


def test_door_release_is_a_real_coupling_between_real_parts():
    graph = garage.build_overhead_door_machine().build_graph()
    coupling = next(edge for edge in graph["edges"]
                    if edge["identity"] == "garage.overhead-door.trolley-coupling")
    assert coupling["constraint"] == "direct-drive-lockup"
    assert coupling["initially_engaged"]
    assert coupling["releasable_by"].startswith("garage.overhead-door.door.release-cord")
    opener = next(node for node in graph["nodes"]
                  if node.get("catalogue_entry") == "door.opener")
    assert opener["drive_condition"] == "mechanically-immobilized"


def test_meter_is_one_real_machine_in_the_authored_location():
    built = garage.build_starter_garage("meter")
    meter = next(machine for machine in built["machines"]
                 if machine.identity == "starter-garage.measure.multimeter")
    graph = meter.build_graph()
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["starting_location"] == "instrument pouch"


def test_electrical_parts_use_existing_hardware_machine_builders():
    machines = garage.build_electrical_machines()
    panel = next(machine for machine in machines if machine.identity == "garage.panel")
    graph = panel.build_graph()
    handles = [node for node in graph["nodes"] if node.get("part_role") == "breaker-handle"]
    assert any(node.get("breaker_identity") == "garage.panel.DRYER" for node in handles)
    dryer = next(machine for machine in machines if machine.identity == "garage.dryer-receptacle")
    assert {node["terminal_role"] for node in dryer.build_graph()["nodes"]} == {
        "line-1", "line-2", "neutral", "protective-earth"}
