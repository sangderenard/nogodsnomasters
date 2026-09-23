"""Starter-garage authoring through the existing game object systems.

There is deliberately no garage runtime, inventory ledger, pressure solver,
door state machine, material receiver, or snapshot implementation here.
Objects are existing ``Machine``/``MachinePart``/``MachineLine`` instances;
the room is the existing ``AirVolume``; the appliances are the existing
cabinet production graphs. ``MachineSim`` and the managed game systems own
runtime state exactly as they do for every other machine.

For present authoring convenience every loose tool and stock item is a
Machine. Unknown dimensions and masses remain explicitly unresolved rather
than acquiring made-up physics.
"""
from __future__ import annotations

import hashlib
import itertools
import random
import re

from air_volumes import AirVolume
from electrical_hardware import BreakerPosition, build_breaker_panel, build_duplex
from machines import Machine, MachineLine, MachinePart
from starter_garage_catalogue import CONSTITUENTS, INVENTORY, InventoryEntry, catalogue_document


DRYER_FAULTS = {
    "drive": ("broken-belt", "high-drag-idler", "high-drag-drum-roller"),
    "control": ("failed-open-door-switch", "misaligned-latch-striker",
                "disconnected-control-connector"),
    "thermal": ("open-thermal-fuse", "open-heater-element"),
}


def permitted_dryer_fault_sets() -> tuple[tuple[str, ...], ...]:
    rows = []
    families = tuple(DRYER_FAULTS)
    for count in range(1, len(families) + 1):
        for selected in itertools.combinations(families, count):
            rows.extend(itertools.product(*(DRYER_FAULTS[name] for name in selected)))
    return tuple(rows)


PERMITTED_DRYER_FAULT_SETS = permitted_dryer_fault_sets()
if len(PERMITTED_DRYER_FAULT_SETS) != 47:
    raise RuntimeError("the fixed dryer model must have exactly 47 fault sets")


def _seed(value: int | str) -> int:
    if isinstance(value, bool):
        raise TypeError("a boolean is not a world seed")
    if isinstance(value, int):
        return value
    return int.from_bytes(hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "big")


def dryer_faults_for_seed(seed: int | str) -> tuple[str, ...]:
    rng = random.Random(_seed(seed))
    return PERMITTED_DRYER_FAULT_SETS[rng.randrange(len(PERMITTED_DRYER_FAULT_SETS))]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "part"


def _entry_attributes(entry: InventoryEntry, index: int) -> dict:
    return {
        "catalogue_entry": entry.identity,
        "catalogue_instance": index,
        "starting_location": entry.location,
        "starting_condition": entry.state,
        "inventory_group": entry.group,
        "pickable": True,
        "geometry_fidelity": "unresolved-authoring-body",
        "mass_unresolved": True,
        "dimensions_unresolved": True,
    }


def _unresolved_part(identity: str, kind: str, attributes: dict, position=(0.0, 0.0, 0.0)):
    """A real MachinePart whose absent product data is declared, not guessed."""
    return MachinePart(identity, kind, position, (0.01, 0.01, 0.01), mass_kg=0.0,
                       material="unresolved-material", part_role=kind,
                       attributes=dict(attributes))


def _constituents(owner: str, machine_identity: str) -> list[MachinePart]:
    parts = []
    records = [record for record in CONSTITUENTS if record.owner == owner]
    for record_index, record in enumerate(records):
        for part_index, label in enumerate(record.parts):
            parts.append(_unresolved_part(
                f"{machine_identity}.constituent.{_slug(record.identity)}.{part_index + 1}",
                _slug(label),
                {"constituent_record": record.identity, "constituent_owner": owner,
                 "display_name": label, "reference_note": record.note},
                (0.0, 0.0, 0.002 * (record_index + part_index))))
    return parts


def build_loose_machine(entry: InventoryEntry, index: int) -> Machine:
    """Represent one independently handleable catalogue object as a Machine."""
    suffix = "" if entry.quantity == 1 else f".{index + 1}"
    identity = f"starter-garage.{entry.identity}{suffix}"
    primary = _unresolved_part(f"{identity}.body", _slug(entry.label),
                               _entry_attributes(entry, index))
    return Machine(identity, entry.label, power="passive", medium="solid",
                   parts=[primary, *_constituents(entry.identity, identity)],
                   note="Real Machine; unspecified product mass and dimensions remain unresolved.")


def _machine_from_production(identity: str, label: str, built) -> Machine:
    return Machine(identity, label, power="shore-supplied", medium="electromechanical",
                   production_graph=built.graph.as_document())


def build_washer_machine() -> Machine:
    import appliance_production
    built = appliance_production.build_washer("laundry.washer")
    for node in built.graph.nodes:
        node.setdefault("catalogue_entry", "laundry.washer")
        node.setdefault("starting_location", "laundry corner")
    return _machine_from_production("laundry.washer", "Front-loading washer", built)


def build_dryer_machine(seed: int | str) -> Machine:
    import appliance_production
    faults = dryer_faults_for_seed(seed)
    built = appliance_production.build_dryer("laundry.dryer", faults=faults)
    for node in built.graph.nodes:
        node.setdefault("catalogue_entry", "laundry.dryer")
        node.setdefault("starting_location", "laundry corner")
        node.setdefault("dryer_model", "Garage Model VED-240")
        node.setdefault("generated_fault_set", list(faults))
    return _machine_from_production("laundry.dryer", "Vented electric dryer", built)


def _part_for_entry(entry: InventoryEntry, index: int, identity: str, position) -> MachinePart:
    attrs = _entry_attributes(entry, index); attrs["display_name"] = entry.label
    return _unresolved_part(identity, _slug(entry.label), attrs, position)


def build_overhead_door_machine() -> Machine:
    entries = [entry for entry in INVENTORY if entry.category == "overhead-door"]
    parts, by_entry = [], {}
    for row_index, entry in enumerate(entries):
        made = []
        for index in range(entry.quantity):
            identity = f"garage.overhead-door.{entry.identity}.{index + 1}"
            parts.append(_part_for_entry(entry, index, identity,
                                         (0.0, row_index * 0.04, index * 0.04)))
            made.append(identity)
        by_entry[entry.identity] = made
        parts.extend(_constituents(entry.identity, "garage.overhead-door"))
    lines = [
        MachineLine("garage.overhead-door.trolley-coupling", by_entry["door.opener"][0],
                    by_entry["door.trolley"][0], "direct-drive-lockup", 0.008,
                    "door-drive", "steel-plate",
                    {"releasable_by": by_entry["door.release-cord"][0],
                     "initially_engaged": True}),
        MachineLine("garage.overhead-door.arm-pin", by_entry["door.trolley"][0],
                    by_entry["door.arm"][0], "pinned-clevis", 0.010,
                    "door-linkage", "steel-plate"),
        MachineLine("garage.overhead-door.arm-to-door", by_entry["door.arm"][0],
                    by_entry["door.sectional"][0], "pinned-clevis", 0.010,
                    "door-linkage", "steel-plate"),
    ]
    next(p for p in parts if p.identity == by_entry["door.opener"][0]).attributes.update(
        drive_condition="mechanically-immobilized")
    return Machine("garage.overhead-door", "Sectional overhead door and opener",
                   power="shore-supplied", medium="electromechanical", parts=parts, lines=lines)


def build_workbench_machine() -> Machine:
    parts = []
    entries = [entry for entry in INVENTORY if entry.category == "workbench"]
    for row_index, entry in enumerate(entries):
        for index in range(entry.quantity):
            parts.append(_part_for_entry(entry, index,
                f"garage.workbench.{entry.identity}.{index + 1}",
                (index * 0.03, row_index * 0.025, 0.0)))
        parts.extend(_constituents(entry.identity, "garage.workbench"))
    return Machine("garage.workbench", "2x4-and-particleboard workbench",
                   power="passive", medium="solid", parts=parts)


def build_exhaust_machine() -> Machine:
    """The building-owned exhaust passage as one real fluid-circuit machine."""
    entries = {entry.identity: entry for entry in INVENTORY if entry.category == "dryer-exhaust"}
    sequence = (("vent.dryer-collar", (0.0, 0.0, 0.0)),
                ("vent.transition", (0.30, 0.0, 0.0)),
                ("vent.wall-collar", (0.60, 0.0, 0.0)),
                ("vent.elbow", (0.60, 0.50, 0.0)),
                ("vent.rigid-run", (1.80, 0.50, 0.0)),
                ("vent.deposit", (2.70, 0.50, 0.0)),
                ("vent.elbow", (2.90, 0.50, 0.0)),
                ("vent.exterior-hood", (3.00, 0.50, 0.0)))
    used, parts, chain = {key: 0 for key in entries}, [], []
    for key, position in sequence:
        index = used[key]; used[key] += 1
        identity = f"garage.dryer-exhaust.{key}.{index + 1}"
        part = _part_for_entry(entries[key], index, identity, position)
        part.material = "lint-debris" if key == "vent.deposit" else "galvanized-sheet"
        if key == "vent.deposit":
            part.attributes.update(fouling_mode="core-debris", blocked_frac=None,
                fouling_calibration_required=True, retained_mass_kg=None,
                moisture_mass_fraction=None, permeability_m2=None,
                attachment_strength_pa=None, contained_by="garage.dryer-exhaust")
        parts.append(part); chain.append(identity)
    for key, entry in entries.items():
        for index in range(used[key], entry.quantity):
            parts.append(_part_for_entry(entry, index,
                f"garage.dryer-exhaust.{key}.{index + 1}", (1.5, 0.65 + index * 0.03, 0.0)))
        parts.extend(_constituents(key, "garage.dryer-exhaust"))
    lines = [MachineLine(f"garage.dryer-exhaust.air-path.{index}", a, b, "air-line",
                0.0508, "garage.dryer-exhaust-air", "galvanized-sheet",
                {"fluid": "air", "medium_rate_state": "air_flow_kg_s",
                 "fluid_route_class": "rigid-orthogonal", "building_owned": True,
                 "through_port": index in (1, len(chain) - 1)})
             for index, (a, b) in enumerate(zip(chain, chain[1:]), 1)]
    return Machine("garage.dryer-exhaust", "Building dryer exhaust passage",
                   power="passive", medium="air", parts=parts, lines=lines,
                   note="Building-owned circuit; every connected machine sees this passage.")


def build_electrical_machines() -> tuple[Machine, ...]:
    breakers = (BreakerPosition("LIGHTS", "CEILING LIGHTS", (1,), 15.0),
                BreakerPosition("BENCH", "BENCH RECEPTACLES", (2,), 20.0),
                BreakerPosition("UTILITY", "UTILITY RECEPTACLES", (3,), 20.0),
                BreakerPosition("WASHER", "WASHER", (4,), 20.0),
                BreakerPosition("DRYER", "DRYER", (5, 7), 30.0),
                BreakerPosition("OPENER", "DOOR OPENER", (6,), 15.0))
    panel = build_breaker_panel(identity="garage.panel", breakers=breakers, spaces=12,
                                phases=("line-1", "line-2"), neutral_to_frame_bond=False)
    for part in panel.parts:
        part.attributes.setdefault("catalogue_entry", "garage.panel")
        part.attributes.setdefault("starting_location", "service wall")
    machines = [panel]
    entries = {entry.identity: entry for entry in INVENTORY}
    outlet_rows = (("garage.bench-receptacle", 3, "garage.panel.BENCH"),
                   ("garage.utility-receptacle", 2, "garage.panel.UTILITY"),
                   ("garage.washer-receptacle", 1, "garage.panel.WASHER"),
                   ("garage.opener-receptacle", 1, "garage.panel.OPENER"))
    for key, count, breaker in outlet_rows:
        for index in range(count):
            machine = build_duplex(identity=f"{key}.{index + 1}")
            for part in machine.parts:
                part.attributes.update(catalogue_entry=key, catalogue_instance=index,
                    starting_location=entries[key].location, upstream_breaker=breaker)
            machines.append(machine)
    dryer_entry = entries["garage.dryer-receptacle"]
    contacts = [_unresolved_part(f"garage.dryer-receptacle.contact.{role}",
        "electrical-contact", {**_entry_attributes(dryer_entry, 0), "terminal_role": role,
        "electrical_service": "garage.120-240v-split-60hz",
        "upstream_breaker": "garage.panel.DRYER"}, ((index - 1.5) * 0.012, 0.0, 0.0))
        for index, role in enumerate(("line-1", "line-2", "neutral", "protective-earth"))]
    machines.append(Machine("garage.dryer-receptacle", "Four-contact dryer receptacle",
                            power="shore-supplied", medium="electrical", parts=contacts))
    return tuple(machines)


_SPECIAL_ENTRY_IDS = frozenset(
    {entry.identity for entry in INVENTORY if entry.category in
     {"overhead-door", "workbench", "dryer-exhaust"}}
    | {"laundry.washer", "laundry.dryer", "garage.panel", "garage.bench-receptacle",
       "garage.utility-receptacle", "garage.washer-receptacle",
       "garage.dryer-receptacle", "garage.opener-receptacle", "garage.room-air"})


def build_starter_garage(seed: int | str = "starter") -> dict:
    """Return existing-engine objects only: Machines plus the room AirVolume."""
    machines = [build_washer_machine(), build_dryer_machine(seed),
                build_overhead_door_machine(), build_workbench_machine(),
                build_exhaust_machine(), *build_electrical_machines()]
    for entry in INVENTORY:
        if entry.identity in _SPECIAL_ENTRY_IDS:
            continue
        for index in range(entry.quantity):
            machines.append(build_loose_machine(entry, index))
    identities = [machine.identity for machine in machines]
    if len(identities) != len(set(identities)):
        raise RuntimeError("starter garage built duplicate Machine identities")
    return {"machines": tuple(machines), "air_volumes": (AirVolume("garage", 94.0),),
            "dryer_faults": dryer_faults_for_seed(seed)}


def garage_document(seed: int | str = "starter") -> dict:
    """Serialize authored content; every graph comes from Machine.build_graph."""
    built = build_starter_garage(seed)
    return {**catalogue_document(), "seed": _seed(seed),
            "dryer_faults": list(built["dryer_faults"]),
            "machine_count": len(built["machines"]),
            "machines": [machine.build_graph() for machine in built["machines"]],
            "air_volumes": [{"name": v.name, "volume_m3": v.volume_m3,
                             "temperature_k": v.temp_k, "co_ppm": v.co_ppm,
                             "oxygen_fraction": v.o2_frac}
                            for v in built["air_volumes"]]}


__all__ = ["DRYER_FAULTS", "PERMITTED_DRYER_FAULT_SETS", "dryer_faults_for_seed",
           "build_loose_machine", "build_washer_machine", "build_dryer_machine",
           "build_overhead_door_machine", "build_workbench_machine",
           "build_exhaust_machine", "build_electrical_machines", "build_starter_garage",
           "garage_document"]
