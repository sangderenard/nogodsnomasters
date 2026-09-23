"""Hardware authorship using the existing Machine/Part/Line graph path.

No replacement circuit, thermal, damage, dt, rendering or interaction engine.
Rich manufacturing declarations ride on the same graph as the coarse bodies.
Current exporters deliberately mark their coarse preview geometry. A stored
seal or contact law is not claimed to be running until a consumer binds it.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable

from hardware_construction import (CableConstruction, ConnectorConstruction,
    Fact, authored, compatibility, document, positive, route_length_m, unique)


def _api():
    # Preserve the production imports, including the workspace's Turing runtime.
    # No substitute Machine class is shipped to work around a missing checkout.
    from machines import Machine, MachineLine, MachinePart
    from assembly_ports import PartPort
    import numpy as np
    return Machine, MachineLine, MachinePart, PartPort, np


def _vec(value):
    value = tuple(float(x) for x in value)
    if len(value) != 3 or not all(math.isfinite(x) for x in value):
        raise ValueError("position must be a finite 3-vector")
    return value


def _add(a, b):
    return tuple(float(x + y) for x, y in zip(a, b))


def _base_attributes(**extra):
    return {"geometry_fidelity": "coarse-authoring-preview",
            "thermal_domain": "none",  # do not invent a polymer thermal law
            "thermal_capacity_basis": "unbound; Machine generic capacity is not a measured property",
            "simulation_binding": "authored-only-except-existing-lumped-copper-path",
            **extra}


def _electrical_service(spec):
    """Materialize the catalogue's explicit installed-service selection."""
    declaration = spec.details.get("electrical_service")
    if not isinstance(declaration, Fact) or declaration.value is None:
        return None
    from electrical_distribution import ElectricalService
    return ElectricalService(**document(declaration.value))


def _unbound_path(**extra):
    """Describe hardware without pretending an unresolved path is live."""
    return {
        "intended_transport_domain": "electrical",
        "electrical_runtime_binding": "unbound",
        "beam_solvable": False,
        "load_bearing": False,
        **extra,
    }


def build_connector(spec: ConnectorConstruction, *, identity=None, position=(0, 0, 0)):
    """A standalone connector as a real Machine with separate body/contact nodes.

    Current geometry is deliberately staged rather than an interchangeable
    manufacturing replica. Rich contact, material and seal declarations are
    retained unchanged. No location-based gasket mating rule is repurposed.
    """
    Machine, MachineLine, MachinePart, PartPort, np = _api()
    identity = identity or spec.identity
    position = _vec(position)
    if not identity:
        raise ValueError("identity is required")
    attrs = spec.graph_attributes()
    service = _electrical_service(spec)
    if service is not None:
        attrs.update(service.graph_attributes())
    attrs.update(_base_attributes(assembly_role="electrical-connector"))
    attrs["display_name"] = spec.label
    attrs["assembly_mass_kg"] = document(authored(0.3, "kg", "Preview assembly mass, not a product mass"))
    size = (0.028, 0.035, 0.025)
    if spec.sector == "military" and spec.form == "receptacle":
        side = spec.details["geometry_reference"]["flange_side_m"].value
        size = (side / 2, side / 2, 0.030)
    body = MachinePart(f"{identity}.body", "electrical-connector-body", position,
                       size, mass_kg=0.3, material=spec.shell_material.value or "unresolved-material",
                       part_role="electrical-connector-body", attributes=attrs)
    parts, lines = [body], []
    for index, contact in enumerate(spec.contacts):
        local = contact.position_m.value
        if local is None:
            angle = 2 * math.pi * index / len(spec.contacts)
            local = (0.014 * math.cos(angle), 0.014 * math.sin(angle), 0)
        p = _add(position, _add(local, (0, 0, size[2])))
        name = f"{identity}.contact.{contact.identity}"
        bus_identity = (f"{body.identity}::{service.identity}::{contact.role}"
                        if service is not None else None)
        port = PartPort(f"{name}.mating", name, "electrical-contact", np.asarray(p),
                        np.asarray((0., 0., 1.)), 0.002, False,
                        fluid="electricity", joint="bolted-flange")
        parts.append(MachinePart(name, "electrical-contact-proxy", p, (.002, .002, .004),
            mass_kg=0.0, material=contact.material.value or "unresolved-material", ports=[port],
            part_role="electrical-contact", attributes=_base_attributes(
                 physical_construction=document(contact), terminal_role=contact.role,
                 mass_accounted_by=body.identity, wrench_point=True,
                 contact_physics_bound=False, proxy_geometry=True,
                 electrical_bus_identity=bus_identity,
                 electrical_service=None if service is None else service.identity)))
        # A declaration of attachment only, not a copper path from contact to shell.
        lines.append(MachineLine(f"{name}.support", body.identity, name,
            "rigid-distance", .0008, "connector-structure", body.material,
            {"transport_domain": "mechanical", "beam_solvable": False,
             "load_bearing": False, "electrically_conductive": False,
             "in_view": False, "attachment_is_preview_only": True}))
    return Machine(identity, spec.label, power="shore-supplied", medium="electrical",
                   parts=parts, lines=lines, note="Manufacturing declarations with explicitly coarse preview geometry")


def build_cable(spec: CableConstruction, *, identity=None, points=((0, 0, 0), (1, 0, 0)),
                preview_radius_m=0.007, temperature_c=20.0):
    """Existing Conductor supplies the coarse resistance; no copied wire solver.

    All routes use the full polyline, not endpoint separation. The coarse
    equivalent uses axial wire length, explicitly not unmeasured strand/pair
    excess length. Signal cables outside the upstream AWG table keep their
    construction but are not falsely exported as supported power conductors.
    """
    Machine, MachineLine, MachinePart, PartPort, np = _api()
    from dc_power import AWG_MM2, Conductor
    identity = identity or spec.identity
    points = tuple(_vec(p) for p in points)
    length = route_length_m(points)
    service = _electrical_service(spec)
    if not math.isfinite(float(temperature_c)) or float(temperature_c) < -273.15:
        raise ValueError("temperature must be finite")
    positive(preview_radius_m, "preview radius")
    endpoints = []
    for suffix, p, neighbor in (("a", points[0], points[1]), ("b", points[-1], points[-2])):
        direction = np.asarray(p) - np.asarray(neighbor)
        direction = direction / np.linalg.norm(direction)
        name = f"{identity}.end.{suffix}"
        port = PartPort(f"{name}.termination", name, "electrical-cable-end", np.asarray(p),
                        direction, preview_radius_m, False, fluid="electricity")
        endpoints.append(MachinePart(name, "cable-end-reference", p, (.001, .001, .001),
            mass_kg=0, material="copper", ports=[port], part_role="cable-end",
            attributes=_base_attributes(wrench_point=True, mass_is_reference_only=True)))
    rows, unresolved = [], []
    for core in spec.cores:
        awg = core.awg.value
        if awg not in AWG_MM2 or core.material.value != "copper":
            unresolved.append(core.identity)
            continue
        conductor = Conductor(identity=f"{identity}.{core.identity}", awg=awg,
            length_m=length, temperature_c=temperature_c, both_directions=False)
        reference = Conductor(identity=f"{identity}.{core.identity}.20c", awg=awg,
            length_m=length, temperature_c=20.0, both_directions=False)
        if conductor.resistance_ohm <= 0 or not math.isfinite(conductor.resistance_ohm):
            raise ValueError("temperature is outside the useful range of the upstream copper resistance model")
        rows.append({"role": core.role, "service_role": core.role,
            "core_identity": core.identity, "awg": awg,
            "area_mm2": conductor.area_mm2, "resistance_ohm": conductor.resistance_ohm,
            "resistance_ohm_at_20c": reference.resistance_ohm,
            "temperature_coefficient_per_k": 0.00393,
            "ampacity_a": 0.0,
            "material": "copper", "length_basis": "coarse-axial-route-not-strand-path",
            "ampacity_unresolved": True})
    attrs = spec.graph_attributes()
    if not unresolved and service is None:
        raise ValueError(f"{spec.identity}: power cable needs an explicit electrical service")
    if service is not None:
        attrs.update(service.graph_attributes())
    attrs.update({"cable_length_m": length, "waypoints": [list(p) for p in points[1:-1]],
        "terminal_roles": [c.role for c in spec.cores],
        "transport_domain": "signal-electrical" if unresolved else "electrical",
        "conductors": [] if unresolved else rows,
        "conductor_roles": [] if unresolved else [row["service_role"] for row in rows],
        "unresolved_electrical_cores": unresolved,
        "electrical_reduction": "axial-copper-only" if not unresolved else "not-bound",
        "beam_solvable": False, "load_bearing": False, "routed": True,
        "render_radius_basis": "authored circular preview; manufacturing section is physical_construction",
        "jacket_mesh_resolved": False, "strand_mesh_resolved": False,
        "cable_mass_bound_to_mechanical_solver": False})
    edge = MachineLine(f"{identity}.run", endpoints[0].identity, endpoints[1].identity,
        "balanced-signal-path" if unresolved else "insulated-copper-wire",
        preview_radius_m, "balanced-signal" if unresolved else identity, "copper", attrs)
    return Machine(identity, spec.label, power="shore-supplied", medium="electrical",
                   parts=endpoints, lines=[edge], note="No blanket code ampacity or qualification inferred")


@dataclass(frozen=True)
class BreakerPosition:
    identity: str
    label: str
    slots: tuple[int, ...]
    rating_a: float
    common_trip: bool = True

    def __post_init__(self):
        if not self.identity or not self.label or not self.slots:
            raise ValueError("breaker needs identity, label and occupied slots")
        positive(self.rating_a, "breaker rating")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in self.slots):
            raise ValueError("slots are positive integers")
        if len(set(self.slots)) != len(self.slots):
            raise ValueError("breaker occupies a slot twice")
        if len(self.slots) > 1 and not self.common_trip:
            raise ValueError("this catalogue's multipole breaker is explicitly common-trip")


def panel_layout(breakers: Iterable[BreakerPosition], *, spaces=12,
                 phases=("line-1", "line-2")) -> list[dict]:
    """Authored two-column board. Each ROW shares one phase; rows alternate.

    Not a claim that every manufacturer's panel has this bus pattern. A multi-
    pole unit occupies consecutive rows on the same side, never adjacent left
    and right handles that happen to sit on the same bus phase.
    """
    if isinstance(spaces, bool) or not isinstance(spaces, int) or spaces < 2 or spaces % 2:
        raise ValueError("spaces must be a positive even integer")
    phases = tuple(phases)
    if len(phases) not in (2, 3):
        raise ValueError("this authored board has two or three phase buses")
    unique(phases, "phase")
    breakers = tuple(breakers)
    unique((b.identity for b in breakers), "breaker")
    used, result = set(), []
    for breaker in breakers:
        slots = tuple(sorted(breaker.slots))
        if slots[-1] > spaces or used.intersection(slots):
            raise ValueError("breaker is out of range or overlaps an occupied slot")
        if any(b - a != 2 for a, b in zip(slots, slots[1:])):
            raise ValueError("multipole breaker needs consecutive rows on the same side")
        chosen = tuple(phases[((s - 1) // 2) % len(phases)] for s in slots)
        if len(set(chosen)) != len(chosen):
            raise ValueError("multipole breaker must land on distinct phase buses")
        used.update(slots)
        result.append({"identity": breaker.identity, "label": breaker.label,
                       "slots": slots, "phases": chosen, "rating_a": float(breaker.rating_a),
                       "common_trip": breaker.common_trip})
    return result


def _sheet_box(identity, center, width, height, depth):
    """Six coarse plates, not a solid block masquerading as an empty cabinet."""
    _, _, Part, _, _ = _api()
    thickness = 0.0015
    x, y, z = center
    rows = (("back", (x, y, z-depth/2), (width/2, height/2, thickness/2)),
            ("door", (x, y, z+depth/2), (width/2, height/2, thickness/2)),
            ("left", (x-width/2, y, z), (thickness/2, height/2, depth/2)),
            ("right", (x+width/2, y, z), (thickness/2, height/2, depth/2)),
            ("top", (x, y+height/2, z), (width/2, thickness/2, depth/2)),
            ("bottom", (x, y-height/2, z), (width/2, thickness/2, depth/2)))
    return [Part(f"{identity}.{name}", "electrical-enclosure-sheet", p, h, mass_kg=0.5,
        material="steel-plate", part_role="enclosure-door" if name == "door" else "enclosure-sheet",
        attributes=_base_attributes(mass_basis="authored 0.5 kg per preview plate, not measured",
            enclosure_identity=identity, door_hinge_axis=(0, 1, 0) if name == "door" else None,
            cutouts_resolved=False, preview_hidden_by_default=(name == "door"))) for name, p, h in rows]


def build_breaker_panel(*, identity="home.panel", sector="home", position=(0, 0, 0),
                        breakers=None, spaces=12, phases=("line-1", "line-2"),
                        main_rating_a=100., neutral_to_frame_bond=False):
    """Authored panel internals with phase, neutral, earth and pole identity.

    Not a replica of a named load-center/PDISE/MCC product. Contacts, handle,
    bimetal, magnetic trip, latch and arc chamber are distinct declared pieces.
    There is no new trip integrator or automatic source conversion here.
    """
    Machine, Line, Part, PartPort, np = _api()
    center = _vec(position)
    positive(main_rating_a, "main rating")
    if sector not in {"home", "military", "industrial"}:
        raise ValueError("invalid sector")
    if breakers is None:
        breakers = (BreakerPosition("CB1", "LIGHTS", (1,), 15),
                    BreakerPosition("CB2", "BENCH OUTLETS", (2,), 20),
                    BreakerPosition("CB3", "SHOP MACHINE", (3, 5), 30))
    layout = panel_layout(breakers, spaces=spaces, phases=phases)
    parts = _sheet_box(identity, center, .36, .55, .13)
    lines = []
    bus_ids = {}
    for i, role in enumerate((*phases, "neutral", "protective-earth")):
        name = f"{identity}.bus.{role}"
        bus_ids[role] = name
        p = _add(center, (-.13 + .065 * i, 0, -.018))
        parts.append(Part(name, "electrical-busbar", p, (.006, .21, .002), mass_kg=.16,
            material="copper", part_role="electrical-busbar",
            attributes=_base_attributes(terminal_role=role, neutral_to_frame_bond=bool(neutral_to_frame_bond),
                mass_basis="authored specimen", section_basis="authored specimen")))
    lines.append(Line(f"{identity}.protective-frame-bond", bus_ids["protective-earth"], f"{identity}.back",
        "declared-contact-path", .004, "authored-panel-path", "copper",
        _unbound_path(resistance_is_unresolved=True,
            bond_purpose="protective-earth-to-enclosure")))
    if neutral_to_frame_bond:
        lines.append(Line(f"{identity}.service-bond", bus_ids["neutral"], bus_ids["protective-earth"],
            "declared-contact-path", .004, "authored-panel-path", "copper",
            _unbound_path(bond_purpose="explicit-neutral-protective-earth-bond",
                resistance_ohm=None, resistance_is_unresolved=True)))
    main_handle = f"{identity}.MAIN.handle"
    parts.append(Part(main_handle, "breaker-handle", _add(center,(0,.245,.032)), (.018,.014,.009),
        mass_kg=.02, material="unresolved-polymer", part_role="breaker-handle",
        attributes=_base_attributes(breaker_identity=f"{identity}.MAIN", common_trip=True,
            breaker_rating_a=float(main_rating_a), interaction_binding="unbound")))
    for i, phase in enumerate(phases):
        incoming = f"{identity}.supply.{phase}"
        line_contact = f"{identity}.MAIN.{phase}.line"
        load_contact = f"{identity}.MAIN.{phase}.load"
        for name, off in ((incoming,-.032),(line_contact,-.014),(load_contact,.014)):
            parts.append(Part(name,"main-breaker-terminal",_add(center,(-.05+.05*i,.22,off)),
                (.004,.004,.004),mass_kg=.01,material="copper",part_role="electrical-contact",
                attributes=_base_attributes(terminal_role=phase,breaker_identity=f"{identity}.MAIN")))
        for suffix, a, b, switch in (("line-feed",incoming,line_contact,False),
                                     ("contact",line_contact,load_contact,True),
                                     ("bus-feed",load_contact,bus_ids[phase],False)):
            attrs = _unbound_path(terminal_role=phase, electrical_reduction="unbound")
            if switch:
                attrs.update({"contact_state_owner":f"{identity}.MAIN", "contact_closed":False,"switch_law_bound":False})
            lines.append(Line(f"{identity}.MAIN.{phase}.{suffix}",a,b,"declared-contact-path",.003,
                "authored-panel-path","copper",attrs))
    schedule = []
    for row in layout:
        base = f"{identity}.{row['identity']}"
        slot = row["slots"][0]
        x = -.055 if slot % 2 else .055
        y = .19 - ((slot-1)//2) * .060
        p = _add(center, (x, y, .015))
        handle = f"{base}.handle"
        parts.append(Part(handle, "breaker-handle", _add(p, (0, 0, .030)), (.009, .015, .009),
            mass_kg=.01, material="unresolved-polymer", part_role="breaker-handle",
            attributes=_base_attributes(breaker_identity=base, breaker_rating_a=row["rating_a"],
                occupied_slots=list(row["slots"]), handle_states=["off", "on", "tripped"],
                interaction_binding="unbound", common_trip=row["common_trip"])))
        for j, phase in enumerate(row["phases"]):
            pp = _add(p, (0, -.060*j, 0))
            pole = f"{base}.pole{j+1}"
            for suffix, kind, mat, offset in (
                ("case", "breaker-insulating-case", "unresolved-polymer", (0,0,0)),
                ("line-contact", "breaker-fixed-contact", "unresolved-contact-alloy", (-.012,.006,.006)),
                ("load-contact", "breaker-moving-contact", "unresolved-contact-alloy", (.012,.006,.006)),
                ("bimetal", "breaker-thermal-trip", "unresolved-bimetal", (0,-.010,.002)),
                ("magnetic-trip", "breaker-magnetic-trip", "steel-plate", (0,.010,.002)),
                ("latch", "breaker-trip-latch", "steel-plate", (.006,0,.015)),
                ("arc-chamber", "breaker-arc-chamber", "unresolved-composite", (0,0,-.012))):
                name = f"{pole}.{suffix}"
                parts.append(Part(name, kind, _add(pp, offset), (.008,.008,.006),
                    mass_kg=.01, material=mat, part_role=kind,
                    attributes=_base_attributes(breaker_identity=base, pole_identity=pole, terminal_role=phase,
                        physical_properties={"trip_curve": None, "alloy": None,
                           "contact_gap_m": None, "spring_rate_n_m": None, "arc_law": None},
                        property_basis="unresolved except authored preview geometry and mass")))
            lines.append(Line(f"{pole}.bus-feed", bus_ids[phase], f"{pole}.line-contact",
                "declared-contact-path", .002, "authored-panel-path", "copper",
                _unbound_path(terminal_role=phase, electrical_reduction="unbound")))
            # Switchable link is declared, never an unconditional zero-ohm bridge.
            lines.append(Line(f"{pole}.switch", f"{pole}.line-contact", f"{pole}.load-contact",
                "declared-contact-path", .002, "authored-panel-path", "unresolved-contact-alloy",
                _unbound_path(contact_state_owner=base, contact_closed=False,
                    switch_law_bound=False)))
        schedule.append({**row, "breaker_identity":base, "handle_identity":handle,
                         "output_terminals":[f"{base}.pole{j+1}.load-contact" for j in range(len(row["phases"]))]})
    parts[0].attributes.update({"panel_schedule": schedule, "sector":sector,
        "panel_geometry_is_manufacturer_verified":False, "main_rating_a":float(main_rating_a),
        "main_breaker_components_declared":True, "main_breaker_runtime_bound":False, "source_conversion_implemented":False,
        "neutral_to_frame_bond":bool(neutral_to_frame_bond), "circuit_schedule_generated_from_layout":True,
        "phase_bus_pattern":"shared-row-cyclic-phases", "spaces":spaces})
    return Machine(identity, f"Authored {sector} breaker-panel interior", power="shore-supplied", medium="electrical",
        parts=parts, lines=lines, note="Declared components; not a certified panel or live protection system")


def patch_channels(count=24, pins=8):
    if any(isinstance(v,bool) or not isinstance(v,int) or v < 1 for v in (count,pins)):
        raise ValueError("positive integer channel and pin counts required")
    return [{"channel":i, "label":f"{i:02d}",
             "map":{f"front.{p}":f"rear.{p}" for p in range(1,pins+1)}} for i in range(1,count+1)]


def build_patch_panel(*, identity="home.patch24", count=24, position=(0,0,0), sector="home"):
    """Passive 8P8C feedthrough panel; no network switch or shared power bus.

    Generic authored 19-inch rack envelope, not a manufacturer-accurate PCB,
    jack contact drawing or an asserted category-performance certification.
    """
    channels = patch_channels(count)
    Machine, Line, Part, PartPort, np = _api()
    center = _vec(position)
    parts = [Part(f"{identity}.panel", "patch-panel-face", center, (.2413,.022225,.001),
        mass_kg=.35, material="steel-plate", part_role="patch-panel-face",
        attributes=_base_attributes(channel_schedule=channels, sector=sector,
            dimensions_basis="authored 19-inch 1U envelope; vendor mounting drawing unresolved",
            passive=True, category_qualification=None, jack_cutouts_resolved=False))]
    lines = []
    for ch in channels:
        x = ((ch["channel"] - .5) / count - .5) * .40
        nodes = {}
        for side, z in (("front", .012), ("rear", -.035)):
            name = f"{identity}.J{ch['channel']:02d}.{side}"
            p = _add(center,(x,0,z))
            nodes[side] = name
            ports = [PartPort(f"{name}.{pin}",name,"signal-contact",np.asarray(_add(p,((pin-4.5)*.001,0,0))),
                np.asarray((0,0,1 if side=="front" else -1)), .0002, False, fluid="signal") for pin in range(1,9)]
            parts.append(Part(name,"8p8c-jack",p,(.007,.007,.008),mass_kg=.006,
                material="unresolved-polymer",ports=ports,part_role="signal-jack",
                attributes=_base_attributes(channel=ch["channel"],label=ch["label"],side=side,
                    electrical_domain="balanced-signals",pin_geometry_basis="preview only",
                    contact_construction={"alloy":None,"gold_plating_m":None,"normal_force_n":None,
                                          "contact_resistance_ohm":None},
                    switch_normal_contacts=False,poe_capability_qualified=False)))
        for pin in range(1,9):
            lines.append(Line(f"{identity}.J{ch['channel']:02d}.pin{pin}",nodes['front'],nodes['rear'],
                "balanced-signal-path",.0002,f"balanced-signal.channel{ch['channel']}.pin{pin}","copper",
                {"terminal_a_identity":f"{nodes['front']}.{pin}","terminal_b_identity":f"{nodes['rear']}.{pin}",
                 "transport_domain":"signal-electrical","beam_solvable":False,"load_bearing":False,
                 "passive_continuity":True,"rf_transfer_function":None,"internal_trace_layout":None}))
    return Machine(identity,"Authored passive patch panel",power="shore-supplied",medium="signal",
                   parts=parts,lines=lines,note="Independent pin-preserving channels; no implicit network switching")


def build_duplex(*, identity="home.duplex", position=(0,0,0)):
    """Two independently identified 5-20R sockets with explicit break-off links.

    This composes the existing Machine objects. No second object registry and
    no boolean 'powered' state is introduced. A tab's removal must be bound by
    the game's interaction/topology system before it changes a live circuit.
    """
    from hardware_catalogue import domestic_5_20
    Machine, Line, Part, _, _ = _api()
    center = _vec(position)
    modules = [build_connector(domestic_5_20(), identity=f"{identity}.{name}",
               position=_add(center,(0,offset,0))) for name,offset in (("upper",.019),("lower",-.019))]
    parts = [p for module in modules for p in module.parts]
    lines = [line for module in modules for line in module.lines]
    for contact, role in (("X","line"),("W","neutral"),("G","protective-earth")):
        lines.append(Line(f"{identity}.link.{role}",f"{identity}.upper.contact.{contact}",
            f"{identity}.lower.contact.{contact}","declared-contact-path",.001,
            "authored-receptacle-path","brass",
            _unbound_path(part_role="breakoff-tab" if role != "protective-earth" else "ground-strap",
                removable=role != "protective-earth", link_present=True,
                resistance_is_unresolved=True, topology_change_binding="unbound")))
    return Machine(identity,"Duplex 5-20R construction assembly",power="shore-supplied",medium="electrical",parts=parts,lines=lines)


def build_cord(cable: CableConstruction, end_a: ConnectorConstruction, end_b: ConnectorConstruction,
               *, identity="service-cord", points=((0,0,0),(1,0,0))):
    """Compose cable + two connectors, retaining explicit core-to-contact maps.

    Mechanical endpoints are NOT treated as equipotential electrical nodes.
    In particular all contacts must never be electrically shorted through a
    cable endpoint proxy or a connector shell. Roles, not list order, bind.
    Cable ampacity, live mating and interlock validation are not inferred here.
    """
    Machine, Line, _, _, _ = _api()
    core_roles = {c.role for c in cable.cores}
    for endpoint in (end_a,end_b):
        if {c.role for c in endpoint.contacts} != core_roles:
            raise ValueError("connector and cable conductor roles differ")
    compatible, reasons = compatibility(end_a, end_b)
    if not compatible:
        raise ValueError(f"connector pair is incompatible: {reasons}")
    services = tuple(_electrical_service(spec) for spec in (cable, end_a, end_b))
    if any(service is None for service in services):
        raise ValueError("cord needs explicit cable and connector electrical services")
    if len(set(services)) != 1:
        raise ValueError("cable and connector electrical services differ")
    points = tuple(_vec(p) for p in points)
    run = build_cable(cable,identity=f"{identity}.cable",points=points)
    a = build_connector(end_a,identity=f"{identity}.a",position=points[0])
    b = build_connector(end_b,identity=f"{identity}.b",position=points[-1])
    contact_a = {c.role:f"{identity}.a.contact.{c.identity}" for c in end_a.contacts}
    contact_b = {c.role:f"{identity}.b.contact.{c.identity}" for c in end_b.contacts}
    edge = run.lines[0]
    service = services[0]
    body_a, body_b = f"{identity}.a.body", f"{identity}.b.body"
    edge.a, edge.b = body_a, body_b
    map_a = {role:f"{body_a}::{service.identity}::{role}" for role in core_roles}
    map_b = {role:f"{body_b}::{service.identity}::{role}" for role in core_roles}
    edge.attributes.update({"terminal_map_a":map_a,"terminal_map_b":map_b,
                           "contact_identity_map_a":contact_a,
                           "contact_identity_map_b":contact_b,
                           "electrical_nodes_are_role_qualified_contact_buses":True})
    for conductor in edge.attributes["conductors"]:
        conductor["terminal_a_identity"] = map_a[conductor["role"]]
        conductor["terminal_b_identity"] = map_b[conductor["role"]]
    attachment_lines = [
        Line(f"{identity}.cable-end-a-support", run.parts[0].identity, body_a,
             "rigid-distance", .001, "connector-structure", "cable-jacket",
             {"transport_domain":"mechanical", "beam_solvable":False,
              "load_bearing":False, "in_view":False}),
        Line(f"{identity}.cable-end-b-support", run.parts[1].identity, body_b,
             "rigid-distance", .001, "connector-structure", "cable-jacket",
             {"transport_domain":"mechanical", "beam_solvable":False,
              "load_bearing":False, "in_view":False}),
    ]
    return Machine(identity,"Authored cord assembly",power="shore-supplied",medium="electrical",
        parts=run.parts+a.parts+b.parts,lines=run.lines+a.lines+b.lines+attachment_lines,
        note="Pin map preserved by role; detailed geometry and runtime plug events remain unbound")
