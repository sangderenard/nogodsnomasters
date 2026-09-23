"""First hardware specimens, ordered home -> military -> industrial.

Published constructions and authored game choices are separate at field level.
No object in this catalogue is certified hardware. Unknown pitches, polymer
formulations and contact dimensions are deliberately left unresolved.
"""
from __future__ import annotations
from dataclasses import replace
from hardware_construction import (CableConstruction, ConnectorConstruction,
    Contact, Core, Fact, Layer, Lay, Seal, authored, known, unknown)

SOURCES = {
    "southwire-nmb": {
        "url": "https://cabletechsupport.southwire.com/en/tile/42/cable/26854/?country=US",
        "title": "Southwire SPEC 10028, NM-B construction and 12 AWG 4/C table",
        "reviewed": "2026-09-23",
        "scope": "Family construction; dimensions on this page are ONLY the 12 AWG 4/C row.",
    },
    "leviton-5362": {
        "url": "https://leviton.com/products/5362",
        "title": "Leviton 5362 technical information",
        "reviewed": "2026-09-23",
        "scope": "5-20R ratings, materials and terminal construction. Page lists conflicting torque ranges; no chosen torque is asserted.",
    },
    "belden-1583a": {
        "url": "https://www.belden.com/products/cable/ethernet-cable/category-5e-cable/1583a",
        "title": "Belden 1583A construction and engineering data",
        "reviewed": "2026-09-23",
        "scope": "Horizontal solid-conductor cable, NOT a flexible patch cord. Polyolefin is not resolved to HDPE.",
    },
    "class-l": {
        "url": "https://www.amphenol-aerospace.com/resources/literature/view/class-l-catalog-section.pdf",
        "title": "Amphenol Class L catalogue",
        "reviewed": "2026-09-23",
        "scope": "Printed pp. 462-464, 468; configuration table and receptacle drawing visually checked.",
    },
    "leviton-460r7w": {
        "url": "https://leviton.com/products/460r7wlev",
        "title": "Leviton 460R7WLEV technical information",
        "reviewed": "2026-09-23",
        "scope": "60 A, 480 V, 3P+PE, 7h; NBR seals and nylon construction. Installation conditions for environmental qualifications remain unresolved.",
    },
    "repo-machines": {
        "url": "https://github.com/sangderenard/nogodsnomasters/blob/d2a66def49fc8f75ff2dcbdea56db0d7b6e5cddf/engine_toy/machines.py",
        "title": "Existing Machine authoring API",
        "reviewed": "2026-09-23", "scope": "Authoring contract, not a manufactured-hardware specification.",
    },
}


def _layer(key, material, function, source=None, **kwargs):
    mat = known(material, "material-key", source) if source else authored(material, "material-key")
    return Layer(key, mat, function, **kwargs)


def _service(identity, arrangement, line_voltage_v, frequency_hz,
             line_neutral_voltage_v=None, note="Selected game service"):
    """An explicit installed service, separate from a component's rating."""
    return authored({
        "identity": identity,
        "arrangement": arrangement,
        "line_voltage_v": line_voltage_v,
        "frequency_hz": frequency_hz,
        "line_neutral_voltage_v": line_neutral_voltage_v,
    }, "electrical-service", note)


def nm_b(awg=12, insulated_count=2):
    if awg not in (12, 14) or insulated_count not in (2, 3, 4):
        raise ValueError("initial NM-B specimens cover 12/14 AWG and 2/3/4 insulated cores")
    colors = ("black", "white", "red", "blue")[:insulated_count]
    roles = ("line", "neutral", "line-2", "line-3")[:insulated_count]
    cores = []
    for i, (color, role) in enumerate(zip(colors, roles), 1):
        cores.append(Core(f"core-{i}", role, known("copper", "material-key", "southwire-nmb"),
            authored(awg, "AWG", "Selected game specimen gauge within the manufacturer's family"),
            known(1, "count", "southwire-nmb"),
            insulation=(_layer("pvc", "pvc", "electrical-insulation", "southwire-nmb"),
                        _layer("nylon", "nylon", "abrasion-sheath", "southwire-nmb")),
            color=known(color, "color", "southwire-nmb"),
            temper=known("soft-annealed", "material-condition", "southwire-nmb"),
            details={"pvc_plus_nylon_total_thickness_m": known(0.000508, "m", "southwire-nmb")
                     if (awg, insulated_count) == (12, 4) else unknown("m"),
                     "material_grade": unknown(), "interface_bond": unknown()}))
    cores.append(Core("ground", "protective-earth", known("copper", "material-key", "southwire-nmb"),
        authored(awg, "AWG", "Full-size grounding conductor selected for this game specimen"),
        known(1, "count", "southwire-nmb"),
        insulation=(), details={"wrap": _layer("ground-wrap", "kraft-paper", "separation", "southwire-nmb", topology="tape"),
                               "bare": known(True, "bool", "southwire-nmb")}))
    exact_row = (awg, insulated_count) == (12, 4)
    return CableConstruction(f"home.nmb.{awg}-{insulated_count}", f"NM-B {awg}/{insulated_count} with ground", "home",
        tuple(cores), (_layer("binder", "kraft-paper", "binder", "southwire-nmb", topology="tape"),
        _layer("jacket", "pvc", "outer-jacket", "southwire-nmb", topology="conformal",
               thickness_m=known(0.000762, "m", "southwire-nmb") if exact_row else unknown("m"))),
        cross_section=unknown(note="Do not infer section packing from the nominal outside dimension"),
        outer_dimensions_m=known((0.010033, 0.010033), "m", "southwire-nmb", "0.395 inch nominal OD") if exact_row else unknown("m"),
        ratings={"voltage_v": known(600, "V", "southwire-nmb"),
                 "location": known("dry", "environment", "southwire-nmb"),
                 "ampacity_basis_temperature_c": known(60, "degC", "southwire-nmb"),
                 "individual_insulation_rating_c": known(90, "degC", "southwire-nmb")},
        details={"core_packing": unknown(), "overall_lay": unknown("m/turn"),
                  "finished_density_kg_m3": unknown("kg/m3"),
                  "electrical_service": _service(
                      "home.120v-1ph-60hz", "single-phase", 120.0, 60.0, 120.0,
                      "Installed game service; distinct from the cable's 600 V rating"),
                 "installation": "fixed-building-wiring", "not_an_extension_cord": True,
                 "jacket_print": authored(f"GAME SPECIMEN - {awg}/{insulated_count} CU NM-B - 600 V"),
                 "mass_per_m": unknown("kg/m"), "bend_radius": unknown("m"),
                 "damage_fields": ["jacket_cut_depth_m", "wet_length_m", "core_insulation_breach", "local_bend_radius_m"]},
        source_ids=("southwire-nmb",))


def flexible_power(identity, sector, awg, roles, service):
    """An explicitly authored specimen, not an assertion about a real cable SKU."""
    cores = tuple(Core(role, role, authored("copper", "material-key"), authored(awg, "AWG"),
        unknown("count", "Supplier stranding schedule not selected"),
        insulation=(_layer("insulation", "epr", "electrical-insulation"),),
        strand_lays=(Lay(f"{role}.strands", "strand", (role,), "unknown"),),
        details={"strand_contact_law": unknown(), "insulation_grade": unknown(),
                 "conductive_area_is_awg_nominal": True}) for role in roles)
    return CableConstruction(identity, f"Authored flexible {len(roles)}-core {awg} AWG service cable", sector,
        cores, (_layer("fillers", "textile", "packing", topology="filler"),
                _layer("binder", "polyester", "binder", topology="tape"),
                _layer("jacket", "chloroprene-rubber", "outer-jacket", topology="annular")),
        lays=(Lay("core-lay", "overall", tuple(c.identity for c in cores), "helix",
                  pitch_m=authored(0.16, "m/turn", "Chosen game manufacturing lay, not a published supplier pitch"),
                  handedness="right", details={"core_center_offsets_m": unknown("m")}),),
        cross_section=authored("round"),
        ratings={"voltage_v": authored(600, "V"), "installed_ampacity_a": unknown("A")},
        details={"fire_qualification": unknown(), "oil_qualification": unknown(),
                  "water_qualification": unknown(), "minimum_bend_radius_m": unknown("m"),
                  "electrical_service": service,
                 "conductor_stranding_and_core_cabling_are_distinct": True,
                 "reel_winding": {"drum_radius_m": unknown("m"), "traverse_pitch_m": unknown("m/turn"),
                                  "wound_length_m": authored(0, "m", "Initially deployed, not on a reel")},
                 "damage_fields": ["broken_strand_fraction", "jacket_abrasion_m", "insulation_resistance_ohm", "local_temperature_k"]})


def belden_1583a():
    colors = ("blue", "orange", "green", "brown")
    cores, lays = [], []
    for i, color in enumerate(colors, 1):
        members = (f"pair-{i}+", f"pair-{i}-")
        for member, marking in zip(members, (f"white/{color}", color)):
            cores.append(Core(member, member, known("copper", "material-key", "belden-1583a"),
                known(24, "AWG", "belden-1583a"), known(1, "count", "belden-1583a"),
                insulation=(_layer("insulation", "polyolefin", "electrical-insulation", "belden-1583a"),),
                color=known(marking, "color", "belden-1583a")))
        lays.append(Lay(f"pair-{i}", "pair", members, "helix",
            details={"pitch_is_not_in_the_reviewed_data_sheet": True}))
    return CableConstruction("home.data.1583a", "Belden 1583A horizontal U/UTP construction reference", "home",
        tuple(cores), (_layer("jacket", "pvc", "outer-jacket", "belden-1583a"),), tuple(lays),
        cross_section=authored("round", note="Simplified section topology; core packing not reconstructed"),
        outer_dimensions_m=known((0.00483, 0.00483), "m", "belden-1583a"),
        ratings={"voltage_v": known(300, "V", "belden-1583a", "CMR marking, not a mains-use permission"),
                 "max_conductor_dcr_ohm_per_m": known(0.0938, "ohm/m", "belden-1583a", "Maximum bound; not a nominal resistivity"),
                 "operating_temperature_c": known((-20, 75), "degC", "belden-1583a"),
                 "min_stationary_bend_radius_m": known(0.025, "m", "belden-1583a"),
                 "min_installation_bend_radius_m": known(0.051, "m", "belden-1583a"),
                 "nominal_mass_per_m": known(0.027, "kg/m", "belden-1583a")},
        details={"shielding": known("none-U/UTP", "construction", "belden-1583a"),
                 "ripcord_present": known(True, "bool", "belden-1583a"),
                 "ripcord_material": unknown(), "flexible_patch_cord": False,
                 "rf_model": "balanced-pair network required; not a DC bus",
                 "pair_to_pair_coupling": unknown(), "moisture_diffusion": unknown(),
                 "signal_service": authored(
                     {"kind": "balanced-four-pair", "nominal_impedance_ohm": 100.0},
                     "signal-service", "Catalogue identity only; no RF law is inferred")},
        source_ids=("belden-1583a",))


def domestic_5_20(form="receptacle"):
    if form not in ("plug", "receptacle"):
        raise ValueError("expected plug or receptacle")
    src = "leviton-5362"
    # Coordinates are render staging only. Manufacturing contact geometry is unresolved.
    positions = ((-0.0064, 0.004, 0.0), (0.0064, 0.004, 0.0), (0, -0.008, 0))
    contacts = tuple(Contact(letter, role,
        authored(position, "m", "Preview contact centers; not a NEMA dimensional drawing"),
        known("brass", "material-key", src) if form == "receptacle" else unknown("material-key"),
        authored("socket" if form == "receptacle" else "blade", note="Symbolic contact form; actual blade/socket geometry unresolved"),
        termination=known("back-or-side-wire", "termination", src) if form == "receptacle" else unknown(),
        details={"normal_force_law": unknown(), "contact_resistance_law": unknown(),
                 "wiping": known("triple-wipe", "construction", src) if form == "receptacle" and role == "line" else unknown(),
                 "torque_conflict": "Manufacturer webpage lists both 14-18 and 14-16 in-lb; verify instruction sheet"})
        for letter, role, position in zip(("X", "W", "G"), ("line", "neutral", "protective-earth"), positions))
    return ConnectorConstruction(f"home.nema5-20.{form}", f"5-20 {form} construction reference", "home", "NEMA-5-20",
        "nema-5-20:125V:20A:reference", form, contacts,
        known("nylon", "material-key", src) if form == "receptacle" else unknown("material-key"),
        known("nylon", "material-key", src) if form == "receptacle" else unknown("material-key"),
        authored("straight-in-friction"),
        ratings={"voltage_v": known(125, "V", src) if form == "receptacle" else authored(
                     125, "V", "Selected mating-interface rating; not a Leviton plug claim"),
                 "current_a": known(20, "A", src) if form == "receptacle" else authored(
                     20, "A", "Selected mating-interface rating; not a Leviton plug claim"),
                 "water_ingress": unknown(), "operating_temperature_c": known((-40, 60), "degC", src) if form == "receptacle" else unknown("degC")},
        details={"counterpart_status": "Plug is an authored mating counterpart, not a Leviton plug product claim",
                 "electrical_service": _service(
                     "home.120v-1ph-60hz", "single-phase", 120.0, 60.0, 120.0),
                 "receptacle_reference": "5362 is DUPLEX; one construction instance represents one socket; build_duplex installs two",
                 "mounting_yoke_material": known("brass", "material-key", src)
                    if form == "receptacle" else authored(
                        "not-applicable", "material-key", "Authored plug has no mounting yoke"),
                 "line_neutral_breakoff_tabs": authored(
                     form == "receptacle", note="Only the duplex receptacle has removable tabs"),
                 "tamper_resistant": authored(False, note="This specimen does not invent TR shutters"),
                 "enclosure_protection_is_separate": True}, source_ids=(src,))


def class_l_32_12(form="receptacle"):
    if form not in ("plug", "receptacle"):
        raise ValueError("expected plug or receptacle")
    src = "class-l"
    # Arrangement verified on printed p.464. Metric pitch-circle dimensions were NOT given.
    xy = ((-0.010, 0.013), (0.010, 0.013), (0.015, -0.007), (-0.015, -0.007), (0, -0.019))
    rows = (("A", "line-1", "4"), ("N", "neutral", "4N"), ("B", "line-2", "4"),
            ("C", "line-3", "4"), ("G", "protective-earth", "6N"))
    contacts = tuple(Contact(letter, role, authored((x, y, 0.0), "m", "Layout follows p.464; metric center distances are preview assumptions"),
        unknown("material-key", "Contact alloy and plating are not established by the reviewed pages"),
        authored("socket" if form == "receptacle" else "pin"),
        termination=known("crimp", "termination", src),
        details={"contact_size": known(size, "contact-size", src),
                 "engagement_group": known("early" if letter in ("N", "G") else "power", "sequence", src),
                 "crimp_barrel_geometry": unknown("m"), "crimp_contact_resistance": unknown("ohm"),
                 "plating_thickness": unknown("m")}) for (letter, role, size), (x, y) in zip(rows, xy))
    spec = ConnectorConstruction(f"military.class-l32-12.{form}", f"Class L shell 32 / insert 12 {form}", "military",
        "MIL-DTL-22992-Class-L", "class-l:32-12:site120-208-60:authored-key", form, contacts,
        known("aluminium", "material-key", src), unknown("material-key"),
        known("double-stub-thread", "coupling", src),
        seals=(Seal("rear-grommet", ("outside", "rear-terminal-space"), unknown("material-key"),
                    "compressed-grommet", ("mated", "unmated-capped", "unmated-uncapped")),
               Seal("insert-seal", ("front-contact-space", "rear-terminal-space"), unknown("material-key"),
                    "insert-barrier", ("mated", "unmated-capped", "unmated-uncapped"))),
        ratings={"current_a": known(60, "A", src),
                 "immersion_duration_s": known(14400, "s", src, "Qualification condition, not a leak-rate function"),
                 "immersion_pressure_difference_pa": known(101325, "Pa", src),
                 "immersion_configurations": known(("mated", "unmated"), "configuration", src),
                 "mating_qualification_cycles": known(500, "count", src)},
        details={"shell_size": known(32, "shell-size", src), "insert_arrangement": known("32-12", "arrangement", src),
                 "electrical_service": _service(
                     "military.120-208v-3ph-60hz", "three-phase-wye", 208.0, 60.0, 120.0,
                     "Selected site service; keyway and insert rotation remain unresolved"),
                 "geometry_reference": {"flange_side_m": known(2.625 * 0.0254, "m", src, "p.468 S; reference dimension"),
                    "mount_hole_spacing_m": known(2.062 * 0.0254, "m", src, "p.468 R"),
                    "mount_hole_diameter_m": known(0.209 * 0.0254, "m", src, "p.468 T"),
                    "front_depth_m": known(2.188 * 0.0254, "m", src, "p.468 J"),
                    "rear_depth_m": known(1.376 * 0.0254, "m", src, "p.468 F, rear nut fully tightened")},
                 "master_keyway_position": unknown(note="Must select from p.467 before claiming a complete MS part number"),
                 "insert_rotation": unknown(note="Must select for the intended frequency; no universal Class L mating"),
                 "accessory_thread_handedness": known("left", "handedness", src),
                 "qualification_is_not_simulated_ingress": True,
                 "coupling_torque": unknown("N m"), "strain_relief_clamp_force": unknown("N"),
                 "surface_finish": unknown(), "seal_compression_set_curve": unknown()}, source_ids=(src,))
    if form == "plug":
        details = dict(spec.details)
        details["geometry_reference"] = {
            "plug_drawing": unknown(note="p.468 describes the wall receptacle, not the plug; plug drawing not transcribed")}
        spec = replace(spec, details=details)
    return spec


def industrial_460(form="receptacle"):
    if form not in ("plug", "receptacle"):
        raise ValueError("expected plug or receptacle")
    src = "leviton-460r7w"
    roles = ("line-1", "line-2", "line-3", "protective-earth")
    contacts = tuple(Contact(f"terminal-{i}", role, unknown("m"), unknown("material-key"),
        authored("socket" if form == "receptacle" else "pin"),
        termination=authored("screw-clamp"),
        terminal_torque_nm=known(50 * 0.112984829, "N m", src) if form == "receptacle" else unknown("N m"))
        for i, role in enumerate(roles, 1))
    spec = ConnectorConstruction(f"industrial.iec60-480-7h.{form}", f"North American IEC 60309 60 A 480 V 7h {form}", "industrial",
        "IEC-60309", "iec60309:NA60A:480V:3P+PE:7h", form, contacts,
        known("impact-modified-nylon", "material-key", src) if form == "receptacle" else unknown("material-key"),
        known("glass-filled-nylon", "material-key", src) if form == "receptacle" else unknown("material-key"),
        authored("pin-sleeve-with-retention"),
        seals=(Seal("gasket", ("outside", "terminal-space"), known("nbr", "material-key", src),
                    "compressed-elastomer", ("installed-as-qualified",)),),
        ratings={"voltage_v": known(480, "V", src) if form == "receptacle" else authored(
                     480, "V", "Selected mating-interface rating; not a Leviton plug claim"),
                 "current_a": known(60, "A", src) if form == "receptacle" else authored(
                     60, "A", "Selected mating-interface rating; not a Leviton plug claim"),
                 "published_ingress_markings": known(("IP66", "IP67", "IP68", "IP69K"), "qualification", src),
                 "qualification_assembly_conditions": unknown(note="Verify instructions before assigning a marking to any mating/cap state")},
        details={"clock_position": known(7, "clock-hour", src) if form == "receptacle" else authored(
                     7, "clock-hour", "Selected authored counterpart key"),
                 "neutral_present": authored(False, note="Selected 3P+PE interface has no neutral"),
                 "electrical_service": _service(
                     "industrial.480v-3ph-delta-60hz", "three-phase-delta", 480.0, 60.0),
                 "hardware_material": known("303-stainless-steel", "material", src)
                    if form == "receptacle" else unknown("material"),
                 "contact_carrier_flammability": known("UL94-V0", "qualification", src)
                    if form == "receptacle" else unknown("qualification"),
                 "plug_is_authored_counterpart": form == "plug",
                 "mechanical_interlock": unknown(), "contact_travel_m": unknown("m")}, source_ids=(src,))
    if form == "plug":
        ratings = dict(spec.ratings)
        ratings["published_ingress_markings"] = unknown(note="Receptacle qualifications do not certify the authored plug counterpart")
        spec = replace(spec, ratings=ratings, seals=(replace(spec.seals[0], material=unknown("material-key")),))
    return spec


def cable_catalogue():
    return {spec.identity: spec for spec in (
        nm_b(14, 2), nm_b(12, 2), nm_b(12, 4), belden_1583a(),
        flexible_power("home.flex.12-3", "home", 12,
            ("line", "neutral", "protective-earth"),
            _service("home.120v-1ph-60hz", "single-phase", 120.0, 60.0, 120.0)),
        flexible_power("military.flex.6-5", "military", 6,
            ("line-1", "line-2", "line-3", "neutral", "protective-earth"),
            _service("military.120-208v-3ph-60hz", "three-phase-wye", 208.0, 60.0, 120.0)),
        flexible_power("industrial.flex.6-4", "industrial", 6,
            ("line-1", "line-2", "line-3", "protective-earth"),
            _service("industrial.480v-3ph-delta-60hz", "three-phase-delta", 480.0, 60.0)),
    )}


def connector_catalogue():
    return {spec.identity: spec for factory in (domestic_5_20, class_l_32_12, industrial_460)
            for spec in (factory("receptacle"), factory("plug"))}
