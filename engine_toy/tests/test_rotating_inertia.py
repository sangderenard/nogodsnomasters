"""Every rotating mass says what it is, and nothing is guessed.

These lock in the replacement of `DrivetrainSolver._fallback_inertia`,
which filled in any missing polar inertia with `mass * 0.01` -- a radius
of gyration of exactly 100 mm on every node in the graph, right to 1.5x
for a flywheel and wrong by 150x for an oil pump, and silent either way.
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compressors
import engines
import gear_trains
import rotating_inertia as ri
from drivetrain_graph import build_drivetrain_graph, DrivetrainSolver
from sound_parts import drive_ratios_to_crank, DRIVELINE_EDGE_KINDS


def _combustion_engines():
    return [e for e in engines.CATALOGUE
            if e.kind == "combustion" and e.architecture.cylinders]


# --- nothing is guessed ----------------------------------------------

@pytest.mark.parametrize("identity", [e.identity for e in engines.CATALOGUE])
def test_no_rotating_part_is_missing_its_inertia(identity):
    """The loud check: a rotating-mass node with no inertia is a defect,
    not something to paper over with a stand-in. Building at all is the
    assertion -- a massless drive component refuses to build."""
    solver = DrivetrainSolver(graph=build_drivetrain_graph(engines.get(identity)))
    assert solver.inertia_shortfalls == []


def test_a_massless_drive_component_is_refused():
    """Physical law, not a warning. A gear with no polar inertia would
    reach any speed in zero time and take no energy doing it."""
    graph = {
        "nodes": [
            {"identity": "powertrain.engine", "kind": "rotating-mass", "inertia_kg_m2": 2.0},
            {"identity": "rotor.ghost", "kind": "rotating-mass", "mass_kg": 3.0},
        ],
        "edges": [{"identity": "drive", "a": "powertrain.engine", "b": "rotor.ghost",
                   "constraint": "geared-timing-drive", "ratio": 1.0}],
    }
    with pytest.raises(ValueError, match="massless drive components"):
        DrivetrainSolver(graph=graph)
    # and the diagnostic path can still list them rather than raising
    solver = DrivetrainSolver(graph=graph, allow_inertia_shortfalls=True)
    assert any("rotor.ghost" in w for w in solver.inertia_shortfalls)


@pytest.mark.parametrize("identity", [e.identity for e in engines.CATALOGUE])
def test_nothing_that_cannot_spin_is_given_an_inertia(identity):
    """A block port, an engine mount and an electrical junction do not
    rotate. The old fallback gave all 284 of them a polar inertia."""
    graph = build_drivetrain_graph(engines.get(identity))
    solver = DrivetrainSolver(graph=graph)
    rotating = {n["identity"] for n in graph["nodes"] if n.get("kind") == "rotating-mass"}
    # a node may legitimately carry an inertia without being a
    # rotating-mass in exactly two cases: it is a CASING with a rotor
    # inside it (a starter motor bolted to the block, whose armature
    # really does spin), or it is a point on a shaft that rotates and
    # inherits that shaft's group. Everything else -- block ports,
    # engine mounts, electrical junctions, coolant faces -- must have
    # none, and under the old fallback all 284 of them had one.
    # everything rigidly joined to a rotor, however many hops away --
    # the crank's front end reaches the crank through two of them
    rigid: dict[str, list[str]] = {}
    for e in graph["edges"]:
        if e.get("constraint") not in DrivetrainSolver.RIGID_HUB_KINDS:
            continue
        a, b = e.get("a"), e.get("b")
        rigid.setdefault(a, []).append(b)
        rigid.setdefault(b, []).append(a)
    # seed from anything that really turns: a rotating-mass node, or a
    # casing with a rotor inside it (a turbine's accessory gearbox is
    # declared a block component and still drives a starter-generator)
    seeds = rotating | {n["identity"] for n in graph["nodes"] if n.get("housed_assembly")}
    keyed_to_a_rotor = set(seeds)
    frontier = list(seeds)
    while frontier:
        cur = frontier.pop()
        for nxt in rigid.get(cur, ()):
            if nxt not in keyed_to_a_rotor:
                keyed_to_a_rotor.add(nxt)
                frontier.append(nxt)
    for n in graph["nodes"]:
        ident = n["identity"]
        if ident in rotating or n.get("housed_assembly") or ident in keyed_to_a_rotor:
            continue
        assert solver._inertia.get(ident, 0.0) == 0.0, ident


def test_a_rack_is_not_a_rotating_mass():
    """An Otto-Langen rack travels in a straight line. It was declared
    `rotating-mass`, which asked the drivetrain solver and the pitched
    sound catalogue to treat a sliding bar as a spinning one."""
    graph = build_drivetrain_graph(engines.get("otto-langen-atmospheric-1867"))
    racks = [n for n in graph["nodes"] if n["identity"].endswith(".rack")]
    assert racks, "the atmospheric engine should still have its rack"
    for r in racks:
        assert r["kind"] == "translating-mass"
        assert r.get("travel_axis") is not None


# --- one number for the crank ----------------------------------------

@pytest.mark.parametrize("identity", [e.identity for e in _combustion_engines()])
def test_the_crank_presents_the_inertia_the_engine_declares(identity):
    """Combustion integrated the crank at `engine.inertia_kg_m2` while
    the accessory belt pulled against `mass * 0.01` -- two numbers for
    one crankshaft, 63% apart on the AMC six."""
    engine = engines.get(identity)
    solver = DrivetrainSolver(graph=build_drivetrain_graph(engine))
    crank = solver._inertia["powertrain.engine"]
    declared = float(engine.inertia_kg_m2)
    # the crank's group may carry extra REAL parts keyed to it that no
    # published crank-assembly figure includes (the Otto-Langen's
    # freewheel drums), so it may exceed the declared figure -- but it
    # must never fall short of it, and never by the old factor
    assert crank >= declared - 1e-9
    assert crank <= declared * 1.05


def test_the_flywheel_is_not_counted_twice():
    """An engine's published inertia already includes its flywheel."""
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    fly = next(n for n in graph["nodes"] if n["identity"] == "powertrain.flywheel")
    assert fly["inertia_kg_m2"] > 0.0          # it is still a real, specified part
    assert fly["included_in_parent_inertia"]   # and it is not added to the crank again


# --- a rigidly keyed part is not a separate body ----------------------

def test_a_belt_feels_the_pulley_it_runs_on():
    """Pulleys hang off `rigid-keyed-hub` edges, which carry no dynamics,
    so their inertia was stored and then never used by anything."""
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    solver = DrivetrainSolver(graph=graph)
    alt = next(n for n in graph["nodes"] if n["identity"] == "electrical.alternator")
    pulley = next(n for n in graph["nodes"] if n["identity"] == "electrical.alternator.pulley")
    assert solver._inertia["electrical.alternator"] == pytest.approx(
        alt["inertia_kg_m2"] + pulley["inertia_kg_m2"])


def test_the_accessory_belt_pulls_against_the_whole_crank():
    """Every belt drives off `crank_shaft.front`, which reaches the crank
    only through a rigid reference edge."""
    solver = DrivetrainSolver(graph=build_drivetrain_graph(engines.get("amc-258-jeep-i6")))
    assert solver._inertia["powertrain.crank_shaft.front"] == pytest.approx(
        solver._inertia["powertrain.engine"])


# --- shapes are physical ---------------------------------------------

def test_every_shape_factor_is_physically_possible():
    """Between all-at-the-hub and all-at-the-rim. A uniform disc is 1/2
    and a thin ring is exactly 1; nothing can exceed a ring."""
    for key, shape in ri.BY_KEY.items():
        assert ri.MIN_FACTOR <= shape.factor <= ri.MAX_FACTOR, key
    assert ri.BY_KEY["solid-disc"].factor == 0.5
    assert ri.BY_KEY["thin-ring"].factor == 1.0


def test_an_undeclared_shape_raises_rather_than_standing_in():
    with pytest.raises(KeyError):
        ri.polar_inertia("something-nobody-built", 1.0, 0.1)


def test_a_part_with_no_size_raises():
    with pytest.raises(ValueError):
        ri.polar_inertia("solid-disc", 1.0, 0.0)


# --- housed assemblies: a casing with things spinning inside ----------

def test_a_casing_does_not_spin():
    """An alternator's 3 kg is case, stator and rectifier as well as
    rotor. Only the rotor turns."""
    alt = ri.housed_assembly("alternator", 3.0)
    assert alt.rotating_mass_kg < alt.unit_mass_kg
    assert alt.casing_mass_kg > 0.0
    assert alt.rotating_mass_kg + alt.casing_mass_kg == pytest.approx(alt.unit_mass_kg)


def test_a_gear_pump_really_has_two_gears():
    assert len(ri.housed_assembly("oil-pump", 1.4).groups) == 2


def test_a_group_reflects_through_the_square_of_its_ratio():
    g = ri.RotatingGroup("r", "solid-disc", 2.0, 0.05, speed_ratio=3.0)
    assert g.inertia_at_input_kg_m2 == pytest.approx(g.inertia_kg_m2 * 9.0)


# --- gear trains ------------------------------------------------------

def test_a_gearbox_in_neutral_is_not_free():
    """Constant mesh means the input shaft, the layshaft and every
    mainshaft gear turn whenever the clutch is engaged -- gear or no
    gear. It is the inertia an engine swings at idle."""
    gb = gear_trains.Gearbox(unit_mass_kg=34.0, case_radius_m=0.115,
                             gear_ratios=(3.54, 2.13, 1.36, 1.03, 0.82), reverse_ratio=3.28)
    assert gb.inertia_at_input(None) > 0.0
    names = {g.name for g in gb.groups(None)}
    assert "layshaft_cluster" in names


def test_the_layshaft_dominates_a_gearbox():
    """It is the heaviest rotating piece AND it turns faster than the
    input shaft, so it reflects up by the square of the constant mesh."""
    gb = gear_trains.Gearbox(unit_mass_kg=34.0, case_radius_m=0.115,
                             gear_ratios=(3.54, 2.13, 1.36, 1.03, 0.82), reverse_ratio=3.28)
    groups = {g.name: g.inertia_at_input_kg_m2 for g in gb.groups(None)}
    assert groups["layshaft_cluster"] > groups["input_shaft"]


def test_low_range_reflects_less_at_the_input():
    """The output side turns slower, so it costs the input shaft less."""
    tc = gear_trains.TransferCase(unit_mass_kg=32.0, case_radius_m=0.11)
    assert tc.inertia_at_input(low_range=True) < tc.inertia_at_input(low_range=False)


def test_a_converter_is_crank_inertia_even_at_a_standstill():
    """An automatic's impeller is bolted to the crank and spinning in
    park, which is why a flexplate is lighter than a manual's flywheel."""
    conv = gear_trains.TorqueConverter(unit_mass_kg=14.0, outer_radius_m=0.135, oil_mass_kg=2.5)
    assert conv.inertia_at_crank(0.0) > 0.0
    assert conv.inertia_at_crank(1.0) > conv.inertia_at_crank(0.0)


# --- the clutch is sized by what it has to hold -----------------------

def test_a_bigger_engine_gets_a_bigger_clutch():
    small = ri.clutch_pack(150.0)
    big = ri.clutch_pack(900.0)
    assert big.disc_diameter_m > small.disc_diameter_m
    assert big.inertia_kg_m2 > small.inertia_kg_m2


def test_a_clutch_goes_multi_plate_rather_than_growing_without_limit():
    """Real heavy-duty practice, and what stops a 7500 N*m engine being
    handed a metre-wide clutch."""
    pack = ri.clutch_pack(7500.0)
    assert pack.plates > 1
    assert pack.disc_diameter_m <= 0.356


# --- a compressor is an expander run backwards ------------------------

def test_a_roots_does_no_internal_compression():
    """Its pockets arrive at the discharge port still at suction
    pressure, so the work is the pressure difference through the whole
    displacement -- which is exactly why nobody builds a Roots shop
    compressor, and why one is fine as a supercharger."""
    p = compressors.P_ATM_PA
    roots = compressors.construction("roots")
    piston = compressors.construction("reciprocating-crank")
    mild = (compressors.compression_mep_pa(p, p * 1.6, roots)
            / compressors.compression_mep_pa(p, p * 1.6, piston))
    steep = (compressors.compression_mep_pa(p, p * 8.0, roots)
             / compressors.compression_mep_pa(p, p * 8.0, piston))
    assert mild < 1.5      # tolerable at supercharger ratios
    assert steep > 3.0     # hopeless at shop-air ratios


def test_clearance_volume_sets_the_single_stage_limit():
    """Gas trapped in the clearance re-expands before the suction valve
    can open. Push the ratio high enough and it fills the whole stroke."""
    con = compressors.construction("reciprocating-crank")
    ev = [compressors.volumetric_efficiency(r, con.clearance_frac, con.polytropic_index)
          for r in (2, 8, 32, 64)]
    assert ev == sorted(ev, reverse=True)
    assert ev[0] > 0.9
    assert ev[-1] == pytest.approx(0.0, abs=1e-6)


def test_a_swash_plate_runs_far_smoother_than_a_crank_machine():
    """Which is precisely why a car's A/C compressor is built that way."""
    p = compressors.P_ATM_PA
    crank = compressors.spec_for_rating("reciprocating-crank", 3000.0, 150.0, 4.5)
    swash = compressors.spec_for_rating("swash-plate", 3000.0, 150.0, 4.5)
    assert swash.torque_ripple_frac(p, p * 8) < crank.torque_ripple_frac(p, p * 8) / 5.0


def test_only_a_piston_machine_has_reciprocating_mass():
    scroll = compressors.spec_for_rating("scroll", 3000.0, 150.0, 4.5)
    crank = compressors.spec_for_rating("reciprocating-crank", 3000.0, 150.0, 4.5)
    assert scroll.reciprocating_mass_kg == 0.0
    assert crank.reciprocating_mass_kg > 0.0


def test_a_compressor_never_asks_what_is_turning_it():
    """The load is a property of the machine and the gas, not of the
    drive -- which is the whole reason an electrically driven one is a
    declaration rather than a second mechanism."""
    p = compressors.P_ATM_PA
    spec = compressors.spec_for_rating("swash-plate", 3000.0, 150.0, 4.5)
    assert spec.mean_torque_nm(p, p * 8) > 0.0
    graph = build_drivetrain_graph(engines.get("toyota-3sfe-camry-1990"))
    ac = next((n for n in graph["nodes"] if n["identity"] == "ac_compressor"), None)
    assert ac is not None and ac["driven_by"] == "crank-belt"
    assert ac["compressor_construction"] == "swash-plate"


# --- custom equipment is not standard equipment -----------------------

def test_a_direct_drive_bypass_is_only_fitted_when_asked_for():
    """A dog collar that fuses the crank to the output shaft is custom
    and military kit; the vehicle subunit used to emit one on every
    build, including a shopping-trolley commuter."""
    graph = build_drivetrain_graph(engines.get("toyota-3sfe-camry-1990"))
    assert not any(n["identity"] == "powertrain.direct_drive_bypass" for n in graph["nodes"])


# --- what is spinning in here ----------------------------------------

def test_a_machine_can_enumerate_its_own_spinning_masses():
    engine = engines.get("amc-258-jeep-i6")
    graph = build_drivetrain_graph(engine)
    ratios = drive_ratios_to_crank(graph, DRIVELINE_EDGE_KINDS)
    masses = ri.spinning_masses(graph, ratios)
    by_id = {m.identity: m for m in masses}
    assert "powertrain.engine" in by_id
    assert "powertrain.transmission" in by_id      # reachable through the clutch
    assert by_id["powertrain.camshaft"].ratio_to_crank == pytest.approx(0.5)
    # sorted by what each one costs the crank, and every one of them
    # names the shape it was derived from
    costs = [m.inertia_at_crank_kg_m2 for m in masses]
    assert costs == sorted(costs, reverse=True)


# --- mesh: an object is drawn because it is an object -------------------

def test_accessory_bodies_are_drawn_not_just_their_pulleys():
    """The keyword allowlist had "_pulley" on it and not the machines
    the pulleys are bolted to, so the view drew four pulleys turning in
    mid air."""
    from engine_mesh import bodies_in_view
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    drawn = bodies_in_view(graph)
    for ident in ("electrical.alternator", "powertrain.water_pump", "powertrain.cooling_fan",
                  "powertrain.clutch", "powertrain.transmission", "powertrain.transfer_case"):
        assert "node_" + ident.replace(".", "_") in drawn, ident


def test_a_part_a_dedicated_mesh_already_draws_is_not_drawn_twice():
    """head_mesh emits the head casting, so drawing the graph's own head
    nodes as well would lay a second casting over the first."""
    from engine_mesh import bodies_in_view
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    drawn = bodies_in_view(graph)
    heads = [n["identity"] for n in graph["nodes"]
             if n.get("kind") == "engine-head-component"]
    assert heads, "the six should still be in the graph"
    for ident in heads:
        assert "node_" + ident.replace(".", "_") not in drawn


def test_connection_points_are_not_objects():
    """A port face, a shaft reference, a wire terminus: in the graph to
    carry physics, with no separate body to look at."""
    from engine_mesh import bodies_in_view
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    drawn = bodies_in_view(graph)
    for n in graph["nodes"]:
        if n.get("kind") in ("engine-block-port", "crank-shaft-endpoint", "electrical-junction"):
            assert "node_" + n["identity"].replace(".", "_") not in drawn


def test_you_can_shoot_what_you_cannot_see():
    """Enclosure is a VISIBILITY answer. The ray mesh takes no argument
    and gets every body, because a round through a casing goes on to hit
    what is behind it."""
    from engine_mesh import build_engine_mesh, enclosed_bodies
    graph = build_drivetrain_graph(engines.get("dual-motor-ev-reference"))
    hidden = enclosed_bodies(graph, frozenset())
    assert hidden, "the EV motor internals should be enclosed"
    rays, _ = build_engine_mesh(graph)                      # no argument: everything
    render, _ = build_engine_mesh(graph, breached=frozenset())
    assert len(rays.vertices) > len(render.vertices)


def test_damage_reveals_what_a_casing_was_hiding():
    from engine_mesh import build_engine_mesh, breached_casings
    graph = build_drivetrain_graph(engines.get("dual-motor-ev-reference"))
    sealed, _ = build_engine_mesh(graph, breached=frozenset())

    class _Hole:
        part = "powertrain.motor_housing"
        through = True

    opened, _ = build_engine_mesh(graph, breached=breached_casings([_Hole()]))
    assert len(opened.vertices) > len(sealed.vertices)


def test_a_round_passes_through_a_casing_into_what_it_hides():
    """The point of culling being visual only: the ray mesh carries the
    enclosed parts, and the penetration model walks wall by wall, so a
    round that goes through the housing goes on to hit the stator and
    the rotor behind it."""
    import numpy as np
    from engine_rays import Ray, RayMesh
    graph = build_drivetrain_graph(engines.get("dual-motor-ev-reference"))
    mesh = RayMesh.from_graph(graph)
    node = {n["identity"]: n for n in graph["nodes"]}
    centre = np.array(node["powertrain.rotor_pack"]["reference_position"], dtype=float)
    across = np.array([0.0, 2.0, 0.0])
    parts = [t.part for t in mesh.traversals(Ray.from_points(centre + across, centre - across))]
    assert any("motor_housing" in p for p in parts)
    assert any("rotor_pack" in p for p in parts), "the hidden rotor must still be hittable"
    # and the casing is crossed before what it encloses
    assert (parts.index(next(p for p in parts if "motor_housing" in p))
            < parts.index(next(p for p in parts if "rotor_pack" in p)))


def test_a_blind_hole_does_not_open_a_case():
    from engine_mesh import breached_casings

    class _Dent:
        part = "powertrain.motor_housing"
        through = False

    assert breached_casings([_Dent()]) == frozenset()


# --- oil and coolant actually reach the head -------------------------

def _circuit(identity, which):
    solver = DrivetrainSolver(graph=build_drivetrain_graph(engines.get(identity)))
    return next((c for c in solver.fluid_circuits if c.circuit_identity == which), None)


def test_oil_reaches_the_head_and_comes_back():
    """A gasketed port face that carries a fluid is a flow path. The
    oil circuit used to run pan -> pump -> filter -> gallery and STOP at
    the block, with the head's feed and both drainbacks in no circuit."""
    oil = _circuit("amc-258-jeep-i6", "oil")
    assert oil is not None
    assert any("head_bank1.oil_feed_face" in n for n in oil.nodes)
    assert any("oil_return_face" in n for n in oil.nodes)
    assert "powertrain.oil_pan" in oil.nodes


def test_the_valvetrain_is_lubricated_at_all():
    oil = _circuit("amc-258-jeep-i6", "oil")
    assert "powertrain.camshaft" in oil.nodes


def test_every_camshaft_is_fed_not_just_the_first():
    """A DOHC head has two, a twin-row radial one per row, the Wasp
    Major four. Feeding cam 1 and leaving three dry is not survivable."""
    oil = _circuit("pw-r4360-wasp-major", "oil")
    cams = [f"powertrain.camshaft{s}" for s in ("", "_2", "_3", "_4")]
    for cam in cams:
        assert cam in oil.nodes, cam


def test_the_cylinder_head_is_in_the_cooling_system():
    """It makes most of the heat. The coolant transfer holes across the
    gasket were 14 separate two-node circuits."""
    coolant = _circuit("amc-258-jeep-i6", "coolant")
    assert coolant is not None
    assert any("head_bank1.coolant_face" in n for n in coolant.nodes)
    assert "powertrain.radiator" in coolant.nodes


def test_the_crank_has_main_bearings_and_they_are_fed():
    """crank_main_1..N were drawn from the beginning and existed nowhere
    else: no bearings in the graph, and a main gallery joined to nothing."""
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    mains = [n for n in graph["nodes"] if n.get("bearing_kind") == "crankshaft-main"]
    assert len(mains) >= 5
    oil = _circuit("amc-258-jeep-i6", "oil")
    for m in mains:
        assert m["identity"] in oil.nodes, m["identity"]


def test_a_wankel_has_no_valvetrain():
    """It breathes through ports the rotor sweeps past. Deriving "sohc"
    had head_mesh ready to draw it a cam box."""
    from cylinder_ports import derive_valvetrain
    assert derive_valvetrain(engines.get("twin-rotor-13b")) == "none"


def test_a_boxer_gets_one_camshaft_per_head():
    """No single cam can reach both banks of a horizontally opposed
    engine."""
    graph = build_drivetrain_graph(engines.get("gt-flat-six-4000"))
    cams = [n for n in graph["nodes"]
            if n["identity"].rsplit(".", 1)[-1].startswith("camshaft")]
    assert len(cams) == 2
    # and they are on opposite sides of the crank, not in the same hole
    assert cams[0]["reference_position"][2] * cams[1]["reference_position"][2] < 0


def test_every_camshaft_carries_its_own_inertia():
    """Only `powertrain.camshaft` was in the identity table, so shafts
    2..4 registered non-zero purely through their cloned sprockets."""
    graph = build_drivetrain_graph(engines.get("pw-r4360-wasp-major"))
    cams = [n for n in graph["nodes"]
            if n["identity"].rsplit(".", 1)[-1].startswith("camshaft")]
    assert len(cams) == 4
    for c in cams:
        assert float(c.get("inertia_kg_m2") or 0.0) > 0.0, c["identity"]


# --- where the oil goes when it leaves a hole -------------------------

def test_oil_thrown_inside_the_engine_is_not_lost():
    """The dipper's spray lands on the crank and runs back to the sump.
    Billing it to `on_loss` drained the sump of an engine sitting still."""
    from engine_rays import RayMesh
    from hole_emitters import HoleEmitterField
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    field = HoleEmitterField()
    assert field.add_splash_from_graph(graph)
    field.resolve_spray(RayMesh.from_graph(graph))
    for em in field.emitters:
        assert em.spray_resolved
        assert em.escaped_frac == 0.0, em.identity
        assert em.spray_hits


def test_a_spray_does_not_count_its_own_hole_as_containment():
    """A port has geometry in the mesh, so a ray from its centre hits it
    immediately -- which reported every jet as perfectly contained."""
    from hole_emitters import HoleEmitterField, HoleEmitter
    em = HoleEmitter(identity="powertrain.head_bank1.oil_fill.open_port",
                     part="head_bank1", circuit="oil", fluid="engine-oil",
                     position=(0.0, 0.0, 0.0), direction=(0.0, 1.0, 0.0),
                     radius_m=0.016, through=True, kind="open-port")
    assert HoleEmitterField._is_self_hit(em, "port_head_bank1_oil_fill")
    assert not HoleEmitterField._is_self_hit(em, "valve_cover_bank1")


# --- a port is closed by the thing that closes it ---------------------

def test_the_engine_is_oil_tight_because_its_caps_are_declared():
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    by_id = {n["identity"]: n for n in graph["nodes"]}
    for ident, closure in (("powertrain.head_bank1.oil_fill", "screw-cap"),
                           ("powertrain.crankcase.main_gallery", "gallery-plug"),
                           ("powertrain.oil_pan.drain_plug", "drain-plug"),
                           ("powertrain.crankcase.dipstick", "dipstick")):
        assert by_id[ident]["closure"] == closure
        assert by_id[ident]["plugged"]
    # a breather really is meant to be open
    assert by_id["powertrain.crankcase.breather"]["closure"] == ""


def test_taking_a_cap_off_a_running_engine_opens_a_real_hole():
    """And putting it back closes it. Live and idempotent: a build that
    is reconfigured behaves the same as one that was shot."""
    from hole_emitters import HoleEmitterField
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    by_id = {n["identity"]: n for n in graph["nodes"]}
    field = HoleEmitterField()
    assert field.sync_open_ports(graph) == (0, 0)      # sealed
    by_id["powertrain.head_bank1.oil_fill"]["plugged"] = False
    assert field.sync_open_ports(graph) == (1, 0)      # cap off
    em = next(e for e in field.emitters if e.kind == "open-port")
    assert em.circuit == "oil" and em.radius_m > 0.0
    by_id["powertrain.head_bank1.oil_fill"]["plugged"] = True
    assert field.sync_open_ports(graph) == (0, 1)      # cap back on


def test_a_valve_seat_is_not_a_leak():
    """Intake, exhaust and spark-plug ports are open by design."""
    from hole_emitters import HoleEmitterField
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    field = HoleEmitterField()
    field.sync_open_ports(graph)
    assert not [e for e in field.emitters if e.kind == "open-port"]


# --- fouling, for every fluid ----------------------------------------

def test_a_blocked_hole_costs_the_square_of_what_it_lost():
    import fouling
    assert fouling.restriction_factor(0.0) == pytest.approx(1.0)
    assert fouling.restriction_factor(0.5) == pytest.approx(4.0)
    assert fouling.restriction_factor(0.75) == pytest.approx(16.0)


def test_wax_clears_and_scale_does_not():
    import fouling
    waxed = fouling.accumulate(0.0, "wax", 2.0)
    assert waxed > 0.0
    assert fouling.accumulate(waxed, "wax", -2.0) == pytest.approx(0.0)
    scaled = fouling.accumulate(0.0, "scale", 300.0)
    assert fouling.accumulate(scaled, "scale", -300.0) == pytest.approx(scaled)


def test_fluid_components_declare_how_they_block():
    import fouling
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    declared = {n["identity"]: n["fouling_mode"]
                for n in graph["nodes"] if n.get("fouling_mode")}
    assert "powertrain.oil_filter" in declared
    assert "powertrain.radiator" in declared
    for m in declared.values():
        assert fouling.mode(m).clears_with


# --- corrosion, and ports that make themselves ------------------------

def test_bad_fuel_is_read_off_the_fuel():
    """`working_fluids` already declares sulfur_frac on every fuel, so
    how aggressive the exhaust is comes from what is in the tank."""
    import fouling
    import working_fluids as wf
    clean = wf.WORKING_FLUIDS["ultra-low-sulfur-diesel"]
    filthy = next(v for k, v in wf.WORKING_FLUIDS.items() if "heavy-fuel" in k)
    assert fouling.severity_from_fuel(clean) == pytest.approx(1.0, abs=0.2)
    assert fouling.severity_from_fuel(filthy) > 20.0


def test_deposits_make_corrosion_worse_not_better():
    """Under-deposit attack: the soot holds the condensate against the
    metal instead of shielding it."""
    import fouling
    bare = fouling.corrode(0.0, "acid-condensate", 100.0, severity=10.0, blocked_frac=0.0)
    under = fouling.corrode(0.0, "acid-condensate", 100.0, severity=10.0, blocked_frac=1.0)
    assert under > bare


def test_a_wall_that_is_gone_grows_its_own_port():
    """Autogenesis: nobody placed this hole, a process did."""
    import fouling
    node = {"identity": "powertrain.exhaust_collector_1.muffler",
            "reference_position": [0.0, 0.2, 0.0], "wall_thickness_m": 0.0014,
            "wall_lost_m": 0.0, "drum_radius_m": 0.05, "corrosion_mode": "acid-condensate",
            "part": "powertrain.exhaust_collector_1"}
    assert fouling.autogenous_port(node) is None          # wall still there
    node["wall_lost_m"] = 0.0014
    port = fouling.autogenous_port(node, circuit="exhaust")
    assert port is not None
    assert port["autogenous"] and not port["plugged"] and port["closure"] == ""
    # it opens at the low point, where the condensate sat
    assert port["reference_position"][1] < node["reference_position"][1]


def test_a_rust_hole_is_just_a_hole():
    """No separate leak path: once it is in the graph the ordinary
    open-port machinery gives it pressure and flow, exactly as it does
    for one that was shot."""
    import fouling
    from hole_emitters import HoleEmitterField
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    field = HoleEmitterField()
    assert field.sync_open_ports(graph) == (0, 0)
    rotten = next(n for n in graph["nodes"] if n.get("corrosion_mode"))
    rotten["wall_lost_m"] = rotten["wall_thickness_m"]
    port = fouling.autogenous_port(rotten, circuit="oil")
    graph["nodes"].append(port)
    assert field.sync_open_ports(graph) == (1, 0)


def test_the_exhaust_declares_what_rots_and_how_thick_it_is():
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    rot = [n for n in graph["nodes"] if n.get("corrosion_mode")]
    assert rot, "an exhaust system should be able to rot through"
    for n in rot:
        assert n["wall_thickness_m"] > 0.0
        assert n["wall_lost_m"] == 0.0     # a new engine starts whole


def test_an_undeclared_corrosion_mode_raises():
    import fouling
    with pytest.raises(KeyError):
        fouling.corrosion_mode("bit-rot")


def test_corrosion_never_heals():
    import fouling
    lost = fouling.corrode(0.0, "acid-condensate", 500.0, severity=20.0)
    assert fouling.corrode(lost, "acid-condensate", -500.0, severity=20.0) >= lost


def test_an_undeclared_fouling_mode_raises():
    import fouling
    with pytest.raises(KeyError):
        fouling.mode("rusting-out-of-spite")


def test_a_bearing_does_not_set_a_shaft_speed():
    """A bearing SUPPORTS a shaft. Treating one as a drive let the
    walker reach the camshaft at ratio 1.0 through its own bearings,
    contradicting the timing drive beside it."""
    assert "rotational-bearing" not in DRIVELINE_EDGE_KINDS


# --- a loop is not a pipe --------------------------------------------

def test_a_recirculating_filter_is_not_filled_by_its_own_flow_rate():
    """Engine oil is the same five litres passing the filter twenty
    times a minute. Counting flow x concentration counts that one charge
    over and over -- it said a 60 g element caught 300 g in 100 hours."""
    import fouling
    # what the engine MAKES fills it; pumping the same oil faster does not
    slow = fouling.captured_recirculating_g(0.18, 300.0)
    assert 40.0 < slow < 60.0                       # a real oil-change interval
    worn = fouling.captured_recirculating_g(0.6, 300.0)
    assert worn > slow * 3                          # a tired engine eats filters


def test_a_once_through_filter_is_filled_by_what_passes_it():
    """Intake air really does arrive dirty and leave."""
    import fouling
    clean = fouling.captured_once_through_g(190.0 * 400.0, 1.5, 20.0)
    dusty = fouling.captured_once_through_g(190.0 * 400.0, 15.0, 20.0)
    assert dusty == pytest.approx(clean * 10.0)


def test_industrial_units_are_the_ones_on_the_datasheet():
    """Beta ratio, ISO 4406 and mm/yr, so a figure can be checked
    against a part number instead of taken on faith."""
    import fouling
    assert fouling.capture_efficiency(200.0) == pytest.approx(0.995)
    assert fouling.capture_efficiency(2.0) == pytest.approx(0.5)
    # each whole ISO 4406 code step doubles the contamination
    assert (fouling.iso4406_mg_per_l("19/17/14")
            == pytest.approx(fouling.iso4406_mg_per_l("18/16/13") * 2.0))
    for m in fouling.CORROSION_BY_KEY.values():
        assert 0.0 < m.rate_mm_per_year < 50.0


def test_a_conditioner_removes_or_prevents_and_says_which():
    import fouling
    # a flush cleans what is there
    left, debris = fouling.apply_conditioner(0.8, "sludge", "engine-flush")
    assert left < 0.8 and debris > 0.0        # and sends some of it downstream
    # a stabiliser cleans nothing, it slows what is coming
    left, debris = fouling.apply_conditioner(0.8, "varnish", "fuel-stabiliser")
    assert left == pytest.approx(0.8) and debris == 0.0
    assert fouling.inhibited_severity(1.0, "varnish", "fuel-stabiliser") < 1.0


def test_the_shelf_can_be_asked_what_fixes_this():
    import fouling
    assert any(c.key == "core-wash" for c in fouling.conditioners_for("core-debris"))
    assert any(c.key == "anti-gel" for c in fouling.conditioners_for("wax"))
    with pytest.raises(KeyError):
        fouling.conditioner("snake-oil")


# --- metal comes off a part and is conserved --------------------------

def _ledger_after(before: dict, after: dict, displacement_l: float):
    import wear_debris as wd
    led = wd.DebrisLedger()
    led.debit(wd.damage_vector(before), wd.damage_vector(after), displacement_l)
    return led


def test_metal_in_the_oil_is_metal_off_a_part():
    """Not a rate anybody chose: the rings were at y and are now at x,
    so exactly (x - y) of their shed allowance is now flake."""
    import wear_debris as wd
    half = _ledger_after({"piston_rings": 0.10}, {"piston_rings": 0.20}, 4.0)
    whole = _ledger_after({"piston_rings": 0.10}, {"piston_rings": 0.30}, 4.0)
    assert whole.entered_g == pytest.approx(half.entered_g * 2.0)
    # and it is the allowance, not an invention
    expected = wd.shed_allowance_g("piston_rings", 4.0) * 0.10 * (1.0 + wd.UNTRACKED_SHARE)
    assert half.entered_g == pytest.approx(expected)


def test_nothing_is_created_or_lost_anywhere():
    led = _ledger_after({"piston_rings": 0.1, "connecting_rod_bearings": 0.05},
                        {"piston_rings": 0.2, "connecting_rod_bearings": 0.09}, 4.0)
    for _ in range(200):
        led.through_filter(0.9, 200.0)
        led.settle(0.001)
    assert led.is_balanced()
    led.oil_change()
    assert led.is_balanced()
    assert led.accounted_g == pytest.approx(led.entered_g)


def test_wear_cannot_un_happen():
    led = _ledger_after({"piston_rings": 0.3}, {"piston_rings": 0.1}, 4.0)
    assert led.entered_g == 0.0


def test_the_mix_is_the_diagnosis():
    """Bearings shed copper and lead; rings shed iron and chromium. An
    oil sample tells them apart because the model actually took the
    metal off different parts."""
    rings = _ledger_after({"piston_rings": 0.0}, {"piston_rings": 0.05}, 4.0)
    shells = _ledger_after({"connecting_rod_bearings": 0.0},
                           {"connecting_rod_bearings": 0.05}, 4.0)
    r = rings.ppm_by_metal(5.0)
    s = shells.ppm_by_metal(5.0)
    assert r.get("Fe", 0.0) > r.get("Cu", 0.0) * 10
    assert s.get("Cu", 0.0) > s.get("Cr", 0.0)


def test_a_filter_cannot_catch_what_it_cannot_see():
    """The fine fraction passes every time, which is the only reason a
    sample from a filtered engine reads anything at all."""
    led = _ledger_after({"piston_rings": 0.0}, {"piston_rings": 0.1}, 4.0)
    for _ in range(500):
        led.through_filter(1.0, 1000.0)      # a very fine element, many passes
    assert led.suspended_fine_g > 0.0
    assert led.suspended_coarse_g == pytest.approx(0.0, abs=1e-9)


def test_a_sample_after_an_oil_change_reads_clean_on_a_dying_engine():
    led = _ledger_after({"connecting_rod_bearings": 0.0},
                        {"connecting_rod_bearings": 0.4}, 4.0)
    before = led.ppm_by_metal(5.0)["Cu"]
    led.oil_change()
    after = led.ppm_by_metal(5.0)["Cu"]
    assert after < before * 0.2
    assert led.is_balanced()          # the metal did not vanish, it drained


def test_the_calibration_engine_reads_healthy_at_its_sampling_interval():
    """ASTM D5185 on the catalogue's Cat C18 at 500 h -- under caution
    thresholds, and in the right order."""
    import crankcase_state as cs
    engine = engines.get("cat-c18-industrial-diesel")
    sump = cs.CrankcaseState.from_engine(engine)
    led = _ledger_after(
        {"piston_rings": 0.20, "connecting_rod_bearings": 0.15, "valve_guides": 0.10},
        {"piston_rings": 0.203, "connecting_rod_bearings": 0.1515, "valve_guides": 0.1012},
        engine.displacement_l)
    for _ in range(500):
        led.through_filter(0.95, 200.0)
        led.settle(0.0006)
    ppm = led.ppm_by_metal(sump.oil_capacity_kg)
    assert ppm["Fe"] < 100.0 and ppm["Cu"] < 40.0 and ppm["Cr"] < 15.0
    assert ppm["Fe"] > ppm["Cr"] > ppm["Pb"]


def test_the_ledger_is_vector_only():
    """One way through the module. A scalar per-component entry point
    would be a second path to keep correct."""
    import wear_debris as wd
    assert not hasattr(wd.DebrisLedger, "debit_wear")
    assert wd.METAL_MATRIX.shape == (len(wd.OIL_WETTED_COMPONENTS), len(wd.ELEMENTS))


# --- wear opens the rings, which is what blow-by is -------------------

def test_worn_rings_leak_and_lose_compression():
    """`ring_leak` was drawn once from build scatter and never moved, so
    an engine with its rings gone blew by exactly like a new one."""
    import crankcase_state as cs
    engine = engines.get("cat-c18-industrial-diesel")
    st = cs.CrankcaseState.from_engine(engine)
    st.apply_ring_wear(0.0)
    new_leak = float(st.ring_leak.mean())
    new_cr = st.effective_compression_ratio(17.0)
    st.apply_ring_wear(1.0)
    worn_leak = float(st.ring_leak.mean())
    worn_cr = st.effective_compression_ratio(17.0)
    assert worn_leak > new_leak * 3
    assert worn_cr < new_cr
    # and it is the same number read two ways, not a second model
    assert worn_cr == pytest.approx(17.0 * (1.0 - worn_leak))


# --- the ledger actually advances from real duty -----------------------

def _run(identity, steps=1500, accel=20000.0, throttle=0.8):
    import wear as wear_module
    from engine_cycle_sim import EngineCycleSim
    sim = EngineCycleSim(engine=engines.get(identity))
    old = wear_module.WEAR_TIME_ACCELERATION
    wear_module.WEAR_TIME_ACCELERATION = accel
    try:
        sim.throttle = throttle
        sim.start()
        for _ in range(steps):
            sim.step(1 / 200.0)
    finally:
        wear_module.WEAR_TIME_ACCELERATION = old
    return sim


def test_a_freshly_built_sim_has_a_wear_ledger():
    """`state.wear` was only ever created by `set_engine`, which a sim
    built straight from an engine never calls -- so the whole durability
    model sat at zero unless the operator switched engines and back."""
    from engine_cycle_sim import EngineCycleSim
    sim = EngineCycleSim(engine=engines.get("mazda-b6ze-miata-1990"))
    assert sim.state.wear
    assert "piston_rings" in sim.state.wear


def test_running_the_engine_wears_it_and_dirties_its_oil():
    sim = _run("mazda-b6ze-miata-1990")
    assert sim.state.wear["piston_rings"] > 0.0
    led = sim._oil_debris
    assert led.entered_g > 0.0
    assert led.is_balanced(1e-6)
    ppm = led.ppm_by_metal(sim._crankcase_state.oil_kg)
    assert ppm["Fe"] > 0.0
    assert ppm["Fe"] > ppm.get("Cu", 0.0)      # rings before shells, on a young engine


def test_the_duty_clock_moves_metal_and_soot_together():
    """One accelerator for everything that accumulates with duty. If it
    sped up wear and not soot, an accelerated engine would wear its
    rings out with clean oil."""
    slow = _run("mazda-b6ze-miata-1990", accel=5000.0)
    fast = _run("mazda-b6ze-miata-1990", accel=20000.0)
    assert fast.state.wear["piston_rings"] > slow.state.wear["piston_rings"]
    assert fast.state.oil_soot_frac > slow.state.oil_soot_frac


def test_a_diesel_soots_its_oil_far_harder_than_a_petrol_engine():
    """Soot yield is a MASS, declared per combustion family -- not the
    flame's luminosity, which is how bright it looks. Reusing luminosity
    said petrol sooted its oil not quite twice as slowly as diesel; the
    real ratio is nearer a hundred to one."""
    import wear_debris as wd
    petrol = wd.soot_into_oil_g(100.0, "gasoline", 0.05)
    diesel = wd.soot_into_oil_g(100.0, "diesel", 0.05)
    clean = wd.soot_into_oil_g(100.0, "methanol", 0.05)
    assert diesel > petrol > clean


def test_worn_rings_blacken_the_oil_faster():
    """The loop that was never closed: ring wear opens the rings, blow-by
    is what carries soot to the sump, so a tired pack soots its own oil."""
    import wear_debris as wd
    tight = wd.soot_into_oil_g(1000.0, "diesel", 0.05)
    worn = wd.soot_into_oil_g(1000.0, "diesel", 0.28)
    assert worn > tight * 4


def test_soot_is_not_reported_as_a_metal():
    """A spectrographic analysis does not report soot -- it is measured
    separately, as a percentage by mass."""
    import wear_debris as wd
    led = wd.DebrisLedger()
    led.debit_soot(500.0)
    assert led.ppm_by_metal(5.0) == {}
    assert led.soot_frac_of_oil(5.0) > 0.0
    assert led.is_balanced(1e-9)


def test_soot_yield_is_a_mass_not_a_brightness():
    """The error this replaced: `soot_luminosity` describes appearance.
    A premixed petrol flame can look yellow and make almost no
    particulate; a diesel diffusion flame makes it by the gram."""
    import wear_debris as wd
    import combustion_kernel as ck
    assert (wd.soot_yield_g_per_kg("diesel")
            > wd.soot_yield_g_per_kg("gasoline") * 50)
    # and the family is resolved by its own accessor, not by a visual
    assert ck.family_for(engines.get("cat-c18-industrial-diesel")) == "diesel"
    assert ck.family_for(engines.get("mazda-b6ze-miata-1990")) == "gasoline"


def test_the_bake_uses_the_engines_own_rating():
    """A kW-per-litre fallback rated the Wartsila at 633 MW against a
    real 80, which put nearly a tonne of soot in its sump."""
    import wear_profile as wp
    row = wp.bake(engines.get("wartsila-rta96c-14cyl-marine-diesel"), 250.0, steps=40)
    assert row.soot_frac < 0.25          # not the 28% the bad rating gave
    assert row.fuel_burned_kg > 0.0


def test_the_bake_orders_engines_the_way_fuels_do():
    """Diesels soot their oil hardest, petrol barely at all."""
    import wear_profile as wp
    diesel = wp.bake(engines.get("cat-c18-industrial-diesel"), 250.0, steps=40)
    petrol = wp.bake(engines.get("mazda-b6ze-miata-1990"), 250.0, steps=40)
    assert diesel.soot_frac > petrol.soot_frac * 5


def test_only_engines_with_published_figures_are_calibration_points():
    """Inventing a spec for the rest would make the calibration a
    circle."""
    import wear_profile as wp
    rows = [wp.bake(engines.get(i), 250.0, steps=20) for i in wp.INDUSTRIAL_SPECS]
    assert wp.calibration_report(rows)
    unspecced = wp.bake(engines.get("superbike-i4-1340"), 250.0, steps=20)
    assert unspecced.spec.source == ""
    assert wp.calibration_report([unspecced]) == []


# --- centrifuges: what is too dense to stay, not too big to pass ------

def test_a_purifier_cuts_sub_micron_where_a_filter_cannot():
    """The whole reason a ship carries one: soot agglomerates to about a
    micrometre, far below any full-flow element, and a disc stack still
    takes most of it."""
    import centrifuges as cf
    unit = cf.Centrifuge(kind="disc-stack-self-cleaning")
    q = 3.0 / 3600.0
    assert unit.cut_size_m(q, 371.15) < 1.0e-6
    assert unit.grade_efficiency(1.0e-6, q, 371.15) > 0.7
    assert 5_000 < unit.g_force < 20_000          # what these are advertised at


def test_cold_oil_nearly_stops_a_centrifuge():
    """Separation goes inversely with viscosity, which is why a purifier
    is fed through a heater."""
    import centrifuges as cf
    unit = cf.Centrifuge(kind="disc-stack-self-cleaning")
    q = 3.0 / 3600.0
    hot = unit.grade_efficiency(1.0e-6, q, 371.15)
    cold = unit.grade_efficiency(1.0e-6, q, 313.15)
    assert hot > cold * 2


def test_oil_viscosity_matches_a_real_sae_40():
    import centrifuges as cf
    # 150 cSt at 40 C, 15 cSt at 100 C
    assert cf.oil_viscosity_pa_s(313.15) / 870.0 * 1e6 == pytest.approx(150.0, rel=0.05)
    assert cf.oil_viscosity_pa_s(373.15) / 870.0 * 1e6 == pytest.approx(15.0, rel=0.10)


def test_a_faster_bigger_bowl_cuts_finer():
    """Everything follows from geometry and speed -- it is not a rated
    efficiency anyone picks."""
    import centrifuges as cf
    q = 3.0 / 3600.0
    slow = cf.Centrifuge(kind="disc-stack-self-cleaning", speed_rpm=4000.0)
    fast = cf.Centrifuge(kind="disc-stack-self-cleaning", speed_rpm=12000.0)
    assert fast.cut_size_m(q) < slow.cut_size_m(q)
    assert fast.sigma_m2 > slow.sigma_m2


def test_a_self_cleaning_bowl_cleans_itself():
    """A continuous machine ejects and carries on; without that it
    silently packed three hundred kilograms into a twenty-seven
    kilogram bowl and kept working."""
    import centrifuges as cf
    import wear_debris as wd
    unit = cf.Centrifuge(kind="disc-stack-self-cleaning")
    led = wd.DebrisLedger()
    led.debit_soot(400_000.0)
    for _ in range(400):
        cf.clean_oil(unit, led, 3.0 / 3600.0, 3600.0, 200.0, 371.15)
    assert unit.discharges > 0
    assert unit.sludge_kg <= unit.sludge_capacity_kg
    assert not unit.is_full


def test_a_single_action_bowl_fills_and_stops():
    """Cheap, effective, and entirely dependent on somebody opening it."""
    import centrifuges as cf
    import wear_debris as wd
    unit = cf.Centrifuge(kind="bypass-spinner", bowl_radius_m=0.055,
                         inner_radius_m=0.020, bowl_height_m=0.12, speed_rpm=6000.0)
    led = wd.DebrisLedger()
    led.debit_soot(200_000.0)
    for _ in range(4000):
        cf.clean_oil(unit, led, 0.3 / 3600.0, 3600.0, 45.0, 363.15)
    assert unit.is_full
    assert unit.discharges == 0                       # it has no discharge mechanism
    assert unit.grade_efficiency(1e-6, 0.3 / 3600.0) == 0.0   # and removes nothing more
    unit.discharge()                                  # somebody scrapes it out
    assert unit.grade_efficiency(1e-6, 0.3 / 3600.0) > 0.0


def test_a_purifier_saves_the_oil_a_marine_engine_would_otherwise_condemn():
    """The gap this module was built for: the RTA96C read near ten
    percent soot because nothing modelled the separator every real
    installation has."""
    import centrifuges as cf
    import wear_debris as wd
    import wear_profile as wp
    row = wp.bake(engines.get("wartsila-rta96c-14cyl-marine-diesel"), 250.0, steps=60)
    assert row.oil_condemned                          # without one
    led = wd.DebrisLedger()
    led.debit_soot(row.soot_frac * row.sump_kg * 1000.0)
    unit = cf.Centrifuge(kind="disc-stack-self-cleaning", bowl_radius_m=0.30,
                         inner_radius_m=0.12, bowl_height_m=0.35, speed_rpm=7000.0)
    for _ in range(200):
        cf.clean_oil(unit, led, 40.0 / 3600.0, 250.0 * 3600.0 / 200, row.sump_kg, 371.15)
    assert led.soot_frac_of_oil(row.sump_kg) < wp.CONDEMN_SOOT_FRAC


def test_an_undeclared_centrifuge_class_raises():
    import centrifuges as cf
    with pytest.raises(KeyError):
        cf.centrifuge_class("salad-spinner")


# --- the seam to a wetting engine -------------------------------------

def test_a_spray_records_where_it_landed_not_just_what_it_hit():
    """A wetting engine puts fluid on TRIANGLES, so it needs the impact
    point. Keeping only the part name threw that away."""
    from engine_rays import RayMesh
    from hole_emitters import HoleEmitterField
    graph = build_drivetrain_graph(engines.get("amc-258-jeep-i6"))
    field = HoleEmitterField()
    field.add_splash_from_graph(graph)
    field.resolve_spray(RayMesh.from_graph(graph))
    em = field.emitters[0]
    assert em.spray_points
    part, point, normal = em.spray_points[0]
    assert isinstance(part, str) and len(point) == 3 and len(normal) == 3


def test_wetting_is_a_mode_not_an_assumption():
    """A liquid jet wets what it lands on; a gas leak reaches the same
    surfaces and wets nothing."""
    from hole_emitters import HoleEmitter, HoleEmitterField, wetting_deposits
    field = HoleEmitterField()
    gas = HoleEmitter(identity="g", part="p", circuit="intake-air", fluid="gas",
                      position=(0, 0, 0), direction=(0, 1, 0), radius_m=0.003,
                      through=True, wets_surfaces=False)
    gas.spray_resolved = True
    gas.spray_points = (("wall", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),)
    gas.mass_flow_kg_s = 0.01
    gas.escaped_frac = 0.0
    field.emitters.append(gas)
    assert wetting_deposits(field) == []
    gas.wets_surfaces = True
    gas.regime = "spray"
    assert wetting_deposits(field)


def test_an_unresolved_spray_is_not_guessed_at():
    """It has no idea where it is pointing yet."""
    from hole_emitters import HoleEmitter, HoleEmitterField, wetting_deposits
    field = HoleEmitterField()
    em = HoleEmitter(identity="x", part="p", circuit="oil", fluid="engine-oil",
                     position=(0, 0, 0), direction=(0, 1, 0), radius_m=0.003, through=True)
    em.mass_flow_kg_s = 0.01
    em.regime = "spray"
    field.emitters.append(em)
    assert wetting_deposits(field) == []      # not resolved
    em.spray_resolved = True
    em.spray_points = (("wall", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),)
    em.escaped_frac = 0.0
    assert wetting_deposits(field)


def test_ejected_sludge_is_its_own_fluid():
    """Treating it as engine oil lost a fifth of every spill in the round
    trip between a centrifuge's kilograms and an emitter's litres."""
    import centrifuges as cf
    from hole_emitters import HoleEmitterField
    field = HoleEmitterField()
    unit = cf.Centrifuge(kind="disc-stack-self-cleaning")
    cf.spill_onto(field, unit, "powertrain.oil_purifier", 684.3, position=(0.4, 0.6, 0.2))
    assert sum(field.wetted_kg.values()) == pytest.approx(684.3, rel=1e-6)
    assert cf.SLUDGE_DENSITY_KG_M3 > 1000.0     # denser than the oil it came from


def test_a_full_sludge_tank_spills_out_of_the_ejection_port():
    """The bowl ejects on its timer whether or not there is anywhere for
    it to go. That is not an error to clamp away -- it is a real mess."""
    import centrifuges as cf
    import wear_debris as wd
    unit = cf.Centrifuge(kind="disc-stack-self-cleaning")
    tank = cf.SludgeReceiver(capacity_kg=2.0)
    led = wd.DebrisLedger()
    led.debit_soot(400_000.0)
    spilled = 0.0
    for _ in range(400):
        spilled += cf.clean_oil(unit, led, 3.0 / 3600.0, 3600.0, 200.0, 371.15,
                                receiver=tank)["spilled_kg"]
    assert tank.is_full
    assert spilled > 0.0
    assert tank.overflowed_kg == pytest.approx(spilled)


def test_no_receiver_means_all_of_it_goes_on_the_floor():
    import centrifuges as cf
    unit = cf.Centrifuge(kind="disc-stack-self-cleaning")
    unit.sludge_kg = unit.sludge_capacity_kg
    assert cf.discharge_to(unit, None) == pytest.approx(unit.sludge_capacity_kg)


# --- what turns a separator -------------------------------------------

def test_an_industrial_separator_runs_on_a_motor():
    import centrifuges as cf
    unit = cf.Centrifuge(kind="disc-stack-self-cleaning", bowl_radius_m=0.30,
                         inner_radius_m=0.12, bowl_height_m=0.35,
                         speed_rpm=7000.0, motor_kw=11.0)
    assert unit.drive.startswith("electric")
    # a bowl that size holds megajoules and takes minutes to come up
    assert cf.stored_energy_j(unit) > 1.0e6
    assert 120.0 < cf.spin_up_time_s(unit) < 1200.0
    # and far less to hold there than to start
    assert cf.steady_power_w(unit) < unit.motor_kw * 1000.0


def test_a_bypass_spinner_has_no_motor_and_needs_a_supply():
    """Its speed is a RESULT of supply pressure -- take the oil away and
    it stops, and stops cleaning."""
    import centrifuges as cf
    unit = cf.Centrifuge(kind="bypass-spinner", bowl_radius_m=0.055,
                         inner_radius_m=0.020, bowl_height_m=0.12,
                         drive="oil-jet", motor_kw=0.0)
    assert cf.jet_equilibrium_rpm(unit, 101_325.0) == 0.0         # no pressure, no spin
    assert cf.spin_up_time_s(unit, 101_325.0) == float("inf")
    four_bar = 101_325.0 + 4.0e5
    rpm = cf.jet_equilibrium_rpm(unit, four_bar)
    assert 4_000.0 < rpm < 7_000.0                                # what these really do
    assert 0.5 < cf.jet_flow_m3_s(unit, four_bar) * 60_000.0 < 4.0  # L/min
    # more pressure, more speed
    assert cf.jet_equilibrium_rpm(unit, 101_325.0 + 6.0e5) > rpm


def test_a_spinner_spins_down_when_the_supply_is_cut():
    import centrifuges as cf
    unit = cf.Centrifuge(kind="bypass-spinner", bowl_radius_m=0.055,
                         inner_radius_m=0.020, bowl_height_m=0.12,
                         drive="oil-jet", motor_kw=0.0)
    four_bar = 101_325.0 + 4.0e5
    for _ in range(400):
        cf.step_drive(unit, 1.0, powered=True, supply_pa=four_bar)
    assert unit.current_rpm > 1000.0
    for _ in range(400):
        cf.step_drive(unit, 1.0, powered=False, supply_pa=0.0)
    assert unit.current_rpm < 100.0


def test_a_light_rotor_is_not_a_steel_bowl():
    """Sizing both as solid steel made a spinner's rotor five times too
    heavy and gave it a six-minute run-up."""
    import centrifuges as cf
    assert (cf.centrifuge_class("bypass-spinner").bowl_density_kg_m3
            < cf.centrifuge_class("disc-stack-self-cleaning").bowl_density_kg_m3)


# --- the station's own recycling --------------------------------------

def test_the_station_can_recycle_its_own_fluids():
    import station_powerplant as sp
    bank = sp.fluid_recycling_bank()
    assert bank.power == "shore-supplied"
    seps = [p for p in bank.parts if p.kind == "centrifugal-separator"]
    assert len(seps) == 3
    fluids = {p.fluid for p in seps}
    assert fluids == {"engine-oil", "diesel", "coolant-water-glycol"}


def test_oil_fuel_and_coolant_never_share_a_bowl():
    """Cross-contaminating any two is worse than never cleaning either."""
    import station_powerplant as sp
    bank = sp.fluid_recycling_bank()
    seps = [p for p in bank.parts if p.kind == "centrifugal-separator"]
    assert len({p.identity for p in seps}) == len(seps)
    for p in seps:
        feeds = {q.fluid for q in p.ports if q.kind.endswith("-in") or q.kind.endswith("-out")}
        assert feeds == {p.fluid}          # one fluid per machine, in and out


def test_the_bank_has_somewhere_for_the_sludge_to_go():
    import station_powerplant as sp
    bank = sp.fluid_recycling_bank()
    tank = next(p for p in bank.parts if p.kind == "sludge-receiver")
    assert tank.capacity_kg > 0.0
    ejects = [q for p in bank.parts for q in p.ports if q.kind == "sludge-discharge"]
    assert len(ejects) == 3
    for q in ejects:
        assert q.connected_to == tank.identity


# --- fluidic, multi-track separation ----------------------------------

def test_a_cyclone_bank_has_no_moving_parts_and_only_wants_pressure():
    """Which is why it cannot be stopped by cutting power."""
    import centrifuges as cf
    bank = cf.CycloneBank(cyclone_diameter_m=0.010, count=60)
    q = 6.0 / 3600.0
    assert bank.pressure_drop_pa(q) > 0.0
    assert bank.cut_size_m(q) > 0.0
    assert not hasattr(bank, "motor_kw")


def test_more_smaller_cyclones_cut_finer():
    """The whole reason banks are multi-track: cut size falls with
    cyclone diameter, throughput falls with its square, so a fine cut at
    a useful flow means many small ones in parallel."""
    import centrifuges as cf
    q = 6.0 / 3600.0
    coarse = cf.CycloneBank(cyclone_diameter_m=0.050, count=4)
    fine = cf.CycloneBank(cyclone_diameter_m=0.010, count=60)
    assert fine.cut_size_m(q) < coarse.cut_size_m(q)


def test_a_cyclone_is_hopeless_at_soot_and_a_disc_stack_is_not():
    """Neither replaces the other, which is why a plant carries both."""
    import centrifuges as cf
    q = 6.0 / 3600.0
    bank = cf.CycloneBank(cyclone_diameter_m=0.010, count=60)
    stack = cf.Centrifuge(kind="disc-stack-self-cleaning")
    soot = cf.PARTICLE_SIZE_M["soot"]
    assert bank.grade_efficiency(soot, q, 371.15) < 0.05
    assert stack.grade_efficiency(soot, 3.0 / 3600.0, 371.15) > 0.7


def test_viscosity_is_the_whole_story_for_a_cyclone():
    import centrifuges as cf
    q = 6.0 / 3600.0
    bank = cf.CycloneBank(cyclone_diameter_m=0.010, count=60)
    water = bank.cut_size_m(q, particle_density_kg_m3=2520.0, viscosity_pa_s=0.001)
    cold_oil = bank.cut_size_m(q, particle_density_kg_m3=1800.0, viscosity_pa_s=0.130)
    assert cold_oil > water * 5
    # a real desilter on water and grit cuts in the teens of micrometres
    assert 5e-6 < water < 30e-6


def test_cyclone_pressure_drop_rises_with_the_square_of_flow():
    """Which is why a bank is sized by adding cyclones, not by pushing
    harder through the ones it has."""
    import centrifuges as cf
    bank = cf.CycloneBank(cyclone_diameter_m=0.010, count=60)
    q = 6.0 / 3600.0
    assert bank.pressure_drop_pa(2 * q) == pytest.approx(bank.pressure_drop_pa(q) * 4.0)


# --- expiry, auto-drains, and somebody to do it -----------------------

def test_an_auto_draining_unit_still_owes_a_manual_service():
    """A self-cleaning bowl ejects for months and still cokes up between
    the discs. The discharge buys a long interval, not exemption."""
    import servicing as sv
    item = sv.ServiceItem("sep", "disc-stack-self-cleaning", hours=2100.0)
    assert item.auto_drain
    assert any(t.key == "bowl-strip" for t, _ in item.due())


def test_a_unit_with_no_drain_is_entirely_dependent_on_somebody():
    import servicing as sv
    solid = sv.ServiceItem("bowl", "solid-bowl-batch", condition=0.95)
    assert not solid.auto_drain
    assert any(t.key == "bowl-scrape" for t, _ in solid.due())


def test_three_different_ways_of_becoming_due():
    """Hours, full, and measured condition are genuinely different
    questions -- a sludge tank does not care how long it took to fill."""
    import servicing as sv
    by_hours = sv.ServiceItem("cyc", "cyclone-bank", hours=800.0)
    by_full = sv.ServiceItem("tank", "sludge-receiver", condition=0.9)
    by_cond = sv.ServiceItem("elem", "filter-element", condition=0.85)
    for item in (by_hours, by_full, by_cond):
        assert item.due(), item.identity
    assert not sv.ServiceItem("cyc2", "cyclone-bank", hours=10.0).due()


def test_a_service_entry_is_what_a_crew_task_queue_needs():
    """Where to go, what to do, how long, whether to stop the machine,
    and what to bring."""
    import servicing as sv
    items = [sv.ServiceItem("e", "filter-element", position=(1.0, 2.0, 3.0),
                            condition=0.9, label="oil element")]
    entry = sv.due_tasks(items)[0]
    assert entry["position"] == (1.0, 2.0, 3.0)
    assert entry["minutes"] > 0.0
    assert entry["requires_shutdown"] is True
    assert entry["consumes"] == "filter-element"
    assert entry["skill"]


def test_the_most_overdue_comes_first():
    import servicing as sv
    items = [sv.ServiceItem("a", "filter-element", condition=0.81, label="a"),
             sv.ServiceItem("b", "filter-element", condition=2.00, label="b")]
    assert [e["label"] for e in sv.due_tasks(items)] == ["b", "a"]


def test_servicing_clears_what_it_was_for():
    import servicing as sv
    item = sv.ServiceItem("sep", "disc-stack-self-cleaning", hours=3000.0)
    assert item.due()
    item.serviced("bowl-strip")
    item.serviced("auto-drain-check")
    assert not item.due()


def test_the_whole_recycling_bank_can_be_asked_what_it_owes():
    import servicing as sv
    import station_powerplant as sp
    bank = sp.fluid_recycling_bank()
    items = []
    for p in bank.parts:
        if p.kind == "centrifugal-separator":
            items.append(sv.ServiceItem(p.identity, "disc-stack-self-cleaning",
                                        p.position, hours=2400.0))
        elif p.kind == "sludge-receiver":
            items.append(sv.ServiceItem(p.identity, "sludge-receiver",
                                        p.position, condition=0.95))
    entries = sv.due_tasks(items)
    assert entries
    assert sv.workload_minutes(entries) > 0.0
    # every entry knows where it is, so somebody can be sent there
    assert all(len(e["position"]) == 3 for e in entries)


def test_an_undeclared_service_profile_raises():
    import servicing as sv
    with pytest.raises(KeyError):
        sv.ServiceItem("x", "wishful-thinking").due()


# --- the station as a plant somebody has to run -----------------------

def test_canvas_stops_weather_and_nothing_else():
    """A shelter is cover from rain and from being seen. Pretending a
    tent is a steel room is how an installation feels safer than it is."""
    from ballistics import MATERIAL_PROFILES
    canvas = MATERIAL_PROFILES["canvas"]
    steel = MATERIAL_PROFILES["cover"]
    assert canvas.toughness_j_m3 < steel.toughness_j_m3 / 100


def test_the_shelter_is_a_frame_and_a_fabric_not_a_building():
    import station_powerplant as sp
    sh = sp.plant_shelter()
    mats = {p.material for p in sh.parts}
    assert mats == {"steel-tube", "canvas"}
    # open at one end: three panels and a roof, not four walls
    assert len([p for p in sh.parts if p.material == "canvas"]) == 4


def test_a_piped_route_costs_nothing_a_day_and_everything_if_cut():
    """Reporting the same portering figure for both said the line was
    worthless."""
    import station_powerplant as sp
    piped = sp.fuel_supply_route(piped=True)
    drums = sp.fuel_supply_route(piped=False)
    assert piped["porter_crew_hours_per_day"] == 0.0
    assert drums["porter_crew_hours_per_day"] > 0.0
    assert piped["fallback_crew_hours_per_day"] == pytest.approx(
        drums["porter_crew_hours_per_day"])
    assert piped["single_point_of_failure"] and not drums["single_point_of_failure"]


def test_a_hose_run_is_sections_and_somebody_has_to_lay_them():
    import station_powerplant as sp
    machine, runs = sp.fuel_hose_set()
    main = next(r for r in runs if r.spec == "bulk-6in")
    assert not main.complete and main.reach_m == 0.0
    need = main.need()
    assert need is not None and need.want == "hose-section"
    assert need.minutes > 60.0                       # it is a real job of work
    main.sections_laid = main.sections_required
    assert main.complete and main.need() is None


def test_a_cut_line_stops_at_the_cut():
    import station_powerplant as sp
    _, runs = sp.fuel_hose_set()
    main = next(r for r in runs if r.spec == "bulk-6in")
    main.sections_laid = main.sections_required
    full = main.reach_m
    main.cut_sections = 2
    assert main.reach_m < full
    assert not main.complete
    assert main.need().quantity == 2


def test_the_station_carries_drums_as_the_fallback():
    import station_powerplant as sp
    machine, _ = sp.fuel_hose_set()
    drums = [p for p in machine.parts if p.kind == "fuel-drum"]
    assert len(drums) >= 4
    for d in drums:
        assert d.fluid_volume_l == pytest.approx(sp.DRUM_CAPACITY_L)
        assert d.mass_kg > 150.0                     # why portering is expensive


def test_oil_can_be_dried_and_only_one_machine_does_it():
    """Free water settles or throws out; dissolved water does neither.
    Vacuum dehydration is the only one of these that removes it."""
    import centrifuges as cf
    d = cf.VacuumDehydrator()
    assert 0.3 < d.removal_fraction_per_pass() < 0.8
    dried = d.step(4 * 3600.0, 400.0, 1200.0)
    assert dried["water_ppm"] < 60.0
    # and it cannot dry below what the vacuum it pulls allows
    assert dried["water_ppm"] > 0.0


def test_a_vacuum_too_shallow_for_the_temperature_does_nothing():
    import centrifuges as cf
    weak = cf.VacuumDehydrator(vacuum_pa=30_000.0, oil_temp_k=313.15)
    assert weak.driving_pressure_pa == 0.0
    assert weak.removal_fraction_per_pass() == 0.0


def test_hot_oil_holds_more_water_in_solution():
    """Which is why a clear hot sample drops free water overnight."""
    import centrifuges as cf
    assert cf.oil_saturation_ppm(363.15) > cf.oil_saturation_ppm(293.15) * 3


def test_the_dehydrator_reclaims_the_water_it_drives_off():
    """It cannot vanish: the vapour is condensed before the pump, because
    water through a vacuum pump destroys it."""
    import centrifuges as cf
    d = cf.VacuumDehydrator()
    pot = cf.CondensateReceiver(capacity_kg=25.0)
    dried = d.step(3600.0, 400.0, 1200.0)
    got = cf.reclaim_condensate(d, dried["removed_g"], pot)
    assert got["water_kg"] > 0.0
    assert pot.held_kg == pytest.approx(got["water_kg"] + got["oil_kg"])
    assert got["condenser_duty_j"] > 0.0


def test_the_reclaimed_water_is_dirty():
    """It carries oil mist, the acids oxidised oil makes, and often the
    glycol that was the reason for drying it."""
    import centrifuges as cf
    demisted = cf.reclaim_condensate(cf.VacuumDehydrator(), 500.0, None, demisted=True)
    bare = cf.reclaim_condensate(cf.VacuumDehydrator(), 500.0, None, demisted=False)
    assert demisted["oil_frac"] > 0.0
    assert bare["oil_frac"] > demisted["oil_frac"] * 10


def test_a_full_condensate_pot_asks_to_be_emptied():
    import centrifuges as cf
    pot = cf.CondensateReceiver(capacity_kg=1.0, held_kg=0.95, oil_kg=0.004)
    need = pot.need("station.dehydrator.pot", (1.0, 0.0, 1.0))
    assert need is not None and need.want == "drain"
    assert need.matches == "oily-water-waste"
    assert cf.CondensateReceiver(capacity_kg=1.0, held_kg=0.1).need("x") is None


def test_an_overflowing_pot_does_not_silently_swallow_it():
    import centrifuges as cf
    pot = cf.CondensateReceiver(capacity_kg=0.10)
    spill = cf.reclaim_condensate(cf.VacuumDehydrator(), 400.0, pot)["overflow_kg"]
    assert pot.is_full
    assert spill > 0.0
    assert pot.overflowed_kg == pytest.approx(spill)


# --- cryogenics -------------------------------------------------------

def test_throttling_hydrogen_from_ambient_heats_it():
    """Its inversion temperature is below room temperature, so a plant
    that throttles it from ambient makes heat, in the one gas whose
    ignition energy is a static spark."""
    import cryogenics as cg
    assert not cg.throttling_cools("liquid-hydrogen", 300.0)
    assert cg.hydrogen_precool_required(300.0)
    assert cg.throttling_cools("liquid-hydrogen", 150.0)
    # nitrogen is the opposite case, which is why it does the pre-cooling
    assert cg.throttling_cools("liquid-nitrogen", 300.0)


def test_a_litre_of_liquid_air_is_a_lot_of_gas():
    import cryogenics as cg
    assert cg.cryogen("liquid-air").expansion_ratio > 600.0


def test_mli_is_astonishing_only_in_a_vacuum():
    """Let air in and it is a stack of conductive foils -- worse than
    perlite, which is why a jacket failure is a cliff, not a slope."""
    import cryogenics as cg
    mli = cg.insulation("mli")
    assert mli.needs_vacuum
    assert mli.k_w_mk < cg.insulation("pu-foam").k_w_mk / 100
    assert mli.k_spoiled_w_mk > cg.insulation("vacuum-perlite").k_w_mk


def test_a_dewar_that_loses_its_vacuum_falls_off_a_cliff():
    import cryogenics as cg
    v = cg.VacuumJacketedVessel(capacity_l=200.0, fill_l=180.0)
    good = v.boil_off_percent_per_day()
    assert 0.05 < good < 3.0                     # what a real vessel is specified at
    v.vacuum_intact = False
    assert v.boil_off_percent_per_day() > good * 50
    assert v.need() is not None                  # and it asks for attention


def test_a_cryogenic_vessel_must_be_able_to_vent():
    """It is a machine for turning liquid into hundreds of times its
    volume of gas. The only question is whether the gas has somewhere
    to go."""
    import cryogenics as cg
    v = cg.VacuumJacketedVessel(capacity_l=50.0, fill_l=45.0, vacuum_intact=False,
                               relief_fitted=True)
    burst = False
    for _ in range(200):
        burst |= v.step(60.0)["burst"]
    assert not burst and v.vented_kg > 0.0
    sealed = cg.VacuumJacketedVessel(capacity_l=50.0, fill_l=45.0, vacuum_intact=False,
                                     relief_fitted=False)
    burst = False
    for _ in range(200):
        burst |= sealed.step(60.0)["burst"]
    assert burst


def test_the_cold_head_cools_harder_than_a_throttle_and_makes_liquid():
    import cryogenics as cg
    x = cg.Turboexpander()
    warm = x.step(1.0, 300.0)
    cold = x.step(1.0, 120.0)
    assert warm["outlet_k"] < 300.0
    assert warm["liquid_frac"] == 0.0            # not cold enough yet
    assert cold["liquid_frac"] > 0.0             # fed cold, it makes liquid
    assert cold["shaft_w"] > 0.0                 # and the work has to go somewhere


def test_air_separates_because_nitrogen_boils_first():
    import cryogenics as cg
    n2, o2 = cg.cryogen("liquid-nitrogen"), cg.cryogen("liquid-oxygen")
    assert o2.boil_k - n2.boil_k > 10.0
    short = cg.separation_products(100.0, 3)
    tall = cg.separation_products(100.0, 40)
    assert tall["nitrogen"]["purity"] > short["nitrogen"]["purity"]
    assert tall["nitrogen"]["litres"] > tall["oxygen"]["litres"] * 3


def test_hydrogen_is_compressed_without_a_sliding_seal():
    """It embrittles steel and leaks through seals nothing else leaks
    through."""
    import cryogenics as cg
    for key in ("hydrogen-diaphragm", "hydrogen-ionic"):
        assert cg.compressor_duty(key).seal_kind != "labyrinth"
    assert cg.compression_power_w("air-plant", 0.5, 293.15, 30.0) > 0.0


# --- clearing work is a held interaction ------------------------------

def test_a_clearing_task_is_a_delayed_touch_job():
    """It finishes when somebody has stayed there long enough, not when
    they arrive."""
    import servicing as sv
    item = sv.ServiceItem("sludge", "sludge-receiver", (0.0, -1.0, 0.0), condition=0.95)
    job = sv.touch_jobs([item])[0]
    assert job.seconds > 60.0
    assert not job.complete
    job.work(job.seconds * 0.5)
    assert 0.4 < job.progress < 0.6
    assert job.work(job.seconds)
    assert job.complete


def test_a_job_that_needed_the_machine_stopped_cannot_be_left_half_done():
    import servicing as sv
    strip = sv.ServiceItem("sep", "disc-stack-self-cleaning", hours=2400.0)
    jobs = sv.touch_jobs([strip])
    hard = next(j for j in jobs if j.requires_shutdown)
    soft = next(j for j in jobs if not j.requires_shutdown)
    assert not hard.interruptible and soft.interruptible
    hard.work(hard.seconds * 0.4)
    assert hard.interrupt() == 0.0                # starts again
    soft.work(soft.seconds * 0.4)
    assert soft.interrupt() > 0.3                 # a bowl half scraped stays half scraped


def test_needs_and_service_tasks_produce_the_same_kind_of_job():
    """One kind of object for an interaction system to path to."""
    import servicing as sv
    item = sv.ServiceItem("elem", "filter-element", (1.0, 0.0, 0.0), condition=0.9)
    need = sv.barrel_need("barrel", (2.0, 0.0, 0.0), 0.1, 205.0, "diesel")
    jobs = sv.touch_jobs([item], [need])
    assert len(jobs) == 2
    for j in jobs:
        assert len(j.position) == 3 and j.seconds > 0.0 and j.clears


# --- the dewar has an atmosphere in it --------------------------------

def test_what_boils_off_is_not_what_is_in_there():
    """Nitrogen boils thirteen degrees colder, so the vapour is far
    richer in it -- which is the whole mechanism of separating air, and
    of a neglected vessel enriching itself in oxygen."""
    import cryogenics as cg
    liquid = {"nitrogen": 0.755, "oxygen": 0.232, "argon": 0.013}
    vap = cg.vapour_composition(liquid)
    assert vap["nitrogen"] > liquid["nitrogen"]
    assert vap["oxygen"] < liquid["oxygen"] / 2


def test_a_standing_vessel_enriches_itself_in_oxygen():
    """Nobody does anything to it. Old liquid air is dangerous in a way
    fresh liquid air is not."""
    import cryogenics as cg
    v = cg.CryogenicVessel(capacity_l=200.0, fill_l=170.0, vacuum_intact=False)
    start = v.oxygen_frac
    for _ in range(400):
        v.step(600.0, humid=False)
    assert v.oxygen_frac > start


def test_an_iced_vent_bursts_the_vessel():
    """The classic failure: the heat leak keeps boiling the contents
    whether or not the gas has anywhere to go."""
    import cryogenics as cg
    v = cg.CryogenicVessel(capacity_l=200.0, fill_l=170.0)
    burst = False
    for _ in range(24 * 5):
        burst |= v.step(3600.0, humid=True)["burst"]
    assert v.vent_iced_shut
    assert burst


def test_deicing_keeps_it_alive():
    import cryogenics as cg
    v = cg.CryogenicVessel(capacity_l=200.0, fill_l=170.0)
    for _ in range(24 * 14):
        v.step(3600.0, humid=True)
        if v.deice_need() is not None:
            v.deice()
    assert not v.burst
    assert v.fill_l > 100.0


def test_deicing_is_a_job_somebody_has_to_go_and_do():
    import cryogenics as cg
    import servicing as sv
    v = cg.CryogenicVessel(capacity_l=200.0, fill_l=170.0, position=(3.0, 0.0, 1.0))
    assert v.deice_need() is None                 # clear vent, nothing owed
    v.vent_blocked_frac = 0.7
    need = v.deice_need()
    assert need is not None and need.want == "deice"
    job = sv.touch_job_for_need(need)
    assert job.seconds > 60.0 and job.position == (3.0, 0.0, 1.0)


def test_frost_on_the_outside_means_the_vacuum_has_gone():
    """It is not a cosmetic detail, it is the diagnostic."""
    import cryogenics as cg
    good = cg.CryogenicVessel(fill_l=100.0, vacuum_intact=True)
    bad = cg.CryogenicVessel(fill_l=100.0, vacuum_intact=False)
    assert not good.exterior_frosts and bad.exterior_frosts
    assert bad.heat_leak_w() > good.heat_leak_w() * 100


def test_the_ullage_is_a_real_mixture_at_a_real_pressure():
    import cryogenics as cg
    v = cg.CryogenicVessel(capacity_l=100.0, fill_l=80.0, vacuum_intact=False,
                           relief_fitted=False)
    v.step(600.0, humid=False)
    assert v.ullage.total_kg > 0.0
    fr = v.ullage.fractions()
    assert fr["nitrogen"] > fr["oxygen"]          # what boiled was N2-rich
    assert v.pressure_pa > cg.ATM_PA


def test_venting_does_not_select():
    """A relief valve passes whatever is in front of it."""
    import cryogenics as cg
    a = cg.DewarAtmosphere()
    a.add("nitrogen", 0.8)
    a.add("oxygen", 0.2)
    went = a.remove_proportional(0.5)
    assert went["nitrogen"] / went["oxygen"] == pytest.approx(4.0)


def test_boil_off_arrives_at_the_boiling_point():
    """Cold gas entering a warm ullage has to chill it.

    Mass without temperature is how a gas space ends up warmer than
    anything feeding it."""
    import cryogenics as cg
    a = cg.DewarAtmosphere(kg={"nitrogen": 1.0}, temp_k=200.0)
    a.add("nitrogen", 1.0, at_k=80.0)
    assert a.temp_k == pytest.approx(140.0)
    warm = cg.DewarAtmosphere(kg={"nitrogen": 1.0}, temp_k=200.0)
    warm.add("nitrogen", 1.0)                     # no temperature offered
    assert warm.temp_k == pytest.approx(200.0)    # so none is invented


def test_a_standing_dewar_does_not_depend_on_the_step_size():
    """The ullage temperature sets the pressure, and the pressure is
    what opens the relief valve and bursts the vessel. If it moves with
    the step size then whether a dewar bursts depends on the frame rate.

    Pinned because it did: a fixed nudge per call gave 162 K at dt=60
    and 91 K at dt=3600 over the same six hours."""
    import cryogenics as cg

    def stand(dt):
        v = cg.CryogenicVessel(fill_l=150.0)
        t = 0.0
        while t < 6 * 3600.0:
            v.step(dt)
            t += dt
        return v

    fine, coarse = stand(60.0), stand(3600.0)
    assert fine.ullage.temp_k == pytest.approx(coarse.ullage.temp_k, rel=0.05)
    assert fine.fill_l == pytest.approx(coarse.fill_l, rel=1e-6)
    assert fine.oxygen_frac == pytest.approx(coarse.oxygen_frac, rel=1e-6)


def test_the_ullage_never_overshoots_ambient_however_coarse_the_step():
    """The relaxation is exponential precisely so that it cannot."""
    import cryogenics as cg
    v = cg.CryogenicVessel(fill_l=50.0, insulation_media="mineral-wool",
                           vacuum_intact=False)
    v.step(60.0, humid=False)
    v.step(30 * 86400.0, humid=False)             # a month in one step
    assert v.ullage.temp_k <= cg.AMBIENT_K + 1e-9
    assert v.ullage.temp_k >= v.boil_k - 1e-9


# --- the cold bank, and the hole it lives in --------------------------

def test_freezing_beats_chilling_by_an_order_of_magnitude():
    import thermal_storage as ts
    water = ts.medium("chilled-water").energy_j_per_kg(10.0, use_phase_change=False)
    ice = ts.medium("ice").energy_j_per_kg(0.0, use_phase_change=True)
    assert ice > water * 7


def test_the_sky_will_freeze_water_but_it_wants_area():
    """The yakhchal mechanism: free, silent, and slow."""
    import thermal_storage as ts
    tonne = 1000.0 * ts.ICE_LATENT_J_KG
    clear = ts.sky_pan_area_for(tonne, 10.0, clear=True)
    humid = ts.sky_pan_area_for(tonne, 10.0, clear=False)
    assert 50.0 < clear < 400.0          # real, and large
    assert humid > clear * 3             # cloud and damp close the window


def test_a_chiller_is_worth_running_at_night():
    """Its efficiency is set by what it has to reject heat INTO."""
    import thermal_storage as ts
    night = ts.chiller_cold_j(3000.0, 10.0, 3.0, ambient_k=288.15)
    day = ts.chiller_cold_j(3000.0, 10.0, 3.0, ambient_k=313.15)
    assert night > day * 1.3


def test_the_pit_holds_a_real_load_for_a_real_time():
    import thermal_storage as ts
    pit = ts.IcePit(capacity_kg=8000.0, stored_kg=8000.0)
    assert pit.hold_hours(2000.0) > 200.0        # days, not hours
    assert pit.heat_gain_w() > 0.0               # and it does leak
    melted = pit.melt(86400.0)
    assert melted > 0.0 and pit.stored_kg < 8000.0


def test_drawing_on_the_bank_melts_it_and_charging_freezes_it_back():
    import thermal_storage as ts
    pit = ts.IcePit(capacity_kg=1000.0, stored_kg=1000.0)
    took = pit.draw_j(100e6)
    assert took > 0.0 and pit.water_kg > 0.0
    back = pit.charge_j(50e6)
    assert back > 0.0 and pit.stored_kg > pit.capacity_kg - 1000.0


def test_the_hole_has_to_be_dug_and_the_ground_decides():
    """The same pit is a two-day job or a three-week one depending
    entirely on where it was sited."""
    import thermal_storage as ts
    vol = ts.pit_excavation_m3(12.0, 3.0)
    assert vol > 12.0 * 3.0                      # battered walls are real volume
    sand = ts.Excavation(required_m3=vol, soil="loose-sand").hours_remaining(2)
    caliche = ts.Excavation(required_m3=vol, soil="caliche").hours_remaining(2)
    assert caliche > sand * 5
    machine = ts.Excavation(required_m3=vol, soil="caliche").hours_remaining(1, machine=True)
    assert machine < caliche / 20                # trivial later, expensive early


def test_digging_is_cumulative_and_blunts_the_tools():
    import thermal_storage as ts
    e = ts.Excavation(required_m3=20.0, soil="caliche")
    for _ in range(10):
        e.dig(8.0, workers=2)
    assert 0.0 < e.progress < 1.0                # a debt, not a switch
    assert e.tool_condition < 1.0
    assert e.need() is not None


def test_a_hole_makes_a_heap_bigger_than_itself():
    """Broken ground does not go back into the space it came from."""
    import earthworks as ew
    pile = ew.DirtPile(kind="hardpan")
    pile.add(41.4)
    assert pile.loose_m3 > pile.in_situ_m3
    assert pile.heap_height_m() > 0.0
    assert pile.heap_footprint_m2() > 0.0


def test_spoil_from_a_dig_lands_on_the_pile_once():
    import thermal_storage as ts
    import earthworks as ew
    e = ts.Excavation(required_m3=30.0, soil="ordinary-soil")
    pile = ew.DirtPile(kind="ordinary-soil")
    e.dig(20.0, workers=2)
    ew.spoil_from_excavation(e, pile)
    first = pile.in_situ_m3
    ew.spoil_from_excavation(e, pile)            # again, with no new digging
    assert pile.in_situ_m3 == pytest.approx(first)


def test_a_berm_is_only_cover_if_its_top_is_thick_enough():
    """The base width is what makes a berm look thicker than it is."""
    import earthworks as ew
    thin = ew.Berm(kind="sand", length_m=10.0, height_m=1.6, top_width_m=0.2)
    thick = ew.Berm(kind="sand", length_m=10.0, height_m=1.6, top_width_m=0.8)
    assert not thin.stops_small_arms and thick.stops_small_arms
    assert thin.base_width_m > thin.top_width_m  # and it still looks substantial


def test_sand_will_not_stand_steeply_so_its_berm_is_wide():
    import earthworks as ew
    sand = ew.Berm(kind="sand", length_m=1.0, height_m=1.6, top_width_m=1.0)
    rock = ew.Berm(kind="rock", length_m=1.0, height_m=1.6, top_width_m=1.0)
    assert sand.base_width_m > rock.base_width_m
    assert sand.required_m3 > rock.required_m3


def test_one_ice_pit_does_not_berm_a_station():
    """Worth knowing before anybody plans on it."""
    import thermal_storage as ts
    import earthworks as ew
    pile = ew.DirtPile(kind="hardpan")
    pile.add(ts.pit_excavation_m3(12.0, 3.0))
    ring = ew.berm_around(46.0, "hardpan")
    ring.build_from(pile, 9999.0)
    assert ring.complete_frac < 0.5
    assert ring.need(pile) is not None


# --- somewhere to sleep -----------------------------------------------

def test_the_bunk_is_sealed_until_there_is_somebody_to_put_in_it():
    import station_powerplant as sp
    b = sp.bunk_shelter()
    assert getattr(b, "sealed") is True
    assert getattr(b, "berths") == 2
    # closed on all four sides: no open working face like the plant shelter
    walls = [p for p in b.parts if p.part_role == "bunk-canvas"]
    assert len(walls) == 5                       # four sides and a roof


def test_a_desert_tent_is_double_skinned():
    """A single-skin canvas tent in a desert is an oven."""
    import station_powerplant as sp
    b = sp.bunk_shelter()
    assert any(p.part_role == "bunk-fly-sheet" for p in b.parts)


def test_the_bunk_is_furnished_so_it_is_not_empty_when_it_opens():
    import station_powerplant as sp
    b = sp.bunk_shelter()
    assert len([p for p in b.parts if p.part_role == "crew-bunk"]) == 2
    assert len([p for p in b.parts if p.part_role == "crew-stowage"]) == 2


# --- one job pays for the other ---------------------------------------

def test_the_pit_is_dug_to_yield_exactly_the_critical_berm():
    """Critical for critical: dig the hole whose spoil builds the wall
    that matters, and the two jobs are one job."""
    import thermal_storage as ts
    import earthworks as ew
    crit = ew.Berm("crit", "hardpan", length_m=14.0, height_m=1.6, top_width_m=0.8)
    spec = ts.pit_sized_for_berm(crit)
    assert spec["excavation_m3"] == pytest.approx(crit.required_m3)
    exc = ts.Excavation(required_m3=spec["excavation_m3"], soil="hardpan",
                        depth_m=spec["depth_m"])
    pile = ew.DirtPile(kind="hardpan")
    while not exc.complete:
        exc.dig(8.0, workers=2)
        ew.spoil_from_excavation(exc, pile)
    crit.build_from(pile, 9999.0)
    assert crit.complete_frac == pytest.approx(1.0, abs=0.02)
    assert spec["ice_capacity_kg"] > 0.0


def test_the_valve_will_not_let_the_ice_room_be_spent():
    """Draw above what recharge replaces and a week of digging goes in
    an afternoon."""
    import thermal_storage as ts
    vent = ts.ColdVent(sustainable_w=3500.0)
    easy = vent.draw_w(1000.0)
    hard = vent.draw_w(9000.0)
    assert easy["passed_w"] == pytest.approx(1000.0)
    assert hard["passed_w"] == pytest.approx(3500.0)     # capped
    assert hard["unmet_w"] > 0.0
    # and wide open, with the valve ignored, it really would take it
    assert vent.draw_w(9000.0, respect_valve=False)["passed_w"] == pytest.approx(9000.0)


def test_a_closed_valve_passes_nothing():
    import thermal_storage as ts
    shut = ts.ColdVent(sustainable_w=3500.0, valve_open_frac=0.0)
    assert shut.draw_w(5000.0)["passed_w"] == 0.0


def test_an_empty_pit_has_nothing_to_give():
    import thermal_storage as ts
    pit = ts.IcePit(capacity_kg=1000.0, stored_kg=0.0)
    assert ts.ColdVent().draw_w(2000.0, pit)["passed_w"] == 0.0


def test_diluting_the_discharge_makes_it_nearly_invisible():
    """Radiated power goes as the fourth power of temperature, so a
    modest dilution is a large reduction in what a sensor sees."""
    import thermal_storage as ts
    vent = ts.ColdVent()
    hot = 673.15
    mixed = vent.mixed_exhaust_k(hot, 0.05, 0.40)
    assert mixed < 330.0
    ratio = ts.ir_signature_ratio(hot, mixed)
    assert ratio < 0.02                       # under two percent of the original
    # and no dilution changes nothing
    assert ts.ir_signature_ratio(hot, hot) == pytest.approx(1.0)


def test_a_plume_at_ambient_is_invisible_however_much_of_it_there_is():
    import thermal_storage as ts
    assert ts.ir_signature_ratio(313.15, 313.15, ambient_k=313.15) == pytest.approx(1.0)
    assert ts.ir_signature_ratio(673.15, 313.15, ambient_k=313.15) == pytest.approx(0.0)
