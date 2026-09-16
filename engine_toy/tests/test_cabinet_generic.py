"""The generic box, and the machines that are one."""
import pytest

import appliance_production as app
import cabinet as cb
import feet
import mode_table as mt
import wrench_paths as wp
from joints import JOINT_TYPES, member_constraint


def test_zero_racks_and_no_window_is_still_a_cabinet():
    b = cb.build(cb.CabinetSpec(identity="t.empty", inside_m=(0.5, 0.5, 0.5),
                                construction="screwed-sheet",
                                door=cb.Door(kind="solid", window=cb.Window(kind="none"))))
    doc = b.graph.as_document()
    assert not [n for n in doc["nodes"] if n.get("kind") == "load-rack"]
    assert not [n for n in doc["nodes"] if n.get("kind") == "pressure-window"]
    assert len(wp.mount_nodes(doc)) == 4
    assert wp.problems(wp.wrench_paths(doc), doc) == []


def test_racks_are_a_vector_each_with_its_own_size_and_load():
    # each rack's headroom is its own, and it has to fit under the next:
    # declared at 30 cm with shelves 25 cm apart, the graph reported the
    # upper shelf standing in the lower one's headroom -- which it was
    racks = (cb.Rack(0.20, half_extent_m=(0.20, 0.01, 0.20), load_kg=5.0, headroom_m=0.22),
             cb.Rack(0.45, half_extent_m=(0.30, 0.02, 0.25), load_kg=40.0, headroom_m=0.22),
             cb.Rack(0.70))
    b = cb.build(cb.CabinetSpec(identity="t.racks", inside_m=(0.8, 0.9, 0.7),
                                working_pressure_pa=2e5, racks=racks))
    doc = b.graph.as_document()
    shelves = sorted((n for n in doc["nodes"] if n.get("kind") == "load-rack"),
                     key=lambda n: n["level"])
    assert [tuple(n["body_half_extent_m"]) for n in shelves[:2]] == [(0.20, 0.01, 0.20), (0.30, 0.02, 0.25)]
    assert shelves[2]["body_half_extent_m"][0] == pytest.approx(0.35)
    charges = cb.charges(b.spec)
    assert charges["t.racks.rack.1"] == 40.0 and len(charges) == 3
    assert [l for l in b.graph.check() if "clear volume" in l] == []


def test_a_piano_hinge_frees_one_rotation_and_the_door_still_reaches_the_feet():
    assert JOINT_TYPES["piano-hinge"].freedoms() == ("ry",)
    assert member_constraint("piano-hinge").freedoms() == ("ry",)
    b = cb.build(cb.CabinetSpec(identity="t.door", inside_m=(0.6, 0.6, 0.6),
                                construction="screwed-sheet"))
    doc = b.graph.as_document()
    hinges = [e for e in doc["edges"] if e["constraint"] == "piano-hinge"]
    assert len(hinges) == 2
    assert wp.wrench_paths(doc)["t.door.door"].kind != "unsupported"


def test_top_and_bottom_compartments_carry_their_loadouts_and_claim_their_space():
    spec = cb.CabinetSpec(
        identity="t.plant", inside_m=(0.6, 0.6, 0.6), construction="screwed-sheet",
        bottom=cb.Compartment(0.15, plant=(cb.PlantItem("motor", "motor", (0.06, 0.06, 0.08), 6.0,
                                                        mount="isolated"),)),
        top=cb.Compartment(0.10, plant=(cb.PlantItem("box", "control-box", (0.1, 0.03, 0.1), 1.0),)))
    b = cb.build(spec)
    doc = b.graph.as_document()
    ids = {n["identity"] for n in doc["nodes"]}
    assert "t.plant.bottom.motor" in ids and "t.plant.top.box" in ids
    paths = wp.wrench_paths(doc)
    assert paths["t.plant.bottom.motor"].kind == "isolated"
    assert [l for l in b.graph.check() if "clear volume" in l] == []
    vols = {v["identity"] for v in b.graph.clear_volumes}
    assert {"t.plant.top_bay", "t.plant.bottom_bay"} <= vols
    # the feet are under the bottom box, not under the chamber floor
    feet_y = min(n["reference_position"][1] for n in doc["nodes"]
                 if n.get("port_role") == "structural-mount")
    assert feet_y < -0.15


def test_a_porthole_is_glass_sized_by_its_own_span():
    spec = cb.CabinetSpec(identity="t.port", inside_m=(0.6, 0.6, 0.6), working_pressure_pa=1e5,
                          door=cb.Door(kind="solid", window=cb.Window(kind="porthole",
                                                                       porthole_radius_m=0.12)))
    doc = cb.build(spec).graph.as_document()
    pane = next(n for n in doc["nodes"] if n.get("kind") == "pressure-window")
    assert pane["span_m"] == pytest.approx(0.24)
    assert pane["glass_thickness_m"] == pytest.approx(cb.glass_thickness_m(1e5, 0.24))


# ---------------------------------------------------------------------
# the appliances
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def washer():
    return app.build_washer()


def test_the_washer_tub_is_suspended_and_its_motor_is_specced_with_it(washer):
    doc = washer.graph.as_document()
    paths = wp.wrench_paths(doc)
    drum = paths["plant.washer.drum"]
    assert drum.kind == "isolated"
    f0 = drum.isolation_hz(9.0)
    assert 3.0 < f0 < 12.0
    # at full spin the cabinet feels a fraction of the towel
    assert drum.transmission(app.SPIN_RPM / 60.0, 9.0) < 0.3
    motor = paths["plant.washer.drum.motor"]
    assert motor.kind in ("rigid", "isolated")
    belt = next(e for e in doc["edges"] if e["identity"] == "plant.washer.drum.belt")
    assert belt["suspension"]["linear_stiffness_n_per_m"] == 20_000.0


def test_a_direct_drive_motor_rides_the_suspension():
    b = app.build_washer(drive="direct")
    doc = b.graph.as_document()
    paths = wp.wrench_paths(doc)
    assert paths["plant.washer.drum.motor"].kind == "isolated"
    assert paths["plant.washer.drum.motor"].isolation_hz(8.0) < 20.0


def test_the_towel_shakes_the_washer_most_passing_through_its_suspension(washer):
    doc = washer.graph.as_document()
    ft = feet.from_mounts(doc, pad_diameter_m=0.05)
    t = mt.evaluate(doc, ft, feet.ground("concrete-slab"), app.washer_cycle(),
                    frame_underside_y=feet.underside_y(doc) + 0.02)
    spin = [r for r in t.rows if r.state == "spin"]
    trans = [next(e for e in r.effects if e.source.endswith(".drum")).transmission for r in spin]
    peak = trans.index(max(trans))
    assert max(trans) > 1.5
    assert 0 < peak < len(trans) - 1
    assert trans[-1] < 0.3
    # and it is the moment a foot leaves the floor, if it ever does
    lifts = [r for r in spin if any("leaves" in h for h in r.hazards)]
    if lifts:
        assert all(abs(spin.index(r) - peak) <= 2 for r in lifts)


def test_without_its_counterweight_the_same_washer_shakes_harder():
    heavy = app.build_washer().graph.as_document()
    light = app.build_washer(counterweight_kg=0.0).graph.as_document()

    def worst(doc):
        ft = feet.from_mounts(doc, pad_diameter_m=0.05)
        t = mt.evaluate(doc, ft, feet.ground("concrete-slab"), app.washer_cycle(),
                        frame_underside_y=feet.underside_y(doc) + 0.02)
        return max(e.response.heave_m for r in t.rows for e in r.effects if e.applied)

    assert worst(light) > worst(heavy)


def test_the_dryer_drum_is_on_rollers_and_only_the_blower_spins_fast():
    b = app.build_dryer()
    doc = b.graph.as_document()
    paths = wp.wrench_paths(doc)
    assert paths["plant.dryer.drum"].kind in ("rigid", "isolated")
    assert paths["plant.dryer.drum"].isolation_hz(6.0) is None or \
        paths["plant.dryer.drum"].isolation_hz(6.0) > 100.0
    from operating_states import sources_of
    fast = [s for s in sources_of(doc) if s.rated_rpm > 1000.0]
    assert {s.identity.split(".")[-1] for s in fast} == {"blower", "motor"}
    assert [l for l in b.graph.check() if "clear volume" in l] == []
