"""The square, glass-faced cabinet: what its shape costs and what it can carry."""
import math

import pytest

import cabinet_autoclave_production as cab
import feet
import mode_table as mt
import wrench_paths as wp


@pytest.fixture(scope="module")
def graph():
    return cab.build()


@pytest.fixture(scope="module")
def doc(graph):
    return graph.as_document()


def test_a_flat_wall_is_sized_by_bending_and_goes_as_root_pressure():
    t3 = cab.wall_thickness_m(300_000.0)
    t6 = cab.wall_thickness_m(600_000.0)
    # the plate beyond the corrosion allowance scales as sqrt(p)
    assert (t6 - cab.CORROSION_ALLOWANCE_M) / (t3 - cab.CORROSION_ALLOWANCE_M) == pytest.approx(math.sqrt(2.0), rel=1e-6)
    # and it is a different order of thing from a tube at the same pressure
    import autoclave_production as tube
    assert t3 > 4.0 * tube.wall_thickness_m(300_000.0)


def test_the_glass_is_mullioned_because_a_single_pane_would_be_absurd():
    whole = cab.glass_thickness_m(300_000.0, pitch_m=cab.INSIDE_W_M)
    strip = cab.glass_thickness_m(300_000.0)
    assert whole > 0.09
    assert 0.025 < strip < 0.035
    # halving the pitch halves the pane
    assert cab.glass_thickness_m(300_000.0, pitch_m=0.20) == pytest.approx(strip / 2.0)


def test_the_cabinet_is_heavy_and_the_walls_are_most_of_it(doc):
    walls = sum(n["mass_kg"] for n in doc["nodes"] if n.get("kind") == "pressure-wall")
    total = sum(n.get("mass_kg", 0.0) for n in doc["nodes"] if not n.get("wrench_point"))
    assert total > 1500.0
    assert walls > 0.6 * total
    panes = [n for n in doc["nodes"] if n.get("kind") == "pressure-window"]
    assert len(panes) == 3
    assert all(n["glass_thickness_m"] == pytest.approx(cab.glass_thickness_m(300_000.0)) for n in panes)


def test_the_case_is_the_frame_and_everything_reaches_the_six_feet(doc):
    paths = wp.wrench_paths(doc)
    assert wp.problems(paths, doc) == []
    mounts = wp.mount_nodes(doc)
    assert len(mounts) == 6
    for wall in ("floor", "ceiling", "back", "left", "right"):
        p = paths[f"plant.cabinet.wall.{wall}"]
        assert p.kind == "rigid" and len(p.reaches) == 6
    assert paths["plant.cabinet.vacuum_pump"].kind == "isolated"
    # the heater's terminal box hangs on a conduit that said it carries it
    heater = paths["plant.cabinet.heater"]
    assert heater.kind == "rigid"          # bolted to the floor as well


def test_nothing_crosses_a_claimed_volume_but_its_own_assembly(graph):
    findings = [l for l in graph.check() if "clear volume" in l]
    assert findings == []


def test_racks_and_the_cradle_exclude_each_other_by_volume_and_only_low_down():
    cfg = cab.configuration(cab.RACK_SET, cab.ENGINE_CRADLE, duty="engine-bake")
    conflicts = cfg.space_conflicts()
    assert conflicts
    assert all("rack_0" in c or "rack_1" in c for c in conflicts)
    assert not any("rack_2" in c or "rack_3" in c for c in conflicts)


def test_glove_ports_are_an_option_that_grows_two_ports_in_the_middle_pane():
    with_gloves = cab.build(options=(cab.RACK_SET, cab.GLOVE_PORTS)).as_document()
    without = cab.build(options=(cab.RACK_SET,)).as_document()
    gloves = [n for n in with_gloves["nodes"] if n.get("closure") == "glove-sleeve"]
    assert len(gloves) == 2
    assert not [n for n in without["nodes"] if n.get("closure") == "glove-sleeve"]
    alone = cab.configuration(cab.GLOVE_PORTS, duty="glove-work")
    assert {"glove-port-left", "glove-port-right"} <= set(alone.required_casing_ports())
    assert alone.check(cab.CATALOGUE)["buildable"]
    # and a full rack set in front of the arms is refused by geometry:
    # the reach volume and two rack levels want the same space
    crowded = cab.configuration(cab.GLOVE_PORTS, cab.RACK_SET, duty="glove-work")
    assert crowded.space_conflicts()
    assert not crowded.check(cab.CATALOGUE)["buildable"]


def test_a_composite_cure_needs_heat_vacuum_and_pressure_and_the_site_cannot_lend_the_pump():
    bare = cab.configuration(cab.INTEGRAL_HEATER, duty="composite-cure")
    assert "vacuum" in bare.missing() and "pressure-hold" in bare.missing()
    full = cab.configuration(cab.INTEGRAL_HEATER, cab.VACUUM_SET, cab.AIR_SET,
                             duty="composite-cure")
    assert full.check(cab.CATALOGUE)["buildable"]
    assert "electricity" in full.from_shore()


def test_the_mode_table_runs_on_the_cabinet_and_the_racks_load_it(doc):
    ft = feet.from_mounts(doc, pad_diameter_m=0.15)
    t = mt.evaluate(doc, ft, feet.ground("concrete-slab"), cab.cycle(),
                    frame_underside_y=feet.underside_y(doc) + 0.06)
    assert t.problems == ()
    empty = t.at("load", 1.0)
    loaded = t.at("hold", 1.0)
    assert loaded.rigid_body.total_mass_kg == pytest.approx(
        empty.rigid_body.total_mass_kg + 4 * cab.RACK_LOAD_KG)
    assert all(f > 0.0 for f in loaded.frequencies_hz)
    pump_rows = [r for r in t.rows if r.state == "evacuate" and r.effects]
    assert pump_rows and all(r.effects[0].path_kind == "isolated" for r in pump_rows)
