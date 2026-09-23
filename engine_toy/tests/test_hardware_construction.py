"""Executable authoring tests; no fake Machine, circuit or dt runtime."""
import json
import math
from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hardware_construction import (Fact, Layer, Lay, authored, known, unknown,
    document, compatibility, route_length_m, unknown_facts)
from hardware_catalogue import (SOURCES, cable_catalogue, connector_catalogue,
    nm_b, belden_1583a, domestic_5_20, class_l_32_12, industrial_460)
from electrical_hardware import BreakerPosition, panel_layout, patch_channels


def test_unknown_is_not_zero():
    f = unknown("ohm")
    assert f.value is None and f.basis == "unknown"
    assert document(f)["value"] is None


@pytest.mark.parametrize("value,basis", [(0,"unknown"),(None,"authored"),(None,"published"),(0,"nonsense")])
def test_inconsistent_evidence_is_rejected(value,basis):
    with pytest.raises(ValueError):
        Fact(value,basis=basis,note="test",sources=("source",))


def test_published_requires_source():
    with pytest.raises(ValueError):
        Fact(20,"A","published")


@pytest.mark.parametrize("v", [float("nan"),float("inf"),-float("inf")])
def test_nonfinite_facts_are_rejected(v):
    with pytest.raises(ValueError):
        authored({"nested":[v]})


def test_document_is_an_independent_copy():
    spec = nm_b()
    data = document(spec)
    data["cores"][0]["material"]["value"] = "wrong"
    assert spec.cores[0].material.value == "copper"


def test_single_helix_length_and_turns():
    lay = Lay("pair","pair",("p+","p-"),"helix",
              authored(.1,"m/turn"),authored(.01,"m"),"right")
    assert lay.turns_for(2) == 20
    assert lay.ideal_path_length_m(2) == pytest.approx(math.hypot(2,20*2*math.pi*.01))


def test_handedness_does_not_change_isolated_helix_length():
    right = Lay("l","overall",("core",),"helix",authored(.12,"m/turn"),authored(.02,"m"),"right")
    assert right.ideal_path_length_m(3) == replace(right,handedness="left").ideal_path_length_m(3)


def test_unknown_lay_is_not_a_zero_twist():
    with pytest.raises(ValueError):
        Lay("l","pair",("a","b"),"helix").turns_for(1)


def test_straight_path_has_no_extra_length():
    lay = Lay("solid","strand",("a",),"straight",handedness="none")
    assert lay.ideal_path_length_m(1.25) == 1.25


@pytest.mark.parametrize("pitch",[0,-1,float("nan")])
def test_invalid_lay_pitch_rejected(pitch):
    with pytest.raises(ValueError):
        Lay("l","overall",("core",),"helix",authored(pitch,"m/turn"))


def test_wrong_pitch_units_rejected():
    with pytest.raises(ValueError):
        Lay("l","strand",("c",),"helix",authored(30,"rpm"))


def test_layer_thickness_is_positive():
    with pytest.raises(ValueError):
        Layer("bad",authored("pvc"),"insulation",authored(0,"m"))


def test_full_polyline_not_chord():
    assert route_length_m(((0,0,0),(1,0,0),(1,2,0))) == 3


@pytest.mark.parametrize("points", [[],[(0,0,0)],[(0,0),(1,1)],[(0,0,0),(0,0,0)],[(0,0,0),(float("nan"),0,0)]])
def test_bad_routes_rejected(points):
    with pytest.raises(ValueError):
        route_length_m(points)


@pytest.mark.parametrize("key", list(cable_catalogue()))
def test_each_cable_is_json_safe_and_has_distinct_core_identities(key):
    spec = cable_catalogue()[key]
    data = spec.graph_attributes()
    json.dumps(data,allow_nan=False)
    assert len(spec.cores) == len({c.identity for c in spec.cores})
    assert all(c.material.basis in {"published","authored"} for c in spec.cores)


@pytest.mark.parametrize("key", list(connector_catalogue()))
def test_each_connector_is_json_safe_with_distinct_contacts(key):
    spec = connector_catalogue()[key]
    json.dumps(spec.graph_attributes(),allow_nan=False)
    assert len(spec.contacts) == len({c.identity for c in spec.contacts})


def test_nmb_core_and_outer_stacks_remain_distinct():
    spec = nm_b()
    assert [x.material.value for x in spec.cores[0].insulation] == ["pvc","nylon"]
    assert [x.material.value for x in spec.layers] == ["kraft-paper","pvc"]
    assert not spec.cores[-1].insulation
    assert spec.cores[-1].role == "protective-earth"


def test_12_4_dimensions_are_not_copied_to_12_2():
    assert nm_b(12,4).outer_dimensions_m.value == (.010033,.010033)
    assert nm_b(12,2).outer_dimensions_m.value is None


def test_data_cable_is_not_a_flexible_patch_cord_or_pe_bus():
    spec = belden_1583a()
    assert len(spec.cores) == 8
    assert len(spec.lays) == 4
    assert all(lay.pitch_m.value is None for lay in spec.lays)
    assert spec.details["flexible_patch_cord"] is False
    assert spec.details["shielding"].value == "none-U/UTP"
    assert all(c.role != "protective-earth" for c in spec.cores)
    assert spec.cores[0].insulation[0].material.value == "polyolefin"


def test_data_sheet_max_dcr_is_marked_as_a_bound():
    f = belden_1583a().ratings["max_conductor_dcr_ohm_per_m"]
    assert f.value == .0938
    assert "Maximum" in f.note


def test_cable_has_distinct_strand_and_overall_lay():
    spec = cable_catalogue()["military.flex.6-5"]
    assert spec.lays[0].level == "overall"
    assert spec.cores[0].strand_lays[0].level == "strand"
    assert spec.lays[0].pitch_m.basis == "authored"
    assert spec.cores[0].strand_count.value is None


@pytest.mark.parametrize("factory",[domestic_5_20,class_l_32_12,industrial_460])
def test_declared_counterparts_match(factory):
    assert compatibility(factory("plug"),factory("receptacle")) == (True,())


def test_same_form_does_not_mate():
    assert not compatibility(domestic_5_20(),domestic_5_20())[0]


def test_same_family_different_key_does_not_mate():
    a = class_l_32_12("plug")
    b = replace(class_l_32_12(),interface_key="class-l:different-voltage-or-insert")
    assert not compatibility(a,b)[0]


def test_contact_record_order_does_not_change_pin_map():
    a,b = class_l_32_12("plug"),class_l_32_12()
    b = replace(b,contacts=tuple(reversed(b.contacts)))
    assert compatibility(a,b)[0]


def test_role_swaps_are_not_silently_accepted():
    a,b = domestic_5_20("plug"),domestic_5_20()
    x,w,g = b.contacts
    b = replace(b,contacts=(replace(x,role=w.role),replace(w,role=x.role),g))
    assert not compatibility(a,b)[0]


def test_class_l_has_verified_contact_sizes_and_early_contacts():
    spec = class_l_32_12()
    contacts = {c.identity:c for c in spec.contacts}
    assert contacts["G"].details["contact_size"].value == "6N"
    assert contacts["N"].details["contact_size"].value == "4N"
    assert contacts["A"].details["engagement_group"].value == "power"
    assert contacts["N"].details["engagement_group"].value == "early"
    assert contacts["G"].details["engagement_group"].value == "early"
    assert spec.details["master_keyway_position"].value is None


def test_class_l_sealing_does_not_require_cap_unconditionally():
    assert all("unmated-uncapped" in s.active_configurations for s in class_l_32_12().seals)


def test_industrial_three_phase_does_not_invent_neutral():
    spec = industrial_460()
    assert len(spec.contacts) == 4
    assert "neutral" not in {c.role for c in spec.contacts}
    assert spec.seals[0].material.value == "nbr"
    assert spec.ratings["qualification_assembly_conditions"].value is None


def test_unknown_inventory_is_reported():
    paths = unknown_facts(belden_1583a())
    assert any("pitch_m" in path for path in paths)


def test_source_ids_resolve():
    def visit(obj):
        if isinstance(obj,dict):
            for s in obj.get("sources",[]):
                assert s in SOURCES
            for value in obj.values(): visit(value)
        elif isinstance(obj,list):
            for value in obj: visit(value)
    for obj in (*cable_catalogue().values(),*connector_catalogue().values()):
        visit(document(obj))


def test_row_phase_pattern_not_left_right_split():
    layout = panel_layout((BreakerPosition("a","A",(1,),15),BreakerPosition("b","B",(2,),20),
                          BreakerPosition("c","C",(3,),20)))
    assert layout[0]["phases"] == layout[1]["phases"] == ("line-1",)
    assert layout[2]["phases"] == ("line-2",)


def test_two_pole_occupies_alternating_phase_rows():
    assert panel_layout((BreakerPosition("a","A",(1,3),30),))[0]["phases"] == ("line-1","line-2")


def test_three_pole_board_layout():
    row = panel_layout((BreakerPosition("a","A",(2,4,6),60),),phases=("line-1","line-2","line-3"))[0]
    assert row["phases"] == ("line-1","line-2","line-3")


@pytest.mark.parametrize("slots",[(1,2),(1,5),(11,13),(1,3,5)])
def test_impossible_two_phase_breaker_placement_rejected(slots):
    with pytest.raises(ValueError):
        panel_layout((BreakerPosition("a","A",slots,30),))


def test_overlapping_breakers_rejected():
    with pytest.raises(ValueError):
        panel_layout((BreakerPosition("a","A",(1,3),30),BreakerPosition("b","B",(3,),20)))


@pytest.mark.parametrize("slots",[(0,),(-1,),(1.0,),(True,),(1,1)])
def test_invalid_slot_types_rejected(slots):
    with pytest.raises(ValueError):
        BreakerPosition("x","X",slots,20)


def test_patch_panel_channels_are_independent_pin_preserving_paths():
    channels = patch_channels()
    assert len(channels) == 24
    assert sum(len(c["map"]) for c in channels) == 192
    assert all(c["map"] == {f"front.{i}":f"rear.{i}" for i in range(1,9)} for c in channels)
    assert len({c["label"] for c in channels}) == 24


def test_wall_receptacle_dimensions_not_assigned_to_plug():
    assert "flange_side_m" in class_l_32_12().details["geometry_reference"]
    assert "flange_side_m" not in class_l_32_12("plug").details["geometry_reference"]


def test_receptacle_qualification_not_inherited_by_plug():
    assert industrial_460().ratings["published_ingress_markings"].basis == "published"
    assert industrial_460("plug").ratings["published_ingress_markings"].basis == "unknown"


def test_authored_counterparts_do_not_claim_receptacle_evidence():
    domestic = domestic_5_20("plug")
    assert domestic.ratings["voltage_v"].basis == "authored"
    assert domestic.ratings["current_a"].basis == "authored"
    assert domestic.details["mounting_yoke_material"].value == "not-applicable"
    assert domestic.details["line_neutral_breakoff_tabs"].value is False

    industrial = industrial_460("plug")
    assert industrial.ratings["voltage_v"].basis == "authored"
    assert industrial.ratings["current_a"].basis == "authored"
    assert industrial.details["hardware_material"].basis == "unknown"
    assert industrial.details["contact_carrier_flammability"].basis == "unknown"


def test_power_specimens_select_services_separately_from_cable_ratings():
    home = cable_catalogue()["home.nmb.12-2"]
    service = home.details["electrical_service"]
    assert service.basis == "authored"
    assert service.value["line_voltage_v"] == 120.0
    assert home.ratings["voltage_v"].value == 600

    military = cable_catalogue()["military.flex.6-5"]
    assert military.details["electrical_service"].value["arrangement"] == "three-phase-wye"
    assert class_l_32_12().details["electrical_service"].value \
        == military.details["electrical_service"].value
