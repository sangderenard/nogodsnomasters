"""Square tube, angle, sheet, and the seams that hold sheet together."""
import math

import pytest

from milspec import AngleSection, SquareTubeSection, SheetSection, TubeSection
from joints import BONDS, JOINT_TYPES, Seam, member_constraint
from turret_production import ProductionGraph


def test_square_tube_in_aluminium_is_lighter_and_softer_than_in_steel():
    al = SquareTubeSection(0.040, 0.003, material="6061t6")
    st = SquareTubeSection(0.040, 0.003, material="a36")
    assert al.area_m2 == st.area_m2
    assert al.mass_per_m_kg < st.mass_per_m_kg / 2.5
    # the same section is a third the stiffness in aluminium: E, not shape
    assert al.buckling_load_n(1.0) / st.buckling_load_n(1.0) == pytest.approx(68.9 / 200.0, rel=0.02)


def test_square_tube_second_moment_is_the_difference_of_fourth_powers():
    s = SquareTubeSection(0.040, 0.003)
    assert s.second_moment_m4 == pytest.approx((0.040 ** 4 - 0.034 ** 4) / 12.0)
    assert s.second_moment_m4 > TubeSection(0.040, 0.003).second_moment_m4


def test_angle_reports_its_weak_axis_and_it_is_much_weaker():
    a = AngleSection(0.040, 0.004, material="a36")
    # the classical values for a 40x40x4 equal angle: I_xx ~ 4.5 cm4,
    # least I ~ 1.85 cm4 (tables give 1.86)
    assert a.second_moment_leg_m4 == pytest.approx(4.5e-8, rel=0.05)
    assert a.second_moment_m4 == pytest.approx(1.86e-8, rel=0.08)
    assert a.second_moment_m4 < 0.5 * a.second_moment_leg_m4
    assert a.mass_per_m_kg == pytest.approx(2.42, rel=0.03)


def test_a_member_may_declare_its_section_and_the_graph_solves_that_section():
    g = ProductionGraph("t")
    g.node("a", (0, 0, 0), "corner", mass_kg=0.0, mass_in_total=False)
    g.node("b", (1, 0, 0), "corner", mass_kg=0.0, mass_in_total=False)
    sec = AngleSection(0.040, 0.004, material="a36", designation="40x40x4 A36")
    g.edge("m", "a", "b", "bolted-flange-mount", section=sec)
    e = g.as_document()["edges"][0]
    assert e["damage"]["section_area_m2"] == pytest.approx(sec.area_m2)
    assert e["damage"]["second_moment_m4"] == pytest.approx(sec.second_moment_m4)
    assert e["damage"]["material"] == "a36"
    assert e["damage"]["section_designation"] == "40x40x4 A36"


def test_a_seam_is_specified_per_inch_and_costed_per_fastener():
    s = Seam(fastener="sheet-metal-screw", per_inch=1.0 / 3.0,
             sheet_thickness_m=0.0012, fastener_diameter_m=0.0048)
    # a metre of seam at a screw every three inches is thirteen screws
    assert s.count(1.0) == 13
    # never fewer than two: one screw is a hinge
    assert s.count(0.01) == 2
    # a #10 in 1.2 mm mild steel tears out at about a kilonewton
    assert 800.0 < s.capacity_per_fastener_n() < 1600.0
    assert s.capacity_n(1.0) == pytest.approx(13 * s.capacity_per_fastener_n())
    # and bears at a fraction of E.t: tens of MN/m per screw
    assert 3e7 < s.stiffness_per_fastener_n_per_m() < 1e8


def test_the_three_sheet_joints_differ_in_the_way_they_should():
    screw = Seam("sheet-metal-screw", 1.0 / 3.0, 0.0012, fastener_diameter_m=0.0048)
    spot = Seam("spot-weld", 1.0 / 2.0, 0.0012, fastener_diameter_m=0.006)
    snap = Seam("snap-fit", 1.0 / 4.0, 0.002, "a36", fastener_diameter_m=0.010,
                release_force_n=60.0)
    tack = Seam("tack-weld", 1.0 / 6.0, 0.0015, fastener_diameter_m=0.010)
    # a spot weld is stiffer than a screw in the same sheet
    assert spot.stiffness_per_fastener_n_per_m() > screw.stiffness_per_fastener_n_per_m()
    # a snap releases at its declared force and says so
    assert snap.capacity_per_fastener_n() == 60.0
    assert snap.pack(1.0)["releases"] is True
    assert screw.pack(1.0)["releases"] is False
    # a tack is a short fillet no bigger than the sheet: a 10 mm tack in
    # 1.5 mm sheet holds about three kilonewtons
    assert 2000.0 < tack.capacity_per_fastener_n() < 4000.0
    assert tack.capacity_per_fastener_n() > screw.capacity_per_fastener_n()


def test_the_joint_types_and_bonds_exist_and_say_what_they_transmit():
    for key in ("screwed-seam", "snap-fastened", "unibody-spot-welded", "tack-welded"):
        assert key in JOINT_TYPES
    assert JOINT_TYPES["unibody-spot-welded"].freedoms() == ()
    assert JOINT_TYPES["tack-welded"].freedoms() == ()
    assert JOINT_TYPES["tack-welded"].bond == "tack-weld"
    assert "sheet-metal-screw" in BONDS and "snap-fit" in BONDS and "tack-weld" in BONDS
    for key in ("screwed-seam", "spot-welded-seam", "snap-seam",
                "tack-welded-bracket", "shell-attachment-weld"):
        mc = member_constraint(key)
        assert mc.welded and not mc.routed


def test_a_tacked_bracket_on_casing_carries_its_tacks_on_the_edge():
    g = ProductionGraph("t")
    g.node("casing", (0, 0, 0), "panel", mass_kg=2.0)
    g.node("bracket", (0.12, 0, 0), "bracket", mass_kg=0.3)
    tacks = Seam("tack-weld", per_inch=1.0 / 2.0, sheet_thickness_m=0.0015,
                 fastener_diameter_m=0.008)
    g.edge("t", "casing", "bracket", "tack-welded-bracket", seam=tacks,
           section=SheetSection(0.05, 0.0015))
    e = g.as_document()["edges"][0]
    assert e["seam"]["fastener"] == "tack-weld"
    assert e["seam"]["count"] == tacks.count(0.12)
    assert e["seam"]["capacity_n"] == pytest.approx(tacks.capacity_n(0.12))
