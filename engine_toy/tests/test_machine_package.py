"""A machine you can pick up: where its ports are, what it stands on,
and what it weighs.

Each test here is anchored to something that was actually wrong, not to
a restatement of the code.
"""
import math

import numpy as np
import pytest

import autoclave_production as ap
import machine_package as mp
import machine_options as mo
from engine_mounts import MountingTechnique


UP = np.array([0.0, 1.0, 0.0])


def _radial(node, graph):
    vessel = next(n for n in graph.nodes if n.get("kind") == "pressure-vessel")
    axis = np.asarray(vessel.get("tube_axis", (0.0, 0.0, 1.0)), float)
    r = np.asarray(node["reference_position"], float) - np.asarray(
        vessel["reference_position"], float)
    r = r - axis * float(np.dot(r, axis))
    return r / float(np.linalg.norm(r))


# ---------------------------------------------------------------------
# where a port sits on a round shell
# ---------------------------------------------------------------------

def test_angle_zero_is_the_crown_and_the_prism_says_so():
    """The fact the first draft got wrong, stated once.

    On a z-axis body the perpendicular the default frame builds angle
    zero from is +Y -- so ninety degrees, where the vent was authored,
    is the FLANK. Declaring the clock does not move anything here; it
    removes the need to know this."""
    from prism_bodies import Prism
    v = Prism(identity="v", kind="k", centre=(0, 0, 0), shape="cylinder",
              axis=(0, 0, 1), radius=1.0, length=1.0, clock=(0.0, 1.0, 0.0))
    assert v.side_point(0.0).position[1] == pytest.approx(1.0)
    assert v.side_point(math.radians(180.0)).position[1] == pytest.approx(-1.0)
    assert abs(v.side_point(math.radians(90.0)).position[1]) < 1e-9

    # and an undeclared clock lands in the same place on this body, which
    # is why nothing caught it
    u = Prism(identity="u", kind="k", centre=(0, 0, 0), shape="cylinder",
              axis=(0, 0, 1), radius=1.0, length=1.0)
    assert np.allclose(u.side_point(0.0).position, v.side_point(0.0).position)


def test_the_vessel_vents_from_the_crown_and_drains_from_the_invert():
    g = ap.build()
    by_kind = {n.get("port_kind"): n for n in g.nodes
               if n.get("kind") == "vessel-port"}
    assert float(np.dot(_radial(by_kind["vent"], g), UP)) > 0.85
    assert float(np.dot(_radial(by_kind["drain"], g), UP)) < -0.85
    # and everything that must not be coverable by standing liquid is above
    for kind in ("relief", "gauge", "vacuum", "steam_in"):
        assert float(np.dot(_radial(by_kind[kind], g), UP)) > 0.0, kind


def test_the_station_check_catches_a_vent_on_the_flank():
    """Proof the check is a check. Put the vent and the drain back where
    the first draft had them and it must complain about both."""
    g = ap.build()
    r = 0.3
    for kind, deg in (("vent", 90.0), ("drain", 270.0)):
        n = next(x for x in g.nodes if x.get("port_kind") == kind)
        rad = math.radians(deg)
        n["reference_position"] = [-math.sin(rad) * r, math.cos(rad) * r,
                                   n["reference_position"][2]]
    faults = ap.check_port_stations(g)
    assert len(faults) == 2
    assert any("vent" in f and "crown" in f for f in faults)
    assert any("drain" in f and "invert" in f for f in faults)


def test_the_basket_rails_are_both_below_the_basket():
    """Authored at 200 and 340 degrees, one rail sat near the crown and
    the basket rested on neither."""
    g = ap.build()
    basket = next(n for n in g.nodes if n.get("kind") == "load-basket")
    rails = [n for n in g.nodes if n.get("kind") == "chamber-rail"]
    assert len(rails) == 2
    for rail in rails:
        assert rail["reference_position"][1] < basket["reference_position"][1]


# ---------------------------------------------------------------------
# what it stands on
# ---------------------------------------------------------------------

def test_a_machine_that_declares_no_mounts_cannot_be_installed():
    """And says so, rather than inventing four corners of its own box."""
    doc = {"identity": "m", "nodes": [
        {"identity": "m.body", "kind": "body", "reference_position": [0, 0, 0],
         "body_half_extent_m": [0.5, 0.5, 0.5], "mass_kg": 100.0}], "edges": []}
    pkg = mp.MachinePackage.build(doc)
    assert pkg.mounts == []
    assert not pkg.stability().ok


def test_the_autoclave_stands_on_four_declared_feet_and_they_carry_its_weight():
    pkg = mp.MachinePackage.build(ap.build())
    assert len(pkg.mounts) == 4
    assert pkg.stability().ok
    loads = pkg.static_mount_loads()
    carried = sum(abs(f[1]) for f in loads.values())
    assert carried == pytest.approx(pkg.rigid_body.total_mass_kg * 9.80665,
                                    rel=1e-3)


def test_one_saddle_is_fixed_and_one_slides_by_more_than_the_shell_grows():
    """Anchor both and the shell cannot grow. What it develops instead
    does not depend on how long it is, which is why "only a millimetre"
    is never the answer."""
    g = ap.build()
    saddles = [n for n in g.nodes if n.get("kind") == "vessel-saddle"]
    assert len(saddles) == 2
    anchors = sorted(s["anchor"] for s in saddles)
    assert anchors == ["round-holes", "slotted-holes"]

    sliding = next(s for s in saddles if s["anchor"] == "slotted-holes")
    fixed = next(s for s in saddles if s["anchor"] == "round-holes")
    assert fixed["slot_travel_m"] == 0.0

    from gas_works import _steam_properties
    t_sat, _ = _steam_properties(600_000.0)
    span = abs(sliding["reference_position"][2] - fixed["reference_position"][2])
    growth = ap.STEEL_EXPANSION_PER_K * span * (t_sat - ap.AMBIENT_K)
    assert sliding["slot_travel_m"] > growth

    # restraining it is not a large force, it is a yielded shell
    assert ap.restrained_growth_stress_pa(t_sat - ap.AMBIENT_K) > 250e6


def test_the_prism_is_measured_from_metal_not_from_centres():
    """EnginePackage takes the min/max of node positions because a
    drivetrain node is a centroid. A production body declares its
    extent, and ignoring it under-reports the crate by the vessel's own
    radius."""
    pkg = mp.MachinePackage.build(ap.build())
    nodes = [n for n in ap.build().nodes if not n.get("wrench_point")]
    centres_only = np.asarray([n["reference_position"] for n in nodes], float)
    bare = centres_only.max(axis=0) - centres_only.min(axis=0)
    assert all(a >= b for a, b in zip(pkg.prism.size_m, bare))
    assert pkg.prism.size_m[0] > bare[0]


def test_only_declared_crossings_count_as_connectors():
    pkg = mp.MachinePackage.build(ap.build())
    assert {c.circuit for c in pkg.connectors} == {"steam", "vacuum"}
    assert all(c.bore_m > 0.0 for c in pkg.connectors)


# ---------------------------------------------------------------------
# which pump, and why it is a configuration question
# ---------------------------------------------------------------------

def test_an_oil_sealed_pump_cannot_be_fitted_for_a_solvent_duty():
    """Not a warning, not a wear multiplier: the machine does not pass."""
    wrong = mo.configuration(mo.SHORE_STEAM, mo.VACUUM_SET, duty="solvent-dewax")
    assert not wrong.check()["buildable"]
    assert "solvent-safe-vacuum" in wrong.missing()

    right = mo.configuration(mo.SHORE_STEAM, mo.DIAPHRAGM_VACUUM_SET,
                             duty="solvent-dewax")
    assert right.check()["buildable"]


def test_a_site_cannot_supply_a_solvent_safe_vacuum_through_a_port():
    """Steam can be piped in and power wired in. A pump that is safe to
    put solvent through is hardware the machine carries or does not."""
    assert "solvent-safe-vacuum" not in mo.SITE_SUPPLIES
    bare = mo.configuration(duty="solvent-dewax")
    assert "solvent-safe-vacuum" in bare.missing()


def test_the_two_pumps_exclude_each_other_by_wanting_the_same_bay():
    both = mo.configuration(mo.VACUUM_SET, mo.DIAPHRAGM_VACUUM_SET,
                            duty="porous-load")
    assert both.space_conflicts()
    assert not both.check()["buildable"]


def test_the_diaphragm_set_grows_an_exhaust_where_the_other_grew_a_drain():
    """An oil-free pump has no oil to drop and does have somewhere its
    exhaust must go."""
    oiled = mo.configuration(mo.VACUUM_SET).required_casing_ports()
    dry = mo.configuration(mo.DIAPHRAGM_VACUUM_SET).required_casing_ports()
    assert "pump-drain" in oiled and "pump-drain" not in dry
    assert "pump-exhaust" in dry and "pump-exhaust" not in oiled


def test_the_diaphragm_pump_is_adequate_for_the_duty_it_is_chosen_for():
    """The swap costs three decades of ultimate vacuum and that buys
    nothing here: a pre-vacuum purge wants tens of millibar, and a
    three-stage diaphragm pump reaches about one."""
    from compressors import VacuumPump
    oiled = VacuumPump(clearance_frac=0.005, stages=2)
    dry = VacuumPump(clearance_frac=0.20, stages=3)
    assert oiled.ultimate_pa() < dry.ultimate_pa() / 100.0
    purge_target_pa = 5_000.0          # 50 mbar absolute
    assert dry.ultimate_pa() < purge_target_pa / 10.0


def test_a_machine_package_reports_a_technique_it_was_told_not_one_it_guessed():
    pkg = mp.MachinePackage.build(ap.build(),
                                  technique=MountingTechnique.RUBBER_ISOLATOR)
    assert pkg.technique is MountingTechnique.RUBBER_ISOLATOR
    assert all(m.hardware.stiffness_n_per_m is not None
               for m in pkg.assignments())
