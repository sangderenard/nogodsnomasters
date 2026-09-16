"""The walk from a body to the mounts, and what the tensor could not see.

Every test here is about the difference between WHERE a mass is and HOW
its moment reaches the feet. The rigid tensor answers the first and
assumes the second; the walk answers the second from the graph.
"""
import math

import numpy as np
import pytest

import autoclave_production as ap
import wrench_paths as wp
from joints import Seam
from turret_production import ProductionGraph


@pytest.fixture(scope="module")
def autoclave():
    return ap.build().as_document()


@pytest.fixture(scope="module")
def paths(autoclave):
    return wp.wrench_paths(autoclave)


def test_every_massive_body_on_the_autoclave_reaches_a_mount(autoclave, paths):
    """The first run of this walk reported the 498 kg vessel held by its
    steam line, because its saddle seats were nodes joined to the saddles
    and to nothing else. Declaring the seats as wrench points on the
    shell is what fixed it, and this is the check that keeps it fixed."""
    assert wp.problems(paths, autoclave) == []
    for body in wp.massive_bodies(autoclave):
        assert paths[body["identity"]].kind != "unsupported", body["identity"]


def test_the_vessel_is_carried_rigidly_and_reaches_all_four_feet(paths):
    p = paths["plant.autoclave.vessel"]
    assert p.kind == "rigid"
    assert p.stiffness_n_per_m == math.inf
    assert len(p.reaches) == 4
    # the lever its moment has is the saddle span, near enough
    assert 0.5 < p.couple_arm_m < 1.0


def test_the_pump_is_isolated_and_says_at_what_frequency(autoclave, paths):
    pump = next(n for n in autoclave["nodes"]
                if n["identity"] == "plant.autoclave.vacuum_pump")
    p = paths[pump["identity"]]
    assert p.kind == "isolated"
    f0 = p.isolation_hz(pump["mass_kg"])
    assert 5.0 < f0 < 60.0
    # below the isolator it transmits everything; well above, a fraction
    assert p.transmission(f0 / 10.0, pump["mass_kg"]) == pytest.approx(1.0, abs=0.02)
    assert p.transmission(f0 * 4.0, pump["mass_kg"]) < 0.2
    # and AT it, more than everything: that is the trade an isolator is
    assert p.transmission(f0, pump["mass_kg"]) > 2.0


def test_a_hose_is_not_a_load_path(autoclave):
    """Take the pump's isolator bracket away and its only edge is the
    vacuum line, which joints marks routed. The walk must then say the
    pump is unsupported rather than quietly running it through the hose."""
    doc = dict(autoclave)
    doc["edges"] = [e for e in autoclave["edges"]
                    if e["identity"] != "plant.autoclave.vacuum_pump.isolator"]
    p = wp.wrench_paths(doc)["plant.autoclave.vacuum_pump"]
    assert p.kind == "unsupported"
    assert p.transmission(10.0, 36.0) == 0.0
    assert any("vacuum_pump" in t for t in wp.problems(wp.wrench_paths(doc), doc))


def test_series_compliance_is_the_softest_link_not_the_hop_count():
    """Forty welded hops cost nothing; one bushing costs its compliance."""
    g = ProductionGraph("t")
    g.node("t.mount", (0.0, 0.0, 0.0), "foot", port_role="structural-mount",
           mass_kg=0.0, mass_in_total=False)
    prev = "t.mount"
    for i in range(20):
        ident = f"t.link.{i}"
        g.node(ident, (0.05 * (i + 1), 0.0, 0.0), "bar", mass_kg=0.0,
               mass_in_total=False)
        g.edge(f"t.e.{i}", prev, ident, "rigid-distance")
        prev = ident
    g.node("t.body", (1.2, 0.0, 0.0), "body", mass_kg=10.0)
    g.edge("t.iso", prev, "t.body", "engine-mount-isolator")
    doc = g.as_document()
    p = wp.wrench_paths(doc)["t.body"]
    assert p.kind == "isolated"
    assert len(p.hops) == 22
    packs = next(e for e in doc["edges"] if e["identity"] == "t.iso")["joint_bushings"]
    k_each = packs["a"]["linear_stiffness_n_per_m"]
    assert p.stiffness_n_per_m == pytest.approx(k_each / 2.0, rel=1e-9)


def test_a_screwed_seam_is_a_compliant_link_with_the_seams_own_stiffness():
    g = ProductionGraph("t")
    g.node("t.mount", (0.0, 0.0, 0.0), "foot", port_role="structural-mount",
           mass_kg=0.0, mass_in_total=False)
    g.node("t.panel", (0.0, 0.5, 0.0), "panel", mass_kg=3.0)
    seam = Seam(fastener="sheet-metal-screw", per_inch=1.0 / 3.0,
                sheet_thickness_m=0.0012)
    g.edge("t.seam", "t.mount", "t.panel", "screwed-seam", seam=seam)
    doc = g.as_document()
    e = doc["edges"][0]
    assert e["seam"]["count"] == seam.count(0.5)
    p = wp.wrench_paths(doc)["t.panel"]
    assert p.kind == "isolated"
    assert p.stiffness_n_per_m == pytest.approx(seam.stiffness_n_per_m(0.5))


def test_a_rigid_drain_carries_its_trap_when_it_says_so():
    """A hose is not a load path. A steel drain declared load-bearing is
    one -- its own cantilever stiffness plus its bracket's -- so the trap
    on its end is supported, and the walk says through what."""
    from milspec import TubeSection

    def graph(load_bearing, help_n_per_m=0.0):
        g = ProductionGraph("t")
        g.node("t.mount", (0.0, 0.0, 0.0), "foot", port_role="structural-mount",
               mass_kg=0.0, mass_in_total=False)
        g.node("t.trap", (0.0, -0.40, 0.0), "drain-trap", mass_kg=2.5)
        kw = {"load_bearing": True} if load_bearing else {}
        if help_n_per_m:
            kw["support_stiffness_n_per_m"] = help_n_per_m
        g.edge("t.drain", "t.mount", "t.trap", "condensate-line", radius=0.019,
               wall_m=0.0025, alloy="a36", **kw)
        return g.as_document()

    hose = wp.wrench_paths(graph(False))["t.trap"]
    assert hose.kind == "unsupported"

    doc = graph(True)
    pipe = wp.wrench_paths(doc)["t.trap"]
    assert pipe.kind == "isolated"
    sec = TubeSection(outer_diameter_m=0.038, wall_m=0.0025, material="a36")
    k_expected = 3.0 * sec.mat.youngs_pa * sec.second_moment_m4 / 0.40 ** 3
    assert pipe.stiffness_n_per_m == pytest.approx(k_expected, rel=1e-6)
    # a 38 mm steel pipe cantilevered 400 mm is stiff: hundreds of kN/m
    assert 1e5 < pipe.stiffness_n_per_m < 1e7

    helped = wp.wrench_paths(graph(True, 2.0e5))["t.trap"]
    assert helped.stiffness_n_per_m == pytest.approx(k_expected + 2.0e5, rel=1e-6)
