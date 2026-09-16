"""An optional assembly is only real if its absence changes the machine.

Three consequences are claimed in machine_options: parts, space, and
requirements. Each is checked here against something that already
enforces it, rather than against a restatement of the claim.
"""
import machine_options as mo
import autoclave_production as ap


def test_removing_the_boiler_forces_a_port_the_other_version_never_needs():
    """The requirement consequence, which is the one that matters.

    Without an integral boiler the machine still needs steam, so the
    casing grows a flanged inlet. With one, it does not -- but it grows
    a flue, a fuel fill and a feedwater fill instead. Neither is the
    lighter version of the other."""
    integral = mo.configuration(mo.INTEGRAL_BOILER)
    shore = mo.configuration(mo.SHORE_STEAM)

    assert "steam-inlet-flange" in shore.required_casing_ports()
    assert "steam-inlet-flange" not in integral.required_casing_ports()

    # and the trade runs the other way too
    assert "flue" in integral.required_casing_ports()
    assert "flue" not in shore.required_casing_ports()

    # the integral boiler did not remove requirements, it moved them
    assert {"fuel", "feedwater"} <= set(integral.from_shore())


def test_a_bare_machine_is_incomplete_when_the_site_cannot_supply_it():
    """No option fitted and no shore steam is not a lighter autoclave,
    it is one that cannot run."""
    stranded = mo.configuration(shore=())
    assert not stranded.check()["buildable"]
    assert "steam" in stranded.missing()


def test_mutual_exclusion_is_found_by_geometry_and_not_declared():
    """Nobody marked the boiler and the long chamber incompatible."""
    for a in (mo.INTEGRAL_BOILER, mo.LARGE_CHAMBER):
        assert not hasattr(a, "excludes")

    both = mo.configuration(mo.INTEGRAL_BOILER, mo.LARGE_CHAMBER)
    assert both.space_conflicts()
    assert not both.check()["buildable"]

    # ... and each alone is fine
    assert mo.configuration(mo.INTEGRAL_BOILER, mo.VACUUM_SET).check()["buildable"]


def test_an_unfitted_option_frees_its_volume_for_something_else():
    with_boiler = mo.configuration(mo.INTEGRAL_BOILER)
    without = mo.configuration(mo.SHORE_STEAM)
    assert without.freed_space_m3(mo.CATALOGUE) > with_boiler.freed_space_m3(mo.CATALOGUE)
    assert without.mass_kg() < with_boiler.mass_kg()


def test_a_claim_asserted_on_a_real_graph_is_enforced_by_the_graph():
    """The claim is not a description: apply_to hands it to the same
    clear_volume check that already catches members crossing a clearance.
    Proved by moving a claim onto the vessel and watching the graph
    complain about its own structure."""
    g = ap.build()
    assert not [p for p in g.check() if "clear volume" in p]

    # a claim placed where the machine's own members run must be caught
    greedy = mo.OptionalAssembly(
        identity="test.greedy", label="greedy option",
        claims=(mo.Claim(identity="test.greedy_bay",
                         min=(-2.0, -2.0, -2.0), max=(2.0, 2.0, 2.0),
                         note="deliberately swallows the whole machine"),))
    mo.configuration(greedy).apply_to(g)

    crossings = [p for p in g.check() if "test.greedy_bay" in p]
    assert crossings, "a claim over the whole machine caught nothing"
