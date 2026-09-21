"""What a material does to a field, on the keys parts already declare.

The table is checked against handbook figures that were arrived at
independently -- copper's skin depth is 66 microns at a megahertz in
every reference there is -- and against the one comparison that explains
why the table needs three numbers rather than one: mu-metal shields a
mains-frequency magnetic field far better than copper while conducting
thirty-seven times worse.
"""

from __future__ import annotations

import math

import pytest

from em_materials import (
    COPPER_CONDUCTIVITY_S_M,
    EM_MATERIALS,
    EMMaterial,
    VACUUM_PERMEABILITY_H_M,
    material_for,
)


# --------------------------------------------------------------------------
# against the handbook
# --------------------------------------------------------------------------

def test_copper_skin_depth_matches_the_published_figure() -> None:
    """66 microns at 1 MHz, 9.3 mm at 50 Hz: the reference case."""
    copper = EM_MATERIALS["copper"]

    assert copper.skin_depth_m(1.0e6) == pytest.approx(66.0e-6, rel=0.02)
    assert copper.skin_depth_m(50.0) == pytest.approx(9.3e-3, rel=0.02)
    assert copper.skin_depth_m(2.45e9) == pytest.approx(1.3e-6, rel=0.05)


def test_skin_depth_follows_the_inverse_root_of_frequency() -> None:
    """Hundredfold the frequency, tenth the depth."""
    copper = EM_MATERIALS["copper"]

    assert copper.skin_depth_m(1.0e6) == pytest.approx(
        copper.skin_depth_m(1.0e8) * 10.0, rel=1e-9)


def test_the_formula_is_what_it_says_it_is() -> None:
    material = EM_MATERIALS["stainless-steel"]
    frequency = 1.0e6
    permeability = material.relative_permeability * VACUUM_PERMEABILITY_H_M

    assert material.skin_depth_m(frequency) == pytest.approx(
        1.0 / math.sqrt(math.pi * frequency * permeability
                        * material.conductivity_s_m))


def test_an_insulator_has_no_skin_depth_rather_than_an_error() -> None:
    """The field goes straight through, which is the correct statement."""
    assert EM_MATERIALS["ptfe"].skin_depth_m(1.0e6) == math.inf
    assert EM_MATERIALS["ptfe"].surface_resistance_ohm(1.0e6) == 0.0
    assert not EM_MATERIALS["ptfe"].is_conductor


def test_zero_frequency_has_no_skin_effect() -> None:
    assert EM_MATERIALS["copper"].skin_depth_m(0.0) == math.inf


# --------------------------------------------------------------------------
# why three numbers and not one
# --------------------------------------------------------------------------

def test_mu_metal_beats_copper_magnetically_while_conducting_far_worse() -> None:
    """The comparison a single "conductivity" figure could never express.

    Skin depth falls with the PRODUCT of conductivity and permeability,
    so fifty thousand times the permeability wins easily against
    thirty-seven times the conductivity.  This is why a low-frequency
    magnetic shield is nickel-iron and a radio-frequency one is copper.
    """
    copper = EM_MATERIALS["copper"]
    mumetal = EM_MATERIALS["mu-metal"]

    assert copper.conductivity_s_m / mumetal.conductivity_s_m > 30.0
    assert mumetal.skin_depth_m(50.0) < copper.skin_depth_m(50.0) / 30.0


def test_mild_steel_shields_better_than_stainless_despite_both_being_steel() -> None:
    """Permeability, again: one is magnetic and the other is not."""
    mild = EM_MATERIALS["steel-plate"]
    stainless = EM_MATERIALS["stainless-steel"]

    assert mild.relative_permeability > 100.0
    assert stainless.relative_permeability < 1.05
    assert mild.skin_depth_m(1.0e3) < stainless.skin_depth_m(1.0e3)


def test_conductivity_is_reported_relative_to_copper_as_the_trade_does() -> None:
    assert EM_MATERIALS["copper"].conductivity_relative_to_copper == pytest.approx(1.0)
    assert EM_MATERIALS["aluminium"].conductivity_relative_to_copper == pytest.approx(
        0.587, abs=0.02)


# --------------------------------------------------------------------------
# surface resistance and dielectric loss
# --------------------------------------------------------------------------

def test_surface_resistance_is_milliohms_per_square_at_microwave() -> None:
    """The number that sets a cavity's Q."""
    copper = EM_MATERIALS["copper"]

    assert copper.surface_resistance_ohm(2.45e9) == pytest.approx(
        12.8e-3, rel=0.05)
    assert copper.surface_resistance_ohm(2.45e9) == pytest.approx(
        1.0 / (copper.conductivity_s_m * copper.skin_depth_m(2.45e9)))


def test_a_worse_conductor_has_a_higher_surface_resistance() -> None:
    at = 2.45e9
    assert (EM_MATERIALS["stainless-steel"].surface_resistance_ohm(at)
            > EM_MATERIALS["aluminium"].surface_resistance_ohm(at)
            > EM_MATERIALS["copper"].surface_resistance_ohm(at))


def test_water_absorbs_where_ptfe_does_not() -> None:
    """Why wet things warm in a field and dry ones do not.

    High permittivity and a large loss tangent together; PTFE has
    neither, and the ratio between them is four orders of magnitude.
    """
    field = 1.0e4
    water = EM_MATERIALS["water"].dielectric_loss_w_m3(2.45e9, field)
    ptfe = EM_MATERIALS["ptfe"].dielectric_loss_w_m3(2.45e9, field)

    assert water > 1.0e8
    assert water / ptfe > 1.0e4


def test_dielectric_loss_goes_with_the_square_of_the_field() -> None:
    water = EM_MATERIALS["water"]

    assert water.dielectric_loss_w_m3(2.45e9, 2.0e4) == pytest.approx(
        4.0 * water.dielectric_loss_w_m3(2.45e9, 1.0e4))


def test_a_lossless_dielectric_absorbs_nothing() -> None:
    assert EM_MATERIALS["air"].dielectric_loss_w_m3(2.45e9, 1.0e5) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# reading it off a part that was never edited
# --------------------------------------------------------------------------

def test_an_existing_part_gains_field_behaviour_from_what_it_already_says() -> None:
    """No node has to be rewritten to acquire a skin depth."""
    node = {"identity": "lab.dewar.jacket.floor", "material": "stainless-steel"}

    material = material_for(node)
    assert material is not None
    assert material.key == "stainless-steel"


def test_a_part_may_declare_its_field_identity_separately() -> None:
    """A plated housing is polymer to stress and copper to a field."""
    node = {"material": "polyethylene", "em_material": "copper"}

    assert material_for(node).key == "copper"


def test_a_material_the_table_does_not_know_says_nothing() -> None:
    """Silence rather than a default: a guessed conductivity is a lie."""
    assert material_for({"material": "unobtainium"}) is None
    assert material_for({}) is None


def test_materials_declare_themselves_for_the_graph() -> None:
    attributes = EM_MATERIALS["copper"].graph_attributes()

    assert attributes["em_material"] == "copper"
    assert attributes["em_conductor"] is True
    assert attributes["conductivity_s_m"] == COPPER_CONDUCTIVITY_S_M
    assert EM_MATERIALS["ptfe"].graph_attributes()["em_conductor"] is False


def test_every_row_is_physically_admissible() -> None:
    """Nothing may have negative conductivity or sub-vacuum permittivity."""
    for key, material in EM_MATERIALS.items():
        assert isinstance(material, EMMaterial)
        assert material.conductivity_s_m >= 0.0, key
        assert material.relative_permittivity >= 1.0, key
        assert material.relative_permeability > 0.0, key
        assert 0.0 <= material.loss_tangent < 1.0, key
