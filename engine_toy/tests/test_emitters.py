"""Emitters and radiators, checked against figures from outside them.

The radiation resistance is the interesting one.  ``eta k^4 A^2 / 6 pi``
is derived here from a magnetic dipole's radiated power, and the trade
quotes ``31171 (A / lambda^2)^2`` from tables; the two must agree, and
their agreeing is what says the derivation is the same physics the
tables came from.

The magnetron is checked against the transformer that was already in
this repository waiting for it -- the one whose label says "magnetron
oven transformer" and which carries a filament winding no power
transformer would need.
"""

from __future__ import annotations

import math

import pytest

from electrical_distribution import standard_services
from emitters import Magnetron, RFSource, RadiationLoad
from transformers import IronCoreTransformer, Winding
from waveguides import SPEED_OF_LIGHT


def _oven_transformer(leakage_inductance_h: float | None = 7.0,
                      filament_turns: int = 2) -> IronCoreTransformer:
    """The same declaration the transformer suite uses."""
    return IronCoreTransformer(
        identity="oven.mot",
        service=standard_services("oven")["120-240v-split-60hz"],
        core="m19",
        core_mass_kg=6.2,
        core_area_m2=0.0032,
        primary=Winding("primary", 83, 1.63, 0.26, 13.0),
        secondary=Winding("secondary", 1450, 0.13, 0.31, 0.55),
        filament=Winding("filament", filament_turns, 6.0, 0.24, 10.0),
        leakage_inductance_h=leakage_inductance_h,
        label="magnetron oven transformer",
    )


# --------------------------------------------------------------------------
# radiation: power that leaves, not power that heats
# --------------------------------------------------------------------------

def test_small_loop_radiation_matches_the_tabulated_constant() -> None:
    """``eta k^4 A^2 / 6 pi`` against ``31171 (A/lambda^2)^2``.

    The table's constant is rounded, so agreement to a tenth of a
    percent is the most this can show -- and is enough, because the two
    were arrived at independently.
    """
    frequency, area = 2.45e9, 1.0e-4
    wavelength = SPEED_OF_LIGHT / frequency

    assert RadiationLoad.small_loop_resistance_ohm(area, frequency) == (
        pytest.approx(31171.0 * (area / wavelength ** 2) ** 2, rel=1e-3))


def test_radiation_resistance_goes_as_area_squared_and_frequency_fourth() -> None:
    """The same fourth power that makes a small hole leak so little.

    A radiator and an aperture are the same physics from the two sides,
    so the exponents have to match.
    """
    base = RadiationLoad.small_loop_resistance_ohm(1.0e-4, 1.0e9)

    assert RadiationLoad.small_loop_resistance_ohm(2.0e-4, 1.0e9) == (
        pytest.approx(4.0 * base, rel=1e-12))
    assert RadiationLoad.small_loop_resistance_ohm(1.0e-4, 2.0e9) == (
        pytest.approx(16.0 * base, rel=1e-12))


def test_a_radiator_declares_that_its_power_is_gone() -> None:
    """The distinction the whole ledger rests on."""
    load = RadiationLoad.small_loop("sky", "out", "gnd", 1.0e-4, 2.45e9)

    assert load.graph_attributes()["part_role"] == "radiation-load"
    assert load.graph_attributes()["radiated"] is True
    assert load.branches()[0].kind == "resistor"
    assert load.injections() == {}


def test_the_small_loop_range_is_a_predicate() -> None:
    small = RadiationLoad.small_loop("ok", "a", "b", 1.0e-4, 2.45e9)
    large = RadiationLoad.small_loop("big", "a", "b", 1.0, 2.45e9)

    assert small.is_small_at(2.45e9)
    assert not large.is_small_at(2.45e9)


def test_a_bare_resistance_has_no_size_to_check() -> None:
    """Refused rather than assumed: a declared ohm value is not a loop."""
    plain = RadiationLoad("load", "a", "b", 50.0)

    with pytest.raises(ValueError, match="not as a loop"):
        plain.is_small_at(2.45e9)


def test_a_radiator_that_radiates_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        RadiationLoad("dead", "a", "b", 0.0)
    with pytest.raises(ValueError, match="positive area"):
        RadiationLoad.small_loop("dead", "a", "b", 0.0, 2.45e9)


# --------------------------------------------------------------------------
# the emitter
# --------------------------------------------------------------------------

def test_available_power_round_trips_through_the_declaration() -> None:
    """Sources are sold by watts, so they can be declared by watts."""
    source = RFSource.from_available_power("tx", "a", "b", 2.45e9, 800.0, 50.0)

    assert source.available_power_w == pytest.approx(800.0)
    assert source.internal_resistance_ohm == 50.0
    assert abs(source.emf_v) == pytest.approx(math.sqrt(8.0 * 50.0 * 800.0))


def test_available_power_is_what_a_matched_load_would_take() -> None:
    """Half the EMF across a matched load, so a quarter of the power the
    EMF would deliver into the source impedance alone."""
    source = RFSource("tx", "a", "b", 2.45e9, 100.0, 50.0)

    assert source.available_power_w == pytest.approx(
        0.5 * abs(50.0) ** 2 / 50.0)


def test_an_emitter_needs_a_frequency_and_a_real_impedance() -> None:
    with pytest.raises(ValueError, match="needs a frequency"):
        RFSource("tx", "a", "b", 0.0, 1.0, 50.0)
    with pytest.raises(ValueError, match="positive source"):
        RFSource("tx", "a", "b", 2.45e9, 1.0, 0.0)


# --------------------------------------------------------------------------
# the tube
# --------------------------------------------------------------------------

def test_the_tubes_numbers_relate_rather_than_float_free() -> None:
    """DC in, RF out and anode heat are one statement, not three."""
    tube = Magnetron("mag", anode_voltage_v=4000.0, anode_current_a=0.3,
                     efficiency=0.7)

    assert tube.dc_input_w == pytest.approx(1200.0)
    assert tube.rf_output_w == pytest.approx(840.0)
    assert tube.anode_heat_w == pytest.approx(360.0)
    assert tube.rf_output_w + tube.anode_heat_w == pytest.approx(
        tube.dc_input_w)


def test_a_tube_cannot_emit_more_than_it_is_fed() -> None:
    with pytest.raises(ValueError, match="cannot emit more"):
        Magnetron("mag", efficiency=1.4)
    with pytest.raises(ValueError, match="real operating point"):
        Magnetron("mag", anode_current_a=0.0)


def test_the_tube_becomes_an_emitter_at_its_own_output() -> None:
    tube = Magnetron("mag")
    source = tube.as_source("feed", "return")

    assert source.frequency_hz == pytest.approx(2.45e9)
    assert source.available_power_w == pytest.approx(tube.rf_output_w)


def test_a_tube_declared_for_this_supply_passes_cleanly() -> None:
    """The transformer was declared with a filament winding and magnetic
    shunts long before there was a tube to plug into it.  A tube declared
    to match what it actually delivers raises nothing."""
    tube = Magnetron("mag", anode_voltage_v=1900.0, filament_voltage_v=2.89)

    assert tube.supply_notes(_oven_transformer()) == ()


def test_the_nominal_tube_is_flagged_against_this_transformer() -> None:
    """A real cross-module finding, recorded rather than smoothed over.

    Two turns off an 83-turn primary on a 120 V leg is 2.89 V, while the
    common magnetron filament nominal is 3.3 V -- fourteen percent low.
    Neither module is wrong on its own; they were declared apart and
    were never checked against each other until now.  Whether the winding
    or the assumed nominal should move is a question for whoever owns
    the oven, so the check reports rather than decides.
    """
    nominal = Magnetron("mag", anode_voltage_v=1900.0, filament_voltage_v=3.3)
    notes = nominal.supply_notes(_oven_transformer())

    assert len(notes) == 1
    assert "filament winding 2.89 V" in notes[0]


def test_a_supply_that_does_not_fit_says_which_way_it_fails() -> None:
    """Three distinct failures, each named rather than lumped."""
    tube = Magnetron("mag", anode_voltage_v=1900.0, filament_voltage_v=3.3)

    unshunted = tube.supply_notes(_oven_transformer(leakage_inductance_h=None))
    assert any("limit its current" in note for note in unshunted)

    wrong_filament = tube.supply_notes(_oven_transformer(filament_turns=6))
    assert any("filament winding" in note for note in wrong_filament)

    thirsty = Magnetron("big", anode_voltage_v=9000.0, filament_voltage_v=3.3)
    assert any("below the" in note
               for note in thirsty.supply_notes(_oven_transformer()))


def test_the_tube_declares_itself_to_the_graph() -> None:
    declared = Magnetron("mag").graph_attributes()

    assert declared["part_role"] == "magnetron"
    assert declared["frequency_hz"] == pytest.approx(2.45e9)
    assert declared["filament_power_w"] == pytest.approx(33.0)
