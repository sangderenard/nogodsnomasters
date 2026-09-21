"""The transformer, and the shunts that make one limit its own current.

A magnetron oven transformer is the reason this module exists: its load
cannot regulate itself, so the supply must, and it does that with
magnetic shunts rather than with electronics.  Model it as an ideal
transformer and the secondary current is bounded only by winding
resistance, which is both wrong and uninformative.

The numbers here are related by physics rather than asserted.  Turns,
core area and frequency fix the flux density through the EMF equation,
and a design either fits inside its steel or does not.
"""

from __future__ import annotations

import math

import pytest

from electrical_distribution import standard_services
from transformers import CORE_STEELS, IronCoreTransformer, Winding


def _service():
    # A consumer oven takes one leg of the split-phase service: 120 V.
    return standard_services("oven")["120-240v-split-60hz"]


def _oven_transformer(leakage_inductance_h: float | None = 7.0):
    return IronCoreTransformer(
        identity="oven.mot",
        service=_service(),
        core="m19",
        core_mass_kg=6.2,
        core_area_m2=0.0032,
        primary=Winding("primary", 83, 1.63, 0.26, 13.0),
        secondary=Winding("secondary", 1450, 0.13, 0.31, 0.55),
        filament=Winding("filament", 2, 6.0, 0.24, 10.0),
        leakage_inductance_h=leakage_inductance_h,
        label="magnetron oven transformer",
    )


# --------------------------------------------------------------------------
# magnetics
# --------------------------------------------------------------------------

def test_a_split_phase_appliance_takes_the_line_to_neutral_leg() -> None:
    """120 V, not 240 V: the service supplies both and the oven takes one."""
    assert _oven_transformer().primary_voltage_v == pytest.approx(120.0)


def test_flux_density_follows_the_transformer_emf_equation() -> None:
    """``B = V / (4.44 f N A)``, computed rather than declared."""
    transformer = _oven_transformer()
    expected = 120.0 / (4.44 * 60.0 * 83 * 0.0032)

    assert transformer.peak_flux_density_tesla == pytest.approx(expected)
    assert transformer.peak_flux_density_tesla == pytest.approx(1.70, abs=0.02)


def test_more_turns_means_less_flux() -> None:
    """The inverse relation is the designer's only real handle on saturation."""
    few = _oven_transformer()
    many = IronCoreTransformer(
        "oven.mot", _service(), "m19", 6.2, 0.0032,
        Winding("primary", 166, 1.63, 0.26, 13.0),
        Winding("secondary", 2900, 0.13, 0.31, 0.55))

    assert many.peak_flux_density_tesla == pytest.approx(
        few.peak_flux_density_tesla / 2.0)
    # Doubling both windings leaves the ratio, hence the HV, untouched.
    assert many.open_circuit_secondary_v == pytest.approx(
        few.open_circuit_secondary_v)


def test_a_core_run_too_hard_is_reported_as_saturating() -> None:
    """Too few turns for the voltage is a design error, and it says so."""
    starved = IronCoreTransformer(
        "bad.mot", _service(), "m19", 6.2, 0.0032,
        Winding("primary", 30, 1.63, 0.26, 13.0),
        Winding("secondary", 520, 0.13, 0.31, 0.55))

    assert starved.saturation_margin > 1.0
    assert any("saturates" in note for note in starved.design_notes())


def test_a_cheap_core_near_saturation_is_called_out_but_allowed() -> None:
    """Running hot to save iron is a real choice, not a fault."""
    transformer = _oven_transformer()

    assert 0.85 < transformer.saturation_margin < 1.0
    assert any("saturation" in note for note in transformer.design_notes())


def test_the_secondary_and_filament_follow_their_turns() -> None:
    transformer = _oven_transformer()

    assert transformer.open_circuit_secondary_v == pytest.approx(
        120.0 * 1450 / 83, rel=1e-9)
    assert transformer.filament_voltage_v == pytest.approx(
        120.0 * 2 / 83, rel=1e-9)


def test_a_transformer_without_a_filament_winding_has_no_filament_voltage() -> None:
    plain = IronCoreTransformer(
        "plain", _service(), "m19", 6.2, 0.0032,
        Winding("primary", 83, 1.63, 0.26, 13.0),
        Winding("secondary", 1450, 0.13, 0.31, 0.55))

    assert plain.filament_voltage_v is None


# --------------------------------------------------------------------------
# what the shunts are for
# --------------------------------------------------------------------------

def test_shunts_set_a_finite_short_circuit_current() -> None:
    """The defining behaviour: shorted, it settles instead of running away."""
    transformer = _oven_transformer()

    reactance = 2.0 * math.pi * 60.0 * 7.0
    impedance = math.hypot(reactance, transformer.secondary.resistance_ohm)
    assert transformer.short_circuit_secondary_a == pytest.approx(
        transformer.open_circuit_secondary_v / impedance)
    # A magnetron oven transformer limits well under an amp.
    assert 0.5 < transformer.short_circuit_secondary_a < 1.0


def test_without_shunts_nothing_limits_the_secondary() -> None:
    """``None`` is the honest answer, and the design notes say why."""
    unshunted = _oven_transformer(leakage_inductance_h=None)

    assert unshunted.short_circuit_secondary_a is None
    assert unshunted.leakage_reactance_ohm is None
    assert any("does not limit" in note for note in unshunted.design_notes())


def test_the_load_line_is_a_circle_not_a_straight_line() -> None:
    """``V^2 + (Z I)^2 = V_oc^2`` -- the drop is taken in quadrature.

    This is what makes the supply hold up under light load and collapse
    near its limit, instead of sagging proportionally from the start.
    """
    transformer = _oven_transformer()
    open_circuit = transformer.open_circuit_secondary_v
    impedance = math.hypot(transformer.leakage_reactance_ohm,
                           transformer.secondary.resistance_ohm)

    for current in (0.0, 0.1, 0.3, 0.5, 0.7):
        terminal = transformer.secondary_voltage_at(current)
        assert terminal ** 2 + (impedance * current) ** 2 == pytest.approx(
            open_circuit ** 2, rel=1e-9)


def test_regulation_is_gentle_at_rated_load_and_collapses_at_the_limit() -> None:
    transformer = _oven_transformer()
    open_circuit = transformer.open_circuit_secondary_v

    rated = transformer.secondary_voltage_at(0.30)
    assert rated / open_circuit > 0.90, "should barely sag at rated current"

    near_limit = transformer.secondary_voltage_at(0.70)
    assert near_limit / open_circuit < 0.55, "should be collapsing by here"

    beyond = transformer.secondary_voltage_at(
        transformer.short_circuit_secondary_a * 1.5)
    assert beyond == 0.0, "past the short-circuit current there is nothing left"


# --------------------------------------------------------------------------
# losses become heat
# --------------------------------------------------------------------------

def test_copper_resistance_climbs_with_temperature() -> None:
    """0.393% per kelvin: a hot winding is a third more resistive."""
    winding = _oven_transformer().primary

    assert winding.resistance_ohm_at(20.0) == pytest.approx(
        winding.resistance_ohm)
    assert winding.resistance_ohm_at(130.0) / winding.resistance_ohm == (
        pytest.approx(1.0 + 0.00393 * 110.0))


def test_copper_loss_grows_with_the_square_of_current() -> None:
    transformer = _oven_transformer()

    single = transformer.copper_loss_w(0.2)
    double = transformer.copper_loss_w(0.4)
    assert double / single == pytest.approx(4.0, rel=1e-9)


def test_core_loss_does_not_depend_on_load() -> None:
    """Magnetising loss is paid for whenever it is energised, load or none."""
    transformer = _oven_transformer()

    assert transformer.heat_w(0.0) == pytest.approx(transformer.core_loss_w)
    assert transformer.core_loss_w > 0.0


def test_core_loss_scales_with_the_square_of_flux_density() -> None:
    """Halving the flux quarters the iron loss, at fixed mass and frequency."""
    hard = _oven_transformer()
    gentle = IronCoreTransformer(
        "gentle", _service(), "m19", 6.2, 0.0032,
        Winding("primary", 166, 1.63, 0.26, 13.0),
        Winding("secondary", 2900, 0.13, 0.31, 0.55))

    assert gentle.core_loss_w == pytest.approx(hard.core_loss_w / 4.0, rel=1e-9)


def test_a_better_steel_loses_less_in_the_same_design() -> None:
    hard = _oven_transformer()
    better = IronCoreTransformer(
        "better", _service(), "m6", 6.2, 0.0032,
        hard.primary, hard.secondary, hard.filament, 7.0)

    assert better.core_loss_w < hard.core_loss_w
    assert CORE_STEELS["m6"].specific_loss_w_kg < CORE_STEELS["m19"].specific_loss_w_kg


def test_heat_is_what_a_thermal_system_would_be_handed() -> None:
    transformer = _oven_transformer()

    assert transformer.heat_w(0.30) == pytest.approx(
        transformer.core_loss_w + transformer.copper_loss_w(0.30))
    assert 30.0 < transformer.heat_w(0.30) < 60.0


# --------------------------------------------------------------------------
# declaration
# --------------------------------------------------------------------------

def test_a_hard_worked_winding_is_called_out() -> None:
    """8 A/mm2 is a duty-cycle limit, which an oven transformer really has."""
    notes = _oven_transformer().design_notes()

    assert any("A/mm2" in note for note in notes)


def test_graph_attributes_declare_the_current_limiting_fact() -> None:
    """A consumer of the graph must not have to infer this from a number."""
    shunted = _oven_transformer().graph_attributes()
    plain = _oven_transformer(leakage_inductance_h=None).graph_attributes()

    assert shunted["current_limiting"] is True
    assert shunted["part_role"] == "transformer"
    assert shunted["short_circuit_secondary_a"] is not None
    assert plain["current_limiting"] is False
    assert plain["short_circuit_secondary_a"] is None
    # The service travels with it, so the graph knows what it plugs into.
    assert shunted["frequency_hz"] == 60.0


def test_an_unknown_core_steel_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown core steel"):
        IronCoreTransformer(
            "bad", _service(), "cheese", 6.2, 0.0032,
            Winding("primary", 83, 1.63, 0.26, 13.0),
            Winding("secondary", 1450, 0.13, 0.31, 0.55))


def test_a_dc_service_cannot_run_a_transformer() -> None:
    with pytest.raises(ValueError, match="AC service"):
        IronCoreTransformer(
            "bad", standard_services("x")["48v-dc"], "m19", 6.2, 0.0032,
            Winding("primary", 83, 1.63, 0.26, 13.0),
            Winding("secondary", 1450, 0.13, 0.31, 0.55))
