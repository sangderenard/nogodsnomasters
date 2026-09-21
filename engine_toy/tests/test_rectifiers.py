"""The half-wave doubler, and why an oven's output is pulsed.

The doubler is the reason a consumer oven's RF is a carrier gated at
mains frequency rather than a steady wave: the capacitor is refilled
once per cycle and the magnetron conducts only while the output is above
its turn-on knee.  ``conduction_fraction`` is therefore the number the
field layer consumes, and it is derived from the geometry of the
waveform rather than declared.

One thing is deliberately ABSENT and pinned as absent: loaded
regulation.  Treating the capacitor as a reservoir draining at ``I t/C``
describes a parallel filter, not this series pump, and at an oven's
ordinary current it predicts a swing larger than the peak it swings
about.  See ``test_the_loaded_window_is_not_claimed``.
"""

from __future__ import annotations

import math

import pytest

from electrical_distribution import standard_services
from transformers import IronCoreTransformer, Winding
from rectifiers import HalfWaveDoubler, HighVoltageCapacitor, HighVoltageDiode


def _mot():
    return IronCoreTransformer(
        "oven.mot", standard_services("oven")["120-240v-split-60hz"],
        "m19", 6.2, 0.0032,
        Winding("primary", 83, 1.63, 0.26, 13.0),
        Winding("secondary", 1450, 0.13, 0.31, 0.55),
        Winding("filament", 2, 6.0, 0.24, 10.0),
        leakage_inductance_h=7.0)


def _doubler(piv_v=12000.0, bleeder=10.0e6, knee_v=4000.0):
    return HalfWaveDoubler(
        "oven.doubler", _mot(),
        HighVoltageCapacitor("oven.hv_cap", 1.05e-6, 2100.0, bleeder),
        HighVoltageDiode("oven.hv_diode", piv_v),
        load_knee_v=knee_v)


# --------------------------------------------------------------------------
# the doubling
# --------------------------------------------------------------------------

def test_the_output_is_twice_the_secondary_peak_less_the_diode_drop() -> None:
    doubler = _doubler()
    peak = doubler.transformer.open_circuit_secondary_v * math.sqrt(2.0)

    assert doubler.secondary_peak_v == pytest.approx(peak)
    assert doubler.open_circuit_output_v == pytest.approx(
        2.0 * peak - doubler.diode.forward_drop_v)
    assert doubler.open_circuit_output_v == pytest.approx(5920.0, abs=5.0)


def test_a_high_voltage_diode_is_a_stack_and_drops_like_one() -> None:
    """Ten dies in series drop nine volts, not the familiar 0.7."""
    diode = HighVoltageDiode("d", 12000.0, series_dies=10, die_forward_drop_v=0.9)

    assert diode.forward_drop_v == pytest.approx(9.0)


def test_the_diode_stands_off_twice_peak_when_it_is_not_conducting() -> None:
    """The constraint that forces the part to be a stack in the first place."""
    doubler = _doubler()

    assert doubler.diode_reverse_v == pytest.approx(
        2.0 * doubler.secondary_peak_v)
    assert doubler.diode_reverse_v > doubler.transformer.open_circuit_secondary_v


def test_a_diode_rated_only_for_the_transformer_peak_is_refused() -> None:
    """Rating for the winding rather than for twice it is the classic error."""
    notes = _doubler(piv_v=4000.0).design_notes()

    assert any("inverse against" in note for note in notes)


def test_a_thin_inverse_margin_is_called_out_without_being_an_error() -> None:
    doubler = _doubler(piv_v=6500.0)
    notes = doubler.design_notes()

    assert doubler.diode.peak_inverse_v > doubler.diode_reverse_v
    assert any("margin" in note for note in notes)


# --------------------------------------------------------------------------
# why the output is pulsed
# --------------------------------------------------------------------------

def test_conduction_is_a_window_not_the_whole_cycle() -> None:
    """``V_pk (1 + sin theta) = V_knee`` opens and closes the tube.

    38.6% of the mains cycle for an ordinary oven: the RF is a 2.45 GHz
    carrier gated at 60 Hz, which is why ovens are described by a duty
    cycle at all.
    """
    doubler = _doubler()
    fraction = doubler.conduction_fraction()

    assert 0.0 < fraction < 1.0
    assert fraction == pytest.approx(0.386, abs=0.005)


def test_a_higher_knee_narrows_the_window() -> None:
    """A harder tube to start conducts for less of each cycle."""
    easy = _doubler(knee_v=3000.0).conduction_fraction()
    hard = _doubler(knee_v=5000.0).conduction_fraction()

    assert easy > hard > 0.0


def test_a_knee_above_the_doubled_peak_never_conducts() -> None:
    """The tube stays dark, and the design notes say so rather than dividing."""
    doubler = _doubler(knee_v=9000.0)

    assert doubler.conduction_fraction() == 0.0
    assert doubler.delivered_power_w(0.3) == 0.0
    assert any("never" in note for note in doubler.design_notes())


def test_the_duty_cycle_governs_timing_not_total_energy() -> None:
    """``V_knee * I_average`` -- the fraction cancels, and that is the point.

    A narrower window means the same energy arrives in sharper bursts,
    not that less of it arrives.
    """
    doubler = _doubler()

    assert doubler.delivered_power_w(0.30) == pytest.approx(4000.0 * 0.30)


def test_charge_per_cycle_is_stated_and_voltage_swing_is_not() -> None:
    doubler = _doubler()

    assert doubler.charge_per_cycle_c(0.30) == pytest.approx(0.30 / 60.0)
    assert not hasattr(doubler, "ripple_v")


def test_the_loaded_window_is_not_claimed() -> None:
    """Conduction takes no current argument, on purpose.

    The reservoir formula ``I t / C`` gives 5000 V of swing at 300 mA on
    a 2965 V peak -- impossible -- because this capacitor is a series
    pump that the transformer drives through, not a parallel filter.
    Refusing to answer is better than answering wrongly; the transient
    layer is where the loaded cycle gets solved.
    """
    import inspect

    signature = inspect.signature(_doubler().conduction_fraction)
    assert not signature.parameters, (
        "conduction_fraction must not pretend to model loaded regulation")


# --------------------------------------------------------------------------
# the capacitor is the hazard
# --------------------------------------------------------------------------

def test_the_capacitor_stores_a_serious_amount_of_energy() -> None:
    """Half CV^2 at several kilovolts is joules, not millijoules."""
    doubler = _doubler()
    stored = doubler.capacitor.stored_energy_j(doubler.open_circuit_output_v)

    assert stored == pytest.approx(
        0.5 * 1.05e-6 * doubler.open_circuit_output_v ** 2)
    assert stored > 10.0


def test_a_bleeder_discharges_it_on_an_rc_decay() -> None:
    doubler = _doubler()
    seconds = doubler.capacitor.bleed_time_s(doubler.open_circuit_output_v)

    tau = 10.0e6 * 1.05e-6
    assert seconds == pytest.approx(
        tau * math.log(doubler.open_circuit_output_v / 50.0))
    assert 30.0 < seconds < 120.0


def test_without_a_bleeder_there_is_no_discharge_time_and_it_is_a_defect() -> None:
    """``None`` means it holds its charge until something else takes it."""
    doubler = _doubler(bleeder=None)

    assert doubler.capacitor.bleed_time_s(5000.0) is None
    assert any("bleeder" in note for note in doubler.design_notes())


def test_an_already_safe_capacitor_needs_no_bleed_time() -> None:
    capacitor = HighVoltageCapacitor("c", 1.05e-6, 2100.0, 10.0e6)

    assert capacitor.bleed_time_s(40.0) is None


def test_an_undersized_capacitor_rating_is_called_out() -> None:
    weak = HalfWaveDoubler(
        "weak", _mot(),
        HighVoltageCapacitor("c", 1.05e-6, 900.0, 10.0e6),
        HighVoltageDiode("d", 12000.0), load_knee_v=4000.0)

    assert any("VAC against" in note for note in weak.design_notes())


def test_a_sound_design_has_nothing_to_complain_about() -> None:
    assert _doubler().design_notes() == []


# --------------------------------------------------------------------------
# declaration
# --------------------------------------------------------------------------

def test_the_graph_declares_that_the_output_is_pulsed() -> None:
    """A consumer must not have to infer this from the topology's name."""
    attributes = _doubler().graph_attributes()

    assert attributes["part_role"] == "half-wave-voltage-doubler"
    assert attributes["pulsed_at_supply_frequency"] is True
    assert attributes["supply_frequency_hz"] == 60.0
    assert attributes["diode_reverse_v"] > attributes["secondary_peak_v"]


def test_the_capacitor_declares_itself_a_hazard() -> None:
    attributes = HighVoltageCapacitor("c", 1.05e-6, 2100.0).graph_attributes()

    assert attributes["stored_energy_hazard"] is True
