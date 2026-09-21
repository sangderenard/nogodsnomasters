"""Copper between components, so circuits are laid out rather than wired.

Every quantity here is derived from geometry and checked against a rule
of thumb that was arrived at independently: a one-ounce trace is about
half a milliohm per square, board inductance is about a nanohenry per
millimetre, and a half-millimetre one-ounce trace carries about an amp
and a half for a ten degree rise.  None of those numbers is typed into
the module; they fall out of resistivity, the Rosa/Grover formula and
IPC-2221.

The architectural point is in ``test_a_layout_has_more_nodes_than_its
_schematic``: modelling copper honestly multiplies the nodes, and Kron
reduction takes them back out while keeping what the copper did.
"""

from __future__ import annotations

import math

import pytest

from circuit_graph import Branch, CircuitGraph
from dc_power import Conductor
from traces import MIL_M, OUNCE_COPPER_M, PcbTrace, WireRun


def _trace(**kwargs):
    defaults = dict(identity="t", tail="a", head="b",
                    length_m=0.010, width_m=0.5e-3)
    defaults.update(kwargs)
    return PcbTrace(**defaults)


# --------------------------------------------------------------------------
# geometry into electrical quantities
# --------------------------------------------------------------------------

def test_copper_weight_is_a_thickness() -> None:
    """One ounce over a square foot is 34.8 microns, and two is twice that."""
    assert _trace().thickness_m == pytest.approx(OUNCE_COPPER_M)
    assert _trace(copper_weight_oz=2.0).thickness_m == pytest.approx(
        2.0 * OUNCE_COPPER_M)


def test_resistance_is_half_a_milliohm_per_square_in_one_ounce() -> None:
    """The board designer's unit, reproduced rather than declared."""
    trace = _trace()

    assert trace.squares == pytest.approx(20.0)
    assert trace.ohms_per_square == pytest.approx(0.5e-3, rel=0.05)
    assert trace.resistance_ohm == pytest.approx(
        trace.ohms_per_square * trace.squares)


def test_a_wider_trace_is_proportionally_less_resistive() -> None:
    narrow = _trace(width_m=0.25e-3)
    wide = _trace(width_m=1.0e-3)

    assert narrow.resistance_ohm == pytest.approx(4.0 * wide.resistance_ohm)


def test_resistance_climbs_with_temperature() -> None:
    cold = _trace(temperature_c=20.0)
    hot = _trace(temperature_c=120.0)

    assert hot.resistance_ohm / cold.resistance_ohm == pytest.approx(
        1.0 + 0.00393 * 100.0, rel=1e-6)


def test_inductance_lands_near_a_nanohenry_per_millimetre() -> None:
    """Rosa/Grover for a straight flat conductor, against the rule of thumb."""
    trace = _trace(length_m=0.010)

    assert trace.inductance_h == pytest.approx(10.0e-9, rel=0.25)
    # Longer is more, and superlinearly so through the logarithm.
    assert _trace(length_m=0.020).inductance_h > 2.0 * trace.inductance_h * 0.9


def test_a_conductor_shorter_than_its_own_section_is_refused() -> None:
    """That is a pad, and its inductance is not this formula's question."""
    with pytest.raises(ValueError, match="that is a pad"):
        _trace(length_m=0.0001, width_m=1.0e-3).inductance_h


def test_ampacity_follows_ipc_2221() -> None:
    trace = _trace()
    area_mils2 = trace.area_m2 / (MIL_M ** 2)

    assert trace.ampacity_a == pytest.approx(
        0.048 * 10.0 ** 0.44 * area_mils2 ** 0.725)
    assert trace.ampacity_a == pytest.approx(1.5, rel=0.15)


def test_a_buried_trace_carries_half_as_much() -> None:
    """It has nowhere to put the heat, and IPC-2221 says so with k."""
    exposed = _trace(layer="external")
    buried = _trace(layer="internal")

    assert buried.ampacity_a == pytest.approx(exposed.ampacity_a / 2.0)


def test_a_hotter_rise_allows_more_current() -> None:
    gentle = _trace(temperature_rise_c=10.0)
    permissive = _trace(temperature_rise_c=40.0)

    assert permissive.ampacity_a > gentle.ampacity_a


# --------------------------------------------------------------------------
# refusals and warnings
# --------------------------------------------------------------------------

def test_an_overloaded_trace_is_called_out() -> None:
    trace = _trace()
    notes = trace.design_notes(current_a=5.0)

    assert any("IPC-2221" in note for note in notes)


def test_an_unetchable_width_is_called_out() -> None:
    notes = _trace(width_m=0.05e-3).design_notes()

    assert any("etch" in note for note in notes)


def test_a_sound_trace_has_nothing_to_complain_about() -> None:
    assert _trace().design_notes(current_a=1.0) == []


def test_an_unknown_layer_is_refused() -> None:
    with pytest.raises(ValueError, match="external or internal"):
        _trace(layer="somewhere")


def test_zero_geometry_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _trace(width_m=0.0)


# --------------------------------------------------------------------------
# wire reuses what already models wire
# --------------------------------------------------------------------------

def test_a_wire_run_takes_its_resistance_from_the_conductor() -> None:
    """``dc_power.Conductor`` already models copper; this places it."""
    conductor = Conductor("batt", awg=10, length_m=3.0)
    run = WireRun("w", "x", "y", conductor)

    assert run.resistance_ohm == conductor.resistance_ohm
    assert run.ampacity_a == conductor.ampacity_a
    # 10 AWG is about 3.3 mohm/m, out and back over three metres.
    assert run.resistance_ohm == pytest.approx(20.0e-3, rel=0.1)


def test_a_wire_run_has_inductance_in_microhenries() -> None:
    """Metres of wire are a very different object from millimetres of trace."""
    run = WireRun("w", "x", "y", Conductor("batt", awg=10, length_m=3.0))

    assert 1.0e-6 < run.inductance_h < 100.0e-6


def test_the_out_and_back_rule_doubles_the_run() -> None:
    there = WireRun("w", "x", "y",
                    Conductor("c", awg=10, length_m=3.0, both_directions=False))
    back = WireRun("w", "x", "y",
                   Conductor("c", awg=10, length_m=3.0, both_directions=True))

    assert back.resistance_ohm == pytest.approx(2.0 * there.resistance_ohm)


# --------------------------------------------------------------------------
# the architectural point
# --------------------------------------------------------------------------

def test_a_trace_is_two_branches_through_an_interior_node() -> None:
    branches = _trace().branches()

    assert len(branches) == 2
    assert {branch.kind for branch in branches} == {"resistor", "inductor"}
    # They meet at a node named after the trace, which nothing else uses.
    assert branches[0].head == branches[1].tail == "t~"


def test_a_layout_has_more_nodes_than_its_schematic_and_reduces_back() -> None:
    """Model the copper honestly, then hand runtime a small matrix.

    The schematic has three nodes.  Laying one connection out as real
    copper makes five.  Kron reduction eliminates the three interior
    ones exactly -- and the copper's own resistance survives into the
    answer, which is the whole reason for modelling it.
    """
    schematic = CircuitGraph("schematic", (
        Branch("r1", "in", "mid", "resistor", 100.0),
        Branch("r2", "mid", "gnd", "resistor", 300.0),
    ), ports=("in", "gnd"))

    trace = PcbTrace("tr", "n1", "mid", 0.020, 0.4e-3)
    laid_out = CircuitGraph("laid-out", (
        Branch("r1", "in", "n1", "resistor", 100.0),
        *trace.branches(),
        Branch("r2", "mid", "gnd", "resistor", 300.0),
    ), ports=("in", "gnd"))

    assert len(schematic.nodes) == 3
    assert len(laid_out.nodes) == 5

    reduced = laid_out.reduce(1000.0)
    assert reduced.eliminated_nodes == 3
    assert reduced.admittance.shape == (2, 2)

    impedance = abs(reduced.two_terminal_impedance_ohm())
    assert impedance == pytest.approx(400.0 + trace.resistance_ohm, rel=1e-4)
    assert impedance > 400.0, "the copper has to show up somewhere"


def test_traces_declare_themselves_for_the_graph() -> None:
    attributes = _trace().graph_attributes()

    assert attributes["part_role"] == "pcb-trace"
    assert attributes["ampacity_a"] > 0.0
    assert attributes["inductance_h"] > 0.0

    wire = WireRun("w", "x", "y", Conductor("c", awg=14, length_m=1.0))
    assert wire.graph_attributes()["part_role"] == "wire-run"
    assert wire.graph_attributes()["awg"] == 14
