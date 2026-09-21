"""Real conductors between components, so a circuit is laid out rather than wired.

A netlist says two pins are connected.  A LAYOUT says how: by fifteen
millimetres of half-millimetre trace in one-ounce copper, or by a metre
of 14 AWG, and those are different circuits.  The difference shows up as
resistance that drops volts, inductance that rings, and a current limit
that is a property of the copper rather than a number somebody typed.

WHAT IS DERIVED, NOT DECLARED

  resistance    rho * L / (w t), with copper's temperature coefficient.
                A one-ounce trace is about half a milliohm per square,
                and this reproduces that without being told it.
  inductance    the Rosa/Grover partial self-inductance of a straight
                conductor.  It lands near the familiar nanohenry per
                millimetre for board geometry, again without being told.
  ampacity      IPC-2221, ``I = k dT^0.44 A^0.725`` with ``A`` in square
                mils and ``k`` differing for an exposed trace and a
                buried one, because a buried one cannot shed heat.

WHY THIS MAKES THE REDUCTION EARN ITS KEEP.  Every trace is two series
branches -- its resistance and its inductance -- with a node between
them, so laying a circuit out MULTIPLIES its nodes.  A schematic with
eight nodes becomes a layout with thirty.  Kron reduction takes them
back out exactly, which is why it is worth having: you model the copper
honestly and still hand the runtime a small matrix.

Wire is not reinvented here.  ``dc_power.Conductor`` already models a run
of copper of an AWG size with the real resistance, the real ampacity and
the real out-and-back gotcha; :class:`WireRun` gives it two endpoints and
an inductance so it can sit in a circuit graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from circuit_graph import Branch
from dc_power import (
    COPPER_RESISTIVITY_OHM_M, COPPER_TEMPCO_PER_C, Conductor,
)

#: One ounce of copper spread over a square foot: the board industry's
#: unit of thickness, which is 1.37 thousandths of an inch.
OUNCE_COPPER_M = 34.8e-6
MIL_M = 25.4e-6
VACUUM_PERMEABILITY = 4.0e-7 * math.pi


def _partial_self_inductance_h(length_m: float, width_m: float,
                               thickness_m: float) -> float:
    """Rosa/Grover partial self-inductance of a straight flat conductor.

    ``L = (mu0/2pi) l [ln(2l/(w+t)) + 1/2 + 0.2235 (w+t)/l]``

    It is a PARTIAL inductance: the loop it belongs to is not closed
    here, because the return path is another branch of the circuit and
    the graph is what closes it.  For ordinary board geometry this lands
    near the familiar nanohenry per millimetre.
    """
    perimeter = width_m + thickness_m
    if length_m <= 0.0 or perimeter <= 0.0:
        raise ValueError("a conductor needs a positive length and section")
    ratio = 2.0 * length_m / perimeter
    if ratio <= 1.0:
        # Shorter than it is wide: the formula's logarithm turns on it,
        # and the honest answer is that this is a pad, not a trace.
        raise ValueError(
            "conductor is shorter than its own section; that is a pad, "
            "and its inductance is not a straight-conductor question")
    return (VACUUM_PERMEABILITY / (2.0 * math.pi)) * length_m * (
        math.log(ratio) + 0.5 + 0.2235 * perimeter / length_m)


@dataclass(frozen=True)
class PcbTrace:
    """Copper on a board, between two named nodes.

    ``layer`` is ``external`` or ``internal`` and is not cosmetic: a
    buried trace carries roughly half the current of an exposed one at
    the same temperature rise, because it has nowhere to put the heat.
    """

    identity: str
    tail: str
    head: str
    length_m: float
    width_m: float
    copper_weight_oz: float = 1.0
    layer: str = "external"
    temperature_c: float = 25.0
    temperature_rise_c: float = 10.0

    def __post_init__(self) -> None:
        if self.layer not in ("external", "internal"):
            raise ValueError(f"{self.identity}: layer must be external or internal")
        if min(self.length_m, self.width_m, self.copper_weight_oz) <= 0.0:
            raise ValueError(f"{self.identity}: trace geometry must be positive")

    @property
    def thickness_m(self) -> float:
        return self.copper_weight_oz * OUNCE_COPPER_M

    @property
    def area_m2(self) -> float:
        return self.width_m * self.thickness_m

    @property
    def squares(self) -> float:
        """Length in widths: the unit a board designer actually thinks in."""
        return self.length_m / self.width_m

    @property
    def resistance_ohm(self) -> float:
        resistivity = COPPER_RESISTIVITY_OHM_M * (
            1.0 + COPPER_TEMPCO_PER_C * (self.temperature_c - 20.0))
        return resistivity * self.length_m / self.area_m2

    @property
    def ohms_per_square(self) -> float:
        return self.resistance_ohm / self.squares

    @property
    def inductance_h(self) -> float:
        return _partial_self_inductance_h(
            self.length_m, self.width_m, self.thickness_m)

    @property
    def ampacity_a(self) -> float:
        """IPC-2221 for the declared temperature rise."""
        constant = 0.048 if self.layer == "external" else 0.024
        area_mils2 = self.area_m2 / (MIL_M ** 2)
        return constant * self.temperature_rise_c ** 0.44 * area_mils2 ** 0.725

    def branches(self) -> tuple[Branch, ...]:
        """The resistance and inductance this trace really is.

        Two branches in series through an interior node named after the
        trace, which Kron reduction is expected to eliminate again.
        """
        interior = f"{self.identity}~"
        return (
            Branch(f"{self.identity}.r", self.tail, interior,
                   "resistor", self.resistance_ohm),
            Branch(f"{self.identity}.l", interior, self.head,
                   "inductor", self.inductance_h),
        )

    def design_notes(self, current_a: float | None = None) -> list[str]:
        notes: list[str] = []
        if current_a is not None and current_a > self.ampacity_a:
            notes.append(
                f"{self.identity} carries {current_a:.2f} A against an "
                f"IPC-2221 limit of {self.ampacity_a:.2f} A for a "
                f"{self.temperature_rise_c:.0f} C rise")
        if self.width_m < 0.15e-3:
            notes.append(
                f"{self.identity} is {self.width_m * 1e3:.2f} mm wide, below "
                "what ordinary board houses etch reliably")
        return notes

    def graph_attributes(self) -> dict:
        return {
            "part_role": "pcb-trace",
            "length_m": float(self.length_m),
            "width_m": float(self.width_m),
            "copper_weight_oz": float(self.copper_weight_oz),
            "layer": self.layer,
            "resistance_ohm": float(self.resistance_ohm),
            "inductance_h": float(self.inductance_h),
            "ampacity_a": float(self.ampacity_a),
        }


@dataclass(frozen=True)
class WireRun:
    """A run of ``dc_power.Conductor`` given two endpoints.

    The conductor already knows its gauge, its length, its resistance
    and its out-and-back behaviour; this adds where it goes and what it
    does to a transient.
    """

    identity: str
    tail: str
    head: str
    conductor: Conductor

    @property
    def resistance_ohm(self) -> float:
        return self.conductor.resistance_ohm

    @property
    def ampacity_a(self) -> float:
        return self.conductor.ampacity_a

    @property
    def inductance_h(self) -> float:
        area = self.conductor.area_mm2 * 1e-6
        diameter = 2.0 * math.sqrt(area / math.pi)
        run = self.conductor.length_m * (
            2.0 if self.conductor.both_directions else 1.0)
        return _partial_self_inductance_h(run, diameter, diameter)

    def branches(self) -> tuple[Branch, ...]:
        interior = f"{self.identity}~"
        return (
            Branch(f"{self.identity}.r", self.tail, interior,
                   "resistor", self.resistance_ohm),
            Branch(f"{self.identity}.l", interior, self.head,
                   "inductor", self.inductance_h),
        )

    def graph_attributes(self) -> dict:
        return {
            "part_role": "wire-run",
            "awg": int(self.conductor.awg),
            "length_m": float(self.conductor.length_m),
            "both_directions": bool(self.conductor.both_directions),
            "resistance_ohm": float(self.resistance_ohm),
            "inductance_h": float(self.inductance_h),
            "ampacity_a": float(self.ampacity_a),
        }


__all__ = [
    "OUNCE_COPPER_M", "MIL_M", "PcbTrace", "WireRun",
]
