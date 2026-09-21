"""High-voltage rectification, and the half-wave doubler that gates a magnetron.

WHY A DOUBLER RATHER THAN A BIGGER TRANSFORMER.  A magnetron wants
several kilovolts and draws a few hundred milliamps.  Winding a
transformer for the full voltage costs turns, insulation and iron; a
doubler buys the second kilovolt with one capacitor and one diode, which
are cheap and small.  Every consumer oven is built this way, and the
choice has a consequence that shapes everything downstream.

THE CONSEQUENCE IS THAT THE OUTPUT IS PULSED.  A half-wave doubler
refills its capacitor once per mains cycle, and the magnetron conducts
only while the output exceeds its turn-on knee.  So the tube runs in a
burst once per cycle and is dark between bursts: the RF is a microwave
carrier gated at mains frequency.  That is why an oven's output is
described as a duty cycle, why food heats unevenly in TIME as well as
space, and why :meth:`HalfWaveDoubler.conduction_fraction` is the number
the field layer actually wants.

WHAT THIS IS NOT.  These are closed-form results about the UNLOADED
circuit: peak, doubled output, the voltage the diode stands off, the
energy the capacitor holds, and the conduction window.  Loaded
regulation is deliberately absent, because the obvious closed form for
it is wrong here -- the capacitor is a series pump rather than a
reservoir, and treating it as one predicts a voltage swing larger than
the peak it swings about.  That needs the cycle solved rather than
averaged, which belongs to the transient layer.  Switching transients,
diode recovery and the first cycles after switch-on are likewise not
modelled.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from transformers import IronCoreTransformer


@dataclass(frozen=True)
class HighVoltageDiode:
    """A rectifier stack, which is what a "high voltage diode" really is.

    No single silicon die stands ten kilovolts, so the part is a series
    string of dies in one body.  ``series_dies`` is why the forward drop
    is volts rather than the familiar 0.7: every die in the string drops
    its own.
    """

    identity: str
    peak_inverse_v: float
    series_dies: int = 10
    die_forward_drop_v: float = 0.9
    rated_average_a: float = 0.5

    @property
    def forward_drop_v(self) -> float:
        return self.series_dies * self.die_forward_drop_v

    def graph_attributes(self) -> dict:
        return {
            "part_role": "high-voltage-rectifier",
            "peak_inverse_v": float(self.peak_inverse_v),
            "series_dies": int(self.series_dies),
            "forward_drop_v": float(self.forward_drop_v),
            "rated_average_a": float(self.rated_average_a),
        }


@dataclass(frozen=True)
class HighVoltageCapacitor:
    """The doubler's reservoir, and the most dangerous part in the machine.

    It is rated in AC volts because that is how it is stressed -- it
    sits across a transformer winding and reverses every cycle -- while
    the energy it holds is a DC question.  A charged oven capacitor
    stores several joules at several kilovolts, which is why a bleeder
    resistor is fitted across it and why its absence is a defect rather
    than an economy.
    """

    identity: str
    capacitance_f: float
    rated_vac: float
    #: Across the terminals, to discharge the part when the oven is off.
    bleeder_resistance_ohm: float | None = 10.0e6

    def stored_energy_j(self, voltage_v: float) -> float:
        return 0.5 * self.capacitance_f * voltage_v ** 2

    def bleed_time_s(self, voltage_v: float, to_volts: float = 50.0) -> float | None:
        """How long until it is safe to touch, which is an RC decay.

        Returns ``None`` when no bleeder is fitted: the part then holds
        its charge until something else discharges it, and that
        something is usually a person.
        """
        if self.bleeder_resistance_ohm is None or voltage_v <= to_volts:
            return None
        tau = self.bleeder_resistance_ohm * self.capacitance_f
        return tau * math.log(voltage_v / to_volts)

    def graph_attributes(self) -> dict:
        return {
            "part_role": "high-voltage-capacitor",
            "capacitance_f": float(self.capacitance_f),
            "rated_vac": float(self.rated_vac),
            "bleeder_resistance_ohm": (
                None if self.bleeder_resistance_ohm is None
                else float(self.bleeder_resistance_ohm)),
            "stored_energy_hazard": True,
        }


@dataclass(frozen=True)
class HalfWaveDoubler:
    """Transformer, capacitor and diode, feeding a load with a turn-on knee.

    ``load_knee_v`` is the voltage below which the load draws nothing.
    A magnetron has a sharp one -- it is a diode with a magnetic field
    on it -- and that knee, against the doubled peak, is what sets the
    conduction angle and therefore the duty cycle.
    """

    identity: str
    transformer: IronCoreTransformer
    capacitor: HighVoltageCapacitor
    diode: HighVoltageDiode
    load_knee_v: float
    label: str = "half-wave voltage doubler"

    # ---- voltages ---------------------------------------------------

    @property
    def supply_frequency_hz(self) -> float:
        return float(self.transformer.service.frequency_hz)

    @property
    def secondary_peak_v(self) -> float:
        return self.transformer.open_circuit_secondary_v * math.sqrt(2.0)

    @property
    def open_circuit_output_v(self) -> float:
        """Twice the peak, less what the diode string drops getting there."""
        return 2.0 * self.secondary_peak_v - self.diode.forward_drop_v

    @property
    def diode_reverse_v(self) -> float:
        """What the diode stands when it is NOT conducting: twice peak.

        This is the constraint that makes the part a stack, and a
        doubler whose diode is rated for the transformer's peak rather
        than for twice it fails the first time it is switched on.
        """
        return 2.0 * self.secondary_peak_v

    # ---- what the load sees ----------------------------------------

    @property
    def capacitance(self) -> float:
        return self.capacitor.capacitance_f

    def charge_per_cycle_c(self, average_current_a: float) -> float:
        """Charge the load takes each mains cycle: ``I t``.

        This much is unambiguous.  Turning it into a voltage swing is
        NOT, and deliberately is not attempted here -- see
        :meth:`conduction_fraction`.
        """
        return average_current_a / self.supply_frequency_hz

    def conduction_fraction(self) -> float:
        """Fraction of the mains cycle the load spends above its knee.

        With the capacitor charged to about the secondary peak, the
        output swings between roughly zero and twice peak as the
        transformer's own EMF adds to it.  Conduction begins where that
        sum crosses the knee:

            V_pk (1 + sin theta) = V_knee

        and runs to its mirror about the crest.  Below the knee the load
        is an open circuit and no RF is produced at all.

        THIS IS THE UNLOADED WINDOW.  The obvious way to load it -- treat
        the capacitor as a reservoir draining at ``I t / C`` -- describes
        a different circuit.  Here the capacitor is a SERIES pump, and
        while the load conducts the transformer is driving through it, so
        the capacitor supplies only the difference rather than the whole
        cycle's charge.  The reservoir formula predicts a 5000 V swing on
        a 2965 V peak at an oven's ordinary current, which is impossible,
        and real ovens run this topology on about a microfarad.  Getting
        the loaded window right needs the cycle solved rather than
        averaged, which is the transient layer's job and not this
        object's.
        """
        peak = self.secondary_peak_v
        if peak <= 0.0:
            return 0.0
        threshold = (self.load_knee_v - peak) / peak
        if threshold >= 1.0:
            return 0.0            # never reaches the knee: the tube stays dark
        if threshold <= -1.0:
            return 1.0            # never drops below it
        start = math.asin(threshold)
        return (math.pi - 2.0 * start) / (2.0 * math.pi)

    def delivered_power_w(self, average_current_a: float) -> float:
        """Cycle-average power into the load.

        A magnetron holds roughly its knee voltage while conducting, so
        the instantaneous power is ``V_knee * I_conducting``.  The
        cycle average multiplies that by the conduction fraction -- and
        the current quoted here is already the cycle average, which is
        the conducting current times that same fraction.  The two
        cancel, leaving ``V_knee * I_average``: the duty cycle governs
        how the energy is DELIVERED IN TIME, not how much of it there
        is.
        """
        if self.conduction_fraction() <= 0.0:
            return 0.0
        return self.load_knee_v * average_current_a

    # ---- is this design admissible? --------------------------------

    def design_notes(self) -> list[str]:
        notes: list[str] = []
        if self.diode.peak_inverse_v < self.diode_reverse_v:
            notes.append(
                f"diode rated {self.diode.peak_inverse_v:.0f} V inverse against "
                f"{self.diode_reverse_v:.0f} V standing off: it will fail on "
                "the first non-conducting half cycle")
        elif self.diode.peak_inverse_v < 1.3 * self.diode_reverse_v:
            notes.append(
                "diode has under 30% inverse-voltage margin, which a mains "
                "transient will find")
        if self.capacitor.rated_vac < self.transformer.open_circuit_secondary_v:
            notes.append(
                f"capacitor rated {self.capacitor.rated_vac:.0f} VAC against a "
                f"{self.transformer.open_circuit_secondary_v:.0f} V winding")
        if self.capacitor.bleeder_resistance_ohm is None:
            notes.append(
                "no bleeder resistor: the capacitor holds "
                f"{self.capacitor.stored_energy_j(self.open_circuit_output_v):.1f} J "
                "at several kilovolts indefinitely once the supply is removed")
        if self.open_circuit_output_v <= self.load_knee_v:
            notes.append(
                f"doubled output {self.open_circuit_output_v:.0f} V never "
                f"reaches the load's {self.load_knee_v:.0f} V knee; the load "
                "will never conduct")
        return notes

    def graph_attributes(self) -> dict:
        return {
            "part_role": "half-wave-voltage-doubler",
            "supply_frequency_hz": self.supply_frequency_hz,
            "secondary_peak_v": float(self.secondary_peak_v),
            "open_circuit_output_v": float(self.open_circuit_output_v),
            "diode_reverse_v": float(self.diode_reverse_v),
            "load_knee_v": float(self.load_knee_v),
            "pulsed_at_supply_frequency": True,
        }


__all__ = [
    "HighVoltageDiode", "HighVoltageCapacitor", "HalfWaveDoubler",
]
