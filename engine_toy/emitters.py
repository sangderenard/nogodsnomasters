"""What drives a cavity, and what carries power away from one.

TWO KINDS OF RESISTANCE, AND THEY MUST NOT BE ADDED.  A cavity's mode
resistor represents copper warming up.  A radiation resistance
represents power leaving the building.  Both are ohms, both are real,
both take power out of the circuit -- and confusing them is how a model
comes to claim a chamber is safe because the energy "went somewhere".
So a radiating element is a distinct declared part, its power is counted
in its own column of the ledger, and nothing sums the two behind the
caller's back.

THE EMITTER IS A THEVENIN SOURCE, BECAUSE EVERY REAL ONE IS.  An ideal
source has no Norton equivalent and no physical counterpart;
:class:`circuit_graph.VoltageSource` already refuses one and says why.
An emitter here therefore declares an EMF and the impedance behind it,
and the useful number -- available power -- follows rather than being
asserted.

A MAGNETRON IS NOT A POWER SUPPLY'S FRIEND.  It conducts hardly at all
below its threshold and then clamps, so a stiff supply would deliver
whatever current the arc asked for.  That is why the transformer next
door has magnetic shunts and a filament winding: the supply does the
regulating.  :meth:`Magnetron.supply_notes` checks a declared
transformer actually meets what the tube declares it needs, rather than
either side assuming.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from circuit_graph import Branch, VoltageSource
from waveguides import FREE_SPACE_IMPEDANCE_OHM, SPEED_OF_LIGHT


@dataclass(frozen=True)
class RadiationLoad:
    """Free space at a port: a resistance whose power is gone, not warm.

    For a small loop of area ``A`` the radiation resistance is

        R = eta k^4 A^2 / (6 pi)

    which is the textbook ``31171 (A / lambda^2)^2`` written so the
    origin is visible: a magnetic dipole ``m = I A`` radiates
    ``eta k^4 |m|^2 / (12 pi)``, and ``R = 2P / |I|^2``.

    The fourth power of ``k`` is why small radiators are bad radiators,
    and it is the same fourth power that makes a small aperture leak so
    little -- see :class:`waveguides.CircularAperture`.  They are the
    same physics seen from the two sides.
    """

    identity: str
    tail: str
    head: str
    resistance_ohm: float
    area_m2: float | None = None

    def __post_init__(self) -> None:
        if self.resistance_ohm <= 0.0:
            raise ValueError(
                f"{self.identity}: a radiation load needs a positive "
                "resistance; a port that radiates nothing is an open circuit "
                "and should be declared as one")

    @staticmethod
    def small_loop_resistance_ohm(area_m2: float, frequency_hz: float) -> float:
        wavenumber = 2.0 * math.pi * float(frequency_hz) / SPEED_OF_LIGHT
        return (FREE_SPACE_IMPEDANCE_OHM * wavenumber ** 4 * area_m2 ** 2
                / (6.0 * math.pi))

    @classmethod
    def small_loop(cls, identity: str, tail: str, head: str,
                   area_m2: float, frequency_hz: float) -> "RadiationLoad":
        if area_m2 <= 0.0:
            raise ValueError(f"{identity}: a loop needs a positive area")
        return cls(identity, tail, head,
                   cls.small_loop_resistance_ohm(area_m2, frequency_hz),
                   area_m2)

    def is_small_at(self, frequency_hz: float,
                    max_size_parameter: float = 0.3) -> bool:
        """Whether the loop is still small against the wavelength.

        The closed form is the leading term of an expansion in ``k a``;
        past a fraction of a wavelength a loop stops being a dipole and
        the result stops meaning anything.  Declared as a predicate so a
        caller can ask.
        """
        if self.area_m2 is None:
            raise ValueError(
                f"{self.identity}: this load was declared as a resistance, "
                "not as a loop, so it has no size to check")
        radius = math.sqrt(self.area_m2 / math.pi)
        wavenumber = 2.0 * math.pi * float(frequency_hz) / SPEED_OF_LIGHT
        return wavenumber * radius < max_size_parameter

    def branches(self) -> tuple[Branch, ...]:
        return (Branch(f"{self.identity}/r", self.tail, self.head,
                       "resistor", self.resistance_ohm),)

    def injections(self) -> dict[str, complex]:
        return {}

    def graph_attributes(self) -> dict:
        return {
            "part_role": "radiation-load",
            "radiated": True,          # power leaves; it does not heat a wall
            "resistance_ohm": float(self.resistance_ohm),
        }


@dataclass(frozen=True)
class RFSource:
    """An emitter: an EMF, the impedance behind it, and a frequency.

    ``available_power_w`` is what this can hand to a conjugately matched
    load, ``|emf|^2 / (8 R)`` for peak-amplitude phasors.  It is a
    property of the source alone and is the honest way to state "a
    kilowatt emitter" without pretending the load does not matter.
    """

    identity: str
    tail: str
    head: str
    frequency_hz: float
    emf_v: complex = 0.0
    internal_resistance_ohm: float = 50.0

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError(f"{self.identity}: an emitter needs a frequency")
        if self.internal_resistance_ohm <= 0.0:
            raise ValueError(
                f"{self.identity}: an emitter needs a positive source "
                "impedance; see circuit_graph.VoltageSource for why")

    @property
    def available_power_w(self) -> float:
        return (abs(complex(self.emf_v)) ** 2
                / (8.0 * self.internal_resistance_ohm))

    @classmethod
    def from_available_power(cls, identity: str, tail: str, head: str,
                             frequency_hz: float, power_w: float,
                             internal_resistance_ohm: float = 50.0
                             ) -> "RFSource":
        """Declare the emitter by what it can deliver, which is how they
        are actually sold."""
        if power_w < 0.0:
            raise ValueError(f"{identity}: power cannot be negative")
        emf = math.sqrt(8.0 * internal_resistance_ohm * power_w)
        return cls(identity, tail, head, frequency_hz, complex(emf),
                   internal_resistance_ohm)

    def as_voltage_source(self) -> VoltageSource:
        return VoltageSource(f"{self.identity}/emf", self.tail, self.head,
                             self.emf_v, self.internal_resistance_ohm)

    def branches(self) -> tuple[Branch, ...]:
        return self.as_voltage_source().branches()

    def injections(self) -> dict[str, complex]:
        return self.as_voltage_source().injections()

    def graph_attributes(self) -> dict:
        return {
            "part_role": "rf-source",
            "frequency_hz": float(self.frequency_hz),
            "available_power_w": self.available_power_w,
            "internal_resistance_ohm": float(self.internal_resistance_ohm),
        }


@dataclass(frozen=True)
class Magnetron:
    """A cavity magnetron, declared by its operating point.

    The tube is treated as an emitter delivering ``rf_output_w`` into
    its design load.  What it is NOT is a linear amplifier: its anode
    characteristic is a clamp, so the anode voltage and current here are
    an operating point, not a curve, and the supply is what holds them
    there.

    The numbers relate rather than float free: DC input is anode volts
    times anode amps, RF out is that times the efficiency, and the
    difference is heat that has to leave through the anode fins.  A
    declared tube either balances or it does not.
    """

    identity: str
    frequency_hz: float = 2.45e9
    anode_voltage_v: float = 4000.0
    anode_current_a: float = 0.3
    efficiency: float = 0.7
    filament_voltage_v: float = 3.3
    filament_current_a: float = 10.0
    load_resistance_ohm: float = 50.0

    def __post_init__(self) -> None:
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError(
                f"{self.identity}: efficiency {self.efficiency} is not a "
                "fraction; a tube cannot emit more than it is fed")
        if min(self.anode_voltage_v, self.anode_current_a) <= 0.0:
            raise ValueError(
                f"{self.identity}: the anode needs a real operating point")

    @property
    def dc_input_w(self) -> float:
        return self.anode_voltage_v * self.anode_current_a

    @property
    def rf_output_w(self) -> float:
        return self.dc_input_w * self.efficiency

    @property
    def anode_heat_w(self) -> float:
        """What the fins have to carry; the reason ovens have a fan."""
        return self.dc_input_w - self.rf_output_w

    @property
    def filament_power_w(self) -> float:
        return self.filament_voltage_v * self.filament_current_a

    def as_source(self, tail: str, head: str) -> RFSource:
        """The tube as an emitter at its port."""
        return RFSource.from_available_power(
            f"{self.identity}/rf", tail, head, self.frequency_hz,
            self.rf_output_w, self.load_resistance_ohm)

    def supply_notes(self, transformer) -> tuple[str, ...]:
        """What a declared transformer fails to provide, one line each.

        Empty means the supply matches the tube.  This is checked rather
        than assumed because the two are declared in different modules by
        different people, and a filament winding that is close but wrong
        is the failure that shows up as a tube that never starts.
        """
        notes: list[str] = []
        secondary = transformer.open_circuit_secondary_v
        if secondary < self.anode_voltage_v:
            notes.append(
                f"secondary open-circuit {secondary:.0f} V is below the "
                f"{self.anode_voltage_v:.0f} V anode operating point")
        filament = transformer.filament_voltage_v
        if filament is None:
            notes.append("no filament winding; the cathode cannot be heated")
        elif abs(filament - self.filament_voltage_v) > 0.1 * self.filament_voltage_v:
            notes.append(
                f"filament winding {filament:.2f} V is more than ten percent "
                f"from the {self.filament_voltage_v:.2f} V the tube wants")
        if transformer.leakage_inductance_h is None:
            notes.append(
                "no declared leakage inductance; a magnetron needs the "
                "supply to limit its current, so an unshunted transformer "
                "is the wrong part")
        return tuple(notes)

    def graph_attributes(self) -> dict:
        return {
            "part_role": "magnetron",
            "frequency_hz": float(self.frequency_hz),
            "rf_output_w": self.rf_output_w,
            "anode_heat_w": self.anode_heat_w,
            "filament_power_w": self.filament_power_w,
        }


__all__ = ["RadiationLoad", "RFSource", "Magnetron"]
