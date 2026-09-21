"""A driven chamber with leaks, assembled from parts that already exist.

NOTHING NEW IS SOLVED HERE.  The cavity is a circuit, a penetration is a
two-port, an emitter is a Thevenin source and free space is a resistor.
All four were built separately and all four produce or consume the same
currency, so the chamber is an assembly, not another solver.  That is
the test of whether the unification was real: if joining them needed new
machinery, they had not actually been unified.

WHAT THE LEDGER IS FOR.  Power in has to equal copper warming plus power
radiated away, and those two are kept in separate columns because they
mean opposite things to somebody standing outside the box.  A model that
summed them could report a perfectly shielded chamber whose leak was
simply being counted as wall loss.  :class:`PowerLedger` closes the
books and says by how much.

CONVENTION.  Phasors are peak amplitudes, so a resistor dissipates
``|V|^2 / 2R``.  Frequencies in hertz, powers in watts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from cavities import CavityMode, LoopPort, RectangularCavity
from circuit_graph import Branch, CircuitGraph, assemble
from em_materials import EMMaterial
from applicators import ApplicatorState
from emitters import RFSource, RadiationLoad
from waveguides import CircularGuide


@dataclass(frozen=True)
class Penetration:
    """A declared leak: a pickup inside, a bore through the wall, a load out.

    This is the realistic leak and the one worth modelling -- a gland
    with a cable in it, a thermocouple, a pipe with a probe.  Something
    inside picks the field up, the bore carries it through as a guide
    below cutoff, and whatever is outside radiates it.

    A BARE hole coupling straight to the mode is a different object and
    is NOT this: its coupling comes from aperture polarizability rather
    than from a declared pickup, and that identification has not been
    derived here.  See :class:`waveguides.CircularAperture` for the
    aperture's own transmission.
    """

    identity: str
    pickup: LoopPort
    bore: CircularGuide
    termination: RadiationLoad

    @property
    def outside_node(self) -> str:
        return f"{self.identity}/outside"

    def graph_attributes(self) -> dict:
        return {
            "part_role": "chamber-penetration",
            "bore_radius_m": float(self.bore.radius_m),
            "wall_thickness_m": float(self.bore.length_m),
            "termination_ohm": float(self.termination.resistance_ohm),
        }


@dataclass(frozen=True)
class PowerLedger:
    """Where the watts went, in columns that mean different things."""

    delivered_w: float
    wall_loss_w: float
    radiated_w: float
    absorbed_w: float = 0.0

    @property
    def accounted_w(self) -> float:
        return self.wall_loss_w + self.radiated_w + self.absorbed_w

    @property
    def balance_error(self) -> float:
        """Fraction of delivered power the books fail to explain."""
        if self.delivered_w == 0.0:
            return 0.0
        return abs(self.delivered_w - self.accounted_w) / abs(self.delivered_w)

    @property
    def leak_fraction(self) -> float:
        return self.radiated_w / self.accounted_w if self.accounted_w else 0.0

    def shielding_db(self) -> float:
        """Delivered power over radiated power, in decibels.

        This is the chamber's figure of merit, and it is a RATIO OF
        POWERS the solve produced -- not the bore's attenuation quoted
        back.  The bore contributes most of it, but the coupling at both
        ends contributes too, and only the assembled circuit knows both.
        """
        if self.radiated_w <= 0.0:
            return math.inf
        return 10.0 * math.log10(self.delivered_w / self.radiated_w)


@dataclass(frozen=True)
class ShieldedChamber:
    """A cavity, an emitter that drives it, and the ways out."""

    identity: str
    cavity: RectangularCavity
    wall_material: EMMaterial
    drive_port: LoopPort
    source: RFSource
    penetrations: tuple[Penetration, ...] = ()
    modes: tuple[CavityMode, ...] | None = None
    loads: tuple = ()

    def carried_modes(self) -> tuple[CavityMode, ...]:
        return (self.modes if self.modes is not None
                else (self.cavity.dominant_mode,))

    def ports(self) -> tuple[LoopPort, ...]:
        return (self.drive_port,) + tuple(leak.pickup
                                          for leak in self.penetrations)

    def circuit(self, frequency_hz: float) -> tuple[CircuitGraph, dict]:
        """Every part's own contribution, assembled once.

        The cavity hands over its mode resonators and couplings, each
        penetration stamps its guide two-port between its pickup and its
        own outside node, and the emitter and the terminations arrive as
        ordinary elements.  ``assemble`` takes them all because they all
        speak the element protocol already.
        """
        modes = self.carried_modes()
        inner = self.cavity.circuit(self.wall_material, self.ports(), modes,
                                    self.loads)
        elements: list = list(inner.branches) + list(inner.couplings)
        elements.append(self.source)
        for leak in self.penetrations:
            network = leak.bore.port_network(frequency_hz)
            elements.append(network.stamp((leak.pickup.identity,
                                           leak.outside_node)))
            elements.append(leak.termination)
        terminals = (self.drive_port.identity, self.cavity.return_node())
        return assemble(f"{self.identity}/chamber", elements, terminals,
                        "shielded-chamber")

    def solve(self, frequency_hz: float) -> dict[str, complex]:
        graph, injections = self.circuit(frequency_hz)
        return graph.solve(frequency_hz,
                           reference=self.cavity.return_node(),
                           injected=injections)

    def power_ledger(self, frequency_hz: float) -> PowerLedger:
        """Close the books on one frequency.

        Delivered power is read at the source's own terminals; loss and
        radiation are read at the branches that represent them.  Nothing
        is inferred from the others by subtraction, which is what lets
        :attr:`PowerLedger.balance_error` mean anything.
        """
        graph, _ = self.circuit(frequency_hz)
        potentials = self.solve(frequency_hz)
        by_identity = {branch.identity: branch for branch in graph.branches}

        def across(branch: Branch) -> complex:
            return (potentials.get(branch.head, 0.0)
                    - potentials.get(branch.tail, 0.0))

        def dissipated(branch: Branch) -> float:
            return 0.5 * abs(across(branch)) ** 2 / branch.value

        wall = 0.0
        absorbed = 0.0
        for mode in self.carried_modes():
            wall += dissipated(
                by_identity[self.cavity.loss_branch_identity(mode)])
            for load in self.loads:
                absorbed += dissipated(by_identity[
                    self.cavity.load_branch_identity(mode, load.identity)])

        radiated = 0.0
        for leak in self.penetrations:
            radiated += dissipated(
                by_identity[f"{leak.termination.identity}/r"])

        terminal = (potentials.get(self.source.head, 0.0)
                    - potentials.get(self.source.tail, 0.0))
        delivered = 0.5 * (
            terminal * (self.source.as_voltage_source().norton_current_a
                        - self.source.as_voltage_source().norton_conductance_s
                        * terminal).conjugate()).real

        return PowerLedger(delivered_w=delivered, wall_loss_w=wall,
                           radiated_w=radiated, absorbed_w=absorbed)

    def mode_current_a(self, frequency_hz: float, mode: CavityMode) -> complex:
        """The current circulating in one mode's equivalent loop.

        Read off its wall resistor, where current is simply voltage over
        resistance.  This is the quantity that carries the circuit back
        into the field: the mode current fixes the field amplitude, and
        the field amplitude is what a load absorbs from.
        """
        graph, _ = self.circuit(frequency_hz)
        potentials = self.solve(frequency_hz)
        branch = {item.identity: item for item in graph.branches}[
            self.cavity.loss_branch_identity(mode)]
        across = (potentials.get(branch.head, 0.0)
                  - potentials.get(branch.tail, 0.0))
        return across / branch.value

    def applicator_state(self, frequency_hz: float) -> ApplicatorState:
        """The published state: what anything inside is actually sitting in.

        A thermal system takes the absorbed powers as source terms, a
        fluid system takes a sublimation rate from them, an interlock
        takes the field.  None of them needs to know there was a circuit.
        """
        mode = self.carried_modes()[0]
        current = self.mode_current_a(frequency_hz, mode)
        ledger = self.power_ledger(frequency_hz)
        return ApplicatorState(
            frequency_hz=float(frequency_hz),
            peak_field_v_m=self.cavity.peak_electric_field_v_m(mode, current),
            mode_current_a=abs(current),
            absorbed_w={load.identity: load.absorbed_w(self.cavity, mode,
                                                       current)
                        for load in self.loads},
            wall_loss_w=ledger.wall_loss_w,
            radiated_w=ledger.radiated_w)

    def graph_attributes(self) -> dict:
        return {
            "part_role": "shielded-chamber",
            "cavity": self.cavity.identity,
            "drive_frequency_hz": float(self.source.frequency_hz),
            "penetrations": [leak.identity for leak in self.penetrations],
        }


__all__ = ["Penetration", "PowerLedger", "ShieldedChamber"]
