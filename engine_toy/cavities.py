"""Resonant cavities, reduced to the circuit their ports actually see.

WHAT THIS FINISHES.  :mod:`waveguides` gives a penetration as a
two-port, but its ports are nodes on a wall and the chamber interior is
not a node at all -- a volume has no single potential.  This module is
what makes "inside" mean something to a circuit: expand the interior in
its modes, give each mode the series LC it already is, couple the ports
to those modes by mutual inductance, and hand the whole thing to
:func:`circuit_graph.CircuitGraph.reduce`.  What comes back is a
``PortNetwork`` like any other.

MODAL EXPANSION IS KRON REDUCTION.  This is not an analogy.  Kron
reduction eliminates interior nodes and keeps terminals, exactly:

    Y_port = Y_pp - Y_pi Y_ii^-1 Y_ip

A modal expansion eliminates interior FIELD degrees of freedom and
keeps the same terminals.  The eliminated coordinates are mode
amplitudes rather than node potentials, and each contributes a rank-one
term to the port matrix -- which is what a sum over resonators is.  So
the cavity does not need a bespoke reduction: it needs to be written as
a circuit, and the reducer that already exists does the rest.

WHAT IS EXACT HERE AND WHAT IS NOT.

  * Resonant frequencies are exact.  ``f_mnp`` is the guide cutoff
    expression with one more index, and a rectangular box is one of the
    shapes whose spectrum is closed form.
  * The equivalent inductance of a mode and a port's coupling to it are
    exact, computed from the analytic TE101 field rather than declared.
    Their common normalisation is a gauge that cancels: only ``M^2/L``
    reaches the ports.
  * The unloaded Q is a PERTURBATION result.  It assumes the fields are
    those of the lossless cavity and computes wall loss from them, which
    holds when Q is large -- the usual case, and checkable.
  * Truncating the expansion to a few modes is the one deliberate
    simplification, and it is the caller's to make: pass the modes that
    matter near the drive frequency.  Nothing is fitted and nothing is
    tabulated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from circuit_graph import Branch, CircuitGraph, MutualInductance, PortNetwork
from em_materials import EMMaterial, VACUUM_PERMEABILITY_H_M
from waveguides import SPEED_OF_LIGHT


@dataclass(frozen=True)
class CavityMode:
    """One standing wave of the box: three indices and a frequency."""

    family: str                 # TE | TM
    m: int
    n: int
    p: int
    frequency_hz: float

    @property
    def label(self) -> str:
        return f"{self.family}{self.m}{self.n}{self.p}"

    @property
    def is_te101(self) -> bool:
        return (self.family, self.m, self.n, self.p) == ("TE", 1, 0, 1)

    def graph_attributes(self) -> dict:
        return {"mode": self.label, "frequency_hz": float(self.frequency_hz)}


@dataclass(frozen=True)
class LoopPort:
    """A coupling loop: the ordinary way to get power into a cavity.

    A loop links magnetic flux, so it couples to a mode through the
    mode's H field at the loop, projected on the loop's normal.  A loop
    whose normal lies along a direction the mode has no field in couples
    to nothing, which is why orientation is declared rather than assumed.

    ``self_inductance_h`` is the loop's own inductance and is a property
    of the probe, not of the cavity, so the part declares it.  Declaring
    one too small to carry the flux it claims to share makes the coupled
    inductance matrix indefinite, and the circuit refuses it.
    """

    identity: str
    position_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    area_m2: float
    self_inductance_h: float

    def __post_init__(self) -> None:
        if self.area_m2 <= 0.0:
            raise ValueError(f"{self.identity}: a loop needs a positive area")
        if self.self_inductance_h <= 0.0:
            raise ValueError(
                f"{self.identity}: a loop needs a positive self inductance")
        length = math.sqrt(sum(component ** 2 for component in self.normal))
        if length <= 0.0:
            raise ValueError(f"{self.identity}: the normal has no direction")

    @property
    def unit_normal(self) -> tuple[float, float, float]:
        length = math.sqrt(sum(component ** 2 for component in self.normal))
        return tuple(component / length for component in self.normal)

    def graph_attributes(self) -> dict:
        return {
            "part_role": "cavity-loop-port",
            "area_m2": float(self.area_m2),
            "self_inductance_h": float(self.self_inductance_h),
        }


@dataclass(frozen=True)
class RectangularCavity:
    """A closed rectangular box, and the reference resonator.

    ``width_m`` runs along x, ``height_m`` along y, ``depth_m`` along z.
    TE and TM are referred to z, which is the convention that makes the
    box a shorted length of :class:`waveguides.RectangularGuide`.
    """

    identity: str
    width_m: float
    height_m: float
    depth_m: float
    relative_permittivity: float = 1.0

    def __post_init__(self) -> None:
        if min(self.width_m, self.height_m, self.depth_m) <= 0.0:
            raise ValueError(f"{self.identity}: a cavity needs positive sides")

    # ----------------------------------------------------------------
    # the spectrum, which is closed form
    # ----------------------------------------------------------------

    @property
    def phase_speed_m_s(self) -> float:
        return SPEED_OF_LIGHT / math.sqrt(self.relative_permittivity)

    @property
    def volume_m3(self) -> float:
        return self.width_m * self.height_m * self.depth_m

    @property
    def surface_area_m2(self) -> float:
        a, b, d = self.width_m, self.height_m, self.depth_m
        return 2.0 * (a * b + b * d + a * d)

    def mode_exists(self, family: str, m: int, n: int, p: int) -> bool:
        """The index rules, which are structural rather than incidental.

        A TE mode referred to z needs at least one half-wave along z, or
        there is no longitudinal H to be transverse-electric about; and
        it cannot be uniform in both transverse directions at once.  A TM
        mode needs a half-wave along BOTH transverse axes, because its
        longitudinal E must vanish on all four side walls.
        """
        if min(m, n, p) < 0:
            return False
        if family == "TE":
            return p >= 1 and (m > 0 or n > 0)
        if family == "TM":
            return m >= 1 and n >= 1
        raise ValueError(f"{self.identity}: no mode family {family!r}")

    def resonance_hz(self, family: str, m: int, n: int, p: int) -> float:
        if not self.mode_exists(family, m, n, p):
            raise ValueError(
                f"{self.identity}: {family}{m}{n}{p} does not exist in a "
                "rectangular cavity")
        return 0.5 * self.phase_speed_m_s * math.sqrt(
            (m / self.width_m) ** 2 + (n / self.height_m) ** 2
            + (p / self.depth_m) ** 2)

    def modes(self, max_index: int = 2) -> tuple[CavityMode, ...]:
        found = []
        for family in ("TE", "TM"):
            for m in range(max_index + 1):
                for n in range(max_index + 1):
                    for p in range(max_index + 1):
                        if self.mode_exists(family, m, n, p):
                            found.append(CavityMode(
                                family, m, n, p,
                                self.resonance_hz(family, m, n, p)))
        return tuple(sorted(found, key=lambda mode: mode.frequency_hz))

    @property
    def dominant_mode(self) -> CavityMode:
        return self.modes()[0]

    def wavenumber_1_m(self, mode: CavityMode) -> float:
        return 2.0 * math.pi * mode.frequency_hz / self.phase_speed_m_s

    # ----------------------------------------------------------------
    # loss, by the perturbation method
    # ----------------------------------------------------------------

    def unloaded_q(self, mode: CavityMode, material: EMMaterial) -> float:
        """Wall-loss Q for TE101, from the surface integral of the mode.

        ``Q = w U / P``, with ``U`` the stored energy and ``P`` the loss
        in the walls, both taken with the LOSSLESS fields.  That is the
        perturbation method: it presumes the loss does not reshape the
        mode, which holds precisely when the answer is large.  Check it
        by looking at the answer.

        The closed form works out to Pozar's

            Q = (k a d)^3 b eta / (2 pi^2 Rs [2a^3 b + 2b d^3 + a^3 d + a d^3])

        which this reproduces by the grouping that shows where each term
        comes from: one from the side walls, one from the end walls, one
        from the broad faces.
        """
        if not mode.is_te101:
            raise ValueError(
                f"{self.identity}: the closed-form Q here is TE101's; "
                f"{mode.label} needs its own surface integral -- see "
                "approximate_unloaded_q for an order-of-magnitude figure")
        surface = material.surface_resistance_ohm(mode.frequency_hz)
        if surface <= 0.0:
            return math.inf
        a, b, d = self.width_m, self.height_m, self.depth_m
        k = self.wavenumber_1_m(mode)
        # eta = sqrt(mu/eps), and v = 1/sqrt(mu eps), so eta = mu v exactly.
        impedance = VACUUM_PERMEABILITY_H_M * self.phase_speed_m_s
        loss = (b * d / a ** 2                 # the two x walls
                + a * b / d ** 2               # the two z walls
                + (a * a + d * d) / (2.0 * a * d))   # the two y faces
        return impedance * a * b * d * k ** 3 / (
            4.0 * math.pi ** 2 * surface * loss)

    def approximate_unloaded_q(self, mode: CavityMode,
                               material: EMMaterial) -> float:
        """``2V / (delta S)``: the estimate, and labelled as one.

        Every cavity's Q is the volume-to-surface ratio measured in skin
        depths, times a geometry factor of order unity.  This drops the
        factor, so it is right to within a small multiple and no more.
        Use :meth:`unloaded_q` wherever the exact integral is available.
        """
        depth = material.skin_depth_m(mode.frequency_hz)
        if depth == math.inf or depth <= 0.0:
            return math.inf
        return 2.0 * self.volume_m3 / (depth * self.surface_area_m2)

    # ----------------------------------------------------------------
    # the mode as a circuit, and the port's grip on it
    # ----------------------------------------------------------------

    def magnetic_shape_at(self, mode: CavityMode,
                          position_m: tuple[float, float, float]
                          ) -> tuple[float, float, float]:
        """The TE101 H pattern, to an arbitrary common scale.

        The scale is a gauge: it divides out of ``M^2 / L``, which is the
        only combination a port ever sees.  Keeping it arbitrary is what
        lets the inductance and the coupling be computed from the same
        expression without either needing a field amplitude.

        There is no y component -- TE101 is uniform along y -- so a loop
        whose normal is y couples to this mode not at all.
        """
        if not mode.is_te101:
            raise ValueError(
                f"{self.identity}: only TE101's pattern is written out here; "
                f"{mode.label} would need its own")
        x, y, z = (float(value) for value in position_m)
        if not (0.0 <= x <= self.width_m and 0.0 <= y <= self.height_m
                and 0.0 <= z <= self.depth_m):
            raise ValueError(
                f"{self.identity}: {position_m} is outside the cavity")
        along_x = math.sin(math.pi * x / self.width_m) * math.cos(
            math.pi * z / self.depth_m) / self.depth_m
        along_z = -math.cos(math.pi * x / self.width_m) * math.sin(
            math.pi * z / self.depth_m) / self.width_m
        return (along_x, 0.0, along_z)

    def electric_shape_at(self, mode: CavityMode,
                          position_m: tuple[float, float, float]
                          ) -> tuple[float, float, float]:
        """The TE101 E pattern, to unit peak: ``sin(pi x/a) sin(pi z/d)``.

        TE101 has one electric component and it points along y, so the
        field is strongest in the middle of the box and vanishes on
        every side wall -- which is where a load has to sit to absorb
        anything, and why an empty corner of an oven stays cold.
        """
        if not mode.is_te101:
            raise ValueError(
                f"{self.identity}: only TE101's pattern is written out here; "
                f"{mode.label} would need its own")
        x, y, z = (float(value) for value in position_m)
        if not (0.0 <= x <= self.width_m and 0.0 <= y <= self.height_m
                and 0.0 <= z <= self.depth_m):
            raise ValueError(
                f"{self.identity}: {position_m} is outside the cavity")
        return (0.0,
                math.sin(math.pi * x / self.width_m)
                * math.sin(math.pi * z / self.depth_m),
                0.0)

    def electric_overlap_m3(self, mode: CavityMode) -> float:
        """``integral |e|^2 dV`` over the whole box, which is ``V/4``.

        The denominator of every filling factor: what fraction of the
        mode's electric energy a load actually sits in.
        """
        if not mode.is_te101:
            raise ValueError(
                f"{self.identity}: {mode.label} has no written pattern")
        return self.volume_m3 / 4.0

    def peak_electric_field_v_m(self, mode: CavityMode,
                                mode_current_a: complex) -> float:
        """``|E0| = |i| w mu / pi`` -- the field the circuit implies.

        This is the bridge the whole heating story runs over.  The
        circuit solves for a mode current in the same gauge the mode
        inductance was computed in, so the field amplitude follows
        exactly rather than being an input.  Peak amplitude, matching
        the phasor convention used everywhere here.
        """
        if not mode.is_te101:
            raise ValueError(
                f"{self.identity}: {mode.label} has no written pattern")
        omega = 2.0 * math.pi * mode.frequency_hz
        return (abs(complex(mode_current_a)) * omega
                * VACUUM_PERMEABILITY_H_M / math.pi)

    def mode_inductance_h(self, mode: CavityMode) -> float:
        """``mu * integral |h|^2 dV`` in the same gauge as the shape above.

        This is the inductance of the equivalent series resonator, and it
        works out to ``mu a b d k^2 / (4 pi^2)``.
        """
        if not mode.is_te101:
            raise ValueError(
                f"{self.identity}: {mode.label} has no equivalent inductance "
                "here; only TE101's field is written out")
        k = self.wavenumber_1_m(mode)
        return (VACUUM_PERMEABILITY_H_M * self.volume_m3 * k ** 2
                / (4.0 * math.pi ** 2))

    def mode_capacitance_f(self, mode: CavityMode) -> float:
        omega = 2.0 * math.pi * mode.frequency_hz
        return 1.0 / (omega ** 2 * self.mode_inductance_h(mode))

    def mode_resistance_ohm(self, mode: CavityMode,
                            material: EMMaterial) -> float:
        """``w L / Q``: the series loss that gives the mode its width."""
        quality = self.unloaded_q(mode, material)
        if quality == math.inf:
            raise ValueError(
                f"{self.identity}: a lossless cavity has no series "
                "resistance and an infinitely narrow resonance; give the "
                "walls a real conductivity")
        omega = 2.0 * math.pi * mode.frequency_hz
        return omega * self.mode_inductance_h(mode) / quality

    def mutual_inductance_h(self, mode: CavityMode, port: LoopPort) -> float:
        """``mu * A * (h . n)``: flux the mode drives through the loop.

        Same gauge as :meth:`mode_inductance_h`, which is what makes the
        pair meaningful together.
        """
        shape = self.magnetic_shape_at(mode, port.position_m)
        normal = port.unit_normal
        projected = sum(s * n for s, n in zip(shape, normal))
        return VACUUM_PERMEABILITY_H_M * port.area_m2 * projected

    def coupling_coefficient(self, mode: CavityMode, port: LoopPort) -> float:
        """``k = M / sqrt(L_mode L_loop)``, which the circuit wants."""
        mutual = self.mutual_inductance_h(mode, port)
        return mutual / math.sqrt(self.mode_inductance_h(mode)
                                  * port.self_inductance_h)

    # ----------------------------------------------------------------
    # the whole thing as a circuit, and then as a port network
    # ----------------------------------------------------------------

    def return_node(self) -> str:
        return f"{self.identity}/return"

    def loss_branch_identity(self, mode: CavityMode) -> str:
        """The branch that IS this mode's wall loss.

        Named here rather than recovered by matching strings later, so a
        power ledger can point at the copper directly instead of
        guessing which resistor was which.
        """
        return f"{self.identity}/{mode.label}/R"

    def load_branch_identity(self, mode: CavityMode, load: str) -> str:
        """The branch that IS this load's absorption, named at build time.

        Wall loss and absorbed power are both resistors carrying the same
        mode current, and telling them apart afterwards by inspecting
        values would be guesswork.  Each is named for what it is.
        """
        return f"{self.identity}/{mode.label}/load/{load}"

    def circuit(self, material: EMMaterial, ports: tuple[LoopPort, ...],
                modes: tuple[CavityMode, ...] | None = None,
                loads: tuple = ()) -> CircuitGraph:
        """Modes as series resonators, ports as loops, coupled by flux.

        Each mode is a closed L-C-R loop hung off the return node, and
        each port is an inductor from its own terminal to the same
        return.  Every port-mode pair gets a ``MutualInductance``, so one
        mode driven by two ports puts THREE branches in one coupled
        group -- which is the case that has to invert as a single matrix.
        """
        carried = modes if modes is not None else (self.dominant_mode,)
        if not carried:
            raise ValueError(f"{self.identity}: no modes to carry")
        if not ports:
            raise ValueError(f"{self.identity}: a cavity needs a port")
        ground = self.return_node()
        branches: list[Branch] = []
        couplings: list[MutualInductance] = []

        for mode in carried:
            stem = f"{self.identity}/{mode.label}"
            branches.append(Branch(f"{stem}/L", ground, f"{stem}/a",
                                   "inductor", self.mode_inductance_h(mode)))
            branches.append(Branch(f"{stem}/C", f"{stem}/a", f"{stem}/b",
                                   "capacitor", self.mode_capacitance_f(mode)))
            # Wall loss and every absorbing body sit in SERIES in the mode
            # loop: they all see the same mode current, and each keeps its
            # own resistor so the ledger can say which took what.
            series = [(f"{stem}/R", self.mode_resistance_ohm(mode, material))]
            series += [(self.load_branch_identity(mode, load.identity),
                        load.resistance_ohm(self, mode)) for load in loads]
            node = f"{stem}/b"
            for position, (name, value) in enumerate(series):
                head = (ground if position == len(series) - 1
                        else f"{stem}/s{position}")
                branches.append(Branch(name, node, head, "resistor", value))
                node = head

        for port in ports:
            branches.append(Branch(f"{port.identity}/L", port.identity, ground,
                                   "inductor", port.self_inductance_h))
            for mode in carried:
                stem = f"{self.identity}/{mode.label}"
                couplings.append(MutualInductance(
                    f"{port.identity}<->{mode.label}",
                    f"{port.identity}/L", f"{stem}/L",
                    self.coupling_coefficient(mode, port)))

        terminals = tuple(port.identity for port in ports) + (ground,)
        return CircuitGraph(f"{self.identity}/cavity", tuple(branches),
                            terminals, "cavity", tuple(couplings), ())

    def port_network(self, frequency_hz: float, material: EMMaterial,
                     ports: tuple[LoopPort, ...],
                     modes: tuple[CavityMode, ...] | None = None,
                     loads: tuple = ()) -> PortNetwork:
        """The bake: a cavity as a small dense matrix on its terminals.

        The interior -- every mode resonator, every internal node -- is
        eliminated by the reducer that already exists, and what survives
        is the same ``PortNetwork`` a waveguide run or any other
        subcircuit produces.  That is the whole point of routing the
        field problem through a circuit: one product type, one consumer.
        """
        graph = self.circuit(material, ports, modes, loads)
        return graph.reduce(frequency_hz)

    def graph_attributes(self) -> dict:
        return {
            "part_role": "rectangular-cavity",
            "width_m": float(self.width_m),
            "height_m": float(self.height_m),
            "depth_m": float(self.depth_m),
            "volume_m3": self.volume_m3,
            "dominant_mode": self.dominant_mode.label,
            "dominant_resonance_hz": self.dominant_mode.frequency_hz,
        }


__all__ = ["CavityMode", "LoopPort", "RectangularCavity"]
