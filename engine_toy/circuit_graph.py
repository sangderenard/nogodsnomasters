"""Circuits as graphs, solved as matrices, and reduced so they need not be.

THE SOLVER IS NOT NEW.  Modified nodal analysis is the exterior calculus
this repository already has, wearing different units:

    KVL     v = d0 phi        branch voltages are node-potential
                              differences -- the gradient
    KCL     d0^T i = 0        current is conserved at every node --
                              the divergence
    Ohm     i = Y v
      =>    (d0^T Y d0) phi = i_source

``d0^T Y d0`` is the weighted graph Laplacian, which is ``laplace0`` with
the branch ADMITTANCE standing in for the Hodge star ``*1``.  The same
operator gives the seven-point stencil on a lattice and the cotangent
Laplacian on a mesh; here it gives the nodal admittance matrix.  One
matrix abstraction, three textbooks.

WHY THE REDUCTION MATTERS MORE THAN THE SOLVE.  A part's internal
circuit is interesting once -- while it is being designed -- and a
liability afterwards, because nothing wants to factor a network every
tick to find out what a two-terminal component does.  Eliminating the
internal nodes and keeping only the declared ports is the Schur
complement of the nodal matrix, known in power systems as KRON
REDUCTION:

    Y_port = Y_pp - Y_pi Y_ii^-1 Y_ip

and for a linear network it is EXACT, not an approximation.  A
seven-branch ladder reduces to a two-by-two that reproduces its
end-to-end impedance to floating-point.  So a component is solved in
full when it is built and carried afterwards as a small dense port
matrix, which is what lets small intricate circuits compose into large
ones without the cost compounding.

WHAT CANNOT BE REDUCED.  Kron reduction eliminates LINEAR nodes.  A
diode, a magnetron's turn-on knee, a saturating core -- these have to
stay.  The move is to reduce everything linear around them and leave a
small nonlinear core, which is usually two or three states rather than
the whole network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys

import numpy as np

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

from src.common.tensors.abstract_convolution.dec_system import (  # noqa: E402
    CellComplex, DECSystem,
)
from src.common.dt_system.dt_scaler import Metrics  # noqa: E402
from src.common.dt_system.engine_api import DtCompatibleEngine  # noqa: E402
from src.common.dt_system.error_channels import (  # noqa: E402
    DT_CHANNEL_NAMES, empty_channels,
)


TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Branch:
    """One two-terminal component between two named nodes.

    ``kind`` is declared rather than guessed from the value's magnitude,
    for the same reason a part declares its role everywhere else here.
    """

    identity: str
    tail: str
    head: str
    kind: str                    # resistor | capacitor | inductor | conductance
    value: float

    def admittance(self, frequency_hz: float) -> complex:
        """``Y(f)`` for this component, in siemens.

        At DC a capacitor is an open circuit and an inductor a short,
        and both are singular in the nodal matrix if taken literally --
        so the caller is expected to keep a DC network resistive, and
        :meth:`CircuitGraph.admittance_matrix` says so when it is not.
        """
        omega = TWO_PI * float(frequency_hz)
        if self.kind == "resistor":
            if self.value <= 0.0:
                raise ValueError(f"{self.identity}: resistance must be positive")
            return complex(1.0 / self.value, 0.0)
        if self.kind == "conductance":
            return complex(self.value, 0.0)
        if self.kind == "capacitor":
            return complex(0.0, omega * self.value)
        if self.kind == "inductor":
            if omega == 0.0:
                return complex(float("inf"), 0.0)
            return complex(0.0, -1.0 / (omega * self.value))
        raise ValueError(f"{self.identity}: unknown component kind {self.kind!r}")


@dataclass(frozen=True)
class NodalStamp:
    """A block contributed straight into the nodal matrix at named nodes.

    A two-terminal branch enters through ``d0^T Y d0``, which is how a
    resistor or a capacitor says what it is.  Anything with more than
    two terminals cannot: a reduced subcircuit, a controlled source, a
    three-winding transformer relate several nodes at once and have no
    single branch to sit on.  Those contribute a dense block over the
    nodes they touch, which is what MNA calls a stamp, and it adds to
    the nodal matrix alongside the branch assembly rather than instead
    of it.

    This is what lets a circuit be solved AGAINST something else.  A
    subcircuit reduced to its ports is exactly such a block, so plugging
    one network into another is stamping and needs no special case --
    and it nests, because the result is a nodal matrix like any other.
    """

    identity: str
    nodes: tuple[str, ...]
    admittance: np.ndarray

    def __post_init__(self) -> None:
        block = np.asarray(self.admittance, dtype=np.complex128)
        if block.ndim != 2 or block.shape[0] != block.shape[1]:
            raise ValueError(f"{self.identity}: a stamp must be square")
        if block.shape[0] != len(self.nodes):
            raise ValueError(
                f"{self.identity}: {block.shape[0]}x{block.shape[0]} block "
                f"over {len(self.nodes)} nodes")
        object.__setattr__(self, "admittance", block)

    @property
    def is_floating(self) -> bool:
        """Whether the block has no internal reference of its own.

        A network reduced from a passive circuit has rows summing to
        zero -- shifting every terminal together drives no current -- so
        it floats and needs the enclosing circuit to supply a ground.
        """
        return bool(np.allclose(self.admittance.sum(axis=1), 0.0, atol=1e-9))

    def graph_attributes(self) -> dict:
        return {
            "part_role": "nodal-stamp",
            "terminals": list(self.nodes),
            "floating": self.is_floating,
        }


@dataclass(frozen=True)
class MutualInductance:
    """Two inductor branches sharing flux, by a coupling coefficient.

        v1 = L1 di1/dt + M di2/dt
        v2 = M  di1/dt + L2 di2/dt        with  M = k sqrt(L1 L2)

    WHAT THIS DOES TO THE MATRIX.  Every other element here contributes
    one diagonal entry to the branch admittance, which is why ``Y`` has
    so far been a vector.  Coupling relates two branches to each other,
    so the pair contributes a 2x2 BLOCK and ``Y`` stops being diagonal.
    That is not a departure from the exterior-calculus reading -- a
    non-diagonal ``*1`` is the Galerkin Hodge star, which is the ordinary
    object whenever the dual mesh is not orthogonal.  It does mean the
    branch admittance has to be built as a matrix rather than scaled as
    a vector.

    ``|k| <= 1`` is not a convention.  The stored energy of the pair is
    ``(L1 i1^2 + 2 M i1 i2 + L2 i2^2) / 2``, and that form is positive
    semi-definite exactly when ``M^2 <= L1 L2``.  A larger ``k`` would
    let the pair store negative energy, so it is refused.
    """

    identity: str
    primary: str
    secondary: str
    coupling: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.coupling <= 1.0:
            raise ValueError(
                f"{self.identity}: coupling {self.coupling} is outside [-1, 1]; "
                "beyond unity the pair would store negative energy")

    def mutual_h(self, primary_h: float, secondary_h: float) -> float:
        return self.coupling * math.sqrt(primary_h * secondary_h)

    def turns_ratio(self, primary_h: float, secondary_h: float) -> float:
        """``sqrt(L1/L2)``: inductance goes as turns squared."""
        return math.sqrt(primary_h / secondary_h)

    def leakage_h(self, primary_h: float) -> float:
        """``L1 (1 - k^2)`` referred to the primary.

        What the secondary cannot see, and therefore what limits current
        when the secondary is shorted.
        """
        return primary_h * (1.0 - self.coupling ** 2)

    def graph_attributes(self) -> dict:
        return {
            "part_role": "mutual-inductance",
            "coupling": float(self.coupling),
            "coupled_branches": [self.primary, self.secondary],
        }


@dataclass(frozen=True)
class CurrentSource:
    """An independent current source, optionally with its own leakage.

    The native element of nodal analysis: it contributes directly to the
    right-hand side and needs no conversion.  Current is taken to flow
    INSIDE the source from ``tail`` to ``head``, so it arrives at
    ``head`` and departs ``tail`` -- the same convention a battery's
    terminals follow.
    """

    identity: str
    tail: str
    head: str
    current_a: complex = 0.0
    parallel_conductance_s: float = 0.0

    def branches(self) -> tuple[Branch, ...]:
        if self.parallel_conductance_s <= 0.0:
            return ()
        return (Branch(f"{self.identity}.g", self.tail, self.head,
                       "conductance", self.parallel_conductance_s),)

    def injections(self) -> dict[str, complex]:
        return {self.head: complex(self.current_a),
                self.tail: -complex(self.current_a)}

    def graph_attributes(self) -> dict:
        return {
            "part_role": "current-source",
            "current_a": complex(self.current_a).real,
            "parallel_conductance_s": float(self.parallel_conductance_s),
        }


@dataclass(frozen=True)
class VoltageSource:
    """An EMF behind a source impedance, which is what a real source is.

    WHY THE IMPEDANCE IS NOT OPTIONAL.  Nodal analysis solves for
    potentials given injected currents; an IDEAL voltage source states a
    potential and lets the current be whatever it must, which is not a
    statement the matrix can hold.  The usual workaround is to enlarge
    the system with a current unknown per source.  The other way is to
    notice that no real source has zero output impedance -- a battery
    has internal resistance, a winding has resistance and leakage, a
    supply has a regulator with finite gain -- and that with any
    impedance at all the Norton equivalent is EXACT:

        I = emf / R   in parallel with   G = 1 / R

    So this refuses a zero source impedance rather than pretending, and
    the refusal is the physically true statement: declare what the
    source cannot deliver.  ``CircuitGraph.solve(driven=...)`` remains
    available for a genuine boundary condition or a probe.
    """

    identity: str
    tail: str
    head: str
    emf_v: complex = 0.0
    internal_resistance_ohm: float = 0.0

    def __post_init__(self) -> None:
        if self.internal_resistance_ohm <= 0.0:
            raise ValueError(
                f"{self.identity}: a voltage source needs a positive internal "
                "resistance. An ideal source has no Norton equivalent and no "
                "physical counterpart; every real one sags under load, and "
                "how much it sags is the number being asked for here.")

    @property
    def norton_current_a(self) -> complex:
        return complex(self.emf_v) / self.internal_resistance_ohm

    @property
    def norton_conductance_s(self) -> float:
        return 1.0 / self.internal_resistance_ohm

    @property
    def short_circuit_current_a(self) -> complex:
        """What it delivers into a dead short: the Norton current."""
        return self.norton_current_a

    def terminal_voltage_v(self, delivered_current_a: complex) -> complex:
        """The EMF less what the internal resistance keeps."""
        return (complex(self.emf_v)
                - complex(delivered_current_a) * self.internal_resistance_ohm)

    def branches(self) -> tuple[Branch, ...]:
        return (Branch(f"{self.identity}.r", self.tail, self.head,
                       "resistor", self.internal_resistance_ohm),)

    def injections(self) -> dict[str, complex]:
        return {self.head: self.norton_current_a,
                self.tail: -self.norton_current_a}

    def graph_attributes(self) -> dict:
        return {
            "part_role": "voltage-source",
            "emf_v": complex(self.emf_v).real,
            "internal_resistance_ohm": float(self.internal_resistance_ohm),
            "short_circuit_current_a": abs(self.norton_current_a),
        }


def assemble(identity: str, elements, ports=(), label: str = "circuit"):
    """Build a circuit and its injections from a mixed list of elements.

    Anything with ``branches()`` contributes copper or components;
    anything with ``injections()`` contributes to the right-hand side.
    A plain :class:`Branch` is taken as itself.  Returns the graph and
    the injection map, because the two are solved together.
    """
    branches: list[Branch] = []
    injections: dict[str, complex] = {}
    couplings: list[MutualInductance] = []
    stamps: list[NodalStamp] = []
    for element in elements:
        if isinstance(element, Branch):
            branches.append(element)
            continue
        if isinstance(element, MutualInductance):
            couplings.append(element)
            continue
        if isinstance(element, NodalStamp):
            stamps.append(element)
            continue
        branches.extend(element.branches())
        for node, value in getattr(element, "injections", dict)().items():
            injections[node] = injections.get(node, 0.0) + value
    return (CircuitGraph(identity, tuple(branches), tuple(ports), label,
                        tuple(couplings), tuple(stamps)),
            injections)


@dataclass(frozen=True)
class PortNetwork:
    """A circuit reduced to its terminals: what runtime actually carries.

    ``admittance`` is the port-to-port matrix the full network is
    equivalent to at ``frequency_hz``.  It is exact for the linear
    network it came from, so nothing is lost by keeping this and
    discarding the interior.
    """

    identity: str
    ports: tuple[str, ...]
    frequency_hz: float
    admittance: np.ndarray
    eliminated_nodes: int

    @property
    def impedance(self) -> np.ndarray:
        return np.linalg.pinv(self.admittance)

    def stamp(self, nodes: tuple[str, ...] | None = None) -> "NodalStamp":
        """This reduced network as a block, ready to plug into another.

        ``nodes`` renames the terminals for the enclosing circuit; by
        default it keeps its own port names.  The result composes
        exactly, because the reduction it came from was exact.
        """
        names = tuple(nodes) if nodes is not None else self.ports
        if len(names) != len(self.ports):
            raise ValueError(
                f"{self.identity}: {len(names)} names for "
                f"{len(self.ports)} ports")
        return NodalStamp(f"{self.identity}/stamp", names, self.admittance)

    def two_terminal_impedance_ohm(self) -> complex:
        """The one number a two-port reduction is usually wanted for."""
        if len(self.ports) != 2:
            raise ValueError(
                f"{self.identity} has {len(self.ports)} ports, not two")
        return complex(1.0 / self.admittance[0, 0])


@dataclass
class CircuitGraph:
    """A named set of branches between named nodes, with declared ports."""

    identity: str
    branches: tuple[Branch, ...]
    ports: tuple[str, ...] = ()
    label: str = "circuit"
    #: Inductor pairs that share flux; see :class:`MutualInductance`.
    couplings: tuple["MutualInductance", ...] = ()
    #: Multi-terminal blocks; see :class:`NodalStamp`.
    stamps: tuple["NodalStamp", ...] = ()
    _nodes: tuple[str, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        seen: list[str] = []
        for branch in self.branches:
            for node in (branch.tail, branch.head):
                if node not in seen:
                    seen.append(node)
        # A stamp's terminals are nodes of this circuit even when no
        # two-terminal branch touches them.
        for stamp in self.stamps:
            for node in stamp.nodes:
                if node not in seen:
                    seen.append(node)
        if not self.branches and not self.stamps:
            raise ValueError(
                f"{self.identity}: a circuit needs branches or stamps")
        object.__setattr__(self, "_nodes", tuple(seen))
        missing = [port for port in self.ports if port not in seen]
        if missing:
            raise ValueError(f"{self.identity}: ports not in the circuit: {missing}")

    @property
    def nodes(self) -> tuple[str, ...]:
        return self._nodes

    @property
    def node_index(self) -> dict[str, int]:
        return {name: position for position, name in enumerate(self._nodes)}

    def incidence(self) -> np.ndarray:
        """``d0``, shape (branches, nodes), built through the DEC complex.

        Routed through :class:`CellComplex` rather than written out here,
        so a circuit and a mesh genuinely share one incidence operator
        instead of two that happen to agree.  A circuit is a 1-complex:
        it has nodes and branches and no faces, which the complex
        accepts as an empty face map.
        """
        index = self.node_index
        if not self.branches:
            return np.zeros((0, len(self._nodes)), dtype=np.float64)
        edges = np.array([[index[branch.tail], index[branch.head]]
                          for branch in self.branches], dtype=np.int64)
        complex_ = CellComplex.from_rings(len(self._nodes), edges, {})
        system = DECSystem(
            complex_,
            star0=np.ones(len(self._nodes)),
            star1=np.ones(len(self.branches)),
            star2=np.zeros(0))
        return system.sparse("d0").toarray()

    def branch_admittance(self, frequency_hz: float) -> np.ndarray:
        """``Y``, shape (branches, branches).

        Diagonal for every element that relates a branch only to itself.
        A :class:`MutualInductance` fills a 2x2 block instead, because
        coupled inductors are described by an impedance MATRIX and its
        inverse is not diagonal.
        """
        values = np.array(
            [branch.admittance(frequency_hz) for branch in self.branches],
            dtype=np.complex128)
        coupled = {position for coupling in self.couplings
                   for position in self._coupling_positions(coupling)}
        if not np.isfinite(values[[p for p in range(len(values))
                                   if p not in coupled]]).all():
            raise ValueError(
                f"{self.identity}: a component is singular at "
                f"{frequency_hz} Hz -- an inductor at DC is a short and a "
                "capacitor an open, and neither belongs in a nodal matrix")

        admittance = np.zeros((len(values), len(values)), dtype=np.complex128)
        for position, value in enumerate(values):
            if position not in coupled:
                admittance[position, position] = value

        omega = 2.0 * math.pi * float(frequency_hz)
        for group in self._coupled_groups():
            for position in group:
                branch = self.branches[position]
                if branch.kind != "inductor":
                    raise ValueError(
                        f"{self.identity}: {branch.identity} is a "
                        f"{branch.kind!r}; only inductors carry mutual flux")
            if omega == 0.0:
                raise ValueError(
                    f"{self.identity}: coupled inductors are a short at "
                    "DC and have no nodal admittance there")
            row_of = {position: row for row, position in enumerate(group)}
            inductance = np.zeros((len(group), len(group)), dtype=float)
            for position in group:
                inductance[row_of[position], row_of[position]] = (
                    self.branches[position].value)
            for coupling in self.couplings:
                first, second = self._coupling_positions(coupling)
                if first not in row_of:
                    continue
                mutual = coupling.mutual_h(self.branches[first].value,
                                           self.branches[second].value)
                inductance[row_of[first], row_of[second]] = mutual
                inductance[row_of[second], row_of[first]] = mutual
            if float(np.linalg.eigvalsh(inductance).min()) <= 0.0:
                raise ValueError(
                    f"{self.identity}: coupled group "
                    f"{[self.branches[p].identity for p in group]} has a "
                    "non-positive-definite inductance matrix, so some "
                    "current pattern would store negative energy; the "
                    "couplings are mutually inconsistent even though each "
                    "one is inside [-1, 1] on its own")
            block = np.linalg.inv(1.0j * omega * inductance)
            admittance[np.ix_(group, group)] += block
        return admittance

    def _coupled_groups(self) -> list[list[int]]:
        """Mutually coupled branches, grouped so each group inverts ONCE.

        A branch may share flux with more than one other: a three-winding
        transformer, or a resonant cavity mode driven by two probes.  The
        whole group is described by a single impedance matrix ``j w L``,
        and its admittance is that matrix's inverse.

        INVERTING EACH PAIR SEPARATELY AND ADDING IS NOT THE SAME THING.
        Matrix inversion does not distribute over the pieces, and a
        branch appearing in two couplings would have its own inductance
        counted once per coupling.  The error is silent and grows with
        the coupling, which is why the grouping is done here rather than
        left to the caller to avoid.

        ``|k| <= 1`` per pair is necessary but no longer sufficient once
        a group exceeds two branches; positive definiteness of the whole
        matrix is the real condition and is checked above.
        """
        parent: dict[int, int] = {}

        def find(position: int) -> int:
            parent.setdefault(position, position)
            while parent[position] != position:
                parent[position] = parent[parent[position]]
                position = parent[position]
            return position

        for coupling in self.couplings:
            first, second = self._coupling_positions(coupling)
            parent[find(first)] = find(second)
        groups: dict[int, list[int]] = {}
        for position in sorted(list(parent)):
            groups.setdefault(find(position), []).append(position)
        return [sorted(group) for group in groups.values()]

    def _coupling_positions(self, coupling: "MutualInductance"):
        index = {branch.identity: position
                 for position, branch in enumerate(self.branches)}
        for name in (coupling.primary, coupling.secondary):
            if name not in index:
                raise ValueError(
                    f"{coupling.identity}: no branch named {name!r}")
        return index[coupling.primary], index[coupling.secondary]

    def admittance_matrix(self, frequency_hz: float) -> np.ndarray:
        """``d0^T Y d0`` plus every stamp: the nodal admittance matrix.

        Two-terminal elements arrive through the incidence assembly and
        multi-terminal ones through their own block.  Both land in the
        same matrix, which is why a reduced subcircuit composes with
        ordinary components without either knowing about the other.
        """
        index = self.node_index
        total = np.zeros((len(self._nodes), len(self._nodes)),
                         dtype=np.complex128)
        if self.branches:
            incidence = self.incidence()
            total += (incidence.T
                      @ self.branch_admittance(frequency_hz) @ incidence)
        for stamp in self.stamps:
            for name in stamp.nodes:
                if name not in index:
                    raise ValueError(
                        f"{stamp.identity}: no node named {name!r}")
            positions = [index[name] for name in stamp.nodes]
            total[np.ix_(positions, positions)] += stamp.admittance
        return total

    def solve(self, frequency_hz: float, *, reference: str,
              driven: dict[str, complex] | None = None,
              injected: dict[str, complex] | None = None) -> dict[str, complex]:
        """Node potentials from held voltages and injected currents.

        ``d0^T Y d0 phi = i`` is the whole statement.  ``injected`` is
        that right-hand side: current arriving at a node from outside
        the passive network, which is what a source IS in nodal
        analysis.  ``driven`` pins a node's potential outright, which is
        an IDEAL source -- convenient for a probe or a boundary, but not
        a component, because nothing real holds a voltage against any
        current whatever.  :class:`VoltageSource` is the component.

        ``reference`` is the ground the rest is measured against.
        """
        driven = dict(driven or {})
        injected = dict(injected or {})
        index = self.node_index
        for name in (*driven, *injected, reference):
            if name not in index:
                raise ValueError(f"{self.identity}: no node named {name!r}")
        matrix = self.admittance_matrix(frequency_hz)

        current = np.zeros(len(self._nodes), dtype=np.complex128)
        for name, value in injected.items():
            current[index[name]] += complex(value)

        held = {index[reference]: complex(0.0)}
        held.update({index[name]: complex(value) for name, value in driven.items()})
        free = [position for position in range(len(self._nodes)) if position not in held]
        if not free:
            return {name: held[index[name]] for name in self._nodes}

        known = np.zeros(len(self._nodes), dtype=np.complex128)
        for position, value in held.items():
            known[position] = value
        rhs = current[free] - matrix[np.ix_(free, list(held))] @ np.array(
            [held[position] for position in held], dtype=np.complex128)
        solved = np.linalg.solve(matrix[np.ix_(free, free)], rhs)

        potentials = known.copy()
        potentials[free] = solved
        return {name: complex(potentials[position])
                for name, position in index.items()}

    def reduce(self, frequency_hz: float,
               ports: tuple[str, ...] | None = None) -> PortNetwork:
        """Kron reduction: eliminate the interior, keep the terminals.

        Exact for this linear network, so the returned object may be
        carried in place of the circuit for every later evaluation at
        this frequency.
        """
        ports = tuple(ports) if ports is not None else self.ports
        if len(ports) < 2:
            raise ValueError(
                f"{self.identity}: reduction needs at least two ports")
        index = self.node_index
        kept = [index[name] for name in ports]
        interior = [position for position in range(len(self._nodes))
                    if position not in kept]

        matrix = self.admittance_matrix(frequency_hz)
        if not interior:
            reduced = matrix[np.ix_(kept, kept)]
        else:
            inner = matrix[np.ix_(interior, interior)]
            reduced = (matrix[np.ix_(kept, kept)]
                       - matrix[np.ix_(kept, interior)]
                       @ np.linalg.solve(inner, matrix[np.ix_(interior, kept)]))
        return PortNetwork(
            identity=f"{self.identity}/reduced",
            ports=ports,
            frequency_hz=float(frequency_hz),
            admittance=reduced,
            eliminated_nodes=len(interior),
        )

    def graph_attributes(self) -> dict:
        return {
            "part_role": "circuit",
            "circuit_identity": self.identity,
            "node_count": len(self._nodes),
            "branch_count": len(self.branches),
            "ports": list(self.ports),
        }


@dataclass
class CompanionAssembly:
    """One timestep's worth of companion models, assembled and kept.

    Trapezoidal integration replaces each reactive component with a
    conductance in parallel with a history current source:

        capacitor   i_n = (2C/h) v_n  -  [ (2C/h) v_(n-1) + i_(n-1) ]
        inductor    i_n = (h/2L) v_n  +  [ i_(n-1) + (h/2L) v_(n-1) ]

    Both are ``i = G v + I_history``, so at a fixed ``h`` every ``G`` is
    constant and the nodal matrix is real and constant.  That is the
    point: it is assembled once per timestep SIZE, not once per step,
    and the history sources are all that move.
    """

    timestep_s: float
    conductance: np.ndarray
    incidence: np.ndarray
    matrix: np.ndarray

    @staticmethod
    def build(circuit: "CircuitGraph", timestep_s: float) -> "CompanionAssembly":
        step = float(timestep_s)
        if step <= 0.0:
            raise ValueError("a companion assembly needs a positive timestep")
        values = []
        for branch in circuit.branches:
            if branch.kind == "resistor":
                values.append(1.0 / branch.value)
            elif branch.kind == "conductance":
                values.append(branch.value)
            elif branch.kind == "capacitor":
                values.append(2.0 * branch.value / step)
            elif branch.kind == "inductor":
                values.append(step / (2.0 * branch.value))
            else:
                raise ValueError(
                    f"{branch.identity}: {branch.kind!r} has no companion model")
        conductance = np.asarray(values, dtype=np.float64)
        incidence = circuit.incidence()
        return CompanionAssembly(
            timestep_s=step,
            conductance=conductance,
            incidence=incidence,
            matrix=incidence.T @ (conductance[:, None] * incidence),
        )


class CircuitEngine(DtCompatibleEngine):
    """A circuit as a dt-system engine, not a loop of its own.

    The dt system owns the clock.  This publishes what it knows -- the
    energy it stores, the power it dissipates, and a ``dt_limit`` from
    its own fastest time constant -- and steps when it is told to.

    A CONSEQUENCE OF LETTING THE CONTROLLER OWN dt: the companion
    conductances depend on the timestep, so a changing ``dt`` means a
    changing nodal matrix.  Assemblies are therefore cached BY TIMESTEP
    rather than built once.  A controller holding dt steady -- which is
    what it does while the metrics are satisfied -- hits that cache every
    step; one that shrinks dt pays a single assembly at the new size and
    then hits the cache again.
    """

    def __init__(self, circuit: CircuitGraph, *, reference: str,
                 driven=None, state_table=None):
        if reference not in circuit.nodes:
            raise ValueError(f"{circuit.identity}: no node named {reference!r}")
        self.circuit = circuit
        self.reference = reference
        #: ``{node: volts}`` or ``{node: callable(world_time) -> volts}``.
        self.driven = dict(driven or {})
        self.state = {branch.identity: (0.0, 0.0) for branch in circuit.branches}
        self.potentials = {name: 0.0 for name in circuit.nodes}
        self.world_time = 0.0
        self.observer_time = 0.0
        self.last_metrics = None
        self._assemblies: dict[float, CompanionAssembly] = {}
        self._state_table = state_table

    # ---- what the dt system asks --------------------------------------

    def assembly_for(self, timestep_s: float) -> CompanionAssembly:
        key = float(timestep_s)
        found = self._assemblies.get(key)
        if found is None:
            found = CompanionAssembly.build(self.circuit, key)
            self._assemblies[key] = found
        return found

    @property
    def assembled_timesteps(self) -> int:
        """How many distinct dt values have needed an assembly."""
        return len(self._assemblies)

    def time_constants_s(self) -> list[float]:
        """Every RC and L/R the circuit can form against its own branches.

        Deliberately crude: the true modes need the full eigenproblem,
        and the fastest branch-local constant is a sound, cheap bound on
        what the integrator has to resolve.
        """
        resistance = [branch.value for branch in self.circuit.branches
                      if branch.kind == "resistor"]
        smallest = min(resistance) if resistance else None
        largest = max(resistance) if resistance else None
        constants: list[float] = []
        for branch in self.circuit.branches:
            if branch.kind == "capacitor" and smallest is not None:
                constants.append(smallest * branch.value)
            elif branch.kind == "inductor" and largest is not None:
                constants.append(branch.value / largest)
        return constants

    def preferred_dt(self):
        """A tenth of the fastest time constant, or nothing to say."""
        constants = [value for value in self.time_constants_s() if value > 0.0]
        return 0.1 * min(constants) if constants else None

    def causal_ceiling_dt(self):
        limit = self.preferred_dt()
        return math.inf if limit is None else float(limit)

    def get_state(self, state=None):
        return self.state if state is None else state

    def snapshot(self):
        return (dict(self.state), dict(self.potentials),
                float(self.world_time), float(self.observer_time),
                self.last_metrics)

    def restore(self, snapshot) -> None:
        (state, potentials, self.world_time, self.observer_time,
         self.last_metrics) = snapshot
        self.state = dict(state)
        self.potentials = dict(potentials)

    # ---- the step ------------------------------------------------------

    def _driven_now(self, when: float) -> dict[str, float]:
        return {name: float(value(when)) if callable(value) else float(value)
                for name, value in self.driven.items()}

    def history_current(self, assembly: CompanionAssembly, state) -> np.ndarray:
        step = assembly.timestep_s
        history = np.zeros(len(self.circuit.branches), dtype=np.float64)
        for position, branch in enumerate(self.circuit.branches):
            voltage, current = state[branch.identity]
            if branch.kind == "capacitor":
                history[position] = -(2.0 * branch.value / step) * voltage - current
            elif branch.kind == "inductor":
                history[position] = current + (step / (2.0 * branch.value)) * voltage
        return history

    def step(self, dt: float, state=None, state_table=None):
        step = float(dt)
        working = self.state if state is None else state
        assembly = self.assembly_for(step)
        index = self.circuit.node_index

        history = self.history_current(assembly, working)
        injection = -assembly.incidence.T @ history

        held = {index[self.reference]: 0.0}
        for name, value in self._driven_now(self.world_time + step).items():
            if name not in index:
                raise ValueError(f"{self.circuit.identity}: no node {name!r}")
            held[index[name]] = value
        free = [position for position in range(len(self.circuit.nodes))
                if position not in held]

        potentials = np.zeros(len(self.circuit.nodes), dtype=np.float64)
        for position, value in held.items():
            potentials[position] = value
        if free:
            rhs = (injection[free]
                   - assembly.matrix[np.ix_(free, list(held))]
                   @ np.array([held[p] for p in held], dtype=np.float64))
            potentials[free] = np.linalg.solve(
                assembly.matrix[np.ix_(free, free)], rhs)

        voltages = assembly.incidence @ potentials
        currents = assembly.conductance * voltages + history
        advanced = {branch.identity: (float(voltages[position]),
                                      float(currents[position]))
                    for position, branch in enumerate(self.circuit.branches)}

        stored = 0.0
        dissipated = 0.0
        for branch in self.circuit.branches:
            voltage, current = advanced[branch.identity]
            if branch.kind == "capacitor":
                stored += 0.5 * branch.value * voltage ** 2
            elif branch.kind == "inductor":
                stored += 0.5 * branch.value * current ** 2
            elif branch.kind == "resistor":
                dissipated += current ** 2 * branch.value
            elif branch.kind == "conductance":
                dissipated += voltage ** 2 * branch.value

        self.state = advanced
        self.potentials = {name: float(potentials[position])
                           for name, position in index.items()}
        self.world_time += step
        if state_table is not None:
            self.publish(state_table)

        channels = empty_channels()
        present = empty_channels()
        channels[DT_CHANNEL_NAMES.index("energy_j")] = stored
        channels[DT_CHANNEL_NAMES.index("power_w")] = dissipated
        present[DT_CHANNEL_NAMES.index("energy_j")] = 1.0
        present[DT_CHANNEL_NAMES.index("power_w")] = 1.0
        limit = self.preferred_dt()
        metrics = Metrics(
            max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0,
            dt_limit=limit,
            error_channels=channels, error_present=present,
            advanced_dt=step,
        )
        self.last_metrics = metrics
        return True, metrics, advanced

    # ---- sharing one state system --------------------------------------

    def publish(self, state_table) -> None:
        """Put node potentials and branch currents where others can read.

        Two engines share a circuit by sharing the ``StateTable``, not by
        holding references to each other: this writes under the scope
        ``circuit`` keyed by the circuit's own identity, and whatever
        wants a node voltage reads it by name.
        """
        for name, value in self.potentials.items():
            state_table.set("circuit", self.circuit.identity, f"v:{name}", value)
        for identity, (_, current) in self.state.items():
            state_table.set("circuit", self.circuit.identity,
                            f"i:{identity}", current)


__all__ = [
    "Branch", "CircuitGraph", "PortNetwork", "CompanionAssembly",
    "CircuitEngine",
]


