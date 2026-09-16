"""What the system graph's own Laplacian says about how it is joined.

This session cost several hours to a bug with a purely structural
signature: derived_torque computed torque offline, beside a simulator
that was already solving the same thing, and read nothing the simulator
produced. Nobody could see it, because nothing in the project looks at
the system AS A GRAPH and asks whether it is actually one system.

A graph Laplacian answers exactly that, and it is not a metaphor here --
the project already assembles one. block_dynamics.assemble_conductance_
matrix is a real conductance graph Laplacian (W/K) over the block,
solved through turing's compiled eigh. This points the same operator at
the drivetrain graph instead of the casting, because the questions a
Laplacian spectrum answers are the ones we keep failing to ask:

  HOW MANY PIECES IS THIS?  The multiplicity of the zero eigenvalue is
      the number of connected components, exactly. Whether that is a
      DEFECT depends on what the circuit is for, and this module does
      not know: a gearbox and a transfer case hold genuinely separate
      oil, so two pieces there is correct, while a pressurised oil
      gallery in two pieces is plumbing that does not meet. The spectrum
      reports the count; a reader who knows the circuit reads it.

  HOW WEAKLY IS IT HELD TOGETHER?  The second eigenvalue, the algebraic
      connectivity or Fiedler value, is small exactly when a graph is
      nearly two graphs. On a fluid circuit weighted by real hydraulic
      conductance that is a BOTTLENECK: the place the network is close
      to not being a network.

  WHERE WOULD IT SPLIT?  The sign pattern of the Fiedler vector names
      the two sides of that weakest cut. Not a guess about which part is
      restrictive -- the actual partition the geometry implies.

TWO WEIGHTINGS, BOTH HONEST, ANSWERING DIFFERENT QUESTIONS.

  "topology"    every edge weight 1. Pure connectivity. A component
                count from this is a fact about the plumbing and
                nothing else, so it cannot be argued away by changing a
                pipe size.

  "conductance" laminar conductance, which for a round pipe goes with
                the FOURTH POWER of radius (Hagen-Poiseuille). The graph
                edges already declare a real radius, so this is read off
                declared geometry rather than assigned. Here a small
                Fiedler value means a genuinely restrictive path, and a
                narrow pipe dominates the answer the way it dominates
                the real flow.

The eigensolve is turing's own compiled symmetric kernel, the same one
block_dynamics uses, not scipy -- for the same reason it gives there.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drivetrain_graph import build_drivetrain_graph

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))
from src.common.tensors.abstraction import AbstractTensor  # noqa: E402
from src.common.tensors import linalg  # noqa: E402

#: How close to the smallest structurally-nonzero eigenvalue a value
#: must be to count as numerically zero. Only used for REPORTING how
#: cleanly the spectrum separated; the component count itself does not
#: depend on it -- see `components` below for why it must not.
ZERO_EIGENVALUE_RELATIVE_TOLERANCE = 1e-9


@dataclass
class Spectrum:
    """One graph's Laplacian spectrum, and what it says."""
    label: str
    weighting: str
    nodes: int
    edges: int
    eigenvalues: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    identities: list = field(repr=False, default_factory=list)
    fiedler_vector: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    components: int = 0

    @property
    def zero_gap(self) -> float:
        """How cleanly the spectrum separated: the first nonzero
        eigenvalue over the largest. Small means the numerical zeros and
        the real eigenvalues are not well separated, which is a fact
        about conditioning worth seeing -- it is NOT how the component
        count is obtained."""
        c, ev = self.components, self.eigenvalues
        if ev.size <= c or float(ev[-1]) <= 0.0:
            return 0.0
        return float(ev[c]) / float(ev[-1])

    @property
    def algebraic_connectivity(self) -> float:
        """The Fiedler value. Zero exactly when the graph is
        disconnected; small when it is nearly so."""
        c = self.components
        if self.eigenvalues.size <= c:
            return 0.0
        return float(self.eigenvalues[c])

    def cut(self):
        """The two sides of the weakest cut, by Fiedler sign."""
        if self.fiedler_vector.size != len(self.identities):
            return list(self.identities), []
        neg = [i for i, v in zip(self.identities, self.fiedler_vector) if v < 0.0]
        pos = [i for i, v in zip(self.identities, self.fiedler_vector) if v >= 0.0]
        return neg, pos


def _edge_weight(edge: dict, weighting: str) -> float:
    if weighting == "topology":
        return 1.0
    if weighting == "conductance":
        # Hagen-Poiseuille: laminar conductance of a round pipe goes with
        # r^4. Edges with no declared radius are structural rather than
        # fluid and carry unit weight -- weighting them by a radius they
        # do not have would be inventing geometry.
        r = float(edge.get("radius") or 0.0)
        return r ** 4 if r > 0.0 else 1.0
    raise KeyError(f"unknown weighting {weighting!r}; declared: topology, conductance")


def connected_components(nodes: list, edges: list) -> int:
    """How many pieces, counted EXACTLY, by traversal.

    In exact arithmetic this equals the multiplicity of the Laplacian's
    zero eigenvalue, and reading it off the spectrum is the elegant
    answer. It is also the wrong one to ship, and the first
    conductance-weighted run proved it: edge weights go with the fourth
    power of radius, so a 3 mm pipe contributes about 8e-11, the whole
    spectrum sits near zero, and a relative threshold could not tell a
    structural zero from a small real eigenvalue. Circuits reported ZERO
    components -- impossible, a graph with nodes has at least one -- and
    the count DISAGREED between weightings.

    That disagreement is the proof. Connectivity is a topological fact:
    it cannot depend on how wide the pipes are. Any method whose answer
    moves when the weights change is measuring the weights. So the count
    comes from traversal, which is exact and integer, and the spectrum
    is used for the quantity it is actually good at -- how STRONGLY the
    pieces are held together, which does properly depend on the
    weights."""
    adj: dict = {}
    for e in edges:
        a, b = e.get("a"), e.get("b")
        if a is None or b is None or a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, count = set(), 0
    for n in nodes:
        if n in seen:
            continue
        count += 1
        stack = [n]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj.get(x, ()) - seen)
    return count


def laplacian(nodes: list, edges: list, weighting: str = "topology") -> np.ndarray:
    """The weighted graph Laplacian, assembled exactly as
    block_dynamics.assemble_conductance_matrix assembles the block's:
    each edge adds +w to both diagonal entries and -w to the
    off-diagonal pair. Same operator, different graph."""
    index = {ident: i for i, ident in enumerate(nodes)}
    n = len(nodes)
    L = np.zeros((n, n))
    for e in edges:
        a, b = index.get(e.get("a")), index.get(e.get("b"))
        if a is None or b is None or a == b:
            continue
        w = _edge_weight(e, weighting)
        L[a, a] += w
        L[b, b] += w
        L[a, b] -= w
        L[b, a] -= w
    return L


def spectrum(nodes: list, edges: list, label: str = "?",
             weighting: str = "topology", eigh_method: str = "auto") -> Spectrum:
    """Solve it. Symmetric by construction, so the ordinary symmetric
    eigenproblem is the right one -- there is no mass matrix here,
    because a Laplacian asks about connection, not about motion."""
    if len(nodes) == 0:
        return Spectrum(label=label, weighting=weighting, nodes=0, edges=0)
    L = laplacian(nodes, edges, weighting)
    L = 0.5 * (L + L.T)   # kill roundoff asymmetry, as block_dynamics does
    A = AbstractTensor.get_tensor(L.tolist())
    w, vecs = linalg.eigh(A, method=eigh_method)
    w = np.array(w.tolist(), dtype=np.float64)
    vecs = np.array(vecs.tolist(), dtype=np.float64)
    order = np.argsort(w)
    w, vecs = w[order], vecs[:, order]
    w = np.maximum(w, 0.0)   # a Laplacian is positive semidefinite; roundoff can dip
    c = connected_components(nodes, edges)
    s = Spectrum(label=label, weighting=weighting, nodes=len(nodes),
                 edges=len(edges), eigenvalues=w, identities=list(nodes),
                 components=c)
    if vecs.shape[1] > c:
        s.fiedler_vector = vecs[:, c]
    return s


def circuit_spectra(engine, weighting: str = "topology") -> list:
    """One spectrum per declared fluid circuit.

    Per circuit rather than whole-graph because a whole-graph component
    count answers nothing -- the graph was never meant to be one piece.
    Per circuit the count is at least ABOUT something nameable, though
    still not automatically a verdict: see the note in report()."""
    g = build_drivetrain_graph(engine)
    by_circuit: dict = {}
    for e in g["edges"]:
        cid = e.get("circuit_identity")
        if cid:
            by_circuit.setdefault(cid, []).append(e)
    out = []
    for cid, edges in sorted(by_circuit.items()):
        touched, seen = [], set()
        for e in edges:
            for k in ("a", "b"):
                v = e.get(k)
                if v is not None and v not in seen:
                    seen.add(v)
                    touched.append(v)
        out.append(spectrum(touched, edges, label=cid, weighting=weighting))
    return out


def report(engine, weighting: str = "topology") -> str:
    rows = circuit_spectra(engine, weighting)
    lines = [f"system spectrum for {getattr(engine, 'identity', '?')}  "
             f"[{weighting} weighting]",
             f"  {'circuit':20s} {'nodes':>6s} {'edges':>6s} {'pieces':>7s} "
             f"{'lambda2':>12s}   verdict"]
    for s in rows:
        # DESCRIBE, DO NOT JUDGE. An earlier version of this line said
        # "SPLIT: should be one network" for any component count above
        # one, and was immediately wrong about gear-oil: a transmission
        # and a transfer case have genuinely separate oil, so two pieces
        # there is correct. The spectrum reports a fact; whether that
        # fact is a defect depends on what the circuit is FOR, which
        # this module does not know and should not pretend to.
        verdict = ("one network" if s.components <= 1
                   else f"{s.components} independent networks -- check "
                        "whether that is intended")
        lines.append(f"  {s.label:20s} {s.nodes:6d} {s.edges:6d} "
                     f"{s.components:7d} {s.algebraic_connectivity:12.4e}   {verdict}")
    lines.append("")
    lines.append("  `pieces` counts independent networks. More than one is not "
                 "automatically wrong --")
    lines.append("  a gearbox and a transfer case really do hold separate oil. It "
                 "IS worth reading when")
    lines.append("  a circuit is meant to be one flow path: on this engine the oil "
                 "circuit reads as five")
    lines.append("  because four of them are crankcase/pan MATING FACES, sealing "
                 "interfaces tagged with the")
    lines.append("  oil circuit because they are oil-wetted, not because they carry "
                 "any flow.")
    lines.append("")
    lines.append("  lambda2 is algebraic connectivity: the smaller it is, the closer "
                 "the circuit is to")
    lines.append("  falling into two, and under conductance weighting that is a real "
                 "hydraulic bottleneck")
    lines.append("  rather than a topological curiosity.")
    return "\n".join(lines)


if __name__ == "__main__":
    import engines
    ident = sys.argv[1] if len(sys.argv) > 1 else "amc-258-jeep-i6"
    weighting = sys.argv[2] if len(sys.argv) > 2 else "topology"
    print(report(engines.get(ident), weighting))
