"""What a machine is doing, as the thing its modes depend on.

A MACHINE DOES NOT HAVE ONE SET OF MODES. It has one set per thing it
can be doing: an autoclave with its vacuum pump running is a different
oscillator from the same autoclave holding at pressure with the pump
stopped and forty kilogrammes of condensate in its collector, and a
centrifuge at speed has a gyroscope in it that the same centrifuge at
rest does not. So the question "what are this machine's modes" has no
answer until "doing what" is attached to it, and this module is the
vocabulary for attaching it.

Two kinds of thing are declared, and they live in different places
because they are known by different people:

  A SOURCE is declared ON THE BODY that spins, in the graph, by whoever
  authored that body -- because the rotor's axis, speed, inertia and
  balance are facts about the part. Read off nodes by `sources_of`.

  A STATE is declared BY THE MACHINE, as part of its operating cycle --
  because which pump runs during which stage, and how full the
  collector is by then, are facts about how the machine is used. A
  cycle is an ordered set of states; each state says which sources run
  at what fraction of rated, what every container holds, and what CURVE
  to sample it along -- a run-up is a state sampled at rising fractions,
  because the damage is done crossing modes and a single point at rated
  speed cannot see a crossing.

BALANCE IS DECLARED, NEVER DEFAULTED. A rotor that has not said how well
it is balanced has not been built, and a silent G6.3 here would make
every centrifuge in the toy as good as a new one forever. A body either
states its residual unbalance directly or states its balance grade and
rotor mass and lets ISO 21940 turn that into one -- e = G / omega, U =
m.e -- which is the same arithmetic a balancing machine's report does.

WHY THE CYCLE CAN SAY WHETHER TO RUN LIVE. Everything here is evaluated
into a table once -- see mode_table -- and a deployment then either
reads that table per frame or runs the reduction live. That is a real
choice a real deployment makes, so it is a field on the cycle rather
than a global.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: The wildcard a source may declare in `runs_in`: it turns in every
#: state of the cycle. Spelled out so that an empty tuple means what it
#: says -- never runs -- rather than quietly meaning always.
ALWAYS = "*"


@dataclass(frozen=True)
class Source:
    """One body that spins, as declared by the body itself."""
    identity: str
    position: tuple
    axis: tuple
    rated_rpm: float
    inertia_kg_m2: float
    unbalance_kg_m: float
    mass_kg: float
    runs_in: tuple

    def runs(self, state_name: str) -> bool:
        return ALWAYS in self.runs_in or state_name in self.runs_in

    def rpm(self, fraction: float) -> float:
        return self.rated_rpm * float(fraction)

    def angular_momentum(self, fraction: float) -> np.ndarray:
        """H = I.omega along the axis. What makes a spinning body resist
        being tilted, and what couples one rocking axis to the other."""
        w = self.rpm(fraction) * 2.0 * math.pi / 60.0
        return np.asarray(self.axis, float) * (self.inertia_kg_m2 * w)

    def unbalance_force_n(self, fraction: float) -> float:
        w = self.rpm(fraction) * 2.0 * math.pi / 60.0
        return self.unbalance_kg_m * w * w


def unbalance_from_grade(mass_kg: float, grade_mm_s: float, rated_rpm: float
                         ) -> float:
    """ISO 21940 balance grade to residual unbalance in kg.m.

    The grade is a velocity: the permissible eccentricity times the
    service speed. So e = G / omega and the residual unbalance is the
    rotor mass at that eccentricity. G2.5 is a machine-tool spindle or
    a centrifuge drum; G6.3 an ordinary pump or fan; G16 a crankshaft."""
    w = rated_rpm * 2.0 * math.pi / 60.0
    if w <= 0.0:
        return 0.0
    e_m = (grade_mm_s / 1000.0) / w
    return float(mass_kg) * e_m


def source_of(node: dict) -> "Source | None":
    """Read a rotor declaration off one node, or None if it does not spin.

    Nothing is inferred from a kind or a name. A body spins because it
    declares `rotor_axis`, and having declared that it must also say how
    fast, how much inertia, and how well balanced -- each one a
    ValueError if missing, because each one changes the answer."""
    if "rotor_axis" not in node:
        return None
    ident = node["identity"]

    def need(key):
        if key not in node:
            raise ValueError(f"{ident} declares a rotor but not {key!r}")
        return node[key]

    axis = np.asarray(need("rotor_axis"), float)
    n = float(np.linalg.norm(axis))
    if n <= 1e-12:
        raise ValueError(f"{ident}: rotor_axis has no direction")
    axis = axis / n
    rated = float(need("rotor_rated_rpm"))
    if "rotor_inertia_kg_m2" in node:
        inertia = float(node["rotor_inertia_kg_m2"])
    else:
        from rotating_inertia import polar_inertia
        inertia = polar_inertia(need("rotor_shape"), need("rotor_mass_kg"),
                                need("rotor_radius_m"))
    if "rotor_unbalance_kg_m" in node:
        unbalance = float(node["rotor_unbalance_kg_m"])
    else:
        unbalance = unbalance_from_grade(float(need("rotor_mass_kg")),
                                         float(need("balance_grade_mm_s")),
                                         rated)
    runs_in = tuple(need("runs_in"))
    return Source(identity=ident,
                  position=tuple(float(v) for v in node["reference_position"]),
                  axis=tuple(float(v) for v in axis), rated_rpm=rated,
                  inertia_kg_m2=inertia, unbalance_kg_m=unbalance,
                  mass_kg=float(node.get("mass_kg", 0.0)), runs_in=runs_in)


def sources_of(doc: dict) -> list:
    out = []
    for n in doc["nodes"]:
        s = source_of(n)
        if s is not None:
            out.append(s)
    return out


@dataclass(frozen=True)
class OperatingState:
    """One thing the machine can be doing."""
    name: str
    #: fraction of rated speed per source, for the sources that run in
    #: this state; a source that runs and is not listed runs at rated
    speeds: dict = field(default_factory=dict)
    #: kilogrammes of contents per container node, in this state
    charges_kg: dict = field(default_factory=dict)
    #: residual unbalance overrides per source -- a bowl with a sludge
    #: cake in it is not the bowl the balancing machine saw
    unbalance_kg_m: dict = field(default_factory=dict)
    #: the fractions this state is sampled at along its own curve. A
    #: steady state is one point; a run-up or a coast is a sweep
    curve: tuple = (1.0,)
    note: str = ""

    def fraction_for(self, source: Source, sample: float) -> float:
        if not source.runs(self.name):
            return 0.0
        return float(self.speeds.get(source.identity, 1.0)) * float(sample)

    def unbalance_for(self, source: Source) -> float:
        return float(self.unbalance_kg_m.get(source.identity,
                                             source.unbalance_kg_m))


def run_up(points: int = 12, top: float = 1.0, start: float = 0.05) -> tuple:
    """A curve that climbs. Dense enough to land near a crossing.

    The ends are the declared ends exactly, not the sum that floating
    point makes of them: a table looked up at "rated" must find the
    row that was evaluated at rated."""
    n = max(points - 1, 1)
    out = [start + (top - start) * i / n for i in range(points)]
    out[0], out[-1] = float(start), float(top)
    return tuple(out)


def coast(points: int = 12, top: float = 1.0, stop: float = 0.05) -> tuple:
    return tuple(reversed(run_up(points, top, stop)))


@dataclass
class Cycle:
    """The states a machine passes through, in the order it does."""
    machine: str
    states: tuple
    #: Whether a deployment of this machine runs the reduction live or
    #: reads the baked table. The table is always produced; this is the
    #: election of what to do per frame.
    live_vibration: bool = False

    def state(self, name: str) -> OperatingState:
        for s in self.states:
            if s.name == name:
                return s
        raise KeyError(f"{self.machine}: no operating state {name!r}; "
                       f"declared: {', '.join(s.name for s in self.states)}")

    def samples(self):
        """Every (state, sample) the table is evaluated at, in order."""
        for s in self.states:
            for f in s.curve:
                yield s, float(f)

    def describe(self) -> str:
        lines = [f"{self.machine}: {len(self.states)} operating states, "
                 f"vibration {'live' if self.live_vibration else 'baked'}"]
        for s in self.states:
            lines.append(f"    {s.name:12s} {len(s.curve):3d} samples"
                         + (f"  {s.note}" if s.note else ""))
        return "\n".join(lines)
