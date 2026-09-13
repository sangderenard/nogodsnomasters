"""PORTS THAT ARE JOINED INSIDE THE METAL.

A port is a hole in a surface. Two ports on the same body are usually
joined by something -- a cored passage in a casting, a cross-drilling, a
gallery -- and until now the only way to say so was to run a LINE
between them, which is a lie in three directions at once:

    it draws a pipe on the outside of a part that has none
    it puts a member in the frame solve where there is only a hole
    and it says the fluid leaves the body and comes back, so anything
        reasoning about leaks, pressure or heat treats one casting as
        two vessels with plumbing between them

AN INTERNAL CHANNEL IS THE OPPOSITE OF A PIPE. It is metal that ISN'T
there. It has a bore, a route and a volume; it has no wall of its own
because the body around it is the wall; it cannot be disconnected,
inspected or replaced separately from the part it is cored into; and it
weakens the section it passes through rather than stiffening it.

So it is declared as a SET OF PORTS THAT SHARE A PASSAGE, on a named
body, and everything follows from that:

    the fluid system sees one node with N openings
    the renderer draws nothing
    the frame solve sees no member, and the body loses the cored mass
    a crack in the body is a crack in the channel
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class InternalChannel:
    """A cored passage joining ports on one body.

    `ports` are node identities that already exist on the body's
    surface. `route` is how they are joined:

        "gallery"   one passage, every port branching off it -- a
                    crankshaft oil gallery, a manifold rail
        "annular"   a groove all the way round, which is what a rotary
                    union has: every port opens into the same ring, so
                    the joint can turn without the passage caring
        "direct"    each port paired to the next, no common volume
    """
    identity: str
    body: str
    ports: tuple
    bore_m: float
    circuit: str
    route: str = "gallery"
    #: a groove's radial width, when the route is annular
    groove_width_m: float = 0.0
    pressure_pa: float = 0.0
    note: str = ""

    def length_m(self, positions: dict) -> float:
        """How far the fluid travels inside the metal."""
        pts = [np.asarray(positions[p], float) for p in self.ports
               if p in positions]
        if len(pts) < 2:
            return 0.0
        if self.route == "annular":
            centre = np.mean(pts, axis=0)
            r = float(np.mean([np.linalg.norm(p - centre) for p in pts]))
            return 2.0 * math.pi * r
        return float(sum(np.linalg.norm(pts[i + 1] - pts[i])
                         for i in range(len(pts) - 1)))

    def volume_m3(self, positions: dict) -> float:
        if self.route == "annular" and self.groove_width_m > 0.0:
            return self.length_m(positions) * self.groove_width_m * self.bore_m
        return self.length_m(positions) * math.pi * (self.bore_m / 2.0) ** 2

    def area_m2(self) -> float:
        if self.route == "annular" and self.groove_width_m > 0.0:
            return self.groove_width_m * self.bore_m
        return math.pi * (self.bore_m / 2.0) ** 2


def emit_channel(g, channel: InternalChannel, *, material: str = "steel-plate",
                 motion_group: str | None = None,
                 assembly: str | None = None) -> str:
    """Write the channel into a graph as ONE node with N openings.

    It is a node rather than a set of edges because that is what it is:
    a single volume the ports all open into. Edges between the ports
    would put N-1 pipes in the document and a pipe is the one thing this
    is not."""
    if motion_group is not None:
        g.motion_group = motion_group
    if assembly is not None:
        g.assembly = assembly
    pos = {n["identity"]: n["reference_position"] for n in g.nodes}
    present = [p for p in channel.ports if p in pos]
    if not present:
        raise KeyError(f"{channel.identity}: none of its ports exist yet")
    centre = np.mean([np.asarray(pos[p], float) for p in present], axis=0)
    length = channel.length_m(pos)
    volume = channel.volume_m3(pos)
    g.node(channel.identity, tuple(float(v) for v in centre), "manifold",
           material=material,
           # THE CHANNEL WEIGHS LESS THAN NOTHING. It is absence: the
           # body it is cored into should lose this metal, and saying so
           # here is what lets that happen exactly once.
           mass_in_total=False, mass_kg=0.01,
           part_role="internal-channel", internal_channel=True,
           cored_into=channel.body, route=channel.route,
           openings=list(present), opening_count=len(present),
           bore_m=channel.bore_m, groove_width_m=channel.groove_width_m,
           flow_area_m2=round(channel.area_m2(), 8),
           passage_length_m=round(length, 4),
           volume_l=round(volume * 1000.0, 4),
           displaced_kg=round(volume * 7850.0, 3),
           circuit_identity=channel.circuit,
           pressure_pa=channel.pressure_pa,
           in_view=False,
           note=channel.note or "cored, not plumbed: the body is the wall",
           half_extent_m=(channel.bore_m, channel.bore_m, channel.bore_m))
    for i, port in enumerate(present):
        g.edge(f"{channel.identity}.opening.{i}", channel.identity, port,
               "port-face-seal", radius=channel.bore_m / 2.0,
               circuit_identity=channel.circuit,
               palette="rollbar-silver", internal=True,
               load_path="a-hole-in-the-surface-opening-into-the-passage")
    return channel.identity


def channels_of(document: dict, body: str | None = None) -> list:
    """Every internal channel, optionally in one body."""
    return [n for n in document["nodes"]
            if n.get("internal_channel")
            and (body is None or n.get("cored_into") == body)]


def cored_mass_kg(document: dict, body: str) -> float:
    """How much metal this body lost to its own passages.

    The number a mass property is wrong by if channels are declared and
    never subtracted."""
    return sum(float(n.get("displaced_kg", 0.0))
               for n in channels_of(document, body))


def describe(document: dict) -> list:
    out = []
    for n in channels_of(document):
        out.append(
            f"  {n['identity']:34s} {n['route']:8s} "
            f"{n['opening_count']} openings  "
            f"{n['flow_area_m2'] * 1e6:7.0f} mm2  "
            f"{n['passage_length_m']:5.3f} m  "
            f"{n['volume_l']:6.3f} L  in {n['cored_into']}")
    return out
