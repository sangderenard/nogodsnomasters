"""How a body's wrench actually reaches the mounts, read off the graph.

THE INERTIA TENSOR SAYS WHERE THE MASS IS. It does not say how a moment
made at that mass gets to the feet, and those are different questions
with different answers. A rigid tensor assumes every body is welded to
every other one, so a pump's unbalance arrives at the frame undiminished
and at once. Real machines are not like that: a pump sits on isolators
and its force arrives attenuated above the isolator's own frequency; a
vessel sits on saddles and its moment arrives in full; and a body that
is connected to the rest of the machine by nothing but its suction hose
does not act on the frame at all -- it acts on the hose.

So this walks the graph from each massive body through the members that
carry LOAD -- not the ones that carry fluid, which `joints` already
marks `routed` -- to the mount ports the machine declares, and reports
for each body the path with the least compliance:

    rigid        every link on the way is welded or bolted: the moment
                 acts in full, at every speed
    isolated     at least one link is a real bushing with a real
                 stiffness: the body's force reaches the frame through a
                 transmissibility that depends on speed, and the
                 isolation frequency is the body's mass on that stiffness
    unsupported  no load-carrying path to any mount at all, which is a
                 body held by its plumbing and a finding, not a number

WHY THIS IS A WALK AND NOT A SOLVE. The full answer is a beam solve of
the whole machine, which is exactly the thing the mode table exists to
avoid running live. What a moment needs to know to be placed correctly
in a reduced model is far less: whether it gets through, and through
what stiffness. A series of springs has one stiffness, so the walk sums
compliance along the path and takes the least; that is a shortest-path
problem, and it is evaluated once.

WHAT `reaches` IS FOR. A moment needs a lever to act on the feet as a
couple. A body whose structure reaches one mount can push on it; a body
that reaches two or more can twist the frame between them, and the
distance between the mounts it reaches is the lever its moment has.
That is the "how the moments are able to act" part, and it is data
rather than a judgement: the mode table decides what to do with it.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from joints import constraint_of, member_constraint
from machine_package import MOUNT_PORT_ROLE

#: A welded link has no compliance to sum. Held as a number rather than
#: an infinity so a path that is all welds is exactly zero compliance,
#: not the difference of two infinities.
RIGID = 0.0


@dataclass(frozen=True)
class Link:
    """One load-carrying member, and what it lets through."""
    edge: str
    a: str
    b: str
    constraint: str
    compliance_m_per_n: float
    damping_ratio: float = 0.0

    @property
    def rigid(self) -> bool:
        return self.compliance_m_per_n == RIGID


@dataclass(frozen=True)
class WrenchPath:
    """The least-compliance route from a body to a mount."""
    body: str
    mount: str | None
    hops: tuple
    compliance_m_per_n: float
    #: every mount this body's structure reaches, however compliantly
    reaches: tuple
    #: the distance between the two furthest-apart mounts it reaches: the
    #: lever a moment from this body has on the frame
    couple_arm_m: float
    #: the smallest damping ratio of any compliant link on the way; the
    #: isolator's own, for an isolated body
    damping_ratio: float = 0.0

    @property
    def kind(self) -> str:
        if self.mount is None:
            return "unsupported"
        return "rigid" if self.compliance_m_per_n == RIGID else "isolated"

    @property
    def stiffness_n_per_m(self) -> float:
        if self.mount is None:
            return 0.0
        if self.compliance_m_per_n == RIGID:
            return math.inf
        return 1.0 / self.compliance_m_per_n

    def isolation_hz(self, mass_kg: float) -> float | None:
        """The frequency the body bounces on its own path stiffness.

        Only an isolated body has one. A rigid path has no frequency to
        speak of and an unsupported body has no path."""
        if self.kind != "isolated" or mass_kg <= 0.0:
            return None
        return math.sqrt(self.stiffness_n_per_m / mass_kg) / (2.0 * math.pi)

    def transmission(self, hz: float, mass_kg: float) -> float:
        """What fraction of a force made at the body reaches the mount.

        The single-degree-of-freedom transmissibility through a spring
        and a damper: unity well below the isolation frequency, a peak
        at it, and falling as one over the square of speed above it --
        which is the entire reason isolators are fitted. A rigid path is
        one at every speed; an unsupported body transmits nothing and
        that is reported separately as the finding it is."""
        if self.kind == "unsupported":
            return 0.0
        f0 = self.isolation_hz(mass_kg)
        if f0 is None or f0 <= 0.0:
            return 1.0
        r = hz / f0
        z = self.damping_ratio
        num = 1.0 + (2.0 * z * r) ** 2
        den = (1.0 - r * r) ** 2 + (2.0 * z * r) ** 2
        return math.sqrt(num / max(den, 1e-18))

    def describe(self) -> str:
        if self.mount is None:
            return f"{self.body}: UNSUPPORTED -- no load path to any mount"
        k = self.stiffness_n_per_m
        tail = ("welded through" if k == math.inf
                else f"{k / 1e6:.2f} MN/m along the path")
        return (f"{self.body}: {self.kind} -> {self.mount} in {len(self.hops) - 1} "
                f"hops, {tail}; reaches {len(self.reaches)} mount(s), "
                f"couple arm {self.couple_arm_m:.2f} m")


def _link_compliance(edge: dict) -> tuple:
    """A link's compliance and damping from what the edge declares.

    A bushed edge carries the bushing pack `turret_production` derived
    from a real polyurethane annulus; both ends carry one, and they are
    in series. Anything else that carries load is welded, bolted or
    pinned -- a pin is free in a rotation, not soft in a translation --
    and contributes nothing to the sum."""
    support = edge.get("line_support")
    if support:
        # a routed line that declared itself load-bearing: the pipe's own
        # cantilever stiffness with its bracket's help
        return 1.0 / float(support["linear_stiffness_n_per_m"]), float(
            support.get("damping_ratio", 0.02))
    seam = edge.get("seam")
    if seam:
        # a fastened seam: one stiffness for the whole seam, from its
        # fastener count, and the sheet's own small damping
        return 1.0 / float(seam["linear_stiffness_n_per_m"]), float(
            seam.get("damping_ratio", 0.03))
    packs = edge.get("joint_bushings") or {}
    if not packs:
        return RIGID, 0.0
    compliance = 0.0
    zeta = None
    for end in ("a", "b"):
        pack = packs.get(end)
        if not pack:
            continue
        k = float(pack["linear_stiffness_n_per_m"])
        c = float(pack.get("linear_damping_n_s_per_m", 0.0))
        compliance += 1.0 / k
        # the pack derives its damping from critical damping against an
        # effective mass; read the ratio back from the same two numbers
        eff = float(pack.get("effective_mass_kg", 0.0))
        if eff <= 0.0:
            loss = float(pack.get("material", {}).get("loss_factor", 0.0))
            z = loss / 2.0
        else:
            z = c / (2.0 * math.sqrt(k * eff))
        zeta = z if zeta is None else min(zeta, z)
    return compliance, (zeta or 0.0)


def structural_links(doc: dict) -> list:
    """Every edge that carries load, with its compliance.

    `routed` is the graph's own word for a line that carries fluid or
    signal and no load -- declared on the edge by joints.member_constraint
    when it was authored -- so a hose is never mistaken for a bracket
    here, and the check is a flag rather than a spelling. A routed line
    that declared itself `load_bearing` carries a `line_support` pack --
    its own pipe stiffness plus its bracket's -- and IS a link, because
    a rigid drain does hold up its trap, with a little help."""
    links = []
    for e in doc["edges"]:
        token = e.get("constraint_token")
        mc = (constraint_of(token) if token is not None and token >= 0
              else member_constraint(e["constraint"]))
        if (mc.routed or e.get("routed")) and not e.get("line_support"):
            continue
        comp, zeta = _link_compliance(e)
        links.append(Link(edge=e["identity"], a=e["a"], b=e["b"],
                          constraint=e["constraint"],
                          compliance_m_per_n=comp, damping_ratio=zeta))
    return links


def mount_nodes(doc: dict) -> list:
    return [n["identity"] for n in doc["nodes"]
            if n.get("port_role") == MOUNT_PORT_ROLE]


def massive_bodies(doc: dict) -> list:
    return [n for n in doc["nodes"]
            if not n.get("wrench_point") and float(n.get("mass_kg", 0.0)) > 0.0
            and not n.get("chassis_side") and not n.get("supplied_elsewhere")]


def wrench_paths(doc: dict, *, mounts=None) -> dict:
    """The least-compliance path from every massive body to a mount.

    Dijkstra over compliance. Every welded hop costs nothing, so a body
    reached through forty welded shell members and one bushing has the
    bushing's compliance and nothing else -- which is the physics: a
    series of springs is as soft as its softest, and a weld is not a
    spring."""
    links = structural_links(doc)
    adjacency: dict = {}
    for l in links:
        adjacency.setdefault(l.a, []).append((l.b, l))
        adjacency.setdefault(l.b, []).append((l.a, l))
    targets = set(mounts if mounts is not None else mount_nodes(doc))
    position = {n["identity"]: np.asarray(n["reference_position"], float)
                for n in doc["nodes"]}

    out = {}
    for body in massive_bodies(doc):
        start = body["identity"]
        best = {start: (0.0, None, 0.0)}    # node -> (compliance, previous, worst zeta)
        heap = [(0.0, 0, start)]
        counter = 1
        reached = {}
        while heap:
            comp, _, node = heapq.heappop(heap)
            if comp > best[node][0]:
                continue
            if node in targets and node not in reached:
                reached[node] = comp
            for nxt, link in adjacency.get(node, ()):
                c = comp + link.compliance_m_per_n
                if nxt not in best or c < best[nxt][0]:
                    z_here = best[node][2]
                    if not link.rigid:
                        z_here = (link.damping_ratio if z_here == 0.0
                                  else min(z_here, link.damping_ratio))
                    best[nxt] = (c, node, z_here)
                    counter += 1
                    heapq.heappush(heap, (c, counter, nxt))
        if not reached:
            out[start] = WrenchPath(body=start, mount=None, hops=(start,),
                                    compliance_m_per_n=math.inf, reaches=(),
                                    couple_arm_m=0.0)
            continue
        mount = min(reached, key=lambda m: (reached[m], m))
        hops = []
        cur = mount
        while cur is not None:
            hops.append(cur)
            cur = best[cur][1]
        reaches = tuple(sorted(reached))
        arm = 0.0
        for i, a in enumerate(reaches):
            for b in reaches[i + 1:]:
                arm = max(arm, float(np.linalg.norm(position[a] - position[b])))
        out[start] = WrenchPath(body=start, mount=mount, hops=tuple(reversed(hops)),
                                compliance_m_per_n=reached[mount],
                                reaches=reaches, couple_arm_m=arm,
                                damping_ratio=best[mount][2])
    return out


def problems(paths: dict, doc: dict) -> list:
    """What the walk found that a tensor would have hidden."""
    mass = {n["identity"]: float(n.get("mass_kg", 0.0)) for n in doc["nodes"]}
    out = []
    for ident, p in sorted(paths.items()):
        if p.kind == "unsupported":
            out.append(f"{ident} ({mass[ident]:.0f} kg) has no load path to a "
                       "mount: it is held by whatever line reaches it")
    return out


def report(paths: dict, doc: dict) -> str:
    mass = {n["identity"]: float(n.get("mass_kg", 0.0)) for n in doc["nodes"]}
    lines = [f"{doc.get('identity', 'machine')}: how each body reaches the mounts"]
    for ident, p in sorted(paths.items(), key=lambda kv: -mass[kv[0]]):
        f0 = p.isolation_hz(mass[ident])
        tail = f"  isolates above {f0:.1f} Hz" if f0 else ""
        lines.append(f"    {mass[ident]:7.1f} kg  {p.kind:11s} "
                     f"{ident.split('.')[-1]:16s} reaches {len(p.reaches)} "
                     f"arm {p.couple_arm_m:.2f} m{tail}")
    for text in problems(paths, doc):
        lines.append(f"    PROBLEM: {text}")
    return "\n".join(lines)
