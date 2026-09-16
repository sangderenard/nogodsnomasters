"""Routing a manifold: reach every port, cross nothing, arrive together.

Making a header is a geometry problem with three constraints that fight
each other, and a fabricator solves all three at once by eye. Written
down they are:

  REACH        every port has to be met square-on. A pipe that leaves
               the head at an angle has a step in it at the flange, and
               a step is a restriction exactly where the gas is fastest.

  CLEAR        no two pipes may occupy the same space. Obvious, and the
               hardest of the three, because the tubes that need to be
               longest are the ones with the least room to be long in.

  EQUAL        all pipes the same length. This is the one people think
               is cosmetic and it is not: a manifold is a set of tuned
               pipes, and pipes of different lengths tune to different
               engine speeds. An unequal header gives each cylinder a
               different torque curve, which the engine feels as
               roughness that no amount of balancing will fix.

THE TRICK IS THAT LENGTH IS FREE AND SPACE IS NOT. Every runner must
reach the length of the LONGEST one -- there is no shortening the far
cylinder, so the near ones must be made to take the long way round. That
is why headers look the way they do: the near-side pipes are the ones
with the theatrical loops in them, and they are not decoration, they are
the pipe buying length it cannot get by going straight.

HOW THIS ROUTES THEM. Each runner is a cubic Hermite curve from the port
(leaving along the port's own normal, so it meets the flange square) to
a slot on the collector (arriving along the collector axis, so the
tubes merge instead of colliding). Two handles control it: how far it
runs straight out of the port, and how far it bows sideways. Bowing adds
length without adding reach, which is exactly the freedom needed -- so
the short runners are bowed until they match the long one.

SLOTS ARE ASSIGNED BY ANGLE, AND THAT IS WHAT STOPS THE BRAID CROSSING.
If runners enter the collector in the same rotational order as their
ports sit along the head, no two need to swap sides, and pipes that
never swap sides never cross. Assigning slots any other way guarantees
at least one crossing and no amount of bowing fixes it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def hermite(p0, m0, p1, m1, t):
    """Cubic Hermite: position at t given endpoints and end tangents.

    Chosen over a plain Bezier because the TANGENTS are the meaningful
    quantity here -- a runner has to leave the port along the port's
    normal and arrive at the collector along the collector's axis, and
    Hermite takes those directly instead of making them be inferred from
    control points."""
    t = np.asarray(t, dtype=float)[:, None]
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return (h00 * np.asarray(p0) + h10 * np.asarray(m0)
            + h01 * np.asarray(p1) + h11 * np.asarray(m1))


def polyline_length(pts) -> float:
    d = np.diff(np.asarray(pts, dtype=float), axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))


@dataclass
class Port:
    """One cylinder's exit: where it is and which way it faces."""
    identity: str
    position: tuple
    normal: tuple
    cylinder: int = 0


@dataclass
class Runner:
    """One routed pipe."""
    port: Port
    points: np.ndarray
    slot: int
    length_m: float
    bow_m: float
    stub_m: float

    @property
    def identity(self) -> str:
        return f"{self.port.identity}.runner"


def collector_slots(centre, axis, n: int, radius_m: float,
                    reference, stagger_m: float = 0.0) -> list:
    """Where each pipe enters the collector.

    A ring of n points around the collector axis. `reference` fixes
    rotation so slot 0 is on a predictable side -- without it the ring
    is arbitrary and the angular ordering below has nothing to order
    against.

    STAGGERED, because a real collector is not a flat plate with holes
    in it. Bringing every tube into one plane crowds them worst exactly
    where they have least room, and on a four-cylinder -- where the ring
    is small and there is little angular spread to use -- that alone is
    enough to make an otherwise fine manifold unbuildable.
    Fabricators stagger the merge for the same reason, alternate tubes
    entering slightly further back so neighbours are never side by side
    at their tightest point."""
    a = _unit(axis)
    ref = np.asarray(reference, dtype=float) - np.asarray(centre, dtype=float)
    u = ref - a * float(np.dot(ref, a))
    if float(np.linalg.norm(u)) < 1e-9:
        helper = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(a, helper)
    u = _unit(u)
    w = np.cross(a, u)
    out = []
    for i in range(max(1, n)):
        th = 2.0 * math.pi * i / max(1, n)
        back = -a * (stagger_m * (i % 2))
        out.append(np.asarray(centre, dtype=float) + back
                   + (u * math.cos(th) + w * math.sin(th)) * radius_m)
    return out


def _angle_about(axis, origin, reference, p) -> float:
    a = _unit(axis)
    ref = np.asarray(reference, dtype=float) - np.asarray(origin, dtype=float)
    u = ref - a * float(np.dot(ref, a))
    if float(np.linalg.norm(u)) < 1e-9:
        helper = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(a, helper)
    u = _unit(u)
    w = np.cross(a, u)
    d = np.asarray(p, dtype=float) - np.asarray(origin, dtype=float)
    d = d - a * float(np.dot(d, a))
    return math.atan2(float(np.dot(d, w)), float(np.dot(d, u))) % (2.0 * math.pi)


def _curve(port: Port, slot, collector_axis, stub_m: float, bow_m: float,
           bow_dir, samples: int = 48) -> np.ndarray:
    """One runner's centreline for a given stub and bow."""
    p0 = np.asarray(port.position, dtype=float)
    p1 = np.asarray(slot, dtype=float)
    n0 = _unit(port.normal)
    a = _unit(collector_axis)
    span = float(np.linalg.norm(p1 - p0)) + 1e-6
    # tangent magnitudes scale with span so the curve stays smooth on
    # both a short and a long runner
    m0 = n0 * (span * 0.55 + stub_m * 3.0)
    m1 = a * (span * 0.55)
    t = np.linspace(0.0, 1.0, samples)
    pts = hermite(p0, m0, p1, m1, t)
    if bow_m > 0.0:
        # push the middle sideways: adds length, keeps both ends and both
        # end tangents exactly where they were
        s = np.sin(math.pi * t)[:, None]
        pts = pts + np.asarray(bow_dir, dtype=float)[None, :] * (bow_m * s)
    return pts


def _bow_direction(port: Port, slot, collector_axis) -> np.ndarray:
    """Which way a runner bows when it needs length.

    Sideways relative to its own run and to the collector axis, so the
    loop opens out into free space rather than into the engine or into
    the neighbouring pipe."""
    p0 = np.asarray(port.position, dtype=float)
    run = _unit(np.asarray(slot, dtype=float) - p0)
    d = np.cross(run, _unit(collector_axis))
    if float(np.linalg.norm(d)) < 1e-6:
        d = np.cross(run, np.array([0.0, 1.0, 0.0]))
    return _unit(d)


def _solve_bow(port, slot, axis, stub_m, bow_dir, target_m,
               samples: int = 48, iters: int = 40) -> tuple:
    """Find the bow that makes this runner exactly the target length.

    Length rises monotonically with bow, so a bisection is exact and
    cannot get stuck -- no need for anything cleverer."""
    lo, hi = 0.0, 0.05
    base = polyline_length(_curve(port, slot, axis, stub_m, 0.0, bow_dir, samples))
    if base >= target_m:
        return 0.0, base
    for _ in range(30):
        if polyline_length(_curve(port, slot, axis, stub_m, hi, bow_dir, samples)) >= target_m:
            break
        hi *= 1.8
        if hi > 5.0:
            break
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        L = polyline_length(_curve(port, slot, axis, stub_m, mid, bow_dir, samples))
        if L < target_m:
            lo = mid
        else:
            hi = mid
    bow = 0.5 * (lo + hi)
    return bow, polyline_length(_curve(port, slot, axis, stub_m, bow, bow_dir, samples))


def min_clearance(a: np.ndarray, b: np.ndarray) -> float:
    """Closest approach between two sampled centrelines."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(d.min())


@dataclass
class Manifold:
    runners: list = field(default_factory=list)
    collector: tuple = (0.0, 0.0, 0.0)
    collector_axis: tuple = (1.0, 0.0, 0.0)
    tube_diameter_m: float = 0.038
    target_length_m: float = 0.0
    #: Length added beyond the natural minimum to get the tubes apart.
    #: None means no amount within the search made it buildable.
    extra_length_m: float | None = 0.0

    @property
    def lengths(self) -> list:
        return [r.length_m for r in self.runners]

    @property
    def length_spread(self) -> float:
        L = self.lengths
        return (max(L) - min(L)) / max(1e-9, sum(L) / len(L)) if L else 0.0

    def worst_clearance(self) -> tuple:
        """Closest any two pipes come, and which pair. Below one tube
        diameter they are touching; below zero they are interpenetrating
        and the manifold cannot be built."""
        worst, pair = float("inf"), (None, None)
        for i in range(len(self.runners)):
            for j in range(i + 1, len(self.runners)):
                d = min_clearance(self.runners[i].points, self.runners[j].points)
                if d < worst:
                    worst, pair = d, (self.runners[i].identity, self.runners[j].identity)
        return worst, pair

    @property
    def buildable(self) -> bool:
        w, _ = self.worst_clearance()
        return w >= self.tube_diameter_m

    def report(self) -> dict:
        w, pair = self.worst_clearance()
        L = self.lengths
        return {"runners": len(self.runners),
                "target_m": self.target_length_m,
                "min_m": min(L) if L else 0.0, "max_m": max(L) if L else 0.0,
                "spread_frac": self.length_spread,
                "worst_clearance_m": w, "worst_pair": pair,
                "tube_diameter_m": self.tube_diameter_m,
                "buildable": self.buildable}


def route_buildable(ports, collector, collector_axis=(1.0, 0.0, 0.0),
                    tube_diameter_m: float = 0.038, max_extra_m: float = 0.45,
                    step_m: float = 0.03, **kw) -> Manifold:
    """Route it, and if the tubes will not clear, make them longer.

    THIS IS WHAT A FABRICATOR DOES. At the minimum length every runner
    is taking the most direct line it can, which is also the line most
    likely to be in another pipe's way -- a cramped header is genuinely
    hard to build and that is not a modelling artefact. The fix on a
    bench is to add length: give the tubes somewhere to go and they stop
    fighting for the same space.

    It is not free. Every millimetre added lowers the speed the manifold
    is tuned to, so `report()` still carries the final length and the
    caller can see what the clearance cost."""
    best = None
    extra = float(kw.pop("extra_length_m", 0.0))
    while extra <= max_extra_m:
        m = route(ports, collector, collector_axis=collector_axis,
                  tube_diameter_m=tube_diameter_m, extra_length_m=extra, **kw)
        w, _ = m.worst_clearance()
        if best is None or w > best[0]:
            best = (w, m)
        if m.buildable:
            m.extra_length_m = extra
            return m
        extra += step_m
    m = best[1]
    m.extra_length_m = None
    return m


def route(ports, collector, collector_axis=(1.0, 0.0, 0.0),
          tube_diameter_m: float = 0.038, collector_radius_m: float = 0.0,
          stub_m: float = 0.035, extra_length_m: float = 0.0,
          samples: int = 48, separation_passes: int = 24) -> Manifold:
    """Route an equal-length manifold from these ports into one collector.

    Three stages, in the order a fabricator would do them:

      1. decide who enters the collector where, by angle -- this is what
         prevents crossings and it is done before anything is bent
      2. find the longest natural run; that is the length everybody has
         to reach, because it is the only one that cannot be changed
      3. bow every other runner out until it matches, then push pipes
         apart until nothing touches

    `extra_length_m` lengthens every runner together, which is how a
    manifold gets tuned to a lower engine speed without changing where
    anything is bolted."""
    ports = list(ports)
    n = len(ports)
    if n == 0:
        return Manifold(collector=tuple(collector), collector_axis=tuple(collector_axis),
                        tube_diameter_m=tube_diameter_m)
    c = np.asarray(collector, dtype=float)
    a = _unit(collector_axis)
    if collector_radius_m <= 0.0:
        # tubes packed round a circle: the radius that just fits them
        collector_radius_m = (tube_diameter_m * 1.08) / (2.0 * math.sin(math.pi / max(2, n))) \
            if n > 1 else 0.0

    # ---- 1. slots by angle, so nothing has to cross anything ----
    ref = np.asarray(ports[0].position, dtype=float)
    order = sorted(range(n), key=lambda i: _angle_about(a, c, ref, ports[i].position))
    # Stagger tried and REJECTED: pulling alternate slots back along the
    # axis routes those tubes straight through the space the unstaggered
    # ones occupy, and measured clearance got worse, not better (a six
    # went from 38.9 mm to 26.9 mm). Real collectors do stagger, but the
    # tubes approach them along their own staggered axes rather than all
    # along one, which this router does not model. Left at zero, with
    # the reason, so it is not tried again.
    slots = collector_slots(c, a, n, collector_radius_m, ref, stagger_m=0.0)
    slot_of = {}
    for rank, i in enumerate(order):
        slot_of[i] = rank

    # ---- 2. the length everyone must reach ----
    #
    # FAN THE BOWS BEFORE SOLVING, rather than fixing collisions after.
    # Every runner bowed in the "natural" sideways direction bows into
    # the same region of space, so they all pile into each other and the
    # separation pass then has to dig them out one at a time -- which it
    # does badly, because by then each one is boxed in by its neighbours.
    #
    # A real header does not do this. Its tubes fan out around the
    # bundle, each taking the arc on its OWN side, and they are laid out
    # that way from the first bend. Seeding each runner's bow rotated to
    # match the slot it is heading for does the same thing: the runner
    # going to the top of the collector loops over the top, the one
    # going to the bottom loops under, and they were never in each
    # other's way to begin with.
    bows = []
    for i in range(n):
        base = _bow_direction(ports[i], slots[slot_of[i]], a)
        run_axis = _unit(np.asarray(slots[slot_of[i]], dtype=float)
                         - np.asarray(ports[i].position, dtype=float))
        bows.append(_rotate_about(base, run_axis,
                                  2.0 * math.pi * slot_of[i] / max(1, n)))
    natural = [polyline_length(_curve(ports[i], slots[slot_of[i]], a, stub_m, 0.0,
                                      bows[i], samples)) for i in range(n)]
    target = max(natural) + max(0.0, extra_length_m)

    # ---- 3. bow to match, then separate ----
    runners = []
    for i in range(n):
        bow, L = _solve_bow(ports[i], slots[slot_of[i]], a, stub_m, bows[i],
                            target, samples)
        pts = _curve(ports[i], slots[slot_of[i]], a, stub_m, bow, bows[i], samples)
        runners.append(Runner(port=ports[i], points=pts, slot=slot_of[i],
                              length_m=L, bow_m=bow, stub_m=stub_m))

    man = Manifold(runners=runners, collector=tuple(c), collector_axis=tuple(a),
                   tube_diameter_m=tube_diameter_m, target_length_m=target)

    # PUSHING PIPES APART WITHOUT CHANGING THEIR LENGTH is the whole
    # difficulty. Rotating a runner's bow direction about its own
    # port-to-slot axis moves the pipe through space while keeping the
    # bow magnitude -- and therefore the length -- exactly as solved.
    # So separation is a search over bow ANGLE, not over bow size, and
    # equality survives it.
    for _ in range(max(0, separation_passes)):
        w, _pair = man.worst_clearance()
        if w >= tube_diameter_m * 1.05:
            break
        moved = False
        for i in range(n):
            worst = min((min_clearance(runners[i].points, runners[j].points)
                         for j in range(n) if j != i), default=float("inf"))
            if worst >= tube_diameter_m * 1.05:
                continue
            run_axis = _unit(np.asarray(slots[slot_of[i]], dtype=float)
                             - np.asarray(ports[i].position, dtype=float))
            best = (worst, bows[i], runners[i].points, runners[i].length_m)
            for step in (0.25, -0.25, 0.5, -0.5, 0.9, -0.9, 1.4, -1.4, 2.1, -2.1):
                cand = _rotate_about(bows[i], run_axis, step)
                bow, L = _solve_bow(ports[i], slots[slot_of[i]], a, stub_m, cand,
                                    target, samples)
                pts = _curve(ports[i], slots[slot_of[i]], a, stub_m, bow, cand, samples)
                score = min((min_clearance(pts, runners[j].points)
                             for j in range(n) if j != i), default=float("inf"))
                if score > best[0]:
                    best = (score, cand, pts, L)
            if best[1] is not bows[i]:
                bows[i] = best[1]
                runners[i].points = best[2]
                runners[i].length_m = best[3]
                moved = True
        if not moved:
            break
    return man


def _rotate_about(v, axis, angle_rad: float) -> np.ndarray:
    """Rodrigues. Turning the bow around the runner's own axis moves the
    pipe without lengthening or shortening it."""
    v = np.asarray(v, dtype=float)
    k = _unit(axis)
    return (v * math.cos(angle_rad)
            + np.cross(k, v) * math.sin(angle_rad)
            + k * float(np.dot(k, v)) * (1.0 - math.cos(angle_rad)))


# ---------------------------------------------------------------------
# EQUAL-LENGTH AS A STYLE ON exhaust_header.HeaderPlan
# ---------------------------------------------------------------------
#
# exhaust_header.py already routes every port: it groups the cylinders
# the way real headers group them (4-into-1, a 6-into-2 tri-Y on an
# inline six, one collector per bank on a V, a log rail for a stock
# manifold, a ring for a radial), sites each collector below and
# outboard of the head, and bends the tube at a real mandrel radius
# taken from its own diameter.
#
# What it deliberately does not do is make the primaries the same
# length. Its docstring says so outright -- "lengths are reported so a
# tuned (equal-length) header can be judged against what this typical
# layout gives" -- which is a good design: the TYPICAL layout is what
# most engines actually have, and it is the thing an equal-length header
# has to be compared against to mean anything.
#
# So this is that comparison, and that upgrade. It takes a HeaderPlan
# that already exists and re-routes each group's primaries to a common
# length without moving the collector, changing the grouping or
# touching the tube size. Everything that made the original plan
# correct is kept; only the pipe between port and collector changes,
# which is exactly what a fabricator changes when they build a tuned set
# for an engine that came with a log.

def group_to_ports(group):
    """Read one CollectorGroup's primaries as router Ports.

    Each primary already starts at its port and leaves along the port's
    own direction -- the first segment of its polyline IS that
    direction, so the port normal does not need to be looked up again
    and cannot disagree with what was already routed."""
    ports = []
    for p in group.primaries:
        pts = [np.asarray(q, dtype=float) for q in p.points]
        if len(pts) < 2:
            continue
        d = pts[1] - pts[0]
        ports.append(Port(identity=getattr(p, "port_name", f"cyl{p.cylinder}"),
                          position=tuple(pts[0]), normal=tuple(_unit(d)),
                          cylinder=int(getattr(p, "cylinder", 0))))
    return ports


def equalise_group(group, tube_diameter_m: float | None = None,
                   extra_length_m: float = 0.0, match_existing: bool = True, **kw):
    """Re-route one collector group to equal primary lengths, in place.

    Returns a before/after report. Refuses nothing and changes nothing
    it does not have to: the collector stays where exhaust_header put
    it, the grouping is untouched, and the tube keeps its radius."""
    ports = group_to_ports(group)
    if len(ports) < 2:
        return {"cylinders": list(getattr(group, "cylinders", [])),
                "changed": False, "why": "a single primary is already equal to itself"}
    before = [p.length_m for p in group.primaries]
    dia = (tube_diameter_m if tube_diameter_m is not None
           else float(getattr(group.primaries[0], "radius_m", 0.019)) * 2.0)
    # EQUALISE UP, NOT DOWN. You cannot shorten the far cylinder -- its
    # pipe is as short as the distance allows -- so a fabricator brings
    # everything up to the LONGEST. Re-routing on this router's own
    # natural paths instead produced a set shorter than the original
    # shortest, which is a different header: it equalises the lengths
    # and moves the tuned speed at the same time, silently. Matching the
    # existing longest keeps whatever the plan was tuned for and changes
    # only the thing that was asked for.
    if match_existing and before:
        natural = max(polyline_length(_curve(
            pt, np.asarray(group.collector_position, dtype=float),
            _unit(np.asarray(group.outlet_direction, dtype=float)), 0.035, 0.0,
            _bow_direction(pt, np.asarray(group.collector_position, dtype=float),
                           _unit(np.asarray(group.outlet_direction, dtype=float)))))
            for pt in ports)
        extra_length_m = max(extra_length_m, max(before) - natural)
    man = route_buildable(ports, tuple(np.asarray(group.collector_position, dtype=float)),
                          collector_axis=tuple(_unit(np.asarray(group.outlet_direction,
                                                                dtype=float))),
                          tube_diameter_m=dia, extra_length_m=extra_length_m, **kw)
    by_cyl = {r.port.cylinder: r for r in man.runners}
    for prim in group.primaries:
        r = by_cyl.get(int(getattr(prim, "cylinder", 0)))
        if r is not None:
            prim.points = [np.asarray(q, dtype=float) for q in r.points]
    after = [p.length_m for p in group.primaries]
    group.style = f"{getattr(group, 'style', 'header')} (equal-length)"
    w, pair = man.worst_clearance()
    return {"cylinders": list(getattr(group, "cylinders", [])),
            "changed": True,
            "before_min_mm": min(before) * 1000.0, "before_max_mm": max(before) * 1000.0,
            "before_spread": (max(before) - min(before)) / max(1e-9, sum(before) / len(before)),
            "after_mm": sum(after) / len(after) * 1000.0,
            "after_spread": (max(after) - min(after)) / max(1e-9, sum(after) / len(after)),
            "added_mm": (man.extra_length_m or 0.0) * 1000.0,
            "buildable": man.buildable, "worst_clearance_mm": w * 1000.0,
            "worst_pair": pair, "tube_mm": dia * 1000.0}


def equalise_plan(plan, tube_diameter_m: float | None = None,
                  match_existing: bool = True, **kw) -> dict:
    """Do it to every group in a HeaderPlan.

    A log manifold's "primaries" are stubs into a rail and equalising
    them is meaningless -- there is no tuned pipe to tune. Those groups
    are left alone and said so, rather than being quietly bent into
    something a log manifold is not."""
    rows = []
    for g in getattr(plan, "groups", ()):
        style = str(getattr(g, "style", ""))
        if "log" in style.lower():
            rows.append({"cylinders": list(getattr(g, "cylinders", [])),
                         "changed": False,
                         "why": "a log manifold has no primaries to equalise -- each "
                                "port enters a shared rail by a short stub, and the "
                                "rail is the manifold. Making the stubs equal would "
                                "not make it a tuned header, it would make it a log "
                                "with odd-looking stubs"})
            continue
        rows.append(equalise_group(g, tube_diameter_m=tube_diameter_m,
                                   match_existing=match_existing, **kw))
    done = [r for r in rows if r.get("changed")]
    return {"groups": len(rows), "equalised": len(done), "rows": rows,
            "all_buildable": all(r.get("buildable", True) for r in done)}


# ---------------------------------------------------------------------
# TIERED MERGES: twin-port heads and tri-Y headers are the same problem
# ---------------------------------------------------------------------
#
# TWO PIPES PER CYLINDER is not a styling choice, it is what a four-valve
# head gives you if the casting keeps its two exhaust valves on separate
# ports instead of siamesing them into one. Most production four-valve
# heads siamese, because one port is cheaper to cast, easier to seal and
# needs half the manifold. A twin-port head does not, and every exhaust
# valve gets its own pipe.
#
# WHY ANYONE BOTHERS. Two smaller pipes have more perimeter than one
# bigger pipe of the same area, so the gas column is shorter and sharper
# and the pulse arrives at the merge more cleanly. It also lets the head
# keep the two ports short and straight instead of bending them together
# inside the casting, which is where a siamesed port loses most of its
# flow coefficient.
#
# AND THE CATCH IS SPECIFIC. Both pipes from one cylinder carry THE SAME
# EXHAUST EVENT. They fire together, always, so they cannot scavenge
# each other -- there is no quiet period on one to lend to the other,
# which is the entire mechanism a collector relies on. So a twin-port
# set has to merge its own pair FIRST, back into one pipe, before that
# pipe can usefully meet the other cylinders. The result is a two-tier
# header: eight pipes into four into one, on a four-cylinder.
#
# WHICH IS EXACTLY WHAT A TRI-Y IS. A 4-2-1 merges two cylinders that
# fire far apart into one pipe, then merges those. Same structure, same
# routing problem, different reason for the pairing -- so this is one
# function and the difference is only which ports get grouped together.
#
# PIPE SIZE HAS TO GROW AT EACH MERGE, and by area, not by diameter.
# Two 32 mm pipes carry the same gas as one 45 mm pipe, not one 64 mm
# pipe. Merging into something too big drops gas speed and kills the
# scavenging the merge was for; too small and the merge is a
# restriction. sqrt(n) is the whole rule.

def merged_diameter_m(single_diameter_m: float, count: int) -> float:
    """Pipe size after merging `count` pipes, conserving AREA.

    The rule fabricators state as "root n". Two into one is 1.41x the
    diameter, not 2x -- and getting this wrong is the commonest way a
    home-built collector ends up slower than the manifold it replaced."""
    return float(single_diameter_m) * math.sqrt(max(1, int(count)))


def tier_junction(ports, collector, fraction: float = 0.45):
    """Where a group's pipes should meet each other.

    Part of the way toward the collector, on the line from the group's
    own centroid. Near the ports the pipes have not separated enough to
    be worth merging; near the collector there is no room left to do it
    in, and the merge becomes the collector."""
    c = np.mean([np.asarray(p.position, dtype=float) for p in ports], axis=0)
    t = np.asarray(collector, dtype=float)
    return c + (t - c) * max(0.05, min(0.95, float(fraction)))


@dataclass
class TieredManifold:
    tiers: list = field(default_factory=list)      # list of Manifold
    collector: tuple = (0.0, 0.0, 0.0)
    final: "Manifold | None" = None

    @property
    def all_runners(self) -> list:
        out = []
        for m in self.tiers:
            out.extend(m.runners)
        if self.final is not None:
            out.extend(self.final.runners)
        return out

    def report(self) -> dict:
        rows = [m.report() for m in self.tiers]
        fin = self.final.report() if self.final is not None else None
        return {"tiers": len(self.tiers), "tier_reports": rows, "final": fin,
                "all_buildable": all(r["buildable"] for r in rows)
                                 and (fin is None or fin["buildable"]),
                "tier_spread": max((r["spread_frac"] for r in rows), default=0.0)}


def route_tiered(groups, collector, collector_axis=(1.0, 0.0, 0.0),
                 tube_diameter_m: float = 0.032, junction_fraction: float = 0.45,
                 **kw) -> TieredManifold:
    """Route a multi-tier merge: groups meet first, then meet each other.

    `groups` is a list of lists of Ports. Give it one list per cylinder
    of that cylinder's two exhaust ports and it builds a twin-port
    header; give it pairs of cylinders that fire far apart and it builds
    a tri-Y. The routing is identical -- only the grouping carries the
    reason, which is as it should be, because the geometry does not care
    why two pipes are being merged."""
    groups = [list(g) for g in groups if g]
    c = np.asarray(collector, dtype=float)
    out = TieredManifold(collector=tuple(c))
    # EQUALISE ACROSS TIERS, NOT JUST WITHIN THEM. Routing each group to
    # its own junction independently makes every pair internally equal
    # and leaves the pairs different from each other -- so cylinder 1's
    # gas travels 678 mm to the collector and cylinder 7's travels 589.
    # Those two cylinders fire at different times and DO have to scavenge
    # each other, so unequal totals defeat the whole purpose. The pairs
    # must all reach the same tier-1 length, which means finding the
    # longest natural one first and bringing the rest up to it.
    junctions = [None if len(g) == 1 else tier_junction(g, c, junction_fraction)
                 for g in groups]
    tier1_target = 0.0
    for g, j in zip(groups, junctions):
        if j is None:
            continue
        axis = _unit(c - j)
        for pt in g:
            bd = _bow_direction(pt, j, axis)
            tier1_target = max(tier1_target,
                               polyline_length(_curve(pt, j, axis, 0.035, 0.0, bd)))
    junction_ports = []
    for gi, g in enumerate(groups):
        if len(g) == 1:
            # nothing to merge: this pipe goes straight on to the final
            junction_ports.append(g[0])
            continue
        j = junctions[gi]
        axis = _unit(c - j)
        natural = max(polyline_length(_curve(pt, j, axis, 0.035, 0.0,
                                             _bow_direction(pt, j, axis)))
                      for pt in g)
        m = route_buildable(g, tuple(j), collector_axis=tuple(axis),
                            tube_diameter_m=tube_diameter_m,
                            extra_length_m=max(0.0, tier1_target - natural), **kw)
        out.tiers.append(m)
        merged = merged_diameter_m(tube_diameter_m, len(g))
        junction_ports.append(Port(identity=f"tier{gi}.merged",
                                   position=tuple(j), normal=tuple(axis),
                                   cylinder=g[0].cylinder))
        junction_ports[-1].merged_diameter_m = merged
    if len(junction_ports) > 1:
        big = merged_diameter_m(tube_diameter_m, max(1, len(groups[0])))
        out.final = route_buildable(junction_ports, tuple(c),
                                    collector_axis=collector_axis,
                                    tube_diameter_m=big, **kw)
    return out


def twin_port_groups(ports) -> list:
    """Group a port list into one group per cylinder.

    For a twin-port head: both of a cylinder's exhaust ports together,
    because they fire together and must be merged before they can be
    useful to anyone else."""
    by = {}
    for p in ports:
        by.setdefault(int(p.cylinder), []).append(p)
    return [by[k] for k in sorted(by)]


def tri_y_groups(ports, firing_angles_by_cylinder: dict) -> list:
    """Pair cylinders that fire FURTHEST apart, for a 4-2-1.

    The whole point of a tri-Y is that the two cylinders sharing a pipe
    should never be exhausting at the same time -- one wants to be mid-
    silence while the other is pulsing. Pairing by firing angle rather
    than by position is what makes it a tri-Y instead of two pipes that
    happen to meet."""
    cyls = sorted(firing_angles_by_cylinder)
    if len(cyls) < 2:
        return [[p] for p in ports]
    by = {}
    for p in ports:
        by.setdefault(int(p.cylinder), []).append(p)
    # MAXIMISE THE WORST PAIR, not the first one. Taking the best
    # partner for each cylinder in turn is greedy and it strands the
    # leftovers: on a bank firing at 0/270/450/540 it paired 0 with 270
    # and left 450 with 540 -- ninety degrees apart, which is the
    # opposite of what a tri-Y is for. The pairing that matters is the
    # one whose WORST pair is best, so it is searched rather than
    # walked. A bank has at most a handful of cylinders, so every
    # perfect matching can simply be enumerated.
    def sep(x, y):
        d = abs(firing_angles_by_cylinder[x] - firing_angles_by_cylinder[y])
        return min(d, 720.0 - d)

    def matchings(rest):
        if len(rest) < 2:
            yield ([], rest[0] if rest else None)
            return
        a = rest[0]
        for i in range(1, len(rest)):
            b = rest[i]
            tail = rest[1:i] + rest[i + 1:]
            for rem, odd in matchings(tail):
                yield ([(a, b)] + rem, odd)

    best_pairs, best_worst, best_odd = None, -1.0, None
    for pairing, odd in matchings(list(cyls)):
        if not pairing:
            continue
        worst = min(sep(x, y) for x, y in pairing)
        if worst > best_worst:
            best_pairs, best_worst, best_odd = pairing, worst, odd
    pairs = [by.get(x, []) + by.get(y, []) for x, y in (best_pairs or ())]
    if best_odd is not None:
        pairs.append(by.get(best_odd, []))
    return pairs
