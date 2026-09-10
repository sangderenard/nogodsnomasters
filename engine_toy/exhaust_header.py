"""Procedural exhaust header routing: each primary leaves its port a
short way along the port's own direction, turns, runs along a rail
parallel to the crank, and joins its GROUP's collector -- the layout
every real tubular header and every log manifold actually has, laid
out from the cylinder layout's exhaust ports (position + outward
direction + throat radius) rather than as a straight line from port
to one manifold point.

Grouping ("typical", by what the engine is):
  - one bank, 1-4 cylinders: all into one collector (4-into-1)
  - one bank, 5-8 cylinders: front half and rear half into two
    collectors (a 6-into-2 "tri-Y" on an inline six)
  - two or more banks: each bank its own collector(s) by the same rule
  - a "stock-manifold" (ExhaustSystem.header_type) is a LOG: a rail
    running the length of the bank with each port entering it by a
    short stub, one outlet at the log's rear
The collector sits below and outboard of the head at the rear (or
front) of its group, and hands off to the downpipe junction the
production graph already carries as "powertrain.exhaust_manifold".

Every path is a polyline of real points: stub, a quarter-bend of a
few segments (bend radius from the primary's own diameter, the way
mandrel-bent tube actually turns), the rail run, and a final bend
into the collector. Lengths are reported so a tuned (equal-length)
header can be judged against what this typical layout gives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

CRANK_AXIS = np.array([1.0, 0.0, 0.0])
BEND_SEGMENTS = 5
STUB_FRAC_OF_RADIUS = 1.6           # the "tiny bit" out of the head before the first turn
BEND_RADIUS_FRAC_OF_DIAMETER = 1.5  # mandrel-bend centreline radius, ~1.5 D


def _unit(v):
    a = np.array(v, dtype=np.float64)
    n = np.linalg.norm(a)
    return a / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])


@dataclass
class PrimaryPath:
    cylinder: int
    port_name: str
    points: list            # list of np.ndarray (3,), port -> collector
    radius_m: float

    @property
    def length_m(self) -> float:
        return float(sum(np.linalg.norm(b - a) for a, b in zip(self.points, self.points[1:])))


@dataclass
class CollectorGroup:
    index: int
    bank_index: int
    cylinders: list
    collector_position: np.ndarray
    outlet_direction: np.ndarray
    primaries: list = field(default_factory=list)
    radius_m: float = 0.03
    style: str = "4-into-1"


@dataclass
class HeaderPlan:
    groups: list

    def summary(self) -> list[str]:
        out = []
        for g in self.groups:
            lens = [p.length_m for p in g.primaries]
            out.append(f"group {g.index} bank {g.bank_index} {g.style}: cylinders {g.cylinders}, "
                       f"primaries {min(lens) * 1000:.0f}-{max(lens) * 1000:.0f} mm, collector r={g.radius_m * 1000:.0f} mm")
        return out


def _quarter_bend(start: np.ndarray, d_in: np.ndarray, d_out: np.ndarray, bend_r: float) -> list:
    """Points along a circular arc turning from direction d_in to d_out
    with centreline radius bend_r, starting at `start` (the arc's
    tangent point). Returns the points AFTER start."""
    d_in = _unit(d_in); d_out = _unit(d_out)
    if abs(float(np.dot(d_in, d_out))) > 0.999:
        return [start + d_out * bend_r]
    normal = _unit(np.cross(d_in, d_out))
    centre = start + _unit(np.cross(normal, d_in)) * bend_r
    ang = math.acos(max(-1.0, min(1.0, float(np.dot(d_in, d_out)))))
    pts = []
    r0 = start - centre
    for k in range(1, BEND_SEGMENTS + 1):
        t = ang * k / BEND_SEGMENTS
        # rotate r0 about `normal` by t (Rodrigues)
        r = r0 * math.cos(t) + np.cross(normal, r0) * math.sin(t) + normal * float(np.dot(normal, r0)) * (1 - math.cos(t))
        pts.append(centre + r)
    return pts


def _bank_groups(layout) -> list[tuple[int, list]]:
    groups: dict[tuple, list] = {}
    for g, ports in layout:
        if g.kind not in ("spark-piston", "compression-piston"):
            continue
        key = tuple(np.round(_unit(g.axis), 3))
        groups.setdefault(key, []).append((g, ports))
    return list(enumerate(groups.values()))


def _is_radial(layout) -> bool:
    gs = [g for g, _ in layout if g.kind in ("spark-piston", "compression-piston")]
    if len(gs) < 3:
        return False
    xs = {round(float(g.crank_centre[0]), 4) for g in gs}
    axes = {tuple(np.round(_unit(g.axis), 2)) for g in gs}
    return len(xs) == 1 and len(axes) >= 3


def _simple_outlet_paths(layout, kinds, plan, up_sign: float, style: str, primary_radius_m: float | None):
    """Non-piston kinds (an Otto-Langen's foot exhaust, an expander's
    chest passages, a Wankel's peripheral ports): a stub, one bend, and
    a short run into a collector/outlet just clear of the body -- there
    is no rail to run, the pipe leaves where the port is."""
    ports = [(g, p) for g, ps in layout for p in ps if g.kind in kinds and p.fluid_role == "exhaust"]
    if not ports:
        return
    bore = max(g.bore_m for g, _ in ports)
    r_prim = primary_radius_m if primary_radius_m else max(p.radius_m for _, p in ports) * 0.9
    bend_r = 2.0 * r_prim * BEND_RADIUS_FRAC_OF_DIAMETER
    up = np.array([0.0, up_sign, 0.0])
    mean_pos = sum(np.array(p.position) for _, p in ports) / len(ports)
    mean_dir = _unit(sum(np.array(p.direction) for _, p in ports))
    collector = mean_pos + mean_dir * (bore * 0.45 + bend_r) + up * (bore * 0.8)
    group = CollectorGroup(index=len(plan.groups), bank_index=0, cylinders=sorted({g.number for g, _ in ports}),
                           collector_position=collector, outlet_direction=up,
                           radius_m=r_prim * math.sqrt(len(ports)) * 0.85, style=style)
    for g, p in ports:
        pos = np.array(p.position); d = _unit(p.direction)
        pts = [pos, pos + d * (p.radius_m * STUB_FRAC_OF_RADIUS)]
        pts.extend(_quarter_bend(pts[-1], d, up, bend_r))
        pts.append(collector.copy())
        group.primaries.append(PrimaryPath(cylinder=g.number, port_name=p.name, points=pts, radius_m=r_prim))
    plan.groups.append(group)


def _radial_ring(layout, plan, primary_radius_m: float | None):
    """A radial's collector is a RING behind the cylinders: every
    primary stubs out of its head, bends rearward and inward, and joins
    the ring at its own angle; the ring exits at the bottom."""
    gs = [(g, ps) for g, ps in layout if g.kind in ("spark-piston", "compression-piston")]
    ports = [(g, p) for g, ps in gs for p in ps if p.fluid_role == "exhaust"]
    bore = max(g.bore_m for g, _ in gs)
    crank_c = np.array(gs[0][0].crank_centre)
    ring_r = max(float(np.linalg.norm(np.array(g.base) - crank_c)) for g, _ in gs) + bore * 0.55
    x_ring = crank_c[0] + bore * 0.9   # behind the cylinders (+x, toward the accessory case)
    r_prim = primary_radius_m if primary_radius_m else max(p.radius_m for _, p in ports) * 0.9
    bend_r = 2.0 * r_prim * BEND_RADIUS_FRAC_OF_DIAMETER
    outlet = np.array([x_ring, crank_c[1] - ring_r, crank_c[2]])
    group = CollectorGroup(index=len(plan.groups), bank_index=0, cylinders=[g.number for g, _ in gs],
                           collector_position=outlet, outlet_direction=np.array([0.0, -1.0, 0.0]),
                           radius_m=r_prim * 2.2, style="radial-ring")
    ring_pts = []
    n_seg = 36
    for k in range(n_seg + 1):
        ang = -math.pi / 2.0 + 2.0 * math.pi * k / n_seg
        ring_pts.append(np.array([x_ring, crank_c[1] + ring_r * math.cos(ang), crank_c[2] + ring_r * math.sin(ang)]))
    group.primaries.append(PrimaryPath(cylinder=0, port_name="ring", points=ring_pts, radius_m=r_prim * 1.5))
    for g, p in ports:
        pos = np.array(p.position); d = _unit(p.direction)
        radial_dir = _unit(np.array(g.axis))
        pts = [pos, pos + d * (p.radius_m * STUB_FRAC_OF_RADIUS)]
        back = np.array([1.0, 0.0, 0.0])
        pts.extend(_quarter_bend(pts[-1], d, back, bend_r))
        join = crank_c + radial_dir * ring_r; join[0] = x_ring
        pts.append(np.array([x_ring - bend_r, pts[-1][1], pts[-1][2]]))
        pts.extend(_quarter_bend(pts[-1], back, _unit(join - pts[-1]), bend_r))
        pts.append(join)
        group.primaries.append(PrimaryPath(cylinder=g.number, port_name=p.name, points=pts, radius_m=r_prim))
    plan.groups.append(group)


def plan_exhaust_header(layout, header_type: str = "tubular", primary_radius_m: float | None = None,
                        collector_end: str = "rear") -> HeaderPlan:
    """Route every exhaust port in the layout. `header_type`
    "stock-manifold" gives a log; anything else a tubular header
    grouped by the typical rule. `collector_end`: "rear" (+x) or
    "front" (-x) -- where the collector sits along the bank."""
    plan = HeaderPlan(groups=[])
    g_index = 0
    # kinds with no rail to run: their pipe leaves where the port is
    _simple_outlet_paths(layout, ("atmospheric",), plan, +1.0, "atmospheric-outlet", primary_radius_m)
    _simple_outlet_paths(layout, ("expander",), plan, +1.0, "expander-blast-pipe", primary_radius_m)
    _simple_outlet_paths(layout, ("rotary",), plan, -1.0, "rotary-2-into-1", primary_radius_m)
    if _is_radial(layout):
        _radial_ring(layout, plan, primary_radius_m)
        return plan
    g_index = len(plan.groups)
    for bank_index, members in _bank_groups(layout):
        members = sorted(members, key=lambda m: m[0].base[0])
        exhaust_ports = []
        for g, ports in members:
            for p in ports:
                if p.fluid_role == "exhaust":
                    exhaust_ports.append((g, p))
        if not exhaust_ports:
            continue
        axis = _unit(members[0][0].axis)
        bore = max(g.bore_m for g, _ in members)
        n_cyl = len(members)
        # the outboard direction: the exhaust ports' own mean outward direction, flattened off the bore axis
        mean_dir = _unit(sum(np.array(p.direction) for _, p in exhaust_ports))
        outboard = mean_dir - axis * float(np.dot(mean_dir, axis))
        outboard = _unit(outboard) if np.linalg.norm(outboard) > 1e-6 else _unit(np.cross(CRANK_AXIS, axis))
        down = -axis
        r_prim = primary_radius_m if primary_radius_m else max(p.radius_m for _, p in exhaust_ports) * 0.9
        bend_r = 2.0 * r_prim * BEND_RADIUS_FRAC_OF_DIAMETER
        # the rail: outboard of the head, a bit below it, running along x
        head_ref = max(np.array(p.position) for _, p in exhaust_ports) if False else None
        port_positions = [np.array(p.position) for _, p in exhaust_ports]
        rail_anchor = sum(port_positions) / len(port_positions)
        rail_point = rail_anchor + outboard * (bore * 0.55 + bend_r) + down * (bore * 0.45)
        # cylinder groups by the typical rule
        cyl_numbers = [g.number for g, _ in members]
        if header_type == "stock-manifold":
            splits = [cyl_numbers]
            style = "log"
        elif n_cyl <= 4:
            splits = [cyl_numbers]
            style = f"{n_cyl}-into-1"
        else:
            half = (n_cyl + 1) // 2
            splits = [cyl_numbers[:half], cyl_numbers[half:]]
            style = f"{n_cyl}-into-2"
        sign = 1.0 if collector_end == "rear" else -1.0
        for split in splits:
            xs = [float(g.base[0]) for g, _ in members if g.number in split]
            if header_type == "stock-manifold" and collector_end != "front":
                # a stock log on a straight engine dumps in the MIDDLE
                # (the runners meet at the centre, the downpipe drops
                # from there), not off one end
                x_end = (min(xs) + max(xs)) / 2.0
            else:
                x_end = (max(xs) if sign > 0 else min(xs)) + sign * (bore * 0.9)
            collector = rail_point.copy(); collector[0] = x_end
            collector = collector + down * (bore * 0.35) + outboard * (bore * 0.15)
            if len(split) == 1 and header_type != "stock-manifold":
                # a single: stub, one bend, straight out -- the outlet is
                # right there, no rail to run along
                g1, p1 = next((g, p) for g, p in exhaust_ports if g.number == split[0])
                pos = np.array(p1.position); d = _unit(p1.direction)
                pts = [pos, pos + d * (p1.radius_m * STUB_FRAC_OF_RADIUS)]
                out_dir = _unit(outboard * 0.6 + down * 1.0)
                pts.extend(_quarter_bend(pts[-1], d, out_dir, bend_r))
                collector = pts[-1] + out_dir * (bore * 0.5)
                pts.append(collector.copy())
                group = CollectorGroup(index=g_index, bank_index=bank_index, cylinders=list(split),
                                       collector_position=collector, outlet_direction=out_dir,
                                       radius_m=r_prim, style="1-into-1")
                group.primaries.append(PrimaryPath(cylinder=g1.number, port_name=p1.name, points=pts, radius_m=r_prim))
                plan.groups.append(group)
                g_index += 1
                continue
            group = CollectorGroup(index=g_index, bank_index=bank_index, cylinders=list(split),
                                   collector_position=collector, outlet_direction=down,
                                   radius_m=r_prim * math.sqrt(max(len(split), 1)) * 0.85, style=style)
            for g, p in exhaust_ports:
                if g.number not in split:
                    continue
                pos = np.array(p.position); d = _unit(p.direction)
                pts = [pos]
                stub_end = pos + d * (p.radius_m * STUB_FRAC_OF_RADIUS)
                pts.append(stub_end)
                # first turn: from the port direction toward outboard-and-down
                turn_dir = _unit(outboard * 1.0 + down * 0.9)
                pts.extend(_quarter_bend(stub_end, d, turn_dir, bend_r))
                # travel to the rail level
                at_rail = pts[-1].copy()
                rail_level = rail_point.copy(); rail_level[0] = at_rail[0]
                to_rail = rail_level - at_rail
                if np.linalg.norm(to_rail) > 1e-6:
                    pts.append(rail_level)
                if header_type == "stock-manifold":
                    # log: the port stub enters the shared rail; the rail
                    # itself runs from the first entry to the outlet
                    pass
                else:
                    # second turn onto the rail, then run to the collector's x
                    run_dir = CRANK_AXIS * sign
                    pts.extend(_quarter_bend(pts[-1], _unit(to_rail) if np.linalg.norm(to_rail) > 1e-6 else turn_dir,
                                             run_dir, bend_r))
                    run_end = pts[-1].copy(); run_end[0] = collector[0] - sign * bend_r
                    if abs(run_end[0] - pts[-1][0]) > 1e-6:
                        pts.append(run_end)
                    # final bend down into the collector
                    pts.extend(_quarter_bend(pts[-1], run_dir, down, bend_r))
                pts.append(collector.copy() if header_type != "stock-manifold" else rail_level)
                group.primaries.append(PrimaryPath(cylinder=g.number, port_name=p.name, points=pts, radius_m=r_prim))
            if header_type == "stock-manifold":
                # the log rail: every entry point along the bank, then the
                # outlet -- from the centre when the collector sits there
                entries = sorted((pr.points[-1] for pr in group.primaries), key=lambda q: q[0])
                if collector_end != "front" and len(entries) > 1:
                    log_pts = list(entries)
                    mid = collector.copy(); mid[0] = x_end; mid[1] = entries[0][1]; mid[2] = entries[0][2]
                    # rail runs end to end; the outlet drops from its middle
                    group.primaries.append(PrimaryPath(cylinder=0, port_name="log", points=log_pts, radius_m=r_prim * 1.6))
                    group.primaries.append(PrimaryPath(cylinder=0, port_name="log_outlet", points=[mid, collector.copy()], radius_m=r_prim * 1.6))
                else:
                    log_pts = list(entries) + [collector.copy()]
                    group.primaries.append(PrimaryPath(cylinder=0, port_name="log", points=log_pts, radius_m=r_prim * 1.6))
            plan.groups.append(group)
            g_index += 1
    return plan


def emit_header_graph(plan: HeaderPlan, node, edge, downstream_node: str, exhaust_radius_m: float,
                      existing_primary_edges: dict) -> None:
    """Write the plan into the drivetrain graph: a chain of
    "exhaust-flow-path" edges per primary through header waypoints, a
    collector node per group, and one edge from each collector to the
    downstream junction. The production graph's own straight
    `.exhaust_primary` edge for a port is re-pointed to run from the
    LAST waypoint into the collector so its identity (what the exhaust
    circuit and the audio path key on) survives."""
    for group in plan.groups:
        coll_id = f"powertrain.exhaust_collector_{group.index + 1}"
        node(coll_id, [float(v) for v in group.collector_position], "exhaust-collector",
             collector_style=group.style, cylinders=list(group.cylinders), mass_kg=1.5)
        for pr in group.primaries:
            if pr.cylinder == 0:
                # a log manifold's shared rail, or a radial's ring
                prev = None
                for k, pt in enumerate(pr.points[:-1]):
                    wid = f"powertrain.exhaust_log_{group.index + 1}.{pr.port_name}_seg_{k}"
                    node(wid, [float(v) for v in pt], "engine-block-port", port_kind="header-waypoint")
                    if prev is not None:
                        edge(f"powertrain.exhaust_log_{group.index + 1}.{pr.port_name}_run_{k}", prev, wid, "exhaust-flow-path",
                             radius=pr.radius_m, circuit_identity="exhaust",
                             medium_rate_state="exhaust-pulse-pressure-and-temperature")
                    prev = wid
                if prev is not None:
                    edge(f"powertrain.exhaust_log_{group.index + 1}.{pr.port_name}_outlet", prev, coll_id, "exhaust-flow-path",
                         radius=pr.radius_m, circuit_identity="exhaust",
                         medium_rate_state="exhaust-pulse-pressure-and-temperature")
                continue
            port_id = f"powertrain.cylinder_{pr.cylinder}.{pr.port_name}"
            prev = port_id
            for k, pt in enumerate(pr.points[1:-1], start=1):
                wid = f"{port_id}.header_{k}"
                node(wid, [float(v) for v in pt], "engine-block-port", port_kind="header-waypoint")
                edge(f"{port_id}.header_seg_{k}", prev, wid, "exhaust-flow-path", radius=pr.radius_m,
                     circuit_identity="exhaust", medium_rate_state="exhaust-pulse-pressure-and-temperature")
                prev = wid
            primary_edge = existing_primary_edges.get(port_id)
            if primary_edge is not None:
                primary_edge["a"] = prev
                primary_edge["b"] = coll_id
                primary_edge["radius"] = pr.radius_m
            else:
                edge(f"{port_id}.exhaust_primary", prev, coll_id, "exhaust-flow-path", radius=pr.radius_m,
                     circuit_identity="exhaust", medium_rate_state="exhaust-pulse-pressure-and-temperature")
        edge(f"powertrain.exhaust_collector_{group.index + 1}.to_downpipe", coll_id, downstream_node,
             "exhaust-flow-path", radius=max(group.radius_m, exhaust_radius_m), circuit_identity="exhaust",
             medium_rate_state="exhaust-pulse-pressure-and-temperature")
