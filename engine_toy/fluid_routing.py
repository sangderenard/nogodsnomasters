"""Shared physical routing for graph-declared fluid connections.

There are three deliberately different promises a fluid edge can make:

``rigid-orthogonal``
    Installed hard line.  It travels on Cartesian runs, with 90 degree
    elbows, and detours around authored body envelopes.  Three or more lines
    meeting at one port declare a tee there.
``flexible-hose``
    A real hose with a requested reserve of slack.  Its displayed and
    transport lengths follow the sagged polyline rather than the chord.
``trivial``
    A schematic connection.  Routing cannot affect mechanics; it reports
    Manhattan length and is rendered only as a thin conceptual trace.

The output is ordinary edge ``waypoints`` plus explicit route/fitting fields.
Consequently render, fluid inventory, heat loss and pressure calculations can
all consume one graph record rather than maintaining separate decorative and
physical paths.
"""
from __future__ import annotations

from collections import defaultdict
import heapq
import itertools
import math

import numpy as np


RIGID_ORTHOGONAL = "rigid-orthogonal"
FLEXIBLE_HOSE = "flexible-hose"
TRIVIAL = "trivial"
ROUTE_CLASSES = frozenset((RIGID_ORTHOGONAL, FLEXIBLE_HOSE, TRIVIAL))


def is_fluid_edge(edge: dict) -> bool:
    constraint = str(edge.get("constraint", "")).lower()
    if any(word in constraint for word in (
            "fluid", "fuel", "coolant", "oil-line", "hydraulic-line",
            "hydraulic-hose", "pneumatic", "refrigerant", "air-line",
            "vent-line", "delivery-line", "supply-line")):
        return True
    return bool(edge.get("circuit_identity")) and not any(
        word in constraint for word in ("wire", "cable", "shaft", "belt"))


def route_class(edge: dict) -> str:
    declared = edge.get("fluid_route_class")
    if declared is not None:
        declared = str(declared)
        if declared not in ROUTE_CLASSES:
            raise ValueError(f"unknown fluid_route_class {declared!r}")
        return declared
    wording = " ".join(str(edge.get(k, "")).lower()
                       for k in ("constraint", "material", "routing"))
    if edge.get("routed_through_metaconduit") or "metaconduit" in wording:
        return FLEXIBLE_HOSE
    if "trivial" in wording or "conceptual" in wording:
        return TRIVIAL
    if "hose" in wording or "flexible" in wording or "rubber" in wording:
        return FLEXIBLE_HOSE
    return RIGID_ORTHOGONAL


def _simplify(points: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if out and np.allclose(out[-1], p, atol=1e-10):
            continue
        if len(out) >= 2:
            u = out[-1] - out[-2]
            v = p - out[-1]
            if np.count_nonzero(np.abs(u) > 1e-10) == 1 and np.count_nonzero(
                    np.abs(v) > 1e-10) == 1 and np.argmax(np.abs(u)) == np.argmax(np.abs(v)):
                out[-1] = p
                continue
        out.append(p)
    return out


def _path_length(points) -> float:
    return sum(float(np.linalg.norm(points[i + 1] - points[i]))
               for i in range(len(points) - 1))


def _segment_clear(a: np.ndarray, b: np.ndarray, obstacles) -> bool:
    changed = np.flatnonzero(np.abs(b - a) > 1e-10)
    if len(changed) != 1:
        return False
    axis = int(changed[0])
    low, high = sorted((float(a[axis]), float(b[axis])))
    eps = 1e-8
    for lo, hi in obstacles:
        if any(not (float(lo[d]) + eps < float(a[d]) < float(hi[d]) - eps)
               for d in range(3) if d != axis):
            continue
        if max(low, float(lo[axis]) + eps) < min(high, float(hi[axis]) - eps):
            return False
    return True


def _orthogonal_candidate(a, b, order) -> list[np.ndarray]:
    p = np.asarray(a, float).copy()
    points = [p.copy()]
    for axis in order:
        p = p.copy(); p[axis] = b[axis]
        points.append(p)
    return _simplify(points)


def _route_clear(points, obstacles) -> bool:
    return all(_segment_clear(points[i], points[i + 1], obstacles)
               for i in range(len(points) - 1))


def _orthogonal_astar(a, b, obstacles) -> list[np.ndarray] | None:
    # Continuous rectilinear paths only need turn coordinates at an endpoint
    # or just outside an obstacle face.  Search that finite coordinate lattice.
    coords = [{float(a[d]), float(b[d])} for d in range(3)]
    for lo, hi in obstacles:
        for d in range(3):
            coords[d].update((float(lo[d]), float(hi[d])))
    coords = [tuple(sorted(c)) for c in coords]
    coord_index = [{value: i for i, value in enumerate(axis_values)}
                   for axis_values in coords]
    start = tuple(float(v) for v in a); goal = tuple(float(v) for v in b)
    queue = [(sum(abs(start[d] - goal[d]) for d in range(3)), 0.0,
              start, -1)]
    best = {(start, -1): 0.0}
    previous = {}
    final = None
    while queue and len(best) < 60000:
        _score, cost, point, incoming = heapq.heappop(queue)
        state = (point, incoming)
        if cost != best.get(state):
            continue
        if point == goal:
            final = state; break
        p = np.asarray(point, float)
        for axis in range(3):
            here = coord_index[axis][point[axis]]
            neighbour_indices = (here - 1, here + 1)
            for neighbour in neighbour_indices:
                if neighbour < 0 or neighbour >= len(coords[axis]):
                    continue
                value = coords[axis][neighbour]
                q = p.copy(); q[axis] = value
                if not _segment_clear(p, q, obstacles):
                    continue
                qt = tuple(float(v) for v in q)
                bend = 1e-5 if incoming not in (-1, axis) else 0.0
                new_cost = cost + abs(value - point[axis]) + bend
                next_state = (qt, axis)
                if new_cost + 1e-12 >= best.get(next_state, math.inf):
                    continue
                best[next_state] = new_cost
                previous[next_state] = state
                heuristic = sum(abs(qt[d] - goal[d]) for d in range(3))
                heapq.heappush(queue, (new_cost + heuristic, new_cost,
                                       qt, axis))
    if final is None:
        return None
    rev = []
    while True:
        rev.append(np.asarray(final[0], float))
        if final not in previous:
            break
        final = previous[final]
    return _simplify(list(reversed(rev)))


def orthogonal_route(a, b, obstacles) -> list[np.ndarray]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    # Bodies far outside the endpoint envelope cannot intersect either a
    # shortest Manhattan candidate or its local detour.  Keeping them out of
    # the coordinate lattice is what makes routing scale with the contents of
    # this bay, rather than every body in the complete machine.
    margin = max(0.12, float(np.linalg.norm(b - a)) * 0.20)
    corridor_lo = np.minimum(a, b) - margin
    corridor_hi = np.maximum(a, b) + margin
    obstacles = [(lo, hi) for lo, hi in obstacles
                 if np.all(hi >= corridor_lo) and np.all(lo <= corridor_hi)]
    if len(obstacles) > 18:
        midpoint = (a + b) * 0.5
        obstacles = sorted(obstacles, key=lambda box: float(np.linalg.norm(
            np.maximum(box[0] - midpoint, np.maximum(0.0, midpoint - box[1])))))[:18]
    candidates = [_orthogonal_candidate(a, b, order)
                  for order in itertools.permutations(range(3))]
    clear = [p for p in candidates if _route_clear(p, obstacles)]
    if clear:
        return min(clear, key=lambda p: (_path_length(p), len(p)))
    # The normal obstruction is a casting or tank in an otherwise open bay.
    # Try one dogleg outside each nearby face before paying for the general
    # coordinate-lattice search.  This remains an exact rectilinear route and
    # is dramatically cheaper for a complete station with hundreds of lines.
    doglegs = []
    for detour_axis in range(3):
        face_coordinates = sorted({float(box[side][detour_axis])
                                   for box in obstacles for side in (0, 1)})
        other = [d for d in range(3) if d != detour_axis]
        for value in face_coordinates:
            for order in (other, list(reversed(other))):
                p = a.copy(); path = [p.copy()]
                p = p.copy(); p[detour_axis] = value; path.append(p.copy())
                for axis in order:
                    p = p.copy(); p[axis] = b[axis]; path.append(p.copy())
                p = p.copy(); p[detour_axis] = b[detour_axis]; path.append(p)
                path = _simplify(path)
                if _route_clear(path, obstacles):
                    doglegs.append(path)
    if doglegs:
        return min(doglegs, key=lambda p: (_path_length(p), len(p)))
    found = _orthogonal_astar(a, b, obstacles)
    if found is None:
        raise ValueError("no collision-free orthogonal fluid route")
    return found


def slack_hose_route(a, b, slack_fraction: float = 0.08,
                     samples: int = 12) -> list[np.ndarray]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    chord = float(np.linalg.norm(b - a))
    if chord <= 1e-10:
        return [a, b]
    direction = (b - a) / chord
    gravity = np.array((0.0, -1.0, 0.0))
    sag_axis = gravity - direction * float(np.dot(gravity, direction))
    if np.linalg.norm(sag_axis) < 1e-7:
        sag_axis = np.array((0.0, 0.0, 1.0))
    sag_axis /= np.linalg.norm(sag_axis)
    target = chord * (1.0 + max(0.0, float(slack_fraction)))

    def path(sag):
        return [a + (b - a) * t + sag_axis * sag * math.sin(math.pi * t)
                for t in np.linspace(0.0, 1.0, max(4, int(samples)))]
    lo, hi = 0.0, max(0.02, chord)
    while _path_length(path(hi)) < target:
        hi *= 2.0
    for _ in range(32):
        mid = (lo + hi) * 0.5
        if _path_length(path(mid)) < target:
            lo = mid
        else:
            hi = mid
    return path(hi)


def _obstacles(graph: dict, endpoint_ids: set[str], endpoint_points,
               clearance: float):
    boxes = []
    for node in graph.get("nodes", ()):
        if node.get("identity") in endpoint_ids or not node.get("in_view", True):
            continue
        if node.get("solver_condensed_into") or node.get("kind") == "engine-block-port":
            continue
        centre = node.get("reference_position")
        half = node.get("body_half_extent_m")
        if centre is None or half is None:
            continue
        centre = np.asarray(centre, float); half = np.asarray(half, float)
        if centre.shape != (3,) or half.shape != (3,):
            continue
        lo = centre - half - clearance; hi = centre + half + clearance
        # A port may be authored on or just inside its parent casting.  That
        # casting is the fitting's launch/landing body, not an obstacle the
        # connected line can somehow route around without first crossing.
        if any(np.all(point >= lo) and np.all(point <= hi)
               for point in endpoint_points):
            continue
        boxes.append((lo, hi))
    return boxes


def annotate_fluid_routes(graph: dict, *, clearance_m: float = 0.018) -> dict:
    """Annotate all fluid edges in-place and return ``graph``.

    Authored waypoints win.  This makes the function idempotent and permits a
    designer to override an automatic route without creating another schema.
    """
    positions = {str(n["identity"]): np.asarray(n["reference_position"], float)
                 for n in graph.get("nodes", ()) if n.get("reference_position") is not None}
    fluid_edges = [e for e in graph.get("edges", ()) if is_fluid_edge(e)]
    degree = defaultdict(int)
    for edge in fluid_edges:
        degree[str(edge["a"])] += 1; degree[str(edge["b"])] += 1
    for edge in fluid_edges:
        a = positions.get(str(edge["a"])); b = positions.get(str(edge["b"]))
        if a is None or b is None:
            continue
        route_was_explicit = "fluid_route_class" in edge
        cls = route_class(edge)
        edge["fluid_route_class"] = cls
        if edge.get("waypoints"):
            points = [a, *(np.asarray(p, float) for p in edge["waypoints"]), b]
        elif cls == RIGID_ORTHOGONAL:
            obstacle_aware = bool(edge.get("obstacle_aware_route",
                                           route_was_explicit))
            if obstacle_aware:
                obstacles = _obstacles(
                    graph, {str(edge["a"]), str(edge["b"])}, (a, b),
                    float(edge.get("route_clearance_m", clearance_m)))
                try:
                    points = orthogonal_route(a, b, obstacles)
                except ValueError as exc:
                    raise ValueError(f"{edge.get('identity')}: {exc}") from exc
            else:
                # With no obstacles every axis order has exactly the same
                # Manhattan length.  Keep one deterministic order rather
                # than spending six geometry passes proving that identity.
                points = _orthogonal_candidate(a, b, (0, 1, 2))
            edge["route_obstacle_aware"] = obstacle_aware
            edge["route_requires_obstacle_audit"] = not obstacle_aware
            edge["waypoints"] = [p.tolist() for p in points[1:-1]]
        elif cls == FLEXIBLE_HOSE:
            points = slack_hose_route(a, b, float(edge.get(
                "hose_slack_fraction", 0.08)))
            edge["waypoints"] = [p.tolist() for p in points[1:-1]]
            edge["hose_slack_fraction"] = float(edge.get(
                "hose_slack_fraction", 0.08))
        else:
            points = [a, b]
            edge["waypoints"] = []
            edge["route_visual"] = "conceptual-thin"
        edge["route_length_m"] = (float(np.abs(b - a).sum()) if cls == TRIVIAL
                                  else _path_length(points))
        edge["route_fittings"] = {
            "a": "tee" if degree[str(edge["a"])] >= 3 else "straight",
            "b": "tee" if degree[str(edge["b"])] >= 3 else "straight",
            "interior_elbows_90_deg": (len(points) - 2
                                         if cls == RIGID_ORTHOGONAL else 0),
        }
    return graph
