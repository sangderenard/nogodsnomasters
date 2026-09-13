"""Shared routed-pipe geometry: a real bent path between two real graph
points, laid down as waypoint nodes + segment edges -- the same real
pattern dressing.py's intake runners pioneered (bend the pipe through
real 3D waypoints instead of one telescoping straight tube), pulled out
here so any duct (a turbo's swept up-pipe, a remote supercharger's
inlet/outlet ducts, a remote air box or snorkel's intake hose) can lay
one down without re-deriving the waypoint/edge bookkeeping each time.

Pure geometry + graph bookkeeping only -- no engine-specific knowledge,
so this has no import risk with either dressing.py or engine_parts.py.
"""
from __future__ import annotations

import numpy as np


def duct_waypoints(a, b, rise: float = 0.05, forward: np.ndarray | None = None,
                   forward_amount: float = 0.0) -> list[np.ndarray]:
    """A real two-bend "over the top" duct path from a to b: lift off
    a, an optional lateral push toward `forward` (a real unit vector --
    the same use a turbo swoop or a snorkel riser has for "lean this
    duct toward the front/up" instead of routing it straight through
    whatever sits between the two points), then down into b. Real
    plumbing practice for anything that has to clear castings/other
    parts between two points that aren't already in each other's line
    of sight -- not just a straight tube.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0])
    mid_height = max(a[1], b[1]) + rise
    p1 = a + up * (mid_height - a[1])
    p2 = b + up * (mid_height - b[1])
    if forward is not None and forward_amount != 0.0:
        push = np.asarray(forward, dtype=np.float64) * forward_amount
        p1 = p1 + push
        p2 = p2 + push
    return [a, p1, p2, b]


def lay_routed_pipe(node, edge, base_id: str, points: list[np.ndarray], radius: float,
                    circuit_identity: str, medium_rate_state: str,
                    start_identity: str, end_identity: str, kind: str = "low-pressure-air-line") -> None:
    """Lays real waypoint nodes for every interior point in `points`
    (points[0] and points[-1] are assumed to already be real, existing
    node identities -- `start_identity`/`end_identity` -- not re-
    created here) and chains segment edges start -> waypoints -> end.
    A 2-point `points` list (no interior waypoints at all) just lays
    the one direct segment -- the degenerate, real case of a duct that
    doesn't need to bend."""
    prev_id = start_identity
    n_interior = len(points) - 2
    for k in range(1, n_interior + 1):
        wid = f"{base_id}_seg_{k}"
        node(wid, [float(v) for v in points[k]], "engine-block-port", port_kind="duct-waypoint")
        edge(f"{base_id}_seg_{k}", prev_id, wid, kind, radius=radius,
             circuit_identity=circuit_identity, medium_rate_state=medium_rate_state)
        prev_id = wid
    edge(f"{base_id}_final", prev_id, end_identity, kind, radius=radius,
         circuit_identity=circuit_identity, medium_rate_state=medium_rate_state)
