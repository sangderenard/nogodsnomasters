"""Real rigid-body mass properties (total mass, center of gravity,
inertia tensor) of the whole engine ASSEMBLY -- built directly off
`mass_in_total` nodes drivetrain_graph.py's own graph already declares
(the real per-component split: block/head/crank shares plus whatever
else the production subunit already marks `mass_in_total=True`, e.g.
the battery), not a separate hand-placed estimate.

Each component is treated as a real point mass at its own real
reference_position -- a genuine, disclosed simplification (a real head
casting has its own local inertia about its own center, ignored here),
but a real, non-degenerate distribution of real masses at real
positions, which is what a genuine center of gravity and a genuine
inertia tensor both need. This is deliberately NOT the same quantity
as any node's own `inertia_kg_m2` (a real, separate thing: a rotating
part's own polar inertia about ITS spin axis, used by DrivetrainSolver
for shaft dynamics) -- this module computes the ASSEMBLY's own rigid-
body inertia tensor about its own center of gravity, for mount-
stability analysis, not shaft torque response.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drivetrain_graph import build_drivetrain_graph


@dataclass(frozen=True)
class RigidBodyProperties:
    total_mass_kg: float
    center_of_gravity: tuple[float, float, float]
    inertia_tensor_kg_m2: np.ndarray   # 3x3, about the center of gravity, in the engine's own local axes


def rigid_body_properties(engine, graph: dict | None = None) -> RigidBodyProperties:
    graph = graph if graph is not None else build_drivetrain_graph(engine)
    masses = []
    positions = []
    for n in graph["nodes"]:
        if not n.get("mass_in_total"):
            continue
        m = float(n.get("mass_kg", 0.0))
        if m <= 0.0:
            continue
        masses.append(m)
        positions.append(n["reference_position"])
    if not masses:
        return RigidBodyProperties(0.0, (0.0, 0.0, 0.0), np.zeros((3, 3)))

    m = np.array(masses, dtype=np.float64)
    p = np.array(positions, dtype=np.float64)
    total_mass = float(m.sum())
    cg = (m[:, None] * p).sum(axis=0) / total_mass

    r = p - cg[None, :]
    r2 = np.sum(r * r, axis=1)
    I = np.zeros((3, 3))
    for axis in range(3):
        I[axis, axis] = float(np.sum(m * (r2 - r[:, axis] ** 2)))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        val = -float(np.sum(m * r[:, i] * r[:, j]))
        I[i, j] = val
        I[j, i] = val

    return RigidBodyProperties(total_mass_kg=total_mass, center_of_gravity=tuple(cg.tolist()),
                               inertia_tensor_kg_m2=I)
