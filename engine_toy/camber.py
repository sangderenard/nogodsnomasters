"""Solve for the drawing you must BUILD to get the shape you WANT.

With a working stiffness solve the problem inverts. Instead of asking
where a structure ends up, ask what structure ends up where you wanted:

    find  Y  such that  Y + u(Y) = target

Y is the unloaded geometry -- the drawing a fabricator works to -- and
the target is the shape the machine should hold while it is carrying
what it carries. This is pre-camber, done by solving rather than by
guessing an allowance, and it is the same iteration bridge and airframe
assembly have always used.

It converges quickly because it is only weakly nonlinear: the stiffness
depends on member geometry, but moving a node ten millimetres barely
changes the stiffness of a member a metre long, so the deflection
predicted at pass two is very nearly the deflection at pass one.

FOUR THINGS THIS IMPLIES, and they are the reason this file is careful
rather than a one-line loop.

  1. CAMBER DOES NOT RELIEVE STRESS. It is the most common
     misunderstanding and the most important to state. A cambered
     structure under load sits at the target shape, and its members are
     carrying exactly the same forces they would have carried uncambered.
     Nothing has been unloaded; the deflection has merely been started
     from somewhere else. If a member was overstressed before, it is
     overstressed now, in the right place.

  2. IT IS SPECIFIC TO ONE LOAD CASE. The camber is the inverse of one
     particular deflection. Change the load -- fit the heavier barrel,
     fill the magazine, add armour -- and the structure is now wrong by
     the DIFFERENCE, which can be larger than the error you started with.
     An uncambered structure is wrong by its sag; a mis-cambered one can
     be wrong by nearly twice it.

  3. IT IS SPECIFIC TO ONE POSE. A turret is a mechanism, not a
     monument. Its shape under load depends on where it is pointed, so
     the camber can only be correct at one elevation, one traverse, one
     state of recoil. Cambering at stowed and firing at forty-two
     degrees means the correction is aimed somewhere the machine is not.
     This is refused rather than fudged: the pose is recorded and a
     mismatch is reported.

  4. SUPPORTS CANNOT BE CAMBERED. A node bolted to the world is where it
     is. Asking for a target that moves one is asking for the ground to
     move, and it is reported instead of quietly ignored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from frame_solver import FrameSolver, DOF_PER_NODE


@dataclass
class CamberSolution:
    """A drawing to build to, and everything needed to judge it."""
    #: the geometry to FABRICATE -- this is the answer
    rest_positions: np.ndarray
    #: the shape it reaches once loaded
    target_positions: np.ndarray
    #: what it actually reached, from a verifying solve
    achieved_positions: np.ndarray
    identities: tuple
    load_case: str
    pose: str
    iterations: int
    converged: bool
    #: worst distance between what was asked for and what was achieved
    residual_m: float
    #: how much each node was moved off the target to build it
    camber_m: np.ndarray
    #: nodes that could not be cambered because they are supports
    refused: tuple = ()

    @property
    def worst_camber_m(self) -> float:
        return float(np.abs(self.camber_m).max()) if len(self.camber_m) else 0.0

    def camber_of(self, identity: str) -> np.ndarray:
        return self.camber_m[self.identities.index(identity)]

    def describe(self) -> list:
        out = [f"  camber for '{self.load_case}' at pose '{self.pose}': "
               f"{'converged' if self.converged else 'DID NOT CONVERGE'} "
               f"in {self.iterations} pass(es)",
               f"    worst camber {self.worst_camber_m * 1000:+7.3f} mm   "
               f"residual after verification {self.residual_m * 1000:.4f} mm"]
        worst = int(np.argmax(np.abs(self.camber_m).max(axis=1)))
        out.append(f"    most cambered member end: {self.identities[worst]} by "
                   f"{np.abs(self.camber_m[worst]).max() * 1000:.3f} mm")
        if self.refused:
            out.append(f"    {len(self.refused)} support node(s) could not be "
                       f"cambered and were left alone: "
                       f"{', '.join(self.refused[:3])}"
                       + (" ..." if len(self.refused) > 3 else ""))
        out.append("    NOTE: this changes where the structure sits, not what "
                   "its members carry. The forces are unchanged.")
        return out


def solve_for_camber(document: dict, *, target: np.ndarray | None = None,
                     load_case: str = "self weight",
                     pose: str = "as authored",
                     loads: dict | None = None,
                     tolerance_m: float = 1e-6,
                     max_iterations: int = 12) -> CamberSolution:
    """Find the unloaded drawing that settles onto `target` under load.

    `target` defaults to the document's own authored geometry, which is
    the usual request: "make it look like the drawing when it is
    working", rather than "make it look like the drawing on the bench".
    """
    nodes = document["nodes"]
    identities = tuple(n["identity"] for n in nodes)
    authored = np.array([n["reference_position"] for n in nodes], float)
    goal = authored.copy() if target is None else np.asarray(target, float).copy()

    supports = [i for i, n in enumerate(nodes)
                if n.get("fixed_to") or n.get("kind") == "structural-body-pin-frame-foot"]
    refused = tuple(identities[i] for i in supports
                    if not np.allclose(goal[i], authored[i], atol=1e-12))
    # a support is where it is: put the goal back on the ground
    for i in supports:
        goal[i] = authored[i]

    rest = goal.copy()
    converged, used = False, 0
    for used in range(1, max_iterations + 1):
        trial = _document_at(document, rest)
        solved = FrameSolver(document=trial, gravity=True,
                             loads=loads or {}).solve()
        u = solved["displacement"][:, :3]
        # THE FIXED POINT: build where the target is, less wherever the
        # load is going to take it.
        new_rest = goal - u
        for i in supports:
            new_rest[i] = authored[i]
        shift = float(np.abs(new_rest - rest).max())
        rest = new_rest
        if shift < tolerance_m:
            converged = True
            break

    # VERIFY BY BUILDING IT. The iteration can converge on its own
    # arithmetic and still be wrong; the only proof is to load the
    # cambered geometry and see where it lands.
    built = _document_at(document, rest)
    check = FrameSolver(document=built, gravity=True, loads=loads or {}).solve()
    achieved = rest + check["displacement"][:, :3]
    residual = float(np.abs(achieved - goal).max())

    return CamberSolution(
        rest_positions=rest, target_positions=goal, achieved_positions=achieved,
        identities=identities, load_case=load_case, pose=pose,
        iterations=used, converged=converged and residual < tolerance_m * 50,
        residual_m=residual, camber_m=rest - goal, refused=refused)


def _document_at(document: dict, positions: np.ndarray) -> dict:
    """The same structure with its nodes moved, and its members' rest
    lengths recomputed to match.

    THE REST LENGTHS MUST FOLLOW. A cambered drawing is not the old
    drawing with the nodes nudged -- the members are genuinely made to
    different lengths, and that is the entire physical content of
    cambering. Moving the nodes without remaking the members would
    describe a structure built to the old lengths and forced into a new
    shape, which is a pre-stressed assembly and a different thing.
    """
    index = {n["identity"]: i for i, n in enumerate(document["nodes"])}
    nodes = [{**n, "reference_position": [float(v) for v in positions[i]]}
             for i, n in enumerate(document["nodes"])]
    edges = []
    for e in document["edges"]:
        if e["a"] not in index or e["b"] not in index:
            edges.append(e)
            continue
        length = float(np.linalg.norm(positions[index[e["b"]]]
                                      - positions[index[e["a"]]]))
        new = {**e, "rest_length": length}
        if e.get("damage"):
            new["damage"] = {**e["damage"], "natural_rest_length": length}
        edges.append(new)
    return {**document, "nodes": nodes, "edges": edges}


def cambered_document(document: dict, solution: CamberSolution) -> dict:
    """The drawing to hand a fabricator."""
    return _document_at(document, solution.rest_positions)


def compare_load_cases(document: dict, solution: CamberSolution, *,
                       other_loads: dict, other_name: str) -> dict:
    """What a camber cut for one load does under a different one.

    This is implication 2, measured rather than asserted: build to the
    camber, apply the load it was NOT cut for, and see how far off the
    target it now sits -- against how far off an uncambered structure
    would have been."""
    built = cambered_document(document, solution)
    under_other = FrameSolver(document=built, gravity=True,
                              loads=other_loads).solve()
    cambered_error = np.abs(solution.rest_positions
                            + under_other["displacement"][:, :3]
                            - solution.target_positions).max()
    plain = FrameSolver(document=document, gravity=True,
                        loads=other_loads).solve()
    uncambered_error = np.abs(plain["displacement"][:, :3]).max()
    return {"load_case": other_name,
            "cambered_error_m": float(cambered_error),
            "uncambered_error_m": float(uncambered_error),
            "camber_helped": cambered_error < uncambered_error}


# =====================================================================
#  CURVED MEMBERS: THE FUNNY UNLOADED SHAPE
# =====================================================================
def subdivide(document: dict, *, max_element_m: float = 0.25,
              only: tuple = ()) -> dict:
    """Split long members so their own curvature can be represented.

    WHY THIS IS NEEDED BEFORE CAMBER MEANS ANYTHING FOR A LONG MEMBER.
    A two-node element is a straight line between its ends. Cambering
    its endpoints moves the line; it cannot bow it. So for a member
    whose worst deflection is at MIDSPAN -- which is most of them, and
    all the simply supported ones -- node camber captures precisely none
    of the sag, because the ends are exactly where the sag is not.

    The fix is not a cleverer formula, it is more nodes. Split a member
    into sub-elements and the camber solve produces a genuinely CURVED
    unloaded shape all by itself, because the intermediate nodes are now
    free to be cambered.

    And that curve is often the whole answer. A beam that sags does not
    necessarily need a bigger section -- bigger costs mass everywhere
    and buys stiffness by the fourth power of a radius nobody has room
    for. It frequently needs the same section rolled to a funny shape,
    which costs nothing but the rolling.
    """
    index = {n["identity"]: i for i, n in enumerate(document["nodes"])}
    position = np.array([n["reference_position"] for n in document["nodes"]], float)
    nodes = [dict(n) for n in document["nodes"]]
    edges = []
    for e in document["edges"]:
        if e["a"] not in index or e["b"] not in index:
            edges.append(dict(e))
            continue
        if only and e["identity"] not in only:
            edges.append(dict(e))
            continue
        a, b = position[index[e["a"]]], position[index[e["b"]]]
        length = float(np.linalg.norm(b - a))
        pieces = max(1, int(math.ceil(length / max(max_element_m, 1e-6))))
        if pieces == 1:
            edges.append(dict(e))
            continue
        previous = e["a"]
        for k in range(1, pieces + 1):
            if k < pieces:
                t = k / pieces
                mid = a + (b - a) * t
                ident = f"{e['identity']}.node{k}"
                nodes.append({"identity": ident, "kind": "chassis-load-node",
                              "reference_position": [float(v) for v in mid],
                              "body_half_extent_m": [0.01, 0.01, 0.01],
                              "mass_kg": 0.0, "in_view": False,
                              "subdivision_of": e["identity"]})
            else:
                ident = e["b"]
            piece = {**e, "identity": f"{e['identity']}.seg{k}",
                     "a": previous, "b": ident,
                     "rest_length": length / pieces}
            if e.get("damage"):
                piece["damage"] = {**e["damage"],
                                   "natural_rest_length": length / pieces}
            edges.append(piece)
            previous = ident
    return {**document, "nodes": nodes, "edges": edges}


def member_shapes(document: dict, solution: CamberSolution) -> dict:
    """The curve each subdivided member must be rolled to.

    Reported as the offset of each intermediate station from the
    straight line between the member's ends -- which is what a
    fabricator actually sets up to, rather than a list of coordinates."""
    by_member: dict = {}
    index = {i: k for k, i in enumerate(solution.identities)}
    for node in document["nodes"]:
        parent = node.get("subdivision_of")
        if parent:
            by_member.setdefault(parent, []).append(node["identity"])
    shapes = {}
    for parent, stations in by_member.items():
        ends = [e for e in document["edges"]
                if e["identity"].startswith(parent + ".seg")]
        if not ends:
            continue
        start, finish = ends[0]["a"], ends[-1]["b"]
        if start not in index or finish not in index:
            continue
        a = solution.rest_positions[index[start]]
        b = solution.rest_positions[index[finish]]
        axis = b - a
        span = float(np.linalg.norm(axis))
        if span < 1e-9:
            continue
        unit = axis / span
        offsets = []
        for ident in sorted(stations):
            p = solution.rest_positions[index[ident]]
            rel = p - a
            along = float(np.dot(rel, unit))
            perpendicular = rel - unit * along
            offsets.append((along / span, float(np.linalg.norm(perpendicular)),
                            perpendicular))
        peak = max(offsets, key=lambda o: o[1])
        shapes[parent] = {"span_m": span, "stations": len(offsets),
                          "peak_offset_m": peak[1], "peak_at_fraction": peak[0],
                          "peak_direction": peak[2]}
    return shapes
