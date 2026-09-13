"""Mesh once, aim every frame.

A mount that aims is a rigid structure in several pieces, each piece
moving as a whole. Rebuilding its triangles every time it moves is
therefore doing the wrong work: the triangles never change, only the
frame each group of them sits in. Meshing this turret costs about a
tenth of a second -- ten frames a second and nothing left over for
drawing it; transforming its vertices costs a fraction of a millisecond.

So: build the mesh ONCE from the document's rest pose, remember which
motion group every vertex belongs to, and then each frame move the
vertices instead of remaking them.

This is not the "bake a handful of poses and pick the nearest" approach,
and deliberately not. Baking quantises the aim, which is exactly the
error a fine-aim stage exists to remove -- so a baked mount cannot show
the one thing a six-leg platform is for. Here the coarse stages and the
fine stage are the same arithmetic, and half a degree of twitch moves
the muzzle by the eleven millimetres it really moves by.

TWO KINDS OF GEOMETRY, AND ONLY ONE OF THEM IS RIGID.

  A BODY belongs to one mechanism and moves with it entirely. Transform
  its vertices by that group's matrix and you are done.

  A MEMBER joins two bodies, and when those bodies are on different
  stages it is not rigid at all -- the elevation ram is a member whose
  whole job is to change length, and so is every leg of the fine
  platform and the recoil slide itself. Transforming such a tube
  rigidly would draw a ram that never extends while the thing it drives
  moves anyway, which is the specific dishonesty this module exists to
  avoid.

  So a spanning member is stretched instead: each of its vertices is
  held at its own fraction along the member's axis with its own radial
  offset, and both ends are then put wherever their own mechanisms put
  them. A tube drawn that way lengthens, shortens and swings exactly as
  the structure does, because it is being told to by the structure.

What makes all of this possible is that every body already DECLARES
which mechanism carries it (`motion_group`, stamped by
`turret_production.ProductionGraph.node`). Nothing here inspects a name
to decide what moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _flat(identity: str) -> str:
    return identity.replace("/", "_").replace(".", "_")


def _rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The minimal rotation taking unit vector u to unit vector v."""
    c = float(np.dot(u, v))
    axis = np.cross(u, v)
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        # antiparallel: any perpendicular axis will do
        pick = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, pick)
        axis /= np.linalg.norm(axis)
        s, c = 0.0, -1.0
    else:
        axis = axis / s
    x, y, z = axis
    t = np.arctan2(s, c)
    ct, st, C = np.cos(t), np.sin(t), 1.0 - np.cos(t)
    return np.array([
        [ct + x * x * C, x * y * C - z * st, x * z * C + y * st],
        [y * x * C + z * st, ct + y * y * C, y * z * C - x * st],
        [z * x * C - y * st, z * y * C + x * st, ct + z * z * C]])


def _rotations_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """`_rotation_between` for a whole stack of axis pairs at once.

    Same Rodrigues construction; the only care needed is the degenerate
    pair, where the cross product vanishes because the two axes are
    parallel or opposed. Parallel is the identity. Opposed cannot happen
    to a real member without it having been turned inside out, so it is
    given a perpendicular axis and a half turn rather than a NaN."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    axis = np.cross(u, v)
    s_ = np.linalg.norm(axis, axis=1)
    c = np.einsum("ni,ni->n", u, v)
    degenerate = s_ < 1e-12
    if degenerate.any():
        pick = np.where(np.abs(u[:, 0:1]) < 0.9,
                        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        alt = np.cross(u, pick)
        alt_n = np.linalg.norm(alt, axis=1)
        axis = np.where(degenerate[:, None], alt, axis)
        s_ = np.where(degenerate, np.where(c > 0.0, 0.0, alt_n), s_)
    axis = axis / np.maximum(np.linalg.norm(axis, axis=1), 1e-12)[:, None]
    t = np.arctan2(s_, c)
    ct, st = np.cos(t), np.sin(t)
    C = 1.0 - ct
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    r = np.empty((len(u), 3, 3))
    r[:, 0, 0] = ct + x * x * C
    r[:, 0, 1] = x * y * C - z * st
    r[:, 0, 2] = x * z * C + y * st
    r[:, 1, 0] = y * x * C + z * st
    r[:, 1, 1] = ct + y * y * C
    r[:, 1, 2] = y * z * C - x * st
    r[:, 2, 0] = z * x * C - y * st
    r[:, 2, 1] = z * y * C + x * st
    r[:, 2, 2] = ct + z * z * C
    return r


@dataclass
class _Span:
    """One member whose two ends are on different mechanisms."""
    a: str
    b: str
    verts: np.ndarray            # vertex indices this member owns
    t: np.ndarray                # (n,) fraction along the rest axis
    radial: np.ndarray           # (n,3) offset from the axis, in rest space
    rest_axis: np.ndarray        # (3,) unit
    rest_length: float


@dataclass
class ArticulatedMesh:
    """A mesh whose geometry knows what carries it."""
    rest_vertices: np.ndarray
    rest_normals: np.ndarray
    rigid: list = field(default_factory=list)      # (group name, vertex indices)
    spans: list = field(default_factory=list)      # _Span
    mesh: object = None
    rest_positions: dict = field(default_factory=dict)
    unclaimed: tuple = ()
    #: keyed per node rather than per motion group: a deflected
    #: structure rather than an articulated one
    per_node: bool = False

    # ---------------------------------------------------------------
    @classmethod
    def build(cls, document: dict, mesh, *,
              unclaimed_group: str = "frame",
              per_node: bool = False) -> "ArticulatedMesh":
        """Attribute every vertex of `mesh` to the structure that moves it.

        A mesh may legitimately hold geometry this document knows
        nothing about -- the demo draws the turret in a scene that also
        contains the engine beside it, in one static mesh. Anything not
        accounted for is put in `unclaimed_group`, which by default is
        the ground frame, and is also listed in `unclaimed` so a body
        that should have been moving and is not can be found rather
        than wondered about."""
        # WHAT MOVES THIS VERTEX. Normally a motion group: the whole
        # elevating mass swings together and one 4x4 carries all of it.
        #
        # PER-NODE IS THE SAME MACHINERY WITH A FINER KEY. A frame solve
        # does not return a transform per mechanism, it returns a
        # displacement per NODE -- every one different, none of them a
        # rigid body. Keyed per node, each node's own geometry
        # translates with it and EVERY member becomes a span stretched
        # between its two ends, which is exactly what the span path
        # below already does and the only reason it could not be used
        # for a deflected structure was the coarseness of the key.
        group_of_node = {}
        for n in document["nodes"]:
            group_of_node[n["identity"]] = (
                n["identity"] if per_node
                else "recoil" if n.get("recoils")
                else str(n.get("motion_group", "frame")))
        rest_positions = {n["identity"]: np.asarray(n["reference_position"], float)
                          for n in document["nodes"]}
        node_part = {"node_" + _flat(i): i for i in group_of_node}
        edge_part = {"edge_" + _flat(e["identity"]): (e["a"], e["b"])
                     for e in document["edges"]}

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        by_group: dict = {}
        spans: list = []
        unclaimed: list = []

        for part_name, (t0, t1) in zip(mesh.part_names, mesh.part_ranges):
            tris = mesh.triangles[t0:t1]
            if len(tris) == 0:
                continue
            idx = np.unique(tris.reshape(-1))
            identity = node_part.get(part_name)
            if identity is not None:
                by_group.setdefault(group_of_node[identity], []).append(idx)
                continue
            ends = edge_part.get(part_name)
            if ends is None:
                unclaimed.append(part_name)
                by_group.setdefault(unclaimed_group, []).append(idx)
                continue
            ga, gb = group_of_node.get(ends[0]), group_of_node.get(ends[1])
            if ga is not None and ga == gb:
                by_group.setdefault(ga, []).append(idx)
                continue
            if ga is None or gb is None:
                unclaimed.append(part_name)
                by_group.setdefault(unclaimed_group, []).append(idx)
                continue
            pa, pb = rest_positions[ends[0]], rest_positions[ends[1]]
            axis = pb - pa
            length = float(np.linalg.norm(axis))
            if length < 1e-9:
                by_group.setdefault(ga, []).append(idx)
                continue
            unit = axis / length
            rel = verts[idx] - pa
            t = rel @ unit / length
            radial = rel - np.outer(t * length, unit)
            spans.append(_Span(a=ends[0], b=ends[1], verts=idx, t=t, radial=radial,
                               rest_axis=unit, rest_length=length))

        rigid = [(name, np.concatenate(parts)) for name, parts in by_group.items()]
        # ONE SET OF ARRAYS FOR EVERY STRETCHING MEMBER. Thirty-three
        # members is a short loop, but a short loop of small numpy calls
        # is still six hundred calls a frame, and it measured ten
        # milliseconds -- two thirds of a frame's whole budget spent on
        # arithmetic that is the same shape for every member. Flattened
        # here, posing them is a dozen whole-array operations.
        flat_spans = None
        if spans:
            flat_spans = {
                "verts": np.concatenate([sp.verts for sp in spans]),
                "owner": np.concatenate([np.full(len(sp.verts), i, dtype=np.int32)
                                         for i, sp in enumerate(spans)]),
                "t": np.concatenate([sp.t for sp in spans]),
                "radial": np.concatenate([sp.radial for sp in spans]),
                "rest_axis": np.array([sp.rest_axis for sp in spans]),
                "a": [sp.a for sp in spans],
                "b": [sp.b for sp in spans],
            }
        built = cls(rest_vertices=verts,
                    rest_normals=np.asarray(mesh.normals, dtype=np.float64),
                    rigid=rigid, spans=spans, mesh=mesh,
                    rest_positions=rest_positions, unclaimed=tuple(unclaimed))
        built._groups = group_of_node
        built._flat = flat_spans
        built.per_node = bool(per_node)
        if flat_spans is not None:
            flat_spans["rest_a"] = np.array([rest_positions[i] for i in flat_spans["a"]])
            flat_spans["rest_b"] = np.array([rest_positions[i] for i in flat_spans["b"]])
            flat_spans["group_a"] = [group_of_node[i] for i in flat_spans["a"]]
            flat_spans["group_b"] = [group_of_node[i] for i in flat_spans["b"]]
        return built

    # ---------------------------------------------------------------
    def pose(self, frames: dict):
        """Move the mesh into the pose `frames` describes -- a mapping
        from motion-group name to a 4x4 transform, as produced by
        `turret_production.pose_frames`.

        Returns the same mesh object with its vertex and normal arrays
        replaced. Triangles, materials, part names and thermal groups
        are untouched, because none of them changed."""
        verts = self.rest_vertices.copy()
        norms = self.rest_normals.copy()
        eye = np.eye(4)

        for name, idx in self.rigid:
            m = np.asarray(frames.get(name, eye), dtype=np.float64)
            r = m[:3, :3]
            verts[idx] = self.rest_vertices[idx] @ r.T + m[:3, 3]
            n = self.rest_normals[idx] @ r.T
            norms[idx] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

        # THE MEMBERS THAT ARE NOT RIGID, because their two ends are not
        # on the same mechanism. Both ends go wherever their own stage
        # puts them and the tube is stretched between: each vertex keeps
        # its fraction along the axis and its radial offset, so the
        # member lengthens, shortens and swings exactly as the structure
        # makes it.
        f = self._flat
        if f is not None:
            ma = np.array([np.asarray(frames.get(g, eye))[:3] for g in f["group_a"]])
            mb = np.array([np.asarray(frames.get(g, eye))[:3] for g in f["group_b"]])
            pa = np.einsum("nij,nj->ni", ma[:, :, :3], f["rest_a"]) + ma[:, :, 3]
            pb = np.einsum("nij,nj->ni", mb[:, :, :3], f["rest_b"]) + mb[:, :, 3]
            axis = pb - pa
            length = np.linalg.norm(axis, axis=1)
            unit = axis / np.maximum(length, 1e-12)[:, None]
            rot = _rotations_between(f["rest_axis"], unit)
            owner = f["owner"]
            idx = f["verts"]
            along = (f["t"] * length[owner])[:, None] * unit[owner]
            verts[idx] = (pa[owner] + along
                          + np.einsum("nij,nj->ni", rot[owner], f["radial"]))
            n = np.einsum("nij,nj->ni", rot[owner], self.rest_normals[idx])
            norms[idx] = n / np.maximum(
                np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

        self.mesh.vertices = verts.astype(self.mesh.vertices.dtype, copy=False)
        self.mesh.normals = norms.astype(self.mesh.normals.dtype, copy=False)
        return self.mesh

    def displace(self, positions):
        """Pose the mesh from a SOLVED node position per node.

        `positions` is either a mapping identity -> xyz or an array in
        the document's own node order, which is what
        `FrameSolver.settled_position` hands back. Requires a mesh built
        with `per_node=True`; without that the key is too coarse to say
        where any individual node went.

        THIS IS NOT A SECOND POSING PATH. It builds the translations the
        existing `pose` already consumes, so members stretch and swing
        through the same flattened span arithmetic -- no rebake, no
        second mesh, no per-frame set_graph. Re-baking a 2236-part mesh
        once per instant of a shot was costing minutes per trial to
        redraw geometry that had not changed shape, only place."""
        if not self.per_node:
            raise ValueError(
                "displace needs ArticulatedMesh.build(..., per_node=True): "
                "a motion-group key cannot say where one node went")
        import numpy as _np
        if not isinstance(positions, dict):
            arr = _np.asarray(positions, dtype=_np.float64)
            order = list(self.rest_positions)
            if len(arr) != len(order):
                raise ValueError(
                    f"{len(arr)} positions for {len(order)} nodes -- pass a "
                    f"mapping if the order is not the document's own")
            positions = {ident: arr[i] for i, ident in enumerate(order)}
        frames = {}
        for ident, rest in self.rest_positions.items():
            now = positions.get(ident)
            if now is None:
                continue
            m = _np.eye(4)
            m[:3, 3] = _np.asarray(now, dtype=_np.float64) - rest
            frames[ident] = m
        return self.pose(frames)

    # ---------------------------------------------------------------
    def _group_of(self, identity: str) -> str:
        return getattr(self, "_groups", {}).get(identity, "frame")

    def coverage(self) -> dict:
        """How many vertices each group claims, plus how many belong to
        members that stretch. A stage with no vertices is one that will
        never visibly move, which is worth knowing before wondering why
        the barrel is not elevating."""
        out = {name: int(len(idx)) for name, idx in self.rigid}
        out["<stretching members>"] = int(sum(len(s.verts) for s in self.spans))
        return out
