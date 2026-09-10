"""Triangulated mesh primitives, ported faithfully from the real in-game
mesh builders -- not reinvented.

`tube_mesh` is a direct Python port of the `tube`/`fixedTube` closures in
`turing/src/compiler/abstract_ui_div_map.py` (lines ~3784, 3837, 3894):
the same perpendicular-basis construction (an arbitrary helper vector
crossed against the tube axis), the same ring-of-`sides`-vertices sweep
between two endpoints, the same per-side two-triangle winding
`[[p0,ra],[p1,rb],[p2,rb],[p0,ra],[p2,rb],[p3,ra]]`, emitted as
non-indexed triangle soup exactly like the real renderer does. This is
what the real game already uses for wiring harnesses, brake/hydraulic
hoses, fuel lines, and powertrain-part shafts.

`cuboid_mesh` is a direct Python port of the `fixedCuboid`/`cuboid`
closures (same file, ~line 3781/3891) plus the real `faces` index table
(line 3557): 8 corners of a centered box, 6 faces of 2 triangles each,
the exact same face-index/normal table.

Both return `(vertices, normals)` as (N, 3) float64 arrays, one row per
triangle-soup vertex (N is a multiple of 3), matching the real renderer's
per-vertex position+normal layout minus the trailing RGB triplet (color
is a rendering concern, applied by the caller, not part of the geometry).
"""
from __future__ import annotations

import numpy as np

# Real face-index/normal table, ported verbatim from abstract_ui_div_map.py
# (`const faces = [...]`, line 3557): each entry is (6 corner indices for
# 2 triangles, face normal), indexing into the 8-corner box
# [-1,-1,-1]..[1,1,1] scaled by half-extent, in the same corner order.
_CUBOID_CORNER_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)

_CUBOID_FACES = (
    ((4, 5, 6, 4, 6, 7), (0, 0, 1)),
    ((1, 0, 3, 1, 3, 2), (0, 0, -1)),
    ((5, 1, 2, 5, 2, 6), (1, 0, 0)),
    ((0, 4, 7, 0, 7, 3), (-1, 0, 0)),
    ((7, 6, 2, 7, 2, 3), (0, 1, 0)),
    ((0, 1, 5, 0, 5, 4), (0, -1, 0)),
)


def _normalized3(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same arbitrary-helper-vector trick the real closures use: pick
    whichever of +Y/+X is less parallel to the axis, cross it in twice
    to get two mutually perpendicular unit vectors spanning the tube's
    cross-section."""
    helper = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.85 else np.array([1.0, 0.0, 0.0])
    u = _normalized3(np.cross(axis, helper))
    v = _normalized3(np.cross(axis, u))
    return u, v


# Global tessellation detail: every tube/drum generator multiplies its
# requested side count by this (floored at 3). The live view bakes at a
# lower detail than an export; set_detail() is the one knob.
DETAIL = 1.0


def set_detail(detail: float) -> None:
    global DETAIL
    DETAIL = max(0.15, float(detail))


def _sides(sides: int) -> int:
    return max(3, int(round(sides * DETAIL)))


def tube_mesh(start, end, radius: float, sides: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """A real cylindrical tube swept between two points -- ported
    algorithm, not approximated: see module docstring."""
    sides = _sides(sides)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    axis = _normalized3(end - start)
    u, v = _perpendicular_basis(axis)

    angles = np.arange(sides + 1) * (2.0 * np.pi / sides)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    # radial[i] = u*cos(angle_i) + v*sin(angle_i), one ring direction per angle
    radial = np.outer(cos_a, u) + np.outer(sin_a, v)

    verts = []
    norms = []
    for segment in range(sides):
        ra, rb = radial[segment], radial[segment + 1]
        p0 = start + ra * radius
        p1 = start + rb * radius
        p2 = end + rb * radius
        p3 = end + ra * radius
        for point, normal in ((p0, ra), (p1, rb), (p2, rb), (p0, ra), (p2, rb), (p3, ra)):
            verts.append(point)
            norms.append(normal)
    return np.array(verts, dtype=np.float64), np.array(norms, dtype=np.float64)


def cuboid_mesh(center, half_extent) -> tuple[np.ndarray, np.ndarray]:
    """A real box mesh, ported algorithm and face table (see module
    docstring) -- the same 8-corner/6-face/2-triangle construction the
    real renderer uses for every boxy vehicle part."""
    center = np.asarray(center, dtype=np.float64)
    half_extent = np.asarray(half_extent, dtype=np.float64)
    corners = center + _CUBOID_CORNER_SIGNS * half_extent

    verts = []
    norms = []
    for indices, normal in _CUBOID_FACES:
        for index in indices:
            verts.append(corners[index])
            norms.append(np.array(normal, dtype=np.float64))
    return np.array(verts, dtype=np.float64), np.array(norms, dtype=np.float64)


_CUBOID_WIREFRAME_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom (z=-1) face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top (z=+1) face
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
)


def tube_wireframe(start, end, radius: float, sides: int = 10) -> list[tuple[np.ndarray, np.ndarray]]:
    """A cheap wireframe cousin of tube_mesh: two end rings plus
    longitudinal connector lines, the same real shape the production
    renderer's own wireframe/preview path uses for a tube-shaped
    primitive (abstract_ui_geometry.py's part_geometry_lines, the
    "axial-structural-casing" case) -- segments, not triangles, so a
    live visualizer can draw orders of magnitude fewer primitives per
    part than the solid mesh would cost."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    axis = _normalized3(end - start)
    u, v = _perpendicular_basis(axis)

    angles = np.arange(sides + 1) * (2.0 * np.pi / sides)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    radial = np.outer(cos_a, u) + np.outer(sin_a, v)

    ring_start = start + radial * radius
    ring_end = end + radial * radius

    segments = []
    for i in range(sides):
        segments.append((ring_start[i], ring_start[i + 1]))
        segments.append((ring_end[i], ring_end[i + 1]))
    # longitudinal connectors: a handful, not one per side, plenty for
    # a wireframe read at typical on-screen sizes
    longitudinal_count = min(sides, 4)
    step = max(1, sides // longitudinal_count)
    for i in range(0, sides, step):
        segments.append((ring_start[i], ring_end[i]))
    return segments


def cuboid_wireframe(center, half_extent) -> list[tuple[np.ndarray, np.ndarray]]:
    """The 12 real edges of a box -- ported corner table (see module
    docstring), just connected as lines instead of triangulated faces."""
    center = np.asarray(center, dtype=np.float64)
    half_extent = np.asarray(half_extent, dtype=np.float64)
    corners = center + _CUBOID_CORNER_SIGNS * half_extent
    return [(corners[a], corners[b]) for a, b in _CUBOID_WIREFRAME_EDGES]


def write_obj(path: str, parts: list[tuple[np.ndarray, np.ndarray, str]]) -> None:
    """Write one .obj (with per-part `o` groups) so the ported geometry
    can actually be opened and looked at in any real 3D tool (Blender,
    an online OBJ viewer, etc.) rather than trusted blind."""
    with open(path, "w") as f:
        vertex_offset = 0
        for vertices, normals, name in parts:
            f.write(f"o {name}\n")
            for p in vertices:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            tri_count = len(vertices) // 3
            for t in range(tri_count):
                i0 = vertex_offset + t * 3 + 1
                i1 = vertex_offset + t * 3 + 2
                i2 = vertex_offset + t * 3 + 3
                f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
            vertex_offset += len(vertices)


def capped_tube_mesh(start, end, radius: float, sides: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """tube_mesh plus a fan of triangles closing each end -- a real
    disc/drum (flywheel, pulley, bulkhead, cooling fin, piston crown)
    rather than an open hoop, which is all an uncapped tube reads as
    when seen end-on."""
    start = np.asarray(start, dtype=np.float64); end = np.asarray(end, dtype=np.float64)
    vtx, nrm = tube_mesh(start, end, radius, sides=sides)
    axis = _normalized3(end - start)
    u, v = _perpendicular_basis(axis)
    verts = []; norms = []
    for centre, n_dir, flip in ((start, -axis, True), (end, axis, False)):
        for i in range(sides):
            a0 = 2.0 * np.pi * i / sides; a1 = 2.0 * np.pi * (i + 1) / sides
            p0 = centre + radius * (np.cos(a0) * u + np.sin(a0) * v)
            p1 = centre + radius * (np.cos(a1) * u + np.sin(a1) * v)
            tri = (centre, p1, p0) if flip else (centre, p0, p1)
            verts.extend(tri); norms.extend([n_dir, n_dir, n_dir])
    return (np.concatenate([vtx, np.array(verts, dtype=np.float64)]),
            np.concatenate([nrm, np.array(norms, dtype=np.float64)]))


def prism_mesh(polygon_uv, centre, axis, half_length: float, u_dir, v_dir) -> tuple[np.ndarray, np.ndarray]:
    """A polygon (2-D points in the plane spanned by u_dir/v_dir, given
    counter-clockwise) extruded +-half_length along `axis`: end caps as
    fans from the polygon centroid plus one quad per edge -- a rotor,
    a bracket, any prismatic part that isn't a box or a drum."""
    centre = np.asarray(centre, dtype=np.float64)
    axis = _normalized3(np.asarray(axis, dtype=np.float64))
    u_dir = _normalized3(np.asarray(u_dir, dtype=np.float64)); v_dir = _normalized3(np.asarray(v_dir, dtype=np.float64))
    pts = [centre + u_dir * float(a) + v_dir * float(b) for a, b in polygon_uv]
    cen = sum(pts) / len(pts)
    verts = []; norms = []
    for sign in (-1.0, 1.0):
        off = axis * (half_length * sign)
        n = axis * sign
        for i in range(len(pts)):
            a = pts[i] + off; b = pts[(i + 1) % len(pts)] + off
            tri = (cen + off, a, b) if sign > 0 else (cen + off, b, a)
            verts.extend(tri); norms.extend([n, n, n])
    for i in range(len(pts)):
        a0 = pts[i] - axis * half_length; b0 = pts[(i + 1) % len(pts)] - axis * half_length
        a1 = a0 + axis * (2 * half_length); b1 = b0 + axis * (2 * half_length)
        edge_dir = b0 - a0
        n = _normalized3(np.cross(edge_dir, axis))
        verts.extend([a0, b0, b1, a0, b1, a1]); norms.extend([n] * 6)
    return np.array(verts, dtype=np.float64), np.array(norms, dtype=np.float64)
