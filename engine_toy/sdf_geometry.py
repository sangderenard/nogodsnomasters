"""Mesh-derived signed-distance and volume kernels.

The expensive inside/outside classification is compiled on demand into a
small voxel grid. Runtime containment and CSG-hole updates are then array
lookups/vector operations rather than repeated triangle-ray tests.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CapsuleSdf:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius_m: float

    @classmethod
    def from_puncture(cls, puncture) -> "CapsuleSdf":
        start = np.asarray(puncture.entry_position_m, dtype=np.float64)
        if puncture.exit_position_m is not None:
            end = np.asarray(puncture.exit_position_m, dtype=np.float64)
        else:
            direction = np.asarray(puncture.direction, dtype=np.float64)
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            end = start + direction * float(puncture.penetration_depth_m)
        return cls(tuple(float(v) for v in start), tuple(float(v) for v in end), float(puncture.radius_m))

    def signed_distance(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        a = np.asarray(self.start, dtype=np.float64)
        axis = np.asarray(self.end, dtype=np.float64) - a
        axis_len_sq = max(float(np.dot(axis, axis)), 1e-18)
        along = np.clip(np.sum((p - a) * axis, axis=-1) / axis_len_sq, 0.0, 1.0)
        closest = a + along[..., None] * axis
        return np.linalg.norm(p - closest, axis=-1) - max(float(self.radius_m), 0.0)


@dataclass
class MeshSdfKernel:
    triangles: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    occupied: np.ndarray
    watertight: bool
    cutouts: list[CapsuleSdf] = field(default_factory=list)

    @classmethod
    def compile(cls, triangles, resolution: int | Sequence[int] = 28) -> "MeshSdfKernel":
        tri = np.asarray(triangles, dtype=np.float64).reshape((-1, 3, 3))
        if tri.shape[0] == 0:
            raise ValueError("cannot compile an SDF from an empty triangle mesh")
        dims = _resolution3(resolution)
        raw_min = tri.reshape((-1, 3)).min(axis=0)
        raw_max = tri.reshape((-1, 3)).max(axis=0)
        extent = np.maximum(raw_max - raw_min, 1e-6)
        bounds_min = raw_min
        bounds_max = raw_min + extent
        occupied = _voxelize_inside_x(tri, bounds_min, bounds_max, dims)
        return cls(
            triangles=tri,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            occupied=occupied,
            watertight=_is_watertight(tri),
        )

    @property
    def voxel_size_m(self) -> np.ndarray:
        return (self.bounds_max - self.bounds_min) / np.asarray(self.occupied.shape, dtype=np.float64)

    @property
    def enclosed_volume_m3(self) -> float:
        if not self.watertight:
            raise ValueError("enclosed volume requires a watertight part mesh")
        return float(np.count_nonzero(self.occupied) * np.prod(self.voxel_size_m))

    def contains(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        original_shape = p.shape[:-1]
        flat = p.reshape((-1, 3))
        rel = (flat - self.bounds_min) / (self.bounds_max - self.bounds_min)
        idx = np.floor(rel * np.asarray(self.occupied.shape)).astype(np.int64)
        valid = np.all((idx >= 0) & (idx < np.asarray(self.occupied.shape)), axis=1)
        result = np.zeros(len(flat), dtype=bool)
        if np.any(valid):
            vi = idx[valid]
            result[valid] = self.occupied[vi[:, 0], vi[:, 1], vi[:, 2]]
        return result.reshape(original_shape)

    def signed_distance(self, points) -> np.ndarray:
        """Exact triangle distance with voxel-compiled inside/outside sign."""
        p = np.asarray(points, dtype=np.float64)
        original_shape = p.shape[:-1]
        flat = p.reshape((-1, 3))
        distances = np.array([_point_mesh_distance(point, self.triangles) for point in flat])
        distances[self.contains(flat)] *= -1.0
        for cutout in self.cutouts:
            distances = np.maximum(distances, -cutout.signed_distance(flat))
        return distances.reshape(original_shape)

    def subtract_capsules(self, capsules: Iterable[CapsuleSdf]) -> float:
        """Subtract finite tool/projectile sweeps and return removed volume.

        Only voxel centres inside each capsule's AABB are evaluated.  A full
        world-sized centre tensor per tooth stroke made narrow tool paths far
        more expensive than the material they can possibly touch.
        """
        additions = list(capsules)
        if not additions:
            return 0.0
        removed_cells = 0
        spacing = self.voxel_size_m
        shape = np.asarray(self.occupied.shape, dtype=np.int64)
        for capsule in additions:
            a = np.asarray(capsule.start, dtype=np.float64)
            b = np.asarray(capsule.end, dtype=np.float64)
            radius = max(float(capsule.radius_m), 0.0)
            lower = np.minimum(a, b) - radius
            upper = np.maximum(a, b) + radius
            i0 = np.maximum(0, np.floor(
                (lower - self.bounds_min) / spacing - 0.5
            ).astype(np.int64))
            i1 = np.minimum(shape, np.ceil(
                (upper - self.bounds_min) / spacing + 0.5
            ).astype(np.int64))
            if np.any(i1 <= i0):
                continue
            axes = [
                self.bounds_min[d] + (np.arange(i0[d], i1[d]) + 0.5) * spacing[d]
                for d in range(3)
            ]
            centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            local = self.occupied[
                i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]
            ]
            remove = local & (capsule.signed_distance(centers) <= 0.0)
            removed_cells += int(np.count_nonzero(remove))
            local[remove] = False
        self.cutouts.extend(additions)
        return float(removed_cells * np.prod(spacing))


def _voxelize_inside_x(
    triangles: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    dims: tuple[int, int, int],
) -> np.ndarray:
    nx, ny, nz = dims
    spacing = (bounds_max - bounds_min) / np.asarray(dims, dtype=np.float64)
    xs = bounds_min[0] + (np.arange(nx) + 0.5) * spacing[0]
    ys = bounds_min[1] + (np.arange(ny) + 0.5) * spacing[1]
    zs = bounds_min[2] + (np.arange(nz) + 0.5) * spacing[2]
    occupied = np.zeros(dims, dtype=bool)

    direction = np.array([1.0, 0.0, 0.0])
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    determinant_ok = np.abs(det) > 1e-12
    inv_det = np.where(determinant_ok, 1.0 / np.where(determinant_ok, det, 1.0), 0.0)
    origin_x = bounds_min[0] - spacing[0]
    merge_epsilon = max(float(spacing[0]) * 1e-5, 1e-10)

    for iy, y in enumerate(ys):
        for iz, z in enumerate(zs):
            origin = np.array([origin_x, y, z])
            delta = origin - v0
            u = np.einsum("ij,ij->i", delta, pvec) * inv_det
            qvec = np.cross(delta, e1)
            v = np.einsum("j,ij->i", direction, qvec) * inv_det
            distance = np.einsum("ij,ij->i", e2, qvec) * inv_det
            mask = determinant_ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (distance > 0.0)
            crossings = np.sort(origin_x + distance[mask])
            crossings = _merge_crossings(crossings, merge_epsilon)
            if len(crossings) < 2:
                continue
            counts = np.searchsorted(crossings, xs, side="right")
            occupied[:, iy, iz] = counts % 2 == 1
    return occupied


def _point_mesh_distance(point: np.ndarray, triangles: np.ndarray) -> float:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal_len_sq = np.einsum("ij,ij->i", normal, normal)
    ap = point - a
    plane_scale = np.einsum("ij,ij->i", ap, normal) / np.maximum(normal_len_sq, 1e-18)
    projected = point - plane_scale[:, None] * normal

    v0 = ab
    v1 = ac
    v2 = projected - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denominator = d00 * d11 - d01 * d01
    bary_v = (d11 * d20 - d01 * d21) / np.maximum(denominator, 1e-18)
    bary_w = (d00 * d21 - d01 * d20) / np.maximum(denominator, 1e-18)
    inside = (bary_v >= 0.0) & (bary_w >= 0.0) & (bary_v + bary_w <= 1.0)
    plane_distance = np.abs(np.einsum("ij,ij->i", ap, normal)) / np.sqrt(np.maximum(normal_len_sq, 1e-18))
    edge_distance = np.minimum(
        _point_segment_distances(point, a, b),
        np.minimum(_point_segment_distances(point, b, c), _point_segment_distances(point, c, a)),
    )
    return float(np.min(np.where(inside, plane_distance, edge_distance)))


def _point_segment_distances(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    scale = np.einsum("ij,ij->i", point - start, segment) / np.maximum(
        np.einsum("ij,ij->i", segment, segment), 1e-18)
    closest = start + np.clip(scale, 0.0, 1.0)[:, None] * segment
    return np.linalg.norm(point - closest, axis=1)


def _voxel_centers(bounds_min: np.ndarray, bounds_max: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    spacing = (bounds_max - bounds_min) / np.asarray(shape, dtype=np.float64)
    axes = [
        bounds_min[i] + (np.arange(shape[i]) + 0.5) * spacing[i]
        for i in range(3)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _merge_crossings(values: np.ndarray, epsilon: float) -> np.ndarray:
    if len(values) < 2:
        return values
    keep = np.ones(len(values), dtype=bool)
    keep[1:] = np.diff(values) > epsilon
    return values[keep]


def _is_watertight(triangles: np.ndarray) -> bool:
    def key(point) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.round(point, decimals=9))

    edges = Counter()
    for triangle in triangles:
        vertices = [key(point) for point in triangle]
        for a, b in ((vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0])):
            edges[tuple(sorted((a, b)))] += 1
    return bool(edges) and all(count == 2 for count in edges.values())


def _resolution3(resolution: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(resolution, int):
        value = max(4, int(resolution))
        return (value, value, value)
    values = tuple(max(4, int(value)) for value in resolution)
    if len(values) != 3:
        raise ValueError("resolution must be an int or three integers")
    return values
