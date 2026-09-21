"""Project one physical circular port onto voxel boundary faces.

The port remains one object.  Face shares are a derived spatial view whose
areas sum to the physical aperture, including when the aperture crosses cell
corners or edges.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VoxelFaceShare:
    voxel: int
    area_m2: float
    area_fraction: float
    normal_axis: int
    normal_sign: int


def project_circular_port(
    position_m: Sequence[float],
    direction: Sequence[float],
    radius_m: float,
    *,
    shape: Sequence[int],
    bounds_min_m: Sequence[float],
    bounds_max_m: Sequence[float],
    grid_world_axes: Sequence[int] = (0, 2, 1),
    samples_per_axis: int = 128,
) -> tuple[VoxelFaceShare, ...]:
    """Return area shares on an axis-aligned voxel-volume boundary.

    ``grid_world_axes`` maps grid x/y/z to world axes.  Engine-toy's world is
    x/right, y/up, z/depth while the chamber grid is x/depth/up, hence the
    default ``(0, 2, 1)``.
    """
    shape = tuple(int(v) for v in shape)
    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError("shape must contain three positive extents")
    axes = tuple(int(v) for v in grid_world_axes)
    if sorted(axes) != [0, 1, 2]:
        raise ValueError("grid_world_axes must be a permutation of 0,1,2")
    radius = float(radius_m)
    if radius <= 0.0:
        return ()

    world_position = np.asarray(position_m, dtype=float)
    world_direction = np.asarray(direction, dtype=float)
    grid_position = world_position[list(axes)]
    grid_direction = world_direction[list(axes)]
    lo = np.asarray(bounds_min_m, dtype=float)[list(axes)]
    hi = np.asarray(bounds_max_m, dtype=float)[list(axes)]
    spacing = (hi - lo) / np.asarray(shape, dtype=float)

    normal_axis = int(np.argmax(np.abs(grid_direction)))
    if abs(grid_direction[normal_axis]) < 1e-12:
        distances = np.minimum(abs(grid_position - lo), abs(grid_position - hi))
        normal_axis = int(np.argmin(distances))
    normal_sign = 1 if abs(grid_position[normal_axis] - hi[normal_axis]) <= abs(
        grid_position[normal_axis] - lo[normal_axis]) else -1
    boundary_index = shape[normal_axis] - 1 if normal_sign > 0 else 0
    tangential = tuple(axis for axis in range(3) if axis != normal_axis)

    # Topology changes are rare, so a deterministic disk quadrature is both
    # cheaper than per-tick geometry and able to represent one small aperture
    # straddling any number of faces.  Normalization below preserves the exact
    # physical area even though the overlap fractions are quadrature estimates.
    n = max(16, int(samples_per_axis))
    q = (np.arange(n, dtype=float) + 0.5) * (2.0 / n) - 1.0
    uu, vv = np.meshgrid(q, q, indexing="ij")
    inside = uu * uu + vv * vv <= 1.0
    u = grid_position[tangential[0]] + radius * uu[inside]
    v = grid_position[tangential[1]] + radius * vv[inside]
    iu = np.floor((u - lo[tangential[0]]) / spacing[tangential[0]]).astype(int)
    iv = np.floor((v - lo[tangential[1]]) / spacing[tangential[1]]).astype(int)
    valid = ((iu >= 0) & (iu < shape[tangential[0]])
             & (iv >= 0) & (iv < shape[tangential[1]]))
    if not np.any(valid):
        return ()
    pairs, counts = np.unique(np.stack((iu[valid], iv[valid]), axis=1),
                              axis=0, return_counts=True)
    fractions = counts.astype(float) / float(np.sum(counts))
    area = math.pi * radius * radius
    shares = []
    for pair, fraction in zip(pairs, fractions):
        index = [0, 0, 0]
        index[normal_axis] = boundary_index
        index[tangential[0]] = int(pair[0])
        index[tangential[1]] = int(pair[1])
        flat = index[0] + shape[0] * (index[1] + shape[1] * index[2])
        shares.append(VoxelFaceShare(
            flat, area * float(fraction), float(fraction),
            normal_axis, normal_sign))
    return tuple(shares)
