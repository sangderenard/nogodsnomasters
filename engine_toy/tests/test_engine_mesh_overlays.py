import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine_mesh import EngineMesh, merge_engine_meshes
from station_reference import _placed_engine_mesh


def _triangle(name: str, group: str | None, x: float = 0.0) -> EngineMesh:
    return EngineMesh(
        vertices=np.array([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0],
                           [x, 1.0, 0.0]]),
        normals=np.tile([0.0, 0.0, 1.0], (3, 1)),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        material_ids=np.array([3], dtype=np.int32),
        part_names=[name], part_ranges=[(0, 1)], moving=False,
        part_groups=[group])


def test_merge_preserves_object_material_and_thermal_records():
    merged = merge_engine_meshes([
        _triangle("case", "oil"), _triangle("turbine", "exhaust", 4.0)])

    assert merged.n_triangles == 2
    assert merged.part_names == ["case", "turbine"]
    assert merged.part_ranges == [(0, 1), (1, 2)]
    assert merged.part_groups == ["oil", "exhaust"]
    assert merged.triangles.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert merged.material_ids.tolist() == [3, 3]


def test_placed_engine_mesh_namespaces_state_channels_and_only_translates():
    source = _triangle("piston", "block_cyl_1")
    placed = _placed_engine_mesh(source, "engine.port",
                                 np.array([2.0, 3.0, 4.0]))

    assert placed.part_names == ["engine.port::piston"]
    assert placed.part_groups == ["engine.port::block_cyl_1"]
    np.testing.assert_allclose(placed.vertices,
                               source.vertices + [2.0, 3.0, 4.0])
    np.testing.assert_allclose(placed.normals, source.normals)
    assert placed.material_ids.tolist() == source.material_ids.tolist()
