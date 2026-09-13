"""A per-part temperature texture with its own surface-conduction
kernel, on top of the lumped per-group temperatures the sim already
carries.

The lumped thermal circuits (`block_cyl_N`, `exhaust`, `coolant`, ...)
give every part ONE temperature. Real parts are not uniform: an exhaust
port's rim glows before the rest of the head, a turbine housing is
hottest at its inlet scroll, a bore is hottest at the top. This module
gives every part its own small 2-D temperature field (a tile in an
atlas texture), runs a diffusion kernel over it every tick -- conduction
along the surface -- with the part's lumped group temperature as the
field's relaxation target (its boundary condition) and real hot spots
injected where the graph says they are (exhaust ports, the combustion
face at the top of a bore, a turbine inlet), and writes the result into
the renderer's per-material emissive UV layer (base_material.frag.glsl:
`e_uv.a` scales the emission, `e_uv.b` its saturation) so the glow is
painted across the triangles instead of one flat value per group.

The UV mapping is a planar projection of each part onto its own two
principal axes (a tube unwraps to length x transverse; a box to its
largest face), which is exact enough for a temperature that varies
along a part far more than around it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from engine_mesh import BLACKBODY_VISIBLE_FLOOR_K, BLACKBODY_REFERENCE_K, _planck_relative, _BLACKBODY_WAVELENGTHS_UM

TILE_PX = 16                 # per-part temperature field resolution
LAYER_PX = 128               # atlas layer edge -> 64 tiles per layer
TILES_PER_ROW = LAYER_PX // TILE_PX
IDENTITY_LAYER = 0           # layer 0 is the shader's identity texel (no spatial modulation)

# conduction: a per-tick Laplacian relaxation, in tile cells; real
# surface conduction in cast iron/steel over ~10 cm parts settles in
# seconds, which is the rate a viewer sees a hot spot spread at
CONDUCTION_CELL2_PER_S = 6.0
RELAX_TO_GROUP_PER_S = 0.8   # how fast the field follows its lumped group temperature
SOURCE_RADIUS_CELLS = 2.5


@dataclass
class Tile:
    layer: int
    ix: int
    iy: int
    group: str
    origin: np.ndarray          # part centroid
    basis: np.ndarray           # (2, 3): the two projection axes
    extent: np.ndarray          # (2,) half-extents along the basis
    temp: np.ndarray = field(default_factory=lambda: np.full((TILE_PX, TILE_PX), 293.15))
    sources: list = field(default_factory=list)   # (cell_x, cell_y, delta_k)

    def uv_for(self, verts: np.ndarray, recenter: bool = False) -> np.ndarray:
        rel = verts - (verts.mean(axis=0) if recenter else self.origin)
        p = rel @ self.basis.T                       # (n, 2)
        local = 0.5 + 0.5 * p / np.maximum(self.extent, 1e-6)
        local = np.clip(local, 0.03, 0.97)           # keep inside the tile's own texels
        u = (self.ix + local[:, 0]) / TILES_PER_ROW
        v = (self.iy + local[:, 1]) / TILES_PER_ROW
        return np.stack([u, v], axis=1).astype(np.float32)

    def cell_for(self, point: np.ndarray) -> tuple[int, int]:
        p = (np.asarray(point, dtype=np.float64) - self.origin) @ self.basis.T
        local = np.clip(0.5 + 0.5 * p / np.maximum(self.extent, 1e-6), 0.0, 0.999)
        return int(local[0] * TILE_PX), int(local[1] * TILE_PX)


class ThermalAtlas:
    """Tiles for every part that carries a thermal group; one atlas layer
    per (material variant row) so each row's texstack emit_uv_layer can
    point at its own layer."""

    def __init__(self) -> None:
        self.tiles: dict[str, Tile] = {}
        self._layer_fill: dict[int, int] = {}      # layer -> tiles allocated
        self._row_layer: dict[int, int] = {}       # material variant row -> layer
        self.n_layers = 1
        self._stack = None

    def layer_for_row(self, row: int) -> int:
        layer = self._row_layer.get(row)
        if layer is None:
            layer = self.n_layers
            self.n_layers += 1
            self._row_layer[row] = layer
            self._layer_fill[layer] = 0
        return layer

    def add_part(self, name: str, group: str, row: int, verts: np.ndarray) -> Tile | None:
        if name in self.tiles:
            return self.tiles[name]
        layer = self.layer_for_row(row)
        n = self._layer_fill[layer]
        if n >= TILES_PER_ROW * TILES_PER_ROW:
            # this row's layer is full: open another layer for the same row
            layer = self.n_layers; self.n_layers += 1
            self._row_layer[row] = layer; self._layer_fill[layer] = 0; n = 0
        pts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        origin = pts.mean(axis=0)
        rel = pts - origin
        if len(pts) >= 3:
            _u, _s, vt = np.linalg.svd(rel, full_matrices=False)
            basis = vt[:2]
        else:
            basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        proj = rel @ basis.T
        extent = np.maximum(np.abs(proj).max(axis=0), 1e-4)
        tile = Tile(layer=layer, ix=n % TILES_PER_ROW, iy=n // TILES_PER_ROW, group=group,
                    origin=origin, basis=basis, extent=extent)
        self._layer_fill[layer] = n + 1
        self.tiles[name] = tile
        self._stack = None
        return tile

    def add_source(self, name: str, point, delta_k: float) -> None:
        tile = self.tiles.get(name)
        if tile is None:
            return
        cx, cy = tile.cell_for(point)
        tile.sources.append((cx, cy, float(delta_k)))
        self._stack = None

    # ---- the kernel, vectorised over every tile at once ----
    def _ensure_stack(self) -> None:
        if getattr(self, "_stack", None) is not None:
            return
        names = list(self.tiles.keys())
        n = len(names)
        temp = np.stack([self.tiles[k].temp for k in names]) if n else np.zeros((0, TILE_PX, TILE_PX))
        yy, xx = np.mgrid[0:TILE_PX, 0:TILE_PX]
        src_w = np.zeros((n, TILE_PX, TILE_PX)); src_d = np.zeros((n, TILE_PX, TILE_PX))
        for i, k in enumerate(names):
            for cx, cy, delta in self.tiles[k].sources:
                w = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * SOURCE_RADIUS_CELLS ** 2))
                src_w[i] = np.maximum(src_w[i], w); src_d[i] = np.maximum(src_d[i], delta * w)
        layers = np.array([self.tiles[k].layer for k in names], dtype=np.int64)
        ix = np.array([self.tiles[k].ix for k in names], dtype=np.int64); iy = np.array([self.tiles[k].iy for k in names], dtype=np.int64)
        # flat texel indices of every tile's 16x16 block inside its layer
        # (transposed: tile rows run along u), for one vectorised scatter
        ty, tx = np.mgrid[0:TILE_PX, 0:TILE_PX]
        gx = ix[:, None, None] * TILE_PX + ty[None]      # transposed placement (a.T)
        gy = iy[:, None, None] * TILE_PX + tx[None]
        flat = (gy * LAYER_PX + gx).reshape(n, -1)
        self._stack = {"names": names, "temp": temp, "src_w": src_w, "src_d": src_d,
                       "groups": [self.tiles[k].group for k in names], "layers": layers, "ix": ix, "iy": iy,
                       "flat": flat, "layer_col": np.repeat(layers[:, None], TILE_PX * TILE_PX, axis=1)}

    def step(self, dt: float, group_temps: dict[str, float], live_sources: dict[str, float] | None = None) -> None:
        """One conduction step on every tile at once: Laplacian diffusion
        along the surface (zero-flux edges), relaxation toward each
        tile's lumped group temperature, and the declared hot spots
        pushed above it (scaled by live_sources[group] when given --
        e.g. a cylinder's live burn strength)."""
        self._ensure_stack()
        st = self._stack
        if not st["names"]:
            return
        dt = max(1e-4, min(dt, 0.1))
        k = min(0.24, CONDUCTION_CELL2_PER_S * dt)
        r = min(1.0, RELAX_TO_GROUP_PER_S * dt)
        f = st["temp"]
        lap = np.roll(f, 1, 1) + np.roll(f, -1, 1) + np.roll(f, 1, 2) + np.roll(f, -1, 2) - 4.0 * f
        lap[:, 0, :] = f[:, 1, :] - f[:, 0, :]; lap[:, -1, :] = f[:, -2, :] - f[:, -1, :]
        lap[:, :, 0] = f[:, :, 1] - f[:, :, 0]; lap[:, :, -1] = f[:, :, -2] - f[:, :, -1]
        target = np.array([group_temps.get(g, 293.15) for g in st["groups"]])[:, None, None]
        gain = np.array([(live_sources or {}).get(g, 1.0) for g in st["groups"]])[:, None, None]
        f = f + k * lap
        f = f + (target - f) * r
        f = f + (target + st["src_d"] * gain - f) * np.minimum(1.0, st["src_w"]) * r
        st["temp"] = f
        for i, name in enumerate(st["names"]):
            self.tiles[name].temp = f[i]

    def to_rgba_layers(self) -> np.ndarray:
        """The emissive UV array: layer 0 the identity texel; every other
        texel (r=0 no directional lobe, g=255 full diffuse, b saturation,
        a intensity) from its tile's temperature -- alpha is the Planck
        red-band radiance relative to BLACKBODY_REFERENCE_K, normalised
        per layer (the row's emit_gain carries that layer's own peak)."""
        self._ensure_stack()
        st = self._stack
        arr = np.zeros((self.n_layers, LAYER_PX, LAYER_PX, 4), dtype=np.uint8)
        arr[IDENTITY_LAYER, :, :, :] = (0, 255, 128, 255)
        arr[1:, :, :, 1] = 255
        arr[1:, :, :, 2] = 128
        self.layer_gain = {}
        if not st["names"]:
            return arr
        ref = _planck_relative(_BLACKBODY_WAVELENGTHS_UM[0], BLACKBODY_REFERENCE_K)
        c2 = 14387.77 / _BLACKBODY_WAVELENGTHS_UM[0]
        t = np.maximum(st["temp"], 1.0)
        rad = np.where(t >= BLACKBODY_VISIBLE_FLOOR_K,
                       (1.0 / np.expm1(np.minimum(c2 / t, 700.0))) / (_BLACKBODY_WAVELENGTHS_UM[0] ** 5) / ref, 0.0)
        layers = st["layers"]
        # per-layer peak radiance (the row's emit_gain carries it)
        peak_by_tile = np.zeros(len(layers))
        for layer in np.unique(layers):
            sel = layers == layer
            peak = float(rad[sel].max())
            self.layer_gain[int(layer)] = peak
            peak_by_tile[sel] = peak
        safe = np.maximum(peak_by_tile, 1e-9)[:, None, None]
        a = (rad / safe * 255.0).astype(np.uint8)
        sat = np.clip(128.0 - 128.0 * np.clip((t - 1150.0) / 500.0, 0.0, 1.0), 0, 128).astype(np.uint8)
        flat_arr = arr.reshape(self.n_layers, LAYER_PX * LAYER_PX, 4)
        flat_arr[st["layer_col"], st["flat"], 3] = a.reshape(len(layers), -1)
        flat_arr[st["layer_col"], st["flat"], 2] = sat.reshape(len(layers), -1)
        return arr
