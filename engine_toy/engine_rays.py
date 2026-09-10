"""Ray interface over the final engine mesh: a ray (a click from the
camera, a projectile's trajectory) resolved to the exact parts it
passes through, in order, with entry and exit points -- and a
penetration model that spends a projectile's energy wall by wall and
leaves a real OPEN PORT in every casting it holes.

  cast(ray)            every triangle hit along the ray (Moller-Trumbore
                       over the whole mesh in one numpy pass; ~20k
                       triangles in a few ms), grouped per part into
                       traversals: entry point, exit point, thickness
                       of material crossed (a tube's wall is crossed
                       twice, once per side, and each crossing is its
                       own traversal)
  pick(ray)            the first part hit -- what a click means
  penetrate(ray, ...)  a projectile: energy is spent per traversal as
                       thickness x the part material's areal toughness;
                       every part it gets through is holed (a
                       PortHole: position, direction, calibre) and the
                       projectile stops inside the part that absorbs the
                       rest. Holes in castings are handed back as open
                       ports for assembly_ports (a holed pan leaks, a
                       holed head loses compression, a holed runner
                       leans that cylinder).
view_basis/screen_to_ray below invert mesh_visualizer.py's old CPU
shear-projection camera; that module is no longer what the live
pygame app draws with (see engine_gl_view.py, a real GL/Phong view
with a real perspective camera) -- the live app's clicks go through
EngineGLView.pick_ray instead, which inverts ITS OWN camera. These two
are kept for mesh_visualizer.py's own standalone/CPU-only use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

# areal toughness: energy to punch a 1 m^2 x 1 m thick slab -- i.e. J per m of
# thickness per m^2 of hole; disclosed order-of-magnitude figures by material
TOUGHNESS_J_PER_M3 = {
    "case": 2.5e9, "cylinder": 2.5e9, "head": 2.2e9, "cover": 0.4e9, "crank": 6.0e9, "rod": 5.0e9, "piston": 1.8e9,
    "valve": 6.0e9, "spring": 3.0e9, "follower": 4.0e9, "cam": 5.0e9, "rotor": 4.0e9,
    "intake": 0.3e9, "exhaust": 0.9e9, "fuel": 0.8e9, "lube": 0.8e9, "ignition": 0.2e9,
    "intake_port": 2.0e9, "exhaust_port": 2.0e9, "igniter": 1.0e9, "injector": 2.0e9, "casting_port": 2.0e9,
    "expander_port": 2.0e9, "other": 1.0e9,
}


# hollow parts: the mesh draws a jacket, a runner, a tank or a cover as a
# closed drum/box, but a projectile crosses two WALLS of it, not a solid
# slab -- the wall thickness that actually resists, by material (m)
SHELL_WALL_M = {
    "cylinder": 0.008, "case": 0.006, "cover": 0.0015, "intake": 0.003, "exhaust": 0.0025, "fuel": 0.0015,
    "lube": 0.002, "head": 0.010, "ignition": 0.001, "other": 0.003,
}


def resisting_thickness(material: str, crossed_m: float) -> float:
    """What a traversal of `crossed_m` through this material actually
    puts in the way: solid parts (crank, rods, valves, pistons) resist
    over their whole thickness; shell parts resist over two walls."""
    wall = SHELL_WALL_M.get(material)
    if wall is None:
        return crossed_m
    return min(crossed_m, 2.0 * wall)


@dataclass(frozen=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray      # unit

    @classmethod
    def from_points(cls, a, b) -> "Ray":
        a = np.asarray(a, dtype=np.float64); d = np.asarray(b, dtype=np.float64) - a
        return cls(a, d / max(np.linalg.norm(d), 1e-12))

    def at(self, t: float) -> np.ndarray:
        return self.origin + self.direction * t


@dataclass
class Hit:
    t: float
    point: np.ndarray
    normal: np.ndarray
    part: str
    material: str
    entering: bool          # ray goes into the part's material here


@dataclass
class Traversal:
    part: str
    material: str
    t_in: float
    t_out: float
    point_in: np.ndarray
    point_out: np.ndarray

    @property
    def thickness_m(self) -> float:
        return max(0.0, self.t_out - self.t_in)


@dataclass
class PortHole:
    part: str
    material: str
    position: np.ndarray
    direction: np.ndarray
    calibre_m: float
    through: bool           # exit hole too (went all the way through)


@dataclass
class Penetration:
    holes: list = field(default_factory=list)
    stopped_in: str | None = None
    energy_left_j: float = 0.0
    log: list = field(default_factory=list)


class RayMesh:
    """The static and moving meshes flattened for ray tests, with each
    triangle's part name and material label."""

    def __init__(self, static, moving) -> None:
        from engine_mesh import MATERIALS
        tv, tn, names, mats = [], [], [], []
        for mesh in (static, moving):
            if mesh is None or mesh.n_triangles == 0:
                continue
            tv.append(mesh.tri_vertices()); tn.append(mesh.tri_normals().mean(axis=1))
            for name, (a, b) in zip(mesh.part_names, mesh.part_ranges):
                names.extend([name] * (b - a))
            mats.extend(MATERIALS[int(i)].name for i in mesh.material_ids)
        self.tri = np.concatenate(tv) if tv else np.zeros((0, 3, 3))
        self.normal = np.concatenate(tn) if tn else np.zeros((0, 3))
        self.part = np.array(names)
        self.material = np.array(mats)
        # precompute edges
        self.v0 = self.tri[:, 0]; self.e1 = self.tri[:, 1] - self.v0; self.e2 = self.tri[:, 2] - self.v0

    @classmethod
    def from_graph(cls, graph: dict, crank_angle_deg: float = 0.0, covers_off: bool = True) -> "RayMesh":
        from engine_mesh import build_engine_mesh
        static, moving = build_engine_mesh(graph, crank_angle_deg=crank_angle_deg, covers_off=covers_off)
        return cls(static, moving)

    # ---- intersection ----
    def hits(self, ray: Ray, t_min: float = 1e-6) -> list:
        if self.tri.shape[0] == 0:
            return []
        d = ray.direction
        p = np.cross(d, self.e2)
        det = np.einsum("ij,ij->i", self.e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = ray.origin - self.v0
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, self.e1)
        v = np.einsum("j,ij->i", d, q) * inv
        t = np.einsum("ij,ij->i", self.e2, q) * inv
        mask = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > t_min)
        idx = np.nonzero(mask)[0]
        out = []
        for i in idx[np.argsort(t[idx])]:
            n = self.normal[i]
            entering = float(np.dot(n, d)) < 0.0
            out.append(Hit(t=float(t[i]), point=ray.at(float(t[i])), normal=n, part=str(self.part[i]),
                           material=str(self.material[i]), entering=entering))
        return out

    def traversals(self, ray: Ray) -> list:
        """Entry/exit pairs per part along the ray, in order. An entry
        with no matching exit (an open tube seen edge-on) closes at the
        next hit of the same part or is dropped as a graze."""
        open_: dict = {}
        out = []
        for h in self.hits(ray):
            if h.entering:
                if h.part not in open_:
                    open_[h.part] = h
            else:
                start = open_.pop(h.part, None)
                if start is not None:
                    out.append(Traversal(h.part, h.material, start.t, h.t, start.point, h.point))
        return sorted(out, key=lambda tr: tr.t_in)

    def pick(self, ray: Ray):
        hs = self.hits(ray)
        return hs[0] if hs else None

    # ---- projectile ----
    def penetrate(self, ray: Ray, energy_j: float, calibre_m: float = 0.0076, toughness=None) -> Penetration:
        """Spend a projectile's energy through successive walls: each
        traversal costs thickness x calibre-area x the part material's
        toughness; a wall it gets through is holed in and out, the one
        that absorbs the rest is holed in only and the projectile
        stops there."""
        tough = dict(TOUGHNESS_J_PER_M3); tough.update(toughness or {})
        area = math.pi * (calibre_m / 2.0) ** 2
        res = Penetration(energy_left_j=float(energy_j))
        for tr in self.traversals(ray):
            solid_m = resisting_thickness(tr.material, tr.thickness_m)
            cost = solid_m * area * tough.get(tr.material, tough["other"])
            if res.energy_left_j >= cost:
                res.energy_left_j -= cost
                res.holes.append(PortHole(tr.part, tr.material, tr.point_in, ray.direction, calibre_m, True))
                res.log.append(f"through {tr.part} ({tr.material}, {solid_m * 1000:.1f} mm of wall) -{cost:.0f} J -> {res.energy_left_j:.0f} J left")
            else:
                depth = res.energy_left_j / max(area * tough.get(tr.material, tough["other"]), 1e-9)
                res.holes.append(PortHole(tr.part, tr.material, tr.point_in, ray.direction, calibre_m, False))
                res.stopped_in = tr.part
                res.log.append(f"stopped in {tr.part} ({tr.material}) after {depth * 1000:.1f} mm of {solid_m * 1000:.1f} mm of wall")
                res.energy_left_j = 0.0
                break
        return res


def holes_as_open_ports(pen: Penetration) -> list[dict]:
    """Every hole as an open port record for the assembly/network layer:
    a leak in the part it is in, of the projectile's calibre."""
    out = []
    for h in pen.holes:
        out.append({"identity": f"{h.part}.hole_{len(out) + 1}", "part": h.part, "material": h.material,
                    "position": [float(v) for v in h.position], "direction": [float(v) for v in h.direction],
                    "radius_m": h.calibre_m / 2.0, "through": h.through, "kind": "projectile-hole", "connected": False})
    return out


# ---- camera model: the live view's projection and its inverse ----

def view_basis(angle_rad: float):
    """mesh_visualizer._project: rotate about y by angle, then y' = 0.86*y - 0.34*z,
    screen x = x', screen y = -y'. The viewing direction is the +z'
    axis (depth), which in world space is (sin a, 0, cos a) rotated back."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    # world -> view: x' = x c - z s ; z' = x s + z c
    right = np.array([c, 0.0, -s])
    forward = np.array([s, 0.0, c])       # +z' (into the screen)
    up = np.array([0.0, 1.0, 0.0])
    return right, up, forward


def screen_to_ray(px: float, py: float, viewport_w: int, viewport_h: int, center, scale: float, angle_rad: float) -> Ray:
    """The world-space ray behind a pixel of the live view (an
    orthographic camera: the view's projection has no perspective).
    The shear y' = 0.86*y - 0.34*z is inverted along the ray direction
    so the ray passes through every world point that projects onto
    the pixel."""
    right, up, forward = view_basis(angle_rad)
    x = (px - viewport_w * 0.5) / scale
    y_s = (viewport_h * 0.5 - py) / scale
    # points on the ray: P(t) = center + right*x + up*(y_s/0.86) + t*(forward + up*(0.34/0.86))
    origin = np.asarray(center) + right * x + up * (y_s / 0.86) - (forward + up * (0.34 / 0.86)) * 10.0
    direction = forward + up * (0.34 / 0.86)
    return Ray(origin, direction / np.linalg.norm(direction))
