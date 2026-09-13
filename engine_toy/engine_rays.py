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
  penetrate(ray, ...)  a projectile: each traversal resolves velocity,
                       incidence angle, yaw/tumble, projected area,
                       material strength/ductility, deformation and any
                       contained-fluid drag. Every part it gets through
                       is holed and the projectile stops, breaks up or
                       ricochets when the resolved state says it must.
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

from ballistics import (
    FluidLayer,
    ImpactResult,
    MATERIAL_PROFILES,
    ProjectileState,
    inferred_fluid,
    material_profile,
    resolve_impact,
    resisting_thickness as ballistic_resisting_thickness,
)

TOUGHNESS_J_PER_M3 = {name: profile.toughness_j_m3 for name, profile in MATERIAL_PROFILES.items()}
SHELL_WALL_M = {
    name: profile.shell_wall_m
    for name, profile in MATERIAL_PROFILES.items()
    if profile.shell_wall_m is not None
}

def resisting_thickness(material: str, crossed_m: float) -> float:
    """What a traversal of `crossed_m` through this material actually
    puts in the way: solid parts (crank, rods, valves, pistons) resist
    over their whole thickness; shell parts resist over two walls."""
    return ballistic_resisting_thickness(material_profile(material), crossed_m)


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
    normal_in: np.ndarray | None = None

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
    exit_position: np.ndarray | None = None
    resisting_thickness_m: float = 0.0
    penetration_depth_m: float = 0.0
    energy_before_j: float = 0.0
    energy_spent_j: float = 0.0
    damage_mode: str = "puncture"
    incidence_angle_deg: float = 0.0
    speed_before_m_s: float = 0.0
    speed_after_m_s: float = 0.0
    projectile_integrity: float = 1.0
    fluid_energy_spent_j: float = 0.0


@dataclass
class Penetration:
    holes: list = field(default_factory=list)
    stopped_in: str | None = None
    energy_left_j: float = 0.0
    log: list = field(default_factory=list)
    impacts: list[tuple[str, ImpactResult]] = field(default_factory=list)
    projectile: ProjectileState | None = None


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
        self._sdf_cache = {}
        # precompute edges
        self.v0 = self.tri[:, 0]; self.e1 = self.tri[:, 1] - self.v0; self.e2 = self.tri[:, 2] - self.v0

    def part_sdf(self, part: str, resolution: int = 28):
        """Compile and cache a vertex-derived SDF/volume kernel for one part."""
        from sdf_geometry import MeshSdfKernel

        key = (part, int(resolution))
        kernel = self._sdf_cache.get(key)
        if kernel is None:
            triangles = self.tri[self.part == part]
            if not len(triangles):
                raise KeyError(f"mesh has no part {part!r}")
            kernel = MeshSdfKernel.compile(triangles, resolution)
            self._sdf_cache[key] = kernel
        return kernel

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
                    out.append(Traversal(h.part, h.material, start.t, h.t, start.point, h.point, start.normal))
        return sorted(out, key=lambda tr: tr.t_in)

    def pick(self, ray: Ray):
        """Return only the nearest surface hit without materializing all hits."""
        if self.tri.shape[0] == 0:
            return None
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
        mask = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6)
        indices = np.nonzero(mask)[0]
        if indices.size == 0:
            return None
        i = int(indices[np.argmin(t[indices])])
        distance = float(t[i])
        normal = self.normal[i]
        return Hit(
            t=distance,
            point=ray.at(distance),
            normal=normal,
            part=str(self.part[i]),
            material=str(self.material[i]),
            entering=float(np.dot(normal, d)) < 0.0,
        )

    # ---- projectile ----
    def penetrate(
        self,
        ray: Ray,
        energy_j: float,
        calibre_m: float = 0.0076,
        toughness=None,
        *,
        projectile_mass_kg: float = 0.0095,
        yaw_rad: float = 0.0,
        tumble_rad_s: float = 0.0,
        projectile: ProjectileState | None = None,
        fluid_by_part=None,
    ) -> Penetration:
        """Spend a projectile's energy through successive walls: each
        traversal costs thickness x calibre-area x the part material's
        toughness; a wall it gets through is holed in and out, the one
        that absorbs the rest is holed in only and the projectile
        stops there."""
        tough = dict(TOUGHNESS_J_PER_M3)
        tough.update(toughness or {})
        state = projectile or ProjectileState.from_energy(
            energy_j,
            calibre_m,
            mass_kg=projectile_mass_kg,
            direction=ray.direction,
            yaw_rad=yaw_rad,
            tumble_rad_s=tumble_rad_s,
        )
        res = Penetration(energy_left_j=state.energy_j, projectile=state)
        for tr in self.traversals(ray):
            if state.energy_j <= 0.0:
                break
            profile = material_profile(tr.material, tough.get(tr.material, tough["other"]))
            normal = tr.normal_in if tr.normal_in is not None else -ray.direction
            incidence_cos = max(0.15, abs(float(np.dot(ray.direction, normal))))
            normal_crossed_m = tr.thickness_m * incidence_cos
            normal_resisting_m = ballistic_resisting_thickness(profile, normal_crossed_m)
            fluid_path_m = max(0.0, tr.thickness_m - normal_resisting_m / incidence_cos)
            if fluid_by_part is None:
                fluid = inferred_fluid(tr.part, tr.material)
            elif callable(fluid_by_part):
                fluid = fluid_by_part(tr.part, tr.material)
            else:
                fluid = fluid_by_part.get(tr.part)
            impact = resolve_impact(
                state,
                profile,
                normal_resisting_m,
                normal,
                fluid=fluid,
                fluid_path_m=fluid_path_m,
            )
            res.impacts.append((tr.part, impact))
            state = impact.projectile_after
            res.projectile = state
            res.energy_left_j = state.energy_j
            if impact.ricocheted:
                res.stopped_in = tr.part
                res.log.append(
                    f"ricochet from {tr.part} ({tr.material}) at {impact.incidence_angle_deg:.0f} deg "
                    f"{impact.speed_before_m_s:.0f}->{impact.speed_after_m_s:.0f} m/s")
                break
            hole = PortHole(
                tr.part,
                tr.material,
                tr.point_in,
                ray.direction,
                impact.diameter_after_m,
                impact.perforated,
                exit_position=tr.point_out if impact.perforated else None,
                resisting_thickness_m=impact.effective_thickness_m,
                penetration_depth_m=impact.penetration_depth_m,
                energy_before_j=impact.energy_before_j,
                energy_spent_j=impact.energy_spent_j,
                damage_mode=impact.mode.value,
                incidence_angle_deg=impact.incidence_angle_deg,
                speed_before_m_s=impact.speed_before_m_s,
                speed_after_m_s=impact.speed_after_m_s,
                projectile_integrity=impact.integrity_after,
                fluid_energy_spent_j=impact.fluid_energy_spent_j,
            )
            res.holes.append(hole)
            fluid_note = f", fluid -{impact.fluid_energy_spent_j:.0f} J" if impact.fluid_energy_spent_j > 0.0 else ""
            if impact.perforated:
                res.log.append(
                    f"through {tr.part} ({tr.material}, {impact.incidence_angle_deg:.0f} deg, "
                    f"{impact.effective_thickness_m * 1000:.1f} mm) "
                    f"{impact.speed_before_m_s:.0f}->{impact.speed_after_m_s:.0f} m/s"
                    f"{fluid_note}, {impact.mode.value}")
                if state.energy_j <= 0.0:
                    res.stopped_in = f"{tr.part} contents"
                    break
                if impact.projectile_disrupted:
                    res.stopped_in = f"{tr.part} (projectile breakup)"
                    res.log.append(f"projectile broke up after {tr.part}; no intact penetrator continues")
                    break
            else:
                res.stopped_in = tr.part
                res.log.append(
                    f"stopped in {tr.part} ({tr.material}, {impact.incidence_angle_deg:.0f} deg) "
                    f"after {impact.penetration_depth_m * 1000:.1f} mm of "
                    f"{impact.effective_thickness_m * 1000:.1f} mm at {impact.speed_before_m_s:.0f} m/s, "
                    f"{impact.mode.value}")
                break
        return res


def holes_as_open_ports(pen: Penetration) -> list[dict]:
    """Through-holes as entry/exit ports for the future network layer."""
    out = []
    for h in pen.holes:
        if not h.through or h.exit_position is None:
            continue
        for side, position, direction in (
            ("entry", h.position, -h.direction),
            ("exit", h.exit_position, h.direction),
        ):
            out.append({"identity": f"{h.part}.hole_{len(out) + 1}_{side}", "part": h.part, "material": h.material,
                        "position": [float(v) for v in position], "direction": [float(v) for v in direction],
                        "radius_m": h.calibre_m / 2.0, "through": True, "kind": "projectile-hole",
                        "connected": False})
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
