"""Live engine view for the pygame app: the dressed, material-coded
engine mesh (engine_mesh.py) shaded with the spectral analyzer's Phong
model on the CPU, moving parts rebuilt from the sim's live crank
angle, translucent castings so the works inside show, and a legend of
the materials on screen.

Runs entirely on a background thread once started -- start()/stop()
only touch thread lifecycle; the main thread's per-frame cost is
pull_surface() (a lock + reference read) plus one blit. The physics
tick hands the thread its readings through ThermalMeshState
(temperatures, crank angle, the per-cylinder stats shown on the
overlay); the thread never touches the sim.

Shading is the base_material.frag.glsl Phong evaluated per triangle:
  colour = albedo * (ambient + kd * max(N.L, 0)) + spec_strength * max(N.H, 0)^shininess
with the material's own ambient / spec_strength / shininess record, a
thermal tint on castings that carry a real circuit temperature, and
opacity from the material -- translucent triangles are composited
back-to-front over the opaque ones.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import pygame
import pygame.gfxdraw

from vehicle_mesh import THERMAL_GROUP_RANGE_K
from engine_mesh import build_engine_mesh, start_animation, MATERIALS
import mesh_primitives

STRUCTURAL_COLOR = (120, 120, 130)
COLD_COLOR = np.array([70, 120, 255], dtype=np.float64)
HOT_COLOR = np.array([255, 60, 40], dtype=np.float64)
LIGHT_DIR = np.array([0.45, 0.7, 0.55]); LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
VIEW_DIR = np.array([0.0, 0.0, 1.0])
BLOCK_METAL_RANGE_K = (293.15, 420.0)


class ThermalMeshState:
    """Thread-safe snapshot of what the visualizer draws from: circuit
    temperatures, the crank angle, and the stats overlay lines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.temps_k: dict[str, float] = {"coolant": 293.15, "oil": 293.15, "exhaust": 293.15, "intake": 293.15}
        self.crank_angle_deg = 0.0
        self.stats: list[str] = []

    def set_from_sim(self, sim) -> None:
        st = sim.state
        stats = [f"{st.rpm:5.0f} rpm  {st.power_kw:5.1f} kW  {st.current_torque_nm:5.0f} Nm",
                 f"coolant {st.coolant_temp_k - 273.15:4.0f} C  oil {st.oil_temp_k - 273.15:4.0f} C  exh {st.exhaust_temp_k - 273.15:4.0f} C"]
        if st.cylinder_valve_factor:
            stats.append("valve factor " + " ".join(f"{v:.3f}" for v in st.cylinder_valve_factor[:8]))
            stats.append("float rpm    " + " ".join(f"{v:5.0f}" for v in st.cylinder_float_rpm[:8]))
            stats.append("carbon %     " + " ".join(f"{v * 100:5.1f}" for v in st.cylinder_carbon_frac[:8]))
        if st.cylinder_oil_film_mg:
            stats.append(f"oil {st.sump_oil_l:.2f} L  burn {st.oil_consumption_ml_per_h:.1f} mL/h  blow-by {st.blowby_l_per_min:.2f} L/min  case {st.crankcase_pressure_kpa:.2f} kPa")
        if getattr(sim.engine, "fuel_network", None) is not None:
            stats.append(f"supply {st.fuel_supply_pressure_pa / 1e5:.2f} bar  avail {st.fuel_availability_frac * 100:3.0f}%")
        with self._lock:
            self.temps_k["coolant"] = st.coolant_temp_k
            self.temps_k["oil"] = st.oil_temp_k
            self.temps_k["exhaust"] = st.exhaust_temp_k
            self.temps_k["intake"] = st.intake_charge_temp_k
            for i, temp_k in enumerate(st.cylinder_block_temps_k):
                self.temps_k[f"block_cyl_{i + 1}"] = temp_k
            self.crank_angle_deg = float(st.crank_angle_deg)
            self.stats = stats

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self.temps_k)

    def snapshot_all(self):
        with self._lock:
            return dict(self.temps_k), self.crank_angle_deg, list(self.stats)


def _thermal_tint(group: str | None, temps_k: dict[str, float]) -> float | None:
    """0..1 cold..hot for a thermally tracked group, else None."""
    if group is None:
        return None
    if group.startswith("block_cyl_"):
        lo, hi = BLOCK_METAL_RANGE_K
    elif group in THERMAL_GROUP_RANGE_K:
        lo, hi = THERMAL_GROUP_RANGE_K[group]
    else:
        return None
    return max(0.0, min(1.0, (temps_k.get(group, lo) - lo) / max(hi - lo, 1e-6)))


def _color_for_temp(group, temps_k):
    frac = _thermal_tint(group, temps_k)
    if frac is None:
        return STRUCTURAL_COLOR
    rgb = COLD_COLOR + (HOT_COLOR - COLD_COLOR) * frac
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


class MeshVisualizer:
    def __init__(self, thermal_state: ThermalMeshState, viewport_w: int = 320, viewport_h: int = 320, fps: float = 12.0,
                 bg_color: tuple[int, int, int] = (18, 18, 22), covers_off: bool = True,
                 animation_divisions=16, detail: float = 0.5) -> None:
        """animation_divisions: how the moving parts are baked -- an
        integer count of frames evenly over the 720-degree cycle, or a
        list of crank angles in degrees. The live view then only picks
        the baked frame nearest the sim's crank angle each render, so
        it scales with rpm without re-deriving any geometry."""
        self.thermal_state = thermal_state
        self.animation_divisions = animation_divisions
        self.detail = detail          # tessellation multiplier for the live view (mesh_primitives.DETAIL)
        self._animation = None
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.bg_color = bg_color
        self.covers_off = covers_off
        self._frame_dt = 1.0 / max(fps, 1.0)
        self._lock = threading.Lock()
        self._geometry_lock = threading.Lock()
        self._pending_graph = None
        self._graph = None
        self._static = None
        self._moving = None
        self._thermal_groups: np.ndarray | None = None
        self._center = np.zeros(3)
        self._scale = 1.0
        self._angle_rad = 0.6
        self._spin = True
        self._draw_surface = pygame.Surface((viewport_w, viewport_h))
        self._ready_surface = pygame.Surface((viewport_w, viewport_h))
        self._draw_surface.fill(bg_color); self._ready_surface.fill(bg_color)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._label_font = pygame.font.SysFont("consolas", 12) if pygame.font.get_init() else None
        self.frame_ms = 0.0
        self.tri_count = 0

    # ---- lifecycle ----
    def set_graph(self, graph: dict) -> None:
        with self._geometry_lock:
            self._pending_graph = graph

    def toggle_covers(self) -> None:
        self.covers_off = not self.covers_off
        with self._geometry_lock:
            self._pending_graph = self._graph

    def set_spin(self, spin: bool) -> None:
        self._spin = spin

    def set_animation_divisions(self, divisions) -> None:
        """Re-bake the moving parts at a new set of crank divisions."""
        self.animation_divisions = divisions
        with self._geometry_lock:
            self._pending_graph = self._graph

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="mesh-visualizer", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def legend(self) -> list[tuple[str, tuple[int, int, int], float]]:
        """(label, rgb, opacity) for every material present in the view."""
        present = set()
        for mesh in (self._static, self._moving):
            if mesh is not None and mesh.n_triangles:
                present.update(int(i) for i in np.unique(mesh.material_ids))
        out = []
        for i in sorted(present):
            m = MATERIALS[i]
            out.append((m.label, tuple(int(c * 255) for c in m.albedo), m.opacity))
        return out

    # ---- geometry ----
    def _rebuild_geometry_if_pending(self) -> None:
        with self._geometry_lock:
            graph = self._pending_graph
            self._pending_graph = None
        if graph is None:
            return
        self._graph = graph
        prev = mesh_primitives.DETAIL
        mesh_primitives.set_detail(self.detail)
        try:
            static, moving = build_engine_mesh(graph, crank_angle_deg=0.0, covers_off=self.covers_off)
        finally:
            mesh_primitives.DETAIL = prev
        self._static, self._moving = static, moving
        # first frame now, the rest one per render loop (never a stall)
        self._animation = start_animation(graph, self.animation_divisions, covers_off=self.covers_off, detail=self.detail,
                                          spring_style="cylinder")
        all_v = np.concatenate([m.vertices for m in (static, moving) if m.n_triangles]) if (static.n_triangles or moving.n_triangles) else np.zeros((1, 3))
        center = all_v.mean(axis=0)
        spread = np.percentile(np.linalg.norm(all_v - center, axis=1), 92)
        self._center = center
        self._scale = (min(self.viewport_w, self.viewport_h) * 0.46) / max(spread, 1e-6)
        # thermal group per static triangle (castings carry a circuit temperature)
        groups = []
        from vehicle_mesh import build_drivetrain_solid_parts  # noqa: F401  (groups come from part names)
        for name, (a, b) in zip(static.part_names, static.part_ranges):
            g = None
            if name.startswith("cyl") and "_" in name:
                try:
                    g = f"block_cyl_{int(name[3:name.index('_')])}"
                except ValueError:
                    g = None
            elif name.startswith(("crankcase_tunnel_", "crankcase_skirt_", "head_casting")):
                g = None
            elif name == "sump" or "oil" in name:
                g = "oil"
            elif "exhaust" in name:
                g = "exhaust"
            elif "intake" in name or "runner" in name:
                g = "intake"
            groups.extend([g] * (b - a))
        self._thermal_groups = groups

    # ---- shading ----
    def _project(self, verts: np.ndarray, normals: np.ndarray):
        cos_a, sin_a = math.cos(self._angle_rad), math.sin(self._angle_rad)
        local = verts - self._center
        x = local[:, 0] * cos_a - local[:, 2] * sin_a
        z = local[:, 0] * sin_a + local[:, 2] * cos_a
        y = local[:, 1] * 0.86 - z * 0.34
        sx = self.viewport_w * 0.5 + x * self._scale
        sy = self.viewport_h * 0.5 - y * self._scale
        nx = normals[:, 0] * cos_a - normals[:, 2] * sin_a
        nz = normals[:, 0] * sin_a + normals[:, 2] * cos_a
        n = np.stack([nx, normals[:, 1], nz], axis=1)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        return sx, sy, z, n

    def _shade(self, mesh, temps_k, groups=None):
        """Returns (screen (T,3,2), depth (T,), rgba (T,4) 0..255) for a mesh."""
        if mesh is None or mesh.n_triangles == 0:
            return None
        tv = mesh.tri_vertices().reshape(-1, 3)
        tn = mesh.tri_normals().reshape(-1, 3)
        sx, sy, z, n = self._project(tv, tn)
        T = mesh.n_triangles
        screen = np.stack([sx, sy], axis=1).reshape(T, 3, 2)
        depth = z.reshape(T, 3).mean(axis=1)
        nf = n.reshape(T, 3, 3).mean(axis=1)
        nf = nf / np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-9)
        # opaque closed parts: back faces are culled; translucent and
        # thin/open parts are drawn two-sided with the normal flipped
        facing = nf @ VIEW_DIR
        nf = nf * np.where(facing < 0.0, -1.0, 1.0)[:, None]
        ndotl = np.clip(nf @ LIGHT_DIR, 0.0, 1.0)
        h = LIGHT_DIR + VIEW_DIR; h = h / np.linalg.norm(h)
        ndoth = np.clip(nf @ h, 0.0, 1.0)
        mats = mesh.material_ids
        albedo = np.array([MATERIALS[i].albedo for i in mats])
        ambient = np.array([MATERIALS[i].ambient for i in mats])
        spec_s = np.array([MATERIALS[i].spec_strength for i in mats])
        shin = np.array([MATERIALS[i].shininess for i in mats])
        opacity = np.array([MATERIALS[i].opacity for i in mats])
        if groups is not None and len(groups) == T:
            tint = np.array([_thermal_tint(g, temps_k) if g is not None else -1.0 for g in groups])
            has = tint >= 0.0
            if has.any():
                tcol = (COLD_COLOR + (HOT_COLOR - COLD_COLOR) * tint[has, None]) / 255.0
                albedo[has] = albedo[has] * 0.55 + tcol * 0.45
        diffuse = ambient[:, None] + 0.85 * ndotl[:, None]
        spec = spec_s * np.power(ndoth, shin)
        rgb = np.clip(albedo * diffuse + spec[:, None], 0.0, 1.0) * 255.0
        rgba = np.concatenate([rgb, (opacity * 255.0)[:, None]], axis=1).astype(np.int32)
        keep = ~((facing < 0.0) & (opacity >= 0.98))
        return screen[keep], depth[keep], rgba[keep]

    def _render_frame(self) -> None:
        temps_k, crank_deg, stats = self.thermal_state.snapshot_all()
        t0 = time.perf_counter()
        self._draw_surface.fill(self.bg_color)
        if self._animation is not None and self._animation.n_frames:
            self._moving = self._animation.frame_for(crank_deg)
        batches = []
        s = self._shade(self._static, temps_k, self._thermal_groups)
        if s is not None: batches.append(s)
        m = self._shade(self._moving, temps_k)
        if m is not None: batches.append(m)
        if batches:
            screen = np.concatenate([b[0] for b in batches]); depth = np.concatenate([b[1] for b in batches])
            rgba = np.concatenate([b[2] for b in batches])
            opaque = rgba[:, 3] >= 250
            order = np.argsort(depth)
            surf = self._draw_surface
            # opaque first (painter's order), then translucent back-to-front over them
            for pass_mask in (opaque, ~opaque):
                for idx in order:
                    if not pass_mask[idx]:
                        continue
                    pts = screen[idx]
                    c = rgba[idx]
                    ipts = [(int(px), int(py)) for px, py in pts]
                    if c[3] >= 250:
                        pygame.draw.polygon(surf, (int(c[0]), int(c[1]), int(c[2])), ipts)
                    else:
                        pygame.gfxdraw.filled_polygon(surf, ipts, (int(c[0]), int(c[1]), int(c[2]), int(c[3])))
            self.tri_count = int(len(order))
        if self._label_font is not None:
            y = 4
            for line in stats[:8]:
                self._draw_surface.blit(self._label_font.render(line, True, (230, 230, 235)), (6, y)); y += 14
            n_fr = self._animation.n_frames if self._animation is not None else 0
            self._draw_surface.blit(self._label_font.render(f"crank {crank_deg % 720:5.0f} deg  frame {self._animation.frame_index(crank_deg) + 1 if n_fr else 0}/{n_fr}  {self.tri_count} tris  {self.frame_ms:.0f} ms  [O covers {'off' if self.covers_off else 'on'}]", True, (150, 150, 160)), (6, self.viewport_h - 16))
        self.frame_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._draw_surface, self._ready_surface = self._ready_surface, self._draw_surface

    def _run(self) -> None:
        last_t = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            self._rebuild_geometry_if_pending()
            if self._animation is not None:
                self._animation.bake_next()
            if self._spin:
                self._angle_rad = (self._angle_rad + dt * 0.25) % (2.0 * math.pi)
            if self._static is not None:
                self._render_frame()
            elapsed = time.perf_counter() - now
            time.sleep(max(0.0, self._frame_dt - elapsed))

    def render_once(self, crank_angle_deg: float | None = None, bake_all: bool = True) -> pygame.Surface:
        """Synchronous single frame (tests / headless exports)."""
        self._rebuild_geometry_if_pending()
        if bake_all and self._animation is not None:
            while self._animation.bake_next():
                pass
        if crank_angle_deg is not None:
            with self.thermal_state._lock:
                self.thermal_state.crank_angle_deg = crank_angle_deg
        self._render_frame()
        return self.pull_surface()

    def pull_surface(self) -> pygame.Surface:
        with self._lock:
            return self._ready_surface
