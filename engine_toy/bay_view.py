"""Standalone engine-bay packing viewer.

Renders the REAL vehicle chassis/firewall graph (turing's production
abstract_ui_vehicles module -- specifically fit_vehicle_chassis_to_power_unit,
the actual function the game/native rig/design tools all share for fitting
a chassis around an installed engine) together with the current engine's
own real drivetrain mesh (vehicle_mesh.build_drivetrain_solid_parts), so
the engine-bay packing problem can actually be looked at: does this
engine's own derived bounding envelope fit inside the bay the production
fitter solved for it, at the real engine_position/orientation the vehicle
graph settled on -- and does the firewall (abstract_ui_vehicles.py's new
body.firewall.* bulkhead) sit clear of it.

Deliberately independent of main.py/main_pygame.py: no audio, no physics
stepping, no sim loop, no dashboard -- a pure geometry/packing tool.
Camera and engine selection are entirely user-paced (arrow keys, [ ] to
cycle engines, O to toggle a transverse-mount test); nothing auto-animates
unless SPACE's slow auto-orbit is turned on.

Run directly: `python bay_view.py`
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pygame

from drivetrain_graph import build_drivetrain_graph
from vehicle_mesh import build_drivetrain_solid_parts
from engines import CATALOGUE
import engine_mounts

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))
from src.compiler.abstract_ui_vehicles import (  # noqa: E402
    load_default_car_configuration, fit_vehicle_chassis_to_power_unit,
    _vehicle_mechanical_graph,
)

WINDOW_W, WINDOW_H = 1100, 820
BG_COLOR = (16, 16, 20)
ENGINE_COLOR = np.array([210, 150, 70], dtype=np.float64)
CHASSIS_COLOR = (90, 140, 220)
FIREWALL_COLOR = (230, 80, 80)
BAY_COLOR = (90, 210, 130)
MOUNT_COLOR = (240, 220, 60)
CV_STUB_COLOR = (220, 120, 230)
LIGHT_DIR = np.array([0.4, 0.6, 0.7])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
AMBIENT_FLOOR = 0.30
TRANSVERSE_YAW_DEG = 90.0


def _rotate_points(points: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """The exact roll -> pitch -> yaw convention
    abstract_ui_vehicles.py's own parameter_defaults() already applies to
    the crank axis (roll about local X, "pitch" about local Z, "yaw"
    about local Y-up) -- generalized here from a single axis vector to a
    whole point cloud, so the real engine mesh can actually be shown
    mounted transverse, not just have its crank direction reported as
    such."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    y, z = y * cr - z * sr, y * sr + z * cr
    x, y = x * cp - y * sp, x * sp + y * cp
    x, z = x * cy - z * sy, x * sy + z * cy
    return np.stack([x, y, z], axis=1)


# build_drivetrain_graph's own node identities (see its printed layout)
# span well past the physical engine: the transmission/transfer-case
# chain, this toy's own dyno_absorber test rig (explicitly "no real
# vehicle would ever have one" -- see drivetrain_graph.py's module
# docstring), and the fuel tank/pump (mounted elsewhere in the vehicle,
# not in the engine bay) all live on the same crank-relative graph but
# are NOT part of what has to physically fit ahead of the firewall.
# Excluded here by identity prefix so the envelope this hands to the
# real fit_vehicle_chassis_to_power_unit means what it claims to mean:
# the engine (+ its bolted-on accessories) alone, not this toy's whole
# test bench.
_ENGINE_ENVELOPE_EXCLUDE_PREFIXES = (
    "dyno_absorber", "fuel.tank", "fuel.pump",
    "powertrain.transmission", "powertrain.transfer_case", "powertrain.direct_drive_bypass",
    "mount.transmission", "mount.transfer_case",
    # a transverse install's transaxle/final-drive/differential/halfshafts
    # are the same real cut as the transmission/transfer-case above --
    # NOT part of the engine's own physical envelope against the firewall
    "powertrain.transaxle", "powertrain.final_drive", "powertrain.differential",
    "powertrain.halfshaft_left", "powertrain.halfshaft_right",
    "mount.transaxle", "mount.torque_rod",
)


def _is_engine_body_node(identity: str) -> bool:
    return not any(identity.startswith(prefix) for prefix in _ENGINE_ENVELOPE_EXCLUDE_PREFIXES)


def _engine_body_subgraph(graph: dict) -> dict:
    keep_nodes = [n for n in graph["nodes"] if _is_engine_body_node(n["identity"])]
    keep_ids = {n["identity"] for n in keep_nodes}
    keep_edges = [e for e in graph["edges"] if e["a"] in keep_ids and e["b"] in keep_ids]
    return {**graph, "nodes": keep_nodes, "edges": keep_edges}


def engine_envelope(engine) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Real derived bounding envelopes for fit_vehicle_chassis_to_power_unit
    -- taken straight from the engine's OWN real mesh geometry (the same
    tube/cuboid parts the pygame mesh view already renders), restricted to
    the physical engine body (see _ENGINE_ENVELOPE_EXCLUDE_PREFIXES above),
    never a guessed number. Returns (engine_envelope_m, oil_pan_envelope_m),
    each (length_x, height_y, width_z), the exact shape that function
    expects."""
    graph = _engine_body_subgraph(build_drivetrain_graph(engine))
    parts = build_drivetrain_solid_parts(graph)
    all_verts = np.concatenate([p.vertices for p in parts], axis=0)
    mins, maxs = all_verts.min(axis=0), all_verts.max(axis=0)
    engine_extent = tuple(float(v) for v in (maxs - mins))
    pan_parts = [p for p in parts if p.name == "node_powertrain_oil_pan"]
    if pan_parts:
        pan_verts = pan_parts[0].vertices
        pan_mins, pan_maxs = pan_verts.min(axis=0), pan_verts.max(axis=0)
        pan_extent = tuple(float(v) for v in (pan_maxs - pan_mins))
    else:
        # No real oil-pan node at all (two-stroke total-loss lubrication
        # -- see build_drivetrain_graph's has_oil_pan gate). A thin,
        # clearly-labeled stand-in envelope rather than a fabricated pan
        # size: fit_vehicle_chassis_to_power_unit requires a positive
        # oil_pan_envelope_m regardless.
        pan_extent = (engine_extent[0] * 0.4, max(engine_extent[1] * 0.12, 0.02), engine_extent[2] * 0.4)
    return engine_extent, pan_extent, mins, maxs


def build_bay_scene(engine, transverse: bool):
    """The real packing solve: derive this engine's own envelope, hand it
    to the production fitter, then build the REAL whole-vehicle graph
    (chassis frame, firewall, everything) around the result. transverse
    only ever changes powertrain.engine_orientation_degrees on a COPY of
    the fitted configuration -- the firewall/chassis themselves never
    move, proving they were never derived from engine placement at all."""
    engine_extent, pan_extent, mesh_min, mesh_max = engine_envelope(engine)
    base_config = load_default_car_configuration()
    fitted_config, bay_info = fit_vehicle_chassis_to_power_unit(
        base_config, engine_envelope_m=engine_extent, oil_pan_envelope_m=pan_extent,
        engine_mass_kg=max(engine.mass_kg, 1.0),
    )
    source = fitted_config.source
    orientation = [0.0, 90.0 if transverse else 0.0, 0.0] if transverse else [0.0, 0.0, 0.0]
    source["powertrain"]["engine_orientation_degrees"] = orientation
    from src.compiler.abstract_ui_vehicles import vehicle_configuration_from_mapping
    fitted_config = vehicle_configuration_from_mapping(source)
    vehicle_graph = _vehicle_mechanical_graph(fitted_config)
    return {
        "vehicle_graph": vehicle_graph,
        "engine_extent": engine_extent,
        "pan_extent": pan_extent,
        "bay_info": bay_info,
        "mesh_min": mesh_min,
        "mesh_max": mesh_max,
        "engine_position": tuple(float(v) for v in source["powertrain"]["engine_position"]),
        "orientation_deg": tuple(orientation),
    }


def _node_pos(graph: dict, identity: str) -> np.ndarray | None:
    for n in graph["nodes"]:
        if n["identity"] == identity:
            return np.array(n["reference_position"], dtype=np.float64)
    return None


class BayViewer:
    def __init__(self) -> None:
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("engine bay packing viewer (standalone -- no main.py)")
        self.font = pygame.font.SysFont("consolas", 15)
        self.small_font = pygame.font.SysFont("consolas", 12)
        self.engine_index = 0
        self.transverse = False
        self.yaw = math.radians(35.0)
        self.pitch = math.radians(18.0)
        self.zoom = 1.0
        self.auto_orbit = False
        self.clock = pygame.time.Clock()
        self._rebuild()

    def _rebuild(self) -> None:
        engine = CATALOGUE[self.engine_index]
        self.engine = engine
        self.scene = build_bay_scene(engine, self.transverse)
        graph = self.scene["vehicle_graph"]
        # the real engine mesh, rotated by the SAME orientation the
        # vehicle graph's own engine_orientation_degrees carries, then
        # translated onto the vehicle's own solved engine_position --
        # two independently real coordinate systems overlaid honestly,
        # not one faked to match the other
        # Rendered mesh keeps the real transmission/transfer-case (useful
        # packaging context behind the engine), but still drops the
        # dyno_absorber test rig and the fuel tank/pump -- neither is
        # actually mounted at the crank-relative coordinates this toy
        # places them at; drawing them here would misrepresent where
        # they really live in the vehicle.
        eng_graph = {**build_drivetrain_graph(engine)}
        eng_graph["nodes"] = [n for n in eng_graph["nodes"]
                              if not n["identity"].startswith(("dyno_absorber", "fuel.tank", "fuel.pump"))]
        keep_ids = {n["identity"] for n in eng_graph["nodes"]}
        eng_graph["edges"] = [e for e in eng_graph["edges"] if e["a"] in keep_ids and e["b"] in keep_ids]
        parts = build_drivetrain_solid_parts(eng_graph)
        verts = np.concatenate([p.vertices for p in parts], axis=0)
        roll, yaw_deg, pitch_deg = self.scene["orientation_deg"]
        verts = _rotate_points(verts, math.radians(roll), math.radians(pitch_deg), math.radians(yaw_deg))
        verts = verts + np.array(self.scene["engine_position"])
        self.engine_verts = verts.reshape(-1, 3, 3)
        normals = np.concatenate([p.normals for p in parts], axis=0)
        normals = _rotate_points(normals, math.radians(roll), math.radians(pitch_deg), math.radians(yaw_deg))
        self.engine_normals = normals.reshape(-1, 3, 3)

        # The real automated mount policy (engine_mounts.assign_mounting)
        # -- not a bespoke set drawn for this viewer. Same real technique/
        # CG-in-hull/TRA machinery EnginePackage.build already runs;
        # include_transmission=True because the mesh above is drawn WITH
        # its transmission/transaxle for packaging context, so the mount
        # set has to match what's actually on screen. Positions are in
        # the engine's own local frame exactly like the mesh vertices
        # above, so the identical rotate-then-translate puts them where
        # they really are relative to the fitted vehicle.
        install_context = "automotive"
        mounting = engine_mounts.assign_mounting(
            engine, install_context=install_context, include_transmission=True,
            transmission_mass_kg=engine.transmission_mass_kg if hasattr(engine, "transmission_mass_kg") else 0.0)
        self.mounting = mounting
        mount_pos = np.array([m.position for m in mounting.mounts], dtype=np.float64) if mounting.mounts else np.zeros((0, 3))
        mount_pos = _rotate_points(mount_pos, math.radians(roll), math.radians(pitch_deg), math.radians(yaw_deg))
        self.mount_points = mount_pos + np.array(self.scene["engine_position"])
        self.mount_labels = [f"{m.role}:{m.hardware.technique.value if hasattr(m.hardware.technique, 'value') else m.hardware.technique}"
                             for m in mounting.mounts]

        # Real CV-joint attachment points: wherever the graph actually
        # declares a halfshaft (transverse installs only -- see
        # drivetrain_graph.py's transaxle block), read straight from the
        # full (untransformed-here) drivetrain graph, same as frame/
        # firewall corners above -- never invented, just picked up when
        # the graph has them.
        # A transverse install's real halfshaft nodes ARE the CV-joint
        # stubs. A longitudinal install has no halfshafts in this graph
        # at all -- it still needs exactly one real torque-output-to-
        # chassis point, though: whichever real driveline node is the
        # most downstream (transfer_case, else direct_drive_bypass, else
        # bare transmission -- the same three identities drivetrain_
        # graph.py already emits for every non-transaxle engine), same
        # real "where does torque leave this crate" question, just
        # answered by a driveshaft flange instead of a pair of CV joints.
        # Never invented geometry -- always a real existing graph node.
        full_graph = build_drivetrain_graph(engine)
        cv_ids = ("powertrain.halfshaft_left", "powertrain.halfshaft_right")
        cv_local = [_node_pos(full_graph, cid) for cid in cv_ids]
        cv_local = [p for p in cv_local if p is not None]
        if not cv_local:
            for fallback_id in ("powertrain.transfer_case", "powertrain.direct_drive_bypass", "powertrain.transmission"):
                p = _node_pos(full_graph, fallback_id)
                if p is not None:
                    cv_local = [p]
                    break
        if cv_local:
            cv_arr = _rotate_points(np.array(cv_local), math.radians(roll), math.radians(pitch_deg), math.radians(yaw_deg))
            self.cv_stub_points = cv_arr + np.array(self.scene["engine_position"])
        else:
            self.cv_stub_points = np.zeros((0, 3))

        # real chassis/firewall reference geometry straight off the
        # actual production graph -- no separate wireframe authored here
        frame_corners = [_node_pos(graph, f"frame.{c}")
                         for c in ("front_left", "front_right", "rear_right", "rear_left")]
        self.frame_loop = np.array([p for p in frame_corners if p is not None])
        fw_upper_left = _node_pos(graph, "body.firewall.upper_left")
        fw_upper_right = _node_pos(graph, "body.firewall.upper_right")
        fw_lower_left = _node_pos(graph, "body.firewall.lower_left")
        fw_lower_right = _node_pos(graph, "body.firewall.lower_right")
        self.firewall_quad = np.array([fw_upper_left, fw_upper_right, fw_lower_right, fw_lower_left])

        bay_length, bay_width = self.scene["bay_info"]["minimum_expanded_bay_m"]
        eh = self.scene["engine_extent"][1]
        pan_depth = self.scene["pan_extent"][1]
        cx, cy, cz = self.scene["engine_position"]
        hl, hh, hw = bay_length / 2.0, (eh + pan_depth) / 2.0, bay_width / 2.0
        self.bay_box = np.array([
            [cx - hl, cy - hh + pan_depth, cz - hw], [cx + hl, cy - hh + pan_depth, cz - hw],
            [cx + hl, cy - hh + pan_depth, cz + hw], [cx - hl, cy - hh + pan_depth, cz + hw],
            [cx - hl, cy + hh + pan_depth, cz - hw], [cx + hl, cy + hh + pan_depth, cz - hw],
            [cx + hl, cy + hh + pan_depth, cz + hw], [cx - hl, cy + hh + pan_depth, cz + hw],
        ])
        all_pts = np.concatenate([self.engine_verts.reshape(-1, 3), self.frame_loop, self.bay_box], axis=0)
        self.center = all_pts.mean(axis=0)
        spread = np.max(np.linalg.norm(all_pts - self.center, axis=1))
        self._base_scale = (min(WINDOW_W, WINDOW_H) * 0.36) / max(spread, 1e-6)

    def _project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = points - self.center
        cos_a, sin_a = math.cos(self.yaw), math.sin(self.yaw)
        x = local[:, 0] * cos_a - local[:, 2] * sin_a
        z = local[:, 0] * sin_a + local[:, 2] * cos_a
        cos_p, sin_p = math.cos(self.pitch), math.sin(self.pitch)
        y = local[:, 1] * cos_p - z * sin_p
        depth = local[:, 1] * sin_p + z * cos_p
        scale = self._base_scale * self.zoom
        screen_x = WINDOW_W * 0.5 + x * scale
        screen_y = WINDOW_H * 0.55 - y * scale
        return np.stack([screen_x, screen_y], axis=1), depth

    def _draw_loop(self, points_3d: np.ndarray, color: tuple[int, int, int], width: int = 2, close: bool = True) -> None:
        screen, _ = self._project(points_3d)
        pts = [(int(p[0]), int(p[1])) for p in screen]
        if close:
            pygame.draw.polygon(self.screen, color, pts, width)
        else:
            pygame.draw.lines(self.screen, color, False, pts, width)

    def _draw_box_edges(self, box: np.ndarray, color: tuple[int, int, int]) -> None:
        bottom = [0, 1, 2, 3]
        top = [4, 5, 6, 7]
        self._draw_loop(box[bottom], color, width=1)
        self._draw_loop(box[top], color, width=1)
        for i in range(4):
            self._draw_loop(box[[bottom[i], top[i]]], color, width=1, close=False)

    def _draw_engine_mesh(self) -> None:
        n_tri = self.engine_verts.shape[0]
        flat = self.engine_verts.reshape(-1, 3)
        screen, depth = self._project(flat)
        screen = screen.reshape(n_tri, 3, 2)
        depth = depth.reshape(n_tri, 3).mean(axis=1)
        nrm = self.engine_normals.reshape(-1, 3)
        cos_a, sin_a = math.cos(self.yaw), math.sin(self.yaw)
        nx = nrm[:, 0] * cos_a - nrm[:, 2] * sin_a
        nz = nrm[:, 0] * sin_a + nrm[:, 2] * cos_a
        rn = np.stack([nx, nrm[:, 1], nz], axis=1)
        rn = rn / np.maximum(np.linalg.norm(rn, axis=1, keepdims=True), 1e-9)
        bright = np.clip(rn @ LIGHT_DIR, AMBIENT_FLOOR, 1.0).reshape(n_tri, 3).mean(axis=1)
        order = np.argsort(depth)
        for idx in order:
            b = bright[idx]
            color = (min(255, int(ENGINE_COLOR[0] * b)), min(255, int(ENGINE_COLOR[1] * b)),
                     min(255, int(ENGINE_COLOR[2] * b)))
            pts = [(int(px), int(py)) for px, py in screen[idx]]
            pygame.draw.polygon(self.screen, color, pts)

    def _draw_points(self, points_3d: np.ndarray, color: tuple[int, int, int], radius: int = 6) -> None:
        if points_3d.shape[0] == 0:
            return
        screen, _ = self._project(points_3d)
        for p in screen:
            pygame.draw.circle(self.screen, color, (int(p[0]), int(p[1])), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (int(p[0]), int(p[1])), radius, 1)

    def _draw_hud(self) -> None:
        eng = self.engine
        bay = self.scene["bay_info"]
        ee = self.scene["engine_extent"]
        lines = [
            f"[{self.engine_index + 1}/{len(CATALOGUE)}] {eng.label}",
            f"engine envelope (L,H,W) m: {ee[0]:.2f} x {ee[1]:.2f} x {ee[2]:.2f}   mass {eng.mass_kg:.0f} kg",
            f"solved bay (L,W) m: {bay['minimum_expanded_bay_m'][0]:.2f} x {bay['minimum_expanded_bay_m'][1]:.2f}"
            f"   chassis half-L/W: {bay['chassis_half_length_m']:.2f} / {bay['chassis_half_width_m']:.2f}",
            f"engine position (vehicle graph): {tuple(round(v, 3) for v in self.scene['engine_position'])}"
            f"   mount: {'TRANSVERSE (yaw 90)' if self.transverse else 'longitudinal (default)'}",
            "-" * 70,
            "[ / ]  cycle engine    O  toggle transverse mount    SPACE  auto-orbit "
            + ("ON" if self.auto_orbit else "off"),
            "arrows  orbit/tilt     +/-  zoom     R  reset view    ESC/Q  quit",
            "orange = real engine mesh (this toy)   blue = real chassis frame corners"
            "   red = firewall (new)   green = solved bay envelope box",
            "yellow = engine_mounts.assign_mounting's own real mount points"
            "   magenta = real torque-output-to-chassis stub"
            " (halfshaft/CV joints if transverse, else driveshaft flange)",
        ]
        for i, text in enumerate(lines):
            surf = self.font.render(text, True, (235, 235, 235)) if i < 4 else self.small_font.render(
                text, True, (170, 170, 175))
            self.screen.blit(surf, (14, 10 + i * 20))

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.engine_index = (self.engine_index - 1) % len(CATALOGUE)
                        self._rebuild()
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.engine_index = (self.engine_index + 1) % len(CATALOGUE)
                        self._rebuild()
                    elif event.key == pygame.K_o:
                        self.transverse = not self.transverse
                        self._rebuild()
                    elif event.key == pygame.K_SPACE:
                        self.auto_orbit = not self.auto_orbit
                    elif event.key == pygame.K_r:
                        self.yaw, self.pitch, self.zoom = math.radians(35.0), math.radians(18.0), 1.0
            keys = pygame.key.get_pressed()
            orbit_speed = 1.2 * dt
            if keys[pygame.K_LEFT]:
                self.yaw -= orbit_speed
            if keys[pygame.K_RIGHT]:
                self.yaw += orbit_speed
            if keys[pygame.K_UP]:
                self.pitch = max(-1.4, self.pitch - orbit_speed)
            if keys[pygame.K_DOWN]:
                self.pitch = min(1.4, self.pitch + orbit_speed)
            if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
                self.zoom = min(4.0, self.zoom * (1.0 + dt))
            if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
                self.zoom = max(0.25, self.zoom * (1.0 - dt))
            if self.auto_orbit:
                self.yaw += 0.12 * dt  # deliberately slow -- a controlled inspection tool, not a spin cycle

            self.screen.fill(BG_COLOR)
            self._draw_box_edges(self.bay_box, BAY_COLOR)
            self._draw_loop(self.frame_loop, CHASSIS_COLOR, width=2)
            self._draw_loop(self.firewall_quad, FIREWALL_COLOR, width=3)
            self._draw_engine_mesh()
            self._draw_points(self.mount_points, MOUNT_COLOR, radius=7)
            self._draw_points(self.cv_stub_points, CV_STUB_COLOR, radius=6)
            self._draw_hud()
            pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    BayViewer().run()
