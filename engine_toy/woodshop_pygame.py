"""First-person Python woodshop using the existing Pluck material shader.

The simulation remains :class:`woodshop.WoodshopSimulation`: this module is
only the Living Data-style player view over its geometry state. Machine parts
are handed to ``BaseGLRenderer`` and therefore use the same
``base_material.vert/frag.glsl`` program as the other engine views. Camera
motion follows the Living Data map's captured-mouse, WASD, perspective view.

Run from ``engine_toy`` with ``python woodshop_pygame.py``.

Controls: mouse looks; WASD moves; E picks up the crosshair target; 1..0
equips the right hand; Shift+1..0 equips the left; mouse buttons use the
corresponding hand; Q/Shift+Q drops; F5 saves; F9 restores; Escape quits.
"""
from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path

import numpy as np

from woodshop import WoodshopSimulation


WINDOW = (1280, 760)
EYE_HEIGHT_M = 1.62
NEAR_M = 0.04
FAR_M = 128.0
# Living Data's camera uses tangentHalfFov=0.70.
FOV_Y_RAD = 2.0 * math.atan(0.70)
MOUSE_SENSITIVITY = 0.0025


def camera_facing(yaw: float, pitch: float) -> np.ndarray:
    """Living Data first-person facing vector in render-world coordinates."""
    horizontal = math.cos(float(pitch))
    return np.asarray((math.sin(float(yaw)) * horizontal,
                       math.sin(float(pitch)),
                       math.cos(float(yaw)) * horizontal), dtype=np.float32)


def camera_floor_right(facing: np.ndarray) -> np.ndarray:
    """Shader screen-right projected into the Machine simulation's XY plane."""
    flat_forward = np.asarray((facing[0], facing[2]), dtype=np.float32)
    flat_forward /= max(float(np.linalg.norm(flat_forward)), 1.0e-12)
    return np.asarray((-flat_forward[1], flat_forward[0]), dtype=np.float32)


def _look_at(eye: np.ndarray, facing: np.ndarray) -> np.ndarray:
    forward = np.asarray(facing, dtype=np.float32)
    forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
    up = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    camera_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(right, eye)
    matrix[1, 3] = -np.dot(camera_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def _perspective(aspect: float) -> np.ndarray:
    f = 1.0 / math.tan(FOV_Y_RAD * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / max(float(aspect), 1.0e-12)
    matrix[1, 1] = f
    matrix[2, 2] = (FAR_M + NEAR_M) / (NEAR_M - FAR_M)
    matrix[2, 3] = 2.0 * FAR_M * NEAR_M / (NEAR_M - FAR_M)
    matrix[3, 2] = -1.0
    return matrix


def _sim_to_render(point) -> np.ndarray:
    """Machine XYZ (Z up) to the shader world's XYZ (Y up)."""
    return np.asarray((float(point[0]), float(point[2]), float(point[1])),
                      dtype=np.float32)


def _ray_box_distance(origin: np.ndarray, direction: np.ndarray,
                      lower: np.ndarray, upper: np.ndarray) -> float | None:
    """Nearest nonnegative intersection with one render-world AABB."""
    t_min, t_max = -math.inf, math.inf
    for axis in range(3):
        if abs(float(direction[axis])) < 1.0e-12:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return None
            continue
        a = float((lower[axis] - origin[axis]) / direction[axis])
        b = float((upper[axis] - origin[axis]) / direction[axis])
        if a > b:
            a, b = b, a
        t_min, t_max = max(t_min, a), min(t_max, b)
        if t_min > t_max:
            return None
    if t_max < 0.0:
        return None
    return max(0.0, t_min)


def targeted_object(sim: WoodshopSimulation, eye: np.ndarray,
                    facing: np.ndarray, reach_m: float = 4.0) -> str | None:
    """Return the first world Machine hit by the centre-camera ray."""
    hits: list[tuple[float, str]] = []
    for identity, item in sim.items.items():
        if item.custody != "world":
            continue
        pivot_sim = item.center_xyz()
        rotation_sim = item.rotation_matrix()
        inverse = rotation_sim.T
        origin_sim = np.asarray((eye[0], eye[2], eye[1]), dtype=float)
        direction_sim = np.asarray((facing[0], facing[2], facing[1]), dtype=float)
        local_origin = pivot_sim + inverse @ (origin_sim - pivot_sim)
        local_direction = inverse @ direction_sim
        for part in item.sim.machine.parts:
            center = np.asarray(part.position, dtype=float)
            half = np.asarray((part.half_extent_m[0], part.half_extent_m[2],
                               part.half_extent_m[1]), dtype=np.float32)
            # Ray-box uses render coordinates; after inverse object rotation,
            # swap the simulation's Y/Z axes in the same way as the renderer.
            local_origin_r = _sim_to_render(local_origin)
            local_direction_r = _sim_to_render(local_direction)
            center_r = _sim_to_render(center)
            distance = _ray_box_distance(
                local_origin_r, local_direction_r, center_r - half, center_r + half)
            if distance is not None and distance <= reach_m:
                hits.append((distance, identity))
    return min(hits)[1] if hits else None


def aim_on_shop_plane(eye: np.ndarray, facing: np.ndarray,
                      fallback_distance_m: float = 2.0) -> tuple[float, float]:
    """Centre ray expressed on the Machine simulation's XY work plane."""
    if facing[1] < -1.0e-5:
        distance = -float(eye[1]) / float(facing[1])
        hit = eye + facing * max(0.0, distance)
    else:
        horizontal = np.asarray((facing[0], 0.0, facing[2]), dtype=np.float32)
        horizontal /= max(float(np.linalg.norm(horizontal)), 1.0e-12)
        hit = eye + horizontal * float(fallback_distance_m)
    return float(hit[0]), float(hit[2])


class GeometryState:
    """Triangle state derived directly from the current live Machine parts."""

    def __init__(self, material_ids: dict[str, int]):
        self.material_ids = material_ids
        self.positions: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []
        self.materials: list[int] = []

    def triangle(self, a, b, c, normal, material: str) -> None:
        for point in (a, b, c):
            self.positions.append(np.asarray(point, dtype=np.float32))
            self.normals.append(np.asarray(normal, dtype=np.float32))
            self.materials.append(self.material_ids[material])

    def quad(self, a, b, c, d, normal, material: str) -> None:
        self.triangle(a, b, c, normal, material)
        self.triangle(a, c, d, normal, material)

    def box(self, center, half, material: str,
            rotation: np.ndarray | None = None) -> None:
        c = np.asarray(center, dtype=np.float32)
        h = np.maximum(np.asarray(half, dtype=np.float32), 0.00035)
        r = np.eye(3, dtype=np.float32) if rotation is None else np.asarray(rotation)
        def point(signs):
            return c + r @ (h * np.asarray(signs, dtype=np.float32))
        faces = (
            ((-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1),(0,0,1)),
            ((1,-1,-1),(-1,-1,-1),(-1,1,-1),(1,1,-1),(0,0,-1)),
            ((1,-1,1),(1,-1,-1),(1,1,-1),(1,1,1),(1,0,0)),
            ((-1,-1,-1),(-1,-1,1),(-1,1,1),(-1,1,-1),(-1,0,0)),
            ((-1,1,1),(1,1,1),(1,1,-1),(-1,1,-1),(0,1,0)),
            ((-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,1),(0,-1,0)),
        )
        for a, b, d, e, normal in faces:
            self.quad(point(a), point(b), point(d), point(e), r @ normal, material)

    def beam(self, a, b, width: float, material: str) -> None:
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1.0e-9:
            return
        x_axis = direction / length
        helper = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        if abs(float(np.dot(x_axis, helper))) > 0.92:
            helper = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        z_axis = np.cross(x_axis, helper)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        self.box((a + b) * 0.5, (length * 0.5, width * 0.5, width * 0.5),
                 material, rotation)

    def gimbal(self, center, radius: float = 0.24) -> None:
        center = np.asarray(center, dtype=np.float32)
        for axes, material in (((1, 2), "gimbal-x"),
                               ((0, 2), "gimbal-y"),
                               ((0, 1), "gimbal-z")):
            points = []
            for index in range(33):
                angle = math.tau * index / 32.0
                point = center.copy()
                point[axes[0]] += radius * math.cos(angle)
                point[axes[1]] += radius * math.sin(angle)
                points.append(point)
            for a, b in zip(points, points[1:]):
                self.beam(a, b, 0.008, material)

    def ribbon(self, a_sim, b_sim, width: float, material: str,
               height_offset: float = 0.001) -> None:
        a, b = _sim_to_render(a_sim), _sim_to_render(b_sim)
        a[1] += height_offset
        b[1] += height_offset
        direction = b - a
        horizontal = np.asarray((direction[2], 0.0, -direction[0]), dtype=np.float32)
        horizontal /= max(float(np.linalg.norm(horizontal)), 1.0e-12)
        side = horizontal * max(float(width), 0.001) * 0.5
        self.quad(a - side, b - side, b + side, a + side,
                  (0, 1, 0), material)

    def rows(self) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(self.positions, dtype=np.float32)
        normals = np.asarray(self.normals, dtype=np.float32)
        uv = np.zeros((len(positions), 2), dtype=np.float32)
        rows = np.ascontiguousarray(np.concatenate((positions, normals, uv), axis=1))
        return rows, np.asarray(self.materials, dtype=np.int32)


def woodshop_geometry(sim: WoodshopSimulation, material_ids: dict[str, int]) \
        -> tuple[np.ndarray, np.ndarray]:
    """Resolve the current object geometry state for the existing shader."""
    state = GeometryState(material_ids)
    state.box((0.0, -0.035, 0.0), (8.0, 0.025, 8.0), "shop-floor")
    for item in sim.items.values():
        if item.custody not in {"world", "inventory"}:
            continue
        pivot_sim = item.center_xyz()
        rotation_sim = item.rotation_matrix()
        # Axis permutation from simulation XYZ/Z-up to render XYZ/Y-up.
        swap = np.asarray(((1,0,0),(0,0,1),(0,1,0)), dtype=np.float32)
        rotation_render = swap @ rotation_sim @ swap
        for part in item.sim.machine.parts:
            role = part.part_role
            material = (
                "pine" if role in {"workpiece", "material-product"}
                else "steel" if role in {"blade", "bar", "rail", "screw"}
                else "handle" if role in {"handle", "jaw-pad"}
                else "steel"
            )
            posed_center = pivot_sim + rotation_sim @ (
                np.asarray(part.position, dtype=float) - pivot_sim)
            center = _sim_to_render(posed_center)
            half = (part.half_extent_m[0], part.half_extent_m[2],
                    part.half_extent_m[1])
            state.box(center, half, material, rotation_render)
        for edge in item.sim.machine.work_edges:
            owner = next(part for part in item.sim.machine.parts
                         if part.identity == edge.owner)
            a0 = np.asarray(owner.position) + np.asarray(edge.a_local)
            b0 = np.asarray(owner.position) + np.asarray(edge.b_local)
            a = pivot_sim + rotation_sim @ (a0 - pivot_sim)
            b = pivot_sim + rotation_sim @ (b0 - pivot_sim)
            state.ribbon(a, b, edge.thickness_m + 2.0 * edge.wave_amplitude_m,
                         "cut-edge")
    for kerf in sim.cutting.state.kerfs:
        state.ribbon(kerf.actual_a, kerf.actual_b, kerf.width_m, "kerf", 0.002)
    if sim.active_drop is not None:
        state.gimbal(_sim_to_render(sim.items[sim.active_drop.identity].center_xyz()))
    return state.rows()


class DynamicShaderMesh:
    """Persistent VAO receiving the latest geometry state."""

    def __init__(self) -> None:
        from OpenGL import GL
        self.gl = GL
        self.vao = int(GL.glGenVertexArrays(1))
        self.buffers = tuple(int(value) for value in GL.glGenBuffers(5))
        self.count = 0
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[0])
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(3)
        GL.glVertexAttribPointer(3, 2, GL.GL_FLOAT, GL.GL_FALSE, 32, ctypes.c_void_p(24))
        for buffer_id, location in zip(self.buffers[1:4], (2, 4, 5)):
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribIPointer(location, 1, GL.GL_INT, 4, ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[4])
        GL.glEnableVertexAttribArray(6)
        GL.glVertexAttribPointer(6, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)

    def update(self, rows: np.ndarray, material_ids: np.ndarray) -> None:
        gl = self.gl
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        count = len(rows)
        columns = (np.ascontiguousarray(material_ids, dtype=np.int32),
                   np.full(count, -1, dtype=np.int32),
                   np.ones(count, dtype=np.int32),
                   np.zeros((count, 3), dtype=np.float32))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffers[0])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, rows.nbytes, rows, gl.GL_DYNAMIC_DRAW)
        for buffer_id, values in zip(self.buffers[1:], columns):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_id)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, values.nbytes, values, gl.GL_DYNAMIC_DRAW)
        self.count = count

    def close(self) -> None:
        self.gl.glDeleteBuffers(len(self.buffers), self.buffers)
        self.gl.glDeleteVertexArrays(1, (self.vao,))


def _renderer_modules():
    spectral = Path(__file__).resolve().parents[1] / "spectral-analyzer"
    if str(spectral) not in sys.path:
        sys.path.insert(0, str(spectral))
    from base_gl_renderer import BaseGLRenderer
    from material_db import MaterialDatabase
    from ordinary_gl_mesh_viewer import _OpenGLTextPanel
    return BaseGLRenderer, MaterialDatabase, _OpenGLTextPanel


def _material_database(MaterialDatabase):
    database = MaterialDatabase()
    definitions = {
        "shop-floor": ((0.20, 0.23, 0.24), 0.88, 0.02),
        "pine": ((0.64, 0.40, 0.18), 0.72, 0.01),
        "steel": ((0.46, 0.50, 0.52), 0.27, 0.78),
        "handle": ((0.28, 0.11, 0.045), 0.65, 0.01),
        "cut-edge": ((0.82, 0.86, 0.88), 0.20, 0.85),
        "kerf": ((0.78, 0.035, 0.018), 0.48, 0.06),
        "gimbal-x": ((0.95, 0.06, 0.04), 0.25, 0.20),
        "gimbal-y": ((0.08, 0.92, 0.20), 0.25, 0.20),
        "gimbal-z": ((0.08, 0.30, 1.00), 0.25, 0.20),
    }
    indices = {}
    for name, (albedo, roughness, metallic) in definitions.items():
        indices[name] = database.register(name, {
            "albedo_rgb": albedo, "roughness": roughness,
            "metallic": metallic, "ior": 1.48, "opacity": 1.0,
            "emission_rgb": (0.0, 0.0, 0.0),
        })
    return database, indices


def main() -> None:
    import pygame
    from OpenGL import GL

    BaseGLRenderer, MaterialDatabase, OpenGLTextPanel = _renderer_modules()
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                    pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode(WINDOW, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption("Engine Toy · first-person Python woodshop")
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    database, material_ids = _material_database(MaterialDatabase)
    renderer = BaseGLRenderer(database, auto_drain=False)
    renderer.init_gl()
    renderer.set_point_lights(
        np.asarray(((2.5, 4.5, -2.5), (-3.0, 2.6, 3.5)), np.float32),
        np.asarray(((1.0, 0.91, 0.78), (0.42, 0.55, 0.78)), np.float32),
        np.asarray((34.0, 15.0), np.float32))
    mesh = DynamicShaderMesh()
    panel = OpenGLTextPanel()
    sim = WoodshopSimulation()
    player = np.asarray((0.0, -2.0), dtype=np.float32)
    yaw, pitch = 0.0, -0.36
    messages = ["Centre the crosshair on an object and press E to pick it up."]
    clock = pygame.time.Clock()
    frame = 0

    def say(message: str) -> None:
        messages.append(message)
        del messages[:-4]

    def start_hand_action(hand: str, aim_xy: tuple[float, float]) -> None:
        fit = sim.begin_sleeve_fit_action(hand, aim_xy)
        if fit is not None:
            say(f"inserted stock {fit.penetration_m * 1000.0:.0f} mm; "
                f"holding force {fit.holding_force_n:.0f} N")
            return
        kerf = sim.begin_saw_action(hand, aim_xy)
        if kerf is not None:
            error_mm = (kerf.actual_a[0] - kerf.intended_a[0]) * 1000.0
            say(f"{hand} saw support {kerf.support_quality:.2f}; line error {error_mm:+.1f} mm")

    running = True
    try:
        while running:
            dt = min(clock.tick(120) / 1000.0, 1.0 / 30.0)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    if sim.active_drop is not None and event.buttons[0]:
                        # Left-drag: roll/pitch the object in the two visible
                        # gimbal axes. A vertical ~quarter-window drag turns a
                        # flat stud onto either edge.
                        sim.adjust_drop((float(event.rel[1]) * 0.35, 0.0,
                                         float(event.rel[0]) * 0.35))
                    elif sim.active_drop is not None and event.buttons[2]:
                        sim.adjust_drop((0.0, float(event.rel[1]) * 0.35, 0.0))
                    elif sim.active_drop is None:
                        # The Living Data / Pluck camera basis defines
                        # screen-right as cross(forward, up). With +Z forward
                        # that is -X, hence the negative yaw input.
                        yaw -= float(event.rel[0]) * MOUSE_SENSITIVITY
                        pitch = min(1.35, max(-1.35,
                                             pitch - float(event.rel[1]) * MOUSE_SENSITIVITY))
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if sim.active_drop is not None:
                            sim.cancel_drop()
                            say("drop placement cancelled")
                        else:
                            running = False
                    elif event.key == pygame.K_F5:
                        try:
                            path = sim.save_world()
                            say(f"world saved: {path}")
                        except Exception as error:
                            say(f"save failed: {error}")
                    elif event.key == pygame.K_F9:
                        try:
                            path = sim.load_world()
                            say(f"world restored: {path}")
                        except FileNotFoundError:
                            say("no saved woodshop world yet")
                        except Exception as error:
                            say(f"restore failed: {error}")
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        dropped = sim.confirm_drop()
                        if dropped:
                            say(f"placed {sim.items[dropped].sim.machine.label}")
                    elif event.key == pygame.K_e:
                        eye = np.asarray((player[0], EYE_HEIGHT_M, player[1]), np.float32)
                        target = targeted_object(sim, eye, camera_facing(yaw, pitch))
                        if target:
                            result = sim.try_put_in_hand(target)
                            if result is None:
                                say("both hands are occupied")
                            else:
                                hand, slot = result
                                say(f"picked up {sim.items[target].sim.machine.label} -> {hand}, slot {slot}")
                    elif event.key == pygame.K_q:
                        if sim.active_drop is not None:
                            dropped = sim.confirm_drop()
                            if dropped:
                                say(f"placed {sim.items[dropped].sim.machine.label}")
                        else:
                            hand = "left" if event.mod & pygame.KMOD_SHIFT else "right"
                            forward = camera_facing(yaw, 0.0)
                            drop_xy = (float(player[0] + forward[0] * 0.9),
                                       float(player[1] + forward[2] * 0.9))
                            placement = sim.begin_drop(hand, drop_xy)
                            if placement:
                                say("drop gimbal: left-drag roll/yaw; right-drag pitch; Enter/Q place")
                    elif pygame.K_0 <= event.key <= pygame.K_9:
                        slot = 10 if event.key == pygame.K_0 else event.key - pygame.K_0
                        if slot in sim.hotbar.slots:
                            hand = "left" if event.mod & pygame.KMOD_SHIFT else "right"
                            sim.hotbar.equip(hand, slot)
                            say(f"slot {slot} equipped in {hand} hand")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if sim.active_drop is not None:
                        continue
                    eye = np.asarray((player[0], EYE_HEIGHT_M, player[1]), np.float32)
                    aim_xy = aim_on_shop_plane(eye, camera_facing(yaw, pitch))
                    if event.button == 1:
                        start_hand_action("left", aim_xy)
                    elif event.button == 3:
                        start_hand_action("right", aim_xy)
                elif (event.type == pygame.MOUSEBUTTONUP and event.button in (1, 3)
                      and sim.active_drop is None):
                    sim.release_saw_action()

            facing = camera_facing(yaw, pitch)
            flat_forward = np.asarray((facing[0], facing[2]), dtype=np.float32)
            flat_forward /= max(float(np.linalg.norm(flat_forward)), 1.0e-12)
            # Same horizontal projection as the shader's
            # cross(forward, up) screen-right vector.
            flat_right = camera_floor_right(facing)
            keys = pygame.key.get_pressed()
            motion = (flat_forward * (float(keys[pygame.K_w]) - float(keys[pygame.K_s]))
                      + flat_right * (float(keys[pygame.K_d]) - float(keys[pygame.K_a])))
            length = float(np.linalg.norm(motion))
            if length:
                player += motion / length * 1.8 * dt

            eye = np.asarray((player[0], EYE_HEIGHT_M, player[1]), np.float32)
            aim_xy = aim_on_shop_plane(eye, facing)
            sim.sync_hands(tuple(float(v) for v in player), aim_xy)
            sim.advance(dt)
            rows, row_materials = woodshop_geometry(sim, material_ids)
            mesh.update(rows, row_materials)

            width, height = pygame.display.get_surface().get_size()
            GL.glViewport(0, 0, width, height)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
            GL.glClearColor(0.035, 0.055, 0.072, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            view = _look_at(eye, facing)
            projection = _perspective(width / max(height, 1))
            # BaseGLRenderer consumes column-major GL memory; transpose at the
            # shader boundary while retaining normal numpy matrices above.
            mv_gl = np.ascontiguousarray(view.T, dtype=np.float32)
            mvp_gl = np.ascontiguousarray((projection @ view).T, dtype=np.float32)
            renderer.draw_mesh(mesh.vao, mesh.count, mvp_gl, mv_gl, enable_blend=False)

            if frame % 4 == 0:
                held_left = sim.hotbar.held("left") or "empty"
                held_right = sim.hotbar.held("right") or "empty"
                slots = "  ".join(
                    f"{slot % 10}:{sim.items[item].sim.machine.label}"
                    for slot, item in sorted(sim.hotbar.slots.items())) or "hotbar empty"
                panel.update(["FIRST-PERSON WOODSHOP · existing Pluck shader",
                              f"left:  {held_left}", f"right: {held_right}", slots,
                              "WASD · mouse look · E pickup · Shift+number left hand",
                              "F5 save world · F9 restore world",
                              ("DROP GIMBAL: L-drag roll/yaw · R-drag pitch · Enter/Q place · Esc cancel"
                               if sim.active_drop is not None else
                               "Q drops right · Shift+Q drops left"),
                              *messages])
            panel.draw((width, height))

            # Crosshair without introducing another scene shader.
            GL.glEnable(GL.GL_SCISSOR_TEST)
            GL.glScissor(width // 2 - 7, height // 2 - 1, 15, 3)
            GL.glClearColor(0.95, 0.76, 0.25, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glScissor(width // 2 - 1, height // 2 - 7, 3, 15)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glDisable(GL.GL_SCISSOR_TEST)
            pygame.display.flip()
            frame += 1
    finally:
        panel.close()
        mesh.close()
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        pygame.quit()


if __name__ == "__main__":
    main()
