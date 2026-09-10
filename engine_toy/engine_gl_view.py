"""The engine mesh drawn by ACTUAL hardware: vertex/triangle/material
buffers uploaded to the GPU once, and the spectral analyzer's real
Phong fragment shader (spectral-analyzer/csrc/shaders/base_material)
draws them every frame via ordinary GL draw calls -- no per-frame CPU
projection, no per-triangle Python loop, no software polygon fill.

This replaces the CPU path in mesh_visualizer.py (which projected and
shaded every triangle in numpy and then called pygame.draw.polygon /
gfxdraw once per triangle -- ~150 ms of Python-level fill for 8000
triangles, the actual bottleneck; the numpy Phong evaluation itself
was only ~35 ms and the shading math was never the problem). Here the
"animation" is what was asked for: every baked crank-angle frame is
uploaded to its own GPU vertex array ONCE, and playing the animation
is choosing which already-uploaded VAO to hand to the shader this
tick -- an index and two draw calls, not a re-derivation.

Everything a triangle needs (position, normal, material id) comes
straight from engine_mesh.EngineMesh; the material table
(engine_mesh.material_table()) is fed to BaseGLRenderer through a
five-line adapter that satisfies its only real contract
(`build_tensors()` returning the PBR/Phong/enamel/texture-stack
chunks) -- the same SSBO layout base_material.frag.glsl reads,
verified against spectral-analyzer/ordinary_gl_mesh_viewer.py's own
reference VAO layout (position/normal/uv interleaved buffer 0;
material id, group id, cull-immune as separate int buffers 2/4/5;
velocity buffer 6 -- matching base_material.vert.glsl's attribute
locations exactly).

A GL context is a THREAD property: every call in this module must run
on the thread that owns the context pygame created (normally the
main thread, since that's where pygame.display.set_mode(OPENGL) is
called) -- there is no background render thread here, unlike the CPU
MeshVisualizer. That is fine: with the CPU work gone, there is nothing
left to move off the main thread.
"""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_SPECTRAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spectral-analyzer")
if _SPECTRAL_DIR not in sys.path:
    sys.path.insert(0, _SPECTRAL_DIR)

from engine_mesh import EngineMesh, MATERIALS, material_table, build_engine_mesh, start_animation, CYCLE_DEG

GL_OK = False
try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_INT, GL_FALSE,
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24,
        GL_RENDERBUFFER, GL_RGBA8, GL_TEXTURE_2D, GL_FRAMEBUFFER_COMPLETE,
        GL_DEPTH_TEST, GL_LEQUAL, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, GL_CLAMP_TO_EDGE,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
        glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer, glBufferData,
        glEnableVertexAttribArray, glVertexAttribPointer, glVertexAttribIPointer,
        glDeleteVertexArrays, glDeleteBuffers,
        glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D, glFramebufferRenderbuffer,
        glGenRenderbuffers, glBindRenderbuffer, glRenderbufferStorage, glCheckFramebufferStatus,
        glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
        glViewport, glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        glEnable, glDisable, glDepthFunc, glReadPixels, glDeleteFramebuffers, glDeleteRenderbuffers,
        glDeleteTextures,
    )
    GL_OK = True
except Exception:
    GL_OK = False

if GL_OK:
    from base_gl_renderer import BaseGLRenderer


# ---------------------------------------------------------------------
# Material adapter: the only contract BaseGLRenderer needs from a
# "material_db" is build_tensors() -> the four SSBO chunks.
# ---------------------------------------------------------------------

class _StaticMaterialDB:
    def __init__(self) -> None:
        t = material_table()
        n = len(t["names"])
        self._tensors = {
            "pbr": t["pbr"].astype(np.float32),
            "phong_compat": t["phong"].astype(np.float32),
            "enamel": np.zeros((n, 8), dtype=np.float32),          # thickness_nm=0 -> enamel coat skipped
            "texture_stack": np.zeros((n, 16), dtype=np.float32),  # all-zero row = identity UV stack (no-op)
        }

    def build_tensors(self) -> dict:
        return self._tensors


# ---------------------------------------------------------------------
# GPU upload: one VAO per EngineMesh, in the exact attribute layout
# base_material.vert.glsl declares (aPos=0, aNorm=1, aMatId=2, aUv=3,
# aGroupId=4, aCullImmune=5, aVelocity=6).
# ---------------------------------------------------------------------

@dataclass
class GLMesh:
    vao: int
    buffers: tuple
    n_vertices: int

    def delete(self) -> None:
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(len(self.buffers), list(self.buffers))


def upload_mesh(mesh: EngineMesh) -> GLMesh | None:
    """Non-indexed triangle soup: 3 vertices per triangle, matching
    EngineMesh.tri_vertices()/tri_normals() directly -- no index
    buffer needed, and it keeps flat per-triangle material ids trivial
    to expand to one value per vertex (repeat 3x)."""
    if not GL_OK:
        raise RuntimeError("PyOpenGL is not available")
    n_tri = mesh.n_triangles
    if n_tri == 0:
        return None
    pos = mesh.tri_vertices().reshape(-1, 3).astype(np.float32)
    nrm = mesh.tri_normals().reshape(-1, 3).astype(np.float32)
    uv = np.zeros((pos.shape[0], 2), dtype=np.float32)
    rows = np.ascontiguousarray(np.concatenate([pos, nrm, uv], axis=1), dtype=np.float32)
    mat_id = np.repeat(mesh.material_ids.astype(np.int32), 3)
    n = rows.shape[0]

    vao = int(glGenVertexArrays(1))
    buffers = tuple(int(b) for b in glGenBuffers(5))
    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
    glBufferData(GL_ARRAY_BUFFER, rows.nbytes, rows, GL_STATIC_DRAW)
    stride = rows.shape[1] * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(3)
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

    int_columns = (mat_id, np.zeros(n, np.int32), np.ones(n, np.int32))   # matId, groupId, cullImmune
    for buf_id, location, values in zip(buffers[1:4], (2, 4, 5), int_columns):
        glBindBuffer(GL_ARRAY_BUFFER, buf_id)
        glBufferData(GL_ARRAY_BUFFER, values.nbytes, values, GL_STATIC_DRAW)
        glEnableVertexAttribArray(location)
        glVertexAttribIPointer(location, 1, GL_INT, 4, ctypes.c_void_p(0))

    velocity = np.zeros((n, 3), np.float32)
    glBindBuffer(GL_ARRAY_BUFFER, buffers[4])
    glBufferData(GL_ARRAY_BUFFER, velocity.nbytes, velocity, GL_STATIC_DRAW)
    glEnableVertexAttribArray(6)
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    glBindVertexArray(0)
    return GLMesh(vao=vao, buffers=buffers, n_vertices=n)


# ---------------------------------------------------------------------
# Offscreen target (FBO) -- so this drops into the existing pygame
# 2D dashboard by one glReadPixels + one Surface blit, same call
# shape as the old CPU view's pull_surface(), without requiring the
# whole app window to run in OPENGL mode.
# ---------------------------------------------------------------------

class OffscreenTarget:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.color_tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.color_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_tex, 0)
        self.depth_rb = int(glGenRenderbuffers(1))
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_rb)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"engine_gl_view: incomplete FBO (status {status})")

    def read_rgba(self) -> np.ndarray:
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        buf = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 4)
        return arr[::-1]   # GL's origin is bottom-left; pygame's is top-left

    def delete(self) -> None:
        glDeleteFramebuffers(1, [self.fbo])
        glDeleteRenderbuffers(1, [self.depth_rb])
        glDeleteTextures([self.color_tex])


def _look_at(eye, target, up):
    f = target - eye; f = f / np.linalg.norm(f)
    s = np.cross(f, up); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye); m[1, 3] = -np.dot(u, eye); m[2, 3] = np.dot(f, eye)
    return m


def _ortho(half_w, half_h, near, far):
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 1.0 / half_w
    m[1, 1] = 1.0 / half_h
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


def _perspective(fov_y_rad, aspect, near, far):
    """Real perspective (foreshortening depth cues, unlike the flat
    orthographic view this replaced -- an object edge-on to an
    orthographic camera visually collapses to a line as it rotates,
    which read as 'warped/flattened'; perspective keeps a sense of
    depth through the whole turn)."""
    f = 1.0 / np.tan(fov_y_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


@dataclass
class EngineGLView:
    """Owns everything GPU-side for one engine's live view: the
    material SSBOs (uploaded once), the static mesh VAO (uploaded
    once), and EVERY baked animation frame as its own VAO (uploaded
    once each, incrementally as bake_next() is called) -- playing the
    animation from then on costs an index lookup and two draw calls,
    never a re-upload.
    """
    width: int
    height: int
    covers_off: bool = True
    animation_divisions: int = 16
    spin_rad_per_s: float = 0.0

    def __post_init__(self) -> None:
        if not GL_OK:
            raise RuntimeError("engine_gl_view requires PyOpenGL and an active GL context")
        self._db = _StaticMaterialDB()
        self._renderer = BaseGLRenderer(self._db)
        self._renderer.init_gl()
        self._target = OffscreenTarget(self.width, self.height)
        self._static_gl: GLMesh | None = None
        self._frame_gl: list = []
        self._animation = None
        self._graph = None
        self._materials_present: set = set()
        self._angle_rad = 0.6
        self._center = np.zeros(3, dtype=np.float32)
        self._scale = 1.0
        self._half_extents = np.ones(3, dtype=np.float32)
        # two real point emitters (key + fill), world-space -- draw_mesh's
        # own mv transforms them to view space each call, so they track
        # the camera correctly as it orbits; there is no scene-wide
        # ambient in this shader (the material's own bounded Phong
        # ambient term is the indirect-light floor), so without these
        # the render is genuinely all-ambient and looks flat
        self._renderer.set_point_lights(
            positions=np.zeros((2, 3), dtype=np.float32),
            colors=np.array([[1.0, 0.98, 0.94], [0.55, 0.62, 0.75]], dtype=np.float32),
            intensities=np.array([3.2, 1.1], dtype=np.float32),
        )

    # ---- geometry: uploaded once per graph / per baked frame ----
    def set_graph(self, graph: dict) -> None:
        self._graph = graph
        self._free_geometry()
        static_mesh, _moving0 = build_engine_mesh(graph, crank_angle_deg=0.0, covers_off=self.covers_off)
        self._static_gl = upload_mesh(static_mesh)
        self._materials_present = set(int(i) for i in np.unique(static_mesh.material_ids)) if static_mesh.n_triangles else set()
        if _moving0.n_triangles:
            self._materials_present |= set(int(i) for i in np.unique(_moving0.material_ids))
        # frame on the BOUNDING BOX midpoint, not a vertex-density-
        # weighted mean -- an engine with more triangles at one end
        # (a flywheel, a dense head casting) has its mean pulled off
        # the true visual center, which is what put the model in "the
        # wrong spot" in the viewport. Both the static castings AND
        # the moving parts (a flywheel, a freewheel assembly, a
        # crosshead frame) count toward the box, since those can
        # extend past the static geometry's own extent.
        parts = [m.vertices for m in (static_mesh, _moving0) if m.n_triangles]
        all_v = np.concatenate(parts) if parts else np.zeros((1, 3))
        box_min = all_v.min(axis=0)
        box_max = all_v.max(axis=0)
        center = (box_min + box_max) / 2.0
        radius = float(np.linalg.norm(box_max - box_min)) / 2.0
        self._center = center.astype(np.float32)
        self._scale = max(radius, 1e-6)
        # half-extents along the object's own axes, for PER-AZIMUTH
        # framing: a straight-six is far longer along its crank axis
        # than it is tall/wide, so sizing the camera off one isotropic
        # radius (the box diagonal) either wastes most of the frame on
        # a 3/4 view or -- worse -- shows almost nothing when the long
        # axis swings edge-on to the camera during the turntable spin.
        # _fit_distance below re-derives the actual on-screen extent
        # for whatever azimuth is current instead.
        self._half_extents = ((box_max - box_min) / 2.0).astype(np.float32)
        self._animation = start_animation(graph, self.animation_divisions, covers_off=self.covers_off,
                                          spring_style="cylinder")
        self._frame_gl = [None] * self._animation.n_frames
        self._upload_pending_frames(limit=1)

    def _free_geometry(self) -> None:
        if self._static_gl is not None:
            self._static_gl.delete(); self._static_gl = None
        for g in self._frame_gl:
            if g is not None:
                g.delete()
        self._frame_gl = []

    def bake_next(self) -> bool:
        """Advance the incremental bake by one frame: derive the next
        moving-parts frame AND upload it to its own VAO immediately --
        after this, that frame never touches the CPU again."""
        if self._animation is None:
            return False
        had_more = self._animation.bake_next()
        self._upload_pending_frames(limit=1)
        return had_more or any(g is None for g in self._frame_gl)

    def _upload_pending_frames(self, limit: int) -> None:
        uploaded = 0
        for i, frame in enumerate(self._animation.frames if self._animation else []):
            if frame is not None and self._frame_gl[i] is None:
                self._frame_gl[i] = upload_mesh(frame)
                uploaded += 1
                if uploaded >= limit:
                    break

    def _frame_index_for(self, crank_angle_deg: float) -> int | None:
        if self._animation is None:
            return None
        angles = self._animation.angles_deg
        a = float(crank_angle_deg) % CYCLE_DEG
        d = np.abs((angles - a + CYCLE_DEG / 2.0) % CYCLE_DEG - CYCLE_DEG / 2.0)
        d = np.where([g is None for g in self._frame_gl], np.inf, d)
        if not np.isfinite(d).any():
            return None
        return int(np.argmin(d))

    # ---- render: two draw calls, no per-frame geometry work ----
    def render(self, crank_angle_deg: float, spin: bool = True, dt: float = 0.0) -> np.ndarray:
        """Same draw as render_gpu(), but reads the result back to a
        numpy array (a CPU round trip) -- for headless PNG export and
        the test harness; the live app uses render_gpu() instead, which
        never leaves the GPU."""
        self._draw(crank_angle_deg, spin=spin, dt=dt)
        rgba = self._target.read_rgba()
        return rgba

    def render_gpu(self, crank_angle_deg: float, spin: bool = True, dt: float = 0.0) -> int:
        """Same draw as render(), but leaves the result sitting in the
        FBO's own colour texture and returns its GL texture id --
        no glReadPixels, no CPU round trip at all. The compositor
        (gl_compositor.py) draws that texture id straight into the
        main window's backbuffer as a quad; the pixels never visit
        Python. This is the path main_pygame.py uses every tick."""
        self._draw(crank_angle_deg, spin=spin, dt=dt)
        return self._target.color_tex

    def _draw(self, crank_angle_deg: float, spin: bool, dt: float) -> None:
        if spin:
            self._angle_rad = (self._angle_rad + dt * 0.25) % (2.0 * np.pi)
        glBindFramebuffer(GL_FRAMEBUFFER, self._target.fbo)
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glClearColor(18 / 255.0, 18 / 255.0, 22 / 255.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # a real perspective camera, refit EVERY frame to the object's
        # ACTUAL on-screen silhouette at the current azimuth -- not one
        # fixed distance sized off the worst-case diagonal. A straight
        # six is far longer along its crank axis than it is tall, so a
        # single isotropic distance either wastes most of the frame on
        # a 3/4 view or nearly loses the object when the long axis
        # swings toward the camera; this instead takes the real 3D
        # bounding-box corners, transforms them through the real
        # camera's own right/up basis vectors (an actual 3D
        # projection, not a 2D approximation), and sizes the distance
        # off THAT measured extent -- the geometry is exactly the same
        # every frame, only the framing adapts to it.
        fov_y = np.radians(32.0)
        aspect = self.width / max(self.height, 1)
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        az = float(self._angle_rad)
        dir_to_eye = np.array([np.sin(az), 0.40, np.cos(az)], dtype=np.float64)
        dir_to_eye = dir_to_eye / np.linalg.norm(dir_to_eye)
        forward = -dir_to_eye
        right = np.cross(forward, up_world); right = right / max(np.linalg.norm(right), 1e-9)
        cam_up = np.cross(right, forward)

        hx, hy, hz = (float(v) for v in self._half_extents)
        cx, cy, cz = (float(v) for v in self._center)
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        corners = np.array([cx, cy, cz]) + signs * np.array([hx, hy, hz])
        rel = corners - np.array([cx, cy, cz])
        half_w = float(np.max(np.abs(rel @ right))) or 1e-6
        half_h = float(np.max(np.abs(rel @ cam_up))) or 1e-6
        # fit both the vertical extent (against fov_y) and the
        # horizontal extent (against the derived horizontal fov)
        fit_v = half_h / np.tan(fov_y / 2.0)
        fit_h = half_w / np.tan(fov_y / 2.0) / aspect
        distance = max(fit_v, fit_h) * 1.25 + 1e-4

        eye = (np.array([cx, cy, cz]) + dir_to_eye * distance).astype(np.float32)
        view = _look_at(eye, self._center, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        depth_extent = float(np.max(np.abs(rel @ forward))) + self._scale * 0.25
        proj = _perspective(fov_y, aspect, max(distance - depth_extent, distance * 0.02), distance + depth_extent)
        mv = view
        mvp = proj @ view
        self._last_eye, self._last_view, self._last_proj = eye, view, proj
        # BaseGLRenderer.draw_mesh hands these straight to
        # glUniformMatrix4fv(..., transpose=GL_FALSE, ...), which reads
        # the raw bytes as COLUMN-major; a plain numpy array built with
        # the ordinary M @ v (column-vector) convention is ROW-major in
        # memory, so passing it unmodified silently uploads the
        # TRANSPOSE of the intended matrix -- a real, different linear
        # map, not just a relabelling (confirmed with an independent
        # three-axis test: the untransposed path visibly compressed
        # and skewed the geometry; transposing here reproduced the
        # hand-computed screen positions exactly). Transpose right
        # before the GL call; every other use of view/proj/mvp in this
        # class (pick_ray, the light rig) keeps the untransposed,
        # ordinary-numpy-convention versions.
        mv_gl = np.ascontiguousarray(mv.T, dtype=np.float32)
        mvp_gl = np.ascontiguousarray(mvp.T, dtype=np.float32)

        key = eye + np.array([-0.3, 0.5, -0.2], dtype=np.float32) * self._scale
        fill = self._center - (eye - self._center) * 0.6 + np.array([0.0, self._scale, 0.0], dtype=np.float32)
        self._renderer.set_point_lights(
            positions=np.stack([key, fill]).astype(np.float32),
            colors=np.array([[1.0, 0.98, 0.94], [0.55, 0.62, 0.75]], dtype=np.float32),
            intensities=np.array([self._scale * 3.5, self._scale * 1.4], dtype=np.float32),
        )
        if self._static_gl is not None:
            self._renderer.draw_mesh(self._static_gl.vao, self._static_gl.n_vertices, mvp_gl, mv_gl, enable_blend=True)
        idx = self._frame_index_for(crank_angle_deg)
        if idx is not None and self._frame_gl[idx] is not None:
            fg = self._frame_gl[idx]
            self._renderer.draw_mesh(fg.vao, fg.n_vertices, mvp_gl, mv_gl, enable_blend=True)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def pick_ray(self, px: float, py: float):
        """The world-space ray behind pixel (px, py) of this view's own
        viewport, inverting the SAME perspective camera render()/
        render_gpu() just used (eye/view/proj cached from the last
        draw) -- what a click on the composited quad actually means in
        world space, for engine_rays.RayMesh.pick/penetrate. Unlike an
        orthographic camera, a perspective ray's ORIGIN is the eye for
        every pixel and the DIRECTION is what varies across the frame."""
        from engine_rays import Ray
        view = getattr(self, "_last_view", None)
        proj = getattr(self, "_last_proj", None)
        eye = getattr(self, "_last_eye", None)
        if view is None:
            return None
        ndc_x = (px / self.width) * 2.0 - 1.0
        ndc_y = 1.0 - (py / self.height) * 2.0
        inv_proj = np.linalg.inv(proj.astype(np.float64))
        inv_view = np.linalg.inv(view.astype(np.float64))
        # a point on the far clip plane at this pixel, unprojected to
        # world space; eye -> that point is the ray direction
        clip_far = np.array([ndc_x, ndc_y, 1.0, 1.0])
        p_view = inv_proj @ clip_far
        p_view = p_view / p_view[3]
        p_world = inv_view @ np.array([p_view[0], p_view[1], p_view[2], 1.0])
        origin = np.asarray(eye, dtype=np.float64)
        direction = p_world[:3] - origin
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        return Ray(origin, direction)

    def render_to_pygame_surface(self, crank_angle_deg: float, spin: bool = True, dt: float = 0.0):
        import pygame
        rgba = self.render(crank_angle_deg, spin=spin, dt=dt)
        return pygame.image.frombuffer(np.ascontiguousarray(rgba).tobytes(), (self.width, self.height), "RGBA")

    def legend(self) -> list[tuple[str, tuple[int, int, int], float]]:
        """(label, rgb 0..255, opacity) for every material actually
        present in the current graph -- the same shape the old CPU
        MeshVisualizer.legend() returned, so main_pygame.py's legend-
        drawing loop needs no logic change, only a source change."""
        out = []
        for i in sorted(self._materials_present):
            m = MATERIALS[i]
            out.append((m.label, tuple(int(c * 255) for c in m.albedo), m.opacity))
        return out

    def close(self) -> None:
        self._free_geometry()
        self._target.delete()
