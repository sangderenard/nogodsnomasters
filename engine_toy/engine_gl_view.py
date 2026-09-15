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
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_SPECTRAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spectral-analyzer")
if _SPECTRAL_DIR not in sys.path:
    sys.path.insert(0, _SPECTRAL_DIR)

from engine_mesh import (EngineMesh, MATERIALS, material_table, build_engine_mesh, merge_engine_meshes, start_animation, CYCLE_DEG,
                        build_throttle_animation, blackbody_emission_rgb, BLACKBODY_VISIBLE_FLOOR_K,
                        significant_crank_angles, BLACKBODY_REFERENCE_K, BLACKBODY_SCENE_GAIN)
from sdf_geometry import CapsuleSdf


def _fragment_shader_with_sdf_cutouts(source: str) -> str:
    uniform_marker = "uniform float uFieldGain = 0.0;"
    main_marker = "void main() {"
    if uniform_marker not in source or main_marker not in source:
        raise RuntimeError("engine cutout shader injection markers not found")
    source = source.replace(
        uniform_marker,
        uniform_marker + """

#define MAX_ENGINE_CUTOUTS 64
uniform int uEngineCutoutCount;
uniform vec4 uEngineCutoutStart[MAX_ENGINE_CUTOUTS];
uniform vec4 uEngineCutoutEnd[MAX_ENGINE_CUTOUTS];

bool inside_engine_cutout(vec3 point_obj) {
    for (int i = 0; i < MAX_ENGINE_CUTOUTS; ++i) {
        if (i >= uEngineCutoutCount) break;
        vec3 start = uEngineCutoutStart[i].xyz;
        vec3 axis = uEngineCutoutEnd[i].xyz - start;
        float axis_len_sq = max(dot(axis, axis), 1e-12);
        float along = clamp(dot(point_obj - start, axis) / axis_len_sq, 0.0, 1.0);
        float radius = uEngineCutoutStart[i].w;
        if (length(point_obj - (start + axis * along)) <= radius) return true;
    }
    return false;
}""",
        1,
    )
    return source.replace(
        main_marker,
        main_marker + """
    if (inside_engine_cutout(vPosObj)) {
        discard;
    }""",
        1,
    )


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
        glDeleteVertexArrays, glDeleteBuffers, glBufferSubData,
        glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D, glFramebufferRenderbuffer,
        glGenRenderbuffers, glBindRenderbuffer, glRenderbufferStorage, glCheckFramebufferStatus,
        glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
        glViewport, glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        glGetIntegerv, GL_VIEWPORT,
        glEnable, glDisable, glDepthFunc, glReadPixels, glDeleteFramebuffers, glDeleteRenderbuffers,
        glGetUniformLocation, glUniform1i, glUniform4fv, glUseProgram,
        glDeleteTextures,
    )
    GL_OK = True
except Exception:
    GL_OK = False

if GL_OK:
    import base_gl_renderer as _base_gl_renderer
    from base_gl_renderer import BaseGLRenderer


    class _SdfCutoutRenderer(BaseGLRenderer):
        """Base material renderer with finite capsule subtraction."""

        MAX_CUTOUTS = 64

        def __init__(self, material_db) -> None:
            self._cutout_capsules = np.zeros((0, 2, 4), dtype=np.float32)
            self._u_cutout_count = -1
            self._u_cutout_start = -1
            self._u_cutout_end = -1
            super().__init__(material_db)

        def _build_program(self) -> None:
            original_read_glsl = _base_gl_renderer._read_glsl

            def read_glsl_with_cutouts(path: str) -> str:
                source = original_read_glsl(path)
                if path != _base_gl_renderer._FRAG_PATH:
                    return source
                return _fragment_shader_with_sdf_cutouts(source)

            _base_gl_renderer._read_glsl = read_glsl_with_cutouts
            try:
                super()._build_program()
            finally:
                _base_gl_renderer._read_glsl = original_read_glsl
            self._u_cutout_count = glGetUniformLocation(self._prog, "uEngineCutoutCount")
            self._u_cutout_start = glGetUniformLocation(self._prog, "uEngineCutoutStart")
            self._u_cutout_end = glGetUniformLocation(self._prog, "uEngineCutoutEnd")

        def set_cutout_capsules(self, capsules: list[tuple]) -> None:
            clipped = capsules[-self.MAX_CUTOUTS:]
            self._cutout_capsules = np.asarray(clipped, dtype=np.float32).reshape((-1, 2, 4))

        def draw_mesh(self, *args, **kwargs) -> None:
            glUseProgram(self._prog)
            count = len(self._cutout_capsules)
            if self._u_cutout_count != -1:
                glUniform1i(self._u_cutout_count, count)
            if count:
                if self._u_cutout_start != -1:
                    starts = np.ascontiguousarray(self._cutout_capsules[:, 0, :])
                    glUniform4fv(self._u_cutout_start, count, starts)
                if self._u_cutout_end != -1:
                    ends = np.ascontiguousarray(self._cutout_capsules[:, 1, :])
                    glUniform4fv(self._u_cutout_end, count, ends)
            super().draw_mesh(*args, **kwargs)


# ---------------------------------------------------------------------
# Material adapter: the only contract BaseGLRenderer needs from a
# "material_db" is build_tensors() -> the four SSBO chunks.
# ---------------------------------------------------------------------

class _ThermalMaterialDB:
    """The material table BaseGLRenderer reads its four SSBO chunks from,
    plus a LIVE layer on top of it: one extra row per (base material,
    thermal group) pair actually present in the current engine, whose
    emission.rgb / emit_gain are rewritten every tick from that group's
    own real temperature this instant (Planck, engine_mesh.blackbody_
    emission_rgb). Nothing about the glow is baked into the mesh or the
    animation frames -- a part's triangles just point at its variant
    row, and the row's emission is whatever the sim's thermal state
    says right now: a piston in a cylinder whose coolant has stopped
    flowing climbs through cherry red toward white exactly as its own
    block_cyl_N temperature does, because that is the number driving
    it. Diagnostic, not decorative."""

    def __init__(self) -> None:
        t = material_table()
        self._n_base = len(t["names"])
        self._base_pbr = t["pbr"].astype(np.float32)
        self._base_phong = t["phong"].astype(np.float32)
        self._variants: dict[tuple[int, str], int] = {}     # (base material id, group) -> row index
        self._variant_rows: list[tuple[int, str]] = []
        self._group_temps: dict[str, float] = {}
        # explicit rows: group -> (emission rgb, opacity, albedo rgb | None)
        # for things whose colour is COMPOSED rather than a plain
        # blackbody of one temperature (combustion_kernel.py's flame:
        # soot blackbody + chemiluminescence; its residue: a declared
        # smoke colour/opacity, no emission at all)
        self._group_overrides: dict[str, tuple[tuple, float, tuple | None]] = {}
        # thermal_atlas.py: row -> (emit_uv_layer, gain) when a per-part
        # temperature texture is painting this row's parts
        self._row_layers: dict[int, tuple[int, float]] = {}
        # False when the renderer's ThermalChunk SSBO does the blackbody
        # per fragment: rows then keep only their base look (and the
        # explicit flame/smoke overrides), no host-side emission
        self.emit_rows = True
        self._tensors: dict | None = None
        self._rebuild()

    def set_row_layers(self, row_layers: dict[int, tuple[int, float]]) -> None:
        self._row_layers = dict(row_layers)
        self._rebuild()

    @property
    def n_base(self) -> int:
        return self._n_base

    def base_of(self, material_id: int) -> int:
        """The plain material a (possibly variant) row stands in for."""
        if material_id < self._n_base:
            return int(material_id)
        return self._variant_rows[material_id - self._n_base][0]

    def variant_index(self, base_material_id: int, group: str) -> int:
        key = (int(base_material_id), group)
        idx = self._variants.get(key)
        if idx is None:
            idx = self._n_base + len(self._variant_rows)
            self._variants[key] = idx
            self._variant_rows.append(key)
            self._rebuild()
        return idx

    def remap(self, mesh: EngineMesh) -> EngineMesh:
        """Point every part that carries a thermal group at its own
        (material, group) variant row. In place; returns the mesh."""
        groups = getattr(mesh, "part_groups", None) or []
        for (t0, t1), group in zip(mesh.part_ranges, groups):
            if group is None or t1 <= t0:
                continue
            base = int(mesh.material_ids[t0])
            if base >= self._n_base:
                base = self.base_of(base)
            mesh.material_ids[t0:t1] = self.variant_index(base, group)
        return mesh

    def set_group_temperatures(self, temps_k: dict[str, float]) -> bool:
        """Returns True when any variant row's emission actually changed
        (quantised to 2 K so an idling engine doesn't re-upload the
        material table every frame for noise)."""
        q = {g: round(float(v) / 2.0) * 2.0 for g, v in temps_k.items()}
        if q == self._group_temps:
            return False
        self._group_temps = q
        self._rebuild()
        return True

    def group_temperature(self, group: str) -> float:
        return self._group_temps.get(group, 0.0)

    def set_group_overrides(self, overrides: dict[str, tuple[tuple, float, tuple | None]]) -> bool:
        """Replace the explicit-row set; True when it actually changed."""
        q = {g: (tuple(round(float(c), 3) for c in e), round(float(o), 3), (None if a is None else tuple(round(float(c), 3) for c in a)))
             for g, (e, o, a) in overrides.items()}
        if q == self._group_overrides:
            return False
        self._group_overrides = q
        self._rebuild()
        return True

    def _rebuild(self) -> None:
        n = self._n_base + len(self._variant_rows)
        pbr = np.zeros((n, 16), dtype=np.float32)
        phong = np.zeros((n, 8), dtype=np.float32)
        texstack = np.zeros((n, 16), dtype=np.float32)      # all-zero row = identity UV stack (no-op)
        pbr[:self._n_base] = self._base_pbr
        phong[:self._n_base] = self._base_phong
        for k, (base, group) in enumerate(self._variant_rows):
            row = self._n_base + k
            pbr[row] = self._base_pbr[base]
            phong[row] = self._base_phong[base]
            override = self._group_overrides.get(group)
            if override is not None:
                (r, g, b), opacity, albedo = override
                pbr[row, 8:11] = (r, g, b)
                pbr[row, 7] = opacity
                texstack[row, 8] = 1.0 if (r > 0.0 or g > 0.0 or b > 0.0) else 0.0
                if albedo is not None:
                    pbr[row, 0:3] = albedo
                    phong[row, 4:7] = albedo
                continue
            if not self.emit_rows:
                continue
            temp = self._group_temps.get(group, 0.0)
            r, g, b = blackbody_emission_rgb(temp)
            pbr[row, 8:11] = (r, g, b)
            texstack[row, 8] = 1.0 if temp >= BLACKBODY_VISIBLE_FLOOR_K else 0.0   # emit_gain
            layer_gain = self._row_layers.get(row)
            if layer_gain is not None and layer_gain[1] > 0.0:
                # the atlas paints this row: its layer, and a gain such
                # that a texel AT the group temperature reproduces the
                # row's own emission while hotter texels exceed it
                texstack[row, 0] = float(layer_gain[0])
                texstack[row, 8] = float(layer_gain[1])
                if temp < BLACKBODY_VISIBLE_FLOOR_K:
                    # the group is cold but a hot spot on it may not be:
                    # give the row a cherry-red base colour to scale
                    pbr[row, 8:11] = blackbody_emission_rgb(BLACKBODY_REFERENCE_K)
            if temp >= BLACKBODY_VISIBLE_FLOOR_K:
                # a glowing surface is no longer see-through in any
                # useful sense -- the glow IS what you see of it
                pbr[row, 7] = max(pbr[row, 7], min(1.0, 0.5 + 0.5 * min(1.0, r)))
        # a NEW dict object every rebuild: BaseGLRenderer._build_ssbos
        # skips the upload when it sees the same tensors object again
        self._tensors = {"pbr": pbr, "phong_compat": phong,
                         "enamel": np.zeros((n, 8), dtype=np.float32), "texture_stack": texstack}

    def build_tensors(self) -> dict:
        return self._tensors


def thermal_groups_from_state(state) -> dict[str, float]:
    """The live per-group temperatures (kelvin) an EngineCycleSim state
    already carries, keyed the way vehicle_mesh._thermal_group_for tags
    every part. Read fresh every tick -- these are the sim's own
    numbers, not a schedule."""
    temps = {
        "exhaust": float(getattr(state, "exhaust_temp_k", 0.0)),
        "intake": float(getattr(state, "intake_runner_temp_k", 0.0)),
        "coolant": float(getattr(state, "coolant_temp_k", 0.0)),
        "oil": float(getattr(state, "oil_temp_k", 0.0)),
    }
    for i, t in enumerate(getattr(state, "cylinder_block_temps_k", []) or []):
        temps[f"block_cyl_{i + 1}"] = float(t)
    return temps


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
    # The interleaved pos/nrm/uv rows exactly as uploaded, kept so that
    # geometry which ARTICULATES can be restaged with a buffer write
    # instead of a teardown (see update_geometry). None on a mesh built
    # before this existed -- callers fall back to a full re-upload.
    rows: "np.ndarray | None" = None

    def delete(self) -> None:
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(len(self.buffers), list(self.buffers))

    def update_geometry(self, pos: np.ndarray, nrm: np.ndarray,
                        material_ids: "np.ndarray | None" = None) -> bool:
        """Push moved vertices (and optionally recoloured triangles) into
        the buffers this mesh already owns.

        For geometry whose TRIANGLES do not change -- an articulating
        mount, a recoiling barrel, a rig re-banded by utilisation -- the
        UVs, the thermal group ids and the cull-immune column are all
        still exactly right, because they are properties of the triangle
        list and the triangle list did not change. Only positions,
        normals and (sometimes) material ids differ, so only those are
        written.

        Returns False, changing nothing, if this mesh cannot take the
        update (no cached rows, or a vertex count that no longer
        matches); the caller is expected to fall back to a full upload.
        That check is the whole safety story -- a topology change is
        refused rather than written past the end of a buffer.
        """
        rows = self.rows
        if rows is None or len(pos) != self.n_vertices or len(nrm) != self.n_vertices:
            return False
        rows[:, 0:3] = pos.astype(np.float32, copy=False)
        rows[:, 3:6] = nrm.astype(np.float32, copy=False)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glBufferSubData(GL_ARRAY_BUFFER, 0, rows.nbytes, rows)
        if material_ids is not None:
            values = np.repeat(np.asarray(material_ids, dtype=np.int32), 3)
            if len(values) != self.n_vertices:
                return False
            values = np.ascontiguousarray(values, dtype=np.int32)
            glBindBuffer(GL_ARRAY_BUFFER, self.buffers[1])
            glBufferSubData(GL_ARRAY_BUFFER, 0, values.nbytes, values)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        return True


def _streak_soup(pos: np.ndarray, vel: np.ndarray, r: float, gas: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """n droplets as n thin 4-sided prisms (8 triangles each) along
    their own velocity, built in one vectorised pass: a streak of
    length ~ |v| x 4 ms (a motion-blurred drop), never shorter than 2r.
    Returns (vertices (24n,3), normals (24n,3)) triangle soup."""
    n = len(pos)
    v = vel.astype(np.float64)
    speed = np.linalg.norm(v, axis=1)
    d = v / np.maximum(speed, 1e-9)[:, None]
    # a motion-blurred drop: the streak is how far it moves in one
    # short exposure, floored at the droplet itself and capped so a
    # detonation product at 900 m/s reads as a fireball rather than a
    # spike across the whole frame
    length = np.clip(speed * 0.0006, 2.0 * r, 0.22)[:, None]
    if gas:
        length = np.maximum(length, 3.0 * r)
    helper = np.where(np.abs(d[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    u = np.cross(d, helper); u /= np.maximum(np.linalg.norm(u, axis=1), 1e-9)[:, None]
    w = np.cross(d, u)
    a = pos - d * length * 0.5
    b = pos + d * length * 0.5
    # four side quads of a square prism, corners at +/-u, +/-w
    c = [u * r + w * r, -u * r + w * r, -u * r - w * r, u * r - w * r]
    tris = []
    for k in range(4):
        c0, c1 = c[k], c[(k + 1) % 4]
        p00, p01, p10, p11 = a + c0, a + c1, b + c0, b + c1
        nrm = (c0 + c1); nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)[:, None]
        tris.append((p00, p10, p11, nrm)); tris.append((p00, p11, p01, nrm))
    verts = np.concatenate([np.stack([t[0], t[1], t[2]], axis=1) for t in tris], axis=1).reshape(-1, 3)
    norms = np.concatenate([np.repeat(t[3][:, None, :], 3, axis=1) for t in tris], axis=1).reshape(-1, 3)
    return verts, norms


# A frame slot that has been baked and turned out to have NOTHING in
# it. This has to be distinguishable from "not uploaded yet", because
# `upload_mesh` returns None for an empty mesh and the bake loop asks
# "are any slots still None?" to decide whether it is finished. A
# machine (machines.py) has no moving parts at all, so every frame is
# empty, every slot stayed None, and the loop spun forever -- which is
# why a machine would never render and never error either.
EMPTY_FRAME = object()


def upload_mesh(mesh: EngineMesh, uv: "np.ndarray | None" = None, group_ids: "np.ndarray | None" = None) -> GLMesh | None:
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
    if uv is None or uv.shape[0] != pos.shape[0]:
        uv = np.zeros((pos.shape[0], 2), dtype=np.float32)
    rows = np.ascontiguousarray(np.concatenate([pos, nrm, uv.astype(np.float32)], axis=1), dtype=np.float32)
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

    gid = np.repeat(group_ids.astype(np.int32), 3) if group_ids is not None and len(group_ids) == n_tri else np.zeros(n, np.int32)
    int_columns = (mat_id, gid, np.ones(n, np.int32))   # matId, groupId (= thermal id), cullImmune
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
    return GLMesh(vao=vao, buffers=buffers, n_vertices=n, rows=rows)


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
    # an int (evenly spaced), an explicit list of degrees, or "dense" --
    # engine_mesh.significant_crank_angles: a 5-degree grid plus every
    # real event angle (dead centres, valve open/peak/close, firing and
    # each burn phase) for every cylinder, so nothing the sim schedules
    # ever falls between two baked frames. The default, since a user's
    # own engine is what gets rendered.
    animation_divisions: "int | list | str" = "dense"
    spin_rad_per_s: float = 0.0
    # The mesh REDUCTION stage, made explicit and optional. Two real knobs:
    #   detail        tessellation multiplier for every tube/drum side count
    #                 (mesh_primitives.DETAIL): 1.0 = full, the exporter's
    #                 own level; the live app may run lower.
    #   spring_style  "cylinder" draws each valve spring as one cheap tube
    #                 (the live default), "helix" draws the real coil.
    # A high-triangle render is simply detail=1.0, spring_style="helix".
    detail: float = 1.0
    spring_style: str = "cylinder"
    # the combustion kernel's own point emitters: off by default -- a
    # tiny light source fully enclosed in a bore is exactly what a Phong
    # point light does worst (it shines straight through the walls it
    # should be trapped behind), and it made that failure obvious. The
    # kernel's SURFACE self-emission stays; only the light it throws on
    # neighbours is gated here.
    flame_lights_enabled: bool = False
    # Per-part heat detail texture (thermal_atlas.py): OFF by default --
    # the per-part SSBO path below is the real one; the texture is an
    # opt-in close-up mode with a Python kernel and a texture upload.
    thermal_texture_enabled: bool = False
    # "ssbo" (default): every part with a thermal group gets a thermal id
    # (its vertices' aGroupId); the renderer holds one float of kelvin per
    # id in an SSBO and the fragment shader does the Planck itself -- the
    # host uploads a float array per tick and nothing else. "rows": the
    # older per-(material, group) emission rows written in Python.
    thermal_mode: str = "ssbo"

    def __post_init__(self) -> None:
        if not GL_OK:
            raise RuntimeError("engine_gl_view requires PyOpenGL and an active GL context")
        self._db = _ThermalMaterialDB()
        self._renderer = _SdfCutoutRenderer(self._db)
        self._renderer.init_gl()
        self._target = OffscreenTarget(self.width, self.height)
        self._static_gl: GLMesh | None = None
        self._frame_gl: list = []
        self._animation = None
        self._throttle_animation = None
        self._throttle_gl: list = []
        self._graph = None
        self._materials_present: set = set()
        self._angle_rad = 0.6
        # Camera elevation and zoom, so a caller that needs to look at
        # something from a chosen angle can. The defaults reproduce
        # exactly what this view did before they existed: a fixed 0.40
        # rise on the eye direction and a distance fitted to the object.
        self._elevation = 0.40
        self._zoom = 1.0
        self._center = np.zeros(3, dtype=np.float32)
        self._scale = 1.0
        self._half_extents = np.ones(3, dtype=np.float32)
        # live thermal emission: (group, world centroid, area) for every
        # static part that carries a thermal group, so a hot one can
        # also be a real point emitter lighting its neighbours -- and
        # the group temperatures set_thermal_state was last given
        self._thermal_parts: list[tuple[str, np.ndarray, float]] = []
        self._group_temps: dict[str, float] = {}
        # combustion_kernel.py's baked per-cylinder burn/residue frames,
        # uploaded once; which one each cylinder shows is a per-tick
        # decision off the sim's own firing bookkeeping
        self._combustion = None
        # thermal_atlas.py: per-part temperature tiles + their conduction kernel
        self._atlas = None
        self._atlas_stamp = None
        self._atlas_tick = 0
        self._flame_gl: dict[tuple[int, int], GLMesh] = {}
        self._smoke_gl: dict[tuple[int, int], GLMesh] = {}
        self._flame_centroid: dict[tuple[int, int], np.ndarray] = {}
        self._leak_gl: GLMesh | None = None   # set_emitter_particles: the damage holes' droplet streaks
        self._leak_rng = np.random.default_rng(77)
        self._live_flames: list = []      # (cyl, phase index, emission rgb)
        self._live_smoke: list = []       # (cyl, phase index)
        self._cutout_capsules: list[tuple] = []
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
        self._cutout_capsules = []
        self._renderer.set_cutout_capsules([])
        self._free_geometry()
        import mesh_primitives as _mp
        _prev_detail = _mp.DETAIL
        _mp.set_detail(self.detail)
        try:
            static_mesh, _moving0 = build_engine_mesh(graph, crank_angle_deg=0.0, covers_off=self.covers_off)
        finally:
            _mp.DETAIL = _prev_detail
        # Other simulation objects may supply their own fully baked meshes.
        # Keep their geometry/material/thermal records intact; the scene only
        # places the object and later swaps its equally-shaped live frame.
        self._mesh_overlay_slices = {}
        overlays = list(graph.get("mesh_overlays") or ())
        if overlays:
            joined = [static_mesh]
            cursor = len(static_mesh.vertices)
            for overlay in overlays:
                mesh = overlay["mesh"]
                joined.append(mesh)
                self._mesh_overlay_slices[str(overlay["identity"])] = slice(
                    cursor, cursor + len(mesh.vertices))
                cursor += len(mesh.vertices)
            static_mesh = merge_engine_meshes(joined)
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
        # live-thermal variant rows: every part with a real thermal group
        # points at its own (material, group) row so its glow tracks that
        # group's temperature each tick (see _ThermalMaterialDB) -- done
        # AFTER the legend's material set was read (legend shows base
        # materials), BEFORE upload (the ids have to be on the GPU)
        self._hotspots = self._hotspot_parts(graph, static_mesh)
        self._thermal_ids = {}          # part name -> thermal id (>= 1)
        self._thermal_groups = []       # id -> (group, hot-spot delta)
        self._db.remap(static_mesh)
        self._db.remap(_moving0)
        self._atlas = self._build_atlas(graph, static_mesh) if self.thermal_texture_enabled else None
        self._static_mesh = static_mesh
        # a fresh graph is a fresh triangle list: whatever was culled off
        # the last engine does not apply to this one
        self._static_tri_keep = None
        self._absent = set()
        self._absent: set = set()
        # which triangles of _static_mesh are actually in _static_gl.
        # None means all of them; set_absent_parts narrows it when a part
        # bursts, and restage_static_mesh has to honour the same subset or
        # it would quietly resurrect the burst parts on the next restage.
        self._static_tri_keep: "np.ndarray | None" = None
        self._static_gl = upload_mesh(static_mesh, uv=self._atlas_uv(static_mesh), group_ids=self._thermal_id_column(static_mesh))
        if self.thermal_mode == "ssbo":
            self._renderer.set_thermal_emission(True, gain=BLACKBODY_SCENE_GAIN, ref_k=BLACKBODY_REFERENCE_K,
                                                floor_k=BLACKBODY_VISIBLE_FLOOR_K)
            self._db.emit_rows = False
        else:
            self._renderer.set_thermal_emission(False)
            self._db.emit_rows = True
        self._thermal_parts = []
        groups = getattr(static_mesh, "part_groups", None) or []
        for (t0, t1), group in zip(static_mesh.part_ranges, groups):
            if group is None or t1 <= t0:
                continue
            tv = static_mesh.tri_vertices()[t0:t1]                     # (n, 3, 3)
            areas = 0.5 * np.linalg.norm(np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]), axis=1)
            total = float(areas.sum())
            if total <= 0.0:
                continue
            centroid = (tv.mean(axis=1) * areas[:, None]).sum(axis=0) / total
            self._thermal_parts.append((group, centroid.astype(np.float32), total))
        self._renderer.update_material_ssbo()
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
        divisions = (significant_crank_angles(graph) if self.animation_divisions == "dense" else self.animation_divisions)
        self._animation = start_animation(graph, divisions, covers_off=self.covers_off,
                                          detail=self.detail, spring_style=self.spring_style)
        self._frame_gl = [None] * self._animation.n_frames
        self._upload_pending_frames(limit=1)

        # a SEPARATE small baked set, keyed by throttle position (0..1)
        # instead of crank angle -- cheap enough (a plate + lever arm
        # per throttle body) to upload all of it right here rather than
        # spreading it across bake_next() the way the crank animation
        # does; composited alongside whatever crank frame is showing,
        # never folded into that frame count.
        _mp.set_detail(self.detail)
        try:
            self._throttle_animation = build_throttle_animation(graph, divisions=9)
        finally:
            _mp.DETAIL = _prev_detail
        for g in self._throttle_gl:
            g.delete()
        self._throttle_gl = [upload_mesh(f) for f in self._throttle_animation.frames]
        if self._throttle_gl:
            self._materials_present |= {int(i) for f in self._throttle_animation.frames if f.n_triangles
                                        for i in np.unique(f.material_ids)}

    def update_mesh_overlay(self, identity: str, mesh: EngineMesh) -> None:
        """Replace one baked object's live vertices without rebuilding it."""
        span = self._mesh_overlay_slices[str(identity)]
        if len(mesh.vertices) != span.stop - span.start:
            raise ValueError(f"{identity}: live overlay changed topology")
        self._static_mesh.vertices[span] = mesh.vertices
        self._static_mesh.normals[span] = mesh.normals

    def set_cutout_punctures(self, punctures) -> int:
        """Replace the rendered cutout set from persistent damage state."""
        self._cutout_capsules = []
        return self.add_cutout_punctures(punctures)

    def add_cutout_punctures(self, punctures) -> int:
        """Add persistent projectile channels to the world-space SDF mask."""
        for puncture in punctures:
            capsule = CapsuleSdf.from_puncture(puncture)
            start = capsule.start
            end = capsule.end
            radius = max(capsule.radius_m, 1e-6)
            self._cutout_capsules.append((
                (start[0], start[1], start[2], radius),
                (end[0], end[1], end[2], radius),
            ))
        dropped = max(0, len(self._cutout_capsules) - self._renderer.MAX_CUTOUTS)
        self._cutout_capsules = self._cutout_capsules[-self._renderer.MAX_CUTOUTS:]
        self._renderer.set_cutout_capsules(self._cutout_capsules)
        return dropped

    def bake_combustion(self, firing_angles_deg: dict[int, float], visual) -> None:
        """Bake and upload the combustion kernel frames for the current
        graph (combustion_kernel.bake_combustion_frames): one flame
        volume per cylinder per burn phase, one residue puff per
        cylinder per exhaust phase when the fuel leaves any."""
        import combustion_kernel as ck
        from cylinder_ports import deserialize_layout
        for g in list(self._flame_gl.values()) + list(self._smoke_gl.values()):
            g.delete()
        self._flame_gl = {}; self._smoke_gl = {}; self._flame_centroid = {}
        self._combustion = None
        layout_data = (self._graph or {}).get("cylinder_layout")
        if not layout_data:
            return
        layout = deserialize_layout(layout_data)
        frames = ck.bake_combustion_frames(layout, firing_angles_deg, visual, ck.exhaust_ports_from_layout(layout))
        self._combustion = frames
        for cyl, fl in frames.flame.items():
            for k, mesh in enumerate(fl):
                self._db.remap(mesh)
                gl = upload_mesh(mesh)
                if gl is not None:
                    self._flame_gl[(cyl, k)] = gl
                    self._flame_centroid[(cyl, k)] = mesh.vertices.mean(axis=0).astype(np.float32)
        for cyl, sl in frames.smoke.items():
            for k, mesh in enumerate(sl):
                self._db.remap(mesh)
                gl = upload_mesh(mesh)
                if gl is not None:
                    self._smoke_gl[(cyl, k)] = gl
        self._renderer.update_material_ssbo()

    @property
    def combustion_burn_duration_deg(self) -> float:
        return self._combustion.burn_duration_deg if self._combustion is not None else 0.0

    def set_combustion_state(self, states) -> None:
        """states: combustion_kernel.combustion_state_from_sim(sim, ...).
        Picks each cylinder's frame and writes its composed emission /
        residue opacity into that cylinder's own material rows."""
        import combustion_kernel as ck
        self._live_flames = []; self._live_smoke = []
        if self._combustion is None:
            return
        vis = self._combustion.visual
        overrides: dict = {}
        for st in states:
            fl = self._combustion.flame.get(st.cylinder)
            if fl and st.burn_phase is not None:
                k = min(len(fl) - 1, int(st.burn_phase * len(fl)))
                rgb = ck.flame_emission_rgb(vis, st.burn_phase, st.strength)
                overrides[f"flame_c{st.cylinder}_p{k}"] = (rgb, 0.85, None)
                self._live_flames.append((st.cylinder, k, rgb))
            sl = self._combustion.smoke.get(st.cylinder)
            if sl and st.smoke_phase is not None:
                k = min(len(sl) - 1, int(st.smoke_phase * len(sl)))
                opacity = vis.residue_opacity * ck.smoke_shape(st.smoke_phase) * min(1.0, st.strength)
                overrides[f"smoke_c{st.cylinder}_p{k}"] = ((0.0, 0.0, 0.0), opacity, tuple(vis.residue_rgb))
                self._live_smoke.append((st.cylinder, k))
        if self._db.set_group_overrides(overrides):
            self._renderer.update_material_ssbo()

    # droplets per emitter by regime: the budget is generous (a few
    # hundred per stream, ~12 triangles each -- tens of thousands of
    # triangles for a badly shot engine, still one draw call)
    PARTICLES_PER_EMITTER = {"spray": 320, "pour": 160, "drip": 24, "gas": 260, "ingest": 90, "splash": 140, "burst": 400}

    def set_absent_parts(self, identities) -> None:
        """Parts that have burst: drop every static triangle whose mesh
        part maps to one of these graph identities and re-upload the
        static mesh (the baked moving frames are untouched -- only
        castings/bolt-ons can be absent)."""
        from damage_state import mesh_part_identity
        wanted = set(identities)
        if wanted == self._absent or self._static_mesh is None:
            return
        self._absent = set(wanted)
        mesh = self._static_mesh
        keep = np.ones(mesh.n_triangles, bool)
        for (t0, t1), name in zip(mesh.part_ranges, mesh.part_names):
            if mesh_part_identity(name, self._graph) in wanted:
                keep[t0:t1] = False
        from engine_mesh import EngineMesh
        tri_keep = np.nonzero(keep)[0]
        self._static_tri_keep = None if len(tri_keep) == mesh.n_triangles else tri_keep
        sub = EngineMesh(mesh.vertices, mesh.normals, mesh.triangles[tri_keep], mesh.material_ids[tri_keep],
                         [], [], mesh.moving, [])
        # the thermal id column follows the same triangle subset
        gid = self._thermal_id_column(mesh)
        gid = gid[tri_keep] if gid is not None and len(gid) == mesh.n_triangles else None
        if self._static_gl is not None:
            self._static_gl.delete()
        self._static_gl = upload_mesh(sub, uv=None, group_ids=gid)

    def restage_static_mesh(self) -> None:
        """Re-upload the static mesh after its vertices have been moved.

        For geometry that ARTICULATES rather than changes: a mount that
        aims has exactly the same triangles, the same materials and the
        same thermal groups from one frame to the next, and only its
        vertex positions differ. Rebuilding the mesh for that costs a
        tenth of a second (see `articulation.ArticulatedMesh`, which
        moves the vertices instead); pushing the moved vertices costs a
        buffer write.

        The caller owns `self._static_mesh` and is expected to have
        written new `vertices` and `normals` into it (and may have
        recoloured triangles by writing `material_ids`). Everything
        derived from the TRIANGLES -- the atlas UVs, the thermal id
        column -- is still valid precisely because the triangles did not
        change, so none of it is recomputed: the moved vertices go
        straight into the buffers the mesh already owns.

        Falls back to a full re-upload whenever that is not safely
        possible -- nothing uploaded yet, or a triangle list that really
        did change underneath us. Both paths honour whatever triangle
        subset `set_absent_parts` last culled, so restaging an engine
        with a burst part does not bring the part back.
        """
        mesh = self._static_mesh
        if mesh is None:
            return
        keep = self._static_tri_keep
        tri_v = mesh.tri_vertices().reshape(-1, 3)
        tri_n = mesh.tri_normals().reshape(-1, 3)
        mat = mesh.material_ids
        if keep is not None:
            tri_v = tri_v.reshape(-1, 3, 3)[keep].reshape(-1, 3)
            tri_n = tri_n.reshape(-1, 3, 3)[keep].reshape(-1, 3)
            mat = mat[keep]
        if self._static_gl is not None and self._static_gl.update_geometry(tri_v, tri_n, material_ids=mat):
            return
        # the fall-back: topology really changed (or nothing is uploaded
        # yet), so the derived columns are no longer valid either
        from engine_mesh import EngineMesh
        staged = mesh
        gid = self._thermal_id_column(mesh)
        if keep is not None:
            staged = EngineMesh(mesh.vertices, mesh.normals, mesh.triangles[keep], mesh.material_ids[keep],
                                [], [], mesh.moving, [])
            gid = gid[keep] if gid is not None and len(gid) == mesh.n_triangles else None
        if self._static_gl is not None:
            self._static_gl.delete()
        self._static_gl = upload_mesh(
            staged, uv=self._atlas_uv(staged), group_ids=gid)

    def set_emitter_particles(self, emitters, t_max_s: float = 0.3, budget: int = 6000) -> None:
        """Droplet clouds for every fluid emitter (hole_emitters.
        HoleEmitter) with something passing: each emitter's own
        droplets() sample -- a random age along the arc, a random
        direction in its flow character's cone -- drawn as short streaks
        along each droplet's own velocity (a motion-blurred drop), in the
        fluid's leak material. Vectorised: one triangle-soup build from
        all droplets at once, one upload, one draw. `budget` caps the
        total droplets per frame across emitters."""
        from types import SimpleNamespace
        from engine_mesh import _from_parts
        if self._leak_gl is not None:
            self._leak_gl.delete(); self._leak_gl = None
        rng = self._leak_rng
        want = [(em, self.PARTICLES_PER_EMITTER.get(em.regime, 100)) for em in emitters]
        total = sum(n for _, n in want)
        scale = min(1.0, budget / max(total, 1))
        parts = []
        for i, (em, n) in enumerate(want):
            pos, vel = em.droplets(rng, max(4, int(n * scale)), t_max_s)
            if not len(pos):
                continue
            r = max(0.0012, min(0.005, em.radius_m * 0.45)) if em.fluid != "gas" else max(0.004, em.radius_m * 1.2)
            if "mist" in em.character:
                r *= 0.5
            verts, norms = _streak_soup(pos, vel, r, gas=(em.fluid == "gas"))
            parts.append(SimpleNamespace(name=f"leak_{em.fluid}_{i}", vertices=verts, normals=norms))
        if not parts:
            return
        mesh = _from_parts(parts, moving=True)
        self._leak_gl = upload_mesh(mesh)

    def _flame_emitters(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        out = []
        if not self.flame_lights_enabled:
            return out
        for cyl, k, rgb in self._live_flames:
            c = self._flame_centroid.get((cyl, k))
            lum = max(rgb[0], rgb[1], rgb[2], 1e-6)
            if c is None or lum <= 1e-4:
                continue
            colour = np.array(rgb, dtype=np.float32) / lum
            out.append((c, colour, float(self._scale) * 1.6 * math.sqrt(min(lum, 25.0))))
        return out

    def _hotspot_parts(self, graph: dict, static_mesh: EngineMesh) -> dict:
        """part name -> (group, delta_k): the parts genuinely hotter than
        their lumped group and by how much -- an exhaust port's bore top
        and chamber roof (the combustion face), each head casting (its
        port rims), a turbine housing (its inlet scroll) -- off the
        graph's own real exhaust-port positions."""
        out: dict = {}
        groups = getattr(static_mesh, "part_groups", None) or []
        group_of = {n: g for n, g in zip(static_mesh.part_names, groups) if g}
        ports = [n for n in graph.get("nodes", []) if n.get("kind") == "engine-block-port" and n.get("fluid_role") == "exhaust"
                 and ".exhaust_port" in n["identity"] and "header" not in n["identity"]]
        for n in ports:
            cyl = n["identity"].split(".")[1].replace("cylinder_", "")
            for name, delta in ((f"cyl{cyl}_bore", 110.0), (f"chamber_roof_cyl{cyl}", 140.0)):
                if name in group_of:
                    out[name] = (group_of[name], delta)
        for name, group in group_of.items():
            if name.startswith("head_casting") or "turbine_housing" in name:
                out[name] = (group, 90.0)
        return out

    def _thermal_id_column(self, mesh: EngineMesh) -> np.ndarray:
        """One thermal id per triangle: the part's own id (allocated on
        first sight; a part that keeps its lumped group temperature and
        a hot-spot part are just different ids with different kelvins)."""
        ids = np.zeros(mesh.n_triangles, dtype=np.int32)
        groups = getattr(mesh, "part_groups", None) or []
        for name, (t0, t1), group in zip(mesh.part_names, mesh.part_ranges, groups):
            if group is None or t1 <= t0:
                continue
            tid = self._thermal_ids.get(name)
            if tid is None:
                tid = len(self._thermal_groups) + 1
                self._thermal_ids[name] = tid
                delta = self._hotspots.get(name, (group, 0.0))[1]
                self._thermal_groups.append((group, float(delta)))
            ids[t0:t1] = tid
        return ids

    def _upload_thermal_kelvins(self, group_temps_k: dict[str, float]) -> None:
        """The whole per-tick thermal work in ssbo mode: one float per id."""
        n = len(self._thermal_groups)
        if n == 0:
            return
        live = {f"block_cyl_{cyl}": 1.0 + 0.6 * min(1.0, max(rgb)) for cyl, _k, rgb in self._live_flames}
        kelvins = np.zeros(n + 1, dtype=np.float32)
        for i, (group, delta) in enumerate(self._thermal_groups, start=1):
            kelvins[i] = group_temps_k.get(group, 0.0) + delta * live.get(group, 1.0)
        self._renderer.set_thermal_kelvins(kelvins)

    def _build_atlas(self, graph: dict, static_mesh: EngineMesh):
        """One tile per part that carries a thermal group, in the layer of
        that part's own material row; real hot spots off the graph:
        every exhaust port rim on its head casting and bore, the
        combustion face at the top of each bore and its chamber roof,
        the turbine inlet on each turbine housing."""
        from thermal_atlas import ThermalAtlas
        atlas = ThermalAtlas()
        tv = static_mesh.tri_vertices()
        groups = getattr(static_mesh, "part_groups", None) or []
        for name, (t0, t1), group in zip(static_mesh.part_names, static_mesh.part_ranges, groups):
            if group is None or t1 <= t0:
                continue
            atlas.add_part(name, group, int(static_mesh.material_ids[t0]), tv[t0:t1].reshape(-1, 3))
        # hot spots
        ports = [n for n in graph.get("nodes", []) if n.get("kind") == "engine-block-port" and n.get("fluid_role") == "exhaust"
                 and ".exhaust_port" in n["identity"] and "header" not in n["identity"]]
        for n in ports:
            pos = np.array(n["reference_position"], dtype=np.float64)
            cyl = n["identity"].split(".")[1].replace("cylinder_", "")
            for name in atlas.tiles:
                if name.startswith("head_casting") or name == f"cyl{cyl}_bore" or name == f"chamber_roof_cyl{cyl}":
                    tile = atlas.tiles[name]
                    d = float(np.linalg.norm(tile.origin - pos))
                    if d < 2.5 * float(np.max(tile.extent)) + 0.05:
                        atlas.add_source(name, pos, 160.0 if name.startswith("head") else 110.0)
        for name, tile in atlas.tiles.items():
            if "turbine_housing" in name:
                atlas.add_source(name, tile.origin + tile.basis[0] * tile.extent[0] * 0.8, 90.0)
        return atlas

    def _atlas_uv(self, mesh: EngineMesh, recenter: bool = False):
        if self._atlas is None or not mesh.n_triangles:
            return None
        tv = mesh.tri_vertices().reshape(-1, 3)
        uv = np.zeros((tv.shape[0], 2), dtype=np.float32)
        groups = getattr(mesh, "part_groups", None) or []
        for name, (t0, t1), group in zip(mesh.part_names, mesh.part_ranges, groups):
            if group is None or t1 <= t0:
                continue
            tile = self._atlas.tiles.get(name)
            if tile is None:
                tile = self._atlas.add_part(name, group, int(mesh.material_ids[t0]), tv[t0 * 3:t1 * 3])
            if tile is not None:
                uv[t0 * 3:t1 * 3] = tile.uv_for(tv[t0 * 3:t1 * 3], recenter=recenter)
        return uv

    def _step_atlas(self, group_temps_k: dict[str, float]) -> None:
        import time as _time
        now = _time.monotonic()
        dt = 1.0 / 60.0 if self._atlas_stamp is None else max(1e-3, min(0.25, now - self._atlas_stamp))
        self._atlas_stamp = now
        live = {}
        for cyl, _k, rgb in self._live_flames:
            live[f"block_cyl_{cyl}"] = 1.0 + 0.6 * min(1.0, max(rgb))
        self._atlas.step(dt, group_temps_k, live_sources=live)
        self._atlas_tick += 1
        if self._atlas_tick % 4 == 0:
            arr = self._atlas.to_rgba_layers()
            self._renderer.set_emit_uv_texture_array(arr)
            row_layers = {}
            # per row: the layer its parts live in and that layer's gain
            # relative to the row's own group-temperature radiance
            for row, layer in self._atlas._row_layer.items():
                peak = self._atlas.layer_gain.get(layer, 0.0)
                group = next((t.group for t in self._atlas.tiles.values() if t.layer == layer), None)
                temp = group_temps_k.get(group, 0.0) if group else 0.0
                base = blackbody_emission_rgb(temp)[0]
                gain = (peak / base) if base > 1e-6 else (peak / max(blackbody_emission_rgb(BLACKBODY_REFERENCE_K)[0], 1e-6))
                row_layers[row] = (layer, float(gain))
            self._db.set_row_layers(row_layers)
            self._renderer.update_material_ssbo()

    def set_thermal_state(self, group_temps_k: dict[str, float]) -> None:
        """Feed the sim's live per-group temperatures in (thermal_groups_
        from_state(sim.state) every tick). Re-uploads the material table
        only when a quantised temperature actually moved."""
        self._group_temps = dict(group_temps_k)
        if self.thermal_mode == "ssbo":
            self._upload_thermal_kelvins(group_temps_k)
        elif self._db.set_group_temperatures(group_temps_k):
            self._renderer.update_material_ssbo()
        if self._atlas is not None:
            self._step_atlas(group_temps_k)

    def _thermal_emitters(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Point emitters for every glowing part cluster: (world
        position, colour, intensity). Parts of one group are merged per
        side of the engine (z sign) so a V engine's two headers light
        their own banks instead of one phantom light inside the block;
        intensity follows the real red-band radiance (Planck, relative
        to BLACKBODY_REFERENCE_K), on the same size-relative scale the
        key/fill rig uses."""
        clusters: dict[tuple[str, int], list[tuple[np.ndarray, float]]] = {}
        for group, centroid, area in self._thermal_parts:
            temp = self._group_temps.get(group, 0.0)
            if temp < BLACKBODY_VISIBLE_FLOOR_K:
                continue
            clusters.setdefault((group, 1 if centroid[2] >= 0.0 else -1), []).append((centroid, area))
        out = []
        for (group, _side), members in clusters.items():
            temp = self._group_temps.get(group, 0.0)
            r, g, b = blackbody_emission_rgb(temp)
            lum = max(r, 1e-6)
            colour = np.array([r, g, b], dtype=np.float32) / lum
            total = sum(a for _, a in members)
            pos = sum(c * a for c, a in members) / max(total, 1e-9)
            # radiant power grows with emitting area; the rig's own key
            # light is self._scale*3.5 -- a cherry-red header (lum 1.0)
            # of ~a tenth of the engine's silhouette lands near the fill
            area_frac = min(1.0, total / max(4.0 * float(self._scale) ** 2, 1e-6))
            # sqrt-compressed: radiance climbs ~T^8 through this band, a
            # light that tracked it linearly would flood the whole bay
            # yellow long before the surface itself read as white
            out.append((pos.astype(np.float32), colour,
                        float(self._scale) * 1.2 * math.sqrt(min(lum, 25.0)) * (0.3 + area_frac)))
        return out

    def _free_geometry(self) -> None:
        for g in list(self._flame_gl.values()) + list(self._smoke_gl.values()):
            g.delete()
        self._flame_gl = {}; self._smoke_gl = {}; self._flame_centroid = {}
        self._combustion = None; self._live_flames = []; self._live_smoke = []
        if self._static_gl is not None:
            self._static_gl.delete(); self._static_gl = None
            self._static_tri_keep = None
        for g in self._frame_gl:
            # EMPTY_FRAME is a sentinel meaning "baked, and there was
            # nothing in it" -- it owns no GPU buffer, so there is
            # nothing to delete. Treating it like a mesh only shows up
            # when a graph is replaced repeatedly, which is exactly what
            # an animation study does.
            if g is not None and g is not EMPTY_FRAME:
                g.delete()
        for g in self._throttle_gl:
            if g is not None:
                g.delete()
        self._throttle_gl = []
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
                self._db.remap(frame)
                gl = upload_mesh(frame, uv=self._atlas_uv(frame, recenter=True),
                                 group_ids=self._thermal_id_column(frame))
                # an empty frame is DONE, not pending
                self._frame_gl[i] = gl if gl is not None else EMPTY_FRAME
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

    def _throttle_frame_for(self, throttle_frac: float):
        if not self._throttle_animation or not self._throttle_gl:
            return None
        idx = self._throttle_animation.frame_index(throttle_frac)
        return self._throttle_gl[idx] if idx < len(self._throttle_gl) else None

    # ---- render: two-or-three draw calls, no per-frame geometry work ----
    def render(self, crank_angle_deg: float, spin: bool = True, dt: float = 0.0,
              throttle_frac: float = 1.0) -> np.ndarray:
        """Same draw as render_gpu(), but reads the result back to a
        numpy array (a CPU round trip) -- for headless PNG export and
        the test harness; the live app uses render_gpu() instead, which
        never leaves the GPU."""
        self._draw(crank_angle_deg, spin=spin, dt=dt, throttle_frac=throttle_frac)
        rgba = self._target.read_rgba()
        return rgba

    def render_gpu(self, crank_angle_deg: float, spin: bool = True, dt: float = 0.0,
                   throttle_frac: float = 1.0) -> int:
        """Same draw as render(), but leaves the result sitting in the
        FBO's own colour texture and returns its GL texture id --
        no glReadPixels, no CPU round trip at all. The compositor
        (gl_compositor.py) draws that texture id straight into the
        main window's backbuffer as a quad; the pixels never visit
        Python. This is the path main_pygame.py uses every tick.

        throttle_frac drives the SEPARATE baked throttle-plate frame
        set (build_throttle_animation) composited alongside whatever
        crank_angle_deg frame is showing -- a real, independent live
        parameter, not folded into the crank-angle bake."""
        self._draw(crank_angle_deg, spin=spin, dt=dt, throttle_frac=throttle_frac)
        return self._target.color_tex

    def _draw(self, crank_angle_deg: float, spin: bool, dt: float, throttle_frac: float = 1.0) -> None:
        if spin:
            self._angle_rad = (self._angle_rad + dt * 0.25) % (2.0 * np.pi)
        # the caller's own viewport (main_pygame's whole window) is
        # restored on the way out -- leaving this FBO's 620x620 viewport
        # bound is what rasterised every later composite (the engine
        # quad itself, the legend) into the window's bottom-left corner
        prev_viewport = tuple(int(v) for v in glGetIntegerv(GL_VIEWPORT))
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
        dir_to_eye = np.array([np.sin(az), float(getattr(self, "_elevation", 0.40)), np.cos(az)],
                              dtype=np.float64)
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
        distance = (max(fit_v, fit_h) * 1.25 + 1e-4) * float(getattr(self, "_zoom", 1.0))

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

        key = eye + np.array([-0.3, 0.6, -0.2], dtype=np.float32) * self._scale
        fill = (self._center - (eye - self._center) * 0.55
                + np.array([0.0, self._scale, 0.0], dtype=np.float32))
        overhead = self._center + np.array(
            [0.0, self._scale * 2.6, 0.0], dtype=np.float32)
        rim = (self._center - dir_to_eye.astype(np.float32) * self._scale * 1.8
               + np.array([self._scale * .5, self._scale * .6, 0.0],
                          dtype=np.float32))
        light_pos = [key, fill, overhead, rim]
        light_col = [np.array([1.0, 0.98, 0.94], dtype=np.float32),
                     np.array([0.62, 0.70, 0.88], dtype=np.float32),
                     np.array([0.96, 0.97, 1.0], dtype=np.float32),
                     np.array([0.72, 0.82, 1.0], dtype=np.float32)]
        # Point-light intensity is an emitter area in the shader and must
        # scale with scene length squared.  The former linear scale left a
        # station-sized graph almost entirely on its 12% ambient floor.
        area_scale = self._scale * self._scale
        light_int = [area_scale * 12.0, area_scale * 7.0,
                     area_scale * 10.0, area_scale * 6.0]
        for pos, colour, intensity in (self._thermal_emitters() + self._flame_emitters())[:96]:
            light_pos.append(pos); light_col.append(colour); light_int.append(intensity)
        self._renderer.set_point_lights(
            positions=np.stack(light_pos).astype(np.float32),
            colors=np.stack(light_col).astype(np.float32),
            intensities=np.array(light_int, dtype=np.float32),
        )
        if self._static_gl is not None:
            self._renderer.draw_mesh(self._static_gl.vao, self._static_gl.n_vertices, mvp_gl, mv_gl, enable_blend=True)
        idx = self._frame_index_for(crank_angle_deg)
        if idx is not None and self._frame_gl[idx] not in (None, EMPTY_FRAME):
            fg = self._frame_gl[idx]
            self._renderer.draw_mesh(fg.vao, fg.n_vertices, mvp_gl, mv_gl, enable_blend=True)
        tg = self._throttle_frame_for(throttle_frac)
        if tg is not None:
            self._renderer.draw_mesh(tg.vao, tg.n_vertices, mvp_gl, mv_gl, enable_blend=True)
        # the live burns and residue: one small draw per active cylinder
        for cyl, k, _rgb in self._live_flames:
            fg = self._flame_gl.get((cyl, k))
            if fg is not None:
                self._renderer.draw_mesh(fg.vao, fg.n_vertices, mvp_gl, mv_gl, enable_blend=True, depth_write=False)
        for cyl, k in self._live_smoke:
            sg = self._smoke_gl.get((cyl, k))
            if sg is not None:
                self._renderer.draw_mesh(sg.vao, sg.n_vertices, mvp_gl, mv_gl, enable_blend=True, depth_write=False)
        if self._leak_gl is not None:
            self._renderer.draw_mesh(self._leak_gl.vao, self._leak_gl.n_vertices, mvp_gl, mv_gl, enable_blend=True)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(*prev_viewport)

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
            if i >= len(MATERIALS):
                continue   # a live-thermal variant row stands in for a base material already listed
            m = MATERIALS[i]
            out.append((m.label, tuple(int(c * 255) for c in m.albedo), m.opacity))
        return out

    def close(self) -> None:
        self._free_geometry()
        self._target.delete()
