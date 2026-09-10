"""A tiny 2-D compositor for an OpenGL-mode pygame window: draws an
arbitrary GL texture as a screen-space quad at pixel coordinates. This
is the ONE extra shader the app needs beyond base_material's Phong
program -- an unlit, orthographic, textured quad -- and it is what
lets everything the window shows (the engine mesh, the dashboard
text, the legend swatches) be a real GPU draw call instead of a
software Surface blit.

Two things get composited through this, both GPU-resident already:
  - the engine view's own FBO colour attachment (rendered by the real
    Phong shader in engine_gl_view.py) -- drawn straight from its
    texture id, no CPU readback, no re-upload
  - text: rendered to a small pygame Surface by ordinary font
    rendering (that was never the bottleneck) and uploaded once per
    change as a GL texture, then redrawn as a quad every frame like
    anything else on the screen

Pixel coordinates match the old pygame blit convention exactly (origin
top-left, +x right, +y down), so replacing `screen.blit(surf, (x,
y))` with `compositor.draw_texture(tex, x, y, w, h)` (or
`text.draw(line, x, y, color)`) preserves every existing layout
constant in main_pygame.py.
"""
from __future__ import annotations

import ctypes

import numpy as np

from OpenGL.GL import (
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS, GL_LINK_STATUS,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE,
    GL_TRIANGLE_STRIP, GL_TEXTURE_2D, GL_TEXTURE0, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_DEPTH_TEST,
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer, glBufferData,
    glEnableVertexAttribArray, glVertexAttribPointer,
    glUseProgram, glGetUniformLocation, glUniform2f, glUniform4f, glUniform1i,
    glActiveTexture, glBindTexture, glDrawArrays, glEnable, glDisable, glBlendFunc,
)

_VERT_SRC = """#version 330 core
layout(location = 0) in vec2 aPos;   // pixel-space corner, 0..1 unit quad
layout(location = 1) in vec2 aUv;
uniform vec2 uRectPos;    // top-left, pixels
uniform vec2 uRectSize;   // width/height, pixels
uniform vec2 uViewport;   // window width/height, pixels
out vec2 vUv;
void main() {
    vec2 px = uRectPos + aPos * uRectSize;
    vec2 ndc = vec2(px.x / uViewport.x, 1.0 - px.y / uViewport.y) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUv = aUv;
}
"""

_FRAG_SRC = """#version 330 core
in vec2 vUv;
out vec4 FragColor;
uniform sampler2D uTex;
uniform vec4 uTint;
uniform bool uFlipV;
void main() {
    vec2 uv = uFlipV ? vec2(vUv.x, 1.0 - vUv.y) : vUv;
    FragColor = texture(uTex, uv) * uTint;
}
"""


def _compile(src: str, kind) -> int:
    sid = glCreateShader(kind)
    glShaderSource(sid, src)
    glCompileShader(sid)
    if not glGetShaderiv(sid, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sid).decode())
    return sid


class Compositor:
    """One shared unit quad + shader program for every 2-D texture
    draw the app makes; construct once after a GL context is current."""

    def __init__(self) -> None:
        vs = _compile(_VERT_SRC, GL_VERTEX_SHADER)
        fs = _compile(_FRAG_SRC, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vs); glAttachShader(prog, fs)
        glLinkProgram(prog)
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(prog).decode())
        glDeleteShader(vs); glDeleteShader(fs)
        self._prog = prog
        self._u_pos = glGetUniformLocation(prog, "uRectPos")
        self._u_size = glGetUniformLocation(prog, "uRectSize")
        self._u_viewport = glGetUniformLocation(prog, "uViewport")
        self._u_tex = glGetUniformLocation(prog, "uTex")
        self._u_tint = glGetUniformLocation(prog, "uTint")
        self._u_flip = glGetUniformLocation(prog, "uFlipV")

        verts = np.array([0, 0, 0, 0,  1, 0, 1, 0,  0, 1, 0, 1,  1, 1, 1, 1], dtype=np.float32)
        self._vao = int(glGenVertexArrays(1))
        self._vbo = int(glGenBuffers(1))
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)

    def draw_texture(self, tex_id: int, x: float, y: float, w: float, h: float,
                     viewport_w: int, viewport_h: int, tint=(1.0, 1.0, 1.0, 1.0), flip_v: bool = False) -> None:
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self._prog)
        glUniform2f(self._u_pos, float(x), float(y))
        glUniform2f(self._u_size, float(w), float(h))
        glUniform2f(self._u_viewport, float(viewport_w), float(viewport_h))
        glUniform4f(self._u_tint, *tint)
        glUniform1i(self._u_flip, 1 if flip_v else 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        if self._u_tex != -1:
            glUniform1i(self._u_tex, 0)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_BLEND)
