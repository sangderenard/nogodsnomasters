"""GPU text: pygame's font rasterizer draws each line to a small CPU
surface (that was never the cost -- font rendering is a few dozen
microseconds), and this module's only job is to get that surface onto
the screen as a GL texture quad through gl_compositor instead of a
software Surface.blit. Every line the dashboard shows becomes one
small texture upload (cheap: a few KB) and one quad draw call.

A tiny cache keyed on (text, color, font id) skips the re-upload for
any line that reads identically to last frame -- the static help text
and the legend labels hit this every tick; the live numbers don't, and
don't need to: uploading a 700x18 RGBA texture costs low tens of
microseconds, nowhere near the 147 ms/frame the old per-triangle
polygon path cost.
"""
from __future__ import annotations

import numpy as np
import pygame

from OpenGL.GL import (
    GL_TEXTURE_2D, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR, GL_CLAMP_TO_EDGE,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glDeleteTextures,
)

from gl_compositor import Compositor


class _TexCache:
    __slots__ = ("tex_id", "w", "h", "key")

    def __init__(self, tex_id: int, w: int, h: int, key) -> None:
        self.tex_id, self.w, self.h, self.key = tex_id, w, h, key


def _upload_surface(surf: "pygame.Surface") -> tuple[int, int, int]:
    w, h = surf.get_size()
    # no pre-flip: row 0 of the buffer stays the surface's own top row,
    # which is exactly what this module's UV mapping expects (v=0 ->
    # quad top, flip_v=False) -- text draws right-side up with no
    # shader-side flip. The engine mesh's FBO texture is the opposite
    # case (GL's own bottom-left origin) and passes flip_v=True instead.
    data = pygame.image.tostring(surf, "RGBA", False)
    tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex, w, h


class TextLayer:
    """Call once per frame per line: draw(text, x, y, color). Owns its
    own Compositor (shared quad shader) so callers need nothing else
    to put a string of pixels on a GL-mode window."""

    def __init__(self, font: "pygame.font.Font", viewport_w: int, viewport_h: int) -> None:
        self.font = font
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.compositor = Compositor()
        self._cache: dict[tuple, _TexCache] = {}
        self._seen_this_frame: set = set()

    def set_viewport(self, w: int, h: int) -> None:
        self.viewport_w, self.viewport_h = w, h

    def begin_frame(self) -> None:
        self._seen_this_frame = set()

    def end_frame(self) -> None:
        """Free textures for strings that didn't appear this frame
        (a log entry that expired, a dashboard line that vanished)."""
        stale = [k for k in self._cache if k not in self._seen_this_frame]
        for k in stale:
            glDeleteTextures([self._cache.pop(k).tex_id])

    def draw(self, text: str, x: float, y: float, color=(255, 255, 255), alpha: float = 1.0) -> tuple[int, int]:
        """Draws `text` with its top-left at (x, y) in the same pixel
        coordinates the old `screen.blit(font.render(...), (x, y))`
        used. Returns (width, height) of the rendered text."""
        if not text:
            return (0, 0)
        key = (text, tuple(color), id(self.font))
        self._seen_this_frame.add(key)
        cached = self._cache.get(key)
        if cached is None:
            surf = self.font.render(text, True, color).convert_alpha()
            tex, w, h = _upload_surface(surf)
            cached = _TexCache(tex, w, h, key)
            self._cache[key] = cached
        self.compositor.draw_texture(cached.tex_id, x, y, cached.w, cached.h,
                                     self.viewport_w, self.viewport_h, tint=(1.0, 1.0, 1.0, alpha))
        return (cached.w, cached.h)

    def draw_surface(self, surf: "pygame.Surface", x: float, y: float, cache_key=None) -> None:
        """For pre-composed surfaces (the legend gradient block): upload
        once (cache_key identifies it) and redraw as a quad every frame
        after that -- the same 'upload once, hand it to the shader
        every tick' rule the engine mesh follows."""
        key = cache_key if cache_key is not None else id(surf)
        self._seen_this_frame.add(key)
        cached = self._cache.get(key)
        if cached is None:
            tex, w, h = _upload_surface(surf.convert_alpha())
            cached = _TexCache(tex, w, h, key)
            self._cache[key] = cached
        self.compositor.draw_texture(cached.tex_id, x, y, cached.w, cached.h, self.viewport_w, self.viewport_h)

    def close(self) -> None:
        for c in self._cache.values():
            glDeleteTextures([c.tex_id])
        self._cache.clear()
