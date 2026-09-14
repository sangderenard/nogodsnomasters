"""The firing trial, rendered: a live window, or an animated PNG.

    python firing_frame_view.py --live            watch it solve
    python firing_frame_view.py                   write firing_trial.png
    python firing_frame_view.py --live --stand-in skip the ballistics kernel

Colour is not decoration. Every member is painted by the fraction of its
own yield it is using at that instant, so the load path through the
machine is the thing you look at rather than something reconstructed
from a table of identities.

ONE BAKE. The first version of this built a whole EngineGLView and baked
the entire 2236-part mesh once per instant of the shot, which is minutes
of work to redraw geometry that never changed shape -- only place and
colour. Both of those already have systems here:

    articulation.ArticulatedMesh   moves vertices through the span
                                   arithmetic, no rebake
    EngineGLView.restage_static_mesh  pushes the moved vertices and the
                                   new material ids as a buffer write

So the mesh is built once, `ArticulatedMesh.build(..., per_node=True)`
attributes every vertex to the node that carries it, and each frame is
a displace and a restage.
"""
from __future__ import annotations

import sys

import numpy as np

import firing_frame_trial as fft


def _band_material_ids():
    """One material per utilisation band, registered with the mesh
    system so the existing renderer uses them unchanged. Returns the
    material INDEX per band, which is what a triangle carries."""
    from engine_mesh import (Material, _rgb, DECLARED_MATERIALS, MATERIALS,
                             MATERIAL_INDEX)
    ids = {}
    for i, (_limit, colour, label) in enumerate(fft.UTILISATION_BANDS):
        name = f"util{i}"
        if name not in MATERIAL_INDEX:
            m = Material(name, label, _rgb(colour), 1.0, 0.30, 0.22,
                         0.55, 40.0, 0.78)
            DECLARED_MATERIALS[f"utilisation::{i}"] = m
            MATERIALS.append(m)
            MATERIAL_INDEX[name] = len(MATERIALS) - 1
        ids[colour] = MATERIAL_INDEX[name]
    return ids


def material_ids_for_colour_mode(base_material_ids, triangle_indices,
                                 member_owner, utilisation, bands,
                                 band_material_ids, mode: str):
    """Select authored assembly colours or live yield-utilisation bands."""
    material_ids = np.array(base_material_ids, copy=True)
    if mode == "assembly":
        return material_ids
    if mode != "yield":
        raise ValueError(f"unknown structural colour mode {mode!r}")
    triangles = np.asarray(triangle_indices, dtype=np.int64)
    if triangles.size:
        owner = np.asarray(member_owner, dtype=np.int64)
        util = np.asarray(utilisation, dtype=float)
        band = np.searchsorted(np.asarray(bands, dtype=float), util)
        palette = np.asarray(band_material_ids, dtype=np.int64)
        material_ids[triangles] = palette[np.minimum(
            band[owner], len(palette) - 1)]
    return material_ids


class Trial:
    """A baked view of one machine, and the shot driven through it."""

    def __init__(self, document: dict, *, width: int = 1280,
                 height: int = 880, hidden: bool = True):
        import pygame
        pygame.display.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.HIDDEN if hidden else 0)
        pygame.display.set_mode((width if not hidden else 64,
                                 height if not hidden else 64), flags)
        if not hidden:
            pygame.display.set_caption(
                "firing trial -- colour is fraction of yield")
        from engine_gl_view import EngineGLView
        from articulation import ArticulatedMesh
        self.document = document
        self.view = EngineGLView(width=width, height=height, covers_off=True,
                                 animation_divisions="dense", detail=0.7,
                                 spring_style="cylinder")
        self.view.set_graph(document)
        while self.view.bake_next():
            pass
        self.mesh = self.view._static_mesh
        self.articulated = ArticulatedMesh.build(document, self.mesh,
                                                 per_node=True)
        self.window = (width, height)
        self.compositor = None
        if not hidden:
            from gl_compositor import Compositor
            self.compositor = Compositor()
        self.band_ids = _band_material_ids()
        self.rest_material_ids = np.array(self.mesh.material_ids, copy=True)
        # WHICH TRIANGLES BELONG TO WHICH MEMBER, resolved once. The
        # mesh names its parts after the edges that made them, so this
        # is a lookup rather than anything that has to be maintained.
        from articulation import _flat
        self.tri_range = {}
        for e in document["edges"]:
            self.tri_range[e["identity"]] = "edge_" + _flat(e["identity"])
        self.part_range = dict(zip(self.mesh.part_names, self.mesh.part_ranges))

    def show(self, frame: dict) -> None:
        """Put the machine where the solve says it is, in the colours the
        loads say it deserves -- one vertex write, one buffer upload."""
        self.articulated.displace(frame["position"])
        mat = self.rest_material_ids.copy()
        for ident, (util, _stress) in frame["utilisation"].items():
            rng = self.part_range.get(self.tri_range.get(ident))
            if rng is None:
                continue
            colour, _label = fft.band_for(util)
            mat[rng[0]:rng[1]] = self.band_ids[colour]
        self.mesh.material_ids = mat
        self.view.restage_static_mesh()

    def draw(self, azimuth: float):
        """Offscreen, read back to a numpy image: the export path."""
        self.view._angle_rad = float(azimuth)
        return self.view.render(crank_angle_deg=0.0, spin=False, dt=0.0,
                                throttle_frac=0.0)

    def present(self, azimuth: float) -> None:
        """Draw into the WINDOW, through the same compositor the live
        app uses.

        EngineGLView.render() leaves its result in an offscreen FBO and
        hands back a numpy copy, which is right for writing a PNG and
        useless for a window: flipping the display then presents a
        default framebuffer nothing ever drew to, and the window is
        black. render_gpu() returns the FBO's colour texture instead and
        gl_compositor.Compositor draws it into the backbuffer as a quad
        -- the path main_pygame.py runs every tick, with no CPU round
        trip."""
        from OpenGL.GL import glViewport, glClearColor, glClear,             GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
        self.view._angle_rad = float(azimuth)
        tex = self.view.render_gpu(crank_angle_deg=0.0, spin=False, dt=0.0,
                                   throttle_frac=0.0)
        glViewport(0, 0, self.window[0], self.window[1])
        glClearColor(0.06, 0.06, 0.07, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.compositor.draw_texture(tex, 0.0, 0.0,
                                     float(self.window[0]),
                                     float(self.window[1]),
                                     self.window[0], self.window[1],
                                     flip_v=True)


def _banner(frames: list) -> None:
    print(f"recoil force from: {frames[0]['source']}", flush=True)
    print(f"{len(frames)} instants, peak "
          f"{max(f['force_n'] for f in frames) / 1e3:.1f} kN, "
          f"peak deflection "
          f"{max(f['max_deflection_m'] for f in frames) * 1000:.1f} mm",
          flush=True)


def animate(document: dict, path: str = "firing_trial.png", *, steps: int = 16,
            width: int = 1280, height: int = 880, azimuth: float = 0.62,
            ballistics: bool = True, frames: list | None = None) -> str:
    """`frames` lets any solver drive this -- the static trial or the
    ring-down -- since both hand back the same shape."""
    trial = Trial(document, width=width, height=height, hidden=True)
    if frames is None:
        frames = fft.trial(document, steps=steps, ballistics=ballistics)
    _banner(frames)
    shots = []
    for fr in frames:
        trial.show(fr)
        shots.append(np.ascontiguousarray(trial.draw(azimuth)[:, :, :3]))
    from PIL import Image
    pages = [Image.fromarray(s) for s in shots]
    pages[0].save(path, save_all=True, append_images=pages[1:],
                  duration=110, loop=0)
    return path


def live(document: dict, *, steps: int = 24, width: int = 1280,
         height: int = 880, ballistics: bool = True) -> None:
    """Watch it, looping, turning. Escape or the window close ends it."""
    import pygame
    trial = Trial(document, width=width, height=height, hidden=False)
    frames = fft.trial(document, steps=steps, ballistics=ballistics)
    _banner(frames)
    clock = pygame.time.Clock()
    angle, i, running = 0.62, 0, True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                          and ev.key == pygame.K_ESCAPE):
                running = False
        fr = frames[i % len(frames)]
        trial.show(fr)
        trial.present(angle)
        pygame.display.flip()
        angle += 0.005
        i += 1
        clock.tick(12)
    pygame.quit()


if __name__ == "__main__":
    import turret_production as tp
    doc = tp.build_gimbal_cannon_station(bore_mm=20.0).as_document()
    use_ballistics = "--stand-in" not in sys.argv
    n = 8 if "--quick" in sys.argv else 24
    if "--live" in sys.argv:
        live(doc, steps=n, ballistics=use_ballistics)
    else:
        print("wrote", animate(doc, steps=n, ballistics=use_ballistics))
