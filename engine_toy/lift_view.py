"""LIFT THE GUN OFF ITS MOUNT so every connection between them shows.

    python lift_view.py                      lift 1.0 m, write lift_*.png
    python lift_view.py --lift 1.6           further
    python lift_view.py --live               watch it, turning
    python lift_view.py --groups fine,elevation    lift only these

A machine drawn in one piece hides the only thing worth looking at,
which is where its parts meet. Separating the moving assembly from the
structure that carries it leaves every member that crosses the gap
stretched across open air, one per connection, and they can then be
counted, named and argued about.

THIS IS NOT AN EXPLODED DIAGRAM DRAWN BY HAND. The lift is applied to
DECLARED motion groups, and every member is stretched by the same span
arithmetic that poses the machine when it elevates -- so a member that
appears in the gap is genuinely a member joining those two groups, and
one that does not is genuinely not. Nothing is placed for the picture.

Colour is by declared assembly, so the thing in the gap can be named.
"""
from __future__ import annotations

import sys

import numpy as np

#: WHAT MOVES WITH THE GUN. Declared motion groups, not a list of part
#: names -- `fine` is the gun and everything bolted to it, `elevation`
#: is the cradle that carries it, `carriage` the rail blocks. The base,
#: the outriggers and the traverse ring stay behind.
GUN_GROUPS = ("fine", "elevation", "carriage")

ASSEMBLY_COLOURS = {
    "base-frame": "#6b7684", "outriggers": "#ffd21f", "ammunition": "#3fa34d",
    "base-stores": "#a9865b", "powerplant": "#ff6b1f", "gun-tube": "#2d6cdf",
    "breech": "#e8eaee", "cradle": "#e0572c", "gun-gimbal": "#ff8c42",
    "gun-support-structure": "#ff5d5d", "recoil-gear": "#ff3ea5",
    "trunnion-thrust-seat": "#ff85c8", "fine-rig-base": "#8a4fff",
    "fine-rig-platform": "#b98cff", "fine-rig-mount": "#ffffff",
    "slew-ring": "#19c3b2", "traverse-ring-drive": "#0f8f86",
    "secondary-mg": "#d4d400", "under-barrel-spine": "#00e5ff",
    "bore-evacuator": "#7cf1ff", "muzzle-reference": "#fff176",
    "barrel-cooling": "#4fc3f7", "recoil-rail": "#ff1744",
    "recoil-absorber": "#ff8a00", "recoil-journal": "#ff1744",
    "turntable-drum": "#19c3b2", "slew-race": "#ffe066",
    "turntable-floor": "#8a4fff",
    "recoil-rail": "#ff1744",
}


def lifted_positions(document: dict, lift_m: float,
                     groups=GUN_GROUPS) -> dict:
    """Where every node sits with the named groups raised."""
    wanted = set(groups)
    out = {}
    for n in document["nodes"]:
        p = np.asarray(n["reference_position"], dtype=np.float64).copy()
        if str(n.get("motion_group")) in wanted:
            p[1] += float(lift_m)
        out[n["identity"]] = p
    return out


def crossings(document: dict, groups=GUN_GROUPS) -> list:
    """Every member with one end in the lifted set and one end out.

    These are the connections the gap exposes, and this is the list the
    picture is a picture OF -- so what is seen and what is reported
    cannot disagree."""
    wanted = set(groups)
    group_of = {n["identity"]: str(n.get("motion_group"))
                for n in document["nodes"]}
    out = []
    for e in document["edges"]:
        a, b = group_of.get(e["a"]), group_of.get(e["b"])
        if (a in wanted) == (b in wanted):
            continue
        out.append(e)
    return out


def _paint(document: dict):
    """Register one material per declared assembly and stamp it on."""
    from engine_mesh import (Material, _rgb, DECLARED_MATERIALS, MATERIALS,
                             MATERIAL_INDEX)
    for i, name in enumerate(sorted(ASSEMBLY_COLOURS)):
        key = f"assembly::{name}"
        if key not in DECLARED_MATERIALS:
            m = Material(f"asm{i}", name, _rgb(ASSEMBLY_COLOURS[name]), 1.0,
                         0.34, 0.24, 0.6, 50.0, 0.75)
            DECLARED_MATERIALS[key] = m
            MATERIALS.append(m)
            MATERIAL_INDEX[m.name] = len(MATERIALS) - 1
    nodes, edges = [], []
    for n in document["nodes"]:
        key = f"assembly::{n['assembly']}"
        nodes.append(dict(n, material=key) if key in DECLARED_MATERIALS else n)
    for e in document["edges"]:
        key = f"assembly::{e['assembly']}"
        edges.append(dict(e, material=key, in_view=not e.get("rigid"))
                     if key in DECLARED_MATERIALS
                     else dict(e, in_view=not e.get("rigid")))
    return dict(document, nodes=nodes, edges=edges)


def build(document: dict, *, width=1500, height=1000, hidden=True):
    import pygame
    pygame.display.init()
    flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.HIDDEN if hidden else 0)
    pygame.display.set_mode((width, height) if not hidden else (64, 64), flags)
    if not hidden:
        pygame.display.set_caption("lifted: every member crossing the gap")
    from engine_gl_view import EngineGLView
    from articulation import ArticulatedMesh
    painted = _paint(document)
    view = EngineGLView(width=width, height=height, covers_off=True,
                        animation_divisions="dense", detail=0.7,
                        spring_style="cylinder")
    view.set_graph(painted)
    while view.bake_next():
        pass
    art = ArticulatedMesh.build(painted, view._static_mesh, per_node=True)
    return view, art


def main(argv) -> None:
    import turret_production as tp
    lift = 1.0
    if "--lift" in argv:
        lift = float(argv[argv.index("--lift") + 1])
    groups = GUN_GROUPS
    if "--groups" in argv:
        groups = tuple(argv[argv.index("--groups") + 1].split(","))
    doc = tp.balanced_station(bore_mm=20.0).as_document()

    cross = crossings(doc, groups)
    print(f"lifting {', '.join(groups)} by {lift:.2f} m", flush=True)
    print(f"{len(cross)} members cross the gap:", flush=True)
    seen = {}
    for e in cross:
        seen.setdefault((e["assembly"], e["constraint"]), []).append(e["identity"])
    for (asm, con), ids in sorted(seen.items()):
        print(f"   {len(ids):3d}  {asm:22s} {con:34s} {ids[0]}"
              + (f" (+{len(ids) - 1} more)" if len(ids) > 1 else ""), flush=True)

    live = "--live" in argv
    view, art = build(doc, hidden=not live)
    art.displace(lifted_positions(doc, lift, groups))
    view.restage_static_mesh()
    if live:
        import pygame
        from gl_compositor import Compositor
        comp = Compositor()
        from OpenGL.GL import (glViewport, glClearColor, glClear,
                               GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT)
        clock, angle, running = pygame.time.Clock(), 0.62, True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                              and ev.key == pygame.K_ESCAPE):
                    running = False
            view._angle_rad = angle
            tex = view.render_gpu(crank_angle_deg=0.0, spin=False, dt=0.0,
                                  throttle_frac=0.0)
            glViewport(0, 0, 1500, 1000)
            glClearColor(0.06, 0.06, 0.07, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            comp.draw_texture(tex, 0.0, 0.0, 1500.0, 1000.0, 1500, 1000,
                              flip_v=True)
            pygame.display.flip()
            angle += 0.006
            clock.tick(30)
        pygame.quit()
        return
    import pygame
    for azimuth, name in ((0.62, "lift_q.png"), (1.571, "lift_side.png"),
                          (2.60, "lift_rear.png")):
        view._angle_rad = azimuth
        img = view.render(crank_angle_deg=0.0, spin=False, dt=0.0,
                          throttle_frac=0.0)
        rgb = np.ascontiguousarray(img[:, :, :3])
        pygame.image.save(pygame.image.frombuffer(
            rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB"), name)
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
