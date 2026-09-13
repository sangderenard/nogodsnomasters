"""THE DRUM JOINT ON ITS OWN, at standard parameters.

    python drum_reference.py --live    solve it live, orbit with the mouse
    python drum_reference.py           spec sheet + drum_ref_*.png

This is the reference article. The turret is one thing built ON this
joint, and there will be others -- anything that has to turn a heavy
overhung mass on a big radius wants exactly this and not a flat track.
It gets a page of its own, away from the gun, because a joint you can
only see buried inside a machine is one nobody can check.

WHAT THE JOINT IS, in the order the load travels:

    the truss bolts anywhere on the TOP ANNULUS
    the OUTER and INNER WALLS close the cell, so the section carries
        torsion as shear flow instead of twisting like a strip
    the MIDDLE is a machined annular volume, pocketed up into the top
        plate and down into the bottom one -- the drive lives in here
        and nothing crosses a face to reach it
    the last band of radius top and bottom stays FULL THICKNESS: that
        is the greased mating floor the two CAPTIVE RACES press against
    the BOTTOM ANNULUS stands on the platform

and the two races at full cell depth are what make an overturning load
a push/pull PAIR rather than compression on one face that can only push.

THE LIVE WINDOW SOLVES. It does not orbit a still picture: the member
bank -- `beam_theory`'s SymPy law, printed into a loop by
`structure_native`, lowered by the compiler and cached by source
digest -- is stepped every frame at the real elapsed dt, and every
member is recoloured by the fraction of its own yield it is using.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from surfaces import BoxedRing, BearingRace, emit_boxed_ring, emit_race
from turret_production import ProductionGraph


#: The standard article. Sized from the turret's own seat, so the
#: reference and the machine are the same joint rather than two that
#: merely look alike.
STANDARD = dict(
    inner_radius_m=0.588,
    outer_radius_m=1.588,          # a metre of annulus
    height_m=0.300,
    plate_thickness_m=0.040,
    wall_thickness_m=0.025,
    material="6061t6",
    rim_material="4340qt",
    drive_pocket_m=0.025,
    mating_band_m=0.180,
    segments=16,
)

PALETTE = {
    "drum-plate": "#19c3b2",
    "drum-wall": "#6b7684",
    "race": "#ffe066",
    "static": "#3a4048",
    "motor": "#e8465a",
    "track-out": "#9aa7b5",
    "track-in": "#c9a227",
}

ROLE = {"drum.upper": "drum-plate", "drum.lower": "drum-plate",
        "drum.motor": "motor", "drum.rotor": "motor",
        "drum.cradle": "motor", "drum.thrust": "motor",
        "drum.hub": "motor", "drum.groove": "motor",
        "drum.track.outer": "track-out", "drum.track.inner": "track-in",
        "drum.contact": "motor", "drum.lockup": "motor",
        "drum.wall": "drum-wall", "drum.shear": "drum-wall",
        "race.": "race", "seat.": "static"}


def build(**overrides):
    """One drum, its two races, and the static ring under them."""
    spec = dict(STANDARD)
    spec.update(overrides)
    g = ProductionGraph(identity="drum-reference")
    segments = spec["segments"]
    g.motion_group, g.assembly = "frame", "static"
    seat_r = spec["outer_radius_m"] + 0.05
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        # AT MID HEIGHT, not on the floor: the static ring straddles the
        # drum's rim and reaches UP to one race and DOWN to the other.
        # On the floor it lands exactly on the lower race and the members
        # between them have zero length.
        g.node(f"seat.{i}", [seat_r * math.cos(a), spec["height_m"] / 2.0,
                             seat_r * math.sin(a)],
               "chassis-load-node", material="steel-plate", fixed_to="world",
               part_role="static-ring-seat",
               half_extent_m=(0.08, 0.055, 0.08))
    for i in range(segments):
        g.edge(f"seat.beam.{i}", f"seat.{i}", f"seat.{(i + 1) % segments}",
               "rigid-distance", radius=0.038, palette="chassis-grey",
               load_path="the-static-ring-the-drum-turns-on")

    box = BoxedRing(identity="drum", centre=(0.0, spec["height_m"] / 2.0, 0.0),
                    **{k: v for k, v in spec.items() if k != "segments"},
                    segments=segments)
    drum = emit_boxed_ring(g, box, motion_group="traverse",
                           assembly="drum-plate")
    races = {}
    for tag, rim in (("upper", drum["upper"]["outer"]),
                     ("lower", drum["lower"]["outer"])):
        y = spec["height_m"] / 2.0 + (spec["height_m"] / 2.0 if tag == "upper"
                                      else -spec["height_m"] / 2.0)
        race = BearingRace(identity=f"race.{tag}", stations=segments,
                           centre=(0.0, y, 0.0), radius_m=seat_r,
                           sweep_deg=360.0, preload_n=54_000.0)
        races[tag] = emit_race(g, race, lambda i, _r=rim: _r[i],
                               motion_group="frame", assembly="race")
        for i in range(segments):
            g.edge(f"race.{tag}.seat.{i}", f"race.{tag}.element.{i}",
                   f"seat.{i}", "rigid-distance", radius=0.042,
                   palette="chassis-grey",
                   load_path=f"{tag}-race-seated-on-the-static-ring")
    return g, box, drum, races, spec


def spec_sheet(box: BoxedRing, document: dict) -> list:
    n = {x["identity"]: x for x in document["nodes"]}
    sec = box._face("upper").section
    mass = sum(float(x["mass_kg"]) for x in document["nodes"]
               if not x.get("wrench_point")
               and str(x.get("motion_group", "")).startswith(("traverse",
                                                              "drive-unit")))
    m0 = n["drum.motor.0"]
    hub = n["drum.hub.0"]
    out = [
        "DRUM JOINT -- standard article",
        f"  annulus            r {box.inner_radius_m:.3f} .. "
        f"{box.outer_radius_m:.3f} m   "
        f"({box.outer_radius_m - box.inner_radius_m:.3f} m radial)",
        f"  cell depth         {box.height_m * 1000:.0f} mm"
        f"   plate {box.plate_thickness_m * 1000:.0f} mm"
        f"   wall {box.wall_thickness_m * 1000:.0f} mm",
        f"  body {box.material}, rim {box.rim_material}",
        f"  rotating mass      {mass:.0f} kg",
        "",
        "  TORSION -- why it is boxed",
        f"    open section     J = {box.open_torsion_m4:.3e} m4",
        f"    closed cell      J = {box.closed_torsion_m4:.3e} m4"
        f"   ({box.closed_torsion_m4 / box.open_torsion_m4:.0f}x)",
        "",
        "  PLATE SECTION -- why a ring is not a beam",
        f"    out of plane     I = {sec['second_moment_out_m4']:.3e} m4"
        f"   (the overhang loads this one)",
        f"    in plane         I = {sec['second_moment_in_m4']:.3e} m4"
        f"   ({sec['second_moment_in_m4'] / sec['second_moment_out_m4']:.0f}x)",
        "",
        "  THE DRIVE -- donuts between two tracks",
        f"    {box.drive_units} units, r {m0['drum_radius_m']:.3f} m, "
        f"pitch {m0['pitch_radius_m']:.3f} m, axes parallel to the drum",
        f"    outer track      r {n['drum.track.outer.0']['track_radius_m']:.3f} m"
        f"  (hangs from the top annulus)",
        f"    inner track      r {n['drum.track.inner.0']['track_radius_m']:.3f} m"
        f"  ({n['drum.track.inner.0']['material']}, on the bottom)",
        f"    rotor inertia    {m0['rotor_inertia_kg_m2']:.1f} kg.m2 each",
        "",
        "  THE HUB -- power, air and oil across a turning joint",
        f"    seal diameter    {hub['seal_diameter_m'] * 1000:.0f} mm"
        f"  (the rim is {box.outer_radius_m * 2000:.0f} mm)",
    ]
    for pt in hub.get("ports", []):
        out.append(f"    {pt['name']:11s} {pt['fluid']:14s} "
                   f"{pt['flow_l_min']:7.1f} L/min at "
                   f"{pt['velocity_m_s']:5.2f} m/s -> "
                   f"{pt['slot_width_m'] * 1000:5.2f} mm slot")
    return out


def _paint(document: dict):
    from engine_mesh import (Material, _rgb, DECLARED_MATERIALS, MATERIALS,
                             MATERIAL_INDEX)
    for i, (name, colour) in enumerate(PALETTE.items()):
        key = f"drumref::{name}"
        if key not in DECLARED_MATERIALS:
            m = Material(f"dr{i}", name, _rgb(colour), 1.0, 0.32, 0.22,
                         0.6, 45.0, 0.8)
            DECLARED_MATERIALS[key] = m
            MATERIALS.append(m)
            MATERIAL_INDEX[m.name] = len(MATERIALS) - 1

    def pick(ident: str) -> str:
        for prefix, name in ROLE.items():
            if ident.startswith(prefix):
                return f"drumref::{name}"
        return "drumref::drum-wall"

    nodes = [dict(x, material=pick(x["identity"])) for x in document["nodes"]]
    edges = [dict(e, material=pick(e["identity"]),
                  in_view=not e.get("rigid")) for e in document["edges"]]
    return dict(document, nodes=nodes, edges=edges)


def _view(painted, W, H, live):
    """Paint BEFORE the view exists.

    A material registered after EngineGLView is constructed is not in
    the table the renderer built, so every part draws with an index that
    resolves to nothing and the image comes back as bare background.
    Nothing errors; the picture is simply empty."""
    import pygame
    pygame.display.init()
    pygame.display.set_mode((W, H) if live else (64, 64),
                            pygame.OPENGL | pygame.DOUBLEBUF
                            | (0 if live else pygame.HIDDEN))
    if live:
        pygame.display.set_caption("drum joint -- colour is fraction of yield")
    from engine_gl_view import EngineGLView
    view = EngineGLView(width=W, height=H, covers_off=True,
                        animation_divisions="dense", detail=0.8,
                        spring_style="cylinder")
    view.set_graph(painted)
    while view.bake_next():
        pass
    return view


def live_window(doc, box, painted, W=1400, H=950) -> None:
    """The bank stepped every frame, the mesh recoloured every frame."""
    import pygame
    import structure_native as sn
    import firing_frame_trial as fft
    from articulation import _flat
    from gl_compositor import Compositor
    from milspec import MATERIAL_BY_KEY
    from OpenGL.GL import (glViewport, glClearColor, glClear,
                           GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT)
    from engine_mesh import (Material, _rgb, DECLARED_MATERIALS, MATERIALS,
                             MATERIAL_INDEX)

    view = _view(painted, W, H, True)
    print("  loading the member bank ...", flush=True)
    bank = sn.bank_for_target(doc, target="python")
    print(f"  {bank.n} members, digest {bank.source_digest}", flush=True)

    edges = {e["identity"]: e for e in doc["edges"]}
    yld, emod, radius, length = [], [], [], []
    for ident in bank.identities:
        d = edges[ident].get("damage") or {}
        mat = MATERIAL_BY_KEY.get(d.get("material", "4130n"),
                                  MATERIAL_BY_KEY["4130n"])
        yld.append(float(getattr(mat, "yield_pa", 460e6)))
        emod.append(float(d.get("youngs_modulus_pa", mat.youngs_pa)))
        radius.append(float(edges[ident].get("radius", 0.012)))
        length.append(float(edges[ident].get("rest_length", 1.0)))
    yld = np.asarray(yld)
    emod = np.asarray(emod)
    radius = np.asarray(radius)
    length = np.maximum(np.asarray(length), 1e-6)

    band_mat = []
    for bi, (_lim, colour, label) in enumerate(fft.UTILISATION_BANDS):
        nm = f"util{bi}"
        if nm not in MATERIAL_INDEX:
            m = Material(nm, label, _rgb(colour), 1.0, 0.30, 0.22,
                         0.55, 40.0, 0.78)
            DECLARED_MATERIALS[f"utilisation::{bi}"] = m
            MATERIALS.append(m)
            MATERIAL_INDEX[nm] = len(MATERIALS) - 1
        band_mat.append(MATERIAL_INDEX[nm])
    band_mat = np.asarray(band_mat)
    bands = np.asarray([lim for lim, _c, _l in fft.UTILISATION_BANDS])
    view._renderer.update_material_ssbo()

    part_range = dict(zip(view._static_mesh.part_names,
                          view._static_mesh.part_ranges))
    ranges = [part_range.get("edge_" + _flat(i)) for i in bank.identities]
    owner = np.concatenate([np.full(r[1] - r[0], i, np.int32)
                            for i, r in enumerate(ranges) if r])
    tris = np.concatenate([np.arange(r[0], r[1]) for r in ranges if r])
    base_mat = np.array(view._static_mesh.material_ids, copy=True)

    # ---- THE FORCE COMES OFF THE JOINTS, not off a dial ----
    # Every term is read from what the parts declared when they were
    # built: the rotors' torque, the donut's own radius, each face's
    # friction coefficient and what the spring presses it with, the
    # lock-up's clamp and plate count. What crosses is min(outer, inner),
    # because two faces in series pass what the weaker one holds.
    n_by = {x["identity"]: x for x in doc["nodes"]}
    faces = {}
    for e in doc["edges"]:
        if e.get("part_role") == "traction-contact":
            faces.setdefault(e["a"], {})[e.get("contact")] = e
    lockups = {e["a"]: e for e in doc["edges"]
               if e.get("part_role") == "inner-lockup"}

    def joint_force(outer, inner, locked, media):
        unit = "drum.motor.0"
        made = sum(float(n_by[k].get("torque_nm", 0.0))
                   for k in (f"drum.rotor.{m}.0" for m in media) if k in n_by)
        wants = made / max(float(n_by[unit].get("drum_radius_m", 1.0)), 1e-9)
        h = {}
        for tag, frac in (("outer", outer), ("inner", inner)):
            f = faces.get(unit, {}).get(tag)
            h[tag] = (float(f.get("friction_coefficient", 0.0))
                      * float(f.get("presses_with_n", 0.0))
                      * frac) if f else 0.0
        if locked and unit in lockups:
            lk = lockups[unit]
            h["inner"] = (float(lk.get("clamp_force_n", 0.0))
                          * float(lk.get("friction_coefficient", 0.35))
                          * float(lk.get("plates", 1)))
        passes = max(min(wants, h["outer"], h["inner"]), 0.0)
        why = ("the rotors" if passes >= wants - 1e-6 else
               "the outer face" if h["outer"] <= h["inner"] else
               "the inner face")
        return passes, why

    driven = np.array([i for i, ident in enumerate(bank.identities)
                       if ".track_hoop." in ident or ".rim_beam." in ident],
                      dtype=np.int64)
    if driven.size == 0:
        driven = np.arange(min(16, bank.n))

    comp = Compositor()
    clock = pygame.time.Clock()
    az, el, zoom = 0.6, 0.45, 1.0
    drag = paused = locked = False
    running = True
    outer_cmd, inner_cmd = 1.0, 1.0
    media = ("mag", "hyd", "pne")
    frames = 0
    print("  SPACE pause   L lock-up   O/I clutches   R reset   ESC quit",
          flush=True)
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                          and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                paused = not paused
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_l:
                locked = not locked
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_o:
                outer_cmd = 0.0 if outer_cmd > 0.5 else 1.0
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_i:
                inner_cmd = 0.0 if inner_cmd > 0.5 else 1.0
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                bank.q[:] = 0.0
                bank.qdot[:] = 0.0
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                drag = True
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (4, 5):
                zoom *= 0.9 if ev.button == 4 else 1.1
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                drag = False
            elif ev.type == pygame.MOUSEMOTION and drag:
                az -= ev.rel[0] * 0.008
                el = max(-0.9, min(1.4, el + ev.rel[1] * 0.005))

        dt = min(clock.tick(60) / 1000.0, 0.05)
        passes, why = joint_force(outer_cmd, inner_cmd, locked, media)
        if not paused:
            bank.arrays["force_n"][:] = 0.0
            bank.arrays["force_n"][driven] = passes / max(driven.size, 1)
            out = bank.step(dt)  # the bank subdivides by its stiffest member
        else:
            out = bank.out.reshape(bank.n, 3)

        # a modal slope is a curvature is a stress, through the section
        stress = np.abs(out[:, 1]) * emod * radius / length
        util = stress / np.maximum(yld, 1.0)
        mat = base_mat.copy()
        band = np.minimum(np.searchsorted(bands, util), len(band_mat) - 1)
        mat[tris] = band_mat[band[owner]]
        view._static_mesh.material_ids = mat
        view.restage_static_mesh()

        view._angle_rad, view._elevation, view._zoom = az, el, zoom
        tex = view.render_gpu(crank_angle_deg=0.0, spin=False, dt=0.0,
                              throttle_frac=0.0)
        glViewport(0, 0, W, H)
        glClearColor(0.06, 0.06, 0.07, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        comp.draw_texture(tex, 0.0, 0.0, float(W), float(H), W, H,
                          flip_v=True)
        pygame.display.flip()
        frames += 1
        if frames % 60 == 0:
            print(f"  {clock.get_fps():5.1f} fps   out {outer_cmd:.0f} "
                  f"in {inner_cmd:.0f} lock {'Y' if locked else 'n'}   "
                  f"{passes / 1000:6.1f} kN/unit ({why})   "
                  f"peak {util.max() * 100:6.2f}% of yield   "
                  f"tip {np.abs(out[:, 0]).max() * 1000:7.4f} mm", flush=True)
    pygame.quit()


def main(argv) -> None:
    over = {}
    for key, flag in (("height_m", "--depth"), ("outer_radius_m", "--outer"),
                      ("mating_band_m", "--band")):
        if flag in argv:
            over[key] = float(argv[argv.index(flag) + 1])
    g, box, drum, races, spec = build(**over)
    doc = g.as_document()
    problems = g.check()
    print("\n".join(spec_sheet(box, doc)), flush=True)
    print(f"\n  check              {'PASS' if not problems else problems[:2]}",
          flush=True)
    print(f"  graph              {len(doc['nodes'])} nodes, "
          f"{len(doc['edges'])} edges", flush=True)
    painted = _paint(doc)

    if "--live" in argv:
        live_window(doc, box, painted)
        return

    import pygame
    view = _view(painted, 1400, 950, False)
    for az, el, name in ((0.62, 0.45, "drum_ref_q.png"),
                         (0.00, 0.06, "drum_ref_side.png"),
                         (0.62, 1.35, "drum_ref_top.png")):
        view._angle_rad, view._elevation, view._zoom = az, el, 1.0
        img = view.render(crank_angle_deg=0.0, spin=False, dt=0.0,
                          throttle_frac=0.0)
        rgb = np.ascontiguousarray(img[:, :, :3])
        pygame.image.save(pygame.image.frombuffer(
            rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB"), name)
        print("  wrote", name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
