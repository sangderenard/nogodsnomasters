"""DRIVE THE DRUM AND WATCH WHAT YIELDS.

    python drum_drive_trial.py            spec + drum_drive.png
    python drum_drive_trial.py --live     watch it, orbit with the mouse

The bottom annulus is bolted down. Everything else is free, and the six
donuts are told what to do -- which clutches, how hard, which media --
and the structure is solved for it, every member coloured by the
fraction of its own yield it is using.

WHAT IS APPLIED, and why it is not just "torque":

    a donut can only deliver what its CONTACTS will carry. The outer
    face is dry at mu 0.35, the inner is traction fluid at 0.08, and
    what crosses is min(outer, inner) of the engagement -- so a unit
    commanded to full output on one clutch delivers NOTHING and simply
    spins. The tangential force is the smaller of what the rotors make
    and what the pressed faces will hold, and both are computed rather
    than assumed.

THE SCENARIO runs through the states that matter: floating, one clutch
in (spin, no drive), both in and rising, and then a pop -- everything
at once, which is the load case the structure is really being asked
about.
"""
from __future__ import annotations

import math
import sys

import numpy as np

import drum_reference as dref
import firing_frame_trial as fft
from frame_solver import FrameSolver, DOF_PER_NODE
from milspec import MATERIAL_BY_KEY
from surfaces import traction_state


#: (label, outer engagement, inner engagement, which media are driving)
SCENARIO = (
    ("floating, nothing touching", 0.0, 0.0, ()),
    ("inner in: the donuts spin", 0.0, 1.0, ("hyd",)),
    ("both in, easing on", 0.35, 0.35, ("hyd",)),
    ("both in, hydraulic hard", 1.0, 0.8, ("hyd",)),
    ("electric joins to hold", 1.0, 1.0, ("hyd", "mag")),
    ("POP -- all three, both faces", 1.0, 1.0, ("hyd", "mag", "pne")),
)


def unit_force(document: dict, box, outer: float, inner: float,
               media) -> dict:
    """What one donut actually puts into the track, this instant."""
    n = {x["identity"]: x for x in document["nodes"]}
    made = sum(float(n[f"drum.rotor.{m}.0"]["torque_nm"]) for m in media)
    r_d = float(n["drum.motor.0"]["drum_radius_m"])
    wants = made / max(r_d, 1e-6)
    state = traction_state({"outer": outer, "inner": inner})
    held = min(box.clutch_engagement_n * 0.35 * outer,
               box.clutch_engagement_n * 0.08 * inner)
    passes = min(wants, held) if state["transmits"] else 0.0
    return {"rotor_torque_nm": made, "wants_n": wants,
            "contacts_hold_n": held, "passes_n": passes,
            "limited_by": ("nothing -- not engaged" if not state["transmits"]
                           else "the contacts" if held < wants
                           else "the rotors"),
            "drum_torque_nm": passes * float(
                n["drum.track.outer.0"]["track_radius_m"]) * box.drive_units}


def loads_for(document: dict, box, passes_n: float) -> dict:
    """Tangential force at each unit, put on the track it pushes.

    Tangential means tangential: the direction is computed from where
    the unit sits, not applied as a convenient global push."""
    n = {x["identity"]: x for x in document["nodes"]}
    centre = np.asarray(box.centre, float)
    out = {}
    for k in range(box.drive_units):
        ident = f"drum.motor.{k}"
        p = np.asarray(n[ident]["reference_position"], float) - centre
        radial = np.array([p[0], 0.0, p[2]])
        radial /= max(np.linalg.norm(radial), 1e-9)
        tangent = np.cross(np.array([0.0, 1.0, 0.0]), radial)
        i = int(round(k * box.segments / box.drive_units)) % box.segments
        out[f"drum.track.outer.{i}"] = tuple(tangent * passes_n)
        out[ident] = tuple(-tangent * passes_n)
    return out


def solve_state(document: dict, box, outer, inner, media) -> dict:
    u = unit_force(document, box, outer, inner, media)
    fs = FrameSolver(document=document, loads=loads_for(document, box,
                                                        u["passes_n"]),
                     gravity=True)
    solved = fs.solve()
    util = {}
    for j, ident in enumerate(solved["identities"]):
        e = next((m for m in fs.members if m["identity"] == ident), None)
        if e is None or e.get("rigid"):
            continue
        d = e.get("damage") or {}
        mat = MATERIAL_BY_KEY.get(d.get("material", "4130n"),
                                  MATERIAL_BY_KEY["4130n"])
        e_pa = float(d.get("youngs_modulus_pa", mat.youngs_pa))
        y = float(getattr(mat, "yield_pa", 460e6))
        stress = (abs(float(solved["axial_strain"][j]))
                  + abs(float(solved["bending_strain"][j]))) * e_pa
        util[ident] = (stress / max(y, 1.0), stress)
    return {**u, "position": fs.settled_position(solved),
            "utilisation": util,
            "max_deflection_m": float(solved["max_deflection_m"]),
            "source": "drive trial", "force_n": u["passes_n"],
            "time_s": 0.0, "index": 0}


def build():
    """The reference drum with its bottom annulus bolted down."""
    g, box, drum, races, spec = dref.build()
    for ident in drum["lower"]["mean"]:
        node = next(x for x in g.nodes if x["identity"] == ident)
        node["fixed_to"] = "world"
    return g, box, drum


def main(argv) -> None:
    g, box, drum = build()
    doc = g.as_document()
    print(f"  bottom annulus locked: {box.segments} stations", flush=True)
    frames, rows = [], []
    for label, outer, inner, media in SCENARIO:
        st = solve_state(doc, box, outer, inner, media)
        frames.append(st)
        worst = max(st["utilisation"].values(), default=(0.0, 0.0))
        rows.append((label, outer, inner, media, st, worst))
    print(f"\n  {'state':30s} {'out':>4s} {'in':>4s} "
          f"{'passes':>9s} {'drum':>10s} {'worst':>7s}  limited by")
    for label, outer, inner, media, st, worst in rows:
        print(f"  {label:30s} {outer:4.2f} {inner:4.2f} "
              f"{st['passes_n'] / 1000:7.1f} kN "
              f"{st['drum_torque_nm'] / 1000:8.1f} kNm "
              f"{worst[0] * 100:6.1f}%  {st['limited_by']}", flush=True)
    over = [i for i, (_l, _o, _i, _m, st, _w) in enumerate(rows)
            if any(u > 1.0 for u, _s in st["utilisation"].values())]
    print(f"\n  states with a member over yield: "
          f"{len(over)} of {len(rows)}", flush=True)

    import firing_frame_view as ffv
    if "--live" in argv:
        import pygame
        trial = ffv.Trial(doc, width=1280, height=880, hidden=False)
        clock, az, el, i, running = pygame.time.Clock(), 0.62, 0.5, 0, True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                              and ev.key == pygame.K_ESCAPE):
                    running = False
                elif ev.type == pygame.MOUSEMOTION and ev.buttons[0]:
                    az -= ev.rel[0] * 0.008
                    el = max(-0.9, min(1.4, el + ev.rel[1] * 0.005))
            trial.show(frames[(i // 12) % len(frames)])
            trial.view._elevation = el
            trial.present(az)
            pygame.display.flip()
            i += 1
            clock.tick(24)
        pygame.quit()
    else:
        held = []
        for st in frames:
            held.extend([st] * 6)
        print("  wrote", ffv.animate(doc, "drum_drive.png", frames=held,
                                     width=1280, height=880), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
