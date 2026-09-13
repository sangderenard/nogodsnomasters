"""WHIP IT SIDE TO SIDE AND SEE WHAT COMPLAINS.

    python slew_trial.py             spec + slew_trial.png
    python slew_trial.py --live      watch it, drag to orbit

The outriggers are on the ground. The drum is given everything the drive
can make and told to reverse, repeatedly, and the structure is solved at
every instant with the loads a slew actually produces:

    TANGENTIAL   m * alpha * r     the mass being accelerated
    CENTRIPETAL  m * omega^2 * r   the mass being held on its circle
    WEIGHT       m * g             which never stops

AND THE TURNTABLE IS NOT BALANCED. It cannot be: the trunnion balances
the gun about its own pivot, which says nothing about where the turret's
mass sits relative to the axis it TURNS on. Five tonnes of gun hangs
four metres forward; a counterweight on the back rim pulls the centre
of mass back toward the axis but does not put it there, and what is
left over is thrown round as a rotating load every time the thing
slews. That residue is the point of this trial -- a perfectly balanced
turntable would tell you nothing you could not get from a static solve.

REVERSALS ARE THE WORST CASE, not the peak speed. At the ends of the
sweep the drive is pushing one way while the mass is still going the
other, so the torque and the momentum add instead of cancelling. A
machine that survives its top speed can still tear itself apart
stopping.
"""
from __future__ import annotations

import math
import sys

import numpy as np

import turret_production as tp
import firing_frame_trial as fft
from frame_solver import FrameSolver, DOF_PER_NODE
from milspec import MATERIAL_BY_KEY


#: What the drive makes with the inner face locked to direct drive and
#: the outer metering: 6 units x 63 kN at the 1.402 m track radius.
DRIVE_TORQUE_NM = 530_000.0
#: How far it sweeps each way before reversing.
SWEEP_DEG = 60.0


def turning_bodies(document: dict) -> list:
    """Everything that goes round, with its mass and its radius.

    Wrench points and internal channels are excluded for the reasons
    they always are: a port is a place, not a body, and a channel is
    metal that is not there."""
    out = []
    for n in document["nodes"]:
        if n.get("wrench_point") or n.get("internal_channel"):
            continue
        if str(n.get("motion_group")) == "frame":
            continue
        p = np.asarray(n["reference_position"], dtype=float)
        r = math.hypot(p[0], p[2])
        out.append((n["identity"], float(n.get("mass_kg", 0.0)), p, r))
    return out


def inertia_kg_m2(bodies) -> float:
    return sum(m * r * r for _i, m, _p, r in bodies)


def sweep(document: dict, *, steps: int = 32) -> list:
    """A full side-to-side cycle, with the state at every instant."""
    bodies = turning_bodies(document)
    I = inertia_kg_m2(bodies)
    mass = sum(m for _i, m, _p, _r in bodies)
    alpha = DRIVE_TORQUE_NM / max(I, 1e-9)
    half = math.radians(SWEEP_DEG) / 2.0
    # accelerate for half the sweep, brake for the other half: the
    # fastest a bang-bang drive can cover it and come to rest
    t_half = math.sqrt(half / max(alpha, 1e-9))
    peak_w = alpha * t_half
    out = []
    for k in range(steps):
        f = k / steps
        phase = f * 4.0                       # four quarters in a cycle
        sign = 1.0 if phase < 2.0 else -1.0
        q = phase % 2.0
        accelerating = q < 1.0
        t = (q if accelerating else 2.0 - q) * t_half
        w = sign * alpha * t
        a = sign * alpha * (1.0 if accelerating else -1.0)
        out.append({"index": k, "time_s": f * 4.0 * t_half,
                    "omega": w, "alpha": a,
                    "accelerating": accelerating,
                    "phase": ("accelerating" if accelerating else "braking")})
    return out, {"inertia_kg_m2": I, "mass_kg": mass, "alpha_rad_s2": alpha,
                 "peak_rad_s": peak_w, "half_time_s": t_half}


def loads_at(bodies, omega: float, alpha: float) -> dict:
    """The load every turning body puts into the structure right now."""
    loads = {}
    for ident, m, p, r in bodies:
        if m <= 0.0 or r < 1e-6:
            continue
        radial = np.array([p[0], 0.0, p[2]]) / r
        tangent = np.cross(np.array([0.0, 1.0, 0.0]), radial)
        f = tangent * (-m * alpha * r) + radial * (-m * omega * omega * r)
        loads[ident] = tuple(f)
    return loads


def solve_step(document: dict, bodies, state) -> dict:
    fs = FrameSolver(document=document,
                     loads=loads_at(bodies, state["omega"], state["alpha"]),
                     gravity=True)
    solved = fs.solve()
    by_id = {e["identity"]: e for e in fs.members}
    util = {}
    for j, ident in enumerate(solved["identities"]):
        e = by_id.get(ident)
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
    return {**state, "position": fs.settled_position(solved),
            "utilisation": util, "source": "slew trial",
            "force_n": abs(state["alpha"]) * 1000.0,
            "max_deflection_m": float(solved["max_deflection_m"])}


def main(argv) -> None:
    g = tp.balanced_station(bore_mm=20.0)
    doc = g.as_document()
    bodies = turning_bodies(doc)
    states, spec = sweep(doc)
    cg_m = sum(m for _i, m, _p, _r in bodies)
    cg = sum(m * p for _i, m, p, _r in bodies) / max(cg_m, 1e-9)
    print(f"  turning mass      {spec['mass_kg']:8.0f} kg", flush=True)
    print(f"  polar inertia     {spec['inertia_kg_m2']:8.0f} kg.m2", flush=True)
    print(f"  centre of mass    {math.hypot(cg[0], cg[2]):8.3f} m off the axis"
          f"  ({cg_m * 9.80665 * math.hypot(cg[0], cg[2]) / 1000:.0f} kN.m"
          f" of standing offset)", flush=True)
    print(f"  drive             {DRIVE_TORQUE_NM / 1000:8.0f} kN.m"
          f"  -> {spec['alpha_rad_s2']:.4f} rad/s2", flush=True)
    print(f"  {SWEEP_DEG:.0f} deg sweep       {2 * spec['half_time_s']:8.2f} s"
          f"  peaking at {math.degrees(spec['peak_rad_s']):.1f} deg/s", flush=True)

    frames = [solve_step(doc, bodies, st) for st in states]
    worst = {}
    for fr in frames:
        for ident, (u, s) in fr["utilisation"].items():
            if u > worst.get(ident, (0.0, 0.0))[0]:
                worst[ident] = (u, s)
    ranked = sorted(worst.items(), key=lambda kv: -kv[1][0])
    over = [k for k, v in ranked if v[0] > 1.0]
    print(f"\n  peak deflection   "
          f"{max(f['max_deflection_m'] for f in frames) * 1000:8.2f} mm")
    print(f"  members over yield: {len(over)} of {len(ranked)}\n")
    for ident, (u, s) in ranked[:10]:
        print(f"   {u * 100:7.1f}%  {s / 1e6:7.1f} MPa  "
              f"{fft.band_for(u)[1]:15s} {ident}")

    import firing_frame_view as ffv
    if "--live" in argv:
        import pygame
        trial = ffv.Trial(doc, width=1360, height=900, hidden=False)
        clock, az, el, i, running = pygame.time.Clock(), 0.62, 0.45, 0, True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                              and ev.key == pygame.K_ESCAPE):
                    running = False
                elif ev.type == pygame.MOUSEMOTION and ev.buttons[0]:
                    az -= ev.rel[0] * 0.008
                    el = max(-0.9, min(1.4, el + ev.rel[1] * 0.005))
            trial.show(frames[i % len(frames)])
            trial.view._elevation = el
            trial.present(az)
            pygame.display.flip()
            i += 1
            clock.tick(14)
        pygame.quit()
    else:
        print("\n  wrote", ffv.animate(doc, "slew_trial.png", frames=frames,
                                       width=1360, height=900), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
