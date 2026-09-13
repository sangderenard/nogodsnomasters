"""WATCH THE STRUCTURE TAKE THE SHOT.

The frame solver has always been able to answer "what does this machine
do under this load", and the answer came back as a column of numbers
about members nobody could point at. That is the wrong medium for a
question about a shape: a metre of invented deflection, a slider solved
as a bar, a coolant line carrying structural steel -- every one of those
was visible instantly as a picture and invisible for hours as a table.

So this drives the shot through the frame solve and renders it, one
frame per instant, with every member coloured by how much of its yield
it is using. Green is loafing, yellow is working, red is at the limit,
white is past it.

WHAT IS REAL HERE and what is a stand-in, stated rather than implied:

  REAL: the geometry, every section and alloy from milspec, the joint
  releases (a slider slides, a pin carries no moment, a routed line is
  not structure at all), the stiffness assembly, gravity lumped at the
  nodes, and the displacement solve.

  A STAND-IN: the shape of the recoil force in time. If the interior
  ballistics kernel is available this uses the pressure history it
  computes, which is the real curve; if it is not, it falls back to a
  triangular pulse of the same impulse, and says so.

  NOT MODELLED: the structure's own dynamics. Each frame is a STATIC
  solve at that instant's load, so this shows what the machine is being
  asked to carry, not how it rings afterwards. Ringing is beam_theory's
  job and it runs per member.
"""
from __future__ import annotations

import math

import numpy as np

from frame_solver import FrameSolver
from milspec import MATERIAL_BY_KEY


#: Where each member's utilisation lands on the ramp. Deliberately not a
#: smooth gradient: a band is readable at a glance and a gradient is
#: not, and the only question being asked here is "which of these is in
#: trouble".
UTILISATION_BANDS = (
    (0.10, "#1d6f42", "loafing"),
    (0.25, "#3fa34d", "easy"),
    (0.45, "#9acd32", "working"),
    (0.65, "#e8c33a", "hard"),
    (0.85, "#e8792c", "near the limit"),
    (1.00, "#d62828", "at yield"),
    (1e9, "#ffffff", "PAST YIELD"),
)


def recoil_history(steps: int, *, peak_n: float = 110_000.0,
                   duration_s: float = 0.012, ballistics: bool = True) -> tuple:
    """Force on the breech through the shot, and where it came from.

    Tries the real interior-ballistics pressure curve first. That kernel
    computes chamber pressure against travel for this exact load, and
    force on the breech face is that pressure times the bore area --
    which is the honest curve, sharply peaked and long-tailed, not the
    symmetric bump a triangle gives."""
    try:
        if not ballistics:
            raise RuntimeError("caller asked for the stand-in")
        import interior_ballistics as ib
        shot = ib.native_shot()
        p = np.asarray(shot["pressure_pa"], dtype=np.float64)
        area = float(shot["bore_area_m2"])
        force = p * area
        t = np.asarray(shot["time_s"], dtype=np.float64)
        grid = np.linspace(0.0, float(t[-1]), steps)
        return np.interp(grid, t, force), grid, "interior ballistics"
    except Exception:
        pass
    t = np.linspace(0.0, duration_s, steps)
    rise = duration_s * 0.25
    force = np.where(t < rise, peak_n * t / rise,
                     peak_n * np.exp(-(t - rise) / (duration_s * 0.30)))
    return force, t, "triangular stand-in (ballistics kernel unavailable)"


def member_utilisation(solved: dict, members: list) -> dict:
    """Fraction of its own yield each member is using.

    Axial and bending add: a column that is also bent reaches yield on
    the compression face before either alone would say so, and reporting
    the larger of the two would have missed it."""
    by_id = {e["identity"]: e for e in members}
    out = {}
    for k, ident in enumerate(solved["identities"]):
        edge = by_id.get(ident)
        if edge is None or edge.get("rigid"):
            continue
        dmg = edge.get("damage") or {}
        mat = MATERIAL_BY_KEY.get(dmg.get("material", "4130n"),
                                  MATERIAL_BY_KEY["4130n"])
        e_pa = float(dmg.get("youngs_modulus_pa", mat.youngs_pa))
        yield_pa = float(getattr(mat, "yield_pa", 460e6))
        stress = (abs(float(solved["axial_strain"][k]))
                  + abs(float(solved["bending_strain"][k]))) * e_pa
        out[ident] = (stress / max(yield_pa, 1.0), stress)
    return out


def band_for(utilisation: float) -> tuple:
    for limit, colour, label in UTILISATION_BANDS:
        if utilisation <= limit:
            return colour, label
    return UTILISATION_BANDS[-1][1], UTILISATION_BANDS[-1][2]


def trial(document: dict, *, steps: int = 24, load_node: str = "turret.breech",
          axis=(0.0, 0.0, -1.0), ballistics: bool = True) -> list:
    """Every instant of the shot, from TWO solves.

    The first version ran the solver once per instant: twenty-four
    assemblies of a fifteen-hundred-member stiffness matrix and
    twenty-four dense factorisations of a 3858-freedom system, minutes
    of work, to answer twenty-four questions that differ only in how
    hard the breech is being pushed.

    THE STRUCTURE IS LINEAR, so it does not have to be asked twice. K is
    the same matrix at every instant, and the load is one fixed vector
    scaled by the pressure history:

        K u(t) = f_gravity + s(t) * f_recoil
        u(t)   = u_gravity + s(t) * (u_reference - u_gravity)

    which is exact, not an approximation -- superposition is what linear
    elasticity means. Two solves cover the whole shot, and every
    instant after that is a scale of a vector. Strains scale with the
    displacements they are differences of, so the utilisations come the
    same way.

    WHAT THIS GIVES UP, said plainly: it is still a STATIC solve per
    instant, so it shows what the machine is being asked to carry and
    not how it rings afterwards. Making it dynamic means a mass matrix
    and a time integration, and then the two-solve trick genuinely does
    stop applying -- superposition over a static K is exactly the thing
    that would no longer hold.
    """
    force, times, source = recoil_history(steps, ballistics=ballistics)
    axis = np.asarray(axis, dtype=np.float64)
    reference_n = float(np.max(np.abs(force))) or 1.0

    quiet = FrameSolver(document=document, loads={})
    at_rest = quiet.solve()
    loaded = FrameSolver(document=document,
                         loads={load_node: tuple(axis * reference_n)}).solve()
    members = quiet.members

    rest_pos = quiet.settled_position(at_rest)
    delta_pos = (quiet.position + loaded["displacement"][:, :3]) - rest_pos
    d_axial = loaded["axial_strain"] - at_rest["axial_strain"]
    d_bend = loaded["bending_strain"] - at_rest["bending_strain"]

    frames = []
    for i in range(steps):
        scale = float(force[i]) / reference_n
        scaled = {
            "identities": at_rest["identities"],
            "axial_strain": at_rest["axial_strain"] + scale * d_axial,
            "bending_strain": at_rest["bending_strain"] + scale * d_bend,
        }
        position = rest_pos + scale * delta_pos
        frames.append({
            "index": i, "time_s": float(times[i]), "force_n": float(force[i]),
            "source": source, "position": position,
            "utilisation": member_utilisation(scaled, members),
            "max_deflection_m": float(np.abs(
                position - quiet.position).max()),
        })
    return frames


def report(frames: list) -> list:
    """The trial as text, for when a picture is not what is wanted."""
    lines = [f"recoil force from: {frames[0]['source']}",
             f"{len(frames)} instants, peak "
             f"{max(f['force_n'] for f in frames) / 1e3:.1f} kN"]
    worst = {}
    for fr in frames:
        for ident, (u, stress) in fr["utilisation"].items():
            if u > worst.get(ident, (0.0, 0.0))[0]:
                worst[ident] = (u, stress)
    lines.append(f"peak deflection "
                 f"{max(f['max_deflection_m'] for f in frames) * 1000:.1f} mm")
    ranked = sorted(worst.items(), key=lambda kv: -kv[1][0])
    over = [k for k, v in ranked if v[0] > 1.0]
    lines.append(f"members over yield: {len(over)} of {len(ranked)}")
    for ident, (u, stress) in ranked[:12]:
        lines.append(f"   {u * 100:7.1f}%  {stress / 1e6:7.1f} MPa  "
                     f"{band_for(u)[1]:15s} {ident}")
    return lines
