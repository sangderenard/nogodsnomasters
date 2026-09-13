"""THE PLATFORM RINGING, INTEGRATED WITH ITSELF.

    python platform_ringdown.py              write platform_ringdown.png
    python platform_ringdown.py --live       watch it
    python platform_ringdown.py --modes 40   how much of the structure to keep

`firing_frame_trial` answers "what is this machine being asked to carry
at this instant" -- a static solve per instant, exact and completely
deaf. It cannot tell you what happens AFTER the shot, and after the shot
is when a gun platform decides whether the next round leaves from a
stationary gun.

This is the other question. One structure, coupled, integrated forward:

    M u.. + C u. + K u = f(t)

projected onto the structure's own mode shapes, which is what makes it
tractable -- 4044 freedoms become the forty modes that carry the motion,
each an independent oscillator, and the coupling is already inside the
shapes. THAT is the sense in which it integrates with itself: every
member's strain at every instant is the sum of what all the modes are
doing, not what that member is doing alone.

    q_i.. + 2 zeta w_i q_i. + w_i^2 q_i = phi_i^T f(t)

integrated semi-implicitly, for the same reason beam_theory does it:
explicit Euler on an oscillator gains energy every step, and a structure
integrated that way rings itself apart instead of down.

WHAT IS REAL: the geometry, the sections and alloys from milspec, the
joint releases, the assembled stiffness and lumped mass, the eigenmodes
of that pair, and the integration. The damping ratio is DECLARED -- a
bolted steel structure is 1-2% of critical and there is no way to
compute it from geometry, so it is stated here rather than derived from
nothing.

WHAT IS NOT: the shot's force history is the same stand-in the static
trial uses unless the ballistics kernel is present, and it says which.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from frame_solver import FrameSolver, DOF_PER_NODE, _local_frame
from milspec import MATERIAL_BY_KEY
import firing_frame_trial as fft


#: Critical damping fraction for a bolted steel frame. Declared, because
#: it cannot be derived from the geometry and pretending otherwise would
#: be inventing the one number that decides how long this rings.
STRUCTURAL_DAMPING = 0.015


def _member_strain(solver, u: np.ndarray) -> tuple:
    """Axial and bending strain in every member, for a displacement.

    Vectorised over members: the static recovery walks them one at a
    time, which is right for one solve and wrong for four hundred
    frames of one."""
    idents, axial, bending = [], [], []
    for edge in solver.members:
        ia, ib = solver.index[edge["a"]], solver.index[edge["b"]]
        a, b = solver.position[ia], solver.position[ib]
        L = float(np.linalg.norm(b - a))
        if L < 1e-9:
            continue
        R = _local_frame(a, b)
        ua = u[ia * DOF_PER_NODE:ia * DOF_PER_NODE + 6]
        ub = u[ib * DOF_PER_NODE:ib * DOF_PER_NODE + 6]
        da, ra = R @ ua[:3], R @ ua[3:]
        db, rb = R @ ub[:3], R @ ub[3:]
        ro = float(edge.get("radius", 0.012))
        released = set(edge.get("freedoms_released") or ())
        ax = 0.0 if "x" in released else (db[0] - da[0]) / L
        bd = 0.0 if released & {"ry", "rz"} else math.hypot(
            (rb[1] - ra[1]) / L, (rb[2] - ra[2]) / L) * ro
        idents.append(edge["identity"])
        axial.append(ax)
        bending.append(bd)
    return idents, np.asarray(axial), np.asarray(bending)


def _yield_and_modulus(solver) -> tuple:
    y, e = [], []
    for edge in solver.members:
        d = edge.get("damage") or {}
        mat = MATERIAL_BY_KEY.get(d.get("material", "4130n"),
                                  MATERIAL_BY_KEY["4130n"])
        y.append(float(getattr(mat, "yield_pa", 460e6)))
        e.append(float(d.get("youngs_modulus_pa", mat.youngs_pa)))
    return np.asarray(y), np.asarray(e)


def ringdown(document: dict, *, modes: int = 40, steps: int = 180,
             duration_s: float = 0.60, load_node: str = "turret.breech",
             axis=(0.0, 0.0, -1.0), damping: float = STRUCTURAL_DAMPING,
             ballistics: bool = True) -> list:
    """Integrate the structure through the shot and past it.

    Returns a frame per step with displaced positions and every member's
    utilisation, the same shape `firing_frame_trial` returns, so the
    renderer does not care which produced it."""
    solver = FrameSolver(document=document, loads={})
    modal = solver.modes(count=modes)
    shapes = modal["shapes"][:, :modes]
    omega = 2.0 * math.pi * np.asarray(modal["all_frequencies_hz"][:modes])
    n_dof = shapes.shape[0]

    # the static settle under gravity: what it rings ABOUT
    rest = solver.solve()
    u_static = rest["displacement"].reshape(-1)

    # the shot, as a force on one node along the bore
    force_t, times, source = fft.recoil_history(steps, ballistics=ballistics)
    t_end = float(times[-1])
    # ...and then silence, which is the half of this that matters
    dt = duration_s / steps
    drive = np.zeros(steps)
    n_shot = max(int(t_end / dt), 1)
    drive[:n_shot] = np.interp(np.linspace(0.0, t_end, n_shot), times, force_t)

    f_dir = np.zeros(n_dof)
    i = solver.index[load_node]
    f_dir[i * DOF_PER_NODE:i * DOF_PER_NODE + 3] = np.asarray(axis, float)
    modal_force = shapes.T @ f_dir            # phi^T f, per mode, per newton

    q = np.zeros(modes)
    qd = np.zeros(modes)
    yield_pa, e_pa = _yield_and_modulus(solver)
    idents_ref = None
    frames = []
    for k in range(steps):
        f = float(drive[k])
        accel = modal_force * f - 2.0 * damping * omega * qd - omega ** 2 * q
        qd = qd + accel * dt                  # semi-implicit: velocity first
        q = q + qd * dt
        u = u_static + shapes @ q
        idents, ax, bd = _member_strain(solver, u)
        if idents_ref is None:
            idents_ref = idents
        stress = (np.abs(ax) + np.abs(bd)) * e_pa[:len(ax)]
        util = stress / np.maximum(yield_pa[:len(ax)], 1.0)
        frames.append({
            "index": k, "time_s": k * dt, "force_n": f, "source": source,
            "position": solver.position + u.reshape(-1, DOF_PER_NODE)[:, :3],
            "utilisation": {ident: (float(uu), float(ss))
                            for ident, uu, ss in zip(idents, util, stress)},
            "max_deflection_m": float(np.abs(
                u.reshape(-1, DOF_PER_NODE)[:, :3]).max()),
            "modal_energy_j": float(0.5 * np.sum(qd ** 2 + (omega * q) ** 2)),
        })
    return frames


def report(frames: list, modal=None) -> list:
    peak = max(f["max_deflection_m"] for f in frames)
    e0 = max(f["modal_energy_j"] for f in frames)
    tail = frames[-1]["modal_energy_j"]
    out = [
        f"recoil from: {frames[0]['source']}",
        f"{len(frames)} steps over {frames[-1]['time_s']:.3f} s, "
        f"damping {STRUCTURAL_DAMPING * 100:.1f}% of critical",
        f"peak deflection {peak * 1000:.2f} mm",
        f"modal energy {e0:.3g} -> {tail:.3g} "
        f"({tail / max(e0, 1e-30) * 100:.1f}% still ringing at the end)",
    ]
    worst = {}
    for fr in frames:
        for ident, (u, s) in fr["utilisation"].items():
            if u > worst.get(ident, (0.0, 0.0))[0]:
                worst[ident] = (u, s)
    ranked = sorted(worst.items(), key=lambda kv: -kv[1][0])
    out.append(f"members over yield: "
               f"{sum(1 for _k, v in ranked if v[0] > 1.0)} of {len(ranked)}")
    for ident, (u, s) in ranked[:10]:
        out.append(f"   {u * 100:7.1f}%  {s / 1e6:7.1f} MPa  "
                   f"{fft.band_for(u)[1]:15s} {ident}")
    return out


if __name__ == "__main__":
    import turret_production as tp
    import firing_frame_view as ffv
    argv = sys.argv[1:]
    n_modes = int(argv[argv.index("--modes") + 1]) if "--modes" in argv else 40
    doc = tp.balanced_station(bore_mm=20.0).as_document()
    frames = ringdown(doc, modes=n_modes,
                      ballistics="--stand-in" not in argv)
    print("\n".join(report(frames)), flush=True)
    if "--live" in argv:
        import pygame
        trial = ffv.Trial(doc, hidden=False)
        clock, angle, i, running = pygame.time.Clock(), 0.62, 0, True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                              and ev.key == pygame.K_ESCAPE):
                    running = False
            trial.show(frames[i % len(frames)])
            trial.present(angle)
            pygame.display.flip()
            angle += 0.002
            i += 1
            clock.tick(24)
        pygame.quit()
    else:
        print("wrote", ffv.animate(doc, "platform_ringdown.png",
                                   frames=frames))
