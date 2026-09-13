"""THE SCENE, LIVE: the game engine ticking, the frame solved every tick.

    python live_scene.py

    drag            orbit          wheel / Z X    zoom
    SPACE           fire           G              gravity settle again
    R               reset          1 / 2          colour: yield | assembly
    ESC             quit

WHAT THIS IS AND WHAT THE OTHER TWO WERE NOT.

`firing_frame_trial` solves a static instant. `platform_ringdown`
precomputes a decay and plays it back. Both are films. Neither is the
machine running, and neither can be steered.

This runs `machines.MachineSim` -- the game engine's own machine tick,
the same one turret_demo drives -- over the PRODUCTION graph, and
solves the structure every frame against the wall clock. Nothing is
precomputed except the parts that cannot change: the stiffness and the
mode shapes, which are properties of the geometry and are assembled
once because assembling them is a second and a half and the geometry is
not moving between frames. Everything that does change -- the forces,
the modal state, every member's strain, every member's colour -- is
computed this frame, from this frame's dt.

    q.. + 2 zeta w q. + w^2 q = phi^T f(t)

stepped at the real elapsed time, substepped to stay under the shortest
mode's period, because a 60 Hz frame and a 90 Hz mode is an integrator
that explodes rather than a structure that rings.

AND IT IS VECTORISED, because it has to be. A hundred and sixty-four
thousand member-strain evaluations a second is not a Python loop: the
member DOF indices, the local frames and the section properties are
gathered into arrays once, and a frame is four einsums and a scatter.
That is what the constraint tokens were for.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

import firing_frame_trial as fft
from frame_solver import FrameSolver, DOF_PER_NODE, _local_frame
from milspec import MATERIAL_BY_KEY


STRUCTURAL_DAMPING = 0.015
MODES = 40


class LiveStructure:
    """The structure, assembled once, solved every tick."""

    def __init__(self, document: dict, *, modes: int = MODES):
        self.solver = FrameSolver(document=document, loads={})
        modal = self.solver.modes(count=modes)
        n = min(modes, modal["shapes"].shape[1])
        self.shapes = np.ascontiguousarray(modal["shapes"][:, :n])
        self.omega = 2.0 * math.pi * np.asarray(
            modal["all_frequencies_hz"][:n], dtype=float)
        self.mass = np.asarray(modal["lumped_mass"], dtype=float)
        self.rest = self.solver.solve()
        self.u_static = self.rest["displacement"].reshape(-1)
        self.q = np.zeros(n)
        self.qd = np.zeros(n)
        # THE ENGINE'S OWN STABILITY RULE, not a margin of my own.
        # drivetrain_graph.DrivetrainSolver already answers exactly this
        # question for its stiff spring-integrated edges -- dt * omega_n
        # must stay under STABILITY_MARGIN -- and engine_cycle_sim's
        # clutch junction defers to the same number rather than picking
        # one. A structure's retained modes are the same kind of stiff
        # oscillator and there is no reason for them to be governed by a
        # different constant. I had written 0.15, which is five times
        # looser than the engine runs, so the fastest modes here would
        # have been integrated past their own stability limit while the
        # engine beside them was not.
        from drivetrain_graph import DrivetrainSolver
        self.stability_margin = DrivetrainSolver.STABILITY_MARGIN
        top = float(self.omega.max()) if len(self.omega) else 1.0
        self.max_dt = self.stability_margin / max(top, 1e-6)

        # ---- everything a strain needs, gathered once ----------------
        members, dof, rot, length, ro, rel_ax, rel_bd = [], [], [], [], [], [], []
        for e in self.solver.members:
            ia, ib = self.solver.index[e["a"]], self.solver.index[e["b"]]
            a, b = self.solver.position[ia], self.solver.position[ib]
            L = float(np.linalg.norm(b - a))
            if L < 1e-9:
                continue
            members.append(e)
            dof.append(list(range(ia * DOF_PER_NODE, ia * DOF_PER_NODE + 6))
                       + list(range(ib * DOF_PER_NODE, ib * DOF_PER_NODE + 6)))
            rot.append(_local_frame(a, b))
            length.append(L)
            ro.append(float(e.get("radius", 0.012)))
            r = set(e.get("freedoms_released") or ())
            rel_ax.append("x" in r)
            rel_bd.append(bool(r & {"ry", "rz"}))
        self.members = members
        self.dof = np.asarray(dof, dtype=np.int64)
        self.rot = np.asarray(rot, dtype=float)
        self.length = np.asarray(length, dtype=float)
        self.ro = np.asarray(ro, dtype=float)
        self.free_axial = np.asarray(rel_ax, dtype=bool)
        self.free_bend = np.asarray(rel_bd, dtype=bool)
        yld, emod = [], []
        for e in members:
            d = e.get("damage") or {}
            mat = MATERIAL_BY_KEY.get(d.get("material", "4130n"),
                                      MATERIAL_BY_KEY["4130n"])
            yld.append(float(getattr(mat, "yield_pa", 460e6)))
            emod.append(float(d.get("youngs_modulus_pa", mat.youngs_pa)))
        self.yield_pa = np.asarray(yld)
        self.e_pa = np.asarray(emod)
        self.identities = [e["identity"] for e in members]
        self.force = np.zeros(self.shapes.shape[0])
        self._modal_force = np.zeros(len(self.omega))

    # ------------------------------------------------------------------
    def apply(self, node: str, vector) -> None:
        i = self.solver.index[node]
        self.force[:] = 0.0
        self.force[i * DOF_PER_NODE:i * DOF_PER_NODE + 3] = vector
        self._modal_force = self.shapes.T @ self.force

    def clear(self) -> None:
        self.force[:] = 0.0
        self._modal_force[:] = 0.0

    def substep_plan(self, dt: float) -> tuple:
        """How many of its own substeps this takes out of the dt it was
        given -- the same shape as engine_cycle_sim._clutch_substep_plan
        and drivetrain_graph's stable sub-dt.

        THE DT IS NOT MINE TO CHOOSE. One outer step is handed to every
        subsystem in the scene and each subdivides THAT, by whatever its
        own stiffest term requires. Subsystems running on clocks of
        their own would drift apart from one another between exchanges,
        which is a different machine every frame."""
        n = max(1, int(math.ceil(dt / self.max_dt)))
        return n, dt / n

    def step(self, dt: float) -> None:
        n, h = self.substep_plan(dt)
        for _ in range(n):
            a = (self._modal_force - 2.0 * STRUCTURAL_DAMPING * self.omega
                 * self.qd - self.omega ** 2 * self.q)
            self.qd += a * h
            self.q += self.qd * h

    def displacement(self) -> np.ndarray:
        return self.u_static + self.shapes @ self.q

    def utilisation(self) -> np.ndarray:
        """Every member's fraction of its own yield, this instant.

        Four einsums over the whole structure: the per-member loop that
        the static recovery uses is right for one solve and is sixty
        times too slow for sixty frames a second."""
        u = self.displacement()
        U = u[self.dof]                                   # (n, 12)
        da = np.einsum("nij,nj->ni", self.rot, U[:, 0:3])
        ra = np.einsum("nij,nj->ni", self.rot, U[:, 3:6])
        db = np.einsum("nij,nj->ni", self.rot, U[:, 6:9])
        rb = np.einsum("nij,nj->ni", self.rot, U[:, 9:12])
        axial = np.where(self.free_axial, 0.0,
                         (db[:, 0] - da[:, 0]) / self.length)
        curv = np.hypot((rb[:, 1] - ra[:, 1]) / self.length,
                        (rb[:, 2] - ra[:, 2]) / self.length) * self.ro
        bending = np.where(self.free_bend, 0.0, curv)
        stress = (np.abs(axial) + np.abs(bending)) * self.e_pa
        return stress / np.maximum(self.yield_pa, 1.0)

    def positions(self) -> np.ndarray:
        u = self.displacement().reshape(-1, DOF_PER_NODE)
        return self.solver.position + u[:, :3]


# =====================================================================
def main(argv) -> None:
    import pygame
    import turret_production as tp
    import machines
    import firing_frame_view as ffv

    print("building the production graph ...", flush=True)
    graph = tp.balanced_station(bore_mm=20.0)
    doc = graph.as_document()

    # THE GAME ENGINE'S OWN MACHINE TICK, over this graph.
    sim = machines.MachineSim(machine=machines.get("gimbal-cannon-station"),
                              graph=doc)
    print("assembling the structure ...", flush=True)
    live = LiveStructure(doc)
    _n60, _h60 = live.substep_plan(1.0 / 60.0)
    print(f"  {len(live.members)} members, {len(live.omega)} modes, "
          f"{live.omega[0] / (2 * math.pi):.2f} .. "
          f"{live.omega[-1] / (2 * math.pi):.1f} Hz", flush=True)
    print(f"  stability margin {live.stability_margin} (the engine's own) "
          f"-> {_n60} substeps of {_h60 * 1000:.3f} ms inside a 60 Hz frame",
          flush=True)

    trial = ffv.Trial(doc, width=1440, height=920, hidden=False)
    # triangle ranges per member, once, so recolouring is a scatter
    ranges = []
    from articulation import _flat
    for ident in live.identities:
        ranges.append(trial.part_range.get("edge_" + _flat(ident)))
    owner = np.concatenate([np.full(r[1] - r[0], i, np.int32)
                            for i, r in enumerate(ranges) if r]) \
        if any(ranges) else np.zeros(0, np.int32)
    tris = np.concatenate([np.arange(r[0], r[1]) for r in ranges if r]) \
        if any(ranges) else np.zeros(0, np.int64)
    bands = np.asarray([lim for lim, _c, _l in fft.UTILISATION_BANDS])
    band_mat = np.asarray([trial.band_ids[c]
                           for _l, c, _n in fft.UTILISATION_BANDS])
    base_mat = trial.rest_material_ids

    clock = pygame.time.Clock()
    azimuth, elevation, zoom = 0.62, 0.40, 1.0
    dragging = False
    running = True
    t0 = time.perf_counter()
    frames = 0
    recoil_left = 0.0
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    recoil_left = 0.012
                    live.apply("turret.breech", (0.0, 0.0, -110_000.0))
                elif ev.key == pygame.K_g:
                    live.q[:] = 0.0
                    live.qd[:] = 0.0
                    live.q[:] = live.shapes.T @ (live.mass * 0.02)
                elif ev.key == pygame.K_r:
                    live.q[:] = 0.0
                    live.qd[:] = 0.0
                    live.clear()
                elif ev.key == pygame.K_z:
                    zoom = max(0.25, zoom * 0.9)
                elif ev.key == pygame.K_x:
                    zoom = min(4.0, zoom * 1.1)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    dragging = True
                elif ev.button == 4:
                    zoom = max(0.25, zoom * 0.9)
                elif ev.button == 5:
                    zoom = min(4.0, zoom * 1.1)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                azimuth -= ev.rel[0] * 0.008
                elevation = max(-0.9, min(1.4, elevation + ev.rel[1] * 0.005))

        dt = min(clock.tick(60) / 1000.0, 0.05)
        # THE GAME ENGINE TICKS FIRST: supply, actuators, damage, holes.
        sim.step(dt)
        # then the structure, at the same dt, from where it actually is
        if recoil_left > 0.0:
            recoil_left -= dt
            if recoil_left <= 0.0:
                live.clear()
        live.step(dt)

        util = live.utilisation()
        band = np.searchsorted(bands, util)
        mat = base_mat.copy()
        if tris.size:
            mat[tris] = band_mat[np.minimum(band[owner], len(band_mat) - 1)]
        trial.mesh.material_ids = mat
        trial.articulated.displace(live.positions())
        trial.view.restage_static_mesh()

        trial.view._elevation = elevation
        trial.view._zoom = zoom
        trial.present(azimuth)
        pygame.display.flip()
        frames += 1
        if frames % 60 == 0:
            hot = int((util > 0.85).sum())
            print(f"  t={time.perf_counter() - t0:6.1f}s  "
                  f"{clock.get_fps():5.1f} fps   peak "
                  f"{util.max() * 100:6.1f}% of yield   {hot} members hot",
                  flush=True)
    pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
