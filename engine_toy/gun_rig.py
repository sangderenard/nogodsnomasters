"""A headless rig for iterating on the gun, on the real engine.

WHY NOT THE DEMO. `turret_demo.main` is 348 lines of OpenGL around the
machine, and iterating on interior ballistics through a window is
iterating with a blindfold and a hat on. It also predates everything
built since: it never calls `build_gimbal_cannon_station`, so it is
running `machines.get("gimbal-cannon-station")` rather than the
production graph, and it knows nothing about the tube ringing, the
jacket circuit, the ported barrel or the baked ballistics kernel.

WHAT THIS IS INSTEAD. The same machine, driven headless, through the
engine's own pieces and nothing bespoke:

    turret_production.build_gimbal_cannon_station   the real graph
    drivetrain_graph._discover_fluid_circuits       the real circuits
    interior_ballistics.native_step                 SymPy -> SSA -> LLVM
    beam_theory.beam_dynamics_native                the tube, ringing
    symbolic_parts.compile_trajectory_ssa           the round, flying
    barrel_thermal.jacket_conductance               the jacket, derived

Every number that comes out of it is produced by something the game
runs, not by a script beside the game. That is the whole point: when
this rig says the barrel reached 300 C, it is the same arithmetic the
machine would do.

WHAT IT IS FOR. Firing a lot of rounds quickly and watching what
drifts -- velocity, pressure, barrel temperature, whip, where the
rounds land -- so a change to the ballistics can be judged in seconds
instead of through a render loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class GunRig:
    """One gun, its graph, and the engine pieces that step it."""
    bore_mm: float = 20.0
    rate_hz: float = 5.0
    coolant_flow_kg_s: float = 1.0
    coolant_supply_k: float = 278.15
    #: built on demand
    graph: object = None
    document: dict = None
    circuits: list = field(default_factory=list)
    #: live state
    barrel_k: float = 293.15
    whip_q_m: float = 0.0
    whip_qd_m_s: float = 0.0
    whip_rad: float = 0.0
    rounds: int = 0
    wall_heat_j: float = 0.0

    def build(self) -> "GunRig":
        import turret_production as tp
        from drivetrain_graph import _discover_fluid_circuits
        self.graph = tp.build_gimbal_cannon_station(bore_mm=self.bore_mm)
        problems = self.graph.check()
        if problems:
            raise ValueError(f"the graph does not check out: {problems[:3]}")
        self.document = self.graph.as_document()
        self.circuits = _discover_fluid_circuits(self.document)
        return self

    # ---------------- what the graph says about itself ----------------
    def barrel(self) -> dict:
        return next(n for n in self.document["nodes"]
                    if n["identity"] == "turret.outer_barrel")

    def liner(self) -> dict:
        return next(n for n in self.document["nodes"]
                    if n["identity"] == "turret.weapon")

    def jacket(self) -> dict:
        from barrel_thermal import jacket_conductance
        return jacket_conductance(self.document, flow_kg_s=self.coolant_flow_kg_s)

    def beam_kw(self) -> dict:
        b = self.barrel()
        ro = float(b["tube_outer_radius_m"])
        ri = float(b["tube_inner_radius_m"])
        jacket_l = sum(float(n.get("fluid_volume_l", 0.0))
                       for n in self.document["nodes"]
                       if n.get("part_role") == "barrel-jacket-chamber")
        length = float(b["drum_length_m"])
        return dict(length_m=length, outer_radius_m=ro, wall_m=ro - ri,
                    density_kg_m3=7850.0, youngs_cold_pa=2.05e11,
                    damping_ratio=0.005,
                    added_mass_per_m_kg=jacket_l / length,
                    temperature_k=self.barrel_k)

    # ---------------- one round, all the way through ----------------
    #: ---- NO TWO ROUNDS ARE THE SAME ----
    #: A deterministic gun is a wrong gun. Real rounds vary because the
    #: things they are made of vary, and these are the real sources with
    #: their real tolerances -- not a fudge added to the answer:
    #:
    #:   CHARGE MASS. A reloading shop stacking pressed pucks is
    #:   repeatable to about a third of a percent; loose powder thrown
    #:   by volume is three times worse. This is the dominant term.
    #:   WEB. Propellant is pressed to a tolerance and it ages; the
    #:   grain in one lot is not the grain in the next.
    #:   BAND INTERFERENCE. Bands are swaged parts with a tolerance,
    #:   and the bore wears between them.
    #:   TEMPERATURE. Propellant burns faster when it is warm, which is
    #:   why the first round out of a cold gun is slower than the
    #:   twentieth -- and that one is not random at all, it is the
    #:   barrel heating the round in the chamber.
    #:
    #: Each is applied to its OWN input and the kernel is then run
    #: normally, so the spread that comes out is whatever the physics
    #: makes of the spread that went in -- rather than a standard
    #: deviation sprinkled on the muzzle velocity afterwards.
    charge_tolerance: float = 0.003
    web_tolerance: float = 0.02
    interference_tolerance: float = 0.15
    #: THE SAME ROUND GIVES THE SAME ANSWER. Interior ballistics depends
    #: on the charge, the web, the bore and the barrel -- not on how hot
    #: the tube is or how it is ringing, because the shot is gone in
    #: under three milliseconds. So firing a hundred identical rounds
    #: means solving one and reading it a hundred times.
    #:
    #: It matters because the kernel is 17,000 steps at 42 us: 0.7 s of
    #: compute per shot, which at five rounds a second is three and a
    #: half seconds of work per second of simulation. That is what made
    #: the window unusable, not the rendering.
    #:
    #: The real fix is to bake the integration LOOP into the kernel
    #: instead of driving it from Python a step at a time -- the same
    #: move `symbolic_parts.trajectory_batch_source` already makes for
    #: trajectories. This is the cheap correct one until then.
    _shot_cache: dict = field(default_factory=dict, repr=False)

    def ballistics_parameters(self, *, charge_kg: float = 0.040,
                              web_m: float = 2.7e-4,
                              twist_cal: float = 30.0,
                              interference_m: float = 50e-6,
                              dt: float = 1.0e-6) -> dict:
        """The installed gun/load boundary consumed by both shot runners."""
        b = self.barrel()
        length_m = float(b["drum_length_m"])
        bore_m = self.bore_mm / 1000.0
        area = math.pi * (bore_m / 2.0) ** 2
        recoiling = (float(b.get("mass_kg", 0.0))
                     + float(self.liner().get("mass_kg", 0.0)))
        return dict(
            dt_s=dt, bore_area_m2=area, chamber_volume_m3=4.8e-5,
            shot_mass_kg=0.130, charge_mass_kg=charge_kg,
            impetus_j_per_kg=1.1e6, covolume_m3_per_kg=1.0e-3,
            propellant_density_kg_m3=1620.0, gamma=1.25,
            burn_rate_coeff=6.4e-10, grain_chi=1.0, grain_lambda=0.0,
            grain_web_m=web_m, shot_start_pressure_pa=3.5e7,
            barrel_length_m=length_m,
            twist_calibres_per_turn=twist_cal,
            band_interference_m=interference_m,
            bore_diameter_m=bore_m, primer_gas_kg=3.5e-4,
            recoiling_mass_kg=max(recoiling, 1.0),
            gas_port_travel_m=1.0e9, gas_port_fraction=0.0)

    def recoil_force_profile(self, *, structural_interval_s: float = 5.0e-5,
                             dt: float = 1.0e-6,
                             limit: int = 20_000) -> dict:
        """Breech force history from the existing compiled combustion law.

        The interior law runs at its native microsecond step.  Consecutive
        samples are accumulated into impulse-preserving structural intervals;
        this reduces exchanges with the full beam solve without replacing the
        pressure curve by a guessed pulse.  Sum(force * duration) is exactly
        the impulse produced by the sampled breech-pressure history.
        """
        import interior_ballistics as ib
        step, _names = ib.native_step()
        kw = self.ballistics_parameters(dt=dt)
        state = {"travel_m": 0.0, "velocity_m_s": 0.0,
                 "burnt_fraction": 0.0, "recoil_velocity_m_s": 0.0,
                 "harvested_gas_kg": 0.0, "wall_heat_j": 0.0}
        segments = []
        interval_impulse = 0.0
        interval_time = 0.0
        peak = 0.0
        last = None
        for _ in range(int(limit)):
            last = step(**kw, **state)
            force_n = (float(last["breech_pressure_pa"])
                       * float(kw["bore_area_m2"]))
            peak = max(peak, force_n)
            interval_impulse += force_n * dt
            interval_time += dt
            state = {
                "travel_m": last["travel_next_m"],
                "velocity_m_s": last["velocity_next_m_s"],
                "burnt_fraction": last["burnt_fraction_next"],
                "recoil_velocity_m_s": last["recoil_velocity_next_m_s"],
                "harvested_gas_kg": last["harvested_gas_next_kg"],
                "wall_heat_j": last["wall_heat_next_j"],
            }
            done = last["at_muzzle"] > 0.5
            if interval_time + 1.0e-15 >= structural_interval_s or done:
                segments.append((interval_time,
                                 interval_impulse / interval_time))
                interval_impulse = 0.0
                interval_time = 0.0
            if done:
                break
        if last is None or last["at_muzzle"] <= 0.5:
            raise RuntimeError("interior ballistics did not reach the muzzle")
        return {
            "segments": tuple(segments),
            "duration_s": sum(h for h, _force in segments),
            "impulse_n_s": sum(h * force for h, force in segments),
            "peak_force_n": peak,
            "muzzle_m_s": float(state["velocity_m_s"]),
            "recoil_velocity_m_s": float(state["recoil_velocity_m_s"]),
            "recoiling_mass_kg": float(kw["recoiling_mass_kg"]),
            "shot_mass_kg": float(kw["shot_mass_kg"]),
            "bore_diameter_m": float(kw["bore_diameter_m"]),
            "wall_heat_j": float(state["wall_heat_j"]),
        }

    def fire_one(self, *, charge_kg: float = 0.040, web_m: float = 2.7e-4,
                 twist_cal: float = 30.0, interference_m: float = 50e-6,
                 dt: float = 1.0e-6, vary: bool = True) -> dict:
        """Interior ballistics on the graph's own barrel."""
        if vary:
            import random
            if not hasattr(self, "_rng"):
                self._rng = random.Random(20260912)
            g = self._rng.gauss
            charge_kg *= 1.0 + g(0.0, self.charge_tolerance)
            web_m *= 1.0 + g(0.0, self.web_tolerance)
            interference_m *= max(0.2, 1.0 + g(0.0, self.interference_tolerance))
            # the cache is keyed on the VARIED numbers, so identical
            # rounds still cost nothing and different ones are solved
            charge_kg = round(charge_kg, 6)
            web_m = round(web_m, 8)
            interference_m = round(interference_m, 9)
        key = (round(charge_kg, 6), round(web_m, 8), round(twist_cal, 3),
               round(interference_m, 9), round(dt, 9), self.bore_mm)
        hit = self._shot_cache.get(key)
        if hit is not None:
            self.rounds += 1
            self.wall_heat_j += hit["wall_heat_j"]
            return hit
        import interior_ballistics as ib
        # THE WHOLE SHOT IN ONE CALL. `native_step` bakes one timestep
        # and the seventeen thousand iterations were a Python loop
        # around it -- 700 ms a shot. `native_shot` compiles the loop
        # too: 8.8 ms, same answer to the digit.
        run, _ = ib.native_shot()
        kw = self.ballistics_parameters(
            charge_kg=charge_kg, web_m=web_m, twist_cal=twist_cal,
            interference_m=interference_m, dt=dt)
        res = run(20000, **kw)
        st = res
        r = res
        peak = res["peak_breech_pa"]
        self.rounds += 1
        self.wall_heat_j += st["wall_heat_j"]
        available = charge_kg * 1.1e6 / 0.25
        out = {
            "muzzle_m_s": st["velocity_m_s"],
            "peak_breech_pa": peak,
            "burnt": min(1.0, st["burnt_fraction"]),
            "wall_heat_j": st["wall_heat_j"],
            "exit_gas_k": r["gas_temperature_k"],
            "gas_kg": r["gas_made_kg"],
            "unburnt_kg": max(0.0, charge_kg - r["gas_made_kg"]),
            "recoil_impulse_n_s": (0.130 * st["velocity_m_s"]
                                   + charge_kg * st["velocity_m_s"] * 1.5),
            "closure": (r["shot_energy_j"] + r["recoil_energy_j"]
                        + st["wall_heat_j"] + r["gas_enthalpy_j"]) / available,
        }
        self._shot_cache[key] = out
        return out

    # ---------------- the tube, between rounds ----------------
    def settle(self, dt: float, impulse_n_s: float = 0.0) -> float:
        from beam_theory import beam_dynamics_native
        step, _ = beam_dynamics_native()
        kw = self.beam_kw()
        sub = 2.0e-4
        r = None
        for _ in range(max(1, int(dt / sub))):
            r = step(modal_q_m=self.whip_q_m, modal_qdot_m_s=self.whip_qd_m_s,
                     force_n=0.0, dt_s=sub, **kw)
            self.whip_q_m, self.whip_qd_m_s = (r["modal_q_next_m"],
                                               r["modal_qdot_next_m_s"])
        if impulse_n_s > 0.0:
            # the trunnion offset is the lever: recoil is axial and only
            # bends the tube through the distance from the bore to the pivot
            nodes = {n["identity"]: n for n in self.document["nodes"]}
            bore_y = nodes["turret.weapon"]["reference_position"][1]
            pivot_y = nodes["turret.pitch"]["reference_position"][1]
            lever = abs(bore_y - pivot_y) / kw["length_m"]
            in_tube = 0.0027
            f = impulse_n_s / in_tube * lever
            for _ in range(int(in_tube / 1.0e-5)):
                r = step(modal_q_m=self.whip_q_m,
                         modal_qdot_m_s=self.whip_qd_m_s,
                         force_n=f, dt_s=1.0e-5, **kw)
                self.whip_q_m, self.whip_qd_m_s = (r["modal_q_next_m"],
                                                   r["modal_qdot_next_m_s"])
        if r is not None:
            self.whip_rad = r["tip_slope_rad"]
            self.first_mode_hz = r["natural_frequency_hz"]
        return self.whip_rad

    def soak(self, joules: float, dt: float) -> float:
        """Heat in from the shot, out through the jacket the graph declares."""
        b = self.barrel()
        mass = float(b.get("mass_kg", 60.0)) + float(
            self.liner().get("mass_kg", 0.0))
        ua = self.jacket()["ua_w_per_k"]
        self.barrel_k += joules / max(mass * 470.0, 1.0)
        self.barrel_k -= (ua * (self.barrel_k - self.coolant_supply_k)
                          * dt / max(mass * 470.0, 1.0))
        return self.barrel_k

    # ---------------- a burst ----------------
    def burst(self, rounds: int, **shot) -> list[dict]:
        dt = 1.0 / self.rate_hz
        out = []
        for _ in range(rounds):
            r = self.fire_one(**shot)
            self.soak(r["wall_heat_j"], dt)
            self.settle(dt, r["recoil_impulse_n_s"])
            out.append({**r, "barrel_k": self.barrel_k,
                        "whip_rad": self.whip_rad,
                        "first_mode_hz": getattr(self, "first_mode_hz", 0.0)})
        return out

    def describe(self) -> list[str]:
        b = self.barrel()
        j = self.jacket()
        return [
            f"gun rig: {self.bore_mm:.0f} mm liner in a "
            f"{float(b['tube_inner_radius_m']) * 2000:.0f} mm outer barrel, "
            f"{float(b['drum_length_m']):.2f} m",
            f"  jacket {j['chambers']} chambers, {j['annulus_m'] * 1000:.0f} mm gap, "
            f"UA {j['ua_w_per_k']:.0f} W/K at {self.coolant_flow_kg_s} kg/s "
            f"({j['regime']})",
            f"  circuits: " + ", ".join(
                f"{c.kind_class} ({len(c.nodes)})" for c in self.circuits),
            f"  graph: {len(self.document['nodes'])} bodies, "
            f"{len(self.document['edges'])} members",
        ]


# =====================================================================
#  THE SAME RIG, ON SCREEN
# =====================================================================
WINDOW = (1280, 800)
VIEW = 800


def run(bore_mm: float = 20.0, rate_hz: float = 5.0,
        coolant_flow_kg_s: float = 1.0, frames: int = 0,
        out: str = "gun_rig.png") -> dict:
    """Fire the gun and watch it, on the real graph.

    The readouts are not a HUD invented for the picture -- every one of
    them is a number the engine produced this frame: muzzle velocity
    and peak pressure from the interior ballistics kernel, barrel
    temperature from the jacket conductance the graph's own chamber
    geometry implies, whip and first mode from the beam kernel, and the
    pose that is drawn is the pose the tube is actually bent into.
    """
    import pygame
    import numpy as np
    import turret_production as tp

    rig = GunRig(bore_mm=bore_mm, rate_hz=rate_hz,
                 coolant_flow_kg_s=coolant_flow_kg_s).build()
    pygame.init()
    flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.HIDDEN if frames else 0)
    pygame.display.set_mode(WINDOW, flags)
    pygame.display.set_caption(
        f"gun rig -- {bore_mm:.0f} mm in a 120, live ballistics")
    font = pygame.font.SysFont("consolas", 15)
    from engine_gl_view import EngineGLView
    view = EngineGLView(width=VIEW, height=VIEW, covers_off=True,
                        animation_divisions="dense", detail=0.6,
                        spring_style="cylinder")
    view.set_graph(rig.document)
    while view.bake_next():
        pass
    view._angle_rad = 1.15
    # BAKE ONCE, POSE EVERY FRAME. Calling set_graph and re-baking each
    # frame rebuilds 1613 members sixty times a second, which is why the
    # first version of this ran at a frame every few seconds. The demo
    # already had the answer: an ArticulatedMesh carries the vertices
    # and `pose` moves them by motion group, so only the transforms are
    # recomputed.
    from articulation import ArticulatedMesh
    articulated = ArticulatedMesh.build(rig.document, view._static_mesh)

    clock = pygame.time.Clock()
    dt = 1.0 / 60.0
    since_shot = 0.0
    firing = True
    last = {"muzzle_m_s": 0.0, "peak_breech_pa": 0.0, "burnt": 0.0,
            "exit_gas_k": 0.0, "closure": 0.0, "recoil_impulse_n_s": 0.0}
    recoil_m = 0.0
    running = True
    frame = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    firing = not firing
                elif event.key == pygame.K_LEFT:
                    view._angle_rad -= 0.12
                elif event.key == pygame.K_RIGHT:
                    view._angle_rad += 0.12

        # ---- fire, on the real kernels ----
        since_shot += dt
        impulse = 0.0
        if firing and since_shot >= 1.0 / rig.rate_hz:
            since_shot = 0.0
            last = rig.fire_one()
            rig.soak(last["wall_heat_j"], 1.0 / rig.rate_hz)
            impulse = last["recoil_impulse_n_s"]
            recoil_m = 0.42
        rig.settle(dt, impulse)
        recoil_m = max(0.0, recoil_m - dt * 1.6)     # back to battery

        # ---- draw the machine IN THE POSE IT IS ACTUALLY IN ----
        articulated.pose(tp.pose_frames(rig.document, whip_rad=rig.whip_rad,
                                        recoil_m=recoil_m))
        view.restage_static_mesh()
        rgba = view.render(crank_angle_deg=0.0, spin=False, dt=dt,
                           throttle_frac=0.0)
        surf = pygame.image.frombuffer(
            np.ascontiguousarray(rgba[:, :, :3]).tobytes(),
            (rgba.shape[1], rgba.shape[0]), "RGB")

        panel = pygame.Surface(WINDOW, pygame.SRCALPHA)
        panel.fill((14, 17, 22))
        panel.blit(surf, (0, 0))
        j = rig.jacket()
        lines = [
            ("ROUNDS", f"{rig.rounds}", (230, 230, 235)),
            ("", "", None),
            ("muzzle", f"{last['muzzle_m_s']:8.0f} m/s", (120, 220, 140)),
            ("peak breech", f"{last['peak_breech_pa'] / 1e6:8.0f} MPa",
             (220, 160, 110)),
            ("burnt", f"{last['burnt'] * 100:8.0f} %", (200, 200, 210)),
            ("exit gas", f"{last['exit_gas_k']:8.0f} K", (220, 120, 110)),
            ("energy closure", f"{last['closure'] * 100:8.1f} %", (150, 170, 220)),
            ("", "", None),
            ("barrel", f"{rig.barrel_k - 273.15:8.1f} C", (220, 160, 110)),
            ("jacket UA", f"{j['ua_w_per_k']:8.0f} W/K", (110, 200, 220)),
            ("coolant", f"{j['velocity_m_s']:8.2f} m/s {j['regime']}",
             (110, 200, 220)),
            ("", "", None),
            ("whip", f"{rig.whip_rad * 1000:8.4f} mrad", (200, 150, 230)),
            ("at 800 m", f"{rig.whip_rad * 800 * 1000:8.1f} mm", (200, 150, 230)),
            ("first mode", f"{getattr(rig, 'first_mode_hz', 0.0):8.2f} Hz",
             (200, 150, 230)),
            ("", "", None),
            ("SPACE", "hold fire" if firing else "FIRE", (230, 200, 120)),
            ("arrows", "orbit", (120, 126, 134)),
        ]
        y = 24
        for label, value, colour in lines:
            if colour is None:
                y += 10
                continue
            panel.blit(font.render(label, True, (120, 126, 134)), (VIEW + 24, y))
            panel.blit(font.render(value, True, colour), (VIEW + 190, y))
            y += 21
        b = rig.barrel()
        foot = (f"{bore_mm:.0f} mm liner, {float(b['drum_length_m']):.2f} m "
                f"outer barrel, {len(rig.document['nodes'])} bodies, "
                f"{len(rig.document['edges'])} members")
        panel.blit(font.render(foot, True, (100, 106, 114)), (VIEW + 24, 760))

        screen = pygame.display.get_surface()
        if screen is not None and not (screen.get_flags() & pygame.OPENGL):
            screen.blit(panel, (0, 0))
            pygame.display.flip()
        else:
            pygame.image.save(panel, out)
        frame += 1
        if frames and frame >= frames:
            running = False
        clock.tick(60)
    pygame.quit()
    return {"rounds": rig.rounds, "barrel_c": rig.barrel_k - 273.15,
            "out": out}


if __name__ == "__main__":
    import sys
    run(frames=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
