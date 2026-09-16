"""State -> picture, with no interactive controls in the way.

    python snapshot.py amc-258-jeep-i6 --rpm 5200 --throttle 1 --run 4 \\
        --shoot powertrain.exhaust_manifold --shoot powertrain.oil_pump --after 0.5 \\
        --azimuth 1.1 --out jeep_redline_shot.png

Builds the real sim, drives it to the asked state (throttle, a dyno
rpm hold, an ignition cut, the room mode), fires the asked shots at
named parts with the same ray/ballistics/damage path the app uses
(engine_rays.RayMesh.penetrate -> EngineCycleSim.apply_penetration, so
holes, emitters and damage sounds are all the real ones), runs
`--after` seconds past the LAST event, and renders one frame of the
same EngineGLView the app shows -- thermal state, live combustion,
puncture cutouts, emitter droplets -- to a PNG, plus a plain-text
report of the state at that instant (rpm, what fired, what is leaking
and how fast, what has been lost). Headless: a hidden pygame GL window.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def render_machine(machine_id: str, *, run_s: float = 3.0, command: float = 1.0,
                   load_n: float = 0.0, shots: list[str] = (), bursts: list[str] = (),
                   calibre: str | None = None, shot_from=(0.0, 0.4, 1.5), after_s: float = 0.5,
                   azimuth: float = 1.1, size: int = 1000, detail: float = 0.6,
                   dt: float = 1 / 60, out: str = "machine.png") -> dict:
    """The same picture-of-a-state tool, for a machine that is not an
    engine (machines.py).

    It is deliberately the same code path: the machine builds the same
    graph document, so the same view bakes it, the same RayMesh shoots
    it and the same emitters leak out of it. Nothing here knows what a
    scissor lift is."""
    import pygame
    pygame.display.init()
    pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
    import machines
    from engine_gl_view import EngineGLView
    from engine_mesh import build_engine_mesh
    from engine_rays import RayMesh, Ray

    m = machines.get(machine_id)
    sim = machines.MachineSim(machine=m)
    for p in m.moving_parts:
        sim.command(p.identity, float(command))
        sim.load(p.identity, float(load_n))

    view = EngineGLView(width=size, height=size, covers_off=True,
                        animation_divisions="dense", detail=detail, spring_style="cylinder")
    view.set_graph(sim.graph)
    while view.bake_next():
        pass

    for _ in range(int(run_s / dt)):
        sim.step(dt)

    log = []
    if shots:
        nodes = {n["identity"]: n for n in sim.graph["nodes"]}
        static_m, moving_m = build_engine_mesh(sim.graph, covers_off=True)
        rm = RayMesh(static_m, moving_m)
        for target in shots:
            if target not in nodes:
                log.append(f"shot at {target}: no such part")
                continue
            p = np.array(nodes[target]["reference_position"], dtype=float)
            origin = p + np.asarray(shot_from, dtype=float)
            if calibre:
                from calibres import get_calibre
                cal = get_calibre(calibre)
                proj = cal.projectile(p - origin, float(np.linalg.norm(shot_from)))
                pen = rm.penetrate(Ray.from_points(origin, p), proj.energy_j, cal.diameter_m, projectile=proj)
            else:
                pen = rm.penetrate(Ray.from_points(origin, p), energy_j=3000.0, calibre_m=0.00762)
            # the same recorder the engine sim uses; a machine has no
            # pumped circuits, so the holes it makes are fed by whatever
            # the part itself contains
            from damage_state import record_penetration
            rec = record_penetration(sim.state.part_damage, sim.graph, (), pen)
            sim.hole_emitters.add_from_punctures(rec, sim.graph, ())
            log.append(f"shot at {target}: {len(rec)} punctures")
            log.extend(pen.log[:3])
    for target in bursts:
        sim.burst_part(target)
        log.append(f"burst {target}")
    if shots or bursts:
        for _ in range(int(after_s / dt)):
            sim.step(dt)
        punctures = [q for st in sim.state.part_damage.values() for q in st.punctures]
        view.set_cutout_punctures(punctures)
    view.set_absent_parts(sim.state.absent_parts)
    view.set_emitter_particles([e for e in sim.hole_emitters.emitters if e.regime != "none"]
                               + sim.bursts.clouds())
    view._angle_rad = float(azimuth)
    rgba = view.render(crank_angle_deg=0.0, spin=False, dt=0.0, throttle_frac=0.0)
    surf = pygame.image.frombuffer(np.ascontiguousarray(rgba[:, :, :3]).tobytes(),
                                   (rgba.shape[1], rgba.shape[0]), "RGB")
    pygame.image.save(surf, out)
    return {"machine": machine_id, "out": out, "summary": sim.summary(),
            "shots": log, "absent": sorted(sim.state.absent_parts)}


def render_snapshot(engine_id: str, *, rpm: float | None = None, throttle: float = 1.0, run_s: float = 3.0,
                    shots: list[str] = (), energy_j: float = 3000.0, calibre_m: float = 0.00762,
                    shot_from=(0.0, 0.0, 1.5), after_s: float = 0.5, azimuth: float = 1.1, size: int = 1000,
                    out: str = "snapshot.png", ignition_cut: bool = False, garage: bool = False,
                    detail: float = 0.6, dt: float = 1 / 60, bursts: list[str] = (), calibre: str | None = None,
                    squad: list[str] = (), charges: list[str] = (), fits: list[str] = ()) -> dict:
    import pygame
    pygame.display.init()
    pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
    import engines
    from engine_cycle_sim import EngineCycleSim
    from engine_gl_view import EngineGLView, thermal_groups_from_state
    from drivetrain_graph import build_drivetrain_graph, engine_mesh_view_graph
    from combustion_kernel import visual_for, combustion_state_from_sim
    from engine_mesh import build_engine_mesh
    from engine_rays import RayMesh, Ray
    from toy_shared import leak_lines

    eng = engines.get(engine_id)
    sim = EngineCycleSim(engine=eng)
    sim.start()
    if garage:
        sim.garage_mode = True
    sim.throttle = float(throttle)
    if rpm is not None:
        sim.brake_target_rpm = float(rpm)
    for _ in range(int(run_s / dt)):
        sim.step(dt)
    if ignition_cut:
        sim.state.ignition_cut = True

    view = EngineGLView(width=size, height=size, covers_off=True, animation_divisions="dense", detail=detail,
                        spring_style="cylinder")
    view.set_graph(engine_mesh_view_graph(build_drivetrain_graph(eng)))
    while view.bake_next():
        pass
    view.bake_combustion(sim._firing_angle_deg, visual_for(eng, sim.fuel_choice or eng.preferred_fuel_profile))

    log = []
    # the shrapnel cascade and the firing squad both need the live geometry
    def ray_mesh_factory():
        # the whole engine, covers on -- see RayMesh.from_graph: what is
        # rendered see-through or taken off for the view is not a
        # statement about what a projectile has to get through
        s_m, m_m = build_engine_mesh(view._graph, crank_angle_deg=sim.state.crank_angle_deg, covers_off=False)
        return RayMesh(s_m, m_m)
    sim.ray_mesh_factory = ray_mesh_factory
    if charges:
        from ordnance import Charge
        for spec in charges:
            # "explosive:mass_kg:target[:fuze[:timer_s[:casing_kg]]]"
            f = spec.split(":")
            ch = Charge(explosive=f[0], mass_kg=float(f[1]),
                        attached_to=(f[2] if len(f) > 2 and not f[2].startswith("@") else None),
                        position=(tuple(float(v) for v in f[2][1:].split(",")) if len(f) > 2 and f[2].startswith("@") else None),
                        fuze=(f[3] if len(f) > 3 else "command"), timer_s=float(f[4]) if len(f) > 4 else 5.0,
                        casing_kg=float(f[5]) if len(f) > 5 else 0.0)
            sim.ordnance.place(ch)
            if ch.fuze == "command":
                rep = sim.ordnance.detonate(ch, sim)
                log.append(f"detonated {ch.identity}: {len(rep.get('failed', []))} parts failed, {rep.get('fragments', 0)} casing fragments")
    if squad:
        from calibres import FiringSquad, Shot
        fs = FiringSquad()
        for spec in squad:
            # "calibre:target:rounds:spread_deg[:dwell_s][:dx,dy,dz]"
            f = spec.split(":")
            shot = Shot(f[0], f[1], rounds=int(f[2]) if len(f) > 2 else 1, spread_deg=float(f[3]) if len(f) > 3 else 1.5,
                        dwell_s=float(f[4]) if len(f) > 4 else 0.0,
                        from_direction=tuple(float(v) for v in f[5].split(",")) if len(f) > 5 else (0.0, 0.3, 1.0))
            fs.shots.append(shot)
        for r in fs.execute(sim, ray_mesh_factory, view._graph, dt):
            log.extend(r.log[:1])
    if shots:
        nodes = {n["identity"]: n for n in view._graph["nodes"]}
        # an edge (a pipe, a line) is a target too: its midpoint
        for e in view._graph["edges"]:
            a, b = nodes.get(e["a"]), nodes.get(e["b"])
            if a is not None and b is not None and e["identity"] not in nodes:
                nodes[e["identity"]] = {"reference_position": [(x + y) / 2.0 for x, y in zip(a["reference_position"], b["reference_position"])]}
        static_m, moving_m = build_engine_mesh(view._graph, crank_angle_deg=sim.state.crank_angle_deg, covers_off=True)
        rm = RayMesh(static_m, moving_m)
        for target in shots:
            if target not in nodes:
                log.append(f"shot at {target}: no such part")
                continue
            p = np.array(nodes[target]["reference_position"], dtype=float)
            if calibre:
                from calibres import get_calibre
                cal = get_calibre(calibre)
                proj = cal.projectile(p - (p + np.asarray(shot_from, dtype=float)), float(np.linalg.norm(shot_from)))
                pen = rm.penetrate(Ray.from_points(p + np.asarray(shot_from, dtype=float), p), proj.energy_j, cal.diameter_m,
                                   projectile=proj, fluid_by_part=sim.ballistic_fluid_for_part)
            else:
                pen = rm.penetrate(Ray.from_points(p + np.asarray(shot_from, dtype=float), p), energy_j=energy_j,
                                   calibre_m=calibre_m, fluid_by_part=sim.ballistic_fluid_for_part)
            rec = sim.apply_penetration(pen)
            log.append(f"shot at {target}: " + ("; ".join(f"{i.split('.')[-1]} {q.damage_mode}" for i, q in rec) or "missed"))
    for target in bursts:
        b = sim.burst_part(target)
        log.append(f"burst {target}: " + (f"{b.energy_j / 1000:.1f} kJ, {sum(len(c.pos) for c in b.clouds)} fragments, fastest {b.max_speed_m_s:.0f} m/s"
                                          if b is not None else "no such part"))
    for spec in fits:
        # "emitter_substring:coupling:appliance[:supply_bar[:fluid]]"
        f = spec.split(":")
        match = [e for e in sim.hole_emitters.emitters if f[0] in e.identity]
        if not match:
            log.append(f"fit {spec}: no hole matching {f[0]!r}")
            continue
        port = sim.fittings.fit(match[0], coupling=(f[1] if len(f) > 1 else "garden-hose"),
                                appliance=(f[2] if len(f) > 2 else "sprinkler"),
                                supply_pressure_pa=(float(f[3]) * 1e5 if len(f) > 3 else 0.0),
                                supply_fluid=(f[4] if len(f) > 4 else None))
        log.append(f"fitted {port.describe()}")
    if shots or bursts or squad or charges or fits:
        for _ in range(int(after_s / dt)):
            sim.step(dt)
        punctures = [q for st in sim.state.part_damage.values() for q in st.punctures]
        view.set_cutout_punctures(punctures)
    view.set_absent_parts(sim.state.absent_parts)

    view.set_thermal_state(thermal_groups_from_state(sim.state))
    view.set_combustion_state(combustion_state_from_sim(sim, view.combustion_burn_duration_deg))
    view.set_emitter_particles([e for e in sim.hole_emitters.emitters if e.regime != "none"] + sim.bursts.clouds())
    view._angle_rad = float(azimuth)
    rgba = view.render(crank_angle_deg=sim.state.crank_angle_deg, spin=False, dt=0.0, throttle_frac=sim.throttle)
    surf = pygame.image.frombuffer(np.ascontiguousarray(rgba[:, :, :3]).tobytes(), (rgba.shape[1], rgba.shape[0]), "RGB")
    pygame.image.save(surf, out)

    st = sim.state
    report = {
        "engine": engine_id, "rpm": round(sim.rpm), "throttle": sim.throttle, "crank_deg": round(st.crank_angle_deg, 1),
        "exhaust_k": round(st.exhaust_temp_k), "coolant_k": round(st.coolant_temp_k), "knock": st.knock_flag,
        "slot_strengths": [round(r[0], 2) for r in st.slot_records],
        "shots": log, "leaks": leak_lines(sim), "fouling": sim.hole_emitters.fouling(), "out": out,
        "bursts": sim.bursts.summary(), "absent": sorted(sim.state.absent_parts),
        "fires": sim.fires.summary(), "ports": sim.fittings.summary(),
        "cascade": list(sim.cascade_log)[:14],
        "ordnance": sim.ordnance.log,
        "engine_dead": st.engine_dead,
        "nodes": (lambda: (__import__("node_effects").condition_lines(sim._node_conditions, sim._effects, limit=12)))(),
        "emitters": [(e.identity.split(".")[-2] if "." in e.identity else e.identity, e.fluid, e.regime,
                      round(e.mass_flow_kg_s * 1000, 1), round(e.jet_speed_m_s, 1), e.character)
                     for e in sim.hole_emitters.emitters if e.regime != "none" and e.kind != "splash"],
        "splash": [(e.cylinder, round(e.mass_flow_kg_s * 1000, 1), round(e.jet_speed_m_s, 1), e.character)
                   for e in sim.hole_emitters.emitters if e.kind == "splash" and e.regime != "none"],
    }
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--rpm", type=float, default=None, help="dyno hold rpm (None = free)")
    ap.add_argument("--throttle", type=float, default=1.0)
    ap.add_argument("--run", type=float, default=3.0, help="seconds to run before the shots")
    ap.add_argument("--shoot", action="append", default=[], help="graph node identity to shoot at (repeatable)")
    ap.add_argument("--energy", type=float, default=3000.0)
    ap.add_argument("--calibre-m", type=float, default=0.00762, help="bare hole diameter for --shoot when no --calibre cartridge")
    ap.add_argument("--after", type=float, default=0.5, help="seconds after the last event")
    ap.add_argument("--azimuth", type=float, default=1.1)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--burst", action="append", default=[], help="graph node identity to burst (repeatable)")
    ap.add_argument("--calibre", default=None, help="a real cartridge for --shoot (calibres.py), e.g. '.50 bmg'")
    ap.add_argument("--squad", action="append", default=[],
                    help="firing-squad itinerary entry 'calibre:target:rounds:spread_deg[:dwell_s][:dx,dy,dz]' (repeatable, in order)")
    ap.add_argument("--calibres", action="store_true", help="list the cartridge catalogue and exit")
    ap.add_argument("--charge", action="append", default=[],
                    help="place a charge: 'explosive:mass_kg:target[:fuze[:timer_s[:casing_kg]]]' "
                         "(target = a node/edge identity, or @x,y,z; fuze = command|timer|delay|contact)")
    ap.add_argument("--explosives", action="store_true", help="list the explosive catalogue and exit")
    ap.add_argument("--fit", action="append", default=[],
                    help="fit a hole: 'emitter_substring:coupling:appliance[:supply_bar[:fluid]]' "
                         "(appliance = burner|fridge|sprinkler|monitor|air-tool|exchanger)")
    ap.add_argument("--cut", action="store_true", help="ignition cut at the snapshot")
    ap.add_argument("--garage", action="store_true")
    ap.add_argument("--out", default="snapshot.png")
    a = ap.parse_args(argv)
    if a.calibres:
        from calibres import describe_catalogue
        print("\n".join(describe_catalogue())); return
    if a.explosives:
        from ordnance import describe_catalogue as _ec
        print("\n".join(_ec())); return
    rep = render_snapshot(a.engine, rpm=a.rpm, throttle=a.throttle, run_s=a.run, shots=a.shoot, energy_j=a.energy,
                          calibre_m=a.calibre_m, after_s=a.after, azimuth=a.azimuth, size=a.size, out=a.out,
                          ignition_cut=a.cut, garage=a.garage, bursts=a.burst, calibre=a.calibre, squad=a.squad,
                          charges=a.charge, fits=a.fit)
    for k, v in rep.items():
        if isinstance(v, list) and v and isinstance(v[0], str):
            print(f"{k}:"); [print("   " + line) for line in v]
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
