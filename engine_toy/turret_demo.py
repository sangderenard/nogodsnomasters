"""Watch it work: an engine running, a turret it powers, and hostiles.

    python turret_demo.py

Nothing to drive and nothing to configure. The point is to look at it
and judge whether the behaviour reads as effective and organic, which
is not a thing you can tell from a table of numbers.

WHAT IS ACTUALLY COUPLED, because that is what makes it look alive
rather than scripted:

  The ENGINE runs its real cycle and drives its real auxiliary plant.
  The plant's hydraulic pump makes flow in proportion to crank speed.

  The TURRET slews on that flow. At idle the pump makes little and the
  mount is sluggish; open the throttle and it whips round. That is one
  number flowing from the engine through the plant to the mount, not a
  slew rate someone typed, and it is the thing most worth watching.

  The FIRE CONTROL acquires the nearest hostile it can actually see,
  iterates a real firing solution against the baked ballistics, and
  commands the mount toward it. The two reticles show the two separate
  time costs -- the computer converging, and the mount catching up.

WHAT YOU ARE LOOKING AT

  Left, the engine, animating on its own baked crank frames.

  Right, a plan view from above: the turret at the centre, its barrel
  as a solid line, the current firing solution as a dashed amber line,
  hostiles as dots with their recent tracks behind them, and range
  rings every two hundred metres. A hostile with no line of sight is
  hollow rather than filled.

  Keys: SPACE throttle between idle and open, H add a hostile,
  C clear them, ESC quit.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pygame
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE

sys.path.insert(0, str(Path(__file__).resolve().parent))

WINDOW = (1560, 820)
VIEW = 690
PLAN_X = VIEW + 24
PLAN_SIZE = 700
BG = (14, 15, 18)
INK = (214, 220, 228)
DIM = (120, 128, 138)
AMBER = (232, 176, 72)
WHITE = (238, 242, 248)
RED = (226, 96, 84)
GREEN = (120, 210, 150)
PLAN_RANGE_M = 1600.0


def _plan_xy(world, centre, scale):
    """World (x, z) to plan pixels, north up."""
    return (PLAN_SIZE / 2 + (world[0] - centre[0]) * scale,
            PLAN_SIZE / 2 - (world[2] - centre[2]) * scale)


def main(frames: int = 0, out_prefix: str = "turret_demo", warm_s: float = 0.0) -> int:
    """`frames` > 0 runs headless, saving a PNG per captured frame --
    which is how this gets checked without a person at the keyboard."""
    pygame.init()
    flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.HIDDEN if frames else 0)
    pygame.display.set_mode(WINDOW, flags)
    pygame.display.set_caption("turret demo -- engine, mount, hostiles")
    font = pygame.font.SysFont("consolas", 15)

    from gl_text import TextLayer
    from engine_gl_view import EngineGLView
    from engines import CATALOGUE
    from engine_cycle_sim import EngineCycleSim
    from drivetrain_graph import build_drivetrain_graph, engine_mesh_view_graph
    from combustion_kernel import visual_for
    import machines
    from fire_control import Hostile, SolutionTable, TurretFireControl
    from engine_mesh import build_engine_mesh
    from engine_rays import RayMesh

    text = TextLayer(font, *WINDOW)

    # ---- the engine that powers it ----
    engine = next(e for e in CATALOGUE if "c18" in e.identity)
    sim = EngineCycleSim(engine=engine)
    sim.start()
    sim.throttle = 0.15

    view = EngineGLView(width=VIEW, height=VIEW, covers_off=True,
                        animation_divisions="dense", detail=0.55)
    engine_view_graph = engine_mesh_view_graph(build_drivetrain_graph(engine))

    # ---- the turret it drives ----
    # THE CANONICAL TURRET. Not the box-list stand-in: the structure
    # authored in `turret_production` -- captive pinions on the ring,
    # the trunnion at the barrel's own centre of gravity, a six-leg fine
    # platform at the back of the tube, and a machine gun with its own
    # yaw and its own trunnion. `machines.gimbal_cannon_station` wraps
    # that document and adds only the plant that drives it.
    turret = machines.get("gimbal-cannon-station")
    turret_sim = machines.MachineSim(machine=turret)
    turret_mesh = RayMesh(*build_engine_mesh(turret_sim.graph, covers_off=False))
    table = SolutionTable.load_or_bake(turret.sight_cartridge,
                                       zero_range_m=turret.sight_zero_range_m)
    # THE STATION: the turret sits on a pole beside the engine, runs its
    # own electronics off its own battery, and starts the engine when
    # one of its stores asks for it.
    from station import WeaponStation, FriendlyMask, EmergencyBattery
    from world_gun import gun_for_world_object
    POLE_HEIGHT = 2.6
    fc = TurretFireControl(table=table, muzzle_position=np.array([0.0, POLE_HEIGHT, 0.0]),
                           ray_mesh=turret_mesh, crosswind_m_s=3.0)
    # its own structure, for the friendly-fire mask: the turret, its
    # pole and the engine sitting next to it are all things it can
    # traverse across and must not shoot
    own_graph = dict(turret_sim.graph)
    own_nodes = [dict(n) for n in own_graph["nodes"]]
    for n in own_nodes:
        n["reference_position"] = [n["reference_position"][0],
                                   n["reference_position"][1] + POLE_HEIGHT,
                                   n["reference_position"][2]]
    # the pole, and the engine crate four metres away
    own_nodes.append({"identity": "station.pole", "kind": "structure",
                      "reference_position": [0.0, POLE_HEIGHT / 2, 0.0],
                      "body_half_extent_m": [0.11, POLE_HEIGHT / 2, 0.11],
                      "material": "steel-plate", "in_view": True})
    # The engine's own envelope, for the friendly-fire mask only. It is
    # deliberately NOT in the view: the real engine geometry is already
    # in the scene, and drawing a box over it hid the engine entirely.
    own_nodes.append({"identity": "station.engine_crate", "kind": "structure",
                      "reference_position": [4.2, 0.9, 0.0],
                      "body_half_extent_m": [1.5, 0.9, 0.8],
                      "material": "steel-plate", "in_view": True,
                      "mask_only": True})
    own_graph = {**own_graph, "nodes": own_nodes}
    own_mesh = RayMesh(*build_engine_mesh(own_graph, covers_off=False))

    # THE VIEW SHOWS BOTH. The engine graph goes through
    # `engine_mesh_view_graph`, which crops to the engine itself -- that
    # is the mode that was hiding the turret, and it is doing its job:
    # it was written to drop chassis-side plumbing from an ENGINE view
    # and has no idea a turret exists. The answer is not to loosen it
    # but to compose the scene afterwards, so the engine is still
    # cropped the way an engine view should be and the station's own
    # parts are added alongside.
    def compose_station_scene():
        nodes = list(engine_view_graph["nodes"])
        edges = list(engine_view_graph["edges"])
        # the turret, lifted onto its pole and set beside the engine
        for n in own_graph["nodes"]:
            if n.get("mask_only"):
                continue           # a collision volume, not a body to draw
            moved = dict(n)
            pos = list(moved["reference_position"])
            moved["reference_position"] = [pos[0] - 4.2, pos[1], pos[2]]
            moved["in_view"] = True
            nodes.append(moved)
        placed = {n["identity"] for n in nodes}
        for e in own_graph["edges"]:
            if e["a"] in placed and e["b"] in placed:
                edges.append({**e, "in_view": True})
        return {**engine_view_graph, "nodes": nodes, "edges": edges}

    view.set_graph(compose_station_scene())
    # ---- THE MOUNT ARTICULATES ----
    # Mesh once, aim every frame. The composed scene's static mesh holds
    # the engine AND the turret; `ArticulatedMesh` attributes each of its
    # vertices to the mechanism that carries it (off the motion group
    # every body already declares) and leaves the engine's own geometry
    # where it is. Posing then costs a few milliseconds instead of the
    # tenth of a second a re-mesh costs, which is the difference between
    # a turret that turns and a turret that is a photograph.
    import turret_production as _tp
    from articulation import ArticulatedMesh
    STATION_OFFSET = np.array([-4.2, POLE_HEIGHT, 0.0])
    _placed = {**turret_sim.graph,
               "nodes": [{**n, "reference_position":
                          [n["reference_position"][0] + STATION_OFFSET[0],
                           n["reference_position"][1] + STATION_OFFSET[1],
                           n["reference_position"][2] + STATION_OFFSET[2]]}
                         for n in turret_sim.graph["nodes"]]}
    articulated = ArticulatedMesh.build(_placed, view._static_mesh)
    # ---- WHERE A ROUND ACTUALLY LEAVES FROM ----
    # The posed barrel's own node plus its rotated bore axis times half
    # its length. Not an offset from the mount and not a heading
    # recomputed from the fire-control angles: the geometry itself, so a
    # round leaves the end of the tube that is drawn on screen.
    from projectiles import muzzle_of
    _pose_state = {"doc": _placed}
    kills = 0
    impacts: list = []
    def aim_frames(*, hoist_m, traverse_deg, elevation_deg, recoil_m=0.0,
                   fine_pitch_deg=0.0, mg_yaw_deg=0.0, mg_pitch_deg=0.0):
        """Where every mechanism is. The document handed to
        `pose_frames` is the PLACED one -- the turret already stands on
        its pole beside the engine -- and the chain reads its own pivots
        out of it, so the answer comes back in world coordinates with no
        second transform to get wrong."""
        frames = _tp.pose_frames(_placed, hoist_m=hoist_m, traverse_deg=traverse_deg,
                                 elevation_deg=elevation_deg, recoil_m=recoil_m,
                                 fine_pitch_deg=fine_pitch_deg,
                                 mg_yaw_deg=mg_yaw_deg, mg_pitch_deg=mg_pitch_deg)
        return frames
    view.bake_combustion(sim._firing_angle_deg,
                         visual_for(engine, sim.fuel_choice or engine.preferred_fuel_profile))
    # THE SOLUTION RUNS ON ITS OWN THREAD. The mount acts on whatever
    # plan is published and never pauses tracking to think; conditions
    # are fed in every tick so a manoeuvring target changes the plan in
    # real time rather than at the next convenient moment.
    from solution_service import SolutionService
    solver_service = SolutionService(table=table, cycle_s=0.05)
    fc.attach_service(solver_service)
    station = WeaponStation(identity="post-01", engine_sim=sim, fire_control=fc,
                            battery=EmergencyBattery(state_of_charge=0.52),
                            friendly=FriendlyMask(own_mesh=own_mesh),
                            pole_height_m=POLE_HEIGHT)
    # the station reads its own muzzle off the posed geometry, so the
    # round leaves the tube that is drawn rather than an assumed point
    station.muzzle_pose = lambda: muzzle_of(_pose_state["doc"])
    # the gun itself, loaded with anti-material ordnance
    gun = gun_for_world_object(own_graph, ".50 ap")
    reload_s = 0.0

    rng = random.Random(7)

    def spawn(n=1):
        for _ in range(n):
            bearing = rng.uniform(0, math.tau)
            distance = rng.uniform(450.0, 1400.0)
            position = np.array([math.sin(bearing) * distance, 1.8,
                                 math.cos(bearing) * distance])
            # a heading that mostly crosses rather than closes, so the
            # tracking problem is a real one
            course = bearing + rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 1.4)
            speed = rng.uniform(6.0, 20.0)
            h = Hostile(f"contact-{len(hostiles) + 1:02d}", position,
                        np.array([math.sin(course) * speed, 0.0,
                                  math.cos(course) * speed]))
            # give it a body, so a round has something to defeat
            h.give_body(turret.sight_cartridge)
            hostiles.append(h)

    hostiles: list = []
    tracks: dict = {}
    spawn(4)

    clock = pygame.time.Clock()
    running = True
    captured = 0
    elapsed = 0.0
    if frames:
        sim.throttle = 0.85            # open it up so the mount is lively
    while running:
        dt = (1 / 60.0) if frames else min(clock.tick(60) / 1000.0, 0.05)
        elapsed += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    sim.throttle = 0.85 if sim.throttle < 0.5 else 0.15
                elif event.key == pygame.K_h:
                    spawn(1)
                elif event.key == pygame.K_c:
                    hostiles.clear(); tracks.clear(); fc.target = None; fc.solver = None

        # ---- hostiles move, and are tracked ----
        # A DESTROYED CONTACT STOPS. It is left on the plot as a wreck
        # rather than deleted, because knowing what you killed and where
        # is the whole reason to run this.
        for h in hostiles:
            if h.destroyed:
                continue
            h.position = h.position + h.velocity * dt
            # turn them gently so the plot does not look like a ruler
            yaw = 0.25 * dt
            vx, vz = float(h.velocity[0]), float(h.velocity[2])
            h.velocity[0] = vx * math.cos(yaw) - vz * math.sin(yaw)
            h.velocity[2] = vx * math.sin(yaw) + vz * math.cos(yaw)
            trail = tracks.setdefault(h.identity, [])
            trail.append((float(h.position[0]), float(h.position[2])))
            del trail[:-90]
        # the station does everything: power, auto-start, acquisition,
        # the mount, and whether firing is allowed
        state = station.step(dt, hostiles)
        reload_s = max(0.0, reload_s - dt)
        if state.get("may_fire") and reload_s <= 0.0 and fc.target is not None:
            station.fire(fc.target, gun)
            reload_s = 60.0 / 330.0   # the 40 mm L/70's real cyclic rate
        # ROUNDS IN THE AIR ARE STEPPED EVERY TICK, and this line is the
        # whole difference between a gun and a noise: without it the
        # rounds were emitted into a field nothing flew, accumulated
        # forever, and never resolved against anything.
        for _ev in station.step_projectiles(dt, hostiles):
            if (_ev.get("effect") or {}).get("destroyed"):
                kills += 1
            impacts.append(_ev)
        del impacts[:-6]
        plant = getattr(sim, "plant", None)
        hydraulics = getattr(plant, "hydraulics", None) if plant else None
        flow = float(getattr(hydraulics, "pump_flow_lpm", 0.0) or 0.0)

        # ---- THE MOUNT GOES WHERE THE FIRE CONTROL PUT IT ----
        # The same angles the solution is driving, on the same tick, so
        # the turret you can see and the turret that takes the shot are
        # one turret. The hoist runs out as soon as the station has a
        # reason to be up; the tube's recoil position comes from the
        # slide's own state, which is real dynamics rather than an
        # animation of a shot.
        hoist_m = 0.850 if state["engine_running"] else 0.0
        recoil_m = float(getattr(getattr(turret, "recoil", None), "compression_m", 0.0) or 0.0)
        # the machine gun takes whatever the cannon is not taking: it
        # leads the same track on its own swivel, a little ahead, so the
        # two mounts are visibly working the same problem separately
        mg_yaw = fc.bearing_deg + 18.0
        mg_pitch = max(-15.0, min(80.0, fc.elevation_deg + 6.0))
        _aim = dict(hoist_m=hoist_m, traverse_deg=fc.bearing_deg,
                    elevation_deg=max(-8.0, min(42.0, fc.elevation_deg)),
                    recoil_m=recoil_m, mg_yaw_deg=mg_yaw - fc.bearing_deg,
                    mg_pitch_deg=mg_pitch)
        articulated.pose(aim_frames(**_aim))
        # the SAME pose as a document, so the muzzle a round leaves from
        # and the muzzle that is drawn cannot disagree
        _pose_state["doc"] = _tp.pose_document(_placed, **_aim)
        view.restage_static_mesh()

        # ---- draw ----
        text.begin_frame()
        tex = view.render_gpu(crank_angle_deg=sim.state.crank_angle_deg, spin=False,
                              dt=dt, throttle_frac=sim.throttle)
        text.compositor.draw_texture(tex, 0, 0, VIEW, VIEW, *WINDOW, flip_v=True)

        plan = pygame.Surface((PLAN_SIZE, PLAN_SIZE), pygame.SRCALPHA)
        plan.fill((20, 22, 27, 255))
        scale = (PLAN_SIZE / 2 - 30) / PLAN_RANGE_M
        centre = (0.0, 0.0, 0.0)
        for ring in range(200, int(PLAN_RANGE_M) + 1, 200):
            pygame.draw.circle(plan, (40, 46, 54), (PLAN_SIZE // 2, PLAN_SIZE // 2),
                               int(ring * scale), 1)
        # the barrel, and the solution
        for name, angle, colour, width in (
                ("aim", state["aim"]["bearing_deg"], WHITE, 3),
                ("solution", (state["solution"] or {}).get("bearing_deg"), AMBER, 2)):
            if angle is None:
                continue
            a = math.radians(angle)
            end = (PLAN_SIZE / 2 + math.sin(a) * PLAN_SIZE * 0.46,
                   PLAN_SIZE / 2 - math.cos(a) * PLAN_SIZE * 0.46)
            pygame.draw.line(plan, colour, (PLAN_SIZE / 2, PLAN_SIZE / 2), end, width)
        for h in hostiles:
            trail = tracks.get(h.identity, [])
            if len(trail) > 2:
                pts = [_plan_xy((x, 0.0, z), centre, scale) for x, z in trail]
                pygame.draw.lines(plan, (58, 66, 76), False, pts, 1)
            p = _plan_xy(h.position, centre, scale)
            visible = True
            if fc.ray_mesh is not None:
                from fire_control import line_of_sight
                visible, _ = line_of_sight(fc.ray_mesh, fc.muzzle_position, h.position)
            colour = RED if h is fc.target else (INK if visible else DIM)
            pygame.draw.circle(plan, colour, (int(p[0]), int(p[1])), 5,
                               0 if visible else 1)
        pygame.draw.circle(plan, GREEN, (PLAN_SIZE // 2, PLAN_SIZE // 2), 7, 2)
        text.draw_surface(plan, PLAN_X, 16, cache_key=("plan", pygame.time.get_ticks()))

        # the readouts sit under the ENGINE, not under the plan: four
        # rows below a 760 px plan ran off the bottom of an 820 px window
        y = VIEW + 14
        rows = [
            f"engine {'RUNNING' if state['engine_running'] else 'stopped ':8s} "
            f"{sim.rpm:5.0f} rpm   pump {flow:5.1f} L/min   "
            f"batt {state['battery'] * 100:5.1f}%   accum {state['accumulator'] * 100:5.1f}%   "
            f"load {state['load_w']:4.0f} W",
            (f"auto-start: {', '.join(state['reasons'])}" if state['reasons']
             else "auto-start: no demand -- station is dark and watching"),
            f"target {state['target'] or '--':12s}  solution {state['iterations']:2d} iter"
            f"{'  converged' if state['converged'] else '  solving  '}"
            f"  error {state['tracking_error_deg']:6.2f} deg"
            f"{'   ON TARGET' if state['on_target'] else ''}"
            f"{'   FIRING' if state.get('may_fire') else ''}",
            f"rounds {station.rounds_fired:3d} of .50 AP   "
            f"plan {state.get('plan_age_s', 0.0) * 1000:4.0f} ms old   "
            + (station.log[-1][:46] if station.log else ""),
            "SPACE throttle    H add contact    C clear    ESC quit",
        ]
        for i, row in enumerate(rows):
            text.draw(row, 18, y + i * 19, AMBER if "ON TARGET" in row else INK)
        text.end_frame()
        if frames:
            buf = glReadPixels(0, 0, WINDOW[0], WINDOW[1], GL_RGB, GL_UNSIGNED_BYTE)
            img = np.frombuffer(buf, dtype=np.uint8).reshape(WINDOW[1], WINDOW[0], 3)[::-1]
            surf = pygame.image.frombuffer(np.ascontiguousarray(img).tobytes(),
                                           WINDOW, "RGB")
            if elapsed >= warm_s:
                pygame.image.save(surf, f"{out_prefix}_{captured:02d}.png")
                captured += 1
                if captured >= frames:
                    running = False
        else:
            pygame.display.flip()

    solver_service.stop()
    text.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0, help="headless: save this many PNGs")
    ap.add_argument("--out", default="turret_demo")
    ap.add_argument("--warm", type=float, default=0.0, help="seconds to run before capturing")
    a = ap.parse_args()
    raise SystemExit(main(frames=a.frames, out_prefix=a.out, warm_s=a.warm))
