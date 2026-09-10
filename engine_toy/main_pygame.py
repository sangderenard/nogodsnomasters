"""Interactive engine-revving toy (pygame frontend).

Same sim, same dual-channel audio, same dashboard content as main.py --
the only difference is input. main.py polls msvcrt one character at a
time, so "holding" a key is really just relying on the OS's keyboard
auto-repeat, and there's no way to know two keys are truly down at the
same instant. pygame.key.get_pressed() gives real per-frame boolean
state for every key at once, so throttle and brake (and the load
resistor) are driven continuously and can be worked simultaneously --
ease off the brake while blipping the throttle, the way an actual dyno
pedal box would let you.

Throttle and brake hold wherever you set them -- no pedal-spring decay,
they only change while you actually hold W/S or E/D.

Controls: same layout as main.py --
  W / S      throttle up / down       SPACE  chop throttle to 0
  E / D      brake load up / down     G      start / restart engine
  [ / ]      previous / next engine   K      kill ignition (coasts down)
  1-9, 0     jump directly to roster slot 1-9, 10
  C          build a custom engine (drops to the console for prompts)
  F          cycle fuel                A     toggle anti-lag
  O          engine view: covers off/on (valves, springs, cams shown)
  B          toggle engine brake       L     cycle listen mode
  U          toggle brake target-rpm mode: a real PI dyno controller
             holds rpm at a target (snapshotted from current rpm when
             turned on); E/D then raise/lower the TARGET instead of
             brake torque directly while it's active
  T          toggle auto-throttle rpm mode: same idea as U but drives
             THROTTLE instead of brake load -- independent of U, either/
             neither/both can be active at once; R/Y raise/lower its
             target while active
  N          toggle rev limiter -- off means a genuine unchecked
             over-rev: watch for valve float, knock, and whatever else
             the limiter actually exists to prevent
  Z / X      load resistor down / up
  , / .      quick-shift down / up -- a real two-stage clutch-out,
             swap gear, clutch-in sequence, one button each way (not
             raw clutch control -- there's no manual clutch pedal
             binding yet, just the driver-side up/down shift request)
  P          WOT dyno pull: neutral -> settle -> real quick-shift into
             top ("H") gear -> floor it to redline. A real inertia-dyno
             pull (brake load goes to 0.0 for the run, power read from
             the drum's own known inertia accelerating -- the DYNO
             dashboard line's own live numbers), captured at the
             transmission exit, not the crank
  V          bake acoustic + vibration + physics-rate snapshots to disk
  H          toggle help               Esc   quit
"""
from __future__ import annotations

import sys
import time

import pygame
import sounddevice as sd

from engines import CATALOGUE
from engine_cycle_sim import EngineCycleSim
from engine_sound import SAMPLE_RATE, BLOCK_SIZE
from audio_stream import LiveAudioState, AudioStreamer
from drivetrain_graph import build_drivetrain_graph, engine_mesh_view_graph
from engine_gl_view import EngineGLView
from gl_text import TextLayer
from toy_shared import (
    Listen, clamp, make_audio_callback, dashboard_lines, bake_snapshot,
    run_engine_builder_form, current_fuel, select_engine_by_index,
    EventLog, poll_engine_events,
)

THROTTLE_RATE_PER_S = 1.3
BRAKE_RATE_FRAC_PER_S = 0.6
LOAD_RESISTOR_RATE_PER_S = 0.35
BRAKE_TARGET_RPM_RATE_PER_S = 600.0

HELP_TEXT = __doc__.strip().split("Controls:")[1].strip()

WINDOW_SIZE = (1760, 780)
BG_COLOR = (18, 18, 22)
TEXT_COLOR = (210, 215, 225)
ACCENT_COLOR = (255, 170, 60)
LOG_COLUMN_X = 700
MESH_COLUMN_X = 1160
MESH_Y0 = 40
MESH_VIEWPORT_SIZE = 560
LOG_KIND_COLOR = {
    "knock": (255, 90, 90), "misfire": (255, 170, 60), "backfire": (255, 90, 90),
    "float": (255, 170, 60), "limiter": (140, 180, 255), "wastegate": (140, 180, 255),
    "surge": (255, 170, 60),
}


def main() -> None:
    # NOT pygame.init() -- that blanket-inits every subsystem, including
    # pygame.mixer, which opens its own real SDL audio device. This
    # program's actual audio is entirely sounddevice/PortAudio (the real
    # synthesis engine, engine_sound.py + AudioStreamer); pygame's mixer
    # was never used for anything here, just silently grabbed an audio
    # device anyway. Two independent real-time audio engines contending
    # for the same device/OS scheduler in one process is a well-known,
    # real cause of exactly the choppiness the console (main.py, no
    # pygame at all) never had -- only initializing what's actually
    # used (display + font) fixes it at the root instead of chasing the
    # symptom further downstream.
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption("Engine Toy (pygame -- continuous multi-key)")
    # OPENGL|DOUBLEBUF: the whole window is a real GL surface now -- the
    # engine mesh is drawn by the spectral analyzer's actual Phong
    # shader (engine_gl_view.py) instead of a software polygon-fill
    # loop, and even the dashboard TEXT goes through the GPU as texture
    # quads (gl_text.py) because an OPENGL-flavored window can't take
    # ordinary Surface blits any more -- there is no software backbuffer
    # to blit onto. Every draw below is therefore a real GL draw call.
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
    from OpenGL.GL import glViewport, glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    custom_engines: list = []

    def roster():
        return CATALOGUE + custom_engines

    sim = EngineCycleSim(engine=CATALOGUE[0])
    # cold on launch -- a real engine doesn't start itself; use this
    # engine's own real starting system (G) to bring it up
    audio_state = LiveAudioState(sim.engine)
    audio_state.set_from_sim(sim)
    streamer = AudioStreamer(audio_state, SAMPLE_RATE)
    streamer.start()
    listen = Listen()
    log = EventLog()

    # the live engine view: vertices/triangles/materials uploaded to the
    # GPU ONCE per engine (the static castings) and once per baked crank
    # angle (the moving parts) -- see engine_gl_view.py. Playing the
    # animation from here on is an index lookup and two draw calls, not
    # a re-derivation; bake_next() is called once per frame below and
    # is a no-op once every angle is uploaded, so switching engines
    # never freezes the window (it just shows fewer baked angles for a
    # few frames while the rest finish uploading).
    view = EngineGLView(width=MESH_VIEWPORT_SIZE, height=MESH_VIEWPORT_SIZE, covers_off=True, animation_divisions=16)
    view.set_graph(engine_mesh_view_graph(build_drivetrain_graph(sim.engine)))
    mesh_viz_engine = sim.engine

    text = TextLayer(font, *WINDOW_SIZE)
    MESH_TITLE = "engine mesh (live: baked crank frames, real Phong shader; O = covers)"

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE, channels=2, dtype="float32",
        blocksize=BLOCK_SIZE, callback=make_audio_callback(streamer, listen),
    )
    stream.start()

    help_on = False
    last_export = ""
    running = True
    try:
        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                    # a click on the live view resolves to the exact part
                    # under it (engine_rays): left = pick, right = a
                    # 7.62 mm / 3 kJ projectile along the same ray
                    mx, my = event.pos
                    if MESH_COLUMN_X <= mx < MESH_COLUMN_X + MESH_VIEWPORT_SIZE and MESH_Y0 <= my < MESH_Y0 + MESH_VIEWPORT_SIZE:
                        from engine_rays import RayMesh
                        from engine_mesh import build_engine_mesh
                        # the ray inverts the SAME real camera the GL view
                        # just drew with (view.pick_ray, from the last
                        # render's cached eye/view/proj) -- a fresh CPU
                        # mesh at the current crank angle is only built on
                        # a click, not every frame, since ray casting is a
                        # rare event and geometry stays on the GPU otherwise
                        ray = view.pick_ray(mx - MESH_COLUMN_X, my - MESH_Y0)
                        if ray is None:
                            log.add("click: view not ready yet", kind="info")
                        else:
                            static_m, moving_m = build_engine_mesh(view._graph, crank_angle_deg=sim.state.crank_angle_deg,
                                                                   covers_off=True)
                            rm = RayMesh(static_m, moving_m)
                            if event.button == 1:
                                hit = rm.pick(ray)
                                log.add(f"click: {hit.part} ({hit.material})" if hit else "click: nothing", kind="info")
                            else:
                                pen = rm.penetrate(ray, energy_j=3000.0, calibre_m=0.00762)
                                for line in pen.log[:4]:
                                    log.add("shot: " + line, kind="warn")
                                if not pen.log:
                                    log.add("shot: missed", kind="info")
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_ESCAPE:
                        running = False
                    elif key == pygame.K_g:
                        # the real start action: the starting system
                        # turns the crank over and drops out on catch
                        sim.engage_starter()
                    elif key == pygame.K_k:
                        sim.state.ignition_cut = True
                    elif key == pygame.K_LEFTBRACKET:
                        r = roster()
                        i = r.index(sim.engine) if sim.engine in r else 0
                        sim.set_engine(r[(i - 1) % len(r)])
                    elif key == pygame.K_RIGHTBRACKET:
                        r = roster()
                        i = r.index(sim.engine) if sim.engine in r else 0
                        sim.set_engine(r[(i + 1) % len(r)])
                    elif key == pygame.K_c:
                        print("\n[switch to this console window for the custom-engine form]")
                        built = run_engine_builder_form(custom_engines)
                        if built is not None:
                            custom_engines.append(built)
                            sim.set_engine(built)
                    elif key == pygame.K_f:
                        fuels = sorted(sim.engine.fuel_compatibility.keys())
                        if fuels:
                            cur = current_fuel(sim)
                            idx = fuels.index(cur) if cur in fuels else -1
                            sim.fuel_choice = fuels[(idx + 1) % len(fuels)]
                    elif key == pygame.K_a:
                        sim.anti_lag_enabled = not sim.anti_lag_enabled
                    elif key == pygame.K_b:
                        sim.engine_brake_enabled = not sim.engine_brake_enabled
                    elif key == pygame.K_u:
                        sim.brake_target_rpm = None if sim.brake_target_rpm is not None else max(sim.rpm, sim.engine.idle_rpm)
                    elif key == pygame.K_n:
                        sim.rev_limiter_enabled = not sim.rev_limiter_enabled
                    elif key == pygame.K_t:
                        sim.throttle_target_rpm = None if sim.throttle_target_rpm is not None else max(sim.rpm, sim.engine.idle_rpm)
                    elif key == pygame.K_l:
                        listen.cycle()
                    elif key == pygame.K_v:
                        last_export = bake_snapshot(sim)
                    elif key == pygame.K_o:
                        view.covers_off = not view.covers_off
                        view.set_graph(view._graph)   # re-bake at the new detail level
                    elif key == pygame.K_h:
                        help_on = not help_on
                    elif key == pygame.K_PERIOD:
                        sim.shift_up()
                    elif key == pygame.K_COMMA:
                        sim.shift_down()
                    elif key == pygame.K_p:
                        sim.start_wot_dyno_pull()
                    elif pygame.K_0 <= key <= pygame.K_9:
                        select_engine_by_index(sim, roster(), key - pygame.K_0)

            keys = pygame.key.get_pressed()
            # true continuous state: throttle and brake can move together,
            # every frame, exactly as long as each key is actually held
            if keys[pygame.K_w]:
                sim.throttle = clamp(sim.throttle + THROTTLE_RATE_PER_S * dt, 0.0, 1.0)
            if keys[pygame.K_s]:
                sim.throttle = clamp(sim.throttle - THROTTLE_RATE_PER_S * dt, 0.0, 1.0)
            if keys[pygame.K_SPACE]:
                sim.throttle = 0.0
            if keys[pygame.K_e]:
                if sim.brake_target_rpm is not None:
                    sim.brake_target_rpm = clamp(
                        sim.brake_target_rpm + BRAKE_TARGET_RPM_RATE_PER_S * dt, 0.0, sim.engine.redline_rpm * 1.1)
                else:
                    cap = sim.engine.peak_torque_nm * 1.3
                    sim.brake_load_nm = clamp(
                        sim.brake_load_nm + BRAKE_RATE_FRAC_PER_S * sim.engine.peak_torque_nm * dt, 0.0, cap)
            if keys[pygame.K_d]:
                if sim.brake_target_rpm is not None:
                    sim.brake_target_rpm = clamp(
                        sim.brake_target_rpm - BRAKE_TARGET_RPM_RATE_PER_S * dt, 0.0, sim.engine.redline_rpm * 1.1)
                else:
                    sim.brake_load_nm = clamp(
                        sim.brake_load_nm - BRAKE_RATE_FRAC_PER_S * sim.engine.peak_torque_nm * dt, 0.0, 1e9)
            if keys[pygame.K_r]:
                if sim.throttle_target_rpm is not None:
                    sim.throttle_target_rpm = clamp(
                        sim.throttle_target_rpm + BRAKE_TARGET_RPM_RATE_PER_S * dt, 0.0, sim.engine.redline_rpm * 1.1)
            if keys[pygame.K_y]:
                if sim.throttle_target_rpm is not None:
                    sim.throttle_target_rpm = clamp(
                        sim.throttle_target_rpm - BRAKE_TARGET_RPM_RATE_PER_S * dt, 0.0, sim.engine.redline_rpm * 1.1)
            if keys[pygame.K_z]:
                sim.electrical_load_frac = clamp(sim.electrical_load_frac - LOAD_RESISTOR_RATE_PER_S * dt, 0.0, 3.0)
            if keys[pygame.K_x]:
                sim.electrical_load_frac = clamp(sim.electrical_load_frac + LOAD_RESISTOR_RATE_PER_S * dt, 0.0, 3.0)

            if not sim.stalled:
                sim.step(dt)
            audio_state.set_from_sim(sim)  # audio-synth thread reads this snapshot, not sim directly
            if sim.engine is not mesh_viz_engine:
                view.set_graph(engine_mesh_view_graph(build_drivetrain_graph(sim.engine)))
                mesh_viz_engine = sim.engine
            view.bake_next()   # advance the incremental GPU upload by one frame; a no-op once complete
            sim.drain_events()
            poll_engine_events(sim, log)

            glViewport(0, 0, *WINDOW_SIZE)
            glClearColor(*(c / 255.0 for c in BG_COLOR), 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            text.begin_frame()

            lines = dashboard_lines(sim, roster(), listen, HELP_TEXT if help_on else None, last_export)
            y = 14
            for line in lines:
                color = ACCENT_COLOR if line.strip().startswith(("KNOCK", "MISFIRE", "BACKFIRE", "TURBO", "BLOWER")) else TEXT_COLOR
                text.draw(line, 16, y, color)
                y += 19

            log_y = 14
            text.draw("recent events", LOG_COLUMN_X, log_y, (140, 140, 150))
            log_y += 26
            for entry in log.active():
                fade = max(0.0, 1.0 - entry.age_s() / log.ttl_s)
                base = LOG_KIND_COLOR.get(entry.kind, TEXT_COLOR)
                color = tuple(int(BG_COLOR[i] + (base[i] - BG_COLOR[i]) * fade) for i in range(3))
                text.draw(entry.text, LOG_COLUMN_X, log_y, color)
                log_y += 20

            text.draw(MESH_TITLE, MESH_COLUMN_X, 14, (140, 140, 150))
            # the real Phong draw: two GL draw calls against geometry
            # already resident on the GPU (uploaded in set_graph/
            # bake_next above, never here) -- composited straight from
            # its own FBO texture, no CPU pixel round trip at all
            mesh_tex = view.render_gpu(crank_angle_deg=sim.state.crank_angle_deg, spin=True, dt=dt)
            text.compositor.draw_texture(mesh_tex, MESH_COLUMN_X, MESH_Y0, MESH_VIEWPORT_SIZE, MESH_VIEWPORT_SIZE,
                                         *WINDOW_SIZE, flip_v=True)

            # the material legend: what every colour in the view is --
            # engine_gl_view tracks which materials are actually present
            # in this engine's graph, the same shape the old CPU
            # visualizer's legend() returned
            legend_y = MESH_Y0 + MESH_VIEWPORT_SIZE + 8
            lx, ly = MESH_COLUMN_X, legend_y
            for label, rgb, opacity in view.legend():
                swatch = pygame.Surface((14, 10), pygame.SRCALPHA)
                swatch.fill((rgb[0], rgb[1], rgb[2], int(255 * max(opacity, 0.35))))
                text.draw_surface(swatch, lx, ly + 3, cache_key=("swatch", rgb, opacity))
                text.draw(label, lx + 18, ly, TEXT_COLOR)
                ly += 17
                if ly > WINDOW_SIZE[1] - 20:
                    break

            text.end_frame()
            pygame.display.flip()
    finally:
        stream.stop()
        stream.close()
        streamer.stop()
        view.close()
        text.close()
        pygame.quit()
        print("Engine off.")


if __name__ == "__main__":
    main()
