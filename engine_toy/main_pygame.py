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
  I          toggle compressed-air idle assist (engines with a reserve
             tank and a declared dump port only, e.g. the C18): below
             the trip rpm the reservoir dumps into the manifold as real
             boost through a real large-bore choked port -- and drains
  M          pull / refit the catalytic converter -- a real part: its
             restriction (real backpressure -> real power), its sound
             damping, its conversion and its precious-metal scrap value
             all leave with it
  J          cycle the room: outdoors -> garage door open -> closed ->
             sealed; the tailpipe empties into whatever room there is
  Q          (with a room) fit / remove the exhaust extractor duct -- the
             big orange high-temp hose over the tailpipe, vented outside
  N          toggle rev limiter -- off means a genuine unchecked
             over-rev: watch for valve float, knock, and whatever else
             the limiter actually exists to prevent
  - / =      captive-ball governor spring preload down / up (atmospheric
             engines only) -- the real adjusting screw on the governor's
             return spring: more preload governs FASTER, less governs
             softer/slower. No effect on engines without this governor
  Z / X      load resistor down / up
  , / .      quick-shift down / up -- a real two-stage clutch-out,
             swap gear, clutch-in sequence, one button each way (not
             raw clutch control -- there's no manual clutch pedal
             binding yet, just the driver-side up/down shift request)
  P          WOT dyno pull: neutral -> settle -> first gear -> real
             quick-shifts through every ratio -> final-gear redline. A real inertia-dyno
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
from engine_cycle_sim import EngineCycleSim, BUTTERFLY_MAX_ANGLE_DEG
from engine_sound import SAMPLE_RATE, BLOCK_SIZE
from audio_stream import LiveAudioState, AudioStreamer
from drivetrain_graph import build_drivetrain_graph, engine_mesh_view_graph
from engine_gl_view import EngineGLView, thermal_groups_from_state
from engine_rays import RayMesh
from calibres import get_calibre
from engine_mesh import build_engine_mesh, enclosed_bodies, breached_casings
from combustion_kernel import visual_for, combustion_state_from_sim
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
GOVERNOR_PRELOAD_RATE_PER_S = 0.15

HELP_TEXT = __doc__.strip().split("Controls:")[1].strip()

WINDOW_SIZE = (1760, 780)
BG_COLOR = (18, 18, 22)
TEXT_COLOR = (210, 215, 225)
ACCENT_COLOR = (255, 170, 60)
LOG_COLUMN_X = 700
MESH_VIEWPORT_SIZE = 620
MESH_Y0 = 16                                          # was 40 -- closer to the top
MESH_COLUMN_X = WINDOW_SIZE[0] - MESH_VIEWPORT_SIZE - 20   # flush against the right edge, computed so it stays that way if WINDOW_SIZE ever changes
# The weapon selector, sitting under the engine view where the shooting
# happens. Right-click has always fired something at the part under the
# cursor; it fired a hard-coded 7.62 mm because there was nowhere to say
# otherwise. These are real cartridges out of calibres.py -- the button
# only chooses which one, and the cartridge itself supplies the mass,
# the speed, the core hardness and the energy.
WEAPON_BUTTONS = ("9mm", ".45 acp", ".500 s&w", "7.62 nato", ".50 bmg", ".50 ap")
WEAPON_BUTTON_H = 20
WEAPON_BUTTON_PAD = 6
# The mesh reduction stage is optional (engine_gl_view.EngineGLView's own
# detail/spring_style): 1.0 + "helix" is the full high-triangle build the
# exporter uses; the live view defaults to full tessellation with cheap
# cylinder springs. Lower MESH_DETAIL if a machine needs the frame rate.
MESH_DETAIL = 1.0
MESH_SPRING_STYLE = "cylinder"
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
    log = EventLog(max_entries=30)

    # the live engine view: vertices/triangles/materials uploaded to the
    # GPU ONCE per engine (the static castings) and once per baked crank
    # angle (the moving parts) -- see engine_gl_view.py. Playing the
    # animation from here on is an index lookup and two draw calls, not
    # a re-derivation; bake_next() is called once per frame below and
    # is a no-op once every angle is uploaded, so switching engines
    # never freezes the window (it just shows fewer baked angles for a
    # few frames while the rest finish uploading).
    view = EngineGLView(width=MESH_VIEWPORT_SIZE, height=MESH_VIEWPORT_SIZE, covers_off=True, animation_divisions="dense",
                        detail=MESH_DETAIL, spring_style=MESH_SPRING_STYLE)
    view.set_graph(engine_mesh_view_graph(build_drivetrain_graph(sim.engine)))
    mesh_viz_engine = sim.engine
    # the shrapnel cascade (burst.py) fires fragments through the same
    # live geometry a right-click uses
    # covers_off=False deliberately, and NOT view.covers_off: what the
    # player has chosen to see through does not change what is bolted to
    # the engine. With covers_off=True here, every shot and every
    # shrapnel fragment was resolved against an engine that had no valve
    # cover, no cam case and no valley cover -- a round came down onto
    # the valvetrain through open air and the cover neither holed nor
    # took any energy out of it.
    sim.ray_mesh_factory = lambda: RayMesh(*build_engine_mesh(view._graph, crank_angle_deg=sim.state.crank_angle_deg, covers_off=False))
    # the combustion kernel: per-cylinder burn/residue frames baked off
    # the sim's OWN firing angles and the live fuel's own visual family
    mesh_viz_fuel = sim.fuel_choice or sim.engine.preferred_fuel_profile
    view.bake_combustion(sim._firing_angle_deg, visual_for(sim.engine, mesh_viz_fuel))

    text = TextLayer(font, *WINDOW_SIZE)
    MESH_TITLE = "engine mesh (live: baked crank frames, real Phong shader; O = covers)"

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE, channels=2, dtype="float32",
        blocksize=BLOCK_SIZE, callback=make_audio_callback(streamer, listen),
    )
    stream.start()

    help_on = False
    last_export = ""
    hover_part: str | None = None
    hover_material: str | None = None
    hover_local_x = -1
    hover_local_y = -1
    hover_angle_bucket = None
    hover_ray_mesh: RayMesh | None = None

    # which cartridge the right mouse button is loaded with, and where
    # its buttons are on screen
    selected_weapon = "7.62 nato"

    def _weapon_button_rects() -> list[tuple[str, pygame.Rect]]:
        """One row of buttons across the bottom of the engine view,
        sized to the column so they stay inside it whatever the window
        is."""
        n = len(WEAPON_BUTTONS)
        gap = 4
        w = (MESH_VIEWPORT_SIZE - gap * (n - 1)) // n
        y = MESH_Y0 + MESH_VIEWPORT_SIZE + 4
        return [(name, pygame.Rect(MESH_COLUMN_X + i * (w + gap), y, w, WEAPON_BUTTON_H))
                for i, name in enumerate(WEAPON_BUTTONS)]

    def _weapon_hit(mx: int, my: int) -> str | None:
        for name, rect in _weapon_button_rects():
            if rect.collidepoint(mx, my):
                return name
        return None

    def _restore_damage_cutouts() -> None:
        punctures = (
            puncture
            for state in sim.state.part_damage.values()
            for puncture in state.punctures
        )
        dropped = view.set_cutout_punctures(punctures)
        if dropped:
            log.add(f"damage view: {dropped} oldest puncture tunnels exceed the 64-cutout GPU limit",
                    kind="warn", cooldown=False)

    def _update_hover_part(mx: int, my: int) -> None:
        nonlocal hover_part, hover_material, hover_local_x, hover_local_y
        nonlocal hover_ray_mesh, hover_angle_bucket
        if not (MESH_COLUMN_X <= mx < MESH_COLUMN_X + MESH_VIEWPORT_SIZE and MESH_Y0 <= my < MESH_Y0 + MESH_VIEWPORT_SIZE):
            hover_part = None
            hover_material = None
            return
        hover_local_x = mx - MESH_COLUMN_X
        hover_local_y = my - MESH_Y0
        ray = view.pick_ray(hover_local_x, hover_local_y)
        if ray is None:
            hover_part = None
            hover_material = None
            return
        angle_bucket = round(float(sim.state.crank_angle_deg), 2)
        mesh_key = (id(view._graph), angle_bucket, view.covers_off)
        if hover_ray_mesh is None or hover_angle_bucket != mesh_key:
            static_m, moving_m = build_engine_mesh(view._graph, crank_angle_deg=float(sim.state.crank_angle_deg), covers_off=view.covers_off)
            hover_ray_mesh = RayMesh(static_m, moving_m)
            hover_angle_bucket = mesh_key
        hit = hover_ray_mesh.pick(ray)
        hover_part = hit.part if hit is not None else None
        hover_material = hit.material if hit is not None else None

    running = True
    try:
        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    _update_hover_part(*event.pos)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and _weapon_hit(*event.pos) is not None:
                    selected_weapon = _weapon_hit(*event.pos)
                    c = get_calibre(selected_weapon)
                    log.add(f"weapon: {c.name} -- {c.mass_kg * 1000:.1f} g at {c.muzzle_m_s:.0f} m/s, "
                            f"{c.muzzle_energy_j / 1000:.2f} kJ", kind="info")
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                    # a click on the live view resolves to the exact part
                    # under it (engine_rays): left = pick, right = a
                    # 7.62 mm / 3 kJ projectile along the same ray
                    mx, my = event.pos
                    if MESH_COLUMN_X <= mx < MESH_COLUMN_X + MESH_VIEWPORT_SIZE and MESH_Y0 <= my < MESH_Y0 + MESH_VIEWPORT_SIZE:
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
                                # the selected cartridge, not a fixed
                                # one: its own real mass, speed, bore
                                # and core hardness
                                c = get_calibre(selected_weapon)
                                shot = c.projectile(ray.direction)
                                pen = rm.penetrate(
                                    ray,
                                    energy_j=shot.energy_j,
                                    calibre_m=c.diameter_m,
                                    projectile=shot,
                                    fluid_by_part=sim.ballistic_fluid_for_part,
                                )
                                punctures = sim.apply_penetration(pen)
                                dropped = view.add_cutout_punctures(puncture for _, puncture in punctures)
                                if dropped:
                                    log.add(f"damage view: {dropped} oldest puncture tunnels exceed the 64-cutout GPU limit",
                                            kind="warn", cooldown=False)
                                for index, line in enumerate(pen.log, start=1):
                                    log.add(f"shot {index:02d}: {line}", kind="warn", cooldown=False)
                                if punctures:
                                    boundary_holes = sum(puncture.boundary_holes for _, puncture in punctures)
                                    through_parts = sum(1 for _, puncture in punctures if puncture.through)
                                    log.add(f"damage: {through_parts} parts penetrated, {boundary_holes} boundary holes recorded",
                                            kind="warn", cooldown=False)
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
                        # off -> exhaust brake -> (+ compression release, if fitted) -> off
                        if not sim.engine_brake_enabled:
                            sim.engine_brake_enabled = True; sim.compression_brake_enabled = False
                        elif sim.engine.compression_release_brake and not sim.compression_brake_enabled:
                            sim.compression_brake_enabled = True
                        else:
                            sim.engine_brake_enabled = False; sim.compression_brake_enabled = False
                    elif key == pygame.K_u:
                        sim.brake_target_rpm = None if sim.brake_target_rpm is not None else max(sim.rpm, sim.engine.idle_rpm)
                    elif key == pygame.K_n:
                        sim.rev_limiter_enabled = not sim.rev_limiter_enabled
                    elif key == pygame.K_m:
                        if sim.engine.exhaust_system.layout_has_catalyst:
                            sim.engine.exhaust_system.catalyst_fitted = not sim.engine.exhaust_system.catalyst_fitted
                            mesh_viz_engine = None   # the pipe changed: rebuild the real graph + mesh
                            cat = sim.catalytic_converter
                            log.add(("catalytic converter refitted" if cat else
                                     "catalytic converter PULLED -- in your hand: "
                                     f"{sim.engine.displacement_l * 0.8:.1f} L brick"), "info")
                    elif key == pygame.K_q:
                        sim.exhaust_extractor_fitted = not sim.exhaust_extractor_fitted
                        log.add("exhaust extractor duct " + ("FITTED over the tailpipe" if sim.exhaust_extractor_fitted else "removed"), "info")
                    elif key == pygame.K_j:
                        modes = ["outdoors", "open", "closed", "sealed"]
                        sim.garage_mode = modes[(modes.index(sim.garage_mode) + 1) % len(modes)]
                        log.add(f"room: {sim.garage_mode}", "info")
                    elif key == pygame.K_i:
                        sim.pneumatic_idle_assist_enabled = not sim.pneumatic_idle_assist_enabled
                    elif key == pygame.K_t:
                        sim.throttle_target_rpm = None if sim.throttle_target_rpm is not None else max(sim.rpm, sim.engine.idle_rpm)
                    elif key == pygame.K_l:
                        listen.cycle()
                    elif key == pygame.K_v:
                        last_export = bake_snapshot(sim)
                    elif key == pygame.K_o:
                        view.covers_off = not view.covers_off
                        view.set_graph(view._graph)   # re-bake at the new detail level
                        _restore_damage_cutouts()
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
            if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
                gov = sim._atmo_governor
                if gov is not None:
                    cap = gov.spring_rate_n_per_m * (gov.r_max_m - gov.r_min_m)
                    gov.spring_preload_n = clamp(gov.spring_preload_n - GOVERNOR_PRELOAD_RATE_PER_S * dt, 0.0, cap)
            if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
                gov = sim._atmo_governor
                if gov is not None:
                    cap = gov.spring_rate_n_per_m * (gov.r_max_m - gov.r_min_m)
                    gov.spring_preload_n = clamp(gov.spring_preload_n + GOVERNOR_PRELOAD_RATE_PER_S * dt, 0.0, cap)

            if not sim.stalled:
                sim.step(dt)
            audio_state.set_from_sim(sim)  # audio-synth thread reads this snapshot, not sim directly
            live_fuel = sim.fuel_choice or sim.engine.preferred_fuel_profile
            if sim.engine is not mesh_viz_engine or live_fuel != mesh_viz_fuel:
                if sim.engine is not mesh_viz_engine:
                    view.set_graph(engine_mesh_view_graph(build_drivetrain_graph(sim.engine)))
                    _restore_damage_cutouts()
                    mesh_viz_engine = sim.engine
                    hover_ray_mesh = None
                    hover_angle_bucket = None
                    hover_part = None
                    hover_material = None
                mesh_viz_fuel = live_fuel
                view.bake_combustion(sim._firing_angle_deg, visual_for(sim.engine, live_fuel))
            view.bake_next()   # advance the incremental GPU upload by one frame; a no-op once complete
            # live blackbody self-emission: every part glows off its own
            # thermal group's REAL temperature this tick (Planck) -- a
            # diagnostic, not a schedule: a bay whose coolant stopped
            # flowing shows it
            view.set_thermal_state(thermal_groups_from_state(sim.state))
            # and each cylinder's own live burn, off the sim's firing events
            view.set_combustion_state(combustion_state_from_sim(sim, view.combustion_burn_duration_deg))
            # the fluid emitters (hole_emitters.py): droplets off every
            # leaking hole and every crank dipper, re-uploaded per tick
            view.set_emitter_particles([e for e in sim.hole_emitters.emitters if e.regime != "none"] + sim.bursts.clouds())
            view.set_absent_parts(sim.state.absent_parts)
            # and the parts that are merely out of sight: anything
            # sealed inside a casing nothing has holed yet. Recomputed
            # per tick because a single through-hole reveals everything
            # behind it, and the set is cheap -- set_hidden_parts is a
            # no-op unless it actually changed.
            view.set_hidden_parts(enclosed_bodies(
                view._graph, breached_casings(sim.hole_emitters.emitters)))
            sim.drain_events()
            poll_engine_events(sim, log)

            glViewport(0, 0, *WINDOW_SIZE)
            glClearColor(*(c / 255.0 for c in BG_COLOR), 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            text.begin_frame()

            # how many lines this window can really draw, computed from
            # the window rather than assumed: 19 px per line from y=14
            fits = max(10, (WINDOW_SIZE[1] - 14) // 19)
            lines = dashboard_lines(sim, roster(), listen, HELP_TEXT if help_on else None,
                                    last_export, max_lines=fits)
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
            # the real live butterfly-plate angle (engine_cycle_sim's
            # own throttle_plate_angle_deg, already computed every
            # tick for the actual airflow physics) drives a SEPARATE
            # baked frame set composited alongside the crank-angle one
            # -- see engine_gl_view.EngineGLView._draw's own comment.
            throttle_frac = sim.state.throttle_plate_angle_deg / BUTTERFLY_MAX_ANGLE_DEG
            mesh_tex = view.render_gpu(crank_angle_deg=sim.state.crank_angle_deg, spin=True, dt=dt,
                                       throttle_frac=throttle_frac)
            text.compositor.draw_texture(mesh_tex, MESH_COLUMN_X, MESH_Y0, MESH_VIEWPORT_SIZE, MESH_VIEWPORT_SIZE,
                                         *WINDOW_SIZE, flip_v=True)

            # Mouse motion performs one nearest-surface pick. Full ray
            # traversal is reserved for an actual projectile click.
            info_y = MESH_Y0 + MESH_VIEWPORT_SIZE + 8
            if hover_part is not None:
                text.draw(f"mouse [{hover_local_x:4d},{hover_local_y:4d}]  first: {hover_part} ({hover_material})",
                          LOG_COLUMN_X, info_y, (170, 190, 210))
            else:
                text.draw("mouse: no surface", LOG_COLUMN_X, info_y, (120, 120, 120))

            # the material legend: what every colour in the view is --
            # engine_gl_view tracks which materials are actually present
            # in this engine's graph, the same shape the old CPU
            # visualizer's legend() returned
            # the weapon row: what the right mouse button is loaded
            # with, selectable without leaving the view you are shooting
            for name, rect in _weapon_button_rects():
                chosen = name == selected_weapon
                swatch = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
                swatch.fill((70, 96, 130, 235) if chosen else (34, 38, 46, 210))
                pygame.draw.rect(swatch, (150, 190, 235) if chosen else (70, 74, 84),
                                 pygame.Rect(0, 0, rect.w, rect.h), 1)
                text.draw_surface(swatch, rect.x, rect.y, cache_key=("weapon_btn", rect.w, rect.h, chosen))
                text.draw(name, rect.x + 5, rect.y + 4,
                          (235, 240, 250) if chosen else (150, 155, 165))

            legend_y = MESH_Y0 + MESH_VIEWPORT_SIZE + 8 + WEAPON_BUTTON_H + WEAPON_BUTTON_PAD
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
