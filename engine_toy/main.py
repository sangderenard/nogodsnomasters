"""Interactive engine-revving toy (terminal frontend).

Load any catalogue engine, watch live stats, rev it with a throttle,
load the output shaft with a brake (dyno-style), and listen to it from
two spatial points at once: left ear = inside the header (the sharp
exhaust/combustion note), right ear = standing in the engine bay (the
softer valvetrain/pump/fan/belt noise). rpm, air delivery, ignition
timing, and electrical state are all owned by a crank-domain physical
simulator (engine_cycle_sim.py) -- lope, misfires, and knock are things
that emerge from it, not scripted audio effects. Boosted engines carry
a turbo (spools with real lag, can wastegate-flutter, surge on a hard
lift, and -- with anti-lag on -- deliberately pop on lift-off) or a
supercharger (boost tracks rpm/throttle directly, no lag); a misfired
charge igniting in a hot/boosted exhaust is an unplanned backfire.
Press V to bake the current rpm/throttle/load snapshot into
seamless-loop WAVs -- all three acoustic points (header, engine bay,
intake) AND every structural mount point -- plus a physics-rate trace
of the same source model.

Keyboard here is msvcrt single-char polling: no true "is this key held
right now" state, so simultaneous multi-key combos rely on the OS's
key-repeat timing. For real continuous multi-key input, use
main_pygame.py instead -- same sim, same audio, a pygame window.

Throttle and brake hold wherever you set them -- no pedal-spring decay,
they only change when you press a key.

Controls:
  W / S      throttle up / down       SPACE  chop throttle to 0
  E / D      brake load up / down     G      start / restart engine
  [ / ]      previous / next engine   K      kill ignition (coasts down)
  0-9        this engine's own DASH ACTUATORS (nitrous/WMI arm, starter
             select, whatever real switches its build actually has --
             see the panel, blank if none)
  i/j/m/o    ACTUATOR MACROS (a preset touching more than one control at
             once -- "AUTO: <switch>" hands that switch to its own real
             automated rule; "SAFE ALL" disarms everything)
  C          build a custom engine (pick cylinders/layout/displacement/etc)
  F          cycle fuel (wrong fuel can induce real knock)
  shift-F    CONVERT this cylinder to the next working fluid (contract shown in the log)
  B          toggle engine brake (compression/pumping braking on lift-off;
             off = freewheel/coast, valvetrain friction still applies)
  U          toggle brake target-rpm mode: a real PI dyno controller
             holds rpm at a target instead of you setting brake torque
             directly. Turning it on snapshots the current rpm as the
             target; E/D then raise/lower the TARGET (not the brake
             torque) while it's active, so you can sweep the throttle
             at a fixed rpm point the way a real dyno test would
  T          toggle auto-throttle rpm mode: a real PI governor holds
             rpm at a target by adjusting THROTTLE instead of brake --
             independent of U's brake-target mode (either, neither, or
             both can be active at once, a real dyno commonly runs a
             throttle servo and a load absorber together). Turning it
             on snapshots the current rpm as the target; R/Y then
             raise/lower that target while it's active
  N          toggle rev limiter -- off means a genuine unchecked
             over-rev: watch for valve float (past the lifter spring's
             real max_safe_rpm), knock, and whatever else the limiter
             actually exists to prevent
  - / =      captive-ball governor spring preload down / up (atmospheric
             engines only) -- the real adjusting screw on the governor's
             return spring: more preload makes the ball fight harder to
             reach trip radius, so the engine runs FASTER before it
             governs; less preload governs SOFTER/slower. No effect on
             engines without this governor (see the GOVERNOR dashboard line)
  Z / X      load resistor down / up  (baseline electrical draw --
             more draw than the alternator can cover at low rpm sags
             the battery, drops spark energy, raises misfire rate)
  , / .      quick-shift down / up -- a real two-stage clutch-out,
             swap gear, clutch-in sequence, one button each way
  P          WOT dyno pull: neutral -> settle -> real quick-shift into
             top ("H") gear -> floor it to redline, brake load released
             to 0.0 (a real inertia-dyno pull, power read off the
             drum's own known inertia accelerating), captured at the
             transmission exit
  A          toggle anti-lag (turbo builds that support it)
  L          cycle listen mode (stereo / header solo / engine-bay solo)
  V          bake acoustic + vibration + physics-rate snapshots to disk
  H          toggle help
  Q / Esc    quit
"""
from __future__ import annotations

import sys
import time

import sounddevice as sd

from engines import CATALOGUE
from engine_cycle_sim import EngineCycleSim
from engine_sound import SAMPLE_RATE, BLOCK_SIZE
from audio_stream import LiveAudioState, AudioStreamer
from toy_shared import (
    Listen, clamp, make_audio_callback, dashboard_lines, bake_snapshot,
    run_engine_builder_form, current_fuel,
    EventLog, poll_engine_events,
)
from control_panel import ControlPanel
from conversion import convert, conversion_targets

if sys.platform != "win32":
    print("This toy's keyboard control uses msvcrt and needs Windows. Try main_pygame.py instead.")
    sys.exit(1)
import msvcrt  # noqa: E402

THROTTLE_STEP = 0.02
BRAKE_STEP_FRAC = 0.08
THROTTLE_TARGET_RPM_STEP = 100.0
LOAD_RESISTOR_STEP = 0.05
BRAKE_TARGET_RPM_STEP = 100.0
GOVERNOR_PRELOAD_STEP_N = 0.05

HELP_TEXT = __doc__.strip().split("Controls:")[1].strip()

_prev_lines = 0


DASHBOARD_COLUMN_WIDTH = 74


def draw_dashboard(sim: EngineCycleSim, roster: list, listen: Listen, help_on: bool,
                    last_export: str, log: EventLog, first: bool, panel=None) -> None:
    global _prev_lines
    lines = dashboard_lines(sim, roster, listen, HELP_TEXT if help_on else None, last_export, panel=panel)
    log_entries = log.active()
    if not log_entries:
        text = "\n".join(lines)
    else:
        composed = []
        for i, line in enumerate(lines):
            if i < len(log_entries):
                entry = log_entries[i]
                age_marker = "*" if entry.age_s() < 0.5 else " "
                composed.append(f"{line:<{DASHBOARD_COLUMN_WIDTH}}{age_marker} {entry.text}")
            else:
                composed.append(line)
        text = "\n".join(composed)
    if not first:
        sys.stdout.write(f"\x1b[{_prev_lines}A")
    sys.stdout.write("\x1b[0J")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    _prev_lines = text.count("\n") + 1


def main() -> None:
    custom_engines: list = []

    def roster():
        return CATALOGUE + custom_engines

    sim = EngineCycleSim(engine=CATALOGUE[0])
    # cold on launch -- a real engine doesn't start itself; use this
    # engine's own real starting system (G) to bring it up
    panel = ControlPanel()
    panel.rebuild(sim)
    _last_panel_engine = sim.engine
    audio_state = LiveAudioState(sim.engine)
    audio_state.set_from_sim(sim)
    streamer = AudioStreamer(audio_state, SAMPLE_RATE)
    streamer.start()
    listen = Listen()
    log = EventLog()

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE, channels=2, dtype="float32",
        blocksize=BLOCK_SIZE, callback=make_audio_callback(streamer, listen),
    )
    stream.start()

    help_on = False
    last_export = ""
    # cylinder conversions (conversion.py): the base engine a converted
    # one came from, and where in the fluid list the next press goes
    conversion_base = None
    converted_engine = None
    conversion_idx = 0
    last_t = time.perf_counter()
    first = True
    try:
        while True:
            now = time.perf_counter()
            dt = now - last_t
            last_t = now

            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                lower = ch.lower()
                if lower == "w":
                    sim.throttle = clamp(sim.throttle + THROTTLE_STEP, 0.0, 1.0)
                elif lower == "s":
                    sim.throttle = clamp(sim.throttle - THROTTLE_STEP, 0.0, 1.0)
                elif ch == " ":
                    sim.throttle = 0.0
                elif lower == "e":
                    if sim.brake_target_rpm is not None:
                        sim.brake_target_rpm = clamp(sim.brake_target_rpm + BRAKE_TARGET_RPM_STEP, 0.0, sim.engine.redline_rpm * 1.1)
                    else:
                        cap = sim.engine.peak_torque_nm * 1.3
                        sim.brake_load_nm = clamp(sim.brake_load_nm + BRAKE_STEP_FRAC * sim.engine.peak_torque_nm, 0.0, cap)
                elif lower == "d":
                    if sim.brake_target_rpm is not None:
                        sim.brake_target_rpm = clamp(sim.brake_target_rpm - BRAKE_TARGET_RPM_STEP, 0.0, sim.engine.redline_rpm * 1.1)
                    else:
                        sim.brake_load_nm = clamp(sim.brake_load_nm - BRAKE_STEP_FRAC * sim.engine.peak_torque_nm, 0.0, 1e9)
                elif lower == "g":
                    # the real start action: the starting system turns
                    # the crank over and drops out on catch
                    sim.engage_starter()
                elif lower == "k":
                    sim.state.ignition_cut = True
                elif ch == ".":
                    sim.shift_up()
                elif ch == ",":
                    sim.shift_down()
                elif lower == "p":
                    sim.start_wot_dyno_pull()
                elif ch == "[":
                    r = roster()
                    i = r.index(sim.engine) if sim.engine in r else 0
                    sim.set_engine(r[(i - 1) % len(r)])
                elif ch == "]":
                    r = roster()
                    i = r.index(sim.engine) if sim.engine in r else 0
                    sim.set_engine(r[(i + 1) % len(r)])
                elif lower == "c":
                    built = run_engine_builder_form(custom_engines)
                    if built is not None:
                        custom_engines.append(built)
                        sim.set_engine(built)
                    first = True
                elif ch == "F":
                    base = conversion_base if (converted_engine is not None and sim.engine is converted_engine) else sim.engine
                    targets = conversion_targets(base)
                    if targets:
                        fluid = targets[conversion_idx % len(targets)]
                        conversion_idx += 1
                        try:
                            contract, built = convert(base, fluid)
                        except ValueError as exc:
                            log.add(f"conversion to {fluid} refused: {exc}", kind="warn")
                        else:
                            if converted_engine in custom_engines:
                                custom_engines.remove(converted_engine)
                            custom_engines.append(built)
                            conversion_base, converted_engine = base, built
                            sim.set_engine(built)
                            for line in contract.describe():
                                log.add(line)
                            first = True
                elif lower == "f":
                    fuels = sorted(sim.engine.fuel_compatibility.keys())
                    if fuels:
                        cur = current_fuel(sim)
                        idx = fuels.index(cur) if cur in fuels else -1
                        sim.fuel_choice = fuels[(idx + 1) % len(fuels)]
                elif lower == "z":
                    sim.electrical_load_frac = clamp(sim.electrical_load_frac - LOAD_RESISTOR_STEP, 0.0, 3.0)
                elif lower == "x":
                    sim.electrical_load_frac = clamp(sim.electrical_load_frac + LOAD_RESISTOR_STEP, 0.0, 3.0)
                elif lower == "b":
                    sim.engine_brake_enabled = not sim.engine_brake_enabled
                elif lower == "u":
                    sim.brake_target_rpm = None if sim.brake_target_rpm is not None else max(sim.rpm, sim.engine.idle_rpm)
                elif lower == "n":
                    sim.rev_limiter_enabled = not sim.rev_limiter_enabled
                elif ch == "-":
                    gov = sim._atmo_governor
                    if gov is not None:
                        cap = gov.spring_rate_n_per_m * (gov.r_max_m - gov.r_min_m)
                        gov.spring_preload_n = clamp(gov.spring_preload_n - GOVERNOR_PRELOAD_STEP_N, 0.0, cap)
                elif ch == "=":
                    gov = sim._atmo_governor
                    if gov is not None:
                        cap = gov.spring_rate_n_per_m * (gov.r_max_m - gov.r_min_m)
                        gov.spring_preload_n = clamp(gov.spring_preload_n + GOVERNOR_PRELOAD_STEP_N, 0.0, cap)
                elif lower == "t":
                    sim.throttle_target_rpm = None if sim.throttle_target_rpm is not None else max(sim.rpm, sim.engine.idle_rpm)
                elif lower == "r":
                    if sim.throttle_target_rpm is not None:
                        sim.throttle_target_rpm = clamp(sim.throttle_target_rpm + THROTTLE_TARGET_RPM_STEP, 0.0, sim.engine.redline_rpm * 1.1)
                elif lower == "y":
                    if sim.throttle_target_rpm is not None:
                        sim.throttle_target_rpm = clamp(sim.throttle_target_rpm - THROTTLE_TARGET_RPM_STEP, 0.0, sim.engine.redline_rpm * 1.1)
                elif ch.isdigit():
                    panel.toggle(ch)
                elif lower == "a":
                    sim.anti_lag_enabled = not sim.anti_lag_enabled
                elif lower == "l":
                    listen.cycle()
                elif lower == "v":
                    last_export = bake_snapshot(sim)
                elif lower == "h":
                    help_on = not help_on
                    first = True
                elif lower == "q" or ch == "\x1b":
                    raise KeyboardInterrupt
                elif lower in ("i", "j", "m", "o"):
                    panel.trigger_macro(lower, sim)

            if sim.engine is not _last_panel_engine:
                panel.rebuild(sim)
                _last_panel_engine = sim.engine
            panel.step(sim)
            if not sim.stalled:
                sim.step(dt)
            audio_state.set_from_sim(sim)  # audio-synth thread reads this snapshot, not sim directly
            sim.drain_events()  # keep the queue from growing
            poll_engine_events(sim, log)

            draw_dashboard(sim, roster(), listen, help_on, last_export, log, first, panel=panel)
            first = False
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        streamer.stop()
        print("\nEngine off.")


if __name__ == "__main__":
    main()
