"""Watch the craft drive, on the real laws, for as long as you like.

    python time_trials/watch.py              a window, until you close it
    python time_trials/watch.py --png 3      three sampled frames to disk

THIS FILE CONTAINS NO PHYSICS. It steps `craft_graph` and draws what comes
back. Every number on screen was produced by turing's own vehicle body and
wheel-contact laws with engine_toy's real `EngineCycleSim` in the engine
slot; nothing here integrates, and nothing here has an opinion about dt.

WHAT TO WATCH. `limit` is the contact law's own stability bound, read back
out of the force and compression it published -- not recomputed from its
internals -- and `sub` is how many pieces the window was taken in to stay
inside it. `Fy` against the craft's weight is the other one: the tires
should be carrying it, and if that sum is twice the weight something is
being supported by a number somebody made up. That number is the
whole lesson of this build: run the contact law slower than its own radial
mode and it returns exactly 0.0 N of ground, silently, and the craft sits
still with its wheels spinning.

The first frame takes a while: the vehicle law is 341 inputs of SymPy and
has to be lowered before anything can move.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import threading                                                     # noqa: E402
import pygame                                                        # noqa: E402

import craft_graph as CG                                             # noqa: E402

MAT = (16, 49, 46)
GRID = (23, 67, 63)
INK = (234, 242, 238)
MUTED = (127, 163, 156)
AMBER = (242, 168, 29)
SHEAR = (255, 91, 74)
COLOURS = ((226, 72, 61), (79, 179, 217), (99, 198, 138), (242, 168, 29))

WIDTH, HEIGHT = 1080, 660
STRIP = 132
#: the view never gets tighter than this, nor wider
MINIMUM_ACROSS = 24.0
MAXIMUM_ACROSS = 400.0

CRAFT = {"roadster": "mazda-b6ze-miata-1990",
         "coupe": "vw-vr6-2800-12v"}


def to_screen(x_m, y_m, centre, scale):
    return (int(WIDTH * 0.5 + (x_m - centre[0]) * scale),
            int((HEIGHT - STRIP) * 0.5 - (y_m - centre[1]) * scale))


def draw_mat(surface, centre, scale):
    surface.fill(MAT)
    step = 5.0
    half_w = (WIDTH * 0.5) / scale
    half_h = ((HEIGHT - STRIP) * 0.5) / scale
    x = math.floor((centre[0] - half_w) / step) * step
    while x < centre[0] + half_w:
        sx, _ = to_screen(x, centre[1], centre, scale)
        pygame.draw.line(surface, GRID, (sx, 0), (sx, HEIGHT - STRIP))
        x += step
    y = math.floor((centre[1] - half_h) / step) * step
    while y < centre[1] + half_h:
        _, sy = to_screen(centre[0], y, centre, scale)
        pygame.draw.line(surface, GRID, (0, sy), (WIDTH, sy))
        y += step


def draw_craft(surface, row, colour, centre, scale, heading):
    body = [(2.1, -0.85), (2.1, 0.85), (-2.1, 0.85), (-2.1, -0.85)]
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    points = [to_screen(row["x"] + px * cos_h - py * sin_h,
                        row["z"] + px * sin_h + py * cos_h, centre, scale)
              for px, py in body]
    pygame.draw.polygon(surface, colour, points)
    nose = to_screen(row["x"] + 2.7 * cos_h, row["z"] + 2.7 * sin_h,
                     centre, scale)
    pygame.draw.circle(surface, INK, nose, 3)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=int, default=0)
    parser.add_argument("--every", type=int, default=40,
                        help="world frames between saved stills. "
                             "At the 1 ms floor 40 frames is 40 ms "
                             "of world time, which is far too little "
                             "to see anything move -- that is what "
                             "the first capture showed.")
    parser.add_argument("--throttle", type=float, default=0.85)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--out", default="time_trials/frames")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    batch = CG.EngineBatch.build(CRAFT)
    fleet, n_in, n_contact, n_carry = CG.build_graphs(list(CRAFT))
    build_seconds = time.perf_counter() - started
    names = list(CRAFT)

    pygame.init()
    font = pygame.font.SysFont("consolas", 19)
    small = pygame.font.SysFont("consolas", 14)
    headless = args.png > 0
    if headless:
        surface = pygame.Surface((WIDTH, HEIGHT))
        Path(args.out).mkdir(parents=True, exist_ok=True)
    else:
        surface = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("craft graph")

    trails = {name: [] for name in names}
    headings = {name: 0.0 for name in names}
    dt = CG.FLOOR_S * args.substeps
    frame = 0
    written = 0
    last_written = -1
    wall0 = time.perf_counter()
    world_s = 0.0

    # THE DISPLAY NEVER WAITS ON THE SIM. One eager `vehicle_step` is
    # 175 ms and the body takes two per substep, so a frame of two craft
    # costs about 1.2 s. Stepping and drawing on one thread means the
    # window pumps events once per frame, Windows marks it unresponsive
    # and stops repainting it, and a sim that IS advancing looks frozen
    # at the first frame it ever drew. That is exactly what happened.
    #
    # So the sim runs on its own thread as fast as it can and publishes a
    # snapshot; the window redraws at its own rate from whatever is
    # current. The clock on screen is then honest about both: `world` is
    # what the physics advanced, `wall` is what it took.
    # A ROW BEFORE THE FIRST STEP. `graph.last` is empty until a step has
    # run, and the display starts immediately -- by design, since it must
    # not wait on the sim. So publish a zeroed row of the right shape
    # rather than let the first redraw read a key that is not there yet.
    def blank_row():
        return {"x": 0.0, "z": 0.0, "speed_x": 0.0, "speed_z": 0.0,
                "wheel_omega": 0.0, "clutch_torque_nm": 0.0,
                "driveline_torque_nm": 0.0, "contact_force_x": 0.0,
                "contact_force_y": 0.0, "patch_cm2": 0.0,
                "compression_m": 0.0, "limit_ms": float("inf"),
                "substeps": 1.0}

    def snapshot(name):
        row = blank_row()
        row.update(fleet.craft[name].last or {})
        return row

    published = {"rows": [snapshot(n) for n in names],
                 "world_s": 0.0, "frames": 0, "frame_ms": 0.0}
    lock = threading.Lock()
    stop = threading.Event()

    # one throttle per craft, so the second one trails and the two
    # actually separate -- the body law has no steering hardware to
    # separate them any other way.
    throttle = {name: args.throttle * (1.0 if index == 0 else 0.55)
                for index, name in enumerate(names)}

    def simulate():
        world = 0.0
        count = 0
        while not stop.is_set():
            started_frame = time.perf_counter()
            batch.step(dt)
            fleet.step(dt,
                       shaft_omega={n: batch.exterior(n)[0] for n in names},
                       throttle=throttle)
            world += dt
            count += 1
            with lock:
                published["rows"] = [snapshot(n) for n in names]
                published["world_s"] = world
                published["frames"] = count
                published["frame_ms"] = (time.perf_counter() - started_frame) * 1000.0

    worker = threading.Thread(target=simulate, daemon=True)
    worker.start()

    clock = pygame.time.Clock()
    running = True
    while running:
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
        with lock:
            rows = list(published["rows"])
            world_s = published["world_s"]
            frame = published["frames"]
            frame_ms = published["frame_ms"]
        for index, name in enumerate(names):
            row = rows[index]
            headings[name] = math.atan2(row["speed_z"], row["speed_x"] + 1e-9)
            trails[name].append((row["x"], row["z"]))
            if len(trails[name]) > 1200:
                trails[name].pop(0)

        # FIT EVERY CRAFT, ALWAYS. The view is framed on what the craft
        # actually span plus a margin, with a floor so a tight pack is not
        # magnified into abstraction and a ceiling so nothing leaves the
        # mat. Taking a fixed width instead left both craft as two pixels
        # in the middle of an empty grid.
        xs = [r["x"] for r in rows]
        zs = [r["z"] for r in rows]
        centre = (0.5 * (min(xs) + max(xs)), 0.5 * (min(zs) + max(zs)))
        span_x = max(max(xs) - min(xs), 1e-6)
        span_z = max(max(zs) - min(zs), 1e-6)
        needed = max(span_x * 1.35 + 12.0, (span_z * 1.35 + 12.0)
                     * WIDTH / max(HEIGHT - STRIP, 1))
        needed = min(max(needed, MINIMUM_ACROSS), MAXIMUM_ACROSS)
        scale = WIDTH / needed

        draw_mat(surface, centre, scale)
        # WHEN THEY ARE ON TOP OF EACH OTHER. Two craft within a few
        # pixels draw over one another and the one underneath is simply
        # never seen -- which is how a red car went missing for a whole
        # session. Any craft sharing screen space with an earlier one gets
        # a ring and a short leader out to its label, so the pile is
        # legible as a pile rather than as a single craft.
        placed = []
        crowded = {}
        for index, name in enumerate(names):
            here = to_screen(rows[index]["x"], rows[index]["z"],
                             centre, scale)
            near = [other for other, spot in placed
                    if math.hypot(spot[0] - here[0], spot[1] - here[1]) < 26]
            crowded[name] = len(near)
            placed.append((name, here))

        for index, name in enumerate(names):
            colour = COLOURS[index % len(COLOURS)]
            if len(trails[name]) > 1:
                pygame.draw.lines(surface, colour, False,
                                  [to_screen(px, py, centre, scale)
                                   for px, py in trails[name]], 2)
            draw_craft(surface, rows[index], colour, centre, scale,
                       headings[name])
            stack = crowded[name]
            if stack:
                spot = to_screen(rows[index]["x"], rows[index]["z"],
                                 centre, scale)
                radius = 16 + 9 * stack
                pygame.draw.circle(surface, colour, spot, radius, 2)
                tip = (spot[0] + radius + 46, spot[1] - radius - 22 * stack)
                pygame.draw.line(surface, colour,
                                 (spot[0] + radius, spot[1]), tip, 1)
                surface.blit(small.render(name, True, colour),
                             (tip[0] + 4, tip[1] - 8))

        top = HEIGHT - STRIP
        pygame.draw.rect(surface, (11, 32, 30), (0, top, WIDTH, STRIP))
        pygame.draw.line(surface, GRID, (0, top), (WIDTH, top))
        wall = time.perf_counter() - wall0
        surface.blit(font.render(f"world {world_s:7.2f} s", True, AMBER),
                     (16, top + 8))
        surface.blit(font.render(f"wall  {wall:7.2f} s", True, MUTED),
                     (16, top + 30))
        surface.blit(small.render(
            f"turing vehicle law {n_in} inputs, contact {n_contact}, "
            f"{n_carry} carried", True, MUTED), (16, top + 58))
        surface.blit(small.render(
            f"engines batched in one dt round; built in "
            f"{build_seconds:.0f}s", True, MUTED), (16, top + 78))
        surface.blit(small.render(
            f"{len(names)} craft, no steering port; "
            f"{frame_ms:.0f} ms per sim frame", True, MUTED), (16, top + 98))

        x = 430
        for index, name in enumerate(names):
            row = rows[index]
            colour = COLOURS[index % len(COLOURS)]
            rpm = float(batch.sims[name].rpm or 0.0)
            surface.blit(small.render(name.upper(), True, colour), (x, top + 8))
            surface.blit(small.render(
                f"{rpm:7.0f} rpm   {row['speed_x']:6.2f} m/s", True, INK),
                (x, top + 28))
            surface.blit(small.render(
                f"Fx {row['contact_force_x']:8.0f} N  Fy {row['contact_force_y']:8.0f} N",
                True, INK), (x, top + 48))
            surface.blit(small.render(
                f"patch {row['patch_cm2']:6.0f} cm2  clutch "
                f"{row['clutch_torque_nm']:7.0f} Nm", True, MUTED),
                (x, top + 68))
            floor_colour = SHEAR if row["substeps"] > 32 else AMBER
            surface.blit(small.render(
                f"limit {row['limit_ms']:5.2f} ms   sub {row['substeps']:.0f}",
                True, floor_colour), (x, top + 88))
            surface.blit(small.render(
                f"travelled {math.hypot(row['x'], row['z']):8.1f} m", True, MUTED),
                (x, top + 108))
            x += 320

        if headless:
            if (written < args.png and frame % args.every == 1
                    and frame != last_written):
                last_written = frame
                path = Path(args.out) / f"craft_{written:02d}.png"
                pygame.image.save(surface, str(path))
                print(f"  wrote {path}  world {world_s:.2f}s  wall {wall:.2f}s")
                written += 1
            if written >= args.png:
                running = False
        else:
            pygame.display.flip()
            clock.tick(30)

    stop.set()
    pygame.quit()


if __name__ == "__main__":
    main()
