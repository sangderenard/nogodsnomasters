"""Top-down view of the two machines, drawing only what the sims publish.

    python time_trials/view.py                 a window, until you close it
    python time_trials/view.py --png 6         six frames to disk, no window

THIS FILE CONTAINS NO PHYSICS. It steps `race.Race` and draws the numbers
that come back. There is no tau here, no dt, no substep count and no
integration -- the last viewer had all four written in JavaScript, which
made it a second time field competing with the real one. If something is
not published by a sim, it is not drawn.

WHAT TO WATCH. The strip along the bottom is the only thing that matters
more than the cars: `world` against `wall`. The machines advance world
seconds; the clock on the wall advances wall seconds; the gap between
them is the drift, and it is real rather than staged. These engines cost
more wall time than the world time they buy, so the gap opens steadily
and the vehicles move in slow motion while the frame itself never
stretches.
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

import pygame                                                       # noqa: E402

import race as race_module                                          # noqa: E402

MAT = (16, 49, 46)
GRID = (23, 67, 63)
INK = (234, 242, 238)
MUTED = (127, 163, 156)
CAR = (226, 72, 61)
PLANE = (79, 179, 217)
AMBER = (242, 168, 29)

WIDTH, HEIGHT = 1000, 640
STRIP = 96
METRES_ACROSS = 70.0


def _world_to_screen(x_m, y_m, centre, scale):
    cx, cy = centre
    return (int(WIDTH * 0.5 + (x_m - cx) * scale),
            int((HEIGHT - STRIP) * 0.5 + (y_m - cy) * scale))


def _draw_mat(surface, centre, scale):
    surface.fill(MAT)
    step = 5.0                       # a 5 m grid, so speed is readable
    cx, cy = centre
    half_w = (WIDTH * 0.5) / scale
    half_h = ((HEIGHT - STRIP) * 0.5) / scale
    first_x = math.floor((cx - half_w) / step) * step
    while first_x < cx + half_w:
        sx, _ = _world_to_screen(first_x, cy, centre, scale)
        pygame.draw.line(surface, GRID, (sx, 0), (sx, HEIGHT - STRIP))
        first_x += step
    first_y = math.floor((cy - half_h) / step) * step
    while first_y < cy + half_h:
        _, sy = _world_to_screen(cx, first_y, centre, scale)
        pygame.draw.line(surface, GRID, (0, sy), (WIDTH, sy))
        first_y += step


def _rotate(points, heading):
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return [(px * cos_h - py * sin_h, px * sin_h + py * cos_h)
            for px, py in points]


def _draw_car(surface, row, centre, scale):
    body = _rotate([(2.0, -0.9), (2.0, 0.9), (-2.0, 0.9), (-2.0, -0.9)],
                   row["heading"])
    pygame.draw.polygon(surface, CAR, [
        _world_to_screen(row["x"] + px, row["y"] + py, centre, scale)
        for px, py in body])
    nose = _rotate([(2.6, 0.0)], row["heading"])[0]
    pygame.draw.circle(surface, INK, _world_to_screen(
        row["x"] + nose[0], row["y"] + nose[1], centre, scale), 3)


def _draw_plane(surface, row, centre, scale):
    fuselage = _rotate([(3.2, 0.0), (-2.0, -1.0), (-1.2, 0.0), (-2.0, 1.0)],
                       row["heading"])
    pygame.draw.polygon(surface, PLANE, [
        _world_to_screen(row["x"] + px, row["y"] + py, centre, scale)
        for px, py in fuselage])
    span = max(0.25, math.cos(row["bank"]))
    wing = _rotate([(0.2, -3.4 * span), (0.9, -3.4 * span),
                    (0.9, 3.4 * span), (0.2, 3.4 * span)], row["heading"])
    pygame.draw.polygon(surface, (143, 214, 240), [
        _world_to_screen(row["x"] + px, row["y"] + py, centre, scale)
        for px, py in wing])


def _draw_strip(surface, font, small, race, world_s, wall_s):
    top = HEIGHT - STRIP
    pygame.draw.rect(surface, (11, 32, 30), (0, top, WIDTH, STRIP))
    pygame.draw.line(surface, GRID, (0, top), (WIDTH, top))

    kept = (world_s / wall_s) if wall_s > 1e-9 else 0.0
    surface.blit(font.render(f"world {world_s:6.2f} s", True, AMBER), (16, top + 8))
    surface.blit(font.render(f"wall  {wall_s:6.2f} s", True, MUTED), (16, top + 32))
    surface.blit(font.render(f"keeping up {kept*100:5.1f}%", True, AMBER),
                 (200, top + 8))
    surface.blit(small.render("the frame never stretches;", True, MUTED),
                 (200, top + 38))
    surface.blit(small.render("the world runs behind", True, MUTED),
                 (200, top + 56))

    x = 470
    for row, colour in zip(race.state()["machines"], (CAR, PLANE)):
        surface.blit(small.render(row["name"].upper(), True, colour), (x, top + 8))
        surface.blit(small.render(
            f"{row['rpm']:7.0f} rpm   gear {row['gear']}", True, INK),
            (x, top + 28))
        surface.blit(small.render(
            f"{row['speed_mps']:6.2f} m/s   {row['torque_nm']:7.1f} Nm",
            True, INK), (x, top + 48))
        surface.blit(small.render(
            f"patch {row['patch_cm2']:.0f} cm2 at {row['normal_load_n']:.0f} N",
            True, MUTED), (x, top + 68))
        x += 265


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=int, default=0,
                        help="write this many sampled frames to disk instead "
                             "of opening a window")
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--out", default="time_trials/frames")
    args = parser.parse_args(argv)

    race = race_module.Race.build()
    race.control("car", throttle=0.9, steer=0.22)
    race.control("plane", throttle=1.0, steer=0.10)

    pygame.init()
    font = pygame.font.SysFont("consolas", 20)
    small = pygame.font.SysFont("consolas", 15)
    headless = args.png > 0
    if headless:
        surface = pygame.Surface((WIDTH, HEIGHT))
        Path(args.out).mkdir(parents=True, exist_ok=True)
    else:
        surface = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("time trials")

    trails = {"car": [], "plane": []}
    started = time.perf_counter()
    sample_every = max(1, args.frames // max(args.png, 1))
    written = 0

    for frame in range(args.frames):
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        race.frame()
        rows = race.state()["machines"]
        for row in rows:
            trails[row["name"]].append((row["x"], row["y"]))
            if len(trails[row["name"]]) > 900:
                trails[row["name"]].pop(0)

        centre = (sum(r["x"] for r in rows) / len(rows),
                  sum(r["y"] for r in rows) / len(rows))
        spread = max(max(abs(r["x"] - centre[0]) for r in rows),
                     max(abs(r["y"] - centre[1]) for r in rows), 6.0)
        # `max`, not `min`. Taking the minimum meant that two machines sitting
        # close together zoomed the view IN until a 4 m car filled the screen.
        # The view is at least METRES_ACROSS wide and only ever widens.
        scale = min(WIDTH, HEIGHT - STRIP) / max(METRES_ACROSS, spread * 2.5)

        _draw_mat(surface, centre, scale)
        for name, colour in (("car", CAR), ("plane", PLANE)):
            if len(trails[name]) > 1:
                pygame.draw.lines(surface, colour, False, [
                    _world_to_screen(px, py, centre, scale)
                    for px, py in trails[name]], 2)
        for row in rows:
            (_draw_plane if row["name"] == "plane" else _draw_car)(
                surface, row, centre, scale)

        world_s = max(r["world_s"] for r in rows)
        _draw_strip(surface, font, small, race, world_s,
                    time.perf_counter() - started)

        if headless:
            if frame % sample_every == 0 and written < args.png:
                path = Path(args.out) / f"frame_{written:02d}.png"
                pygame.image.save(surface, str(path))
                print(f"  wrote {path}  world {world_s:5.2f} s  "
                      f"wall {time.perf_counter()-started:5.2f} s")
                written += 1
        else:
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
