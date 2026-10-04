"""Earth-Moon-craft demo for the managed all-body Green coast.

Run ``python orbital_green_demo.py`` for the live view, or
``python orbital_green_demo.py --frames 8 --every 4 --out shots/green`` to
save eight headless PNGs, advancing four requested windows between captures.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

from orbital_green_world import OrbitalGreenWorld


G = 6.67430e-11
EARTH_MASS_KG = 5.9722e24
MOON_MASS_KG = 7.342e22
CRAFT_MASS_KG = 1.0e3
EARTH_MOON_DISTANCE_M = 384_400_000.0
CRAFT_ORBIT_RADIUS_M = 7_000_000.0
OUTER_WINDOW_S = 600.0
LENGTH_SCALE_M = 50_000.0
WINDOW = (1120, 760)
MAP_RECT = (18, 18, 730, 724)
PANEL_RECT = (770, 18, 332, 724)
BODY_COLOURS = ((92, 160, 255), (205, 210, 222), (255, 176, 73))
BODY_NAMES = ("Earth", "Moon", "craft")


def make_world() -> OrbitalGreenWorld:
    """Create one barycentric, mutually gravitating Earth-Moon-craft world."""
    total_mass = EARTH_MASS_KG + MOON_MASS_KG
    separation = EARTH_MOON_DISTANCE_M
    omega = math.sqrt(G * total_mass / separation**3)
    earth_x = -MOON_MASS_KG / total_mass * separation
    moon_x = EARTH_MASS_KG / total_mass * separation

    craft_angle = math.radians(35.0)
    craft_offset = CRAFT_ORBIT_RADIUS_M * np.array(
        [math.cos(craft_angle), math.sin(craft_angle), 0.0])
    craft_speed = math.sqrt(G * EARTH_MASS_KG / CRAFT_ORBIT_RADIUS_M)
    craft_tangent = craft_speed * np.array(
        [-math.sin(craft_angle), math.cos(craft_angle), 0.0])

    positions = np.array([
        [earth_x, 0.0, 0.0],
        [moon_x, 0.0, 0.0],
        [earth_x + craft_offset[0], craft_offset[1], craft_offset[2]],
    ], dtype=np.float64)
    velocities = np.array([
        [0.0, -omega * MOON_MASS_KG / total_mass * separation, 0.0],
        [0.0, omega * EARTH_MASS_KG / total_mass * separation, 0.0],
        [0.0, -omega * MOON_MASS_KG / total_mass * separation, 0.0],
    ], dtype=np.float64)
    velocities[2] += craft_tangent
    return OrbitalGreenWorld(
        position_m=positions,
        velocity_m_s=velocities,
        mass_kg=(EARTH_MASS_KG, MOON_MASS_KG, CRAFT_MASS_KG),
        gravitational_constant=G,
        window_s=OUTER_WINDOW_S,
        length_scale_m=LENGTH_SCALE_M,
        cfl=0.5,
    )


def _telemetry_rows(world: OrbitalGreenWorld, telemetry) -> list[tuple[str, float]]:
    """Decode fixed telemetry and the registered participant publications."""
    values = np.asarray(telemetry, dtype=np.float64).reshape(-1)
    from llvm_dt_system import TELEMETRY_FIELDS

    rows = []
    for name, value in zip(TELEMETRY_FIELDS, values):
        if name in {"advanced", "dt_next", "residual"} or name.startswith(
                "orbital_green_"):
            rows.append((name, float(value)))
    rows.extend(world.published_metrics().items())
    return rows


def _fmt_time(seconds: float) -> str:
    hours, remainder = divmod(max(0.0, seconds), 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    return f"{hours:.0f} h {minutes:02.0f} m {seconds:04.1f} s"


def main(*, frames: int = 0, every: int = 1, out_prefix: str = "orbital_green",
         window_s: float = OUTER_WINDOW_S, headless: bool = False) -> int:
    if frames < 0:
        raise ValueError("--frames must be non-negative")
    if every < 1:
        raise ValueError("--every must be at least one")
    if not math.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("--window must be finite and positive")

    if frames or headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    world = make_world()
    pygame.init()
    screen = pygame.display.set_mode(WINDOW, pygame.HIDDEN if frames or headless else 0)
    pygame.display.set_caption("Earth-Moon-craft | Green coast on managed dt")
    font = pygame.font.SysFont("consolas", 15)
    title_font = pygame.font.SysFont("consolas", 19, bold=True)
    clock = pygame.time.Clock()
    map_surface = pygame.Surface((MAP_RECT[2], MAP_RECT[3]))
    trails = [[], [], []]
    frame = 0
    captured = 0
    latest_rows: list[tuple[str, float]] = []
    latest_dt_next = float("nan")
    latest_advanced = 0.0
    running = True

    def to_screen(relative_position):
        half = MAP_RECT[2] / 2.0
        scale = (half - 30.0) / (EARTH_MOON_DISTANCE_M * 1.03)
        return (int(half + relative_position[0] * scale),
                int(MAP_RECT[3] / 2 - relative_position[1] * scale))

    try:
        while running:
            if not (frames or headless):
                clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (
                        pygame.K_ESCAPE, pygame.K_q):
                    running = False
            if not running:
                break

            latest_advanced, dt_next, telemetry = world.advance(window_s)
            latest_dt_next = float(dt_next)
            latest_rows = _telemetry_rows(world, telemetry)
            positions = world.positions_m()
            origin = positions[0].copy()
            for index, position in enumerate(positions):
                trails[index].append(to_screen(position - origin))
                del trails[index][:-1800]

            map_surface.fill((14, 18, 27))
            pygame.draw.rect(map_surface, (28, 34, 47), map_surface.get_rect(), 1)
            for index, trail in enumerate(trails):
                if len(trail) > 1:
                    pygame.draw.lines(map_surface, (*BODY_COLOURS[index], 110),
                                      False, trail, 1)
            for index, position in enumerate(positions):
                point = to_screen(position - origin)
                radius = (12, 7, 5)[index]
                pygame.draw.circle(map_surface, BODY_COLOURS[index], point, radius)
                pygame.draw.circle(map_surface, (232, 237, 246), point, radius, 1)
                pygame.draw.circle(map_surface, BODY_COLOURS[index], point,
                                   radius + 4, 1)
            screen.fill((8, 10, 16))
            screen.blit(map_surface, MAP_RECT[:2])

            panel = pygame.Rect(PANEL_RECT)
            pygame.draw.rect(screen, (17, 21, 29), panel, border_radius=8)
            pygame.draw.rect(screen, (48, 58, 73), panel, 1, border_radius=8)
            x, y = panel.x + 16, panel.y + 16

            def draw(text: str, colour=(208, 216, 228), line_height=23,
                     use_title=False):
                nonlocal y
                screen.blit((title_font if use_title else font).render(
                    text, True, colour), (x, y))
                y += line_height

            draw("ORBITAL GREEN COAST", (255, 190, 93), 31, True)
            draw("one Earth-Moon-craft system", (160, 178, 202))
            draw(f"time   {_fmt_time(world.time_s)}")
            draw(f"window {latest_advanced:,.1f} s")
            draw(f"dt proposal {latest_dt_next:,.6g} s", (124, 220, 164))
            y += 12
            draw("GREEN / DT TELEMETRY", (255, 190, 93), 28, True)
            for name, value in latest_rows:
                if name == "orbital_green_collocation_residual_m":
                    label = "collocation residual"
                    text = f"{value:.4e} m"
                elif name == "orbital_green_total_energy_relative_residual":
                    label, text = "energy defect", f"{value:.4e}"
                elif name == "orbital_green_linear_momentum_relative_residual":
                    label, text = "momentum defect", f"{value:.4e}"
                elif name == "orbital_green_angular_momentum_relative_residual":
                    label, text = "angular defect", f"{value:.4e}"
                elif name == "residual":
                    label, text = "dt residual", f"{value:.4e}"
                elif name == "dt_next":
                    label, text = "proposal", f"{value:.6g} s"
                elif name == "advanced":
                    label, text = "advanced", f"{value:.6g} s"
                else:
                    continue
                draw(f"{label:<23} {text}", (169, 190, 211))
            y += 12
            draw("BODIES", (255, 190, 93), 28, True)
            for name, mass, colour in zip(
                    BODY_NAMES,
                    (EARTH_MASS_KG, MOON_MASS_KG, CRAFT_MASS_KG),
                    BODY_COLOURS):
                draw(f"{name:<9} {mass:.4e} kg", colour)
            y += 12
            draw("Trails show motion relative to Earth.", (148, 162, 181))
            draw("ESC or Q closes the live window.", (148, 162, 181))
            pygame.display.flip()

            if frames and frame % every == 0:
                output = Path(f"{out_prefix}_{captured:03d}.png")
                output.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(screen, str(output))
                print(f"saved {output}", flush=True)
                captured += 1
                if captured >= frames:
                    running = False
            frame += 1
        return 0
    finally:
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=0,
                        help="save this many headless PNG screenshots")
    parser.add_argument("--every", type=int, default=1,
                        help="simulation windows between screenshots")
    parser.add_argument("--out", default="orbital_green",
                        help="screenshot filename prefix")
    parser.add_argument("--window", type=float, default=OUTER_WINDOW_S,
                        help="requested outer window in seconds")
    parser.add_argument("--headless", action="store_true",
                        help="run without a visible window; defaults to one screenshot")
    args = parser.parse_args()
    raise SystemExit(main(
        frames=args.frames if args.frames else (1 if args.headless else 0),
        every=args.every, out_prefix=args.out, window_s=args.window,
        headless=args.headless,
    ))
