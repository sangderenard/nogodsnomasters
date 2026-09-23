"""A tiny LLVM-powered Wooly Willy in a Pygame OpenGL window.

The filings do not use a hand-written magnetic approximation.  Their field
and attraction magnitude are the catalogue's Faraday equations F16.9, F16.10
and F16.15, compiled together through the repository's required route:

    SymPy equations -> repository SSA -> AbstractTensor Python with a batch
    shape -> whole-program SSA -> LLVM -> one native call over every filing.

OpenGL only draws the resulting positions and field directions.  Contact is
resolved in fixed eight-filing cohorts: N5.7 supplies the zero-restitution
normal impulse, N5.4/N5.5 bound the tangential Coulomb impulse, and WI5.7
accounts for dissipated friction work.  Magnetically touching filings enter a
persistent bond graph; three boolean matrix squarings close every eight-member
component and its members share momentum, so clumps really become inseparable.

Controls
--------
Hold left mouse    attract and comb the filings
Hold right mouse   repel filings (useful as an eraser)
R                  refill the tray
S                  shake the toy
Escape             quit

For a noninteractive smoke/render run::

    python wooly_willy.py --frames 180 --screenshot build/wooly_willy.png
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
TURING_ROOT = HERE.parent / "turing"
if str(TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(TURING_ROOT))


@lru_cache(maxsize=1)
def compile_wooly_willy_equations():
    """Compile the exact catalogue equations that drive the filings."""

    from honorary_engine_equation_catalogue import (
        eq_F16_9,
        eq_F16_10,
        eq_F16_15,
    )
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations

    return compile_sympy_equations(
        (eq_F16_9, eq_F16_10, eq_F16_15),
        name="wooly_willy_magnetic_field",
    )


@lru_cache(maxsize=8)
def compiled_filing_kernel(count: int):
    """The batch-shaped AbstractTensor stage compiled into LLVM."""

    from src.compiler.native_law_kernels import law_kernel

    return law_kernel(
        compile_wooly_willy_equations(),
        "wooly_willy_magnetic_field",
        int(count),
        "llvm",
    )


class Filings:
    """Movable iron filings whose magnetic quantities come from LLVM."""

    def __init__(self, count: int, seed: int = 19):
        self.count = int(count)
        if self.count % 8:
            raise ValueError("filing count must be divisible by eight")
        self.groups = self.count // 8
        self.rng = np.random.default_rng(seed)
        self.kernel = compiled_filing_kernel(self.count)
        self.ones = np.ones(self.count, dtype=np.float64)
        self.positions = np.empty((self.count, 2), dtype=np.float64)
        self.velocity = np.zeros_like(self.positions)
        self.direction = np.zeros_like(self.positions)
        self.direction[:, 1] = 1.0
        self.strength = np.zeros(self.count, dtype=np.float64)
        self.bond = np.zeros((self.groups, 8, 8), dtype=bool)
        self.friction_work = 0.0
        self.reset()

    def reset(self) -> None:
        # The loose powder tray along the bottom of the classic toy.
        self.positions[:, 0] = self.rng.uniform(-0.78, 0.78, self.count)
        self.positions[:, 1] = self.rng.uniform(-0.88, -0.72, self.count)
        self.velocity.fill(0.0)
        self.bond.fill(False)
        diagonal = np.arange(8)
        self.bond[:, diagonal, diagonal] = True
        self.friction_work = 0.0
        self.direction[:, 0] = self.rng.uniform(-1.0, 1.0, self.count)
        self.direction[:, 1] = self.rng.uniform(-0.25, 0.25, self.count)
        self._normalize_directions()

    def shake(self) -> None:
        self.velocity += self.rng.normal(0.0, 0.8, self.velocity.shape)

    def _normalize_directions(self) -> None:
        lengths = np.sqrt(np.sum(self.direction * self.direction, axis=1))
        lengths = np.maximum(lengths, 1e-12)
        self.direction /= lengths[:, None]

    def _resolve_contacts(self, active: bool, dt: float) -> None:
        """Apply N5 contact/friction impulses and close magnetic bonds."""

        position = self.positions.reshape(self.groups, 8, 2)
        velocity = self.velocity.reshape(self.groups, 8, 2)
        diameter = 0.012
        filing_mu = 0.42
        for left in range(7):
            for right in range(left + 1, 8):
                separation = position[:, right] - position[:, left]
                distance = np.sqrt(np.sum(separation * separation, axis=1))
                touching = distance < diameter
                if not np.any(touching):
                    continue
                safe_distance = np.maximum(distance, 1e-12)
                normal = separation / safe_distance[:, None]
                relative = velocity[:, right] - velocity[:, left]
                normal_speed = np.sum(relative * normal, axis=1)
                # N5.7 with equal masses, no angular lever arm and e_r=0.
                normal_impulse = np.where(
                    touching, np.maximum(-normal_speed * 0.5, 0.0), 0.0,
                )
                tangent = relative - normal_speed[:, None] * normal
                tangent_speed = np.sqrt(np.sum(tangent * tangent, axis=1))
                tangent_direction = tangent / np.maximum(
                    tangent_speed, 1e-12,
                )[:, None]
                # N5.4/N5.5: oppose slip, capped by mu times normal impulse.
                tangent_impulse = np.minimum(
                    tangent_speed * 0.5, filing_mu * normal_impulse,
                )
                impulse = (
                    normal_impulse[:, None] * normal
                    - tangent_impulse[:, None] * tangent_direction
                )
                impulse *= touching[:, None]
                velocity[:, left] -= impulse
                velocity[:, right] += impulse
                # Position projection prevents tunnelling/overlap without
                # inventing spring energy.
                correction = (
                    np.maximum(diameter - distance, 0.0) * 0.5
                )[:, None] * normal
                position[:, left] -= correction
                position[:, right] += correction
                self.friction_work += float(np.sum(
                    tangent_impulse * tangent_speed
                ))
                if active:
                    self.bond[:, left, right] |= touching
                    self.bond[:, right, left] |= touching

        # Eight members need only three squarings for transitive closure.
        closure = self.bond
        for _ in range(3):
            closure = np.matmul(closure, closure) > 0
        self.bond = closure
        weights = closure.astype(np.float64)
        component_velocity = np.matmul(weights, velocity)
        component_size = np.sum(weights, axis=2, keepdims=True)
        velocity[:] = component_velocity / np.maximum(component_size, 1.0)

    def step(
        self,
        magnet: tuple[float, float],
        active: bool,
        repel: bool,
        dt: float,
    ) -> None:
        dx = self.positions[:, 0] - float(magnet[0])
        dy = self.positions[:, 1] - float(magnet[1])
        distance = np.sqrt(dx * dx + dy * dy)
        # A finite stylus radius is geometry, not a modification of the law.
        radius = np.maximum(distance, 0.055)
        theta = np.arctan2(dx, dy)  # catalogue polar angle, measured from +y

        produced = self.kernel({
            "m": self.ones,
            "m_1": self.ones,
            "m_2": self.ones,
            "mu_0": self.ones,
            "r": np.ascontiguousarray(radius),
            "theta": np.ascontiguousarray(theta),
        })
        b_r = np.asarray(produced["B_r"], dtype=np.float64)
        b_theta = np.asarray(produced["B_theta"], dtype=np.float64)
        attraction = np.asarray(produced["F_dd"], dtype=np.float64)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        # e_r=(sin(theta), cos(theta)); e_theta=(cos(theta), -sin(theta)).
        bx = b_r * sin_theta + b_theta * cos_theta
        by = b_r * cos_theta - b_theta * sin_theta
        field_length = np.sqrt(bx * bx + by * by)
        valid = field_length > 1e-15
        target = self.direction.copy()
        target[valid, 0] = bx[valid] / field_length[valid]
        target[valid, 1] = by[valid] / field_length[valid]
        # A filing has no arrowhead, so choose the equivalent orientation that
        # turns through the smaller angle instead of flipping by pi.
        flip = np.sum(target * self.direction, axis=1) < 0.0
        target[flip] *= -1.0
        self.direction = 0.72 * self.direction + 0.28 * target
        self._normalize_directions()
        self.strength = field_length

        if active:
            radial_x = dx / radius
            radial_y = dy / radius
            # F16.15 has its physical r^-4 dynamic range.  The display-scale
            # gain and acceleration ceiling express toy mass/friction and keep
            # a filing from tunnelling through the stylus in one frame.
            acceleration = np.minimum(attraction * 0.0025, 24.0)
            sign = 1.0 if repel else -1.0
            self.velocity[:, 0] += sign * acceleration * radial_x * dt
            self.velocity[:, 1] += sign * acceleration * radial_y * dt

        # Gravity makes loose filings settle and pile.  Coulomb tray friction
        # removes a fixed speed mu*g*dt (rather than exponential drag), the
        # force law's correct dry-friction integration.
        self.velocity[:, 1] -= 0.42 * dt
        on_tray = self.positions[:, 1] < -0.865
        tray_speed = np.sqrt(np.sum(self.velocity * self.velocity, axis=1))
        speed_loss = 0.38 * 0.42 * dt
        tray_scale = np.maximum(0.0, 1.0 - speed_loss / np.maximum(tray_speed, 1e-12))
        self.velocity[on_tray] *= tray_scale[on_tray, None]
        self.friction_work += float(np.sum(
            speed_loss * tray_speed[on_tray]
        ))
        self.velocity *= 0.997 ** (dt * 60.0)
        speed = np.sqrt(np.sum(self.velocity * self.velocity, axis=1))
        too_fast = speed > 1.6
        self.velocity[too_fast] *= (1.6 / speed[too_fast])[:, None]
        self.positions += self.velocity * dt
        self._resolve_contacts(active, dt)

        # Keep the filings inside the sealed display cavity.
        for axis, limit in ((0, 0.82), (1, 0.91)):
            low = self.positions[:, axis] < -limit
            high = self.positions[:, axis] > limit
            self.positions[low, axis] = -limit
            self.positions[high, axis] = limit
            self.velocity[low | high, axis] *= -0.25


def _circle(gl, x: float, y: float, rx: float, ry: float, color, segments=64):
    gl.glColor3f(*color)
    gl.glBegin(gl.GL_TRIANGLE_FAN)
    gl.glVertex2f(x, y)
    angles = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    for angle in angles:
        gl.glVertex2f(x + rx * np.cos(angle), y + ry * np.sin(angle))
    gl.glEnd()


def _line_loop(gl, points, color, width=2.0):
    gl.glColor3f(*color)
    gl.glLineWidth(width)
    gl.glBegin(gl.GL_LINE_LOOP)
    for x, y in points:
        gl.glVertex2f(x, y)
    gl.glEnd()


def draw_toy(gl, filings: Filings, magnet, active: bool, repel: bool) -> None:
    gl.glClearColor(0.035, 0.045, 0.055, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    # Powder-blue plastic body and cream drawing cavity.
    gl.glColor3f(0.16, 0.48, 0.68)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(-0.98, -0.98); gl.glVertex2f(0.98, -0.98)
    gl.glVertex2f(0.98, 0.98); gl.glVertex2f(-0.98, 0.98)
    gl.glEnd()
    _circle(gl, 0.0, 0.03, 0.86, 0.91, (0.91, 0.88, 0.73), 96)
    _circle(gl, 0.0, -0.82, 0.83, 0.13, (0.10, 0.34, 0.50), 64)

    # Willy's face is deliberately simple so the filings remain the star.
    for eye_x in (-0.24, 0.24):
        _circle(gl, eye_x, 0.27, 0.105, 0.075, (0.98, 0.98, 0.94), 36)
        _circle(gl, eye_x, 0.27, 0.035, 0.047, (0.12, 0.20, 0.22), 24)
    gl.glColor3f(0.60, 0.42, 0.30)
    gl.glLineWidth(3.0)
    gl.glBegin(gl.GL_LINE_STRIP)
    gl.glVertex2f(0.0, 0.20); gl.glVertex2f(-0.055, -0.04)
    gl.glVertex2f(0.05, -0.05)
    gl.glEnd()
    angles = np.linspace(np.pi * 1.12, np.pi * 1.88, 30)
    gl.glBegin(gl.GL_LINE_STRIP)
    for angle in angles:
        gl.glVertex2f(0.24 * np.cos(angle), -0.13 + 0.15 * np.sin(angle))
    gl.glEnd()

    # One GL batch for all iron filings.
    positions = filings.positions
    directions = filings.direction
    local_strength = np.clip(np.log1p(filings.strength) / 8.0, 0.0, 1.0)
    half_length = 0.010 + 0.010 * local_strength
    gl.glLineWidth(1.35)
    gl.glBegin(gl.GL_LINES)
    for index in range(filings.count):
        shade = 0.055 + 0.10 * local_strength[index]
        gl.glColor3f(shade, shade * 0.90, shade * 0.72)
        sx = directions[index, 0] * half_length[index]
        sy = directions[index, 1] * half_length[index]
        gl.glVertex2f(positions[index, 0] - sx, positions[index, 1] - sy)
        gl.glVertex2f(positions[index, 0] + sx, positions[index, 1] + sy)
    gl.glEnd()

    if active:
        x, y = magnet
        direction = -1.0 if repel else 1.0
        gl.glLineWidth(8.0)
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(0.90, 0.16, 0.12)
        gl.glVertex2f(x, y); gl.glVertex2f(x, y + 0.075 * direction)
        gl.glColor3f(0.10, 0.28, 0.88)
        gl.glVertex2f(x, y); gl.glVertex2f(x, y - 0.075 * direction)
        gl.glEnd()

    _line_loop(
        gl,
        [(-0.86, -0.93), (0.86, -0.93), (0.91, -0.85),
         (0.91, 0.80), (0.82, 0.94), (-0.82, 0.94),
         (-0.91, 0.80), (-0.91, -0.85)],
        (0.04, 0.18, 0.28),
        3.0,
    )


def _automatic_magnet(frame: int) -> tuple[float, float]:
    """A pleasant combing path for screenshots and unattended demos."""

    phase = frame * 0.035
    x = 0.52 * np.sin(phase)
    # Alternate between scalp and moustache height.
    band = 0.58 if (frame // 90) % 2 == 0 else -0.22
    y = band + 0.09 * np.sin(phase * 2.3)
    return float(x), float(y)


def _save_frame(gl, pygame, size: tuple[int, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = gl.glReadPixels(0, 0, size[0], size[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    surface = pygame.image.fromstring(bytes(pixels), size, "RGB", True)
    pygame.image.save(surface, path)


def run_demo(
    *,
    filings_count: int = 4096,
    frames: int | None = None,
    screenshot: Path | None = None,
    hidden: bool = False,
) -> None:
    import pygame
    from OpenGL import GL as gl

    pygame.init()
    size = (900, 900)
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    if hidden:
        flags |= pygame.HIDDEN
    pygame.display.set_mode(size if not hidden else (900, 900), flags)
    pygame.display.set_caption(
        "LLVM Wooly Willy — drag LMB to comb, RMB to erase, R refill, S shake"
    )
    gl.glViewport(0, 0, *size)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    filings = Filings(filings_count)
    clock = pygame.time.Clock()
    running = True
    frame = 0
    magnet = (0.0, 0.55)
    active = frames is not None
    repel = False

    while running:
        dt = min(clock.tick(60) / 1000.0, 1.0 / 30.0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    filings.reset()
                elif event.key == pygame.K_s:
                    filings.shake()

        if frames is None:
            mx, my = pygame.mouse.get_pos()
            magnet = (2.0 * mx / size[0] - 1.0, 1.0 - 2.0 * my / size[1])
            buttons = pygame.mouse.get_pressed(3)
            active = bool(buttons[0] or buttons[2])
            repel = bool(buttons[2])
        else:
            magnet = _automatic_magnet(frame)
            active = True
            repel = False

        filings.step(magnet, active, repel, dt)
        draw_toy(gl, filings, magnet, active, repel)
        pygame.display.flip()
        frame += 1
        if frames is not None and frame >= frames:
            running = False

    if screenshot is not None:
        # Draw once after the final update so the requested image is exactly
        # the final state even on a hidden surface.
        draw_toy(gl, filings, magnet, active, repel)
        gl.glFinish()
        _save_frame(gl, pygame, size, screenshot)
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filings", type=int, default=4096)
    parser.add_argument(
        "--frames", type=int,
        help="run an automatic magnet path for this many frames, then exit",
    )
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument(
        "--hidden", action="store_true",
        help="create a hidden OpenGL surface (useful with --frames)",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="compile a small LLVM batch, print its contract, and exit",
    )
    args = parser.parse_args()
    if args.filings <= 0:
        parser.error("--filings must be positive")
    if args.filings % 8:
        parser.error("--filings must be divisible by eight")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.probe:
        kernel = compiled_filing_kernel(args.filings)
        print(f"entry={kernel.artifact.name}")
        print(f"arguments={kernel.argument_names}")
        print(f"outputs={tuple(kernel.output_ids)}")
        print(f"llvm_shortfalls={kernel.artifact.shortfalls}")
        return
    run_demo(
        filings_count=args.filings,
        frames=args.frames,
        screenshot=args.screenshot,
        hidden=args.hidden,
    )


if __name__ == "__main__":
    main()
