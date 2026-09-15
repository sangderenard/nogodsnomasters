"""SIMPLE CRAFT, EACH RUNNING A REAL ENGINE.

    python time_trials/craft.py --seconds 20

A craft is a small declared thing: a mass, a frontal area, a tire, the
gearbox it actually has, and a real `EngineCycleSim` out of the catalogue.
Nothing else. No 341-input body law, no vehicle configuration with 29
sections and 120 mass components, and no symbolic layer -- the engine is
the real one and the body is a few lines of arithmetic.

WHAT COMES FROM THE ENGINE. Crank torque and rpm, every tick, from the
cycle sim with its own cylinders, fuel, spark, thermal state and drivetrain
graph. The body turns that into wheel force through the gearbox, caps it
at what the tire can hold, and writes the road load back as
`brake_load_nm` so the engine answers to the hill it is climbing. That
feedback is the difference between an engine sim and a torque curve.

WHAT COMES FROM THE TIRE. Normal load and contact patch, from turing's
reduced contact law integrated over the tire's own four ring stations --
so a kart on 135 mm wheels and a taildragger on 260 mm ones carry
different loads because their sections differ, not because someone typed
two grip numbers.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ENGINE_TOY = Path(__file__).resolve().parent.parent
if str(_ENGINE_TOY) not in sys.path:
    sys.path.insert(0, str(_ENGINE_TOY))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import engines                                                       # noqa: E402
from engine_cycle_sim import EngineCycleSim, FIXED_PHYSICS_DT_S      # noqa: E402
from time_allocator import derive_dt_limit_s                         # noqa: E402

from tires import KART, AIRCRAFT, TireSection                        # noqa: E402

FLOOR_S = FIXED_PHYSICS_DT_S
AIR_DENSITY = 1.225
GRAVITY = 9.81


@dataclass(frozen=True)
class CraftSpec:
    """One craft, declared in one place.

    The gearbox is per craft because it has to be: a string trimmer is
    crank, clutch, head, with nothing to shift, and handing it a
    five-speed is inventing hardware.
    """

    name: str
    engine: str
    mass_kg: float
    drag_area_m2: float
    tire: TireSection
    ratios: tuple[float, ...]
    final_drive: float
    wheel_radius_m: float = 0.23
    rolling_coefficient: float = 0.013


FLEET = (
    CraftSpec("kart", "25cc-two-stroke-trimmer", 90.0, 0.25, KART,
              (1.0,), 1.9, wheel_radius_m=0.135, rolling_coefficient=0.016),
    CraftSpec("roadster", "mazda-b6ze-miata-1990", 520.0, 0.62, KART,
              (3.14, 1.89, 1.33, 1.00, 0.81), 4.10),
    CraftSpec("coupe", "vw-vr6-2800-12v", 780.0, 0.70, KART,
              (3.14, 1.89, 1.33, 1.00, 0.81), 3.94),
    CraftSpec("taildragger", "pw-r1340-wasp", 1430.0, 0.95, AIRCRAFT,
              (1.0,), 0.667, wheel_radius_m=0.26, rolling_coefficient=0.020),
)


@dataclass
class Craft:
    """A craft: its engine, its tire, and where it is."""

    spec: CraftSpec
    sim: EngineCycleSim
    dt_limit_s: float

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed_mps: float = 0.0
    gear_index: int = 1
    distance_m: float = 0.0
    world_s: float = 0.0

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0

    normal_load_n: float = 0.0
    patch_area_m2: float = 0.0
    drive_force_n: float = 0.0
    slip: float = 0.0

    def settle_tire(self) -> None:
        """Compression that carries a corner's share, then the load and
        patch the contact law gives there. Bisected: the law is a clamped
        quadrature and has no clean inverse."""
        target = self.spec.mass_kg * GRAVITY / 4.0
        low, high = 0.0, self.spec.tire.max_compression_m
        for _ in range(24):
            mid = 0.5 * (low + high)
            _area, force = self.spec.tire.contact(mid)
            low, high = (mid, high) if force < target else (low, mid)
        self.patch_area_m2, self.normal_load_n = self.spec.tire.contact(
            0.5 * (low + high))

    @property
    def ratio(self) -> float:
        if not 1 <= self.gear_index <= len(self.spec.ratios):
            return 0.0
        return self.spec.ratios[self.gear_index - 1] * self.spec.final_drive

    @property
    def grip_limit_n(self) -> float:
        return 1.35 * self.normal_load_n * 2.0

    def step(self, dt: float) -> None:
        """One tick: run the real engine, then move on what it produced."""
        self.sim.throttle = self.throttle
        self.sim.step(dt)

        torque = float(self.sim.state.torque_rms_nm or 0.0) * self.throttle
        ratio = self.ratio
        wanted = (torque * ratio * 0.92 / self.spec.wheel_radius_m
                  if ratio > 0.0 else 0.0)
        limit = self.grip_limit_n
        self.drive_force_n = max(-limit, min(wanted, limit))
        self.slip = (0.0 if abs(wanted) < 1e-9
                     else 1.0 - abs(self.drive_force_n) / abs(wanted))

        drag = 0.5 * AIR_DENSITY * self.spec.drag_area_m2 * self.speed_mps ** 2
        rolling = (self.spec.rolling_coefficient * self.spec.mass_kg * GRAVITY
                   if self.speed_mps > 0.01 else 0.0)
        braking = self.brake * limit
        net = self.drive_force_n - drag - rolling - braking
        self.speed_mps = max(0.0, self.speed_mps + net / self.spec.mass_kg * dt)

        # cornering, bounded by the same grip the tire is publishing
        wish = self.steer * 2.4 * min(1.0, self.speed_mps / 6.0)
        lateral = abs(wish) * max(self.speed_mps, 0.1) * self.spec.mass_kg
        if lateral > limit:
            wish *= limit / lateral
        self.heading += wish * dt
        self.x += math.cos(self.heading) * self.speed_mps * dt
        self.y += math.sin(self.heading) * self.speed_mps * dt
        self.distance_m += self.speed_mps * dt
        self.world_s += dt

        # the road, reflected back to the crank
        if ratio > 0.0:
            self.sim.brake_load_nm = ((drag + rolling + braking)
                                      * self.spec.wheel_radius_m / (ratio * 0.92))
        self._shift()

    def _shift(self) -> None:
        redline = float(getattr(self.sim.engine, "redline_rpm", 6500) or 6500)
        rpm = float(self.sim.rpm or 0.0)
        if rpm > redline * 0.92 and self.gear_index < len(self.spec.ratios):
            self.gear_index += 1
            self.sim.gear_index = self.gear_index
        elif rpm < redline * 0.35 and self.gear_index > 1:
            self.gear_index -= 1
            self.sim.gear_index = self.gear_index


def build(spec: CraftSpec, lane: int) -> Craft:
    engine = engines.get(spec.engine)
    sim = EngineCycleSim(engine=engine)
    sim.start()
    sim.gear_index = 1
    craft = Craft(spec=spec, sim=sim,
                  dt_limit_s=derive_dt_limit_s(engine) or 1.0 / 240.0,
                  gear_index=min(3, len(spec.ratios)))
    craft.settle_tire()
    craft.y = -3.0 * lane
    craft.speed_mps = 12.0
    return craft


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="wall-clock seconds to run for")
    parser.add_argument("--throttle", type=float, default=0.85)
    args = parser.parse_args(argv)

    fleet = [build(spec, lane) for lane, spec in enumerate(FLEET)]
    for craft in fleet:
        craft.throttle = args.throttle
        craft.steer = 0.05
    print(f"{len(fleet)} craft, each on its own engine\n")
    for craft in fleet:
        print(f"   {craft.spec.name:<13}{craft.spec.engine:<26}"
              f"{len(craft.spec.ratios):>2} gear(s)"
              f"  dt_limit {craft.dt_limit_s * 1e6:8.1f} us"
              f"  patch {craft.patch_area_m2 * 1e4:6.1f} cm2"
              f" at {craft.normal_load_n:7.1f} N")
    print()

    started = time.perf_counter()
    deadline = started + args.seconds
    frames = 0
    while time.perf_counter() < deadline:
        for craft in fleet:
            craft.step(FLOOR_S)
        frames += 1
    wall = time.perf_counter() - started

    world_s = frames * FLOOR_S
    print(f"   {'craft':<13}{'rpm':>8}{'gear':>6}{'m/s':>8}{'metres':>10}"
          f"{'force_n':>10}{'slip':>7}")
    for craft in fleet:
        print(f"   {craft.spec.name:<13}{float(craft.sim.rpm or 0.0):>8.0f}"
              f"{craft.gear_index:>6}{craft.speed_mps:>8.2f}"
              f"{craft.distance_m:>10.1f}{craft.drive_force_n:>10.1f}"
              f"{craft.slip:>7.3f}")
    print(f"\n{frames} ticks, world {world_s:.3f} s in {wall:.2f} s wall "
          f"({world_s / wall * 100:.1f}% of real time)")


if __name__ == "__main__":
    main()
