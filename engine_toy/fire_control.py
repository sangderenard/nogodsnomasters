"""Fire control: a baked solution, a converging solve, and a turret that
moves while it thinks.

FOUR THINGS LIVE HERE, and they are the same thing at four scales.

1. THE BAKE. Tracing a trajectory step by step to answer one question
   about one range is absurd when the question gets asked every frame.
   `SolutionTable` traces each cartridge ONCE across a grid of ranges,
   stores what it found, and answers afterwards by interpolation. The
   physics is identical -- it is the same compiled trajectory kernel --
   but a solution costs a lookup instead of four hundred thousand
   integration steps.

   Wind is handled separately and cheaply because it deserves to be:
   drift is very nearly LINEAR in crosswind speed at a fixed range, so
   one extra trace at a reference wind gives a drift-per-metre-per-
   second coefficient that covers every wind. That is not a
   simplification of convenience -- it falls out of drag being a
   function of the relative velocity, which a modest crosswind perturbs
   almost linearly.

2. THE SOLVE IS ITERATIVE, AND THAT IS WHY IT TAKES TIME. A firing
   solution is a fixed point: where the target will be depends on the
   flight time, and the flight time depends on how far away that is.
   You cannot write it down, you iterate it -- guess the range, get a
   time, move the target by that time, get a new range. Each pass is
   closer. So the "time penalty for perfect aim" is not a made-up delay
   on a progress bar; it is a real iteration count, and showing the
   successive iterates IS showing the computer working.

3. LINE OF SIGHT IS A RAY AGAINST THE WORLD. A hostile behind a wall is
   not a target. The same `RayMesh` that resolves a shot resolves
   whether anything is in the way, so occlusion costs nothing extra and
   cannot disagree with what a bullet would do.

4. THE TURRET SLEWS AT ITS OWN RATE. It does not snap to the answer. The
   traverse and elevation actuators have real flow-limited speeds
   (actuators.py), so the mount chases the moving solution and arrives
   when the hydraulics allow, not when the arithmetic finishes.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

BAKE_DIR = Path(__file__).resolve().parent / "baked"
REFERENCE_CROSSWIND_M_S = 10.0       # the wind the drift coefficient is measured at


# =====================================================================
#  1. THE BAKE
# =====================================================================
@dataclass
class SolutionTable:
    """A cartridge's ballistics, solved once and kept."""
    calibre: str
    zero_range_m: float
    ranges_m: list = field(default_factory=list)
    drop_m: list = field(default_factory=list)          # below the line of sight
    drift_per_m_s: list = field(default_factory=list)   # metres of drift per m/s of crosswind
    time_s: list = field(default_factory=list)
    speed_m_s: list = field(default_factory=list)
    energy_j: list = field(default_factory=list)
    zero_elevation_rad: float = 0.0

    # ---------------- baking ----------------
    @classmethod
    def bake(cls, calibre: str, *, zero_range_m: float = 100.0,
             ranges=(50, 100, 150, 200, 300, 400, 500, 600, 800, 1000),
             dt: float = 5e-4) -> "SolutionTable":
        from scope import scoped
        s = scoped(calibre, zero_range_m=zero_range_m)
        table = cls(calibre=calibre, zero_range_m=zero_range_m,
                    zero_elevation_rad=s.zero_elevation_rad)
        for r in ranges:
            drop, _, speed, energy, tof = s._trace(s.zero_elevation_rad, 0.0, float(r), (0, 0, 0))
            _, drift, _, _, _ = s._trace(s.zero_elevation_rad, 0.0, float(r),
                                         (REFERENCE_CROSSWIND_M_S, 0.0, 0.0))
            table.ranges_m.append(float(r))
            table.drop_m.append(float(drop))
            table.drift_per_m_s.append(float(drift) / REFERENCE_CROSSWIND_M_S)
            table.time_s.append(float(tof))
            table.speed_m_s.append(float(speed))
            table.energy_j.append(float(energy))
        return table

    # ---------------- lookup ----------------
    def _interp(self, values, range_m: float) -> float:
        return float(np.interp(float(range_m), self.ranges_m, values))

    def drop_at(self, range_m: float) -> float:
        return self._interp(self.drop_m, range_m)

    def drift_at(self, range_m: float, crosswind_m_s: float) -> float:
        return self._interp(self.drift_per_m_s, range_m) * float(crosswind_m_s)

    def time_at(self, range_m: float) -> float:
        return self._interp(self.time_s, range_m)

    def speed_at(self, range_m: float) -> float:
        return self._interp(self.speed_m_s, range_m)

    def energy_at(self, range_m: float) -> float:
        return self._interp(self.energy_j, range_m)

    # ---------------- persistence ----------------
    def path(self) -> Path:
        key = self.calibre.replace(" ", "-").replace(".", "")
        return BAKE_DIR / f"solution-{key}-zero{self.zero_range_m:.0f}.json"

    def save(self) -> Path:
        BAKE_DIR.mkdir(parents=True, exist_ok=True)
        p = self.path()
        p.write_text(json.dumps(self.__dict__, indent=1), encoding="utf-8")
        return p

    @classmethod
    def load_or_bake(cls, calibre: str, *, zero_range_m: float = 100.0) -> "SolutionTable":
        probe = cls(calibre=calibre, zero_range_m=zero_range_m)
        p = probe.path()
        if p.is_file():
            return cls(**json.loads(p.read_text(encoding="utf-8")))
        table = cls.bake(calibre, zero_range_m=zero_range_m)
        table.save()
        return table


# =====================================================================
#  2. THE CONVERGING SOLVE
# =====================================================================
@dataclass
class SolutionEstimate:
    """One iterate of the solve -- what the computer believes so far."""
    iteration: int
    range_m: float
    elevation_mrad: float
    windage_mrad: float
    time_of_flight_s: float
    aim_point: tuple
    residual_m: float             # how far this moved from the last iterate
    converged: bool = False


@dataclass
class FiringSolver:
    """The fixed-point solve, one iteration at a time.

    Held as an object rather than a function precisely so the caller can
    watch it: `step()` advances one iteration and hands back what the
    computer believes now, which is what both the turret and the second
    reticle follow."""
    table: SolutionTable
    target_position: np.ndarray
    target_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    muzzle_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    crosswind_m_s: float = 0.0
    tolerance_m: float = 0.02
    max_iterations: int = 8
    # live
    iteration: int = 0
    aim_point: np.ndarray = None
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        # the naive first guess is the target itself: straight at it,
        # no drop, no lead. Every iteration after this is the computer
        # correcting that, which is exactly what the moving reticle shows.
        self.aim_point = np.asarray(self.target_position, dtype=float).copy()

    @property
    def converged(self) -> bool:
        return bool(self.history and self.history[-1].converged)

    def step(self) -> SolutionEstimate:
        """One pass of the fixed point."""
        previous = self.aim_point.copy()
        # where the target will be after the CURRENT estimate of flight
        straight = np.linalg.norm(self.aim_point - self.muzzle_position)
        tof = self.table.time_at(straight)
        led = np.asarray(self.target_position, dtype=float) + np.asarray(
            self.target_velocity, dtype=float) * tof
        # the range to THAT point, which is what the ballistics answer for
        offset = led - self.muzzle_position
        range_m = float(np.linalg.norm(offset))
        unit = offset / max(range_m, 1e-9)
        drop = self.table.drop_at(range_m)
        drift = self.table.drift_at(range_m, self.crosswind_m_s)
        elevation_mrad = drop / max(range_m, 1e-9) * 1000.0
        windage_mrad = -drift / max(range_m, 1e-9) * 1000.0

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(unit, world_up)
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, unit)
        total_elevation = self.table.zero_elevation_rad + elevation_mrad / 1000.0
        self.aim_point = (led + up * (total_elevation * range_m)
                          + right * (windage_mrad / 1000.0 * range_m))

        self.iteration += 1
        residual = float(np.linalg.norm(self.aim_point - previous))
        est = SolutionEstimate(
            iteration=self.iteration, range_m=range_m,
            elevation_mrad=elevation_mrad, windage_mrad=windage_mrad,
            time_of_flight_s=tof, aim_point=tuple(float(v) for v in self.aim_point),
            residual_m=residual,
            converged=(residual <= self.tolerance_m or self.iteration >= self.max_iterations))
        self.history.append(est)
        return est

    def solve(self) -> SolutionEstimate:
        """Run it out, for callers that do not want to watch."""
        est = self.step()
        while not est.converged:
            est = self.step()
        return est


# =====================================================================
#  3. LINE OF SIGHT
# =====================================================================
@dataclass
class Hostile:
    """Anything the world has labelled as a target."""
    identity: str
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    label: str = "hostile"
    # A TARGET NEEDS A BODY. Without one a shot has nothing to resolve
    # against, which is precisely why nothing was ever destroyed: the
    # round traced against the SHOOTER's mesh, found the target absent,
    # and reported no effect. `targets.TargetBody` gives it a real
    # graph, real plating and declared critical parts.
    body: object = None

    @property
    def destroyed(self) -> bool:
        return bool(getattr(self.body, "destroyed", False))

    def give_body(self, calibre: str, kind: str = "light-armour"):
        from targets import TargetBody
        self.body = TargetBody(identity=self.identity, kind=kind)
        self.body.arm(calibre)
        return self.body


def line_of_sight(ray_mesh, muzzle, hostile_position, *, ignore_within_m: float = 0.6):
    """Is anything in the way?

    The same mesh a shot resolves against, so occlusion cannot disagree
    with what a bullet would actually do. Hits within `ignore_within_m`
    of the muzzle are the mount's own structure and are not obstacles --
    a turret can always see past its own barrel."""
    from engine_rays import Ray
    muzzle = np.asarray(muzzle, dtype=float)
    target = np.asarray(hostile_position, dtype=float)
    span = target - muzzle
    distance = float(np.linalg.norm(span))
    if distance < 1e-6:
        return True, None
    hit = ray_mesh.pick(Ray.from_points(muzzle, target))
    if hit is None:
        return True, None
    reach = float(np.linalg.norm(np.asarray(hit.point, dtype=float) - muzzle))
    if reach <= ignore_within_m or reach >= distance - 1e-3:
        return True, None
    return False, hit.part


# =====================================================================
#  4. THE TURRET THAT MOVES WHILE IT THINKS
# =====================================================================
@dataclass
class Reticle2D:
    """One crosshair, in the angles a gunner actually reads."""
    name: str
    bearing_deg: float
    elevation_deg: float
    colour: str

    def to_data(self) -> dict:
        return {"name": self.name, "bearing_deg": round(self.bearing_deg, 3),
                "elevation_deg": round(self.elevation_deg, 3), "colour": self.colour}


def _bearing_elevation(vector) -> tuple:
    """A direction as (bearing about the vertical, elevation above the
    horizontal), in degrees."""
    v = np.asarray(vector, dtype=float)
    horizontal = math.hypot(float(v[0]), float(v[2]))
    return (math.degrees(math.atan2(float(v[0]), float(v[2]))),
            math.degrees(math.atan2(float(v[1]), max(horizontal, 1e-12))))


def _angle_difference(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _approach(current: float, goal: float, step: float, *, wrap: bool = False) -> float:
    delta = _angle_difference(goal, current) if wrap else (goal - current)
    if abs(delta) <= step:
        return current + delta
    return current + math.copysign(step, delta)


@dataclass
class TurretFireControl:
    """A turret that acquires, solves, and slews -- all at real rates.

    The time between seeing something and being able to hit it is made
    of two honest parts, kept separate here because they have different
    causes and different cures:

      COMPUTER TIME. The solve is a fixed-point iteration and each pass
      costs a cycle. A faster computer shortens this; a faster turret
      does not.

      SLEW TIME. The mount has to physically get there, at whatever rate
      its hydraulics allow under the load it carries. A faster computer
      does nothing at all for this.

    The two reticles are those two things made visible: the SOLUTION
    reticle is where the computer currently believes the gun should
    point, and it settles as the iteration converges; the AIM reticle is
    where the barrel actually is, and it chases the first at the mount's
    own speed. The gap between them IS the time penalty.
    """
    table: SolutionTable
    muzzle_position: np.ndarray
    ray_mesh: object = None
    traverse_rate_deg_s: float = 18.0
    elevation_rate_deg_s: float = 9.0
    computer_cycle_s: float = 0.05
    crosswind_m_s: float = 0.0
    # live state
    bearing_deg: float = 0.0
    elevation_deg: float = 0.0
    target: object = None
    solver: object = None
    _cycle_accumulator: float = 0.0
    tracking_error_deg: float = 180.0
    on_target: bool = False
    service: object = None
    plan_age_s: float = 0.0
    plan_iterations: int = 0
    plan_converged: bool = False
    log: list = field(default_factory=list)

    def acquire(self, hostiles):
        """Pick the nearest hostile this turret can actually see.

        Labelled hostile AND visible: something behind a wall is not a
        target, and the wall is found with the same ray a shot takes."""
        best, best_range = None, float("inf")
        for h in hostiles:
            if h.label != "hostile":
                continue
            offset = np.asarray(h.position, dtype=float) - self.muzzle_position
            distance = float(np.linalg.norm(offset))
            if self.ray_mesh is not None:
                clear, blocker = line_of_sight(self.ray_mesh, self.muzzle_position, h.position)
                if not clear:
                    self.log.append(f"{h.identity}: no line of sight, {blocker} is in the way")
                    continue
            if distance < best_range:
                best, best_range = h, distance
        if best is not self.target:
            self.target = best
            self.solver = None
            if best is not None:
                self.log.append(f"acquired {best.identity} at {best_range:.0f} m")
        return best

    def attach_service(self, service) -> "TurretFireControl":
        """Run the solution asynchronously.

        With a service attached the mount stops solving inline and
        starts acting on whatever plan is published, which is the whole
        point: it never pauses tracking to think."""
        self.service = service
        service.start()
        return self

    def step(self, dt: float) -> dict:
        """Advance the mount by one tick. Never waits for a solution."""
        if self.target is None:
            if self.service is not None:
                self.service.update(valid=False)
            return self.reticles()

        if self.service is not None:
            # FEED THE CONDITIONS IN, every tick, and take whatever plan
            # is currently published. If the solver is mid-pass we act
            # on the previous plan rather than stalling, and we AGE it
            # so it points where the target is now rather than where it
            # was when the plan was made.
            self.service.update(
                target_identity=self.target.identity,
                target_position=tuple(float(v) for v in self.target.position),
                target_velocity=tuple(float(v) for v in self.target.velocity),
                muzzle_position=tuple(float(v) for v in self.muzzle_position),
                crosswind_m_s=float(self.crosswind_m_s), valid=True)
            plan = self.service.latest()
            if plan is None:
                plan = self.service.solve_once()     # the very first tick
            if plan is None:
                return self.reticles()
            import time as _time
            now = _time.monotonic()
            want_bearing, want_elevation = plan.bearing_elevation_now(now, self.muzzle_position)
            self.plan_age_s = plan.age(now)
            self.plan_iterations = plan.iterations
            self.plan_converged = plan.converged
        else:
            if self.solver is None:
                self.solver = FiringSolver(
                    table=self.table,
                    target_position=np.asarray(self.target.position, dtype=float),
                    target_velocity=np.asarray(self.target.velocity, dtype=float),
                    muzzle_position=self.muzzle_position, crosswind_m_s=self.crosswind_m_s)
            else:
                self.solver.target_position = np.asarray(self.target.position, dtype=float)
            # COMPUTER TIME: one iteration per cycle, no faster
            self._cycle_accumulator += dt
            while self._cycle_accumulator >= self.computer_cycle_s:
                self._cycle_accumulator -= self.computer_cycle_s
                if not self.solver.converged:
                    self.solver.step()
            solution_dir = np.asarray(self.solver.aim_point, dtype=float) - self.muzzle_position
            want_bearing, want_elevation = _bearing_elevation(solution_dir)

        # SLEW TIME: the mount moves at its own rate toward that
        self.bearing_deg = _approach(self.bearing_deg, want_bearing,
                                     self.traverse_rate_deg_s * dt, wrap=True)
        self.elevation_deg = _approach(self.elevation_deg, want_elevation,
                                       self.elevation_rate_deg_s * dt)
        self.tracking_error_deg = math.hypot(
            _angle_difference(want_bearing, self.bearing_deg),
            want_elevation - self.elevation_deg)
        converged = self.plan_converged if self.service is not None else bool(
            self.solver and self.solver.converged)
        self.on_target = bool(self.tracking_error_deg < 0.05 and converged)
        return self.reticles()

    def reticles(self) -> dict:
        """What a gunner sees: where the gun is, and where it should be."""
        solution = None
        iterations = 0
        converged = False
        if self.service is not None:
            plan = self.service.latest()
            if plan is not None:
                import time as _time
                b, e = plan.bearing_elevation_now(_time.monotonic(), self.muzzle_position)
                solution = Reticle2D("solution", b, e, "amber")
                iterations, converged = plan.iterations, plan.converged
        elif self.solver is not None:
            d = np.asarray(self.solver.aim_point, dtype=float) - self.muzzle_position
            b, e = _bearing_elevation(d)
            solution = Reticle2D("solution", b, e, "amber")
            iterations, converged = self.solver.iteration, self.solver.converged
        return {
            "aim": Reticle2D("aim", self.bearing_deg, self.elevation_deg, "white").to_data(),
            "solution": solution.to_data() if solution else None,
            "tracking_error_deg": round(self.tracking_error_deg, 4),
            "iterations": iterations,
            "converged": bool(converged),
            "plan_age_s": round(self.plan_age_s, 4),
            "on_target": self.on_target,
            "target": None if self.target is None else self.target.identity,
        }
