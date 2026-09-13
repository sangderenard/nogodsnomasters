"""The firing solution as a service: it runs on its own clock, and the
mount always acts on whatever plan is currently published.

WHY THIS IS THE RIGHT SHAPE, and not just a threading trick.

Real fire control is already built this way, because the two jobs have
nothing to do with each other. The SERVO LOOP has to run fast and never
stop -- it is holding a mass against wind and recoil, and if it pauses
the barrel drifts. The BALLISTIC SOLUTION is slow, iterative, and only
needs to be roughly current. Tying them together means the mount stops
tracking every time the computer thinks, which is exactly backwards:
the moment you most need the mount moving is the moment the computer
has the most work to do.

So:

  THE MOUNT NEVER BLOCKS. Every tick it reads the latest published plan
  and slews toward it. If no new plan has arrived, it keeps acting on
  the old one. There is no state in which it waits.

  CONDITIONS FLOW IN CONTINUOUSLY. Target position and velocity, wind,
  where the muzzle actually is -- pushed in as they change, not polled.
  The solver picks up whatever is current when it starts its next pass.

  A PLAN CARRIES ITS OWN ASSUMPTIONS. This is the part that makes an
  asynchronous solution usable rather than merely late. The plan records
  the target state it was computed from and when. The consumer can then
  AGE it -- advance the aim by the target's own motion since the plan
  was made -- so a solution that is two hundred milliseconds old is
  still pointing at where the target is now, not where it was. Without
  that, asynchrony just buys you a lag.

  PUBLICATION IS A SINGLE REFERENCE SWAP. No lock is held while the
  mount reads, because reading a reference and replacing a reference are
  each atomic. The worker builds a whole new immutable plan and swaps it
  in; the reader either sees the old one or the new one, never a
  half-written one.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Conditions:
    """Everything the solver needs, as of one instant."""
    stamp: float
    target_identity: str = ""
    target_position: tuple = (0.0, 0.0, 0.0)
    target_velocity: tuple = (0.0, 0.0, 0.0)
    muzzle_position: tuple = (0.0, 0.0, 0.0)
    crosswind_m_s: float = 0.0
    valid: bool = False


@dataclass(frozen=True)
class Plan:
    """One published firing solution, with the assumptions behind it."""
    stamp: float
    conditions: Conditions
    aim_point: tuple
    bearing_deg: float
    elevation_deg: float
    range_m: float
    time_of_flight_s: float
    iterations: int
    converged: bool
    solve_seconds: float = 0.0

    def age(self, now: float) -> float:
        return max(0.0, now - self.stamp)

    def aim_now(self, now: float) -> tuple:
        """The aim point, advanced for the target's motion since this
        plan was computed.

        A solution is about where the target WILL be; if the plan is
        old, the whole prediction is shifted by however far the target
        has travelled since. Advancing it is not extrapolating the
        solution -- the ballistics are unchanged -- it is advancing the
        thing the solution was aimed at."""
        dt = self.age(now)
        if dt <= 0.0:
            return self.aim_point
        velocity = np.asarray(self.conditions.target_velocity, dtype=float)
        return tuple(np.asarray(self.aim_point, dtype=float) + velocity * dt)

    def bearing_elevation_now(self, now: float, muzzle) -> tuple:
        d = np.asarray(self.aim_now(now), dtype=float) - np.asarray(muzzle, dtype=float)
        horizontal = math.hypot(float(d[0]), float(d[2]))
        return (math.degrees(math.atan2(float(d[0]), float(d[2]))),
                math.degrees(math.atan2(float(d[1]), max(horizontal, 1e-12))))


@dataclass
class SolutionService:
    """A worker that solves continuously and publishes as it goes."""
    table: object
    cycle_s: float = 0.05             # the computer's own rate
    max_iterations: int = 8
    # published state -- swapped, never mutated
    plan: Plan | None = field(default=None, init=False)
    conditions: Conditions = field(
        default_factory=lambda: Conditions(stamp=time.monotonic()), init=False)
    solves: int = field(default=0, init=False)
    _thread: object = field(default=None, init=False, repr=False)
    _stop: object = field(default_factory=threading.Event, init=False, repr=False)

    # ---------------- the interface the sim uses ----------------
    def update(self, **kwargs) -> None:
        """Feed the current state in. Cheap, non-blocking, and safe to
        call every tick: it builds one immutable record and swaps it."""
        self.conditions = Conditions(stamp=time.monotonic(), **kwargs)

    def latest(self) -> Plan | None:
        """Whatever is published right now. Never waits."""
        return self.plan

    def start(self) -> "SolutionService":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="firing-solution",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # ---------------- the worker ----------------
    def _run(self) -> None:
        while not self._stop.is_set():
            began = time.monotonic()
            conditions = self.conditions          # one atomic read
            if conditions.valid:
                plan = self._solve(conditions)
                if plan is not None:
                    self.plan = plan              # one atomic publish
                    self.solves += 1
            else:
                self.plan = None
            elapsed = time.monotonic() - began
            self._stop.wait(max(0.0, self.cycle_s - elapsed))

    def solve_once(self) -> Plan | None:
        """Run one pass synchronously -- for tests, and for a caller that
        wants a plan before the first cycle has landed."""
        conditions = self.conditions
        if not conditions.valid:
            return None
        plan = self._solve(conditions)
        if plan is not None:
            self.plan = plan
            self.solves += 1
        return plan

    def _solve(self, conditions: Conditions) -> Plan | None:
        from fire_control import FiringSolver
        began = time.monotonic()
        solver = FiringSolver(
            table=self.table,
            target_position=np.asarray(conditions.target_position, dtype=float),
            target_velocity=np.asarray(conditions.target_velocity, dtype=float),
            muzzle_position=np.asarray(conditions.muzzle_position, dtype=float),
            crosswind_m_s=float(conditions.crosswind_m_s),
            max_iterations=self.max_iterations)
        estimate = solver.solve()
        aim = np.asarray(solver.aim_point, dtype=float)
        muzzle = np.asarray(conditions.muzzle_position, dtype=float)
        d = aim - muzzle
        horizontal = math.hypot(float(d[0]), float(d[2]))
        return Plan(
            stamp=time.monotonic(), conditions=conditions,
            aim_point=tuple(float(v) for v in aim),
            bearing_deg=math.degrees(math.atan2(float(d[0]), float(d[2]))),
            elevation_deg=math.degrees(math.atan2(float(d[1]), max(horizontal, 1e-12))),
            range_m=estimate.range_m, time_of_flight_s=estimate.time_of_flight_s,
            iterations=estimate.iteration, converged=estimate.converged,
            solve_seconds=time.monotonic() - began)
