"""Real centrifugal governors, as actual mass-spring-centrifugal
dynamical systems -- not a lookup against a target rpm (the existing
catalogue's `governor_mode="hit_and_miss"` / `governor_target_rpm`
pair is exactly that abstraction; this is what real hardware sits
underneath it).

CAPTIVE BALL GOVERNOR -- a real, distinct mechanism from the classic
Watt/flyball governor (pivoting weighted arms). Here one or more balls
are CAPTURED in a straight radial track machined into a disc keyed to
a shaft driven off the engine's own output (real period hardware on
early stationary gas/atmospheric engines, including engines of
exactly the Otto-Langen type this was built for -- a small pulley or
gear off the flywheel spun a governor shaft at a real, disclosed
ratio to the engine speed). Two real forces act on each ball along
its track:

  centrifugal force    F_c = m * omega_governor^2 * r   (outward)
  return spring        F_s = preload + k * (r - r_min)  (inward)

genuinely integrated as a one-DOF mass-spring system under those
forces (plus real sliding friction in the track) -- NOT solved as an
instantaneous force balance. That distinction matters: a real captive
ball governor has real inertia, a real natural frequency, and can
genuinely hunt/overshoot around its setpoint before settling, the
same way any other underdamped mechanical governor can. Reducing it
to instantaneous equilibrium would throw that away.

The ball's own radial position drives a real trip lever with
hysteresis (a rising trip_radius_m and a lower release_radius_m,
same physical reason a thermostat or pressure switch needs a band
instead of one point -- a single trip radius sitting right at the
ball's own equilibrium would chatter every cycle). Once tripped, the
governor holds the exhaust valve open / blocks the next ignition --
a real hit-and-miss "miss", driven by the ball's own real position
rather than a raw rpm comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GovernorSpec:
    """The real, DECLARED build for one captive-ball governor --
    separate from CaptiveBallGovernor itself, which carries live
    runtime state (ball position/velocity, tripped). `build()`
    constructs a fresh runtime object from this spec."""
    ball_mass_kg: float
    r_min_m: float
    r_max_m: float
    spring_rate_n_per_m: float
    spring_preload_n: float
    trip_radius_m: float
    release_radius_m: float
    drive_ratio: float = 1.0
    friction_n: float = 0.3

    def build(self) -> "CaptiveBallGovernor":
        return CaptiveBallGovernor(
            ball_mass_kg=self.ball_mass_kg, r_min_m=self.r_min_m, r_max_m=self.r_max_m,
            spring_rate_n_per_m=self.spring_rate_n_per_m, spring_preload_n=self.spring_preload_n,
            trip_radius_m=self.trip_radius_m, release_radius_m=self.release_radius_m,
            drive_ratio=self.drive_ratio, friction_n=self.friction_n)


@dataclass
class CaptiveBallGovernor:
    ball_mass_kg: float
    r_min_m: float
    r_max_m: float
    spring_rate_n_per_m: float
    spring_preload_n: float
    trip_radius_m: float
    release_radius_m: float
    drive_ratio: float = 1.0     # governor shaft omega / engine shaft omega -- a real belt/gear step
    friction_n: float = 0.3      # real sliding friction in the ball's track/guide

    r_m: float = field(init=False)
    v_m_s: float = field(init=False)
    tripped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.r_m = self.r_min_m
        self.v_m_s = 0.0

    def reset(self) -> None:
        self.r_m = self.r_min_m
        self.v_m_s = 0.0
        self.tripped = False

    def step(self, dt: float, engine_shaft_omega: float) -> bool:
        """Advances the ball one tick under its own real dynamics;
        returns whether the governor is presently tripped (miss this
        cycle)."""
        omega_gov = max(0.0, engine_shaft_omega) * self.drive_ratio
        centrifugal_n = self.ball_mass_kg * omega_gov * omega_gov * self.r_m
        spring_n = self.spring_preload_n + self.spring_rate_n_per_m * max(0.0, self.r_m - self.r_min_m)
        friction_n = -math.copysign(self.friction_n, self.v_m_s) if self.v_m_s != 0.0 else 0.0
        net_n = centrifugal_n - spring_n + friction_n
        self.v_m_s += net_n / self.ball_mass_kg * dt
        self.r_m += self.v_m_s * dt

        # real hard stops at either end of the track
        if self.r_m <= self.r_min_m:
            self.r_m = self.r_min_m
            self.v_m_s = max(0.0, self.v_m_s)
        elif self.r_m >= self.r_max_m:
            self.r_m = self.r_max_m
            self.v_m_s = min(0.0, self.v_m_s)

        if not self.tripped and self.r_m >= self.trip_radius_m:
            self.tripped = True
        elif self.tripped and self.r_m <= self.release_radius_m:
            self.tripped = False
        return self.tripped
