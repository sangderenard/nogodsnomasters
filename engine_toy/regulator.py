"""Real regulators as installable line accessories -- generic enough to
sit on any real metered supply (a fuel line, an air line, an injection
line), not bespoke to one system. Two genuinely different real
regulator types, because real hardware comes in both kinds:

  ProgressiveRegulator  -- OPEN LOOP: doses output as a direct ramp of
                          some other real signal (a WMI kit's boost-
                          referenced controller: 0 output at/below an
                          onset point, full output at/above a ceiling,
                          linear between -- no feedback on what the
                          dosed quantity actually did). This is exactly
                          the math drivetrain_graph.py's own WMI
                          regulator branch already runs inline; kept
                          there since it's tied tightly to that one real
                          fluid-circuit step, but the same real law,
                          named once here for anything else that wants
                          the identical real mechanism.

  ClosedLoopRegulator   -- CLOSED LOOP: a real PI controller reading
                          back the thing it's actually trying to hold
                          (a turbine's N1, a piston engine's idle rpm,
                          an oil system's regulated pressure) and
                          adjusting its own output until the error
                          closes -- the same real control law already
                          used for this catalogue's piston-engine idle
                          governor (ecu.py) and the dyno's brake/
                          throttle target loops, generalized into one
                          reusable line accessory instead of being
                          written inline again for every new consumer.
                          gas_turbine.py's own real N1 fuel governor is
                          built directly from this -- install one of
                          these on the fuel line, hand it a target and
                          a feedback reading each tick, done.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class ProgressiveRegulator:
    """Open-loop ramp: output = rated * clamp((signal-onset)/(full-onset), 0, 1)."""
    rated_output: float
    onset: float
    full: float

    def duty(self, signal: float) -> float:
        span = max(1e-9, self.full - self.onset)
        return _clamp((signal - self.onset) / span, 0.0, 1.0)

    def output(self, signal: float) -> float:
        return self.rated_output * self.duty(signal)


@dataclass
class ClosedLoopRegulator:
    """A real installable PI regulator: `regulate(dt, target, feedback)`
    drives its output toward closing (target - feedback), with real
    conditional-integration anti-windup (the same technique ecu.py's
    idle governor uses) so the integral term doesn't wind up while the
    output sits pinned at either bound."""
    kp: float
    ki: float
    output_min: float
    output_max: float
    integral_clamp: float = 1e9

    _integral: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._integral = 0.0

    def regulate(self, dt: float, target: float, feedback: float, base: float = 0.0) -> float:
        error = target - feedback
        unsat = base + self.kp * error + self.ki * self._integral
        saturated_high = unsat >= self.output_max and error > 0.0
        saturated_low = unsat <= self.output_min and error < 0.0
        if not (saturated_high or saturated_low):
            self._integral = _clamp(self._integral + error * dt, -self.integral_clamp, self.integral_clamp)
        raw = base + self.kp * error + self.ki * self._integral
        return _clamp(raw, self.output_min, self.output_max)
