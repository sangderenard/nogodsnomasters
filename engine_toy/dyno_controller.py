"""The test cell's controller -- NOT a vehicle box.

The brake-target and throttle-target PI loops used to sit inline in
EngineCycleSim next to the engine's own physics. They're real controls,
but they belong to the dyno cell: a load-absorber controller driving
the brake to hold an rpm point, and a throttle servo governor doing the
same on the pedal. Neither exists on the vehicle, so they don't go in
the ECU -- they get their own box here, moved behavior-identical, and
the sim delegates. Either loop, neither, or both can be active at once,
the way a real cell runs a throttle servo and an absorber together.
"""
from __future__ import annotations

from engines import Engine

BRAKE_TARGET_KP_FRAC = 1.0 / 0.30   # brake reaches peak torque over ~30% of redline of rpm error
BRAKE_TARGET_KI_FRAC = 0.5           # integral gain as a fraction of the proportional gain
BRAKE_TARGET_INTEGRAL_CLAMP_FRAC = 2.0
THROTTLE_TARGET_KP_FRAC = 1.0 / 0.30  # throttle swings its full 0..1 range over ~30% of redline of rpm error
THROTTLE_TARGET_KI_FRAC = 0.5
THROTTLE_TARGET_INTEGRAL_CLAMP_FRAC = 2.0


class DynoController:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._brake_integral = 0.0
        self._brake_was_active = False
        self._throttle_integral = 0.0
        self._throttle_was_active = False

    def brake_load_nm(self, engine: Engine, target_rpm: float | None, rpm: float,
                      current_brake_load_nm: float, dt: float) -> float:
        """A real dyno-style PI loop: when target_rpm is set, drives the
        absorber load so rpm converges there and holds -- idle-protect is
        just this with target=idle_rpm. Releasing the target has to
        actually release the real absorber (and the wound-up integral),
        not leave the last command silently fighting whatever comes next."""
        if target_rpm is None:
            if self._brake_was_active:
                self._brake_integral = 0.0
                self._brake_was_active = False
                return 0.0
            return current_brake_load_nm
        self._brake_was_active = True
        kp = engine.peak_torque_nm * BRAKE_TARGET_KP_FRAC / max(engine.redline_rpm, 1.0)
        ki = kp * BRAKE_TARGET_KI_FRAC
        error = rpm - target_rpm   # positive = too fast, need more brake
        integral_clamp = engine.peak_torque_nm * BRAKE_TARGET_INTEGRAL_CLAMP_FRAC / max(ki, 1e-9)
        self._brake_integral = max(-integral_clamp, min(integral_clamp, self._brake_integral + error * dt))
        raw = kp * error + ki * self._brake_integral
        return max(0.0, min(engine.peak_torque_nm * 3.0, raw))

    def throttle(self, engine: Engine, target_rpm: float | None, rpm: float,
                 current_throttle: float, dt: float) -> float:
        """The throttle servo governor: drives the pedal itself to hold an
        rpm point on an unloaded (or lightly loaded) engine, independent
        of the absorber loop. Same real-release rule."""
        if target_rpm is None:
            if self._throttle_was_active:
                self._throttle_integral = 0.0
                self._throttle_was_active = False
                return 0.0
            return current_throttle
        self._throttle_was_active = True
        kp = THROTTLE_TARGET_KP_FRAC / max(engine.redline_rpm, 1.0)
        ki = kp * THROTTLE_TARGET_KI_FRAC
        error = target_rpm - rpm   # positive = too slow, need more throttle
        integral_clamp = THROTTLE_TARGET_INTEGRAL_CLAMP_FRAC / max(ki, 1e-9)
        self._throttle_integral = max(-integral_clamp, min(integral_clamp, self._throttle_integral + error * dt))
        raw = kp * error + ki * self._throttle_integral
        return max(0.0, min(1.0, raw))
