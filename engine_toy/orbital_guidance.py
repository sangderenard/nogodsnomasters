"""Outer managed-dt guidance participant for the machine craft.

The participant is intentionally a dt-system Piece rather than another
tracker loop.  It reads and writes the same registered spans as the nested
craft round.  Each outer attempt therefore recomputes the existing joint
thruster/gimbal/wheel allocation from the live state; an outer rejection
restores both the commands and the nested physical state.
"""
from __future__ import annotations

import math

import numpy as np

from orbital_actuation import AXES, allocate_wrench
from src.common.dt_system.time_contracts import BIND


GUIDANCE_COMMITMENT = "orbital_guidance_commitment"


def _one(value) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _rotation(values, prefix: str) -> np.ndarray:
    return np.asarray([[values[f"{prefix}_{a}{b}"] for b in AXES]
                       for a in AXES], dtype=float)


def _symmetric(values, prefix: str) -> np.ndarray:
    xx, xy, xz = (values[f"{prefix}_{name}"]
                  for name in ("xx", "xy", "xz"))
    yy, yz, zz = (values[f"{prefix}_{name}"]
                  for name in ("yy", "yz", "zz"))
    return np.asarray(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)), float)


class MachineGuidancePiece:
    """The machine's stateful guidance/allocation law at the outer dt.

    The host tracker publishes a desired attitude and force, not a timestep.
    This law recomputes the attitude PD and the existing constrained allocator
    for the attempted ``dt``.  Its commitment measure is the fraction of the
    locally predicted time to reversal of the torque request consumed by the
    attempt.  A value above one rejects and retries through the ordinary
    named-channel machinery.
    """

    entry = "machine-guidance"
    contract = BIND

    def __init__(self, craft, *, batch: int = 1):
        if int(batch) != 1:
            raise ValueError("machine guidance currently requires one craft lane")
        self.craft = craft
        self.batch = 1
        gimballed = [k for k, thruster in enumerate(craft.thrusters)
                     if thruster.gimballed]
        self._gimballed = tuple(gimballed)
        names = ["guidance_enabled", "guidance_time_s",
                 "guidance_force_start_s", "guidance_force_gate_rad",
                 "guidance_attitude_frequency", "guidance_attitude_damping"]
        names.extend(f"guidance_force_{axis}" for axis in AXES)
        names.extend(f"guidance_target_{a}{b}" for a in AXES for b in AXES)
        names.extend(f"centre_of_mass_{axis}" for axis in AXES)
        names.extend(f"angular_velocity_{axis}" for axis in AXES)
        names.extend(f"inertia_{pair}" for pair in
                     ("xx", "xy", "xz", "yy", "yz", "zz"))
        names.extend(f"attitude_{a}{b}" for a in AXES for b in AXES)
        names.extend(f"tank{t}_propellant" for t in range(len(craft.tanks)))
        for k in range(craft.thruster_count):
            names.extend((f"thruster{k}_throttle",
                          f"thruster{k}_throttle_state"))
            if k in self._gimballed:
                names.extend((f"thruster{k}_gimbal_a_command",
                              f"thruster{k}_gimbal_b_command",
                              f"thruster{k}_gimbal_a",
                              f"thruster{k}_gimbal_b"))
        for w in range(len(craft.wheels)):
            names.extend((f"wheel{w}_momentum",
                          f"wheel{w}_torque_command"))
        names.append("dt")
        self.argument_names = tuple(names)

        outputs = [f"thruster{k}_throttle_next"
                   for k in range(craft.thruster_count)]
        for k in self._gimballed:
            outputs.extend((f"thruster{k}_gimbal_a_command_next",
                            f"thruster{k}_gimbal_b_command_next"))
        outputs.extend(f"wheel{w}_torque_command_next"
                       for w in range(len(craft.wheels)))
        outputs.extend(("guidance_time_s_next",
                        "guidance_attitude_error_rad_next", "dt_limit",
                        GUIDANCE_COMMITMENT))
        self.output_names = tuple(outputs)

    def instantiate(self, columns, outputs=None):
        # All lifetime state is in the containing PieceState spans.
        return None

    def __call__(self, *columns):
        values = {name: _one(value)
                  for name, value in zip(self.argument_names, columns)}
        enabled = values["guidance_enabled"] > 0.5
        dt = values["dt"]
        time_s = values["guidance_time_s"]
        old_throttle = np.asarray(
            [values[f"thruster{k}_throttle"]
             for k in range(self.craft.thruster_count)], float)
        old_gimbal = np.zeros((self.craft.thruster_count, 2))
        for k in self._gimballed:
            old_gimbal[k] = (values[f"thruster{k}_gimbal_a_command"],
                             values[f"thruster{k}_gimbal_b_command"])
        old_wheel = np.asarray(
            [values[f"wheel{w}_torque_command"]
             for w in range(len(self.craft.wheels))], float)

        if not enabled:
            return self._outputs(old_throttle, old_gimbal,
                                 old_wheel,
                                 time_s + dt, 0.0, 1.0e300, 0.0)

        rotation = _rotation(values, "attitude")
        target = _rotation(values, "guidance_target")
        omega = np.asarray([values[f"angular_velocity_{a}"] for a in AXES])
        inertia = _symmetric(values, "inertia")
        spin_free = inertia.copy()
        for wheel in self.craft.wheels:
            axis = np.asarray(wheel.axis, dtype=float)
            spin_free -= wheel.rotor_inertia_kg_m2 * np.outer(axis, axis)
        skew_error = target.T @ rotation - rotation.T @ target
        error = 0.5 * np.asarray((skew_error[2, 1], skew_error[0, 2],
                                  skew_error[1, 0]))
        angle = math.acos(max(-1.0, min(1.0, 0.5 *
                          (float(np.trace(target.T @ rotation)) - 1.0))))
        wa = values["guidance_attitude_frequency"]
        za = values["guidance_attitude_damping"]
        gyro = np.cross(omega, spin_free @ omega)
        torque = (spin_free
                  @ (-wa * wa * error - 2.0 * za * wa * omega) + gyro)
        intended_force = np.asarray(
            [values[f"guidance_force_{a}"] for a in AXES])
        force = intended_force
        if (time_s < values["guidance_force_start_s"]
                or angle > values["guidance_force_gate_rad"]):
            force = np.zeros(3)

        throttle_state = np.asarray(
            [values[f"thruster{k}_throttle_state"]
             for k in range(self.craft.thruster_count)], float)
        gimbal_state = np.zeros((self.craft.thruster_count, 2))
        for k in self._gimballed:
            gimbal_state[k] = (values[f"thruster{k}_gimbal_a"],
                               values[f"thruster{k}_gimbal_b"])
        tanks = {tank.identity: values[f"tank{t}_propellant"]
                 for t, tank in enumerate(self.craft.tanks)}
        momenta = np.asarray([values[f"wheel{w}_momentum"]
                              for w in range(len(self.craft.wheels))])
        allocation = allocate_wrench(
            self.craft.thrusters, force, torque,
            centre_of_mass_m=np.asarray(
                [values[f"centre_of_mass_{a}"] for a in AXES]),
            attitude=rotation, throttle_state=throttle_state,
            gimbal_state=gimbal_state, tank_propellant_kg=tanks,
            force_direction_n=intended_force,
            round_s=dt, wheels=self.craft.wheels,
            wheel_momentum_n_m_s=momenta,
            angular_velocity_rad_s=omega)

        achieved = np.asarray(allocation.achieved.torque_n_m, float)
        try:
            # Allocation.achieved already includes the wheel-momentum
            # gyroscopic reaction.  The remaining Euler term is the body's
            # spin-free tensor, exactly as the nested craft law carries it.
            body_gyro = np.cross(omega, spin_free @ omega)
            alpha = np.linalg.solve(spin_free, achieved - body_gyro)
        except np.linalg.LinAlgError:
            alpha = np.zeros(3)
        # Local derivative of the same PD law.  The gyroscopic term is
        # re-evaluated at the next accepted interval; this predictor's job is
        # only to prevent a held command crossing its own reversal.
        torque_rate = (spin_free
                       @ (-wa * wa * omega - 2.0 * za * wa * alpha))
        size = float(np.linalg.norm(torque))
        horizon = 1.0e300
        if size > 0.0:
            axis = torque / size
            slope = float(axis @ torque_rate)
            if slope < 0.0:
                candidate = -float(axis @ torque) / slope
                if math.isfinite(candidate) and candidate > 0.0:
                    horizon = candidate
        until_force = values["guidance_force_start_s"] - time_s
        if until_force > 0.0:
            horizon = min(horizon, until_force)
        commitment = 0.0 if horizon >= 1.0e299 else dt / horizon
        return self._outputs(np.asarray(allocation.throttles, float),
                             np.asarray(allocation.gimbal_rad, float),
                             np.asarray(allocation.wheel_torque_n_m, float),
                             time_s + dt, angle, horizon, commitment)

    def _outputs(self, throttles, gimbals, wheel_torque, time_s, angle,
                 horizon, commitment):
        values = [*np.asarray(throttles, float)]
        for k in self._gimballed:
            values.extend(np.asarray(gimbals[k], float))
        values.extend(np.asarray(wheel_torque, float))
        values.extend((float(time_s), float(angle), float(horizon),
                       float(commitment)))
        return tuple(np.full(self.batch, value, dtype=float) for value in values)


__all__ = ["GUIDANCE_COMMITMENT", "MachineGuidancePiece"]
