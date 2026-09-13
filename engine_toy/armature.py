"""Armatures: real linkages, real actuators, and a solver that finds the
activations to reach a pose.

An armature here is not a skeleton with angles. It is the same thing
every other structure in this project is -- nodes joined by members --
with the single addition that SOME MEMBERS CAN CHANGE THEIR LENGTH.
That is what an actuator is, mechanically: a member with a commanded
rest length. Everything else follows from it.

TWO SOLVES, NESTED, AND THEY ARE DIFFERENT PROBLEMS

  FORWARD. Given what every actuator is set to, where does the
  structure end up? This is not a chain of transforms, because a real
  armature is not a chain -- a tripod mount, a parallel platform or a
  triangulated arm all have loops in them, and a loop has no root to
  walk outward from. So the forward answer is a SOLVE: find the node
  positions that satisfy every member's length at once, holding the
  anchored nodes still. Gauss-Newton on the length residuals, which is
  the same least-squares a surveyor uses on a truss.

  INVERSE. What must the actuators be set to for the end effector to
  reach a target? Also a solve, one level up: the pose error as a
  function of the activation vector, differentiated numerically, and
  stepped with damped least squares. The damping is not a fudge -- it
  is what keeps the step finite when the armature is near a singularity,
  which is exactly where a naive inverse blows up and throws the mount
  across the room.

WHY DAMPED LEAST SQUARES AND NOT AN ANALYTIC INVERSE

An analytic inverse exists for a few famous geometries and for nothing
else. The moment an armature has a redundant actuator, an awkward
attachment point or a joint limit, the closed form is gone. Damped
least squares does not care: it handles redundancy by taking the
smallest activation change that helps, handles limits by clamping and
re-solving, and degrades into "get as close as you can" rather than
failing, which is what a real mount does when you ask for something it
cannot reach.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# =====================================================================
#  ACTUATOR TECHNOLOGIES
# =====================================================================
@dataclass
class ElectricLinearActuator:
    """A motor turning a screw.

    The screw is the whole story. Force is torque geared up by the
    lead: a small motor makes enormous thrust through a fine pitch, and
    pays for it in speed, because the same lead that multiplies force
    divides velocity. That trade is fixed by the hardware and no amount
    of control changes it.

    BALL versus LEAD SCREW is the other real choice, and it decides
    something a specification sheet often buries: a ball screw is
    efficient (around 0.9) and BACKDRIVES, so it needs a brake to hold
    a load; an ACME lead screw is inefficient (around 0.35) and SELF-
    LOCKS, so it holds with no power at all. For a gun mount that spends
    most of its life pointed somewhere and still, that is not a detail.
    """
    identity: str = "electric-ram"
    stroke_m: float = 0.200
    lead_m: float = 0.005                 # travel per revolution
    motor_torque_nm: float = 0.85
    motor_rated_rpm: float = 3000.0
    screw_kind: str = "ball"              # "ball" | "acme"
    gear_ratio: float = 1.0
    # live
    position_m: float = 0.0

    @property
    def efficiency(self) -> float:
        return 0.90 if self.screw_kind == "ball" else 0.35

    @property
    def self_locking(self) -> bool:
        """An ACME screw at a normal pitch will not backdrive. A ball
        screw always will."""
        return self.screw_kind != "ball"

    @property
    def max_force_n(self) -> float:
        return (self.motor_torque_nm * self.gear_ratio * 2.0 * math.pi
                / max(self.lead_m, 1e-9) * self.efficiency)

    @property
    def max_speed_m_s(self) -> float:
        return self.motor_rated_rpm / 60.0 / max(self.gear_ratio, 1e-9) * self.lead_m

    def power_for(self, force_n: float, speed_m_s: float) -> float:
        """Electrical input for this working point, including the screw
        and a motor that is not perfect either."""
        mechanical = abs(force_n * speed_m_s)
        return mechanical / max(self.efficiency * 0.85, 1e-9)

    def holding_power_w(self, force_n: float) -> float:
        """What it costs to just stay there. A self-locking screw costs
        NOTHING, which is the reason to accept its poor efficiency."""
        if self.self_locking:
            return 0.0
        # a ball screw has to be held by motor current against the load
        torque = abs(force_n) * self.lead_m / (2.0 * math.pi * self.efficiency)
        return (torque / max(self.motor_torque_nm, 1e-9)) ** 2 * 40.0   # I^2 R at rated

    def describe(self) -> list[str]:
        return [f"  {self.identity}: {self.screw_kind} screw, {self.lead_m * 1000:.1f} mm lead, "
                f"{self.stroke_m * 1000:.0f} mm stroke",
                f"    {self.max_force_n / 1000:6.2f} kN at {self.max_speed_m_s * 1000:5.0f} mm/s, "
                f"efficiency {self.efficiency:.2f}, "
                f"{'self-locking (free to hold)' if self.self_locking else 'backdrives (needs a brake)'}"]


@dataclass
class ServoMotor:
    """A rotary servo through a gearbox.

    Quoted by two torques that matter differently: CONTINUOUS is what it
    can do all day without cooking, PEAK is what it can do for a second
    or two. A mount sized on peak torque works beautifully for the first
    minute and then thermally folds, which is why both are here."""
    identity: str = "servo"
    continuous_torque_nm: float = 1.2
    peak_torque_nm: float = 3.6
    rated_rpm: float = 3000.0
    gear_ratio: float = 100.0
    gear_efficiency: float = 0.75         # harmonic drive, honestly
    holding_current_fraction: float = 0.35

    @property
    def output_torque_nm(self) -> float:
        return self.continuous_torque_nm * self.gear_ratio * self.gear_efficiency

    @property
    def output_peak_torque_nm(self) -> float:
        return self.peak_torque_nm * self.gear_ratio * self.gear_efficiency

    @property
    def output_rate_deg_s(self) -> float:
        return self.rated_rpm / self.gear_ratio * 6.0

    def power_for(self, torque_nm: float, rate_rad_s: float) -> float:
        mechanical = abs(torque_nm * rate_rad_s)
        return mechanical / max(self.gear_efficiency * 0.88, 1e-9)

    def holding_power_w(self, torque_nm: float) -> float:
        """A servo holds by making torque, and making torque costs
        current whether anything moves or not. This is the quiet cost
        that makes people fit brakes."""
        fraction = abs(torque_nm) / max(self.output_torque_nm, 1e-9)
        return (fraction ** 2) * 45.0 * self.holding_current_fraction

    def describe(self) -> list[str]:
        return [f"  {self.identity}: {self.continuous_torque_nm:.1f} Nm motor through "
                f"{self.gear_ratio:.0f}:1",
                f"    {self.output_torque_nm:6.0f} Nm continuous, {self.output_peak_torque_nm:6.0f} Nm peak, "
                f"{self.output_rate_deg_s:5.0f} deg/s"]


# =====================================================================
#  THE ARMATURE
# =====================================================================
@dataclass
class ArmatureNode:
    identity: str
    position: np.ndarray
    anchored: bool = False
    mass_kg: float = 1.0


@dataclass
class ArmatureMember:
    """One member. If `actuator` is set, its length is an activation."""
    identity: str
    a: str
    b: str
    rest_length_m: float
    actuator: object = None
    minimum_m: float = 0.0
    maximum_m: float = 0.0
    activation_m: float = 0.0        # the commanded length, for an actuated member

    @property
    def actuated(self) -> bool:
        return self.actuator is not None

    def target_length(self) -> float:
        return self.activation_m if self.actuated else self.rest_length_m


@dataclass
class Armature:
    """Nodes and members, some of which can change length."""
    identity: str = "armature"
    nodes: list = field(default_factory=list)
    members: list = field(default_factory=list)
    end_effector: str = ""
    # a direction fixed in the end effector's own body, for pointing
    effector_tip: str = ""

    # ---------------- bookkeeping ----------------
    def index(self) -> dict:
        return {n.identity: i for i, n in enumerate(self.nodes)}

    @property
    def actuators(self) -> list:
        return [m for m in self.members if m.actuated]

    def activations(self) -> np.ndarray:
        return np.array([m.activation_m for m in self.actuators], dtype=float)

    def set_activations(self, values) -> None:
        for m, v in zip(self.actuators, np.asarray(values, dtype=float)):
            m.activation_m = float(np.clip(v, m.minimum_m, m.maximum_m))

    # ---------------- forward: where does it end up ----------------
    def solve_pose(self, *, iterations: int = 60, tolerance: float = 1e-9) -> float:
        """Find node positions satisfying every member length.

        Gauss-Newton on the length residuals, anchors held fixed. This
        is the forward kinematics of a structure with loops in it, and
        it is a solve rather than a walk because a loop has no root."""
        idx = self.index()
        free = [i for i, n in enumerate(self.nodes) if not n.anchored]
        if not free:
            return 0.0
        free_slot = {i: k for k, i in enumerate(free)}
        positions = np.array([n.position for n in self.nodes], dtype=float)

        residual_norm = 0.0
        for _ in range(iterations):
            rows, residuals = [], []
            for m in self.members:
                ia, ib = idx[m.a], idx[m.b]
                delta = positions[ia] - positions[ib]
                length = float(np.linalg.norm(delta))
                if length < 1e-12:
                    continue
                unit = delta / length
                residuals.append(length - m.target_length())
                row = np.zeros(len(free) * 3)
                if ia in free_slot:
                    row[free_slot[ia] * 3:free_slot[ia] * 3 + 3] = unit
                if ib in free_slot:
                    row[free_slot[ib] * 3:free_slot[ib] * 3 + 3] = -unit
                rows.append(row)
            if not rows:
                break
            jac = np.asarray(rows)
            res = np.asarray(residuals)
            residual_norm = float(np.linalg.norm(res))
            if residual_norm < tolerance:
                break
            # damped, because a nearly-singular truss is normal
            lam = 1e-9 + 1e-3 * residual_norm
            step, *_ = np.linalg.lstsq(
                jac.T @ jac + lam * np.eye(jac.shape[1]), -jac.T @ res, rcond=None)
            step = step.reshape(-1, 3)
            for i in free:
                positions[i] += step[free_slot[i]]
        for i, n in enumerate(self.nodes):
            n.position = positions[i]
        return residual_norm

    # ---------------- what the pose IS ----------------
    def effector_position(self) -> np.ndarray:
        idx = self.index()
        return np.array(self.nodes[idx[self.end_effector]].position, dtype=float)

    def effector_direction(self) -> np.ndarray:
        """The pointing direction: tip minus base, normalised."""
        idx = self.index()
        base = np.array(self.nodes[idx[self.end_effector]].position, dtype=float)
        tip = np.array(self.nodes[idx[self.effector_tip]].position, dtype=float)
        d = tip - base
        return d / max(float(np.linalg.norm(d)), 1e-12)


# =====================================================================
#  THE INVERSE SOLVER
# =====================================================================
@dataclass
class PoseSolver:
    """Find the activations that put the armature in a target pose.

    Damped least squares on a numerical Jacobian. The Jacobian is
    numerical on purpose: an armature is authored data, not a formula,
    so there is nothing to differentiate symbolically, and with a
    handful of actuators the finite differences are cheap.
    """
    armature: Armature
    damping: float = 1e-3
    step_limit_m: float = 0.02        # how far an actuator may move per iteration
    tolerance: float = 1e-4
    max_iterations: int = 60
    history: list = field(default_factory=list)

    # ---------------- goals ----------------
    def _error(self, goal_kind: str, goal) -> np.ndarray:
        if goal_kind == "position":
            return np.asarray(goal, dtype=float) - self.armature.effector_position()
        if goal_kind == "direction":
            want = np.asarray(goal, dtype=float)
            want = want / max(float(np.linalg.norm(want)), 1e-12)
            return want - self.armature.effector_direction()
        raise ValueError(f"unknown goal kind {goal_kind!r}")

    def solve(self, goal_kind: str, goal) -> dict:
        arm = self.armature
        arm.solve_pose()
        error = self._error(goal_kind, goal)
        best = float(np.linalg.norm(error))
        self.history = [best]

        for iteration in range(self.max_iterations):
            if best <= self.tolerance:
                break
            activations = arm.activations()
            if activations.size == 0:
                break
            # numerical Jacobian: what each actuator does to the error
            jac = np.zeros((error.size, activations.size))
            step = 1e-4
            for k in range(activations.size):
                probe = activations.copy()
                probe[k] += step
                saved = [np.array(n.position) for n in arm.nodes]
                arm.set_activations(probe)
                arm.solve_pose(iterations=25)
                jac[:, k] = (self._error(goal_kind, goal) - error) / step
                for n, p in zip(arm.nodes, saved):
                    n.position = p
                arm.set_activations(activations)

            # damped least squares: the smallest change that helps, and
            # finite even at a singularity
            # SIGN. `jac` is the derivative of the ERROR with respect to
            # the activations, not of the pose, so the Newton step that
            # drives the error to zero is NEGATIVE of the usual form.
            # Getting this backwards makes every iteration move away
            # from the goal, get rejected, and raise the damping -- so
            # the solver runs its full iteration count, reports failure,
            # and leaves the actuators exactly where they started. Which
            # is precisely what it did.
            lam = self.damping * (1.0 + best)
            delta = -np.linalg.solve(jac.T @ jac + lam * np.eye(activations.size),
                                     jac.T @ error)
            norm = float(np.linalg.norm(delta))
            if norm > self.step_limit_m:
                delta *= self.step_limit_m / norm
            arm.set_activations(activations + delta)
            arm.solve_pose()
            error = self._error(goal_kind, goal)
            now = float(np.linalg.norm(error))
            self.history.append(now)
            if now > best:
                # overshot: back off and damp harder, which is what the
                # damping parameter is for
                self.damping *= 2.5
                arm.set_activations(activations)
                arm.solve_pose()
                error = self._error(goal_kind, goal)
            else:
                self.damping = max(self.damping * 0.7, 1e-6)
                best = now
        return {
            "reached": best <= self.tolerance,
            "error": best,
            "iterations": len(self.history) - 1,
            "activations": {m.identity: m.activation_m for m in arm.actuators},
        }
