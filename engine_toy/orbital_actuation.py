"""Orbital craft, build step 2: the actuation matrix.

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``.
Decision 2: craft laws assemble a matrix from control signals (here
throttles) to forces, and any thruster and mass design plugs in.  A design
is a declared set of :class:`Thruster` records plus a mass
(:class:`CraftDesign`); nothing here knows how many thrusters there are.

The law.  Each thruster's thrust is the catalogue's reactive-thrust law
``eq_TS1_2`` (Tsiolkovsky, ``F_thrust = m_dot * c``).  The throttle is
DECLARED as the fraction of the thruster's maximum propellant mass flow,
``m_dot = clamp(u, u_min, u_max) * m_dot_max``, and the record declares
``max_thrust = m_dot_max * c``; ``c`` (the effective exhaust velocity, which
``eq_TS1_4`` ties to the specific impulse of decision 9) cancels, and the
substitution is checked to cancel.  The thrust acts along the thruster's
declared unit direction in the craft frame.  Attitude is fixed at identity
until decision 9 (build step 7), so craft frame = world frame and

    applied_force = B @ clamp(u),    B[:, k] = max_thrust_k * direction_k

The piece reads the design as columns (``thruster{k}_max_thrust``,
``thruster{k}_direction_{x,y,z}``, ``thruster{k}_throttle_min/max``) and
the control signal as ``thruster{k}_throttle``, so one compiled piece serves
every design with the same thruster count.  :func:`actuation_matrix` builds
the same ``B`` numerically for planners and tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

import honorary_engine_equation_catalogue as honorary

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Thruster:
    """One thruster, declared the way a thruster sheet lists it.

    ``direction`` is the unit direction of the FORCE ON THE CRAFT in the
    craft frame (the exhaust leaves the other way).  ``position_m`` is the
    mount point in the craft frame relative to the centre of mass; it is
    carried for the attitude dynamics of step 7 (torque = r x F) and is not
    read by the step-2 piece.  ``kind`` names the thruster type whose
    specific impulse step 7 will look up (decision 9); it is not read yet.
    ``throttle_min``/``throttle_max`` bound the throttle, a fraction of the
    maximum mass flow, inside [0, 1].
    """

    identity: str
    position_m: tuple[float, float, float]
    direction: tuple[float, float, float]
    max_thrust_n: float
    throttle_min: float = 0.0
    throttle_max: float = 1.0
    kind: str = "ideal"

    def __post_init__(self):
        direction = np.asarray(self.direction, dtype=float).reshape(3)
        if not math.isclose(float(np.linalg.norm(direction)), 1.0,
                            rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"thruster {self.identity!r}: direction "
                             f"{self.direction} is not a unit vector")
        if len(tuple(self.position_m)) != 3:
            raise ValueError(f"thruster {self.identity!r}: position is 3-D")
        if not self.max_thrust_n > 0.0:
            raise ValueError(f"thruster {self.identity!r}: max thrust "
                             "must be positive")
        if not 0.0 <= self.throttle_min <= self.throttle_max <= 1.0:
            raise ValueError(f"thruster {self.identity!r}: throttle range "
                             f"[{self.throttle_min}, {self.throttle_max}] "
                             "is not inside [0, 1]")


@dataclass(frozen=True)
class CraftDesign:
    """A craft: a set of thrusters and a mass (constant until step 7)."""

    thrusters: tuple[Thruster, ...]
    mass_kg: float
    identity: str = "craft"

    def __post_init__(self):
        object.__setattr__(self, "thrusters", tuple(self.thrusters))
        names = [thruster.identity for thruster in self.thrusters]
        if len(set(names)) != len(names):
            raise ValueError(f"craft {self.identity!r}: duplicate thruster "
                             f"identities {names}")
        if not self.mass_kg > 0.0:
            raise ValueError(f"craft {self.identity!r}: mass must be positive")

    @property
    def thruster_count(self) -> int:
        return len(self.thrusters)


def actuation_matrix(design: CraftDesign) -> np.ndarray:
    """``B`` (3 x n): column k is ``max_thrust_k * direction_k``; attitude
    is identity until decision 9."""
    columns = [thruster.max_thrust_n * np.asarray(thruster.direction,
                                                   dtype=float)
               for thruster in design.thrusters]
    if not columns:
        return np.zeros((3, 0))
    return np.stack(columns, axis=1)


def clamp_throttles(design: CraftDesign, throttles) -> np.ndarray:
    """The throttles the piece applies: each clamped to its declared range."""
    u = np.asarray(throttles, dtype=float).reshape(design.thruster_count)
    low = np.asarray([t.throttle_min for t in design.thrusters], dtype=float)
    high = np.asarray([t.throttle_max for t in design.thrusters], dtype=float)
    return np.minimum(np.maximum(u, low), high)


def six_axis_jumper(max_thrust_n: float, mass_kg: float, *,
                    arm_m: float = 1.0, kind: str = "ideal") -> CraftDesign:
    """The prototype jumper (decision 3): one thruster per +/- axis.

    Each thruster is mounted on the face opposite its force direction, on
    the axis through the centre of mass, so it makes no torque.
    """
    thrusters = []
    for axis_index, axis in enumerate(AXES):
        for sign, label in ((1.0, "+"), (-1.0, "-")):
            direction = [0.0, 0.0, 0.0]
            direction[axis_index] = sign
            thrusters.append(Thruster(
                identity=f"{label}{axis}",
                position_m=tuple(-arm_m * value for value in direction),
                direction=tuple(direction),
                max_thrust_n=float(max_thrust_n),
                kind=kind,
            ))
    return CraftDesign(tuple(thrusters), float(mass_kg),
                       identity="six-axis jumper")


# ---------------------------------------------------------------- the law
def thruster_symbols(index: int) -> dict:
    """The declared columns of thruster ``index``."""
    prefix = f"thruster{index}"
    return {
        "throttle": sp.Symbol(f"{prefix}_throttle"),
        "max_thrust": sp.Symbol(f"{prefix}_max_thrust"),
        "throttle_min": sp.Symbol(f"{prefix}_throttle_min"),
        "throttle_max": sp.Symbol(f"{prefix}_throttle_max"),
        "impulse": sp.Symbol(f"{prefix}_impulse"),
        "direction": {axis: sp.Symbol(f"{prefix}_direction_{axis}")
                      for axis in AXES},
    }


def thrust_magnitude(index: int):
    """``eq_TS1_2`` for thruster ``index`` with the declared throttle.

    ``m_dot = clamp(u) * m_dot_max`` and ``m_dot_max = max_thrust / c``;
    the exhaust velocity ``c`` cancels (checked)."""
    law = honorary.eq_TS1_2.rhs
    if not law.has(honorary.m_dot) or not law.has(honorary.c_eff):
        raise RuntimeError("eq_TS1_2 is no longer m_dot * c")
    symbols = thruster_symbols(index)
    throttle = sp.Min(sp.Max(symbols["throttle"], symbols["throttle_min"]),
                      symbols["throttle_max"])
    magnitude = law.xreplace({
        honorary.m_dot: throttle * symbols["max_thrust"] / honorary.c_eff,
    })
    if magnitude.has(honorary.c_eff):
        raise RuntimeError("TS1.2: c did not cancel against max_thrust / c")
    return magnitude


def actuation_force_rhs(axis: str, thruster_count: int):
    """Row ``axis`` of ``B @ clamp(u)``: each thruster's TS1.2 thrust along
    its declared direction (attitude identity)."""
    total = sp.Integer(0)
    for index in range(thruster_count):
        total += (thruster_symbols(index)["direction"][axis]
                  * thrust_magnitude(index))
    return total


def thruster_columns(design: CraftDesign) -> dict:
    """Initial dt-state columns for ``design``: the declared values, zero
    throttle, zero accumulated impulse."""
    columns = {}
    for index, thruster in enumerate(design.thrusters):
        prefix = f"thruster{index}"
        columns[f"{prefix}_throttle"] = np.zeros(1)
        columns[f"{prefix}_impulse"] = np.zeros(1)
        columns[f"{prefix}_max_thrust"] = np.full(1, thruster.max_thrust_n)
        columns[f"{prefix}_throttle_min"] = np.full(1, thruster.throttle_min)
        columns[f"{prefix}_throttle_max"] = np.full(1, thruster.throttle_max)
        for slot, axis in enumerate(AXES):
            columns[f"{prefix}_direction_{axis}"] = np.full(
                1, float(thruster.direction[slot]))
    return columns
