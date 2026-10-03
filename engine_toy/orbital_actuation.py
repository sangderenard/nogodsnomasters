"""Orbital craft, build steps 2 and 7: the actuation matrix, propellant,
attitude.

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``.
Decision 2: craft laws assemble a matrix from control signals (here
throttles) to forces, and any thruster and mass design plugs in.  A design
is a declared set of :class:`Thruster` records plus a mass
(:class:`CraftDesign`); nothing here knows how many thrusters there are.

The thrust law.  Each thruster's thrust is the catalogue's reactive-thrust
law ``eq_TS1_2`` (Tsiolkovsky, ``F_thrust = m_dot * c``).  The throttle is
DECLARED as the fraction of the thruster's maximum propellant mass flow,
``m_dot = clamp(u, u_min, u_max) * m_dot_max``, and the record declares
``max_thrust = m_dot_max * c``; ``c`` cancels in the thrust (checked).

Propellant (decision 9, step 7).  The thruster KIND declares its specific
impulse (:data:`THRUSTER_KINDS`); ``eq_TS1_4`` makes it the exhaust velocity,
``c = I_sp * g_0``, and ``eq_TS1_2`` solved for ``m_dot`` is the propellant
flow, ``m_dot = F / c``.  The piece reads ``1 / c`` per thruster
(``thruster{k}_propellant_per_impulse``, kg per N*s; zero for the reactionless
``ideal`` kind of steps 2-3), the way the machine sim reads inverse inertias.
Thrust exists only while propellant remains: the continuous law is
``m_dot = D * H(propellant_mass)`` for the demanded flow ``D``.  Its exact
average over a step of length ``dt`` is ``D * min(1, propellant / (D dt))``
(the burn ends inside the step), so every thruster delivers its demand times
``propellant_supply = max(0, min(1, propellant / (D dt)))`` (``1`` when
nothing that burns propellant is lit).  A step draws exactly what the tank
holds and never more: thrust stops at empty by the law, not by a branch in
a stepper.

Attitude (decision 9, step 7).  The craft frame is carried to the world by
the attitude ``R`` (catalogue ``eq_N1_3``'s rotation matrix, columns
``attitude_{ab}``, world = R @ craft).  Thrust acts along the thruster's
declared unit direction in the craft frame, so the world force is
``R @ B_craft @ (supply * clamp(u))``: the actuation matrix is
orientation-dependent (:func:`actuation_matrix` takes the attitude).  Each
thruster's moment about the centre of mass, ``r_k x F_k`` in the craft frame
(the machine sim's ``cross(position, force)``; the catalogue has no r x F
law), is the torque that drives the attitude dynamics.

Mass properties follow the machine sim's simplified convention
(``turing/src/compiler/abstract_ui_vehicles.py`` ``mass_properties``): the
residual dry mass is a uniform box, each declared part (here a thruster with
a declared mass) a point mass (``eq_N10_1``), the propellant a point mass at
the centre of mass (it changes the mass, not the inertia), and the craft
axes are principal axes -- a design whose products of inertia are not zero
is refused, not approximated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

import honorary_engine_equation_catalogue as honorary

AXES = ("x", "y", "z")

#: ``g_0`` of ``eq_TS1_4``: the defined standard gravity (m/s^2) that turns a
#: specific impulse in seconds into an exhaust velocity.
STANDARD_GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class ThrusterKind:
    """A thruster type and its specific impulse (vacuum, seconds).

    ``math.inf`` is the reactionless idealization of steps 2-3: it delivers
    thrust and burns no propellant."""

    identity: str
    specific_impulse_s: float

    def __post_init__(self):
        if not self.specific_impulse_s > 0.0:
            raise ValueError(f"thruster kind {self.identity!r}: specific "
                             "impulse must be positive")

    @property
    def exhaust_velocity_m_s(self) -> float:
        """``eq_TS1_4``: ``c = I_sp * g_0``."""
        return self.specific_impulse_s * STANDARD_GRAVITY_M_S2

    @property
    def propellant_per_impulse_kg_n_s(self) -> float:
        """``1 / c``: propellant mass per newton-second of thrust (TS1.2
        solved for ``m_dot``); zero for the reactionless kind."""
        return 1.0 / self.exhaust_velocity_m_s


#: The declared thruster kinds (decision 9).  Typical vacuum values:
#: nitrogen cold gas 65-75 s, catalytic hydrazine 220-235 s, storable
#: MMH/NTO bipropellant 300-320 s.
THRUSTER_KINDS = {kind.identity: kind for kind in (
    ThrusterKind("ideal", math.inf),
    ThrusterKind("cold-gas", 70.0),
    ThrusterKind("monopropellant", 230.0),
    ThrusterKind("bipropellant", 310.0),
)}


@dataclass(frozen=True)
class Thruster:
    """One thruster, declared the way a thruster sheet lists it.

    ``direction`` is the unit direction of the FORCE ON THE CRAFT in the
    craft frame (the exhaust leaves the other way).  ``position_m`` is the
    mount point in the craft frame relative to the centre of mass; the
    thrust's moment about the centre of mass is ``position x force``.
    ``kind`` names the :data:`THRUSTER_KINDS` entry whose specific impulse
    sets the propellant flow.  ``throttle_min``/``throttle_max`` bound the
    throttle, a fraction of the maximum mass flow, inside [0, 1].
    ``mass_kg`` is the thruster hardware, a point mass at its mount (part of
    the craft's dry mass).
    """

    identity: str
    position_m: tuple[float, float, float]
    direction: tuple[float, float, float]
    max_thrust_n: float
    throttle_min: float = 0.0
    throttle_max: float = 1.0
    kind: str = "ideal"
    mass_kg: float = 0.0

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
        if self.kind not in THRUSTER_KINDS:
            raise ValueError(f"thruster {self.identity!r}: kind "
                             f"{self.kind!r} is not declared in "
                             f"THRUSTER_KINDS {sorted(THRUSTER_KINDS)}")
        if self.mass_kg < 0.0:
            raise ValueError(f"thruster {self.identity!r}: negative mass")

    @property
    def thruster_kind(self) -> ThrusterKind:
        return THRUSTER_KINDS[self.kind]


@dataclass(frozen=True)
class CraftDesign:
    """A craft: thrusters, a wet mass, the propellant inside it, and the
    body box the residual dry mass fills.

    ``mass_kg`` is the WET mass at the start (dry + propellant).
    ``body_size_m`` is the full (x, y, z) extent of the uniform box the
    residual dry mass (dry minus declared thruster masses) is spread over.
    """

    thrusters: tuple[Thruster, ...]
    mass_kg: float
    identity: str = "craft"
    propellant_kg: float = 0.0
    body_size_m: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "thrusters", tuple(self.thrusters))
        object.__setattr__(self, "body_size_m",
                           tuple(float(v) for v in self.body_size_m))
        names = [thruster.identity for thruster in self.thrusters]
        if len(set(names)) != len(names):
            raise ValueError(f"craft {self.identity!r}: duplicate thruster "
                             f"identities {names}")
        if not self.mass_kg > 0.0:
            raise ValueError(f"craft {self.identity!r}: mass must be positive")
        if not 0.0 <= self.propellant_kg < self.mass_kg:
            raise ValueError(f"craft {self.identity!r}: propellant "
                             f"{self.propellant_kg} kg must be inside "
                             f"[0, mass {self.mass_kg} kg)")
        if len(self.body_size_m) != 3 or not all(v > 0.0
                                                for v in self.body_size_m):
            raise ValueError(f"craft {self.identity!r}: body size is three "
                             "positive extents")
        if not self.residual_mass_kg > 0.0:
            raise ValueError(f"craft {self.identity!r}: thruster masses must "
                             "leave positive residual dry mass")

    @property
    def thruster_count(self) -> int:
        return len(self.thrusters)

    @property
    def dry_mass_kg(self) -> float:
        return self.mass_kg - self.propellant_kg

    @property
    def residual_mass_kg(self) -> float:
        return self.dry_mass_kg - sum(t.mass_kg for t in self.thrusters)


# ------------------------------------------------------- mass properties
def inertia_tensor(design: CraftDesign) -> np.ndarray:
    """The craft's 3x3 inertia tensor about its centre of mass (craft axes).

    The machine sim's convention: the residual dry mass as a uniform box,
    ``m (b^2 + c^2) / 12`` per axis, plus every declared thruster mass as a
    point mass, ``eq_N10_1`` ``I_ij = sum m (|r|^2 delta_ij - r_i r_j)``.
    The propellant sits at the centre of mass and adds nothing.  Mount
    points are relative to the centre of mass, so the point masses must not
    move it (checked)."""
    a, b, c = design.body_size_m
    m = design.residual_mass_kg
    tensor = np.diag([m * (b * b + c * c) / 12.0,
                      m * (a * a + c * c) / 12.0,
                      m * (a * a + b * b) / 12.0])
    moment = np.zeros(3)
    for thruster in design.thrusters:
        r = np.asarray(thruster.position_m, dtype=float)
        tensor += thruster.mass_kg * (float(r @ r) * np.eye(3)
                                      - np.outer(r, r))
        moment += thruster.mass_kg * r
    if np.linalg.norm(moment) > 1.0e-12 * max(1.0, design.mass_kg):
        raise ValueError(f"craft {design.identity!r}: declared thruster "
                         "masses move the centre of mass off the mount "
                         "origin")
    return tensor


def principal_inertia(design: CraftDesign) -> np.ndarray:
    """The diagonal of :func:`inertia_tensor` (kg m^2).  Simplified: the
    craft axes are principal axes; a design with products of inertia is
    refused rather than approximated."""
    tensor = inertia_tensor(design)
    off = tensor - np.diag(np.diag(tensor))
    if np.abs(off).max() > 1.0e-12 * np.trace(tensor):
        raise ValueError(f"craft {design.identity!r}: craft axes are not "
                         "principal axes (products of inertia "
                         f"{off[0, 1]}, {off[0, 2]}, {off[1, 2]})")
    return np.diag(tensor).copy()


# ------------------------------------------------------ host-side matrices
def craft_force_matrix(design: CraftDesign) -> np.ndarray:
    """``B_craft`` (3 x n), craft frame: column k is
    ``max_thrust_k * direction_k``."""
    columns = [thruster.max_thrust_n * np.asarray(thruster.direction,
                                                   dtype=float)
               for thruster in design.thrusters]
    if not columns:
        return np.zeros((3, 0))
    return np.stack(columns, axis=1)


def actuation_matrix(design: CraftDesign, attitude=None) -> np.ndarray:
    """``B`` (3 x n), world frame: ``R @ B_craft`` for the attitude ``R``
    (world = R @ craft).  ``attitude=None`` is the identity attitude, the
    frame steps 2-3 assumed."""
    craft = craft_force_matrix(design)
    if attitude is None:
        return craft
    rotation = np.asarray(attitude, dtype=float).reshape(3, 3)
    return rotation @ craft


def torque_matrix(design: CraftDesign) -> np.ndarray:
    """Craft-frame torque per unit throttle (3 x n): column k is
    ``position_k x (max_thrust_k * direction_k)``."""
    columns = [np.cross(np.asarray(thruster.position_m, dtype=float),
                        thruster.max_thrust_n
                        * np.asarray(thruster.direction, dtype=float))
               for thruster in design.thrusters]
    if not columns:
        return np.zeros((3, 0))
    return np.stack(columns, axis=1)


def propellant_flow_per_throttle(design: CraftDesign) -> np.ndarray:
    """Propellant flow (kg/s) per unit throttle, per thruster:
    ``max_thrust / (I_sp g_0)``."""
    return np.asarray([thruster.max_thrust_n
                       * thruster.thruster_kind.propellant_per_impulse_kg_n_s
                       for thruster in design.thrusters], dtype=float)


def clamp_throttles(design: CraftDesign, throttles) -> np.ndarray:
    """The throttles the piece applies: each clamped to its declared range."""
    u = np.asarray(throttles, dtype=float).reshape(design.thruster_count)
    low = np.asarray([t.throttle_min for t in design.thrusters], dtype=float)
    high = np.asarray([t.throttle_max for t in design.thrusters], dtype=float)
    return np.minimum(np.maximum(u, low), high)


def six_axis_jumper(max_thrust_n: float, mass_kg: float, *,
                    arm_m: float = 1.0, kind: str = "ideal",
                    propellant_kg: float = 0.0,
                    body_size_m=(1.0, 1.0, 1.0)) -> CraftDesign:
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
                       identity="six-axis jumper",
                       propellant_kg=float(propellant_kg),
                       body_size_m=tuple(body_size_m))


# ---------------------------------------------------------------- the law
PROPELLANT_MASS = sp.Symbol("propellant_mass")
PROPELLANT_SUPPLY = sp.Symbol("propellant_supply")
PROPELLANT_FLOW = sp.Symbol("propellant_flow")
DT = sp.Symbol("dt")


def attitude_symbols() -> dict:
    """``R`` as columns: ``attitude_{a}{b}`` is row a, column b."""
    return {(row, col): sp.Symbol(f"attitude_{a}{b}")
            for row, a in enumerate(AXES) for col, b in enumerate(AXES)}


def attitude_matrix_symbol() -> sp.Matrix:
    symbols = attitude_symbols()
    return sp.Matrix(3, 3, lambda row, col: symbols[(row, col)])


def thruster_symbols(index: int) -> dict:
    """The declared columns of thruster ``index``."""
    prefix = f"thruster{index}"
    return {
        "throttle": sp.Symbol(f"{prefix}_throttle"),
        "max_thrust": sp.Symbol(f"{prefix}_max_thrust"),
        "throttle_min": sp.Symbol(f"{prefix}_throttle_min"),
        "throttle_max": sp.Symbol(f"{prefix}_throttle_max"),
        "impulse": sp.Symbol(f"{prefix}_impulse"),
        "propellant_per_impulse": sp.Symbol(
            f"{prefix}_propellant_per_impulse"),
        "direction": {axis: sp.Symbol(f"{prefix}_direction_{axis}")
                      for axis in AXES},
        "position": {axis: sp.Symbol(f"{prefix}_position_{axis}")
                     for axis in AXES},
    }


def thrust_magnitude(index: int):
    """``eq_TS1_2`` for thruster ``index`` with the declared throttle: the
    DEMANDED thrust (before the propellant supply).

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


def propellant_demand(index: int):
    """Thruster ``index``'s demanded propellant flow: ``eq_TS1_2`` solved
    for ``m_dot`` with its demanded thrust and ``1 / c`` from the kind
    (``eq_TS1_4``, evaluated on the host)."""
    (m_dot,) = sp.solve(honorary.eq_TS1_2, honorary.m_dot)
    symbols = thruster_symbols(index)
    flow = m_dot.xreplace({
        honorary.F_thrust: thrust_magnitude(index),
        honorary.c_eff: 1 / symbols["propellant_per_impulse"],
    })
    if flow.has(honorary.c_eff, honorary.F_thrust):
        raise RuntimeError("TS1.2 solved for m_dot did not take F and 1/c")
    return flow


def total_propellant_demand(thruster_count: int):
    return sum((propellant_demand(index) for index in range(thruster_count)),
               sp.Integer(0))


def propellant_supply_rhs(thruster_count: int):
    """The fraction of every demand the step delivers: the exact step
    average of ``H(propellant_mass)`` at the demanded flow ``D``,
    ``max(0, min(1, propellant / (D dt)))``, and ``1`` when ``D`` is zero
    (nothing lit that burns propellant)."""
    demand = total_propellant_demand(thruster_count)
    if demand == 0:
        return sp.Integer(1)
    return sp.Piecewise(
        (sp.Max(sp.Integer(0),
                sp.Min(sp.Integer(1), PROPELLANT_MASS / (DT * demand))),
         demand > 0),
        (sp.Integer(1), True))


def delivered_thrust(index: int):
    """Thruster ``index``'s delivered thrust: demand times the supply."""
    return thrust_magnitude(index) * PROPELLANT_SUPPLY


def propellant_flow_rhs(thruster_count: int):
    """The propellant flow the step draws: ``supply * D``."""
    return PROPELLANT_SUPPLY * total_propellant_demand(thruster_count)


def craft_force_rhs(axis: str, thruster_count: int):
    """Row ``axis`` of ``B_craft @ (supply * clamp(u))``, craft frame."""
    total = sp.Integer(0)
    for index in range(thruster_count):
        total += (thruster_symbols(index)["direction"][axis]
                  * delivered_thrust(index))
    return total


def actuation_force_rhs(axis: str, thruster_count: int):
    """Row ``axis`` of the world thrust ``R @ B_craft @ (supply * clamp(u))``:
    each thruster's delivered TS1.2 thrust along its declared direction,
    carried to the world by the attitude."""
    rotation = attitude_symbols()
    row = AXES.index(axis)
    return sum((rotation[(row, col)] * craft_force_rhs(b, thruster_count)
                for col, b in enumerate(AXES)), sp.Integer(0))


def actuation_torque_rhs(axis: str, thruster_count: int):
    """Row ``axis`` of the craft-frame torque ``sum_k r_k x F_k``."""
    total = sp.Integer(0)
    i = AXES.index(axis)
    j, k = AXES[(i + 1) % 3], AXES[(i + 2) % 3]
    for index in range(thruster_count):
        symbols = thruster_symbols(index)
        r, d = symbols["position"], symbols["direction"]
        total += (r[j] * d[k] - r[k] * d[j]) * delivered_thrust(index)
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
        columns[f"{prefix}_propellant_per_impulse"] = np.full(
            1, thruster.thruster_kind.propellant_per_impulse_kg_n_s)
        for slot, axis in enumerate(AXES):
            columns[f"{prefix}_direction_{axis}"] = np.full(
                1, float(thruster.direction[slot]))
            columns[f"{prefix}_position_{axis}"] = np.full(
                1, float(thruster.position_m[slot]))
    return columns
