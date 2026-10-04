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

import dataclasses
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


#: Declared thruster roles (step 8); ``""`` is the undeclared role of the
#: steps 2-7 designs.
THRUSTER_ROLES = ("", "navigation", "brake", "main")


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

    Machine-craft declarations (step 8, :mod:`orbital_craft_machine`; the
    defaults are the fixed, instant, unfed thruster of steps 2-7):

    ``role``                ``navigation`` (RCS), ``brake`` (retro, against
                            the main thrust) or ``main``.
    ``cone_half_angle_rad`` the gimbal's cone of available directions about
                            ``direction`` (0 = fixed).  A gimballed thruster
                            is a Cardan gimbal: actuator angle ``a`` turns
                            the thrust about ``gimbal_axis`` (``e1``), then
                            ``b`` about ``e2 = direction x e1``, so
                            ``d = cos a cos b n + cos a sin b e1 - sin a e2``
                            and ``n . d = cos a cos b``: the cone is exactly
                            ``cos a cos b >= cos(cone)``.
    ``gimbal_slew_rad_s``   each gimbal actuator's angular rate limit.
    ``throttle_slew_per_s`` the throttle state's rate limit (spool-up and
                            shut-down), throttle fraction per second.
    ``deadband``            the minimum stable throttle: a throttle state
                            below it delivers no thrust (0 = none).
    ``feeds``               ``((tank identity, mass fraction), ...)``: which
                            tanks this thruster's propellant flow is drawn
                            from, in what mass shares (a bipropellant engine
                            draws fuel and oxidiser at its mixture ratio).
    """

    identity: str
    position_m: tuple[float, float, float]
    direction: tuple[float, float, float]
    max_thrust_n: float
    throttle_min: float = 0.0
    throttle_max: float = 1.0
    kind: str = "ideal"
    mass_kg: float = 0.0
    role: str = ""
    cone_half_angle_rad: float = 0.0
    gimbal_axis: tuple[float, float, float] | None = None
    gimbal_slew_rad_s: float = math.inf
    throttle_slew_per_s: float = math.inf
    deadband: float = 0.0
    feeds: tuple = ()

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
        if self.role not in THRUSTER_ROLES:
            raise ValueError(f"thruster {self.identity!r}: role "
                             f"{self.role!r} is not one of {THRUSTER_ROLES}")
        if not 0.0 <= self.cone_half_angle_rad < 0.5 * math.pi:
            raise ValueError(f"thruster {self.identity!r}: cone half-angle "
                             "must be inside [0, pi/2)")
        if self.cone_half_angle_rad > 0.0:
            if self.gimbal_axis is None:
                raise ValueError(f"thruster {self.identity!r}: a gimballed "
                                 "thruster declares its gimbal_axis")
            axis = np.asarray(self.gimbal_axis, dtype=float).reshape(3)
            if (not math.isclose(float(np.linalg.norm(axis)), 1.0,
                                 rel_tol=0.0, abs_tol=1.0e-12)
                    or abs(float(axis @ direction)) > 1.0e-12):
                raise ValueError(f"thruster {self.identity!r}: gimbal_axis "
                                 "must be a unit vector normal to direction")
        if not (self.gimbal_slew_rad_s > 0.0
                and self.throttle_slew_per_s > 0.0):
            raise ValueError(f"thruster {self.identity!r}: slew rates must "
                             "be positive (math.inf: instant)")
        if not 0.0 <= self.deadband <= self.throttle_max:
            raise ValueError(f"thruster {self.identity!r}: deadband must be "
                             "inside [0, throttle_max]")
        object.__setattr__(self, "feeds", tuple(
            (str(tank), float(share)) for tank, share in self.feeds))
        if self.feeds:
            shares = [share for _tank, share in self.feeds]
            tanks = [tank for tank, _share in self.feeds]
            if (len(set(tanks)) != len(tanks) or min(shares) <= 0.0
                    or not math.isclose(sum(shares), 1.0, rel_tol=0.0,
                                        abs_tol=1.0e-12)):
                raise ValueError(f"thruster {self.identity!r}: feeds are "
                                 "distinct tanks with positive mass shares "
                                 "summing to 1")

    @property
    def thruster_kind(self) -> ThrusterKind:
        return THRUSTER_KINDS[self.kind]

    @property
    def gimballed(self) -> bool:
        return self.cone_half_angle_rad > 0.0

    def gimbal_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(n, e1, e2)``: the nominal axis, the first gimbal axis and
        ``e2 = n x e1`` (right-handed).  A fixed thruster has ``e1 = e2 =
        0``."""
        n = np.asarray(self.direction, dtype=float).reshape(3)
        if not self.gimballed:
            return n, np.zeros(3), np.zeros(3)
        e1 = np.asarray(self.gimbal_axis, dtype=float).reshape(3)
        return n, e1, np.cross(n, e1)

    def direction_at(self, a: float = 0.0, b: float = 0.0) -> np.ndarray:
        """The thrust direction (craft frame) at gimbal angles ``(a, b)``:
        ``cos a cos b n + cos a sin b e1 - sin a e2``."""
        n, e1, e2 = self.gimbal_frame()
        if not self.gimballed:
            return n
        return (math.cos(a) * math.cos(b) * n + math.cos(a) * math.sin(b) * e1
                - math.sin(a) * e2)


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


# =====================================================================
# Step 8: the machine craft's thrusters -- gimbals, slews, deadband, feeds
# =====================================================================
# The laws below are the per-thruster state machines of
# :mod:`orbital_craft_machine`.  The seam's ``thruster{k}_throttle`` column
# stays the COMMAND; what the thruster delivers is its own STATE:
#
#   throttle state   s <- s + clamp(clamp(u, u_min, u_max) - s, -r dt, r dt)
#   gimbal actuator  a <- a + clamp(clamp(a_cmd, -cone, cone) - a, -g dt, g dt)
#                    (and b likewise): each actuator slews at its rate
#   delivered        v = s if s >= deadband else 0 (its own column,
#                    written with the state by the slew piece)
#   direction        d = cos a cos b n + cos a sin b e1 - sin a e2
#   supply           per TANK, the step average of H(tank) at its demand
#                    (the step-7 law, one tank at a time); a thruster
#                    delivers the least supply of the tanks it draws from
#   thrust           TS1.2 at m_dot = v * supply * m_dot_max
#   torque           sum_k (r_k - c) x F_k about the CURRENT centre of mass
#   tank flow        sum over thrusters of share_kt * TS1.2 flow
#
# The structure (which thrusters gimbal, which tanks feed which thruster)
# is the machine's declaration and decides which columns an equation
# names; every number is a column.


def machine_thruster_symbols(index: int) -> dict:
    """Thruster ``index``'s columns: the step-7 ones plus its state
    machine's."""
    prefix = f"thruster{index}"
    symbols = thruster_symbols(index)
    symbols.update({
        "state": sp.Symbol(f"{prefix}_throttle_state"),
        "delivered": sp.Symbol(f"{prefix}_delivered"),
        "throttle_slew": sp.Symbol(f"{prefix}_throttle_slew"),
        "deadband": sp.Symbol(f"{prefix}_deadband"),
        "gimbal_a": sp.Symbol(f"{prefix}_gimbal_a"),
        "gimbal_b": sp.Symbol(f"{prefix}_gimbal_b"),
        "gimbal_a_command": sp.Symbol(f"{prefix}_gimbal_a_command"),
        "gimbal_b_command": sp.Symbol(f"{prefix}_gimbal_b_command"),
        "cone": sp.Symbol(f"{prefix}_cone"),
        "gimbal_slew": sp.Symbol(f"{prefix}_gimbal_slew"),
        "e1": {axis: sp.Symbol(f"{prefix}_gimbal_e1_{axis}")
               for axis in AXES},
        "e2": {axis: sp.Symbol(f"{prefix}_gimbal_e2_{axis}")
               for axis in AXES},
    })
    return symbols


def feed_symbol(index: int, tank: int) -> sp.Symbol:
    """The mass share of thruster ``index``'s flow drawn from tank ``tank``."""
    return sp.Symbol(f"thruster{index}_feed{tank}")


def tank_symbols(tank: int) -> dict:
    prefix = f"tank{tank}"
    return {"propellant": sp.Symbol(f"{prefix}_propellant"),
            "supply": sp.Symbol(f"{prefix}_supply"),
            "flow": sp.Symbol(f"{prefix}_flow")}


CENTRE_OF_MASS = {axis: sp.Symbol(f"centre_of_mass_{axis}") for axis in AXES}


def _slew(current, target, rate):
    """``current`` moved toward ``target`` by at most ``rate * dt``."""
    step = rate * DT
    return current + sp.Max(-step, sp.Min(step, target - current))


def throttle_state_rhs(index: int):
    """The throttle state slews toward the clamped command at its rate."""
    s = machine_thruster_symbols(index)
    command = sp.Min(sp.Max(s["throttle"], s["throttle_min"]),
                     s["throttle_max"])
    return _slew(s["state"], command, s["throttle_slew"])


def gimbal_state_rhs(index: int, actuator: str):
    """Gimbal actuator ``actuator`` ('a' or 'b') slews toward its command,
    held to the actuator travel ``[-cone, cone]``."""
    s = machine_thruster_symbols(index)
    command = sp.Max(-s["cone"], sp.Min(s["cone"],
                                        s[f"gimbal_{actuator}_command"]))
    return _slew(s[f"gimbal_{actuator}"], command, s["gimbal_slew"])


def delivered_throttle_rhs(index: int):
    """The throttle the thruster delivers at its NEW state (the slew law's
    result this substep): the state, or nothing below the deadband.  Written
    by the slew piece as ``thruster{k}_delivered``; every later law reads
    that column (spelling the deadband inside each tank's supply condition
    made SymPy's Piecewise construction recurse without bound)."""
    s = machine_thruster_symbols(index)
    state = throttle_state_rhs(index)
    return sp.Piecewise((state, state >= s["deadband"]),
                        (sp.Integer(0), True))


def delivered_throttle_area_rhs(index: int):
    """Exact time integral of the authored bounded ramp and its deadband.

    ``throttle_state_rhs`` gives the ramp endpoint for an elapsed ``DT``.
    Its moving interval lasts ``abs(end - start) / rate``; any remaining
    time holds the endpoint. On the moving interval, integrating the
    delivered throttle is integrating ``u du / rate`` above the deadband.
    The factored difference of squares avoids cancellation for a short
    interval. This observable has no integration substeps or dt proposal.

    Rate is positive by ``Thruster``'s contract, with infinity denoting an
    instantaneous actuator. A zero duration delivers zero and leaves the
    state unchanged, including that instantaneous case.
    """
    s = machine_thruster_symbols(index)
    start = s["state"]
    end = throttle_state_rhs(index)
    low = sp.Max(sp.Min(start, end), s["deadband"])
    high = sp.Max(start, end, s["deadband"])
    moving_area = (high - low) * (high + low) / (2 * s["throttle_slew"])
    moving_time = sp.Abs(end - start) / s["throttle_slew"]
    area = moving_area + (DT - moving_time) * delivered_throttle_rhs(index)
    return sp.Piecewise((area, DT > 0), (sp.Integer(0), True))


def throttle_delivery(thrusters, states, commands, elapsed_s: float) -> tuple:
    """Native ``(delivered throttle-seconds, endpoint states)`` for a command.

    This is an unfed actuator observable: tank availability and the changing
    thrust direction belong to the craft's actual integrated impulse. It is
    suitable for pricing a requested command interval without recreating the
    dt controller's internal partition.
    """
    count = len(thrusters)
    duration = float(elapsed_s)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("actuator duration must be finite and nonnegative")
    states = np.asarray(states, dtype=float).reshape(count)
    commands = np.asarray(commands, dtype=float).reshape(count)
    if count == 0 or duration == 0.0:
        return np.zeros(count), states.copy()
    # A batch lane is one thruster, using the same authored scalar law for
    # every lane. equation_piece owns compilation, caching and book receipts.
    s = machine_thruster_symbols(0)
    piece = honorary.equation_piece("orbital_actuator_delivery", (
        sp.Eq(sp.Symbol("throttle_area"), delivered_throttle_area_rhs(0),
              evaluate=False),
        sp.Eq(sp.Symbol("throttle_state_at_end"), throttle_state_rhs(0),
              evaluate=False),
    ), batch=count)
    values = {
        str(s["state"]): states,
        str(s["throttle"]): commands,
        str(s["throttle_min"]): np.asarray([t.throttle_min for t in thrusters]),
        str(s["throttle_max"]): np.asarray([t.throttle_max for t in thrusters]),
        str(s["throttle_slew"]): np.asarray([t.throttle_slew_per_s for t in thrusters]),
        str(s["deadband"]): np.asarray([t.deadband for t in thrusters]),
        str(DT): np.full(count, duration),
    }
    outputs = dict(zip(piece.output_names, piece(*(
        values[name] for name in piece.argument_names))))
    return (np.asarray(outputs["throttle_area"]).reshape(count),
            np.asarray(outputs["throttle_state_at_end"]).reshape(count))


def delivered_throttle(index: int):
    """The delivered-throttle column (``delivered_throttle_rhs``)."""
    return machine_thruster_symbols(index)["delivered"]


def machine_direction(index: int, gimballed: bool) -> dict:
    """Thruster ``index``'s thrust direction (craft frame), per axis."""
    s = machine_thruster_symbols(index)
    n = s["direction"]
    if not gimballed:
        return dict(n)
    a, b = s["gimbal_a"], s["gimbal_b"]
    return {axis: (sp.cos(a) * sp.cos(b) * n[axis]
                   + sp.cos(a) * sp.sin(b) * s["e1"][axis]
                   - sp.sin(a) * s["e2"][axis]) for axis in AXES}


def _ts12_flow(index: int, thrust):
    """``eq_TS1_2`` solved for ``m_dot`` at ``thrust``, ``1/c`` from the
    kind (the step-7 propellant_demand, at a given thrust)."""
    (m_dot,) = sp.solve(honorary.eq_TS1_2, honorary.m_dot)
    s = machine_thruster_symbols(index)
    return m_dot.xreplace({honorary.F_thrust: thrust,
                           honorary.c_eff: 1 / s["propellant_per_impulse"]})


def _ts12_thrust(index: int, flow_fraction):
    """``eq_TS1_2`` at ``m_dot = flow_fraction * max_thrust / c``; ``c``
    cancels (the step-7 thrust_magnitude, at a given flow fraction)."""
    s = machine_thruster_symbols(index)
    thrust = honorary.eq_TS1_2.rhs.xreplace({
        honorary.m_dot: flow_fraction * s["max_thrust"] / honorary.c_eff})
    if thrust.has(honorary.c_eff):
        raise RuntimeError("TS1.2: c did not cancel against max_thrust / c")
    return thrust


def _feeds(thruster: Thruster, tank_index: dict) -> list:
    try:
        return [tank_index[tank] for tank, _share in thruster.feeds]
    except KeyError as missing:
        raise ValueError(f"thruster {thruster.identity!r} is fed by "
                         f"undeclared tank {missing}") from None


def tank_supply_rhs(tank: int, thrusters, tank_index: dict):
    """Tank ``tank``'s supply fraction this step: the step-7 exact step
    average of ``H(propellant)`` at the demanded draw ``D`` (the TS1.2
    flows of the delivered throttles, in their declared shares)."""
    demand = sp.Integer(0)
    for index, thruster in enumerate(thrusters):
        if tank in _feeds(thruster, tank_index):
            demand += feed_symbol(index, tank) * _ts12_flow(
                index, _ts12_thrust(index, delivered_throttle(index)))
    if demand == 0:
        return sp.Integer(1)
    propellant = tank_symbols(tank)["propellant"]
    return sp.Piecewise(
        (sp.Max(sp.Integer(0), sp.Min(sp.Integer(1),
                                      propellant / (DT * demand))),
         demand > 0),
        (sp.Integer(1), True))


def thruster_supply(index: int, thruster: Thruster, tank_index: dict):
    """The least supply of the tanks thruster ``index`` draws from (a
    bipropellant engine stops when either propellant does); ``1`` for a
    reactionless thruster.  A thruster that burns propellant must declare
    its feeds."""
    tanks = _feeds(thruster, tank_index)
    if not tanks:
        if math.isinf(thruster.thruster_kind.specific_impulse_s):
            return sp.Integer(1)
        raise ValueError(f"thruster {thruster.identity!r} burns propellant "
                         "and declares no feed tank")
    supplies = [tank_symbols(tank)["supply"] for tank in tanks]
    return supplies[0] if len(supplies) == 1 else sp.Min(*supplies)


def machine_thrust(index: int, thruster: Thruster, tank_index: dict):
    """Thruster ``index``'s delivered TS1.2 thrust."""
    return _ts12_thrust(index, delivered_throttle(index)
                        * thruster_supply(index, thruster, tank_index))


def machine_force_rhs(axis: str, thrusters, tank_index: dict):
    """Row ``axis`` of the world thrust ``R @ sum_k F_k d_k``."""
    craft = {b: sp.Integer(0) for b in AXES}
    for index, thruster in enumerate(thrusters):
        thrust = machine_thrust(index, thruster, tank_index)
        direction = machine_direction(index, thruster.gimballed)
        for b in AXES:
            craft[b] += thrust * direction[b]
    rotation = attitude_symbols()
    row = AXES.index(axis)
    return sum((rotation[(row, col)] * craft[b]
                for col, b in enumerate(AXES)), sp.Integer(0))


def machine_torque_rhs(axis: str, thrusters, tank_index: dict):
    """Row ``axis`` of ``sum_k (r_k - c) x F_k`` (craft frame, about the
    current centre of mass ``c``)."""
    i = AXES.index(axis)
    j, k = AXES[(i + 1) % 3], AXES[(i + 2) % 3]
    total = sp.Integer(0)
    for index, thruster in enumerate(thrusters):
        s = machine_thruster_symbols(index)
        arm = {a: s["position"][a] - CENTRE_OF_MASS[a] for a in AXES}
        thrust = machine_thrust(index, thruster, tank_index)
        d = machine_direction(index, thruster.gimballed)
        total += (arm[j] * d[k] - arm[k] * d[j]) * thrust
    return total


def tank_flow_rhs(tank: int, thrusters, tank_index: dict):
    """Tank ``tank``'s draw: the declared share of every thruster's TS1.2
    flow at its delivered thrust."""
    total = sp.Integer(0)
    for index, thruster in enumerate(thrusters):
        if tank in _feeds(thruster, tank_index):
            total += feed_symbol(index, tank) * _ts12_flow(
                index, machine_thrust(index, thruster, tank_index))
    return total


def machine_thruster_columns(thrusters, tank_index: dict) -> dict:
    """The step-8 columns of every thruster (one lane): states and commands
    at zero; declared rates, cone, deadband, gimbal frame and feeds."""
    columns = {}
    for index, thruster in enumerate(thrusters):
        prefix = f"thruster{index}"
        _n, e1, e2 = thruster.gimbal_frame()
        values = {
            "throttle_state": 0.0,
            "delivered": 0.0,
            "throttle_slew": thruster.throttle_slew_per_s,
            "deadband": thruster.deadband,
            "gimbal_a": 0.0, "gimbal_b": 0.0,
            "gimbal_a_command": 0.0, "gimbal_b_command": 0.0,
            "cone": thruster.cone_half_angle_rad,
            "gimbal_slew": thruster.gimbal_slew_rad_s,
        }
        for slot, axis in enumerate(AXES):
            values[f"gimbal_e1_{axis}"] = float(e1[slot])
            values[f"gimbal_e2_{axis}"] = float(e2[slot])
        for tank, share in thruster.feeds:
            values[f"feed{tank_index[tank]}"] = share
        for name, value in values.items():
            columns[f"{prefix}_{name}"] = np.full(1, float(value))
    return columns


# ======================================================================
# Reaction wheels: electric flywheels on the craft's own axes
# ======================================================================
# A wheel is a rotor DECLARED on its body (operating_states: rotor_axis,
# rotor_rated_rpm, rotor_inertia_kg_m2, balance, runs_in) driven by a
# permanent-magnet motor the body also declares (motor_max_torque_n_m,
# motor_torque_constant_n_m_a, motor_winding_resistance_ohm).  Its state is
# the wheel's ABSOLUTE axial angular momentum h (N m s): the motor torque
# tau_m acts on it directly, dh/dt = tau_m (eq_N1_6 about the rotor's own
# symmetry axis, where the w x I w term vanishes; the bearings carry the
# transverse reaction), and -tau_m a acts on the craft.  The craft then
# carries its inertia WITHOUT the rotors' axial spin, I' = I - sum I_w a a^T,
# and the attitude law is eq_N1_6 on the total angular momentum
#
#     I' dw/dt = tau_ext - sum tau_m a - w x (I' w + sum h a)
#
# so the motor torque is an exact exchange between the two stores and
# H = I' w + sum h a changes only by tau_ext (in the inertial frame).
# The motor sees the RELATIVE speed W = h / I_w - a . w.
#
# Saturation is a law: the delivered torque is the command held to the
# motor's torque limit and to what keeps W inside [-W_max, W_max] over the
# step -- the exact step average of a torque that stops at the limit (the
# propellant supply's form).  Nothing branches in a stepper.
#
# Power (the simple motor law): eq_F9_4's Lorentz law F = Bl i, rotary
# (tau = K_t i), and eq_F9_5's motional back-EMF v = Bl dx/dt, rotary
# (v = K_t W), give the winding current and EMF; the electrical power is
# v i plus the copper loss i^2 R (eq_F8_11, one winding):
#
#     P = tau W + R tau^2 / K_t^2
#
# (negative when braking returns more than the copper loss: regeneration).


@dataclass(frozen=True)
class ReactionWheel:
    """One wheel, as its node declares it (:mod:`orbital_craft_machine`).

    ``max_speed_rad_s`` is the rotor's rated speed (relative to the craft);
    ``dump_fraction``: above that fraction of it the allocation dumps the
    excess momentum (desaturation) with time constant ``dump_time_s``, the
    RCS supplying the torque that holds the craft while it does."""

    identity: str
    axis: tuple[float, float, float]
    rotor_inertia_kg_m2: float
    max_speed_rad_s: float
    max_torque_n_m: float
    torque_constant_n_m_a: float
    winding_resistance_ohm: float
    dump_fraction: float = 0.8
    dump_time_s: float = 20.0
    mass_kg: float = 0.0
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        axis = np.asarray(self.axis, dtype=float).reshape(3)
        if not math.isclose(float(np.linalg.norm(axis)), 1.0, rel_tol=0.0,
                            abs_tol=1.0e-12):
            raise ValueError(f"wheel {self.identity!r}: axis is not a unit "
                             "vector")
        for name in ("rotor_inertia_kg_m2", "max_speed_rad_s",
                     "max_torque_n_m", "torque_constant_n_m_a",
                     "dump_time_s"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"wheel {self.identity!r}: {name} must be "
                                 "positive")
        if self.winding_resistance_ohm < 0.0:
            raise ValueError(f"wheel {self.identity!r}: negative resistance")
        if not 0.0 < self.dump_fraction <= 1.0:
            raise ValueError(f"wheel {self.identity!r}: dump_fraction is "
                             "inside (0, 1]")

    @property
    def max_momentum_n_m_s(self) -> float:
        return self.rotor_inertia_kg_m2 * self.max_speed_rad_s

    def electrical_power_w(self, torque_n_m, speed_rad_s):
        """The motor law (module comment), on the host."""
        current = np.asarray(torque_n_m, float) / self.torque_constant_n_m_a
        emf = self.torque_constant_n_m_a * np.asarray(speed_rad_s, float)
        return emf * current + current**2 * self.winding_resistance_ohm


def wheel_symbols(index: int) -> dict:
    prefix = f"wheel{index}"
    return {
        "momentum": sp.Symbol(f"{prefix}_momentum"),
        "command": sp.Symbol(f"{prefix}_torque_command"),
        "inertia": sp.Symbol(f"{prefix}_rotor_inertia"),
        "max_speed": sp.Symbol(f"{prefix}_max_speed"),
        "max_torque": sp.Symbol(f"{prefix}_max_torque"),
        "torque_constant": sp.Symbol(f"{prefix}_torque_constant"),
        "resistance": sp.Symbol(f"{prefix}_winding_resistance"),
        "axis": {axis: sp.Symbol(f"{prefix}_axis_{axis}") for axis in AXES},
    }


def _craft_rate_along(axis_symbols) -> sp.Expr:
    return sum((axis_symbols[a] * sp.Symbol(f"angular_velocity_{a}")
                for a in AXES), sp.Integer(0))


def wheel_relative_speed(index: int, momentum=None):
    """``W = h / I_w - a . w``: the rotor's speed relative to the craft."""
    s = wheel_symbols(index)
    h = s["momentum"] if momentum is None else momentum
    return h / s["inertia"] - _craft_rate_along(s["axis"])


def wheel_torque_rhs(index: int):
    """The motor torque the wheel delivers this step: the command held to
    the speed band (the torque that reaches +/-W_max exactly at the step's
    end) and to the motor's torque limit."""
    s = wheel_symbols(index)
    base = s["inertia"] * _craft_rate_along(s["axis"])
    upper = (base + s["inertia"] * s["max_speed"] - s["momentum"]) / DT
    lower = (base - s["inertia"] * s["max_speed"] - s["momentum"]) / DT
    banded = sp.Max(lower, sp.Min(upper, s["command"]))
    return sp.Max(-s["max_torque"], sp.Min(s["max_torque"], banded))


def wheel_rate_rhs(index: int, torque):
    """``eq_N1_6`` for the rotor about its own symmetry axis: ``d h / dt =
    tau_m`` (the w x I w term has no axial component for an axisymmetric
    rotor; checked to be the catalogue's I^-1 (tau - w x I w) shape)."""
    t = honorary.t
    (rate,) = sp.solve(honorary.eq_N1_6,
                       sp.Derivative(honorary.omega_B(t), t))
    crosses = [term for term in rate.atoms(sp.Function)
               if term.func == honorary.cross]
    if len(crosses) != 1:
        raise RuntimeError("eq_N1_6 no longer has one w x I w term")
    inertia = wheel_symbols(index)["inertia"]
    # the axial momentum rate: I_w * (axial angular acceleration)
    return inertia * rate.xreplace({crosses[0]: sp.Integer(0),
                                    honorary.tau_B(t): torque,
                                    honorary.I_B(t): inertia})


def wheel_power_rhs(index: int, torque, speed):
    """The motor law: eq_F9_4 rotary for the current, eq_F9_5 rotary for
    the back-EMF, eq_F8_11 (one winding) for the copper loss."""
    s = wheel_symbols(index)
    by_name = {sym.name: sym for law in (honorary.eq_F9_4, honorary.eq_F9_5,
                                         honorary.eq_F8_11)
               for sym in law.free_symbols}
    bl, i_vc = by_name["Bl"], by_name["i_{vc}"]
    # eq_F9_4 (F = Bl i) solved for the winding current, rotary
    (current,) = sp.solve(honorary.eq_F9_4, i_vc)
    current = current.xreplace({honorary.eq_F9_4.lhs: torque,
                                bl: s["torque_constant"]})
    # eq_F9_5 (v = Bl dx/dt), rotary: the back-EMF at the rotor's speed
    (derivative,) = honorary.eq_F9_5.rhs.atoms(sp.Derivative)
    emf = honorary.eq_F9_5.rhs.xreplace({derivative: speed,
                                         bl: s["torque_constant"]})
    # eq_F8_11, one winding (I_2 = 0): the copper loss
    loss = honorary.eq_F8_11.rhs.xreplace({
        by_name["I_1"]: current, by_name["R_1"]: s["resistance"],
        by_name["I_2"]: sp.Integer(0)})
    if loss.free_symbols - current.free_symbols - {s["resistance"]}:
        raise RuntimeError("eq_F8_11 left an unspelled winding symbol")
    return emf * current + loss


def wheel_columns(wheels) -> dict:
    """The wheels' columns (one lane): every declared number, zero
    momentum and zero command."""
    columns = {}
    for index, wheel in enumerate(wheels):
        prefix = f"wheel{index}"
        values = {"momentum": 0.0, "torque_command": 0.0,
                  "rotor_inertia": wheel.rotor_inertia_kg_m2,
                  "max_speed": wheel.max_speed_rad_s,
                  "max_torque": wheel.max_torque_n_m,
                  "torque_constant": wheel.torque_constant_n_m_a,
                  "winding_resistance": wheel.winding_resistance_ohm}
        for slot, axis in enumerate(AXES):
            values[f"axis_{axis}"] = float(wheel.axis[slot])
        for name, value in values.items():
            columns[f"{prefix}_{name}"] = np.full(1, float(value))
    return columns


def wheel_spin_inertia(wheels) -> np.ndarray:
    """``sum I_w a a^T``: the rotors' axial inertia, carried by the wheel
    momenta rather than by the craft's tensor."""
    out = np.zeros((3, 3))
    for wheel in wheels:
        a = np.asarray(wheel.axis, dtype=float)
        out += wheel.rotor_inertia_kg_m2 * np.outer(a, a)
    return out


# ------------------------------------------------- host-side machine wrench
def delivered_throttles(thrusters, throttles) -> np.ndarray:
    """The deadband applied to throttle states (or commands)."""
    u = np.asarray(throttles, dtype=float).reshape(len(thrusters))
    band = np.asarray([t.deadband for t in thrusters], dtype=float)
    return np.where(u >= band, u, 0.0)


def machine_wrench(thrusters, throttles, gimbal_rad, centre_of_mass_m,
                   attitude=None):
    """``(force (world), torque (craft, about c))`` delivered at the given
    throttle and gimbal STATES (deadband applied; full supply)."""
    rotation = (np.eye(3) if attitude is None
                else np.asarray(attitude, dtype=float).reshape(3, 3))
    centre = np.asarray(centre_of_mass_m, dtype=float).reshape(3)
    v = delivered_throttles(thrusters, throttles)
    angles = np.asarray(gimbal_rad, dtype=float).reshape(len(thrusters), 2)
    force = np.zeros(3)
    torque = np.zeros(3)
    for k, thruster in enumerate(thrusters):
        f = thruster.max_thrust_n * v[k] * thruster.direction_at(*angles[k])
        force += f
        torque += np.cross(np.asarray(thruster.position_m, float) - centre, f)
    return rotation @ force, torque


@dataclass(frozen=True)
class Allocation:
    """One allocation: the commands and the wrench they achieve.

    ``throttles`` (n,) and ``gimbal_rad`` (n, 2) are the COMMANDS (the
    states they bring the thrusters to within the round).  ``force_n``
    (world) and ``torque_n_m`` (craft frame, about the centre of mass) are
    the wrench delivered once the states arrive; the shortfalls are request
    minus achieved.  ``branches`` is the number of deadband on/off
    combinations solved."""

    throttles: np.ndarray
    gimbal_rad: np.ndarray
    force_n: np.ndarray
    torque_n_m: np.ndarray
    force_shortfall_n: np.ndarray
    torque_shortfall_n_m: np.ndarray
    cost: float
    branches: int
    #: the reaction wheels' motor torque COMMANDS (N m, on each rotor;
    #: the craft feels ``-tau a``); empty for a craft without wheels
    wheel_torque_n_m: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0))

    @property
    def wrench(self) -> tuple[np.ndarray, np.ndarray]:
        return self.force_n, self.torque_n_m

    @property
    def achieved(self) -> "AchievedWrench":
        """The achieved wrench in the tracker seam's shape (``.force_n``,
        ``.torque_n_m``)."""
        return AchievedWrench(self.force_n, self.torque_n_m)


@dataclass(frozen=True)
class AchievedWrench:
    """A force (N, world) and a torque (N m, craft frame, about the centre
    of mass): the shape of ``orbital_tracker.Wrench``."""

    force_n: np.ndarray
    torque_n_m: np.ndarray


#: The roles that make the craft's FORCE (thrust and its brake); the
#: ``navigation`` role (RCS) answers the attitude.
PROPULSIVE_ROLES = ("main", "brake")


def allocate_wrench(thrusters, force_n, torque_n_m, *, centre_of_mass_m,
                    attitude=None, throttle_state=None, gimbal_state=None,
                    tank_propellant_kg: dict | None = None,
                    force_direction_n=None,
                    round_s: float | None = None, force_weight: float = 1.0,
                    torque_weight: float = 1.0, fuel_weight: float = 1.0e-4,
                    max_branches: int = 64, wheels=(),
                    wheel_momentum_n_m_s=None, angular_velocity_rad_s=None,
                    power_weight: float = 1.0e-9) -> Allocation:
    """The commands that best achieve a wrench: ``force_n`` (world) and
    ``torque_n_m`` (craft frame, about ``centre_of_mass_m``).

    ``force_direction_n`` retains a pending translation direction when its
    requested magnitude is temporarily zero (for example while rotating into
    a burn).  Achieved force may be lateral but may not project backwards
    along that direction.  With no explicit direction, a nonzero ``force_n``
    supplies it.

    With ``wheels`` (:class:`ReactionWheel`), throttle, gimbal, navigation
    thrust and wheel motor torque are one bounded wrench solve.  A wheel's
    column is torque-only, ``[0; -axis]``; a thruster's column is its world
    force and moment about the live centre of mass.  The stored wheel
    momentum contributes ``-w x sum(h axis)`` to the achieved body torque.
    Each motor command is held to its torque limit and to the speed band it
    can reach over ``round_s``.  Above ``dump_fraction`` the existing dump
    command remains the preferred origin of the wheel decision, so a zero
    requested wrench does not silently cancel desaturation.

    A craft whose thrusters declare ROLES is allocated by them:

    * a propulsive set (``main`` OR ``brake``: the brake opposes the main
      thrust, and lighting both is propellant spent against itself) lights
      only when the FORCE asks for it -- decided by the same problem over
      that set with the torque unweighted;
    * if one lights, that set, the ``navigation`` set (RCS), its free
      gimbals and the wheels solve the requested force and torque together.
      RCS may supply translation or compensate a propulsive engine's
      lateral force and moment; every solve also starts from the geometric
      gimbal trim;
    * if none lights from force alone, navigation-only, main-assisted and
      brake-assisted answers are compared by that same wrench and resource
      cost; zero-force requests constrain every candidate to an exact
      force-free answer.

    A design without declared roles is the single problem below.
    """
    common = dict(centre_of_mass_m=centre_of_mass_m, attitude=attitude,
                  throttle_state=throttle_state, gimbal_state=gimbal_state,
                  tank_propellant_kg=tank_propellant_kg, round_s=round_s,
                  force_direction_n=force_direction_n,
                  force_weight=force_weight, torque_weight=torque_weight,
                  fuel_weight=fuel_weight, max_branches=max_branches,
                  wheels=tuple(wheels),
                  wheel_momentum_n_m_s=wheel_momentum_n_m_s,
                  angular_velocity_rad_s=angular_velocity_rad_s,
                  power_weight=power_weight)
    return _allocate_thrusters(thrusters, force_n, torque_n_m, **common)


def wheel_allocation(wheels, request, momentum, omega, *, round_s=None,
                     power_weight: float = 1.0e-6):
    """``(motor torques, w x sum h a)``: the wheels' share of the craft
    torque ``request`` (:func:`allocate_wrench`'s convention) by the compiled
    AbstractTensor bounded least-squares solve over each motor's box -- its torque
    limit and, over ``round_s``, its speed band -- with the dump torque of
    any wheel above its dump fraction added on top (the share is solved
    inside what the box leaves after the dump)."""
    from orbital_allocator_native import bounded_least_squares_solver

    axes = np.stack([np.asarray(w.axis, float) for w in wheels], axis=1)
    inertia = np.asarray([w.rotor_inertia_kg_m2 for w in wheels])
    top = np.asarray([w.max_speed_rad_s for w in wheels])
    limit = np.asarray([w.max_torque_n_m for w in wheels])
    gyroscopic = np.cross(omega, axes @ momentum)
    speed = momentum / inertia - axes.T @ omega
    low, high = -limit, limit.copy()
    if round_s is not None:
        h = float(round_s)
        low = np.maximum(low, inertia * (-top - speed) / h)
        high = np.minimum(high, inertia * (top - speed) / h)
        high = np.maximum(high, low)
    band = np.asarray([w.dump_fraction for w in wheels]) * top
    excess = speed - np.clip(speed, -band, band)
    dump = np.clip(-inertia * excess
                   / np.asarray([w.dump_time_s for w in wheels]), low, high)
    # the share: -A tau = request + w x A h, priced by the copper loss
    target = request + gyroscopic
    scale = float(limit.max())
    copper = np.asarray([w.winding_resistance_ohm
                         / w.torque_constant_n_m_a**2 for w in wheels])
    reference = float((copper * limit**2).max()) or 1.0
    rows = np.vstack([-axes / scale,
                      np.diag(np.sqrt(2.0 * power_weight * copper
                                      / reference))])
    rhs = np.concatenate([target / scale, np.zeros(len(wheels))])
    lower, upper = low - dump, high - dump
    solver = bounded_least_squares_solver(rows.shape[0], len(wheels))
    share = solver.solve(rows, rhs, lower, upper,
                         initial=np.clip(np.zeros(len(wheels)), lower, upper),
                         regularization=1.0e-12, tolerance=1.0e-12)
    return np.clip(share + dump, low, high), gyroscopic


def _allocate_thrusters(thrusters, force_n, torque_n_m, *, centre_of_mass_m,
                        attitude=None, throttle_state=None, gimbal_state=None,
                        tank_propellant_kg: dict | None = None,
                        force_direction_n=None,
                        round_s: float | None = None,
                        force_weight: float = 1.0, torque_weight: float = 1.0,
                        fuel_weight: float = 1.0e-4,
                        max_branches: int = 64, wheels=(),
                        wheel_momentum_n_m_s=None,
                        angular_velocity_rad_s=None,
                        power_weight: float = 1.0e-9) -> Allocation:
    """:func:`allocate_wrench` over the thrusters alone (its role
    structure, below)."""
    thrusters = tuple(thrusters)
    roles = {t.role for t in thrusters}
    navigation = [k for k, t in enumerate(thrusters)
                  if t.role == "navigation"]
    propulsive = [k for k, t in enumerate(thrusters)
                  if t.role in PROPULSIVE_ROLES]
    common = dict(centre_of_mass_m=centre_of_mass_m, attitude=attitude,
                  throttle_state=throttle_state, gimbal_state=gimbal_state,
                  round_s=round_s, force_weight=force_weight,
                  force_direction_n=force_direction_n,
                  torque_weight=torque_weight, fuel_weight=fuel_weight,
                  max_branches=max_branches, wheels=tuple(wheels),
                  wheel_momentum_n_m_s=wheel_momentum_n_m_s,
                  angular_velocity_rad_s=angular_velocity_rad_s,
                  power_weight=power_weight)
    if (not navigation or not propulsive
            or roles - {"navigation", *PROPULSIVE_ROLES}):
        return _allocate(thrusters, force_n, torque_n_m,
                         tank_propellant_kg=tank_propellant_kg, **common)
    target_f = np.asarray(force_n, dtype=float).reshape(3)
    # the brake opposes the main thrust: burning both at once is propellant
    # spent against itself, so a lit stage uses one set, the better kept
    sets = [[k for k in propulsive if thrusters[k].role == role]
            for role in PROPULSIVE_ROLES]
    sets = [group for group in sets if group]
    # does the FORCE light a propulsive set?  (torque unweighted: a set the
    # force alone leaves dark is not lit to make torque)
    force_only = dict(common, torque_weight=0.0, wheels=(),
                      wheel_momentum_n_m_s=None,
                      angular_velocity_rad_s=None)
    lighting = []
    for group in sets:
        push = _allocate(thrusters, target_f, np.zeros(3), active=group,
                         tank_propellant_kg=tank_propellant_kg, **force_only)
        if np.any(delivered_throttles(thrusters,
                                      push.throttles)[group] > 0.0):
            lighting.append(group)
    # Navigation-only, main-assisted and brake-assisted answers are evaluated
    # by the same constrained wrench solve.  The allocator, not a role
    # shortcut, decides which available actuator graph best answers the need.
    trials = [_allocate(thrusters, force_n, torque_n_m, active=navigation,
                        tank_propellant_kg=tank_propellant_kg, **common)]
    candidates = lighting if lighting else sets
    for group in candidates:
        # One graph: the propulsive gimbals, navigation translation/couples,
        # and reaction wheels may all compensate one another.  The propulsive
        # roles stay mutually exclusive; opposing engines are not burned
        # against each other merely to manufacture a moment.
        trials.append(_allocate(thrusters, force_n, torque_n_m,
                                active=group + navigation,
                                tank_propellant_kg=tank_propellant_kg,
                                **common))
    best = min(trials, key=lambda trial: trial.cost)
    return dataclasses.replace(
        best, branches=sum(trial.branches for trial in trials))


def _allocate(thrusters, force_n, torque_n_m, *, centre_of_mass_m,
              attitude=None, throttle_state=None, gimbal_state=None,
              tank_propellant_kg: dict | None = None,
              force_direction_n=None,
              round_s: float | None = None, force_weight: float = 1.0,
              torque_weight: float = 1.0, fuel_weight: float = 1.0e-4,
              max_branches: int = 64, active=None,
              net_zero=(), hold_gimbals: bool = False, wheels=(),
              wheel_momentum_n_m_s=None, angular_velocity_rad_s=None,
              power_weight: float = 1.0e-9) -> Allocation:
    """The exact problem below over the ``active`` thrusters (all when
    ``None``; the rest off).  ``hold_gimbals``: every gimbal
    stays at ``gimbal_state``.  ``net_zero``: fixed thrusters whose summed
    force must vanish (an equality: they may only make couples).

    Decision variables: each thruster's throttle, each gimballed thruster's
    two actuator angles ``(a, b)``, and each wheel's motor-torque share
    about its desaturation command.  Cost (the tracker's form):

        J = W_F |F(x) - F*|^2 / (2 T_max^2) + W_tau |tau(x) - tau*|^2
            / (2 tau_max^2) + fuel_weight sum_k q_k u_k / q_max

    with ``q_k`` thruster k's propellant flow at full throttle,
    ``T_k / (I_sp g_0)``, and ``q_max`` that of the strongest burner
    (decision 7: the price is the propellant consumption rate)
    with ``F = R sum_k T_k u_k d_k(a_k, b_k)`` and ``tau = sum_k (r_k - c)
    x T_k u_k d_k``.  Constraints, all exact:

    * throttle box: the declared range intersected, over the round
      ``round_s``, with the slew reach ``state +/- rate * round_s``;
    * deadband: a thruster whose reach straddles its deadband is either
      OFF (delivers nothing) or ON in ``[max(low, deadband), high]``; the
      choice is enumerated (``2^m`` branches for ``m`` such thrusters), each
      branch a smooth problem, the best kept;
    * gimbal: the actuator box ``state +/- slew * round_s`` inside the
      travel ``[-cone, cone]``, and the cone ``cos a cos b >= cos(cone)``;
    * fuel: per tank, ``round_s * sum_k share_kt * flow_k(u_k) <= tank``;
      a thruster with an empty feed tank delivers nothing.

    Each branch is solved by a fixed-topology bounded Gauss-Newton iteration.
    Its linear step is the repository's compiled AbstractTensor linear solve,
    with one prepared native execution reused for every branch and guidance
    retry of the same actuator topology.  Without ``round_s`` there is no
    slew or fuel limit (the static allocation).
    """
    from orbital_allocator_native import bounded_least_squares_solver

    thrusters = tuple(thrusters)
    count = len(thrusters)
    target_f = np.asarray(force_n, dtype=float).reshape(3)
    target_t = np.asarray(torque_n_m, dtype=float).reshape(3)
    rotation = (np.eye(3) if attitude is None
                else np.asarray(attitude, dtype=float).reshape(3, 3))
    centre = np.asarray(centre_of_mass_m, dtype=float).reshape(3)
    state = (np.zeros(count) if throttle_state is None
             else np.asarray(throttle_state, dtype=float).reshape(count))
    angles0 = (np.zeros((count, 2)) if gimbal_state is None
               else np.asarray(gimbal_state, dtype=float).reshape(count, 2))
    tanks = dict(tank_propellant_kg or {})
    h = None if round_s is None else float(round_s)
    if h is not None and not h > 0.0:
        raise ValueError("round_s must be positive")

    wheels = tuple(wheels)
    wheel_count = len(wheels)
    if wheel_count:
        axes = np.stack([np.asarray(w.axis, float) for w in wheels], axis=1)
        momentum = (np.zeros(wheel_count)
                    if wheel_momentum_n_m_s is None else
                    np.asarray(wheel_momentum_n_m_s, float).reshape(
                        wheel_count))
        omega = (np.zeros(3) if angular_velocity_rad_s is None else
                 np.asarray(angular_velocity_rad_s, float).reshape(3))
        wheel_inertia = np.asarray(
            [w.rotor_inertia_kg_m2 for w in wheels])
        wheel_top = np.asarray([w.max_speed_rad_s for w in wheels])
        wheel_limit = np.asarray([w.max_torque_n_m for w in wheels])
        wheel_speed = momentum / wheel_inertia - axes.T @ omega
        wheel_low, wheel_high = -wheel_limit, wheel_limit.copy()
        if h is not None:
            wheel_low = np.maximum(
                wheel_low, wheel_inertia * (-wheel_top - wheel_speed) / h)
            wheel_high = np.minimum(
                wheel_high, wheel_inertia * (wheel_top - wheel_speed) / h)
            wheel_high = np.maximum(wheel_high, wheel_low)
        wheel_band = np.asarray([w.dump_fraction for w in wheels]) * wheel_top
        wheel_excess = wheel_speed - np.clip(
            wheel_speed, -wheel_band, wheel_band)
        wheel_dump = np.clip(
            -wheel_inertia * wheel_excess
            / np.asarray([w.dump_time_s for w in wheels]),
            wheel_low, wheel_high)
        wheel_share_low = wheel_low - wheel_dump
        wheel_share_high = wheel_high - wheel_dump
        wheel_gyroscopic = np.cross(omega, axes @ momentum)
        wheel_copper = np.asarray([
            w.winding_resistance_ohm / w.torque_constant_n_m_a**2
            for w in wheels])
        wheel_reference = float(
            (wheel_copper * wheel_limit**2).max()) or 1.0
        wheel_price = power_weight * wheel_copper / wheel_reference
    else:
        axes = np.zeros((3, 0))
        wheel_dump = np.zeros(0)
        wheel_share_low = wheel_share_high = np.zeros(0)
        wheel_gyroscopic = np.zeros(3)
        wheel_price = np.zeros(0)

    thrust = np.asarray([t.max_thrust_n for t in thrusters], dtype=float)
    arms = np.asarray([np.asarray(t.position_m, float) - centre
                       for t in thrusters]).reshape(count, 3)
    force_scale = float(thrust.max()) if count else 1.0
    torque_scale = max((float(np.linalg.norm(np.cross(
        arms[k], thrust[k] * np.asarray(t.direction, float))))
        for k, t in enumerate(thrusters)), default=0.0) or 1.0
    wf = math.sqrt(force_weight) / force_scale
    wt = math.sqrt(torque_weight) / torque_scale
    flow = np.asarray([t.max_thrust_n
                       * t.thruster_kind.propellant_per_impulse_kg_n_s
                       for t in thrusters], dtype=float)
    # Decision 7 prices the propellant consumption RATE: a thruster costs
    # its TS1.2 flow per unit throttle, |F| / (I_sp g_0), relative to the
    # flow of the strongest propellant-burning thruster at full throttle
    # (so ``fuel_weight`` is still a dimensionless fraction of the largest
    # burner).  A reactionless thruster burns nothing and costs nothing; a
    # craft with no propellant at all is priced per newton, as before.
    burners = [k for k in range(count) if flow[k] > 0.0]
    if burners:
        reference = flow[max(burners, key=lambda k: thrust[k])]
        fuel_price = fuel_weight * flow / reference
    else:
        fuel_price = fuel_weight * thrust / force_scale

    low = np.empty(count)
    high = np.empty(count)
    alive = np.ones(count, dtype=bool)
    for k, t in enumerate(thrusters):
        lo, hi = t.throttle_min, t.throttle_max
        if h is not None and math.isfinite(t.throttle_slew_per_s):
            reach = t.throttle_slew_per_s * h
            lo, hi = max(lo, state[k] - reach), min(hi, state[k] + reach)
        low[k], high[k] = lo, max(lo, hi)
        if flow[k] > 0.0 and any(tanks.get(tank, math.inf) <= 0.0
                                 for tank, _ in t.feeds):
            alive[k] = False
    gimbals = [k for k, t in enumerate(thrusters)
               if t.gimballed and not hold_gimbals]
    gimbal_box = {}
    for k in gimbals:
        t = thrusters[k]
        cone = t.cone_half_angle_rad
        reach = (math.inf if h is None or math.isinf(t.gimbal_slew_rad_s)
                 else t.gimbal_slew_rad_s * h)
        gimbal_box[k] = [(max(-cone, angles0[k, j] - reach),
                          min(cone, angles0[k, j] + reach)) for j in (0, 1)]

    # the deadband's discrete choice: "free" (no deadband, one continuous
    # range through zero), "on", "off", or both of the last two
    options = []
    for k, t in enumerate(thrusters):
        band = t.deadband
        if not alive[k] or (active is not None and k not in active):
            options.append(("off",))
        elif band <= 0.0:
            options.append(("free",))
        else:
            can_on = high[k] >= max(low[k], band)
            can_off = low[k] < band
            options.append(tuple(name for name, ok in (("off", can_off),
                                                       ("on", can_on)) if ok)
                           or ("off",))
    split = [k for k, o in enumerate(options) if len(o) == 2]
    if 2 ** len(split) > max_branches:
        raise ValueError(f"{len(split)} thrusters straddle their deadband: "
                         f"{2 ** len(split)} branches > {max_branches}")

    def solve_branch(choice):
        on = [k for k in range(count) if choice[k] != "off"]
        u_bounds = {k: ((max(low[k], thrusters[k].deadband), high[k])
                        if choice[k] == "on" else (low[k], high[k]))
                    for k in on}
        g_on = [k for k in gimbals if k in u_bounds]
        n_u, n_g, n_w = len(on), 2 * len(g_on), wheel_count
        bounds = [u_bounds[k] for k in on]
        for k in g_on:
            bounds.extend(gimbal_box[k])
        bounds.extend(zip(wheel_share_low, wheel_share_high))

        def unpack(x):
            u = np.zeros(count)
            ang = angles0.copy()
            for i, k in enumerate(on):
                u[k] = x[i]
            for i, k in enumerate(g_on):
                ang[k] = x[n_u + 2 * i:n_u + 2 * i + 2]
            share = np.asarray(x[n_u + n_g:n_u + n_g + n_w])
            return u, ang, share

        price = np.asarray([fuel_price[k] for k in on])

        # The wrench is linear in the throttles: column i of ``columns`` is
        # thruster ``on[i]``'s weighted (world force, craft torque) per unit
        # throttle.  Fixed thrusters' columns are constants; a gimballed
        # thruster's column is rebuilt from its angles each evaluation.
        columns = np.zeros((6, n_u))
        for i, k in enumerate(on):
            f = thrust[k] * thrusters[k].direction_at(*angles0[k])
            columns[:3, i] = wf * (rotation @ f)
            columns[3:, i] = wt * np.cross(arms[k], f)
        target = np.concatenate([
            wf * target_f,
            wt * (target_t + wheel_gyroscopic + axes @ wheel_dump)])
        slot_of = {k: i for i, k in enumerate(on)}
        frames = [thrusters[k].gimbal_frame() for k in g_on]

        def wrench_jacobian(x):
            u = x[:n_u]
            jac = np.zeros((6, n_u + n_g + n_w))
            jac[:, :n_u] = columns
            jac[3:, n_u + n_g:] = -wt * axes
            for i, k in enumerate(g_on):
                n, e1, e2 = frames[i]
                a, b = x[n_u + 2 * i], x[n_u + 2 * i + 1]
                ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b),                     math.sin(b)
                d = ca * cb * n + ca * sb * e1 - sa * e2
                da = -sa * cb * n - sa * sb * e1 - ca * e2
                db = -ca * sb * n + ca * cb * e1
                column = slot_of[k]
                scale = thrust[k]
                for j, dd in ((None, d), (0, da), (1, db)):
                    f = scale * dd if j is None else scale * u[column] * dd
                    r = arms[k]
                    block = np.concatenate([
                        wf * (rotation @ f),
                        wt * np.asarray((r[1] * f[2] - r[2] * f[1],
                                         r[2] * f[0] - r[0] * f[2],
                                         r[0] * f[1] - r[1] * f[0]))])
                    if j is None:
                        jac[:, column] = block
                    else:
                        jac[:, n_u + 2 * i + j] = block
            return jac

        def fun(x):
            u = x[:n_u]
            share = x[n_u + n_g:n_u + n_g + n_w]
            jac = wrench_jacobian(x)
            residual = (jac[:, :n_u] @ u
                        + jac[:, n_u + n_g:] @ share - target)
            value = (0.5 * float(residual @ residual) + float(price @ u)
                     + float(wheel_price @ (share * share)))
            grad = jac.T @ residual
            grad[:n_u] += price
            grad[n_u + n_g:] += 2.0 * wheel_price * share
            return value, grad

        constraints = []
        # Rotation retains the pending translation direction even while its
        # force magnitude is gated to zero.  The solve remains free to make
        # torque with every actuator, but incidental thrust may never oppose
        # that direction.  The constraint uses the same live gimballed force
        # columns as the objective, rather than nominal thruster directions.
        direction_force = (target_f if force_direction_n is None else
                           np.asarray(force_direction_n, dtype=float).reshape(3))
        direction_force_size = float(np.linalg.norm(direction_force))

        def achieved_force_and_jac(x):
            u = x[:n_u]
            jac = wrench_jacobian(x)
            return (jac[:3, :n_u] @ u / wf,
                    jac[:3, :] / wf)

        if direction_force_size > 0.0:
            requested_direction = direction_force / direction_force_size

            def forward_force(x, direction=requested_direction):
                force, _jac = achieved_force_and_jac(x)
                return np.asarray([direction @ force])

            def forward_force_jac(x, direction=requested_direction):
                _force, jac = achieved_force_and_jac(x)
                return (direction @ jac)[None, :]

            constraints.append({"type": "ineq", "fun": forward_force,
                                "jac": forward_force_jac})
        for i, k in enumerate(g_on):
            cos_cone = math.cos(thrusters[k].cone_half_angle_rad)
            slot = n_u + 2 * i

            def cone(x, slot=slot, cos_cone=cos_cone):
                return np.asarray([math.cos(x[slot]) * math.cos(x[slot + 1])
                                   - cos_cone])

            def cone_jac(x, slot=slot):
                row = np.zeros(n_u + n_g + n_w)
                row[slot] = -math.sin(x[slot]) * math.cos(x[slot + 1])
                row[slot + 1] = -math.cos(x[slot]) * math.sin(x[slot + 1])
                return row[None, :]
            constraints.append({"type": "ineq", "fun": cone,
                                "jac": cone_jac})
        couple = [i for i, k in enumerate(on) if k in set(net_zero)]
        if couple:
            rows = np.zeros((3, n_u + n_g + n_w))
            for i in couple:
                rows[:, i] = thrust[on[i]] * np.asarray(
                    thrusters[on[i]].direction, dtype=float)
            constraints.append({"type": "eq",
                                "fun": lambda x, rows=rows: rows @ x,
                                "jac": lambda x, rows=rows: rows})
        budgets = []
        if h is not None:
            for tank, kg in tanks.items():
                row = np.zeros(n_u + n_g + n_w)
                for i, k in enumerate(on):
                    for name, share in thrusters[k].feeds:
                        if name == tank:
                            row[i] += h * share * flow[k]
                if np.any(row):
                    budgets.append((row, float(kg)))
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda x, row=row, kg=kg: np.asarray(
                            [kg - row @ x]),
                        "jac": lambda x, row=row: -row[None, :]})
        if not bounds:
            u, ang, share = unpack(np.zeros(0))
            value = fun(np.zeros(0))[0]
        else:
            lower = np.asarray([b[0] for b in bounds])
            upper = np.asarray([b[1] for b in bounds])
            warm = np.concatenate([np.asarray([state[k] for k in on]),
                                   *[angles0[k] for k in g_on],
                                   np.zeros(n_w)])
            warm = np.clip(warm, lower, upper)
            centred = warm.copy()
            centred[n_u:n_u + n_g] = np.clip(
                0.0, lower[n_u:n_u + n_g], upper[n_u:n_u + n_g])
            # the geometric trim: each gimballed engine's line through the
            # centre of mass (a start, not an answer)
            trim = warm.copy()
            for i, k in enumerate(g_on):
                trim[n_u + 2 * i:n_u + 2 * i + 2] = _trim_angles(
                    thrusters[k], -arms[k])
            trim = np.clip(trim, lower, upper)
            best = None
            # Every role/deadband branch pads to the same fixed actuator
            # topology.  The compiled solver is therefore built once for the
            # craft rather than once for each active subset.
            solver_columns = count + 2 * len(gimbals) + wheel_count
            solver = bounded_least_squares_solver(6, solver_columns)
            starts = [warm]
            for extra in (centred, trim):
                if not any(np.array_equal(extra, seen) for seen in starts):
                    starts.append(extra)
            for x0 in starts:
                x = np.asarray(x0, dtype=float).copy()
                for _iteration in range(12):
                    jac = wrench_jacobian(x)
                    u_now = x[:n_u]
                    share_now = x[n_u + n_g:n_u + n_g + n_w]
                    achieved_weighted = (jac[:, :n_u] @ u_now
                                         + jac[:, n_u + n_g:] @ share_now)
                    residual = target - achieved_weighted
                    padded = np.zeros((6, solver_columns))
                    padded[:, :x.size] = jac
                    step_low = np.zeros(solver_columns)
                    step_high = np.zeros(solver_columns)
                    step_low[:x.size] = lower - x
                    step_high[:x.size] = upper - x
                    linear = np.zeros(solver_columns)
                    linear[:n_u] = price
                    if n_w:
                        linear[n_u + n_g:n_u + n_g + n_w] = (
                            2.0 * wheel_price * share_now)
                    delta = solver.solve(
                        padded, residual, step_low, step_high,
                        initial=np.zeros(solver_columns),
                        regularization=1.0e-12, linear_term=linear,
                        tolerance=1.0e-12)[:x.size]
                    if not np.all(np.isfinite(delta)):
                        break
                    # A pending translational plan is an inequality, not a
                    # force target while the craft is still turning.  Limit
                    # the Newton step at its zero-projection boundary instead
                    # of allowing a torque solution to thrust backwards.
                    if direction_force_size > 0.0:
                        direction = direction_force / direction_force_size
                        force_now, force_jac = achieved_force_and_jac(x)
                        projection = float(direction @ force_now)
                        slope = float(direction @ (force_jac @ delta))
                        if slope < 0.0 and projection + slope < 0.0:
                            delta *= max(0.0, min(1.0,
                                projection / -slope))
                    before = fun(x)[0]
                    accepted = None
                    scale = 1.0
                    for _line_search in range(12):
                        trial = np.clip(x + scale * delta, lower, upper)
                        trial = _onto_cones(trial, n_u, g_on, thrusters)
                        trial = _within_budgets(trial, n_u, budgets)
                        if direction_force_size > 0.0:
                            force_trial, _ = achieved_force_and_jac(trial)
                            if float(direction @ force_trial) < -1.0e-10:
                                scale *= 0.5
                                continue
                        trial_value = fun(trial)[0]
                        if trial_value <= before:
                            accepted = (trial_value, trial)
                            break
                        scale *= 0.5
                    if accepted is None:
                        break
                    trial_value, trial = accepted
                    motion = float(np.max(np.abs(trial - x)))
                    x = trial
                    if motion <= 1.0e-11 or before - trial_value <= 1.0e-14:
                        break
                value = fun(x)[0]
                if best is None or value < best[0]:
                    best = (value, x)
            value, x = best
            u, ang, share = unpack(x)
        for k in range(count):
            if choice[k] == "off":
                u[k] = thrusters[k].throttle_min
        return value, u, ang, share

    best = None
    branches = 0
    for bits in range(2 ** len(split)):
        choice = [o[0] for o in options]
        for position, k in enumerate(split):
            choice[k] = "on" if bits >> position & 1 else "off"
        value, u, ang, share = solve_branch(choice)
        branches += 1
        if best is None or value < best[0]:
            best = (value, u, ang, share, choice)
    value, u, ang, share, choice = best
    delivered = np.where([c != "off" for c in choice], u, 0.0)
    force, thruster_torque = machine_wrench(
        thrusters, delivered, ang, centre, rotation)
    wheel_torque = wheel_dump + share
    torque = (thruster_torque - axes @ wheel_torque
              - wheel_gyroscopic)
    return Allocation(throttles=u, gimbal_rad=ang, force_n=force,
                      torque_n_m=torque,
                      force_shortfall_n=target_f - force,
                      torque_shortfall_n_m=target_t - torque,
                      cost=float(value), branches=branches,
                      wheel_torque_n_m=wheel_torque)


def _trim_angles(thruster: Thruster, toward) -> np.ndarray:
    """The Cardan angles ``(a, b)`` that point ``thruster`` along
    ``toward`` (craft frame), inverted from ``direction_at``:
    ``sin a = -d . e2`` and ``tan b = (d . e1) / (d . n)``."""
    n, e1, e2 = thruster.gimbal_frame()
    d = np.asarray(toward, dtype=float)
    d = d / np.linalg.norm(d)
    a = math.asin(max(-1.0, min(1.0, -float(d @ e2))))
    b = math.atan2(float(d @ e1), float(d @ n))
    return np.asarray((a, b))


def _onto_cones(x, n_u, g_on, thrusters):
    """Pull a gimbal the solver left a rounding outside its cone back onto
    it along its own ray (the cone region is convex and contains the
    centre)."""
    x = x.copy()
    for i, k in enumerate(g_on):
        slot = n_u + 2 * i
        cos_cone = math.cos(thrusters[k].cone_half_angle_rad)
        a, b = x[slot], x[slot + 1]
        if math.cos(a) * math.cos(b) >= cos_cone:
            continue
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if math.cos(mid * a) * math.cos(mid * b) >= cos_cone:
                lo = mid
            else:
                hi = mid
        x[slot], x[slot + 1] = lo * a, lo * b
    return x


def _within_budgets(x, n_u, budgets):
    """Scale the throttles down onto a tank budget the solver overshot by a
    rounding (each budget is linear in the throttles, through zero)."""
    x = x.copy()
    for row, kg in budgets:
        used = float(row @ x)
        if used > kg and used > 0.0:
            x[:n_u] *= kg / used
    return x
