"""Orbital craft, build step 3a: the fixed flight plan (a Hohmann transfer).

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``.  This
module is self-contained: it knows nothing about the jumper, the dt system
or the tracker.  Its public surface is

    HohmannPlan                 frozen record of one planned transfer
    hohmann_plan(mu, r1, r2)    builds it (one compiled plan piece)
    reference(plan, t)          (r, v) the plan prescribes at time t
    kepler_residual(plan, t)    the final Newton stage's residual there
    two_body_acceleration(mu, positions)
                                the plan's acceleration law at any points
    KEPLER_LAWS, KEPLER_SCALE   the laws (eq_KE*_*) and their validity

Laws (section KE, "Kepler").  Every law is derived from the original
symbolic orbital library ``turing/src/transmogrifier/orbital.py`` (decision
5), imported and used unchanged:

    eq_KE1_1  a_t = (r_1 + r_2)/2       apsides of Orbit.orbital_radius_theta
    eq_KE1_2  e_t = (r_a - r_p)/(r_a + r_p)        (theta = 0 and theta = pi)
    eq_KE1_3  v = sqrt(mu (2/r - 1/a))  Orbit.vis_viva, positive root
    eq_KE1_4  dv_1 = v(r_1; a_t) - v(r_1; r_1)     burn 1 (vis-viva twice)
    eq_KE1_5  dv_2 = v(r_2; r_2) - v(r_2; a_t)     burn 2
    eq_KE1_6  n = sqrt(mu / a^3)        mean motion (Kepler's third law; the
                                        one law the original does not state)
    eq_KE1_7  t_transfer = M(E = pi) / n(a_t)      Orbit.kepler_equation
    eq_KE2_1  M = n (t - t_p)           mean anomaly since periapsis
    eq_KE2_2  E_k1 = E_k - f / f'       ONE Newton stage on Orbit.kepler_equation
                                        (f = E - e sin E - M, f' = df/dE);
                                        iterate symbols explicit, as the
                                        catalogue's N11 stages
    eq_KE2_3  kepler_residual = |E_k1 - e sin E_k1 - M|
    eq_KE2_4  cos_nu = (cos E - e) / (1 - e cos E)   true anomaly, by its
    eq_KE2_5  sin_nu = sqrt(1 - e^2) sin E / (1 - e cos E)   cosine and sine
    eq_KE2_6  r = Orbit.orbital_radius_theta with cos(theta) = cos_nu
    eq_KE2_7  position_x = r (cos_nu cos w - sin_nu sin w)   (w = omega_p,
    eq_KE2_8  position_y = r (sin_nu cos w + cos_nu sin w)    periapsis angle)
    eq_KE2_9  velocity_x = d position_x / dE * dE/dt, with dE/dt = n / f'
    eq_KE2_10 velocity_y = d position_y / dE * dE/dt     (Kepler's equation
                                        differentiated in time; derived here)
    eq_KE2_11..13 acceleration_* = -dH/d position_*, H = Orbit.hamiltonian

The Newton stage is ONE law; the iteration belongs to the consumer
(:func:`reference`), which applies it ``KEPLER_STAGES`` times from the
declared starter ``E_0 = M`` and refuses a result whose residual exceeds
``KEPLER_RESIDUAL_BOUND``.  Measured: 6 stages from ``E_0 = M`` leave a max
residual of 4.4e-16 for e <= 0.85 on a 400k-point M grid (2.3e-12 at
e = 0.9), hence ``KEPLER_MAX_ECCENTRICITY = 0.85``, a validity predicate on
``KEPLER_SCALE`` that :func:`hohmann_plan` checks.

Geometry.  The plan lies in the xy plane of the central body's frame (body
at the origin), prograde = counterclockwise about +z.  Three legs, each a
Kepler conic (a, e, w, t_p):

    t <  t_burn1             circle r1:  (r1, 0, phase, t_burn1)
    t_burn1 <= t < t_burn2   transfer:   (a_t, e_t, w_t, t_p)
    t >= t_burn2             circle r2:  (r2, 0, phase + pi, t_burn2)

``phase`` is the craft's polar angle at burn 1.  Raising (r2 > r1): the
burn is at periapsis, ``w_t = phase``, ``t_p = t_burn1``.  Lowering: the
burn is at apoapsis, ``w_t = phase + pi``, ``t_p = t_burn1 - t_transfer``.

Every evaluation runs compiled ``equation_piece``s (batched): one plan piece
(batch 1) and four reference pieces (batch ``REFERENCE_BATCH``; longer t
arrays are evaluated in chunks).
"""
from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

from src.transmogrifier.orbital import Orbit

from honorary_engine_equation_catalogue import LawScale, equation_piece

#: Newton stages per evaluation, from the declared starter E_0 = M.
KEPLER_STAGES = 6
#: Largest |E - e sin E - M| a reference evaluation may return.
KEPLER_RESIDUAL_BOUND = 1.0e-12
#: Largest eccentricity the stage count is validated for (measured).
KEPLER_MAX_ECCENTRICITY = 0.85
#: Samples per compiled reference call.
REFERENCE_BATCH = 64

_AXES = ("x", "y", "z")


def kepler_laws(orbit=Orbit) -> dict:
    """The KE section: ``{'eq_KE1_1': Eq, ...}`` derived from ``orbit``
    (the original library's ``Orbit``), catalogue spelling."""
    mu = sp.Symbol("mu", positive=True)
    a, e, r = sp.symbols("a e r", positive=True)
    r_1, r_2, r_p, r_a = sp.symbols("r_1 r_2 r_p r_a", positive=True)
    a_t, e_t, n = sp.symbols("a_t e_t n", positive=True)
    t, t_p, w = sp.symbols("t t_p omega_p", real=True)
    M, E, E_k, E_k1 = sp.symbols("M E E_k E_k1", real=True)
    theta = sp.Symbol("theta", real=True)
    cos_nu, sin_nu = sp.symbols("cos_nu sin_nu", real=True)
    eqs = {}

    # KE1: the transfer ellipse from its apsides (orbital_radius_theta at
    # theta = 0 is periapsis, at theta = pi apoapsis)
    shape = orbit.orbital_radius_theta(a, e, theta)
    (apsides,) = sp.solve([sp.Eq(shape.rhs.subs(theta, 0), r_p),
                           sp.Eq(shape.rhs.subs(theta, sp.pi), r_a)],
                          [a, e], dict=True)
    low, high = sp.Min(r_1, r_2), sp.Max(r_1, r_2)
    # a is symmetric in the apsides: either burn radius may be periapsis
    eqs["eq_KE1_1"] = sp.Eq(a_t, apsides[a].xreplace({r_p: r_1, r_a: r_2}))
    eqs["eq_KE1_2"] = sp.Eq(e_t, apsides[e].xreplace({r_p: low, r_a: high}))
    vis = orbit.vis_viva(mu, a, r)
    (v,) = vis.lhs.free_symbols
    if vis.lhs != v**2:
        raise RuntimeError("Orbit.vis_viva is no longer v**2 = ...")
    eqs["eq_KE1_3"] = sp.Eq(v, sp.sqrt(vis.rhs))
    speed = eqs["eq_KE1_3"].rhs
    eqs["eq_KE1_4"] = sp.Eq(sp.Symbol("dv_1", real=True),
                            speed.xreplace({r: r_1, a: a_t})
                            - speed.xreplace({r: r_1, a: r_1}))
    eqs["eq_KE1_5"] = sp.Eq(sp.Symbol("dv_2", real=True),
                            speed.xreplace({r: r_2, a: r_2})
                            - speed.xreplace({r: r_2, a: a_t}))
    eqs["eq_KE1_6"] = sp.Eq(n, sp.sqrt(mu / a**3))
    kepler = orbit.kepler_equation(M, E, e)
    # half a period: M advances by Kepler's equation at E = pi
    eqs["eq_KE1_7"] = sp.Eq(sp.Symbol("t_transfer", positive=True),
                            kepler.rhs.subs(E, sp.pi)
                            / eqs["eq_KE1_6"].rhs.xreplace({a: a_t}))

    # KE2: the reference state at time t on one Kepler leg
    eqs["eq_KE2_1"] = sp.Eq(M, n * (t - t_p))
    residual = kepler.rhs - kepler.lhs                  # E - e sin E - M
    stage = residual.xreplace({E: E_k})
    eqs["eq_KE2_2"] = sp.Eq(E_k1, E_k - stage / sp.diff(stage, E_k))
    eqs["eq_KE2_3"] = sp.Eq(sp.Symbol("kepler_residual", real=True),
                            sp.Abs(residual.xreplace({E: E_k1})))
    eqs["eq_KE2_4"] = sp.Eq(cos_nu, (sp.cos(E) - e) / (1 - e * sp.cos(E)))
    eqs["eq_KE2_5"] = sp.Eq(sin_nu, sp.sqrt(1 - e**2) * sp.sin(E)
                            / (1 - e * sp.cos(E)))
    if shape.lhs != r:
        raise RuntimeError("Orbit.orbital_radius_theta no longer gives r")
    eqs["eq_KE2_6"] = sp.Eq(r, shape.rhs.xreplace({sp.cos(theta): cos_nu}))
    position = {
        "x": r * (cos_nu * sp.cos(w) - sin_nu * sp.sin(w)),
        "y": r * (sin_nu * sp.cos(w) + cos_nu * sp.sin(w)),
    }
    eqs["eq_KE2_7"] = sp.Eq(sp.Symbol("position_x", real=True), position["x"])
    eqs["eq_KE2_8"] = sp.Eq(sp.Symbol("position_y", real=True), position["y"])
    # velocity: d/dt = dE/dt d/dE; Kepler's equation in time gives
    # dM/dt = n = f'(E) dE/dt
    rate = n / sp.diff(residual, E)
    in_E = {r: eqs["eq_KE2_6"].rhs}
    anomaly = {cos_nu: eqs["eq_KE2_4"].rhs, sin_nu: eqs["eq_KE2_5"].rhs}
    for index, axis in ((9, "x"), (10, "y")):
        along = position[axis].xreplace(in_E).xreplace(anomaly)
        eqs[f"eq_KE2_{index}"] = sp.Eq(sp.Symbol(f"velocity_{axis}",
                                                 real=True),
                                       sp.diff(along, E) * rate)
    coordinates = sp.Matrix([sp.Symbol(f"position_{axis}", real=True)
                             for axis in _AXES])
    momenta = sp.Matrix([sp.Symbol(f"velocity_{axis}", real=True)
                         for axis in _AXES])
    hamiltonian = orbit.hamiltonian(coordinates, momenta, mu)
    for index, axis in zip((11, 12, 13), _AXES):
        eqs[f"eq_KE2_{index}"] = sp.Eq(
            sp.Symbol(f"acceleration_{axis}", real=True),
            -sp.diff(hamiltonian, sp.Symbol(f"position_{axis}", real=True)))
    return eqs


KEPLER_LAWS = kepler_laws()
globals().update(KEPLER_LAWS)

KEPLER_SCALE = LawScale(
    "Keplerian two-body reference, fixed Newton stage count",
    tuple(KEPLER_LAWS),
    (sp.Symbol("e") >= 0, sp.Symbol("e") <= KEPLER_MAX_ECCENTRICITY),
    source=(f"{KEPLER_STAGES} Newton stages from E_0 = M leave "
            f"|E - e sin E - M| <= 4.4e-16 for e <= "
            f"{KEPLER_MAX_ECCENTRICITY} (measured, 400k-point M grid); "
            "point masses, no perturbations"),
)


def _closed(expression, *law_names):
    """``expression`` with the named laws' left sides replaced by their
    right sides until none remains (composition, not new physics)."""
    replacements = {KEPLER_LAWS[name].lhs: KEPLER_LAWS[name].rhs
                    for name in law_names}
    while True:
        composed = expression.xreplace(replacements)
        if composed == expression:
            return composed
        expression = composed


def _output(name: str, law_name: str, *composing):
    return sp.Eq(sp.Symbol(name), _closed(KEPLER_LAWS[law_name].rhs,
                                          *composing), evaluate=False)


# ------------------------------------------------------------- the pieces
@functools.lru_cache(maxsize=None)
def _plan_piece():
    return equation_piece("orbital_plan_hohmann", (
        _output("a_t", "eq_KE1_1"),
        _output("e_t", "eq_KE1_2"),
        _output("dv_1", "eq_KE1_4", "eq_KE1_1"),
        _output("dv_2", "eq_KE1_5", "eq_KE1_1"),
        _output("t_transfer", "eq_KE1_7", "eq_KE1_1"),
    ), batch=1)


@functools.lru_cache(maxsize=None)
def _reference_pieces():
    batch = REFERENCE_BATCH
    mean_anomaly = equation_piece("orbital_plan_mean_anomaly", (
        _output("M", "eq_KE2_1", "eq_KE1_6"),), batch=batch)
    stage = equation_piece("orbital_plan_kepler_stage", (
        _output("E_k1", "eq_KE2_2"),
        _output("kepler_residual", "eq_KE2_3", "eq_KE2_2"),
    ), batch=batch)
    state = equation_piece("orbital_plan_state", tuple(
        _output(name, law, "eq_KE2_6", "eq_KE2_4", "eq_KE2_5", "eq_KE1_6")
        for name, law in (("position_x", "eq_KE2_7"),
                          ("position_y", "eq_KE2_8"),
                          ("velocity_x", "eq_KE2_9"),
                          ("velocity_y", "eq_KE2_10"))), batch=batch)
    gravity = equation_piece("orbital_plan_two_body", tuple(
        _output(f"acceleration_{axis}", f"eq_KE2_{index}")
        for index, axis in zip((11, 12, 13), _AXES)), batch=batch)
    return mean_anomaly, stage, state, gravity


def _call(piece, columns: dict) -> dict:
    """Run a compiled piece on named columns; outputs by name (copied)."""
    outputs = piece(*(np.ascontiguousarray(columns[name], dtype=np.float64)
                      for name in piece.argument_names))
    return {name: np.array(value, dtype=np.float64)
            for name, value in zip(piece.output_names, outputs)}


# ---------------------------------------------------------------- the plan
@dataclass(frozen=True)
class HohmannPlan:
    """A planned Hohmann transfer between coplanar circular orbits.

    Times in s, radii in m, speeds in m/s; ``dv1``/``dv2`` are signed along
    the prograde direction (negative = retrograde, a lowering transfer).
    """

    mu: float
    r1: float
    r2: float
    phase: float
    t_burn1: float
    t_burn2: float
    dv1: float
    dv2: float
    a_transfer: float
    e_transfer: float
    transfer_time: float

    @property
    def ideal_delta_v(self) -> float:
        """|dv1| + |dv2|: the impulsive transfer's total delta-v."""
        return abs(self.dv1) + abs(self.dv2)

    def legs(self):
        """((t_start, a, e, w, t_p), ...) for the three Kepler legs."""
        raising = self.r2 >= self.r1
        transfer = (self.a_transfer, self.e_transfer,
                    self.phase if raising else self.phase + np.pi,
                    self.t_burn1 if raising
                    else self.t_burn1 - self.transfer_time)
        return ((-np.inf, self.r1, 0.0, self.phase, self.t_burn1),
                (self.t_burn1, *transfer),
                (self.t_burn2, self.r2, 0.0, self.phase + np.pi,
                 self.t_burn2))


def hohmann_plan(mu: float, r1: float, r2: float, *,
                 t_burn1: float = 0.0, phase: float = 0.0) -> HohmannPlan:
    """The Hohmann transfer r1 -> r2 with burn 1 at ``t_burn1`` at polar
    angle ``phase``; refuses a transfer outside ``KEPLER_SCALE``."""
    if not (mu > 0.0 and r1 > 0.0 and r2 > 0.0):
        raise ValueError("mu, r1 and r2 must be positive")
    values = _call(_plan_piece(), {"mu": [mu], "r_1": [r1], "r_2": [r2]})
    e_t = float(values["e_t"][0])
    if not all(KEPLER_SCALE.validity({"e": e_t})):
        raise ValueError(f"transfer eccentricity {e_t} is outside the "
                         f"validated Kepler stage domain "
                         f"(e <= {KEPLER_MAX_ECCENTRICITY})")
    transfer_time = float(values["t_transfer"][0])
    return HohmannPlan(
        mu=float(mu), r1=float(r1), r2=float(r2), phase=float(phase),
        t_burn1=float(t_burn1), t_burn2=float(t_burn1) + transfer_time,
        dv1=float(values["dv_1"][0]), dv2=float(values["dv_2"][0]),
        a_transfer=float(values["a_t"][0]), e_transfer=e_t,
        transfer_time=transfer_time)


def _chunks(values: np.ndarray):
    """(start, stop, padded batch) over ``values`` in REFERENCE_BATCH."""
    for start in range(0, values.size, REFERENCE_BATCH):
        chunk = values[start:start + REFERENCE_BATCH]
        padded = np.full(REFERENCE_BATCH, chunk[-1])
        padded[:chunk.size] = chunk
        yield start, start + chunk.size, padded


def _solve_kepler(plan: HohmannPlan, batch_t: np.ndarray):
    """Leg columns at ``batch_t`` with the mean anomaly, then
    ``KEPLER_STAGES`` applications of the stage law from E_0 = M.
    Returns (columns, E, final residual)."""
    mean_anomaly, stage, _, _ = _reference_pieces()
    legs = plan.legs()
    starts = np.asarray([leg[0] for leg in legs])
    elements = np.asarray([leg[1:] for leg in legs])
    leg = elements[np.searchsorted(starts, batch_t, side="right") - 1]
    columns = {"mu": np.full(batch_t.size, plan.mu), "t": batch_t,
               "a": leg[:, 0], "e": leg[:, 1], "omega_p": leg[:, 2],
               "t_p": leg[:, 3]}
    columns["M"] = _call(mean_anomaly, columns)["M"]
    anomaly = columns["M"]                           # the declared starter
    for _ in range(KEPLER_STAGES):
        step = _call(stage, {**columns, "E_k": anomaly})
        anomaly = step["E_k1"]
    return columns, anomaly, step["kepler_residual"]


def reference(plan: HohmannPlan, t):
    """The plan's position and velocity at time(s) ``t``.

    Scalar ``t`` -> two shape-(3,) arrays; array ``t`` -> two (len, 3).
    Raises if a Kepler residual exceeds ``KEPLER_RESIDUAL_BOUND``.
    """
    state = _reference_pieces()[2]
    times = np.asarray(t, dtype=np.float64)
    flat = np.atleast_1d(times).ravel()
    position = np.zeros((flat.size, 3))
    velocity = np.zeros((flat.size, 3))
    for start, stop, batch_t in _chunks(flat):
        columns, anomaly, residual = _solve_kepler(plan, batch_t)
        worst = float(np.max(residual))
        if not worst <= KEPLER_RESIDUAL_BOUND:
            raise RuntimeError(f"Kepler residual {worst} exceeds "
                               f"{KEPLER_RESIDUAL_BOUND}")
        values = _call(state, {**columns, "E": anomaly})
        count = stop - start
        for slot, axis in enumerate(("x", "y")):
            position[start:stop, slot] = values[f"position_{axis}"][:count]
            velocity[start:stop, slot] = values[f"velocity_{axis}"][:count]
    if times.ndim == 0:
        return position[0], velocity[0]
    return position, velocity


def kepler_residual(plan: HohmannPlan, t) -> np.ndarray:
    """The final-stage Kepler residual at each time in ``t`` (diagnostic)."""
    flat = np.atleast_1d(np.asarray(t, dtype=np.float64)).ravel()
    out = np.zeros(flat.size)
    for start, stop, batch_t in _chunks(flat):
        out[start:stop] = _solve_kepler(plan, batch_t)[2][:stop - start]
    return out


def two_body_acceleration(mu: float, positions) -> np.ndarray:
    """eq_KE2_11..13 at each row of ``positions`` (n, 3) -> (n, 3)."""
    *_, gravity = _reference_pieces()
    points = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    out = np.zeros_like(points)
    for start in range(0, points.shape[0], REFERENCE_BATCH):
        chunk = points[start:start + REFERENCE_BATCH]
        padded = np.repeat(chunk[-1:], REFERENCE_BATCH, axis=0)
        padded[:chunk.shape[0]] = chunk
        values = _call(gravity, {
            "mu": np.full(REFERENCE_BATCH, float(mu)),
            **{f"position_{axis}": padded[:, slot]
               for slot, axis in enumerate(_AXES)}})
        for slot, axis in enumerate(_AXES):
            out[start:start + chunk.shape[0], slot] = (
                values[f"acceleration_{axis}"][:chunk.shape[0]])
    return out
