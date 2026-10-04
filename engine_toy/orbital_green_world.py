"""Persistent all-body Green-collocation world on the managed dt system.

The generated orbital law owns the coupled coast equations.  This module
declares its columns and metrics, gives the compiled pieces to the existing
dt graph, and advances a requested window through ``advance_round``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import sympy as sp

_ROOT = Path(__file__).resolve().parents[1]
_TURING = _ROOT / "turing"
_EXAMPLES = _TURING / "examples"
for _path in (_TURING, _EXAMPLES):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from honorary_engine_equation_catalogue import equation_piece
from llvm_dt_system import advance_round, instantiate_system, piece_leaf
from orbital_green import AXES, equation_sets, initial_columns
from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_graph import ControllerNode, RoundNode
from src.common.dt_system.error_channels import DT_CHANNEL_NAMES, channel_fields
from src.common.dt_system.time_contracts import BIND


GREEN_WORLD_ERROR_LIMITS = {
    "orbital_green_collocation_residual_m": 1.0e-3,
    "orbital_green_total_energy_relative_residual": 1.0e-10,
    "orbital_green_linear_momentum_relative_residual": 1.0e-10,
    "orbital_green_angular_momentum_relative_residual": 1.0e-10,
}
GREEN_WORLD_CHANNEL_NAMES = (*DT_CHANNEL_NAMES, *GREEN_WORLD_ERROR_LIMITS)


def _world_metric_equations(body_count: int, endpoint_equations):
    """Declare attempt-local N-body conservation defects from the proposed
    endpoint expressions, before the endpoint piece commits registered state.

    Difference forms avoid subtracting two large, nearly equal energy and
    angular-momentum totals.  The relative channels divide by the sum of
    magnitudes of the corresponding stored terms, so the limits are usable
    for systems with very different masses and orbital scales.
    """
    next_values = {}
    for equation in endpoint_equations:
        lhs = str(equation.lhs)
        if not lhs.endswith("_next"):
            raise ValueError(f"unexpected Green endpoint output {lhs!r}")
        next_values[lhs[:-5]] = equation.rhs

    vector = lambda values: sp.Matrix([values[a] for a in AXES])
    position0, position1, momentum0, momentum1, masses = [], [], [], [], []
    kinetic_delta, kinetic_scale = sp.Integer(0), sp.Integer(0)
    angular_delta = sp.zeros(3, 1)
    angular_scale = sp.Integer(0)
    momentum_delta = sp.zeros(3, 1)
    momentum_scale = sp.Integer(0)
    for body in range(body_count):
        mass = sp.Symbol(f"body{body}_mass_kg")
        x0 = vector({a: sp.Symbol(f"body{body}_position_{a}") for a in AXES})
        p0 = vector({a: sp.Symbol(f"body{body}_momentum_{a}") for a in AXES})
        x1 = vector({a: next_values[f"body{body}_position_{a}"] for a in AXES})
        p1 = vector({a: next_values[f"body{body}_momentum_{a}"] for a in AXES})
        dx, dp = x1 - x0, p1 - p0
        position0.append(x0)
        position1.append(x1)
        momentum0.append(p0)
        momentum1.append(p1)
        masses.append(mass)
        kinetic_delta += (p1 + p0).dot(dp) / (2 * mass)
        kinetic_scale += (p0.dot(p0) + p1.dot(p1)) / (2 * mass)
        momentum_delta += dp
        momentum_scale += sp.sqrt(p0.dot(p0)) + sp.sqrt(p1.dot(p1))
        angular_delta += dx.cross(p0) + x0.cross(dp) + dx.cross(dp)
        # Use |r||p| as the scale.  The conserved angular momentum can be
        # exactly zero for radial motion, but then normalizing by |r x p|
        # would turn harmless roundoff into a unit-sized error.
        angular_scale += (sp.sqrt(x0.dot(x0) * p0.dot(p0))
                          + sp.sqrt(x1.dot(x1) * p1.dot(p1)))

    # Match orbital_green's canonical column identity, including assumptions;
    # SymPy treats otherwise identical spellings with different assumptions
    # as distinct symbols and the equation compiler correctly rejects them.
    gravitational_constant = sp.Symbol("gravitational_constant", positive=True)
    energy_delta = kinetic_delta
    potential_scale = sp.Integer(0)
    for first in range(body_count):
        for second in range(first + 1, body_count):
            mass_product = masses[first] * masses[second]
            separation0 = position0[second] - position0[first]
            separation1 = position1[second] - position1[first]
            delta = separation1 - separation0
            radius0 = sp.sqrt(separation0.dot(separation0))
            radius1 = sp.sqrt(separation1.dot(separation1))
            # U1-U0 = G*m1*m2*(r1-r0)/(r0*r1), with r1-r0
            # expressed as a dot product to retain close small changes.
            energy_delta += (gravitational_constant * mass_product
                             * (separation1 + separation0).dot(delta)
                             / ((radius1 + radius0) * radius0 * radius1))
            potential_scale += gravitational_constant * mass_product * (
                1 / radius0 + 1 / radius1)

    energy_scale = kinetic_scale + potential_scale
    momentum_norm = sp.sqrt(momentum_delta.dot(momentum_delta))
    angular_norm = sp.sqrt(angular_delta.dot(angular_delta))
    tiny = sp.Float("1e-300")
    return (
        sp.Eq(sp.Symbol("orbital_green_total_energy_relative_residual"),
              sp.Abs(energy_delta) / sp.Max(energy_scale, tiny), evaluate=False),
        sp.Eq(sp.Symbol("orbital_green_linear_momentum_relative_residual"),
              momentum_norm / sp.Max(momentum_scale, tiny), evaluate=False),
        sp.Eq(sp.Symbol("orbital_green_angular_momentum_relative_residual"),
              angular_norm / sp.Max(angular_scale, tiny), evaluate=False),
    )


def _ordered_equation_sets(body_count: int):
    sets = dict(equation_sets(body_count))
    prefix = ("orbital_green_ballistic_guess", "orbital_green_picard1",
              "orbital_green_picard2")
    residual = "orbital_green_residual"
    endpoint = "orbital_green_endpoint"
    expected = {*prefix, residual, endpoint}
    if sets.keys() != expected:
        raise RuntimeError(f"unexpected generated Green piece set: {tuple(sets)}")
    metrics = _world_metric_equations(body_count, sets[endpoint])
    # The residual piece publishes its named error channel directly.
    # Conservation metrics read the old physical state and candidate endpoint
    # equations before the final commit.
    return tuple((name, sets[name]) for name in prefix) + (
        (residual, sets[residual]),
        (f"orbital_green_world_metrics_b{body_count}", metrics),
        (endpoint, sets[endpoint]),
    )


class OrbitalGreenWorld:
    """One rollback-managed state containing every mutually gravitating body.

    Body arrays have shape ``(body_count, 3)`` and describe entities inside
    this one system.  They are not dt-system batch lanes; the batch is fixed to
    one so each Green pair is part of the same coupled world.
    """

    def __init__(self, *, position_m, velocity_m_s,
                 mass_kg: Sequence[float], gravitational_constant: float,
                 window_s: float, length_scale_m: float, cfl: float = 0.5,
                 error_limits: Mapping[str, float] | None = None,
                 retain_compilation: bool = False):
        positions = np.asarray(position_m, dtype=np.float64)
        velocities = np.asarray(velocity_m_s, dtype=np.float64)
        masses = np.asarray(mass_kg, dtype=np.float64).reshape(-1)
        self.body_count = int(masses.size)
        self.window_s = float(window_s)
        self.length_scale_m = float(length_scale_m)
        if not math.isfinite(self.window_s) or self.window_s <= 0.0:
            raise ValueError("window_s must be finite and positive")
        if not math.isfinite(self.length_scale_m) or self.length_scale_m <= 0.0:
            raise ValueError("length_scale_m must be finite and positive")
        if not math.isfinite(float(cfl)) or float(cfl) <= 0.0:
            raise ValueError("cfl must be finite and positive")

        self.error_limits = dict(GREEN_WORLD_ERROR_LIMITS)
        if error_limits is not None:
            undeclared = set(error_limits) - self.error_limits.keys()
            if undeclared:
                raise ValueError(f"undeclared Green error channels: {sorted(undeclared)}")
            for name, value in error_limits.items():
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"{name}: error limit must be finite and positive")
                self.error_limits[name] = float(value)

        equation_groups = _ordered_equation_sets(self.body_count)
        pieces = tuple(equation_piece(
            name, equations, batch=1,
            retain_compilation=retain_compilation)
            for name, equations in equation_groups)
        for piece in pieces:
            piece.contract = BIND
        columns = initial_columns(
            self.body_count, position_m=positions, velocity_m_s=velocities,
            mass_kg=masses, gravitational_constant=gravitational_constant,
            batch=1)
        targets = Targets(
            cfl=float(cfl), div_max=1.0e9, mass_max=1.0e-3,
            **channel_fields(self.error_limits,
                             names=GREEN_WORLD_CHANNEL_NAMES, limits=True))
        speed = float(np.max(np.linalg.norm(velocities, axis=1)))
        dt_init = (self.window_s if speed <= 0.0 else min(
            self.window_s, targets.cfl * self.length_scale_m / speed))
        self.pieces = pieces
        self.piece_labels = tuple(piece.entry for piece in pieces)
        self.dt_graph = RoundNode(
            plan=SuperstepPlan(round_max=self.window_s, dt_init=dt_init,
                               allow_increase_mid_round=True),
            controller=ControllerNode(
                ctrl=STController(dt_min=None), targets=targets,
                dx=self.length_scale_m),
            children=[piece_leaf(piece, label=label)
                      for piece, label in zip(pieces, self.piece_labels)],
            schedule="sequential",
            label="orbital-green-world",
        )
        self.dt_state = instantiate_system(
            self.dt_graph, columns, channel_names=GREEN_WORLD_CHANNEL_NAMES)
        self.time_s = 0.0

    def _scalar(self, name: str) -> float:
        return float(np.asarray(getattr(self.dt_state, name),
                                dtype=np.float64).reshape(-1)[0])

    def positions_m(self) -> np.ndarray:
        return np.asarray([
            [self._scalar(f"body{body}_position_{axis}") for axis in AXES]
            for body in range(self.body_count)], dtype=np.float64)

    def momenta_kg_m_s(self) -> np.ndarray:
        return np.asarray([
            [self._scalar(f"body{body}_momentum_{axis}") for axis in AXES]
            for body in range(self.body_count)], dtype=np.float64)

    def velocities_m_s(self) -> np.ndarray:
        momenta = self.momenta_kg_m_s()
        masses = np.asarray([self._scalar(f"body{body}_mass_kg")
                             for body in range(self.body_count)])
        return momenta / masses[:, None]

    def published_metrics(self) -> dict[str, float]:
        """Last attempt's present dt publications, reduced by channel name."""
        values = np.asarray(self.dt_state.pub_values, dtype=np.float64).reshape(
            len(self.pieces), len(GREEN_WORLD_CHANNEL_NAMES))
        present = np.asarray(self.dt_state.pub_present, dtype=np.float64).reshape(
            len(self.pieces), len(GREEN_WORLD_CHANNEL_NAMES))
        return {
            name: float(np.max(values[present[:, channel] > 0.0, channel]))
            for channel, name in enumerate(GREEN_WORLD_CHANNEL_NAMES)
            if np.any(present[:, channel] > 0.0)
        }

    def advance(self, window_s: float | None = None):
        """Advance one requested world window through the dt coordinator."""
        requested = self.window_s if window_s is None else float(window_s)
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("requested world window must be finite and positive")
        advanced, dt_next, telemetry = advance_round(self.dt_state, requested)
        advanced = float(advanced)
        if not math.isclose(advanced, requested, rel_tol=0.0,
                            abs_tol=1.0e-12 * max(1.0, requested)):
            raise RuntimeError(
                f"orbital Green world advanced {advanced} of {requested}")
        self.time_s += advanced
        return advanced, dt_next, telemetry


__all__ = ["GREEN_WORLD_ERROR_LIMITS", "GREEN_WORLD_CHANNEL_NAMES",
           "OrbitalGreenWorld"]
