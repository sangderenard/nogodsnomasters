"""Temporal Green-function collocation foundation for all-body orbital coast.

This module builds a small, fixed-shape SymPy piece graph for the point-mass
N-body equations.  Its three Lobatto nodes are ``0, 1/2, 1``; two bounded
Picard corrections construct the interior position stages.  Every generated
piece is an ordinary ``equation_piece`` candidate, so its scratch outputs can
be registered as dt-system columns and participate in the existing
PieceState rollback transaction.

The module does not create or advance a dt system.  A caller must register the
returned pieces and all columns from :func:`initial_columns` in the existing
dt graph, then call the existing round path.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import sympy as sp

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

from src.common.tensors.abstraction import AbstractTensor

from honorary_engine_equation_catalogue import equation_piece

AXES = ("x", "y", "z")
NODE_COUNT = 3
PICARD_CORRECTIONS = 2
LOBATTO_NODES = (0.0, 0.5, 1.0)

_DT = sp.Symbol("dt")
_G = sp.Symbol("gravitational_constant", positive=True)


def _integration_weights(nodes: tuple[float, ...]):
    """Return single- and double-integral collocation weights.

    ``V C = I`` determines each Lagrange polynomial's monomial coefficients.
    The solve is deliberately performed through the repository's
    ``AbstractTensor.linalg.solve`` at preparation time; the resulting fixed
    coefficients are constants in the generated law, not a runtime solver.
    """
    node_array = np.asarray(nodes, dtype=np.float64)
    count = len(nodes)
    vandermonde = np.stack([node_array ** degree
                            for degree in range(count)], axis=1)
    V = AbstractTensor.tensor(vandermonde, dtype=np.float64)
    identity = AbstractTensor.tensor(np.eye(count, dtype=np.float64),
                                     dtype=np.float64)
    coefficients = AbstractTensor.linalg.solve(V, identity).numpy()

    q = np.zeros((count, count), dtype=np.float64)
    a = np.zeros((count, count), dtype=np.float64)
    for row, upper in enumerate(node_array):
        for node in range(count):
            q[row, node] = sum(
                coefficients[degree, node] * upper ** (degree + 1)
                / (degree + 1) for degree in range(count))
            a[row, node] = sum(
                coefficients[degree, node] * upper ** (degree + 2)
                / ((degree + 1) * (degree + 2))
                for degree in range(count))
    return tuple(tuple(float(v) for v in row) for row in q), \
        tuple(tuple(float(v) for v in row) for row in a)


@functools.lru_cache(maxsize=1)
def collocation_weights():
    """Prepared weights ``(Q, A)`` for the fixed three-node law."""
    return _integration_weights(LOBATTO_NODES)


def _symbol(name: str):
    return sp.Symbol(name)


def _position(body: int, axis: str):
    return _symbol(f"body{body}_position_{axis}")


def _momentum(body: int, axis: str):
    return _symbol(f"body{body}_momentum_{axis}")


def _mass(body: int):
    return _symbol(f"body{body}_mass_kg")


def _acceleration(body: int, axis: str, stage_positions):
    """Exact pairwise Newtonian acceleration for one body at one stage."""
    total = sp.Integer(0)
    for source in range(len(stage_positions)):
        if source == body:
            continue
        delta = stage_positions[source][axis] - stage_positions[body][axis]
        radius_sq = sum(
            (stage_positions[source][component]
             - stage_positions[body][component]) ** 2
            for component in AXES)
        # This is the point-mass Green kernel's gradient.  No softening or
        # radius clamp is applied; coincident stages are outside the law.
        total += _G * _mass(source) * delta / radius_sq ** sp.Rational(3, 2)
    return total


def _stage_symbols(body_count: int, iteration: int, node: int):
    return tuple({axis: _symbol(
        f"green_iter{iteration}_node{node}_body{body}_position_{axis}")
                  for axis in AXES} for body in range(body_count))


def _stage_output_eq(symbol: sp.Symbol, rhs):
    return sp.Eq(_symbol(f"{symbol.name}_next"), rhs, evaluate=False)


def equation_sets(body_count: int):
    """Build partitioned SymPy equalities for one candidate coast step.

    Returns ordered ``(piece_id, equations)`` pairs: ballistic stage guess,
    two Picard corrections, a collocation residual publication, and endpoint
    commit.  All stage scratch outputs use the established ``*_next``
    equation-piece convention and have matching names from
    :func:`initial_columns`.
    """
    if int(body_count) != body_count or body_count < 2:
        raise ValueError("all-body gravity needs at least two bodies")
    body_count = int(body_count)
    q_weights, a_weights = collocation_weights()
    positions0 = tuple({axis: _position(body, axis) for axis in AXES}
                       for body in range(body_count))
    velocities0 = tuple({axis: _momentum(body, axis) / _mass(body)
                         for axis in AXES}
                        for body in range(body_count))

    pieces = []

    # Start at the ballistic path, then apply the fixed number of integral
    # Picard corrections.  The c=0 node is exactly the initial position.
    equations = []
    for node in (1, 2):
        guess = _stage_symbols(body_count, 0, node)
        for body in range(body_count):
            c = LOBATTO_NODES[node]
            for axis in AXES:
                equations.append(_stage_output_eq(
                    guess[body][axis], positions0[body][axis]
                    + _DT * c * velocities0[body][axis]))
    pieces.append(("orbital_green_ballistic_guess", tuple(equations)))
    # The previous stage set omits c=0; insert the exact initial node.
    for correction in range(1, PICARD_CORRECTIONS + 1):
        # Both interior nodes from the last iterate must be sampled. Build an
        # explicit per-node sequence for every body before evaluating gravity.
        prior_nodes = tuple(
            positions0 if node == 0 else
            _stage_symbols(body_count, correction - 1, node)
            for node in range(NODE_COUNT))
        equations = []
        for node in (1, 2):
            current = _stage_symbols(body_count, correction, node)
            for body in range(body_count):
                for axis in AXES:
                    integral = sp.Integer(0)
                    for source_node in range(NODE_COUNT):
                        acceleration = _acceleration(
                            body, axis, prior_nodes[source_node])
                        integral += a_weights[node][source_node] * acceleration
                    equations.append(_stage_output_eq(
                        current[body][axis], positions0[body][axis]
                        + _DT * LOBATTO_NODES[node] * velocities0[body][axis]
                        + _DT ** 2 * integral))
        pieces.append((f"orbital_green_picard{correction}", tuple(equations)))

    # Final force samples at all three collocation nodes produce the endpoint
    # position and momentum columns in the same registered transaction.
    final_nodes = tuple(
        positions0 if node == 0 else
        _stage_symbols(body_count, PICARD_CORRECTIONS, node)
        for node in range(NODE_COUNT))
    endpoint = []
    defect_squares = []
    for body in range(body_count):
        for axis in AXES:
            integral_a = sp.Integer(0)
            integral_q = sp.Integer(0)
            for node in range(NODE_COUNT):
                acceleration = _acceleration(body, axis, final_nodes[node])
                integral_a += a_weights[-1][node] * acceleration
                integral_q += q_weights[-1][node] * acceleration
            position_next = (positions0[body][axis]
                             + _DT * velocities0[body][axis]
                             + _DT ** 2 * integral_a)
            momentum_next = (_momentum(body, axis)
                             + _mass(body) * _DT * integral_q)
            endpoint.extend((
                _stage_output_eq(_symbol(f"body{body}_position_{axis}"),
                                 position_next),
                _stage_output_eq(_symbol(f"body{body}_momentum_{axis}"),
                                 momentum_next),
            ))
            for stage_node in (1, 2):
                collocation_rhs = (
                    positions0[body][axis]
                    + _DT * LOBATTO_NODES[stage_node]
                    * velocities0[body][axis]
                    + _DT ** 2 * sum(
                        a_weights[stage_node][source_node]
                        * _acceleration(body, axis,
                                        final_nodes[source_node])
                        for source_node in range(NODE_COUNT)))
                defect = final_nodes[stage_node][body][axis] - collocation_rhs
                defect_squares.append(defect ** 2)
    # Publish the defect while the canonical body columns still hold the
    # attempt's initial positions. The endpoint piece is the physical commit.
    pieces.append(("orbital_green_residual", (
        sp.Eq(_symbol("orbital_green_collocation_residual_m"),
              sp.sqrt(sum(defect_squares, sp.Integer(0))), evaluate=False),
    )))
    pieces.append(("orbital_green_endpoint", tuple(endpoint)))
    return tuple(pieces)


def craft_coast_equation_sets(center_count: int, *, inertia=None,
                              inverse_inertia=None,
                              internal_angular_momentum=None):
    """Green candidates for a reduced MachineCraft inertial coast.

    The current game represents gravity wells as registered ``center{i}_*``
    parameters.  This law samples every one of those wells at each Lobatto
    node.  When the caller supplies its existing rigid-body reduction, the
    same collocation also carries attitude and body rate as a torque-free
    gyrostat: ``H_body = I' omega + H_internal``.  This is the engine-machine
    baking boundary -- one summed mass/inertia and the sum of every internal
    rotor's declared angular momentum, rather than a second model of parts.

    All results are ``green_coast_*`` candidates.  MachineCraft's existing
    commit remains the sole owner of canonical state and selects candidates
    only when its attempted external wrench and reduced inertial properties
    say the aggregate really was static over the interval.
    """
    if int(center_count) != center_count or center_count < 1:
        raise ValueError("Green coast needs at least one gravity center")
    center_count = int(center_count)
    q_weights, a_weights = collocation_weights()
    tag = f"orbital_green_coast_c{center_count}"
    position0 = {axis: _symbol(f"position_{axis}") for axis in AXES}
    momentum0 = {axis: _symbol(f"momentum_{axis}") for axis in AXES}
    mass = _symbol("mass")
    velocity0 = {axis: momentum0[axis] / mass for axis in AXES}

    def stage(iteration: int, node: int):
        return {axis: _symbol(
            f"green_coast_iter{iteration}_node{node}_position_{axis}")
                for axis in AXES}

    def acceleration(axis: str, position):
        total = sp.Integer(0)
        for source in range(center_count):
            delta = {_axis: _symbol(f"center{source}_{_axis}")
                     - position[_axis] for _axis in AXES}
            radius_sq = sum((delta[_axis] ** 2 for _axis in AXES),
                            sp.Integer(0))
            total += (_symbol(f"center{source}_mu") * delta[axis]
                      / radius_sq ** sp.Rational(3, 2))
        return total

    pieces = []
    equations = []
    for node in (1, 2):
        guess = stage(0, node)
        for axis in AXES:
            equations.append(_stage_output_eq(
                guess[axis], position0[axis]
                + _DT * LOBATTO_NODES[node] * velocity0[axis]))
    pieces.append((f"{tag}_ballistic_guess", tuple(equations)))

    for correction in range(1, PICARD_CORRECTIONS + 1):
        prior_nodes = tuple(position0 if node == 0
                            else stage(correction - 1, node)
                            for node in range(NODE_COUNT))
        equations = []
        for node in (1, 2):
            current = stage(correction, node)
            for axis in AXES:
                integral = sum(
                    (a_weights[node][source_node]
                     * acceleration(axis, prior_nodes[source_node])
                     for source_node in range(NODE_COUNT)), sp.Integer(0))
                equations.append(_stage_output_eq(
                    current[axis], position0[axis]
                    + _DT * LOBATTO_NODES[node] * velocity0[axis]
                    + _DT ** 2 * integral))
        pieces.append((f"{tag}_picard{correction}", tuple(equations)))

    rigid_arguments = (inertia, inverse_inertia,
                       internal_angular_momentum)
    if any(value is not None for value in rigid_arguments):
        if any(value is None for value in rigid_arguments):
            raise ValueError("rigid Green coast needs inertia, inverse "
                             "inertia, and internal angular momentum")
        inertia = sp.Matrix(inertia)
        inverse_inertia = sp.Matrix(inverse_inertia)
        internal_angular_momentum = sp.Matrix(internal_angular_momentum)
        if (inertia.shape != (3, 3) or inverse_inertia.shape != (3, 3)
                or internal_angular_momentum.shape != (3, 1)):
            raise ValueError("invalid reduced rigid-body Green shapes")

        attitude0 = {(row, col): _symbol(f"attitude_{AXES[row]}{AXES[col]}")
                     for row in range(3) for col in range(3)}
        omega0 = {axis: _symbol(f"angular_velocity_{axis}") for axis in AXES}

        def rigid_state(iteration, node):
            prefix = f"green_coast_rigid_iter{iteration}_node{node}"
            return {
                **{f"omega_{axis}": _symbol(f"{prefix}_angular_velocity_{axis}")
                   for axis in AXES},
                **{f"attitude_{row}{col}":
                   _symbol(f"{prefix}_attitude_{AXES[row]}{AXES[col]}")
                   for row in range(3) for col in range(3)},
            }

        rigid0 = {
            **{f"omega_{axis}": omega0[axis] for axis in AXES},
            **{f"attitude_{row}{col}": attitude0[(row, col)]
               for row in range(3) for col in range(3)},
        }

        def rigid_rate(state):
            omega = sp.Matrix([state[f"omega_{axis}"] for axis in AXES])
            rotation = sp.Matrix(
                3, 3, lambda row, col: state[f"attitude_{row}{col}"])
            angular_momentum = inertia * omega + internal_angular_momentum
            omega_rate = -inverse_inertia * omega.cross(angular_momentum)
            skew = sp.Matrix(((0, -omega[2], omega[1]),
                              (omega[2], 0, -omega[0]),
                              (-omega[1], omega[0], 0)))
            rotation_rate = rotation * skew
            return {
                **{f"omega_{axis}": omega_rate[index]
                   for index, axis in enumerate(AXES)},
                **{f"attitude_{row}{col}": rotation_rate[row, col]
                   for row in range(3) for col in range(3)},
            }

        initial_rate = rigid_rate(rigid0)
        equations = []
        for node in (1, 2):
            guess = rigid_state(0, node)
            c = LOBATTO_NODES[node]
            for name, symbol in guess.items():
                equations.append(_stage_output_eq(
                    symbol, rigid0[name] + _DT * c * initial_rate[name]))
        pieces.append((f"{tag}_rigid_guess", tuple(equations)))

        for correction in range(1, PICARD_CORRECTIONS + 1):
            prior = tuple(rigid0 if node == 0 else
                          rigid_state(correction - 1, node)
                          for node in range(NODE_COUNT))
            rates = tuple(rigid_rate(state) for state in prior)
            equations = []
            for node in (1, 2):
                current = rigid_state(correction, node)
                for name, symbol in current.items():
                    integral = sum(
                        (q_weights[node][source] * rates[source][name]
                         for source in range(NODE_COUNT)), sp.Integer(0))
                    equations.append(_stage_output_eq(
                        symbol, rigid0[name] + _DT * integral))
            pieces.append((f"{tag}_rigid_picard{correction}",
                           tuple(equations)))

        final_rigid = tuple(
            rigid0 if node == 0 else
            rigid_state(PICARD_CORRECTIONS, node)
            for node in range(NODE_COUNT))
        final_rates = tuple(rigid_rate(state) for state in final_rigid)
        candidate = []
        for name, initial in rigid0.items():
            endpoint = initial + _DT * sum(
                (q_weights[-1][source] * final_rates[source][name]
                 for source in range(NODE_COUNT)), sp.Integer(0))
            if name.startswith("omega_"):
                output = _symbol("green_coast_angular_velocity_"
                                 + name.removeprefix("omega_"))
            else:
                row, col = int(name[-2]), int(name[-1])
                output = _symbol(
                    f"green_coast_attitude_{AXES[row]}{AXES[col]}")
            candidate.append(_stage_output_eq(output, endpoint))
        pieces.append((f"{tag}_rigid_candidate", tuple(candidate)))

    final_nodes = tuple(position0 if node == 0
                        else stage(PICARD_CORRECTIONS, node)
                        for node in range(NODE_COUNT))
    candidate = []
    defects = []
    for axis in AXES:
        integral_a = sum(
            (a_weights[-1][node] * acceleration(axis, final_nodes[node])
             for node in range(NODE_COUNT)), sp.Integer(0))
        integral_q = sum(
            (q_weights[-1][node] * acceleration(axis, final_nodes[node])
             for node in range(NODE_COUNT)), sp.Integer(0))
        candidate.extend((
            _stage_output_eq(_symbol(f"green_coast_position_{axis}"),
                             position0[axis] + _DT * velocity0[axis]
                             + _DT ** 2 * integral_a),
            _stage_output_eq(_symbol(f"green_coast_momentum_{axis}"),
                             momentum0[axis] + mass * _DT * integral_q),
        ))
        for node in (1, 2):
            collocation_rhs = (
                position0[axis]
                + _DT * LOBATTO_NODES[node] * velocity0[axis]
                + _DT ** 2 * sum(
                    (a_weights[node][source_node]
                     * acceleration(axis, final_nodes[source_node])
                     for source_node in range(NODE_COUNT)), sp.Integer(0)))
            defects.append((final_nodes[node][axis] - collocation_rhs) ** 2)
    pieces.append((f"{tag}_residual", (
        _stage_output_eq(_symbol("green_coast_collocation_residual_m"),
                         sp.sqrt(sum(defects, sp.Integer(0)))),
    )))
    pieces.append((f"{tag}_candidate", tuple(candidate)))
    return tuple(pieces)


def initial_columns(body_count: int, *, position_m, velocity_m_s,
                    mass_kg: Sequence[float], gravitational_constant: float,
                    batch: int = 1):
    """Initial registered columns for an all-body coast graph.

    ``position_m`` and ``velocity_m_s`` are ``(body_count, 3)`` arrays;
    ``mass_kg`` contains one positive mass per body.  State arrays are
    broadcast over ``batch`` lanes, matching the existing equation-piece
    batch convention.  Stage columns are initialized explicitly so the dt
    transaction owns them from its first attempt.  Their names are
    ``green_iter{0,1,2}_node{1,2}_body{i}_position_{axis}``.
    """
    if int(body_count) != body_count or body_count < 2:
        raise ValueError("all-body gravity needs at least two bodies")
    body_count = int(body_count)
    if int(batch) != batch or batch < 1:
        raise ValueError("batch must be a positive integer")
    positions = np.asarray(position_m, dtype=np.float64)
    velocities = np.asarray(velocity_m_s, dtype=np.float64)
    masses = np.asarray(mass_kg, dtype=np.float64).reshape(-1)
    if positions.shape != (body_count, 3) or velocities.shape != (body_count, 3):
        raise ValueError("position_m and velocity_m_s must have shape (body_count, 3)")
    if masses.shape != (body_count,) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("mass_kg must contain one finite positive mass per body")
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("gravitational_constant must be finite and positive")

    columns = {_G.name: np.full(batch, float(gravitational_constant))}
    for body in range(body_count):
        columns[f"body{body}_mass_kg"] = np.full(batch, masses[body])
        for axis_index, axis in enumerate(AXES):
            columns[f"body{body}_position_{axis}"] = np.full(
                batch, positions[body, axis_index])
            columns[f"body{body}_momentum_{axis}"] = np.full(
                batch, masses[body] * velocities[body, axis_index])
    # Match every generated *_next output to a physical registered column.
    for _piece_name, equations in equation_sets(body_count):
        for equation in equations:
            output = str(equation.lhs)
            if output.endswith("_next"):
                columns.setdefault(output[:-5], np.zeros(batch))
    return columns


def dt_pieces(body_count: int, *, batch: int = 1,
              retain_compilation: bool = False):
    """Compile the ordered collocation pieces using the existing piece API."""
    if int(batch) != batch or batch < 1:
        raise ValueError("batch must be a positive integer")
    pieces = tuple(equation_piece(name, equations, batch=int(batch),
                                  retain_compilation=retain_compilation)
                   for name, equations in equation_sets(body_count))
    return pieces, tuple(piece.entry for piece in pieces)


__all__ = ["AXES", "NODE_COUNT", "PICARD_CORRECTIONS", "LOBATTO_NODES",
           "collocation_weights", "equation_sets", "craft_coast_equation_sets", "initial_columns",
           "dt_pieces"]
