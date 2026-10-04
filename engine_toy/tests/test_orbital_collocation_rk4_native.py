"""Bounded c0/t1 native seven-row gate, not production row-bank acceptance."""
import pickle
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

import orbital_collocation as oc
from orbital_actuation import CraftDesign, Thruster
from src.compiler.identity_concordance import concordance_report
from src.compiler.process_graph_autograd import reverse_compile_book
from src.compiler import ssa_llvm_backend


def _original_slice_defect_laws(center_count, thruster_count):
    """Independent pre-reduction oracle: the original seven authored laws.

    Keep the actual library RK4 call, including its binary coefficients.
    Neither this reference nor its derivatives use the reduced source.
    """
    supply = {oc.PROPELLANT_SUPPLY: sp.Integer(1)}
    flow = oc.propellant_flow_rhs(thruster_count).xreplace(supply)
    applied = {a: oc.actuation_force_rhs(a, thruster_count).xreplace(supply)
               for a in oc.AXES}
    gravity = {a: oc.oj.gravity_force_rhs(a, center_count) for a in oc.AXES}
    n72 = {a: oc.oj.variable_mass_momentum_rate(a) for a in oc.AXES}

    def derivative(_time, state):
        at = dict(zip(oc.STATE, state))
        mass = oc._DRY + at[oc.PROPELLANT_MASS]
        position = {a: at[oc._POSITION[a]] for a in oc.AXES}
        momentum = {a: at[oc._MOMENTUM[a]] for a in oc.AXES}
        return sp.Matrix(
            [n72[a].xreplace({
                oc._FORCE[a]: gravity[a].xreplace({
                    **{oc._POSITION[b]: position[b] for b in oc.AXES},
                    oc._MASS: mass}) + applied[a],
                oc.PROPELLANT_FLOW: flow, oc._MOMENTUM[a]: momentum[a],
                oc._MASS: mass}) for a in oc.AXES]
            + [oc.honorary.eq_N1_1.rhs.xreplace({
                oc.honorary.m_i(oc.honorary.t): mass,
                oc.honorary.p_i(oc.honorary.t): momentum[a]}) for a in oc.AXES]
            + [-flow])

    advanced = oc.RK4Integrator().step(derivative, sp.Integer(0),
                                      sp.Matrix(oc.STATE), oc._DT)
    return [oc._NEXT[s] - advanced[i] for i, s in enumerate(oc.STATE)]


def test_affine_rk4_relations_preserve_all_binary_coefficients_and_finite_flow():
    momentum = sp.symbols("p_x p_y p_z")
    position = sp.symbols("r_x r_y r_z")
    propellant, dry, flow, dt = sp.symbols("P dry q h")
    initial = sp.Matrix([*momentum, *position, propellant])
    forces = []

    def derivative(_time, state):
        stage_force = sp.symbols(f"F{len(forces)}_x F{len(forces)}_y F{len(forces)}_z")
        forces.append(stage_force)
        mass = dry + state[6]
        return sp.Matrix([
            oc.oj.variable_mass_momentum_rate(a).xreplace({
                oc._FORCE[a]: stage_force[i], oc.PROPELLANT_FLOW: flow,
                oc._MOMENTUM[a]: state[i], oc._MASS: mass})
            for i, a in enumerate(oc.AXES)] + [
                oc.honorary.eq_N1_1.rhs.xreplace({
                    oc.honorary.m_i(oc.honorary.t): mass,
                    oc.honorary.p_i(oc.honorary.t): state[i]})
                for i in range(3)] + [-flow])

    original = oc.RK4Integrator().step(derivative, sp.Integer(0), initial, dt)
    variables = (*momentum, *(f for stage in forces for f in stage))
    for i, before in enumerate(original):
        after = oc._cancel_affine_physical_law(before, variables)
        exact_before = before.xreplace({atom: sp.Rational(atom)
                                       for atom in before.atoms(sp.Float)})
        assert sp.cancel(after - exact_before) == 0
        if i in (3, 4, 5):
            axis = i - 3
            ballistic = after.xreplace({stage[axis]: sp.Integer(0)
                                        for stage in forces})
            assert not ballistic.has(flow)
            coefficient = sp.cancel((ballistic - position[axis])
                                    * (dry + propellant) / (dt * momentum[axis]))
            assert coefficient == sp.Rational(18014398509481983, 18014398509481984)
            assert coefficient != 1


@pytest.fixture(scope="module")
def gravity_law_pair():
    return oc.slice_defect_laws(1, 1), _original_slice_defect_laws(1, 1)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_reduced_laws_equal_original_with_finite_propellant_and_gravity(gravity_law_pair, axis):
    reduced, original = gravity_law_pair
    direction = tuple(int(i == axis) for i in range(3))
    design = CraftDesign((Thruster("finite", (0, 0, 0), direction, 64.0,
                                   kind="cold-gas"),),
                         mass_kg=64.0, propellant_kg=16.0)
    feeds = {sp.Symbol(name): sp.Rational(*float(value[0]).as_integer_ratio())
             for name, value in oc.thruster_columns(design).items()}
    feeds[oc._DRY] = sp.Rational(design.dry_mass_kg)
    feeds[oc._DT] = sp.Rational(1, 32)
    feeds[oc.thruster_symbols(0)["throttle"]] = sp.Rational(3, 8)
    for (row, column), symbol in oc.attitude_symbols().items():
        feeds[symbol] = sp.Integer(row == column)
    center, mu = oc.oj._center_symbols(0)
    feeds.update({symbol: sp.Integer(0) for symbol in center.values()})
    feeds[mu] = sp.Integer(1)  # Nonzero gravity, away from its singularity.
    state = [2 * d for d in direction] + [16 * d for d in direction] + [16]
    for symbol, value in zip(oc.STATE, state):
        feeds[symbol] = sp.Integer(value)
        feeds[oc._NEXT[symbol]] = sp.Integer(0)
    for before, after in zip(original, reduced):
        exact = {atom: sp.Rational(atom)
                 for expression in (before, after) for atom in expression.atoms(sp.Float)}
        assert sp.cancel(after.xreplace({**exact, **feeds})
                         - before.xreplace({**exact, **feeds})) == 0
    assert set.union(*(e.free_symbols for e in reduced)) == set.union(
        *(e.free_symbols for e in original))


def _exact_reference_substitutions(substitutions):
    # These are the same float64 ABI inputs, represented as exact binary
    # rationals so xreplace cannot round intermediate arithmetic at 53 bits.
    # Resolve inputs once before evalf instead of revisiting substitutions
    # throughout each adaptive high-precision expression evaluation.
    return {symbol: (sp.oo if value == math.inf else
                     -sp.oo if value == -math.inf else
                     sp.Rational(*float(value).as_integer_ratio()))
            for symbol, value in substitutions.items()}


def _reference_eval(expression, exact_substitutions):
    # Library-created Float coefficients also need their exact binary value.
    # Leaving them as Float would let reconstruction during xreplace fold
    # numeric operations at the coefficient's original precision.
    coefficients = {atom: sp.Rational(atom) for atom in expression.atoms(sp.Float)}
    return expression.xreplace({**coefficients, **exact_substitutions}).evalf(80)


def test_exact_reference_preserves_inputs_and_float_coefficients():
    x, y, z = sp.symbols("x y z")
    inputs = {x: float(2**54), y: 0.25, z: float(2**54)}
    exact = _exact_reference_substitutions(inputs)
    coefficient = sp.Float(0.1)
    expression = coefficient*x + coefficient*y - coefficient*z
    expected = (sp.Rational(coefficient)*sp.Rational(1, 4)).evalf(80)
    assert _reference_eval(expression, exact) == expected
    assert _reference_eval(expression, exact) == expression.evalf(80, subs=inputs)
    assert _exact_reference_substitutions({x: 0.1})[x] == sp.Rational(*0.1.as_integer_ratio())
    assert _exact_reference_substitutions({x: math.inf, y: -math.inf}) == {x: sp.oo, y: -sp.oo}


def test_library_rk4_seven_rows_and_native_jacobian(tmp_path, monkeypatch):
    # Test-only persistent cache: every reuse still takes the existing
    # expression-key/compiler-record/library validation path.
    monkeypatch.setattr(oc, "CACHE_DIRECTORY",
                        Path(tempfile.gettempdir()) / "orbital_collocation_native_gate_rows")
    began = time.perf_counter()
    print("RK4 gate: constructing authored expressions", flush=True)
    expressions = oc.slice_defect_laws(0, 1)
    reference_expressions = _original_slice_defect_laws(0, 1)
    wrt = oc._slice_wrt(1)
    name = "colloc_library_rk4_c0_t1_gate"
    key = oc._cache_key(name, expressions, wrt)
    print(f"RK4 gate: expressions ready after {time.perf_counter()-began:.3f}s; "
          f"preparing validated row artifact key={key}", flush=True)
    modules = []
    emit = ssa_llvm_backend.emit_ssa_function_to_llvm

    def capture(module, *args, **kwargs):
        modules.append(module)
        print(f"RK4 gate: native emission begins at {time.perf_counter()-began:.3f}s", flush=True)
        return emit(module, *args, **kwargs)

    monkeypatch.setattr(ssa_llvm_backend, "emit_ssa_function_to_llvm", capture)
    with reverse_compile_book():
        rows = oc.compile_reverse_rows(name, expressions, wrt)
    print(f"RK4 gate: artifact ready at {time.perf_counter()-began:.3f}s", flush=True)
    directory = oc.CACHE_DIRECTORY / key
    receipt_path = directory / f"concordance.{rows.compiler.digest}.txt"
    if modules:
        receipt = concordance_report(modules[0], limit=2)
        assert rows.artifact.emission is not None
        with (directory / f"ssa.{rows.compiler.digest}.pkl").open("wb") as handle:
            pickle.dump(modules[0], handle)
        receipt_path.write_text(f"cache-key {key}\nfresh emitted audit\n{receipt}\n",
                                encoding="utf-8")
    else:
        # The manifest loader does not reconstruct the compiler module/book.
        # An absent prior receipt is a refusal, not an invented audit.
        assert receipt_path.is_file(), "cached gate needs its archived emitted audit"
        archived = receipt_path.read_text(encoding="utf-8")
        assert archived.startswith(f"cache-key {key}\n")
        print(f"PRIOR emitted audit evidence: {receipt_path}; not reconstructed", flush=True)
        receipt = archived[archived.index("identity concordance:"):]
    print(receipt, flush=True)
    assert receipt.splitlines()[0].endswith(", 0 finding(s)")
    print(f"RK4 gate: audit ready at {time.perf_counter()-began:.3f}s; "
          "constructing symbolic reference derivatives", flush=True)
    by_name = {symbol.name: symbol for symbol in wrt}
    derivatives = []
    for index, expression in enumerate(reference_expressions):
        derivatives.append([sp.diff(expression, by_name[name])
                            for name in rows.gradient_ids])
        print(f"RK4 gate: derivative row {index+1}/7 ready at "
              f"{time.perf_counter()-began:.3f}s", flush=True)
    for kind, propellant, thrust, throttle, dt in (
            ("ideal", 0.0, 64.0, 0.5, 0.25),
            ("cold-gas", 16.0, 4096.0, 0.375, 0.5)):
        design = CraftDesign((Thruster("one", (0, 0, 0), (1, 0, 0), thrust,
                                       kind=kind),),
                             mass_kg=64.0, propellant_kg=propellant)
        feeds = {name: float(value[0]) for name, value in
                 oc.thruster_columns(design).items()}
        feeds[oc._DRY.name] = design.dry_mass_kg
        for (r, c), symbol in oc.attitude_symbols().items():
            feeds[symbol.name] = float(r == c)
        for symbol, value in zip(oc.STATE, (32., 64., 96., 1., 2., 3., propellant)):
            feeds[symbol.name] = value
            feeds[oc._NEXT[symbol].name] = 0.0
        feeds[oc.thruster_symbols(0)["throttle"].name] = throttle
        feeds[oc._DT.name] = dt
        assert set(rows.input_ids) <= set(feeds)
        substitutions = {symbol: feeds[symbol.name] for expression in reference_expressions
                         for symbol in expression.free_symbols}
        exact_substitutions = _exact_reference_substitutions(substitutions)
        print(f"RK4 gate: {kind} reference evaluation begins at "
              f"{time.perf_counter()-began:.3f}s", flush=True)
        expected = []
        for index, expression in enumerate(reference_expressions):
            print(f"RK4 gate: {kind} value row {index+1}/7 evalf at "
                  f"{time.perf_counter()-began:.3f}s", flush=True)
            expected.append(float(_reference_eval(expression, exact_substitutions)))
        expected = np.asarray(expected)
        expected_J = []
        for index, row in enumerate(derivatives):
            expected_row = []
            for column, expression in enumerate(row):
                print(f"RK4 gate: {kind} Jacobian row {index+1}/7 "
                      f"column {column+1}/{len(row)} ({tuple(rows.gradient_ids)[column]}) "
                      f"evalf at {time.perf_counter()-began:.3f}s", flush=True)
                expected_row.append(float(_reference_eval(expression, exact_substitutions)))
            expected_J.append(expected_row)
        expected_J = np.asarray(expected_J)
        print(f"RK4 gate: {kind} reference ready at {time.perf_counter()-began:.3f}s; "
              "calling native Jacobian", flush=True)
        values, J = rows.jacobian(feeds)
        print(f"c0/t1 RK4 {kind}: 7 native rows, {J.shape[1]} derivative columns; "
              f"max value discrepancy {np.max(np.abs(values - expected)):.17g}; "
              f"max Jacobian discrepancy {np.max(np.abs(J - expected_J)):.17g}",
              flush=True)
        np.testing.assert_array_max_ulp(values, expected, maxulp=4)
        np.testing.assert_array_max_ulp(J, expected_J, maxulp=4)
        np.testing.assert_array_equal(rows.values(feeds), values)
        if propellant > 0.0:
            assert 0.0 < -values[-1] < propellant
        print(f"RK4 gate: {kind} native checks complete at "
              f"{time.perf_counter()-began:.3f}s", flush=True)
