"""Structural completeness checks for the honorary Timoshenko law family."""

import sympy as sp

import honorary_engine_equation_catalogue as catalogue


def _section(equations, number):
    prefix = f"eq_T{number}_"
    return {name: law for name, law in equations.items()
            if name.startswith(prefix)}


def test_timoshenko_has_complete_linear_space_beam_state_and_problem_data():
    equations = catalogue.equations_by_engine("Timoshenko")

    # Six strains, six resultants plus isotropic closure/stress recovery, and
    # six balance laws are the irreducible 3-D linear Timoshenko statement.
    assert len(_section(equations, 13)) == 6
    assert len(_section(equations, 14)) == 8
    assert len(_section(equations, 15)) == 6

    # A field equation without energies, weak form, boundary/initial data and
    # its consistent element operators is not a complete solvable problem.
    assert len(_section(equations, 16)) == 2
    assert len(_section(equations, 17)) == 8
    assert len(_section(equations, 18)) == 5

    assert all(isinstance(law, sp.Equality) for law in equations.values())
    assert catalogue.check_law_declarations() == []


def test_timoshenko_keeps_shear_and_rotary_inertia_in_reference_witnesses():
    equations = catalogue.equations_by_engine("Timoshenko")

    static_text = " ".join(map(str, _section(equations, 19).values()))
    wave_text = str(equations["eq_T20_1"])
    assert "kGA" in static_text
    assert "rhoI" in wave_text
    assert "kGA_w" in wave_text


def test_large_rigid_rotation_uses_objective_shear_deformable_extension():
    equations = catalogue.equations_by_engine("Timoshenko")
    exact = _section(equations, 21)

    assert len(exact) == 9
    orthogonality = str(exact["eq_T21_1"])
    assert "R_g(x, t)" in orthogonality
    assert "transpose(R_g(x, t))" in orthogonality
    assert "Derivative(R_g(x, t), x)" in str(exact["eq_T21_4"])
    assert "Derivative(R_g(x, t), t)" in str(exact["eq_T21_5"])
    assert "cross" in str(exact["eq_T21_9"])
