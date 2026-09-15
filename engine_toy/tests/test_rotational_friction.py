from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drivetrain_graph import DrivetrainSolver, FluidCircuit


def _solver(kind="rotational-bearing", *, crank=False):
    a = "powertrain.engine" if crank else "rotor.a"
    graph = {
        "nodes": [
            {"identity": a, "kind": "rotating-mass", "inertia_kg_m2": 2.0},
            {"identity": "rotor.b", "kind": "rotating-mass", "inertia_kg_m2": 0.01},
        ],
        "edges": [{"identity": "bearing", "a": a, "b": "rotor.b",
                   "constraint": kind, "bearing_drag_nm": 100.0,
                   "max_torque_nm": 100.0, "normal_force_n": 1000.0,
                   "friction_coefficient": 1.0, "contact_radius_m": 0.1}],
    }
    return DrivetrainSolver(graph), a


@pytest.mark.parametrize("kind", ["rotational-bearing", "friction-clutch-shaft",
                                  "rolling-friction-contact"])
@pytest.mark.parametrize("speed", [-10.0, 10.0])
def test_friction_stops_relative_motion_without_creating_energy(kind, speed):
    solver, a = _solver(kind)
    solver.omega[a] = speed
    before_energy = 0.5 * 2.0 * speed ** 2
    before_momentum = 2.0 * speed
    result = solver.step(0.1, 0.0)
    after_energy = (0.5 * 2.0 * solver.omega[a] ** 2
                    + 0.5 * 0.01 * solver.omega["rotor.b"] ** 2)
    assert solver.omega[a] == pytest.approx(solver.omega["rotor.b"])
    assert 2.0 * solver.omega[a] + 0.01 * solver.omega["rotor.b"] == pytest.approx(before_momentum)
    heat = solver._edge_state["bearing"].dissipated_heat_j
    assert after_energy + heat == pytest.approx(before_energy)
    assert heat > 0.0
    assert result["friction_heat_w"] == pytest.approx(heat / 0.1)
    assert solver._edge_state["bearing"].unrejected_heat_j == heat


def test_prescribed_crank_supplies_work_and_receives_bearing_reaction():
    solver, a = _solver(crank=True)
    result = solver.step(0.1, 10.0)
    assert solver.omega[a] == 10.0
    assert solver.omega["rotor.b"] == pytest.approx(10.0)
    work = result["crank_reaction_torque_nm"] * 10.0 * 0.1
    heat = solver._edge_state["bearing"].dissipated_heat_j
    assert work == pytest.approx(0.5 * 0.01 * 10.0 ** 2 + heat)


def test_friction_heat_enters_only_its_connected_liquid_circuit():
    solver, a = _solver()
    circuit = FluidCircuit("thermal-liquid", frozenset({a}), (),
                           thermal_mass_kj_per_k=10.0)
    solver.fluid_circuits.append(circuit)
    solver.omega[a] = 10.0
    initial_temp = circuit.temp_k
    # Isolate heat delivery from the circuit's separate ambient rejection.
    solver._step_fluid_circuits = lambda *args, **kwargs: None
    solver.step(0.1, 0.0)
    state = solver._edge_state["bearing"]
    assert (circuit.temp_k - initial_temp) * 10_000.0 == pytest.approx(state.dissipated_heat_j)
    assert state.unrejected_heat_j == 0.0
