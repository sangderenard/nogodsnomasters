import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivetrain_graph import build_drivetrain_graph
from engine_cycle_sim import EngineCycleSim
from engine_mesh import build_engine_mesh, build_moving_mesh
from engines import get
from turbine_parts import modules_for
from station_reference import drivetrain_power_path


def _agt():
    return get("agt1500-abrams-turbine")


def test_agt1500_modules_own_the_physical_package_without_proxy_mass():
    engine = _agt()
    graph = build_drivetrain_graph(engine)
    expected = modules_for(engine)
    records = [n for n in graph["nodes"] if n.get("of_engine") == engine.identity]

    assert {n["identity"] for n in records} == {m.identity for m in expected}
    assert sum(n["mass_kg"] for n in records) == engine.mass_kg == 1134.0
    assert all(n["kind"] == "engine-block-component" for n in records)
    assert all(n["mass_in_total"] for n in records)

    old_shells = [n for n in graph["nodes"]
                  if n.get("drawn_by") == "turbine_parts:module-records"]
    assert old_shells
    assert all(not n["mass_in_total"] for n in old_shells)


def test_agt1500_mesh_is_module_derived_and_rotors_are_bakeable():
    graph = build_drivetrain_graph(_agt())
    static, moving = build_engine_mesh(graph, crank_angle_deg=0.0,
                                       covers_off=True)
    later = build_moving_mesh(graph, crank_angle_deg=37.0, covers_off=True)

    assert static.n_triangles > 5000
    assert moving.n_triangles > 1000
    assert moving.part_names == later.part_names
    assert moving.vertices.shape == later.vertices.shape
    assert not np.allclose(moving.vertices, later.vertices)
    assert "node_turbine_recuperator_left" in static.part_names
    assert "node_turbine_power_turbine_rotor" in moving.part_names


def test_agt1500_live_cycle_advances_the_render_phase():
    sim = EngineCycleSim(_agt())
    sim.start()
    sim.throttle = 0.35
    before = sim.state.crank_angle_deg
    sim.step(1.0 / 60.0)

    assert sim.state.rpm > 0.0
    assert sim.state.crank_angle_deg != before


def test_powerplant_rigid_graph_reaches_transfer_case_through_clutch_and_box():
    for identity in ("ldt465-multifuel-deuce",
                     "agt1500-abrams-turbine"):
        path = drivetrain_power_path(build_drivetrain_graph(get(identity)))
        assert path == (
            "powertrain.engine", "powertrain.crank_shaft.rear",
            "powertrain.clutch", "powertrain.transmission",
            "powertrain.transfer_case")
