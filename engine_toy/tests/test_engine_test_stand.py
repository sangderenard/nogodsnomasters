import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_test_stand import EngineTestStand, TestStandCooling as StandCooling
from engines import get


def _sim(*, pump=None, exchanger=0.0, oil=False):
    circuit = SimpleNamespace(
        kind_class="thermal-liquid", nodes={"oil" if oil else "coolant"},
        pump_node=pump, active_heat_exchange_w_per_k=exchanger,
        flow_lpm=0.0, heat_share=0.3, temp_k=370.0,
        thermal_mass_kj_per_k=40.0)
    return SimpleNamespace(
        engine=SimpleNamespace(peak_power_kw=100.0),
        _drivetrain=SimpleNamespace(fluid_circuits=[circuit]),
        _drivetrain_out={}, state=SimpleNamespace())


@pytest.mark.parametrize("oil", [False, True])
def test_external_cooling_uses_finite_inventory_and_conserves_heat(oil):
    sim = _sim(oil=oil)
    circuit = sim._drivetrain.fluid_circuits[0]
    cooling = StandCooling.for_engine(sim)
    before = (circuit.temp_k * 40_000.0
              + cooling.reservoir_temp_k * cooling.thermal_capacity_j_per_k)
    cooling.advance(0.005, sim)
    after = (circuit.temp_k * 40_000.0
             + cooling.reservoir_temp_k * cooling.thermal_capacity_j_per_k)
    assert after + cooling.rejected_heat_w * 0.005 == pytest.approx(before)
    assert circuit.temp_k < 370.0
    assert cooling.reservoir_temp_k > cooling.ambient_k
    assert cooling.pump_flow_l_min > 0.0
    assert circuit.pump_node is None
    if oil:
        assert not hasattr(sim.state, "coolant_temp_k")


def test_complete_cooling_and_air_cooled_machines_do_not_gain_equipment():
    sim = _sim(pump="own-pump", exchanger=1500.0)
    assert StandCooling.for_engine(sim) is None
    sim._drivetrain.fluid_circuits.clear()
    assert StandCooling.for_engine(sim) is None


def test_real_turbine_test_stand_keeps_machine_spec_and_shared_snapshot_state():
    engine = get("agt1500-abrams-turbine")
    original_accessories = copy.deepcopy(engine.accessories)
    sim = EngineTestStand(engine)
    assert sim.test_stand_cooling is not None
    assert sim.engine.accessories == original_accessories
    circuit, interface = sim.test_stand_cooling._engine_circuit(sim)
    assert interface == "turbine-oil-to-coolant"
    circuit.temp_k = 370.0
    sim.step(0.005)
    assert circuit.temp_k < 370.0
    snapshot = copy.deepcopy(sim)
    assert snapshot.test_stand_cooling is snapshot.state.test_stand_cooling
    assert snapshot.test_stand_cooling is not sim.test_stand_cooling
