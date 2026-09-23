from types import SimpleNamespace

import pytest

from dt_benchmark import CycleEngine
from machines import Machine, MachineSim, MachineSystem
from src.common.dt_system.state_table import StateTable
from src.common.dt_system.time_contracts import HOLD, SUBCYCLE


def _register(engine):
    table = StateTable()
    engine.register(
        table, lambda _: {"pos": (0.0, 0.0, 0.0), "mass": 0.0},
        (engine,), group_label="mode-contract")
    return table


class _CycleProbe:
    def __init__(self):
        self.rpm = 1200.0
        self.advanced = 0.0

    def step(self, dt):
        self.advanced += float(dt)

    def sync_state_span(self):
        import numpy as np
        return np.asarray([self.rpm, self.advanced])

    def load_state_span(self, span):
        self.rpm, self.advanced = map(float, span)


def test_cycle_engine_declares_its_internal_fixed_step_as_subcycle():
    sim = _CycleProbe()
    engine = CycleEngine(sim)
    table = _register(engine)

    ok, metrics, _ = engine.step_with_state(
        {}, 0.01, realtime=True, state_table=table)

    assert ok
    assert sim.advanced == pytest.approx(0.01)
    assert metrics.pub_contract.tolist() == [SUBCYCLE]
    assert metrics.pub_exchange_time_present.tolist() == [1.0]
    assert metrics.pub_exchange_time.tolist() == pytest.approx([0.001])
    assert engine.causal_ceiling_dt() == pytest.approx(0.05)


def test_machine_system_is_transactional_and_honestly_holds_without_a_tau():
    sim = MachineSim(Machine(
        identity="test.machine", label="test machine", medium="electric"))
    engine = MachineSystem(sim)
    table = _register(engine)
    checkpoint = engine.snapshot()

    ok, metrics, _ = engine.step_with_state(
        sim, 0.02, realtime=False, state_table=table)

    assert ok
    assert sim.elapsed_s == pytest.approx(0.02)
    assert metrics.pub_contract.tolist() == [HOLD]
    assert metrics.pub_exchange_time_present.tolist() == [0.0]
    engine.restore(checkpoint)
    assert sim.elapsed_s == 0.0
    assert engine.world_time == 0.0

