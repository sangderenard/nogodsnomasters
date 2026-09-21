"""The dt adapter must not throw away a save/restore the engine has.

`CycleEngine.snapshot`/`restore` returned `None`. Read from the dt
system's side that says "this engine has no restorable state", which is
false: `engine_state.py` carries the whole live sim as one flat span with
an exact round trip, and the two OTHER engines in `dt_benchmark.py` -- the
spring bank and the fluid bank -- have always snapshotted their real
arrays. Only the one with the best save/restore in the tree returned
nothing.

The cost of that stub is not theoretical. A `None` snapshot is why the
engine batch was put on the realtime lane: the scientific lane takes a
checkpoint per attempt and rolls back, and there was nothing to roll back
to. The lane choice was made from a stub rather than from the engine.

NOTE ON COMPARISON: the span legitimately carries NaN (one slot, on the
engines here), so `np.array_equal` without `equal_nan=True` reports every
exact round trip as a failure. `span_diff.py` handles the same thing with
an explicit `both_nan` mask.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engines
from dt_benchmark import CycleEngine
from engine_cycle_sim import EngineCycleSim

DT = 1.0 / 1024.0
IDENTITY = "vw-vr6-2800-12v"


def _running(steps: int = 60) -> CycleEngine:
    sim = EngineCycleSim(engine=engines.get(IDENTITY))
    sim.start()
    sim.throttle = 0.9
    sim.gear_index = 1
    engine = CycleEngine(sim)
    for _ in range(steps):
        engine.step(DT)
    return engine


def test_a_snapshot_is_the_whole_declared_span():
    engine = _running()
    snapshot = engine.snapshot()
    assert snapshot is not None, "the adapter is discarding the engine's state"
    assert snapshot.size == engine.sim.state_span_length()
    assert snapshot.size > 1000, snapshot.size


def test_fresh_engine_can_be_checkpointed_before_its_first_tick():
    """A dt round checkpoints every participant before advancing any one."""
    sim = EngineCycleSim(engine=engines.get(IDENTITY))
    engine = CycleEngine(sim)
    snapshot = engine.snapshot()
    assert snapshot.size == sim.state_span_length()




def test_restoring_puts_the_engine_back():
    """Step away, put it back, and the crank is where it was."""
    engine = _running()
    snapshot = engine.snapshot()
    at_snapshot = engine.sim.rpm
    for _ in range(40):
        engine.step(DT)
    assert engine.sim.rpm != pytest.approx(at_snapshot), "it never moved"
    engine.restore(snapshot)
    assert engine.sim.rpm == at_snapshot


def test_the_restored_span_is_exact():
    engine = _running()
    snapshot = engine.snapshot()
    for _ in range(40):
        engine.step(DT)
    engine.restore(snapshot)
    assert np.array_equal(engine.sim.sync_state_span(), snapshot, equal_nan=True)


def test_a_snapshot_survives_the_steps_it_protects_against():
    """THE COPY IS NOT OPTIONAL.

    `sync_state_span` packs INTO `self.state_span` and returns that same
    array. A snapshot that handed it back without copying would be
    rewritten in place by the very steps it was taken to guard against --
    and would then restore the present as though it were the past, with
    nothing anywhere reporting a problem.
    """
    engine = _running()
    held = engine.snapshot()
    before = held.copy()
    for _ in range(10):
        engine.step(DT)
    assert np.array_equal(held, before, equal_nan=True)


def test_restore_tolerates_nothing_to_restore():
    """The scientific lane can ask before anything was ever taken."""
    engine = _running(steps=1)
    assert engine.restore(None) is None
