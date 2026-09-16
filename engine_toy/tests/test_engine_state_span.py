"""The engine's whole state is a flat span, and it comes back exactly.

These pin `engine_state.py`, which closed the gap `compile_contract.py`
had been naming for a long time: "EngineCycleSim is not declared here yet
-- its state is a graph of Python objects rather than typed spans". It is
declared now, and the declaration is only worth anything if it is
COMPLETE. A span that carries most of the state round-trips perfectly and
still fails to reproduce a trajectory, which is the failure these exist
to catch.

The history is the reason for the shape of the tests. The declaration
began at 304 slots against a real 3154 -- about a tenth. What was missing
was not obscure: 1784 doubles of per-edge torsional wind-up (the solver
declared its shaft SPEEDS and none of its ANGLES, so it had the
driveline's momentum and none of its potential), the generator positions,
the 53 accumulators the sim carries on itself, and the whole 103-field
state record. None of it was found by reading names. All of it was found
by diffing a restored sim against the original and looking at what moved.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engines
from engine_abi import engine_graph_abi
from engine_cycle_sim import EngineCycleSim
from engine_state import coverage, pack, unpack, round_trip, UNMAPPED


ENGINE = "mazda-b6ze-miata-1990"


def _warmed(identity: str = ENGINE, ticks: int = 120) -> EngineCycleSim:
    """A sim with something interesting in it.

    A cold sim has zeros nearly everywhere and would pass a span test that
    carried almost nothing, so every test here runs the engine first.
    """
    sim = EngineCycleSim(engine=engines.get(identity))
    sim.start()
    sim.throttle = 0.6
    for _ in range(ticks):
        sim.step(1.0 / 240.0)
    return sim


def test_every_declared_slot_is_carried():
    """No holes. A declared span nobody fills is worse than none."""
    sim = _warmed()
    report = coverage(engine_graph_abi(sim.engine))
    assert report["hole_slots"] == 0, report["hole_spans"]
    assert report["carried_slots"] == report["stride"]
    assert UNMAPPED == ()


def test_the_span_is_not_mostly_empty():
    """The layout must describe a running engine, not an idea of one.

    A stride of three thousand slots proves nothing if the sim only ever
    writes thirty of them; the first version of this declaration was 304
    slots wide and the engine's real state was ten times that.
    """
    sim = _warmed()
    span = pack(sim, engine_graph_abi(sim.engine))
    assert span.size > 3000
    assert np.count_nonzero(span) > 300


def test_round_trip_is_exact():
    """pack -> unpack -> pack changes nothing."""
    sim = _warmed()
    assert round_trip(sim, engine_graph_abi(sim.engine)) == 0.0


def test_restore_is_exact_after_stepping_away():
    """Stronger than a round trip, which never advances the sim.

    This is the one that catches a span that READS a field correctly and
    WRITES it into the wrong place -- the class of bug that put the firing
    order (1, 3, 4, 2) into the combustion state, because four of the
    per-cylinder collections are dicts keyed by cylinder number and
    iterating a mapping walks its keys.
    """
    sim = _warmed()
    abi = engine_graph_abi(sim.engine)
    saved = pack(sim, abi)
    for _ in range(25):
        sim.step(1.0 / 240.0)
    unpack(sim, abi, saved)
    again = pack(sim, abi)
    both_nan = np.isnan(saved) & np.isnan(again)
    delta = np.nan_to_num(np.abs(np.where(both_nan, 0.0, saved - again)),
                          nan=np.inf)
    assert int(np.count_nonzero(delta)) == 0


def test_the_generator_position_travels_with_the_state():
    """Seeding gives determinism only from a POSITION.

    Carrying the seed and not the position was worth 18.8 rpm of
    divergence on its own: two replays from one span took different random
    draws, and knock and misfire follow those draws.
    """
    sim = _warmed()
    abi = engine_graph_abi(sim.engine)
    saved = pack(sim, abi)
    first = [sim._rng.random() for _ in range(5)]
    unpack(sim, abi, saved)
    assert [sim._rng.random() for _ in range(5)] == first


def test_replay_from_one_span_is_nearly_deterministic():
    """Two replays from one span follow the same trajectory.

    NOT YET EXACT, and deliberately asserted as a bound rather than as
    equality. Everything reachable in the object graph restores exactly
    and the residual is something outside it that has not been found; see
    `engine_state.py` for what has been ruled out. The bound is here to
    stop it getting worse, and it is tight enough that any of the causes
    already eliminated would blow through it -- the generator alone was
    18.8 rpm, the edge state 1.1e-3.
    """
    sim = _warmed()
    abi = engine_graph_abi(sim.engine)
    saved = pack(sim, abi)

    def replay():
        unpack(sim, abi, saved)
        out = []
        for _ in range(40):
            sim.step(1.0 / 240.0)
            out.append(float(sim.rpm))
        return out

    first, second = replay(), replay()
    worst = max(abs(a - b) for a, b in zip(first, second))
    assert worst < 1.0e-3, worst


def test_the_class_carries_its_declared_span():
    """`state_span` is the field `program_abi.records` declares."""
    sim = _warmed()
    span = sim.sync_state_span()
    assert span.dtype == np.float64
    assert span.size == sim.state_span_length()
    rpm = float(sim.rpm)
    for _ in range(30):
        sim.step(1.0 / 240.0)
    sim.load_state_span()
    sim.step(1.0 / 240.0)
    assert abs(float(sim.rpm) - rpm) < 25.0


@pytest.mark.parametrize("identity", ["mazda-b6ze-miata-1990",
                                      "vw-vr6-2800-12v"])
def test_stride_is_a_property_of_the_topology(identity):
    """Two engines of different shape declare different strides.

    A batch is one topology because the lane stride is a compile-time
    constant, so a stride that did not move with the machine would make
    `abi_negotiator` bucket things that cannot share an assembly.
    """
    sim = _warmed(identity, ticks=40)
    abi = engine_graph_abi(sim.engine)
    assert abi.state_stride == pack(sim, abi).size


def test_two_engines_have_distinct_strides():
    strides = {
        identity: engine_graph_abi(engines.get(identity)).state_stride
        for identity in ("mazda-b6ze-miata-1990", "vw-vr6-2800-12v")
    }
    assert len(set(strides.values())) == 2, strides
