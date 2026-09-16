"""The inertial solve as a table: one row per thing the machine is doing."""
import json
import math

import numpy as np
import pytest

import autoclave_production as ap
import feet
import mode_table as mt
from operating_states import (Cycle, OperatingState, Source, run_up, sources_of,
                              unbalance_from_grade)


@pytest.fixture(scope="module")
def doc():
    return ap.build().as_document()


@pytest.fixture(scope="module")
def table(doc):
    ft = feet.from_mounts(doc, pad_diameter_m=0.12)
    return mt.evaluate(doc, ft, feet.ground("tarmac"), ap.cycle(),
                       frame_underside_y=feet.underside_y(doc) + 0.06)


# ---------------------------------------------------------------------
# declarations
# ---------------------------------------------------------------------

def test_the_pump_declares_its_rotor_and_it_is_read_not_inferred(doc):
    srcs = sources_of(doc)
    assert [s.identity for s in srcs] == ["plant.autoclave.vacuum_pump"]
    pump = srcs[0]
    assert pump.axis == (1.0, 0.0, 0.0)
    assert pump.rated_rpm == 1450.0
    assert pump.runs("evacuate") and not pump.runs("sterilise")
    # G6.3 on a 9 kg rotor at 1450 rpm: e = 6.3 mm/s / 151.8 rad/s = 41 um
    assert pump.unbalance_kg_m == pytest.approx(9.0 * 6.3e-3 / (1450 * 2 * math.pi / 60), rel=1e-6)


def test_a_rotor_that_has_not_said_how_it_is_balanced_is_refused():
    from operating_states import source_of
    node = {"identity": "x", "reference_position": [0, 0, 0], "rotor_axis": (0, 1, 0),
            "rotor_rated_rpm": 3000.0, "rotor_inertia_kg_m2": 0.1, "runs_in": ("*",)}
    with pytest.raises(ValueError, match="rotor_mass_kg"):
        source_of(node)


def test_balance_grade_arithmetic_is_iso_21940():
    # G2.5 at 3000 rpm is an eccentricity of 8 um; on 10 kg that is 80 g.mm
    u = unbalance_from_grade(10.0, 2.5, 3000.0)
    assert u == pytest.approx(10.0 * 2.5e-3 / (3000 * 2 * math.pi / 60), rel=1e-9)
    assert u * 1e6 == pytest.approx(79.6, rel=0.01)


def test_the_cycle_samples_a_run_up_and_holds_a_steady_state():
    cyc = ap.cycle()
    names = [s.name for s in cyc.states]
    assert names == ["purge", "evacuate", "sterilise", "exhaust"]
    samples = list(cyc.samples())
    assert sum(1 for s, f in samples if s.name == "evacuate") == 10
    assert sum(1 for s, f in samples if s.name == "purge") == 1


# ---------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------

def test_every_body_has_a_load_path_so_the_table_has_no_problems(table):
    assert table.problems == ()


def test_the_pump_is_applied_only_in_the_state_it_runs_in(table):
    for r in table.rows:
        running = [e for e in r.effects if e.applied]
        if r.state == "evacuate":
            assert len(running) == 1
            assert running[0].path_kind == "isolated"
        else:
            assert running == []


def test_the_pumps_force_reaches_the_frame_through_its_isolators(table):
    """Transmission is one at the bottom of the run-up, peaks near the
    isolator frequency, and is above one at rated -- the pump is fitted
    on isolators that are too soft to isolate it at 24 Hz, which is the
    table telling the designer something rather than hiding it."""
    rows = [r for r in table.rows if r.state == "evacuate"]
    trans = [r.effects[0].transmission for r in rows]
    assert trans[0] == pytest.approx(1.0, abs=0.01)
    assert max(trans) > 2.0
    assert trans.index(max(trans)) not in (0, len(trans) - 1)


def test_a_full_collector_moves_the_modes(table):
    """Two states differ by nothing but water in a container, and that
    is enough to lower every rocking frequency."""
    empty = table.at("purge", 1.0)
    full = table.at("exhaust", 1.0)
    assert full.rigid_body.total_mass_kg == pytest.approx(
        empty.rigid_body.total_mass_kg + ap.COLLECTOR_FULL_KG, abs=1e-6)
    assert all(f < e for f, e in zip(full.frequencies_hz, empty.frequencies_hz))


def test_a_horizontal_rotor_does_not_split_the_modes(table):
    """The pump's axis is along x. Its gyroscopic moment would yaw the
    frame, and vertical feet do not resist yaw: the model's stated
    boundary, so the modes at full speed equal the modes at rest."""
    rest = table.at("evacuate", 0.05)
    rated = table.at("evacuate", 1.0)
    assert rated.angular_momentum_y == 0.0
    assert rated.frequencies_hz == pytest.approx(rest.frequencies_hz)


def test_a_vertical_rotor_splits_the_rocking_modes():
    import machine_shake as shake
    from engine_mass_properties import RigidBodyProperties
    rb = RigidBodyProperties(200.0, (0.0, 0.3, 0.0), np.diag([12.0, 8.0, 10.0]))
    ft = [feet.LevellingFoot(f"f.{n}", p, pad_diameter_m=0.08)
          for n, p in (("a", (-0.3, 0, -0.25)), ("b", (-0.3, 0, 0.25)),
                       ("c", (0.3, 0, -0.25)), ("d", (0.3, 0, 0.25)))]
    g = feet.ground("firm-soil")
    st = feet.stand(ft, rb, g, frame_underside_y=0.06)
    model = shake.build("t", st, ft, rb, g)
    still = shake.whirl_modes(model, 0.0)
    assert [m.frequency_hz for m in still] == pytest.approx(
        [m.frequency_hz for m in model.modes])
    fast = shake.whirl_modes(model, 300.0)
    rock_still = sorted(m.frequency_hz for m in still if m.kind != "heave")
    rock_fast = sorted(m.frequency_hz for m in fast if m.kind != "heave")
    # backward whirl falls, forward rises: they move apart
    assert rock_fast[0] < rock_still[0]
    assert rock_fast[-1] > rock_still[-1]


# ---------------------------------------------------------------------
# the prebake
# ---------------------------------------------------------------------

def test_the_baked_table_is_plain_data_and_round_trips_through_json(table):
    baked = mt.bake(table)
    text = json.dumps(baked)
    back = json.loads(text)
    assert back["schema"] == "engine-toy-mode-table-v1"
    assert len(back["rows"]) == len(table.rows)
    assert back["paths"]["plant.autoclave.vessel"]["kind"] == "rigid"
    assert back["paths"]["plant.autoclave.vacuum_pump"]["kind"] == "isolated"


def test_lookup_is_nearest_row_and_a_frame_is_one_multiply(table):
    baked = mt.bake(table)
    row = mt.lookup(baked, "evacuate", 0.97)
    assert row["sample"] == 1.0
    assert row["sources"][0]["rpm"] == 1450.0
    motion = mt.frame_motion(row, 0.0)
    assert motion["heave_m"] == pytest.approx(row["sources"][0]["heave_m"])
    with pytest.raises(KeyError):
        mt.lookup(baked, "no-such-state", 1.0)


def test_the_cycle_elects_live_or_baked_and_the_table_carries_it(doc):
    ft = feet.from_mounts(doc, pad_diameter_m=0.12)
    live = mt.evaluate(doc, ft, feet.ground("tarmac"), ap.cycle(live_vibration=True),
                       frame_underside_y=feet.underside_y(doc) + 0.06)
    assert live.live_vibration is True
    assert mt.bake(live)["live_vibration"] is True
