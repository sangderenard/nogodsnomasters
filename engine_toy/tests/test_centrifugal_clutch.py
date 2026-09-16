"""A centrifugal clutch cannot stall its engine, and now it exists.

`engines.py` has always described the trimmer as having "a real
centrifugal clutch (engages above idle to spin the cutting head -- an
exact fit for this toy's existing ClutchPort model)". Nothing implemented
one: the word appeared in `couplings.py` not at all, and in the sim only
for superchargers. The trimmer got a plain dry plate, which its load can
stall -- measured, it died 31 times in 90 seconds of racing and sat across
the track as a wall for everyone behind it.

The defining property is the zero. Below its engagement speed a
centrifugal clutch transmits NOTHING, so the engine idles on whatever the
head is doing. Everything here is about that zero being real and being
respected by the things that consume it.
"""
from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import couplings
import engines
from engine_cycle_sim import EngineCycleSim

RPM_TO_RAD = math.pi / 30.0
TRIMMER = "25cc-two-stroke-trimmer"


def test_a_small_two_stroke_gets_a_centrifugal_clutch():
    """Derived from what the engine DECLARES, not from its name.

    Two-stroke, and small enough that a clutch is the only thing between
    the crank and the tool. Matching on the identity string would be the
    substring rule this project keeps removing.
    """
    spec = couplings.recommended_for(engines.get(TRIMMER))
    assert spec.kind == "centrifugal"
    assert spec.apply_source == "none"
    assert spec.engage_rad_s > 0.0
    assert spec.rated_rad_s > spec.engage_rad_s


def test_a_road_engine_still_gets_a_plate():
    for identity in ("mazda-b6ze-miata-1990", "vw-vr6-2800-12v"):
        assert couplings.recommended_for(engines.get(identity)).kind != "centrifugal"


def _trimmer_clutch():
    engine = engines.get(TRIMMER)
    spec = couplings.recommended_for(engine)
    return spec, spec.build(engine.clutch_torque_nm)


def test_capacity_is_exactly_zero_below_engagement():
    """THE WHOLE POINT. Not small -- zero.

    A clutch that transmits a little below engagement can still be dragged
    down by its load, which is the behaviour this replaces.
    """
    spec, clutch = _trimmer_clutch()
    for fraction in (0.0, 0.25, 0.5, 0.9, 0.999):
        clutch.drive_omega_rad_s = spec.engage_rad_s * fraction
        assert clutch.torque_capacity_nm() == 0.0


def test_capacity_climbs_with_the_square_of_speed():
    """Shoes thrown outward: `k * (w**2 - w_engage**2)`, and monotonic."""
    spec, clutch = _trimmer_clutch()
    speeds = [spec.engage_rad_s * m for m in (1.1, 1.3, 1.6, 2.0, 2.4)]
    capacities = []
    for omega in speeds:
        clutch.drive_omega_rad_s = omega
        capacities.append(clutch.torque_capacity_nm())
    assert all(b > a for a, b in zip(capacities, capacities[1:])), capacities

    # the ratio of two capacities follows the ratio of (w^2 - engage^2)
    engage_sq = spec.engage_rad_s ** 2
    predicted = ((speeds[-1] ** 2 - engage_sq) / (speeds[0] ** 2 - engage_sq))
    assert capacities[-1] / capacities[0] == pytest.approx(predicted, rel=1e-9)


def test_it_holds_its_rating_at_the_speed_it_is_rated_for():
    spec, clutch = _trimmer_clutch()
    clutch.drive_omega_rad_s = spec.rated_rad_s
    assert clutch.torque_capacity_nm() == pytest.approx(
        clutch.rated_torque_nm, rel=1e-9)


def test_a_resting_zero_is_not_read_as_no_opinion():
    """`cap = capacity() or peak * 3` swallowed a legitimate 0.0.

    At rest a centrifugal clutch correctly reports zero capacity, which is
    falsy, so `or` read it as "no opinion" and substituted three times
    peak torque. The clutch that cannot stall its engine was being built
    as one that could hold anything.
    """
    sim = EngineCycleSim(engine=engines.get(TRIMMER))
    assert sim.coupling.kind == "centrifugal"
    peak = max(sim.engine.peak_torque_nm, 1.0)
    assert sim._brake_junction.max_torque_nm < peak * 3.0


def test_the_coupling_is_told_the_drive_speed_each_tick():
    """A speed-dependent capacity needs the speed.

    `Coupling.step` sets `drive_omega_rad_s` itself, but the friction path
    never calls it -- the port does the integrating and only `set_apply`
    is consulted -- so a centrifugal clutch would be asked for its
    capacity at a standstill forever.

    The junction is only solved when a ratio is engaged, so this engages
    one. In neutral `_current_gear_ratio()` is 0 and the coupling is never
    consulted at all, which matters for a trimmer because a trimmer has no
    gearbox: see the test below.
    """
    sim = EngineCycleSim(engine=engines.get(TRIMMER))
    sim.start()
    sim.throttle = 0.8
    sim.gear_index = 1
    for _ in range(60):
        sim.step(1.0 / 240.0)
    assert sim.coupling.drive_omega_rad_s > 0.0
    # Close but not equal, and it should not be: the coupling is told the
    # speed at the moment it is consulted, and `_omega` keeps integrating
    # through the remaining combustion substeps of the same tick.
    assert sim.coupling.drive_omega_rad_s == pytest.approx(
        abs(sim._omega), rel=0.01)


def test_it_idles_with_the_clutch_open():
    """The behaviour the whole kind exists for.

    A trimmer idles at 2800 rpm and engages around 4060, so at idle the
    clutch holds nothing, the head does not turn, and no load can reach
    the crank. Measured: 2786 rpm with a capacity of exactly 0.000 Nm.
    """
    sim = EngineCycleSim(engine=engines.get(TRIMMER))
    sim.start()
    sim.throttle = 0.8
    sim.gear_index = 1
    for _ in range(60):
        sim.step(1.0 / 240.0)
    assert sim.coupling.capacity_nm == 0.0
    assert sim.rpm > sim.engine.idle_rpm * 0.9


def test_the_junction_is_not_solved_in_neutral():
    """A limitation worth pinning rather than discovering twice.

    `_coupled_junction_torque` is reached only when a gear ratio is
    engaged. A trimmer has no gearbox -- crank, clutch, head -- so with
    `gear_index` at its default of neutral its centrifugal clutch is never
    consulted and never reports a speed. That is a real gap for any
    machine whose drive is a clutch and nothing else.
    """
    sim = EngineCycleSim(engine=engines.get(TRIMMER))
    sim.start()
    sim.throttle = 0.8
    for _ in range(60):
        sim.step(1.0 / 240.0)
    assert sim.gear_index == 0
    assert sim.coupling.drive_omega_rad_s == 0.0


def test_the_engagement_speed_sits_above_idle():
    """It has to: a clutch that grips at idle spins the head at idle."""
    engine = engines.get(TRIMMER)
    spec = couplings.recommended_for(engine)
    assert spec.engage_rad_s > engine.idle_rpm * RPM_TO_RAD
