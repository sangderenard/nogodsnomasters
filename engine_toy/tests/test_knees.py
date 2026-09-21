"""Knees: where a thing stops being fine, with a width rather than a line.

These are the consequence vocabulary -- yield, cracking pressure,
saturation, turn-on, cut-in, ampacity -- so they have to be right in
three separate ways.  They must reduce to the hard threshold in the
limit, so a sharp knee is still a knee.  They must be smooth and
differentiable everywhere, because a branch has no derivative exactly
where the interesting physics is.  And where the sharpness has a
physical origin, it must come from the physics instead of being tuned:
``DiodeKnee`` is handed a saturation current and returns 0.714 V at an
ampere without ever being told about 0.7 V.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from knees import (
    BOLTZMANN_OVER_CHARGE_V_K,
    Ceiling,
    DiodeKnee,
    Knee,
    harmonic_blend,
    logistic,
    smooth_maximum,
    smooth_minimum,
    softplus,
)


# --------------------------------------------------------------------------
# the smooth hinge
# --------------------------------------------------------------------------

def test_softplus_becomes_the_hard_hinge_as_sharpness_vanishes() -> None:
    values = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])

    assert np.allclose(softplus(values, 1.0e-9), np.maximum(values, 0.0))
    assert np.allclose(softplus(values, 0.0), np.maximum(values, 0.0))


def test_softplus_rounds_the_corner_by_exactly_s_ln2() -> None:
    """A real knee is not a point, and this says how much it is not."""
    for sharpness in (0.1, 1.0, 7.0):
        assert softplus(0.0, sharpness) == pytest.approx(
            sharpness * math.log(2.0))


def test_softplus_does_not_overflow_on_a_large_argument() -> None:
    """Spelled through logaddexp, so far past the knee it just returns x."""
    assert softplus(1.0e5, 1.0) == pytest.approx(1.0e5)
    assert np.isfinite(softplus(1.0e12, 2.0))


def test_the_hinge_identity_holds_at_any_sharpness() -> None:
    """``excess - deficit == value - location``, exactly.

    ``softplus(z) - softplus(-z) = z`` for every sharpness, so however
    rounded the corner is, what is past the knee minus what is short of
    it is still the distance to it.  No tolerance-fitting can fake this.
    """
    knee = Knee("yield", location=250.0e6, sharpness=8.0e6, units="Pa")
    values = np.array([0.0, 100.0e6, 250.0e6, 400.0e6])

    assert np.allclose(knee.excess(values) - knee.deficit(values),
                       values - knee.location)


def test_logistic_is_the_derivative_of_softplus() -> None:
    """Not merely similar: the same function, to eight decimals."""
    sharpness, step = 0.7, 1.0e-6
    for point in (-2.0, -0.3, 0.0, 1.5, 3.0):
        numeric = (softplus(point + step, sharpness)
                   - softplus(point - step, sharpness)) / (2.0 * step)
        assert numeric == pytest.approx(
            float(logistic(point, sharpness)), abs=1e-8)


def test_the_gate_runs_from_zero_to_one_through_a_half() -> None:
    knee = Knee("crack", location=3.0e5, sharpness=1.0e4, units="Pa")

    assert float(knee.gate(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(knee.gate(knee.location)) == pytest.approx(0.5)
    assert float(knee.gate(1.0e6)) == pytest.approx(1.0)
    rising = knee.gate(np.linspace(2.5e5, 3.5e5, 50))
    assert np.all(np.diff(rising) > 0.0)


def test_the_slope_of_the_excess_is_the_gate() -> None:
    knee = Knee("k", location=10.0, sharpness=2.0)

    assert np.allclose(knee.slope(np.array([5.0, 10.0, 15.0])),
                       knee.gate(np.array([5.0, 10.0, 15.0])))


# --------------------------------------------------------------------------
# what a knee implies
# --------------------------------------------------------------------------

def test_hardness_says_how_knee_like_a_knee_is() -> None:
    soft = Knee("relief", 3.0e5, 1.0e4)
    hard = Knee("relief", 3.0e5, 1.0e2)

    assert hard.hardness == pytest.approx(3000.0)
    assert soft.hardness == pytest.approx(30.0)
    assert Knee("ideal", 3.0e5, 0.0).hardness == math.inf


def test_a_sharper_knee_demands_a_smaller_timestep() -> None:
    """Stiffness with a number on it, for the dt system to read."""
    soft = Knee("relief", 3.0e5, 1.0e4)
    hard = Knee("relief", 3.0e5, 1.0e2)

    assert hard.resolving_dt(1.0e6) == pytest.approx(
        soft.resolving_dt(1.0e6) / 100.0)
    # Nothing approaching the knee never crosses it.
    assert soft.resolving_dt(0.0) == math.inf


def test_a_negative_sharpness_is_refused() -> None:
    with pytest.raises(ValueError, match="sharpness cannot be negative"):
        Knee("bad", 1.0, -1.0)


# --------------------------------------------------------------------------
# saturation, the knee seen from the other side
# --------------------------------------------------------------------------

def test_a_ceiling_tracks_its_input_then_stops() -> None:
    ceiling = Ceiling("m19", limit=1.95, units="T")

    assert float(ceiling.apply(0.02)) == pytest.approx(0.02, rel=1e-3)
    assert float(ceiling.apply(100.0)) == pytest.approx(1.95, rel=1e-6)
    assert float(ceiling.apply(-100.0)) == pytest.approx(-1.95, rel=1e-6)


def test_a_ceiling_is_monotonic_and_never_exceeded() -> None:
    """Non-decreasing everywhere, and strictly rising while that is resolvable.

    Deep in saturation ``tanh`` is within a machine epsilon of one, so
    consecutive outputs become bit-identical and the difference is
    exactly zero rather than merely small.  That is the right behaviour
    for a saturation model -- the ceiling is reached, not approached
    forever -- so the assertion is non-decreasing, with strict increase
    asserted only where float64 can still tell the values apart.
    """
    ceiling = Ceiling("m19", limit=1.95)
    driven = np.linspace(0.0, 50.0, 200)
    applied = ceiling.apply(driven)

    assert np.all(np.diff(applied) >= 0.0)
    assert np.all(applied <= ceiling.limit)

    resolvable = ceiling.apply(np.linspace(0.0, 2.0 * ceiling.limit, 200))
    assert np.all(np.diff(resolvable) > 0.0)
    assert np.all(resolvable < ceiling.limit)


def test_headroom_falls_to_nothing() -> None:
    ceiling = Ceiling("m19", limit=1.95)

    assert float(ceiling.headroom(0.0)) == pytest.approx(1.0)
    assert float(ceiling.headroom(50.0)) == pytest.approx(0.0, abs=1e-6)


def test_headroom_survives_deep_saturation() -> None:
    """How far past the ceiling a part is must not round away to zero.

    ``1 - tanh(u)`` subtracts nearly equal numbers and collapses to
    exactly 0.0 by about ``u = 19``.  The identity ``2 / (e^2u + 1)``
    subtracts nothing.  This is the range reduction the proof cores
    require anyway -- they will not evaluate ``tanh`` beyond a radius of
    0.5 -- so the reduced form is the prerequisite for extra limbs, not
    a way of avoiding them.
    """
    ceiling = Ceiling("m19", limit=1.95)

    # tanh reaches 1.0 in float64 near u = 19, so a drive of 100 against
    # a 1.95 limit (u = 51) is well past where the subtraction dies.
    assert 1.0 - np.tanh(100.0 / 1.95) == 0.0, (
        "the naive form is expected to have collapsed by here")

    assert float(ceiling.headroom(25.0)) > 1e-12
    assert float(ceiling.headroom(100.0)) > 1e-50
    assert float(ceiling.headroom(600.0)) > 0.0

    # Still monotonic, and still bounded, all the way down.
    deep = ceiling.headroom(np.array([25.0, 100.0, 600.0]))
    assert np.all(np.diff(deep) < 0.0)
    assert np.all(deep > 0.0)


def test_the_bend_is_reported_as_a_region_not_a_threshold() -> None:
    """Nobody can name the point where tanh stops being straight."""
    ceiling = Ceiling("m19", limit=1.95, units="T")
    bend = ceiling.knee

    assert 0.5 * ceiling.limit < bend.location < 0.8 * ceiling.limit
    assert bend.sharpness > 0.0
    assert bend.units == "T"


def test_a_ceiling_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Ceiling("bad", limit=0.0)


# --------------------------------------------------------------------------
# where the sharpness is physics
# --------------------------------------------------------------------------

def test_the_thermal_voltage_is_kt_over_q() -> None:
    diode = DiodeKnee("d", ideality=1.0, temperature_k=300.0)

    assert diode.thermal_voltage_v == pytest.approx(
        BOLTZMANN_OVER_CHARGE_V_K * 300.0)
    assert diode.thermal_voltage_v == pytest.approx(0.02585, abs=1e-4)


def test_the_familiar_seven_tenths_of_a_volt_is_derived() -> None:
    """Nothing here is told about 0.7 V; it appears at an ampere.

    And 0.536 V appears at a milliamp, which is the point: a diode's
    "knee voltage" was never a constant, and a model that takes one as a
    parameter has thrown the dependence away.
    """
    diode = DiodeKnee("1n4148", saturation_current_a=1.0e-12, ideality=1.0)

    assert float(diode.voltage_v(1.0)) == pytest.approx(0.714, abs=0.005)
    assert float(diode.voltage_v(1.0e-3)) == pytest.approx(0.536, abs=0.005)
    assert float(diode.voltage_v(1.0e-6)) == pytest.approx(0.357, abs=0.005)


def test_current_and_voltage_invert_each_other() -> None:
    diode = DiodeKnee("d", saturation_current_a=1.0e-12)

    for current in (1.0e-6, 1.0e-3, 0.5, 2.0):
        assert float(diode.current_a(diode.voltage_v(current))) == pytest.approx(
            current, rel=1e-6)


def test_reverse_bias_saturates_at_minus_the_saturation_current() -> None:
    """``expm1`` keeps this exact instead of cancelling it to zero."""
    diode = DiodeKnee("d", saturation_current_a=2.5e-12)

    assert float(diode.current_a(-5.0)) == pytest.approx(-2.5e-12, rel=1e-9)


def test_a_stack_multiplies_the_voltage_and_the_sharpness() -> None:
    """Ten dies in series stand off ten times as much and turn on as slowly."""
    single = DiodeKnee("one", series_dies=1)
    stack = DiodeKnee("ten", series_dies=10)

    assert float(stack.voltage_v(0.3)) == pytest.approx(
        10.0 * float(single.voltage_v(0.3)), rel=1e-9)
    assert stack.knee_at(0.3).sharpness == pytest.approx(
        10.0 * single.knee_at(0.3).sharpness)


def test_the_diodes_knee_carries_its_own_provenance() -> None:
    knee = DiodeKnee("d").knee_at(0.3)

    assert knee.units == "V"
    assert "Shockley" in knee.note
    assert knee.hardness > 1.0


def test_a_hotter_junction_turns_on_more_gently() -> None:
    """Sharpness is kT/q, so heat literally blunts the knee."""
    cold = DiodeKnee("d", temperature_k=250.0)
    hot = DiodeKnee("d", temperature_k=400.0)

    assert hot.thermal_voltage_v > cold.thermal_voltage_v


def test_an_impossible_junction_is_refused() -> None:
    with pytest.raises(ValueError, match="saturation current must be positive"):
        DiodeKnee("bad", saturation_current_a=0.0)
    with pytest.raises(ValueError, match="ideality must be positive"):
        DiodeKnee("bad", ideality=0.0)


# --------------------------------------------------------------------------
# blends, and the one that is not a minimum
# --------------------------------------------------------------------------

def test_the_harmonic_blend_returns_the_smaller_when_they_are_separated() -> None:
    assert float(harmonic_blend(1.0, 100.0)) == pytest.approx(1.0, rel=1e-3)
    assert float(harmonic_blend(100.0, 1.0)) == pytest.approx(1.0, rel=1e-3)


def test_the_harmonic_blend_undershoots_by_root_two_at_the_crossover() -> None:
    """Pinned because it is a trap, not because it is a bug.

    ``a b / sqrt(a^2 + b^2)`` with ``a == b`` gives ``a / sqrt 2`` -- 29%
    under.  For two competing regimes both half-active that is arguably
    right; for a cap or a supply limit it is wrong, and this test exists
    so nobody reaches for it expecting a minimum.
    """
    assert float(harmonic_blend(5.0, 5.0)) == pytest.approx(
        5.0 / math.sqrt(2.0))
    assert float(harmonic_blend(5.0, 5.0)) < 0.72 * 5.0


def test_smooth_minimum_actually_is_one() -> None:
    """Its error is bounded by the sharpness, not by a fraction of the value."""
    assert smooth_minimum(5.0, 5.0, 0.0) == pytest.approx(5.0)
    assert float(smooth_minimum(5.0, 5.0, 0.1)) == pytest.approx(
        5.0 - 0.1 * math.log(2.0))
    assert float(smooth_minimum(2.0, 9.0, 0.01)) == pytest.approx(2.0, abs=1e-3)


def test_smooth_maximum_is_the_same_the_other_way_up() -> None:
    assert smooth_maximum(2.0, 9.0, 0.0) == pytest.approx(9.0)
    assert float(smooth_maximum(5.0, 5.0, 0.1)) == pytest.approx(
        5.0 + 0.1 * math.log(2.0))


def test_the_smooth_extremes_bracket_the_true_ones() -> None:
    values = [(1.0, 4.0), (7.0, 7.0), (0.5, 100.0)]
    for a, b in values:
        assert float(smooth_minimum(a, b, 0.05)) <= min(a, b)
        assert float(smooth_maximum(a, b, 0.05)) >= max(a, b)


def test_a_knee_declares_itself_for_the_graph() -> None:
    attributes = Knee("crack", 3.0e5, 1.0e4, units="Pa").graph_attributes()

    assert attributes["knee_location"] == 3.0e5
    assert attributes["knee_sharpness"] == 1.0e4
    assert attributes["knee_hardness"] == pytest.approx(30.0)
    assert attributes["knee_units"] == "Pa"
