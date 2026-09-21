"""Guides checked against dimensions the trade publishes, not against us.

The whole claim of this module is that a penetration's shielding is a
consequence of geometry rather than a fitted number, so the tests have
to come from outside it.  Three independent sources are used: the
standard WR series, whose cutoffs are printed on every datasheet; the
Bessel roots that set a round pipe's modes; and the shielding rule of
thumb that a round hole costs 32 dB per diameter below cutoff, which
this module had better reproduce without being told.

The last one is the interesting test.  ``(20/ln 10) * 1.8412 * 2`` is
31.98, and if the module returns that from the TE11 mode alone then the
rule of thumb was never an empirical fit -- it was the evanescent decay
of the dominant mode over two radii, and we have just derived it.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from waveguides import (
    BESSEL_TE_ROOTS,
    BESSEL_TM_ROOTS,
    FREE_SPACE_IMPEDANCE_OHM,
    NEPERS_TO_DB,
    SPEED_OF_LIGHT,
    CircularAperture,
    CircularGuide,
    GuideMode,
    RectangularGuide,
)


# --------------------------------------------------------------------------
# against the published WR series
# --------------------------------------------------------------------------

#: Standard rectangular guides: inside width, inside height, and the
#: TE10 cutoff the datasheet prints.  Chosen to span three decades so a
#: scaling mistake cannot hide in the middle of the range.
WR_SERIES = [
    ("WR-975", 0.24765, 0.12383, 0.605e9),
    ("WR-284", 0.07214, 0.03404, 2.078e9),
    ("WR-90", 0.02286, 0.01016, 6.557e9),
    ("WR-28", 0.00711, 0.00356, 21.08e9),
]


@pytest.mark.parametrize("name,width,height,published", WR_SERIES)
def test_te10_cutoff_matches_the_datasheet(
        name: str, width: float, height: float, published: float) -> None:
    guide = RectangularGuide(name, width_m=width, height_m=height)

    assert guide.dominant_mode.label == "TE10"
    assert guide.dominant_mode.cutoff_hz == pytest.approx(published, rel=1e-3)


def test_wr90_higher_modes_order_as_published() -> None:
    """TE10, TE20, TE01, TE11 -- and TE20 is exactly twice TE10."""
    guide = RectangularGuide("WR-90", width_m=0.02286, height_m=0.01016)

    assert guide.cutoff_hz("TE", 2, 0) == pytest.approx(13.114e9, rel=1e-3)
    assert guide.cutoff_hz("TE", 0, 1) == pytest.approx(14.754e9, rel=1e-3)
    assert guide.cutoff_hz("TE", 1, 1) == pytest.approx(16.145e9, rel=1e-3)
    assert guide.cutoff_hz("TE", 2, 0) == pytest.approx(
        2.0 * guide.cutoff_hz("TE", 1, 0), rel=1e-12)


def test_the_recommended_band_is_single_mode() -> None:
    """WR-90 is sold for 8.2-12.4 GHz: TE10 alone gets through there.

    This is why the band is what it is, and the module reproduces the
    reasoning rather than the interval.
    """
    guide = RectangularGuide("WR-90", width_m=0.02286, height_m=0.01016)
    carried = [mode for mode in guide.modes() if mode.propagates_at(10.0e9)]

    assert [mode.label for mode in carried] == ["TE10"]
    assert not guide.dominant_mode.propagates_at(0.5 * 8.2e9)
    assert guide.dominant_mode.propagates_at(12.4e9)


# --------------------------------------------------------------------------
# the round pipe, which is what a drilled penetration is
# --------------------------------------------------------------------------

def test_te11_is_dominant_in_a_round_guide() -> None:
    """1.8412 is the smallest root in either table, so TE11 turns on first."""
    guide = CircularGuide("vent", radius_m=0.0127)
    ordered = guide.modes()

    assert ordered[0].label == "TE11"
    assert ordered[0].cutoff_hz == guide.dominant_mode.cutoff_hz
    assert min(BESSEL_TE_ROOTS.values()) < min(BESSEL_TM_ROOTS.values())


def test_a_one_inch_pipe_cuts_off_where_the_root_says() -> None:
    """fc = 1.8412 c / (2 pi a): 6.92 GHz for a 25.4 mm bore."""
    guide = CircularGuide("one-inch", radius_m=0.0127)

    assert guide.dominant_mode.cutoff_hz == pytest.approx(6.92e9, rel=2e-3)
    assert guide.dominant_mode.cutoff_hz == pytest.approx(
        1.8411838 * SPEED_OF_LIGHT / (2.0 * math.pi * 0.0127))


def test_te01_and_tm11_are_degenerate() -> None:
    """Both sit on 3.8317; the degeneracy is real, not a typo in the table."""
    guide = CircularGuide("pipe", radius_m=0.01)

    assert guide.cutoff_hz("TE", 0, 1) == pytest.approx(
        guide.cutoff_hz("TM", 1, 1), rel=1e-12)


def test_cutoff_scales_inversely_with_bore() -> None:
    """Halve the hole, double the cutoff -- the reason small holes shield."""
    wide = CircularGuide("wide", radius_m=0.02)
    narrow = CircularGuide("narrow", radius_m=0.01)

    assert narrow.dominant_mode.cutoff_hz == pytest.approx(
        2.0 * wide.dominant_mode.cutoff_hz, rel=1e-12)


def test_a_dielectric_fill_lowers_every_cutoff_by_the_root_of_epsilon() -> None:
    empty = CircularGuide("dry", radius_m=0.01)
    filled = CircularGuide("potted", radius_m=0.01, relative_permittivity=4.0)

    assert filled.dominant_mode.cutoff_hz == pytest.approx(
        0.5 * empty.dominant_mode.cutoff_hz, rel=1e-12)


# --------------------------------------------------------------------------
# the rule of thumb, derived
# --------------------------------------------------------------------------

def test_deep_below_cutoff_gives_32_db_per_diameter() -> None:
    """The trade's number, out of the mode and nothing else."""
    guide = CircularGuide("penetration", radius_m=0.005)
    far_below = guide.dominant_mode.cutoff_hz / 1000.0

    assert guide.attenuation_db_per_diameter(far_below) == pytest.approx(
        NEPERS_TO_DB * 1.8411838 * 2.0, rel=1e-5)
    assert guide.attenuation_db_per_diameter(far_below) == pytest.approx(
        31.98, rel=1e-3)


def test_the_rule_weakens_as_cutoff_is_approached() -> None:
    """At half the cutoff the decay has already lost 13 percent.

    The rule of thumb is an asymptote, and treating it as a constant
    over-predicts shielding exactly where a designer most needs the
    truth -- near the top of the band a hole is expected to block.
    """
    guide = CircularGuide("penetration", radius_m=0.005)
    cutoff = guide.dominant_mode.cutoff_hz

    deep = guide.attenuation_db_per_diameter(cutoff / 1000.0)
    near = guide.attenuation_db_per_diameter(cutoff / 2.0)

    assert near == pytest.approx(deep * math.sqrt(1.0 - 0.25), rel=1e-4)
    assert near < deep


def test_attenuation_is_linear_in_wall_thickness() -> None:
    """Which is why a thick wall shields and a foil one does not."""
    guide = CircularGuide("bore", radius_m=0.002)
    frequency = 2.45e9

    thin = guide.attenuation_db(frequency, 0.003)
    thick = guide.attenuation_db(frequency, 0.012)

    assert thick == pytest.approx(4.0 * thin, rel=1e-12)
    assert guide.attenuation_db(frequency, 0.0) == 0.0


def test_a_mesh_hole_at_2_45_ghz_is_far_below_cutoff() -> None:
    """Why a Faraday mesh works, stated as the number that makes it work.

    A millimetre bore cuts off near 88 GHz, so an oven's 2.45 GHz sits
    some thirty-six times below it and the decay is at its asymptote.
    This is also why voxelising the hole would be hopeless: the feature
    is three orders below the wavelength.
    """
    hole = CircularGuide("mesh-hole", radius_m=0.001, length_m=0.001)

    assert hole.dominant_mode.cutoff_hz == pytest.approx(87.9e9, rel=5e-3)
    assert hole.dominant_mode.cutoff_hz / 2.45e9 == pytest.approx(35.9, rel=0.02)
    assert hole.attenuation_db(2.45e9) == pytest.approx(16.0, rel=0.02)


# --------------------------------------------------------------------------
# above cutoff the guide is a pipe, not a shield
# --------------------------------------------------------------------------

def test_a_propagating_mode_loses_nothing_to_cutoff() -> None:
    """Zero is the correct answer here, not a small number.

    Wall loss is real but is a separate and much smaller effect; this
    figure accounts only for being below cutoff, and saying so keeps
    the two from being silently summed.
    """
    guide = CircularGuide("duct", radius_m=0.05, length_m=0.2)
    frequency = 10.0 * guide.dominant_mode.cutoff_hz

    assert guide.dominant_mode.propagates_at(frequency)
    assert guide.attenuation_db(frequency) == 0.0
    assert guide.dominant_mode.wavenumber_1_m(frequency).imag == 0.0


def test_the_wavenumber_switches_branch_at_cutoff_without_a_gap() -> None:
    """Real above, imaginary below, and zero exactly on it."""
    mode = GuideMode("TE", 1, 1, cutoff_hz=1.0e9)

    assert mode.wavenumber_1_m(1.0e9) == complex(0.0, 0.0)
    assert mode.wavenumber_1_m(1.001e9).real > 0.0
    assert mode.wavenumber_1_m(1.001e9).imag == 0.0
    assert mode.wavenumber_1_m(0.999e9).imag > 0.0
    assert mode.wavenumber_1_m(0.999e9).real == 0.0


def test_the_wavenumber_is_the_free_space_one_far_above_cutoff() -> None:
    mode = GuideMode("TE", 1, 0, cutoff_hz=1.0e9)
    frequency = 1.0e12

    assert mode.wavenumber_1_m(frequency).real == pytest.approx(
        2.0 * math.pi * frequency / SPEED_OF_LIGHT, rel=1e-6)


def test_a_penetration_crosses_over_as_the_bore_opens() -> None:
    """One sweep showing the hole become a pipe.

    Nothing switches mode in the code; the same expression covers both
    ends, which is the point of carrying the wavenumber as a complex
    number.
    """
    thickness = 0.003
    shielding = [
        CircularGuide(f"r{radius}", radius_m=radius).attenuation_db(
            2.45e9, thickness)
        for radius in (0.0005, 0.002, 0.008, 0.02, 0.05)
    ]

    # The tightest bore is 0.5 mm in a 3 mm wall: three diameters deep,
    # and far enough below cutoff to be sitting on the asymptote, so it
    # has to come out at three times the rule of thumb.
    assert shielding[0] == pytest.approx(3.0 * 31.98, rel=1e-3)
    assert shielding[-1] == 0.0
    assert all(a >= b for a, b in zip(shielding, shielding[1:]))


# --------------------------------------------------------------------------
# what the graph is told
# --------------------------------------------------------------------------

def test_a_guide_declares_itself_to_the_graph() -> None:
    """Parts are identified by declared attributes, never by name."""
    guide = CircularGuide("cable-gland", radius_m=0.004, length_m=0.006)
    declared = guide.graph_attributes()

    assert declared["part_role"] == "circular-waveguide"
    assert declared["te11_cutoff_hz"] == pytest.approx(
        guide.dominant_mode.cutoff_hz)
    assert declared["length_m"] == pytest.approx(0.006)

    rect = RectangularGuide("feed", width_m=0.02286, height_m=0.01016)
    assert rect.graph_attributes()["part_role"] == "rectangular-waveguide"
    assert rect.graph_attributes()["te10_cutoff_hz"] == pytest.approx(
        6.557e9, rel=1e-3)


def test_a_mode_declares_its_label_and_cutoff() -> None:
    mode = GuideMode("TM", 0, 1, cutoff_hz=3.0e9)

    assert mode.label == "TM01"
    assert mode.graph_attributes() == {"mode": "TM01", "cutoff_hz": 3.0e9}


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_a_guide_needs_real_dimensions() -> None:
    with pytest.raises(ValueError, match="positive radius"):
        CircularGuide("bad", radius_m=0.0)
    with pytest.raises(ValueError, match="length cannot be negative"):
        CircularGuide("bad", radius_m=0.01, length_m=-1.0)
    with pytest.raises(ValueError, match="positive sides"):
        RectangularGuide("bad", width_m=0.02, height_m=0.0)


def test_an_untabulated_root_is_refused_rather_than_guessed() -> None:
    guide = CircularGuide("pipe", radius_m=0.01)

    with pytest.raises(ValueError, match="no tabulated root"):
        guide.cutoff_hz("TE", 7, 3)


def test_a_tm_mode_needs_both_indices() -> None:
    """TM10 and TM01 do not exist in a rectangular guide.

    Saying so is the difference between a model and a formula: the
    expression would happily return a number for either.
    """
    guide = RectangularGuide("feed", width_m=0.02, height_m=0.01)

    with pytest.raises(ValueError, match="both indices at least one"):
        guide.cutoff_hz("TM", 1, 0)
    with pytest.raises(ValueError, match="both indices at least one"):
        guide.cutoff_hz("TM", 0, 1)
    with pytest.raises(ValueError, match="no mode"):
        guide.cutoff_hz("TE", 0, 0)


def test_the_rectangular_mode_list_omits_the_modes_that_do_not_exist() -> None:
    guide = RectangularGuide("feed", width_m=0.02, height_m=0.01)
    modes = guide.modes()
    labels = [mode.label for mode in modes]

    assert "TE00" not in labels
    assert "TM10" not in labels
    assert "TM01" not in labels
    assert "TE10" in labels and "TM11" in labels

    cutoffs = [mode.cutoff_hz for mode in modes]
    assert cutoffs == sorted(cutoffs)


# --------------------------------------------------------------------------
# the medium, which must enter k and k_c together or not at all
# --------------------------------------------------------------------------

def test_a_fill_survives_into_the_propagation_constant() -> None:
    """Regression: the cutoff hid this, because at cutoff it cancels.

    ``k`` taken in vacuum while ``k_c`` carries the fill still gives
    beta = 0 in exactly the right place, so a suite that checks only the
    cutoff frequency passes while every propagating result is wrong by
    the root of epsilon.  This checks the slope, not the zero.
    """
    guide = CircularGuide("potted", radius_m=0.01, relative_permittivity=4.0)
    frequency = 1.0e12                       # far above any cutoff here
    beta = guide.dominant_mode.wavenumber_1_m(frequency).real

    assert beta == pytest.approx(
        2.0 * math.pi * frequency * 2.0 / SPEED_OF_LIGHT, rel=1e-4)


def test_a_fill_slows_the_guide_by_the_root_of_epsilon() -> None:
    empty = CircularGuide("dry", radius_m=0.01)
    filled = CircularGuide("potted", radius_m=0.01, relative_permittivity=4.0)
    frequency = 1.0e12

    assert (filled.dominant_mode.wavenumber_1_m(frequency).real
            == pytest.approx(
                2.0 * empty.dominant_mode.wavenumber_1_m(frequency).real,
                rel=1e-4))


def test_a_mode_recovers_its_permittivity_from_its_speed() -> None:
    guide = CircularGuide("potted", radius_m=0.01, relative_permittivity=9.0)

    assert guide.dominant_mode.relative_permittivity == pytest.approx(9.0)


# --------------------------------------------------------------------------
# wave impedance: where the energy goes, not just how much gets through
# --------------------------------------------------------------------------

def test_te_impedance_tends_to_free_space_far_above_cutoff() -> None:
    guide = CircularGuide("duct", radius_m=0.01)
    impedance = guide.dominant_mode.wave_impedance_ohm(
        1000.0 * guide.dominant_mode.cutoff_hz)

    assert impedance.real == pytest.approx(FREE_SPACE_IMPEDANCE_OHM, rel=1e-5)
    assert impedance.imag == pytest.approx(0.0, abs=1e-9)


def test_te_is_above_free_space_and_tm_below_it() -> None:
    """The standard pair of limits, and they bracket eta0 from both sides."""
    guide = CircularGuide("duct", radius_m=0.01)
    te = guide.dominant_mode
    tm = next(mode for mode in guide.modes() if mode.family == "TM")
    frequency = 3.0 * tm.cutoff_hz

    assert te.wave_impedance_ohm(frequency).real > FREE_SPACE_IMPEDANCE_OHM
    assert 0.0 < tm.wave_impedance_ohm(frequency).real < FREE_SPACE_IMPEDANCE_OHM


def test_below_cutoff_the_impedance_is_purely_reactive() -> None:
    """The claim that sub-cutoff shielding is reflective, made checkable.

    A real part of exactly zero is the statement that an evanescent mode
    dissipates nothing.  If this ever grows a real part, the model has
    started absorbing energy that physics says is reflected.
    """
    guide = CircularGuide("penetration", radius_m=0.001)
    te = guide.dominant_mode
    tm = next(mode for mode in guide.modes() if mode.family == "TM")

    te_z = te.wave_impedance_ohm(te.cutoff_hz / 100.0)
    tm_z = tm.wave_impedance_ohm(tm.cutoff_hz / 100.0)

    assert te_z.real == 0.0 and te_z.imag > 0.0       # inductive
    assert tm_z.real == 0.0 and tm_z.imag < 0.0       # capacitive


def test_the_impedance_is_refused_exactly_at_cutoff() -> None:
    mode = GuideMode("TE", 1, 1, cutoff_hz=1.0e9)

    with pytest.raises(ValueError, match="singular exactly at cutoff"):
        mode.wave_impedance_ohm(1.0e9)


# --------------------------------------------------------------------------
# the two-port: the same object a Kron reduction produces
# --------------------------------------------------------------------------

def _scattering(network, reference_ohm: complex) -> np.ndarray:
    """S from Y at a stated reference impedance, by the textbook form."""
    normalised = reference_ohm * network.admittance
    identity = np.eye(len(network.ports))
    return (identity - normalised) @ np.linalg.inv(identity + normalised)


def test_the_two_port_is_a_transmission_line_and_says_so() -> None:
    """S21 out of the admittance matrix must be exp(-gamma l), exactly.

    This is the real check on the Y-matrix: the admittances were built
    from coth and csch, and the scattering parameter is recovered by a
    matrix inversion that knows nothing about where they came from.  Two
    independent routes to one number.
    """
    guide = CircularGuide("bore", radius_m=0.002, length_m=0.006)
    frequency = 2.45e9
    network = guide.port_network(frequency)
    mode = guide.dominant_mode

    scattering = _scattering(network, mode.wave_impedance_ohm(frequency))
    expected = cmath.exp(-mode.propagation_constant_1_m(frequency) * 0.006)

    assert scattering[1, 0] == pytest.approx(expected, rel=1e-9)
    assert scattering[0, 1] == pytest.approx(expected, rel=1e-9)


def test_the_two_port_agrees_with_the_decibels() -> None:
    """The circuit object and the scalar must not drift apart."""
    guide = CircularGuide("bore", radius_m=0.002, length_m=0.006)
    frequency = 2.45e9
    network = guide.port_network(frequency)

    scattering = _scattering(
        network, guide.dominant_mode.wave_impedance_ohm(frequency))
    from_matrix = -20.0 * math.log10(abs(scattering[1, 0]))

    assert from_matrix == pytest.approx(guide.attenuation_db(frequency),
                                        rel=1e-9)


def test_the_two_port_is_reciprocal_and_symmetric() -> None:
    """A passive guide has no preferred direction; Y must be symmetric."""
    guide = CircularGuide("bore", radius_m=0.002, length_m=0.006)
    network = guide.port_network(2.45e9)

    assert np.allclose(network.admittance, network.admittance.T)
    assert network.eliminated_nodes == 0
    assert len(network.ports) == 2


def test_the_two_port_below_cutoff_is_purely_reactive() -> None:
    """No resistance anywhere in the matrix: nothing is being dissipated."""
    guide = CircularGuide("penetration", radius_m=0.001, length_m=0.01)
    network = guide.port_network(2.45e9)

    assert np.allclose(network.admittance.real, 0.0)


def test_the_two_port_stamps_into_an_enclosing_circuit() -> None:
    """The handoff itself: a penetration becomes an ordinary circuit block."""
    guide = CircularGuide("gland", radius_m=0.002, length_m=0.006)
    network = guide.port_network(2.45e9)
    stamp = network.stamp(("chamber-interior", "chassis"))

    assert stamp.nodes == ("chamber-interior", "chassis")
    assert np.allclose(stamp.admittance, network.admittance)


def test_a_zero_length_line_is_refused_and_names_the_alternative() -> None:
    """The failure mode that would otherwise be silent.

    A line of no length attenuates nothing, so a naive model would call
    a thin-screen hole perfectly transparent.  It is not; it is an
    aperture, and the refusal says where to go.
    """
    guide = CircularGuide("hole", radius_m=0.001)

    with pytest.raises(ValueError, match="aperture polarizability"):
        guide.port_network(2.45e9, 0.0)


def test_the_two_port_is_refused_exactly_at_cutoff() -> None:
    guide = CircularGuide("bore", radius_m=0.01, length_m=0.01)

    with pytest.raises(ValueError, match="singular exactly at cutoff"):
        guide.port_network(guide.dominant_mode.cutoff_hz)


def test_a_rectangular_run_also_reduces_to_a_two_port() -> None:
    guide = RectangularGuide("feed", width_m=0.02286, height_m=0.01016,
                             length_m=0.05)
    network = guide.port_network(10.0e9)

    assert network.ports == ("feed/near", "feed/far")
    assert np.allclose(network.admittance, network.admittance.T)


def test_single_mode_validity_is_a_predicate_not_a_docstring() -> None:
    """A thick wall earns the single-mode model; a foil one does not."""
    thick = CircularGuide("thick", radius_m=0.001, length_m=0.02)
    thin = CircularGuide("thin", radius_m=0.001, length_m=0.0001)

    assert thick.dominant_mode_holds(2.45e9)
    assert not thin.dominant_mode_holds(2.45e9)


# --------------------------------------------------------------------------
# the aperture: what the tunnel cannot see
# --------------------------------------------------------------------------

def test_the_polarizabilities_are_bethes() -> None:
    aperture = CircularAperture("hole", radius_m=0.003)

    assert aperture.magnetic_polarizability_m3 == pytest.approx(
        4.0 * 0.003 ** 3 / 3.0)
    assert aperture.electric_polarizability_m3 == pytest.approx(
        2.0 * 0.003 ** 3 / 3.0)
    assert aperture.magnetic_polarizability_m3 == pytest.approx(
        2.0 * aperture.electric_polarizability_m3)


def test_transmission_goes_as_the_sixth_power_of_the_radius() -> None:
    """The exponent is the physics, and it comes out exact.

    Doubling the hole multiplies the transmitted power by 64.  This is
    the one number in the aperture model that is not an approximation:
    it follows from a dipole going as a^3 and radiated power as its
    square.
    """
    frequency = 2.45e9
    small = CircularAperture("small", radius_m=0.001)
    big = CircularAperture("big", radius_m=0.002)

    assert (big.transmission_cross_section_m2(frequency)
            / small.transmission_cross_section_m2(frequency)
            == pytest.approx(64.0, rel=1e-12))


def test_the_transmitted_fraction_goes_as_the_fourth_power_of_frequency() -> None:
    """Bethe's headline result: the fraction scales as (a/lambda)^4."""
    aperture = CircularAperture("hole", radius_m=0.001)
    low = aperture.transmission_coefficient(1.0e9)
    high = aperture.transmission_coefficient(2.0e9)

    assert high / low == pytest.approx(16.0, rel=1e-12)


def test_transmission_matches_the_closed_form() -> None:
    aperture = CircularAperture("hole", radius_m=0.001)
    frequency = 2.45e9
    k = 2.0 * math.pi * frequency / SPEED_OF_LIGHT

    assert aperture.transmission_coefficient(frequency) == pytest.approx(
        64.0 * (k * 0.001) ** 4 / (27.0 * math.pi ** 3))
    assert aperture.shielding_db(frequency) == pytest.approx(62.75, rel=1e-3)


def test_the_aperture_dominates_a_thin_wall_by_a_wide_margin() -> None:
    """Why the two terms must be composed rather than substituted.

    A millimetre hole in a millimetre wall: the tunnel accounts for
    16 dB, the aperture for 63.  Taking the guide term alone would
    under-predict the shielding by some forty-seven decibels, and would
    do it quietly.
    """
    frequency = 2.45e9
    tunnel = CircularGuide("mesh", radius_m=0.001,
                           length_m=0.001).attenuation_db(frequency)
    aperture = CircularAperture("mesh", radius_m=0.001,
                                thickness_m=0.001).shielding_db(frequency)

    assert tunnel == pytest.approx(16.0, rel=0.02)
    assert aperture == pytest.approx(62.75, rel=1e-3)
    assert aperture - tunnel > 40.0


def test_the_validity_range_is_declared_and_enforced() -> None:
    """Both conditions, each refused on its own."""
    fine = CircularAperture("mesh", radius_m=0.001, thickness_m=0.0005)
    assert fine.is_valid_at(2.45e9)
    assert fine.size_parameter(2.45e9) == pytest.approx(0.0513, rel=1e-2)

    # Small against the wavelength, but the screen is no longer a screen.
    thick = CircularAperture("bore", radius_m=0.001, thickness_m=0.01)
    assert not thick.is_valid_at(2.45e9)

    # Thin, but the hole is now a sizeable fraction of a wavelength.
    wide = CircularAperture("port", radius_m=0.05, thickness_m=0.001)
    assert not wide.is_valid_at(2.45e9)


def test_an_aperture_needs_a_radius() -> None:
    with pytest.raises(ValueError, match="needs a radius"):
        CircularAperture("bad", radius_m=0.0)


def test_an_aperture_declares_itself_to_the_graph() -> None:
    aperture = CircularAperture("vent-hole", radius_m=0.002, thickness_m=0.001)
    declared = aperture.graph_attributes()

    assert declared["part_role"] == "circular-aperture"
    assert declared["magnetic_polarizability_m3"] == pytest.approx(
        4.0 * 0.002 ** 3 / 3.0)
