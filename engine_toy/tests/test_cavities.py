"""A cavity reduced to a circuit, checked against what it must already be.

The chain here is long -- analytic mode, surface-integral Q, equivalent
resonator, mutual coupling, Kron reduction -- so every link is pinned by
something computed a different way:

  * the Q closed form against Pozar's published grouping, AND against a
    direct numerical integration of the very field the module exposes;
  * the equivalent inductance against the same integral;
  * the REDUCED port impedance against the textbook transformer-coupled
    resonator, ``Re Z = w0 M^2 Q / L`` and ``Im Z = w0 L_loop``, which
    the reduction knows nothing about;
  * the lineshape against a Lorentzian of the declared Q.

If the reduction ever stops being exact, the third of those is what
notices, because it reaches the answer without passing through the
matrix at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cavities import CavityMode, LoopPort, RectangularCavity
from em_materials import EM_MATERIALS, VACUUM_PERMEABILITY_H_M
from waveguides import RectangularGuide


#: A WR-90 cross-section closed at 20 mm: the standard worked example,
#: and X-band, so the numbers are ones a datasheet would recognise.
def _cavity() -> RectangularCavity:
    return RectangularCavity("box", width_m=0.02286, height_m=0.01016,
                             depth_m=0.02)


def _drive_loop(identity: str = "drive", z: float = 0.001) -> LoopPort:
    """A loop near the z wall, normal along x, where H_x is strongest."""
    return LoopPort(identity, (0.01143, 0.00508, z), (1.0, 0.0, 0.0),
                    area_m2=1.0e-5, self_inductance_h=1.0e-8)


# --------------------------------------------------------------------------
# the spectrum
# --------------------------------------------------------------------------

def test_te101_is_dominant_and_sits_where_the_geometry_says() -> None:
    cavity = _cavity()
    mode = cavity.dominant_mode

    assert mode.label == "TE101"
    assert mode.frequency_hz == pytest.approx(9.9583e9, rel=1e-4)
    assert mode.frequency_hz == pytest.approx(
        0.5 * 299792458.0 * math.hypot(1 / 0.02286, 1 / 0.02))


def test_a_cavity_is_a_guide_with_one_more_index() -> None:
    """Stretch the box and its resonance falls to the guide's cutoff.

    The claim in the module docstring, made checkable: a cavity is a
    shorted length of guide, so as the short recedes the standing wave
    relaxes onto the travelling mode's cutoff.
    """
    guide = RectangularGuide("WR-90", width_m=0.02286, height_m=0.01016)
    stretched = RectangularCavity("long", width_m=0.02286, height_m=0.01016,
                                  depth_m=5.0)

    assert stretched.dominant_mode.frequency_hz == pytest.approx(
        guide.dominant_mode.cutoff_hz, rel=1e-4)


def test_the_index_rules_are_structural() -> None:
    """A TE mode needs a half-wave along z; a TM mode needs one on both
    transverse axes.  Neither rule is incidental, so both are enforced."""
    cavity = _cavity()

    assert cavity.mode_exists("TE", 1, 0, 1)
    assert not cavity.mode_exists("TE", 1, 0, 0)     # no half-wave along z
    assert not cavity.mode_exists("TE", 0, 0, 1)     # uniform across both
    assert cavity.mode_exists("TM", 1, 1, 0)         # p = 0 is allowed here
    assert not cavity.mode_exists("TM", 1, 0, 1)     # needs both transverse

    with pytest.raises(ValueError, match="does not exist"):
        cavity.resonance_hz("TE", 1, 0, 0)
    with pytest.raises(ValueError, match="no mode family"):
        cavity.mode_exists("TEM", 1, 0, 1)


def test_the_mode_list_is_ordered_and_contains_only_real_modes() -> None:
    cavity = _cavity()
    modes = cavity.modes()
    frequencies = [mode.frequency_hz for mode in modes]

    assert frequencies == sorted(frequencies)
    assert modes[0].label == "TE101"
    assert all(cavity.mode_exists(mode.family, mode.m, mode.n, mode.p)
               for mode in modes)


# --------------------------------------------------------------------------
# Q, by three routes that must agree
# --------------------------------------------------------------------------

def _pozar_q(cavity: RectangularCavity, mode: CavityMode,
             surface_ohm: float) -> float:
    """The published grouping, written out independently of the module."""
    a, b, d = cavity.width_m, cavity.height_m, cavity.depth_m
    k = cavity.wavenumber_1_m(mode)
    eta = VACUUM_PERMEABILITY_H_M * cavity.phase_speed_m_s
    return ((k * a * d) ** 3 * b * eta
            / (2.0 * math.pi ** 2 * surface_ohm
               * (2.0 * a ** 3 * b + 2.0 * b * d ** 3
                  + a ** 3 * d + a * d ** 3)))


def test_the_q_closed_form_is_the_published_one() -> None:
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    mode = cavity.dominant_mode
    surface = copper.surface_resistance_ohm(mode.frequency_hz)

    assert cavity.unloaded_q(mode, copper) == pytest.approx(
        _pozar_q(cavity, mode, surface), rel=1e-12)
    assert cavity.unloaded_q(mode, copper) == pytest.approx(7931.0, rel=1e-3)


def test_q_matches_a_direct_integration_of_the_exposed_field() -> None:
    """The independent route: integrate the mode rather than trust the
    grouping.  ``Q = w mu |h|^2 dV / (Rs |h_tan|^2 dS)`` uses nothing but
    the shape function the module hands out."""
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    mode = cavity.dominant_mode
    a, b, d = cavity.width_m, cavity.height_m, cavity.depth_m
    steps = 80
    xs = (np.arange(steps) + 0.5) * a / steps
    zs = (np.arange(steps) + 0.5) * d / steps

    stored = 0.0
    broad = 0.0
    for x in xs:
        for z in zs:
            hx, _, hz = cavity.magnetic_shape_at(mode, (x, 0.0, z))
            stored += hx * hx + hz * hz
            broad += 2.0 * (hx * hx + hz * hz)
    stored *= (a / steps) * (d / steps) * b
    broad *= (a / steps) * (d / steps)

    walls = 0.0
    for z in zs:                       # x = 0 and x = a: H_z is tangential
        for edge in (0.0, a):
            _, _, hz = cavity.magnetic_shape_at(mode, (edge, 0.0, z))
            walls += hz * hz * (d / steps) * b
    for x in xs:                       # z = 0 and z = d: H_x is tangential
        for edge in (0.0, d):
            hx, _, _ = cavity.magnetic_shape_at(mode, (x, 0.0, edge))
            walls += hx * hx * (a / steps) * b

    omega = 2.0 * math.pi * mode.frequency_hz
    surface = copper.surface_resistance_ohm(mode.frequency_hz)
    integrated = (omega * VACUUM_PERMEABILITY_H_M * stored
                  / (surface * (walls + broad)))

    assert integrated == pytest.approx(cavity.unloaded_q(mode, copper),
                                       rel=1e-6)
    assert VACUUM_PERMEABILITY_H_M * stored == pytest.approx(
        cavity.mode_inductance_h(mode), rel=1e-6)


def test_q_goes_inversely_with_surface_resistance() -> None:
    """Which is the whole reason cavities are silver-plated."""
    cavity = _cavity()
    mode = cavity.dominant_mode
    copper = EM_MATERIALS["copper"]
    steel = EM_MATERIALS["stainless-steel"]

    assert cavity.unloaded_q(mode, copper) > cavity.unloaded_q(mode, steel)
    assert (cavity.unloaded_q(mode, copper)
            * copper.surface_resistance_ohm(mode.frequency_hz)
            == pytest.approx(
                cavity.unloaded_q(mode, steel)
                * steel.surface_resistance_ohm(mode.frequency_hz)))


def test_the_estimate_is_labelled_and_lands_within_a_small_factor() -> None:
    """``2V/(delta S)`` drops a geometry factor of order one, and says so."""
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    mode = cavity.dominant_mode

    ratio = (cavity.approximate_unloaded_q(mode, copper)
             / cavity.unloaded_q(mode, copper))
    assert 0.5 < ratio < 2.0


def test_the_exact_q_is_refused_for_modes_it_was_not_derived_for() -> None:
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    other = CavityMode("TM", 1, 1, 0, cavity.resonance_hz("TM", 1, 1, 0))

    with pytest.raises(ValueError, match="approximate_unloaded_q"):
        cavity.unloaded_q(other, copper)
    with pytest.raises(ValueError, match="only TE101"):
        cavity.magnetic_shape_at(other, (0.01, 0.005, 0.01))


# --------------------------------------------------------------------------
# what a loop grips
# --------------------------------------------------------------------------

def test_a_loop_normal_to_a_field_the_mode_lacks_couples_to_nothing() -> None:
    """TE101 is uniform along y and has no H_y, so a y-facing loop is deaf.

    Zero here is a statement about the mode, not a small number: the
    field component simply does not exist.
    """
    cavity = _cavity()
    mode = cavity.dominant_mode
    deaf = LoopPort("deaf", (0.01143, 0.00508, 0.001), (0.0, 1.0, 0.0),
                    area_m2=1.0e-5, self_inductance_h=1.0e-8)

    assert cavity.coupling_coefficient(mode, deaf) == 0.0


def test_a_loop_at_a_field_null_couples_to_nothing() -> None:
    """H_x goes as sin(pi x / a), so a loop on the x = 0 wall sees none."""
    cavity = _cavity()
    mode = cavity.dominant_mode
    at_null = LoopPort("null", (0.0, 0.00508, 0.001), (1.0, 0.0, 0.0),
                       area_m2=1.0e-5, self_inductance_h=1.0e-8)

    assert cavity.coupling_coefficient(mode, at_null) == pytest.approx(
        0.0, abs=1e-15)


def test_coupling_is_proportional_to_loop_area() -> None:
    cavity = _cavity()
    mode = cavity.dominant_mode
    small = _drive_loop()
    big = LoopPort("big", (0.01143, 0.00508, 0.001), (1.0, 0.0, 0.0),
                   area_m2=2.0e-5, self_inductance_h=1.0e-8)

    assert cavity.mutual_inductance_h(mode, big) == pytest.approx(
        2.0 * cavity.mutual_inductance_h(mode, small))


def test_a_declared_loop_must_be_physical() -> None:
    with pytest.raises(ValueError, match="positive area"):
        LoopPort("bad", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0, 1e-8)
    with pytest.raises(ValueError, match="positive self inductance"):
        LoopPort("bad", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1e-5, 0.0)
    with pytest.raises(ValueError, match="no direction"):
        LoopPort("bad", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1e-5, 1e-8)


def test_a_loop_outside_the_box_is_refused() -> None:
    cavity = _cavity()

    with pytest.raises(ValueError, match="outside the cavity"):
        cavity.magnetic_shape_at(cavity.dominant_mode, (0.05, 0.005, 0.01))


# --------------------------------------------------------------------------
# the reduction, against the textbook it never consults
# --------------------------------------------------------------------------

def test_the_reduced_impedance_is_the_coupled_resonator() -> None:
    """The keystone test.

    A transformer-coupled series resonator has, at resonance,
    ``Z = j w0 L_loop + w0 M^2 Q / L_mode``.  The right-hand side is
    written here from the coupling and the Q alone.  The left-hand side
    comes out of assembling a circuit and Kron-reducing it, which
    touches none of that algebra.  They agree exactly or something in
    between is wrong.
    """
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    mode = cavity.dominant_mode
    port = _drive_loop()
    frequency = mode.frequency_hz

    impedance = cavity.port_network(
        frequency, copper, (port,)).two_terminal_impedance_ohm()

    omega = 2.0 * math.pi * frequency
    mutual = cavity.mutual_inductance_h(mode, port)
    expected_real = (omega * mutual ** 2 * cavity.unloaded_q(mode, copper)
                     / cavity.mode_inductance_h(mode))

    assert impedance.real == pytest.approx(expected_real, rel=1e-9)
    assert impedance.imag == pytest.approx(omega * port.self_inductance_h,
                                           rel=1e-9)


def test_the_resonance_lands_on_the_mode_frequency() -> None:
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    mode = cavity.dominant_mode
    quality = cavity.unloaded_q(mode, copper)
    port = _drive_loop()

    span = np.linspace(mode.frequency_hz * (1.0 - 2.0 / quality),
                       mode.frequency_hz * (1.0 + 2.0 / quality), 401)
    response = [cavity.port_network(
        float(f), copper, (port,)).two_terminal_impedance_ohm().real
        for f in span]

    assert span[int(np.argmax(response))] == pytest.approx(
        mode.frequency_hz, rel=1e-5)


def test_the_lineshape_carries_the_declared_q() -> None:
    """Half power at half a linewidth either side, which is what Q means.

    Read off the reduced network rather than the resonator it was built
    from, so the reduction has to preserve the width as well as the
    centre.
    """
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    mode = cavity.dominant_mode
    quality = cavity.unloaded_q(mode, copper)
    port = _drive_loop()
    centre = mode.frequency_hz

    def real_part(frequency: float) -> float:
        return cavity.port_network(
            frequency, copper, (port,)).two_terminal_impedance_ohm().real

    peak = real_part(centre)
    below = real_part(centre * (1.0 - 1.0 / (2.0 * quality)))
    above = real_part(centre * (1.0 + 1.0 / (2.0 * quality)))

    assert below / peak == pytest.approx(0.5, rel=2e-3)
    assert above / peak == pytest.approx(0.5, rel=2e-3)


def test_the_bake_is_an_ordinary_port_network() -> None:
    """Whatever else it is, what comes out is what a circuit consumes."""
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    network = cavity.port_network(cavity.dominant_mode.frequency_hz, copper,
                                  (_drive_loop(),))

    assert network.ports == ("drive", "box/return")
    assert network.eliminated_nodes == 2          # the mode's two interior nodes
    assert np.allclose(network.admittance, network.admittance.T)
    stamp = network.stamp(("magnetron-anode", "chassis"))
    assert stamp.nodes == ("magnetron-anode", "chassis")


def test_two_ports_on_one_mode_reduce_to_a_three_terminal_block() -> None:
    """The case that forced the coupled group to invert as one matrix.

    Both loops share flux with the same mode branch, so the three sit in
    a single coupled group.  A passive three-terminal block has a
    symmetric admittance whose rows sum to zero -- current in equals
    current out, whatever the interior did.
    """
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    network = cavity.port_network(
        cavity.dominant_mode.frequency_hz, copper,
        (_drive_loop("drive", 0.001), _drive_loop("pickup", 0.019)))

    assert network.ports == ("drive", "pickup", "box/return")
    assert np.allclose(network.admittance, network.admittance.T)
    assert np.allclose(network.admittance.sum(axis=1), 0.0,
                       atol=1e-6 * np.abs(network.admittance).max())


def test_two_loops_cannot_each_claim_most_of_the_flux() -> None:
    """``k1^2 + k2^2 < 1`` -- the condition a pairwise check cannot see.

    Each coupling is inside [-1, 1] on its own, so the per-pair rule
    passes both.  Together they would let the group store negative
    energy, and only the whole inductance matrix knows that.
    """
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    mode = cavity.dominant_mode
    # Tiny self-inductance makes each loop claim a large share of the mode.
    greedy = [LoopPort(name, (0.01143, 0.00508, 0.001), (1.0, 0.0, 0.0),
                       area_m2=1.0e-5, self_inductance_h=1.0e-10)
              for name in ("one", "two")]
    couplings = [abs(cavity.coupling_coefficient(mode, port))
                 for port in greedy]

    assert all(value < 1.0 for value in couplings)
    assert sum(value ** 2 for value in couplings) > 1.0

    with pytest.raises(ValueError, match="negative energy"):
        cavity.port_network(mode.frequency_hz, copper, tuple(greedy))


def test_the_expansion_stops_at_the_one_mode_whose_field_is_written() -> None:
    """The honest limit of this module, asserted rather than implied.

    The machinery is general -- the circuit, the coupling and the group
    inversion all take any number of modes -- but only TE101's field
    pattern has been written out, so any other mode is refused at the
    point where its coupling would have to be computed.  Carrying more
    modes is a matter of adding their patterns, not of changing this.
    """
    cavity, copper = _cavity(), EM_MATERIALS["copper"]
    modes = cavity.modes()
    port = _drive_loop()
    second = next(mode for mode in modes if not mode.is_te101)

    assert cavity.port_network(modes[0].frequency_hz, copper, (port,),
                               (modes[0],)) is not None
    with pytest.raises(ValueError, match="only TE101|no equivalent inductance"):
        cavity.port_network(modes[0].frequency_hz, copper, (port,),
                            (modes[0], second))


def test_a_cavity_needs_a_port_and_a_mode() -> None:
    cavity, copper = _cavity(), EM_MATERIALS["copper"]

    with pytest.raises(ValueError, match="needs a port"):
        cavity.circuit(copper, ())
    with pytest.raises(ValueError, match="no modes to carry"):
        cavity.circuit(copper, (_drive_loop(),), ())


def test_a_cavity_needs_positive_sides() -> None:
    with pytest.raises(ValueError, match="positive sides"):
        RectangularCavity("bad", width_m=0.02, height_m=0.0, depth_m=0.02)


def test_a_cavity_declares_itself_to_the_graph() -> None:
    cavity = _cavity()
    declared = cavity.graph_attributes()

    assert declared["part_role"] == "rectangular-cavity"
    assert declared["dominant_mode"] == "TE101"
    assert declared["dominant_resonance_hz"] == pytest.approx(9.9583e9,
                                                              rel=1e-4)
    assert declared["volume_m3"] == pytest.approx(0.02286 * 0.01016 * 0.02)
