"""Absorbing bodies, and the states their absorption drives.

The gauge test is the one that matters.  A load filling a cavity that is
itself filled with the same dielectric must show ``Q = 1/tan d`` -- the
textbook dielectric Q.  That number is reached here through a chain that
never mentions it: a mode inductance from a magnetic shape integral, a
field amplitude from a circuit current, an absorbed power from a volume
integral, a resistance from that power.  If any link in the chain had
the wrong normalisation the identity would not close.

A second check pins the phasor convention, which is the kind of thing
that silently costs a factor of two: the absorbed density computed here
must equal :meth:`EMMaterial.dielectric_loss_w_m3`, whose argument is an
RMS field, once the root two is applied.
"""

from __future__ import annotations

import math

import pytest

from applicators import (
    WATER_LATENT_HEAT_SUBLIMATION_J_KG,
    ApplicatorState,
    DielectricLoad,
    SublimationDuty,
    Susceptor,
)
from cavities import RectangularCavity, LoopPort
from chambers import ShieldedChamber
from em_materials import EM_MATERIALS, water_em
from emitters import Magnetron


def _oven() -> RectangularCavity:
    return RectangularCavity("oven", width_m=0.30, height_m=0.20,
                             depth_m=0.25)


def _centre() -> tuple[float, float, float]:
    return (0.15, 0.10, 0.125)


# --------------------------------------------------------------------------
# the gauge: field to circuit and back
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["ptfe", "fr4", "water-ice", "water"])
def test_a_filled_cavity_shows_the_textbook_dielectric_q(key: str) -> None:
    """``Q = 1/tan d``, reached without ever writing that formula down."""
    material = EM_MATERIALS[key]
    cavity = RectangularCavity(
        "filled", 0.30, 0.20, 0.25,
        relative_permittivity=material.relative_permittivity)
    mode = cavity.dominant_mode
    load = DielectricLoad("fill", material, cavity.volume_m3,
                          fills_cavity=True)

    assert load.quality_factor(cavity, mode) == pytest.approx(
        1.0 / material.loss_tangent, rel=1e-8)


def test_a_load_in_an_air_mode_shows_one_over_the_loss_permittivity() -> None:
    """The distinction that catches people, stated rather than tripped over.

    ``1/tan d`` is the Q of a cavity that IS the dielectric.  A body
    dropped into an air-filled cavity sees the air mode, whose
    wavenumber carries no ``eps'``, so the same algebra gives ``1/eps''``
    instead.  Both are right; they are answers to different questions.
    """
    cavity = _oven()
    mode = cavity.dominant_mode
    material = EM_MATERIALS["fr4"]
    load = DielectricLoad("fill", material, cavity.volume_m3,
                          fills_cavity=True)

    loss_permittivity = (material.relative_permittivity
                         * material.loss_tangent)
    assert load.quality_factor(cavity, mode) == pytest.approx(
        1.0 / loss_permittivity, rel=1e-8)


def test_the_absorbed_density_agrees_with_the_material_formula() -> None:
    """Pins the phasor convention against a factor of two.

    ``dielectric_loss_w_m3`` takes an RMS field; everything here carries
    peak phasors.  The two agree only through the root two, and this is
    where that is checked rather than assumed.
    """
    cavity = _oven()
    mode = cavity.dominant_mode
    material = water_em("liquid")
    load = DielectricLoad("cup", material, 2.0e-4, _centre())
    current = 1.0

    peak = load.field_v_m(cavity, mode, current)
    assert load.power_density_w_m3(cavity, mode, current) == pytest.approx(
        material.dielectric_loss_w_m3(mode.frequency_hz,
                                      peak / math.sqrt(2.0)), rel=1e-9)


def test_a_body_at_a_wall_absorbs_nothing() -> None:
    """TE101's electric field vanishes on every side wall.

    Which is why an oven has a turntable, and why a corner of a chamber
    is a place to put something you do not want heated.
    """
    cavity = _oven()
    mode = cavity.dominant_mode
    at_wall = DielectricLoad("corner", water_em("liquid"), 2.0e-4,
                             (0.0, 0.10, 0.125))

    assert at_wall.shape_factor(cavity, mode) == pytest.approx(0.0, abs=1e-15)
    assert at_wall.resistance_ohm(cavity, mode) == pytest.approx(0.0, abs=1e-20)


# --------------------------------------------------------------------------
# selectivity: the point of the whole exercise
# --------------------------------------------------------------------------

def test_ice_absorbs_thousands_of_times_less_than_liquid_water() -> None:
    """At equal field, which is the material statement.

    This is what makes a field selective, and also what makes defrosting
    run away: the product of melting absorbs far harder than the
    feedstock, so the first film to melt takes the power.
    """
    cavity = _oven()
    mode = cavity.dominant_mode
    liquid = DielectricLoad("l", water_em("liquid"), 2.0e-4, _centre())
    ice = DielectricLoad("i", water_em("ice"), 2.0e-4, _centre())

    ratio = (liquid.resistance_ohm(cavity, mode)
             / ice.resistance_ohm(cavity, mode))
    assert ratio == pytest.approx(4367.0, rel=0.01)


def test_vapour_is_effectively_transparent() -> None:
    """A field keeps working on what has not boiled yet."""
    cavity = _oven()
    mode = cavity.dominant_mode
    vapour = DielectricLoad("v", water_em("vapour"), 2.0e-4, _centre())
    liquid = DielectricLoad("l", water_em("liquid"), 2.0e-4, _centre())

    assert (vapour.resistance_ohm(cavity, mode)
            < 1.0e-5 * liquid.resistance_ohm(cavity, mode))


def test_dissolved_ions_absorb_harder_than_pure_water() -> None:
    """A drain's contents are never pure water, and it matters."""
    cavity = _oven()
    mode = cavity.dominant_mode
    brine = DielectricLoad("b", water_em("brine"), 2.0e-4, _centre())
    pure = DielectricLoad("p", water_em("liquid"), 2.0e-4, _centre())

    assert brine.resistance_ohm(cavity, mode) > (
        3.0 * pure.resistance_ohm(cavity, mode))


def test_the_state_reports_selectivity_between_two_bodies() -> None:
    cavity = _oven()
    liquid = DielectricLoad("melt", water_em("liquid"), 1.0e-5, _centre())
    ice = DielectricLoad("block", water_em("ice"), 1.0e-3, _centre())
    state = ApplicatorState(2.45e9, 1000.0, 1.0,
                            {"melt": 2.0, "block": 0.1}, 5.0, 0.0)

    assert state.total_absorbed_w == pytest.approx(2.1)
    assert state.density_w_m3(liquid) == pytest.approx(2.0e5)
    assert state.selectivity(liquid, ice) == pytest.approx(2000.0)


# --------------------------------------------------------------------------
# the perturbation, and where it stops
# --------------------------------------------------------------------------

def test_a_small_low_permittivity_body_is_a_perturbation() -> None:
    cavity = _oven()
    mode = cavity.dominant_mode
    chip = DielectricLoad("chip", EM_MATERIALS["ptfe"], 1.0e-6, _centre())

    assert chip.is_perturbation(cavity, mode)
    assert abs(chip.frequency_pull_fraction(cavity, mode)) < 0.01


def test_a_cup_of_water_is_not_a_perturbation_and_says_so() -> None:
    """The honest limit of this model, asserted.

    Two hundred millilitres in a fifteen-litre cavity is five percent of
    the mode's electric volume at eighty times the permittivity, and the
    perturbation estimate comes out over unity -- which is the estimate
    reporting its own collapse.  A real oven answers this by being
    multimode and by a magnetron that pulls; neither is modelled here,
    so the honest thing is to refuse the claim rather than print a
    number.
    """
    cavity = _oven()
    mode = cavity.dominant_mode
    cup = DielectricLoad("cup", water_em("liquid"), 2.0e-4, _centre())

    assert not cup.is_perturbation(cavity, mode)
    assert abs(cup.frequency_pull_fraction(cavity, mode)) > 1.0


def test_a_load_needs_a_volume() -> None:
    with pytest.raises(ValueError, match="needs a volume"):
        DielectricLoad("empty", water_em("liquid"), 0.0)


# --------------------------------------------------------------------------
# the susceptor: heating what will not absorb
# --------------------------------------------------------------------------

def test_a_susceptor_must_declare_what_it_heats() -> None:
    """Otherwise it is a part that gets hot for no stated reason."""
    with pytest.raises(ValueError, match="must declare what it heats"):
        Susceptor("s", EM_MATERIALS["fr4"], 1.0e-5, _centre(),
                  conductance_w_k=0.5)
    with pytest.raises(ValueError, match="thermal conductance"):
        Susceptor("s", EM_MATERIALS["fr4"], 1.0e-5, _centre(),
                  serves="cold-finger", conductance_w_k=0.0)


def test_a_susceptor_runs_hotter_than_what_it_serves() -> None:
    """And by exactly what the bond cannot carry."""
    susceptor = Susceptor("s", EM_MATERIALS["fr4"], 1.0e-5, _centre(),
                          serves="cold-finger", conductance_w_k=0.5)

    assert susceptor.delivered_temperature_rise_k(10.0) == pytest.approx(20.0)
    assert susceptor.graph_attributes()["part_role"] == "susceptor"
    assert susceptor.graph_attributes()["serves"] == "cold-finger"


def test_a_susceptor_absorbs_like_any_other_body() -> None:
    """It is the declaration that differs, not the physics."""
    cavity = _oven()
    mode = cavity.dominant_mode
    plain = DielectricLoad("p", EM_MATERIALS["fr4"], 1.0e-5, _centre())
    susceptor = Susceptor("s", EM_MATERIALS["fr4"], 1.0e-5, _centre(),
                          serves="cryogen", conductance_w_k=0.5)

    assert susceptor.resistance_ohm(cavity, mode) == pytest.approx(
        plain.resistance_ohm(cavity, mode))


# --------------------------------------------------------------------------
# sublimation: the state the absorption drives
# --------------------------------------------------------------------------

def test_sublimation_costs_fusion_plus_vaporisation() -> None:
    """Under vacuum a solid never passes through liquid, and pays for both."""
    duty = SublimationDuty()

    assert duty.latent_heat_j_kg == pytest.approx(
        WATER_LATENT_HEAT_SUBLIMATION_J_KG)
    assert duty.mass_rate_kg_s(2.834e6) == pytest.approx(1.0, rel=1e-3)


def test_the_clearing_time_is_mass_over_rate() -> None:
    duty = SublimationDuty()

    assert duty.time_to_clear_s(0.01, 100.0) == pytest.approx(
        0.01 * WATER_LATENT_HEAT_SUBLIMATION_J_KG / 100.0)


def test_a_field_that_does_not_couple_never_clears_the_deposit() -> None:
    """Infinity is the correct answer, and more useful than an error."""
    duty = SublimationDuty()

    assert duty.time_to_clear_s(0.01, 0.0) == math.inf
    assert duty.mass_rate_kg_s(-5.0) == 0.0


# --------------------------------------------------------------------------
# the assembled applicator
# --------------------------------------------------------------------------

def _chamber(load) -> ShieldedChamber:
    cavity = _oven()
    frequency = cavity.dominant_mode.frequency_hz
    return ShieldedChamber(
        "oven", cavity, EM_MATERIALS["copper"],
        LoopPort("feed", (0.15, 0.10, 0.01), (1.0, 0.0, 0.0), 1.0e-4, 1.0e-8),
        Magnetron("mag", frequency_hz=frequency).as_source(
            "feed", cavity.return_node()),
        loads=(load,))


def test_the_ledger_closes_with_a_load_absorbing() -> None:
    """Delivered equals copper plus contents, in separate columns."""
    load = DielectricLoad("block", water_em("ice"), 1.0e-3, _centre())
    ledger = _chamber(load).power_ledger(_oven().dominant_mode.frequency_hz)

    assert ledger.absorbed_w > 0.0
    assert ledger.wall_loss_w > 0.0
    assert ledger.balance_error < 1.0e-9


def test_the_published_state_carries_what_anything_inside_would_feel() -> None:
    load = DielectricLoad("block", water_em("ice"), 1.0e-3, _centre())
    chamber = _chamber(load)
    frequency = _oven().dominant_mode.frequency_hz
    state = chamber.applicator_state(frequency)

    assert state.peak_field_v_m > 0.0
    assert state.absorbed_w["block"] == pytest.approx(
        chamber.power_ledger(frequency).absorbed_w, rel=1e-9)
    assert state.graph_attributes()["part_role"] == "applicator-state"


def test_an_absorbing_load_lowers_the_field_the_cavity_reaches() -> None:
    """A lossy body spoils the cavity, and the field it sits in falls.

    This is the feedback that makes an applicator self-limiting, and it
    only appears because the load is inside the circuit rather than a
    number applied afterwards.
    """
    frequency = _oven().dominant_mode.frequency_hz
    quiet = DielectricLoad("q", water_em("vapour"), 1.0e-3, _centre())
    lossy = DielectricLoad("l", EM_MATERIALS["fr4"], 1.0e-3, _centre())

    assert (_chamber(lossy).applicator_state(frequency).peak_field_v_m
            < _chamber(quiet).applicator_state(frequency).peak_field_v_m)
