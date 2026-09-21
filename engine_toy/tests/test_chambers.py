"""The join: a driven cavity leaking through a bore, solved once.

The point of these tests is that nothing new was solved.  A cavity was
already a circuit, a penetration was already a two-port, an emitter is a
Thevenin source and free space is a resistor -- so if the unification
was real, assembling them should produce a chamber whose books close and
whose shielding is the bore's own attenuation plus the coupling at each
end.  Both are checked, and the second is checked in the regime where
the bore is weak enough not to load the cavity, because outside that
regime it is not supposed to hold.
"""

from __future__ import annotations

import pytest

from cavities import LoopPort, RectangularCavity
from chambers import Penetration, PowerLedger, ShieldedChamber
from em_materials import EM_MATERIALS
from emitters import Magnetron, RadiationLoad
from waveguides import CircularGuide


def _oven() -> RectangularCavity:
    return RectangularCavity("oven", width_m=0.30, height_m=0.20,
                             depth_m=0.25)


def _chamber(wall_m: float = 0.006, bore_m: float = 0.002,
             leaks: bool = True) -> ShieldedChamber:
    cavity = _oven()
    copper = EM_MATERIALS["copper"]
    frequency = cavity.dominant_mode.frequency_hz
    tube = Magnetron("mag", frequency_hz=frequency)
    drive = LoopPort("feed", (0.15, 0.10, 0.01), (1.0, 0.0, 0.0),
                     area_m2=1.0e-4, self_inductance_h=1.0e-8)
    penetrations = ()
    if leaks:
        penetrations = (Penetration(
            "gland",
            LoopPort("gland-pickup", (0.15, 0.10, 0.24), (1.0, 0.0, 0.0),
                     area_m2=1.0e-5, self_inductance_h=1.0e-8),
            CircularGuide("bore", radius_m=bore_m, length_m=wall_m),
            RadiationLoad.small_loop("sky", "gland/outside",
                                     cavity.return_node(), 1.0e-4,
                                     frequency)),)
    return ShieldedChamber(
        "oven", cavity, copper, drive,
        tube.as_source("feed", cavity.return_node()), penetrations)


def _frequency() -> float:
    return _oven().dominant_mode.frequency_hz


# --------------------------------------------------------------------------
# the books
# --------------------------------------------------------------------------

def test_the_ledger_closes_to_machine_precision() -> None:
    """Delivered equals copper warming plus power radiated away.

    Each column is read at the element that represents it; none is got
    by subtracting the others, so this is a real conservation check on
    the assembled solve rather than an identity.
    """
    ledger = _chamber().power_ledger(_frequency())

    assert ledger.delivered_w > 0.0
    assert ledger.balance_error < 1.0e-12


def test_wall_loss_and_radiation_are_separate_columns() -> None:
    """The distinction a summed model would lose.

    Almost all the power warms the copper; a vanishing sliver leaves the
    box.  If these were added together the chamber would look identical
    to one with no leak at all.
    """
    ledger = _chamber().power_ledger(_frequency())

    assert ledger.wall_loss_w > 1.0
    assert 0.0 < ledger.radiated_w < 1.0e-3
    assert ledger.leak_fraction < 1.0e-6


def test_a_sealed_chamber_radiates_nothing() -> None:
    ledger = _chamber(leaks=False).power_ledger(_frequency())

    assert ledger.radiated_w == 0.0
    assert ledger.shielding_db() == float("inf")
    assert ledger.balance_error < 1.0e-12


# --------------------------------------------------------------------------
# the shielding, and where it comes from
# --------------------------------------------------------------------------

def test_a_thicker_wall_shields_more() -> None:
    thin = _chamber(wall_m=0.002).power_ledger(_frequency())
    thick = _chamber(wall_m=0.012).power_ledger(_frequency())

    assert thick.radiated_w < thin.radiated_w
    assert thick.shielding_db() > thin.shielding_db() + 60.0


def test_the_shielding_increment_is_the_bores_own_attenuation() -> None:
    """The end-to-end check, and the reason the join is worth anything.

    Deepen the bore and the extra shielding the assembled solve reports
    must be exactly what the standalone two-port says that extra length
    costs.  The circuit never consults the decibel figure, and the
    decibel figure knows nothing about cavities or loops.

    Taken between 6 and 12 mm, where the bore is far enough below cutoff
    that it does not load the cavity.  Nearer the mouth it does, and
    then the increment is NOT the bore's alone -- which is the honest
    reason this is not asserted at 2 mm.
    """
    frequency = _frequency()
    shallow = _chamber(wall_m=0.006).power_ledger(frequency).shielding_db()
    deep = _chamber(wall_m=0.012).power_ledger(frequency).shielding_db()

    bore = lambda length: CircularGuide(
        "b", radius_m=0.002, length_m=length).attenuation_db(frequency)
    assert deep - shallow == pytest.approx(bore(0.012) - bore(0.006),
                                           rel=1e-3)


def test_the_chamber_shields_by_more_than_the_bore_alone() -> None:
    """Coupling at both ends counts too, and only the assembly sees it.

    A small pickup loop grips little of the mode and a small radiator
    sheds little of what reaches it.  Quoting the bore's attenuation as
    the chamber's shielding would understate it substantially, which is
    the mirror of the thin-wall error on the aperture side.
    """
    frequency = _frequency()
    ledger = _chamber(wall_m=0.006).power_ledger(frequency)
    bore = CircularGuide("b", radius_m=0.002,
                         length_m=0.006).attenuation_db(frequency)

    assert ledger.shielding_db() > bore + 30.0


def test_a_wider_bore_leaks_more() -> None:
    """Cutoff scales inversely with radius, so the bore is the lever."""
    frequency = _frequency()
    narrow = _chamber(bore_m=0.002).power_ledger(frequency)
    wide = _chamber(bore_m=0.006).power_ledger(frequency)

    assert wide.radiated_w > narrow.radiated_w


# --------------------------------------------------------------------------
# the ledger as an object
# --------------------------------------------------------------------------

def test_the_ledger_arithmetic_is_what_it_claims() -> None:
    ledger = PowerLedger(delivered_w=100.0, wall_loss_w=90.0,
                         radiated_w=10.0)

    assert ledger.accounted_w == pytest.approx(100.0)
    assert ledger.balance_error == pytest.approx(0.0)
    assert ledger.leak_fraction == pytest.approx(0.1)
    assert ledger.shielding_db() == pytest.approx(10.0)


def test_an_unbalanced_ledger_says_so_rather_than_hiding_it() -> None:
    ledger = PowerLedger(delivered_w=100.0, wall_loss_w=50.0,
                         radiated_w=10.0)

    assert ledger.balance_error == pytest.approx(0.4)


def test_the_chamber_declares_itself_to_the_graph() -> None:
    declared = _chamber().graph_attributes()

    assert declared["part_role"] == "shielded-chamber"
    assert declared["penetrations"] == ["gland"]
