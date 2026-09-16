"""Where the metal goes: a closed mass balance from worn part to filter.

An engine's oil gets dirty because the engine is wearing out, and the
two numbers are the same number. `wear.py` already tracks every real
wearing part as a damage fraction, 0.0 new to 1.0 worn out. This module
says what that damage IS in grams of metal, debits it from the part, and
follows it -- into suspension, onto the filter, into the sludge, out
with the drain -- without ever creating or losing any of it.

THE RULE, AND WHY IT IS WORTH THE TROUBLE. Metal in the oil is not a
rate anybody gets to choose. If the rings were at damage y and are now
at damage x, then exactly (x - y) of their shed allowance became flake,
and that flake is somewhere. So:

    entered = captured + suspended + deposited + drained

holds exactly, every tick, and `balance_error_g` is checked rather than
hoped for. A generation rate guessed in grams per hour cannot tell you
that a filter filled early because the bearings are going; a mass
balance can, because the grams came from the bearings.

WHAT THIS BUYS THAT A RATE DOES NOT. Real engines are diagnosed by
spectrographic oil analysis, which reports each metal separately in
parts per million, and the mix is the diagnosis:

    iron + chromium   rings and bore -- the ring pack is going
    copper + lead     bearing overlay -- the shells are giving up
    aluminium         piston skirt or crown
    tin               bearing overlay on an older engine

Because every source declares WHICH metals it sheds, `ppm_by_metal`
reports the same thing a real lab does, off the same mass balance. An
engine that has eaten its bearings reads high copper and lead, and it
reads that way because the copper and lead really did come off the
bearings in this model, not because a rule said it should.

HOW THE SHED ALLOWANCES ARE CALIBRATED, AND AGAINST WHAT. Not against
a generic "used oil runs 20 to 100 ppm iron" band, which is an average
over every engine ever sampled and calibrates nothing. Against one real
engine, by the real test method, at the real sampling interval:

    engine    cat-c18-industrial-diesel, in this catalogue: 18.1 L, 6
              cylinders, sump as CrankcaseState sizes it
    method    ASTM D5185 -- wear metals in used lubricating oil by
              inductively coupled plasma atomic emission spectrometry,
              reported as ppm by mass, which is what `ppm_by_metal`
              returns and what a lab sends back
    interval  500 hours, one Caterpillar S.O.S sampling interval
    check     a healthy engine at that interval must read comfortably
              under the published caution thresholds for its class, and
              the MIX must be right: iron highest, chromium with it from
              the ring facing, copper and lead well below in the ratio a
              bearing overlay actually sheds

The allowances are then set BACKWARDS from that reading rather than
guessed forwards: a healthy heavy diesel at 500 hours with a ~45 kg sump
reads tens of ppm of iron, which is one to two grams of iron actually in
suspension, and the allowances are whatever puts that much there once
the fine/coarse split and the filter have had their say. Guessed
forwards they came out four times too high and a sound engine read past
its own caution threshold.

The same allowances are then scaled by displacement for every other
engine, which is disclosed extrapolation from one calibrated point, not
a second calibration.

ESTIMATION WHERE IT IS HONEST TO ESTIMATE. Not every gram is traceable:
cylinder bore wear pairs with the rings, gasket fretting and general
debris have no single owner. `UNTRACKED_SHARE` is a disclosed fraction
added alongside the tracked sources to stand in for those, and it is
counted in the ledger like everything else, so the balance still closes.
It is an estimate and it is labelled as one -- what is NOT estimated is
the part that a damaged component actually contributed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DebrisSource:
    """One wearing part, and the metal it gives up on the way out."""
    component: str              # a name in wear.COMPONENTS
    shed_g_full_life: float     # grams lost going from damage 0.0 to 1.0, at reference size
    metals: tuple               # ((element, mass fraction), ...), summing to 1
    reaches_oil: bool           # False: it wears somewhere the oil never sees
    why: str
    # THE SHARE TOO FINE FOR THE FILTER TO EVER CATCH, by mass.
    #
    # This is the reason spectrographic oil analysis exists. Rubbing
    # wear makes particles from well under a micrometre up to tens of
    # micrometres, and a full-flow element rated at ten simply does not
    # see the bottom of that range -- it passes straight through, every
    # pass, for the life of the oil. So a healthy engine's oil carries
    # tens of ppm of iron with a perfectly good filter fitted, and a lab
    # can read the wear rate off a sample precisely BECAUSE the fine
    # fraction is never removed.
    #
    # Assuming the filter caught everything made the oil read near zero
    # and the whole diagnostic worthless. Bearing overlay wears finest
    # (soft metals, smeared); ring and bore scuffing throws coarser
    # flake that a filter does stop.
    fine_fraction: float = 0.45


# Reference engine for the shed masses below: a 4 litre, 6 cylinder
# engine. A bigger engine has bigger rings and more of them, so the
# allowance scales with displacement -- disclosed, and much closer than
# pretending a 25 cc trimmer and a marine diesel shed the same gram.
REFERENCE_DISPLACEMENT_L = 4.0

# What is NOT individually tracked -- bore wear paired with the rings,
# gasket fretting, timing gear, general debris -- as a share added on
# top of the tracked sources. An estimate, declared as one.
UNTRACKED_SHARE = 0.25


_SOURCES: tuple[DebrisSource, ...] = (
    DebrisSource(
        "piston_rings", shed_g_full_life=18.0,
        metals=(("Fe", 0.72), ("Cr", 0.25), ("Mo", 0.03)),
        fine_fraction=0.35, reaches_oil=True,
        why="a ring loses a few tenths of a millimetre of radial thickness over its "
            "life, and there are three of them per cylinder; the chromium is the "
            "face plating, which is why Fe and Cr rise together when a ring pack goes"),
    DebrisSource(
        "oil_control_ring", shed_g_full_life=5.5,
        metals=(("Fe", 0.80), ("Cr", 0.20)),
        fine_fraction=0.35, reaches_oil=True,
        why="thinner and more lightly loaded than the compression rings, but it is the "
            "one whose wear shows up as oil consumption long before anything seizes"),
    DebrisSource(
        "connecting_rod_bearings", shed_g_full_life=5.0,
        metals=(("Cu", 0.55), ("Pb", 0.30), ("Sn", 0.12), ("Al", 0.03)),
        fine_fraction=0.7, reaches_oil=True,
        why="the overlay goes first -- a few tens of micrometres of lead-tin over a "
            "copper-lead lining -- which is exactly why rising copper and lead in an "
            "oil sample is the classic warning that the shells are on their way out"),
    DebrisSource(
        "crankshaft", shed_g_full_life=2.0,
        metals=(("Fe", 1.0),),
        fine_fraction=0.3, reaches_oil=True,
        why="a hardened journal gives up very little until something else has already "
            "failed; iron with no copper alongside it points at the shaft, not the shell"),
    DebrisSource(
        "valve_guides", shed_g_full_life=1.5,
        metals=(("Cu", 0.60), ("Zn", 0.25), ("Fe", 0.15)),
        fine_fraction=0.55, reaches_oil=True,
        why="bronze guides shed copper and zinc into the oil that drains past them, "
            "which is why guide wear can be mistaken for bearing wear on copper alone"),
    DebrisSource(
        "piston_crown", shed_g_full_life=3.0,
        metals=(("Al", 0.94), ("Si", 0.06)),
        fine_fraction=0.4, reaches_oil=True,
        why="crown and skirt erosion puts aluminium and its silicon alloying into the "
            "oil; silicon ALSO comes in as road dust, which is why the two are read "
            "together and a high Si with no Al means a leaking air filter"),
    DebrisSource(
        "apex_seals", shed_g_full_life=11.0,
        metals=(("Fe", 0.55), ("C", 0.45)),
        fine_fraction=0.3, reaches_oil=True,
        why="a rotary's seals scrape the housing directly and are famously the short-"
            "lived part; carbon-composite seals put carbon in the oil as well as iron"),
    DebrisSource(
        "valve_seats", shed_g_full_life=1.0,
        metals=(("Fe", 0.85), ("Cr", 0.15)),
        fine_fraction=0.25, reaches_oil=True,
        why="recession debris mostly leaves down the exhaust, but the share that gets "
            "past the guide reaches the oil"),
    # --- wears where the oil never sees it
    DebrisSource(
        "spark_plug", shed_g_full_life=0.5,
        metals=(("Pt", 0.5), ("Ni", 0.5)), reaches_oil=False,
        why="electrode erosion goes out of the exhaust, not into the sump"),
    DebrisSource(
        "fuel_injector", shed_g_full_life=0.4,
        metals=(("Fe", 1.0),), reaches_oil=False,
        why="nozzle erosion leaves with the spray"),
    DebrisSource(
        "engine_oil", shed_g_full_life=0.0,
        metals=(), reaches_oil=False,
        why="the oil's own degradation is not metal; it is tracked as oil condition, "
            "and it is what the sludge mode in fouling.py is about"),
)

BY_COMPONENT: dict[str, DebrisSource] = {s.component: s for s in _SOURCES}


def source(component: str) -> DebrisSource:
    s = BY_COMPONENT.get(str(component))
    if s is None:
        raise KeyError(
            f"no debris source declared for wear component {component!r}; "
            f"declared: {', '.join(sorted(BY_COMPONENT))}")
    return s


def shed_allowance_g(component: str, displacement_l: float) -> float:
    """Grams this part gives up over its whole life, at this engine's size."""
    s = source(component)
    if not s.reaches_oil:
        return 0.0
    scale = max(0.05, float(displacement_l)) / REFERENCE_DISPLACEMENT_L
    return s.shed_g_full_life * scale


# ---------------------------------------------------------------------
# THE VOCABULARY, AS MATRICES
# ---------------------------------------------------------------------
#
# An engine debits every wearing component at once into a fixed element
# vocabulary, and that is a matrix product, not a loop: a vector of
# damage steps times a component-by-element matrix of mass fractions
# gives the whole tick's metal in one operation. The ledger's own pools
# are vectors over the same vocabulary for the same reason -- there is
# one way through this module, and it is the vector way.

import numpy as _np

# The fixed element vocabulary. Order is the matrix's column order and
# is part of the contract: anything reading these arrays indexes by it.
ELEMENTS: tuple = ("Fe", "Cr", "Cu", "Pb", "Sn", "Al", "Si", "Mo", "Zn", "C", "Pt", "Ni")
_ELEMENT_INDEX: dict = {el: i for i, el in enumerate(ELEMENTS)}

# Components that actually reach the oil, in a fixed row order.
OIL_WETTED_COMPONENTS: tuple = tuple(s.component for s in _SOURCES if s.reaches_oil)
_COMPONENT_INDEX: dict = {c: i for i, c in enumerate(OIL_WETTED_COMPONENTS)}


def _build_matrices():
    n, m = len(OIL_WETTED_COMPONENTS), len(ELEMENTS)
    metals = _np.zeros((n, m), dtype=_np.float64)
    shed = _np.zeros(n, dtype=_np.float64)
    fine = _np.zeros(n, dtype=_np.float64)
    for comp in OIL_WETTED_COMPONENTS:
        s = BY_COMPONENT[comp]
        r = _COMPONENT_INDEX[comp]
        shed[r] = s.shed_g_full_life
        fine[r] = s.fine_fraction
        for el, frac in s.metals:
            metals[r, _ELEMENT_INDEX[el]] = float(frac)
    return metals, shed, fine


#: (components x elements) mass fractions, (components,) shed allowance
#: at reference displacement, (components,) unfilterable fine share.
METAL_MATRIX, SHED_G, FINE_FRAC = _build_matrices()
_FE = _ELEMENT_INDEX["Fe"]


def damage_vector(wear_state: dict) -> "_np.ndarray":
    """A wear.py damage ledger as a vector in OIL_WETTED_COMPONENTS order."""
    return _np.array([float(wear_state.get(c, 0.0)) for c in OIL_WETTED_COMPONENTS],
                     dtype=_np.float64)


_FE = _ELEMENT_INDEX["Fe"]


@dataclass
class DebrisLedger:
    """Every gram that has come off the engine, and where it is now.

    `debit` is the only way in: it moves mass out of the worn parts and
    into suspension. Everything after that moves it between places it
    can be, and `balance_error_g` is checked rather than hoped for:

        entered = captured + suspended + deposited + drained

    Metal is carried COARSE and FINE separately, per element, because
    that is what makes the diagnosis work. Bearing overlay wears fine
    and stays in suspension; ring and bore flake is coarse and the
    filter takes it out. Apply one average suspended share to every
    element and a worn set of shells reads the same as a worn ring
    pack, which is the one thing an oil sample is for.

    Both pools are vectors over ELEMENTS and every operation on them is
    a whole-array operation.
    """
    entered_g: float = 0.0
    captured_g: float = 0.0
    deposited_g: float = 0.0
    drained_g: float = 0.0
    fine: "_np.ndarray" = field(default_factory=lambda: _np.zeros(len(ELEMENTS)))
    coarse: "_np.ndarray" = field(default_factory=lambda: _np.zeros(len(ELEMENTS)))
    estimated_g: float = 0.0
    # Soot, carried apart from the metals because it is not one: a
    # spectrographic analysis does not report it (it is measured
    # separately, by thermogravimetry or infrared, as a percentage by
    # mass) and it outweighs every metal here put together.
    soot_fine_g: float = 0.0
    soot_coarse_g: float = 0.0

    def debit(self, damage_before, damage_after,
              displacement_l: float = REFERENCE_DISPLACEMENT_L,
              include_untracked: bool = True) -> float:
        """Debit every component's wear for this tick, in one operation.

        The rings were at `damage_before` and are now at `damage_after`,
        so exactly that share of their shed allowance is now flake, and
        the flake is somewhere. Wear cannot un-happen, so a negative
        step contributes nothing. Both arguments are vectors in
        OIL_WETTED_COMPONENTS order -- see `damage_vector`."""
        before = _np.asarray(damage_before, dtype=_np.float64)
        after = _np.asarray(damage_after, dtype=_np.float64)
        step = _np.maximum(0.0, after - before)
        if not step.any():
            return 0.0
        scale = max(0.05, float(displacement_l)) / REFERENCE_DISPLACEMENT_L
        grams = SHED_G * scale * step
        per_element = grams @ METAL_MATRIX
        fine_per_element = (grams * FINE_FRAC) @ METAL_MATRIX
        extra = grams * UNTRACKED_SHARE if include_untracked else _np.zeros_like(grams)
        extra_total = float(extra.sum())
        if extra_total > 0.0:
            per_element = per_element.copy()
            fine_per_element = fine_per_element.copy()
            per_element[_FE] += extra_total
            fine_per_element[_FE] += float((extra * FINE_FRAC).sum())
        self.fine = self.fine + fine_per_element
        self.coarse = self.coarse + (per_element - fine_per_element)
        total = float(grams.sum()) + extra_total
        self.entered_g += total
        self.estimated_g += extra_total
        return total

    def debit_soot(self, grams: float) -> float:
        """Add soot delivered to the sump by blow-by (`soot_into_oil_g`).

        It enters the same ledger as the metal because it is the same
        question -- mass in the oil, loading the same element -- but it
        is kept apart from the metals so an oil analysis still reads as
        an oil analysis."""
        g = max(0.0, float(grams))
        if g <= 0.0:
            return 0.0
        self.soot_fine_g += g * SOOT_FINE_FRACTION
        self.soot_coarse_g += g * (1.0 - SOOT_FINE_FRACTION)
        self.entered_g += g
        return g

    def soot_frac_of_oil(self, oil_mass_kg: float) -> float:
        """Soot as a fraction of the charge, which is how a used oil
        report states it: a diesel runs one to five percent by the end
        of a drain interval, and past about five it is condemned."""
        m = max(1e-6, float(oil_mass_kg))
        return (self.soot_fine_g + self.soot_coarse_g) / 1000.0 / m

    def through_filter(self, passes_fraction: float, beta_ratio: float = 200.0) -> float:
        """Send some of what is suspended past the element.

        Only the COARSE flake is on offer: the fine fraction is below
        what the element rates at and goes through every pass, for the
        life of the oil. That is why a healthy engine's oil still reads
        tens of ppm of iron with a good filter fitted, and why a lab can
        read a wear rate off a sample at all."""
        from fouling import capture_efficiency
        share = max(0.0, min(1.0, float(passes_fraction))) * capture_efficiency(beta_ratio)
        caught = self.coarse * share
        self.coarse = self.coarse - caught
        soot_caught = self.soot_coarse_g * share
        self.soot_coarse_g -= soot_caught
        got = float(caught.sum()) + soot_caught
        self.captured_g += got
        return got

    def settle(self, fraction: float) -> float:
        """What drops into sludge where no filter reaches -- the sump
        floor, the galleries, under the cover. Heavy flake settles; the
        fine stuff largely stays up."""
        f = max(0.0, min(1.0, float(fraction)))
        drop_c = self.coarse * f
        drop_f = self.fine * (f * 0.25)
        self.coarse = self.coarse - drop_c
        self.fine = self.fine - drop_f
        soot_drop = self.soot_coarse_g * f + self.soot_fine_g * (f * 0.25)
        self.soot_coarse_g -= self.soot_coarse_g * f
        self.soot_fine_g -= self.soot_fine_g * (f * 0.25)
        moved = float(drop_c.sum() + drop_f.sum()) + soot_drop
        self.deposited_g += moved
        return moved

    def oil_change(self, drain_fraction: float = 0.9,
                   replace_filter: bool = True) -> float:
        """Out with the old. A drain takes the suspended metal and leaves
        the settled sludge exactly where it is, which is why a change
        does not reset an engine's history."""
        d = max(0.0, min(1.0, float(drain_fraction)))
        taken = float((self.coarse * d).sum() + (self.fine * d).sum())
        taken += (self.soot_coarse_g + self.soot_fine_g) * d
        self.coarse = self.coarse * (1.0 - d)
        self.fine = self.fine * (1.0 - d)
        self.soot_coarse_g *= (1.0 - d)
        self.soot_fine_g *= (1.0 - d)
        self.drained_g += taken
        if replace_filter:
            self.drained_g += self.captured_g
            self.captured_g = 0.0
        return taken

    def ppm_by_metal(self, oil_mass_kg: float) -> dict:
        """A spectrographic oil analysis: ASTM D5185, ppm by mass.

        Only what is IN the oil shows up in a sample -- metal on the
        filter and metal in the sludge is out of the bottle, which is
        why a sample taken after a change reads clean on an engine that
        is still destroying itself. A dict purely because that is how a
        report reads; the arithmetic above it is all whole-array."""
        m = max(1e-6, float(oil_mass_kg))
        total = (self.fine + self.coarse) / m * 1000.0
        return {el: float(v) for el, v in zip(ELEMENTS, total) if v > 0.0}

    @property
    def suspended_fine_g(self) -> float:
        return float(self.fine.sum()) + self.soot_fine_g

    @property
    def suspended_coarse_g(self) -> float:
        return float(self.coarse.sum()) + self.soot_coarse_g

    @property
    def suspended_g(self) -> float:
        return self.suspended_fine_g + self.suspended_coarse_g

    @property
    def accounted_g(self) -> float:
        return self.suspended_g + self.captured_g + self.deposited_g + self.drained_g

    @property
    def balance_error_g(self) -> float:
        """How much has been created or lost. Must stay at zero."""
        return self.accounted_g - self.entered_g

    def is_balanced(self, tolerance_g: float = 1e-9) -> bool:
        return abs(self.balance_error_g) <= tolerance_g


# ---------------------------------------------------------------------
# SOOT: what actually fills a diesel's oil
# ---------------------------------------------------------------------
#
# Wear metal is grams. Soot is KILOGRAMS, and leaving it out was the
# single biggest hole in this ledger. A heavy diesel's oil carries one
# to five percent soot by mass at the end of a drain interval -- on a
# forty-five kilogram sump that is well over a kilogram, against about
# eight grams of wear metal in the same period. Soot is a hundred and
# fifty times the wear debris, it is what blackens the oil in hours, it
# is what thickens it, and it is most of what a diesel's filter is
# actually holding.
#
# WHERE IT COMES FROM, and why it closes a loop that is already here.
# Soot is made in the cylinder, and the share of it that reaches the
# SUMP rather than the exhaust gets there by going past the rings --
# which is blow-by, which is `ring_leak`, which now rises with ring
# wear (crankcase_state.apply_ring_wear). So:
#
#     worn rings -> more blow-by -> more soot in the oil
#                -> filter blocks sooner AND the oil thickens
#
# every step of which was already modelled separately and none of which
# was connected. An engine with a tired ring pack really does black its
# oil faster, and now it does so for the real reason.
#
# Soot is also the case that makes the coarse/fine split earn itself.
# Individual soot particles are tens of nanometres and they agglomerate
# to around a micrometre -- far below what a full-flow element rates at
# -- so a filter takes very little of it however good the filter is.
# That is exactly why diesel oil goes black and STAYS black with a
# perfectly good filter fitted, and why soot is controlled by the oil's
# dispersant additives and the drain interval rather than by filtration.

# HOW MUCH SOOT A FUEL ACTUALLY MAKES -- which is NOT how bright it
# burns. `combustion_kernel.soot_luminosity` describes a flame's
# appearance, and using it as a mass yield says a petrol engine soots
# its oil not quite twice as slowly as a diesel. The real ratio is
# nearer a hundred to one: a diesel's diffusion flame makes particulate
# by the gram per kilogram of fuel, while a premixed petrol flame makes
# almost none however yellow it looks. Reusing the visual number put
# seven percent soot in a flat-six's sump, which would condemn the oil
# of an engine that in reality never darkens it.
#
# Grams of soot per kilogram of fuel burned, by combustion family --
# the same families combustion_kernel keys its visuals on, so a fuel
# says both things in the same vocabulary without one standing in for
# the other.
SOOT_G_PER_KG_FUEL_BY_FAMILY: dict = {
    "diesel": 2.0,          # pre-aftertreatment diffusion flame: the sooty case
    "multifuel": 2.5,       # worse, and the reason these engines smoke
    "kerosene": 0.8,
    "two-stroke": 3.0,      # it is burning its own lubricant
    "nitromethane": 0.05,
    "gasoline": 0.03,       # premixed and nearly clean, whatever the flame looks like
    "coal-gas": 0.05,
    "methanol": 0.002,      # essentially none: no carbon-carbon bonds to build soot from
}
DEFAULT_SOOT_G_PER_KG_FUEL = 0.5

#: Almost none of it is big enough for a full-flow element to stop.
SOOT_FINE_FRACTION = 0.92


def soot_yield_g_per_kg(family: str) -> float:
    """Soot made per kilogram of this family's fuel."""
    return SOOT_G_PER_KG_FUEL_BY_FAMILY.get(str(family), DEFAULT_SOOT_G_PER_KG_FUEL)


def soot_into_oil_g(fuel_burned_kg: float, family: str,
                    ring_leak_frac: float) -> float:
    """Soot delivered to the sump by blow-by.

    `family` is the combustion family the fuel belongs to, the same key
    combustion_kernel uses. A clean-burning fuel blackens the oil far
    more slowly for the same work -- which is why a methanol engine's
    oil stays clear and a diesel's does not survive an hour looking
    new -- and the share that reaches the sump at all is whatever gets
    past the rings, so a worn pack soots its own oil."""
    return (max(0.0, float(fuel_burned_kg)) * soot_yield_g_per_kg(family)
            * max(0.0, min(1.0, float(ring_leak_frac))))
