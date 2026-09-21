"""What a running engine actually puts into the air around it, by species.

WHAT WAS ALREADY HERE, AND THE HOLE BETWEEN THE PIECES

`emissions.py` counts what went WRONG -- carbon monoxide, unburnt
hydrocarbon, oxides of nitrogen -- off the standard Heywood trends
against equivalence ratio, and prices what the carbon monoxide does to a
person standing in the bay. `combustion_efficiency.py` derives, from
chamber geometry alone, what fraction of the charge burnt at all.
`crankcase_state.py` tracks, per cylinder, the milligrams of oil the
rings let past and burn. `air_treatment.py` has had a `Contaminant`
carrying soot, SO2, NOx, CO and HC since it was written, and a filter
that removes them.

Between those four there is no accounting. Nothing states the CO2 and
water an engine makes, which is most of the mass leaving the pipe.
Nothing turns `combustion_efficiency`'s fraction into a quantity of
anything. Nothing connects `working_fluids.sulfur_frac` -- declared on
nine catalogued fuels, down to ultra-low-sulphur diesel's fifteen parts
per million -- to `Contaminant.so2_kg_per_kg`, which has been sitting
empty on the other side of the gap. And soot has existed in this project
only as a render colour, a particle size and a ledger of grams that
something else was expected to fill.

This module is that accounting, and it is a CONSERVATION LAW rather than
a table of yields. Every carbon atom that enters in the fuel leaves as
carbon dioxide, carbon monoxide, unburnt hydrocarbon or soot. Every
hydrogen leaves as water or is locked in one of the other three. Every
sulphur leaves as sulphur dioxide or as sulphate. `closure()` reports
what did not balance, and on a correct run that number is float noise.
Making carbon dioxide the RESIDUAL rather than a modelled term is the
whole design: it means soot cannot be quietly invented, because every
gram of it has to be a gram that carbon dioxide did not get.

THE ONE CALIBRATION, STATED

Sooting propensity is measured and ordered (organic_species.sooting_index,
the Calcote-Manos scale), but an index is not a mass. Exactly one
absolute anchor turns it into one:

    SOOT_ANCHOR_FRAC -- an uncontrolled direct-injection diesel at full
    load emits about three parts per thousand of its fuel mass as
    particulate.

That is a disclosed order-of-magnitude figure of the same kind as every
heating value and carbon fraction already in this project, and it is the
only absolute number in the soot path. Everything else -- how a gasoline
engine compares, how a bunker-fuel engine compares, what happens as the
mixture goes rich or the load comes off -- is relative to it through
quantities that are independently measured.

WHY A DIESEL SMOKES AT AN AIR/FUEL RATIO A PETROL ENGINE WOULD CALL LEAN

Because global equivalence ratio is the wrong variable for it, and this
is the single most important structural fact in the module. A spark
engine burns a PREMIXED charge: the whole cylinder is at one mixture,
and it cannot soot until that one mixture goes properly rich. A
compression-ignition engine burns a DIFFUSION flame: fuel sprays into
air and burns at the surface of the spray, so there are fuel-rich cores
at every load no matter how much excess air the cylinder holds overall.
Those two get different treatment here, selected by
`working_fluids.ignition`, which the catalogue already declares. Feeding
a diesel's global phi into a premixed sooting criterion predicts a
diesel never smokes, which is the opposite of the thing everyone knows
about diesels.

SMOKE IS NOT ONE SUBSTANCE

Black smoke is fuel soot. Blue smoke is lubricating oil that got past a
ring and left as droplets rather than burning. White smoke is condensed
fuel or water. They come from different places, look different, and are
removed by different hardware, so they are separate species here and
`crankcase_state.burn_kg_s` feeds the oil one directly.

SMOG IS A THIRD QUESTION AGAIN

Neither mass of soot nor mass of hydrocarbon predicts photochemical
smog, because smog is ozone made from hydrocarbon AND nitrogen oxide AND
sunlight, and a kilogram of methane and a kilogram of xylene are not
remotely the same input to it. `organic_species.ozone_reactivity_mir`
carries the per-species reactivity that distinction needs, and
`ozone_potential` below applies it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import organic_species as chem
import working_fluids


# ---------------------------------------------------------------------
# the one absolute anchor, and the relative physics around it
# ---------------------------------------------------------------------

#: Particulate as a fraction of fuel mass for an uncontrolled
#: direct-injection diesel at full load. Every other soot number in this
#: module is relative to this one.
SOOT_ANCHOR_FRAC = 0.003

#: Sooting index of the fuel the anchor was measured on, so the anchor
#: transfers to other fuels by their own measured propensity rather than
#: by assertion.
SOOT_ANCHOR_INDEX = chem.MOLECULES["diesel-blend"].sooting_index

#: Share of fuel sulphur that oxidises past SO2 to SO3 and ends up as a
#: sulphate droplet instead of a gas. The standard inventory figure, and
#: the reason taking sulphur out of diesel cut the measured particulate
#: number before anyone fitted a filter.
SULFUR_TO_SULFATE_FRAC = 0.02

#: Share of lubricating oil reaching the chamber that leaves as
#: condensed droplets rather than burning. The balance burns and is
#: accounted as ordinary products. A worn engine's visible blue haze is
#: this fraction of a larger number, not a larger fraction.
OIL_ESCAPE_FRAC = 0.10

#: Premixed sooting onset, as an equivalence ratio, for a fuel of zero
#: sooting index; an aromatic reaches its own onset far leaner. The
#: span between these two is the measured spread of critical
#: carbon-to-oxygen ratios across fuel structures, not a shaping knob.
PREMIXED_ONSET_PHI_CLEAN = 1.60
PREMIXED_ONSET_PHI_SOOTY = 1.00


def _sooting_propensity(mol: chem.Molecule) -> float:
    """Measured sooting index, normalised so naphthalene is one."""
    return max(0.0, min(1.0, mol.sooting_index / 100.0))


def soot_fraction_of_fuel(mol: chem.Molecule, phi: float, load_frac: float,
                          compression_ignition: bool,
                          completeness: float = 1.0) -> float:
    """Fraction of fuel mass leaving as soot.

    Two regimes, chosen by how the fuel meets its air, because they are
    genuinely different flames and not two settings of one curve."""
    prop = _sooting_propensity(mol)
    if mol.carbon <= 0.0:
        return 0.0                     # hydrogen cannot soot, and does not
    load = max(0.0, min(1.0, float(load_frac)))
    p = max(1e-6, float(phi))
    unburnt = max(0.0, 1.0 - max(0.0, min(1.0, float(completeness))))

    if compression_ignition:
        # A diffusion flame soots from its own rich cores. What sets how
        # much survives is how much fuel is crowded into the available
        # mixing time, which is load -- and how much excess air is left
        # to burn it back up on the way out, which is 1/phi. Squaring
        # load is the observed shape of a smoke curve: a diesel at half
        # load is far cleaner than half of a diesel at full load.
        formed = SOOT_ANCHOR_FRAC * (prop / max(1e-9, SOOT_ANCHOR_INDEX / 100.0))
        formed *= load * load
        # oxidation on the way out, strong when there is spare oxygen
        oxidised = min(0.95, 0.85 * max(0.0, 1.0 - p))
        return max(0.0, formed * (1.0 - oxidised) + formed * unburnt)

    # Premixed: nothing soots until the whole charge is rich enough, and
    # where that threshold sits is a property of the fuel's structure.
    onset = (PREMIXED_ONSET_PHI_CLEAN
             + (PREMIXED_ONSET_PHI_SOOTY - PREMIXED_ONSET_PHI_CLEAN) * prop)
    # A premixed engine still makes a little particulate from fuel films
    # on the wall and from oil, well below any sooting limit. It is small
    # and it is not zero, and calling it zero is how a gasoline direct
    # injection engine ends up looking cleaner than it measures. This
    # floor is present at ALL mixtures and the rich term ADDS to it --
    # the two are different mechanisms, and subsuming the floor the
    # moment the sooting limit is crossed would put a step down in the
    # middle of a curve that only ever rises.
    floor = 0.05 * SOOT_ANCHOR_FRAC * prop * load
    if p <= onset:
        return floor
    excess = p - onset
    return floor + SOOT_ANCHOR_FRAC * prop * (excess ** 1.5) * max(0.2, load) * 4.0


# ---------------------------------------------------------------------
# the balance
# ---------------------------------------------------------------------

@dataclass
class ProductRates:
    """Everything leaving the pipe, kg/s, by species key.

    Keys are `organic_species` keys throughout, so any consumer can ask
    the species table what each one is rather than knowing by name."""
    kg_s: dict = field(default_factory=dict)
    fuel_kg_s: float = 0.0
    oil_kg_s: float = 0.0
    air_kg_s: float = 0.0
    phi: float = 1.0
    #: Which organic_species row the fuel was, so escaped fuel can be
    #: read back off the axis without the caller having to remember.
    fuel_species: str = ""
    note: str = ""

    def get(self, species: str) -> float:
        return float(self.kg_s.get(species, 0.0))

    @property
    def hydrocarbon_kg_s(self) -> float:
        """The regulatory total, read back off the species axis.

        "Hydrocarbon" is a reporting category, not a substance, so it is
        derived here rather than carried as a key. For a carbon-free
        fuel it is correctly zero and the escaped fuel is still on the
        axis under its own name."""
        mol = chem.MOLECULES.get(self.fuel_species)
        if mol is None or mol.carbon <= 0.0:
            return 0.0
        return self.get(self.fuel_species)

    @property
    def total_kg_s(self) -> float:
        return sum(self.kg_s.values())

    @property
    def particulate_kg_s(self) -> float:
        """Everything that is a particle or a droplet, which is what a
        filter sees and what a particulate limit counts."""
        return (self.get("soot") + self.get("lubricating-oil")
                + self.get("sulfuric-acid"))

    @property
    def smoke_opacity_proxy(self) -> float:
        """Particulate per unit exhaust mass. Not an opacity meter
        reading -- opacity needs a path length and a size distribution,
        which belong with the plume rather than the engine -- but it is
        the quantity an opacity meter is responding to, and it is what
        `symbolic_atmosphere`'s Beer-Lambert extinction wants fed to it
        as a mass loading."""
        t = self.total_kg_s
        return self.particulate_kg_s / t if t > 0.0 else 0.0

    def ozone_potential_kg_s(self, fuel_species: str) -> float:
        """Grams of ozone per second this exhaust can make downwind.

        Weighted by the ESCAPED hydrocarbon's own reactivity. Using the
        fuel's reactivity for the escaped hydrocarbon is an approximation
        and a stated one: real exhaust hydrocarbon is partly cracked and
        partly oxygenated, and is typically MORE reactive than the fuel
        it came from, so this is a floor rather than an estimate."""
        mol = chem.MOLECULES.get(fuel_species)
        mir = mol.ozone_reactivity_mir if mol else 0.0
        return self.hydrocarbon_kg_s * mir

    def closure(self) -> dict:
        """What did not balance. Float noise on a correct run."""
        return dict(self._closure)

    _closure: dict = field(default_factory=dict, repr=False)


#: Every element this boundary can carry. Nothing a flame makes lies
#: outside it, and the ledger walks all of them rather than the two that
#: happen to be interesting.
ELEMENTS_TRACKED = ("C", "H", "O", "N", "S", "Ar")


def _element_kg(species: str, kg: float, element: str) -> float:
    mol = chem.MOLECULES.get(species)
    if mol is None:
        raise KeyError(
            f"{species!r} is on the product axis but has no declared "
            f"composition, so no element balance can be written for it. "
            f"Every key leaving this module must be an organic_species row.")
    return kg * mol.mass_fractions.get(element, 0.0)


def _charge_of(species: str) -> float:
    mol = chem.MOLECULES.get(species)
    return float(mol.charge) if mol is not None else 0.0


def burn(fuel_species: str, fuel_kg_s: float, phi: float,
         load_frac: float = 1.0, compression_ignition: bool = False,
         completeness: float = 1.0,
         co_kg_s: float = 0.0, hc_kg_s: float = 0.0, nox_kg_s: float = 0.0,
         oil_kg_s: float = 0.0) -> ProductRates:
    """The full species balance for an interval's fuel and oil.

    `co_kg_s`, `hc_kg_s` and `nox_kg_s` are NOT re-derived here. They
    come from `emissions.engine_out`, which already models them against
    the same equivalence ratio off the standard trends, and duplicating
    that with a second curve is how two numbers for one thing start
    disagreeing. This module takes them as given and closes the balance
    around them -- which is also what makes the balance a check on them:
    a carbon monoxide rate that implied more carbon than the fuel
    contained would show up in `closure()` immediately."""
    mol = chem.molecule(fuel_species)
    fuel = max(0.0, float(fuel_kg_s))
    oil = max(0.0, float(oil_kg_s))
    p = max(1e-6, float(phi))

    out: dict = {}

    # -- 1. soot, from the fuel's own measured propensity -------------
    soot_frac = soot_fraction_of_fuel(mol, p, load_frac, compression_ignition,
                                      completeness)
    soot = fuel * soot_frac

    # -- 2. oil: what escapes as droplets, and what burns -------------
    oil_escaped = oil * OIL_ESCAPE_FRAC
    oil_burnt = oil - oil_escaped
    if oil_escaped > 0.0:
        out["lubricating-oil"] = oil_escaped

    # -- 3. sulphur: every atom of it, and only once -----------------
    # Oil that escaped as a droplet carries its own sulphur out with it,
    # still bound in the oil. Oxidising that same sulphur to SO2 as well
    # would emit it twice, which the element ledger catches as a
    # negative sulphur residual -- so what escaped is reserved first and
    # only the sulphur that actually burnt is oxidised. The deferred
    # sulphur in unburnt fuel is handled the same way, below, once the
    # escape rates are known.
    lube = chem.MOLECULES["lubricating-oil"]

    # -- 4. the incomplete products, taken from emissions.py ----------
    # ...but never more carbon than the fuel brought. `emissions.py`
    # derives carbon monoxide and hydrocarbon from equivalence ratio
    # alone and never asks what the fuel is made of, so a HYDROGEN
    # engine comes back from it carrying about a tenth of a gram per
    # second of carbon monoxide -- out of a fuel with no carbon atom in
    # it at all. That is a real defect in a module this one deliberately
    # does not duplicate, so the balance clamps the impossible part to
    # what the fuel can actually supply, keeps the numbers physical for
    # everything downstream, and reports the shortfall in `closure()`
    # and in `note` rather than absorbing it quietly.
    co = max(0.0, float(co_kg_s))
    hc = max(0.0, float(hc_kg_s))
    nox = max(0.0, float(nox_kg_s))
    lube_pre = chem.MOLECULES["lubricating-oil"]
    co_c = chem.MOLECULES["carbon-monoxide"].mass_fractions["C"]
    fuel_c = mol.mass_fractions.get("C", 0.0)
    unburnt_fuel = 0.0

    # A fuel with no carbon in it cannot emit a hydrocarbon. What
    # `emissions.py` returns in that slot is real escaping fuel -- a
    # hydrogen engine does slip unburnt hydrogen -- so it is moved to
    # the fuel's own species key, where the hydrogen balance picks it up
    # and where nothing downstream can read it as a carbon species.
    if fuel_c <= 0.0 and hc > 0.0:
        unburnt_fuel = hc
        hc = 0.0

    # Soot and escaped lubricating oil have already claimed their carbon
    # above, and they are claims this module made itself from
    # conservation. Only what is LEFT after them is available to the
    # rates handed in from outside, so the reservation happens first --
    # clamping against gross carbon instead would let the oil's carbon be
    # spent twice and leave a residual that looks like a balance error
    # when it is really a double count.
    carbon_total = (fuel * fuel_c + oil * lube_pre.mass_fractions.get("C", 0.0))
    carbon_reserved = (_element_kg("soot", soot, "C")
                       + _element_kg("lubricating-oil", oil_escaped, "C"))
    carbon_available = max(0.0, carbon_total - carbon_reserved)

    claimed_c = co * co_c + hc * fuel_c
    carbon_overdrawn = 0.0
    if claimed_c > carbon_available:
        carbon_overdrawn = claimed_c - carbon_available
        scale = carbon_available / claimed_c if claimed_c > 0.0 else 0.0
        co *= scale
        hc *= scale
    if co > 0.0:
        out["carbon-monoxide"] = co
    # Escaped fuel goes on the axis under THE FUEL'S OWN SPECIES, never
    # under a category word like "hydrocarbon". A master solver's state
    # axis is made of species, and "hydrocarbon" is not one -- it has no
    # formula, no molar mass and no phase, so it cannot be balanced, and
    # a boundary that spoke it would be handing the solver a name it
    # could not resolve. `hydrocarbon_kg_s` below is the accessor for
    # anyone who wants the regulatory total back.
    if hc > 0.0:
        out[fuel_species] = out.get(fuel_species, 0.0) + hc
    if nox > 0.0:
        out["nitrogen-dioxide"] = nox
    if soot > 0.0:
        out["soot"] = soot
    if unburnt_fuel > 0.0:
        out[fuel_species] = out.get(fuel_species, 0.0) + unburnt_fuel

    # -- 5. what came in, once, so every sink can be a residual -------
    air_kg_s = fuel * mol.stoichiometric_afr() / p
    o2_in = air_kg_s * chem.AIR_O2_MASS_FRACTION
    supplied = {fuel_species: fuel, "lubricating-oil": oil,
                "oxygen": o2_in,
                "nitrogen": air_kg_s * (1.0 - chem.AIR_O2_MASS_FRACTION)}

    def _left(element: str) -> float:
        """What of this element has not yet been spoken for."""
        went_in = sum(_element_kg(k, v, element) for k, v in supplied.items())
        spoken = sum(_element_kg(k, v, element) for k, v in out.items())
        return went_in - spoken

    # -- 6. sulphur that actually burnt, split gas and droplet --------
    # Everything above has already put escaped oil and unburnt fuel on
    # the axis, so `_left("S")` is exactly the sulphur that went through
    # the flame and nothing else.
    s_burnt = _left("S")
    if s_burnt > 0.0:
        s_to_sulfate = s_burnt * SULFUR_TO_SULFATE_FRAC
        so2 = chem.MOLECULES["sulfur-dioxide"]
        h2so4 = chem.MOLECULES["sulfuric-acid"]
        out["sulfur-dioxide"] = (s_burnt - s_to_sulfate) / so2.mass_fractions["S"]
        out["sulfuric-acid"] = s_to_sulfate / h2so4.mass_fractions["S"]

    # -- 7. the four sinks, each the residual of its own element ------
    # Carbon dioxide, water, molecular nitrogen and molecular oxygen are
    # not modelled: they are what is LEFT of carbon, hydrogen, nitrogen
    # and oxygen once every escape above has taken its share. Deriving
    # all four this way rather than two of them is what closes the whole
    # ledger instead of half of it -- the earlier version computed
    # leftover oxygen from stoichiometric demand and passed the air's
    # nitrogen straight through, so the nitrogen in NOx arrived from
    # nowhere and the oxygen bound in CO and NOx was spent twice. Both
    # showed up the moment N and O were added to the ledger.
    #
    # ORDER MATTERS AND IS NOT ARBITRARY. Oxygen is taken last, because
    # every other sink consumes oxygen and none of them consume what is
    # left of it.
    for element, sink in (("C", "carbon-dioxide"), ("H", "water"),
                          ("N", "nitrogen"), ("O", "oxygen")):
        remaining = _left(element)
        if remaining <= 0.0:
            continue
        out[sink] = out.get(sink, 0.0) + (
            remaining / chem.MOLECULES[sink].mass_fractions[element])

    # Argon is genuinely inert: it arrives in the air and leaves
    # untouched, and saying so is cheaper than leaving a hole in the
    # ledger that looks like a leak.
    argon_in = air_kg_s * chem.AIR_MOLE_FRAC["argon"] * (
        chem.MOLECULES["argon"].molar_mass_g_mol / chem.AIR_MOLAR_MASS_G_MOL)
    if argon_in > 0.0:
        out["argon"] = argon_in
        supplied["argon"] = argon_in

    o2_left = out.get("oxygen", 0.0)
    # A RICH mixture can leave this negative, and when it does the
    # products handed in are not achievable with the air supplied. That
    # is a real finding about the rates, not a number to patch: the
    # physical resolution is that more carbon stays as CO and less
    # becomes CO2, which is equilibrium speciation and belongs to the
    # master solver's reaction rows, not to a boundary flux. So it is
    # reported and left alone. `emissions.py`'s rich CO and HC trends,
    # combined with complete conversion of everything they do not
    # claim, demand more oxygen than the charge holds above about
    # phi 1.4 -- the same class of defect as the carbon one above, found
    # the same way.
    oxygen_overdrawn = max(0.0, -_left("O"))

    rates = ProductRates(kg_s=out, fuel_kg_s=fuel, oil_kg_s=oil,
                         air_kg_s=air_kg_s, phi=p, fuel_species=fuel_species)

    # -- 8. the ledger, every element and the charge ------------------
    # Not a hand-written carbon line and a hand-written hydrogen line:
    # one walk over what went in and one over what came out, through the
    # species table, so adding a species to the balance cannot leave an
    # element silently unaccounted. This is the same shape the master
    # compendium checks every reaction with -- elements and charge
    # together (`ChemicalCompendium.reaction_balance`) -- and it is here
    # in that shape deliberately, so a boundary flux and a reaction row
    # are audited by the same standard before either reaches the solver.
    ledger = {}
    for element in ELEMENTS_TRACKED:
        went_in = sum(_element_kg(k, v, element) for k, v in supplied.items())
        came_out = sum(_element_kg(k, v, element) for k, v in out.items())
        ledger[f"{element}_in_kg_s"] = went_in
        ledger[f"{element}_residual_kg_s"] = went_in - came_out
    charge_in = sum(_charge_of(k) * v for k, v in supplied.items())
    charge_out = sum(_charge_of(k) * v for k, v in out.items())

    rates._closure = dict(ledger)
    rates._closure.update({
        "charge_residual_kg_s": charge_in - charge_out,
        "mass_in_kg_s": sum(supplied.values()),
        "mass_residual_kg_s": sum(supplied.values()) - rates.total_kg_s,
        "oxygen_short_kg_s": max(
            0.0, (mol.oxygen_required_kg(fuel)
                  + lube.oxygen_required_kg(oil_burnt)) - o2_in),
        "oxygen_overdrawn_kg_s": oxygen_overdrawn,
        "soot_frac_of_fuel": soot_frac,
        "carbon_overdrawn_kg_s": carbon_overdrawn,
    })
    bits = [f"soot {soot_frac * 1e4:.2f} parts per ten thousand of fuel"]
    if carbon_overdrawn > 0.0:
        bits.append(
            f"CARBON OVERDRAWN by {carbon_overdrawn * 1e6:.3f} mg/s: the CO and "
            f"HC rates handed in account for more carbon than "
            f"{mol.label} contains. They have been scaled back to what the "
            f"fuel can supply; the source of those rates needs to consult "
            f"the fuel's composition")
    short = rates._closure["oxygen_short_kg_s"]
    if short > 0.0:
        bits.append(f"oxygen short by {short * 1e3:.2f} g/s at phi {p:.2f}: "
                    "the charge cannot burn everything it was given")
    if oxygen_overdrawn > 0.0:
        bits.append(
            f"OXYGEN OVERDRAWN by {oxygen_overdrawn * 1e6:.1f} mg/s: the "
            f"products implied by the CO and HC rates need more oxygen than "
            f"the charge holds at phi {p:.2f}. Rich speciation is the "
            f"solver's to resolve through its own equilibrium rows; this "
            f"boundary reports the inconsistency rather than inventing a "
            f"shift reaction to hide it")
    rates.note = "; ".join(bits)
    return rates


# ---------------------------------------------------------------------
# wiring to the things that already exist
# ---------------------------------------------------------------------

def for_engine_out(fuel_name: str, fuel_kg_s: float, phi: float,
                   load_frac: float, rates, *, completeness: float = 1.0,
                   oil_kg_s: float = 0.0) -> ProductRates:
    """Close the balance around an `emissions.EngineOutRates`.

    This is the one-line join: whatever the sim already computed for CO,
    HC and NOx stays authoritative, and everything that was missing --
    carbon dioxide, water, sulphur dioxide, sulphate, soot, oil smoke,
    the leftover oxygen -- comes out of conservation around it."""
    fluid = working_fluids.WORKING_FLUIDS.get(fuel_name)
    species = chem.FUEL_COMPOSITION.get(fuel_name)
    if species is None:
        raise KeyError(
            f"{fuel_name!r} has no declared composition. It is a MIXTURE "
            f"(see organic_species.unmapped_fuels) and needs a mixture row "
            f"before a species balance can be written for it -- giving it a "
            f"single formula would be a fiction.")
    ci = bool(fluid is not None and fluid.ignition == "compression")
    return burn(species, fuel_kg_s, phi, load_frac, ci, completeness,
                co_kg_s=getattr(rates, "co_kg_s", 0.0),
                hc_kg_s=getattr(rates, "hc_kg_s", 0.0),
                nox_kg_s=getattr(rates, "nox_kg_s", 0.0),
                oil_kg_s=oil_kg_s)


def to_contaminant(rates: ProductRates, carrier_kg_s: float | None = None):
    """As an `air_treatment.Contaminant`, per kilogram of carrier air.

    `air_treatment` already knows how to move these through a filter, a
    wet scrubber and a reagent bed, and already knows which stage takes
    which species. Handing it a loaded `Contaminant` is all that was
    missing between an engine running in a bay and the treatment train
    in the same bay."""
    from air_treatment import Contaminant

    carrier = float(carrier_kg_s if carrier_kg_s is not None else rates.air_kg_s)
    if carrier <= 0.0:
        return Contaminant()
    c = Contaminant()
    c.set("soot", rates.get("soot") / carrier)
    c.set("so2", rates.get("sulfur-dioxide") / carrier)
    c.set("nox", rates.get("nitrogen-dioxide") / carrier)
    c.set("co", rates.get("carbon-monoxide") / carrier)
    c.set("hc", rates.hydrocarbon_kg_s / carrier)
    c.set("oil", rates.get("lubricating-oil") / carrier)
    c.set("water", rates.get("water") / carrier)
    # sulphate is a liquid droplet, not a gas and not a dry solid: it is
    # the one product here that `Contaminant` has no slot for, and it is
    # reported as condensed liquid rather than silently folded into soot
    c.set("liquid", rates.get("sulfuric-acid") / carrier)
    return c


def report(fuel_name: str = "ultra-low-sulfur-diesel", fuel_kg_s: float = 0.002,
           phi: float = 0.7, load_frac: float = 1.0) -> str:
    """One case, fully itemised, with the conservation check."""
    import emissions

    fluid = working_fluids.working_fluid(fuel_name)
    ci = fluid.ignition == "compression"
    species = chem.FUEL_COMPOSITION[fuel_name]
    mol = chem.molecule(species)
    air = fuel_kg_s * mol.stoichiometric_afr() / max(1e-6, phi)
    eo = emissions.engine_out(air + fuel_kg_s, fuel_kg_s, phi, load_frac, ci)
    r = for_engine_out(fuel_name, fuel_kg_s, phi, load_frac, eo,
                       oil_kg_s=fuel_kg_s * 0.002)

    lines = [f"{fluid.name}  as {species} ({mol.label})",
             f"  phi {phi:.2f}  load {load_frac:.2f}  "
             f"{'compression' if ci else 'spark'} ignition",
             f"  fuel {fuel_kg_s * 1e3:.3f} g/s   air {r.air_kg_s * 1e3:.1f} g/s",
             ""]
    for k in sorted(r.kg_s, key=lambda k: -r.kg_s[k]):
        m = chem.MOLECULES.get(k)
        label = m.label if m else k
        lines.append(f"  {k:<20} {r.kg_s[k] * 1e3:>12.6f} g/s   {label}")
    lines += ["",
              f"  particulate      {r.particulate_kg_s * 1e6:>10.3f} mg/s",
              f"  opacity proxy    {r.smoke_opacity_proxy * 1e6:>10.3f} mg/kg exhaust",
              f"  ozone potential  {r.ozone_potential_kg_s(species) * 1e6:>10.3f} mg O3/s",
              "", f"  {r.note}", "", "  CONSERVATION"]
    for k, v in r.closure().items():
        lines.append(f"    {k:<26} {v:+.6e}")
    return "\n".join(lines)


if __name__ == "__main__":
    for case in (("ultra-low-sulfur-diesel", 0.002, 0.70, 1.0),
                 ("ultra-low-sulfur-diesel", 0.002, 0.70, 0.3),
                 ("heavy-fuel-oil", 0.002, 0.80, 1.0),
                 ("pump-gasoline-87", 0.002, 1.00, 0.8),
                 ("pump-gasoline-87", 0.002, 1.45, 0.8),
                 ("hydrogen", 0.0005, 0.60, 1.0)):
        print(report(*case))
        print("=" * 72)
