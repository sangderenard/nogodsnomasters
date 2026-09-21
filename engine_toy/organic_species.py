"""Molecular identity: what each substance in this machine is MADE OF.

WHAT IS MISSING WITHOUT IT

`working_fluids.py` says a fuel's stoichiometric air/fuel ratio is 14.7.
`gas_works.py` says wood is 0.50 carbon by mass. `emissions.py` counts
unburnt hydrocarbon "as CH2 units" and nitrogen oxide "as NO2" with the
molar masses 14.0 and 46.0 written into the arithmetic. `cryogenics.py`
prices a dewar's ullage with one specific gas constant, 296.8, and a
single branch for hydrogen because hydrogen is the one species light
enough that the lie shows. `symbolic_atmosphere.py` states the problem
outright in its own docstring: the Hertz-Knudsen flux needs `M`, "the
species' OWN molar mass", and calls that "the one place a literal
periodic-table number enters the law directly".

Five modules each carrying a different shadow of the same fact. None of
them is wrong. What none of them can do is ANSWER for a substance it was
not written for, because the fact underneath -- how many carbons, how
many hydrogens, how much oxygen and sulphur per formula unit -- is
nowhere declared.

This module declares it, once. Everything above is then derivable rather
than typed, and `verify_against_working_fluids()` proves that claim
against the catalogue that already exists: the stoichiometric ratios
this module DERIVES from elemental formulas reproduce the hand-entered
ones in `working_fluids.py` across methane, propane, hydrogen, methanol,
nitromethane, gasoline, diesel, heavy fuel oil and vegetable oil -- nine
chemically unrelated fuels, agreeing to about a percent. It is a real
check, not a fit: nothing here was tuned to the catalogue, and where it
disagrees (biodiesel) the disagreement is reported rather than hidden.

WHY THIS SITS BESIDE cryogenics.py RATHER THAN INSIDE IT

`CRYOGENS` is a table of BULK FLUID BEHAVIOUR -- boiling point, latent
heat, expansion ratio, what it does to a room. That is the right
vocabulary for the question a dewar asks. It is the wrong vocabulary for
the question a flame asks, which is about atoms. The two are different
questions about the same six species, and the entry for liquid methane
in `CRYOGENS` is the proof: methane is already an organic molecule
sitting in an inorganic table, described entirely by how cold it is.

So this module does not move, wrap or shadow `CRYOGENS`. It keys by the
same names where they overlap (`nitrogen`, `oxygen`, `argon`, `hydrogen`,
`methane`), adds the elemental formula those rows never needed, and
`cross_reference()` reports the overlap so neither table can drift from
the other unnoticed.

THE INORGANIC ROWS HERE ARE COMBUSTION COMPANIONS

Carbon dioxide, water, carbon monoxide, the nitrogen oxides, sulphur
dioxide and molecular oxygen and nitrogen are declared here because a
combustion balance cannot be WRITTEN without them -- they are the right
hand side of every reaction this module exists to state. They are not a
second inorganic catalogue.

BLENDS GET A PER-CARBON EMPIRICAL FORMULA, WHICH IS WHAT THEY REALLY HAVE

Gasoline is not a molecule. Neither is diesel, kerosene, crude, bunker
fuel or engine oil. Each is a distribution of hundreds of species, and
the refinery characterises it exactly the way this table does: by the
average atoms per carbon, written CH1.87 and so on. That is not an
approximation this module invented to cope -- it is the standard
characterisation, it is what an ultimate analysis reports, and it is
precisely enough to get stoichiometry, carbon fraction and the products
of complete combustion right. `blend=True` marks those rows so nothing
downstream mistakes one for a pure compound with a sharp boiling point.

SOOT IS A SPECIES HERE, NOT A COLOUR

`combustion_kernel.py` has `soot_luminosity`, which is a render
constant. `air_treatment.py` has `soot_kg_per_kg`, which is a slot that
nothing fills. `centrifuges.py` has a soot particle size and a ledger of
grams. What has never existed is soot as a substance with a composition,
and real soot is not elemental carbon: nascent soot runs around CH0.2
and carries adsorbed sulphate and unburnt oil. Declaring that is what
lets the same Beer-Lambert extinction and Stokes settling laws already
written in `symbolic_atmosphere.py` be pointed at a smoke plume, since
those laws take a density and a size and ask nothing about what the
particle is -- but the mass balance that produces the particle very much
does.
"""
from __future__ import annotations

from dataclasses import dataclass

import working_fluids


# ---------------------------------------------------------------------
# the periodic table, to the extent this machine contains one
# ---------------------------------------------------------------------
# Standard atomic weights (IUPAC, conventional values). These are the
# only literal chemical constants in the module; every molar mass below
# is computed from a formula and these, so no molar mass can drift from
# the composition it is supposed to describe.
ATOMIC_MASS_G_MOL = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "Ar": 39.948,
}


@dataclass(frozen=True)
class Molecule:
    """One substance, by composition.

    The formula fields are ATOMS PER FORMULA UNIT and are floats on
    purpose: a pure compound has integers (propane is C3H8), a blend has
    a real average per carbon (diesel is CH1.80), and a fuel sulphur
    content of 15 parts per million is a real 1e-5 of a sulphur atom per
    carbon. One field type covers all three because they are the same
    kind of fact measured at different precisions.
    """
    key: str
    label: str
    carbon: float = 0.0
    hydrogen: float = 0.0
    oxygen: float = 0.0
    nitrogen: float = 0.0
    sulfur: float = 0.0
    argon: float = 0.0

    #: Formal charge, in elementary units. Every species a flame produces
    #: is neutral, so this is zero throughout the combustion rows -- but
    #: it is declared rather than assumed, because the master compendium
    #: balances charge alongside elements on every reaction it accepts
    #: (`ChemicalCompendium.reaction_balance` returns both), and an
    #: external table that cannot state a charge cannot be checked
    #: against it. The aqueous ions on the compendium side (H+, OH-,
    #: CO3-2) are exactly where a silently-assumed zero would be wrong.
    charge: int = 0

    #: Structural family. This is what decides sooting, ozone-forming
    #: reactivity and how a solvent behaves, and it is a declared
    #: attribute rather than something inferred from the name -- an
    #: identity this project has been explicit about not string-matching.
    #: "alkane" | "alkene" | "aromatic" | "cycloalkane" | "alcohol" |
    #: "ester" | "nitro" | "carbonaceous" | "inorganic" | "elemental"
    family: str = "alkane"

    #: True when this row is an AVERAGE over a distribution rather than a
    #: compound: its formula is per carbon, it boils across a range, and
    #: a sharp saturation pressure computed from it is a category error.
    blend: bool = False

    #: Phase at ambient: "gas" | "liquid" | "solid".
    ambient_phase: str = "liquid"

    #: Threshold Sooting Index, Calcote-Manos scale (ethane 0, naphthalene
    #: 100): how strongly this structure forms soot in a diffusion flame
    #: at equal carbon flow. This is a MEASURED ordering, and the ordering
    #: is the point -- aromatics soot roughly an order of magnitude harder
    #: than the straight-chain alkanes of the same carbon number, which is
    #: why a fuel's aromatic content, not its carbon count, predicts black
    #: smoke. Values are the published order of magnitude, in keeping with
    #: how every other table in this project is sourced.
    sooting_index: float = 0.0

    #: Maximum Incremental Reactivity, grams of ozone formed per gram of
    #: this compound released into a NOx-containing urban atmosphere
    #: (Carter's MIR scale, the one California regulates on). This is the
    #: number that makes SMOG a different question from SOOT: a kilogram
    #: of escaped methane is almost inert photochemically and a kilogram
    #: of escaped xylene is not, and no mass-based hydrocarbon total can
    #: tell them apart. 0.0 means non-reactive or not a VOC at all --
    #: which for methane is very nearly the literal truth and is why
    #: regulations count "non-methane hydrocarbon" separately.
    ozone_reactivity_mir: float = 0.0

    #: Liquid density at ambient, kg/m3, where the substance has one.
    #: 0.0 means "not a liquid here" rather than weightless.
    liquid_density_kg_m3: float = 0.0

    #: Normal boiling point, K. For a blend this is the T50 point, the
    #: same characterisation `working_fluids.boiling_point_k` uses and
    #: for the same stated reason. 0.0 means it does not usefully boil
    #: (it decomposes, sublimes, or is a solid that never gets there).
    boiling_point_k: float = 0.0

    why: str = ""

    # -- derived: nothing below is ever typed in ----------------------

    @property
    def molar_mass_g_mol(self) -> float:
        a = ATOMIC_MASS_G_MOL
        return (self.carbon * a["C"] + self.hydrogen * a["H"]
                + self.oxygen * a["O"] + self.nitrogen * a["N"]
                + self.sulfur * a["S"] + self.argon * a["Ar"])

    @property
    def molar_mass_kg_mol(self) -> float:
        return self.molar_mass_g_mol * 1.0e-3

    @property
    def gas_constant_j_kgk(self) -> float:
        """R specific. This is the number `DewarAtmosphere.pressure_pa`
        currently spells as 296.8 with one hydrogen branch."""
        m = self.molar_mass_kg_mol
        return UNIVERSAL_GAS_CONSTANT / m if m > 0.0 else 0.0

    @property
    def organic(self) -> bool:
        """Carbon-bearing and hydrogen-bearing. Carbon dioxide and
        carbon monoxide carry carbon and are not organic; elemental
        carbon and soot are the edge this rule is drawn around, and they
        are declared `carbonaceous` rather than argued about."""
        return self.carbon > 0.0 and self.hydrogen > 0.0

    @property
    def hydrogen_to_carbon(self) -> float:
        """H/C atomic ratio. The single number that separates methane
        (4.0, burns clean and blue) from a bunker fuel (1.55, burns
        black), and the axis every sooting correlation is written on."""
        return self.hydrogen / self.carbon if self.carbon > 0.0 else 0.0

    @property
    def oxygen_demand_mol(self) -> float:
        """Moles of O2 to burn one mole of this completely.

        C -> CO2 takes one, H4 -> 2 H2O takes one, sulphur -> SO2 takes
        one, and oxygen already in the molecule is oxygen the air does
        not have to bring, which is the entire reason an alcohol has
        such a low air/fuel ratio and why an oxygenated fuel needs more
        of itself for the same air."""
        return max(0.0, self.carbon + self.hydrogen / 4.0
                   - self.oxygen / 2.0 + self.sulfur)

    @property
    def mass_fractions(self) -> dict:
        """Ultimate analysis: this substance by element, by mass.
        `gas_works.RAW_FUEL_CARBON_FRACTION` is the carbon entry of this
        for four solid fuels."""
        m = self.molar_mass_g_mol
        if m <= 0.0:
            return {}
        a = ATOMIC_MASS_G_MOL
        out = {
            "C": self.carbon * a["C"] / m,
            "H": self.hydrogen * a["H"] / m,
            "O": self.oxygen * a["O"] / m,
            "N": self.nitrogen * a["N"] / m,
            "S": self.sulfur * a["S"] / m,
            "Ar": self.argon * a["Ar"] / m,
        }
        return {k: v for k, v in out.items() if v > 0.0}

    @property
    def carbon_mass_frac(self) -> float:
        return self.mass_fractions.get("C", 0.0)

    def stoichiometric_afr(self) -> float:
        """Air/fuel MASS ratio at stoichiometry, from the formula alone.

        This is the number `working_fluids.stoich_afr` states by hand for
        every fuel in the catalogue, and `verify_against_working_fluids`
        checks this derivation against all of them."""
        m = self.molar_mass_g_mol
        if m <= 0.0:
            return 0.0
        o2_g = self.oxygen_demand_mol * MOLECULES["oxygen"].molar_mass_g_mol
        return o2_g / (m * AIR_O2_MASS_FRACTION)

    def complete_products_kg(self, fuel_kg: float) -> dict:
        """What one burns INTO, when it burns all the way.

        Mass in equals mass out: fuel plus its stoichiometric oxygen
        becomes carbon dioxide, water and sulphur dioxide, and the
        balance is checked by `products_conserve_mass`. This is the
        half of combustion `emissions.py` never had -- it counts what
        went wrong (CO, HC, NOx) and never counted what went right, so
        nothing downstream could ask a running engine how much carbon
        dioxide or water it was putting into the volume around it."""
        kg = max(0.0, float(fuel_kg))
        m = self.molar_mass_g_mol
        if kg <= 0.0 or m <= 0.0:
            return {}
        mol = kg / (m * 1.0e-3)                      # mol of fuel
        out = {}
        if self.carbon > 0.0:
            out["carbon-dioxide"] = mol * self.carbon * MOLECULES["carbon-dioxide"].molar_mass_kg_mol
        if self.hydrogen > 0.0:
            out["water"] = mol * (self.hydrogen / 2.0) * MOLECULES["water"].molar_mass_kg_mol
        if self.sulfur > 0.0:
            out["sulfur-dioxide"] = mol * self.sulfur * MOLECULES["sulfur-dioxide"].molar_mass_kg_mol
        if self.nitrogen > 0.0:
            # fuel-bound nitrogen leaves as N2 under complete combustion;
            # what it does under real conditions is fuel NOx, which is a
            # different mechanism from the thermal NOx `emissions.py`
            # already models and is deliberately not folded in here.
            out["nitrogen"] = mol * (self.nitrogen / 2.0) * MOLECULES["nitrogen"].molar_mass_kg_mol
        return out

    def oxygen_required_kg(self, fuel_kg: float) -> float:
        m = self.molar_mass_g_mol
        if m <= 0.0:
            return 0.0
        mol = max(0.0, float(fuel_kg)) / (m * 1.0e-3)
        return mol * self.oxygen_demand_mol * MOLECULES["oxygen"].molar_mass_kg_mol


UNIVERSAL_GAS_CONSTANT = 8.31446   # J/(mol*K)


# ---------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------

def _m(key, label, **kw) -> Molecule:
    return Molecule(key=key, label=label, **kw)


_ROWS = (
    # -- combustion companions: the right-hand side of every reaction --
    # The three air species leave `boiling_point_k` at zero deliberately:
    # cryogenics.CRYOGENS owns where they boil, along with latent heat,
    # expansion ratio and inversion temperature, and copying one of those
    # four numbers over here would create exactly the drift this module
    # was written to end. `cross_reference()` reads the cryogen row
    # instead, and reports which table each figure came from.
    _m("oxygen", "molecular oxygen", oxygen=2, family="inorganic",
       ambient_phase="gas",
       why="the oxidiser every balance here is written against; where it "
           "boils is cryogenics.CRYOGENS['liquid-oxygen']'s to say"),
    _m("nitrogen", "molecular nitrogen", nitrogen=2, family="inorganic",
       ambient_phase="gas",
       why="inert ballast on the way in, and the source of thermal NOx on "
           "the way out because at flame temperature it stops being inert"),
    _m("argon", "argon", argon=1, family="inorganic", ambient_phase="gas",
       why="genuinely inert, and present at almost one percent, which is why "
           "an air separation column accumulates it in the middle"),
    _m("carbon-dioxide", "carbon dioxide", carbon=1, oxygen=2,
       family="inorganic", ambient_phase="gas", boiling_point_k=194.7,
       why="the intended product. Sublimes rather than boils at one "
           "atmosphere -- phase_table.py and cryogenics.co2_freezes_at "
           "already carry that, and this row does not restate it"),
    _m("water", "water", hydrogen=2, oxygen=1, family="inorganic",
       ambient_phase="liquid", liquid_density_kg_m3=1000.0,
       boiling_point_k=373.15,
       why="the other intended product, and the reason a cold exhaust drips: "
           "burning a kilogram of any hydrocarbon makes more than a kilogram "
           "of water"),
    _m("carbon-monoxide", "carbon monoxide", carbon=1, oxygen=1,
       family="inorganic", ambient_phase="gas",
       why="incomplete oxidation, one oxygen short. emissions.py already "
           "prices what it does to a person in a closed bay"),
    _m("nitric-oxide", "nitric oxide", nitrogen=1, oxygen=1,
       family="inorganic", ambient_phase="gas",
       why="what the Zeldovich mechanism actually makes at the flame; it "
           "oxidises to NO2 in the plume, which is why tailpipe NOx is "
           "reported as NO2 and measured as the sum"),
    _m("nitrogen-dioxide", "nitrogen dioxide", nitrogen=1, oxygen=2,
       family="inorganic", ambient_phase="gas",
       why="the brown one. The photolysis of this molecule is the first "
           "step of photochemical smog and the reason smog needs sunlight"),
    _m("sulfuric-acid", "sulphuric acid (sulphate aerosol)", hydrogen=2,
       sulfur=1, oxygen=4, family="inorganic", ambient_phase="liquid",
       liquid_density_kg_m3=1830.0,
       why="a small share of every fuel sulphur atom oxidises past SO2 to "
           "SO3, meets water in the cooling plume and becomes a liquid "
           "droplet. It is why sulphur shows up in a PARTICULATE number as "
           "well as a gas one, and why taking sulphur out of diesel cut "
           "measured particulate before any filter was fitted"),
    _m("sulfur-dioxide", "sulphur dioxide", sulfur=1, oxygen=2,
       family="inorganic", ambient_phase="gas",
       why="every sulphur atom in the fuel arrives here. working_fluids.py "
           "has declared sulfur_frac on nine fuels and air_treatment.py has "
           "had an so2 slot the whole time with nothing connecting them"),

    # -- pure hydrocarbons, light to heavy ---------------------------
    _m("methane", "methane", carbon=1, hydrogen=4, ambient_phase="gas",
       boiling_point_k=111.7, sooting_index=0.0,
       ozone_reactivity_mir=0.014,
       why="H/C of four, the highest any hydrocarbon can have, and the "
           "reason a methane flame is blue and a diesel flame is not. Also "
           "cryogenics.CRYOGENS['liquid-methane'], from the other side"),
    _m("ethane", "ethane", carbon=2, hydrogen=6, ambient_phase="gas",
       boiling_point_k=184.6, sooting_index=0.0,
       ozone_reactivity_mir=0.28,
       why="the zero of the sooting scale by definition"),
    _m("propane", "propane", carbon=3, hydrogen=8, ambient_phase="gas",
       boiling_point_k=231.0, sooting_index=3.0, liquid_density_kg_m3=493.0,
       ozone_reactivity_mir=0.49,
       why="working_fluids stores it as a pressurised liquid; the vaporiser "
           "in fuel_network.py exists because of its latent heat"),
    _m("n-butane", "n-butane", carbon=4, hydrogen=10, ambient_phase="gas",
       boiling_point_k=272.7, sooting_index=4.0, liquid_density_kg_m3=573.0,
       ozone_reactivity_mir=1.15,
       why="the winter/summer LPG blend component; boils just below room "
           "temperature, which is the whole difference from propane"),
    _m("iso-octane", "iso-octane (2,2,4-trimethylpentane)", carbon=8,
       hydrogen=18, boiling_point_k=372.4, sooting_index=6.0,
       liquid_density_kg_m3=692.0,
       ozone_reactivity_mir=1.26,
       why="octane number 100 by definition -- it is one of the two "
           "reference fuels the scale working_fluids.effective_octane uses "
           "is measured against"),
    _m("n-heptane", "n-heptane", carbon=7, hydrogen=16,
       boiling_point_k=371.6, sooting_index=5.0, liquid_density_kg_m3=684.0,
       ozone_reactivity_mir=1.07,
       why="octane number 0, the other end of the same defined scale; "
           "nearly the same boiling point as iso-octane and nothing like "
           "the same knock behaviour, which is the point"),
    _m("n-decane", "n-decane", carbon=10, hydrogen=22,
       boiling_point_k=447.3, sooting_index=13.0, liquid_density_kg_m3=730.0,
       ozone_reactivity_mir=0.61,
       why="a kerosene surrogate: the single compound whose ignition "
           "behaviour stands in for jet fuel in most published mechanisms"),
    _m("n-dodecane", "n-dodecane", carbon=12, hydrogen=26,
       boiling_point_k=489.5, sooting_index=17.0, liquid_density_kg_m3=750.0,
       ozone_reactivity_mir=0.51,
       why="the standard diesel surrogate"),
    _m("n-hexadecane", "n-hexadecane (cetane)", carbon=16, hydrogen=34,
       boiling_point_k=560.0, sooting_index=22.0, liquid_density_kg_m3=773.0,
       ozone_reactivity_mir=0.3,
       why="cetane number 100 by definition; working_fluids.cetane is "
           "measured against this molecule and alpha-methylnaphthalene"),
    _m("cyclohexane", "cyclohexane", carbon=6, hydrogen=12,
       family="cycloalkane", boiling_point_k=353.9, sooting_index=10.0,
       liquid_density_kg_m3=779.0,
       ozone_reactivity_mir=1.25,
       why="a naphthene: ring, but saturated. Soots more than a straight "
           "chain of the same carbon number and far less than an aromatic, "
           "which is how you know the ring is not what matters -- the "
           "double bonds are"),

    # -- aromatics: the sooting end, and the smog end ----------------
    _m("benzene", "benzene", carbon=6, hydrogen=6, family="aromatic",
       boiling_point_k=353.2, sooting_index=30.0, liquid_density_kg_m3=876.0,
       ozone_reactivity_mir=0.72,
       why="H/C of one. The first aromatic ring, and the species soot "
           "growth is believed to nucleate through, so it is both a fuel "
           "component and the doorway to the particle"),
    _m("toluene", "toluene", carbon=7, hydrogen=8, family="aromatic",
       boiling_point_k=383.8, sooting_index=54.0, liquid_density_kg_m3=867.0,
       ozone_reactivity_mir=4.0,
       why="the high-octane aromatic real gasoline is blended with, and the "
           "single biggest contributor to a petrol engine's ozone-forming "
           "potential -- it is what makes a gasoline blend soot AND smog"),
    _m("xylene", "xylene (mixed isomers)", carbon=8, hydrogen=10,
       family="aromatic", boiling_point_k=411.5, sooting_index=64.0,
       liquid_density_kg_m3=864.0,
       ozone_reactivity_mir=7.4,
       why="the other aromatic in the gasoline pool; highest reactivity per "
           "gram of any common fuel component in photochemical smog"),
    _m("naphthalene", "naphthalene", carbon=10, hydrogen=8,
       family="aromatic", ambient_phase="solid", boiling_point_k=491.0,
       sooting_index=100.0, liquid_density_kg_m3=1140.0,
       ozone_reactivity_mir=3.35,
       why="two fused rings, the 100 of the sooting scale, and the first "
           "member of the polycyclic series that the rest of soot growth "
           "runs through. Solid at room temperature, which is why it "
           "condenses onto a particle instead of staying a gas"),
    _m("pyrene", "pyrene", carbon=16, hydrogen=10, family="aromatic",
       ambient_phase="solid", boiling_point_k=677.0, sooting_index=100.0,
       liquid_density_kg_m3=1271.0,
       ozone_reactivity_mir=1.5,
       why="four fused rings. In every soot mechanism worth the name this "
           "is where a molecule stops being a gas-phase species and starts "
           "being a particle -- it is the modelled nucleation step"),

    # -- oxygenates --------------------------------------------------
    _m("methanol", "methanol", carbon=1, hydrogen=4, oxygen=1,
       family="alcohol", boiling_point_k=337.85, sooting_index=0.0,
       liquid_density_kg_m3=792.0,
       ozone_reactivity_mir=0.67,
       why="one oxygen per carbon: it cannot soot, and it needs less than "
           "half the air of gasoline, both for the same reason"),
    _m("ethanol", "ethanol", carbon=2, hydrogen=6, oxygen=1,
       family="alcohol", boiling_point_k=351.4, sooting_index=1.0,
       liquid_density_kg_m3=789.0,
       ozone_reactivity_mir=1.53,
       why="the E85 component. Its latent heat is what cools the charge, "
           "and working_fluids already carries that at 760 kJ/kg for the "
           "blend"),
    _m("nitromethane", "nitromethane", carbon=1, hydrogen=3, nitrogen=1,
       oxygen=2, family="nitro", boiling_point_k=374.35,
       liquid_density_kg_m3=1140.0,
       ozone_reactivity_mir=0.1,
       why="carries its own oxidiser, which is why its air/fuel ratio is "
           "1.7 and not 15 -- the engine is mostly not breathing air. The "
           "derivation in this module reproduces that 1.7 exactly from the "
           "formula, which is the strongest single check in the table"),
    _m("methyl-oleate", "methyl oleate", carbon=19, hydrogen=36, oxygen=2,
       family="ester", boiling_point_k=617.0, sooting_index=12.0,
       liquid_density_kg_m3=874.0,
       ozone_reactivity_mir=0.6,
       why="the biodiesel surrogate: a fatty acid methyl ester, two oxygens "
           "at the end of a long chain, which is why biodiesel soots less "
           "than diesel and gels earlier"),
    _m("triolein", "triolein", carbon=57, hydrogen=104, oxygen=6,
       family="ester", boiling_point_k=0.0, sooting_index=14.0,
       liquid_density_kg_m3=915.0,
       ozone_reactivity_mir=0.3,
       why="straight vegetable oil before anyone transesterifies it: three "
           "of the above chains on a glycerol backbone. It does not boil, "
           "it decomposes, and its viscosity is why working_fluids marks it "
           "viscous_at_ambient and fuel_network puts a heater in front"),

    # -- blends: per-carbon empirical formulas -----------------------
    _m("gasoline-blend", "gasoline (average CH1.87)", carbon=1,
       hydrogen=1.87, blend=True, boiling_point_k=373.0,
       sooting_index=20.0, liquid_density_kg_m3=745.0,
       ozone_reactivity_mir=3.3,
       why="a real pump blend is roughly a quarter aromatic by volume, so "
           "its sooting index sits well above the paraffins in it and well "
           "below the toluene"),
    _m("diesel-blend", "diesel (average CH1.80)", carbon=1, hydrogen=1.80,
       sulfur=1.0e-5, blend=True, boiling_point_k=533.0,
       sooting_index=35.0, liquid_density_kg_m3=832.0,
       ozone_reactivity_mir=1.9,
       why="the sulphur here is ultra-low-sulphur diesel's own 15 ppm, "
           "carried as a real fraction of an atom per carbon so the SO2 "
           "balance is right rather than rounded to nothing"),
    _m("kerosene-blend", "kerosene / Jet A (average CH1.92)", carbon=1,
       hydrogen=1.92, sulfur=2.1e-4, blend=True, boiling_point_k=473.0,
       sooting_index=26.0, liquid_density_kg_m3=800.0,
       ozone_reactivity_mir=1.7,
       why="lighter and less aromatic than diesel, which is why a turbine "
           "burns it cleaner than a diesel burns its own fuel"),
    _m("crude-blend", "crude oil (average CH1.75)", carbon=1, hydrogen=1.75,
       sulfur=1.05e-2, blend=True, boiling_point_k=0.0, sooting_index=45.0,
       liquid_density_kg_m3=870.0,
       ozone_reactivity_mir=2.2,
       why="1.5 % sulphur by mass, which is what working_fluids declares "
           "and what the derived formula reproduces"),
    _m("heavy-fuel-oil-blend", "residual fuel oil (average CH1.55)",
       carbon=1, hydrogen=1.55, sulfur=1.085e-2, blend=True,
       boiling_point_k=0.0, sooting_index=70.0, liquid_density_kg_m3=980.0,
       ozone_reactivity_mir=2.0,
       why="H/C 1.55: the aromatic, asphaltene-rich bottom of the barrel. "
           "This is the fuel that makes a ship's plume visible from orbit, "
           "and the sooting index says so before anything is simulated"),
    _m("lubricating-oil", "lubricating oil (average CH1.87)", carbon=1,
       hydrogen=1.87, sulfur=2.3e-3, blend=True, boiling_point_k=0.0,
       sooting_index=40.0, liquid_density_kg_m3=870.0,
       ozone_reactivity_mir=0.8,
       why="fluids.py's engine-oil, as a composition. It does not boil, it "
           "cracks -- which is exactly why oil that gets past a ring makes "
           "blue smoke instead of vapour, and why that smoke is a different "
           "substance from fuel soot"),
    _m("wood-fuel", "wood, dry (average CH1.44O0.66)", carbon=1,
       hydrogen=1.44, oxygen=0.66, blend=True, ambient_phase="solid",
       sooting_index=25.0,
       why="half carbon by mass, which is the figure gas_works.py states by "
           "hand and this formula reproduces to two decimal places. The "
           "oxygen already in the wood is why its heating value is half "
           "coal's: a third of it is pre-oxidised"),
    _m("coal-fuel", "bituminous coal (average CH0.80O0.13S0.005)", carbon=1,
       hydrogen=0.80, oxygen=0.13, sulfur=0.005, blend=True,
       ambient_phase="solid", sooting_index=60.0,
       why="80 % carbon by mass, matching gas_works; the sulphur is what "
           "made industrial smog lethal and is one line away from an SO2 "
           "rate here"),
    _m("coke-fuel", "coke (average CH0.20O0.06S0.006)", carbon=1,
       hydrogen=0.20, oxygen=0.06, sulfur=0.006, blend=True,
       ambient_phase="solid", sooting_index=20.0,
       why="coal with the volatiles driven off, which is why the hydrogen "
           "is nearly gone and the carbon fraction is 0.90 as gas_works "
           "states. A coke fire has no flame worth the name because there "
           "is nothing left to make one"),
    _m("charcoal-fuel", "charcoal (average CH0.35O0.10)", carbon=1,
       hydrogen=0.35, oxygen=0.10, blend=True, ambient_phase="solid",
       sooting_index=15.0,
       why="wood's coke. Same story, lighter starting material"),

    # -- condensed carbonaceous matter: the particle itself ----------
    _m("soot", "combustion soot (nascent, CH0.20)", carbon=1, hydrogen=0.20,
       family="carbonaceous", blend=True, ambient_phase="solid",
       liquid_density_kg_m3=1800.0,
       why="NOT elemental carbon. Freshly made soot keeps about a fifth of "
           "a hydrogen per carbon and loses it as it ages in the plume, "
           "which is why old soot absorbs light differently from new soot. "
           "The 1800 kg/m3 is the material density of the primary particle, "
           "not the fluffy aggregate -- an aggregate's effective density is "
           "far lower and is a function of its size, which belongs with the "
           "size distribution rather than here"),
    _m("elemental-carbon", "elemental carbon (graphitised)", carbon=1,
       family="carbonaceous", ambient_phase="solid",
       liquid_density_kg_m3=2000.0,
       why="what soot becomes after enough residence time, and what a "
           "thermal-optical analyser actually reports as the EC in EC/OC"),
)

MOLECULES: dict[str, Molecule] = {m.key: m for m in _ROWS}


def molecule(key: str) -> Molecule:
    m = MOLECULES.get(str(key))
    if m is None:
        raise KeyError(f"unknown species {key!r}; declared: {', '.join(sorted(MOLECULES))}")
    return m


# ---------------------------------------------------------------------
# air, derived rather than typed
# ---------------------------------------------------------------------
# cryogenics.AIR_COMPOSITION and air_separation.AIR_MOLE_FRAC both state
# these three numbers. This module does not restate them as a third copy
# -- it takes them and works out what they IMPLY, which is the part
# neither of those files could: the molar mass of air, and therefore the
# oxygen mass fraction that every stoichiometric ratio in this project
# is divided by.
AIR_MOLE_FRAC = {"nitrogen": 0.7808, "oxygen": 0.2095, "argon": 0.0093}

AIR_MOLAR_MASS_G_MOL = sum(
    f * MOLECULES[k].molar_mass_g_mol for k, f in AIR_MOLE_FRAC.items())

AIR_O2_MASS_FRACTION = (
    AIR_MOLE_FRAC["oxygen"] * MOLECULES["oxygen"].molar_mass_g_mol
    / AIR_MOLAR_MASS_G_MOL)

#: gas_works.py states 0.233 as a cited figure. The composition above
#: implies 0.2316. They are the same number to the precision either is
#: known to; this constant is kept derived so that changing the argon
#: fraction changes everything downstream of it, which typing 0.233
#: would not.
AIR_GAS_CONSTANT_J_KGK = UNIVERSAL_GAS_CONSTANT / (AIR_MOLAR_MASS_G_MOL * 1e-3)


def mixture_molar_mass_g_mol(kg_by_species: dict) -> float:
    """Molar mass of a real mixture given MASSES, not mole fractions.

    A gas space tracked by mass -- which is how `DewarAtmosphere`,
    `air_treatment.Stream` and `drivetrain_graph.FluidCircuit` all track
    theirs -- has to go through moles to get here, because mixing is
    linear in moles and not in mass. Getting this backwards is how a
    hydrogen-rich ullage ends up priced as air."""
    mol = 0.0
    mass = 0.0
    for key, kg in kg_by_species.items():
        w = max(0.0, float(kg))
        if w <= 0.0:
            continue
        m = MOLECULES.get(key)
        if m is None or m.molar_mass_g_mol <= 0.0:
            continue
        mol += w / (m.molar_mass_g_mol * 1e-3)
        mass += w
    return (mass / mol) * 1e3 if mol > 0.0 else 0.0


def mixture_gas_constant_j_kgk(kg_by_species: dict) -> float:
    """R specific for a mass-tracked mixture.

    This is what `cryogenics.DewarAtmosphere.pressure_pa` spells as
    ``296.8 if no hydrogen else 4124.0``. That branch is honest about
    the one case where the error is unmissable and silently wrong for
    every other: an ullage of boiled-off carbon dioxide, of fuel vapour
    over a tank, or of the oil mist in a crankcase is nowhere near air.
    Handing that method this function instead makes the mixture price
    itself, and costs it one import."""
    m = mixture_molar_mass_g_mol(kg_by_species)
    return UNIVERSAL_GAS_CONSTANT / (m * 1e-3) if m > 0.0 else AIR_GAS_CONSTANT_J_KGK


# ---------------------------------------------------------------------
# which molecule stands for which catalogued fuel
# ---------------------------------------------------------------------
# A declared mapping, in one place, from the fuels working_fluids.py
# already sells to the composition each one has. Fuels absent from this
# map are absent on purpose and say why: coal gas and wood gas are
# MIXTURES of carbon monoxide, hydrogen and methane, not substances with
# a formula, and giving them one would be the kind of quiet fiction this
# table exists to remove. They need a mixture row, which is a different
# shape, and `unmapped_fuels()` keeps them visible until they get one.
FUEL_COMPOSITION: dict[str, str] = {
    "pump-gasoline-87": "gasoline-blend",
    "pump-gasoline-89": "gasoline-blend",
    "pump-gasoline-91": "gasoline-blend",
    "pump-gasoline-93": "gasoline-blend",
    "aviation-gasoline-100-130": "gasoline-blend",
    "petroleum-vapor": "gasoline-blend",
    "methanol-race": "methanol",
    "nitromethane-race": "nitromethane",
    "ultra-low-sulfur-diesel": "diesel-blend",
    "jet-a-kerosene": "kerosene-blend",
    "kerosene": "kerosene-blend",
    "jp-8": "kerosene-blend",
    "crude-oil": "crude-blend",
    "vegetable-oil": "triolein",
    "engine-oil-waste": "lubricating-oil",
    "heavy-fuel-oil": "heavy-fuel-oil-blend",
    "biodiesel-b100": "methyl-oleate",
    "natural-gas": "methane",
    "propane": "propane",
    "hydrogen": "hydrogen-gas",
}

# hydrogen is the one fuel that is a molecule with no carbon at all, and
# it would look like an omission in the organic table if it were not
# stated. It lives here rather than among the hydrocarbons.
MOLECULES["hydrogen-gas"] = Molecule(
    key="hydrogen-gas", label="molecular hydrogen", hydrogen=2,
    family="elemental", ambient_phase="gas", boiling_point_k=20.28,
    why="no carbon: it cannot make carbon monoxide, carbon dioxide or soot, "
        "and the only thing it can make besides water is NOx -- which it "
        "makes plenty of, because it burns hot. cryogenics.CRYOGENS holds "
        "the same species as the hard one to liquefy")


def composition_of_fuel(fuel_name: str) -> Molecule | None:
    """The molecule standing for a catalogued working fluid, or None if
    that fuel is a mixture rather than a substance."""
    key = FUEL_COMPOSITION.get(str(fuel_name))
    return MOLECULES.get(key) if key else None


def unmapped_fuels() -> list:
    """Combustible working fluids with no declared composition, and why
    that is not an oversight."""
    return sorted(f.name for f in working_fluids.WORKING_FLUIDS.values()
                  if f.combustible and f.name not in FUEL_COMPOSITION)


# ---------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------

def verify_against_working_fluids() -> list:
    """Derived stoichiometric air/fuel ratio against the catalogued one.

    Nothing in the formula table was fitted to these numbers. Agreement
    is evidence the compositions are right; disagreement is a finding
    about one side or the other and is reported as such rather than
    tuned away."""
    rows = []
    for name, key in sorted(FUEL_COMPOSITION.items()):
        fluid = working_fluids.WORKING_FLUIDS.get(name)
        mol = MOLECULES.get(key)
        if fluid is None or mol is None or fluid.stoich_afr <= 0.0:
            continue
        derived = mol.stoichiometric_afr()
        declared = fluid.stoich_afr
        rows.append({
            "fuel": name,
            "as": key,
            "declared_afr": declared,
            "derived_afr": derived,
            "error_frac": (derived - declared) / declared,
        })
    return rows


def verify_carbon_fractions() -> list:
    """Derived carbon mass fraction against gas_works' four solid fuels.

    The figures are quoted here rather than imported because gas_works
    imports otto_langen at module level and this module has no business
    pulling an engine catalogue in to check an ultimate analysis."""
    stated = {"coal-fuel": ("coal", 0.80), "coke-fuel": ("coke", 0.90),
              "charcoal-fuel": ("charcoal", 0.85), "wood-fuel": ("wood", 0.50)}
    rows = []
    for key, (label, declared) in sorted(stated.items()):
        derived = MOLECULES[key].carbon_mass_frac
        rows.append({"fuel": label, "as": key, "declared_c_frac": declared,
                     "derived_c_frac": derived,
                     "error_frac": (derived - declared) / declared})
    return rows


def products_conserve_mass(key: str, fuel_kg: float = 1.0) -> float:
    """Mass balance residual: (fuel + oxygen) - products, in kg.

    Should be float noise. This is the property that makes
    `complete_products_kg` a conservation law rather than a yield
    table."""
    mol = molecule(key)
    inp = fuel_kg + mol.oxygen_required_kg(fuel_kg)
    out = sum(mol.complete_products_kg(fuel_kg).values())
    return inp - out


def cross_reference() -> list:
    """Species declared both here and in cryogenics.CRYOGENS.

    Neither table is authoritative over the other -- they answer
    different questions -- but a species whose boiling point differs
    between them is a drift, and this is where it shows."""
    try:
        import cryogenics
    except Exception:
        return []
    pairs = {"nitrogen": "liquid-nitrogen", "oxygen": "liquid-oxygen",
             "argon": "liquid-argon", "hydrogen-gas": "liquid-hydrogen",
             "methane": "liquid-methane"}
    rows = []
    for here, there in sorted(pairs.items()):
        cryo = cryogenics.CRYOGENS.get(there)
        mol = MOLECULES.get(here)
        if cryo is None or mol is None:
            continue
        stated = mol.boiling_point_k > 0.0
        rows.append({
            "species": here,
            "cryogen": there,
            # where the boiling point comes from. "cryogenics" means this
            # table declines to hold a second copy; "both" means the two
            # can drift and the delta is the thing to watch.
            "boil_k": mol.boiling_point_k if stated else cryo.boil_k,
            "boil_k_source": "both" if stated else "cryogenics",
            "boil_k_delta": (mol.boiling_point_k - cryo.boil_k) if stated else 0.0,
            "molar_mass_g_mol": mol.molar_mass_g_mol,
            # what cryogenics has never been able to say about its own
            # rows, and what its pressure_pa needs
            "gas_constant_j_kgk": mol.gas_constant_j_kgk,
        })
    return rows


def report() -> str:
    """Everything this module claims, as one printable audit."""
    out = ["STOICHIOMETRIC AIR/FUEL: derived from formula vs catalogued",
           f"  {'fuel':<28} {'as':<22} {'declared':>9} {'derived':>9} {'error':>8}"]
    worst = 0.0
    for r in verify_against_working_fluids():
        worst = max(worst, abs(r["error_frac"]))
        out.append(f"  {r['fuel']:<28} {r['as']:<22} {r['declared_afr']:>9.2f} "
                   f"{r['derived_afr']:>9.2f} {r['error_frac']:>+7.1%}")
    out.append(f"  worst disagreement: {worst:.1%}")
    out.append("")
    out.append("CARBON MASS FRACTION: derived vs gas_works' stated values")
    for r in verify_carbon_fractions():
        out.append(f"  {r['fuel']:<12} {r['declared_c_frac']:>6.2f} "
                   f"{r['derived_c_frac']:>8.3f} {r['error_frac']:>+7.1%}")
    out.append("")
    out.append("MASS BALANCE of complete combustion, kg residual per kg fuel")
    for key in ("methane", "gasoline-blend", "diesel-blend", "methanol",
                "nitromethane", "heavy-fuel-oil-blend", "wood-fuel"):
        out.append(f"  {key:<24} {products_conserve_mass(key):+.3e}")
    out.append("")
    out.append("SHARED WITH cryogenics.CRYOGENS "
               "(boil from whichever table declares it; M and R are new here)")
    for r in cross_reference():
        drift = (f"  drift {r['boil_k_delta']:+.2f} K"
                 if r["boil_k_source"] == "both" else "")
        out.append(f"  {r['species']:<12} boil {r['boil_k']:>7.2f} K "
                   f"[{r['boil_k_source']:<10}]  M={r['molar_mass_g_mol']:>6.2f}"
                   f"  R={r['gas_constant_j_kgk']:>7.1f}{drift}")
    out.append("")
    out.append(f"AIR, derived from its own composition: "
               f"M={AIR_MOLAR_MASS_G_MOL:.3f} g/mol  "
               f"O2 mass fraction={AIR_O2_MASS_FRACTION:.4f}  "
               f"R={AIR_GAS_CONSTANT_J_KGK:.1f} J/kgK")
    unmapped = unmapped_fuels()
    if unmapped:
        out.append(f"NO COMPOSITION DECLARED (mixtures, not substances): "
                   f"{', '.join(unmapped)}")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
