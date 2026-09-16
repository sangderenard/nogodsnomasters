"""Which transitions each material has, where, and what goes wrong.

The physics of phase change is already in this project three times over,
each solving its own case correctly and none of them able to answer for
a material they were not written for:

    thermal_storage   freezing_point_k (with real colligative depression
                      from salt and glycol), freeze_kg, HeatBalance's
                      melt/freeze rates, LiquidTote bursting
    cryogenics        CO2_SUBLIMATION_K, the triple point, co2_freezes_at,
                      water frosting onto an 80 K fitting
    surface_wetting   boiling and diffusion-limited drying

What has never existed is the TABLE: one declared place that says, for a
given material, which transitions it can undergo at all, at what
temperature, with what latent heat, and -- the part that actually
matters -- what each transition DOES beyond changing state.

This is a table, not an engine. It does not integrate anything. The
solvers above keep their jobs; they stop each carrying their own private
copy of where water freezes.

WHY THE COMPLICATION FIELD IS THE POINT

A transition temperature alone is nearly useless for the materials this
machine actually contains, because the interesting ones do not undergo
clean transitions:

  WATER expands 9.09% on freezing, which is the only reason freezing
  matters mechanically -- it splits whatever holds it. thermal_storage
  already bursts totes on exactly this.

  ENGINE OIL has no freezing point. It has a POUR POINT: the wax gels
  and it stops flowing while remaining, thermodynamically, a liquid. An
  engine below its oil's pour point will not get oil to the bearings
  even though nothing is frozen, and a pump pulling against gelled oil
  blows the filter bypass open and feeds unfiltered oil to the mains.
  Modelling this as "freezing at some temperature" gets the behaviour
  exactly backwards -- nothing solidifies and the engine dies anyway.

  DIESEL is worse, because it has TWO thresholds above anything anyone
  would call freezing. At the CLOUD POINT wax crystals appear and the
  fuel goes hazy. At the COLD FILTER PLUGGING POINT those crystals mat
  across the filter and starve the engine. Both are well above the
  temperature at which diesel could be said to solidify, which is why
  winter diesel is a different blend rather than a heated tank.

  CO2 has no liquid phase at atmospheric pressure at all. Below its
  triple point it goes straight from gas to solid, which is what dry ice
  is and why it "melts" into nothing. cryogenics.co2_freezes_at already
  knows this; the table records that the liquid row simply does not
  exist below 5.18 bar.

  COOLANT's freezing point is a function of its glycol fraction, not a
  constant -- a real colligative property, already solved in
  thermal_storage.freezing_point_k, and referenced here rather than
  re-derived.

LATENT HEAT HAS A SIGN, AND IT MATTERS. A transition absorbs or releases
energy, and that energy goes into or out of whatever the material is
touching. Condensation HEATS the surface it forms on; evaporation COOLS
it. That is why a boiling coolant leak scalds and an evaporating fuel
spill frosts. The sign convention here is: positive latent heat is
absorbed BY the material (melting, boiling, subliming), negative is
released INTO its surroundings (freezing, condensing, depositing).
"""
from __future__ import annotations

from dataclasses import dataclass

# --- the six transitions, named once ----------------------------------
MELT = "melt"              # solid -> liquid, absorbs
FREEZE = "freeze"          # liquid -> solid, releases
VAPORISE = "vaporise"      # liquid -> gas, absorbs
CONDENSE = "condense"      # gas -> liquid, releases
SUBLIME = "sublime"        # solid -> gas, absorbs
DEPOSIT = "deposit"        # gas -> solid, releases (frost)

#: The reverse of each, since every pair shares a temperature and a
#: magnitude of latent heat and differs only in sign.
REVERSE = {MELT: FREEZE, FREEZE: MELT, VAPORISE: CONDENSE,
           CONDENSE: VAPORISE, SUBLIME: DEPOSIT, DEPOSIT: SUBLIME}


@dataclass(frozen=True)
class Transition:
    """One material changing state, and what that costs or causes.

    `temp_k` of 0.0 means THIS TRANSITION DOES NOT HAPPEN for this
    material under conditions this machine sees -- engine oil does not
    boil, it cracks. A consumer must read 0.0 as "never", not as a very
    cold transition.

    `latent_j_per_kg` is signed: positive absorbs heat from the
    surroundings, negative releases it into them.

    `volume_change_frac` is signed the same way and declared FOR THIS
    DIRECTION. Water's famous 9.09% belongs to FREEZING, not melting --
    ice is the bigger phase, so going to it expands and coming back
    contracts. Declaring it on the wrong one and then testing for a
    negative reads correctly and means the opposite, which is exactly
    what happened here once."""
    kind: str
    temp_k: float
    latent_j_per_kg: float
    complication: str = ""
    volume_change_frac: float = 0.0     # +expands, -contracts, on this transition


@dataclass(frozen=True)
class MaterialPhases:
    """Every transition one material has, with its complications."""
    material: str
    transitions: tuple
    note: str = ""

    def get(self, kind: str) -> Transition | None:
        return next((t for t in self.transitions if t.kind == kind), None)

    def happens(self, kind: str) -> bool:
        t = self.get(kind)
        return t is not None and t.temp_k > 0.0

    def state_at(self, temp_k: float) -> str:
        """solid | liquid | gas at this temperature, one atmosphere.

        Reads the table rather than deciding anything: whichever
        transitions exist bound the ranges."""
        t = float(temp_k)
        sub = self.get(SUBLIME)
        if sub is not None and sub.temp_k > 0.0:
            # no liquid phase at all at this pressure
            return "solid" if t < sub.temp_k else "gas"
        melt = self.get(MELT)
        boil = self.get(VAPORISE)
        if melt is not None and melt.temp_k > 0.0 and t < melt.temp_k:
            return "solid"
        if boil is not None and boil.temp_k > 0.0 and t >= boil.temp_k:
            return "gas"
        return "liquid"


def _pair(kind: str, temp_k: float, latent: float, complication: str = "",
          volume_change: float = 0.0) -> tuple:
    """A transition and its exact reverse, so the two can never drift.

    They share a temperature and a magnitude; the reverse flips the sign
    of both the latent heat and the volume change, because it is the
    same event run backwards."""
    fwd = Transition(kind, temp_k, latent, complication, volume_change)
    rev = Transition(REVERSE[kind], temp_k, -latent, complication, -volume_change)
    return (fwd, rev)


PHASE_TABLE: dict = {}


def _add(m: MaterialPhases) -> None:
    PHASE_TABLE[m.material] = m


_add(MaterialPhases(
    material="water",
    transitions=(
        *_pair(FREEZE, 273.15, -333_550.0,
               "ICE EXPANDS 9.09% ON FREEZING and does it with enough force to "
               "split cast iron. This is the only reason freezing matters "
               "mechanically -- see thermal_storage.LiquidTote, which already "
               "bursts containers on exactly this, and VesselClass.freeze_safe_l "
               "for how much a vessel may hold and still survive going solid.",
               volume_change=0.0909),   # on FREEZING; melting is the -9.09% reverse
        *_pair(VAPORISE, 373.15, 2_256_000.0,
               "A very large latent heat, which is why a coolant leak scalds far "
               "past what its temperature alone suggests: condensing on skin "
               "releases every joule of it back."),
    ),
    note="Freezing point is NOT fixed: thermal_storage.freezing_point_k depresses "
         "it with salt or glycol, a real colligative effect. Use that function "
         "rather than the 273.15 here whenever the water has anything in it."))

_add(MaterialPhases(
    material="coolant",
    transitions=(
        *_pair(FREEZE, 236.15, -300_000.0,
               "A 50/50 glycol mix, so this is already depressed about 37 K from "
               "water. The real number depends on the mix and belongs to "
               "thermal_storage.freezing_point_k(glycol_frac=...) -- this row is "
               "the common case, not a constant to prefer over that function. "
               "Glycol also SUPPRESSES the expansion that makes freezing "
               "dangerous, which is half of why it is used.",
               volume_change=0.02),     # on FREEZING, and much smaller than water's
        *_pair(VAPORISE, 380.15, 2_000_000.0,
               "Above water's, which is the other half of why glycol is used: it "
               "raises the boiling point as well as lowering the freezing point. "
               "A pressurised system raises it further still."),
    )))

_add(MaterialPhases(
    material="engine-oil",
    transitions=(
        Transition(MELT, 0.0, 0.0,
                   "NO MELTING POINT. Oil has a POUR POINT instead, around 245 K "
                   "for a typical multigrade: the wax gels and it stops FLOWING "
                   "while still being a liquid. An engine below it will not get "
                   "oil to the bearings though nothing has frozen, and the pump "
                   "pulling against gelled oil opens the filter bypass and feeds "
                   "the mains unfiltered. Treating this as freezing gets the "
                   "behaviour backwards -- see POUR_POINT_K."),
        Transition(VAPORISE, 0.0, 0.0,
                   "DOES NOT BOIL. It CRACKS -- thermally decomposes into lighter "
                   "fractions and carbon somewhere past 600 K, which is a chemical "
                   "change and not a phase change. This is why oil never dries off "
                   "a surface: surface_wetting reads a 0.0 boiling point as "
                   "non-volatile, which is correct. An oil leak stays a mess."),
    ),
    note="The one material here whose important cold behaviour is not a phase "
         "transition at all."))

_add(MaterialPhases(
    material="diesel",
    transitions=(
        Transition(MELT, 0.0, 0.0,
                   "TWO THRESHOLDS WELL ABOVE ANY FREEZING POINT, and both stop "
                   "the engine long before the fuel solidifies. CLOUD POINT "
                   "(~267 K for summer diesel): wax crystals appear, fuel goes "
                   "hazy, nothing has happened yet. COLD FILTER PLUGGING POINT "
                   "(~264 K): those crystals mat across the filter and the engine "
                   "starves. This is why winter diesel is a different blend rather "
                   "than a heated tank. See CLOUD_POINT_K / CFPP_K."),
        *_pair(VAPORISE, 533.0, 250_000.0,
               "The T50 point of a distillation range, not a sharp boil -- diesel "
               "is a blend. Far above gasoline's, which is why the same hot "
               "manifold flashes off petrol and leaves diesel sitting there."),
    )))

_add(MaterialPhases(
    material="gasoline",
    transitions=(
        Transition(MELT, 0.0, 0.0,
                   "Does not freeze at any temperature this machine sees; the "
                   "lightest fractions are still liquid past 210 K."),
        *_pair(VAPORISE, 373.0, 350_000.0,
               "T50 of a wide distillation range -- the light ends are already "
               "evaporating at 310 K, which is why a spill smells immediately and "
               "why vapour, not liquid, is what actually ignites."),
    )))

_add(MaterialPhases(
    material="carbon-dioxide",
    transitions=(
        *_pair(SUBLIME, 194.65, 571_000.0,
               "NO LIQUID PHASE AT ATMOSPHERIC PRESSURE AT ALL. Below the triple "
               "point (518 kPa) CO2 goes gas straight to solid and back, which is "
               "what dry ice is and why it leaves no puddle. Above 5.18 bar a "
               "liquid row exists and this one does not apply -- "
               "cryogenics.co2_freezes_at already switches on exactly that "
               "pressure, and should be used instead of this row whenever the "
               "pressure is known."),
    ),
    note="The clearest case for why a table needs a `happens` test rather than a "
         "temperature comparison: asking this material for its melting point is a "
         "question with no answer."))

_add(MaterialPhases(
    material="nitrogen",
    transitions=(
        *_pair(MELT, 63.15, 25_700.0),
        *_pair(VAPORISE, 77.36, 199_000.0,
               "The cold everything in the cryogenic plant is measured against. "
               "Cold enough that atmospheric water and CO2 both DEPOSIT directly "
               "onto any fitting it touches -- see cryogenics.CryogenicVessel."
               "ice_vent, which is that icing happening to a real valve."),
    )))

_add(MaterialPhases(
    material="oxygen",
    transitions=(
        *_pair(MELT, 54.36, 13_900.0),
        *_pair(VAPORISE, 90.19, 213_000.0,
               "BOILS 12.8 K WARMER THAN NITROGEN, which is the entire basis of "
               "air separation: distil liquid air and the nitrogen leaves first. "
               "It is also why a boiling-off cryogenic air spill becomes "
               "OXYGEN-ENRICHED rather than inert as it ages."),
    )))

# ---------------------------------------------------------------------
# METALS, whose melting is a TRANSFORMATION INTO A FLUID
#
# Every row above describes something that was already a fluid changing
# state. These describe something that was a PART. A melted casting does
# not stop being matter: it pours, wets, drips and sprays, and every one
# of those paths already exists for oil and fuel. fluids.py carries the
# resulting liquid (molten-aluminium and friends, with real densities,
# viscosities and surface tensions at pouring temperature); this carries
# the transition into it.
#
# AND THEY EMIT. That is the one genuinely new thing about a molten
# metal compared with every other fluid in the registry: it is far
# hotter than anything it touches, so it radiates in its own right.
# fluids.Fluid.emission_k is that temperature, and a renderer already
# doing blackbody for exhaust and flame needs nothing else.
# ---------------------------------------------------------------------

_add(MaterialPhases(
    material="aluminium",
    transitions=(
        *_pair(MELT, 933.47, 397_000.0,
               "Becomes fluids.BY_KEY['molten-aluminium'] -- a real liquid at "
               "2380 kg/m3 with a surface tension of 0.87 N/m, thirty times "
               "water's. That is why a spill beads into fat blobs instead of "
               "spreading: surface_wetting's capillary length goes with "
               "sqrt(sigma/rho.g) and lands near 6 mm.",
               volume_change=0.065),    # MELTING expands; the paired FREEZE contracts -- shrinkage porosity
        Transition(VAPORISE, 2743.0, 10_900_000.0,
                   "Reachable only in an arc or a thermite reaction, not by "
                   "anything an engine does to itself."),
    ),
    note="Contracts about 6.5% on solidifying, which is why castings need "
         "risers and why a cooled repair pulls away from the parent metal. "
         "NOTE THE SIGN: liquid metal is LESS dense than solid, so melting "
         "expands and freezing contracts -- the opposite of water, which is "
         "the anomaly here and not the rule. Declared on MELT and paired, so "
         "the two directions cannot disagree."))

_add(MaterialPhases(
    material="magnesium",
    transitions=(
        *_pair(MELT, 923.0, 349_000.0,
               "ALREADY ABOVE ITS OWN IGNITION TEMPERATURE WHEN IT MELTS. "
               "Magnesium ignites at about the same 923 K it melts at, so a "
               "magnesium pour in air is not a hot liquid that might catch -- "
               "it IS a fire, burning at 3400 K. fluids marks the molten row "
               "flammable with exactly this autoignition temperature. See "
               "spectacle.ScrapePlate.extinguishing_note for why it cannot be "
               "put out with water, CO2, or an inert blanket.",
               volume_change=0.041),
    ),
    note="The one metal here whose liquid phase is a combustion event rather "
         "than merely a hot fluid."))

_add(MaterialPhases(
    material="iron",
    transitions=(
        *_pair(MELT, 1811.0, 247_000.0,
               "Becomes molten-iron at 7000 kg/m3 and 1.87 N/m -- the highest "
               "surface tension of anything in the registry, which is why a "
               "weld pool stands up in a bead instead of running flat, and why "
               "slag floats off it so cleanly.",
               volume_change=0.030),
    )))

_add(MaterialPhases(
    material="lead",
    transitions=(
        *_pair(MELT, 600.61, 23_200.0,
               "A tiny latent heat -- a fortieth of aluminium's -- and a low "
               "melting point, which together are the whole reason lead is the "
               "metal you can melt on a camp stove and pour into a mould by "
               "hand. It is also why a babbitt bearing is repairable in place "
               "and an aluminium one is not.",
               volume_change=0.035),
    )))


#: Cold behaviour that is NOT a phase transition, kept beside the table
#: because that is precisely where someone will look for it and not find
#: it in the transitions above.
POUR_POINT_K = {
    "engine-oil": 245.0,        # typical multigrade; a straight SAE 30 is ~258
    "gear-oil": 248.0,
    "hydraulic-oil": 243.0,
}
CLOUD_POINT_K = {"diesel": 267.0, "jet-a-kerosene": 226.0}
CFPP_K = {"diesel": 264.0, "jet-a-kerosene": 224.0}


def phases(material: str) -> MaterialPhases | None:
    return PHASE_TABLE.get(material)


def flows_at(material: str, temp_k: float) -> bool:
    """Whether this material will actually MOVE at this temperature.

    Deliberately a different question from whether it is a liquid, and
    the reason this module exists in the form it does: oil at 240 K is
    thermodynamically liquid and completely immobile, and a pump asking
    `state_at` would be told everything is fine."""
    t = float(temp_k)
    pour = POUR_POINT_K.get(material)
    if pour is not None and t < pour:
        return False
    plug = CFPP_K.get(material)
    if plug is not None and t < plug:
        return False
    m = phases(material)
    if m is None:
        return True
    return m.state_at(t) != "solid"


def report() -> str:
    lines = ["material         state@253K  state@293K  state@393K   notable"]
    for name, m in PHASE_TABLE.items():
        flags = []
        if not flows_at(name, 253.0):
            flags.append("will not flow at 253 K")
        fr = m.get(FREEZE)
        if fr is not None and fr.volume_change_frac > 0.01:
            flags.append(f"expands {abs(fr.volume_change_frac) * 100:.2f}% freezing")
        if m.happens(SUBLIME):
            flags.append("no liquid phase at 1 atm")
        if not m.happens(VAPORISE):
            flags.append("never boils")
        lines.append(f"  {name:14s} {m.state_at(253.0):10s} {m.state_at(293.0):11s} "
                     f"{m.state_at(393.0):10s}   {'; '.join(flags)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
