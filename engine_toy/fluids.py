"""One registry of the fluids this machine can actually contain.

Everything that used to be five parallel dictionaries -- surface
tension here, density there, a circuit-name mapping somewhere else, a
guess-the-fluid-from-a-string function next to it, and a leak colour in
the mesh module -- is one row per fluid.

The point is that adding a fluid is adding a row. Transmission fluid
went in as exactly that: a density, a viscosity, a surface tension, the
circuit names that carry it, the words that name it, and the colour it
leaks. Nothing else in the leak, spray, burst or damage-sound path
needed editing, because all of them ask this registry rather than
carrying their own copy of the table.

The physical numbers are real and at working temperature, which is the
condition a leak actually happens in: hot engine oil is far thinner
than the cold SAE grade on the bottle, and ATF is thinner still, which
is exactly why a transmission line lets go as a fine spray where an oil
gallery makes a rope.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fluid:
    key: str                       # the property key used everywhere else
    label: str                     # what to call it on screen
    phase: str                     # "liquid" | "gas"
    density_kg_m3: float
    viscosity_pa_s: float          # dynamic, at working temperature
    surface_tension_n_m: float
    circuits: tuple[str, ...] = ()      # circuit identities that carry it
    keywords: tuple[str, ...] = ()      # words in a node/circuit name that mean it
    leak_material: str = "leak_fuel"    # engine_mesh.py row for the streaks
    flammable: bool = False
    autoignition_k: float = 0.0         # what a spray needs to touch to light
    ullage_vapour: bool = False         # a part-empty vessel holds explosive vapour
    # ---- THE OTHER SIDE OF THE REACTION ----
    # This registry described only FUELS: `flammable` and an ignition
    # temperature. That is half a combustion model, and the missing half
    # is not cosmetic -- without it nothing can tell the difference
    # between venting nitrogen into a space and venting oxygen into it,
    # and a fire needs both sides declared to happen at all.
    #: Declared oxidiser. Not the same as "contains oxygen": water
    #: contains oxygen and will not oxidise anything.
    oxidiser: bool = False
    #: Kilograms of O2-equivalent this fluid can deliver per kilogram of
    #: itself. Air is 0.232, oxygen 1.0, nitrous oxide 0.364 -- because
    #: N2O -> N2 + 1/2 O2 gives 16 g of oxygen from 44 g of nitrous.
    oxygen_equivalence: float = 0.0
    #: Energy released by the oxidiser DECOMPOSING, with no fuel present
    #: at all. Zero for oxygen, which is stable. Large for nitrous oxide,
    #: which is a monopropellant: this is why an N2O bottle can detonate
    #: on its own and an O2 bottle cannot.
    decomposition_j_kg: float = 0.0
    #: Whether that decomposition, once started, propagates through the
    #: bulk without further ignition.
    self_sustaining: bool = False

    @property
    def oxidising_power(self) -> float:
        """How hard this pushes a fire, with AIR as 1.0.

        Two terms. The first is simply how much oxygen it can hand over
        relative to air. The second is the energy it brings itself,
        which raises the local temperature at the reaction front and so
        makes ignition easier independently of how much oxygen is
        present -- and it is the term that makes nitrous oxide a worse
        actor than pure oxygen despite carrying barely a third of the
        oxygen per kilogram."""
        if not self.oxidiser:
            return 0.0
        supply = self.oxygen_equivalence / 0.232
        carried = self.decomposition_j_kg / 600_000.0
        return supply + carried
    # ---- THERMAL ----
    # This registry carried density and viscosity and not one thermal
    # property, so a fluid circuit could say how a leak ran down a
    # casting and could not say how much heat the fluid moved -- which
    # is the only question anyone actually asks a coolant. Declared
    # here so every circuit in the project reads the same numbers.
    specific_heat_j_per_kg_k: float = 4180.0
    conductivity_w_per_m_k: float = 0.60
    #: AT ONE ATMOSPHERE. Pressurising a loop raises this, and that is
    #: the entire reason cooling systems are pressurised.
    boiling_point_k: float = 373.15
    freezing_point_k: float = 273.15


#: cp (J/kg.K), conductivity (W/m.K), boiling point, freezing point --
#: all at one atmosphere, all real published figures for the ordinary
#: commercial grade of each.
THERMAL_PROPERTIES = {
    "engine-oil": (1900.0, 0.145, 560.0, 248.0),
    "transmission-fluid": (2000.0, 0.140, 540.0, 233.0),
    "gear-oil": (1900.0, 0.140, 560.0, 253.0),
    "hydraulic-oil": (1900.0, 0.135, 550.0, 233.0),
    # 50/50 ethylene glycol: barely half water's specific heat, and
    # that is the price of the freeze protection
    "coolant": (3400.0, 0.400, 380.0, 236.0),
    "fuel": (2100.0, 0.120, 420.0, 200.0),
    "water": (4180.0, 0.600, 373.15, 273.15),
    "gas": (1005.0, 0.026, 0.0, 0.0),
}

_BASE_FLUIDS: tuple[Fluid, ...] = (
    Fluid("engine-oil", "engine oil", "liquid", 870.0, 0.045, 0.031,
          circuits=("oil",), keywords=("engine-oil", "motor oil", "lube"),
          leak_material="leak_engine-oil", flammable=True, autoignition_k=633.15),
    # WHAT COMES OUT OF A SEPARATOR is not the oil that went in. It is
    # oil plus everything taken out of it -- soot, water, wear metal --
    # so it is markedly denser and thicker, and it is worth being its
    # own fluid because a kilogram of it is not a litre of oil. Treating
    # ejected sludge as engine oil lost a fifth of every spill in the
    # round trip between a centrifuge's kilograms and an emitter's
    # litres.
    Fluid("sludge", "separator sludge", "liquid", 1100.0, 0.30, 0.033,
          circuits=("oil",), keywords=("sludge", "separator sludge", "bowl solids"),
          leak_material="leak_engine-oil", flammable=True, autoignition_k=633.15),
    # ATF and hydraulic oil are genuinely close relatives -- a light
    # mineral oil with friction modifiers, dyed red so a puddle under a
    # machine tells you at a glance which system it came out of. They
    # look alike on purpose here, because they look alike in a workshop.
    Fluid("transmission-fluid", "transmission fluid (ATF)", "liquid", 850.0, 0.021, 0.030,
          circuits=("transmission", "torque-converter", "atf"),
          # "transmission" alone counts: an automatic is full of ATF,
          # and a manual's gear oil is close enough in density and
          # surface behaviour that a leak looks and pours the same
          keywords=("atf", "transmission", "torque converter", "dexron", "mercon"),
          leak_material="leak_transmission-fluid", flammable=True, autoignition_k=613.15),
    # GEAR OIL is its own fluid and not a thick engine oil: an EP
    # (extreme-pressure) sulphur-phosphorus package in a much heavier
    # base, because a hypoid tooth face slides as well as rolls and
    # needs a film that survives it. At working temperature it is still
    # roughly twice as viscous as hot engine oil, which is exactly why a
    # differential leak makes a slow crawling rope rather than a spray.
    Fluid("gear-oil", "gear oil", "liquid", 900.0, 0.090, 0.030,
          circuits=("gear-oil", "final-drive", "differential"),
          keywords=("gear oil", "gearbox", "differential", "final drive",
                    "transfer case", "transaxle", "hypoid"),
          leak_material="leak_gear-oil", flammable=True, autoignition_k=643.15),
    Fluid("hydraulic-oil", "hydraulic oil", "liquid", 870.0, 0.032, 0.032,
          circuits=("hydraulic", "hydronic"),
          keywords=("hydraulic", "hydronic"),
          leak_material="leak_hydraulic-oil", flammable=True, autoignition_k=633.15),
    Fluid("coolant", "coolant", "liquid", 1050.0, 0.0012, 0.060,
          circuits=("coolant",), keywords=("coolant", "glycol", "antifreeze"),
          leak_material="leak_coolant"),
    Fluid("fuel", "fuel", "liquid", 745.0, 0.0006, 0.024,
          circuits=("fuel",),
          keywords=("fuel", "gasoline", "petrol", "diesel", "methanol",
                    "ethanol", "kerosene", "nitro"),
          leak_material="leak_fuel", flammable=True, autoignition_k=553.15,
          ullage_vapour=True),
    Fluid("water", "water", "liquid", 1000.0, 0.0010, 0.072,
          circuits=("water", "condensate"), keywords=("water", "condensate"),
          leak_material="leak_water"),
    Fluid("oxygen", "gaseous oxygen", "gas", 1.429, 2.04e-5, 0.0,
          circuits=("oxygen", "asu-product"), keywords=("oxygen", "lox", "o2"),
          leak_material="leak_gas", flammable=False,
          specific_heat_j_per_kg_k=918.0, conductivity_w_per_m_k=0.0263,
          boiling_point_k=90.19, freezing_point_k=54.36,
          oxidiser=True, oxygen_equivalence=1.0),
    Fluid("nitrous-oxide", "nitrous oxide", "gas", 1.977, 1.47e-5, 0.0,
          circuits=("nitrous",), keywords=("nitrous", "n2o", "nos"),
          leak_material="leak_gas", flammable=False,
          specific_heat_j_per_kg_k=880.0, conductivity_w_per_m_k=0.0173,
          boiling_point_k=184.7, freezing_point_k=182.3,
          oxidiser=True, oxygen_equivalence=0.364,
          decomposition_j_kg=1_864_000.0, self_sustaining=True),
    Fluid("gas", "gas", "gas", 1.2, 1.8e-5, 0.0,
          circuits=("intake-air", "exhaust", "pneumatic-reserve", "pneumatic",
                    "nitrous", "boost", "refrigerant", "blanket", "nitrogen"),
          keywords=("air", "gas", "exhaust", "steam", "nitrous", "boost",
                    "vapour", "vapor", "refrigerant", "nitrogen"),
          leak_material="leak_gas"),
)


#: the same fluids, carrying their thermal properties
FLUIDS: tuple[Fluid, ...] = tuple(
    dataclasses.replace(
        f,
        specific_heat_j_per_kg_k=THERMAL_PROPERTIES[f.key][0],
        conductivity_w_per_m_k=THERMAL_PROPERTIES[f.key][1],
        boiling_point_k=THERMAL_PROPERTIES[f.key][2],
        freezing_point_k=THERMAL_PROPERTIES[f.key][3],
    ) if f.key in THERMAL_PROPERTIES else f
    for f in _BASE_FLUIDS
)

BY_KEY: dict[str, Fluid] = {f.key: f for f in FLUIDS}
LIQUIDS: tuple[Fluid, ...] = tuple(f for f in FLUIDS if f.phase == "liquid")

# the derived lookups the rest of the engine asks for, built once from
# the rows above rather than hand-maintained beside them
DENSITY_KG_M3: dict[str, float] = {f.key: f.density_kg_m3 for f in FLUIDS}
VISCOSITY_PA_S: dict[str, float] = {f.key: f.viscosity_pa_s for f in FLUIDS}
SURFACE_TENSION_N_M: dict[str, float] = {f.key: f.surface_tension_n_m for f in LIQUIDS}
LIQUID_OF_CIRCUIT: dict[str, str] = {c: f.key for f in LIQUIDS for c in f.circuits}
GAS_CIRCUITS: tuple[str, ...] = tuple(c for f in FLUIDS if f.phase == "gas" for c in f.circuits)
LEAK_MATERIAL: dict[str, str] = {f.key: f.leak_material for f in FLUIDS}


def fluid_key(name) -> str | None:
    """A declared fluid label -> the property key.

    An EXACT key wins, always, and that is the path everything built
    now takes: a node says `fluid="gear-oil"` and gets gear oil. The
    keyword pass below exists only for labels that were written as
    prose before this registry did ("motor oil", "hypoid"), and it is
    longest-match so a qualified name beats a bare one -- a
    "transmission oil" is ATF and not engine oil. Guessing a part's
    identity from its name is a different thing entirely and is not
    done anywhere: parts declare what they are.
    """
    if not name:
        return None
    raw = str(name).strip().lower()
    if raw in BY_KEY:
        return raw
    n = raw.replace("_", " ").replace("-", " ")
    best: tuple[int, str] | None = None
    for f in FLUIDS:
        for kw in f.keywords:
            k = kw.replace("-", " ")
            if k in n and (best is None or len(k) > best[0]):
                best = (len(k), f.key)
    if best is not None:
        return best[1]
    return "engine-oil" if "oil" in n else None


def of_circuit(circuit_identity: str | None) -> Fluid | None:
    """Which fluid a circuit identity carries."""
    if not circuit_identity:
        return None
    cid = str(circuit_identity).lower()
    for f in FLUIDS:
        if cid in f.circuits:
            return f
    for f in FLUIDS:
        if any(cid.startswith(c) for c in f.circuits):
            return f
    key = fluid_key(cid)
    return BY_KEY.get(key) if key else None


def leak_material_for(fluid_key_or_none: str | None) -> str:
    """The engine_mesh.py material row a leak of this should be drawn
    with, so a puddle's colour is decided in one place."""
    return LEAK_MATERIAL.get(fluid_key_or_none or "", "leak_fuel")
