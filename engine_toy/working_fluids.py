"""THE one registry of working fluids -- every liquid fuel, gaseous
fuel, and non-combusting expander fluid (steam, compressed air) this
toy can run a cylinder on, in one table with one real property
vocabulary, so that "converting a cylinder to another fuel" is a
change of fuel NETWORK (fuel_network.py: what hardware sits between
the store and the admission point) plus a lookup here -- never a
second engine model per fuel.

engines.py's _FUEL_PROPERTIES / FUEL_DENSITY_KG_M3 / FUEL_LATENT_HEAT_
J_PER_KG and otto_langen.py's FUEL_GAS_PROPERTIES are all VIEWS of
this table now (see the bottom of this module), not parallel copies
that can drift.

Every number is a real, disclosed order-of-magnitude property, not a
fitted knob:
  - energy_density_j_per_kg: lower heating value.
  - stoich_afr: stoichiometric air/fuel MASS ratio.
  - effective_octane: knock resistance (methane ~130 RON, hydrogen
    behaves ~130 lean but pre-ignites easily -- that's the min_
    ignition_energy row, a different real failure than knock).
  - density_kg_m3: liquid at ambient for liquids; gas at 1 atm / 20 C
    for gases (what a mixer's port areas and a holder's volume see).
  - latent_heat_j_per_kg: heat of vaporization -- charge cooling for
    liquids; for LPG/LNG it's the heat a VAPORIZER has to pull out of
    the coolant to deliver gas at all (fuel_network.Vaporizer).
  - lfl / ufl / stoich_vol_frac: flammability limits and stoichiometric
    gas VOLUME fraction in air (Coward & Jones, USBM Bulletin 503).
  - laminar_flame_speed_m_s: near-stoichiometric laminar flame speed --
    hydrogen's ~3 m/s vs methane's ~0.4 is the real reason timing and
    backfire behaviour differ so sharply between them.
  - min_ignition_energy_mj: minimum spark energy to light a stoich
    mixture -- hydrogen's 0.017 mJ (a static discharge) vs ~0.25 mJ for
    hydrocarbons is why hydrogen port injection backfires into a hot
    intake and why its network mandates a flame arrestor.
  - storage / storage_pressure_pa: how the real store holds it.
  - gamma / gas_constant_j_per_kgk: for expander fluids, the isentropic
    exponent and R the cutoff cylinder expands with.
"""
from __future__ import annotations

from dataclasses import dataclass

LIQUID_FUEL = "liquid-fuel"
GASEOUS_FUEL = "gaseous-fuel"
EXPANDER_FLUID = "expander-fluid"

ATMOSPHERIC_TANK = "atmospheric-tank"        # a vented liquid tank
PRESSURIZED_LIQUID = "pressurized-liquid"    # LPG: liquid under its own vapour pressure, needs a vaporizer
HIGH_PRESSURE_GAS = "high-pressure-gas"      # CNG / H2 bottle, needs staged regulation
CRYOGENIC = "cryogenic"                      # LNG / LH2, needs a vaporizer (boil-off, not coolant-limited)
GASHOLDER = "gasholder"                      # low-pressure holder fed by a generator or a main
BOILER = "boiler"                            # steam dome
RECEIVER = "receiver"                        # compressed-air receiver


@dataclass(frozen=True)
class WorkingFluid:
    name: str
    phase: str
    energy_density_j_per_kg: float = 0.0
    stoich_afr: float = 0.0
    effective_octane: float = 0.0
    density_kg_m3: float = 0.0
    latent_heat_j_per_kg: float = 0.0
    #: Temperature at which this fluid boils at one atmosphere, K.
    #:
    #: NOT A SHARP NUMBER FOR ANY PETROLEUM FUEL, and that is worth
    #: knowing rather than hiding. Gasoline, diesel and kerosene are
    #: BLENDS: they boil across a whole distillation range, gasoline
    #: from around 308 K to 480 K. The single figure here is the
    #: standard T50 characterisation point -- the temperature at which
    #: half the sample has evaporated -- which is the number refiners
    #: actually specify and the one that governs how fast a spill dries.
    #: Pure compounds (methanol, nitromethane) have a real sharp value
    #: and get it exactly.
    #:
    #: 0.0 means "does not usefully boil here": engine oil cracks before
    #: it distils, so anything reading this must treat 0.0 as
    #: non-volatile rather than as absolute zero.
    boiling_point_k: float = 0.0
    lfl: float = 0.0
    ufl: float = 0.0
    stoich_vol_frac: float = 0.0
    laminar_flame_speed_m_s: float = 0.0
    min_ignition_energy_mj: float = 0.0
    storage: str = ATMOSPHERIC_TANK
    storage_pressure_pa: float = 101_325.0
    gamma: float = 1.4
    gas_constant_j_per_kgk: float = 287.0
    flame_arrestor_required: bool = False
    hardened_valve_seats_required: bool = False
    # how a cylinder lights it: "spark" | "compression" (a cetane fuel:
    # diesel, kerosene, crude, vegetable oil) | "none" (expander fluids)
    ignition: str = "spark"
    # too viscous to pump/atomize cold (straight vegetable oil, crude):
    # a real conversion needs a fuel heater (fuel_network.FuelHeater)
    viscous_at_ambient: bool = False

    # ---- WHAT MAKES A FUEL HARD ON AN ENGINE -------------------------
    # Octane already says how a fuel resists knock in a spark engine.
    # These say the other things a fuel does to the hardware it is burnt
    # in, so tolerance can be DERIVED per engine instead of typed per
    # pair (see fuel_tolerance below).
    #: Ignition quality under compression. The diesel counterpart of
    #: octane, and the opposite of it -- a good diesel fuel is a bad
    #: spark fuel and the reverse. 0 means "no opinion declared".
    cetane: float = 0.0
    #: Incombustible mineral content, mass fraction. This is the one that
    #: destroys turbines: sodium and vanadium in residual fuels attack a
    #: hot section far faster than heat alone. A piston engine mostly
    #: just wears.
    ash_frac: float = 0.0
    #: Tendency to leave gum, lacquer and carbon behind. Rings, injector
    #: tips, valve stems, turbine nozzles.
    gum_tendency: float = 0.0
    #: Sulphur, mass fraction. Acid in the crankcase, and why oil change
    #: intervals were short before ULSD.
    sulfur_frac: float = 0.0
    #: Temperature this must be heated to before it will atomise at all.
    #: Ambient means none needed; a residual oil is unusable cold.
    preheat_c: float = 0.0

    @property
    def combustible(self) -> bool:
        return self.phase != EXPANDER_FLUID

    @property
    def gaseous(self) -> bool:
        return self.phase == GASEOUS_FUEL


def _liquid(name, energy, afr, octane, density, latent=350_000.0, flame=0.40, mie=0.25,
            ignition="spark", viscous=False, cetane=0.0, ash=0.0, gum=0.0,
            sulfur=0.0, preheat=0.0, boiling=0.0) -> WorkingFluid:
    return WorkingFluid(name=name, phase=LIQUID_FUEL, energy_density_j_per_kg=energy, stoich_afr=afr,
                        effective_octane=octane, density_kg_m3=density, latent_heat_j_per_kg=latent,
                        boiling_point_k=boiling,
                        laminar_flame_speed_m_s=flame, min_ignition_energy_mj=mie, ignition=ignition,
                        viscous_at_ambient=viscous, cetane=cetane, ash_frac=ash,
                        gum_tendency=gum, sulfur_frac=sulfur, preheat_c=preheat)


WORKING_FLUIDS: dict[str, WorkingFluid] = {f.name: f for f in (
    # -- liquid fuels (the catalogue's existing rows, unchanged values) --
    _liquid("pump-gasoline-87", 44.0e6, 14.7, 87.0, 745.0, boiling=373.0),
    _liquid("pump-gasoline-89", 44.0e6, 14.7, 89.0, 745.0, boiling=373.0),
    _liquid("pump-gasoline-91", 44.0e6, 14.7, 91.0, 745.0, boiling=373.0),
    _liquid("pump-gasoline-93", 44.0e6, 14.7, 93.0, 745.0, boiling=373.0),
    _liquid("aviation-gasoline-100-130", 43.5e6, 14.7, 100.0, 715.0, boiling=367.0),
    _liquid("methanol-race", 19.9e6, 6.4, 105.0, 792.0, latent=1_100_000.0, flame=0.45, boiling=337.85),
    _liquid("nitromethane-race", 11.3e6, 1.7, 110.0, 1140.0, latent=330_000.0, boiling=374.35),
    _liquid("ultra-low-sulfur-diesel", 45.5e6, 14.5, 100.0, 832.0, latent=250_000.0,
            ignition="compression", cetane=48.0, sulfur=0.000015, boiling=533.0),
    _liquid("jet-a-kerosene", 43.0e6, 14.5, 100.0, 800.0, latent=250_000.0,
            ignition="compression", cetane=43.0, sulfur=0.0003, boiling=473.0),
    _liquid("kerosene", 43.0e6, 14.5, 100.0, 800.0, latent=250_000.0,
            ignition="compression", cetane=40.0, sulfur=0.0004, gum=0.05, boiling=473.0),
    # JP-8 is kerosene with additives; the military multifuel standard.
    _liquid("jp-8", 43.0e6, 14.5, 100.0, 800.0, latent=250_000.0,
            ignition="compression", cetane=43.0, sulfur=0.0003, boiling=473.0),
    # Crude is the hard one and the ash is why: sodium and vanadium in a
    # hot section corrode it far faster than temperature alone does.
    _liquid("crude-oil", 42.0e6, 14.0, 100.0, 870.0, latent=250_000.0,
            ignition="compression", viscous=True, cetane=35.0, ash=0.0008,
            gum=0.45, sulfur=0.015, preheat=45.0),
    _liquid("vegetable-oil", 37.5e6, 12.5, 100.0, 920.0, latent=250_000.0,
            ignition="compression", viscous=True, cetane=38.0, gum=0.55, preheat=70.0),
    # USED ENGINE OIL. A real waste-oil fuel: burnt in heaters, in big
    # marine and stationary diesels, and acceptable to a continuous
    # burner. It carries the metal it wore off the engine it came out
    # of, which is exactly the ash that eats a turbine's hot section --
    # so it is "yes, at a cost", not "yes".
    _liquid("engine-oil-waste", 40.0e6, 13.5, 100.0, 890.0, latent=250_000.0,
            ignition="compression", viscous=True, cetane=30.0, ash=0.011,
            gum=0.70, sulfur=0.006, preheat=80.0),
    # Residual/bunker fuel: the bottom of the barrel, and unusable cold.
    _liquid("heavy-fuel-oil", 40.5e6, 13.8, 100.0, 980.0, latent=250_000.0,
            ignition="compression", viscous=True, cetane=32.0, ash=0.0015,
            gum=0.60, sulfur=0.025, preheat=120.0),
    _liquid("biodiesel-b100", 37.8e6, 13.8, 100.0, 880.0, latent=250_000.0,
            ignition="compression", cetane=55.0, gum=0.30, sulfur=0.00001),
    # E85: high octane, big latent heat, poor energy density -- it needs
    # far more fuel for the same air and cools the charge doing it.
    _liquid("ethanol-e85", 29.2e6, 9.8, 105.0, 785.0, latent=760_000.0, flame=0.44),
    # -- gaseous fuels --
    # Victorian town gas: H2/CH4/CO, light, wide band, fast flame (H2-rich).
    WorkingFluid("coal-gas", GASEOUS_FUEL, energy_density_j_per_kg=30.0e6, stoich_afr=8.6,
                 effective_octane=110.0, density_kg_m3=0.70, lfl=0.053, ufl=0.31, stoich_vol_frac=0.17,
                 laminar_flame_speed_m_s=0.9, min_ignition_energy_mj=0.05, storage=GASHOLDER,
                 storage_pressure_pa=101_325.0 + 1_500.0, flame_arrestor_required=True,
                 hardened_valve_seats_required=True),
    # producer / wood gas: ~half nitrogen by mass, very low LHV, needs ~1:1 air.
    WorkingFluid("wood-gas", GASEOUS_FUEL, energy_density_j_per_kg=4.8e6, stoich_afr=1.2,
                 effective_octane=100.0, density_kg_m3=1.15, lfl=0.20, ufl=0.74, stoich_vol_frac=0.47,
                 laminar_flame_speed_m_s=0.3, min_ignition_energy_mj=0.3, storage=GASHOLDER,
                 storage_pressure_pa=101_325.0 + 1_500.0, hardened_valve_seats_required=True),
    WorkingFluid("natural-gas", GASEOUS_FUEL, energy_density_j_per_kg=50.0e6, stoich_afr=17.2,
                 effective_octane=130.0, density_kg_m3=0.68, lfl=0.05, ufl=0.15, stoich_vol_frac=0.095,
                 laminar_flame_speed_m_s=0.38, min_ignition_energy_mj=0.28, storage=HIGH_PRESSURE_GAS,
                 storage_pressure_pa=22.0e6, hardened_valve_seats_required=True),
    # LPG: stored liquid at ~8.5 bar, vaporized by coolant heat.
    WorkingFluid("propane", GASEOUS_FUEL, energy_density_j_per_kg=46.4e6, stoich_afr=15.7,
                 effective_octane=105.0, density_kg_m3=1.9, latent_heat_j_per_kg=430_000.0,
                 lfl=0.021, ufl=0.095, stoich_vol_frac=0.040, laminar_flame_speed_m_s=0.43,
                 min_ignition_energy_mj=0.25, storage=PRESSURIZED_LIQUID, storage_pressure_pa=850_000.0,
                 hardened_valve_seats_required=True),
    WorkingFluid("hydrogen", GASEOUS_FUEL, energy_density_j_per_kg=120.0e6, stoich_afr=34.3,
                 effective_octane=130.0, density_kg_m3=0.084, lfl=0.04, ufl=0.75, stoich_vol_frac=0.295,
                 laminar_flame_speed_m_s=3.0, min_ignition_energy_mj=0.017, storage=HIGH_PRESSURE_GAS,
                 storage_pressure_pa=70.0e6, flame_arrestor_required=True,
                 hardened_valve_seats_required=True),
    WorkingFluid("petroleum-vapor", GASEOUS_FUEL, energy_density_j_per_kg=44.0e6, stoich_afr=14.7,
                 effective_octane=90.0, density_kg_m3=3.0, lfl=0.014, ufl=0.076, stoich_vol_frac=0.018,
                 laminar_flame_speed_m_s=0.40, min_ignition_energy_mj=0.25, storage=ATMOSPHERIC_TANK),
    # -- expander fluids: no combustion, the cylinder is a cutoff expander (expander.py) --
    WorkingFluid("compressed-air", EXPANDER_FLUID, density_kg_m3=1.20, storage=RECEIVER,
                 storage_pressure_pa=827_000.0, gamma=1.4, gas_constant_j_per_kgk=287.0, ignition="none"),
    # saturated steam at a traction-engine ~6 bar; latent heat is what the boiler must supply per kg
    WorkingFluid("steam", EXPANDER_FLUID, density_kg_m3=0.60, latent_heat_j_per_kg=2_086_000.0,
                 storage=BOILER, storage_pressure_pa=600_000.0, gamma=1.30, gas_constant_j_per_kgk=461.5,
                 ignition="none"),
)}


def working_fluid(name: str) -> WorkingFluid:
    try:
        return WORKING_FLUIDS[name]
    except KeyError:
        raise KeyError(f"unknown working fluid {name!r}; known: {sorted(WORKING_FLUIDS)}") from None


# ---------------------------------------------------------------------
# Views the older per-module tables are now built from (single source)
# ---------------------------------------------------------------------
FUEL_PROPERTIES: dict[str, dict] = {
    f.name: dict(energy_density_j_per_kg=f.energy_density_j_per_kg, stoich_afr=f.stoich_afr,
                 effective_octane=f.effective_octane)
    for f in WORKING_FLUIDS.values() if f.combustible
}
FUEL_DENSITY_KG_M3: dict[str, float] = {f.name: f.density_kg_m3 for f in WORKING_FLUIDS.values() if f.combustible}
FUEL_LATENT_HEAT_J_PER_KG: dict[str, float] = {
    f.name: f.latent_heat_j_per_kg for f in WORKING_FLUIDS.values() if f.combustible and f.latent_heat_j_per_kg > 0.0
}
FUEL_GAS_PROPERTIES: dict[str, dict[str, float]] = {
    f.name: dict(density_kg_m3=f.density_kg_m3, lfl=f.lfl, ufl=f.ufl, stoich_vol_frac=f.stoich_vol_frac)
    for f in WORKING_FLUIDS.values() if f.gaseous
}


# ---------------------------------------------------------------------
# tolerance, DERIVED rather than typed per engine/fuel pair
# ---------------------------------------------------------------------

#: How an engine meets its fuel. This is the property that decides most
#: of the tolerance, and it is not the same question as "petrol or
#: diesel".
SPARK_IGNITION = "spark"            #: a flame kernel, so octane is everything
COMPRESSION_IGNITION = "compression"  #: autoignition, so cetane is everything
CONTINUOUS_BURNER = "continuous"    #: a flame in a can that never goes out


def fuel_tolerance(fluid: "WorkingFluid", *, burner: str,
                   compression_ratio: float = 10.0,
                   hot_section: bool = False,
                   fuel_heater: bool = False) -> tuple[float, tuple[str, ...]]:
    """How well this engine tolerates this fuel, and WHY.

    Returns a 0..1 factor and the reasons that moved it, so a bad score
    is explainable rather than a number someone chose. The reasons are
    the point: "0.25" tells you nothing, "cetane 30 against a 22:1
    compression ratio" tells you what to change.

    The three burners want genuinely different things, which is why one
    table of per-pair numbers could never be right:

      SPARK          lives or dies on octane against its compression
                     ratio. Knock is the failure and it is immediate.
      COMPRESSION    wants cetane, and does not care about octane at
                     all -- a high-octane fuel is a BAD diesel fuel,
                     which is why petrol in a diesel is a problem of
                     ignition delay and not of knock.
      CONTINUOUS     barely cares about ignition quality either way,
                     because the flame is already lit and stays lit.
                     This is the real reason a gas turbine is the
                     multifuel engine and a piston engine is not. What
                     it cares about instead is ASH, because a hot
                     section corrodes.

    Viscosity is a hard gate rather than a penalty: a fuel that will not
    atomise cold will not run at all without a heater, however good it
    is once warm.
    """
    reasons: list[str] = []
    factor = 1.0

    if fluid.preheat_c > 0.0 and not fuel_heater:
        factor *= 0.05
        reasons.append(
            f"needs {fluid.preheat_c:.0f} C of preheat and there is no fuel heater")
    elif fluid.viscous_at_ambient and not fuel_heater:
        factor *= 0.55
        reasons.append("viscous cold: poor atomisation, hard starting")

    if burner == SPARK_IGNITION:
        # a real, disclosed octane requirement from the compression the
        # piston/head choice actually runs
        required = 80.0 + 6.0 * max(compression_ratio - 8.0, 0.0)
        margin = fluid.effective_octane - required
        if margin < 0.0:
            factor *= max(0.10, 1.0 + margin / 40.0)
            reasons.append(
                f"octane {fluid.effective_octane:.0f} against {required:.0f} required "
                f"at {compression_ratio:.1f}:1 -- knock")
        if fluid.cetane >= 40.0:
            factor *= 0.45
            reasons.append(
                f"cetane {fluid.cetane:.0f}: pre-ignites before the plug fires")
    elif burner == COMPRESSION_IGNITION:
        required_cetane = 52.0 - 0.8 * max(compression_ratio - 16.0, 0.0)
        if fluid.cetane <= 0.0:
            factor *= 0.08
            reasons.append("no compression ignition quality at all: it will not light")
        else:
            margin = fluid.cetane - required_cetane
            if margin < 0.0:
                factor *= max(0.15, 1.0 + margin / 30.0)
                reasons.append(
                    f"cetane {fluid.cetane:.0f} against {required_cetane:.0f} wanted at "
                    f"{compression_ratio:.1f}:1 -- long ignition delay, hard knock")
    elif burner == CONTINUOUS_BURNER:
        # almost anything burns. What it costs is the hot section.
        if fluid.ash_frac > 0.0 and hot_section:
            factor *= max(0.25, 1.0 - fluid.ash_frac * 45.0)
            reasons.append(
                f"ash {fluid.ash_frac*100:.2f}%: hot-section corrosion, shortened life")
        elif fluid.ash_frac > 0.0:
            factor *= max(0.70, 1.0 - fluid.ash_frac * 12.0)
            reasons.append(f"ash {fluid.ash_frac*100:.2f}%: deposits")

    # everything gunks and everything sulphurs, in proportion
    if fluid.gum_tendency > 0.0:
        factor *= max(0.60, 1.0 - fluid.gum_tendency * 0.30)
        reasons.append(f"gum {fluid.gum_tendency:.2f}: deposits on tips and rings")
    if fluid.sulfur_frac > 0.0005:
        factor *= max(0.75, 1.0 - fluid.sulfur_frac * 8.0)
        reasons.append(f"sulphur {fluid.sulfur_frac*100:.2f}%: acid wear, shorter oil life")

    return max(0.0, min(1.0, factor)), tuple(reasons)


if __name__ == "__main__":
    CASES = (
        ("gas turbine (hot section)", CONTINUOUS_BURNER, 10.0, True, True),
        ("multifuel diesel 22:1", COMPRESSION_IGNITION, 22.0, False, True),
        ("road diesel 17:1", COMPRESSION_IGNITION, 17.0, False, False),
        ("petrol engine 10:1", SPARK_IGNITION, 10.0, False, False),
    )
    FUELS = ("jet-a-kerosene", "ultra-low-sulfur-diesel", "pump-gasoline-87",
             "crude-oil", "engine-oil-waste", "vegetable-oil", "heavy-fuel-oil")
    for label, burner, cr, hot, heater in CASES:
        print(f"\n{label}" + ("  [fuel heater fitted]" if heater else ""))
        for name in FUELS:
            factor, why = fuel_tolerance(WORKING_FLUIDS[name], burner=burner,
                                         compression_ratio=cr, hot_section=hot,
                                         fuel_heater=heater)
            head = why[0] if why else "no objection"
            print(f"   {name:<26}{factor:5.2f}   {head}")
