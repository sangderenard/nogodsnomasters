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

    @property
    def combustible(self) -> bool:
        return self.phase != EXPANDER_FLUID

    @property
    def gaseous(self) -> bool:
        return self.phase == GASEOUS_FUEL


def _liquid(name, energy, afr, octane, density, latent=350_000.0, flame=0.40, mie=0.25,
            ignition="spark", viscous=False) -> WorkingFluid:
    return WorkingFluid(name=name, phase=LIQUID_FUEL, energy_density_j_per_kg=energy, stoich_afr=afr,
                        effective_octane=octane, density_kg_m3=density, latent_heat_j_per_kg=latent,
                        laminar_flame_speed_m_s=flame, min_ignition_energy_mj=mie, ignition=ignition,
                        viscous_at_ambient=viscous)


WORKING_FLUIDS: dict[str, WorkingFluid] = {f.name: f for f in (
    # -- liquid fuels (the catalogue's existing rows, unchanged values) --
    _liquid("pump-gasoline-87", 44.0e6, 14.7, 87.0, 745.0),
    _liquid("pump-gasoline-89", 44.0e6, 14.7, 89.0, 745.0),
    _liquid("pump-gasoline-91", 44.0e6, 14.7, 91.0, 745.0),
    _liquid("pump-gasoline-93", 44.0e6, 14.7, 93.0, 745.0),
    _liquid("aviation-gasoline-100-130", 43.5e6, 14.7, 100.0, 715.0),
    _liquid("methanol-race", 19.9e6, 6.4, 105.0, 792.0, latent=1_100_000.0, flame=0.45),
    _liquid("nitromethane-race", 11.3e6, 1.7, 110.0, 1140.0, latent=330_000.0),
    _liquid("ultra-low-sulfur-diesel", 45.5e6, 14.5, 100.0, 832.0, latent=250_000.0, ignition="compression"),
    _liquid("jet-a-kerosene", 43.0e6, 14.5, 100.0, 800.0, latent=250_000.0, ignition="compression"),
    _liquid("kerosene", 43.0e6, 14.5, 100.0, 800.0, latent=250_000.0, ignition="compression"),
    _liquid("crude-oil", 42.0e6, 14.0, 100.0, 870.0, latent=250_000.0, ignition="compression", viscous=True),
    _liquid("vegetable-oil", 37.5e6, 12.5, 100.0, 920.0, latent=250_000.0, ignition="compression", viscous=True),
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
