"""Fire as a thing that lasts, spreads, and can be put out.

Burning is not an event here, it is a state. Something lights a fuel
(ordnance fireball, a spray on a hot part -- `EngineCycleSim.
ignite_within`), and from then on there is a `PoolFire`: a real
burning area fed by real fuel at a real rate, with a heat release rate
that heats what is near it, that can light its neighbours, and that
goes out when it runs out of fuel, runs out of air, or has enough
water thrown at it.

The numbers are the standard fire-engineering ones, disclosed:

  burning rate   a hydrocarbon pool burns at a mass flux per unit area
                 that is a property of the fuel (petrol ~0.055 kg/m2 s,
                 diesel ~0.035, oil ~0.030) times its area
  heat release   Q = m_dot * LHV * combustion efficiency (0.7 for an
                 open pool -- real pool fires are sooty and incomplete)
  spread         anything flammable within the flame's own radiant
                 reach lights: the radiant flux at distance r from a
                 fire of heat release Q is roughly chi Q / (4 pi r^2)
                 with a radiative fraction chi ~ 0.3, and a liquid
                 fuel needs about 10 kW/m2 to light
  suppression    water put on a fire takes heat away as it boils:
                 2.6 MJ per kg to go from ambient to steam, at a
                 fireground application efficiency of ~0.3. When the
                 heat taken exceeds the heat released, the fire goes
                 out. This is exactly the "critical flow rate"
                 calculation the fire service uses.
  air            a fire in a closed volume uses its own oxygen up:
                 ~3.1 kg of air per kg of hydrocarbon burned. In the
                 bay/room stack (`air_volumes.py`) that is a real
                 limit, not a decoration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

# per-fuel pool burning flux and heat content
POOL_FLUX_KG_M2_S = {"fuel": 0.055, "engine-oil": 0.030, "diesel": 0.035, "methanol": 0.017}
LHV_J_KG = {"fuel": 43.5e6, "engine-oil": 42.0e6, "diesel": 42.6e6, "methanol": 19.9e6}
POOL_COMBUSTION_EFFICIENCY = 0.7
RADIATIVE_FRACTION = 0.3
PILOTED_IGNITION_FLUX_W_M2 = 10_000.0      # a liquid fuel lights at about this radiant flux
WATER_HEAT_CAPACITY_J_KG = 2.6e6           # ambient water to steam
FIREFIGHTING_EFFICIENCY = 0.3              # the fraction of applied water that actually absorbs fire heat
AIR_PER_FUEL_KG = 3.1 * 4.76               # kg of air per kg of hydrocarbon (stoichiometric, by mass)
SPREAD_POOL_M2_PER_KG = 0.25               # how much floor a kilogram of spilled liquid covers


@dataclass
class PoolFire:
    """One burning pool or part."""
    identity: str
    position: tuple
    fuel: str = "fuel"
    fuel_kg: float = 1.0
    area_m2: float = 0.2
    source: str = ""
    burning: bool = True
    age_s: float = 0.0
    water_applied_kg: float = 0.0
    suppressed_frac: float = 0.0     # 0 free-burning .. 1 out
    starved: bool = False
    burned_kg: float = 0.0

    @property
    def flux_kg_m2_s(self) -> float:
        return POOL_FLUX_KG_M2_S.get(self.fuel, 0.03)

    @property
    def burn_rate_kg_s(self) -> float:
        if not self.burning or self.fuel_kg <= 0.0:
            return 0.0
        return self.flux_kg_m2_s * self.area_m2 * (1.0 - self.suppressed_frac)

    @property
    def heat_release_w(self) -> float:
        return self.burn_rate_kg_s * LHV_J_KG.get(self.fuel, 42e6) * POOL_COMBUSTION_EFFICIENCY

    def radiant_flux_at(self, point) -> float:
        r = max(float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(self.position, dtype=float))), 0.15)
        return RADIATIVE_FRACTION * self.heat_release_w / (4.0 * math.pi * r * r)

    def critical_water_kg_s(self) -> float:
        """The flow that would just put this fire out."""
        return self.heat_release_w / (WATER_HEAT_CAPACITY_J_KG * FIREFIGHTING_EFFICIENCY)

    def apply_water(self, kg_s: float, dt: float) -> None:
        if kg_s <= 0.0:
            return
        self.water_applied_kg += kg_s * dt
        crit = self.critical_water_kg_s()
        if crit <= 0.0:
            return
        # knocking a fire down is not instant: the suppressed fraction
        # follows the applied-over-critical ratio with a real time lag
        target = min(1.0, kg_s / crit)
        self.suppressed_frac += (target - self.suppressed_frac) * min(1.0, dt / 3.0)
        if self.suppressed_frac >= 0.98:
            self.burning = False

    def step(self, dt: float, oxygen_frac: float = 0.2095) -> float:
        """Burn for dt. Returns the fuel consumed (kg)."""
        self.age_s += dt
        if not self.burning:
            return 0.0
        if oxygen_frac < 0.13:             # a hydrocarbon flame goes out below roughly this
            self.starved = True
            self.burning = False
            return 0.0
        used = min(self.fuel_kg, self.burn_rate_kg_s * dt)
        self.fuel_kg -= used
        self.burned_kg += used
        if self.fuel_kg <= 1e-6:
            self.burning = False
        return used

    def describe(self) -> str:
        if not self.burning:
            why = ("out: starved of air" if self.starved else
                   "out: extinguished" if self.suppressed_frac > 0.5 else "out: fuel exhausted")
            return f"{self.identity.split('.')[-1]}: {why} after {self.age_s:.0f} s, {self.burned_kg:.2f} kg burned"
        return (f"{self.identity.split('.')[-1]}: {self.heat_release_w / 1000:.0f} kW over {self.area_m2:.2f} m2, "
                f"{self.fuel_kg:.2f} kg left, needs {self.critical_water_kg_s() * 60:.1f} L/min to knock down"
                + (f", {self.suppressed_frac * 100:.0f}% knocked down" if self.suppressed_frac > 0.02 else ""))


@dataclass
class FireField:
    fires: list = field(default_factory=list)
    total_burned_kg: float = 0.0

    @property
    def burning(self) -> list:
        return [f for f in self.fires if f.burning]

    @property
    def heat_release_w(self) -> float:
        return sum(f.heat_release_w for f in self.fires if f.burning)

    def light(self, identity: str, position, fuel: str, fuel_kg: float, source: str) -> PoolFire | None:
        if any(f.identity == identity and f.burning for f in self.fires):
            return None
        f = PoolFire(identity=identity, position=tuple(float(v) for v in position), fuel=fuel,
                     fuel_kg=max(fuel_kg, 1e-3), area_m2=max(0.05, fuel_kg * SPREAD_POOL_M2_PER_KG), source=source)
        self.fires.append(f)
        return f

    def step(self, dt: float, oxygen_frac: float = 0.2095) -> float:
        burned = 0.0
        for f in self.fires:
            burned += f.step(dt, oxygen_frac)
        self.total_burned_kg += burned
        return burned

    def spread_targets(self, candidates) -> list:
        """(identity, position, flux) for every candidate a burning fire
        is radiating enough at to light."""
        out = []
        for ident, pos in candidates:
            flux = sum(f.radiant_flux_at(pos) for f in self.fires if f.burning)
            if flux >= PILOTED_IGNITION_FLUX_W_M2:
                out.append((ident, pos, flux))
        return out

    def air_demand_kg_s(self) -> float:
        return sum(f.burn_rate_kg_s for f in self.fires if f.burning) * AIR_PER_FUEL_KG

    def summary(self) -> list[str]:
        return [f"  FIRE {f.describe()}" for f in self.fires[-6:]]
