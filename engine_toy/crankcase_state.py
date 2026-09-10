"""Crankcase participation, per cylinder, as flat arrays: every
crankcase splash-lubricates its bores (the crank throws oil at the
cylinder bottoms whether or not a pump also feeds the galleries), and
every cylinder is an OBSTRUCTED PATH between the crankcase and the
combustion chamber in which oil, fuel and gas each have a finite
participation:

  oil film on the bore   deposited by splash (crank speed x how much
                         oil is in the sump/trough), scraped down by
                         the rings (rpm), and burned past them into the
                         chamber (film x cylinder pressure x ring
                         leak) -- that burned oil is the engine's oil
                         consumption, and it is a real carbon source
                         for the valves (valve_state.age's oil term)
  blow-by gas            combustion gas past the rings into the case
                         (per fire, x combustion strength x ring leak):
                         it raises crankcase pressure, which vents
                         through the breather; with no open breather
                         the case pressurises and pushes oil past the
                         seals
  fuel dilution          unburned fuel washing past the rings into the
                         oil on cold, rich running; it boils back out
                         once the oil is hot
The sump (or splash trough) holds a finite oil mass that consumption
and seal loss draw down and a top-up restores.

Ring seal carries machine error per cylinder (the same seeded draw
the valvetrain uses), so consumption and blow-by differ by cylinder.
All of it is a handful of numpy operations per tick.

Graph: the oil source (sump or trough) is joined to every cylinder's
bore bottom by an "oil-splash-path" edge -- a real, visible path that
the fluid-circuit solver does not treat as a line (participation is
computed here, not as pipe flow), and the reason a splash-bath
engine legitimately has no pump: the crankcase IS the pump.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

OIL_DENSITY_KG_L = 0.87
SPLASH_DEPOSIT_KG_PER_S_PER_RAD_S = 2.0e-6      # per cylinder at full sump, per rad/s of crank
FILM_SCRAPE_PER_RAD = 0.004                      # fraction of film scraped down per crank radian
FILM_BURN_PER_S_AT_FULL_LOAD = 0.006             # fraction of film burned per second at full combustion
BLOWBY_KG_PER_FIRE_PER_LITRE = 1.5e-6            # per litre of cylinder, per firing event, at full strength
FUEL_DILUTION_KG_PER_S_PER_RICH = 4.0e-6         # per cylinder, per unit richness beyond stoich, when cold
DILUTION_BOILOFF_PER_S = 1.0 / 900.0             # once the oil is past ~360 K
BREATHER_CONDUCTANCE_KG_PER_S_PER_PA = 2.0e-8
SEAL_LOSS_KG_PER_S_PER_PA = 5.0e-11
TOL_RING_SEAL_FRAC = 0.06


def _seed(identity: str) -> int:
    return int(hashlib.sha1((identity + ":rings").encode()).hexdigest()[:8], 16)


@dataclass
class CrankcaseState:
    n_cyl: int
    cylinder_litres: float
    ring_leak: np.ndarray                 # per cylinder, 0 tight .. ~0.3 worn (machine error + wear)
    film_kg: np.ndarray = field(default=None)
    oil_kg: float = 4.0
    oil_capacity_kg: float = 4.0
    fuel_in_oil_kg: float = 0.0
    case_gas_kg: float = 0.0
    breather_open: bool = True
    identity: str = ""
    # last-tick rates for reporting
    burn_kg_s: np.ndarray = field(default=None)
    blowby_kg_s: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.film_kg is None:
            self.film_kg = np.full(self.n_cyl, 0.0004)
        if self.burn_kg_s is None:
            self.burn_kg_s = np.zeros(self.n_cyl)
        if self.blowby_kg_s is None:
            self.blowby_kg_s = np.zeros(self.n_cyl)

    @classmethod
    def from_engine(cls, engine, oil_capacity_l: float | None = None, seed: int | None = None) -> "CrankcaseState":
        arch = engine.architecture
        n = max(1, int(arch.cylinders or 1))
        rng = np.random.default_rng(_seed(engine.identity) if seed is None else seed)
        ring = np.clip(0.03 + TOL_RING_SEAL_FRAC * np.abs(rng.standard_normal(n)) * 0.5, 0.01, 0.3)
        cap_l = oil_capacity_l if oil_capacity_l is not None else max(0.5, engine.displacement_l * 1.1)
        return cls(n_cyl=n, cylinder_litres=engine.displacement_l / n, ring_leak=ring,
                   oil_kg=cap_l * OIL_DENSITY_KG_L, oil_capacity_kg=cap_l * OIL_DENSITY_KG_L, identity=engine.identity)

    @property
    def crankcase_pressure_pa(self) -> float:
        # ideal gas in a case volume of ~2x displacement: p - p_atm from the gas that has not vented
        vol_m3 = max(self.cylinder_litres * self.n_cyl * 2.0, 0.2) / 1000.0
        return self.case_gas_kg * 287.0 * 350.0 / vol_m3

    @property
    def oil_level_frac(self) -> float:
        return self.oil_kg / max(self.oil_capacity_kg, 1e-9)

    @property
    def dilution_frac(self) -> float:
        return self.fuel_in_oil_kg / max(self.oil_kg + self.fuel_in_oil_kg, 1e-9)

    def step(self, dt: float, rpm: float, strengths: np.ndarray, richness: float, oil_temp_k: float,
             fires_per_s: float) -> None:
        """One tick. `strengths` is each cylinder's combustion strength
        this tick (0..1, the piston loop's own per-cylinder figure);
        `fires_per_s` the engine's firing rate."""
        omega = rpm * 2.0 * np.pi / 60.0
        level = self.oil_level_frac
        # film: d/dt film = deposit - (scrape + burn) * film -> exact
        # exponential relaxation toward deposit/(scrape+burn), so any
        # step size (a 2 ms sim tick or an hour of accelerated wear)
        # lands on the same physics
        deposit = SPLASH_DEPOSIT_KG_PER_S_PER_RAD_S * omega * min(1.0, level * 1.25)
        k_scrape = FILM_SCRAPE_PER_RAD * omega
        k_burn = FILM_BURN_PER_S_AT_FULL_LOAD * np.clip(strengths, 0.0, 1.5) * (1.0 + 4.0 * self.ring_leak)
        k = k_scrape + k_burn
        film0 = self.film_kg
        with np.errstate(divide="ignore", invalid="ignore"):
            steady = np.where(k > 0.0, deposit / np.maximum(k, 1e-12), film0)
            decay = np.exp(-k * dt)
        film1 = np.where(k > 0.0, steady + (film0 - steady) * decay, film0 + deposit * dt)
        # the burn over the step is the mean film times the burn rate
        film_mean = np.where(k > 0.0, steady + (film0 - steady) * (1.0 - decay) / np.maximum(k * dt, 1e-12), film0)
        self.film_kg = np.maximum(0.0, film1)
        self.burn_kg_s = k_burn * film_mean
        # blow-by in, breather out: d/dt gas = inflow - lam * gas, exact
        self.blowby_kg_s = BLOWBY_KG_PER_FIRE_PER_LITRE * self.cylinder_litres * fires_per_s / self.n_cyl             * np.clip(strengths, 0.0, 1.5) * (1.0 + 6.0 * self.ring_leak)
        inflow = float(self.blowby_kg_s.sum())
        vol_m3 = max(self.cylinder_litres * self.n_cyl * 2.0, 0.2) / 1000.0
        lam = (BREATHER_CONDUCTANCE_KG_PER_S_PER_PA * 287.0 * 350.0 / vol_m3) if self.breather_open else 0.0
        gas0 = self.case_gas_kg
        if lam > 0.0:
            g_steady = inflow / lam
            self.case_gas_kg = g_steady + (gas0 - g_steady) * np.exp(-lam * dt)
            gas_mean = g_steady + (gas0 - g_steady) * (1.0 - np.exp(-lam * dt)) / (lam * dt)
        else:
            self.case_gas_kg = gas0 + inflow * dt
            gas_mean = gas0 + inflow * dt / 2.0
        p_mean = gas_mean * 287.0 * 350.0 / vol_m3
        seal_loss = SEAL_LOSS_KG_PER_S_PER_PA * max(0.0, p_mean - 2000.0)
        self.oil_kg = max(0.0, self.oil_kg - (float(self.burn_kg_s.sum()) + seal_loss) * dt)
        cold = max(0.0, (330.0 - oil_temp_k) / 40.0)
        self.fuel_in_oil_kg += FUEL_DILUTION_KG_PER_S_PER_RICH * self.n_cyl * max(0.0, richness - 1.0) * (0.3 + cold) * dt
        if oil_temp_k > 360.0:
            self.fuel_in_oil_kg *= float(np.exp(-DILUTION_BOILOFF_PER_S * dt))

    def top_up(self, litres: float) -> None:
        self.oil_kg = min(self.oil_capacity_kg, self.oil_kg + litres * OIL_DENSITY_KG_L)

    def change_oil(self) -> None:
        self.oil_kg = self.oil_capacity_kg
        self.fuel_in_oil_kg = 0.0

    def oil_burn_frac(self) -> np.ndarray:
        """Per cylinder 0..1: how hard oil is burning relative to a
        badly worn engine -- the valve carbon source term."""
        return np.clip(self.burn_kg_s / 2.0e-5, 0.0, 1.0)   # ~80 mL/h per cylinder = badly worn

    def report(self) -> list[str]:
        out = [f"sump {self.oil_kg / OIL_DENSITY_KG_L:.2f} L ({self.oil_level_frac * 100:.0f}%), fuel dilution {self.dilution_frac * 100:.2f}%, "
               f"case {self.crankcase_pressure_pa / 1000:.2f} kPa, consumption {self.burn_kg_s.sum() * 3600 / OIL_DENSITY_KG_L * 1000:.1f} mL/h, "
               f"blow-by {self.blowby_kg_s.sum() / 1.2 * 60000:.2f} L/min"]
        for c in range(self.n_cyl):
            out.append(f"  cyl {c + 1}: film {self.film_kg[c] * 1e6:6.0f} mg  rings leak {self.ring_leak[c] * 100:.1f}%  burn {self.burn_kg_s[c] * 3.6e6 / OIL_DENSITY_KG_L:.2f} mL/h")
        return out
