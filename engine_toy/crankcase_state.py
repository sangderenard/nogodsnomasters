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


# HOW MUCH OIL AN ENGINE ACTUALLY HOLDS.
#
# This was `displacement_l * 1.1` for every engine ever built, which is
# about right for a small petrol car and badly wrong for anything that
# works for a living: a Cat C18 is 18.1 litres of displacement and holds
# close to sixty litres of oil, not twenty. The ratio is not a constant,
# it is a property of what the engine is for --
#
#   petrol car        ~1 litre of oil per litre swept. A shallow sump,
#                     short drain intervals, and a sump designed around
#                     ground clearance more than oil life.
#   heavy diesel      ~3. A deep sump because the drain interval is
#                     measured in hundreds of hours and the oil has to
#                     carry soot and acid for all of it.
#   large/marine      ~4 and up, often with a separate tank; the oil is
#                     a system, not a fill.
#
# It matters well beyond a dipstick reading: every wear metal
# concentration is grams divided by this mass, so getting it wrong by
# three scales an entire oil analysis by three and makes a healthy
# engine read like a failing one.
# IT DOES NOT SCALE LINEARLY, and a ratio cannot be made to fit. A
# 1.6 litre Miata holds about 3.8 litres of oil -- well over twice its
# displacement -- while a 4.2 litre AMC six holds about 5.7, which is
# barely more than one. Sump volume grows much more slowly than swept
# volume, because a small engine still needs a usable depth of oil under
# the pickup, a pump that will not suck air on a corner, and enough mass
# to carry heat. Fitting those two real engines gives an exponent near
# 0.45, and the same exponent lands a modern 2.0 litre four at about
# 4.2 litres, which is what they hold.
#
# The coefficient is what the class changes: a heavy diesel carries far
# more oil for the same exponent, because its drain interval is hundreds
# of hours and the oil has to hold that much soot and acid for all of
# them.
#
# This matters well past a dipstick reading: every wear metal
# concentration is grams divided by this mass, so being wrong by three
# scales an entire oil analysis by three and makes a healthy engine read
# like a failing one.
OIL_CAPACITY_EXPONENT = 0.45
OIL_CAPACITY_COEFFICIENT = {
    "light-petrol": 3.1,       # fits the Miata at 3.8 L and the AMC six at 5.9 L
    "heavy-diesel": 14.0,      # fits the Cat C18 near its real ~57 L
    "large-slow-speed": 40.0,  # a circulating tank, not a sump: its own architecture
}


def derive_oil_capacity_l(engine) -> float:
    """Sump capacity in litres, from what kind of engine this is.

    An engine that declares `oil_capacity_l` is believed; everything
    else is classed by the things that actually decide a sump's depth --
    compression ignition and sheer size -- and scaled sub-linearly."""
    declared = float(getattr(engine, "oil_capacity_l", 0.0) or 0.0)
    if declared > 0.0:
        return declared
    disp = max(0.05, float(getattr(engine, "displacement_l", 0.0) or 0.0))
    if disp >= 100.0:
        k = OIL_CAPACITY_COEFFICIENT["large-slow-speed"]
    elif bool(getattr(engine, "compression_ignition", False)) or disp >= 8.0:
        k = OIL_CAPACITY_COEFFICIENT["heavy-diesel"]
    else:
        k = OIL_CAPACITY_COEFFICIENT["light-petrol"]
    return max(0.35, k * disp ** OIL_CAPACITY_EXPONENT)


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
        cap_l = (oil_capacity_l if oil_capacity_l is not None
                 else derive_oil_capacity_l(engine))
        return cls(n_cyl=n, cylinder_litres=engine.displacement_l / n, ring_leak=ring,
                   oil_kg=cap_l * OIL_DENSITY_KG_L, oil_capacity_kg=cap_l * OIL_DENSITY_KG_L, identity=engine.identity)

    # A worn ring pack leaks. `from_engine` draws `ring_leak` once, from
    # build scatter, and nothing moved it afterwards -- so an engine with
    # its rings 90% gone blew by exactly like a new one, and the wear
    # ledger that knew better was never consulted.
    #
    # The chain is real and it is short: the rings lose radial thickness
    # (which wear_debris.py now accounts for in grams), the pack stops
    # sealing, blow-by rises, and the charge that leaks past is charge
    # that never gets compressed -- so effective compression falls out of
    # the same number rather than being tracked separately.
    #
    # A thoroughly worn pack reaching about 30% leak is the top of the
    # range `from_engine` already clips to, so wear takes a ring from
    # wherever it was built to that same ceiling and no further.
    WORN_RING_LEAK = 0.30

    def apply_ring_wear(self, ring_damage_frac: float) -> None:
        """Open the rings up by how far gone they actually are."""
        d = max(0.0, min(1.0, float(ring_damage_frac)))
        if d <= 0.0:
            return
        base = getattr(self, "_built_ring_leak", None)
        if base is None:
            base = self.ring_leak.copy()
            self._built_ring_leak = base
        self.ring_leak = np.clip(base + (self.WORN_RING_LEAK - base) * d, 0.01, 1.0)

    #: A recessed seat leaks past the VALVE, which is a different path
    #: from past the rings and has a different fix -- the head comes off
    #: and the seats are re-cut, rather than a rebore. Tracked separately
    #: for that reason, and because a compression test that shows loss
    #: which a wet test does NOT restore is exactly how a mechanic tells
    #: the two apart.
    WORN_SEAT_LEAK = 0.22

    def apply_valve_seat_wear(self, seat_damage_frac: float) -> None:
        """Let a recessed seat stop sealing."""
        d = max(0.0, min(1.0, float(seat_damage_frac)))
        if not hasattr(self, "valve_leak"):
            self.valve_leak = np.zeros_like(self.ring_leak)
        self.valve_leak = np.clip(
            np.zeros_like(self.ring_leak) + self.WORN_SEAT_LEAK * d, 0.0, 1.0)

    @property
    def seat_leak_mean(self) -> float:
        v = getattr(self, "valve_leak", None)
        return 0.0 if v is None else float(np.mean(v))

    def effective_compression_ratio(self, geometric_ratio: float) -> float:
        """What the engine actually compresses to, past the rings AND
        past the valves.

        Charge that goes by the rings is charge that is not there at top
        dead centre, so the trapped mass falls with blow-by and the ratio
        the cylinder really achieves falls with it. This is the same leak
        number, read the other way round -- not a second wear model.

        A recessed valve seat loses charge the same way and adds to it,
        but through a path the rings never see. The two are combined as
        independent leaks rather than added, because neither can take
        more than what is left after the other."""
        leak = float(np.mean(self.ring_leak))
        seat = self.seat_leak_mean
        combined = 1.0 - (1.0 - leak) * (1.0 - seat)
        return max(1.5, float(geometric_ratio) * (1.0 - combined))

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
             fires_per_s: float, deposit_kg_s=None) -> None:
        """One tick. `strengths` is each cylinder's combustion strength
        this tick (0..1, the piston loop's own per-cylinder figure);
        `fires_per_s` the engine's firing rate. `deposit_kg_s`: the
        per-cylinder film deposit the splash emitters actually delivered
        (hole_emitters.HoleEmitterField.splash_deposit_kg_s -- the
        dipper's own flung flow, its retained fraction, per bore);
        None keeps the uniform crank-speed law below."""
        omega = rpm * 2.0 * np.pi / 60.0
        level = self.oil_level_frac
        # film: d/dt film = deposit - (scrape + burn) * film -> exact
        # exponential relaxation toward deposit/(scrape+burn), so any
        # step size (a 2 ms sim tick or an hour of accelerated wear)
        # lands on the same physics
        if deposit_kg_s is not None:
            deposit = np.asarray(deposit_kg_s, dtype=float)[:self.n_cyl]
            if len(deposit) < self.n_cyl:
                deposit = np.concatenate([deposit, np.zeros(self.n_cyl - len(deposit))])
        else:
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
