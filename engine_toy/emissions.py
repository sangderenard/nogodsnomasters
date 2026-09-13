"""What comes out of the pipe, and what it does to the people near it.

Engine-out CO / HC / NOx from the real mixture (equivalence ratio),
load and fuel flow the sim already computes; a catalytic converter as
a REAL part (engines.CatalyticConverter: a substrate sized off the
displacement, a real precious-metal loading with a real scrap value,
a brick that has to light off before it converts anything and that
only converts what the mixture lets it); and the toxicology of the
result -- carboxyhaemoglobin uptake in whoever is standing in the
volume the tailpipe empties into (air_volumes.py), by the Stewart
relation used in real CO-exposure assessment.

Every curve here is the standard textbook shape (Heywood-style engine-
out trends against phi; a light-off sigmoid and a lambda window for a
three-way brick; the Stewart COHb equation), not tuned to look good.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

CO_MOLAR_MASS = 28.01
EXHAUST_MOLAR_MASS = 28.9            # close to air for a lean/stoich burn
CO_KG_PER_M3_PPM = 1.145e-6          # 1 ppm CO = 1.145 mg/m^3 at 25 C, 1 atm
CO_HEAT_OF_OXIDATION_J_PER_KG = 10.1e6
HC_HEAT_OF_OXIDATION_J_PER_KG = 43.0e6


@dataclass(frozen=True)
class EngineOutRates:
    co_kg_s: float
    hc_kg_s: float
    nox_kg_s: float
    exhaust_kg_s: float
    o2_frac_in_exhaust: float        # volume fraction of oxygen left in the exhaust gas


def engine_out(exhaust_kg_s: float, fuel_kg_s: float, phi: float, load_frac: float,
               compression_ignition: bool, misfire_frac: float = 0.0, crevice_factor: float = 1.0) -> EngineOutRates:
    """Engine-out pollutant mass rates. phi = fuel/air over stoichiometric
    (1.0 stoich, >1 rich). load_frac: manifold pressure fraction (NOx
    scales with charge density / peak temperature)."""
    ex = max(exhaust_kg_s, 0.0)
    if ex <= 0.0:
        return EngineOutRates(0.0, 0.0, 0.0, 0.0, 0.2095)
    if compression_ignition:
        # a diesel always burns lean overall: CO and HC are low, NOx
        # comes with load, and the leftover oxygen is real
        co_vol = 0.0003 + 0.0025 * max(0.0, phi - 0.7)
        hc_vol = 0.0001
        nox_vol = 0.0008 * max(0.2, load_frac)
        o2_left = max(0.02, 0.2095 * (1.0 - min(phi, 0.95)))
    else:
        rich = max(0.0, phi - 1.0)
        lean = max(0.0, 1.0 - phi)
        co_vol = 0.0025 + 0.30 * rich + (0.0010 if lean > 0.0 else 0.0)
        hc_vol = 0.0004 + 0.020 * rich + 0.030 * max(0.0, lean - 0.15) ** 1.5
        nox_vol = 0.0025 * math.exp(-((phi - 0.9) / 0.12) ** 2) * max(0.15, load_frac)
        o2_left = 0.2095 * max(0.0, 1.0 - min(phi, 1.0)) * 0.9 + 0.005
    co = ex * co_vol * (CO_MOLAR_MASS / EXHAUST_MOLAR_MASS)
    hc = ex * hc_vol * (14.0 / EXHAUST_MOLAR_MASS) * max(0.2, crevice_factor)   # as CH2 units; the chamber's crevices
    nox = ex * nox_vol * (46.0 / EXHAUST_MOLAR_MASS)     # as NO2
    # a misfire pushes the whole unburnt charge out the port
    hc += max(0.0, min(1.0, misfire_frac)) * max(fuel_kg_s, 0.0)
    return EngineOutRates(co, hc, nox, ex, o2_left)


@dataclass
class CatalystState:
    """A real brick with real thermal inertia: it converts nothing until
    it is hot, and its own oxidation exotherm helps keep it hot."""
    brick_temp_k: float = 293.15
    co_efficiency: float = 0.0
    nox_efficiency: float = 0.0
    converted_co_kg: float = 0.0

    def step(self, dt: float, spec, inlet_temp_k: float, exhaust_kg_s: float, phi: float,
             rates: EngineOutRates) -> tuple[float, float, float]:
        """Returns (co, hc, nox) kg/s leaving the brick."""
        if spec is None:
            self.co_efficiency = 0.0; self.nox_efficiency = 0.0
            return rates.co_kg_s, rates.hc_kg_s, rates.nox_kg_s
        # light-off: a sigmoid across ~+-60 K around the brick's own
        # light-off temperature
        lit = 1.0 / (1.0 + math.exp(-(self.brick_temp_k - spec.light_off_k) / 25.0))
        if spec.kind == "three-way":
            # oxidation starves rich of oxygen; reduction starves lean of CO
            ox_window = 1.0 if phi <= 1.02 else max(0.3, 1.0 - (phi - 1.02) * 4.0)
            red_window = math.exp(-((phi - 1.0) / 0.035) ** 2) if phi < 1.0 else min(1.0, 0.85 + 0.15 * min(1.0, (phi - 1.0) / 0.05))
            co_eff = spec.peak_conversion * lit * ox_window
            nox_eff = spec.peak_conversion * lit * red_window
        else:   # diesel oxidation catalyst: oxidises, never reduces
            co_eff = spec.peak_conversion * lit
            nox_eff = 0.05 * lit
        self.co_efficiency = co_eff; self.nox_efficiency = nox_eff
        co_out = rates.co_kg_s * (1.0 - co_eff)
        hc_out = rates.hc_kg_s * (1.0 - co_eff)
        nox_out = rates.nox_kg_s * (1.0 - nox_eff)
        # thermal: gas heating toward inlet temp, plus the exotherm of
        # what it just burned, against the brick's real heat capacity
        exotherm_w = (rates.co_kg_s - co_out) * CO_HEAT_OF_OXIDATION_J_PER_KG + (rates.hc_kg_s - hc_out) * HC_HEAT_OF_OXIDATION_J_PER_KG
        gas_coupling_w_per_k = 1100.0 * max(exhaust_kg_s, 0.0) * 0.6      # cp_exhaust * flow * contact fraction
        loss_w_per_k = 4.0                                                 # shell convection to the bay
        c_brick = spec.brick_mass_kg * 900.0
        d_temp = ((inlet_temp_k - self.brick_temp_k) * gas_coupling_w_per_k
                  + exotherm_w - (self.brick_temp_k - 293.15) * loss_w_per_k) / max(c_brick, 1.0)
        self.brick_temp_k += d_temp * dt
        self.converted_co_kg += (rates.co_kg_s - co_out) * dt
        return co_out, hc_out, nox_out


# ---------------------------------------------------------------------
# the person in the room
# ---------------------------------------------------------------------

STEWART_COEFF = 3.317e-5              # %COHb per minute = k * ppm^1.036 * RMV(L/min) * t(min)
STEWART_EXPONENT = 1.036
COHB_HALF_LIFE_MIN_IN_AIR = 320.0     # elimination half-life breathing room air
COHB_LADDER = (
    (10.0, "headache, mild -- the first real sign"),
    (20.0, "throbbing headache, nausea, impaired judgement"),
    (30.0, "severe headache, dizziness, confusion"),
    (40.0, "collapse likely; unable to self-rescue"),
    (50.0, "coma, convulsions -- death without rescue"),
    (60.0, "fatal"),
)


@dataclass
class Occupant:
    """Carboxyhaemoglobin of one person breathing an AirVolume's CO."""
    cohb_pct: float = 0.0
    respiratory_minute_volume_l: float = 12.0    # light activity; 8 resting, 20+ working hard
    exposure_min: float = 0.0
    peak_ppm: float = 0.0

    def step(self, dt: float, co_ppm: float) -> None:
        minutes = dt / 60.0
        uptake = STEWART_COEFF * max(co_ppm, 0.0) ** STEWART_EXPONENT * self.respiratory_minute_volume_l * minutes
        elimination = self.cohb_pct * (1.0 - 0.5 ** (minutes / COHB_HALF_LIFE_MIN_IN_AIR))
        self.cohb_pct = max(0.0, min(100.0, self.cohb_pct + uptake - elimination))
        self.exposure_min += minutes if co_ppm > 50.0 else 0.0
        self.peak_ppm = max(self.peak_ppm, co_ppm)

    @property
    def condition(self) -> str:
        out = "no symptoms"
        for threshold, text in COHB_LADDER:
            if self.cohb_pct >= threshold:
                out = text
        return out

    def minutes_to_lethal(self, co_ppm: float, lethal_pct: float = 55.0) -> float | None:
        """Time at this ppm to reach a likely-fatal COHb from the current
        level (elimination ignored -- a conservative, real estimate).
        None when the ppm can't get there in a day."""
        rate = STEWART_COEFF * max(co_ppm, 0.0) ** STEWART_EXPONENT * self.respiratory_minute_volume_l
        if rate <= 0.0:
            return None
        minutes = (lethal_pct - self.cohb_pct) / rate
        return minutes if 0.0 <= minutes <= 1440.0 else None
