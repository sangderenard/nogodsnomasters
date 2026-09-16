"""Emitters carry heat, not only mass.

A hole that vents gas is moving enthalpy as surely as it is moving
kilograms, and where that enthalpy lands is a real effect: a burst line
in a sealed room warms the room, a cryogenic vent chills it, and a
carbon-dioxide discharge does both in the wrong order. This puts the
temperature on the stream and mixes it into whatever receives it.

THE PART THAT IS USUALLY GOT BACKWARDS. "Decompression cooling" sounds
like it happens at the hole. It mostly does not.

A free jet through an orifice into the atmosphere is adiabatic and
extracts no work, so its TOTAL enthalpy is unchanged. The gas at the
throat really is colder -- a choked orifice sits at 83% of source
temperature for a diatomic gas -- but that is static temperature, bought
entirely with velocity, and the jet gives it straight back as it slows
down. Stand in front of it and the stagnation temperature you measure is
very close to the temperature in the pipe.

What is left at the hole is the JOULE-THOMSON effect, a real-gas
departure of about a fifth of a kelvin per bar for air at ordinary
conditions. On a ten-bar line that is two kelvin and not worth
dramatising. On a two-hundred-bar cylinder it is NOT small, and treating
the coefficient as a constant over that range is wrong twice over --
mu_JT falls steeply as pressure rises, so a linear extrapolation from a
one-bar figure badly overstates the drop. The coefficient here therefore
carries a pressure dependence, and past a few hundred bar it should be
read as indicative rather than quantitative.

It also goes the OTHER WAY for hydrogen and helium, which warm on
throttling because they sit above their inversion temperature at
anything like room conditions.

THE BIG EFFECT IS AT THE SOURCE, AND IT IS BIG. The vessel that is
emptying cools itself, because the gas remaining inside expands and does
work pushing the rest out. That is nearly isentropic and it follows the
pressure ratio to the power (gamma-1)/gamma -- so a vessel blowing down
from ten bar to one loses nearly half its absolute temperature. This is
why a scuba cylinder gets cold, why a CO2 extinguisher frosts over, and
why a line that has been venting for a minute delivers much colder gas
than the same line in its first second.

So the honest arrangement is: the SOURCE cools as it empties, the stream
carries whatever temperature the source had when it left plus a small
throttling term, and the RECEIVER mixes it in by heat capacity. All three
of those are cheap, and the whole thing is one matrix multiply per step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as _np

ATM_PA = 101_325.0

# ---------------------------------------------------------------------
# species properties, as columns so the whole field mixes at once
# ---------------------------------------------------------------------
#
# Ordered arrays rather than a dict of scalars, because the operation
# this module exists to do is "deposit N streams into M volumes" and
# that is a reduction, not a loop.

SPECIES: tuple = ("air", "nitrogen", "oxygen", "argon", "carbon-dioxide",
                  "hydrogen", "helium", "methane", "steam", "exhaust")

#: Specific heat at constant pressure, J/kg.K.
CP_J_KGK = _np.array([1005.0, 1040.0, 918.0, 520.0, 846.0,
                      14300.0, 5193.0, 2220.0, 1996.0, 1150.0])
#: Ratio of specific heats. Sets how hard a vessel cools as it empties.
GAMMA = _np.array([1.400, 1.400, 1.395, 1.667, 1.289,
                   1.410, 1.667, 1.306, 1.330, 1.350])
#: Joule-Thomson coefficient, K per bar, near room temperature.
#: NEGATIVE means the gas WARMS when throttled -- hydrogen and helium
#: are above their inversion temperature at ordinary conditions, which
#: is the fact that makes hydrogen liquefaction hard.
MU_JT_K_PER_BAR = _np.array([0.22, 0.22, 0.29, 0.43, 1.10,
                             -0.03, -0.06, 0.44, 0.00, 0.15])
#: Specific gas constant, J/kg.K.
R_J_KGK = _np.array([287.0, 296.8, 259.8, 208.1, 188.9,
                     4124.0, 2077.0, 518.3, 461.5, 290.0])

_INDEX = {s: i for i, s in enumerate(SPECIES)}


def species_index(name: str) -> int:
    i = _INDEX.get(str(name))
    if i is None:
        raise KeyError(f"unknown species {name!r}; declared: {', '.join(SPECIES)}")
    return i


def indices(names) -> _np.ndarray:
    """Vector of indices for a list of species names."""
    return _np.array([species_index(n) for n in names], dtype=_np.intp)


# ---------------------------------------------------------------------
# the three temperature effects, each on the whole field at once
# ---------------------------------------------------------------------

#: Pressure at which the Joule-Thomson coefficient has fallen to about
#: half its low-pressure value. Real mu_JT drops as the gas is squeezed
#: -- the attractive intermolecular term that causes the effect is
#: progressively beaten by the repulsive one -- so integrating a
#: constant coefficient over a cylinder's worth of pressure is a
#: substantial overestimate.
JT_HALF_PRESSURE_PA = 120e5


def throttled_temp_k(source_k, source_pa, ambient_pa=ATM_PA, idx=None):
    """Temperature the stream ARRIVES at, after Joule-Thomson.

    Isenthalpic throttling, which is the only temperature change a free
    jet actually keeps. Integrated with a falling coefficient rather than
    a constant one, because over a big pressure drop the difference is
    tens of kelvin."""
    t = _np.asarray(source_k, dtype=float)
    p1 = _np.asarray(source_pa, dtype=float)
    p0 = _np.asarray(ambient_pa, dtype=float)
    mu = MU_JT_K_PER_BAR[idx] if idx is not None else MU_JT_K_PER_BAR[0]
    # integral of mu0 / (1 + P/Ph) dP, which is mu0 * Ph * ln(...)
    ph = JT_HALF_PRESSURE_PA
    drop_bar = (ph / 1e5) * _np.log((ph + _np.maximum(p0, p1))
                                    / (ph + _np.minimum(p0, p1)))
    return _np.maximum(1.0, t - mu * drop_bar)


def choked_static_temp_k(source_k, idx=None):
    """Static temperature AT THE THROAT of a choked orifice.

    Reported because it is what the metal at the hole actually sees, and
    it is what frosts a fitting -- but it is NOT what the receiving
    volume gets, because the jet re-stagnates. Using this as the delivery
    temperature is the commonest way to overstate vent cooling."""
    g = GAMMA[idx] if idx is not None else GAMMA[0]
    return _np.asarray(source_k, dtype=float) * (2.0 / (g + 1.0))


def blowdown_temp_k(initial_k, pressure_ratio, idx=None, isentropic_eff=0.92):
    """What is LEFT IN THE VESSEL after it blows down. The big one.

    Near-isentropic expansion of the gas that stays behind, which is
    doing the work of pushing the rest out. A vessel going from ten bar
    to one loses close to half its absolute temperature, and every
    kilogram that leaves after that leaves cold."""
    g = GAMMA[idx] if idx is not None else GAMMA[0]
    r = _np.maximum(1e-6, _np.asarray(pressure_ratio, dtype=float))
    ideal = _np.asarray(initial_k, dtype=float) * r ** ((g - 1.0) / g)
    # real vessels are not adiabatic: the walls give heat back
    return ideal + (_np.asarray(initial_k, dtype=float) - ideal) * (1.0 - isentropic_eff)


# ---------------------------------------------------------------------
# receiving volumes
# ---------------------------------------------------------------------

@dataclass
class ThermalVolume:
    """Somewhere a stream can be deposited: a bay, a room, a pit's air.

    Holds a heat capacity and a temperature, and mixes by energy. It also
    tracks what SPECIES have arrived, because a volume that has been
    filled with nitrogen is a different problem from a warm one and the
    two questions share a bookkeeping."""
    identity: str = "bay.air"
    volume_m3: float = 60.0
    temp_k: float = 293.15
    #: Mass of gas in it. Air at 1.2 kg/m3 unless told otherwise.
    mass_kg: float = 0.0
    cp_j_kgk: float = 1005.0
    #: Thermal mass of the STRUCTURE the gas is in contact with. A room
    #: is not just its air: the walls, floor and everything standing in
    #: it hold far more heat than the air does, and they are why a
    #: draught of hot gas does not simply raise the room by the number
    #: an air-only calculation gives.
    structure_kg: float = 0.0
    structure_cp_j_kgk: float = 500.0
    #: Coupling between gas and structure, W/K. Finite, so a fast
    #: transient really does swing the air before the walls catch up.
    structure_ua_w_k: float = 200.0
    structure_k: float = 293.15
    species_kg: dict = field(default_factory=dict)
    sealed: bool = False
    air_changes_per_hour: float = 0.5

    def __post_init__(self):
        if self.mass_kg <= 0.0:
            self.mass_kg = self.volume_m3 * 1.204

    @property
    def heat_capacity_j_k(self) -> float:
        return max(1.0, self.mass_kg * self.cp_j_kgk)

    def deposit(self, mass_kg: float, temp_k: float, cp_j_kgk: float = 1005.0,
                species: str = "air") -> dict:
        """Mix one stream in. Energy balance, not a temperature average.

        Averaging temperatures is wrong whenever the heat capacities
        differ, and here they differ by a factor of fourteen between
        hydrogen and argon."""
        m = max(0.0, float(mass_kg))
        if m <= 0.0:
            return {"temp_k": self.temp_k, "delta_k": 0.0}
        before = self.temp_k
        c_room = self.heat_capacity_j_k
        c_in = m * float(cp_j_kgk)
        self.temp_k = (c_room * self.temp_k + c_in * float(temp_k)) / (c_room + c_in)
        self.species_kg[species] = self.species_kg.get(species, 0.0) + m
        if self.sealed:
            self.mass_kg += m
        return {"temp_k": self.temp_k, "delta_k": self.temp_k - before,
                "energy_j": c_in * (float(temp_k) - before)}

    def deposit_batch(self, mass_kg, temp_k, idx, species_names=None) -> dict:
        """Mix MANY streams in one operation.

        The whole reason this module keeps species as arrays. Order does
        not matter to the result because the mix is a single energy
        balance over all streams at once -- which is both faster and more
        correct than folding them in one at a time, since sequential
        mixing quietly makes the first stream matter more."""
        m = _np.maximum(0.0, _np.asarray(mass_kg, dtype=float))
        if m.size == 0 or not _np.any(m > 0.0):
            return {"temp_k": self.temp_k, "delta_k": 0.0, "mass_kg": 0.0}
        t = _np.asarray(temp_k, dtype=float)
        cp = CP_J_KGK[_np.asarray(idx, dtype=_np.intp)]
        c_in = m * cp
        before = self.temp_k
        c_room = self.heat_capacity_j_k
        self.temp_k = float((c_room * self.temp_k + float(_np.dot(c_in, t)))
                            / (c_room + float(c_in.sum())))
        total = float(m.sum())
        if species_names is not None:
            for name, kg in zip(species_names, m):
                if kg > 0.0:
                    self.species_kg[name] = self.species_kg.get(name, 0.0) + float(kg)
        if self.sealed:
            self.mass_kg += total
        return {"temp_k": self.temp_k, "delta_k": self.temp_k - before,
                "mass_kg": total, "streams": int(m.size)}

    def settle(self, dt_s: float, ambient_k: float = 293.15) -> dict:
        """Let the room come back: structure first, then ventilation.

        Without this a bay that was flooded with hot exhaust stays hot
        for ever, and the structure term is what makes the recovery have
        the right SHAPE -- a fast swing followed by a slow return."""
        dt = max(0.0, float(dt_s))
        c_gas = self.heat_capacity_j_k
        c_str = max(1.0, self.structure_kg * self.structure_cp_j_kgk)
        q = self.structure_ua_w_k * (self.temp_k - self.structure_k) * dt
        q = _clamp_exchange(q, self.temp_k, self.structure_k, c_gas, c_str)
        self.temp_k -= q / c_gas
        self.structure_k += q / c_str
        if not self.sealed and self.air_changes_per_hour > 0.0:
            frac = min(1.0, self.air_changes_per_hour * dt / 3600.0)
            self.temp_k += (float(ambient_k) - self.temp_k) * frac
            for k in list(self.species_kg):
                self.species_kg[k] *= (1.0 - frac)
        return {"temp_k": self.temp_k, "structure_k": self.structure_k}

    def oxygen_frac(self) -> float:
        """What the air in here is now, after everything vented into it.

        The reason species are tracked alongside temperature: a volume
        that has been thermally charged by a nitrogen vent has also been
        made unbreathable, and those are the same event."""
        inert = sum(kg for s, kg in self.species_kg.items()
                    if s in ("nitrogen", "argon", "helium", "carbon-dioxide", "methane"))
        base = max(1e-6, self.volume_m3 * 1.204)
        displaced = max(0.0, base - inert)
        added_o2 = self.species_kg.get("oxygen", 0.0)
        return (0.209 * displaced + added_o2) / max(1e-6, base + added_o2)


def _clamp_exchange(q: float, hot_k: float, cold_k: float,
                    c_hot: float, c_cold: float) -> float:
    """Never let an exchange overshoot equilibrium in one step.

    The bug this prevents is the classic explicit-integration one: a
    large dt drives the two temperatures past each other and the next
    step drives them back harder, and the whole thing rings itself
    apart."""
    if abs(hot_k - cold_k) < 1e-12:
        return 0.0
    q_eq = (hot_k - cold_k) / (1.0 / c_hot + 1.0 / c_cold)
    return max(-abs(q_eq), min(abs(q_eq), q)) if q_eq >= 0.0 else \
        max(-abs(q_eq), min(abs(q_eq), q))


# ---------------------------------------------------------------------
# harvesting the field
# ---------------------------------------------------------------------

@dataclass
class EmissionThermals:
    """One step's worth of thermal streams off an emitter field.

    Built as arrays so the deposit is one reduction. Nothing here loops
    over emitters doing arithmetic."""
    mass_kg: _np.ndarray
    temp_k: _np.ndarray
    idx: _np.ndarray
    names: tuple = ()

    @property
    def total_kg(self) -> float:
        return float(self.mass_kg.sum()) if self.mass_kg.size else 0.0

    @property
    def enthalpy_j(self) -> float:
        """Energy carried, relative to 273.15 K. Signed the obvious way:
        a cryogenic stream is negative."""
        if not self.mass_kg.size:
            return 0.0
        cp = CP_J_KGK[self.idx]
        return float(_np.dot(self.mass_kg * cp, self.temp_k - 273.15))


def emission_thermals(field, dt: float, default_species: str = "air",
                      species_of=None, ambient_pa: float = ATM_PA) -> EmissionThermals:
    """Pull the thermal streams out of a HoleEmitterField for one step.

    Reads what the emitters already track -- mass flow and source
    temperature -- and adds the throttling term. It does not modify the
    field, so it is safe to call after the field's own step."""
    masses, temps, idxs, names = [], [], [], []
    for em in getattr(field, "emitters", ()):
        m = float(getattr(em, "mass_flow_kg_s", 0.0)) * max(0.0, dt)
        if m <= 0.0 or getattr(em, "regime", "none") not in ("gas", "spray", "pour"):
            continue
        sp = default_species
        if species_of is not None:
            sp = species_of(em) or default_species
        elif getattr(em, "circuit", None) == "exhaust":
            sp = "exhaust"
        i = species_index(sp)
        src_k = float(getattr(em, "gas_temp_k", 293.15))
        src_pa = float(getattr(em, "source_pressure_pa", ambient_pa))
        masses.append(m)
        temps.append(float(throttled_temp_k(src_k, src_pa, ambient_pa, i)))
        idxs.append(i)
        names.append(sp)
    if not masses:
        e = _np.empty(0, dtype=float)
        return EmissionThermals(e, e, _np.empty(0, dtype=_np.intp), ())
    return EmissionThermals(_np.array(masses, dtype=float),
                            _np.array(temps, dtype=float),
                            _np.array(idxs, dtype=_np.intp),
                            tuple(names))


def vent_into(volume: ThermalVolume, thermals: EmissionThermals) -> dict:
    """Deposit a whole step's emissions into one volume, in one go."""
    return volume.deposit_batch(thermals.mass_kg, thermals.temp_k,
                                thermals.idx, thermals.names)


@dataclass
class BlowdownSource:
    """A vessel that cools itself as it empties.

    Carried separately from the emitter because it is a property of the
    SOURCE, not of the hole -- two holes in one vessel share this, and
    the vessel cools according to the total leaving through both."""
    identity: str = "vessel"
    species: str = "air"
    volume_m3: float = 0.05
    pressure_pa: float = 2.0e6
    temp_k: float = 293.15
    #: The vessel's own walls, which fight the cooling and are the reason
    #: a real cylinder does not reach the isentropic temperature.
    wall_kg: float = 8.0
    wall_cp_j_kgk: float = 470.0
    wall_ua_w_k: float = 12.0
    wall_k: float = 293.15

    @property
    def idx(self) -> int:
        return species_index(self.species)

    @property
    def gas_kg(self) -> float:
        r = float(R_J_KGK[self.idx])
        return self.pressure_pa * self.volume_m3 / (r * max(1.0, self.temp_k))

    def vent(self, mass_kg: float, dt_s: float = 0.0) -> dict:
        """Take mass out and let the remainder cool by expanding.

        Returns the temperature the departing gas HAD, which is what the
        receiving volume should be given -- not the vessel's new, colder
        temperature, because that gas has already gone."""
        held = self.gas_kg
        m = min(held * 0.999, max(0.0, float(mass_kg)))
        if m <= 0.0 or held <= 0.0:
            return {"vented_kg": 0.0, "delivered_k": self.temp_k,
                    "vessel_k": self.temp_k, "pressure_pa": self.pressure_pa}
        delivered_k = self.temp_k
        g = float(GAMMA[self.idx])
        # the gas that stays behind expands from held to (held - m)
        ratio = max(1e-6, (held - m) / held)
        self.temp_k = max(20.0, self.temp_k * ratio ** (g - 1.0))
        r = float(R_J_KGK[self.idx])
        self.pressure_pa = max(1.0, (held - m) * r * self.temp_k / self.volume_m3)
        if dt_s > 0.0:
            c_gas = max(1.0, (held - m) * float(CP_J_KGK[self.idx]))
            c_wall = max(1.0, self.wall_kg * self.wall_cp_j_kgk)
            q = self.wall_ua_w_k * (self.wall_k - self.temp_k) * dt_s
            q = _clamp_exchange(q, self.wall_k, self.temp_k, c_wall, c_gas)
            self.temp_k += q / c_gas
            self.wall_k -= q / c_wall
            self.pressure_pa = max(1.0, (held - m) * r * self.temp_k / self.volume_m3)
        return {"vented_kg": m, "delivered_k": delivered_k,
                "vessel_k": self.temp_k, "wall_k": self.wall_k,
                "pressure_pa": self.pressure_pa,
                "frosting": self.wall_k < 273.15}
