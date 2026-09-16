"""Storing cold: pits, ice, and how you get the ice in the first place.

A thermal battery is the cheapest trick in this whole plant. It stores
no energy in any useful sense -- you cannot drive anything with it --
but it lets a machine put heat somewhere that is not the sky, which is
the entire problem when the thing you are hiding from is looking for
heat.

WHY A PIT. Three reasons and they compound:

  ground coupling   deep soil sits near the MEAN ANNUAL temperature, not
                    the daily one. A desert that swings 40 C by day and
                    5 C by night is about 22 C two metres down, all year.
                    That is a nearly infinite sink at a temperature the
                    surface never sees.
  insulation        earth is mediocre insulation and there is an
                    enormous amount of it. A pit is surrounded on five
                    sides for free and only has to be capped.
  signature         a buried mass has no surface to radiate from. This
                    is the whole point: heat that goes into a pit does
                    not leave a thermal image.

THE PRECEDENT IS OLD. A yakhchal is a dug pit with an insulated cap
that held ice through a Persian desert summer, built from about four
hundred BC, using no power whatsoever. It worked on exactly the
mechanisms below, and the one that does the surprising work is
radiative sky cooling.

HOW THE ICE GETS MADE, and these are genuinely different machines:

  radiative     a shallow pan open to a clear night sky radiates
                through the atmospheric window at eight to thirteen
                micrometres, straight out to space at an effective
                temperature far below ambient. It will freeze water on
                a night that never drops below freezing. Free, silent,
                needs no power and no fuel -- and SLOW, because it is
                limited to a few tens of watts per square metre, so a
                tonne of ice wants a lot of pan.
  mechanical    a chiller run at night, when the ambient is lowest and
                so is its condensing temperature. Fast and reliable and
                it costs power, and running it is a thing that can be
                heard and seen.
  harvested     if it freezes outdoors, cut it and carry it in. Free,
                seasonal, and entirely at the mercy of the weather.

LATENT BEATS SENSIBLE BY AN ORDER OF MAGNITUDE. Chilling a tonne of
water ten degrees stores 42 MJ. FREEZING that tonne stores 334. Any
store that can be built around a phase change should be.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

WATER_CP_J_KGK = 4186.0
ICE_CP_J_KGK = 2100.0
ICE_LATENT_J_KG = 334_000.0
STEFAN_BOLTZMANN = 5.670374419e-8


# ---------------------------------------------------------------------
# what a store is made of
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class StorageMedium:
    key: str
    label: str
    density_kg_m3: float
    cp_j_kgk: float
    latent_j_kg: float            # 0 = sensible only
    phase_change_k: float         # meaningless when latent is 0
    why: str

    def energy_j_per_kg(self, delta_k: float, use_phase_change: bool = True) -> float:
        """What a kilogram holds over this temperature swing."""
        sensible = self.cp_j_kgk * abs(float(delta_k))
        return sensible + (self.latent_j_kg if (use_phase_change and self.latent_j_kg > 0) else 0.0)


MEDIA: dict[str, StorageMedium] = {
    "chilled-water": StorageMedium(
        "chilled-water", "chilled water", 1000.0, WATER_CP_J_KGK, 0.0, 0.0,
        why="the simplest store there is: a tank of cold water. Low density, no "
            "freezing to manage, and you can drink it"),
    "ice": StorageMedium(
        "ice", "water ice", 917.0, ICE_CP_J_KGK, ICE_LATENT_J_KG, 273.15,
        why="eight times the storage of chilling the same water ten degrees, because "
            "the phase change is where the energy is. Also still drinkable afterwards, "
            "which no other storage medium here can say"),
    "glaubers-salt": StorageMedium(
        "glaubers-salt", "sodium sulphate decahydrate", 1460.0, 1930.0, 254_000.0, 305.0,
        why="a salt hydrate melting at about 32 C: for storing WARMTH rather than cold, "
            "and it needs a nucleator or it supercools and never freezes when you want "
            "the heat back"),
    "paraffin-pcm": StorageMedium(
        "paraffin-pcm", "paraffin phase-change wax", 880.0, 2100.0, 200_000.0, 291.0,
        why="chemically dull, which is its virtue: it does not corrode anything, it does "
            "not supercool badly, and the melting point is chosen by picking the chain "
            "length. It also burns"),
    "soil": StorageMedium(
        "soil", "the ground itself", 1800.0, 1000.0, 0.0, 0.0,
        why="not a tank at all: the pit walls. Terrible per kilogram and there is an "
            "unlimited quantity of it, already in place, already buried"),
    "rock-bed": StorageMedium(
        "rock-bed", "packed rock bed", 2600.0, 900.0, 0.0, 0.0,
        why="for heat rather than cold, and the only one here that will take a high "
            "temperature without boiling or decomposing"),
}


def medium(key: str) -> StorageMedium:
    m = MEDIA.get(str(key))
    if m is None:
        raise KeyError(f"unknown storage medium {key!r}; declared: {', '.join(sorted(MEDIA))}")
    return m


# ---------------------------------------------------------------------
# making cold without power: the sky
# ---------------------------------------------------------------------

#: Net cooling a good broadband radiator achieves against a clear night
#: sky, in watts per square metre. A selective emitter tuned to the
#: eight-to-thirteen-micrometre atmospheric window does better; humidity
#: and cloud destroy it, which is why this is a desert technique.
SKY_COOLING_W_M2_CLEAR = 60.0
SKY_COOLING_W_M2_HUMID = 15.0


def sky_cooling_w(area_m2: float, clear: bool = True, cloud_frac: float = 0.0) -> float:
    """Radiative cooling to the night sky.

    The mechanism a yakhchal ran on: a clear desert sky has an effective
    radiative temperature tens of degrees below ambient, and a surface
    facing it loses heat straight out through the atmospheric window
    regardless of how warm the air is. It will freeze water on a night
    that never reaches freezing.

    Cloud closes the window. So does humidity. This is why it is a
    desert trick and not a general one."""
    base = SKY_COOLING_W_M2_CLEAR if clear else SKY_COOLING_W_M2_HUMID
    return max(0.0, float(area_m2)) * base * max(0.0, 1.0 - max(0.0, min(1.0, cloud_frac)))


def sky_pan_area_for(target_j: float, night_hours: float = 10.0,
                     clear: bool = True) -> float:
    """How much pan it takes to make this much cold in one night.

    The honest answer to "can we just use the sky": yes, and it wants
    area. This is the number that decides whether the idea is practical
    at the scale you had in mind."""
    per_m2 = sky_cooling_w(1.0, clear) * max(0.0, night_hours) * 3600.0
    return 0.0 if per_m2 <= 0.0 else max(0.0, float(target_j)) / per_m2


# ---------------------------------------------------------------------
# the pit
# ---------------------------------------------------------------------

#: Deep ground sits near the mean ANNUAL air temperature, whatever the
#: surface is doing. This is the whole reason to dig.
DESERT_DEEP_GROUND_K = 295.15          # about 22 C

#: WHAT THE GROUND IS WORTH AS INSULATION.
#:
#: There is no such thing as "the" conductivity of soil. The same ground
#: spans a factor of ten between oven-dry and saturated, and the desert
#: end of that range is the good end: dry sand is about 0.27 W/m.K, which
#: is 3.70 m2.K/W per metre -- R-21 imperial per metre of it, R-0.53 per
#: inch. Thin, per inch, against fibreglass at R-3.5; but nobody has one
#: inch of ground, they have metres of it, and a metre of dry sand is a
#: 2x6 insulated wall. The pit is not merely hidden by the ground, it is
#: genuinely insulated by it, and only because the ground is dry.
#:
#: This matters in the other direction too. Wet the ground -- a burst
#: line, a spill, rain finding the voids in hardpan spoil -- and the
#: walls get several times more conductive without anything visibly
#: changing. A cold store that quietly stops holding after a leak is a
#: real failure and it should be a reachable one here.
#:
#: The conductivity therefore lives on the soil class, with a saturation
#: figure, and the pit carries its own wetness.


@dataclass
class IcePit:
    """A dug, capped store: the station's cold bank.

    Ground on five sides and insulation on the sixth, which is both the
    cheapest insulation available and the only one that also hides the
    thermal signature of whatever is inside it."""
    identity: str = "station.ice_pit"
    depth_m: float = 3.0
    floor_area_m2: float = 12.0
    capacity_kg: float = 8000.0
    medium_key: str = "ice"
    stored_kg: float = 0.0           # mass in the CHARGED phase (frozen)
    water_kg: float = 0.0            # mass present but melted
    cap_media: str = "mineral-wool"
    cap_thickness_m: float = 0.30
    ground_k: float = DESERT_DEEP_GROUND_K
    soil: str = "hardpan"
    #: Degree of saturation of the surrounding ground, 0 to 1. Desert
    #: default is near-dry, which is why the pit works at all.
    soil_moisture: float = 0.02
    position: tuple = (0.0, -3.0, 0.0)

    @property
    def medium(self) -> StorageMedium:
        return medium(self.medium_key)

    @property
    def wall_area_m2(self) -> float:
        side = math.sqrt(max(0.5, self.floor_area_m2))
        return 4.0 * side * self.depth_m + self.floor_area_m2

    @property
    def stored_j(self) -> float:
        """Cold actually banked: the latent heat of what is frozen."""
        return self.stored_kg * self.medium.latent_j_kg

    @property
    def charge_frac(self) -> float:
        return 0.0 if self.capacity_kg <= 0.0 else min(1.0, self.stored_kg / self.capacity_kg)

    @property
    def soil_spec(self) -> "SoilClass":
        return soil_class(self.soil)

    @property
    def soil_k_w_mk(self) -> float:
        return self.soil_spec.conductivity_w_mk(self.soil_moisture)

    @property
    def equivalent_radius_m(self) -> float:
        """Radius of the sphere that conducts like this pit does.

        Ground is not a slab and there is no honest thickness to divide
        by -- heat spreads outward into a half-space until it reaches
        ground that is at the far-field temperature. The exact steady
        result for a sphere buried in an infinite medium is
        Q = 4.pi.k.r.dT, with no thickness in it at all, and a compact
        pit is close enough to a sphere for that to be the right shape
        of answer. This replaces an arbitrary division by two."""
        v = max(1e-3, self.floor_area_m2 * self.depth_m)
        return (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)

    @property
    def ground_r_si(self) -> float:
        """The ground's resistance, as an R-value against the buried area.

        Quoted this way so it can be compared directly with the cap
        insulation, which is the comparison that decides which of the two
        is worth spending on."""
        r_path = 1.0 / (4.0 * math.pi * max(1e-6, self.soil_k_w_mk)
                        * self.equivalent_radius_m)
        return r_path * self.wall_area_m2

    def heat_gain_w(self, surface_k: float = 313.15) -> float:
        """What leaks in: earth through the walls, and the cap.

        The walls see deep ground, not the surface, which is the point of
        the depth -- they are the small term. The cap faces the day and
        is the one that has to be insulated."""
        import cryogenics as cg
        store_k = self.medium.phase_change_k or 277.0
        buried_frac = self.wall_area_m2 / max(1e-6, self.wall_area_m2 + self.floor_area_m2)
        walls = (4.0 * math.pi * self.soil_k_w_mk * self.equivalent_radius_m
                 * max(0.0, self.ground_k - store_k) * buried_frac)
        cap = cg.wrap_heat_leak_w(self.cap_media, self.floor_area_m2,
                                  self.cap_thickness_m, store_k, surface_k)
        return walls + cap

    def melt(self, dt_s: float, surface_k: float = 313.15) -> float:
        """Standing loss: how much thaws while nobody is using it."""
        if self.stored_kg <= 0.0:
            return 0.0
        melted = min(self.stored_kg,
                     self.heat_gain_w(surface_k) * max(0.0, dt_s) / self.medium.latent_j_kg)
        self.stored_kg -= melted
        self.water_kg += melted
        return melted

    def draw_j(self, joules: float) -> float:
        """Take cold out of the bank -- dump heat into it. Returns what
        it could actually absorb."""
        want = max(0.0, float(joules))
        can = self.stored_j
        took = min(want, can)
        melted = took / self.medium.latent_j_kg
        self.stored_kg -= melted
        self.water_kg += melted
        return took

    def charge_j(self, joules: float) -> float:
        """Put cold in: freeze some of the meltwater back."""
        want = max(0.0, float(joules))
        room_kg = min(self.water_kg, max(0.0, self.capacity_kg - self.stored_kg))
        can = room_kg * self.medium.latent_j_kg
        took = min(want, can)
        frozen = took / self.medium.latent_j_kg
        self.stored_kg += frozen
        self.water_kg -= frozen
        return took

    def hold_hours(self, load_w: float) -> float:
        """How long the bank can absorb this load before it is water."""
        return 0.0 if load_w <= 0.0 else self.stored_j / float(load_w) / 3600.0

    def need(self):
        """A bank that has run down wants recharging before it is needed."""
        import servicing as sv
        if self.charge_frac >= 0.35:
            return None
        return sv.Need(
            identity=self.identity, want="recharge-cold", position=tuple(self.position),
            quantity=(self.capacity_kg - self.stored_kg), unit="kg",
            matches="freezing-capacity", urgency=0.35 / max(0.01, self.charge_frac),
            minutes=30.0, skill="operator",
            label=f"{self.identity} at {self.charge_frac * 100:.0f}% charge",
            why="a thermal bank is only useful full; an empty one is a hole with water "
                "in it and the heat has to go somewhere visible instead")


# ---------------------------------------------------------------------
# getting the ice in there
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class IceMaker:
    key: str
    label: str
    powered: bool
    cop: float                    # cold delivered per unit of work; 0 for unpowered
    why: str


ICE_MAKERS: dict[str, IceMaker] = {
    "sky-pan": IceMaker(
        "sky-pan", "radiative sky-cooling pan", False, 0.0,
        why="a shallow pan open to the night sky, shielded from the ground. No power, "
            "no noise, no signature of its own -- and limited to tens of watts per "
            "square metre, so it is a trickle charger and wants a great deal of area"),
    "night-chiller": IceMaker(
        "night-chiller", "vapour-compression chiller, run at night", True, 3.0,
        why="run when the ambient is lowest, because a chiller's efficiency is set by "
            "what it has to reject INTO. A tonne of ice a night on a few kilowatts, at "
            "the cost of running a machine that can be heard"),
    "cryo-bleed": IceMaker(
        "cryo-bleed", "cold-gas bleed from the cryogenic plant", False, 0.0,
        why="the cold nitrogen gas a dewar vents is worth more than the boiling was "
            "(see cryogenics): running it through the pit on its way to atmosphere "
            "recovers refrigeration that is otherwise thrown away"),
    "harvest": IceMaker(
        "harvest", "cut and carry natural ice", False, 0.0,
        why="the oldest answer. Free, seasonal, entirely at the mercy of the weather, "
            "and pure labour -- which is a real cost when there are two of you"),
}


def ice_maker(key: str) -> IceMaker:
    m = ICE_MAKERS.get(str(key))
    if m is None:
        raise KeyError(f"unknown ice maker {key!r}; declared: {', '.join(sorted(ICE_MAKERS))}")
    return m


def freeze_kg(joules: float) -> float:
    """Kilograms of ice a given amount of cold will make."""
    return max(0.0, float(joules)) / ICE_LATENT_J_KG


def chiller_cold_j(input_w: float, hours: float, cop: float = 3.0,
                   ambient_k: float = 288.15) -> float:
    """Cold delivered by a powered chiller over a run.

    COP is not a constant: it falls as the machine has to reject heat
    into something hotter, which is exactly why ice is made at night.
    A desert afternoon is the worst possible time to run one."""
    derate = max(0.4, min(1.3, 1.0 + (293.15 - float(ambient_k)) / 60.0))
    return max(0.0, float(input_w)) * max(0.0, float(hours)) * 3600.0 * cop * derate


# ---------------------------------------------------------------------
# THE HOLE HAS TO BE DUG
# ---------------------------------------------------------------------
#
# A pit is the cheapest thermal store there is and it is not free: it is
# paid for in advance, in labour, before it stores anything at all. That
# debt is the interesting part of the design rather than an obstacle to
# it -- an ice pit is a decision to spend a week of somebody's arms now
# so that months later there is somewhere to put heat.
#
# THE GROUND DECIDES EVERYTHING. A cubic metre of loose sand is half an
# hour with a shovel; the same cubic metre of desert hardpan is four
# hours and a pick, and caliche will break tools. So the same pit is a
# two-day job or a three-week one depending entirely on where it was
# sited, which makes siting a real decision rather than a cosmetic one.
#
# AND IT IS A DEFICIT, not a switch. Digging is interruptible and
# cumulative: whoever has time puts hours into the hole, the hole
# remembers them, and the store only works once the whole volume is out.
# Later, with a machine, the same hole is an afternoon -- which is the
# right shape for a thing that should feel expensive early and trivial
# later.

@dataclass(frozen=True)
class SoilClass:
    key: str
    label: str
    hand_m3_per_hour: float       # one person, pick and shovel
    machine_m3_per_hour: float    # a small excavator
    tool_wear_per_m3: float       # fraction of a tool's life per cubic metre
    why: str
    dry_k_w_mk: float = 1.0       # oven-dry conductivity
    sat_k_w_mk: float = 2.2       # every pore full of water
    coarse: bool = False          # sand/gravel grading, not silt/clay
    porosity: float = 0.35

    def conductivity_w_mk(self, moisture_frac: float = 0.0) -> float:
        """Johansen: k = k_dry + (k_sat - k_dry) * Ke.

        `moisture_frac` is DEGREE OF SATURATION -- the fraction of the
        pore space holding water, not the fraction of the soil that is
        water. The Kersten number is logarithmic because the first drops
        of water do most of the work: they bridge the grain contacts,
        and conduction through a point contact is what was limiting.
        Which is why the interesting range is 0 to 0.1 and everything
        above that is a slow climb."""
        sr = min(1.0, max(0.0, float(moisture_frac)))
        floor = 0.05 if self.coarse else 0.10
        if sr <= floor:
            ke = 0.0
        elif self.coarse:
            ke = max(0.0, 0.7 * math.log10(sr) + 1.0)
        else:
            ke = max(0.0, math.log10(sr) + 1.0)
        return self.dry_k_w_mk + (self.sat_k_w_mk - self.dry_k_w_mk) * min(1.0, ke)

    def r_per_metre_si(self, moisture_frac: float = 0.0) -> float:
        """m2.K/W per metre of it. This is the number to compare against
        a building material, and dry sand wins that comparison."""
        return 1.0 / max(1e-6, self.conductivity_w_mk(moisture_frac))

    def r_per_foot_imperial(self, moisture_frac: float = 0.0) -> float:
        """The R-number as a builder says it: ft2.F.h/Btu per foot."""
        return self.r_per_metre_si(moisture_frac) * 5.678263 * 0.3048


SOIL_CLASSES: dict[str, SoilClass] = {
    "loose-sand": SoilClass(
        "loose-sand", "loose sand", 1.20, 40.0, 0.002,
        why="fast to move and it will not stand up: a pit in sand needs shoring or it "
            "becomes a shallow bowl, which is its own work",
        dry_k_w_mk=0.27, sat_k_w_mk=2.75, coarse=True, porosity=0.40),
    "ordinary-soil": SoilClass(
        "ordinary-soil", "ordinary soil", 0.60, 30.0, 0.004,
        why="the reference: a fit person with a shovel moves something over half a "
            "cubic metre an hour, all day, and is very tired",
        dry_k_w_mk=0.35, sat_k_w_mk=1.90, coarse=False, porosity=0.38),
    "hardpan": SoilClass(
        "hardpan", "desert hardpan", 0.25, 18.0, 0.010,
        why="sun-baked and cemented. A pick job rather than a shovel job, and it is "
            "what most of a desert actually is once the surface sand is off",
        dry_k_w_mk=0.55, sat_k_w_mk=2.20, coarse=False, porosity=0.28),
    "caliche": SoilClass(
        "caliche", "caliche", 0.12, 9.0, 0.030,
        why="calcium carbonate cemented into something close to weak concrete. It "
            "blunts tools as fast as it yields, and it is the reason siting a pit "
            "matters more than sizing it",
        dry_k_w_mk=1.10, sat_k_w_mk=2.40, coarse=False, porosity=0.18),
    "rock": SoilClass(
        "rock", "rock", 0.04, 4.0, 0.060,
        why="not a digging job at all. Hand-cutting a pit in rock is measured in weeks "
            "and is usually the wrong answer",
        dry_k_w_mk=2.50, sat_k_w_mk=3.20, coarse=False, porosity=0.05),
}


def soil_class(key: str) -> SoilClass:
    s = SOIL_CLASSES.get(str(key))
    if s is None:
        raise KeyError(f"unknown soil {key!r}; declared: {', '.join(sorted(SOIL_CLASSES))}")
    return s


#: Digging is not sustainable at full rate for a whole day. Real earthwork
#: planning assumes a fraction of the clock is actually productive --
#: rest, water, heat, and the fact that a pit gets deeper and the spoil
#: has to go further.
DIG_DUTY_FRACTION = 0.55
#: Every metre of depth means lifting the spoil further. Real pits slow
#: down as they deepen.
DEPTH_PENALTY_PER_M = 0.08


@dataclass
class Excavation:
    """The debt a pit owes before it is a pit.

    Cumulative and interruptible: hours go in, the hole remembers them,
    and nothing works until the volume is out."""
    identity: str = "station.ice_pit.dig"
    required_m3: float = 36.0
    excavated_m3: float = 0.0
    soil: str = "hardpan"
    position: tuple = (0.0, 0.0, 0.0)
    depth_m: float = 3.0
    tool_condition: float = 1.0        # 1.0 new, 0.0 finished

    @property
    def spec(self) -> SoilClass:
        return soil_class(self.soil)

    @property
    def remaining_m3(self) -> float:
        return max(0.0, self.required_m3 - self.excavated_m3)

    @property
    def complete(self) -> bool:
        return self.remaining_m3 <= 1e-9

    @property
    def progress(self) -> float:
        return 0.0 if self.required_m3 <= 0.0 else min(1.0, self.excavated_m3 / self.required_m3)

    def rate_m3_per_hour(self, workers: int = 1, machine: bool = False) -> float:
        """How fast it actually comes out.

        Slows as it deepens, because the spoil has further to go, and a
        blunt tool digs slowly whatever the ground is."""
        s = self.spec
        base = s.machine_m3_per_hour if machine else s.hand_m3_per_hour * max(0, workers)
        depth_frac = self.progress * self.depth_m
        penalty = max(0.35, 1.0 - DEPTH_PENALTY_PER_M * depth_frac)
        tool = 1.0 if machine else max(0.25, self.tool_condition)
        return base * penalty * tool * (1.0 if machine else DIG_DUTY_FRACTION)

    def hours_remaining(self, workers: int = 1, machine: bool = False) -> float:
        r = self.rate_m3_per_hour(workers, machine)
        return float("inf") if r <= 0.0 else self.remaining_m3 / r

    def dig(self, hours: float, workers: int = 1, machine: bool = False) -> float:
        """Put somebody's time into the hole. Returns cubic metres moved."""
        moved = min(self.remaining_m3,
                    self.rate_m3_per_hour(workers, machine) * max(0.0, float(hours)))
        self.excavated_m3 += moved
        if not machine:
            self.tool_condition = max(0.0, self.tool_condition
                                      - moved * self.spec.tool_wear_per_m3)
        return moved

    def need(self):
        """An unfinished hole is a job, and a big one."""
        import servicing as sv
        if self.complete:
            return None
        hours = self.hours_remaining(workers=1, machine=False)
        return sv.Need(
            identity=self.identity, want="excavate", position=tuple(self.position),
            quantity=self.remaining_m3, unit="m3", matches="digging-tools",
            urgency=1.0, minutes=min(600.0, hours * 60.0), skill="labourer",
            label=f"{self.identity} ({self.spec.label}, {self.remaining_m3:.1f} m3 to go)",
            why="a thermal store is paid for before it stores anything: this is the "
                "week of somebody's arms that buys months of somewhere to put heat")

    def tool_need(self):
        """Blunt tools are their own job, and caliche makes them fast."""
        import servicing as sv
        if self.tool_condition > 0.3:
            return None
        return sv.Need(
            identity=f"{self.identity}.tools", want="tool-service",
            position=tuple(self.position), quantity=1.0, unit="each",
            matches="digging-tools", urgency=1.0 / max(0.05, self.tool_condition),
            minutes=25.0, skill="labourer",
            label=f"{self.identity} tools at {self.tool_condition * 100:.0f}%",
            why="caliche blunts a pick about as fast as it yields to one")


def pit_excavation_m3(floor_area_m2: float, depth_m: float,
                      batter: float = 1.15) -> float:
    """Spoil to move for a pit of this size.

    `batter` is the slope the walls need so they do not fall in, which
    is real volume nobody costs for until they are digging it."""
    return max(0.0, float(floor_area_m2)) * max(0.0, float(depth_m)) * max(1.0, batter)


# ---------------------------------------------------------------------
# ONE JOB PAYS FOR THE OTHER
# ---------------------------------------------------------------------
#
# The spoil from an ice pit is protective material, and a station does
# not need protecting equally in every direction. So the pit should be
# sized to the arc that actually matters -- the CRITICAL berm, the one
# covering the approach -- rather than to a perimeter nobody can afford.
# Dig exactly the hole whose spoil builds exactly that wall, and the two
# jobs are one job.
#
# What that buys is the real prize: once the ice house exists, the
# station has somewhere to put heat that is not the sky. An engine bay
# that vents into a pit is not radiating, and a position that is not
# radiating is not there.

def pit_sized_for_berm(berm, floor_depth_m: float = 3.0,
                       batter: float = 1.15) -> dict:
    """Size a pit so its spoil is exactly the critical berm.

    Returns the floor area to dig and what it will store, so the two
    decisions -- how much cover, how much cold -- are made as one."""
    need_m3 = max(0.0, getattr(berm, "required_m3", 0.0))
    depth = max(0.5, float(floor_depth_m))
    floor_area = need_m3 / (depth * max(1.0, batter))
    # ice fills the pit less than completely: freeboard, the cap, and
    # room to work in it
    usable_frac = 0.70
    ice_kg = floor_area * depth * usable_frac * 917.0
    return {
        "floor_area_m2": floor_area,
        "depth_m": depth,
        "excavation_m3": need_m3,
        "ice_capacity_kg": ice_kg,
        "stored_j": ice_kg * ICE_LATENT_J_KG,
        "note": "dug to yield exactly the critical berm's material",
    }


# ---------------------------------------------------------------------
# DRAWING ON IT WITHOUT KILLING IT
# ---------------------------------------------------------------------
#
# A pit full of ice is a bank, and a bank can be emptied. The vent that
# takes cold out of it has to be throttled to what the bank can stand,
# or the first hot day spends everything a week of digging bought.
#
# The proportioning valve is that throttle, and the number it is set to
# is not arbitrary: it is the RECHARGE rate. Draw at or below what the
# sky pan and the night chiller put back and the bank holds level
# indefinitely; draw above it and the pit is on a countdown that can be
# stated in days.

@dataclass
class ColdVent:
    """A pipe from the ice pit to the diffuser, with a valve on it.

    The cold air is worth more than its temperature suggests: mixed
    into a hot exhaust it drops the discharge temperature, and radiated
    power goes as the FOURTH POWER of that, so a modest dilution is a
    large reduction in what an infrared sensor sees."""
    identity: str = "station.cold_vent"
    pipe_area_m2: float = 0.020
    valve_open_frac: float = 1.0
    sustainable_w: float = 3_500.0     # what recharge can actually replace
    pit_air_k: float = 277.0           # air off the ice, a few degrees above freezing
    position: tuple = (0.0, -1.0, 0.0)

    def max_draw_w(self, respect_valve: bool = True) -> float:
        """What the valve will pass.

        Wide open it will happily take more than the bank can stand --
        which is why the setting matters and why it is the recharge rate
        that sets it."""
        if not respect_valve:
            return float("inf")
        return self.sustainable_w * max(0.0, min(1.0, self.valve_open_frac))

    def draw_w(self, demand_w: float, pit: "IcePit | None" = None,
               respect_valve: bool = True) -> dict:
        """Take cold for a heat load. Returns what was actually passed
        and whether the bank is being spent faster than it is filled."""
        want = max(0.0, float(demand_w))
        passed = min(want, self.max_draw_w(respect_valve))
        if pit is not None and pit.stored_j <= 0.0:
            passed = 0.0
        return {
            "demand_w": want,
            "passed_w": passed,
            "unmet_w": want - passed,
            "overdrawn": passed > self.sustainable_w * 1.001,
            "melt_kg_s": passed / ICE_LATENT_J_KG,
        }

    def mixed_exhaust_k(self, exhaust_k: float, exhaust_kg_s: float,
                        cold_kg_s: float, cp_j_kgk: float = 1005.0) -> float:
        """Discharge temperature after the cold air is blended in.

        A plain energy balance -- no cleverness, and none needed."""
        m1, m2 = max(0.0, float(exhaust_kg_s)), max(0.0, float(cold_kg_s))
        if m1 + m2 <= 0.0:
            return float(exhaust_k)
        return (m1 * float(exhaust_k) + m2 * self.pit_air_k) / (m1 + m2)


def ir_signature_ratio(hot_k: float, cooled_k: float, ambient_k: float = 313.15) -> float:
    """How much less a cooled discharge radiates, against ambient.

    Radiated power goes as the fourth power of absolute temperature, so
    cooling a plume is worth far more than the temperature drop looks.
    Measured as excess over the background, because that is what a
    sensor actually discriminates: a plume at ambient is invisible no
    matter how much of it there is."""
    hot = max(0.0, float(hot_k) ** 4 - float(ambient_k) ** 4)
    cold = max(0.0, float(cooled_k) ** 4 - float(ambient_k) ** 4)
    if hot <= 0.0:
        return 1.0
    return cold / hot


# ---------------------------------------------------------------------
# EVERYTHING TERMINATES HERE
# ---------------------------------------------------------------------
#
# Once the pit exists it stops being a store and becomes the station's
# heat sink, and the architecture follows: anything that would otherwise
# reject heat to atmosphere rejects it here instead. A radiator is a
# lamp to an infrared sensor; a coolant loop that terminates in a buried
# pit is not.
#
# COOLANT IS THE BIG ONE. It is the whole point of a coolant circuit
# that it carries energy away from something, and where it puts that
# energy down is normally a finned core in the open air. Terminating it
# in the ice instead moves the entire rejection underground -- which is
# the difference between a position with a warm rectangle on it and a
# position without.
#
# AND THE DEWAR VENTS IN HERE TOO, for the reason the cryogenics module
# gives: over half a cryogen's cooling is in the cold GAS, and venting
# that to atmosphere throws it away. Piping it through the pit recovers
# it.
#
# BUT BE HONEST ABOUT THE SCALE, AND "NOTHING" IS A NUMBER. A dewar's
# standing boil-off is well under a kilogram a day. At 0.4 kg/day the
# recovered gas is 0.96 W: against a 9 kW station load that is 0.011%,
# and over a day it is 83 kJ against the 785 MJ that went in. It is not
# zero and the ledger does not zero it -- it is four decimal places down,
# which is a different claim and a checkable one.
#
# The recovery becomes real when the plant is CONSUMING liquid nitrogen
# rather than merely holding it: boiling 5 kg/h off for a process is
# 290 W, three hundred times the standing figure and a fifth of the pit's
# own leak. The ledger keeps the two terms apart for exactly that reason,
# and `share_of_load` below says which case you are in instead of leaving
# it to a shrug.

N2_GAS_CP_J_KGK = 1040.0


@dataclass
class HeatBalance:
    """What goes into the bank and what comes back out of it.

    Every term is a rate in watts, signed the way it is felt: positive
    melts ice, negative freezes it."""
    coolant_w: float = 0.0        # a circuit terminating here
    exhaust_w: float = 0.0        # discharge routed through
    ambient_w: float = 0.0        # what leaks in through cap and walls
    cryo_recovery_w: float = 0.0  # cold gas piped in: negative, it helps
    sky_w: float = 0.0            # radiative pan: negative
    chiller_w: float = 0.0        # powered: negative

    @property
    def net_w(self) -> float:
        return (self.coolant_w + self.exhaust_w + self.ambient_w
                - abs(self.cryo_recovery_w) - abs(self.sky_w) - abs(self.chiller_w))

    @property
    def melting(self) -> bool:
        return self.net_w > 0.0

    def melt_kg_s(self) -> float:
        return max(0.0, self.net_w) / ICE_LATENT_J_KG

    def freeze_kg_s(self) -> float:
        return max(0.0, -self.net_w) / ICE_LATENT_J_KG

    def share_of_load(self, term_w: float) -> float:
        """What fraction of the gross heat in a term actually is.

        For saying "the vented dewar is 0.011% of this" rather than "the
        vented dewar is nothing", which is a claim that cannot be wrong
        and therefore is not worth making."""
        gross = self.coolant_w + self.exhaust_w + self.ambient_w
        return 0.0 if gross <= 0.0 else abs(float(term_w)) / gross

    def endurance_hours(self, stored_j: float) -> float:
        """How long the bank lasts at this balance. Infinite if the
        recharge terms win, which is the whole design target."""
        if self.net_w <= 0.0:
            return float("inf")
        return stored_j / self.net_w / 3600.0


def cryo_vent_recovery_w(boil_off_kg_s: float, gas_k: float = 77.36,
                         room_k: float = 277.0,
                         cp_j_kgk: float = N2_GAS_CP_J_KGK) -> float:
    """Cooling recovered by warming a cryogen's vented gas in the pit.

    This is the sensible heat the gas takes up between the temperature it
    ARRIVES at and the store's temperature -- and `gas_k` is the arrival
    temperature, which is emphatically not the boiling point unless the
    pipe is short and insulated.

    That default of 77 K is the best case and it is usually a lie. Gas
    that has sat in a dewar's ullage has already given its cold to the
    ullage and the vessel walls; gas that has run twenty metres of bare
    copper through a warm workshop arrives at workshop temperature and
    recovers exactly zero. Pass the real vent temperature -- the vessel
    knows it -- rather than accepting this default."""
    return max(0.0, float(boil_off_kg_s)) * cp_j_kgk * max(0.0, float(room_k) - float(gas_k))


#: Air is 20.9% oxygen and the number that matters is not where breathing
#: gets unpleasant but where it stops working. Below about 18% judgement
#: goes; below 16% people lose consciousness without warning and without
#: feeling short of breath first, because the urge to breathe is driven
#: by carbon dioxide and an inert gas does not raise it. That is the
#: entire mechanism of the commonest cryogenic fatality: someone walks
#: into a low place, does not notice anything wrong, and falls over.
OXYGEN_AMBIENT_FRAC = 0.209
OXYGEN_IMPAIRED_FRAC = 0.18
OXYGEN_LETHAL_FRAC = 0.16
#: Nitrogen gas at room temperature. Cold gas is DENSER than air and a
#: pit is the lowest point on the site, so it does not disperse -- it
#: fills from the floor and stays.
N2_GAS_DENSITY_KG_M3 = 1.16


def vented_oxygen_frac(room_m3: float, inert_kg: float,
                       gas_density_kg_m3: float = N2_GAS_DENSITY_KG_M3) -> float:
    """Oxygen fraction after venting inert gas into an unventilated room.

    Simple displacement, which is the right model for a pit: the cold gas
    is heavier than air and the pit is a hole, so it does not mix away,
    it pools."""
    room = max(1e-3, float(room_m3))
    inert_m3 = max(0.0, float(inert_kg)) / max(1e-6, gas_density_kg_m3)
    remaining_air = max(0.0, room - inert_m3)
    return OXYGEN_AMBIENT_FRAC * remaining_air / room


def vent_into_pit(pit: "IcePit", vented_kg: float, gas_k: float,
                  cp_j_kgk: float = N2_GAS_CP_J_KGK,
                  room_m3: float | None = None) -> dict:
    """Route a dewar's vent into the pit and take the cold off it.

    Returns both halves of the truth: what it bought thermally, and what
    it did to the air in the hole. The second is the one that kills
    people, and it is the reason this is a function with a hazard field
    rather than a line that adds joules."""
    store_k = pit.medium.phase_change_k or 277.0
    kg = max(0.0, float(vented_kg))
    j = kg * cp_j_kgk * max(0.0, store_k - float(gas_k))
    if j > 0.0:
        pit.charge_j(j)
    volume = float(room_m3) if room_m3 is not None else pit.floor_area_m2 * pit.depth_m
    o2 = vented_oxygen_frac(volume, kg)
    return {"recovered_j": j, "gas_k": float(gas_k), "vented_kg": kg,
            "room_m3": volume, "oxygen_frac": o2,
            "breathable": o2 >= OXYGEN_IMPAIRED_FRAC,
            "lethal": o2 < OXYGEN_LETHAL_FRAC,
            "why": ("cold gas is denser than air and a pit is the lowest point on the "
                    "site, so an inert vent into it pools on the floor instead of "
                    "dispersing. Nobody feels it happening: the breathing reflex "
                    "answers to carbon dioxide, not to oxygen")}


def terminate_coolant_w(flow_kg_s: float, in_k: float, out_k: float,
                        cp_j_kgk: float = 3800.0) -> float:
    """Heat a coolant circuit lays down when it terminates in the pit.

    Positive, because it melts ice -- which is exactly what it is for.
    The default specific heat is a water-glycol mix, not pure water."""
    return max(0.0, float(flow_kg_s)) * cp_j_kgk * max(0.0, float(in_k) - float(out_k))


def station_thermal_balance(pit: "IcePit", *, coolant_w: float = 0.0,
                            exhaust_w: float = 0.0,
                            dewar_boil_off_kg_s: float = 0.0,
                            ln2_consumed_kg_s: float = 0.0,
                            sky_pan_m2: float = 0.0, clear_night: bool = True,
                            chiller_w: float = 0.0,
                            surface_k: float = 313.15) -> HeatBalance:
    """The whole station's heat, in one place.

    `dewar_boil_off_kg_s` is a vessel merely standing; `ln2_consumed_kg_s`
    is nitrogen actually being used for something. They are separated
    because the first is negligible and the second is not, and rolling
    them together hides which is which."""
    recovery = (cryo_vent_recovery_w(dewar_boil_off_kg_s)
                + cryo_vent_recovery_w(ln2_consumed_kg_s))
    return HeatBalance(
        coolant_w=max(0.0, coolant_w),
        exhaust_w=max(0.0, exhaust_w),
        ambient_w=pit.heat_gain_w(surface_k),
        cryo_recovery_w=recovery,
        sky_w=sky_cooling_w(sky_pan_m2, clear_night),
        chiller_w=max(0.0, chiller_w))


@dataclass
class PitExchanger:
    """Where a coolant circuit stops being the atmosphere's problem.

    A circuit that terminates here has no radiator, and a position with
    no radiator has no warm rectangle on it. That is the whole reason to
    do this and it is worth the plumbing.

    WHAT AN EXCHANGER CANNOT DO. It cannot return coolant at the store
    temperature. Every real exchanger has an APPROACH -- the gap between
    the cold side and the coldest the hot side gets -- and a coil in an
    ice bath is not a good exchanger, so the approach is several kelvin,
    not a fraction of one. Ignoring that is how a model quietly returns
    coolant at zero degrees and makes the engine look better than it is.

    AND IT DEGRADES. Once the ice around the coil has melted, the coil is
    sitting in water, and water it has already warmed. The effective
    approach widens as the bank discharges, which is why a bank that is
    half gone does not cool half as well -- it cools rather worse than
    that, and the failure is gradual rather than sudden."""
    identity: str = "station.pit_exchanger"
    area_m2: float = 6.0
    u_w_m2k: float = 320.0             # coil in ice water, fouled realistically
    approach_k: float = 4.0            # the best it will ever do
    coolant_cp_j_kgk: float = 3800.0   # water-glycol, not water
    fouling_frac: float = 0.0

    def effective_approach_k(self, pit: "IcePit") -> float:
        """Widens as the bank empties and as the coil fouls."""
        charge = pit.charge_frac
        starve = 1.0 + 3.0 * (1.0 - charge) ** 2
        return self.approach_k * starve / max(0.15, 1.0 - self.fouling_frac)

    def outlet_k(self, pit: "IcePit", inlet_k: float, flow_kg_s: float) -> float:
        """Real NTU, so that flowing faster does NOT get you colder.

        The temptation is to return inlet minus approach and be done. But
        an exchanger has a fixed UA, and pushing more mass through it
        gives each kilogram less time -- effectiveness falls. A circuit
        that speeds up moves more heat in total and returns WARMER, and
        that trade is the thing worth being able to see."""
        store_k = (pit.medium.phase_change_k or 277.0) + self.effective_approach_k(pit)
        c_min = max(1e-3, float(flow_kg_s) * self.coolant_cp_j_kgk)
        ua = self.area_m2 * self.u_w_m2k * max(0.05, 1.0 - self.fouling_frac)
        # one side is changing phase, so C_r = 0 and effectiveness is exact
        eff = 1.0 - math.exp(-ua / c_min)
        return float(inlet_k) - eff * max(0.0, float(inlet_k) - store_k)

    def duty_w(self, pit: "IcePit", inlet_k: float, flow_kg_s: float) -> float:
        out = self.outlet_k(pit, inlet_k, flow_kg_s)
        return max(0.0, float(flow_kg_s)) * self.coolant_cp_j_kgk * max(0.0, float(inlet_k) - out)

    def terminate(self, pit: "IcePit", inlet_k: float, flow_kg_s: float,
                  dt_s: float) -> dict:
        """Run the circuit into the bank for a step and melt what it melts."""
        w = self.duty_w(pit, inlet_k, flow_kg_s)
        offered = w * max(0.0, dt_s)
        absorbed = pit.draw_j(offered)
        spilled = max(0.0, offered - absorbed)
        return {"outlet_k": self.outlet_k(pit, inlet_k, flow_kg_s), "duty_w": w,
                "absorbed_j": absorbed, "spilled_j": spilled,
                "approach_k": self.effective_approach_k(pit),
                "charge_frac": pit.charge_frac,
                # heat the bank could NOT take has to go somewhere visible.
                # Querying with dt=0 offers nothing and so is short of
                # nothing -- the old form called that a shortfall.
                "radiator_required": spilled > 0.0,
                "radiator_w": spilled / dt_s if dt_s > 0.0 else 0.0}


#: Oxygen ENRICHMENT, the other end of the same axis. Above roughly 23.5%
#: things that merely burn start to burn violently, and things that do not
#: normally burn at all -- steel wool, an oily rag, the cotton of a
#: coverall -- catch and do not go out. A dug room shored with timber and
#: hung with canvas is an ignition-rich environment, and enriching it is
#: worse than the asphyxiation case because asphyxiation stops when you
#: leave and a fire does not.
OXYGEN_ENRICHED_FRAC = 0.235


def vented_oxygen_frac_enriching(room_m3: float, oxygen_kg: float,
                                 gas_density_kg_m3: float = 1.354) -> float:
    """Oxygen fraction after venting OXYGEN into a room. Goes up, not down."""
    room = max(1e-3, float(room_m3))
    added = max(0.0, float(oxygen_kg)) / max(1e-6, gas_density_kg_m3)
    displaced = max(0.0, room - added)
    return (OXYGEN_AMBIENT_FRAC * displaced + added) / room


def vent_verdict(cryogen_key: str, pit: "IcePit", kg: float,
                 room_m3: float | None = None, gas_k: float | None = None) -> dict:
    """Should THIS cryogen's vent be piped into the ice room?

    The question is not one question. There is a thermal answer, which is
    about how much of the cooling is in the gas rather than the liquid;
    and there is a room answer, which is about what the gas does to the
    air in a hole that people climb into. They disagree, and where they
    disagree the room wins, because the thermal upside of any of these is
    a few hundred kilojoules and the downside is a body at the bottom of
    a pit.

    There is also a third answer nobody expects, and it is the one that
    actually justifies the pipe: a vent that terminates in a cold DRY
    room does not ice shut. Atmospheric humidity freezing onto an
    eighty-kelvin fitting is what closes vents, and a vent that closes
    turns a standing dewar into a pressure vessel. The ice room has
    almost no vapour pressure in it. Piping the vent there is worth doing
    for the ICING alone, and the recovered cold is a bonus."""
    import cryogenics as cg
    c = cg.cryogen(cryogen_key)
    volume = float(room_m3) if room_m3 is not None else pit.floor_area_m2 * pit.depth_m
    arrival = float(gas_k) if gas_k is not None else c.boil_k
    store_k = pit.medium.phase_change_k or 277.0
    recovered = max(0.0, float(kg)) * c.gas_cp_j_kgk * max(0.0, store_k - arrival)

    if c.hazard == "flammable":
        return {"admit": False, "cryogen": c.key, "recovered_j": recovered,
                "oxygen_frac": OXYGEN_AMBIENT_FRAC, "reason": "flammable",
                "why": f"{c.label} is a fuel. A confined, unventilated, below-grade "
                       "room is the worst place on the site to put one, and the "
                       "thermal prize does not come close to paying for it. "
                       "Vent it high, outside, and downwind"}

    if c.hazard == "oxidiser":
        o2 = vented_oxygen_frac_enriching(volume, kg, c.gas_density_kg_m3)
        ok = o2 <= OXYGEN_ENRICHED_FRAC
        return {"admit": ok, "cryogen": c.key, "recovered_j": recovered,
                "oxygen_frac": o2, "enriched": not ok,
                "reason": "within limits" if ok else "oxygen-enriched",
                "why": (f"{c.label} enriches rather than displaces: {o2 * 100:.1f}% "
                        f"oxygen against a {OXYGEN_ENRICHED_FRAC * 100:.1f}% limit. "
                        "Enrichment is not survivable-by-leaving the way asphyxiation "
                        "is -- timber shoring and canvas that were merely combustible "
                        "become things that cannot be put out. And liquid air becomes "
                        "an oxidiser by standing, so a vessel that was safe to vent "
                        "here in the morning may not be by evening")}

    o2 = vented_oxygen_frac(volume, kg, c.gas_density_kg_m3)
    ok = o2 >= OXYGEN_IMPAIRED_FRAC
    return {"admit": ok, "cryogen": c.key, "recovered_j": recovered,
            "oxygen_frac": o2, "lethal": o2 < OXYGEN_LETHAL_FRAC,
            "reason": "within limits" if ok else "asphyxiant",
            "why": (f"{c.label} is inert, so the only question is displacement: "
                    f"{o2 * 100:.1f}% oxygen. " +
                    ("It also sinks once warm, so it does not clear a pit on its own "
                     "-- it has to be swept out. "
                     if c.sinks_in_a_pit else
                     "It is close to air's density once warm, so it will eventually "
                     "clear. ") +
                    f"Thermally it is worth {c.vent_worth_frac * 100:.0f}% of this "
                    "cryogen's total cooling" +
                    (", which is not much and is why argon's vent is barely worth the "
                     "pipe on thermal grounds alone"
                     if c.vent_worth_frac < 0.45 else ", which is over half of it"))}


@dataclass
class CryoBoilerCoil:
    """Work a cryogen to death in the ice room WITHOUT venting it there.

    This is the answer to the obvious idea. Byproduct oxygen has to go
    somewhere, the ice room wants cold, and boiling the oxygen in the ice
    room is free refrigeration -- but `vent_verdict` refuses more than a
    couple of kilograms of it into the air of a fifty cubic metre hole,
    because oxygen ENRICHES and an enriched dug room full of timber and
    canvas is a fire that cannot be put out.

    A coil fixes the whole problem and costs a pipe. The liquid boils
    inside the tube, the room never sees the gas, and the gas leaves at
    the far end as a warm product that is worth money. You get MORE cold
    this way, not less, because the coil harvests the latent heat as well
    as the sensible -- venting into the room only ever gave you the
    sensible half.

    The coil is also the thing that makes a by-product into a product. An
    oxygen stream that has already surrendered its cold is still oxygen,
    and oxygen is what a field hospital and a welding set both want."""
    identity: str = "station.cryo_coil"
    cryogen_key: str = "liquid-oxygen"
    area_m2: float = 3.0
    #: Boiling cryogen against ice water is a good exchanger on the
    #: inside and a poor one on the outside, and the outside governs.
    u_w_m2k: float = 260.0
    frost_frac: float = 0.0        # ice grows on the coil and insulates it
    boiled_kg: float = 0.0
    position: tuple = (0.0, -2.0, 0.0)

    @property
    def cryo(self):
        import cryogenics as cg
        return cg.cryogen(self.cryogen_key)

    def duty_w(self, pit: "IcePit") -> float:
        """Heat the room hands to the coil -- which is cold gained.

        FROST IS THE LIMIT, not the area. A coil at 90 K sitting in melt
        water grows an ice shell, the shell is the resistance, and the
        coil throttles ITSELF. Real cryogenic vaporisers are sized for
        this and are still periodically defrosted."""
        store_k = pit.medium.phase_change_k or 277.0
        ua = self.area_m2 * self.u_w_m2k * max(0.05, 1.0 - self.frost_frac)
        return max(0.0, ua * (store_k - self.cryo.boil_k))

    def boil_rate_kg_s(self, pit: "IcePit") -> float:
        """Latent only: the gas is still at boiling point at the tube
        exit, and whatever sensible heat it picks up on the way out is
        cold that also came from the room."""
        return self.duty_w(pit) / max(1.0, self.cryo.latent_heat_j_kg)

    def step(self, pit: "IcePit", dt_s: float, available_kg: float,
             outlet_k: float | None = None) -> dict:
        """Boil what there is and freeze what that buys."""
        dt = max(0.0, float(dt_s))
        kg = min(max(0.0, float(available_kg)), self.boil_rate_kg_s(pit) * dt)
        c = self.cryo
        out_k = float(outlet_k) if outlet_k is not None else (pit.medium.phase_change_k or 277.0)
        latent = kg * c.latent_heat_j_kg
        sensible = kg * c.gas_cp_j_kgk * max(0.0, out_k - c.boil_k)
        charged = pit.charge_j(latent + sensible)
        self.boiled_kg += kg
        # the shell thickens while it is working and thaws when it is not
        self.frost_frac = min(0.85, self.frost_frac + 2.0e-5 * dt) if kg > 0.0 \
            else max(0.0, self.frost_frac - 1.0e-5 * dt)
        return {"boiled_kg": kg, "latent_j": latent, "sensible_j": sensible,
                "charged_j": charged, "frost_frac": self.frost_frac,
                "gas_out_kg": kg, "gas_species": c.key,
                "room_oxygen_frac": OXYGEN_AMBIENT_FRAC,
                "why": "the room's air never meets the gas, so there is no enrichment "
                       "and no displacement -- the coil is what makes an oxygen "
                       "by-product usable as refrigeration instead of a hazard"}

    def defrost_need(self):
        import servicing as sv
        if self.frost_frac < 0.5:
            return None
        return sv.Need(
            identity=self.identity, want="defrost", position=tuple(self.position),
            quantity=self.frost_frac, unit="frac", matches="heat-gun",
            urgency=0.5, minutes=25.0, skill="labourer",
            label=f"{self.identity} {self.frost_frac * 100:.0f}% frosted, "
                  f"duty down to {(1.0 - self.frost_frac) * 100:.0f}%",
            why="a cryogenic coil grows its own insulation out of the water it is "
                "cooling, and throttles itself long before it runs out of area")


# ---------------------------------------------------------------------
# WHAT FREEZING DOES TO A CONTAINER
# ---------------------------------------------------------------------
#
# Water is the substance that expands when it freezes, which is why it
# floats and why it destroys anything it is sealed inside. Ice is 917
# kg/m3 against water's 1000, so a litre of water becomes 1.091 litres of
# ice, and 9.1% is not a margin anything rigid has.
#
# AN IBC IS NOT A TANK, IT IS A BOTTLE IN A CAGE. The blow-moulded HDPE
# bottle carries the pressure; the steel cage only stops it bulging
# sideways off a pallet. The cage contributes nothing to burst strength
# and people consistently assume it does. A "1000 litre" tote is brimful
# somewhere around 1050 to 1070 litres, so a nominally FULL one has about
# five per cent of headroom against a nine per cent demand -- it bursts,
# and it bursts every time, not sometimes.
#
# AND IT BURSTS LONG BEFORE IT IS FROZEN THROUGH. This is the part that
# surprises people. Freezing starts at the walls and the surface, because
# that is where the heat leaves, so a shell of ice forms first and seals
# the remaining water inside a rigid container of its own making. That
# trapped water then freezes with nowhere to go, and the pressure it
# makes is hydraulic. A tote that is a third frozen is already a tote
# under pressure. The same mechanism splits pipes, and it is why a split
# pipe is never at the coldest point.
#
# THE VALVE GOES FIRST AND YOU DO NOT FIND OUT UNTIL THE THAW. The
# discharge is at the bottom, it is thin, it is metal or hard plastic,
# it is uninsulated and it sticks out into the air. It freezes ahead of
# the body and cracks quietly, and the tote still looks fine -- until
# somebody opens it in the morning and a thousand litres leaves through
# a fitting that failed two nights ago.

#: Ice at 0 C against water at 0 C. The entire problem, in one ratio.
ICE_EXPANSION_RATIO = 999.84 / 916.7


@dataclass(frozen=True)
class VesselClass:
    key: str
    label: str
    nominal_l: float
    brimful_l: float               # what it actually holds to the top
    #: How much the wall will stretch before it splits. Bulk HDPE
    #: elongates enormously; a blow-moulded BOTTLE does not, because it
    #: fails at the thin corners and the pinch weld long before the flat
    #: of the wall is doing any work. A container allowance is one or two
    #: per cent, not the material datasheet figure.
    strain_allowance: float
    valve_exposed: bool
    #: Open or vented at the top, so ice can heave UPWARD. This removes
    #: the volumetric constraint entirely and is the reason an open stock
    #: tank survives what a sealed tote does not.
    open_top: bool = False
    why: str = ""

    @property
    def freeze_safe_l(self) -> float:
        """Most water it can hold and still freeze solid without bursting.

        An open vessel is not limited by this at all: the ice grows out
        of the top. It is returned as brimful so the comparison still
        works, but the mechanism is different and the body does not
        split."""
        if self.open_top:
            return self.brimful_l
        room = self.brimful_l * (1.0 + self.strain_allowance)
        return room / ICE_EXPANSION_RATIO

    @property
    def safe_fill_frac(self) -> float:
        """As a fraction of the NOMINAL rating, which is what gets used."""
        return self.freeze_safe_l / max(1e-6, self.nominal_l)


VESSELS: dict[str, VesselClass] = {
    "ibc-tote": VesselClass(
        "ibc-tote", "1000 L IBC tote", 1000.0, 1055.0, 0.015, True, False,
        why="a blow-moulded HDPE bottle in a steel cage. The cage is a pallet and a "
            "handhold, not a pressure vessel, and the bottle is what has to hold. "
            "Filled to its rating it has about half the headroom freezing needs"),
    "drum-205": VesselClass(
        "drum-205", "205 L steel drum", 205.0, 216.0, 0.004, False, False,
        why="steel, so it barely strains at all -- it deforms at the rolling hoops and "
            "then splits a seam. The closed head is the strongest part and the chime "
            "weld is the weakest, which is where it goes"),
    "jerrycan-20": VesselClass(
        "jerrycan-20", "20 L jerrycan", 20.0, 22.0, 0.018, False, False,
        why="the pressed ribs are there to let it flex with temperature, which makes "
            "it the one container here designed with this in mind"),
    "poly-tank-2500": VesselClass(
        "poly-tank-2500", "2500 L poly water tank", 2500.0, 2600.0, 0.020, True, True,
        why="vented to atmosphere and open at the top, so the ice has somewhere to go "
            "-- it heaves upward instead of outward. The body usually survives and the "
            "outlet fitting usually does not"),
}


def vessel_class(key: str) -> VesselClass:
    v = VESSELS.get(str(key))
    if v is None:
        raise KeyError(f"unknown vessel {key!r}; declared: {', '.join(sorted(VESSELS))}")
    return v


def freezing_point_k(salt_frac: float = 0.0, glycol_frac: float = 0.0) -> float:
    """Where it actually freezes, which is not always 273.15.

    The cheapest fix for every problem below is to make the water not be
    water. Glycol is the deliberate version and a brine is the improvised
    one -- and dirty water freezes lower than clean water for the same
    reason, which is why the reclaim tank survives nights the clean tank
    does not."""
    dep = 0.0
    dep += 58.0 * max(0.0, min(0.26, float(salt_frac)))      # NaCl, to eutectic
    dep += 40.0 * max(0.0, min(0.60, float(glycol_frac)))    # ethylene glycol
    return 273.15 - dep


@dataclass
class LiquidTote:
    """A container of water that can freeze, and break when it does."""
    identity: str = "station.tote"
    kind: str = "ibc-tote"
    fill_l: float = 1000.0
    temp_k: float = 293.15
    frozen_frac: float = 0.0
    salt_frac: float = 0.0
    glycol_frac: float = 0.0
    burst: bool = False
    valve_cracked: bool = False
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def spec(self) -> VesselClass:
        return vessel_class(self.kind)

    @property
    def freeze_k(self) -> float:
        return freezing_point_k(self.salt_frac, self.glycol_frac)

    @property
    def fill_frac(self) -> float:
        return self.fill_l / max(1e-6, self.spec.brimful_l)

    @property
    def safe_to_freeze(self) -> bool:
        """Whether it could freeze SOLID without splitting."""
        return self.fill_l <= self.spec.freeze_safe_l

    def ice_volume_l(self) -> float:
        """Volume the contents occupy at the current frozen fraction."""
        f = max(0.0, min(1.0, self.frozen_frac))
        return self.fill_l * ((1.0 - f) + f * ICE_EXPANSION_RATIO)

    def strain(self) -> float:
        """How far past its container the contents are trying to go.

        TWO DIFFERENT MECHANISMS, and using the first one for both is the
        mistake that makes a tote look like it survives until it is
        frozen through.

        Before a shell closes, the ice and the remaining water share the
        whole vessel and the demand is the volume-average -- gentle, and
        it grows with the frozen fraction.

        After the shell closes, the water left inside is in a rigid
        container of ice and ALL of its expansion has to be taken by that
        shell. The demand is set by how much liquid is trapped, which is
        largest right after sealing. So the pressure spike comes EARLY,
        while there is still plenty of water, and a tote that is a third
        frozen is a tote already under load. This is why a burst pipe is
        never at the coldest point on the run."""
        if not self.sealed_by_ice():
            return max(0.0, self.ice_volume_l() / max(1e-6, self.spec.brimful_l) - 1.0)
        trapped_l = self.fill_l * (1.0 - self.frozen_frac)
        demand_l = trapped_l * (ICE_EXPANSION_RATIO - 1.0)
        return demand_l / max(1e-6, self.spec.brimful_l)

    def sealed_by_ice(self) -> bool:
        """Whether a shell has closed over the remaining water.

        Once this is true the liquid inside is in a rigid container of
        its own making and its expansion has nowhere to go -- which is
        why the burst comes at a third frozen and not at fully frozen."""
        return 0.25 <= self.frozen_frac < 1.0

    #: A bare container in still air: roughly its surface area times a
    #: natural-convection coefficient of about 4 W/m2K. For a 1000 L
    #: tote that is six square metres and some twenty-five watts per
    #: kelvin -- NOT the handful of watts an insulated vessel would show.
    #: Getting this wrong makes a tonne of water look unfreezable.
    def default_ua_w_k(self) -> float:
        area = 6.0 * (max(1.0, self.spec.brimful_l) / 1000.0) ** (2.0 / 3.0)
        return 4.0 * area

    def step(self, dt_s: float, ambient_k: float,
             ua_w_k: float | None = None) -> dict:
        """Let it stand in the cold and see what it does."""
        if ua_w_k is None:
            ua_w_k = self.default_ua_w_k()
        if self.burst:
            return {"burst": True, "frozen_frac": self.frozen_frac,
                    "temp_k": self.temp_k, "valve_cracked": self.valve_cracked,
                    "strain": self.strain(), "sealed": self.sealed_by_ice()}
        dt = max(0.0, float(dt_s))
        mass = self.fill_l * 0.9998
        fk = self.freeze_k
        q = ua_w_k * (self.temp_k - float(ambient_k)) * dt      # positive: losing heat
        if self.temp_k > fk:
            self.temp_k = max(fk, self.temp_k - q / max(1e-3, mass * WATER_CP_J_KGK))
        elif q > 0.0 and self.frozen_frac < 1.0:
            self.frozen_frac = min(1.0, self.frozen_frac
                                   + q / max(1.0, mass * ICE_LATENT_J_KG))
        # THE VALVE GOES FIRST: thin, metal, uninsulated, sticking out.
        if (self.spec.valve_exposed and not self.valve_cracked
                and self.frozen_frac > 0.06):
            self.valve_cracked = True
        # and the body goes when the ice shell has sealed the water in.
        # An open vessel never gets there: the ice heaves out of the top.
        if self.spec.open_top:
            pass
        elif not self.safe_to_freeze and self.sealed_by_ice():
            if self.strain() > self.spec.strain_allowance:
                self.burst = True
        elif self.frozen_frac >= 1.0 and not self.safe_to_freeze:
            self.burst = True
        return {"burst": self.burst, "frozen_frac": self.frozen_frac,
                "temp_k": self.temp_k, "valve_cracked": self.valve_cracked,
                "strain": self.strain(), "sealed": self.sealed_by_ice()}

    def freeze_risk_need(self):
        import servicing as sv
        if self.safe_to_freeze or self.burst:
            return None
        drain_to = self.spec.freeze_safe_l
        return sv.Need(
            identity=self.identity, want="draw-down", position=tuple(self.position),
            quantity=self.fill_l - drain_to, unit="l", matches="water",
            urgency=0.7, minutes=20.0, skill="labourer",
            label=(f"{self.identity} at {self.fill_l:.0f} L will split if it freezes; "
                   f"safe fill is {drain_to:.0f} L"),
            why="ice is nine per cent bigger than the water it came from, and a tote "
                "filled to its rating has about five per cent of headroom. Drawing it "
                "down or dosing it with glycol are the only two answers, and the "
                "second one ruins it for drinking")


# ---------------------------------------------------------------------
# USING THE WHOLE ROOM
# ---------------------------------------------------------------------
#
# A pit holds a volume, and there are two completely different ways to
# put cold in it, with a factor of two between them.
#
# FREEZE IT IN PLACE and the pit is one solid block of ice at the density
# of ice. That is the maximum the hole can possibly hold and it is
# unbeatable as a thermal store -- and it is not a store of ICE, because
# there is no way to get any of it out. You can draw heat into it through
# a coil and that is all. It is a battery with no removable cells.
#
# FREEZE IT IN CANS and you get blocks you can carry, stack, sell and
# move to where the cold is wanted. You also get an aisle to walk down,
# air gaps between blocks, headroom to lift in, and a brine tank taking
# floor area -- and by the time all that is subtracted you are storing
# something like half of what the solid version held.
#
# THAT IS THE DECISION, and it is not obvious which way it goes. Half
# the capacity for the ability to take cold out of the room in your
# hands is a real trade, and it is exactly the trade the ice trade made
# for two centuries.
#
# CAN ICE IS THE PART WORTH KNOWING. You do not freeze blocks by putting
# water in a cold room, because water is a poor conductor and ice is
# worse, so a block freezes from the outside in and the last of it takes
# forever. Real ice plants use galvanised steel CANS standing in a bath
# of circulating brine: the brine is the cold side, the can is the heat
# path, and the block grows inward from all four walls at once. Harvest
# is a dip in warm water for a few seconds, which frees the block without
# melting any of it worth mentioning. That dip is also where a block
# picks up whatever is in the thaw tank, which is why ice plants care
# about the thaw tank and nobody thinks about it.

#: Ice blocks do not stack solid. Air gaps, dunnage and the fact that
#: blocks are not perfectly square.
BLOCK_PACKING_FRACTION = 0.75
#: Aisle a person can actually work in while carrying fifty kilos.
WORKING_AISLE_M = 0.90
#: Headroom to lift a block out of a stack without taking the roof off.
HANDLING_HEADROOM_M = 0.80


@dataclass(frozen=True)
class IceCan:
    """A galvanised can that makes one block."""
    key: str
    label: str
    block_kg: float
    length_m: float
    width_m: float
    height_m: float
    why: str = ""

    @property
    def footprint_m2(self) -> float:
        return self.length_m * self.width_m

    def freeze_hours(self, brine_k: float = 263.15) -> float:
        """Plank's equation, roughly: freezing time goes as the SQUARE of
        the smallest dimension.

        This is why blocks are slabs and not cubes. Doubling the
        thickness quadruples the time, so an ice plant makes thin blocks
        even though thick ones would be easier to handle."""
        thickness = min(self.length_m, self.width_m)
        dt = max(1.0, 273.15 - float(brine_k))
        # rho * L * (a*d/h + b*d^2/k), with h and k for a can in brine
        rho, lat, k_ice, h_surf = 917.0, ICE_LATENT_J_KG, 2.2, 500.0
        secs = (rho * lat / dt) * (0.5 * thickness / h_surf
                                   + 0.125 * thickness * thickness / k_ice)
        return secs / 3600.0


ICE_CANS: dict[str, IceCan] = {
    "can-25": IceCan("can-25", "25 kg can", 25.0, 0.38, 0.19, 0.42,
                     why="one person, one hand, no argument. The size ice is actually "
                         "sold in"),
    "can-50": IceCan("can-50", "50 kg can", 50.0, 0.48, 0.24, 0.50,
                     why="two hands or a hook. The commonest plant size, and about the "
                         "limit of what gets carried down a ladder"),
    "can-135": IceCan("can-135", "135 kg block can", 135.0, 0.56, 0.28, 0.94,
                      why="the classic ice-house block, and it needs tongs and a hoist. "
                          "It also keeps far better than small blocks, because what "
                          "melts a store is surface area"),
}


def ice_can(key: str) -> IceCan:
    c = ICE_CANS.get(str(key))
    if c is None:
        raise KeyError(f"unknown ice can {key!r}; declared: {', '.join(sorted(ICE_CANS))}")
    return c


@dataclass
class ColdRoomLayout:
    """What actually fits in the hole once people have to work in it."""
    identity: str = "station.cold_room.layout"
    floor_area_m2: float = 16.8
    depth_m: float = 3.0
    can_key: str = "can-50"
    #: Footprint taken by plant: the boiler coil, the vent, the brine
    #: tank, the dewar standing in the corner.
    plant_footprint_m2: float = 2.6
    brine_tank_m2: float = 2.0
    aisle_m: float = WORKING_AISLE_M

    @property
    def room_volume_m3(self) -> float:
        return self.floor_area_m2 * self.depth_m

    @property
    def side_m(self) -> float:
        return math.sqrt(max(0.5, self.floor_area_m2))

    @property
    def aisle_area_m2(self) -> float:
        return self.aisle_m * self.side_m

    @property
    def stack_area_m2(self) -> float:
        return max(0.0, self.floor_area_m2 - self.aisle_area_m2
                   - self.plant_footprint_m2 - self.brine_tank_m2)

    @property
    def stack_height_m(self) -> float:
        return max(0.0, self.depth_m - HANDLING_HEADROOM_M)

    @property
    def block_capacity_kg(self) -> float:
        v = self.stack_area_m2 * self.stack_height_m * BLOCK_PACKING_FRACTION
        return v * 917.0

    @property
    def solid_capacity_kg(self) -> float:
        """Frozen in place, wall to wall. The ceiling nothing beats."""
        return self.room_volume_m3 * 917.0

    @property
    def blocks(self) -> int:
        return int(self.block_capacity_kg // max(1e-6, ice_can(self.can_key).block_kg))

    @property
    def penalty_frac(self) -> float:
        """What being able to carry it out costs."""
        return 1.0 - self.block_capacity_kg / max(1e-6, self.solid_capacity_kg)

    def report(self) -> dict:
        c = ice_can(self.can_key)
        return {"room_m3": self.room_volume_m3,
                "floor_m2": self.floor_area_m2,
                "aisle_m2": self.aisle_area_m2,
                "plant_m2": self.plant_footprint_m2 + self.brine_tank_m2,
                "stack_m2": self.stack_area_m2,
                "stack_height_m": self.stack_height_m,
                "solid_kg": self.solid_capacity_kg,
                "block_kg": self.block_capacity_kg,
                "blocks": self.blocks, "can": c.label,
                "freeze_hours": c.freeze_hours(),
                "penalty_frac": self.penalty_frac,
                "solid_mj": self.solid_capacity_kg * ICE_LATENT_J_KG / 1e6,
                "block_mj": self.block_capacity_kg * ICE_LATENT_J_KG / 1e6}


def cold_room_containers(layout: "ColdRoomLayout") -> list:
    """The right vessels for a room that goes below freezing.

    Every choice here is downstream of one fact: water expands when it
    freezes and a sealed rigid container cannot accommodate that. So
    nothing in a cold room is a sealed full vessel of water."""
    return [
        {"what": "open-top poly tank", "vessel": "poly-tank-2500", "count": 1,
         "holds": "make-up and thaw water",
         "why": "vented and open, so ice heaves upward instead of splitting the body. "
                "The one water vessel that can be left full without a thought"},
        {"what": "brine tank, insulated", "vessel": None, "count": 1,
         "holds": "calcium chloride brine at about -10 C",
         "why": "brine is the cold side the cans stand in. It must not freeze, which "
                "is the whole reason it is brine -- and its concentration is a "
                "maintenance item, because water carried out on the cans dilutes it"},
        {"what": "galvanised ice cans", "vessel": None, "count": 24,
         "holds": "one block each",
         "why": "steel because the can is the heat path; galvanised because it lives "
                "in brine, and a rusted can contaminates every block it makes"},
        {"what": "steel drums, filled to 97%", "vessel": "drum-205", "count": 4,
         "holds": "reclaimed fluid awaiting processing",
         "why": "oil does not expand on freezing, but any water IN it does, and a drum "
                "filled to the brim is the one that splits a seam"},
        {"what": "NOT an IBC of water", "vessel": "ibc-tote", "count": 0,
         "holds": "nothing, if there is any sense in the place",
         "why": "a full tote splits at a quarter frozen and the valve cracks long "
                "before that. If one must live here it goes in at 982 L, not 1000, "
                "and it is the first thing to check after a cold night"},
    ]


# ---------------------------------------------------------------------
# DEFROSTING THE COIL, AND WHAT EACH WAY COSTS
# ---------------------------------------------------------------------

CO2_SUBLIME_K = 194.65
CO2_LATENT_SUBLIME_J_KG = 571_000.0
CO2_GAS_CP_J_KGK = 850.0
CO2_GAS_DENSITY_KG_M3 = 1.84
#: Carbon dioxide is not an inert asphyxiant and treating it as one is a
#: category error. Nitrogen kills by displacing oxygen; CO2 is TOXIC at
#: concentrations where there is still plenty of oxygen, because it
#: drives blood pH. Four per cent is dangerous, ten per cent is
#: unconsciousness in minutes. The one mercy is that unlike nitrogen you
#: feel it -- CO2 is what triggers the urge to breathe, so victims know
#: something is wrong. That is the difference between a warning and no
#: warning, and it is the only good thing about it.
CO2_DANGEROUS_FRAC = 0.04
CO2_LETHAL_FRAC = 0.10


def defrost_cycle(frost_kg: float, method: str = "dry-ice-blast",
                  room_m3: float = 50.0, coil_k: float = 95.0,
                  co2_kg: float = 0.0) -> dict:
    """What it costs to get the ice shell off a cryogenic coil.

    Three ways, and they are not small variations on each other -- the
    heat they put into the room differs by two orders of magnitude,
    because two of them MELT the frost and one of them does not."""
    frost = max(0.0, float(frost_kg))
    melt_j = frost * ICE_LATENT_J_KG

    if method == "electric":
        # a heater element, and every joule of it lands in the room
        added = melt_j / 0.55            # most of the heat misses the frost
        return {"method": "electric heater", "heat_into_room_j": added,
                "bank_ice_melted_kg": added / ICE_LATENT_J_KG,
                "frost_removed_as": "melt water", "downtime_min": 25.0,
                "co2_frac": 0.0, "safe_to_occupy": True,
                "why": "simple, slow, and it charges the room the full latent heat of "
                       "the frost plus whatever the element wasted. The melt water "
                       "then refreezes onto the bank, so it is not lost -- but the "
                       "ELEMENT'S energy is, and that is most of it"}

    if method == "warm-gas":
        # divert warm product gas through the coil
        added = melt_j * 1.15
        return {"method": "warm product gas", "heat_into_room_j": added,
                "bank_ice_melted_kg": added / ICE_LATENT_J_KG,
                "frost_removed_as": "melt water", "downtime_min": 15.0,
                "co2_frac": 0.0, "safe_to_occupy": True,
                "why": "faster than a heater because the heat arrives where the frost "
                       "is, and it costs product gas warmth that was going to be "
                       "thrown away anyway. Still melts the frost, so still pays the "
                       "latent heat"}

    # dry ice blasting: pellets at 195 K hit the shell, the thermal shock
    # spalls it off SOLID, and the CO2 sublimes away leaving nothing.
    kg = float(co2_kg) if co2_kg > 0.0 else max(1.5, frost * 0.8)
    # the CO2 arrives colder than the room, so it is a net COOLING term
    cooling = kg * CO2_GAS_CP_J_KGK * max(0.0, 273.15 - CO2_SUBLIME_K)
    # what little heat there is comes from the compressed air carrier
    carrier_j = kg * 120_000.0
    net = carrier_j - cooling
    co2_frac = (kg / CO2_GAS_DENSITY_KG_M3) / max(1e-3, float(room_m3))
    return {"method": "dry ice blast", "co2_kg": kg,
            "heat_into_room_j": net,
            "bank_ice_melted_kg": max(0.0, net) / ICE_LATENT_J_KG,
            "frost_removed_as": "solid spall -- it never melts",
            "downtime_min": 8.0, "co2_frac": co2_frac,
            "safe_to_occupy": co2_frac < CO2_DANGEROUS_FRAC,
            "lethal": co2_frac >= CO2_LETHAL_FRAC,
            "why": "the frost comes off as SOLID, so the latent heat is never paid at "
                   "all -- that is the whole trick and it is why this is nearly free "
                   "thermally. What it is not is free in the room: carbon dioxide is "
                   "toxic rather than merely asphyxiating, it is heavier than air, and "
                   "this is a hole in the ground. The room is evacuated and purged "
                   "before anyone goes back in, or it is not done"}


# ---------------------------------------------------------------------
# WHAT THE FRONT END COSTS TO OWN
# ---------------------------------------------------------------------

#: Heat of desorption for water off 13X sieve. NOT the latent heat of
#: vaporisation -- it is substantially more, because the sieve holds the
#: water harder than a liquid surface does. Using the latent heat here
#: underestimates regeneration energy by about a third, and that error
#: makes molecular sieve look cheaper to run than it is.
WATER_DESORPTION_J_KG = 3_000_000.0
SIEVE_CP_J_KGK = 920.0
SIEVE_COST_PER_KG = 6.50
#: Sieve does not last forever: it loses capacity with every thermal
#: cycle, and faster if oil ever reaches it.
SIEVE_CYCLE_LIFE = 4000.0


def purifier_cost(bed_kg: float = 50.0, water_held_kg: float = 12.0,
                  co2_held_kg: float = 1.8, regen_k: float = 473.15,
                  ambient_k: float = 288.15, heater_efficiency: float = 0.85,
                  purge_frac: float = 0.06, product_kwh_nm3: float = 1.10,
                  feed_nm3_per_cycle: float = 3000.0) -> dict:
    """What one regeneration costs, and what the bed costs to own.

    THE PURGE IS A REAL COST AND IT IS EASY TO PRICE WRONGLY. A bed is
    swept with waste NITROGEN off the low-pressure column -- gas that was
    going to be vented regardless -- so the purge does not cost what the
    plant's product costs. Valuing it at the oxygen production figure
    overstates the whole front end by an order of magnitude.

    What the purge genuinely costs is HEATING it: several hundred
    kilograms of gas taken up to two hundred degrees and then thrown
    away, every cycle. That is comparable to heating the bed itself, and
    it is the term that actually belongs here."""
    sensible = bed_kg * SIEVE_CP_J_KGK * max(0.0, regen_k - ambient_k)
    desorb = water_held_kg * WATER_DESORPTION_J_KG + co2_held_kg * 600_000.0
    heat_kwh = (sensible + desorb) / 3.6e6 / max(0.1, heater_efficiency)
    purge_nm3 = feed_nm3_per_cycle * purge_frac
    # waste nitrogen: free to make, not free to warm up
    purge_kg = purge_nm3 * 1.2504
    purge_kwh = (purge_kg * 1040.0 * max(0.0, regen_k - ambient_k)
                 / 3.6e6 / max(0.1, heater_efficiency))
    bed_cost = bed_kg * SIEVE_COST_PER_KG
    return {"bed_kg": bed_kg, "bed_cost": bed_cost,
            "sensible_kwh": sensible / 3.6e6,
            "desorption_kwh": desorb / 3.6e6,
            "heat_kwh": heat_kwh, "purge_nm3": purge_nm3, "purge_kg": purge_kg,
            "purge_kwh": purge_kwh,
            "total_kwh_per_regen": heat_kwh + purge_kwh,
            "purge_share": purge_kwh / max(1e-6, heat_kwh + purge_kwh),
            "cycles_of_life": SIEVE_CYCLE_LIFE,
            "replacement_cost_per_cycle": bed_cost / SIEVE_CYCLE_LIFE,
            "why": "the bed is cheap to buy and the running cost is nearly all HEAT -- "
                   "half of it spent warming purge gas that is then vented. The sieve "
                   "itself amortises to pennies a cycle, so the front end is an energy "
                   "item and not a consumable one"}
