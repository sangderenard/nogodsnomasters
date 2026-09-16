"""Making, keeping and using things that are very cold.

Liquid air is the case that makes this worth building: it is dense,
bitterly cold, and it is stored energy -- a litre of it expands to about
seven hundred litres of gas, so a vessel full of it is a pressure vessel
that has not decided to be one yet. Everything in this module exists
because of that.

WHAT ACTUALLY MAKES COLD. Compressing a gas and letting it back down
again does nothing on its own; you get the heat back. Two things break
that, and real plants use both:

  Joule-Thomson    throttle a gas through a valve and, BELOW its
                   inversion temperature, it cools. Simple, no moving
                   parts, and not very efficient. This is the Linde-
                   Hampson cycle and it is how small plants work.
  work extraction  let the gas push a turbine on the way down and it
                   loses the energy it hands over, which cools it far
                   harder than throttling. That turbine is the cold head
                   -- a turboexpander -- and adding one is what makes a
                   Claude cycle. It is how anything that has to produce
                   liquid in quantity does it.

THE HYDROGEN TRAP, which is the reason this module names inversion
temperatures at all. Hydrogen's Joule-Thomson inversion temperature is
about 200 K. ABOVE that, throttling hydrogen makes it HOTTER. A plant
built on the assumption that expanding a gas cools it will, with
hydrogen from ambient, warm it -- and since hydrogen's ignition energy
is a static spark, warming it while leaking it is a specific and
well-known way to have a fire. Hydrogen has to be pre-cooled below its
inversion temperature, classically with liquid nitrogen, before any
throttling stage will do anything but the opposite of what is wanted.

INSULATION IS A MATERIAL CHOICE WITH A VACUUM PRECONDITION. Multi-layer
insulation is astonishing -- effective conductivity a thousand times
below foam -- and it is astonishing ONLY in a vacuum. Let air into the
annulus and MLI is worse than perlite, because it is then a stack of
conductive foils. That is why a vacuum-jacketed vessel that loses its
vacuum does not degrade gracefully: it falls off a cliff, the contents
boil, and the relief valve becomes the only thing between the vessel and
a failure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# the vent is an orifice like any other, so it uses the orifice law
# the pneumatic system already runs on rather than a second one
from drivetrain_graph import choked_orifice_mass_flow_kg_s

STEFAN_BOLTZMANN = 5.670374419e-8
AMBIENT_K = 293.15
ATM_PA = 101_325.0


# ---------------------------------------------------------------------
# what is cold, and what it takes to make it
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Cryogen:
    key: str
    label: str
    boil_k: float                 # normal boiling point at 1 atm
    liquid_density_kg_m3: float
    latent_heat_j_kg: float
    gas_density_kg_m3: float      # at 1 atm, 15 C
    inversion_k: float            # above this, throttling WARMS it
    why: str
    #: Gas specific heat at constant pressure, near room temperature.
    #: This -- not the latent heat -- decides what a VENT is worth,
    #: because the vent carries gas and the gas still has its cold.
    gas_cp_j_kgk: float = 1040.0
    #: What it does to a room, which is a different question from what
    #: it does to a process. "inert", "oxidiser" or "flammable".
    hazard: str = "inert"

    @property
    def sensible_j_kg(self) -> float:
        """Cold carried by the gas from boiling to a cold room."""
        return self.gas_cp_j_kgk * max(0.0, 277.0 - self.boil_k)

    @property
    def vent_worth_frac(self) -> float:
        """Share of this cryogen's total cooling that is in the GAS.

        Above a half and the vent is throwing away more than the liquid
        delivered; well below a half and piping it somewhere is plumbing
        for its own sake."""
        tot = self.latent_heat_j_kg + self.sensible_j_kg
        return 0.0 if tot <= 0.0 else self.sensible_j_kg / tot

    @property
    def sinks_in_a_pit(self) -> bool:
        """Heavier than air at room temperature, so it pools in a hole
        instead of leaving one. Cold gas of ANY species sinks at first;
        this is about where it ends up once it has warmed."""
        return self.gas_density_kg_m3 > 1.225

    @property
    def expansion_ratio(self) -> float:
        """Gas volume per unit of liquid volume. This is the number that
        makes a cryogenic vessel dangerous rather than merely cold."""
        return self.liquid_density_kg_m3 / max(1e-9, self.gas_density_kg_m3)


CRYOGENS: dict[str, Cryogen] = {
    "liquid-air": Cryogen(
        "liquid-air", "liquid air", 78.8, 874.0, 205_000.0, 1.225, 603.0,
        why="not a pure substance: it boils over a range and the nitrogen leaves first, "
            "so a vessel of liquid air left standing becomes oxygen-enriched -- which "
            "is why old liquid air is more dangerous than fresh",
        gas_cp_j_kgk=1005.0, hazard="oxidiser"),
    "liquid-nitrogen": Cryogen(
        "liquid-nitrogen", "liquid nitrogen", 77.36, 807.0, 199_000.0, 1.185, 621.0,
        why="the workhorse: cheap, inert, and cold enough to pre-cool hydrogen below "
            "its inversion temperature",
        gas_cp_j_kgk=1040.0, hazard="inert"),
    "liquid-oxygen": Cryogen(
        "liquid-oxygen", "liquid oxygen", 90.19, 1141.0, 213_000.0, 1.354, 761.0,
        why="boils thirteen degrees warmer than nitrogen, which is the entire basis of "
            "separating air; also violently reactive with anything organic",
        gas_cp_j_kgk=918.0, hazard="oxidiser"),
    "liquid-argon": Cryogen(
        "liquid-argon", "liquid argon", 87.30, 1394.0, 161_000.0, 1.669, 723.0,
        why="between the two, which is why it accumulates in the middle of a column",
        gas_cp_j_kgk=520.0, hazard="inert"),
    "liquid-hydrogen": Cryogen(
        "liquid-hydrogen", "liquid hydrogen", 20.28, 70.8, 449_000.0, 0.0852, 202.0,
        why="the hard one. Its inversion temperature is BELOW ambient, so throttling it "
            "from room temperature heats it; it must be pre-cooled first. Then it has a "
            "latent heat so small relative to its cold that almost anything boils it",
        gas_cp_j_kgk=14300.0, hazard="flammable"),
    "liquid-methane": Cryogen(
        "liquid-methane", "liquid methane (LNG)", 111.7, 422.0, 511_000.0, 0.680, 968.0,
        why="the one that is also a fuel, which is why its boil-off is worth catching "
            "rather than venting",
        gas_cp_j_kgk=2220.0, hazard="flammable"),
}


def cryogen(key: str) -> Cryogen:
    c = CRYOGENS.get(str(key))
    if c is None:
        raise KeyError(f"unknown cryogen {key!r}; declared: {', '.join(sorted(CRYOGENS))}")
    return c


def throttling_cools(key: str, inlet_k: float) -> bool:
    """Whether a Joule-Thomson valve will cool this gas at this
    temperature, or warm it.

    The question nobody asks until a hydrogen rig behaves backwards."""
    return float(inlet_k) < cryogen(key).inversion_k


# ---------------------------------------------------------------------
# insulation: a material, and a vacuum it may depend on
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class InsulationMedia:
    key: str
    label: str
    k_w_mk: float                 # effective conductivity in its intended state
    k_spoiled_w_mk: float         # what it becomes if its vacuum is lost
    needs_vacuum: bool
    max_wrap_thickness_m: float
    density_kg_m3: float
    why: str


INSULATION: dict[str, InsulationMedia] = {
    "mli": InsulationMedia(
        "mli", "multi-layer insulation (aluminised film)", 5.0e-5, 0.045, True, 0.050, 60.0,
        why="dozens of radiation shields with almost nothing touching between them. A "
            "thousand times better than foam IN A VACUUM, and worse than perlite "
            "without one, because it is then just a stack of conductive foils"),
    "vacuum-perlite": InsulationMedia(
        "vacuum-perlite", "evacuated perlite", 1.0e-3, 0.040, True, 0.300, 60.0,
        why="cheap expanded volcanic glass in an evacuated annulus: far less good than "
            "MLI and far more tolerant of a poor vacuum, which is why big field tanks "
            "use it"),
    "aerogel-blanket": InsulationMedia(
        "aerogel-blanket", "aerogel blanket", 0.015, 0.018, False, 0.100, 150.0,
        why="the wrap you put round a machine rather than inside a jacket: no vacuum "
            "needed, flexible, and it works cold or hot"),
    "pu-foam": InsulationMedia(
        "pu-foam", "polyurethane foam", 0.025, 0.030, False, 0.200, 40.0,
        why="ordinary, cheap, and it embrittles and cracks at cryogenic temperatures, "
            "which is why it belongs on warm things"),
    "mineral-wool": InsulationMedia(
        "mineral-wool", "mineral wool", 0.040, 0.045, False, 0.250, 100.0,
        why="for hot machines, not cold ones: it will happily condense and then freeze "
            "atmospheric water inside itself and become a block of ice"),
}


def insulation(key: str) -> InsulationMedia:
    m = INSULATION.get(str(key))
    if m is None:
        raise KeyError(f"unknown insulation {key!r}; declared: {', '.join(sorted(INSULATION))}")
    return m


def wrap_heat_leak_w(media: str, area_m2: float, thickness_m: float,
                     inner_k: float, outer_k: float = AMBIENT_K,
                     vacuum_intact: bool = True) -> float:
    """Conduction through an insulating wrap around a machine.

    Plain Fourier through the thickness. The interesting parameter is
    `vacuum_intact`, because for a vacuum-dependent medium losing it is
    not a degradation, it is a different material."""
    m = insulation(media)
    k = m.k_w_mk if (vacuum_intact or not m.needs_vacuum) else m.k_spoiled_w_mk
    t = max(1e-4, min(float(thickness_m), m.max_wrap_thickness_m))
    return k * max(0.0, float(area_m2)) * abs(float(outer_k) - float(inner_k)) / t


# ---------------------------------------------------------------------
# the vessel
# ---------------------------------------------------------------------

@dataclass
class VacuumJacketedVessel:
    """A dewar: inner vessel, evacuated annulus, insulation, relief.

    IT MUST BE ABLE TO VENT. A cryogenic vessel is a machine for slowly
    turning liquid into several hundred times its volume of gas, and the
    only question is whether that gas has somewhere to go. A sealed one
    does not hold -- it bursts, and it does so at a pressure and on a
    schedule that the heat leak decides."""
    identity: str = "cryo.vessel"
    contents: str = "liquid-air"
    capacity_l: float = 200.0
    fill_l: float = 0.0
    surface_area_m2: float = 2.4
    insulation_media: str = "mli"
    insulation_thickness_m: float = 0.025
    vacuum_intact: bool = True
    relief_pressure_pa: float = 250_000.0
    relief_fitted: bool = True
    pressure_pa: float = ATM_PA
    vented_kg: float = 0.0
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def cryo(self) -> Cryogen:
        return cryogen(self.contents)

    @property
    def fill_frac(self) -> float:
        return 0.0 if self.capacity_l <= 0.0 else min(1.0, self.fill_l / self.capacity_l)

    @property
    def liquid_kg(self) -> float:
        return self.fill_l / 1000.0 * self.cryo.liquid_density_kg_m3

    def heat_leak_w(self, ambient_k: float = AMBIENT_K) -> float:
        return wrap_heat_leak_w(self.insulation_media, self.surface_area_m2,
                                self.insulation_thickness_m, self.cryo.boil_k,
                                ambient_k, self.vacuum_intact)

    def boil_off_kg_s(self, ambient_k: float = AMBIENT_K) -> float:
        if self.liquid_kg <= 0.0:
            return 0.0
        return self.heat_leak_w(ambient_k) / self.cryo.latent_heat_j_kg

    def boil_off_percent_per_day(self, ambient_k: float = AMBIENT_K) -> float:
        """How these are actually specified. A good vessel is a fraction
        of a percent a day; one that has lost its vacuum is tens."""
        if self.liquid_kg <= 0.0:
            return 0.0
        return self.boil_off_kg_s(ambient_k) * 86_400.0 / self.liquid_kg * 100.0

    def step(self, dt: float, ambient_k: float = AMBIENT_K) -> dict:
        """Let it stand for `dt`. Boil-off leaves as gas; if it cannot
        leave, the pressure rises."""
        boiled_kg = self.boil_off_kg_s(ambient_k) * max(0.0, dt)
        boiled_kg = min(boiled_kg, self.liquid_kg)
        if boiled_kg <= 0.0:
            return {"boiled_kg": 0.0, "vented_kg": 0.0, "pressure_pa": self.pressure_pa,
                    "burst": False}
        self.fill_l -= boiled_kg / self.cryo.liquid_density_kg_m3 * 1000.0
        ullage_m3 = max(1e-4, (self.capacity_l - self.fill_l) / 1000.0)
        # the gas it just became, in the space above the liquid
        added_pa = boiled_kg / self.cryo.gas_density_kg_m3 / ullage_m3 * ATM_PA
        self.pressure_pa += added_pa
        vented = 0.0
        burst = False
        if self.relief_fitted and self.pressure_pa > self.relief_pressure_pa:
            vented = boiled_kg
            self.vented_kg += vented
            self.pressure_pa = self.relief_pressure_pa
        elif not self.relief_fitted and self.pressure_pa > self.relief_pressure_pa * 4.0:
            burst = True
        return {"boiled_kg": boiled_kg, "vented_kg": vented,
                "pressure_pa": self.pressure_pa, "burst": burst}

    def drain(self, litres: float) -> float:
        """Draw liquid off the bottom. Returns what was actually given."""
        take = max(0.0, min(float(litres), self.fill_l))
        self.fill_l -= take
        return take

    def need(self):
        """A vessel that has lost its vacuum wants attention before it
        wants anything else."""
        import servicing as sv
        if self.vacuum_intact:
            return None
        return sv.Need(
            identity=self.identity, want="vacuum-pump-down", position=tuple(self.position),
            quantity=1.0, unit="each", matches="vacuum-service",
            urgency=2.0, minutes=90.0, skill="mechanic", label=self.identity,
            why="a jacket that has lost its vacuum is not a worse vessel, it is a "
                "different one: the contents boil away in days instead of months and "
                "the relief valve is the only thing holding it together")


# ---------------------------------------------------------------------
# THE COLD HEAD: a turboexpander
# ---------------------------------------------------------------------
#
# This is the part that collects liquid air. Gas at pressure drives a
# small, very fast wheel and LEAVES ITS ENERGY BEHIND in the shaft; what
# comes out is colder than any throttle could make it, because the gas
# has actually done work rather than merely spread out.
#
# Two consequences worth having. The shaft work has to go somewhere, and
# on a real machine it goes into a brake -- a blower, or a generator --
# and recovering it is most of what makes a Claude cycle efficient
# rather than merely effective. And the outlet can be INSIDE the two-
# phase region: expand far enough and liquid forms in the wheel, which
# is how you collect it and also how you destroy the wheel if you
# overdo it.

RATIO_GAMMA_AIR = 1.40


@dataclass
class Turboexpander:
    """A cryogenic expansion turbine: the cold head of a Claude cycle."""
    identity: str = "cryo.expander"
    isentropic_efficiency: float = 0.82
    mass_flow_kg_s: float = 0.05
    inlet_pressure_pa: float = 4.0e6
    outlet_pressure_pa: float = 1.5e5
    gamma: float = RATIO_GAMMA_AIR
    cp_j_kgk: float = 1005.0
    max_liquid_frac: float = 0.10     # past this the wheel is being eroded

    @property
    def pressure_ratio(self) -> float:
        return max(1.0, self.inlet_pressure_pa / max(1.0, self.outlet_pressure_pa))

    def outlet_temp_k(self, inlet_k: float) -> float:
        """The isentropic drop, less whatever the machine's inefficiency
        puts back as heat."""
        t_in = float(inlet_k)
        t_isen = t_in * self.pressure_ratio ** (-(self.gamma - 1.0) / self.gamma)
        return t_in - self.isentropic_efficiency * (t_in - t_isen)

    def shaft_power_w(self, inlet_k: float) -> float:
        """What it takes OUT -- which is why it cools, and what a real
        machine recovers rather than throwing away."""
        drop = float(inlet_k) - self.outlet_temp_k(inlet_k)
        return self.mass_flow_kg_s * self.cp_j_kgk * drop

    def liquid_fraction(self, inlet_k: float, product: str = "liquid-air") -> float:
        """How much of the stream comes out as liquid.

        Below the boiling point the excess cold goes into latent heat.
        This is the yield of the cold head, and also the thing to watch:
        too much liquid in the wheel and the wheel does not last."""
        c = cryogen(product)
        t_out = self.outlet_temp_k(inlet_k)
        if t_out >= c.boil_k:
            return 0.0
        sensible = self.cp_j_kgk * (c.boil_k - t_out)
        return max(0.0, min(1.0, sensible / c.latent_heat_j_kg))

    def step(self, dt: float, inlet_k: float, product: str = "liquid-air") -> dict:
        frac = self.liquid_fraction(inlet_k, product)
        liquid_kg = self.mass_flow_kg_s * frac * max(0.0, dt)
        c = cryogen(product)
        return {
            "outlet_k": self.outlet_temp_k(inlet_k),
            "shaft_w": self.shaft_power_w(inlet_k),
            "liquid_frac": frac,
            "liquid_kg": liquid_kg,
            "liquid_l": liquid_kg / c.liquid_density_kg_m3 * 1000.0,
            "wheel_eroding": frac > self.max_liquid_frac,
        }


# ---------------------------------------------------------------------
# COMPRESSION, and why hydrogen is its own problem
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CompressorDuty:
    key: str
    label: str
    stages: int
    per_stage_ratio: float
    isothermal_efficiency: float
    seal_kind: str
    why: str


GAS_COMPRESSORS: dict[str, CompressorDuty] = {
    "air-plant": CompressorDuty(
        "air-plant", "air plant main compressor", 4, 3.2, 0.62, "labyrinth",
        why="four stages with intercooling between each: compressing in one jump would "
            "put the discharge somewhere the machine cannot survive"),
    "hydrogen-diaphragm": CompressorDuty(
        "hydrogen-diaphragm", "hydrogen diaphragm compressor", 3, 4.0, 0.55,
        "hydraulic-diaphragm",
        why="hydrogen gets into steel and embrittles it, and it leaks through seals "
            "nothing else leaks through -- so it is compressed behind a metal diaphragm "
            "driven hydraulically, with no sliding seal touching the gas at all"),
    "hydrogen-ionic": CompressorDuty(
        "hydrogen-ionic", "ionic liquid piston compressor", 5, 5.0, 0.70, "ionic-liquid",
        why="the piston is a column of ionic liquid: almost no dead volume, no sliding "
            "seal, and it carries the heat of compression away as it works"),
    "boil-off-recovery": CompressorDuty(
        "boil-off-recovery", "boil-off recovery compressor", 2, 3.0, 0.58, "labyrinth",
        why="takes what a vessel vents and puts it back rather than letting it go; on a "
            "methane vessel that is fuel, and on any of them it is the work already "
            "spent making it"),
}


def compressor_duty(key: str) -> CompressorDuty:
    d = GAS_COMPRESSORS.get(str(key))
    if d is None:
        raise KeyError(
            f"unknown compressor duty {key!r}; declared: {', '.join(sorted(GAS_COMPRESSORS))}")
    return d


def compression_power_w(duty: str, mass_flow_kg_s: float, inlet_k: float,
                        pressure_ratio: float, gas_constant_j_kgk: float = 287.0) -> float:
    """Shaft power to compress this flow through this ratio.

    Multi-stage with intercooling, which is why the stage count matters:
    isothermal work is the floor and a real machine approaches it only by
    taking the heat out between stages."""
    d = compressor_duty(duty)
    r = max(1.0, float(pressure_ratio))
    ideal = (max(0.0, float(mass_flow_kg_s)) * gas_constant_j_kgk
             * float(inlet_k) * math.log(r))
    return ideal / max(0.05, d.isothermal_efficiency)


def hydrogen_precool_required(inlet_k: float) -> bool:
    """Whether a hydrogen stream must be pre-cooled before any throttle.

    Above its inversion temperature a throttle HEATS hydrogen. A plant
    that throttles it from ambient does not merely fail to make cold --
    it makes heat, in the one gas whose ignition energy is a static
    spark."""
    return float(inlet_k) >= cryogen("liquid-hydrogen").inversion_k


# ---------------------------------------------------------------------
# SEPARATION: boiling points do the work
# ---------------------------------------------------------------------

#: Air, by volume, as it arrives.
AIR_COMPOSITION = {"nitrogen": 0.7808, "oxygen": 0.2095, "argon": 0.0093}


def separation_products(liquid_air_l: float, reflux_stages: int = 20) -> dict:
    """Split liquid air into its parts by distillation.

    Nitrogen boils thirteen degrees below oxygen, and that gap is the
    entire mechanism: boil the mixture gently and the vapour is richer in
    nitrogen; condense and re-boil it enough times and the two separate.
    More stages, better separation -- which is why an air separation
    column is tall. Purity is asymptotic in stage count, so this reports
    what the column achieves rather than pretending a short one gets
    pure product."""
    stages = max(1, int(reflux_stages))
    purity = 1.0 - math.exp(-stages / 6.0)
    litres = max(0.0, float(liquid_air_l))
    out: dict = {"stages": stages}
    for name, frac in AIR_COMPOSITION.items():
        out[name] = {
            "litres": litres * frac,
            "purity": purity if name != "argon" else purity * 0.75,
        }
    return out


# ---------------------------------------------------------------------
# THE ATMOSPHERE INSIDE A DEWAR
# ---------------------------------------------------------------------
#
# A cryogenic vessel is not a bucket with a number in it. Above the
# liquid there is a real gas space -- the ullage -- and almost everything
# interesting about these vessels happens there: the pressure that
# builds, the composition that shifts, the vent that has to pass it, and
# the ice that stops the vent passing it.
#
# TWO THINGS FALL OUT OF DOING IT PROPERLY, and neither has to be
# special-cased:
#
#   OXYGEN ENRICHMENT. Liquid air is a mixture, nitrogen boils thirteen
#   degrees colder than oxygen, and the vapour leaving is therefore much
#   richer in nitrogen than the liquid it left. So a vessel standing
#   quietly gets MORE OXYGEN-RICH the longer it stands, without anybody
#   doing anything -- which is why old liquid air is dangerous in a way
#   fresh liquid air is not, and why a vessel that has been sitting is
#   not the vessel that was filled. This is just tracking the two
#   species separately; the hazard appears on its own.
#
#   VENT ICING. The vent carries gas at eighty kelvin through fittings
#   sitting in humid air, and atmospheric water freezes onto them. A
#   partly iced vent still vents; a fully iced one does not, and the
#   pressure behind it keeps rising because the heat leak does not care.
#   That is the classic way one of these fails, and it is a blockage
#   fraction on an orifice -- the same arithmetic as a fouled filter.
#
# AND THE DIAGNOSTIC THAT COMES FREE: with the jacket intact the OUTER
# wall sits near ambient and stays dry. Frost on the outside of a dewar
# means the vacuum has gone. It is not a cosmetic detail, it is the
# tell.

#: Relative volatility of nitrogen to oxygen at these temperatures --
#: how much richer the vapour is than the liquid. This one number is
#: what separates air, in a column or in a neglected vessel.
N2_O2_RELATIVE_VOLATILITY = 3.6
#: Frost accretion on a cryogenic fitting in humid air, as a fraction of
#: the vent area lost per hour. Real vents ice shut in hours, not days.
VENT_ICING_FRAC_PER_HOUR = 0.08
#: Oxygen fraction above which the liquid is an oxidiser first and a
#: coolant second.
OXYGEN_ENRICHED_THRESHOLD = 0.30
#: Specific heat of the ullage gas. Nitrogen, oxygen and argon are close
#: enough that one figure is honest here, the same way one specific gas
#: constant is honest for the pressure.
ULLAGE_CP_J_KG_K = 1040.0


@dataclass
class DewarAtmosphere:
    """The gas space above the liquid, as a real mixture.

    Mass per species, in the same shape the rest of the machine tracks
    fluid mixtures -- so a vessel's contents can be asked what they are,
    not just how much."""
    kg: dict = field(default_factory=dict)
    temp_k: float = 90.0

    @property
    def total_kg(self) -> float:
        return sum(self.kg.values())

    def add(self, species: str, kg: float, at_k: float | None = None) -> None:
        """Take gas in, and take its TEMPERATURE in with it.

        Boil-off arrives at the liquid's boiling point, which is far
        colder than the ullage it joins, so it does not merely add mass
        -- it chills the space. Adding the mass without the temperature
        is how an ullage ends up warmer than anything putting gas into
        it, and since pressure is mass times temperature, that error
        goes straight to the relief valve."""
        if kg <= 0.0:
            return
        if at_k is not None:
            have = self.total_kg
            if have > 0.0:
                self.temp_k = ((self.temp_k * have + float(at_k) * kg)
                               / (have + kg))
            else:
                self.temp_k = float(at_k)
        self.kg[species] = self.kg.get(species, 0.0) + kg

    def remove_proportional(self, kg: float) -> dict:
        """Vent some of it. Everything leaves in the proportion it is
        present, because a vent does not select."""
        total = self.total_kg
        if total <= 0.0 or kg <= 0.0:
            return {}
        f = min(1.0, kg / total)
        went = {}
        for k in list(self.kg):
            g = self.kg[k] * f
            self.kg[k] -= g
            went[k] = g
        return went

    def fractions(self) -> dict:
        t = self.total_kg
        return {k: v / t for k, v in self.kg.items() if t > 0.0 and v > 0.0}

    def pressure_pa(self, volume_m3: float) -> float:
        """Ideal gas on the ullage volume. Nitrogen and oxygen are close
        enough in molar mass that one specific gas constant is honest
        here; hydrogen is not, and says so."""
        v = max(1e-5, float(volume_m3))
        r_specific = 296.8 if self.kg.get("hydrogen", 0.0) <= 0.0 else 4124.0
        return self.total_kg * r_specific * max(1.0, self.temp_k) / v


def vapour_composition(liquid_fracs: dict) -> dict:
    """What boils off a liquid mixture, by mass fraction.

    Raoult with a relative volatility: the more volatile species is
    enriched in the vapour by that factor. For air that factor is about
    three and a half, which is why the vapour is nearly pure nitrogen
    and why what is left behind is not."""
    n2 = max(0.0, float(liquid_fracs.get("nitrogen", 0.0)))
    o2 = max(0.0, float(liquid_fracs.get("oxygen", 0.0)))
    ar = max(0.0, float(liquid_fracs.get("argon", 0.0)))
    a = N2_O2_RELATIVE_VOLATILITY
    denom = a * n2 + o2 + ar * 1.15
    if denom <= 0.0:
        return dict(liquid_fracs)
    return {"nitrogen": a * n2 / denom, "oxygen": o2 / denom, "argon": ar * 1.15 / denom}


@dataclass
class CryogenicVessel:
    """A dewar with a real atmosphere in it and a real vent out of it.

    Replaces treating the contents as one number: the liquid has a
    composition, the ullage is a gas mixture at a pressure, and the vent
    is an orifice that ice can close. Everything the earlier version
    asserted -- enrichment, venting, bursting -- now happens BECAUSE of
    those three rather than alongside them."""
    identity: str = "cryo.dewar"
    capacity_l: float = 200.0
    fill_l: float = 0.0
    liquid: dict = field(default_factory=lambda: {"nitrogen": 0.755,
                                                  "oxygen": 0.232,
                                                  "argon": 0.013})
    surface_area_m2: float = 2.4
    insulation_media: str = "mli"
    insulation_thickness_m: float = 0.025
    vacuum_intact: bool = True
    vent_area_m2: float = 3.0e-5
    vent_blocked_frac: float = 0.0        # ice
    relief_pressure_pa: float = 250_000.0
    burst_pressure_pa: float = 1.2e6
    relief_fitted: bool = True
    ullage: DewarAtmosphere = field(default_factory=DewarAtmosphere)
    vented_kg: float = 0.0
    burst: bool = False
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def liquid_density_kg_m3(self) -> float:
        by = {"nitrogen": 807.0, "oxygen": 1141.0, "argon": 1394.0}
        return sum(by.get(k, 900.0) * f for k, f in self.liquid.items()) or 874.0

    @property
    def liquid_kg(self) -> float:
        return self.fill_l / 1000.0 * self.liquid_density_kg_m3

    @property
    def ullage_m3(self) -> float:
        return max(1e-4, (self.capacity_l - self.fill_l) / 1000.0)

    @property
    def pressure_pa(self) -> float:
        return max(ATM_PA, self.ullage.pressure_pa(self.ullage_m3))

    @property
    def oxygen_frac(self) -> float:
        return float(self.liquid.get("oxygen", 0.0))

    @property
    def oxygen_enriched(self) -> bool:
        """Past this the liquid is an oxidiser first and a coolant
        second. A vessel that has stood long enough gets here on its
        own, with nobody doing anything to it."""
        return self.oxygen_frac >= OXYGEN_ENRICHED_THRESHOLD

    @property
    def boil_k(self) -> float:
        by = {"nitrogen": 77.36, "oxygen": 90.19, "argon": 87.30}
        return sum(by.get(k, 80.0) * f for k, f in self.liquid.items()) or 78.8

    @property
    def latent_heat_j_kg(self) -> float:
        by = {"nitrogen": 199_000.0, "oxygen": 213_000.0, "argon": 161_000.0}
        return sum(by.get(k, 205_000.0) * f for k, f in self.liquid.items()) or 205_000.0

    def heat_leak_w(self, ambient_k: float = AMBIENT_K) -> float:
        return wrap_heat_leak_w(self.insulation_media, self.surface_area_m2,
                                self.insulation_thickness_m, self.boil_k,
                                ambient_k, self.vacuum_intact)

    @property
    def exterior_frosts(self) -> bool:
        """THE TELL. With the jacket intact the outer wall sits near
        ambient and stays dry; frost on the OUTSIDE of a dewar means the
        vacuum has gone."""
        return not self.vacuum_intact

    @property
    def open_vent_area_m2(self) -> float:
        return self.vent_area_m2 * max(0.0, 1.0 - min(1.0, self.vent_blocked_frac))

    @property
    def vent_iced_shut(self) -> bool:
        return self.vent_blocked_frac >= 0.995

    def _warm_ullage(self, dt: float, ambient_k: float) -> None:
        """The wall above the liquid line lets heat in, and it warms the
        gas space toward ambient.

        Done as a relaxation with a time constant taken FROM THE HEAT
        LEAK -- the same leak that boils the liquid, shared out by how
        much of the vessel is gas -- rather than as a fixed nudge per
        call. A fixed nudge per call makes the answer depend on the step
        size, which for this vessel means the pressure at the relief
        valve depends on the frame rate. The exponential form is exact
        for the differential equation at any step and cannot overshoot
        ambient however large the step is."""
        m = self.ullage.total_kg
        if m <= 0.0 or dt <= 0.0:
            return
        gap = ambient_k - self.ullage.temp_k
        if gap <= 0.0:
            return
        share = self.ullage_m3 / max(1e-4, self.capacity_l / 1000.0)
        ua = self.heat_leak_w(ambient_k) * share / gap          # W/K
        if ua <= 0.0:
            return
        tau = m * ULLAGE_CP_J_KG_K / ua                          # s
        self.ullage.temp_k = max(
            self.boil_k,
            min(ambient_k,
                self.ullage.temp_k + gap * (1.0 - math.exp(-dt / max(1e-6, tau)))))

    def ice_vent(self, dt_s: float, humid: bool = True) -> float:
        """Atmospheric water freezing onto a fitting at eighty kelvin."""
        if not humid or self.liquid_kg <= 0.0:
            return self.vent_blocked_frac
        self.vent_blocked_frac = min(
            1.0, self.vent_blocked_frac + VENT_ICING_FRAC_PER_HOUR * (dt_s / 3600.0))
        return self.vent_blocked_frac

    def deice(self) -> float:
        """Clear the vent. Returns how much ice was on it."""
        was = self.vent_blocked_frac
        self.vent_blocked_frac = 0.0
        return was

    def deice_need(self):
        import servicing as sv
        if self.vent_blocked_frac < 0.35:
            return None
        return sv.Need(
            identity=self.identity, want="deice", position=tuple(self.position),
            quantity=self.vent_blocked_frac, unit="frac", matches="heat-gun",
            urgency=self.vent_blocked_frac / 0.35, minutes=20.0, skill="operator",
            label=f"{self.identity} vent {self.vent_blocked_frac * 100:.0f}% iced",
            why="a blocked vent on a cryogenic vessel is the classic way one bursts: "
                "the heat leak keeps boiling the contents whether or not the gas has "
                "anywhere to go")

    def step(self, dt: float, ambient_k: float = AMBIENT_K, humid: bool = True) -> dict:
        """Let it stand. Heat leaks in, the volatile species leaves the
        liquid first, the ullage takes it, the vent passes what it can."""
        if self.burst or dt <= 0.0:
            return {"boiled_kg": 0.0, "vented_kg": 0.0, "pressure_pa": self.pressure_pa,
                    "burst": self.burst, "oxygen_frac": self.oxygen_frac,
                    "vent_blocked_frac": self.vent_blocked_frac,
                    "oxygen_enriched": self.oxygen_enriched}
        self.ice_vent(dt, humid)
        boiled = 0.0
        if self.liquid_kg > 0.0:
            boiled = min(self.liquid_kg,
                         self.heat_leak_w(ambient_k) / self.latent_heat_j_kg * dt)
        if boiled > 0.0:
            # WHAT LEAVES IS NOT WHAT IS THERE: the vapour is richer in
            # the volatile species, so what stays behind is richer in
            # the other one. The enrichment hazard is this line.
            vap = vapour_composition(self.liquid)
            start_kg = self.liquid_kg
            remaining_kg = start_kg - boiled
            new_liquid = {}
            for sp, frac in self.liquid.items():
                new_liquid[sp] = max(0.0, frac * start_kg - vap.get(sp, 0.0) * boiled)
            total = sum(new_liquid.values())
            if total > 0.0:
                self.liquid = {k: v / total for k, v in new_liquid.items()}
            for sp, frac in vap.items():
                # at the boiling point, because that is where it came from
                self.ullage.add(sp, frac * boiled, at_k=self.boil_k)
            self.fill_l = max(0.0, remaining_kg / self.liquid_density_kg_m3 * 1000.0)
        self._warm_ullage(dt, ambient_k)

        vented = 0.0
        p = self.pressure_pa
        if self.relief_fitted and p > self.relief_pressure_pa:
            flow = choked_orifice_mass_flow_kg_s(self.open_vent_area_m2, p,
                                                 max(60.0, self.ullage.temp_k))
            vented = min(self.ullage.total_kg, flow * dt)
            self.ullage.remove_proportional(vented)
            self.vented_kg += vented
        if self.pressure_pa >= self.burst_pressure_pa:
            self.burst = True
        return {"boiled_kg": boiled, "vented_kg": vented, "pressure_pa": self.pressure_pa,
                "burst": self.burst, "oxygen_frac": self.oxygen_frac,
                "vent_blocked_frac": self.vent_blocked_frac,
                "oxygen_enriched": self.oxygen_enriched}


# ---------------------------------------------------------------------
# THE FRONT END: what has to come out before the cold box will run
# ---------------------------------------------------------------------
#
# Air is not nitrogen and oxygen. It is nitrogen and oxygen and argon and
# water and carbon dioxide, and the last two are the ones that decide
# whether the plant runs for a week or for an hour.
#
# THE MECHANISM. Carbon dioxide has no liquid phase at atmospheric
# pressure -- its triple point is at 5.18 bar, so below that it goes
# straight from gas to solid at about 195 K. The cold box runs at 100 K
# and below. So every molecule of CO2 that reaches the exchanger becomes
# a SOLID inside a passage a millimetre wide, and it stays there. Water
# does the same thing thirty degrees earlier and in far greater quantity.
#
# This is why every real air separation plant has a front end that is
# nothing to do with separation: a molecular sieve bed, or in older
# plants a reversing exchanger, whose entire job is to take out the water
# and the CO2 before the air is allowed to get cold. Skip it and the
# plant does not make bad product, it BLOCKS, and it blocks from the cold
# end where you cannot reach it.
#
# AND THE BLOCKAGE IS THE PRODUCT. What the front end catches is dry ice
# and distilled water, and both are worth having: dry ice is a cold store
# at 195 K that needs no vessel and no power, it is a fire suppressant,
# and it is the cold chain for anything perishable. A station that throws
# it away is throwing away the only refrigeration it can carry in a sack.

CO2_SUBLIMATION_K = 194.65
CO2_TRIPLE_PA = 518_000.0
CO2_LATENT_SUBLIME_J_KG = 571_000.0
DRY_ICE_DENSITY_KG_M3 = 1562.0
#: Atmospheric CO2 is ~420 ppm by volume; by MASS it is more, because CO2
#: is heavier than air. This is the conversion people skip.
CO2_AMBIENT_PPM_MASS = 420e-6 * (44.01 / 28.96)


def co2_freezes_at(pressure_pa: float = ATM_PA) -> float:
    """Where CO2 leaves the gas phase. Below the triple point there is no
    liquid at all -- it frosts directly onto the metal."""
    if pressure_pa >= CO2_TRIPLE_PA:
        return 216.59
    return CO2_SUBLIMATION_K


@dataclass
class FrontEndPurifier:
    """Molecular sieve: takes out the water and the CO2, and keeps them.

    Sized the way a real one is -- by how much it can hold before it has
    to be regenerated, not by a flow rate. It saturates, and a saturated
    bed passes what it used to catch straight into the cold box, which is
    why the plant fails downstream of the part that is actually worn
    out."""
    identity: str = "cryo.front_end"
    water_capacity_kg: float = 12.0
    co2_capacity_kg: float = 1.8
    water_held_kg: float = 0.0
    co2_held_kg: float = 0.0
    #: What the bed catches while it has room. Sieve beds are very good
    #: and this is not a fudge -- but they are not perfect, and the slip
    #: is what eventually frosts the exchanger.
    capture_frac: float = 0.999
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def co2_saturation(self) -> float:
        return min(1.0, self.co2_held_kg / max(1e-6, self.co2_capacity_kg))

    @property
    def water_saturation(self) -> float:
        return min(1.0, self.water_held_kg / max(1e-6, self.water_capacity_kg))

    @property
    def saturation(self) -> float:
        return max(self.co2_saturation, self.water_saturation)

    def effective_capture(self) -> float:
        """A bed that is nearly full stops working before it is full.

        The mass transfer zone reaches the outlet and breakthrough starts
        -- gradually, which is the dangerous way for it to happen."""
        s = self.saturation
        if s >= 1.0:
            return 0.0
        return self.capture_frac * (1.0 - max(0.0, (s - 0.85) / 0.15) ** 2 if s > 0.85 else 1.0)

    def process(self, air_kg: float, humidity_kg_per_kg: float = 0.004) -> dict:
        """Push air through. Returns what was caught and what SLIPPED.

        The slip is the important number: it is what arrives at a 100 K
        exchanger as a solid."""
        air = max(0.0, float(air_kg))
        co2_in = air * CO2_AMBIENT_PPM_MASS
        h2o_in = air * max(0.0, humidity_kg_per_kg)
        eff = self.effective_capture()
        co2_caught = min(co2_in * eff, max(0.0, self.co2_capacity_kg - self.co2_held_kg))
        h2o_caught = min(h2o_in * eff, max(0.0, self.water_capacity_kg - self.water_held_kg))
        self.co2_held_kg += co2_caught
        self.water_held_kg += h2o_caught
        return {"co2_caught_kg": co2_caught, "co2_slip_kg": co2_in - co2_caught,
                "water_caught_kg": h2o_caught, "water_slip_kg": h2o_in - h2o_caught,
                "saturation": self.saturation, "capture_frac": eff,
                "cold_box_fouling_kg": (co2_in - co2_caught) + (h2o_in - h2o_caught)}

    def regenerate(self) -> dict:
        """Heat it and purge it. THIS is where the dry ice comes from.

        Real regeneration drives the CO2 off as warm gas and it is vented;
        catching it as solid means taking the purge stream and cooling it
        below 195 K, which the plant can do because the plant is already
        the coldest thing on the site."""
        co2, h2o = self.co2_held_kg, self.water_held_kg
        self.co2_held_kg = 0.0
        self.water_held_kg = 0.0
        return {"co2_kg": co2, "water_kg": h2o,
                "dry_ice_l": co2 / DRY_ICE_DENSITY_KG_M3 * 1000.0,
                "cold_in_dry_ice_j": co2 * CO2_LATENT_SUBLIME_J_KG,
                "why": "the bed's contents are the plant's only two by-products that "
                       "need no vessel: solid CO2 that keeps itself cold, and water "
                       "that has been through a molecular sieve and is therefore "
                       "cleaner than anything else on site"}

    def need(self):
        import servicing as sv
        if self.saturation < 0.75:
            return None
        return sv.Need(
            identity=self.identity, want="regenerate", position=tuple(self.position),
            quantity=self.saturation, unit="frac", matches="heat-purge",
            urgency=1.0 if self.saturation > 0.95 else 0.6,
            minutes=90.0, skill="operator",
            label=(f"{self.identity} sieve {self.saturation * 100:.0f}% saturated "
                   f"({self.co2_held_kg:.2f} kg CO2, {self.water_held_kg:.1f} kg water)"),
            why="a saturated bed passes CO2 into a 100 K exchanger where it becomes a "
                "solid plug at the end you cannot reach; regenerating recovers dry ice "
                "and distilled water as well as saving the cold box")


# ---------------------------------------------------------------------
# THE PLANT: a dewar receives, it does not produce
# ---------------------------------------------------------------------
#
# A vacuum-jacketed vessel is a bucket. It holds what arrives and it
# slows down what leaves, and that is the whole of its contribution. The
# liquid has to be MADE, and making it is a work-input process with no
# way around the bill.
#
# WHERE THE COLD ACTUALLY COMES FROM. Not from the expander. The expander
# converts; the COMPRESSOR is what pays. Work goes in at the compressor,
# comes straight back out as heat in the aftercooler, and what is left
# behind is gas at pressure -- which is gas holding less entropy than it
# had. The expander then spends that stored entropy deficit as
# temperature drop. Skip the compressor and the expander has nothing to
# spend, so it does nothing at all.
#
# A LIQUEFIER WITHOUT A COMPRESSOR IS A PERPETUAL MOTION MACHINE, and
# this refuses to build one, in the same way and for the same reason
# that the driveline refuses a massless gear. A default value on an
# `inlet_pressure_pa` field is not a supply; it is an assumption wearing
# a supply's clothes, and it will silently produce free liquid air.
#
# AND THE EXPANDER MUST BE BRAKED. A turbine that is not absorbing shaft
# work does not cool its gas -- it freewheels, the gas comes out very
# nearly as warm as it went in, and the wheel overspeeds. The brake is
# not an accessory, it is where the energy LEAVES, and on a real machine
# it is a brake compressor (which usefully boosts the feed) or a
# generator. An unbraked expander is the second way to get nothing.
#
# THE RECUPERATOR IS THE THIRD, AND IT IS A CLIFF RATHER THAN A SLOPE.
# One expansion from ambient does not reach 79 K and nothing like it.
# The cycle BOOTSTRAPS: returning cold gas precools the incoming feed
# through a counterflow exchanger, so each pass starts colder than the
# last and the cold end walks down over hours. That only works if the
# exchanger is very good. Below about eighty-five per cent effectiveness
# the losses per pass exceed the gain per pass, the cascade never
# converges, and the plant simply runs for ever at some warm equilibrium
# making nothing. It does not make less liquid -- it makes none.
#
# Which is also why a real plant has a COOL-DOWN measured in hours to
# days, and why nobody starts an air separation unit casually.

#: Reversible work to liquefy air from ambient: its exergy, about
#: 745 kJ/kg. No plant beats this, and one that appears to is a bug in
#: the plant's arithmetic, not a good plant.
AIR_LIQUEFACTION_EXERGY_J_KG = 745_000.0
#: A COARSE PRE-CHECK, NOT THE REAL CLIFF.
#:
#: This started life as the asserted threshold below which a cascade
#: will not converge. It is superseded: Liquefier.step now works the
#: refrigeration balance directly, and the cliff falls out of that on
#: its own -- at around 0.92 to 0.95 for a plant of this size, which is
#: higher than this figure and is why real air liquefiers are built with
#: recuperator effectiveness of 0.95 to 0.98 and are genuinely that
#: fragile.
#:
#: It is kept only as a cheap guard that catches an exchanger so poor it
#: is not worth running the balance on. Anything between this and the
#: real cliff is caught by the balance and reported as a STALL, with the
#: two competing terms in the result so the reason is visible rather
#: than asserted.
RECUPERATOR_CLIFF = 0.85


class UnpoweredLiquefier(ValueError):
    """Raised when something would make cold without paying for it."""


@dataclass
class Recuperator:
    """The counterflow exchanger the whole cycle stands on."""
    identity: str = "cryo.recuperator"
    effectiveness: float = 0.95
    #: Cold-box thermal mass. This is what makes cool-down take hours:
    #: several hundred kilos of aluminium exchanger and piping has to be
    #: taken down two hundred kelvin before any product appears.
    cold_mass_kg: float = 350.0
    cold_mass_cp_j_kgk: float = 500.0
    #: Frozen CO2 or water blocking passages. The front end exists to
    #: keep this at zero; when it is not zero the exchanger stops being
    #: good enough and the cliff does the rest.
    fouled_frac: float = 0.0

    @property
    def working_effectiveness(self) -> float:
        return max(0.0, self.effectiveness * (1.0 - self.fouled_frac))

    @property
    def converges(self) -> bool:
        """Whether this exchanger can bootstrap a cascade at all."""
        return self.working_effectiveness >= RECUPERATOR_CLIFF

    @property
    def heat_capacity_j_k(self) -> float:
        return max(1.0, self.cold_mass_kg * self.cold_mass_cp_j_kgk)


@dataclass
class Liquefier:
    """Compressor, recuperator, expander, separator -- then the dewar.

    Refuses to exist without a compressor, because that is the part that
    pays for the cold and everything downstream merely spends it."""
    identity: str = "cryo.liquefier"
    #: NO DEFAULT. The compressor must be declared. This is the law.
    compressor_duty: str = ""
    expander: "Turboexpander" = None
    recuperator: Recuperator = None
    mass_flow_kg_s: float = 0.05
    #: Fraction of the compressed feed sent through the expander rather
    #: than the throttle. The Claude cycle's one real design knob.
    expander_split: float = 0.6
    brake: str = "brake-compressor"     # or "generator", or "" for none
    ambient_k: float = 300.0
    #: Heat leaking into the cold box through its perlite and its pipe
    #: penetrations. Small in watts and decisive at the end of cool-down,
    #: because by then the net refrigeration is small too.
    ambient_leak_w: float = 120.0
    #: State: how cold the cold end has actually got. Starts at ambient,
    #: which is why a cold start makes nothing.
    cold_end_k: float = 300.0
    running_hours: float = 0.0
    produced_kg: float = 0.0
    energy_kwh: float = 0.0
    position: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.expander is None:
            self.expander = Turboexpander()
        if self.recuperator is None:
            self.recuperator = Recuperator()
        self.cold_end_k = float(self.ambient_k)
        problems = self.shortfalls()
        if problems:
            raise UnpoweredLiquefier(
                "a liquefier cannot make cold it has not paid for:\n  "
                + "\n  ".join(problems))

    def shortfalls(self) -> list:
        """Everything about this plant that would produce free cold."""
        out = []
        if not self.compressor_duty:
            out.append(
                f"{self.identity}: no compressor declared. The dewar is a receiver "
                "and the expander only spends what compression stored -- with no "
                "compressor there is no pressure, no entropy deficit, and no liquid. "
                "Declare a compressor_duty from CRYO_COMPRESSORS")
        else:
            try:
                compressor_duty(self.compressor_duty)
            except KeyError as e:
                out.append(f"{self.identity}: {e}")
        if not self.brake:
            out.append(
                f"{self.identity}: expander has no brake. An unloaded turbine does "
                "not cool its gas, it freewheels -- the shaft work is where the "
                "energy leaves, and with nowhere for it to go the gas comes out as "
                "warm as it went in")
        if self.expander is not None and self.expander.pressure_ratio <= 1.001:
            out.append(
                f"{self.identity}: expander pressure ratio is 1.0 -- nothing to "
                "expand. The inlet pressure has to be SUPPLIED by the compressor, "
                "not assumed by a field default")
        return out

    @property
    def pressure_ratio(self) -> float:
        return self.expander.pressure_ratio

    def compressor_power_w(self) -> float:
        """What the plant actually draws. The bill for all of it."""
        return compression_power_w(self.compressor_duty, self.mass_flow_kg_s,
                                   self.ambient_k, self.pressure_ratio)

    def aftercooler_duty_w(self) -> float:
        """Heat rejected at the aftercooler.

        Worth reporting because it is LARGER than the cooling the plant
        produces -- every watt of compression comes straight back out
        here, and it has to go somewhere. On this station that somewhere
        should be the ice pit rather than a radiator."""
        return self.compressor_power_w()

    def step(self, dt_s: float, product: str = "liquid-air") -> dict:
        """Run the cycle for an interval, including the cool-down.

        Produces nothing until the cold end has actually walked down to
        the boiling point, which is the honest behaviour and takes
        hours."""
        dt = max(0.0, float(dt_s))
        power = self.compressor_power_w()
        self.energy_kwh += power * dt / 3.6e6
        self.running_hours += dt / 3600.0

        if not self.recuperator.converges:
            return {"liquid_kg": 0.0, "cold_end_k": self.cold_end_k,
                    "cooling_down": True, "converges": False,
                    "power_w": power,
                    "why": (f"recuperator at {self.recuperator.working_effectiveness:.2f} "
                            f"is below the {RECUPERATOR_CLIFF:.2f} cliff: each pass "
                            "loses more than it gains, so the cascade never converges "
                            "and the plant runs for ever making nothing")}

        # THE CASCADE, DONE AS A BALANCE RATHER THAN A RATE.
        #
        # The expander's cooling does NOT go into the cold box. Nearly
        # all of it goes into cooling the incoming feed stream, which
        # arrives at ambient every single pass and has to be taken down
        # the whole span. What is left over after that -- the NET
        # refrigeration -- is the only thing that can pull the box down,
        # and it is a small difference between two large numbers.
        #
        # The loss term is what the counterflow exchanger fails to
        # recover: (1 - effectiveness) of the full warm-to-cold span,
        # every pass. So as the cold end falls the span opens and the
        # loss GROWS, while the expander's contribution does not grow as
        # fast. They meet, and where they meet is the plant's floor.
        #
        # This is also where the cliff comes from, without having to be
        # asserted anywhere: at poor effectiveness the loss term beats
        # the expander term before the span ever reaches a boiling
        # point, and the plant simply sits warm.
        cp = self.expander.cp_j_kgk
        out_k = self.expander.outlet_temp_k(self.cold_end_k)
        expander_w = (self.mass_flow_kg_s * self.expander_split * cp
                      * max(0.0, self.cold_end_k - out_k))
        span = max(0.0, self.ambient_k - self.cold_end_k)
        warm_end_loss_w = (self.mass_flow_kg_s * cp
                           * (1.0 - self.recuperator.working_effectiveness) * span)
        net_w = expander_w - warm_end_loss_w - self.ambient_leak_w
        c = cryogen(product)
        floor = c.boil_k
        if net_w <= 0.0:
            # the plant has found its equilibrium and it is warm
            self.cold_end_k = min(self.ambient_k,
                                  self.cold_end_k - net_w * dt
                                  / self.recuperator.heat_capacity_j_k)
            return {"liquid_kg": 0.0, "liquid_frac": 0.0,
                    "cold_end_k": self.cold_end_k, "outlet_k": out_k,
                    "cooling_down": False, "converges": True, "stalled": True,
                    "power_w": power, "net_refrigeration_w": net_w,
                    "expander_w": expander_w, "warm_end_loss_w": warm_end_loss_w,
                    "aftercooler_w": self.aftercooler_duty_w(),
                    "hours": self.running_hours,
                    "why": "net refrigeration has gone negative: the warm-end loss now "
                           "exceeds what the expander delivers, so this is as cold as "
                           "this plant gets and it will sit here indefinitely"}
        drop = net_w * dt / self.recuperator.heat_capacity_j_k
        # never step past the target in one interval -- the same explicit
        # -integration overshoot that would otherwise ring
        drop = min(drop, max(0.0, self.cold_end_k - floor))
        self.cold_end_k = max(floor, self.cold_end_k - drop)

        liquid_kg = 0.0
        frac = 0.0
        if self.cold_end_k <= floor + 1.0:
            frac = self.expander.liquid_fraction(self.cold_end_k, product)
            liquid_kg = self.mass_flow_kg_s * frac * dt
            self.produced_kg += liquid_kg
        return {"liquid_kg": liquid_kg, "liquid_frac": frac,
                "cold_end_k": self.cold_end_k, "outlet_k": out_k,
                "cooling_down": self.cold_end_k > floor + 1.0,
                "converges": True, "stalled": False, "power_w": power,
                "net_refrigeration_w": net_w, "expander_w": expander_w,
                "warm_end_loss_w": warm_end_loss_w,
                "aftercooler_w": self.aftercooler_duty_w(),
                "hours": self.running_hours}

    def specific_work_kwh_kg(self) -> float:
        """What this plant actually spends per kilogram of product."""
        if self.produced_kg <= 0.0:
            return float("inf")
        return self.energy_kwh / self.produced_kg

    def efficiency_report(self) -> dict:
        """Against the thermodynamic floor, which nothing beats."""
        floor = AIR_LIQUEFACTION_EXERGY_J_KG / 3.6e6
        actual = self.specific_work_kwh_kg()
        return {"specific_kwh_kg": actual, "reversible_kwh_kg": floor,
                "second_law_efficiency": (floor / actual) if actual > 0 else 0.0,
                "violates_second_law": actual < floor,
                "why": "the reversible work to liquefy air is its exergy, about "
                       "0.21 kWh/kg. Real small plants land between 1.0 and 1.5, so a "
                       "second-law efficiency near 0.15-0.20 is honest and anything "
                       "above 1.0 is an arithmetic error"}


def feed_dewar(plant: Liquefier, vessel, dt_s: float,
               product: str = "liquid-air") -> dict:
    """Run the plant and put what it made into the vessel.

    The only sanctioned way liquid enters a dewar: it has to come from a
    plant that paid for it."""
    r = plant.step(dt_s, product)
    kg = r.get("liquid_kg", 0.0)
    if kg > 0.0:
        c = cryogen(product)
        litres = kg / c.liquid_density_kg_m3 * 1000.0
        room = max(0.0, vessel.capacity_l - vessel.fill_l)
        took = min(litres, room)
        vessel.fill_l += took
        r["into_vessel_l"] = took
        r["overflow_l"] = litres - took
    else:
        r["into_vessel_l"] = 0.0
        r["overflow_l"] = 0.0
    return r
