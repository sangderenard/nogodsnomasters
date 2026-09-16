"""The column, the condenser-reboiler, and the bottles at the end of it.

Air separation is not a function that turns litres of liquid air into
litres of products. It is a building full of very specific machinery, and
almost every interesting failure and cost lives in the machinery rather
than in the thermodynamics.

WHAT ACTUALLY SEPARATES THINGS. Nitrogen boils 13 K below oxygen, so the
vapour above boiling air is richer in nitrogen than the liquid is. Do
that once and you have moved the composition a little; the measure of
how much is the RELATIVE VOLATILITY, and for nitrogen over oxygen it is
about 3.5 near atmospheric pressure. Do it fifty times in series, each
stage handing its vapour up and its liquid down, and you get pure
product. That series of stages is the column, and its height is the
separation.

ARGON IS WHAT MAKES COLUMNS ENORMOUS. Argon boils between the two, at
87.3 K, and its relative volatility against oxygen is about 1.5. The
minimum stage count goes as 1/ln(alpha), so a job with alpha 1.5 needs
roughly four and a half times the stages of the same job at alpha 3.5 --
and because argon is only one per cent of air to begin with, the crude
argon column is the tallest thing on the site by a wide margin. Real ones
run to two hundred trays and forty metres. A station that wants argon is
choosing to build a landmark.

THE DOUBLE COLUMN IS THE TRICK WORTH UNDERSTANDING. You would think you
need a refrigerator at the top of the column to make reflux and a heater
at the bottom to make boil-up. You need neither, because of one clever
arrangement: run the lower column at about 5.5 bar and the upper column
at about 1.4 bar. Nitrogen condensing at the top of the high-pressure
column sits at roughly 94 K. Oxygen boiling at the bottom of the
low-pressure column sits at roughly 92 K. So the same heat exchanger is
a condenser on one side and a reboiler on the other, and the entire
column runs on a two-kelvin temperature difference that exists ONLY
because of the pressure difference. That single piece of kit is what
makes air separation cheap enough to be worth doing.

IF THE CONDENSER-REBOILER FLOODS OR FOULS, EVERYTHING STOPS. It lives
under a head of liquid oxygen, and liquid oxygen with any hydrocarbon
dissolved in it is an explosive. This is why the front end exists and why
it is not optional: the sieve is not protecting product quality, it is
protecting the reboiler. Air separation plants have historically been
destroyed by exactly this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ATM_PA = 101_325.0
#: Molar volume at 0 C and 1 atm -- "normal" cubic metres, which is how
#: industrial gas is actually sold. Not the same as a cubic metre of gas
#: at room temperature, and the difference is about 10%.
NORMAL_MOLAR_VOLUME_M3 = 0.022414

AIR_MOLE_FRAC = {"nitrogen": 0.7808, "oxygen": 0.2095, "argon": 0.0093}
GAS_DENSITY_KG_NM3 = {"nitrogen": 1.2504, "oxygen": 1.4290, "argon": 1.7840}


# ---------------------------------------------------------------------
# separations, as real column arithmetic
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Separation:
    """One binary split, with the volatility that decides its cost."""
    key: str
    label: str
    light: str
    heavy: str
    alpha: float                 # relative volatility, light over heavy
    #: Height equivalent to a theoretical plate. Structured packing beats
    #: trays and is what modern columns use; it is also what makes a
    #: column that would have been sixty metres into one that is twenty.
    hetp_m: float
    why: str = ""


SEPARATIONS: dict[str, Separation] = {
    "n2-o2-lp": Separation(
        "n2-o2-lp", "nitrogen from oxygen, low pressure", "nitrogen", "oxygen",
        alpha=3.5, hetp_m=0.25,
        why="the easy one, and the reason the low-pressure column is a sane height. "
            "Thirteen kelvin of boiling point difference is a lot to work with"),
    "n2-o2-hp": Separation(
        "n2-o2-hp", "nitrogen from oxygen, high pressure", "nitrogen", "oxygen",
        alpha=2.6, hetp_m=0.25,
        why="the same split done at 5.5 bar, where it is HARDER -- volatility falls "
            "as pressure rises, and everything converges at the critical point. The "
            "high-pressure column is not run at pressure because that helps the "
            "separation; it is run at pressure so its condenser can boil the other "
            "column's sump"),
    "ar-o2": Separation(
        "ar-o2", "argon from oxygen", "argon", "oxygen",
        alpha=1.5, hetp_m=0.20,
        why="the expensive one. Volatility 1.5 against 3.5 means four and a half "
            "times the stages for the same purity, and argon is one per cent of the "
            "feed, so the column is both very tall and very thin"),
    "n2-ar": Separation(
        "n2-ar", "nitrogen from argon", "nitrogen", "argon",
        alpha=2.3, hetp_m=0.20,
        why="the second half of the argon problem: crude argon comes off the belly of "
            "the low-pressure column carrying nitrogen as well as oxygen, so it has to "
            "be cleaned at both ends"),
}


def separation(key: str) -> Separation:
    s = SEPARATIONS.get(str(key))
    if s is None:
        raise KeyError(f"unknown separation {key!r}; declared: {', '.join(sorted(SEPARATIONS))}")
    return s


def fenske_min_stages(alpha: float, x_distillate: float, x_bottoms: float) -> float:
    """Minimum theoretical stages, at total reflux.

    Fenske. This is the floor: no column achieves this purity in fewer
    stages no matter how much energy is thrown at it, because at total
    reflux no product is being withdrawn at all. Everything real is
    worse, and the whole design question is how much worse."""
    a = max(1.0001, float(alpha))
    xd = min(0.999999, max(1e-6, float(x_distillate)))
    xb = min(0.999999, max(1e-6, float(x_bottoms)))
    return math.log((xd / (1.0 - xd)) * ((1.0 - xb) / xb)) / math.log(a)


def underwood_min_reflux(alpha: float, x_feed: float, x_distillate: float,
                         q: float = 1.0) -> float:
    """Minimum reflux ratio for a binary, saturated-liquid feed.

    The other floor, and the one that costs energy rather than steel. A
    column can trade stages against reflux -- a taller column runs on
    less boil-up -- and these two equations are the ends of that trade."""
    a = max(1.0001, float(alpha))
    xf = min(0.999999, max(1e-6, float(x_feed)))
    xd = min(0.999999, max(1e-6, float(x_distillate)))
    return (1.0 / (a - 1.0)) * (xd / xf - a * (1.0 - xd) / (1.0 - xf))


def gilliland_stages(n_min: float, r_min: float, reflux_ratio: float) -> float:
    """Actual stages from the two minima and the reflux you chose.

    Gilliland's correlation: empirical, fitted to real columns, and the
    reason column design is not simply solved. It captures the thing that
    matters -- that going from 1.1x minimum reflux to 1.5x buys a lot of
    stages, and going from 1.5x to 2x buys almost none."""
    r = max(float(r_min) * 1.001, float(reflux_ratio))
    x = (r - r_min) / (r + 1.0)
    y = 1.0 - math.exp(((1.0 + 54.4 * x) / (11.0 + 117.2 * x))
                       * ((x - 1.0) / math.sqrt(max(1e-9, x))))
    return (float(n_min) + y) / max(1e-6, 1.0 - y)


@dataclass
class Column:
    """A distillation column sized the way a real one is.

    Stages, reflux and height are not independent and this refuses to
    pretend they are: choose a purity and a reflux ratio and the height
    follows, and if the height is absurd the purity was."""
    identity: str = "asu.column"
    separation_key: str = "n2-o2-lp"
    feed_frac: float = 0.2095        # mole fraction of the LIGHT key in feed
    distillate_purity: float = 0.995
    bottoms_impurity: float = 0.01
    reflux_over_minimum: float = 1.25
    tray_efficiency: float = 0.75    # packing is better; trays are not ideal
    diameter_m: float = 0.5
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def spec(self) -> Separation:
        return separation(self.separation_key)

    @property
    def min_stages(self) -> float:
        return fenske_min_stages(self.spec.alpha, self.distillate_purity,
                                 self.bottoms_impurity)

    @property
    def min_reflux(self) -> float:
        return max(0.05, underwood_min_reflux(self.spec.alpha, self.feed_frac,
                                              self.distillate_purity))

    @property
    def reflux_ratio(self) -> float:
        return self.min_reflux * max(1.01, self.reflux_over_minimum)

    @property
    def theoretical_stages(self) -> float:
        return gilliland_stages(self.min_stages, self.min_reflux, self.reflux_ratio)

    @property
    def actual_stages(self) -> float:
        """Real stages, because no tray reaches equilibrium."""
        return self.theoretical_stages / max(0.1, self.tray_efficiency)

    @property
    def height_m(self) -> float:
        """What has to be built, and the number that ends arguments."""
        return self.actual_stages * self.spec.hetp_m

    @property
    def boilup_ratio(self) -> float:
        """Vapour up the column per unit of product out of the top.

        This is the energy demand: everything that goes up has to be
        boiled and everything that comes down has to be condensed."""
        return self.reflux_ratio + 1.0

    def report(self) -> dict:
        return {"identity": self.identity, "separation": self.spec.label,
                "alpha": self.spec.alpha, "min_stages": self.min_stages,
                "min_reflux": self.min_reflux, "reflux_ratio": self.reflux_ratio,
                "theoretical_stages": self.theoretical_stages,
                "actual_stages": self.actual_stages, "height_m": self.height_m,
                "boilup_ratio": self.boilup_ratio}


# ---------------------------------------------------------------------
# the piece of kit that makes it all work
# ---------------------------------------------------------------------

@dataclass
class CondenserReboiler:
    """One exchanger, condensing on one face and boiling on the other.

    THE TEMPERATURE CROSSOVER IS THE ENTIRE DESIGN. Nitrogen condensing
    at high-column pressure must be WARMER than oxygen boiling at
    low-column pressure, or nothing moves. That margin is a couple of
    kelvin, and it is bought entirely with the pressure ratio: drop the
    high column's pressure and the crossover closes and the plant stops
    -- not gradually, but as a hard stall with no heat flowing.

    AND IT SITS UNDER LIQUID OXYGEN. Any hydrocarbon that got past the
    front end concentrates in that sump, because the oxygen boils away
    and the hydrocarbon does not. That is the mechanism behind the
    industry's worst accidents and the reason the sieve bed's service
    interval is not a housekeeping matter."""
    identity: str = "asu.condenser_reboiler"
    hp_pressure_pa: float = 550_000.0
    lp_pressure_pa: float = 140_000.0
    area_m2: float = 40.0
    u_w_m2k: float = 900.0           # boiling/condensing both sides: very good
    hydrocarbon_ppm: float = 0.0     # what slipped past the sieve
    flooded_frac: float = 0.0

    @staticmethod
    def boiling_k(species: str, pressure_pa: float) -> float:
        """Clausius-Clapeyron off the normal boiling point.

        Crude, but it has the right slope, and the slope is the only
        thing this is used for."""
        nbp = {"nitrogen": 77.36, "oxygen": 90.19, "argon": 87.30}[species]
        lat = {"nitrogen": 199_000.0, "oxygen": 213_000.0, "argon": 161_000.0}[species]
        mw = {"nitrogen": 0.028, "oxygen": 0.032, "argon": 0.040}[species]
        r_spec = 8.314 / mw
        p = max(1_000.0, float(pressure_pa))
        inv = 1.0 / nbp - (r_spec / lat) * math.log(p / ATM_PA)
        return 1.0 / max(1e-6, inv)

    @property
    def condensing_k(self) -> float:
        return self.boiling_k("nitrogen", self.hp_pressure_pa)

    @property
    def boiling_o2_k(self) -> float:
        return self.boiling_k("oxygen", self.lp_pressure_pa)

    @property
    def crossover_k(self) -> float:
        """The margin the whole plant runs on. Two kelvin is normal."""
        return self.condensing_k - self.boiling_o2_k

    @property
    def stalled(self) -> bool:
        return self.crossover_k <= 0.0

    def duty_w(self) -> float:
        if self.stalled:
            return 0.0
        ua = self.area_m2 * self.u_w_m2k * max(0.0, 1.0 - self.flooded_frac)
        return ua * self.crossover_k

    @property
    def hazard(self) -> str:
        """Hydrocarbons concentrate in a boiling LOX sump. They do not
        pass through; the oxygen leaves and they stay."""
        if self.hydrocarbon_ppm <= 0.05:
            return "clean"
        if self.hydrocarbon_ppm < 0.5:
            return "watch: sieve slipping"
        if self.hydrocarbon_ppm < 5.0:
            return "DRAIN THE SUMP: enrichment in progress"
        return "EXPLOSIVE: liquid oxygen with dissolved hydrocarbon"

    def report(self) -> dict:
        return {"condensing_k": self.condensing_k, "boiling_o2_k": self.boiling_o2_k,
                "crossover_k": self.crossover_k, "stalled": self.stalled,
                "duty_kw": self.duty_w() / 1000.0, "hazard": self.hazard}


# ---------------------------------------------------------------------
# the whole plant, and what it costs
# ---------------------------------------------------------------------

#: A real large air separation unit lands near 0.35 kWh per normal cubic
#: metre of oxygen. Small plants are much worse, because the losses that
#: scale with surface area do not shrink as fast as the throughput does.
LARGE_PLANT_KWH_PER_NM3_O2 = 0.35


def specific_power_kwh_nm3(nm3_per_hour: float) -> float:
    """What a plant of THIS size really costs per unit of product.

    Small cryogenic plants are inefficient and it is not a small effect:
    heat leak is a surface-area term and throughput is a volume term, so
    halving the plant does not halve the loss. A station-scale unit runs
    at two to three times the textbook figure and pretending otherwise
    makes the whole economy wrong."""
    q = max(1.0, float(nm3_per_hour))
    scale_penalty = 1.0 + 2.2 * (1000.0 / (1000.0 + q))
    return LARGE_PLANT_KWH_PER_NM3_O2 * scale_penalty


@dataclass
class AirSeparationUnit:
    """The plant: front end, double column, argon side, cold box.

    Products are what the COLUMNS achieve, at the purities their heights
    support -- there is no path here that produces gas without a column
    tall enough to have made it."""
    identity: str = "asu.plant"
    feed_nm3_per_hour: float = 120.0
    hp_column: Column = field(default_factory=lambda: Column(
        identity="asu.hp_column", separation_key="n2-o2-hp", feed_frac=0.7808,
        distillate_purity=0.99, bottoms_impurity=0.40, diameter_m=0.45))
    lp_column: Column = field(default_factory=lambda: Column(
        identity="asu.lp_column", separation_key="n2-o2-lp", feed_frac=0.60,
        distillate_purity=0.9999, bottoms_impurity=0.005, diameter_m=0.60))
    argon_column: Column | None = None
    reboiler: CondenserReboiler = field(default_factory=CondenserReboiler)
    #: Argon is optional and expensive, and a plant without it simply
    #: sends the argon out with the oxygen, which is why "99.5% oxygen"
    #: is the normal grade and 99.999% is a different plant.
    argon_fitted: bool = False
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def cold_box_height_m(self) -> float:
        """Everything stands inside one perlite-filled steel box, and the
        box is as tall as the tallest thing in it plus its sump."""
        h = max(self.hp_column.height_m + self.lp_column.height_m, 1.0) + 2.0
        if self.argon_fitted and self.argon_column is not None:
            h = max(h, self.argon_column.height_m + 2.0)
        return h

    def products_nm3_per_hour(self) -> dict:
        """What comes out, at the purity the columns actually reach."""
        feed = max(0.0, self.feed_nm3_per_hour)
        if self.reboiler.stalled:
            return {"stalled": True, "oxygen": {}, "nitrogen": {}, "argon": {}}
        # Recovery is not 100%: some oxygen leaves with the waste
        # nitrogen, and a small plant is worse at this than a large one.
        o2_recovery = 0.85 if not self.argon_fitted else 0.95
        o2 = feed * AIR_MOLE_FRAC["oxygen"] * o2_recovery
        n2 = feed * AIR_MOLE_FRAC["nitrogen"] * 0.55     # the rest is waste N2
        out = {
            "stalled": False,
            "oxygen": {"nm3_h": o2, "purity": self.lp_column.distillate_purity
                       if False else self._oxygen_purity(),
                       "kg_h": o2 * GAS_DENSITY_KG_NM3["oxygen"]},
            "nitrogen": {"nm3_h": n2, "purity": self.hp_column.distillate_purity,
                         "kg_h": n2 * GAS_DENSITY_KG_NM3["nitrogen"]},
        }
        if self.argon_fitted and self.argon_column is not None:
            ar = feed * AIR_MOLE_FRAC["argon"] * 0.70
            out["argon"] = {"nm3_h": ar,
                            "purity": self.argon_column.distillate_purity,
                            "kg_h": ar * GAS_DENSITY_KG_NM3["argon"]}
        else:
            out["argon"] = {"nm3_h": 0.0, "purity": 0.0, "kg_h": 0.0,
                            "note": "argon leaves with the oxygen, which is what caps "
                                    "oxygen purity at about 99.5%"}
        return out

    def _oxygen_purity(self) -> float:
        """Without an argon column, oxygen cannot beat about 99.5%.

        Not because the low-pressure column is short, but because argon
        boils between nitrogen and oxygen and has nowhere else to go. The
        column will happily reject the nitrogen and keep the argon."""
        if self.argon_fitted:
            return min(0.99999, self.lp_column.distillate_purity)
        return min(0.995, self.lp_column.distillate_purity)

    def power_kw(self) -> float:
        o2 = self.products_nm3_per_hour().get("oxygen", {}).get("nm3_h", 0.0)
        if o2 <= 0.0:
            return 0.0
        return specific_power_kwh_nm3(o2) * o2

    def report(self) -> dict:
        p = self.products_nm3_per_hour()
        return {"identity": self.identity, "feed_nm3_h": self.feed_nm3_per_hour,
                "cold_box_height_m": self.cold_box_height_m,
                "hp": self.hp_column.report(), "lp": self.lp_column.report(),
                "argon": (self.argon_column.report() if
                          (self.argon_fitted and self.argon_column) else None),
                "reboiler": self.reboiler.report(),
                "products": p, "power_kw": self.power_kw(),
                "kwh_per_nm3_o2": specific_power_kwh_nm3(
                    p.get("oxygen", {}).get("nm3_h", 1.0))}


def argon_column(purity: float = 0.995) -> Column:
    """The tall thin one. Built separately because it is a decision."""
    return Column(identity="asu.argon_column", separation_key="ar-o2",
                  feed_frac=0.10, distillate_purity=purity,
                  bottoms_impurity=0.002, reflux_over_minimum=1.15,
                  tray_efficiency=0.55, diameter_m=0.30)


# ---------------------------------------------------------------------
# bottling
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CylinderSpec:
    key: str
    label: str
    water_volume_l: float
    service_pressure_pa: float
    tare_kg: float
    why: str = ""

    def capacity_nm3(self, species: str = "oxygen") -> float:
        """Gas in it, allowing for the fact that it is not ideal at 200 bar.

        Real compressibility at service pressure is a few per cent the
        wrong way for oxygen, and it is the difference between a cylinder
        holding what the label says and holding less."""
        z = {"oxygen": 1.06, "nitrogen": 1.04, "argon": 1.02}.get(species, 1.05)
        return (self.water_volume_l / 1000.0) * (self.service_pressure_pa / ATM_PA) / z

    def gas_kg(self, species: str = "oxygen") -> float:
        return self.capacity_nm3(species) * GAS_DENSITY_KG_NM3.get(species, 1.3)

    def gas_fraction_of_weight(self, species: str = "oxygen") -> float:
        """Why gas is shipped as liquid whenever there is enough of it."""
        g = self.gas_kg(species)
        return g / max(1e-6, g + self.tare_kg)


CYLINDERS: dict[str, CylinderSpec] = {
    "industrial-50l": CylinderSpec(
        "industrial-50l", "50 L industrial cylinder", 50.0, 200e5, 62.0,
        why="the standard one. Sixty kilos of steel around fourteen kilos of oxygen, "
            "which is a ratio that decides how gas moves around a theatre: as liquid "
            "if there is a dewar at both ends, and as steel if there is not"),
    "medical-10l": CylinderSpec(
        "medical-10l", "10 L medical cylinder", 10.0, 137e5, 14.0,
        why="lower pressure and smaller because somebody has to carry it, and because "
            "medical oxygen is regulated on purity rather than on quantity"),
    "portable-2l": CylinderSpec(
        "portable-2l", "2 L portable", 2.0, 200e5, 3.0,
        why="a few hundred litres of gas: minutes of breathing, or one small weld"),
}


def cylinder(key: str) -> CylinderSpec:
    c = CYLINDERS.get(str(key))
    if c is None:
        raise KeyError(f"unknown cylinder {key!r}; declared: {', '.join(sorted(CYLINDERS))}")
    return c


def compression_work_kwh_nm3(to_pressure_pa: float = 200e5, stages: int = 4,
                             isothermal_efficiency: float = 0.62,
                             inlet_k: float = 293.15) -> float:
    """Energy to put a normal cubic metre into a cylinder.

    Multi-stage with intercooling, which is not a refinement -- it is the
    only way to do this. Single-stage compression to 200 bar would leave
    the gas at well over a thousand kelvin, and in oxygen service that is
    not an efficiency problem, it is an ignition source."""
    n = max(1, int(stages))
    ratio = max(1.0001, float(to_pressure_pa) / ATM_PA)
    per_stage = ratio ** (1.0 / n)
    mol = 1.0 / NORMAL_MOLAR_VOLUME_M3
    work_j = n * mol * 8.314 * float(inlet_k) * math.log(per_stage)
    return work_j / 3.6e6 / max(0.05, isothermal_efficiency)


@dataclass
class BottlingPlant:
    """Compressor, intercoolers, and a filling manifold.

    OXYGEN SERVICE IS A DIFFERENT MACHINE. Everything wetted has to be
    degreased and stay degreased; the lubricant cannot be hydrocarbon;
    and the fill has to be slow, because rapid pressurisation of a
    cylinder heats the gas adiabatically at the valve and that is how
    oxygen fires start. A nitrogen compressor pressed into oxygen service
    is the single most likely way to burn this station down."""
    identity: str = "asu.bottling"
    stages: int = 4
    capacity_nm3_per_hour: float = 30.0
    service_pressure_pa: float = 200e5
    oxygen_clean: bool = True        # degreased, oxygen-compatible lubricant
    filled: dict = field(default_factory=dict)
    energy_kwh: float = 0.0
    position: tuple = (0.0, 0.0, 0.0)

    def fill(self, species: str, nm3: float, kind: str = "industrial-50l") -> dict:
        spec = cylinder(kind)
        if species == "oxygen" and not self.oxygen_clean:
            return {"filled": 0, "refused": True,
                    "why": "compressor is not in oxygen service: hydrocarbon lubricant "
                           "plus 200 bar of oxygen plus the heat of compression is an "
                           "ignition, not a risk of one"}
        per = spec.capacity_nm3(species)
        n = int(max(0.0, float(nm3)) // max(1e-6, per))
        kwh = compression_work_kwh_nm3(self.service_pressure_pa, self.stages) * n * per
        self.energy_kwh += kwh
        self.filled[species] = self.filled.get(species, 0) + n
        return {"filled": n, "refused": False, "cylinder": spec.label,
                "nm3_each": per, "kg_each": spec.gas_kg(species),
                "leftover_nm3": max(0.0, float(nm3)) - n * per,
                "energy_kwh": kwh, "hours": n * per / max(1e-6, self.capacity_nm3_per_hour),
                "gas_share_of_weight": spec.gas_fraction_of_weight(species)}


# ---------------------------------------------------------------------
# RUNNING A DEWAR DOWN THROUGH THE COIL, THEN BOTTLING WHAT COMES OFF
# ---------------------------------------------------------------------
#
# The sequence that makes a by-product pay twice: liquid oxygen goes into
# a coil in the ice room, where it boils and freezes the bank; the gas
# that comes off the far end is caught, buffered, compressed and sold.
# The same kilogram is refrigeration first and product second, and
# nothing is vented at any point.
#
# WHY THERE HAS TO BE A BUFFER. The coil boils continuously at whatever
# rate the room's heat allows, and the compressor runs in batches
# whenever there is power to spare. Without something in between, the
# coil's output either backs up and stalls the boiling or is vented --
# and venting it is the failure this whole arrangement exists to avoid.
# A low-pressure receiver is the cheapest part of the system and the one
# that makes the rest work.
#
# AND YOU CANNOT RUN IT DRY. The last of the liquid is the worst of it:
# as the level falls the wetted area falls with it, so the boiling rate
# collapses while the heat leak does not, and the vessel starts warming
# instead of boiling. Past that point the coil is a heat leak with a
# pump attached. The run stops with a heel in the bottom -- which is
# also what keeps the vessel cold for the next fill, so the heel is not
# waste.

#: Below this fill fraction the wetted area has fallen far enough that
#: the vessel warms rather than boils. Stopping here is not caution, it
#: is where the physics stops cooperating.
DEWAR_HEEL_FRAC = 0.08


@dataclass
class GasReceiver:
    """Low-pressure buffer between a coil that boils and a compressor
    that runs in batches."""
    identity: str = "asu.receiver"
    volume_m3: float = 4.0
    pressure_pa: float = ATM_PA
    max_pressure_pa: float = 8e5
    species: str = "oxygen"
    temp_k: float = 288.15

    @property
    def contents_nm3(self) -> float:
        return self.volume_m3 * (self.pressure_pa / ATM_PA)

    @property
    def full_frac(self) -> float:
        return self.pressure_pa / max(1.0, self.max_pressure_pa)

    def admit_kg(self, kg: float) -> dict:
        """Put gas in. Returns what had to be vented because it would not
        fit, which is the number the buffer exists to keep at zero."""
        rho = GAS_DENSITY_KG_NM3.get(self.species, 1.3)
        nm3 = max(0.0, float(kg)) / rho
        room_nm3 = max(0.0, (self.max_pressure_pa - self.pressure_pa)
                       / ATM_PA * self.volume_m3)
        taken = min(nm3, room_nm3)
        self.pressure_pa += taken / max(1e-6, self.volume_m3) * ATM_PA
        return {"admitted_nm3": taken, "vented_nm3": nm3 - taken,
                "full_frac": self.full_frac,
                "backpressure": nm3 > room_nm3}

    def draw_nm3(self, nm3: float) -> float:
        got = min(self.contents_nm3, max(0.0, float(nm3)))
        self.pressure_pa = max(ATM_PA, self.pressure_pa
                               - got / max(1e-6, self.volume_m3) * ATM_PA)
        return got


@dataclass
class RunDown:
    """Boil a vessel of cryogen down through a coil, catching everything.

    Orchestrates the three parts that have to agree: the vessel that is
    emptying, the coil that sets the rate, and the receiver that has to
    hold what the coil makes."""
    identity: str = "station.rundown"
    species: str = "oxygen"
    start_kg: float = 400.0
    remaining_kg: float = 400.0
    receiver: GasReceiver = field(default_factory=GasReceiver)
    cold_delivered_j: float = 0.0
    vented_kg: float = 0.0
    bottled: int = 0
    hours: float = 0.0
    #: METERING VALVE, and it is not optional.
    #:
    #: A coil that can absorb a hundred kilowatts from ice water boils
    #: liquid oxygen at over two tonnes an hour. A compressor that can
    #: bottle thirty normal cubic metres an hour takes forty-three
    #: kilograms of it. Those two numbers are a factor of fifty apart,
    #: and an unvalved coil resolves that difference by venting -- which
    #: is the exact failure this arrangement was built to avoid.
    #:
    #: So the rate is set at the LIQUID side by a metering valve, sized
    #: to whatever the downstream can actually swallow.
    #:
    #: AND METERING COSTS NOTHING IN COLD, which is not the obvious
    #: answer. The cooling is the latent heat of the liquid and it does
    #: not care how fast the liquid boils -- the same four hundred
    #: kilograms delivers the same hundred and fifty megajoules whether
    #: it goes in an hour or in nine. What changes is only the RATE at
    #: which the bank charges. So there is no trade between cold and
    #: product here at all: metering buys twenty-seven cylinders for
    #: nothing but patience, and the only reason to open the valve is
    #: needing the bank frozen tonight.
    feed_rate_kg_s: float | None = None

    def metered_rate_kg_s(self, plant: "BottlingPlant | None" = None) -> float:
        """What the valve is actually set to."""
        if self.feed_rate_kg_s is not None:
            return max(0.0, self.feed_rate_kg_s)
        if plant is None:
            return float("inf")
        return (plant.capacity_nm3_per_hour
                * GAS_DENSITY_KG_NM3.get(self.species, 1.3) / 3600.0)

    @property
    def fill_frac(self) -> float:
        return self.remaining_kg / max(1e-6, self.start_kg)

    @property
    def at_heel(self) -> bool:
        return self.fill_frac <= DEWAR_HEEL_FRAC

    def wetted_derate(self) -> float:
        """Boiling rate falls with the liquid level, and falls fast at
        the end because the wetted area goes with it."""
        f = self.fill_frac
        if f >= 0.3:
            return 1.0
        return max(0.0, (f / 0.3) ** 0.6)

    def step(self, coil, pit, dt_s: float,
             plant: "BottlingPlant | None" = None) -> dict:
        """One interval: boil, catch, charge the bank.

        The rate is the SMALLER of what the coil could boil and what the
        valve is letting through, which is the whole point of the valve."""
        if self.at_heel:
            return {"stopped": True, "reason": "at heel", "boiled_kg": 0.0,
                    "fill_frac": self.fill_frac}
        rate = min(coil.boil_rate_kg_s(pit), self.metered_rate_kg_s(plant))
        want = min(self.remaining_kg, rate * max(0.0, dt_s))
        r = coil.step(pit, dt_s, want)
        got = r["boiled_kg"]
        self.remaining_kg -= got
        self.cold_delivered_j += r["latent_j"] + r["sensible_j"]
        a = self.receiver.admit_kg(got)
        if a["vented_nm3"] > 0.0:
            self.vented_kg += a["vented_nm3"] * GAS_DENSITY_KG_NM3.get(self.species, 1.3)
        self.hours += dt_s / 3600.0
        return {"stopped": False, "boiled_kg": got, "fill_frac": self.fill_frac,
                "receiver_full": a["full_frac"], "backpressure": a["backpressure"],
                "cold_mj": self.cold_delivered_j / 1e6}

    def bottle(self, plant: "BottlingPlant", kind: str = "industrial-50l") -> dict:
        """Empty the receiver into cylinders."""
        avail = self.receiver.contents_nm3
        f = plant.fill(self.species, avail, kind)
        if not f["refused"]:
            self.receiver.draw_nm3(avail - f["leftover_nm3"])
            self.bottled += f["filled"]
        return f

    def report(self, pit=None) -> dict:
        ice = self.cold_delivered_j / 334_000.0
        return {"hours": self.hours, "boiled_kg": self.start_kg - self.remaining_kg,
                "heel_kg": self.remaining_kg, "fill_frac": self.fill_frac,
                "cold_mj": self.cold_delivered_j / 1e6,
                "ice_made_kg": ice, "bottled": self.bottled,
                "vented_kg": self.vented_kg,
                "cold_per_kg_mj": (self.cold_delivered_j / 1e6
                                   / max(1e-6, self.start_kg - self.remaining_kg))}
