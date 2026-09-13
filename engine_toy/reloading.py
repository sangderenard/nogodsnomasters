"""The facility's own reloading shop: casings, primers, propellant pucks.

THE POINT OF RELOADING IS NOT THRIFT, IT IS INDEPENDENCE. A gun that
only fires factory rounds is a gun whose rate of fire is set by a supply
convoy. A gun whose outpost can make its own ammunition is limited by
what it can carry in BULK -- and the bulk items are far better than the
finished round in every way that matters:

  propellant and primers are compact, stable, and stack densely
  casings come back after every shot and are reusable many times
  projectiles can be turned from bar stock, or cast from slag

so an outpost stores a lot of shots per cubic metre and per kilo of
convoy if it stores them as components.

PUCKS, NOT LOOSE POWDER. A pressed propellant puck is a disc of grain
of a known mass and a known web, and you build a charge by STACKING
them. That is the single most important idea here:

  the charge is quantised, so it is repeatable without weighing
  anything in a dusty tent;
  the same puck serves every calibre the shop loads, so there is one
  stock item rather than a shelf of tins;
  and the charge becomes a decision made AT THE GUN. A shot at a close,
  soft target does not need the pressure that a long shot at armour
  does -- fewer pucks, less barrel wear, less recoil, less noise, less
  flash. Loading for the shot is what a field reloading shop is FOR.

THE WEB IS SIZED TO THE GUN, and this is what an earlier run of the
ballistics model got wrong. Burn rate is r = beta p, so the time to
burn through a grain's web is web / (beta p). Make the web too thick
and the charge is still burning when the shot leaves -- the first
validation run burnt 37% of the charge and threw the rest out of the
muzzle as flash, which is exactly what a badly matched propellant does
in reality. Make it too thin and it all burns at once against a
stationary shot, and the pressure spike wrecks the gun.

The right answer is neither, and it is not a number to be tuned: the
web is chosen so the charge burns out at roughly two thirds of the
shot's travel. Long barrel, thicker web. That is a real design rule
and `size_charge` below applies it with the real ballistics model
rather than a rule of thumb.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from interior_ballistics import PROPELLANTS, GRAINS, native_step


# =====================================================================
#  THE COMPONENTS
# =====================================================================
@dataclass(frozen=True)
class Casing:
    """A case, sized for one calibre. It comes back after the shot."""
    key: str
    calibre_mm: float
    case_volume_l: float          # the chamber it makes when seated
    case_mass_kg: float
    material: str = "brass"
    reloads: int = 12             # before the neck work-hardens and splits
    note: str = ""

    @property
    def chamber_volume_m3(self) -> float:
        return self.case_volume_l / 1000.0


@dataclass(frozen=True)
class Primer:
    """A primer, by the gas it actually makes.

    This is not a detail: the burn law is proportional to pressure and
    the pressure comes from gas, so a charge with no igniter never
    lights at all. The model returned a muzzle velocity of exactly zero
    until the primer was a real term in it."""
    key: str
    label: str
    gas_kg: float
    fits_case_head_mm: float
    note: str = ""


@dataclass(frozen=True)
class PropellantPuck:
    """A pressed disc of propellant: one stock item, stacked to taste."""
    key: str
    propellant: str               # a key in interior_ballistics.PROPELLANTS
    grain: str                    # a key in interior_ballistics.GRAINS
    mass_kg: float
    diameter_mm: float
    thickness_mm: float
    note: str = ""

    @property
    def volume_l(self) -> float:
        return (math.pi * (self.diameter_mm / 2000.0) ** 2
                * self.thickness_mm / 1000.0 * 1000.0)


CASINGS = {
    # 48 cc is RIGHT. The impossible load was 55 g of propellant in it
    # (1.15 g/cc, whose gas needs 55 cc of covolume in a 48 cc
    # chamber); the real charge is about 40 g, which is 0.83 g/cc and
    # leaves 8 cc free at burnout. Widening the case to 70 cc "fixed"
    # the explosion by making the gun unable to fire at all -- the
    # pressure then builds in 45 cc of ullage instead of 23 and never
    # bootstraps. A small chamber is not a problem to be designed away;
    # it is what lets a gun come up to pressure.
    "20x139": Casing("20x139", 20.0, 0.048, 0.118, "brass", 12,
                     note="the autocannon case. Light enough that a "
                          "hundred come back in one bin"),
    "40x365": Casing("40x365", 40.0, 0.330, 0.760, "brass", 10,
                     note="Bofors-class"),
    "76x636": Casing("76x636", 76.0, 2.400, 3.900, "steel", 8,
                     note="steel rather than brass: cheaper to make on "
                          "site and it survives more reloads than brass "
                          "does at this size"),
    "120x570": Casing("120x570", 120.0, 9.800, 7.300, "combustible", 1,
                      note="A COMBUSTIBLE CASE, which is the odd one out: "
                           "it burns with the charge and only the stub "
                           "base comes back. Nothing to eject, nothing to "
                           "store -- and nothing to reload either, so the "
                           "shop must be able to make the case itself"),
}

PRIMERS = {
    "small-percussion": Primer("small-percussion", "small percussion cap",
                               0.00035, 6.5,
                               note="autocannon and small arms"),
    "large-percussion": Primer("large-percussion", "large percussion primer",
                               0.0060, 15.0,
                               note="medium calibre"),
    "electric": Primer("electric", "electric primer", 0.0300, 22.0,
                       note="fired by a current rather than a striker, so "
                            "the gun has no firing pin to break and the "
                            "fire-control computer decides the instant of "
                            "ignition to the microsecond -- which is what "
                            "a fine-aim stage exists to exploit"),
}

PUCKS = {
    "p10-fast": PropellantPuck("p10-fast", "double-base", "ball",
                               0.010, 18.0, 4.0,
                               note="fast, degressive: short barrels and "
                                    "low charges"),
    "p10-medium": PropellantPuck("p10-medium", "double-base", "tubular",
                                 0.010, 18.0, 4.0,
                                 note="the standard puck. Pressed to a "
                                      "0.20-0.30 mm web for autocannon: "
                                      "the shot is only in the tube for "
                                      "about 3 ms, so a thicker web is "
                                      "still burning when it leaves"),
    "p50-slow": PropellantPuck("p50-slow", "triple-base", "seven-perf",
                               0.050, 32.0, 8.0,
                               note="progressive and cool-burning: long "
                                    "barrels, and the one that does not "
                                    "eat the liner"),
    "p250-slow": PropellantPuck("p250-slow", "triple-base", "nineteen-perf",
                                0.250, 58.0, 14.0,
                                note="large calibre"),
}


@dataclass(frozen=True)
class BreechShim:
    """A chamber insert: one breech, any calibre in the family.

    THE BREECH DOES NOT HAVE TO CHANGE. It is the heaviest, most
    expensive, most carefully made part of the gun -- a 297 kg block
    bored and lapped for the largest round the mount will ever fire --
    and swapping it to shoot something smaller is absurd when the thing
    that actually has to change is the SHAPE OF THE HOLE.

    So the calibre-change kit is three pieces and none of them is the
    breech: an inner barrel, a chamber shim, and the right case.

    AND THE SHIM IS NOT OPTIONAL PACKAGING. It sets the chamber volume,
    which is the single most consequential number in interior
    ballistics. Fire a 55 g charge into the 120 mm block's own 9.8 litre
    chamber and the expansion ratio is two hundred to one before the
    shot has moved at all -- the pressure never builds, the propellant
    never gets hot enough to burn properly, and what comes out of the
    muzzle is an unburnt cloud and a slow projectile. The shim takes
    that 9.8 litres down to the 48 cc the round was designed around.

    It also has to do the unglamorous jobs the breech was doing: hold
    headspace, carry the extractor, and take the firing load into the
    breech face over an area big enough not to brinell it.
    """
    key: str
    calibre_mm: float
    casing: str                   # which case it chambers
    chamber_volume_l: float       # what the round actually sees
    mass_kg: float
    material: str = "gun-steel"
    #: the breech it drops into -- the shim's outside, not its inside
    fits_breech_bore_mm: float = 120.0
    note: str = ""

    @property
    def volume_displaced_l(self) -> float:
        """How much of the big chamber this fills in. The shim is mostly
        solid steel, and that is where its mass comes from."""
        return max(0.0, 9.8 - self.chamber_volume_l)


SHIMS = {
    "shim-20": BreechShim(
        "shim-20", 20.0, "20x139", 0.048, 41.0,
        note="the extreme case: it fills 9.75 of the 9.8 litres. Nearly "
             "a solid plug with a small hole and an extractor in it"),
    "shim-40": BreechShim(
        "shim-40", 40.0, "40x365", 0.330, 36.0,
        note="Bofors-class"),
    "shim-76": BreechShim(
        "shim-76", 76.0, "76x636", 2.400, 24.0,
        note="thinner walls, less steel, but the firing load per unit of "
             "shim face is much higher -- this is the one that needs the "
             "bearing area checked rather than assumed"),
    None: None,
}
del SHIMS[None]


@dataclass(frozen=True)
class FillerSleeve:
    """What packs out the annulus so the jacket is a jacket.

    THE BATH DOES NOT COOL. A 20 mm liner inside a 120 mm outer barrel
    leaves 45 mm of water all the way round -- and computed properly
    that gives 26 W/K, against the 919 W/K needed to hold the tube at
    40 C under sustained fire. The coolant crawls at 0.03 m/s and the
    film coefficient goes as Reynolds to the 0.8, so a big annulus is
    the worst possible shape: maximum water, minimum velocity.

    Pack it out to a narrow gap and the same pump gives 1.66 m/s and
    3451 W/K. Same flow, same fluid, same length of tube -- a hundred
    and thirty times the heat transfer, purely from making the channel
    small. This is why real water-cooled barrels have tight jackets
    rather than baths, and it is worth a part of its own.

    It earns its place three more times over: it carries far less water,
    so the tube is lighter and droops less and rings closer to its dry
    frequency; it displaces the mass that was making the barrel a
    pendulum; and it gives the washers something to seal against.
    """
    key: str
    for_calibre_mm: float
    #: the flow gap it leaves against the outer barrel's bore
    gap_m: float
    mass_kg: float
    material: str = "aluminium"
    note: str = ""

    def annulus_volume_l(self, outer_bore_m: float, length_m: float) -> float:
        r_o = outer_bore_m / 2.0
        r_i = r_o - self.gap_m
        return math.pi * (r_o ** 2 - r_i ** 2) * length_m * 1000.0


SLEEVES = {
    "sleeve-20": FillerSleeve("sleeve-20", 20.0, 0.005, 11.4,
                              note="the extreme case: it fills 40 of the "
                                   "45 mm annulus and leaves a 5 mm "
                                   "channel. Nearly all of the jacket's "
                                   "volume becomes metal"),
    "sleeve-40": FillerSleeve("sleeve-40", 40.0, 0.005, 8.9),
    "sleeve-76": FillerSleeve("sleeve-76", 76.0, 0.005, 3.1,
                              note="barely a liner: the 76 mm barrel "
                                   "nearly fills the bore already"),
}


@dataclass(frozen=True)
class CalibreKit:
    """Everything that changes to shoot something different.

    Three items and a case. The breech, the cradle, the recoil gear,
    the mount and the fire control do not appear on this list, which is
    the entire point of building it this way."""
    key: str
    inner_barrel_bore_mm: float
    inner_barrel_length_m: float
    shim: str
    casing: str
    primer: str
    puck: str
    #: what packs the jacket out to a working gap
    sleeve: str = ""
    note: str = ""

    def mass_kg(self) -> float:
        return SHIMS[self.shim].mass_kg


KITS = {
    "20mm": CalibreKit("20mm", 20.0, 1.70, "shim-20", "20x139",
                       "small-percussion", "p10-medium", "sleeve-20",
                       note="the long thin one: high rate, flat, and the "
                            "liner is cheap to replace when it wears"),
    "40mm": CalibreKit("40mm", 40.0, 2.80, "shim-40", "40x365",
                       "large-percussion", "p50-slow", "sleeve-40"),
    "76mm": CalibreKit("76mm", 76.0, 3.80, "shim-76", "76x636",
                       "large-percussion", "p50-slow", "sleeve-76"),
    "120mm": CalibreKit("120mm", 120.0, 5.28, "", "120x570",
                        "electric", "p250-slow", "",
                        note="NO SHIM AND NO INNER BARREL: at maximum "
                             "ordnance the breech's own chamber and the "
                             "outer barrel's own bore are the gun"),
}


#: Practical maximum loading density, kg of propellant per litre of
#: chamber. Above this the charge cannot be got into the case, and
#: above the covolume limit it cannot burn there at all whatever you
#: do -- the gas needs more room than the chamber has.
#: 0.72 was too cautious -- real autocannon and tank rounds run
#: 0.80-0.95. The binding limit is not this at all, it is the COVOLUME
#: check below: the gas's own volume must fit in the chamber, and that
#: is a hard physical wall with nothing on the other side of it.
MAX_LOADING_DENSITY_KG_PER_L = 0.95


def check_load(casing: str, puck: str, pucks: int, shim: str = None) -> dict:
    """Is this stack physically loadable, and can it burn?

    Two separate limits, and conflating them is how a nonsense load
    gets as far as the ballistics model:

      PACKING  the pucks have to fit in the case at all.
      COVOLUME the gas they become has to fit in the chamber. This is
               the harder limit and it is the one with no way round it:
               you cannot compress your way out of the volume the gas
               molecules themselves occupy.
    """
    from interior_ballistics import PROPELLANTS
    c = CASINGS[casing]
    pk = PUCKS[puck]
    prop = PROPELLANTS[pk.propellant]
    chamber_l = (SHIMS[shim].chamber_volume_l if shim else c.case_volume_l)
    charge = pk.mass_kg * pucks
    density = charge / max(chamber_l, 1e-9)
    solid_l = charge / prop.density_kg_m3 * 1000.0
    covolume_l = charge * prop.covolume_m3_per_kg * 1000.0
    puck_stack_l = pk.volume_l * pucks
    return {
        "charge_kg": charge,
        "chamber_l": chamber_l,
        "loading_density_kg_per_l": density,
        "puck_stack_l": puck_stack_l,
        "solid_l": solid_l,
        "covolume_at_burnout_l": covolume_l,
        "free_at_burnout_l": chamber_l - covolume_l,
        "fits_in_case": puck_stack_l <= chamber_l * 0.92,
        "can_burn": covolume_l < chamber_l * 0.92,
        "within_density": density <= MAX_LOADING_DENSITY_KG_PER_L,
        "ok": (puck_stack_l <= chamber_l * 0.92
               and covolume_l < chamber_l * 0.92
               and density <= MAX_LOADING_DENSITY_KG_PER_L),
    }


@dataclass
class Load:
    """One assembled round: case, primer, a stack of pucks, a shot."""
    casing: str
    primer: str
    puck: str
    pucks: int
    shot_mass_kg: float
    web_m: float                  # what the shop pressed the grain to
    label: str = ""

    @property
    def charge_kg(self) -> float:
        return PUCKS[self.puck].mass_kg * self.pucks

    def describe(self) -> list[str]:
        c = CASINGS[self.casing]; p = PUCKS[self.puck]
        return [
            f"{self.label or self.casing}: {c.calibre_mm:.0f} mm",
            f"  case {c.key} ({c.material}, {c.reloads} reloads), "
            f"primer {self.primer}",
            f"  {self.pucks} x {p.key} = {self.charge_kg * 1000:.0f} g of "
            f"{PROPELLANTS[p.propellant].label}",
            f"  grain {p.grain}, web pressed to {self.web_m * 1000:.2f} mm",
            f"  shot {self.shot_mass_kg:.3f} kg",
        ]


# =====================================================================
#  FIRING ONE, WITH THE REAL MODEL
# =====================================================================
def fire(load: Load, barrel_m: float, *, dt: float = 1.0e-6,
         shot_start_pa: float = 3.5e7, bore_resistance_pa: float = 1.5e7,
         recoiling_mass_kg: float = 1.0e9,
         gas_port_travel_m: float = 1.0e9,
         gas_port_fraction: float = 0.0,
         limit: int = 200000) -> dict:
    """Run the baked interior-ballistics kernel over one round."""
    step, _ = native_step()
    c = CASINGS[load.casing]
    pk = PUCKS[load.puck]
    prop = PROPELLANTS[pk.propellant]
    grain = GRAINS[pk.grain]
    area = math.pi * (c.calibre_mm / 2000.0) ** 2
    st = {"travel_m": 0.0, "velocity_m_s": 0.0, "burnt_fraction": 0.0,
          "recoil_velocity_m_s": 0.0, "harvested_gas_kg": 0.0}
    peak = 0.0
    burnout_m = None
    n = 0
    while n < limit:
        r = step(dt_s=dt, bore_area_m2=area,
                 chamber_volume_m3=c.chamber_volume_m3,
                 shot_mass_kg=load.shot_mass_kg,
                 charge_mass_kg=load.charge_kg,
                 impetus_j_per_kg=prop.impetus_j_per_kg,
                 covolume_m3_per_kg=prop.covolume_m3_per_kg,
                 propellant_density_kg_m3=prop.density_kg_m3,
                 gamma=prop.gamma, burn_rate_coeff=prop.burn_rate_coeff,
                 grain_chi=grain.chi, grain_lambda=grain.lam,
                 grain_web_m=load.web_m,
                 shot_start_pressure_pa=shot_start_pa,
                 bore_resistance_pa=bore_resistance_pa,
                 barrel_length_m=barrel_m,
                 primer_gas_kg=PRIMERS[load.primer].gas_kg,
                 recoiling_mass_kg=recoiling_mass_kg,
                 gas_port_travel_m=gas_port_travel_m,
                 gas_port_fraction=gas_port_fraction, **st)
        peak = max(peak, r["breech_pressure_pa"])
        st = {"travel_m": r["travel_next_m"],
              "velocity_m_s": r["velocity_next_m_s"],
              "burnt_fraction": r["burnt_fraction_next"],
              "recoil_velocity_m_s": r["recoil_velocity_next_m_s"],
              "harvested_gas_kg": r["harvested_gas_next_kg"]}
        if burnout_m is None and r["gas_made_kg"] >= load.charge_kg * 0.99:
            burnout_m = st["travel_m"]
        n += 1
        if r["at_muzzle"] > 0.5:
            break
    return {
        "muzzle_m_s": st["velocity_m_s"],
        "peak_breech_pa": peak,
        "time_s": n * dt,
        "burnt_fraction": min(1.0, st["burnt_fraction"]),
        "burnout_travel_m": burnout_m,
        "burnout_fraction_of_barrel": (burnout_m / barrel_m
                                       if burnout_m else None),
        "muzzle_energy_j": 0.5 * load.shot_mass_kg * st["velocity_m_s"] ** 2,
        "recoil_velocity_m_s": st["recoil_velocity_m_s"],
        "harvested_gas_kg": st["harvested_gas_kg"],
    }


def size_charge(casing: str, primer: str, puck: str, shot_mass_kg: float,
                barrel_m: float, *, peak_limit_pa: float = 4.5e8,
                burnout_at: float = 0.66, max_pucks: int = 400,
                webs=None) -> tuple[Load, dict]:
    """What the shop should press and stack for this gun and this shot.

    THIS IS DESIGN, NOT FITTING. Nothing here adjusts a physical
    constant to make an answer come out: the propellant's impetus,
    covolume, density and burn-rate coefficient are what they are. What
    is being chosen is what a reloading shop actually chooses -- HOW
    MANY PUCKS and WHAT WEB TO PRESS THEM TO -- against two real
    constraints: the chamber must not exceed its pressure limit, and
    the charge should be burnt out by about two thirds of the travel so
    the rest of the barrel is expansion rather than flash.
    """
    if webs is None:
        webs = [0.05e-3 * i for i in range(1, 61)]
    best = None
    for n_pucks in range(1, max_pucks + 1):
        # REFUSE, do not simulate. A load that cannot burn in this
        # chamber will still produce a number if you hand it to the
        # integrator, and that number will be enormous and wrong.
        gate = check_load(casing, puck, n_pucks)
        if not gate["ok"]:
            break
        for web in webs:
            load = Load(casing, primer, puck, n_pucks, shot_mass_kg, web)
            r = fire(load, barrel_m)
            if r["peak_breech_pa"] > peak_limit_pa:
                continue
            frac = r["burnout_fraction_of_barrel"]
            if frac is None:
                continue
            score = (r["muzzle_m_s"], -abs(frac - burnout_at))
            if best is None or score > best[2]:
                best = (load, r, score)
    if best is None:
        raise ValueError(
            f"no stack of {puck} in a {casing} case fires a "
            f"{shot_mass_kg:.3f} kg shot down {barrel_m:.2f} m under "
            f"{peak_limit_pa / 1e6:.0f} MPa -- the shop needs a different "
            f"puck, not a different number of them")
    return best[0], best[1]
