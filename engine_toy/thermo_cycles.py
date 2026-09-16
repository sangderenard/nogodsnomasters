"""One law, several cycles: what each kind of engine can ideally do, and
how much of it real hardware gets.

derived_torque had a single fitted constant -- OTTO_REALISATION -- doing
two jobs it could not do at once. It was standing in for the gap between
the ideal cycle and a real engine, AND for every difference between a
1900s hit-and-miss and a modern turbo diesel. One number cannot express
both, which is why small old engines came out far too strong and
forced-induction ones too weak.

This splits it into the two things it actually is.

FIRST, THE IDEAL, WHICH IS DIFFERENT FOR EACH CYCLE. Every engine here
converts heat to work, but they do not all do it the same way and they
do not all have a compression ratio:

  OTTO          spark ignition, heat added at constant volume.
                eta = 1 - r^(1-gamma). Compression ratio is everything,
                and knock is what limits it.

  DIESEL        compression ignition, heat added at roughly constant
                PRESSURE while the piston is already moving down. That
                cutoff costs efficiency at the same compression ratio --
                a diesel is a worse cycle than Otto, point for point.
                It wins anyway because it can run compression ratios
                Otto never could, having no fuel in the cylinder to
                knock.

  BRAYTON       a turbine. There is no compression RATIO because there
                is no piston; there is a compressor PRESSURE ratio, and
                the ideal follows it the same way. A regenerator -- the
                AGT1500 has one -- recovers exhaust heat into the
                compressed air before the burner and lifts efficiency
                substantially at modest pressure ratios, which is the
                whole reason a tank engine has one.

  ATMOSPHERIC   the Otto-Langen. Burns at atmospheric pressure, throws a
                free piston up, and takes its work on the way DOWN from
                atmospheric pressure and gravity. There is no
                compression stroke at all, so a compression ratio is not
                a small number for this engine -- it is not a property
                it has. About eleven per cent, which sounds dreadful and
                was roughly triple the steam engines it competed with.

  RANKINE       steam. Efficiency comes from boiler and condenser
                conditions, not from anything in the cylinder, which is
                why a traction engine's cylinder is almost incidental to
                how efficient it is.

SECOND, THE REALISATION, AND MOST OF IT IS DERIVED RATHER THAN CHOSEN.
Real engines fall short of their ideal cycle for reasons that are not
all the same size, and the largest one is geometric:

  HEAT LOSS SCALES WITH AREA AND WORK SCALES WITH VOLUME. A cylinder's
  surface-to-volume ratio goes as 1/bore, so a 25 cc trimmer loses an
  enormous share of its heat into the walls and a one-metre marine
  cylinder loses almost none. That is not a fudge, it is the reason
  large slow diesels reach fifty per cent thermal efficiency and small
  two-strokes reach fifteen, and it is the single biggest term missing
  when every engine is given the same realisation factor.

  Then era: chamber shape, mixture preparation and how precisely the
  burn can be timed. A flathead with a carburettor and a fixed advance
  genuinely cannot burn its charge as completely or as well-timed as a
  pentroof with port injection and knock feedback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Bore at which the surface-to-volume heat loss term is normalised to
#: 1.0. Chosen as a typical automotive bore so an ordinary car engine
#: sits at its era factor and nothing else -- the scaling then reads
#: directly as "better or worse than a car engine because of size".
REFERENCE_BORE_M = 0.090


def scale_realisation(bore_m: float) -> float:
    """How much of its heat an engine of this size keeps.

    Surface-to-volume goes as 1/bore, so heat loss per unit of work does
    too. Expressed as a factor against a reference automotive bore. A
    small engine is genuinely, unavoidably worse at this and no amount
    of development fixes it -- which is why model aircraft engines are
    inefficient and ship engines are the most efficient heat engines
    ever built.

    The exponent is well below 1 because wall heat loss is only part of
    the total loss; blowdown, friction and incomplete combustion do not
    scale the same way."""
    b = max(0.005, float(bore_m))
    return (b / REFERENCE_BORE_M) ** 0.18


@dataclass(frozen=True)
class Era:
    key: str
    label: str
    #: Share of the ideal cycle the technology of the period reaches,
    #: at the reference bore.
    realisation: float
    why: str = ""


ERAS: dict[str, Era] = {
    "hit-and-miss": Era(
        "hit-and-miss", "1900s hit-and-miss / early stationary", 0.42,
        why="an open chamber, a mixture made by a wick or a crude jet, fixed ignition "
            "timing and a governor that simply skips firing strokes. It is not trying "
            "to be efficient, it is trying to be reliable and repairable in a field"),
    "brass-era": Era(
        "brass-era", "pre-1915 automotive", 0.46,
        why="atmospheric inlet valves in some cases, poor chamber shape, fuel of "
            "unpredictable volatility and no way to know what the engine was doing"),
    "flathead": Era(
        "flathead", "sidevalve / flathead", 0.55,
        why="the charge has to turn a corner into a chamber sitting beside the bore "
            "rather than above it, which makes a long slow burn path and a lot of "
            "exposed surface. Cheap to build and fundamentally limited"),
    "pushrod-carb": Era(
        "pushrod-carb", "overhead valve, carburetted", 0.66,
        why="the chamber is above the piston now and the burn path is short. Mixture "
            "is still made by a venturi and distributed by a manifold that cannot "
            "feed every cylinder the same"),
    "efi-2v": Era(
        "efi-2v", "port injection, two valve", 0.72,
        why="each cylinder gets its own metered fuel, and closed-loop mixture control "
            "means the engine can actually hold the ratio it wants"),
    "modern-4v": Era(
        "modern-4v", "four valve, port or direct injection", 0.78,
        why="a pentroof chamber with a central plug burns fastest of these, and knock "
            "feedback lets it run the timing it wants rather than the timing that is "
            "safe on the worst fuel it might see"),
    "aero-supercharged": Era(
        "aero-supercharged", "supercharged aero piston", 0.70,
        why="developed to an extraordinary degree on unlimited budgets, and then asked "
            "to run enormous boost on 100-octane at altitude. Efficient for its era and "
            "spending most of it driving its own supercharger"),
    "industrial-diesel": Era(
        "industrial-diesel", "modern industrial diesel", 0.80,
        why="direct injection at very high pressure, turbocharged, and designed around "
            "a duty cycle that sits at one speed for thousands of hours"),
    "large-slow-diesel": Era(
        "large-slow-diesel", "slow-speed marine diesel", 0.88,
        why="the most efficient heat engines ever built. Enormous bores, very long "
            "strokes, uniflow scavenging, turbocharged, and turning a hundred rpm -- "
            "every loss that scales with speed or with surface area is minimised"),
    "two-stroke-small": Era(
        "two-stroke-small", "small crankcase-scavenged two-stroke", 0.34,
        why="loses raw mixture straight out of the exhaust port during scavenging, "
            "which is a fuel loss before it is an efficiency loss, and runs total-loss "
            "lubrication on top"),
    "turbine": Era(
        "turbine", "gas turbine", 0.72,
        why="component efficiencies are high but the cycle is unforgiving: the "
            "compressor eats most of the turbine's work, so small losses in either "
            "swing the net enormously"),
    "steam": Era(
        "steam", "steam reciprocating", 0.55,
        why="the cylinder is the smallest part of the problem. Boiler and condenser "
            "conditions decide almost everything"),
}


def era(key: str) -> Era:
    e = ERAS.get(str(key))
    if e is None:
        raise KeyError(f"unknown era {key!r}; declared: {', '.join(sorted(ERAS))}")
    return e


# ---------------------------------------------------------------------
# the ideal cycles
# ---------------------------------------------------------------------

def otto_efficiency(compression_ratio: float, gamma: float = 1.35) -> float:
    r = max(1.5, float(compression_ratio))
    return 1.0 - r ** (1.0 - float(gamma))


def diesel_efficiency(compression_ratio: float, cutoff_ratio: float = 2.0,
                      gamma: float = 1.35) -> float:
    """Heat added at constant pressure while the piston already moves.

    Worse than Otto at the SAME compression ratio -- the cutoff term is
    always greater than one -- and better in practice because it can use
    compression ratios Otto cannot. The cutoff ratio rises with load,
    which is why a diesel is most efficient part-loaded and an Otto
    engine is most efficient near full throttle."""
    r = max(1.5, float(compression_ratio))
    rc = max(1.001, float(cutoff_ratio))
    g = float(gamma)
    return 1.0 - (1.0 / r ** (g - 1.0)) * ((rc ** g - 1.0) / (g * (rc - 1.0)))


def brayton_efficiency(pressure_ratio: float, gamma: float = 1.4,
                       regenerator_effectiveness: float = 0.0,
                       turbine_inlet_k: float = 1200.0,
                       compressor_inlet_k: float = 288.0) -> float:
    """A turbine. Pressure ratio, not compression ratio.

    Without a regenerator this is the same shape as Otto with pressure
    ratio in place of volume ratio. WITH one, exhaust heat is fed back
    into the compressed air before the burner, and the result inverts
    the usual advice: a regenerated turbine gets WORSE as pressure ratio
    rises past a point, because a higher compressor outlet temperature
    leaves less room for the exhaust to heat it further. That is why
    regenerated engines run modest pressure ratios and why the AGT1500
    is not simply a scaled-down jet."""
    rp = max(1.01, float(pressure_ratio))
    g = float(gamma)
    simple = 1.0 - rp ** ((1.0 - g) / g)
    eff = max(0.0, min(0.95, float(regenerator_effectiveness)))
    if eff <= 0.0:
        return simple
    t = max(1.1, float(turbine_inlet_k) / max(1.0, float(compressor_inlet_k)))
    x = rp ** ((g - 1.0) / g)
    ideal_regen = 1.0 - (x / t)
    return simple + (ideal_regen - simple) * eff


def atmospheric_efficiency(expansion_ratio: float = 3.0, gamma: float = 1.35) -> float:
    """Otto-Langen: no compression, work taken on the descent.

    The charge burns at atmospheric pressure and drives a free piston
    up; useful work comes from atmospheric pressure pushing it back down
    against the load. There is nothing to compress, so the only ratio
    available is how far the gas expands before the cylinder is opened."""
    r = max(1.05, float(expansion_ratio))
    return (1.0 - r ** (1.0 - float(gamma))) * 0.55


def rankine_efficiency(boiler_pressure_pa: float = 8e5,
                       condenser_pressure_pa: float = 101_325.0) -> float:
    """Steam, from the pressures it works between.

    A traction engine exhausting to atmosphere has no condenser at all,
    which caps it brutally low -- most of the energy leaves up the
    chimney as the latent heat of the steam, and no cylinder design
    recovers it."""
    pb = max(1.05 * condenser_pressure_pa, float(boiler_pressure_pa))
    pc = max(1000.0, float(condenser_pressure_pa))
    # Carnot between the saturation temperatures the two pressures imply
    t_b = 373.15 * (pb / 101_325.0) ** 0.25
    t_c = 373.15 * (pc / 101_325.0) ** 0.25
    return max(0.02, (1.0 - t_c / t_b) * 0.62)


# ---------------------------------------------------------------------
# putting the two halves together
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CycleResult:
    cycle: str
    ideal: float
    era_factor: float
    scale_factor: float
    realised: float
    why: str = ""


def efficiency(cycle: str, *, era_key: str, bore_m: float = REFERENCE_BORE_M,
               compression_ratio: float = 9.0, cutoff_ratio: float = 2.0,
               pressure_ratio: float = 14.0, regenerator: float = 0.0,
               expansion_ratio: float = 3.0, gamma: float = 1.35,
               boiler_pressure_pa: float = 8e5,
               condenser_pressure_pa: float = 101_325.0) -> CycleResult:
    """Ideal cycle efficiency, times what this era and this size achieve.

    The two factors are kept separate in the result because they answer
    different questions: the ideal says what the cycle allows, the era
    says what the technology of the period got out of it, and the scale
    says what its size costs it no matter what anyone does."""
    c = str(cycle)
    if c == "otto":
        ideal = otto_efficiency(compression_ratio, gamma)
    elif c == "diesel":
        ideal = diesel_efficiency(compression_ratio, cutoff_ratio, gamma)
    elif c == "brayton":
        ideal = brayton_efficiency(pressure_ratio, 1.4, regenerator)
    elif c == "atmospheric":
        ideal = atmospheric_efficiency(expansion_ratio, gamma)
    elif c == "rankine":
        ideal = rankine_efficiency(boiler_pressure_pa, condenser_pressure_pa)
    else:
        raise KeyError(f"unknown cycle {cycle!r}; declared: otto, diesel, brayton, "
                       "atmospheric, rankine")
    e = era(era_key)
    scale = scale_realisation(bore_m)
    return CycleResult(cycle=c, ideal=ideal, era_factor=e.realisation,
                       scale_factor=scale,
                       realised=max(0.01, min(0.72, ideal * e.realisation * scale)),
                       why=e.why)


def cycle_of(engine) -> str:
    """Which cycle this engine runs, from what it actually is.

    Derived rather than declared wherever the engine already says enough
    to know: a turbine is Brayton, an atmospheric engine is atmospheric,
    an expander on steam is Rankine, and a piston engine is Otto or
    Diesel according to whether it compresses hard enough to ignite
    without a spark."""
    kind = str(getattr(engine, "kind", "") or "")
    arch = getattr(engine, "architecture", None)
    if kind in ("turbine", "turboshaft"):
        return "brayton"
    if getattr(engine, "atmospheric", None) is not None:
        return "atmospheric"
    if getattr(engine, "expander", None) is not None:
        return "rankine"
    cr = float(getattr(arch, "compression_ratio", 9.0) or 9.0)
    return "diesel" if cr >= 14.0 else "otto"
