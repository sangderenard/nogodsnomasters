"""Fouling: how a fluid passage closes up, and what closes it.

Every fluid component in this machine has a flow area, and every one of
them can lose it. An oil filter cakes with what it caught. A radiator
core packs with dirt from the outside and scales from the inside. A fuel
filter waxes in the cold and gums with varnish. A gallery sludges. A
turbine's oil feed cokes shut because the shaft behind it soaked. Those
are not one phenomenon with one rate -- they have different drivers,
different speeds, and different things that clear them -- but they are
all the same consequence: the hole is smaller than it was.

WHAT IS DECLARED HERE, AND WHY. A component says which fouling MODE it
is subject to, and the mode says what actually forms the blockage:

    mode          what forms it                      what clears it
    -------------------------------------------------------------------
    filter-cake   the debris the filter was fitted    replacing the
                  to catch, so it grows with the      element
                  flow that has passed through it
    core-debris   airborne dirt packing an exposed    cleaning the core
                  core from outside, so it grows
                  with the air that has gone through
    scale         dissolved solids dropping out of    flushing
                  hot coolant, so it grows with
                  time spent above a threshold
    coking        oil baked onto a hot wall, so it    replacing the line
                  grows with time spent above a
                  much higher threshold
    varnish       fuel oxidising in a warm system,    cleaning
                  so it grows with time
    wax           cold fuel dropping wax, which is    warming the fuel
                  the one that comes BACK
    sludge        cold-running oil emulsifying with   an oil change
                  combustion water, so it grows
                  when the oil never gets hot

THE ONE NUMBER THAT MATTERS is `blocked_frac`, 0 clear to 1 shut, and it
multiplies the component's flow area like any other restriction. It is
deliberately not a special case per component: a 40% blocked oil filter,
a 40% blocked radiator and a 40% blocked gallery are all just a hole
with 40% of it gone, and whatever solves flow through the clear version
solves the fouled one unchanged.

WHY THIS IS A MODULE AND NOT A FIELD ON ONE CLASS. It applies to every
fluid there is -- oil, coolant, fuel, air, hydraulic, refrigerant -- and
to nodes (filters, coolers, pumps) as readily as to lines and ports. A
part declares `fouling_mode` when it is built and carries `blocked_frac`
as state; nothing else has to know which kind of part it is.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FoulingMode:
    key: str
    label: str
    driver: str          # the parameter whose accumulation forms the blockage
    unit: str            # what `rate` is per
    rate: float          # blocked_frac gained per unit of driver, at reference duty
    reversible: bool     # whether the blockage can go away on its own
    clears_with: str     # the real service action that restores it
    why: str
    # Whether what has already deposited helps catch more. True for a
    # medium that filters (the cake is the filter); false for something
    # dropping out of solution onto a wall, which does not care what is
    # already there.
    self_accelerating: bool = False


_MODES: tuple[FoulingMode, ...] = (
    FoulingMode(
        "filter-cake", "filter cake", driver="throughput", unit="m^3 of fluid passed",
        rate=2.0e-3, reversible=False, clears_with="replace-element",
        self_accelerating=True,
        why="a filter blocks BECAUSE it is working: everything it catches stays on the "
            "element, so its restriction is a measure of how much it has already done"),
    FoulingMode(
        "core-debris", "packed core", driver="air-throughput", unit="m^3 of air passed",
        rate=1.2e-6, reversible=True, clears_with="clean-core",
        self_accelerating=True,
        why="an exposed core packs with what the air carries -- dust, chaff, insects -- "
            "on the outside of the fins, where a wash gets it back"),
    FoulingMode(
        "scale", "scale", driver="hot-hours", unit="hours above 95 C",
        rate=1.5e-3, reversible=False, clears_with="flush",
        why="dissolved solids come out of solution on the hottest metal they touch, "
            "which is why a head's jacket scales before a radiator does"),
    FoulingMode(
        "coking", "coke", driver="very-hot-hours", unit="hours above 230 C",
        rate=9.0e-3, reversible=False, clears_with="replace-line",
        why="oil left on a wall that hot bakes to hard carbon; the classic case is a "
            "turbo's feed after a hot shutdown, and it does not wash out"),
    FoulingMode(
        "varnish", "varnish", driver="hours", unit="running hours",
        rate=1.0e-4, reversible=False, clears_with="clean",
        why="fuel oxidises slowly in a warm system and leaves a hard film on everything "
            "it sits in, narrowing jets and seats"),
    FoulingMode(
        "wax", "wax", driver="cold-hours", unit="hours below the cloud point",
        rate=0.25, reversible=True, clears_with="warm-the-fuel",
        why="diesel drops wax crystals when it gets cold enough and they mat across the "
            "filter -- the one blockage that clears itself, because warming the fuel "
            "puts the wax back into solution"),
    FoulingMode(
        "sludge", "sludge", driver="cold-hours", unit="hours below 70 C oil temperature",
        rate=8.0e-4, reversible=False, clears_with="oil-change",
        why="an engine that never gets hot never boils the combustion water out of its "
            "oil, and the emulsion that makes is what blocks a pickup screen"),
)

# How much more a loaded medium catches than a clean one, at full load.
# THE SHAPE IS REAL AND THE NUMBER IS NOT MEASURED -- see `accumulate`.
LOADING_FEEDBACK = 2.0

# ---------------------------------------------------------------------
# INDUSTRIAL UNITS
# ---------------------------------------------------------------------
#
# Filtration and corrosion are both measured, and measured in units that
# are on the datasheet of every real component. Working in those units
# instead of inventing our own means a figure here can be checked
# against a part number, and a capacity can be typed in from a catalogue
# rather than tuned until it looks right.
#
#   DIRT-HOLDING CAPACITY (DHC), grams. What a filter holds before it
#   reaches its terminal pressure drop, measured by multi-pass test:
#   ISO 16889 for oil and hydraulic elements, ISO 5011 for air. This is
#   THE number a filter is sold on, and it is a mass, not a volume of
#   fluid -- which is exactly why one "capacity in cubic metres" could
#   never fit both an oil filter and an air filter.
#
#   BETA RATIO (beta_x), dimensionless. Particles above x micrometres
#   upstream divided by the same downstream, per ISO 16889. beta_200 at
#   10 um means 199 of every 200 such particles are caught, so capture
#   efficiency is 1 - 1/beta. Datasheets quote beta_2, beta_20, beta_75,
#   beta_200, beta_1000 -- those are 50%, 95%, 98.7%, 99.5%, 99.9%.
#
#   ISO 4406 CLEANLINESS CODE, three numbers like 18/16/13. Each is a
#   range code for the count of particles per millilitre above 4, 6 and
#   14 um. It is how the cleanliness of a real oil sample is reported,
#   and it maps to a contaminant concentration -- which is what actually
#   decides how fast a filter fills.
#
#   CORROSION RATE, millimetres per year (mm/yr), or mils per year in
#   imperial. The standard way every corrosion table states a rate, so
#   the corrosion modes below are quoted in mm/yr and converted here
#   rather than carried as metres per second of nobody's experience.

HOURS_PER_YEAR = 8766.0          # 365.25 days, the basis of a mm/yr figure


def capture_efficiency(beta_ratio: float) -> float:
    """Fraction caught, from the beta ratio on the datasheet."""
    b = max(1.0, float(beta_ratio))
    return 1.0 - 1.0 / b


def iso4406_mg_per_l(code: str) -> float:
    """Approximate contaminant concentration for an ISO 4406 code.

    The code's first number is a range code for particles >4 um per mL;
    each step doubles the count. Converting a count to a mass needs an
    assumed size and density distribution, so this is a real, disclosed
    correlation rather than an identity -- but it is anchored where the
    industry anchors it: 18/16/13 is ordinary used engine oil at a few
    milligrams per litre, and each whole code step doubles it."""
    first = str(code).split("/")[0].strip()
    try:
        c = float(first)
    except ValueError:
        raise ValueError(f"not an ISO 4406 code: {code!r}")
    # anchor: code 18 ~ 2.5 mg/L, doubling per code
    return 2.5 * (2.0 ** (c - 18.0))


def captured_once_through_g(flow_l: float, concentration_mg_per_l: float,
                            beta_ratio: float = 200.0) -> float:
    """Grams caught out of fluid the machine sees ONCE and does not see
    again: intake air, raw water, fuel on its way from tank to burner.

    Here the fluid really does arrive dirty and leave, so the honest
    arithmetic is litres past the element times how dirty they were
    times how much of it the element stops."""
    mg = max(0.0, float(flow_l)) * max(0.0, float(concentration_mg_per_l))
    return mg * capture_efficiency(beta_ratio) / 1000.0


def captured_recirculating_g(generation_g_per_h: float, hours: float,
                             beta_ratio: float = 200.0) -> float:
    """Grams caught out of a fluid that goes ROUND, not through.

    THIS IS THE ONE THAT IS EASY TO GET WRONG, and getting it wrong is
    not subtle. An engine's oil is the same five litres passing the
    filter twenty times a minute. Multiplying flow by concentration
    counts that one charge of oil over and over and says a 60 g filter
    catches three hundred grams in a hundred hours, which is five times
    its own capacity -- the arithmetic of a once-through system applied
    to a loop.

    What actually fills the filter is what the ENGINE MAKES: wear metal,
    soot blown past the rings, oxidation products. The oil's steady
    concentration is just the balance between that generation and the
    filter's removal, and over any real interval essentially everything
    generated ends up on the element. So the driver is a generation RATE
    in grams per hour, and the recirculating flow rate does not appear
    at all -- pumping the same dirt past the filter faster does not
    create more of it.

    Real order of magnitude: a sound engine makes a few tenths of a gram
    an hour of filterable material, a worn or sooty one several times
    that, which is why a 60 g element is a few hundred hours either way
    and why a tired engine eats filters."""
    caught = max(0.0, float(generation_g_per_h)) * max(0.0, float(hours))
    return caught * capture_efficiency(beta_ratio)


# Kept as the old name so nothing that already calls it breaks; it is
# the once-through form, which is what every existing caller meant.
captured_g = captured_once_through_g

BY_KEY: dict[str, FoulingMode] = {m.key: m for m in _MODES}


def mode(key: str) -> FoulingMode:
    m = BY_KEY.get(str(key))
    if m is None:
        raise KeyError(f"unknown fouling mode {key!r}; declare one of: {', '.join(sorted(BY_KEY))}")
    return m


def accumulate(blocked_frac: float, mode_key: str, driver_amount: float,
               severity: float = 1.0, capacity: float = 0.0) -> float:
    """Advance a component's blockage by what its driver did this tick.

    `driver_amount` is in the mode's own unit -- cubic metres through a
    filter, hours above a threshold for scale -- so the caller measures
    the real thing rather than passing a timestep and hoping. `severity`
    scales for duty that is worse than the reference (dusty air, dirty
    fuel, an engine on short cold trips).

    A reversible mode takes a NEGATIVE driver amount to come back: warm
    the fuel and the wax goes away, wash the core and the dirt does.

    HOW MUCH OF WHAT FLOWS THROUGH STAYS BEHIND -- and how sure we are.
    This is deposition per unit of throughput, and for a medium that
    catches things it is deliberately NOT linear: a partly loaded filter
    catches a larger share of what reaches it than a clean one does,
    because the cake is itself the filter. That is real and it is why
    loading curves bend upward rather than running straight to the end.
    `LOADING_FEEDBACK` is how strongly, and it is the honest weak point
    of this whole module: the SHAPE is right and the COEFFICIENT is not
    measured against anything. Nothing here has been calibrated against
    a real filter, a real core or a real gallery -- the rates are set so
    that service intervals land in the right decade, not so that any
    particular component is right. Treat a specific number out of this
    as an order of magnitude until something real is fitted to it.
    """
    m = mode(mode_key)
    b = max(0.0, min(1.0, float(blocked_frac)))
    # A COMPONENT'S OWN CAPACITY, not one rate for every part. An oil
    # filter and an air filter foul by the same mechanism and are not
    # remotely comparable in throughput: an engine puts a few cubic
    # metres of oil an hour through one and a couple of hundred cubic
    # metres of air through the other. One shared constant made the air
    # filter block in about two hours. So the MODE carries the shape and
    # the PART carries how much it can hold before it is shut, in that
    # mode's own units. A part that does not say falls back to the
    # mode's reference rate, which is the old behaviour.
    per_unit = (1.0 / float(capacity)) if float(capacity) > 0.0 else m.rate
    delta = per_unit * float(driver_amount) * max(0.0, float(severity))
    if m.self_accelerating and delta > 0.0:
        # the cake is the filter: capture rises with what is already on it
        delta *= 1.0 + LOADING_FEEDBACK * b
    if delta < 0.0 and not m.reversible:
        delta = 0.0
    return max(0.0, min(1.0, b + delta))


def open_area_frac(blocked_frac: float) -> float:
    """The share of the original flow area still open."""
    return max(0.0, 1.0 - max(0.0, min(1.0, float(blocked_frac))))


def restriction_factor(blocked_frac: float) -> float:
    """How much harder it is to push the same flow through.

    Pressure drop through an orifice goes as 1/area^2, so a hole that is
    half shut costs four times the drop, not twice. That squared law is
    the whole reason a filter that looks 'only half blocked' is already
    starving what is behind it."""
    a = open_area_frac(blocked_frac)
    if a <= 1e-6:
        return float("inf")
    return 1.0 / (a * a)


def clears_with(mode_key: str) -> str:
    return mode(mode_key).clears_with


# ---------------------------------------------------------------------
# WHAT A MECHANIC ACTUALLY POURS IN
# ---------------------------------------------------------------------
#
# Every one of these is a real product off a real shelf, and each does
# one of two quite different things: it REMOVES what has already formed,
# or it SLOWS what is still forming. Mixing those up is how people end
# up disappointed -- a stabiliser will not clean anything, and a flush
# does nothing at all about the next thousand hours.
#
# Several also carry a real hazard, and they are declared here because
# the hazard is the interesting part in a machine you can break: a
# strong detergent flush in a high-mileage engine lifts sludge that was
# doing no harm where it sat, and sends it to the pickup screen. That is
# not a folk tale, it is the standard warning, and this model can
# actually express it -- `debris_release` is fouling handed to the part
# DOWNSTREAM.


@dataclass(frozen=True)
class Conditioner:
    key: str
    label: str
    fluid: str                  # which working fluid it goes into
    removes: tuple              # fouling modes it takes back off, with effectiveness
    inhibits: tuple             # fouling/corrosion modes whose rate it cuts, with factor
    debris_release: float       # fraction of removed deposit handed downstream as loose debris
    why: str


_CONDITIONERS: tuple[Conditioner, ...] = (
    Conditioner(
        "valve-saver", "valve saver / top lube", fluid="fuel",
        removes=(), inhibits=(("seat-recession", 0.08),),
        debris_release=0.0,
        why="a metered potassium or phosphorus dose into the intake of a gas-converted "
            "engine. It does the job tetraethyl lead used to do by accident -- laying "
            "a soft sacrificial film on the exhaust seat so the valve closes onto "
            "something other than bare iron. The one additive on this list that is not "
            "optional: without it a soft-seat head on dry gaseous fuel recesses its "
            "seats in tens of hours of sustained load, and the fix for that is a "
            "cylinder head off the engine"),
    Conditioner(
        "engine-flush", "detergent engine flush", fluid="engine-oil",
        removes=(("sludge", 0.7), ("varnish", 0.5)), inhibits=(),
        debris_release=0.35,
        why="a strong detergent run before a drain lifts sludge and varnish off the "
            "galleries -- and sends a good part of it to the pickup screen, which is "
            "exactly why it is the one additive with a reputation for killing tired "
            "engines rather than saving them"),
    Conditioner(
        "coolant-descaler", "chelating descaler", fluid="coolant-water-glycol",
        removes=(("scale", 0.8),), inhibits=(),
        debris_release=0.15,
        why="citric or EDTA chemistry binds the calcium and iron in scale and carries "
            "it out in the flush water; it cannot touch a packed core from the outside, "
            "which is a wash, not a chemistry problem"),
    Conditioner(
        "coolant-inhibitor", "inhibitor recharge (SCA/OAT)", fluid="coolant-water-glycol",
        removes=(), inhibits=(("coolant-galvanic", 0.08), ("cavitation-erosion", 0.35)),
        debris_release=0.0,
        why="it removes nothing -- it restores the sacrificial chemistry that stops the "
            "cooling system being a battery, and the film that damps liner cavitation. "
            "A diesel run on plain water and glycol eats its liners for want of this"),
    Conditioner(
        "injector-cleaner", "PEA injector cleaner", fluid="pump-gasoline-93",
        removes=(("varnish", 0.6),), inhibits=(("varnish", 0.5),),
        debris_release=0.05,
        why="polyether-amine detergent survives the hot end of the injector and cleans "
            "the tip it is stuck to, which is where fuel varnish does its damage"),
    Conditioner(
        "fuel-stabiliser", "fuel stabiliser", fluid="pump-gasoline-93",
        removes=(), inhibits=(("varnish", 0.25),),
        debris_release=0.0,
        why="an antioxidant: it does not clean, it stops the fuel gumming while it sits, "
            "which is the whole problem with a machine that runs twice a year"),
    Conditioner(
        "anti-gel", "cold-flow improver", fluid="ultra-low-sulfur-diesel",
        removes=(("wax", 0.9),), inhibits=(("wax", 0.2),),
        debris_release=0.0,
        why="it changes the shape the wax crystals grow in so they pass the filter "
            "instead of matting across it -- both a cure and a preventative, because "
            "wax is the reversible one"),
    Conditioner(
        "water-remover", "alcohol water remover", fluid="pump-gasoline-93",
        removes=(), inhibits=(("ethanol-water", 0.3),),
        debris_release=0.0,
        why="it carries the separated water through to be burnt rather than leaving it "
            "lying in the bottom of the tank corroding it"),
    Conditioner(
        "core-wash", "core wash", fluid="air",
        removes=(("core-debris", 0.9),), inhibits=(),
        debris_release=0.0,
        why="the one that is not chemistry at all: a hose on the outside of the fins, "
            "which is the only thing that gets a packed core back"),
)

CONDITIONER_BY_KEY: dict[str, Conditioner] = {c.key: c for c in _CONDITIONERS}


def conditioner(key: str) -> Conditioner:
    c = CONDITIONER_BY_KEY.get(str(key))
    if c is None:
        raise KeyError(
            f"unknown conditioner {key!r}; stocked: {', '.join(sorted(CONDITIONER_BY_KEY))}")
    return c


def conditioners_for(mode_key: str) -> list[Conditioner]:
    """What is on the shelf for this particular blockage."""
    return [c for c in _CONDITIONERS
            if any(m == mode_key for m, _ in c.removes)
            or any(m == mode_key for m, _ in c.inhibits)]


def apply_conditioner(blocked_frac: float, mode_key: str, key: str) -> tuple[float, float]:
    """Pour it in. Returns (blockage left, debris sent downstream).

    The debris is the honest half: what a flush takes off a gallery wall
    does not vanish, it goes round the circuit to whatever filters or
    restricts next. A caller that ignores the second return value is
    modelling the advertisement rather than the product."""
    c = conditioner(key)
    b = max(0.0, min(1.0, float(blocked_frac)))
    taken = 0.0
    for m, eff in c.removes:
        if m == mode_key:
            taken = b * max(0.0, min(1.0, float(eff)))
    return b - taken, taken * c.debris_release


def inhibited_severity(severity: float, mode_key: str, key: str) -> float:
    """How much slower it forms with this in the fluid."""
    c = conditioner(key)
    for m, factor in c.inhibits:
        if m == mode_key:
            return max(0.0, float(severity)) * max(0.0, float(factor))
    return max(0.0, float(severity))


# ---------------------------------------------------------------------
# CORROSION, AND PORTS THAT MAKE THEMSELVES
# ---------------------------------------------------------------------
#
# Fouling takes the hole away. Corrosion takes the WALL away, and it
# does not stop when the wall runs out -- it makes a new hole where
# there was not one. That is the part worth building: a port nobody
# placed, that exists because a process put it there.
#
# The two belong in one module because they are the same story told
# from opposite sides, and because they drive each other. A deposit is
# not inert: it holds condensate against the metal, keeps it wet between
# runs, and concentrates whatever was dissolved in it. Under-deposit
# attack is the classic result, and it is why a fouled exhaust rots from
# the inside at the low point rather than evenly. So corrosion severity
# reads the component's own `blocked_frac`, and a part that is fouling
# is a part that is corroding faster.
#
# AUTOGENESIS. When a wall is gone, `autogenous_port` hands back a port
# specification -- position, outward direction, radius, the circuit
# behind it. Dropped into the graph it is an ordinary open port with
# nothing closing it, which the emitter field already turns into a real
# hole with real pressure behind it. No separate "corrosion leak" path:
# a hole that rusted through and a hole that was shot are the same hole.


@dataclass(frozen=True)
class CorrosionMode:
    key: str
    label: str
    agent: str           # what actually attacks the metal
    driver: str          # the exposure that lets it
    unit: str
    rate_mm_per_year: float  # ISO/NACE-style corrosion rate at reference severity,
                             # for a component exposed to this mode continuously
    deposit_coupled: bool  # whether a fouling layer over it makes it worse
    why: str


_CORROSION: tuple[CorrosionMode, ...] = (
    CorrosionMode(
        "acid-condensate", "acid condensate attack", agent="sulphuric acid",
        driver="cold-wet-hours", unit="hours with the exhaust below its dew point",
        rate_mm_per_year=0.13, deposit_coupled=True,
        why="sulphur in the fuel leaves the cylinder as SO2 and SO3, and the moment the "
            "exhaust falls below its dew point that plates out as sulphuric acid on the "
            "pipe wall. It is why an engine doing short cold runs on bad fuel rots its "
            "exhaust from the inside while one that gets hot does not"),
    CorrosionMode(
        "ethanol-water", "phase-separated alcohol attack", agent="wet ethanol",
        driver="wet-hours", unit="hours with separated water in the fuel",
        rate_mm_per_year=0.07, deposit_coupled=False,
        why="ethanol pulls water out of the air and, once enough collects, the mixture "
            "separates and sits in the bottom of the tank and the lines as a conductive, "
            "aggressive layer against steel and some alloys"),
    CorrosionMode(
        "chloride-pitting", "chloride pitting", agent="chloride",
        driver="salt-hours", unit="hours of salt exposure",
        rate_mm_per_year=0.35, deposit_coupled=True,
        why="salt -- off a road or out of the sea -- breaks down the passive film in "
            "spots rather than evenly, so it bores pits that go through a wall long "
            "before the wall as a whole is thin"),
    CorrosionMode(
        "coolant-galvanic", "galvanic attack", agent="spent inhibitor",
        driver="hours", unit="running hours on exhausted coolant",
        rate_mm_per_year=0.018, deposit_coupled=False,
        why="coolant inhibitor is consumed, and once it is gone the dissimilar metals in "
            "a cooling system are a battery with the weakest one as the anode"),
    CorrosionMode(
        "cavitation-erosion", "cavitation erosion", agent="collapsing vapour bubbles",
        driver="cavitating-hours", unit="hours cavitating against the liner",
        rate_mm_per_year=9.6, deposit_coupled=False,
        why="a wet liner rings when the piston slaps it, the coolant against it flashes "
            "and the bubbles collapse on the metal -- it drills through a liner from the "
            "COOLANT side, which is why it is a famous way to fill a sump with water"),
)

CORROSION_BY_KEY: dict[str, CorrosionMode] = {m.key: m for m in _CORROSION}

# How much faster a wall goes under a deposit that holds the electrolyte
# on it. Real under-deposit attack runs several times bare-metal rates.
DEPOSIT_ACCELERATION = 3.0


def corrosion_mode(key: str) -> CorrosionMode:
    m = CORROSION_BY_KEY.get(str(key))
    if m is None:
        raise KeyError(
            f"unknown corrosion mode {key!r}; declare one of: {', '.join(sorted(CORROSION_BY_KEY))}")
    return m


def corrode(wall_lost_m: float, mode_key: str, driver_amount: float,
            severity: float = 1.0, blocked_frac: float = 0.0) -> float:
    """Advance the wall loss. Corrosion never comes back.

    `severity` carries how aggressive the conditions actually are -- how
    much sulphur is in this fuel, how salty the air is. `blocked_frac`
    is the component's own fouling, which for a deposit-coupled mode
    makes the attack markedly worse rather than shielding it."""
    m = corrosion_mode(mode_key)
    sev = max(0.0, float(severity))
    if m.deposit_coupled:
        sev *= 1.0 + (DEPOSIT_ACCELERATION - 1.0) * max(0.0, min(1.0, float(blocked_frac)))
    # mm/yr is how every corrosion table states a rate, so that is what
    # the mode carries; the driver is in HOURS of exposure, so the year
    # is divided out here and the millimetres turned into metres.
    per_hour_m = m.rate_mm_per_year / 1000.0 / HOURS_PER_YEAR
    return max(0.0, float(wall_lost_m) + per_hour_m * max(0.0, float(driver_amount)) * sev)


def wall_remaining_m(wall_thickness_m: float, wall_lost_m: float) -> float:
    return max(0.0, float(wall_thickness_m) - float(wall_lost_m))


def has_perforated(wall_thickness_m: float, wall_lost_m: float) -> bool:
    return wall_remaining_m(wall_thickness_m, wall_lost_m) <= 0.0


# WHAT THE RATES ARE CALIBRATED AGAINST. Real service life, not
# drama: a mild-steel exhaust running heavy fuel with a lot of cold
# wet running perforates its 1.4 mm wall somewhere around two to
# three thousand hours BELOW THE DEW POINT -- years of ordinary use
# -- and the same system on clean fuel outlives the engine. The one
# genuinely fast mode is liner cavitation, which really does bore
# through several millimetres of iron in a few thousand hours and is
# why it has its own famous reputation.
#
# Reference fuel for corrosion severity: modern ultra-low-sulphur
# diesel, 15 ppm. Every rate above is quoted at this, so a fuel is
# scored by how much worse than clean it is.
REFERENCE_SULFUR_FRAC = 0.000015


def severity_from_fuel(fluid) -> float:
    """How aggressive this fuel's combustion products are, from what the
    fuel itself declares.

    `working_fluids` already carries `sulfur_frac` on every fuel -- 15
    ppm for ULSD, 0.6% for bunker, 1.5% for heavy fuel oil -- so the
    answer is read off the fuel in the tank rather than set per engine.
    Burning heavy fuel oil really does eat an exhaust roughly a thousand
    times faster than clean diesel does, and that is the ratio those
    declared numbers give.

    The square root is deliberate and disclosed: acid attack scales with
    how much acid forms, but the wall is only wet for as long as the
    exhaust is below its dew point, so the rate rises much more slowly
    than sulphur content alone would suggest."""
    s = float(getattr(fluid, "sulfur_frac", 0.0) or 0.0)
    if s <= 0.0:
        return 0.0
    return math.sqrt(s / REFERENCE_SULFUR_FRAC)


def severity_from_gum(fluid) -> float:
    """Fouling severity from a fuel's declared gum tendency -- the same
    `working_fluids` field that already describes how much varnish a
    fuel leaves behind it."""
    return 1.0 + 4.0 * max(0.0, float(getattr(fluid, "gum_tendency", 0.0) or 0.0))


def autogenous_port(node: dict, circuit: str | None = None) -> dict | None:
    """The port a corroded-through component has just grown.

    Returns a node specification ready to drop into the graph, or None
    if the wall is still there. It comes out OPEN -- no closure, nothing
    plumbed to it -- because that is what a rust hole is, and from that
    moment the ordinary open-port machinery gives it pressure, flow and
    a spray direction like any other hole in the engine.

    The hole is put at the component's own low point and aimed outward
    and down: corrosion perforates where the condensate collects and
    sits, which is the bottom, and that is also the way what comes out
    is going to go."""
    if not has_perforated(float(node.get("wall_thickness_m") or 0.0),
                          float(node.get("wall_lost_m") or 0.0)):
        return None
    pos = list(node.get("reference_position") or (0.0, 0.0, 0.0))
    # the low point of its own body, where anything wet ends up
    drop = float(node.get("drum_radius_m") or node.get("port_radius_m") or 0.02)
    extent = node.get("body_half_extent_m")
    if extent:
        drop = float(extent[1])
    pos[1] -= drop
    mode_key = str(node.get("corrosion_mode") or "")
    return dict(
        identity=f"{node['identity']}.perforation",
        reference_position=[float(v) for v in pos],
        kind="engine-block-port",
        port_kind="corrosion-perforation",
        port_direction=[0.0, -1.0, 0.0],
        # a rust hole starts as a pinhole and opens up from there
        port_radius_m=max(0.0008, float(node.get("perforation_radius_m") or 0.0015)),
        fluid_role=str(circuit or node.get("fluid_role") or "exhaust"),
        part=str(node.get("part") or node["identity"]),
        mating=False, connected=False, plugged=False, closure="",
        autogenous=True, corrosion_mode=mode_key,
    )


def is_service_due(blocked_frac: float, threshold: float = 0.5) -> bool:
    """Whether this component has lost enough area to want attention."""
    return float(blocked_frac) >= threshold
