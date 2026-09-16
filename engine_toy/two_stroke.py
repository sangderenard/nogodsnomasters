"""Two-strokes breathe through holes in the wall, not through valves.

port_flow models an intake poppet valve: a curtain that opens, a throat
behind it, and a Mach index that says when gas speed stops the cylinder
filling. All of that is the right model for a four-stroke and the wrong
model for a two-stroke, which has no intake valve at all. A Wartsila
RTA96C was being derived as though it had poppet intake valves it does
not have, and came out at a twentieth of its real output.

WHAT ACTUALLY HAPPENS. There is no intake stroke. The piston uncovers a
row of ports cut through the cylinder liner near bottom dead centre, and
fresh air is pushed in THROUGH those ports by something upstream -- a
turbocharger, a blower, or on a small engine the underside of the piston
using the crankcase as a pump. The incoming air has to physically shove
the burnt gas out ahead of it. That is scavenging, and it is a fluid
problem, not a valve problem.

WHICH GIVES TWO NUMBERS INSTEAD OF ONE, and the difference between them
is the whole character of a two-stroke:

  DELIVERY RATIO      air supplied to the cylinder, over what the swept
                      volume would hold at ambient. Can exceed 1.0
                      easily -- a turbocharged marine engine delivers
                      half as much again as its own displacement,
                      because a blower is forcing it through.

  TRAPPING EFFICIENCY what fraction of that air is still in the
                      cylinder when the ports shut. The rest went
                      straight out of the exhaust without ever being
                      burnt, taking any fuel mixed into it along.

  charging efficiency = delivery ratio x trapping efficiency, and THAT
                      is the number equivalent to a four-stroke's
                      volumetric efficiency.

THE TRAPPING NUMBER IS WHY SMALL TWO-STROKES ARE DIRTY. A crankcase-
scavenged engine mixes fuel into the scavenge air, and a third of that
air goes out of the exhaust port unburnt. It is not a combustion
failure -- the fuel never had a chance to burn, it simply left. That is
the blue haze, the smell, and the reason the type is legislated out of
almost everything.

AND WHY THE BIG ONES ARE THE MOST EFFICIENT ENGINES EVER BUILT. Uniflow
scavenging puts the ports at the bottom and a single poppet valve in the
head, so the gas flows one way through the cylinder and the fresh charge
pushes the burnt gas ahead of it like a piston rather than mixing with
it. Trapping reaches the high nineties. Combine that with an enormous
bore, a hundred rpm, and fuel injected only after the ports have shut --
so nothing can short-circuit -- and the result is over fifty per cent
thermal efficiency.

Both facts come from the same number, read at its two extremes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScavengeType:
    key: str
    label: str
    #: WHERE THIS GEOMETRY SITS BETWEEN THE TWO EXACT LIMITS, 0 to 1.
    #:
    #: Trapping efficiency is not something to fit a slope to. Hopkinson
    #: solved it in closed form in 1914 by bounding it between two ideal
    #: cases, and every real engine lies between them:
    #:
    #:   perfect displacement   the fresh charge pushes the burnt gas
    #:                          ahead of it like a piston, with no
    #:                          mixing at all. Nothing fresh can leave
    #:                          until everything burnt has, so
    #:                          TE = min(1, 1/DR) -- exactly.
    #:
    #:   perfect mixing         the incoming charge mixes instantly and
    #:                          uniformly with the cylinder contents, so
    #:                          what leaves is always the current mixture
    #:                          and some fresh charge always escapes.
    #:                          Integrating that gives
    #:                          TE = (1 - e^-DR)/DR -- exactly.
    #:
    #: So the only thing a scavenge geometry contributes is how close to
    #: plug flow it manages to be. One number, physically meaningful,
    #: replacing a pair of fitted constants.
    plug_flow: float
    #: What the incoming air is carrying. If the fuel is mixed in
    #: upstream, everything that short-circuits takes fuel with it.
    fuel_in_scavenge: bool
    why: str = ""


SCAVENGE_TYPES: dict[str, ScavengeType] = {
    "uniflow": ScavengeType(
        "uniflow", "uniflow, liner ports and head valve", 0.92, False,
        why="ports all round the bottom of the liner, one big poppet valve in the "
            "head, and the gas goes one way. The fresh charge pushes the burnt gas "
            "ahead of it instead of mixing with it, which is why trapping is near "
            "perfect -- and the fuel is injected after the ports shut, so none of it "
            "can leave unburnt"),
    "loop": ScavengeType(
        "loop", "loop (Schnuerle) scavenged", 0.55, True,
        why="transfer ports aimed so the incoming charge sweeps up one side of the "
            "bore, across the head and down the other to the exhaust. Far better than "
            "a deflector piston and still mixing with what it is trying to expel"),
    "cross": ScavengeType(
        "cross", "cross scavenged, deflector piston", 0.30, True,
        why="a hump on the piston crown meant to steer the incoming charge upward. It "
            "steers some of it. The deflector also wrecks the chamber shape and adds "
            "mass exactly where a two-stroke wants none"),
    "crankcase": ScavengeType(
        "crankcase", "crankcase scavenged", 0.40, True,
        why="the underside of the piston is the pump. Beautifully simple, nothing "
            "extra to drive, and it means the crankcase is full of fuel-air mixture "
            "rather than oil -- so the engine has to be lubricated by oil mixed into "
            "its own fuel, and burns it"),
}


def scavenge_type(key: str) -> ScavengeType:
    s = SCAVENGE_TYPES.get(str(key))
    if s is None:
        raise KeyError(f"unknown scavenge type {key!r}; declared: "
                       f"{', '.join(sorted(SCAVENGE_TYPES))}")
    return s


#: Port height as a fraction of stroke, typical. The ports are uncovered
#: for the part of the stroke below them, so this sets how long they are
#: open -- and a two-stroke's "cam timing" is nothing but where these
#: holes were cut.
DEFAULT_PORT_HEIGHT_FRAC = 0.22


def port_open_deg(port_height_frac: float = DEFAULT_PORT_HEIGHT_FRAC,
                  rod_stroke_ratio: float = 1.9) -> float:
    """Crank degrees the scavenge ports are open, symmetric about BDC.

    Falls out of the slider-crank: the ports are uncovered whenever the
    piston is below them. A two-stroke's port timing is not adjustable
    without cutting metal, which is why porting IS tuning on these
    engines."""
    f = max(0.01, min(0.9, float(port_height_frac)))
    r = max(1.2, float(rod_stroke_ratio)) * 2.0
    # piston position from BDC, as a fraction of stroke, at angle th
    # from BDC: solve for where it equals the port height
    lo, hi = 0.0, 180.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        th = math.radians(180.0 - mid)
        x = ((1.0 - math.cos(th)) + (r - math.sqrt(max(0.0, r * r - math.sin(th) ** 2)))) / 2.0
        if (1.0 - x) < f:
            lo = mid
        else:
            hi = mid
    return 2.0 * lo


#: Crankcase compression ratio: (crankcase volume + swept) / crankcase
#: volume. A real, measurable geometric property of the casting, and the
#: thing that decides how well the underside of the piston pumps. Small
#: engines run 1.4 to 1.5; any higher and the case gets so small that
#: pumping losses and heat outweigh the delivery gained.
DEFAULT_CRANKCASE_COMPRESSION = 1.45


#: Fraction of the liner circumference that is actually port rather
#: than the bridges between ports. Geometric and measurable: a real
#: uniflow liner is close to half holes at the scavenge belt.
DEFAULT_PORT_CIRCUMFERENCE_FRAC = 0.40


def port_area_m2(bore_m: float, stroke_m: float,
                 port_height_frac: float = DEFAULT_PORT_HEIGHT_FRAC,
                 circumference_frac: float = DEFAULT_PORT_CIRCUMFERENCE_FRAC) -> float:
    """Open area of the scavenge belt, from the castings.

    Circumference times the fraction of it that is port, times the port
    height. Nothing here is chosen except two ratios that can be
    measured off a liner."""
    return (math.pi * max(1e-6, float(bore_m))
            * max(0.0, min(0.95, float(circumference_frac)))
            * max(0.0, float(port_height_frac)) * max(1e-6, float(stroke_m)))


def port_limited_delivery(bore_m: float, stroke_m: float, rpm: float,
                          scavenge_pressure_pa: float,
                          port_height_frac: float = DEFAULT_PORT_HEIGHT_FRAC,
                          circumference_frac: float = DEFAULT_PORT_CIRCUMFERENCE_FRAC,
                          scavenge_temp_k: float = 320.0) -> float:
    """Delivery ratio the PORTS alone will allow.

    Uses drivetrain_graph.choked_orifice_mass_flow_kg_s -- the one
    orifice relation this project has, the same one governing a
    wastegate or a relief valve -- over the time the ports are actually
    open. No parallel flow model.

    On a well-designed engine this comes out enormous, and that is the
    correct answer rather than a broken one: marine scavenge ports are
    deliberately huge so that they are never the restriction. It only
    binds when somebody has cut small ports, which is exactly when it
    should."""
    import drivetrain_graph as dg
    area = port_area_m2(bore_m, stroke_m, port_height_frac, circumference_frac)
    flow = dg.choked_orifice_mass_flow_kg_s(area, float(scavenge_pressure_pa),
                                            float(scavenge_temp_k))
    if flow <= 0.0:
        return 0.0
    # a two-stroke scavenges once per revolution
    open_s = port_open_deg(port_height_frac) / (6.0 * max(1.0, float(rpm)))
    swept = math.pi * 0.25 * bore_m ** 2 * stroke_m
    return (flow * open_s) / max(1e-12, swept * 1.184)


def delivery_ratio(boost_frac: float = 0.0, crankcase: bool = False,
                   crankcase_compression: float = DEFAULT_CRANKCASE_COMPRESSION,
                   transfer_pressure_ratio: float = 1.25,
                   bore_m: float = 0.0, stroke_m: float = 0.0,
                   rpm: float = 0.0) -> float:
    """Air offered to the cylinder, over what its swept volume holds.

    A BLOWN ENGINE simply offers what the blower delivers: one
    atmosphere plus the boost.

    A CRANKCASE-SCAVENGED ONE is pumped by the underside of its own
    piston, which is a reciprocating pump with a very large clearance
    volume -- and the exact relation for that is already in this
    project. compressors.volumetric_efficiency is the standard
    clearance-re-expansion result:

        eta_v = 1 + c - c . r^(1/n)

    where c is clearance volume over swept volume and r the pressure
    ratio. A crankcase's clearance ratio is 1/(CCR - 1), which for a
    typical 1.45 crankcase compression is well over two -- an enormous
    dead volume by pump standards, and exactly why a crankcase-scavenged
    engine cannot deliver even its own displacement.

    This replaces a fitted bell curve over rpm. The rpm dependence a
    real one shows is not a property of the pump at all -- it comes from
    the tuned expansion chamber on the exhaust reflecting escaped charge
    back in, which belongs to the exhaust and not here."""
    if not crankcase:
        # TWO REAL CONSTRAINTS, AND THE ANSWER IS THE SMALLER.
        #
        # The blower can only deliver what its pressure ratio implies,
        # and the ports can only pass what an orifice of that area
        # passes while they are open. A real engine is limited by one or
        # the other, and which one is a design statement: marine
        # practice makes the ports so large that the blower always
        # binds, and an engine with small ports is port-limited no
        # matter what is bolted to it.
        #
        # 1 + boost alone was only ever the first of these.
        blower = 1.0 + max(0.0, float(boost_frac))
        if bore_m > 0.0 and stroke_m > 0.0 and rpm > 0.0:
            ports = port_limited_delivery(
                bore_m, stroke_m, rpm,
                scavenge_pressure_pa=101_325.0 * blower)
            return min(blower, ports)
        return blower
    import compressors as _cp
    ccr = max(1.05, float(crankcase_compression))
    clearance = 1.0 / (ccr - 1.0)
    return max(0.05, _cp.volumetric_efficiency(
        max(1.001, float(transfer_pressure_ratio)), clearance, 1.3))


def perfect_displacement_trapping(dr: float) -> float:
    """The upper bound. Exact.

    No mixing at all: fresh charge sweeps burnt gas ahead of it, so
    until the cylinder is full nothing fresh can escape. Past a delivery
    ratio of one, everything extra goes straight out."""
    d = max(1e-9, float(dr))
    return min(1.0, 1.0 / d)


def perfect_mixing_trapping(dr: float) -> float:
    """The lower bound. Also exact.

    The incoming charge mixes instantly with the cylinder contents, so
    what leaves is always the current mixture and some fresh charge
    always escapes -- even at a delivery ratio well under one. The
    integral of that is (1 - e^-DR)/DR, which is why a badly scavenged
    engine cannot be fixed by simply blowing harder."""
    d = max(1e-9, float(dr))
    return (1.0 - math.exp(-d)) / d


def trapping_efficiency(scavenge: str, dr: float) -> float:
    """Where this geometry actually lands between the two exact limits.

    Not a fitted curve: both bounds are closed-form, and the scavenge
    type contributes only how close to plug flow it manages to be."""
    s = scavenge_type(scavenge)
    k = max(0.0, min(1.0, s.plug_flow))
    return (k * perfect_displacement_trapping(dr)
            + (1.0 - k) * perfect_mixing_trapping(dr))


@dataclass
class ScavengeResult:
    delivery_ratio: float
    trapping: float
    charging: float
    port_open_deg: float
    fuel_short_circuited: float
    why: str = ""


def charge(scavenge: str, *, boost_frac: float = 0.0, rpm_frac: float = 1.0,
           port_height_frac: float = DEFAULT_PORT_HEIGHT_FRAC,
           bore_m: float = 0.0, stroke_m: float = 0.0,
           rpm: float = 0.0) -> ScavengeResult:
    """What a two-stroke cylinder actually ends up holding.

    The charging efficiency returned is the direct equivalent of a
    four-stroke's volumetric efficiency and should be used in its
    place -- not multiplied by it."""
    s = scavenge_type(scavenge)
    crank = (scavenge == "crankcase")
    dr = delivery_ratio(boost_frac, crankcase=crank, bore_m=bore_m,
                        stroke_m=stroke_m, rpm=rpm)
    tr = trapping_efficiency(scavenge, dr)
    lost = (1.0 - tr) if s.fuel_in_scavenge else 0.0
    # THE CYLINDER IS NOT FULL SIZE WHEN IT SHUTS. Scavenge ports are
    # uncovered symmetrically about bottom dead centre, so they close
    # on the way back UP -- by which point the piston has already
    # travelled the height of the ports. The volume actually sealed in
    # is the swept volume less that, which is why a two-stroke's
    # EFFECTIVE stroke is shorter than its mechanical one and why its
    # real compression ratio is always lower than the geometric figure
    # anyone quotes.
    #
    # Leaving this out gave the Wartsila a charging efficiency of 2.52
    # and nearly three times its real torque.
    # Exact, not chosen: ports cut at height h above bottom dead centre
    # close when the piston reaches h, so the volume sealed in is the
    # swept volume less that height. 1 - h/stroke, by definition of
    # where the holes are.
    trapped_frac = max(0.35, 1.0 - float(port_height_frac))
    return ScavengeResult(
        delivery_ratio=dr, trapping=tr, charging=dr * tr * trapped_frac,
        port_open_deg=port_open_deg(port_height_frac),
        fuel_short_circuited=lost,
        why=("fuel is injected after the ports shut, so nothing that short-circuits "
             "is carrying any" if not s.fuel_in_scavenge else
             f"{lost * 100:.0f}% of the fuel leaves unburnt with the scavenge air -- "
             "not a combustion failure, it simply left"))


def scavenge_of(engine) -> str:
    """Which kind this engine uses, from what it is.

    A large slow marine two-stroke is uniflow; a small one is crankcase
    scavenged. The dividing line is whether there is anything to drive
    a blower with, and on anything under a few litres there is not."""
    arch = getattr(engine, "architecture", None)
    if arch is None or not getattr(arch, "two_stroke", False):
        return ""
    fi = getattr(engine, "forced_induction", None)
    blown = fi is not None and getattr(fi, "kind", None) in ("turbo", "supercharger")
    if blown and float(getattr(engine, "displacement_l", 0.0) or 0.0) > 20.0:
        return "uniflow"
    return "crankcase"
