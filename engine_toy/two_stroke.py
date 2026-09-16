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
    #: Fraction of delivered air still aboard when the ports close, at
    #: a delivery ratio of 1.0. The single most important number here.
    trapping_at_unity: float
    #: How fast trapping falls as more air is pushed through. Shove
    #: harder and a greater share of it short-circuits straight out.
    short_circuit_slope: float
    #: What the incoming air is carrying. If the fuel is mixed in
    #: upstream, everything that short-circuits takes fuel with it.
    fuel_in_scavenge: bool
    why: str = ""


SCAVENGE_TYPES: dict[str, ScavengeType] = {
    "uniflow": ScavengeType(
        "uniflow", "uniflow, liner ports and head valve", 0.97, 0.10, False,
        why="ports all round the bottom of the liner, one big poppet valve in the "
            "head, and the gas goes one way. The fresh charge pushes the burnt gas "
            "ahead of it instead of mixing with it, which is why trapping is near "
            "perfect -- and the fuel is injected after the ports shut, so none of it "
            "can leave unburnt"),
    "loop": ScavengeType(
        "loop", "loop (Schnuerle) scavenged", 0.78, 0.30, True,
        why="transfer ports aimed so the incoming charge sweeps up one side of the "
            "bore, across the head and down the other to the exhaust. Far better than "
            "a deflector piston and still mixing with what it is trying to expel"),
    "cross": ScavengeType(
        "cross", "cross scavenged, deflector piston", 0.65, 0.38, True,
        why="a hump on the piston crown meant to steer the incoming charge upward. It "
            "steers some of it. The deflector also wrecks the chamber shape and adds "
            "mass exactly where a two-stroke wants none"),
    "crankcase": ScavengeType(
        "crankcase", "crankcase scavenged", 0.70, 0.34, True,
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


def delivery_ratio(boost_frac: float = 0.0, crankcase: bool = False,
                   rpm_frac: float = 1.0) -> float:
    """Air offered to the cylinder, over what its swept volume holds.

    A blown engine can offer far more than its own displacement; a
    crankcase-scavenged one cannot even manage its own, because the
    pump is a piston underside with a large dead volume and no valves
    worth the name."""
    if crankcase:
        # the crankcase is a poor pump and gets worse away from the
        # speed its transfer ports and reed were tuned for
        f = max(0.0, min(1.5, float(rpm_frac)))
        return 0.62 + 0.28 * math.exp(-((f - 0.75) / 0.42) ** 2)
    return 1.0 + max(0.0, float(boost_frac))


def trapping_efficiency(scavenge: str, dr: float) -> float:
    """Share of delivered air still aboard when the ports close.

    Falls as delivery ratio rises: pushing more air through a cylinder
    that is already full means the extra goes out of the exhaust. This
    is the term that makes "just add more boost" stop working on a
    two-stroke long before it does on a four-stroke."""
    s = scavenge_type(scavenge)
    excess = max(0.0, float(dr) - 1.0)
    return max(0.15, min(0.99, s.trapping_at_unity - s.short_circuit_slope * excess))


@dataclass
class ScavengeResult:
    delivery_ratio: float
    trapping: float
    charging: float
    port_open_deg: float
    fuel_short_circuited: float
    why: str = ""


def charge(scavenge: str, *, boost_frac: float = 0.0, rpm_frac: float = 1.0,
           port_height_frac: float = DEFAULT_PORT_HEIGHT_FRAC) -> ScavengeResult:
    """What a two-stroke cylinder actually ends up holding.

    The charging efficiency returned is the direct equivalent of a
    four-stroke's volumetric efficiency and should be used in its
    place -- not multiplied by it."""
    s = scavenge_type(scavenge)
    crank = (scavenge == "crankcase")
    dr = delivery_ratio(boost_frac, crankcase=crank, rpm_frac=rpm_frac)
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
