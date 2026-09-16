"""Variable valve control, and the clearance it is taking up.

Two things live here that look unrelated and are the same mechanism seen
from opposite ends.

THE FIRST IS DELIBERATE. Every variable valve system is, underneath, a
way of changing WHEN the intake valve closes -- and when the intake
valve closes decides how much charge is actually trapped. Close it late
and some of the charge is pushed back out into the port before
compression starts, so the cylinder compresses a smaller charge through
the same mechanical stroke. That is the Miller and Atkinson idea, and it
means an engine's EFFECTIVE compression ratio is not a property of its
geometry at all. It is a property of its cam timing, adjustable while
running.

THE SECOND IS ACCIDENTAL, and it arrives at the same number. A recessed
valve seat also lowers effective compression, by leaking charge past a
valve that no longer seals. crankcase_state.effective_compression_ratio
already carries both terms because they are genuinely the same quantity
-- what the cylinder actually compresses -- reached by a design choice
in one case and by damage in the other.

AND THE TWO INTERACT BADLY, WHICH IS THE POINT OF PUTTING THEM IN ONE
FILE. Valve seat recession sinks the valve into the head, which takes up
the valve clearance. What happens next depends entirely on what is
holding that clearance:

  SOLID lifters or shims give you a WARNING, and a good one. The
  clearance closes gradually, the valvetrain gets quieter as it goes,
  and a mechanic who checks clearances finds the seats sinking long
  before anything burns. When clearance reaches zero the valve is held
  off its seat and compression collapses -- loud, obvious, and
  survivable because the head comes off before the valve does.

  HYDRAULIC lash adjusters take the recession up AUTOMATICALLY and
  SILENTLY. That is their whole job: they maintain zero lash. So the
  engine never gets noisy, the clearance never needs checking, nothing
  changes -- right up until the adjuster runs out of travel or the valve
  simply burns because it has been sealing on a worn seat with poor heat
  transfer. The first symptom is a dead cylinder. The convenience
  feature deletes the warning.

That is not a quirk worth a comment -- it inverts which engine is the
dangerous one to convert. The older, cruder, solid-lifter engine is the
SAFER conversion, because it tells you what is happening to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------
# how the clearance is held
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class LashAdjustment:
    key: str
    label: str
    #: Does it silently absorb seat recession?
    self_adjusting: bool
    #: How much recession it can take up before it runs out.
    travel_mm: float
    #: Cold clearance, where fitted. Zero for hydraulic.
    nominal_clearance_mm: float
    #: Does a mechanic get an audible or measurable warning first?
    gives_warning: bool
    why: str = ""


LASH_ADJUSTMENTS: dict[str, LashAdjustment] = {
    "solid-shim": LashAdjustment(
        "solid-shim", "shim under bucket", False, 0.0, 0.25, True,
        why="a stack of ground shims. Clearance is checked with a feeler gauge and "
            "changed by swapping a shim, which is tedious and is exactly why the "
            "engine gets inspected on a schedule -- the inspection is the safety "
            "feature, not the shim"),
    "solid-adjuster": LashAdjustment(
        "solid-adjuster", "screw and locknut", False, 0.0, 0.30, True,
        why="the old arrangement, and the forgiving one: clearance is checked and "
            "reset with a spanner in minutes, so it gets done. A recessing seat shows "
            "up as a clearance that keeps closing up between services"),
    "hydraulic": LashAdjustment(
        "hydraulic", "hydraulic lash adjuster", True, 4.0, 0.0, False,
        why="an oil-fed plunger that maintains zero lash for the life of the engine. "
            "It is a genuine improvement in every respect except one: it absorbs seat "
            "recession without a sound, so the engine gives no warning at all and the "
            "first symptom is a burnt valve"),
    "desmodromic": LashAdjustment(
        "desmodromic", "desmodromic (positive closing)", False, 0.0, 0.10, True,
        why="the valve is pulled shut by a second rocker rather than by a spring. "
            "Clearance matters at both ends and there are two to set per valve, so "
            "this is the most labour-intensive head here and the one that tells you "
            "most about its own condition"),
}


def lash_adjustment(key: str) -> LashAdjustment:
    a = LASH_ADJUSTMENTS.get(str(key))
    if a is None:
        raise KeyError(f"unknown lash adjustment {key!r}; declared: "
                       f"{', '.join(sorted(LASH_ADJUSTMENTS))}")
    return a


#: How far a seat sinks between new and fully recessed.
#:
#: CALIBRATED AGAINST WHAT "DAMAGE 1.0" MEANS, which is the thing that
#: makes this number not arbitrary. wear.py's valve_seats component
#: reaching 1.0 means the seat is finished -- and for a seat, finished
#: means the valve is being held off it. So full recession has to be
#: about one nominal valve clearance, not several: at 1.2 mm the valve
#: was held open at a quarter of declared damage and the other three
#: quarters of the scale meant nothing.
#:
#: Documented valve seat recession failures run 0.25 to 0.5 mm, which
#: is the same number from the other direction.
FULL_RECESSION_MM = 0.45


def recession_mm(seat_damage_frac: float) -> float:
    return max(0.0, min(1.0, float(seat_damage_frac))) * FULL_RECESSION_MM


def clearance_state(seat_damage_frac: float, lash_key: str = "solid-adjuster") -> dict:
    """What the recession has done to the valve clearance, and whether
    anyone is going to find out before it burns."""
    a = lash_adjustment(lash_key)
    sunk = recession_mm(seat_damage_frac)
    if a.self_adjusting:
        used = min(a.travel_mm, sunk)
        return {"lash": a.label, "recession_mm": sunk,
                "clearance_mm": 0.0, "adjuster_used_mm": used,
                "adjuster_exhausted": sunk >= a.travel_mm,
                "held_open": sunk >= a.travel_mm,
                "warned": False, "audible_change": False,
                "why": ("the adjuster is taking the recession up silently. Nothing "
                        "sounds different, nothing measures different, and the seat is "
                        f"{sunk:.2f} mm into the head")}
    clear = max(0.0, a.nominal_clearance_mm - sunk)
    return {"lash": a.label, "recession_mm": sunk,
            "clearance_mm": clear, "adjuster_used_mm": 0.0,
            "adjuster_exhausted": False,
            "held_open": clear <= 0.0,
            "warned": clear < a.nominal_clearance_mm * 0.6,
            "audible_change": clear < a.nominal_clearance_mm * 0.8,
            "why": ("clearance closes as the seat sinks, so the valvetrain quietens "
                    "and a feeler gauge finds it. At zero the valve is held off its "
                    "seat, cannot dump heat into the head, and burns")}


# ---------------------------------------------------------------------
# variable valve control
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class VVCSystem:
    key: str
    label: str
    #: Crank degrees of intake phase authority. This is the one that
    #: moves effective compression.
    phase_authority_deg: float
    #: Can it change LIFT as well as phase?
    variable_lift: bool
    #: Discrete profiles rather than continuous.
    discrete_steps: int
    #: Does it need oil pressure to work? Almost all of them do, which
    #: makes them a hot-idle and cold-start liability.
    oil_driven: bool
    why: str = ""


VVC_SYSTEMS: dict[str, VVCSystem] = {
    "none": VVCSystem(
        "none", "fixed cam", 0.0, False, 0, False,
        why="one profile, one phase, chosen as a compromise and living with it. "
            "Everything below exists because that compromise is expensive"),
    "cam-phaser": VVCSystem(
        "cam-phaser", "cam phaser", 50.0, False, 0, True,
        why="a vane actuator in the cam sprocket rotates the whole cam against the "
            "crank. Cheap, continuous, and it moves intake closing -- which is the "
            "lever on effective compression. It is also oil-driven, so it does not "
            "work until the oil is warm and does not work at all on a worn pump"),
    "lift-switching": VVCSystem(
        "lift-switching", "switching rocker (two profiles)", 15.0, True, 2, True,
        why="two cam lobes and a pin that locks a rocker to the other one. A step "
            "change rather than a sweep, which is why these engines have a distinct "
            "and famous second personality rather than a smooth spread"),
    "continuous-lift": VVCSystem(
        "continuous-lift", "continuously variable lift", 60.0, True, 0, False,
        why="an eccentric shaft and an intermediate lever vary lift continuously, so "
            "the valve itself does the throttling and the throttle plate can be "
            "deleted. That removes pumping loss at part load, which is where an "
            "engine spends its life"),
    "camless": VVCSystem(
        "camless", "electrohydraulic camless", 180.0, True, 0, False,
        why="each valve on its own actuator, every event independently commanded. "
            "Total authority, and the thing it is really for is cylinder deactivation "
            "and running different cycles on different strokes"),
}


def vvc_system(key: str) -> VVCSystem:
    v = VVC_SYSTEMS.get(str(key))
    if v is None:
        raise KeyError(f"unknown VVC system {key!r}; declared: "
                       f"{', '.join(sorted(VVC_SYSTEMS))}")
    return v


def cylinder_volume_frac(crank_deg: float, rod_stroke_ratio: float = 1.75) -> float:
    """Swept volume above the piston at this crank angle, as a fraction
    of full swept volume. 0 at TDC, 1 at BDC.

    The real slider-crank, not a cosine. The rod length matters: a short
    rod moves the piston away from TDC faster, which is why rod ratio
    changes an engine's character and not merely its geometry."""
    th = math.radians(float(crank_deg))
    r = max(1.2, float(rod_stroke_ratio)) * 2.0     # rod / crank throw
    x = (1.0 - math.cos(th)) + (r - math.sqrt(max(0.0, r * r - math.sin(th) ** 2)))
    return max(0.0, min(1.0, x / 2.0))


def effective_compression_ratio(geometric_ratio: float, ivc_deg_after_bdc: float,
                                rod_stroke_ratio: float = 1.75) -> float:
    """Compression ratio the cylinder ACTUALLY achieves, from cam timing.

    Compression does not begin when the piston starts up -- it begins
    when the intake valve shuts. Everything before that is pushed back
    out into the port. So the trapped volume is the volume at INTAKE
    VALVE CLOSING, and the effective ratio is that over the clearance
    volume.

    This is the Miller/Atkinson lever, and it is worth being clear about
    what it buys: the EXPANSION ratio is unchanged, because the exhaust
    valve still opens where it always did. So the engine expands the
    charge further than it compressed it, which is thermodynamically
    free efficiency paid for in power density."""
    g = max(1.5, float(geometric_ratio))
    vc = 1.0 / (g - 1.0)                       # clearance, in swept units
    trapped = cylinder_volume_frac(180.0 + max(0.0, float(ivc_deg_after_bdc)),
                                   rod_stroke_ratio)
    return (trapped + vc) / vc


def expansion_over_compression(geometric_ratio: float, ivc_deg_after_bdc: float,
                               rod_stroke_ratio: float = 1.75) -> float:
    """How much more the charge expands than it was compressed.

    1.0 is an ordinary Otto cycle. Above 1.0 is Atkinson/Miller, and the
    number is roughly the efficiency the trick is buying."""
    eff = effective_compression_ratio(geometric_ratio, ivc_deg_after_bdc,
                                      rod_stroke_ratio)
    return max(1.0, float(geometric_ratio)) / max(1.0, eff)


@dataclass
class ValveControl:
    """A head's valve arrangement: what varies, and what holds the lash."""
    identity: str = "powertrain.valve_control"
    system: str = "none"
    lash: str = "solid-adjuster"
    #: Where intake closing sits with the mechanism at rest.
    base_ivc_deg_after_bdc: float = 40.0
    #: Commanded phase, within the system's authority. Negative advances.
    phase_deg: float = 0.0
    oil_pressure_pa: float = 300_000.0
    oil_temp_k: float = 363.15

    @property
    def spec(self) -> VVCSystem:
        return vvc_system(self.system)

    @property
    def lash_spec(self) -> LashAdjustment:
        return lash_adjustment(self.lash)

    @property
    def available(self) -> bool:
        """Oil-driven systems do not work cold or on low pressure.

        Not a detail: a cam phaser on a tired engine parks at its rest
        position and the engine simply runs as though it had a fixed
        cam, which is a real and commonly misdiagnosed fault."""
        if not self.spec.oil_driven:
            return True
        return self.oil_pressure_pa >= 150_000.0 and self.oil_temp_k >= 333.0

    @property
    def ivc_deg(self) -> float:
        if not self.available:
            return self.base_ivc_deg_after_bdc
        a = self.spec.phase_authority_deg
        return self.base_ivc_deg_after_bdc + max(-a, min(a, self.phase_deg))

    def effective_ratio(self, geometric_ratio: float,
                        rod_stroke_ratio: float = 1.75) -> float:
        return effective_compression_ratio(geometric_ratio, self.ivc_deg,
                                           rod_stroke_ratio)

    def recession_report(self, seat_damage_frac: float) -> dict:
        return clearance_state(seat_damage_frac, self.lash)

    def report(self, geometric_ratio: float, seat_damage_frac: float = 0.0) -> dict:
        r = self.recession_report(seat_damage_frac)
        return {"system": self.spec.label, "available": self.available,
                "ivc_deg_after_bdc": self.ivc_deg,
                "effective_cr": self.effective_ratio(geometric_ratio),
                "geometric_cr": geometric_ratio,
                "atkinson_factor": expansion_over_compression(
                    geometric_ratio, self.ivc_deg),
                "clearance": r}
