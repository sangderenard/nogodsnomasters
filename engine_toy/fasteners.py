"""Tapped holes, machined clamps, and the two ways threads go wrong.

NAMED `fasteners` AND NOT `threading`, because a module called
`threading` on the path shadows Python's own, and every import of the
standard library's threading in the whole process silently gets this
instead. It surfaced as an AttributeError at interpreter shutdown, which
is a long way from the cause.

A tapped hole is the joining method a pair of hands can actually perform
with a drill and a tap, and it carries a decision the others do not: the
hole is IN THE PART. A bolt that breaks is a consumable; threads that
strip have destroyed the component they were cut into. That asymmetry is
the whole of thread design.

THE RULE THAT FOLLOWS FROM IT: engage enough thread that THE BOLT BREAKS
FIRST. A broken bolt is drilled out and replaced in an afternoon; a
stripped boss is a new part. The minimum engagement is not a convention,
it is wherever the female thread's shear capacity overtakes the bolt's
tensile capacity -- which is why the rule of thumb is one diameter into
steel and two into aluminium. It is the same calculation with a weaker
female material.

THE TAP DRILL IS THE OTHER DECISION, and it is counter-intuitive. A
bigger hole means less thread engaged -- 75% instead of 100% -- and
costs only about five per cent of the strength, while nearly HALVING the
torque needed to cut it. Chasing 100% thread buys almost nothing and is
the single most common way to snap a tap.

AND TAPS BREAK. A small tap in a deep hole in tough material is very
near its own torsional limit while cutting, and a broken tap is hardened
steel sitting in a hole that cannot be drilled out. That is the risk the
player is actually managing, and it is worse for SMALL threads, because
tap strength falls with the cube of diameter while cutting torque falls
only with the square.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from milspec import MATERIAL_BY_KEY


@dataclass(frozen=True)
class Thread:
    """An ISO metric coarse thread."""
    designation: str
    major_mm: float
    pitch_mm: float

    @property
    def tensile_stress_area_m2(self) -> float:
        """The bolt's effective section: the standard ISO expression,
        taken at the mean of pitch and minor diameters."""
        d = (self.major_mm - 0.9382 * self.pitch_mm) / 1000.0
        return math.pi * d * d / 4.0

    @property
    def pitch_diameter_mm(self) -> float:
        return self.major_mm - 0.6495 * self.pitch_mm


THREADS = {t.designation: t for t in (
    Thread("M3", 3.0, 0.5), Thread("M4", 4.0, 0.7), Thread("M5", 5.0, 0.8),
    Thread("M6", 6.0, 1.0), Thread("M8", 8.0, 1.25), Thread("M10", 10.0, 1.5),
    Thread("M12", 12.0, 1.75), Thread("M16", 16.0, 2.0), Thread("M20", 20.0, 2.5),
)}

#: Bolt property classes, ultimate tensile in Pa. Real values.
BOLT_CLASS = {"4.8": 420e6, "8.8": 830e6, "10.9": 1040e6, "12.9": 1220e6}

#: HSS tap torsional strength coefficient: taps break at roughly this
#: shear stress across their core, which is about 0.7 of the major
#: diameter. Brittle and hard, so it snaps rather than yields.
TAP_SHEAR_PA = 1150e6
TAP_CORE_FRACTION = 0.70

#: Calibrated so an M10 x 1.5 at 75% thread, 1.5 diameters deep, in 4130
#: lands near the published 15 N.m. Empirical, and labelled as such
#: rather than dressed up as a derivation. The first version of this
#: expression carried a stray factor of a thousand and reported 633 N.m
#: to tap an M3, which is roughly the torque of a car engine.
TAPPING_TORQUE_COEFFICIENT = 3.8e-4


def tap_drill_mm(thread: Thread, percent_thread: float = 75.0) -> float:
    """The hole to drill before tapping.

    Percent thread is how much of the full thread form is actually cut.
    The relation is the standard one, and the useful thing it shows is
    how little the hole has to grow to take a lot of work out of the
    cut."""
    return thread.major_mm - (percent_thread / 76.98) * thread.pitch_mm


def percent_thread_for(thread: Thread, drill_mm: float) -> float:
    return 76.98 * (thread.major_mm - drill_mm) / thread.pitch_mm


@dataclass
class TappedHole:
    """A thread cut into a part, and what it will and will not hold."""
    thread: str = "M10"
    material: str = "4130n"
    engagement_mm: float = 15.0
    percent_thread: float = 75.0
    bolt_class: str = "10.9"
    insert: bool = False              # a wire thread insert (Helicoil)

    @property
    def spec(self) -> Thread:
        return THREADS[self.thread]

    @property
    def mat(self):
        return MATERIAL_BY_KEY[self.material]

    @property
    def drill_mm(self) -> float:
        return tap_drill_mm(self.spec, self.percent_thread)

    # ------------------------------------------------------------------
    def thread_shear_area_m2(self) -> float:
        """The cylindrical area the female threads shear across.

        Machinery's Handbook / FED-STD-H28 internal-thread shear area,

            A_n = pi n L_e D_s [ 1/(2n) + 0.57735 (D_s - E_n) ],

        n threads per unit length, D_s the external thread's major
        diameter, E_n the internal thread's pitch diameter.  For the ISO
        basic profile (E_n = D - 0.6495 p) the bracket is 0.875 p, so the
        area is 0.875 pi D L_e -- the 0.5 pi D L_e shortcut understates
        pull-out by 43%.  Scaled by how much thread was actually cut
        (engineering approximation: the basic formula assumes full form)."""
        t = self.spec
        n = 1.0 / t.pitch_mm                            # threads per mm
        D_s, E_n = t.major_mm, t.pitch_diameter_mm
        area_mm2 = (math.pi * n * self.engagement_mm * D_s
                    * (1.0 / (2.0 * n) + 0.57735 * (D_s - E_n)))
        return area_mm2 * 1e-6 * (self.percent_thread / 100.0)

    def pull_out_n(self) -> float:
        """Force to strip the threads out of the PART."""
        shear = self.mat.yield_pa * 0.577
        if self.insert:
            # A WIRE INSERT IS STRONGER THAN THE PARENT THREAD, which
            # sounds like cheating and is not: the coil spreads load over
            # a larger diameter and more turns, so soft materials gain
            # the most. It is why aluminium castings are built with them
            # from new rather than only repaired with them.
            shear *= 1.0 + 0.9 * (1.0 - min(1.0, self.mat.yield_pa / 460e6))
            shear *= 1.35
        return self.thread_shear_area_m2() * shear

    def bolt_break_n(self) -> float:
        """Force to break the BOLT."""
        return self.spec.tensile_stress_area_m2 * BOLT_CLASS[self.bolt_class]

    def minimum_engagement_mm(self) -> float:
        """How deep before the bolt becomes the weaker element.

        This reproduces the rules of thumb rather than restating them:
        about one diameter into steel, about two into aluminium, and the
        reason is entirely the female material's shear strength."""
        shear = self.mat.yield_pa * 0.577
        if self.insert:
            shear *= 1.35
        per_mm = (0.5 * math.pi * (self.spec.major_mm / 1000.0)
                  * 0.001 * (self.percent_thread / 100.0) * shear)
        return self.bolt_break_n() / max(per_mm, 1e-9)

    def bolt_breaks_first(self) -> bool:
        """The condition you actually want to be true."""
        return self.bolt_break_n() <= self.pull_out_n()

    # ------------------------------------------------------------------
    def tapping_torque_nm(self) -> float:
        """What it takes to cut the thread.

        Rises with the square of diameter, with depth, and steeply with
        percent thread -- which is the term the tap drill controls."""
        t = self.spec
        d = t.major_mm / 1000.0
        depth_ratio = (self.engagement_mm / 1000.0) / max(d, 1e-9)
        shear = self.mat.yield_pa * 0.577
        return (TAPPING_TORQUE_COEFFICIENT * shear * d * d * depth_ratio
                * (self.percent_thread / 75.0) ** 1.7)

    def tap_breaking_torque_nm(self) -> float:
        """What the tap itself will take before it snaps.

        Falls with the CUBE of diameter, which is why small taps are
        dangerous and large ones are not."""
        core = self.spec.major_mm / 1000.0 * TAP_CORE_FRACTION
        return math.pi * core ** 3 / 16.0 * TAP_SHEAR_PA

    def tap_margin(self) -> float:
        """How much room there is between cutting and snapping."""
        return self.tap_breaking_torque_nm() / max(self.tapping_torque_nm(), 1e-9)

    # ------------------------------------------------------------------
    def report(self) -> list:
        s = self.spec
        out = [f"  {self.thread} x {s.pitch_mm} in {self.mat.label}, "
               f"{self.engagement_mm:.0f} mm deep at {self.percent_thread:.0f}% thread",
               f"    tap drill {self.drill_mm:5.2f} mm"
               + ("   (wire thread insert fitted)" if self.insert else ""),
               f"    strips at {self.pull_out_n() / 1000:7.1f} kN   "
               f"bolt {self.bolt_class} breaks at {self.bolt_break_n() / 1000:7.1f} kN"
               f"   -> {'BOLT breaks first (good)' if self.bolt_breaks_first() else 'THREADS STRIP FIRST -- the part is lost'}",
               f"    minimum engagement for that: {self.minimum_engagement_mm():5.1f} mm "
               f"= {self.minimum_engagement_mm() / s.major_mm:.2f} x diameter"]
        margin = self.tap_margin()
        risk = ("safe" if margin > 3.0 else "tight" if margin > 1.6
                else "MARGINAL" if margin > 1.0 else "THE TAP WILL BREAK")
        out.append(f"    tapping {self.tapping_torque_nm():6.2f} N.m against a tap that "
                   f"snaps at {self.tap_breaking_torque_nm():6.2f} N.m  -> {margin:4.1f}x, {risk}")
        return out


# =====================================================================
#  MACHINED CLAMPS
# =====================================================================
#: Dry friction coefficients for clamped steel joints. Design practice
#: uses the low end, because a clamp that slips has lost nothing but a
#: clamp that was sized on optimism has lost the setting.
CLAMP_FRICTION = {
    "steel-on-steel": 0.15,
    "steel-on-aluminium": 0.20,
    "knurled-or-serrated": 0.35,
    "oily": 0.08,
}

#: Nut factor relating bolt torque to preload: F = T / (K d). 0.2 dry.
NUT_FACTOR = 0.20


@dataclass
class MachinedClamp:
    """A split clamp machined to a bore, closed by bolts.

    THE PUREST FORM OF THE REVERSIBLE JOINT, and the reason to treat it
    as its own class rather than a bolted joint with extra steps: it
    transmits by FRICTION over a machined interface rather than by form
    or by fusion. Nothing melts, nothing cures, nothing is consumed, and
    it can be loosened, slid along, re-aimed and retightened -- which no
    weld, braze, rivet or bond can do at all.

    AND IT FAILS BENIGNLY. Every other joint here fails by fracture: the
    weld cracks, the thread strips, the rivet shears, and the part is
    damaged. A clamp SLIPS. The setting is lost, nothing is broken, and
    tightening it again restores it. For anything that can be re-datumed
    -- a shaft collar, a yoke, a tube joint, a sight mount -- that is a
    different category of failure rather than a milder one.

    THE PRICE IS PRECISION. Friction needs contact, and contact needs the
    bore to actually fit. A clamp machined loose bears on two lines
    instead of an arc, and its capacity falls with the contact it really
    has, not the contact it was drawn with. This is the one joint in the
    set whose strength depends on how well it was MADE rather than on
    what it was made of.
    """
    bore_diameter_m: float = 0.040
    width_m: float = 0.020
    bolts: int = 2
    bolt: str = "M8"
    bolt_class: str = "10.9"
    preload_fraction: float = 0.70       # of proof load, standard practice
    contact: str = "steel-on-steel"
    #: how much of the bore actually touches. A properly machined clamp
    #: is near 1.0; a bored-out-of-round one is far less.
    contact_fraction: float = 0.95
    #: two-piece clamps pull from both sides; a single split pulls one
    two_piece: bool = True

    @property
    def spec(self) -> Thread:
        return THREADS[self.bolt]

    def bolt_preload_n(self) -> float:
        """What each bolt actually pulls with."""
        proof = self.spec.tensile_stress_area_m2 * BOLT_CLASS[self.bolt_class] * 0.9
        return proof * self.preload_fraction

    def bolt_torque_nm(self) -> float:
        """What the wrench reads to get there. F = T / (K d)."""
        return self.bolt_preload_n() * NUT_FACTOR * self.spec.major_mm / 1000.0

    def normal_force_n(self) -> float:
        """The radial load pressing the bore onto the shaft.

        A two-piece clamp closes from both sides, so both bolt groups
        contribute; a single-split clamp hinges on its uncut side and
        gets roughly half the benefit."""
        pull = self.bolts * self.bolt_preload_n()
        return pull * (2.0 if self.two_piece else 1.0) * self.contact_fraction

    def torque_capacity_nm(self) -> float:
        """Torque it holds before it SLIPS. T = mu N r."""
        mu = CLAMP_FRICTION[self.contact]
        return mu * self.normal_force_n() * self.bore_diameter_m / 2.0

    def axial_capacity_n(self) -> float:
        """And how hard it resists being pushed along the shaft."""
        return CLAMP_FRICTION[self.contact] * self.normal_force_n()

    def report(self) -> list:
        return [
            f"  {self.bore_diameter_m * 1000:.0f} mm bore, {self.bolts} x {self.bolt} "
            f"{self.bolt_class}{'  two-piece' if self.two_piece else '  single split'}, "
            f"{self.contact}",
            f"    each bolt {self.bolt_preload_n() / 1000:5.1f} kN at "
            f"{self.bolt_torque_nm():5.1f} N.m on the wrench",
            f"    contact {self.contact_fraction * 100:.0f}% -> normal "
            f"{self.normal_force_n() / 1000:6.1f} kN",
            f"    holds {self.torque_capacity_nm():7.1f} N.m of torque, "
            f"{self.axial_capacity_n() / 1000:6.1f} kN axially, then SLIPS "
            f"(nothing breaks)",
        ]
