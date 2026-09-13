"""The assembly rig, the datum, and what a repair locks in.

THE VALIDATOR'S ASSEMBLY STAGE IS THE DATUM. Parts are clamped at their
authored positions, the machine loads them the same way every time, and
the weld is made against that. Everything instantiated from it starts at
one hundred per cent: rest lengths exactly as drawn, no residual stress,
weld quality the rig's own. That is not a convenience, it is what a
fixture IS -- a way of making the same part twice.

EVERYTHING FALLS OFF THAT STANDARD, and the interesting part is HOW.

A weld does not join two parts where they ought to be. It joins them
WHERE THEY ARE WHEN IT SOLIDIFIES. Weld metal contracts one to two per
cent as it cools, and what that contraction does depends entirely on
what is holding the work:

  HELD RIGIDLY, the part cannot move, so the shrinkage has nowhere to go
  and becomes RESIDUAL STRESS. As-welded residual stress in a restrained
  joint typically reaches the weld metal's own yield -- the part is the
  right shape and is already carrying a large load before anything is
  put on it.

  HELD LOOSELY OR NOT AT ALL, the part relieves itself by MOVING. The
  stress is small and the geometry is wrong: the member's natural rest
  length is now whatever length it happened to be at, and it will hold
  the structure at that shape forever.

You may have the right geometry or a stress-free part. Not both. That is
the whole reason a fixture is worth building, and it is why a field
repair on a sagging mount is never the same part again even when the
weld itself is perfect.

AND THE ERRORS ACCUMULATE. Each repair locks in the deflection that was
present at that moment. Repair a mount while it is elevated and it is
wrong when stowed; repair it while it is stowed and it is wrong when
elevated; repair it twice and it is wrong in two ways that do not cancel.
The structure drifts from its drawing one repair at a time, and nothing
short of going back to the rig resets it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


#: Linear coefficients of thermal expansion, per kelvin. Real values,
#: and the reason a brazed joint between two different metals is not the
#: same joint as one between two of the same.
CTE_PER_K = {
    "4130n": 11.7e-6, "4340qt": 12.3e-6, "a36": 11.7e-6,
    "hy80": 11.9e-6, "hy100": 11.9e-6, "rha": 11.7e-6, "300m": 11.5e-6,
    "6061t6": 23.1e-6, "7075t6": 23.6e-6, "ti64": 8.6e-6,
    "bronze": 18.0e-6, "copper": 16.5e-6, "stainless": 17.3e-6,
}


@dataclass(frozen=True)
class BondMechanism:
    """HOW a joint is made, and therefore what it locks in.

    A SINGLE SHRINKAGE CONSTANT WAS WRONG, and brazing is the clearest
    case. These are not degrees of the same thing; they are different
    mechanisms that happen to end with two parts stuck together.

    FUSION WELDING melts the parent. A substantial bead of filler
    solidifies from about 1500 C and contracts as it freezes and cools,
    and because it is fused to the parent it drags the parent with it.
    That deposit contraction is large, local, and the origin of both
    weld distortion and as-welded residual stress.

    BRAZING DOES NOT MELT THE PARENT. Filler is drawn into a capillary
    gap a tenth of a millimetre wide by wetting, and freezes at 620-760 C
    while the tubes stay solid throughout. There is almost no deposit to
    shrink -- so a brazed joint's locked-in strain comes from somewhere
    else entirely: DIFFERENTIAL THERMAL CONTRACTION as the whole
    assembly cools from brazing temperature. Two members of the same
    alloy contract together and the joint is very nearly stress-free,
    which is precisely why brazing is chosen for thin-wall assemblies
    that fusion welding would distort. Two DIFFERENT alloys contract by
    different amounts and the difference is locked in permanently.

    ADHESIVE BONDING cures rather than freezes. Epoxy shrinks one to
    three per cent by volume as it cross-links, which is a few tenths of
    a per cent linear, and it happens at room temperature so there is no
    thermal term at all.

    MECHANICAL JOINTS have no solidification and no cure. A bolted joint's
    preload is chosen and applied, not inherited. And a HOT-DRIVEN RIVET
    inverts the whole idea: it is placed at red heat precisely so that
    its contraction clamps the joint, so here the shrinkage is not an
    error to be managed but the entire point of the process.
    """
    key: str
    label: str
    parent_melts: bool
    #: linear contraction of the deposit as it solidifies, if there is a
    #: bulk deposit at all
    deposit_shrinkage: float
    #: temperature drop from process to ambient, which drives the
    #: differential-contraction term
    delta_t_k: float
    #: a capillary film is too thin for its own shrinkage to matter
    capillary: bool = False
    #: cure shrinkage, for anything that sets rather than freezes
    cure_shrinkage: float = 0.0
    #: whether the contraction is the POINT rather than a side effect
    deliberate: bool = False
    note: str = ""


BOND_MECHANISMS = {
    "fusion-weld": BondMechanism(
        "fusion-weld", "fusion weld", True, 0.012, 1480.0,
        note="the parent melts and a real bead freezes onto it; the "
             "deposit's own contraction dominates everything else"),
    "braze": BondMechanism(
        "braze", "capillary braze", False, 0.0, 660.0, capillary=True,
        note="parent stays solid, filler is a film. Nearly stress-free "
             "between like metals; between unlike ones the difference in "
             "contraction is the whole of the residual strain"),
    "bronze-braze": BondMechanism(
        "bronze-braze", "bronze braze", False, 0.0, 870.0, capillary=True,
        note="hotter than silver, so a larger temperature drop and more "
             "differential contraction to lock in"),
    "solder": BondMechanism(
        "solder", "soft solder", False, 0.0, 190.0, capillary=True,
        note="barely warm by comparison: almost nothing is locked in, and "
             "almost nothing is carried either"),
    "adhesive": BondMechanism(
        "adhesive", "structural adhesive", False, 0.0, 0.0,
        cure_shrinkage=0.006,
        note="cures rather than freezes: a few tenths of a per cent of "
             "cure shrinkage and no thermal term at all"),
    "bolted": BondMechanism(
        "bolted", "bolted joint", False, 0.0, 0.0,
        note="nothing is inherited. The preload is a number someone "
             "chose and applied with a wrench"),
    "hot-rivet": BondMechanism(
        "hot-rivet", "hot-driven rivet", False, 0.009, 900.0, deliberate=True,
        note="driven at red heat SO THAT it contracts: the shrinkage is "
             "the clamping force and the reason the joint works"),
}


def locked_strain(mechanism: str, *, restraint: float,
                  parent_alloy: str = "4130n", other_alloy: str | None = None,
                  stress_relieved: bool = False) -> dict:
    """What this joining process leaves behind, and where it came from.

    Returns the two contributions separately because they are physically
    different and respond differently: deposit contraction is what a
    stress-relief anneal removes, while differential contraction between
    unlike metals is geometric and comes back every time the joint
    changes temperature.
    """
    m = BOND_MECHANISMS[mechanism]
    r = max(0.0, min(1.0, restraint))

    # --- the deposit's own contraction -------------------------------
    deposit = 0.0 if m.capillary else r * (m.deposit_shrinkage + m.cure_shrinkage)
    if stress_relieved:
        deposit *= 0.16

    # --- differential contraction between the two members ------------
    # ZERO FOR LIKE METALS. This is the term that makes brazing sensible
    # on a same-alloy assembly and treacherous on a mixed one.
    a = CTE_PER_K.get(parent_alloy, 11.7e-6)
    b = CTE_PER_K.get(other_alloy or parent_alloy, a)
    differential = r * abs(a - b) * m.delta_t_k

    return {
        "mechanism": m.key,
        "deposit_strain": deposit,
        "differential_strain": differential,
        "total_strain": deposit + differential,
        "deliberate": m.deliberate,
        "relievable": deposit > 0.0,
        "dissimilar": (other_alloy or parent_alloy) != parent_alloy,
    }


#: Kept for callers that want the fusion-weld figure by name.
WELD_SHRINKAGE = BOND_MECHANISMS["fusion-weld"].deposit_shrinkage


# =====================================================================
#  WHAT THE DRAWING MEANS
# =====================================================================
#: A structure sags under its own weight, so the coordinates on a
#: drawing cannot describe both the unloaded part and the working one.
#: Somebody has to say which -- and rather than decide it here, it is a
#: setting, because the two are genuinely different engineering choices
#: with different consequences and the only way to know which you wanted
#: is usually to have built one of each.
DATUM_CONVENTIONS = {
    "unloaded": (
        "the drawing is the shape with NO load on it. Simple to fixture "
        "and simple to inspect -- you can measure the part on a bench and "
        "it agrees with the paper. The assembled machine then sits BELOW "
        "its drawing by however much it sags, and the aim datum moves "
        "again every time the load changes."),
    "in-service": (
        "the drawing is the shape UNDER LOAD. Rest lengths are made "
        "shorter than the drawing -- pre-cambered -- so the structure "
        "settles onto the drawing when it is carrying what it is meant "
        "to. Right where it matters, and visibly wrong on the bench, "
        "which makes inspection an argument. And the camber is correct "
        "for ONE load case: camber a mount for a 40 mm tube, hang a "
        "76 mm one on it, and it is now wrong in the other direction by "
        "more than it was before."),
    "as-built": (
        "no intent at all: the part is whatever it was when it was "
        "joined. This is what a field repair produces and it is not a "
        "convention so much as the absence of one."),
}


@dataclass
class Datum:
    """Which shape the drawing describes, and for what load.

    `sag` is supplied by whoever knows -- an equilibrium solve, a
    measurement, a survey. It is NOT invented here: with no solve and no
    measurement the camber is zero and this says so rather than quietly
    guessing a deflection.
    """
    convention: str = "unloaded"
    #: the load case the camber was cut for, if any. A structure used
    #: under a different one is cambered wrong, and knowing which one it
    #: was is the only way to notice.
    cambered_for: str = ""
    #: member identity -> how far that member's span sags under the
    #: nominal load, in metres. Empty until something measures it.
    sag_m: dict = field(default_factory=dict)

    @property
    def note(self) -> str:
        return DATUM_CONVENTIONS[self.convention]

    def camber_for(self, member: str) -> float:
        """How much shorter than the drawing this member is made.

        Negative lengthens, positive shortens. Only the in-service
        convention cambers at all; the others build to the drawing and
        accept whatever the load then does."""
        if self.convention != "in-service":
            return 0.0
        return float(self.sag_m.get(member, 0.0))

    def rest_length_for(self, member: str, drawing_length_m: float) -> float:
        """The length the part is actually MADE to."""
        return drawing_length_m - self.camber_for(member)

    def mismatch(self, load_case: str) -> str:
        """Whether this structure is being used as it was built for."""
        if self.convention != "in-service" or not self.cambered_for:
            return ""
        if load_case == self.cambered_for:
            return ""
        return (f"cambered for '{self.cambered_for}' and being used under "
                f"'{load_case}': the correction is now pointing the wrong way")

    def describe(self) -> list:
        out = [f"  datum: {self.convention}"
               + (f", cambered for '{self.cambered_for}'" if self.cambered_for else ""),
               f"    {self.note}"]
        if self.convention == "in-service" and not self.sag_m:
            out.append("    NO SAG SUPPLIED, so the camber is zero and this is "
                       "presently identical to building unloaded. It needs an "
                       "equilibrium solve or a measurement to mean anything.")
        elif self.sag_m:
            worst = max(self.sag_m.items(), key=lambda kv: abs(kv[1]))
            out.append(f"    {len(self.sag_m)} member(s) cambered, worst "
                       f"{worst[0]} by {worst[1] * 1000:+.2f} mm")
        return out


@dataclass
class AssemblyRig:
    """The validator's fixture: where parts are held while worked on.

    `restraint` is the whole character of the rig. One is a proper
    fixture -- clamps at every node, the work forced to datum. Zero is
    two trestles and hope."""
    identity: str = "validator.assembly"
    restraint: float = 1.0
    #: WHICH SHAPE THE DRAWING DESCRIBES. Configurable on purpose: the
    #: two conventions are real alternatives, and which one someone
    #: wanted is usually discovered by building it the other way.
    datum: object = None
    #: the load the machine applies while welding, as it did when the
    #: part was first printed. Repeating it is what makes a repair
    #: return to datum instead of to some other shape.
    load_case: str = "as-printed"
    #: what the rig can achieve, because a fixture is also a clean,
    #: still, well-lit place with the right consumables in reach
    weld_process: str = "tig"
    weld_filler: str = "er80sd2"
    operator_skill: float = 0.95
    preparation: float = 0.97
    shielding: float = 0.97
    post_weld_heat_treated: bool = True
    inspection: str = "radiographic"

    def __post_init__(self) -> None:
        if self.datum is None:
            self.datum = Datum(convention="unloaded")

    def quality(self, parent_alloy: str = "4130n"):
        from welding import WeldQuality
        return WeldQuality(process=self.weld_process, filler=self.weld_filler,
                           parent_alloy=parent_alloy, preparation=self.preparation,
                           shielding=self.shielding, operator_skill=self.operator_skill,
                           post_weld_heat_treated=self.post_weld_heat_treated,
                           inspection=self.inspection)

    def describe(self) -> list:
        return self.datum.describe() + [
                f"  {self.identity}: restraint {self.restraint * 100:.0f}%, "
                f"load case '{self.load_case}'",
                f"    {self.weld_process.upper()} / {self.weld_filler}, "
                f"skill {self.operator_skill * 100:.0f}%, "
                f"{'heat treated' if self.post_weld_heat_treated else 'NOT heat treated'}, "
                f"{self.inspection} inspection"]


#: What you get in the field, by contrast: whatever is holding it up.
FIELD_TRESTLES = AssemblyRig(
    identity="field.trestles", restraint=0.25, load_case="as-it-sits",
    weld_process="improvised", weld_filler="er70s2", operator_skill=0.45,
    preparation=0.25, shielding=0.35, post_weld_heat_treated=False,
    inspection="visual")

#: A workshop that is not the rig: better than a berm, not the datum.
FIELD_WORKSHOP = AssemblyRig(
    identity="field.workshop", restraint=0.70, load_case="partly-supported",
    weld_process="mig", weld_filler="er80sd2", operator_skill=0.80,
    preparation=0.75, shielding=0.80, post_weld_heat_treated=False,
    inspection="visual")



@dataclass
class WeldEvent:
    """One joint, made once, at a particular moment in the structure's
    life. What it locked in is permanent until it is cut out again."""
    member: str
    datum_length_m: float
    length_at_weld_m: float
    restraint: float
    mechanism: str = "fusion-weld"
    parent_alloy: str = "4130n"
    #: the OTHER member's alloy. Different from the parent means a
    #: brazed or soldered joint carries differential contraction that a
    #: same-alloy one does not, and no anneal will take it out.
    other_alloy: str | None = None
    youngs_pa: float = 205e9
    weld_yield_pa: float = 690e6
    rig: str = "validator.assembly"
    #: WHAT POST-WELD HEAT TREATMENT IS ACTUALLY FOR. Without it the rig
    #: looks pointless: perfect restraint and perfect fit-up STILL leave
    #: the joint at yield, because the weld's own shrinkage has nowhere
    #: to go. A stress-relief anneal takes residual stress down to
    #: roughly a sixth while leaving the geometry alone -- which is the
    #: only way out of the geometry-or-stress trade, and the reason it
    #: is worth an oven and six hours.
    stress_relieved: bool = False

    @property
    def fit_up_error_m(self) -> float:
        """How far from datum the part was when it was joined."""
        return self.length_at_weld_m - self.datum_length_m

    @property
    def new_rest_length_m(self) -> float:
        """THE MEMBER'S NEW NATURAL LENGTH.

        Full restraint pulls it back to datum, so the rest length is the
        drawing's. No restraint leaves it wherever it was, so the rest
        length becomes that. In between, it lands in between -- and that
        is the number the structure will hold forever after."""
        return (self.length_at_weld_m
                + self.restraint * (self.datum_length_m - self.length_at_weld_m))

    @property
    def geometry_error_m(self) -> float:
        """How wrong the part's shape now is, permanently."""
        return self.new_rest_length_m - self.datum_length_m

    @property
    def residual_strain(self) -> float:
        """The strain locked into the metal as it cooled.

        Two sources, and restraint decides how much of each: the weld's
        own shrinkage, and whatever fit-up error the clamps forced out of
        the part. Unrestrained, both relieve themselves as movement;
        restrained, both stay in as stress."""
        forced = -self.restraint * self.fit_up_error_m / max(self.datum_length_m, 1e-9)
        bond = locked_strain(self.mechanism, restraint=self.restraint,
                             parent_alloy=self.parent_alloy,
                             other_alloy=self.other_alloy,
                             stress_relieved=self.stress_relieved)
        return forced + bond["total_strain"]

    @property
    def bond(self) -> dict:
        """Where the locked-in strain came from, split by mechanism.

        THE STRAIN IS THE ANSWER, not a stress. An earlier version
        computed `residual_stress_pa` here and capped it at yield, which
        is reimplementing plasticity beside the plasticity law: the
        strain goes into the member's rest length, the solver finds the
        stress, and the return map relieves whatever exceeds yield. This
        reports the provenance and nothing else."""
        return locked_strain(self.mechanism, restraint=self.restraint,
                             parent_alloy=self.parent_alloy,
                             other_alloy=self.other_alloy,
                             stress_relieved=self.stress_relieved)

    def describe(self) -> list:
        return [
            f"  {self.member} joined by {self.mechanism} in {self.rig} at {self.restraint * 100:.0f}% restraint",
            f"    fit-up was {self.fit_up_error_m * 1000:+7.2f} mm off datum",
            f"    permanent geometry error {self.geometry_error_m * 1000:+7.2f} mm",
            f"    locked-in strain {self.residual_strain:+.5f}  "
            f"(deposit {self.bond['deposit_strain']:+.5f}, "
            f"differential {self.bond['differential_strain']:+.5f})",
        ]


@dataclass
class BuildHistory:
    """Every weld a structure has ever had, and what they add up to.

    The point of keeping it: errors do not cancel. A mount repaired
    elevated and then repaired stowed is wrong in two ways at once, and
    only going back to the rig clears either."""
    events: list = field(default_factory=list)

    def weld(self, member: str, *, datum_length_m: float, length_at_weld_m: float,
             rig: AssemblyRig, youngs_pa: float = 205e9,
             weld_yield_pa: float = 690e6) -> WeldEvent:
        ev = WeldEvent(member=member, datum_length_m=datum_length_m,
                       length_at_weld_m=length_at_weld_m, restraint=rig.restraint,
                       youngs_pa=youngs_pa, weld_yield_pa=weld_yield_pa, rig=rig.identity,
                       stress_relieved=rig.post_weld_heat_treated)
        self.events.append(ev)
        return ev

    def current(self, member: str) -> "WeldEvent | None":
        """The last time this member was joined is the one that counts."""
        for ev in reversed(self.events):
            if ev.member == member:
                return ev
        return None

    def rest_length_of(self, member: str, datum_length_m: float) -> float:
        ev = self.current(member)
        return datum_length_m if ev is None else ev.new_rest_length_m

    def at_datum(self) -> bool:
        """True only if every weld was made in a full-restraint rig."""
        return all(ev.restraint >= 0.999 for ev in self.events)

    def summary(self) -> list:
        if not self.events:
            return ["  never repaired: every member at its authored length"]
        worst = max(self.events, key=lambda e: abs(e.geometry_error_m))
        drift = sum(abs(e.geometry_error_m) for e in self.events)
        worst_strain = max(abs(e.residual_strain) for e in self.events)
        dissimilar = sum(1 for e in self.events if e.bond["dissimilar"])
        return [
            f"  {len(self.events)} weld(s); "
            f"{'still at datum' if self.at_datum() else 'OFF DATUM'}",
            f"    accumulated geometry drift {drift * 1000:6.2f} mm across all members",
            f"    worst single member {worst.member} at "
            f"{worst.geometry_error_m * 1000:+.2f} mm",
            f"    highest locked-in strain {worst_strain:+.5f} "
            f"(the solver turns that into stress; the return map relieves "
            f"what exceeds yield)",
        ] + ([f"    {dissimilar} joint(s) between UNLIKE metals: their "
              f"differential contraction returns with every temperature "
              f"change and no anneal removes it"] if dissimilar else [])
