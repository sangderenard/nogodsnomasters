"""Weld quality, and what bad practice actually costs you.

A joint's capacity in `joints.py` is computed from its nominal geometry
-- throat, perimeter, engagement, pin diameter. That is what the joint
would carry IF IT WERE MADE PROPERLY. This module is the difference
between the drawing and the thing that came out of the field.

THE MECHANIC, AND WHY IT IS SHAPED THIS WAY. A bad weld looks like a
good weld. That is not a game contrivance, it is the central fact about
welding: lack of fusion is invisible from outside, porosity is invisible
from outside, and an unnormalised 4130 heat-affected zone looks exactly
like a normalised one. So a poor joint passes inspection, passes the
proof load, and then fails -- later, under cycling, at a fraction of the
load it was signed off for.

That delayed, load-history-dependent failure is the whole incentive
structure. Good practice costs time and consumables up front: cleaning,
purge gas, the correct filler, post-weld normalising, a competent hand.
Bad practice costs nothing today and takes the mount away at round three
hundred. You cannot test your way out of it with one shot, which is
exactly why real shops pay for procedure qualification instead.

NOTHING HERE IS A PENALTY OR A MULTIPLIER, and an earlier version of
this file was nothing but. It computed a strength factor, a stress
concentration and a fatigue life, and multiplied honest numbers by them
-- which is building a shadow of the physics beside the physics, at the
same cost as doing it properly and with none of the consequences that
make it worth doing.

A weld is a short member of filler metal. Give it its real section, its
real alloy and its real rest length, put it in the graph, and:

  POROSITY AND LACK OF FUSION are less metal, so the element's section
  is smaller. Strength follows without being told to.

  THE STRESS CONCENTRATION is that smaller section sitting in series
  with a full-size tube: the same force through less area is a higher
  stress, which is what a notch does, arrived at geometrically.

  THE RESIDUAL STRESS is a rest length shorter than the gap the weld
  spans, because the metal shrank as it cooled. The solver finds the
  tension itself -- and when that tension exceeds yield it relieves
  itself plastically, which is exactly what real residual stress does
  and exactly what a J2 return map already does.

  THE FATIGUE is accumulated plastic strain. The member law already
  returns `accumulated_plastic_strain_next` and `remaining_ductility_
  next`; cycle a thin, low-ductility element past its local yield and it
  wears out and fractures on its own. No S-N curve, no detail category.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WeldProcess:
    """How the metal was actually joined."""
    key: str
    label: str
    #: what fraction of the nominal throat a competent operator achieves
    throat_efficiency: float
    #: baseline porosity fraction with good preparation
    base_porosity: float
    #: how badly it degrades when preparation is poor
    contamination_sensitivity: float
    #: how exposed it is to wind stealing the shielding gas
    wind_sensitivity: float
    deposition_rate_kg_h: float
    note: str = ""


PROCESSES = {
    "tig": WeldProcess("tig", "GTAW / TIG", 0.98, 0.004, 0.9, 1.0, 0.6,
                       note="the cleanest and the slowest. Full control of "
                            "heat and filler, which is why thin-wall 4130 "
                            "and anything structural gets welded this way"),
    "mig": WeldProcess("mig", "GMAW / MIG", 0.94, 0.012, 1.2, 1.3, 2.8,
                       note="four times the deposition and more porosity "
                            "risk; shielding gas blows away in wind"),
    "flux-core": WeldProcess("flux-core", "FCAW self-shielded", 0.90, 0.020, 1.0, 0.3, 3.5,
                             note="its own shielding, so it works outdoors "
                                  "where MIG cannot -- at the cost of slag "
                                  "inclusions and a rougher profile"),
    "stick": WeldProcess("stick", "SMAW / stick", 0.92, 0.015, 0.8, 0.2, 1.4,
                         note="field-proof and forgiving of dirt and wind; "
                              "slag inclusion is the defect to fear"),
    "oxy": WeldProcess("oxy", "oxy-acetylene", 0.88, 0.010, 1.4, 0.8, 0.3,
                       note="the traditional way to join thin 4130 tube, and "
                            "still sound: slow, hot, a wide heat-affected "
                            "zone, and no electricity needed"),
    "improvised": WeldProcess("improvised", "improvised field weld", 0.62, 0.070, 2.2, 2.0, 1.0,
                              note="whatever was to hand, whatever gas was "
                                   "left, no cleaning and no preheat. It will "
                                   "hold the thing together and it will not "
                                   "hold it together for long"),
}


@dataclass(frozen=True)
class FillerMetal:
    """What was fed into the puddle, which is frequently the wrong thing."""
    key: str
    label: str
    tensile_pa: float
    suits: tuple                # parent alloys it is correct for
    hydrogen_risk: float        # 0 clean, 1 cellulose-in-the-rain
    note: str = ""


FILLERS = {
    "er70s2": FillerMetal("er70s2", "ER70S-2 mild steel", 480e6,
                          ("a36", "4130n"), 0.10,
                          note="correct for mild steel. On 4130 it is "
                               "UNDERMATCHED -- the weld is weaker than the "
                               "tube, which is a deliberate and acceptable "
                               "choice for a cage and a mistake in a load path"),
    "er80sd2": FillerMetal("er80sd2", "ER80S-D2 low-alloy", 690e6,
                           ("4130n", "4340qt"), 0.12,
                           note="matched to 4130: the standard choice for "
                                "chromoly structure"),
    "er312": FillerMetal("er312", "ER312 stainless", 760e6,
                         ("4130n", "4340qt", "hy80", "rha"), 0.05,
                         note="crack-tolerant on dissimilar and hardenable "
                              "steels: what you reach for when you cannot "
                              "post-weld heat treat"),
    "e11018": FillerMetal("e11018", "E11018-M low-hydrogen", 760e6,
                          ("hy80", "hy100", "rha", "4340qt"), 0.04,
                          note="armour and HY plate. Low-hydrogen, and it must "
                               "be kept in a rod oven or it is not"),
    "er4043": FillerMetal("er4043", "ER4043 aluminium", 165e6,
                          ("6061t6",), 0.02,
                          note="silicon filler for 6061: flows well, low crack "
                               "risk, and it does not respond to heat treat, so "
                               "the joint stays annealed-soft"),
    "er5356": FillerMetal("er5356", "ER5356 aluminium", 265e6,
                          ("6061t6",), 0.02,
                          note="magnesium filler: stronger than 4043 and more "
                               "crack-sensitive. Never on 7075, which does not "
                               "fusion weld at all"),
}


@dataclass
class WeldQuality:
    """How well this particular joint was actually made.

    Everything here is a CHOICE someone made in the field, and every one
    of them has a price in time or consumables and a payoff in whether
    the mount survives its own gun."""
    process: str = "tig"
    filler: str = "er80sd2"
    parent_alloy: str = "4130n"
    #: surface preparation: 1.0 ground bright, 0.0 oily mill scale
    preparation: float = 0.9
    #: shielding integrity: 1.0 still air or a purge tent, 0.0 in a gale
    shielding: float = 0.9
    #: the hand that made it: 1.0 coded and qualified, 0.0 first attempt
    operator_skill: float = 0.85
    #: was it normalised or stress-relieved afterwards?
    post_weld_heat_treated: bool = True
    #: did anyone actually look at it with anything?
    inspection: str = "visual"      # none | visual | dye-penetrant | radiographic

    # ------------------------------------------------------------------
    @property
    def proc(self) -> WeldProcess:
        return PROCESSES[self.process]

    @property
    def rod(self) -> FillerMetal:
        return FILLERS[self.filler]

    @property
    def filler_is_matched(self) -> bool:
        return self.parent_alloy in self.rod.suits

    # ------------------------------------------------------------------
    def porosity_fraction(self) -> float:
        """Gas left in the metal. Contamination and lost shielding are
        what put it there."""
        p = self.proc
        dirty = (1.0 - max(0.0, min(1.0, self.preparation))) * p.contamination_sensitivity
        exposed = (1.0 - max(0.0, min(1.0, self.shielding))) * p.wind_sensitivity
        unskilled = (1.0 - max(0.0, min(1.0, self.operator_skill))) * 0.5
        # CAPPED AT A QUARTER, because beyond that it is not a weld. The
        # first calibration let an improvised joint reach 60% voids and
        # 2.5% of rated strength -- which breaks the mechanic rather than
        # expressing it. A bad weld that cannot hold the part on at all
        # is never chosen, never used, and never teaches anything; the
        # failure that matters is the one you get to rely on first.
        return min(0.25, p.base_porosity * (1.0 + 2.2 * (dirty + exposed + unskilled)))

    def lack_of_fusion(self) -> float:
        """The dangerous defect: metal that never joined, invisible from
        outside, and a crack the day it was made."""
        skill = max(0.0, min(1.0, self.operator_skill))
        return min(0.45, (1.0 - skill) ** 1.6 * 0.55
                   + (1.0 - self.proc.throat_efficiency) * 0.4)

    def hydrogen_cracking_risk(self) -> float:
        """HY plate and 4130 crack in the heat-affected zone if hydrogen
        gets in and the joint is not treated. This is the failure that
        appears DAYS later, which is why it has its own term."""
        hardenable = self.parent_alloy in ("4130n", "4340qt", "hy80", "hy100", "rha", "300m")
        if not hardenable:
            return 0.0
        risk = self.rod.hydrogen_risk + (1.0 - self.preparation) * 0.35
        if not self.post_weld_heat_treated:
            risk += 0.40
        return min(1.0, risk)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  WHAT THE DEFECTS ACTUALLY ARE
    # ------------------------------------------------------------------
    # Not multipliers on a result. A weld with porosity has LESS METAL in
    # it, and a weld with lack of fusion has less metal STILL. That is a
    # smaller section, and a smaller section in series with a full-size
    # tube carries the same force at a higher stress all by itself --
    # which is the stress concentration, arrived at by being thin rather
    # than by a factor called Kt.
    #
    # Everything the earlier version computed as a coefficient is
    # obtained instead by handing the solver a truthful element:
    #
    #   strength          the element's own section and its own yield
    #   concentration     a thin link in a chain, geometrically
    #   residual stress   a rest length shorter than the gap it spans
    #   fatigue           accumulated plastic strain in the return map,
    #                     which the member law already returns
    #
    # None of those needed a model of their own. They needed the weld to
    # be in the graph.
    def sound_fraction(self) -> float:
        """How much of the nominal throat is actually metal."""
        lost = self.porosity_fraction() + self.lack_of_fusion()
        return max(0.05, self.proc.throat_efficiency * (1.0 - lost))

    def effective_area_m2(self, nominal_throat_area_m2: float) -> float:
        """The section that is really there to carry load."""
        return nominal_throat_area_m2 * self.sound_fraction()

    def weld_metal_yield_pa(self) -> float:
        """The FILLER'S strength, not the parent's -- which is the whole
        of what "wrong wire" means. An undermatched rod makes a weld
        weaker than the tube it joins no matter how well it is laid."""
        return self.rod.tensile_pa * 0.85

    def fracture_strain(self) -> float:
        """Ductility left in the joint. Hydrogen embrittlement and an
        untreated hardenable heat-affected zone both spend it, and a
        joint with none cracks instead of yielding."""
        base = 0.18
        base *= (1.0 - 0.75 * self.hydrogen_cracking_risk())
        base *= (1.0 - 0.45 * self.lack_of_fusion())
        return max(0.004, base)

    def detectable_by(self) -> str:
        """What inspection would ACTUALLY have found this.

        The point of the mechanic: visual inspection cannot see the
        defect that matters. Porosity and lack of fusion are internal,
        and the joint looks finished either way."""
        if self.lack_of_fusion() > 0.05 or self.porosity_fraction() > 0.05:
            return "radiographic"
        if self.hydrogen_cracking_risk() > 0.4:
            return "dye-penetrant (once it has cracked, days later)"
        return "visual"

    def hidden(self) -> bool:
        """True when the chosen inspection would not catch what is wrong."""
        order = {"none": 0, "visual": 1, "dye-penetrant": 2, "radiographic": 3}
        needed = self.detectable_by().split()[0]
        return order.get(self.inspection, 0) < order.get(needed, 3)

    # ------------------------------------------------------------------
    def describe(self) -> list:
        out = [f"  {self.proc.label} with {self.rod.label} on {self.parent_alloy}"
               f"{'' if self.filler_is_matched else '  -- FILLER NOT MATCHED'}",
               f"    porosity {self.porosity_fraction() * 100:5.1f}%   "
               f"lack of fusion {self.lack_of_fusion() * 100:5.1f}%   "
               f"-> {self.sound_fraction() * 100:5.1f}% of the throat is metal",
               f"    weld metal yields at {self.weld_metal_yield_pa() / 1e6:.0f} MPa, "
               f"ductility {self.fracture_strain() * 100:.1f}%"]
        if self.hydrogen_cracking_risk() > 0.3:
            out.append(f"    HYDROGEN CRACKING RISK {self.hydrogen_cracking_risk() * 100:.0f}%"
                       f"{' -- and it was never heat treated' if not self.post_weld_heat_treated else ''}")
        if self.hidden():
            out.append(f"    {self.inspection} inspection WILL NOT FIND THIS; "
                       f"needs {self.detectable_by()}")
        return out


def weld_edge(identity: str, a: str, b: str, *, quality: "WeldQuality",
              throat_m: float, perimeter_m: float, gap_m: float,
              restraint: float = 1.0, stress_relieved: bool = False) -> dict:
    """A weld, as a MEMBER in the production graph.

    This is the whole point of the module. The weld is not a property of
    the joint it sits in and not a factor applied to that joint's rating
    -- it is a short element of filler metal spanning the gap, with:

      ITS OWN SECTION, which is the sound throat area: the nominal throat
      less whatever porosity and lack of fusion took out of it. Thinner
      than the tube on either side, so the same load runs through it at a
      higher stress, which is the stress concentration without anyone
      having to name one.

      ITS OWN ALLOY, the filler's, not the parent's. Undermatched wire
      makes a weak element and the solve finds out.

      ITS OWN REST LENGTH, shorter than the gap it spans because the
      metal contracted as it cooled. That difference IS the residual
      stress, and the solver computes it the same way it computes every
      other strain. When it exceeds yield the return map relieves it
      plastically, exactly as a real weld does.

      ITS OWN DUCTILITY, spent by hydrogen and by an untreated
      heat-affected zone -- so a bad joint has little left to give and
      cracks where a good one would yield and survive.

    Cycle the structure and the member law accumulates plastic strain in
    this element and depletes its remaining ductility until it fractures.
    That is fatigue, and nothing here had to model it.
    """
    from assembly import WELD_SHRINKAGE
    throat_area = throat_m * perimeter_m
    area = quality.effective_area_m2(throat_area)
    yield_pa = quality.weld_metal_yield_pa()
    # THE SHRINKAGE GOES IN AS A REST LENGTH, and restraint decides how
    # much of it the part is forced to keep rather than relieve by
    # moving. A stress-relief anneal lets the metal creep most of it out
    # without changing the geometry, which is what the oven is for.
    locked = restraint * WELD_SHRINKAGE * (0.16 if stress_relieved else 1.0)
    rest_length = max(gap_m * (1.0 - locked), gap_m * 0.5)
    return {
        "identity": identity, "a": a, "b": b,
        "constraint": "welded-joint-element",
        "radius": math.sqrt(max(area, 1e-9) / math.pi),
        "rest_length": rest_length,
        "material": quality.filler,
        "in_view": True,
        "weld_process": quality.process,
        "weld_filler": quality.filler,
        "sound_fraction": quality.sound_fraction(),
        "porosity_fraction": quality.porosity_fraction(),
        "lack_of_fusion": quality.lack_of_fusion(),
        "inspection_would_miss_it": quality.hidden(),
        "damage": {
            "model": "elastic-plastic-member-with-shear-fracture",
            # the gap it spans, NOT its own shortened natural length:
            # the difference between these two is the residual stress
            "natural_rest_length": rest_length,
            "geometric_span_m": gap_m,
            "section_area_m2": area,
            "youngs_modulus_pa": 205e9,
            "shear_modulus_pa": 79e9,
            "yield_strength_pa": yield_pa,
            "ultimate_strength_pa": yield_pa * 1.18,
            "fracture_strain": quality.fracture_strain(),
            "plastic_strain_limit": 0.0025,
            "axial_yield_force_n": area * yield_pa,
            "shear_force_limit_n": area * yield_pa * 0.577,
            "failure_response": "constraint-opens-and-load-path-is-removed",
            "respawn_response": "restore-authored-natural-length-and-health",
        },
    }


def shop_weld(parent_alloy: str = "4130n") -> WeldQuality:
    """What a competent shop with time and a procedure produces."""
    return WeldQuality(process="tig", filler="er80sd2", parent_alloy=parent_alloy,
                       preparation=0.97, shielding=0.97, operator_skill=0.95,
                       post_weld_heat_treated=True, inspection="radiographic")


def field_repair(parent_alloy: str = "4130n") -> WeldQuality:
    """What you get behind a berm with whatever is in the truck."""
    return WeldQuality(process="improvised", filler="er70s2", parent_alloy=parent_alloy,
                       preparation=0.25, shielding=0.35, operator_skill=0.45,
                       post_weld_heat_treated=False, inspection="visual")
