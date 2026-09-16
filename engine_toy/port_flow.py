"""What a head can actually breathe, and what the tract in front of it
costs.

This exists because the engine's intake flow ceiling used to be:

    flow_capacity_kg_s = displacement * redline / 120 * 1.2

which is not a restriction at all -- it is a restatement of the engine's
own demand at redline. It says "the ports can flow exactly what the
engine wants, at the speed the engine was rated to", by construction. No
valve size, no port throat, no throttle bore, no filter, no manifold. A
cylinder head with valves the size of coins would flow exactly as well
as a race head, and porting an engine would do nothing.

WHAT ACTUALLY LIMITS BREATHING is gas speed through the valve. The
piston sweeps its bore area; all of that has to pass through a much
smaller hole, so the gas moves faster than the piston by the ratio of
those areas -- typically eight to one. Once that gas speed approaches
about half the speed of sound the port stops being able to fill the
cylinder in the time available, and volumetric efficiency falls away
sharply no matter how much throttle is open.

That ratio is the whole of head design:

    v_gas = v_piston * (bore^2) / (n_valves * throat^2)

Everything an engine builder does -- bigger valves, more of them, a
shorter stroke, a straighter port -- is an attack on one of those terms.
It also explains why a four-valve head is worth the complexity: two
smaller valves fit more curtain and more throat into the same bore than
one big one can, and the gain is around a fifth.

TWO DIFFERENT THINGS RESTRICT AT DIFFERENT LIFTS. Near the seat the gas
squeezes through a CURTAIN -- a cylindrical slot of circumference pi.D
and height equal to the lift. That grows as lift grows. Past about a
quarter of valve diameter the curtain is bigger than the port THROAT
behind it, and the throat takes over and never gets any bigger. Which is
why real cams lift to roughly L/D = 0.25 to 0.30 and no further: past
that the lobe is buying valvetrain stress and nothing else.

AND THE MANIFOLD IS NOT PLUMBING, IT IS A TUNED PIPE. A runner is an
organ pipe closed by the valve, and the pressure wave that bounces down
it arrives back at the valve either helpfully or unhelpfully depending
on engine speed. Long runners fill the cylinder at low rpm, short ones
at high rpm, and that single choice is most of an engine's character.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

AIR_DENSITY_KG_M3 = 1.204
SPEED_OF_SOUND_M_S = 343.0
#: Mach index at which a port stops keeping up. Above this, volumetric
#: efficiency falls off a cliff rather than tapering -- it is the number
#: real head designers work to and the reason peak power rpm lands where
#: it does.
CHOKE_MACH_INDEX = 0.60
#: Lift over valve diameter at which the curtain stops being the
#: restriction and the throat takes over. Falls straight out of the
#: geometry: pi.D.L = pi/4.D^2 when L/D = 0.25.
CURTAIN_THROAT_CROSSOVER = 0.25
#: Where volumetric efficiency starts giving ground. Well below the
#: choke index: the port is still passing everything asked of it, but
#: the time available to fill the cylinder is no longer generous.
VE_KNEE_MACH_INDEX = 0.30


# ---------------------------------------------------------------------
# the head
# ---------------------------------------------------------------------

def valve_fraction_of_bore(n_valves_total: int, side: str = "intake") -> float:
    """Valve head diameter as a fraction of bore, per valve.

    Not a single constant, because it genuinely depends on how many
    valves have to fit in the bore. One big intake valve reaches about
    half the bore; two smaller ones reach about 0.37 each -- and two at
    0.37 beat one at 0.50 on both curtain and throat, which is the
    entire argument for a four-valve head."""
    per_side = max(1, int(n_valves_total) // 2)
    if side == "intake":
        return {1: 0.50, 2: 0.37, 3: 0.30}.get(per_side, 0.28)
    return {1: 0.43, 2: 0.32, 3: 0.26}.get(per_side, 0.24)


@dataclass
class PortSet:
    """One cylinder's intake or exhaust side, sized from the bore."""
    side: str = "intake"
    bore_m: float = 0.095
    stroke_m: float = 0.099
    valves_total: int = 2
    lift_m: float = 0.0095
    #: Port throat as a fraction of valve head diameter. A good port is
    #: 0.88; a restrictive cast one nearer 0.80; a fully ported head can
    #: reach 0.92 before the seat has nothing left to sit on.
    throat_frac: float = 0.88
    #: Discharge coefficient at the valve. This is what porting buys and
    #: it is the honest way to express "a better head" -- a number
    #: between about 0.55 for a bad casting and 0.75 for a race port.
    discharge_coeff: float = 0.65

    @property
    def valves_this_side(self) -> int:
        return max(1, int(self.valves_total) // 2)

    @property
    def valve_diameter_m(self) -> float:
        return self.bore_m * valve_fraction_of_bore(self.valves_total, self.side)

    @property
    def throat_diameter_m(self) -> float:
        return self.valve_diameter_m * self.throat_frac

    @property
    def curtain_area_m2(self) -> float:
        """The slot between valve head and seat, at full lift."""
        return (math.pi * self.valve_diameter_m * self.lift_m) * self.valves_this_side

    @property
    def throat_area_m2(self) -> float:
        """The hole behind the valve. Does not grow with lift."""
        return (math.pi * 0.25 * self.throat_diameter_m ** 2) * self.valves_this_side

    @property
    def effective_area_m2(self) -> float:
        """Whichever is smaller, times what the port actually flows."""
        return min(self.curtain_area_m2, self.throat_area_m2) * self.discharge_coeff

    @property
    def lift_over_diameter(self) -> float:
        return self.lift_m / max(1e-6, self.valve_diameter_m)

    @property
    def curtain_limited(self) -> bool:
        """True when more lift would still buy flow."""
        return self.lift_over_diameter < CURTAIN_THROAT_CROSSOVER

    @property
    def piston_area_m2(self) -> float:
        return math.pi * 0.25 * self.bore_m ** 2

    @property
    def area_ratio(self) -> float:
        """How much faster the gas moves than the piston. The number."""
        return self.piston_area_m2 / max(1e-9, self.effective_area_m2)

    def mean_piston_speed_m_s(self, rpm: float) -> float:
        return 2.0 * self.stroke_m * max(0.0, float(rpm)) / 60.0

    def gas_speed_m_s(self, rpm: float) -> float:
        return self.mean_piston_speed_m_s(rpm) * self.area_ratio

    def mach_index(self, rpm: float) -> float:
        return self.gas_speed_m_s(rpm) / SPEED_OF_SOUND_M_S

    @property
    def choke_rpm(self) -> float:
        """Where this port runs out of breath.

        NOT the redline -- the redline is a decision, this is a
        consequence. A head whose choke rpm is below its redline is an
        engine that stops making power before it stops turning, which is
        the commonest thing wrong with a stock engine."""
        mps = CHOKE_MACH_INDEX * SPEED_OF_SOUND_M_S / max(1e-9, self.area_ratio)
        return mps * 60.0 / (2.0 * max(1e-6, self.stroke_m))

    def volumetric_efficiency(self, rpm: float) -> float:
        """VE from breathing alone, before any manifold tuning.

        Flat while the port keeps up, then falling hard as the Mach
        index goes past the choke point. The shape is what matters: an
        engine does not taper off gently, it hits a wall."""
        # THE KNEE IS EARLIER THAN THE CHOKE. Volumetric efficiency does
        # not sit at 100% until the port chokes and then fall off a
        # cliff -- it starts giving ground as soon as gas speed becomes
        # significant, around a third of Mach, and is badly down by the
        # time the choke index is reached. Putting the knee at 0.45
        # meant a 4.2-litre pushrod six held 100% VE to six thousand
        # rpm, which is not an engine anyone has driven.
        z = self.mach_index(rpm)
        knee = VE_KNEE_MACH_INDEX
        if z <= knee:
            return 1.0
        over = (z - knee) / max(1e-6, CHOKE_MACH_INDEX - knee)
        return max(0.12, 1.0 / (1.0 + 1.35 * over * over))

    def report(self, rpm: float) -> dict:
        return {"side": self.side, "valves_this_side": self.valves_this_side,
                "valve_mm": self.valve_diameter_m * 1000.0,
                "throat_mm": self.throat_diameter_m * 1000.0,
                "curtain_cm2": self.curtain_area_m2 * 1e4,
                "throat_cm2": self.throat_area_m2 * 1e4,
                "effective_cm2": self.effective_area_m2 * 1e4,
                "L_over_D": self.lift_over_diameter,
                "curtain_limited": self.curtain_limited,
                "area_ratio": self.area_ratio,
                "gas_m_s": self.gas_speed_m_s(rpm),
                "mach_index": self.mach_index(rpm),
                "choke_rpm": self.choke_rpm,
                "ve": self.volumetric_efficiency(rpm)}


# ---------------------------------------------------------------------
# MANIFOLD TUNING LIVES IN engines.IntakeSystem, NOT HERE
# ---------------------------------------------------------------------
#
# This module briefly carried its own Helmholtz runner model. It was a
# duplicate, and the one it duplicated is better:
#
#   engines.IntakeSystem already declares runner_length_m,
#   runner_diameter_mm and plenum_volume_l, resolves tuned_rpm from
#   them, and returns resonance_gain as a proper damped-resonance
#   Lorentzian with a disclosed runner Q of 3. engines.ExhaustSystem
#   does the same on the other side, including the header-length
#   scavenging band. BOTH already feed the torque model.
#
# And the existing one is more correct in two specific ways this one was
# not. It uses the PLENUM as the resonator cavity, which is what the
# cavity physically is -- the runner is the neck and the plenum is the
# volume behind it, and using cylinder volume instead was simply the
# wrong chamber. And it derives the speed of sound from CHARGE
# TEMPERATURE rather than assuming 343 m/s, which matters, because hot
# intake air moves the tuned point by hundreds of rpm.
#
# Two tuning models that disagree is worse than either alone, so the
# duplicate is gone. What remains in this module is the part nothing
# else does: valve and port AREA, and the gas speed through it.

# ---------------------------------------------------------------------
# what is in front of the throttle
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class IntakeElement:
    key: str
    label: str
    #: Pressure drop at the engine's rated airflow, Pa.
    restriction_pa: float
    #: How much hotter than ambient the air arrives. THE term that
    #: decides whether a "less restrictive" filter actually helps.
    charge_heating_k: float
    #: Fraction of coarse dust it stops.
    filtration: float
    induction_noise_db: float
    why: str = ""


INTAKE_ELEMENTS: dict[str, IntakeElement] = {
    "airbox-panel": IntakeElement(
        "airbox-panel", "airbox with panel filter and cold duct", 900.0, 4.0, 0.995, 0.0,
        why="a sealed box fed by a duct from in front of the radiator. Slightly more "
            "restrictive on a flow bench than an open filter and usually FASTER on the "
            "car, because it is breathing air that is thirty degrees colder -- and "
            "charge density is what makes power, not the reading on the bench"),
    "open-cone": IntakeElement(
        "open-cone", "open conical filter", 450.0, 35.0, 0.97, 6.0,
        why="the dinner plate on a stick. Genuinely less restrictive, genuinely louder, "
            "and sitting in the hottest air in the engine bay -- which gives away more "
            "charge density than the restriction it saved. It measures well and it "
            "tests badly, which is why it sells"),
    "oil-bath": IntakeElement(
        "oil-bath", "oil bath air cleaner", 3200.0, 8.0, 0.998, -3.0,
        why="air is turned sharply over a pool of oil and the dust stays in the oil. "
            "Heavy, restrictive, and effectively infinite capacity -- it does not block, "
            "it just needs the oil changing. Still the right answer in real dust"),
    "cyclone-panel": IntakeElement(
        "cyclone-panel", "cyclonic pre-cleaner and panel", 2100.0, 6.0, 0.9995, 1.0,
        why="a multi-tube pre-cleaner throwing the coarse dust out before the paper "
            "sees it (see centrifuges.GasCyclone). Costs restriction and multiplies "
            "element life by thirty in real dust"),
    "velocity-stacks": IntakeElement(
        "velocity-stacks", "bare velocity stacks", 80.0, 30.0, 0.0, 12.0,
        why="no filter at all: a radiused bellmouth and nothing else. The best "
            "breathing on this list and it feeds the engine whatever is in the air, "
            "which on a road is grit. A race-only answer that people fit to road cars"),
}


def intake_element(key: str) -> IntakeElement:
    e = INTAKE_ELEMENTS.get(str(key))
    if e is None:
        raise KeyError(f"unknown intake element {key!r}; declared: "
                       f"{', '.join(sorted(INTAKE_ELEMENTS))}")
    return e


def charge_density_ratio(restriction_pa: float, heating_k: float,
                         ambient_k: float = 293.15) -> float:
    """What the tract actually delivers, relative to free air.

    Both terms at once, which is the only honest way to compare filters:
    pressure drop costs density directly, and heating costs it through
    the gas law. An element that saves half a kilopascal and adds thirty
    kelvin has made things worse, and nothing but this ratio shows it."""
    p = (101_325.0 - max(0.0, restriction_pa)) / 101_325.0
    t = ambient_k / max(1.0, ambient_k + max(0.0, heating_k))
    return p * t


@dataclass
class IntakeTract:
    """Filter, duct, throttle and plenum, in series."""
    element: str = "airbox-panel"
    throttle_diameter_m: float = 0.060
    duct_diameter_m: float = 0.070
    duct_length_m: float = 0.50
    #: Is the duct drawing from outside the engine bay?
    cold_feed: bool = True
    # No runner here: the runner and its tuning belong to
    # engines.IntakeSystem, which owns runner_length_m,
    # runner_diameter_mm and plenum_volume_l and resolves the tuned
    # point from them. This class is only what sits in FRONT of the
    # throttle -- filter, duct, throttle bore -- which nothing else
    # models as a charge-density loss.

    @property
    def spec(self) -> IntakeElement:
        return intake_element(self.element)

    @property
    def heating_k(self) -> float:
        """Underbonnet air is 30-50 K over ambient. A cold duct is the
        single cheapest power modification there is, and it is free."""
        h = self.spec.charge_heating_k
        return h if self.cold_feed else max(h, 32.0)

    def throttle_area_m2(self, open_frac: float = 1.0) -> float:
        a = math.pi * 0.25 * self.throttle_diameter_m ** 2
        return a * max(0.0, min(1.0, open_frac))

    def restriction_pa(self, flow_kg_s: float, rated_kg_s: float) -> float:
        """Total drop, scaling as the square of flow like every real
        restriction does."""
        r = (max(0.0, flow_kg_s) / max(1e-6, rated_kg_s)) ** 2
        duct = 0.5 * AIR_DENSITY_KG_M3 * (
            flow_kg_s / AIR_DENSITY_KG_M3 / max(1e-9, math.pi * 0.25 * self.duct_diameter_m ** 2)) ** 2
        return self.spec.restriction_pa * r + duct * (self.duct_length_m / 0.5) * 0.15

    def density_ratio(self, flow_kg_s: float, rated_kg_s: float,
                      ambient_k: float = 293.15) -> float:
        return charge_density_ratio(self.restriction_pa(flow_kg_s, rated_kg_s),
                                    self.heating_k, ambient_k)


# ---------------------------------------------------------------------
# the honest flow ceiling
# ---------------------------------------------------------------------

def head_flow_capacity_kg_s(bore_m: float, stroke_m: float, cylinders: int,
                            valves_total: int = 2, lift_m: float = 0.0095,
                            throat_frac: float = 0.88,
                            discharge_coeff: float = 0.65,
                            tract: "IntakeTract | None" = None,
                            ambient_k: float = 293.15) -> dict:
    """What this engine can actually pass, from its own hardware.

    The replacement for `displacement * redline / 120 * 1.2`. Nothing in
    here mentions the redline: the ceiling comes from valve size, lift,
    port quality and what is bolted in front of it, so a small-valve
    head and a ported one are different engines and porting does
    something."""
    ports = PortSet(side="intake", bore_m=bore_m, stroke_m=stroke_m,
                    valves_total=valves_total, lift_m=lift_m,
                    throat_frac=throat_frac, discharge_coeff=discharge_coeff)
    disp_l = math.pi * 0.25 * bore_m ** 2 * stroke_m * 1000.0 * max(1, cylinders)
    choke = ports.choke_rpm
    cap = (disp_l / 1000.0) * (choke / 120.0) * AIR_DENSITY_KG_M3
    dens = 1.0
    if tract is not None:
        dens = tract.density_ratio(cap, cap, ambient_k)
        cap *= dens
    return {"choke_rpm": choke, "flow_capacity_kg_s": cap,
            "displacement_l": disp_l, "density_ratio": dens,
            "area_ratio": ports.area_ratio,
            "valve_mm": ports.valve_diameter_m * 1000.0,
            "curtain_limited": ports.curtain_limited,
            "ports": ports}


# ---------------------------------------------------------------------
# CALIBRATION QUEUE: anchors for the intake resonance constants
# ---------------------------------------------------------------------
#
# RETARGETED. This queue originally aimed at a tuning constant in this
# module, which has since been deleted as a duplicate of
# engines.IntakeSystem. The calibration is still worth doing -- it just
# belongs to the SURVIVING model, whose own two disclosed constants are
# in exactly the position wear life was in before INDUSTRIAL_SPECS:
#
#   engines.INTAKE_RUNNER_Q             declared as "a typical intake
#                                       runner Q of 3", unverified
#   engines.INTAKE_RESONANCE_PEAK_GAIN  how much a tuned runner is
#                                       actually worth at its peak
#
# Collect anchors, solve both per anchor, look at the spread. If it is
# tight the constants are defensible; if not, they are standing in for a
# real dependency.
#
# WHAT A VALID ANCHOR NEEDS. All four, from the same engine:
#
#   runner length      plenum face to valve, along the centreline. The
#                      number everybody quotes loosely and few measure.
#   runner diameter    or area, if it is not round. Area is what the
#                      resonator equation wants.
#   cylinder volume    swept, per cylinder.
#   torque peak rpm    where the manifold is actually helping. Peak
#                      TORQUE, not peak power -- power peaks past the
#                      ram peak because it is fighting the VE curve.
#
# THE BEST ANCHORS ARE VARIABLE-LENGTH MANIFOLDS, and they are worth
# chasing first. An engine with a switching intake -- BMW's DISA,
# Toyota's ACIS, Porsche's VarioRam -- gives TWO lengths on the SAME
# cylinder volume with the same head and the same cam. That isolates
# the constant completely, because everything the ratio is absorbing
# stays fixed between the two points and only L changes. One such engine
# is worth more than five single-length ones.
#
# DELIBERATELY EMPTY. Sourced runner geometry is not something to
# reconstruct from memory, and a fabricated anchor is worse than no
# anchor -- it would make the calibration report look converged while
# certifying a guess. Entries go in when the geometry is actually in
# hand.

@dataclass(frozen=True)
class ManifoldSpec:
    """One engine's measured intake tract and where it makes torque."""
    runner_length_m: float
    runner_diameter_m: float
    cylinder_volume_m3: float
    torque_peak_rpm: float
    source: str = ""
    #: For a switching manifold: the other length, and its peak.
    alternate_length_m: float = 0.0
    alternate_peak_rpm: float = 0.0


MANIFOLD_SPECS: dict[str, ManifoldSpec] = {}


def solve_tuning_ratio(spec: ManifoldSpec,
                       sound_speed_m_s: float = SPEED_OF_SOUND_M_S) -> dict:
    """Solve a tuning ratio from one anchor, for comparison against
    engines.IntakeSystem's own tuned_rpm on the same hardware.

    Inverts tuned_rpm: K = 120 . f_helmholtz / rpm_peak. A switching
    manifold yields two independent values, and the gap between them is
    the honest measure of whether the resonator model is carrying the
    physics or the constant is."""
    area = math.pi * 0.25 * spec.runner_diameter_m ** 2
    V = max(1e-9, spec.cylinder_volume_m3)
    f = (sound_speed_m_s / (2.0 * math.pi)) * math.sqrt(
        area / (max(0.02, spec.runner_length_m) * V))
    k = 120.0 * f / max(1.0, spec.torque_peak_rpm)
    out = {"k": k, "helmholtz_hz": f, "source": spec.source}
    if spec.alternate_length_m > 0.0 and spec.alternate_peak_rpm > 0.0:
        f2 = (sound_speed_m_s / (2.0 * math.pi)) * math.sqrt(
            area / (max(0.02, spec.alternate_length_m) * V))
        k2 = 120.0 * f2 / max(1.0, spec.alternate_peak_rpm)
        out["k_alternate"] = k2
        out["self_consistency"] = abs(k2 - k) / max(1e-6, (k + k2) * 0.5)
    return out


def calibration_report() -> dict:
    """Where the constant stands against whatever anchors exist.

    Mirrors wear_profile.calibration_report: says plainly how many
    anchors there are, what each demands, and whether one number can
    serve them all."""
    rows = {k: solve_tuning_ratio(v) for k, v in MANIFOLD_SPECS.items()}
    ks = [r["k"] for r in rows.values()]
    if not ks:
        return {"anchors": 0, "in_use": None, "rows": {},
                "verdict": ("no anchors. engines.INTAKE_RUNNER_Q and "
                            "INTAKE_RESONANCE_PEAK_GAIN are both declared as typical "
                            "values and neither has been checked against a real "
                            "engine. Historical candidates: Chrysler's 1960 Ram "
                            "Induction published its long-runner tuning target, and "
                            "pre-war supercharged racing engines documented induction "
                            "lengths carefully")}
    lo, hi = min(ks), max(ks)
    mean = sum(ks) / len(ks)
    spread = (hi - lo) / max(1e-6, mean)
    return {"anchors": len(ks), "in_use": None,
            "mean": mean, "low": lo, "high": hi, "spread": spread,
            "rows": rows,
            "verdict": ("one constant serves these" if spread < 0.20 else
                        "spread too wide for one constant: it is standing in for a "
                        "real dependency, most likely valve timing or plenum volume")}
