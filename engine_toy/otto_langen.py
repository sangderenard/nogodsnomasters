"""The Otto & Langen atmospheric engine (1867) -- the real free-piston,
crankless predecessor to the four-stroke Otto cycle.

The real mechanism, and why it needs a genuinely different sim loop
from every other engine in this catalogue (all of which assume a
crank that ties piston position to a fixed rotational phase): there is
no crank. The piston is free. A vertical cylinder is closed at the
bottom (the working space) and open to atmosphere at the top; a rack
fixed to the piston rod runs up through that open top and meshes a
pinion, which drives the flywheel/output shaft through a one-way
(overrunning/ratchet) clutch -- real power take-off ONLY on the
downward stroke.

One real cycle:
  1. IGNITE -- a charge trapped below the piston (near-atmospheric,
     already drawn in) is ignited at essentially constant volume,
     spiking its pressure well above atmospheric.
  2. FREE FLIGHT UP -- that overpressure accelerates the piston
     upward, genuinely unloaded (the ratchet only catches on the way
     down -- the real point of the mechanism: nothing drains the
     piston's kinetic energy before it's had a chance to overshoot).
     As the piston rises the trapped gas expands (real adiabatic
     relation, P*V^gamma = const, fixed mass -- nothing crosses the
     piston boundary), so pressure falls past atmospheric, and
     inertia carries the piston on past that point too, pulling the
     gas below into a real vacuum.
  3. PEAK -- the piston decelerates under (atmosphere above) minus
     (falling pressure below) and its velocity crosses zero at peak
     height, over the real vacuum it just created.
  4. POWER STROKE DOWN -- atmosphere above (unopposed by the now-
     sub-atmospheric charge below) drives the piston back down. The
     ratchet catches -- THIS is the only phase that ever loads the
     flywheel: real work is extracted from atmospheric pressure
     re-filling the vacuum the combustion stroke created, not from
     the combustion pressure itself (combustion's whole job was
     lofting the piston high enough to make that vacuum).
  5. EXHAUST -- a real cam on the rack/pinion mechanism trips an
     exhaust valve once the power stroke has done its useful work
     (the piston has fallen behind the flywheel's own speed and the
     ratchet has overrun) -- venting the remaining trapped charge to
     atmosphere so the piston coasts the rest of the way down under
     gravity alone instead of recompressing its own spent charge into
     a trapped spring (found empirically: without a position/event-
     triggered valve, a piston that overruns above the ignition
     height just recompresses what's left of its own charge and
     settles into a silent equilibrium a good distance short of
     bottom, never reaching a fresh charge at all -- exactly why a
     real engine's exhaust valve is a timed mechanical event, not
     something that waits for the gas to sort itself out). The
     working volume then resets to atmospheric and the next charge is
     ready to ignite once the piston reaches the bottom.

The one-way ratchet is solved as an instantaneous momentum-conserving
catch (angular momentum conservation between the piston's own inertia,
reflected through the pinion radius, and the shaft's), NOT as a
saturating spring/tanh clutch (drivetrain_port.py's ClutchPort, the
usual real-game formula reused everywhere else in this toy). Found
empirically why that reuse doesn't work here: a real ratchet is
rigid, and this mechanism's piston is light (kilograms) while the
lever amplification through a small pinion radius makes even a modest
holding torque correspond to an enormous force on the rack -- any
spring-style engagement integrated at a finite timestep massively
overshoots the lock-up in a single substep, then instantly unlocks
again, chattering open/shut every few substeps instead of catching.
A real ratchet/pawl/sprag doesn't have that failure mode because it
IS a rigid mechanical catch, not a spring -- so it's modeled as one:
solve the exact shared "locked" speed the two real inertias settle to
by momentum conservation (clamped to the ratchet's own real torque
rating for a genuine overload/slip case), the standard, correct way a
physics engine handles a one-way velocity constraint, rather than
force-integrating a stiff spring through it.

Disclosed simplifications (real mechanism, one or two named
constants, same house style as the rest of this catalogue):
  - Ignition is modeled as instantaneous constant-volume heat
    addition, not flame-kernel propagation (engine_cycle_sim's real
    flame-growth model assumes a fast-spinning crank sweeping the
    burn across a few degrees; nothing here spins fast enough for
    that geometry to mean anything before the piston has already
    left). IGNITION_PRESSURE_RATIO is the one disclosed number this
    adds: a real, historically-consistent ~3x pressure (and
    temperature) rise for a lean coal-gas/petroleum-vapor charge
    ignited near atmospheric density -- these engines ran deliberately
    lean/low-compression charges, nowhere near stoichiometric adiabatic
    flame temperature, which is why atmospheric engines were real but
    thermally poor (~10% efficiency at best; most of the combustion
    energy leaves as hot exhausted gas, not as piston work).
  - The exhaust/recharge event is instantaneous once the piston
    returns to the ignition height -- a real engine's port timing and
    charge-draw dynamics aren't modeled, only the working volume's
    thermodynamic reset.
  - Gas is treated as air (R, cv, gamma) throughout, not distinguished
    from the actual fuel-air/exhaust mixture -- the same simplification
    the rest of this catalogue makes for intake/exhaust gas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- real gas/atmosphere constants --------------------------------------
P_ATM_PA = 101_325.0
T_ATM_K = 293.15
R_AIR_J_PER_KGK = 287.0
GAMMA_COMBUSTION_GAS = 1.30   # real: hot combustion products, below cold air's 1.4
CV_J_PER_KGK = R_AIR_J_PER_KGK / (GAMMA_COMBUSTION_GAS - 1.0)

# the one disclosed combustion number -- see module docstring. Real,
# but fuel-dependent: this is the historical coal-gas default, kept as
# the module-level fallback for any caller that doesn't specify a
# fuel. Different real fuels genuinely produce a different real
# pressure rise from the same lean, near-atmospheric-density charge --
# see IGNITION_PRESSURE_RATIO_BY_FUEL below.
IGNITION_PRESSURE_RATIO = 3.0

# Real, per-fuel ignition pressure ratios for this same lean,
# atmospheric-density charge. coal-gas is the historically-cited
# baseline (see module docstring); the others are real, physically-
# reasoned estimates (disclosed as such, not directly cited the way
# coal-gas is) -- wood/producer gas is genuinely diluted by the real
# N2/CO2 the gasification process leaves in it (commonly ~40-50% the
# heating value of coal gas), so a similarly-lean charge produces a
# proportionally smaller real pressure rise; natural gas and petroleum
# vapor are real, higher-energy-density alternatives a modern rebuild
# might actually run, each scaled the same real way off their own
# heating value relative to coal gas.
IGNITION_PRESSURE_RATIO_BY_FUEL: dict[str, float] = {
    "coal-gas": 3.0,
    "wood-gas": 2.2,             # disclosed estimate: real producer-gas dilution
    "natural-gas": 3.4,          # disclosed estimate: real higher heating value
    "petroleum-vapor": 3.6,      # disclosed estimate: real higher heating value
}

# Real fuel-gas COMPOSITION properties per profile -- what turns "how
# much of what's in the gasholder is actually fuel gas" into "does this
# charge fire at all, and how hard". All fractions are VOLUME fractions
# of fuel gas in air at ~atmospheric pressure (the only regime this
# engine ever admits a charge in). lfl/ufl are the real lower/upper
# flammability limits (the standard Bureau of Mines tables, Coward &
# Jones "Limits of Flammability of Gases and Vapors", widely
# reproduced: town/coal gas ~5-31%, producer gas ~20-74%, methane
# 5-15%, gasoline vapour 1.4-7.6%); stoich_vol_frac is the real
# stoichiometric gas fraction from each gas's own air requirement
# (coal gas ~5 vol air per vol gas; producer gas ~1.1 -- it's already
# half nitrogen; methane 9.5; gasoline vapour ~55). density_kg_m3 is
# the real gas density at ambient (coal gas is the same figure
# gas_works.py's own production accounting uses -- one source).
#
# Outside [lfl, ufl] a flame genuinely cannot propagate through the
# charge -- the cylinder admits the mixture, the igniter lights, and
# NOTHING happens (a real misfire, the charge just gets pushed out
# unburned on the next cycle). This is the actual physical reason a
# gasholder full of pure fuel gas is NOT itself an explosion hazard
# (it's far above ufl -- no oxygen in it to burn with) while the same
# holder half-way through being purged of air on first fill genuinely
# IS one (it passes straight through the flammable band on the way),
# and why the ONLY place a real plant ever deliberately makes a
# flammable mixture is inside the engine's own mixer, one charge at a
# time.
# (values live in working_fluids.WORKING_FLUIDS -- one registry for every
# fuel and expander fluid; this name is a view of its gaseous rows)
from working_fluids import FUEL_GAS_PROPERTIES  # noqa: E402
AIR_DENSITY_KG_M3 = P_ATM_PA / (R_AIR_J_PER_KGK * T_ATM_K)   # ~1.20 at this module's own ambient

# The real, fixed mixer setting an Otto-Langen's slide valve admits
# gas and air at -- port areas, not a needle: one disclosed number,
# the same way IGNITION_PRESSURE_RATIO is. Nineteenth-century gas
# engines ran deliberately lean, commonly quoted around 1 part coal
# gas to 9-10 parts air; against coal gas's own ~1:5 stoichiometric
# requirement that is an equivalence ratio near 0.55, which is what
# IGNITION_PRESSURE_RATIO_BY_FUEL's full-strength figures are DEFINED
# at: a charge of pure (concentration 1.0) fuel gas through the design
# mixer produces exactly the profile's cited ratio, and since a real
# gasholder can never deliver a concentration ABOVE 1.0, the design
# mixer is the real physical ceiling on ignition strength -- which is
# exactly what resolve_stroke_m's safety margin is measured against.
# Opening the mixer richer than this (AtmosphericEngineSpec.mixer_gas_
# volume_fraction) genuinely lets a real charge exceed the cited ratio
# (see charge_ignition_pressure_ratio) with no margin left for it.
#
# Per fuel, because the real lean flammability limit is per fuel: a
# coal-gas-style 1:10 mixer setting (phi 0.55) is comfortably inside
# coal gas's and producer gas's own wide bands, but sits right ON
# methane's lean limit (phi ~0.53) and well OUTSIDE gasoline vapour's
# (phi ~0.78) -- a real Otto-Langen re-plumbed for either would need
# its slide-valve ports re-cut richer just to fire at all, so each of
# those profiles' cited full-strength ratios is DEFINED at its own
# real lean-but-safely-flammable setting instead of one setting for
# all four.
MIXER_DESIGN_EQUIVALENCE_RATIO = 0.55
MIXER_DESIGN_EQUIVALENCE_RATIO_BY_FUEL: dict[str, float] = {
    "coal-gas": 0.55,
    "wood-gas": 0.55,
    "natural-gas": 0.70,
    "petroleum-vapor": 0.90,
}


def mixer_design_equivalence_ratio(fuel_profile: str) -> float:
    return MIXER_DESIGN_EQUIVALENCE_RATIO_BY_FUEL.get(fuel_profile, MIXER_DESIGN_EQUIVALENCE_RATIO)


def fuel_gas_properties(fuel_profile: str) -> dict[str, float]:
    return FUEL_GAS_PROPERTIES.get(fuel_profile, FUEL_GAS_PROPERTIES["coal-gas"])


def design_mixer_gas_volume_fraction(fuel_profile: str) -> float:
    """The design mixer's real gas volume fraction for this fuel --
    the profile's own stoichiometric fraction leaned to
    MIXER_DESIGN_EQUIVALENCE_RATIO (phi = g/stoich, the lean-in-air
    volume-basis relation, which is the whole regime here)."""
    return fuel_gas_properties(fuel_profile)["stoich_vol_frac"] * mixer_design_equivalence_ratio(fuel_profile)


def fuel_mass_to_volume_fraction(fuel_mass_frac: float, fuel_density_kg_m3: float,
                                 air_density_kg_m3: float = AIR_DENSITY_KG_M3) -> float:
    """Real conversion of a fuel-gas/air mixture's MASS fraction (what a
    mass-balanced reservoir tracks) to its VOLUME (molar) fraction
    (what flammability limits and a mixer's port areas are stated
    in): both components at the same pressure/temperature, so volume
    is mass over density for each."""
    c = max(0.0, min(1.0, fuel_mass_frac))
    v_fuel = c / max(fuel_density_kg_m3, 1e-9)
    v_air = (1.0 - c) / max(air_density_kg_m3, 1e-9)
    return v_fuel / max(v_fuel + v_air, 1e-12)


def charge_is_flammable(fuel_profile: str, gas_volume_fraction: float) -> bool:
    props = fuel_gas_properties(fuel_profile)
    return props["lfl"] <= gas_volume_fraction <= props["ufl"]


def charge_ignition_pressure_ratio(fuel_profile: str, gas_volume_fraction: float,
                                   design_ratio: float | None = None) -> float:
    """The real constant-volume pressure rise for a charge of the given
    fuel-gas volume fraction in air. Outside the flammability limits:
    1.0 -- no flame, no pressure rise, a real misfire. Inside them, a
    first-order real heat-release scaling off the profile's cited
    full-strength ratio (which is defined at design_mixer_gas_volume_
    fraction, see MIXER_DESIGN_EQUIVALENCE_RATIO): lean of stoich all
    the fuel burns, so heat release is proportional to fuel fraction
    (phi); rich of stoich the available oxygen limits it, so it falls
    as 1/phi -- continuous at phi = 1. A disclosed first-order law
    (real charges near the limits also burn slower and less
    completely -- not modeled beyond the hard cutoff), not a fitted
    curve to any specific engine's indicator diagram."""
    props = fuel_gas_properties(fuel_profile)
    if design_ratio is None:
        design_ratio = IGNITION_PRESSURE_RATIO_BY_FUEL.get(fuel_profile, IGNITION_PRESSURE_RATIO)
    g = gas_volume_fraction
    if not (props["lfl"] <= g <= props["ufl"]):
        return 1.0
    phi = g / max(props["stoich_vol_frac"], 1e-9)
    heat_release_rel = (phi if phi <= 1.0 else 1.0 / phi) / mixer_design_equivalence_ratio(fuel_profile)
    return 1.0 + (design_ratio - 1.0) * heat_release_rel


# Real, disclosed safe-impact-speed rating for the top-of-stroke
# bracket, by what it's actually made of -- the same real "several
# full size have been destroyed" failure mode (see STROKE_STOP_
# DESTRUCTIVE_SPEED_M_S's own comment) at a real, materially-different
# threshold. The period default matches the actual historical outcome
# (real engines really were destroyed by this); a modern precision
# buffer is a real, higher-rated alternative a game or a rebuild could
# choose to fit instead, at whatever real cost/weight/complexity that
# choice implies elsewhere -- not free, just a genuine option.
BUMP_STOP_SAFE_IMPACT_SPEED_M_S: dict[str, float] = {
    "cast-iron-period": 1.8,            # the real 1867 baseline -- genuinely fragile
    "hardened-steel-modern": 4.5,       # a modern precision-machined rigid stop
    "elastomer-hydraulic-buffer": 8.0,  # a modern engineered snubber/damper
}

# real mechanical friction on the piston/rack/pinion train -- a
# disclosed, modest viscous+Coulomb figure (packing gland, rack guide,
# pinion bearings), not zero (a frictionless free piston never settles)
RACK_FRICTION_N = 40.0
RACK_VISCOUS_N_PER_MPS = 25.0
GRAVITY_M_S2 = 9.81

# Real, and researched rather than invented: an Otto-Langen-class free
# piston reaching the top of its own travel under real power ("bottoms
# out on the frame") is a documented, genuinely destructive failure in
# real machines, not a survivable bump -- "Several full size have been
# destroyed in that way" (Home Model Engine Machinist forum; Old
# Machine Press's own account of the mechanism). The real governor (and,
# on earlier production engines, a dedicated safety slide valve later
# judged unnecessary once the governor/pawl combination proved reliable
# enough) existed specifically to keep the piston from ever reaching
# this point under real power -- it is a failure boundary, not an
# operating limit, and this sim treats it as one: a genuine impact
# below a real light-contact threshold gets a modest, mostly-inelastic
# bounce (a light tap against the frame, not a power-stroke slam); above
# it, the real documented consequence is structural failure, not a
# bounce back into normal operation. See OttoLangenCylinder.destroyed.
STROKE_STOP_RESTITUTION = 0.15
# below this real impact speed: a light contact, survivable, matching
# the modest restitution above. At or above it: the real documented
# failure case (a genuine power-stroke slam), not a survivable event.
# Module-level fallback for any caller that doesn't specify a bump-stop
# material -- real, per-material values now live in
# BUMP_STOP_SAFE_IMPACT_SPEED_M_S above; this is just the period
# default (cast-iron-period) kept as a bare constant for compatibility.
STROKE_STOP_DESTRUCTIVE_SPEED_M_S = BUMP_STOP_SAFE_IMPACT_SPEED_M_S["cast-iron-period"]


@dataclass(frozen=True)
class NaturalCycleEstimate:
    """The real, derived (not tuned) top-end this cylinder's own gas-
    law/gravity/friction physics can produce -- see
    AtmosphericEngineSpec.natural_cycle_estimate."""
    period_s: float
    peak_height_m: float
    peak_power_stroke_force_n: float
    peak_power_stroke_speed_m_s: float
    max_output_rpm: float


@dataclass(frozen=True)
class AtmosphericEngineSpec:
    """The real, DECLARED build for one Otto-Langen cylinder -- what
    goes in the catalogue (engines.Engine.atmospheric) -- separate from
    OttoLangenCylinder itself, which also carries live runtime state
    (piston position/velocity, ignition phase, cycle count). `build()`
    constructs a fresh runtime object from this spec.

    `fuel_profile` and `bump_stop_material` are real, named design
    choices -- not free parameters an engine's own physics ignores.
    fuel_profile sets the real ignition pressure ratio (IGNITION_
    PRESSURE_RATIO_BY_FUEL), which sets the real, derived peak stroke
    height (natural_cycle_estimate) -- "more bang" genuinely means a
    higher real peak, no separate knob for it. bump_stop_material sets
    the real safe-impact-speed rating (BUMP_STOP_SAFE_IMPACT_SPEED_
    M_S) for the top-of-stroke bracket -- the actual real consequence
    of choosing a hotter fuel without also giving it enough real
    stroke margin or a better real bracket to survive hitting it. See
    resolve_stroke_m for deriving stroke_m from a real safety margin
    instead of guessing it."""
    bore_m: float
    stroke_m: float
    ignition_height_m: float
    piston_mass_kg: float
    pinion_radius_m: float
    max_shaft_torque_nm: float = 4_000.0
    fuel_profile: str = "coal-gas"
    bump_stop_material: str = "cast-iron-period"
    # the slide valve's real fixed gas:air admission, as the gas volume
    # fraction of the charge it draws. None = the design setting for
    # this fuel (design_mixer_gas_volume_fraction -- the one
    # ignition_pressure_ratio is cited at). Richer than design is a
    # real, allowed, genuinely dangerous choice: see charge_ignition_
    # pressure_ratio and MIXER_DESIGN_EQUIVALENCE_RATIO's own comment.
    mixer_gas_volume_fraction: float | None = None

    @property
    def ignition_pressure_ratio(self) -> float:
        """Full-strength (pure fuel gas through the design mixer)."""
        return IGNITION_PRESSURE_RATIO_BY_FUEL.get(self.fuel_profile, IGNITION_PRESSURE_RATIO)

    @property
    def mixer_gas_volume_fraction_resolved(self) -> float:
        if self.mixer_gas_volume_fraction is not None:
            return max(0.0, min(1.0, self.mixer_gas_volume_fraction))
        return design_mixer_gas_volume_fraction(self.fuel_profile)

    @property
    def fuel_gas_density_kg_m3(self) -> float:
        return fuel_gas_properties(self.fuel_profile)["density_kg_m3"]

    def charge_ignition_pressure_ratio(self, gas_volume_fraction: float) -> float:
        """This cylinder's real pressure rise for an ACTUAL admitted
        charge (see module-level charge_ignition_pressure_ratio): the
        mixer's gas share times whatever fraction of the gas line's
        contents is genuinely fuel gas, times how much of the demanded
        gas the supply actually delivered (the shortfall is air -- the
        piston's own vacuum fills the cylinder regardless)."""
        return charge_ignition_pressure_ratio(self.fuel_profile, gas_volume_fraction,
                                              design_ratio=self.ignition_pressure_ratio)

    @property
    def destructive_impact_speed_m_s(self) -> float:
        return BUMP_STOP_SAFE_IMPACT_SPEED_M_S.get(self.bump_stop_material, STROKE_STOP_DESTRUCTIVE_SPEED_M_S)

    def build(self) -> "OttoLangenCylinder":
        return OttoLangenCylinder(
            bore_m=self.bore_m, stroke_m=self.stroke_m,
            ignition_height_m=self.ignition_height_m,
            piston_mass_kg=self.piston_mass_kg, pinion_radius_m=self.pinion_radius_m,
            max_shaft_torque_nm=self.max_shaft_torque_nm,
            ignition_pressure_ratio=self.ignition_pressure_ratio,
            destructive_impact_speed_m_s=self.destructive_impact_speed_m_s)

    def natural_cycle_estimate(self, dt: float = 2e-4, max_time_s: float = 30.0) -> "NaturalCycleEstimate":
        """The real, ungoverned top cycle rate this cylinder's own gas
        law/gravity/friction physics can produce -- a genuine, derived
        quantity, not a tuned number: this IS a gravity engine (see
        module docstring -- the power stroke is atmosphere re-filling
        the vacuum combustion just created, not combustion pressure
        itself), so its cycle time is set entirely by how long the
        piston's own free-flight/fall under P(x), gravity, and friction
        takes, the same way a pendulum's period is a real derived
        quantity and not a dial setting.

        Solved by reusing OttoLangenCylinder.step ITSELF (not a
        parallel formula), coupled to a real, negligibly-light shaft
        (shaft_inertia_kg_m2 tiny, shaft_load_torque_nm=0.0) -- a real
        "nothing but the bare pinion on the end of the rack" case, not
        a fully disconnected one: the ratchet still genuinely catches
        (needed for the real event-triggered exhaust valve to ever
        fire at all -- see module docstring on why that valve exists),
        but a near-massless shaft offers no real resistance, so almost
        none of the piston's own momentum is lost to it. This is the
        fastest this cylinder's own real cycle can ever go -- any
        actual flywheel/dyno load only ever slows it down from here,
        never speeds it up.

        The piston itself never revolves -- it's a free-flying linear
        mass, not a crank throw -- so "shaft rpm" isn't a cycle-rate
        question at all (that would wrongly average a linear distance
        over a whole intermittent catch-and-release cycle, as if the
        pinion stayed continuously meshed and spinning the entire
        time, which it doesn't). The real, physically honest bridge is
        the rack-and-pinion's own rigid instantaneous relation, omega =
        v/r: the piston's own real PEAK velocity during the power
        stroke (a genuine gas-law/gravity quantity -- how fast it's
        moving at the instant it's recovering the deepest real vacuum)
        is exactly the peak instantaneous speed the pinion -- and so
        the output shaft -- could ever be spun to, at that one instant
        of best mechanical advantage. A real flywheel never actually
        reaches this (its own inertia means each catch only nudges it
        partway there), but nothing catching against this rack can
        ever exceed it either -- it's the genuine physical ceiling."""
        probe = OttoLangenCylinder(bore_m=self.bore_m, stroke_m=self.stroke_m,
                                    ignition_height_m=self.ignition_height_m,
                                    piston_mass_kg=self.piston_mass_kg,
                                    pinion_radius_m=self.pinion_radius_m,
                                    max_shaft_torque_nm=self.max_shaft_torque_nm,
                                    ignition_pressure_ratio=self.ignition_pressure_ratio,
                                    destructive_impact_speed_m_s=self.destructive_impact_speed_m_s)
        negligible_shaft_inertia_kg_m2 = 1e-9
        t = 0.0
        peak_height_m = self.ignition_height_m
        min_pressure_pa = P_ATM_PA
        peak_power_stroke_speed_m_s = 0.0
        shaft_omega = 0.0
        while t < max_time_s:
            shaft_omega = probe.step(dt, shaft_omega, negligible_shaft_inertia_kg_m2, 0.0, fire=True)
            peak_height_m = max(peak_height_m, probe.x_m)
            min_pressure_pa = min(min_pressure_pa, probe.working_pressure_pa)
            if probe.phase in ("free_flight", "exhausting") and probe.v_m_s < 0.0:
                peak_power_stroke_speed_m_s = max(peak_power_stroke_speed_m_s, -probe.v_m_s)
            t += dt
            if probe.cycle_count >= 1:
                break
        # the real maximum driving force this cycle ever produces --
        # atmosphere pushing on one face against the deepest real
        # vacuum the expansion reached on the other, at the cylinder's
        # own bore area -- occurring right as the power stroke begins
        peak_force_n = max(0.0, P_ATM_PA - min_pressure_pa) * (math.pi * (self.bore_m / 2.0) ** 2)
        # omega = v/r, the pinion's own rigid geometric relation -- see
        # this method's own docstring for why this (not a cycle-rate
        # average) is the real bridge from a linear, non-revolving
        # piston to a shaft rpm ceiling
        peak_omega_rad_s = peak_power_stroke_speed_m_s / max(self.pinion_radius_m, 1e-6)
        return NaturalCycleEstimate(
            period_s=max(t, 1e-6), peak_height_m=peak_height_m,
            peak_power_stroke_force_n=peak_force_n,
            peak_power_stroke_speed_m_s=peak_power_stroke_speed_m_s,
            max_output_rpm=peak_omega_rad_s * 60.0 / (2.0 * math.pi))


def resolve_stroke_m(bore_m: float, ignition_height_m: float, piston_mass_kg: float,
                      pinion_radius_m: float, fuel_profile: str = "coal-gas",
                      max_shaft_torque_nm: float = 4_000.0, safety_margin_frac: float = 1.2) -> float:
    """Derives the real stroke this cylinder actually NEEDS, instead of
    guessing one and hoping it's long enough. "More bang" from a
    hotter fuel (a higher real IGNITION_PRESSURE_RATIO_BY_FUEL entry)
    genuinely raises the real, gas-law-derived peak height -- run the
    exact same free-flight physics natural_cycle_estimate already uses
    (via a temporary spec built with a deliberately huge probe stroke,
    so the probe never clips its own measurement against a too-short
    stop) to find that real unconstrained peak, then require the real
    declared stroke to clear it by `safety_margin_frac` (1.2 = 20% real
    clearance). This is the real, physical reason a fuel switch without
    also lengthening the stroke (or fitting a better bump stop -- see
    AtmosphericEngineSpec.bump_stop_material) breaks the engine: the
    real peak moved, the real hardware didn't."""
    probe_spec = AtmosphericEngineSpec(
        bore_m=bore_m, stroke_m=bore_m * 1000.0,   # deliberately huge -- never clips the real peak
        ignition_height_m=ignition_height_m, piston_mass_kg=piston_mass_kg,
        pinion_radius_m=pinion_radius_m, max_shaft_torque_nm=max_shaft_torque_nm,
        fuel_profile=fuel_profile)
    natural = probe_spec.natural_cycle_estimate()
    return natural.peak_height_m * safety_margin_frac


@dataclass
class OttoLangenCylinder:
    """One real cylinder of the atmospheric engine. `bore_m`/`stroke_m`
    are this cylinder's own real geometry; `ignition_height_m` is how
    far above absolute bottom the piston sits when the charge ignites
    (real engines ignited with a small initial volume, not at zero --
    a completely collapsed working volume has nowhere for the flame to
    establish); `piston_mass_kg` is the real reciprocating mass
    (piston + rod + rack, the whole free-flying assembly).

    Owns the joint piston<->shaft ratchet solve directly (see module
    docstring for why): `step()` takes the shaft's own inertia and
    external load and returns the shaft's new omega -- the shaft is
    real, shared state the cylinder and its caller both touch, exactly
    like this toy's existing dyno-absorber load shaft pattern
    elsewhere, just solved exactly instead of via a separately-
    integrated returned torque (which is what chattered)."""
    bore_m: float
    stroke_m: float
    ignition_height_m: float
    piston_mass_kg: float
    pinion_radius_m: float
    max_shaft_torque_nm: float = 4000.0   # the ratchet's own real holding capacity before it'd slip/skip a tooth
    # real, per-instance (not bare module constants) -- see
    # AtmosphericEngineSpec's own docstring for what sets these: the
    # chosen fuel's real pressure rise, and what the top-of-stroke
    # bracket is actually made of
    ignition_pressure_ratio: float = IGNITION_PRESSURE_RATIO
    destructive_impact_speed_m_s: float = STROKE_STOP_DESTRUCTIVE_SPEED_M_S

    x_m: float = field(init=False)          # piston height above absolute bottom
    v_m_s: float = field(init=False)
    working_pressure_pa: float = field(init=False)
    _post_ignition_pv_gamma: float = field(default=0.0, init=False)   # P2 * V2^gamma, the isentrope's constant
    phase: str = field(default="awaiting_ignition", init=False)
    locked: bool = field(default=False, init=False)
    cycle_count: int = field(default=0, init=False)
    # the real top-of-stroke bracket's own last impact -- 0.0 whenever
    # the piston didn't reach the stop this tick, real m/s otherwise
    # (the speed it was carrying INTO the impact, before the stop's own
    # real coefficient of restitution absorbed most of it). A caller
    # (audio, wear/damage tracking) reads this to know a real hard-stop
    # event just happened, rather than the energy having just silently
    # vanished the way an unconditional velocity-to-zero clamp would.
    last_stroke_stop_impact_speed_m_s: float = field(default=0.0, init=False)
    # the real, documented consequence of a hard top-of-stroke impact
    # (see STROKE_STOP_DESTRUCTIVE_SPEED_M_S's own comment) -- once
    # True, a real broken machine: step() stops integrating the piston
    # at all. The real repair path already exists at the caller level
    # (engine_cycle_sim.py rebuilds a fresh OttoLangenCylinder from its
    # own declared spec on stop() -- the same real "swap the wrecked
    # part for a new one" this dataclass has no reason to duplicate).
    destroyed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.x_m = self.ignition_height_m
        self.v_m_s = 0.0
        self.working_pressure_pa = P_ATM_PA

    @property
    def piston_area_m2(self) -> float:
        return math.pi * (self.bore_m / 2.0) ** 2

    def ignite(self) -> None:
        """Real constant-volume heat addition at the current (small)
        working volume: P2 = P1 * ratio, and since V is momentarily
        unchanged, that fixes the isentrope P*V^gamma = const this
        cylinder's free flight will follow until the next exhaust
        event -- no separate mass/temperature bookkeeping needed
        because nothing crosses the piston boundary in between."""
        v1 = self.piston_area_m2 * self.x_m
        p2 = P_ATM_PA * self.ignition_pressure_ratio
        self._post_ignition_pv_gamma = p2 * v1 ** GAMMA_COMBUSTION_GAS
        self.working_pressure_pa = p2
        self.phase = "free_flight"

    def _pressure_at(self, x_m: float) -> float:
        v = self.piston_area_m2 * max(x_m, 1e-6)
        return self._post_ignition_pv_gamma / (v ** GAMMA_COMBUSTION_GAS)

    def step(self, dt: float, shaft_omega: float, shaft_inertia_kg_m2: float,
             shaft_load_torque_nm: float = 0.0, fire: bool = True) -> float:
        """Advances the free piston and the shaft it drives one tick
        together; returns the shaft's new omega. `shaft_load_torque_nm`
        is whatever the flywheel/output is doing besides this cylinder
        (a brake, an accessory, other cylinders on the same shaft) --
        positive opposes positive rotation, same sign convention as
        this catalogue's other torque ports.

        `fire=False` is a real governor's "miss": the exhaust valve is
        simply held open instead of a charge being ignited (the same
        real mechanism a captive-ball governor -- governor.py -- or
        this catalogue's other hit-and-miss engine already use). The
        piston stays parked at rest at ignition height under
        atmospheric pressure on both faces -- there's no gas spring to
        fight and nothing driving it, so the flywheel simply coasts
        through this cycle on its own stored momentum, exactly a real
        miss."""
        if self.destroyed:
            # a real wrecked machine -- the piston genuinely doesn't
            # move again (see this field's own docstring); the shaft
            # just coasts under whatever else is loading it, same as a
            # governed miss above, since there's nothing left driving it
            return shaft_omega - shaft_load_torque_nm / max(shaft_inertia_kg_m2, 1e-9) * dt
        if self.phase == "awaiting_ignition":
            if not fire:
                return shaft_omega - shaft_load_torque_nm / max(shaft_inertia_kg_m2, 1e-9) * dt
            self.ignite()

        area = self.piston_area_m2
        if self.phase == "exhausting":
            # the valve is open: the working volume is vented straight
            # to atmosphere, no gas spring left to fight
            self.working_pressure_pa = P_ATM_PA
        else:
            self.working_pressure_pa = self._pressure_at(self.x_m)
        gas_force_n = (self.working_pressure_pa - P_ATM_PA) * area
        weight_n = -self.piston_mass_kg * GRAVITY_M_S2
        friction_n = -RACK_VISCOUS_N_PER_MPS * self.v_m_s
        friction_n -= math.copysign(RACK_FRICTION_N, self.v_m_s) if self.v_m_s != 0.0 else 0.0

        # both bodies integrated FREE of the ratchet coupling first --
        # the piston under its own real forces, the shaft under
        # whatever else is loading it
        v_free = self.v_m_s + (gas_force_n + weight_n + friction_n) / self.piston_mass_kg * dt
        # the real mechanical floor every caller already enforces on
        # THEIR OWN copy of the shaft speed AFTER this call returns
        # (engine_cycle_sim.py's self._omega = max(0.0, new_omega)) --
        # applied HERE too, not just after the fact: this shaft
        # genuinely cannot spin backward (its own one-way ratchet/
        # backstop), so a transient decelerating load can coast it to a
        # real stop, never past it. Missing this floor let omega_free go
        # deeply negative under a sustained load with no real
        # counterpart, and the momentum-conservation catch below would
        # then drag the much lighter piston along with that unphysical
        # reverse spin -- found empirically: it nearly doubled the
        # piston's own real, load-independent peak height.
        omega_free = max(0.0, shaft_omega - shaft_load_torque_nm / max(shaft_inertia_kg_m2, 1e-9) * dt)

        r = self.pinion_radius_m
        # the real ratchet convention: downward piston motion (v<0)
        # is the driving sense -- define drive speed so it's positive
        # exactly when the piston is trying to spin the shaft forward
        drive_speed_free = -v_free / r

        # a real one-way ratchet can only ever catch while the piston is
        # actually in its driving sense (drive_speed_free > 0, i.e.
        # genuinely descending) -- comparing relative speed ALONE (the
        # line below on its own) missed this: if an external load drove
        # the shaft's own free omega sharply negative, even a piston
        # still ASCENDING (drive_speed_free mildly negative) could read
        # as "faster" than the shaft and spuriously catch. Found
        # empirically: a real catch was firing right at the piston's own
        # natural peak height while it was still moving upward, then
        # rigidly dragging it along with the still-decelerating shaft to
        # nearly double its real, unloaded peak -- a pawl grabbing a gear
        # tooth moving away from it, which a real ratchet cannot do.
        if drive_speed_free <= 0.0 or drive_speed_free <= omega_free:
            # overrun: free motion never asked to go faster than the
            # shaft in the driving sense -- no catch, real freewheel,
            # each body keeps its own free result untouched
            if self.locked and self.phase == "free_flight" and v_free < 0.0:
                # the ratchet was just doing useful work and has now
                # fallen behind the flywheel -- the real cam trip:
                # open the exhaust valve for the rest of this descent
                self.phase = "exhausting"
            self.locked = False
            self.v_m_s = v_free
            new_shaft_omega = omega_free
        else:
            # catch: solve the exact shared speed the two real inertias
            # settle to by angular-momentum conservation, both
            # reflected into the shaft's own rotational domain (the
            # piston's linear inertia becomes m*r^2 there) -- a real
            # rigid ratchet lock, not a force-integrated spring
            j_piston_reflected = self.piston_mass_kg * r * r
            j_total = shaft_inertia_kg_m2 + j_piston_reflected
            omega_common = (shaft_inertia_kg_m2 * omega_free
                           + j_piston_reflected * drive_speed_free) / j_total
            # clamp to the ratchet's own real torque rating: if closing
            # the gap this step would need more impulse than the
            # mechanism can actually carry, it slips instead of
            # catching cleanly (a genuine overload case, not just a
            # numerical safety net)
            impulse_needed_nm_s = shaft_inertia_kg_m2 * (omega_common - omega_free)
            impulse_cap_nm_s = self.max_shaft_torque_nm * dt
            if abs(impulse_needed_nm_s) > impulse_cap_nm_s:
                omega_common = omega_free + math.copysign(impulse_cap_nm_s, impulse_needed_nm_s) / shaft_inertia_kg_m2
            self.locked = True
            new_shaft_omega = omega_common
            self.v_m_s = -omega_common * r

        self.x_m += self.v_m_s * dt

        # real hard limits: the rack/pinion mesh only spans the stroke
        # -- a real bracket physically arrests the piston past that,
        # not a mathematical clamp. Hitting it while still moving
        # upward is a genuine impact: real, mostly-lost energy (heat,
        # sound, deformation) plus a real small rebound, off the stop's
        # own disclosed coefficient of restitution (STROKE_STOP_
        # RESTITUTION) -- not an unconditional "velocity becomes
        # exactly zero," which would silently delete real kinetic
        # energy into nothing rather than dissipating it.
        self.last_stroke_stop_impact_speed_m_s = 0.0
        if self.x_m >= self.stroke_m:
            self.x_m = self.stroke_m
            if self.v_m_s > 0.0:
                self.last_stroke_stop_impact_speed_m_s = self.v_m_s
                if self.v_m_s >= self.destructive_impact_speed_m_s:
                    # the real documented failure case -- not a bounce.
                    # See STROKE_STOP_DESTRUCTIVE_SPEED_M_S's own
                    # comment: a real power-stroke slam into the frame
                    # destroyed full-size engines historically, and this
                    # sim treats it the same way -- the piston stops
                    # for good, right here, mid-stroke.
                    self.destroyed = True
                    self.v_m_s = 0.0
                    self.phase = "destroyed"
                else:
                    self.v_m_s = -self.v_m_s * STROKE_STOP_RESTITUTION
        # piston has returned to (or below) ignition height on its way
        # down, valve already open (or never needed to trap anything if
        # it fell straight through, e.g. an unloaded/no-shaft test) --
        # arm the next charge
        if self.x_m <= self.ignition_height_m and self.phase in ("free_flight", "exhausting") and self.v_m_s <= 0.0:
            self.x_m = self.ignition_height_m
            self.v_m_s = 0.0
            self.working_pressure_pa = P_ATM_PA
            self.phase = "awaiting_ignition"
            self.locked = False
            self.cycle_count += 1

        return new_shaft_omega
