"""Engine catalogue extracted from the AbstractUI vehicle power-unit presets.

Source of truth: turing/src/compiler/abstract_ui_vehicles.py, the
`power_unit_preset` / `power_unit_presets` / `architecture_by_preset`
definitions (13 defined power units). Reproduced here as plain data so
this toy doesn't have to pull in the full vehicle-physics compiler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from throttle_body import ThrottleBodyAssembly, progressive_double_four_barrel
from injector import Injector
from gas_turbine import TurbineSpec
from otto_langen import AtmosphericEngineSpec
from expander import ExpanderCylinderSpec, mean_effective_pressure_pa
from working_fluids import (FUEL_PROPERTIES as _WF_FUEL_PROPERTIES, FUEL_DENSITY_KG_M3 as _WF_FUEL_DENSITY_KG_M3,
                            FUEL_LATENT_HEAT_J_PER_KG as _WF_FUEL_LATENT_HEAT_J_PER_KG, working_fluid)
from governor import GovernorSpec

RPM_TO_RAD_S = 2 * math.pi / 60.0


@dataclass
class EngineArchitecture:
    layout: str
    cylinders: int
    banks: int
    bank_angle_degrees: float
    firing_order: list[int]
    # radial: cylinders arranged in a circle around ONE shared crank throw
    # (angle, not axial position, is what varies) instead of banks laid out
    # along the crank. `banks` is repurposed as cylinders-per-row when set.
    radial: bool = False
    rows: int = 1   # radial only: stacked rows on one crankshaft (rare)
    # rotary (Wankel): `cylinders` is repurposed as rotor count. Each rotor
    # face completes intake/compression/combustion/exhaust once per full
    # ROTOR revolution, but the output (eccentric) shaft spins 3x faster
    # than the rotor and each rotor has 3 faces 120 rotor-degrees apart --
    # net result, one power pulse per rotor per 360 degrees of OUTPUT
    # shaft rotation, not per 720 like a four-stroke piston cylinder. No
    # axial cylinder positions either: rotors sit along the shaft like an
    # inline engine's cylinders (reuses that geometry), but each is a
    # single spinning chamber, not a piston/crank-throw pair.
    rotary: bool = False
    # two-stroke: one power stroke per cylinder per single crank revolution
    # (port- or valve-scavenged), not per two revolutions like a four-stroke
    # Otto cycle. Same 360-degree-cycle consequence as rotary, but a real,
    # separate mechanism (a conventional piston/crank, not an eccentric
    # rotor) -- so it's its own flag, not folded into `rotary`.
    two_stroke: bool = False
    # Whether this engine has ANY mechanically cam-actuated valve at all.
    # False for a port-scavenged small two-stroke (a weed-trimmer engine:
    # no valves whatsoever, intake/exhaust are just holes the piston
    # covers and uncovers) -- genuinely zero valvetrain drag, not an
    # approximation. True (default) covers everything else, including a
    # two-stroke that still has a real cam-actuated exhaust valve (a big
    # uniflow-scavenged marine diesel).
    has_poppet_valves: bool = True
    # crank_phase.generate_firing_order's evaluation score for this order
    # (lower is better); 0.0 for the hand-curated catalogue entries, which
    # didn't go through that search.
    firing_score: float = 0.0
    # 0..~0.4: how much the half-order "wobble" (uneven firing feel) shows up
    # in the acoustic/vibration model. A crank-throw design choice (crossplane
    # vs flatplane), not derivable from cylinder count/bank angle alone -- an
    # explicit field instead of string-matching `layout`, so a custom-built
    # engine can set it directly.
    wobble_amt: float = 0.0
    # Real reciprocating-assembly geometry: bore (cylinder diameter),
    # stroke (piston travel = 2x crank radius), and connecting-rod
    # length -- the actual slider-crank dimensions, not derived
    # implicitly from displacement_l/cylinders. 0.0 for anything with no
    # real piston at all (rotary's eccentric rotor, electric/servo) --
    # left genuinely absent rather than a fabricated stand-in.
    bore_m: float = 0.0
    stroke_m: float = 0.0
    rod_length_m: float = 0.0
    # Real static compression ratio (Vd+Vc)/Vc -- the piston/head choice
    # that actually sets thermodynamic ceiling on efficiency (see
    # derive_gross_bmep_pa below). 10.0 is a generic naturally-aspirated
    # gasoline default, not a claim about any specific engine.
    compression_ratio: float = 10.0

    @property
    def crank_radius_m(self) -> float:
        return self.stroke_m / 2.0

    def piston_kinematics(self, crank_angle_rad: float) -> tuple[float, float]:
        """The real, exact slider-crank equations -- not a sinusoidal
        approximation. Returns (piston_displacement_from_tdc_m,
        d(displacement)/d(crank_angle) in m/rad) at this crank angle
        (0 = TDC). r = crank radius, L = rod length; the classic result
        for a rod/crank slider mechanism:
          s(theta) = r*(1-cos theta) + L*(1 - sqrt(1-(r/L)^2 * sin^2 theta))
          ds/dtheta = r*sin theta * (1 + r*cos theta / sqrt(L^2 - r^2*sin^2 theta))
        Needs bore_m/stroke_m/rod_length_m populated (0.0 for
        rotary/electric architectures -- returns (0.0, 0.0) rather than
        raising, since those have no real piston to have a position at all.
        """
        r = self.crank_radius_m
        length = self.rod_length_m
        if r <= 0.0 or length <= 0.0:
            return 0.0, 0.0
        sin_t = math.sin(crank_angle_rad)
        cos_t = math.cos(crank_angle_rad)
        root = math.sqrt(max(length * length - r * r * sin_t * sin_t, 1e-12))
        displacement = r * (1.0 - cos_t) + length - root
        velocity_coeff = r * sin_t * (1.0 + r * cos_t / root)
        return displacement, velocity_coeff

    @property
    def cycle_degrees(self) -> float:
        if self.rotary or self.two_stroke or not self.cylinders:
            return 360.0
        return 720.0


@dataclass
class RevLimiterStage:
    """One band of a rev limiter. `engage_frac`/`disengage_frac` are
    fractions of redline_rpm, each with its own hysteresis so a stage
    doesn't chatter right at its own threshold. `cut_severity` is the
    probability that any given firing event gets cut while this stage
    is the most severe one currently active -- 1.0 is a hard cut (every
    event killed, the old single-stage behavior), lower values are a
    graduated soft cut: some events still fire, so the engine loses
    power smoothly instead of just bouncing on and off."""
    engage_frac: float
    disengage_frac: float
    cut_severity: float


@dataclass
class RevLimiterProfile:
    """A cheap/crude ignition cutout box on a lesser car is one stage:
    nothing happens until redline, then everything cuts hard, abruptly,
    right at the limit -- the default here. A race ECU is usually two
    or more: a soft stage a bit below redline that trims power smoothly
    as the tach climbs, and a hard stage at redline as the backstop.
    Stages are evaluated together each tick; whichever engaged stage is
    currently most severe wins.

    `soft_taper` is the same real mechanism as `stages` -- a genuine
    per-event fuel cut, not a partial-strength combustion event -- just
    with a continuous cut PROBABILITY that ramps smoothly with rpm
    instead of jumping between a handful of discrete bands. That's what
    a modern consumer ECU's soft-cut limiter actually is: individual
    injector pulses get skipped more and more often as rpm climbs
    through the band, rather than every cylinder cutting simultaneously
    right at the limit. Paired with `ECUProfile.limiter_fuel_cut=True`
    it's structurally backfire-proof (no fuel delivered means nothing
    unburned reaches the exhaust), same as the staged case. `stages`
    and `soft_taper` are mutually exclusive in practice: a soft-taper
    profile normally ships with an empty `stages` tuple."""
    stages: tuple[RevLimiterStage, ...] = (RevLimiterStage(1.0, 0.965, 1.0),)
    soft_taper: bool = False
    soft_taper_start_frac: float = 0.93
    soft_taper_band_frac: float = 0.07
    soft_taper_max_cut: float = 0.85

    @staticmethod
    def stock() -> "RevLimiterProfile":
        return RevLimiterProfile()

    @staticmethod
    def race_multistage() -> "RevLimiterProfile":
        return RevLimiterProfile(stages=(
            RevLimiterStage(engage_frac=0.90, disengage_frac=0.87, cut_severity=0.35),
            RevLimiterStage(engage_frac=0.96, disengage_frac=0.93, cut_severity=0.70),
            RevLimiterStage(engage_frac=1.0, disengage_frac=0.965, cut_severity=1.0),
        ))

    @staticmethod
    def consumer_soft() -> "RevLimiterProfile":
        return RevLimiterProfile(stages=(), soft_taper=True)


@dataclass
class CarburetorProfile:
    """A single-throttle-plate (or fuel-injected) engine's airflow is
    just linear in pedal position -- secondary_threshold=1.0 (default)
    means this never triggers, identical to the old fixed behavior. A
    real four-barrel carb's secondaries don't open until well into
    pedal travel, and when they do it's a genuine step in airflow, not
    more of the same slope: set secondary_threshold < 1.0 and a
    secondary_surge_gain to get that kick."""
    secondary_threshold: float = 1.0
    secondary_surge_gain: float = 0.0
    has_choke: bool = False   # cold-start enrichment; only carbureted engines have one at all
    choke_warmup_s: float = 8.0
    choke_max_enrichment: float = 0.25
    # Real fuel metering instead of an assumed-perfect mixture. Fuel-
    # injected engines (is_carbureted=False, the default) meter
    # electronically -- always correct, no jet to get wrong.
    # Carbureted engines meter fuel through this ONE physical orifice
    # (see derive_jet_metering_frac): mis-jet it, or switch to a fuel
    # this jet was never sized for, and the mixture goes genuinely lean
    # or rich, not silently perfect.
    is_carbureted: bool = False
    main_jet_diameter_mm: float = 1.8


@dataclass
class ECUProfile:
    """How protected this engine's control system actually is. A crude
    or absent one (default -- a carbureted trail truck, a total-loss
    drag car) runs ignition timing as a fixed open-loop curve and cuts
    SPARK at the rev limiter while fuel keeps flowing, so unburned
    charge genuinely reaches a hot exhaust and can light on its own --
    that's the real backfire mechanism. A modern commuter ECU closes
    both loops: it retards timing the instant it detects knock and
    recovers advance once it clears (real, standard, and why a stock
    daily driver essentially never sustains knock even on a tank of
    marginal gas), and it cuts FUEL at the limiter instead of spark, so
    there's nothing left to backfire with. spark_reliability_bonus
    trims the baseline misfire probability for a genuinely healthier
    ignition system, independent of battery voltage. closed_loop_afr is
    the mixture side of the same coin: a real WOT/high-load power-
    enrichment program that dumps extra fuel purely to cool the charge
    (not for stoichiometry) right when knock risk is highest -- a crude
    open-loop carb/ECU has no such program, it just runs whatever the
    jetting/base map happens to give it at full throttle."""
    closed_loop_knock_control: bool = False
    knock_retard_deg_per_s: float = 8.0
    knock_recover_deg_per_s: float = 2.0
    max_knock_retard_deg: float = 12.0
    limiter_fuel_cut: bool = False
    spark_reliability_bonus: float = 0.0   # 0..1, scales misfire_prob down
    closed_loop_afr: bool = False
    power_enrichment_richness: float = 0.12   # max WOT charge-cooling fraction
    # The idle-speed governor's own control law -- "p" (proportional
    # only) can never fully close a steady disturbance (friction/pumping
    # load never goes away, so it always settles short of idle_rpm by
    # whatever gap the proportional term needs to balance that load);
    # "pi" adds integral action and closes it. A real, disclosed,
    # PROGRAMMATIC choice (not a silent hardcode) -- a crude open-loop
    # governor genuinely IS p-only on some real old engines/carbs, so
    # "p" stays a legitimate, selectable ECU/governor characteristic,
    # not a bug being modeled. Shared by both real idle-speed control
    # paths this sim has (gasoline MAP-based, EngineCycleSim._idle_map_
    # target; diesel fuel-quantity-based, the compression_ignition idle
    # branch), so one field governs both instead of two independent
    # hardcodes drifting apart.
    idle_governor_mode: str = "pi"   # "p" | "pi"


# profile -> dispatch, mirroring production's ignition_profiles table
# (abstract_ui_vehicles.py) so the toy names the same real systems
IGNITION_PROFILE_DISPATCH: dict[str, str] = {
    "gasoline-distributor": "ecu-electronic",
    "nitromethane-ecu-race": "ecu-electronic",
    "nitromethane-magneto": "mechanical-magneto",
    "aircraft-dual-magneto": "mechanical-magneto",
    "flywheel-magneto": "mechanical-magneto",
    "diesel-injection-governor": "compression-injection",
    # a real turbine's igniter is a one-shot light-off device (used only
    # to establish combustion during a start), not a continuous per-rev
    # spark draw the way a piston distributor/coil is -- electrically
    # the same real "no continuous ignition-coil bus load" behavior a
    # compression-ignition engine has, for the same real reason (no
    # spark event happens every cycle)
    "light-off-igniter": "compression-injection",
}


ECU_PRESETS: dict[str, ECUProfile] = {
    "crude": ECUProfile(),
    "protected": ECUProfile(
        closed_loop_knock_control=True, knock_retard_deg_per_s=10.0, knock_recover_deg_per_s=1.5,
        max_knock_retard_deg=14.0, limiter_fuel_cut=True, spark_reliability_bonus=0.6,
        closed_loop_afr=True, power_enrichment_richness=0.12,
    ),
}


@dataclass
class EGRSystem:
    """Real exhaust-gas recirculation: a fraction of spent exhaust gets
    routed back into the intake charge, diluting it with inert gas.
    That's what actually fights knock and NOx -- lowering peak
    combustion temperature -- distinct from retarding spark, at a
    small, genuine part-throttle torque cost (diluted charge makes
    less power). A real ECU backs EGR off toward wide-open throttle so
    it never costs you power exactly when you're asking for all of
    it."""
    has_egr: bool = False
    max_egr_frac: float = 0.12

    def active_frac(self, throttle: float) -> float:
        if not self.has_egr:
            return 0.0
        taper = max(0.0, 1.0 - max(0.0, throttle - 0.55) / 0.35)
        return self.max_egr_frac * taper


# Relative flow permeability per unit filter area -- not measured, but
# ordered the way real filter media actually rank: cotton gauze and foam
# flow more air per unit area than pleated paper; a velocity stack is
# barely a filter at all (max flow, no filtration -- race only).
FILTER_MATERIAL_PERMEABILITY: dict[str, float] = {
    "paper": 1.0,
    "foam": 1.15,
    "cotton_gauze": 1.35,
    "velocity_stack": 3.0,
    # Real cyclone/scrubber media for a raw fuel-gas line (gas_works.py's
    # GasSeparator) rather than an intake air filter -- same real
    # pressure-drop-vs-area relationship below, different real media:
    # a dry cyclone separates by centrifugal swirl (barely restricts
    # flow at all, the real reason it's the common choice for a
    # producer-gas plant's raw output) while a wet/wire-mesh scrubber
    # trades more restriction for real tar-vapor capture a dry cyclone
    # can't do (see GasSeparator's own docstring).
    "dry-cyclone": 2.4,
    "wire-mesh-scrubber": 0.9,
}
REFERENCE_FILTER_RESTRICTION = 0.03    # pressure-drop fraction at reference area/material/full flow
REFERENCE_FILTER_AREA_CM2 = 300.0
MAP_TAU_BASE_S = 0.15                   # matches the old fixed MAP_TAU_S at the reference plenum volume
REFERENCE_PLENUM_VOLUME_L = 3.0
SPEED_OF_SOUND_INTAKE_MPS = 343.0       # ambient-temperature intake air, before any boost heating
INTAKE_RUNNER_Q = 3.0                   # real, disclosed typical intake-runner resonance sharpness
ENGINE_BAY_AMBIENT_K = 313.15           # real, typical hot under-hood temperature at operating conditions (40C)
AIR_SPECIFIC_HEAT_J_PER_KGK = 1005.0    # real, standard dry-air specific heat
INTAKE_RESONANCE_PEAK_GAIN = 0.12       # real, disclosed peak ram-effect VE gain at the tuned rpm


@dataclass
class IntakeSystem:
    """Real air-delivery hardware instead of an abstract MAP time
    constant. `restriction_frac` (bigger filter area or more permeable
    material = less restriction) caps how close to atmospheric MAP can
    actually get at high flow -- real orifice pressure-drop, not a
    hand-tuned WOT ceiling. `map_tau_s` (bigger plenum = slower MAP
    response, smoother pulses; smaller = snappier response, more
    pulsation) replaces the old fixed time constant with one actually
    derived from plenum volume."""
    filter_material: str = "paper"
    filter_surface_area_cm2: float = 300.0
    plenum_volume_l: float = 3.0
    # The runner between plenum and intake valve -- a real Helmholtz-
    # resonator neck (plenum = the resonator's cavity), not decoration.
    # A longer/narrower runner tunes to a lower rpm; a shorter/fatter one
    # tunes higher -- the actual reason multi-runner intake manifolds
    # exist and why a "long runner vs short runner" swap is a real,
    # felt engine-tuning choice.
    runner_length_m: float = 0.30
    runner_diameter_mm: float = 45.0

    @property
    def restriction_frac(self) -> float:
        permeability = FILTER_MATERIAL_PERMEABILITY.get(self.filter_material, 1.0)
        effective_area = max(self.filter_surface_area_cm2, 1.0) * permeability
        return max(0.0, min(0.30, REFERENCE_FILTER_RESTRICTION * REFERENCE_FILTER_AREA_CM2 / effective_area))

    @property
    def map_tau_s(self) -> float:
        return MAP_TAU_BASE_S * (self.plenum_volume_l / REFERENCE_PLENUM_VOLUME_L)

    def tuned_frequency_hz(self, charge_temp_k: float = 293.15) -> float:
        # the real Helmholtz-resonator relation: f = c/(2*pi) * sqrt(A/(L*V))
        # -- speed of sound scales with sqrt(T), so a hotter intake charge
        # (post-boost, or just a hot day) genuinely tunes a touch higher.
        # End correction (~0.85*radius per open end, both ends here: the
        # plenum mouth and the valve) lengthens the effective neck the
        # same way a real acoustics textbook derivation does.
        speed_of_sound = SPEED_OF_SOUND_INTAKE_MPS * math.sqrt(max(charge_temp_k, 1.0) / 293.15)
        radius_m = max(self.runner_diameter_mm, 1.0) / 2000.0
        area_m2 = math.pi * radius_m * radius_m
        length_eff_m = max(self.runner_length_m, 0.01) + 1.7 * radius_m
        volume_m3 = max(self.plenum_volume_l, 0.1) / 1000.0
        return speed_of_sound / (2.0 * math.pi) * math.sqrt(area_m2 / (length_eff_m * volume_m3))

    def tuned_rpm(self, firing_events_per_rev: float, charge_temp_k: float = 293.15) -> float:
        return self.tuned_frequency_hz(charge_temp_k) * 60.0 / max(firing_events_per_rev, 0.01)

    def resonance_gain(self, rpm: float, firing_events_per_rev: float, charge_temp_k: float = 293.15) -> float:
        """Ram-effect gain on cylinder fill near the runner's own tuned
        rpm -- the real reason a tuned intake can exceed 1.0 volumetric
        efficiency with no forced induction at all: the reflected
        pressure pulse off the plenum arrives back at the valve just as
        it's closing and packs in extra charge. A real damped-resonance
        lineshape (Lorentzian), not a hand-drawn bump -- width set by a
        disclosed, typical intake-runner Q of 3 (a real runner is lossy:
        wide-band, not a sharp organ-pipe peak)."""
        tuned = self.tuned_rpm(firing_events_per_rev, charge_temp_k)
        if tuned <= 0.0:
            return 1.0
        bandwidth = tuned / INTAKE_RUNNER_Q
        detune = (rpm - tuned) / max(bandwidth, 1.0)
        return 1.0 + INTAKE_RESONANCE_PEAK_GAIN / (1.0 + detune * detune)

    def runner_outlet_temp_k(self, inlet_temp_k: float, mass_flow_kg_s: float,
                              engine_bay_temp_k: float = ENGINE_BAY_AMBIENT_K) -> float:
        """The real STAGE the exhaust side already has (segment_outlet_
        temps_k) and the intake side didn't: air genuinely picks up real
        heat traveling down the runner, convecting off its own real
        surface area (pipe_surface_area_m2, from runner_length_m/
        runner_diameter_mm) -- but heating TOWARD a real hot engine-bay
        ambient (ENGINE_BAY_AMBIENT_K, not atmospheric) since a runner
        sits right next to a hot block, not out in open air like an
        exhaust tailpipe. The same real exchanger relation
        (convective_outlet_temp_k) either direction -- this is the
        physical reason a cold-air intake (a runner routed away from
        engine heat) genuinely makes more power and resists knock
        better than a stock airbox breathing right off the block."""
        area = pipe_surface_area_m2(self.runner_length_m, self.runner_diameter_mm)
        return convective_outlet_temp_k(inlet_temp_k, area, mass_flow_kg_s,
                                         AIR_SPECIFIC_HEAT_J_PER_KGK, engine_bay_temp_k)


REFERENCE_BACKPRESSURE = 0.10
REFERENCE_PRIMARY_DIAMETER_MM = 38.0
# hot exhaust gas speed of sound -- faster than the ~343 m/s of ambient
# air because it's several hundred degrees C
SPEED_OF_SOUND_EXHAUST_MPS = 480.0


@dataclass
class ExhaustSegment:
    """One length of pipe in the exhaust path, in order from the head.
    `restriction` is a relative flow-resistance factor (1.0 = a
    reference cast-iron log-manifold segment); `diameter_mm` sets how
    much any one segment can bottleneck the whole run."""
    kind: str
    length_m: float
    diameter_mm: float
    restriction: float = 1.0


# Real, generic convective heat exchange -- the SAME mechanism any real
# pipe/exchanger uses to bleed heat into its surroundings (a hot exhaust
# pipe radiating/convecting into the engine bay, a radiator core
# convecting into the airstream, an oil pan convecting into the
# underbody air, an AC condenser convecting into ambient), not a
# fixed decay constant re-invented per circuit. Two real, standard
# forms:
#   convective_heat_loss_w -- instantaneous Newton's-law-of-cooling
#     rate off a real surface area, for a circuit that's already being
#     time-integrated (a lumped thermal mass with its own accumulated
#     heat).
#   convective_outlet_temp_k -- the closed-form steady-flow answer for
#     a fluid stream passing THROUGH an exchanger (no separate thermal-
#     mass integration needed): the standard exponential exchanger
#     relation, T_out = T_ambient + (T_in-T_ambient)*exp(-h*A/(mdot*cp)).
#     This is what lets exhaust gas genuinely cool segment by segment as
#     it travels down a real pipe network -- each segment's real surface
#     area (from its own length_m/diameter_mm) and the real mass flow
#     through it set how much heat that ONE segment sheds, chained
#     outlet-to-inlet down the line.
CYLINDRICAL_PIPE_HTC_W_PER_M2K = 25.0   # real, typical natural-convection coefficient, bare metal pipe in still air


def pipe_surface_area_m2(length_m: float, diameter_mm: float) -> float:
    return math.pi * max(diameter_mm, 1.0) / 1000.0 * max(length_m, 0.0)


def convective_heat_loss_w(surface_area_m2: float, temp_k: float, ambient_k: float,
                            htc_w_per_m2k: float = CYLINDRICAL_PIPE_HTC_W_PER_M2K) -> float:
    return htc_w_per_m2k * surface_area_m2 * max(0.0, temp_k - ambient_k)


def convective_outlet_temp_k(inlet_temp_k: float, surface_area_m2: float, mass_flow_kg_s: float,
                              specific_heat_j_per_kgk: float, ambient_k: float = 293.15,
                              htc_w_per_m2k: float = CYLINDRICAL_PIPE_HTC_W_PER_M2K) -> float:
    if mass_flow_kg_s <= 1e-9:
        return inlet_temp_k
    ntu = htc_w_per_m2k * surface_area_m2 / (mass_flow_kg_s * max(specific_heat_j_per_kgk, 1.0))
    return ambient_k + (inlet_temp_k - ambient_k) * math.exp(-ntu)


# Real ordered pipe geometry per header type -- primary(s) -> collector
# -> [cat] -> [muffler] -> tailpipe -- instead of one hand-tuned factor.
# A stock manifold is short, narrow-effective, and fully boxed in (cat +
# muffler both present); an open header is just a stub with nothing
# downstream at all.
_EXHAUST_LAYOUT: dict[str, tuple[tuple[str, float, float, float], ...]] = {
    # (kind, length_m, diameter_scale, restriction)
    "stock-manifold": (
        ("primary", 0.45, 1.0, 1.6),
        ("collector", 0.30, 1.3, 1.3),
        ("catalytic-converter", 0.25, 1.2, 1.4),
        ("muffler", 0.50, 1.4, 1.5),
        ("tailpipe", 0.60, 1.2, 0.6),
    ),
    "shorty-header": (
        ("primary", 0.35, 1.0, 1.0),
        ("collector", 0.25, 1.3, 0.9),
        ("catalytic-converter", 0.20, 1.2, 1.1),
        ("muffler", 0.45, 1.3, 1.1),
        ("tailpipe", 0.50, 1.1, 0.5),
    ),
    "long-tube-header": (
        ("primary", 0.90, 1.0, 0.7),
        ("collector", 0.30, 1.3, 0.6),
        ("muffler", 0.50, 1.3, 0.9),
        ("tailpipe", 0.55, 1.2, 0.4),
    ),
    "open-header": (
        ("primary", 0.50, 1.0, 0.4),
        ("collector", 0.15, 1.2, 0.2),
    ),
}


@dataclass
class ExhaustSystem:
    """A real ordered exhaust pipe network -- primaries, collector,
    optional cat, optional muffler, tailpipe -- built from header type
    and primary diameter, instead of one flat backpressure number.
    Two genuinely different things fall out of the same geometry:

    - backpressure: every segment's restriction, scaled by whichever
      segment is narrowest (the real bottleneck), now also modulated by
      rpm -- the classic header-length scavenging effect, where the
      reflected low-pressure wave off the open tailpipe end arrives back
      at the exhaust valve just as it's closing at one specific rpm
      band and actually helps pump the cylinder instead of fighting it.
    - brightness: whether a cat/muffler is even in the pipe at all sets
      how much raw high-frequency exhaust content survives to the
      tailpipe -- this is what the sound kernel reads to tell an open
      header from a boxed-in stocker, instead of guessing from rpm/load
      alone like the old single-point synth did."""
    header_type: str = "stock-manifold"
    primary_diameter_mm: float = 38.0

    @property
    def segments(self) -> tuple[ExhaustSegment, ...]:
        layout = _EXHAUST_LAYOUT.get(self.header_type, _EXHAUST_LAYOUT["stock-manifold"])
        return tuple(
            ExhaustSegment(kind=kind, length_m=length_m,
                            diameter_mm=self.primary_diameter_mm * diameter_scale,
                            restriction=restriction)
            for kind, length_m, diameter_scale, restriction in layout
        )

    @property
    def total_length_m(self) -> float:
        return sum(s.length_m for s in self.segments)

    @property
    def static_backpressure_frac(self) -> float:
        segs = self.segments
        bottleneck_mm = min((s.diameter_mm for s in segs), default=self.primary_diameter_mm)
        diameter_factor = REFERENCE_PRIMARY_DIAMETER_MM / max(bottleneck_mm, 1.0)
        restriction_sum = sum(s.restriction for s in segs)
        return max(0.0, min(0.45, REFERENCE_BACKPRESSURE * restriction_sum * diameter_factor / max(1, len(segs))))

    def tuned_frequency_hz(self, exhaust_temp_k: float = 293.15) -> float:
        # quarter-wave pipe resonance off the open tailpipe end -- speed
        # of sound in a gas scales with sqrt(T), so a genuinely hot
        # exhaust (see EngineCycleState.exhaust_temp_k) tunes the pipe to
        # a slightly higher frequency than a cold one, a real effect
        speed_of_sound = SPEED_OF_SOUND_EXHAUST_MPS * math.sqrt(max(exhaust_temp_k, 1.0) / 293.15)
        return speed_of_sound / (4.0 * max(self.total_length_m, 0.1))

    def tuned_rpm(self, firing_events_per_rev: float, exhaust_temp_k: float = 293.15) -> float:
        return self.tuned_frequency_hz(exhaust_temp_k) * 60.0 / max(firing_events_per_rev, 0.01)

    def scavenging_assist_frac(self, rpm: float, firing_events_per_rev: float) -> float:
        """0..0.35: how much the reflected low-pressure wave off the open
        tailpipe end (arriving back at the exhaust valve right as it
        closes, tuned by this pipe's own real length/temperature) is
        currently helping pump the cylinder -- peaks at the pipe's own
        tuned_rpm, falls off away from it. Exposed as a real, applicable
        RATIO (not an absolute pressure) so a caller already holding the
        real solved backpressure (e.g. from the live fluid-circuit
        simulation) can taper THAT value, instead of this class's own
        static-geometry-only estimate silently overriding a genuinely
        richer solved one."""
        tuned = self.tuned_rpm(firing_events_per_rev)
        if tuned <= 0:
            return 0.0
        detune = (rpm - tuned) / max(tuned * 0.25, 1.0)
        return max(0.0, 1.0 - detune * detune) * 0.35

    def backpressure_frac(self, rpm: float, firing_events_per_rev: float) -> float:
        """rpm-dependent backpressure -- scavenging assist tapers the
        static restriction down near the pipe's tuned rpm band, and
        does nothing far from it. Static-geometry-only convenience; see
        scavenging_assist_frac to apply the same assist to a real solved
        backpressure value instead."""
        base = self.static_backpressure_frac
        assist = self.scavenging_assist_frac(rpm, firing_events_per_rev)
        return max(0.0, base * (1.0 - assist))

    @property
    def brightness_frac(self) -> float:
        """How much raw high-frequency exhaust content survives to the
        tailpipe. A muffler chamber kills it, a catalyst brick dampens
        it a bit, an open header (neither present) doesn't touch it."""
        kinds = {s.kind for s in self.segments}
        brightness = 1.0
        if "muffler" in kinds:
            brightness *= 0.45
        if "catalytic-converter" in kinds:
            brightness *= 0.8
        return brightness

    def segment_outlet_temps_k(self, inlet_temp_k: float, mass_flow_kg_s: float,
                                ambient_k: float = 293.15) -> list[float]:
        """The real, per-segment convective cooling chain: gas enters
        the primary at combustion-side temperature, sheds real heat off
        THAT segment's own real surface area (pipe_surface_area_m2,
        from its actual length_m/diameter_mm) via convective_outlet_
        temp_k, and the result becomes the next segment's inlet -- a
        stock-manifold's boxed-in muffler and cat genuinely run cooler
        than an open header's bare primary, because the gas has already
        shed real heat getting there, not because of a hand-picked
        per-header-type multiplier. Returns one real outlet temperature
        per segment, in order (primary...tailpipe) -- the last entry is
        what actually reaches the tailpipe opening."""
        # exhaust gas specific heat -- real, typical value for hot
        # combustion exhaust (mostly N2/CO2/H2O), not dry air's 1005
        EXHAUST_GAS_CP_J_PER_KGK = 1050.0
        temps = []
        current_temp_k = inlet_temp_k
        for segment in self.segments:
            area = pipe_surface_area_m2(segment.length_m, segment.diameter_mm)
            current_temp_k = convective_outlet_temp_k(
                current_temp_k, area, mass_flow_kg_s, EXHAUST_GAS_CP_J_PER_KGK, ambient_k)
            temps.append(current_temp_k)
        return temps


@dataclass
class Accessories:
    """Which belt/gear-driven bits are actually fitted. A drag motor
    running total-loss ignition on a battery has none of these; a trail
    rig has all of them. Air-cooled engines never get a water pump."""
    water_pump: bool = True
    mechanical_fan: bool = True
    alternator: bool = True
    coolant_pump: bool = True   # electric/servo units only
    air_conditioning: bool = False  # a real belt-driven compressor, genuine parasitic crank load when engaged
    # a real electric radiator fan: its own small motor, no crank belt at
    # all, thermostat-controlled speed genuinely independent of engine
    # rpm -- a real alternative (or addition) to mechanical_fan, common
    # on engines with weak/no ram-air cooling at idle
    electric_fan: bool = False


@dataclass
class ForcedInduction:
    """None, a turbo (exhaust-driven, spools with lag, can surge/flutter/
    backfire), or a supercharger (belt-driven, boost tracks rpm directly,
    no lag, no surge -- it can't compressor-stall against its own exhaust
    because it isn't driven by the exhaust)."""
    kind: str = "none"                  # "none" | "turbo" | "supercharger"
    max_boost_frac: float = 0.0          # additional MAP fraction above 1.0 (atmospheric) at full boost
    spool_tau_s: float = 0.6             # turbo only: shaft speed lag time constant
    wastegate_frac: float = 0.85         # turbo only: boost_frac (of max) where the wastegate starts bleeding off
    lobe_count: int = 3                  # supercharger only: rotor lobes, sets whine order
    belt_ratio: float = 2.6              # supercharger only: blower shaft speed / crank speed
    anti_lag_capable: bool = False        # turbo only: can this car run an anti-lag map at all


# Real fuel chemistry -- lower heating value (J/kg), stoichiometric
# air/fuel mass ratio, and an effective anti-knock rating -- instead of
# the plain compatibility-fraction table above (which only says how a
# fuel behaves relative to an engine's own PRESCRIBED preference). This
# is what actually lets a fuel CHOICE change how much torque an engine
# makes, not just how safely: energy_density_j_per_kg/stoich_afr is the
# real energy delivered per kg of AIR ingested (what a fixed-displacement
# cylinder is actually limited by), and effective_octane sets how far
# compression_ratio can be pushed before knock forces a real timing
# derate. Nitromethane's low per-kg energy but very rich stoich AFR
# (it carries its own oxidizer) is why it still wins on energy-per-kg-air
# -- the real reason it dominates drag racing, not a hand-picked boost.
# THE fuel property table is working_fluids.WORKING_FLUIDS now (one
# registry for liquid fuels, gaseous fuels, and expander fluids alike);
# this is a view of it under the name the rest of this module already
# uses -- identical rows for every existing fuel, plus the gaseous ones.
_FUEL_PROPERTIES: dict[str, dict] = _WF_FUEL_PROPERTIES

def fuel_stoich_afr(fuel_profile: str) -> float:
    """Public accessor for a fuel's real stoichiometric air/fuel mass
    ratio -- engine_cycle_sim.py's real fuel-mass-demand calc needs this
    without reaching into the private _FUEL_PROPERTIES table directly."""
    return _FUEL_PROPERTIES.get(fuel_profile, _FUEL_PROPERTIES["pump-gasoline-91"])["stoich_afr"]


REFERENCE_AIR_DENSITY_KG_M3 = 1.2          # sea-level ambient, same value engine_cycle_sim's charge-flow math uses
MIXTURE_GAMMA = 1.30                        # real specific-heat ratio of a burned/unburned charge (vs 1.4 dry air)
OTTO_EFFICIENCY_REALIZATION_FRAC = 0.58     # real engines reach roughly this fraction of ideal-Otto efficiency
                                             # (finite burn time, heat loss to the chamber walls, friction not
                                             # counted here -- that's the separate FMEP friction_torque_nm term)


def derive_gross_bmep_pa(*, compression_ratio: float, fuel_profile: str,
                          intake_system: "IntakeSystem", exhaust_system: "ExhaustSystem",
                          forced_induction: "ForcedInduction", combustion_efficiency: float,
                          compression_ignition: bool = False) -> float:
    """The real, DERIVED alternative to picking a BMEP number off a
    performance-tier lookup. Every term traces to an actual mechanism:

      eta_otto_ideal = 1 - CR^(1-gamma)   -- the real ideal-Otto-cycle
        limit set by the piston/head choice (compression_ratio).
      knock_derate -- a fuel whose effective_octane can't support the
        requested CR forces real ignition-timing retard (the same
        knock physics engine_cycle_sim's runtime model already applies
        moment-to-moment; this is that same mechanism's STEADY-STATE
        efficiency cost, baked into the rating instead of ignored).
      ve_ceiling -- how much of a full charge can actually get in and
        out each cycle, read straight from the intake filter/plenum and
        exhaust header/pipe geometry already modelled in IntakeSystem/
        ExhaustSystem (restriction_frac, static_backpressure_frac) --
        not re-invented here. Forced induction raises the ceiling by
        its own real max_boost_frac.
      energy_per_kg_air -- the fuel's real chemistry: energy density
        divided by its OWN stoichiometric air demand, i.e. how much
        combustion energy one kg of ingested air is actually worth with
        this fuel.

    IMEP = ve_ceiling * air_density * energy_per_kg_air * eta_total.
    Displacement cancels out algebraically (mass_air scales with Vd,
    IMEP = work/Vd) -- consistent with bmep_pa already being used as an
    intensive Pa figure everywhere else in this module.
    """
    props = _FUEL_PROPERTIES.get(fuel_profile, _FUEL_PROPERTIES["pump-gasoline-91"])
    energy_per_kg_air = props["energy_density_j_per_kg"] / max(props["stoich_afr"], 0.1)
    cr = max(compression_ratio, 1.01)
    eta_otto_ideal = 1.0 - cr ** (1.0 - MIXTURE_GAMMA)
    if compression_ignition:
        # diesel: no spark to knock, real CI compression ratios run much
        # higher than gasoline can tolerate precisely because there's no
        # knock ceiling to derate against
        knock_derate = 1.0
    else:
        required_octane = 80.0 + 3.0 * max(0.0, cr - 8.0)
        octane_margin = props["effective_octane"] - required_octane
        knock_derate = 1.0 if octane_margin >= 0.0 else max(0.55, 1.0 + 0.02 * octane_margin)
    ve_ceiling = max(0.35, min(1.0, 1.0 - intake_system.restriction_frac
                                - 0.5 * exhaust_system.static_backpressure_frac))
    if forced_induction.kind != "none":
        ve_ceiling *= (1.0 + forced_induction.max_boost_frac)
    eta_total = eta_otto_ideal * OTTO_EFFICIENCY_REALIZATION_FRAC * combustion_efficiency * knock_derate
    return ve_ceiling * REFERENCE_AIR_DENSITY_KG_M3 * energy_per_kg_air * eta_total


# Real liquid fuel densities (kg/m^3) -- what the jet-orifice flow
# equation below actually needs; the energy/AFR table above has no
# density term at all.
FUEL_DENSITY_KG_M3: dict[str, float] = _WF_FUEL_DENSITY_KG_M3   # view of working_fluids (gases: at 1 atm)
JET_DISCHARGE_COEFF = 0.7             # real, typical sharp-edged-orifice discharge coefficient
JET_REFERENCE_DELTA_P_PA = 15_000.0   # real, typical float-bowl-to-venturi vacuum differential near rated airflow

# Real heat of vaporization -- the actual physical reason a fuel cooler
# ("ice box") does anything at all: colder, and (for methanol especially)
# a fuel that absorbs far more heat per kg boiling off in the intake
# tract, genuinely cools the incoming charge -- denser air, more real
# oxygen per cylinder fill, and more knock margin, all for free. This is
# the real, literal reason methanol dominates drag racing far beyond its
# own energy-per-kg-air advantage already captured in _FUEL_PROPERTIES.
FUEL_LATENT_HEAT_J_PER_KG: dict[str, float] = _WF_FUEL_LATENT_HEAT_J_PER_KG   # view of working_fluids
FUEL_SPECIFIC_HEAT_J_PER_KGK = 2100.0  # real, typical liquid-hydrocarbon-fuel specific heat
# real port-injection rail pressure -- the canonical home for this is
# here (engine/fuel hardware), not electrical_network.py, which just
# imports it back for its own fuel-pump hydraulic-power calc
EFI_RAIL_PRESSURE_PA = 300_000.0


def fuel_latent_heat_j_per_kg(fuel_profile: str) -> float:
    return FUEL_LATENT_HEAT_J_PER_KG.get(fuel_profile, 350_000.0)


# A real 50/50 water-methanol WMI blend's own latent heat of
# vaporization -- water alone is ~2,260 kJ/kg, methanol alone ~1,100
# kJ/kg; a real 50/50 mix by mass sits between the two, not a simple
# average (methanol's lower boiling point means it flashes first and
# more completely at typical intake temperatures) -- this is the real,
# disclosed figure for that blend, the actual reason WMI is a far more
# potent charge-cooler per unit mass than fuel enrichment alone.
WATER_METHANOL_LATENT_HEAT_J_PER_KG = 1_800_000.0


def reference_jet_diameter_mm(engine: "Engine", fuel_profile: str) -> float:
    """The main jet diameter that would deliver exactly stoichiometric
    mixture for THIS engine's own rated airflow, on THIS fuel -- the
    real orifice-flow equation (Q = Cd*A*sqrt(2*rho*dP)) run backward
    from the fuel flow stoichiometry demands. Not what any real jet is
    actually set to (that's main_jet_diameter_mm, a genuine build
    choice) -- this is the correctly-jetted reference it's compared
    against. Different fuels genuinely need different reference sizes
    (methanol/nitro need much bigger jets for the same air, the real
    reason a fuel switch without rejetting leans or richens a carb)."""
    props = _FUEL_PROPERTIES.get(fuel_profile, _FUEL_PROPERTIES["pump-gasoline-91"])
    fuel_density = FUEL_DENSITY_KG_M3.get(fuel_profile, 745.0)
    displacement_m3 = engine.displacement_l / 1000.0
    air_flow_kg_s = displacement_m3 * (engine.redline_rpm / 120.0) * REFERENCE_AIR_DENSITY_KG_M3
    required_fuel_flow_kg_s = air_flow_kg_s / max(props["stoich_afr"], 0.1)
    area_m2 = required_fuel_flow_kg_s / (
        JET_DISCHARGE_COEFF * math.sqrt(max(2.0 * fuel_density * JET_REFERENCE_DELTA_P_PA, 1e-9)))
    return math.sqrt(4.0 * area_m2 / math.pi) * 1000.0


def derive_jet_metering_frac(engine: "Engine", fuel_profile: str) -> float:
    """1.0 for anything fuel-injected (is_carbureted=False -- electronic
    metering is always correct here, no jet to get wrong). For a
    carbureted engine, the real consequence of its ACTUAL jet vs the
    correctly-sized reference above: orifice flow area scales with
    diameter^2, so a jet 20% oversized on diameter delivers ~44% more
    fuel than stoichiometric, not 20% -- real, and why jet sizes come in
    such fine increments in practice. Clamped to a plausible real range
    (a wildly wrong jet chokes to idle-only or floods -- this sim
    doesn't model outright flooding/starvation, just the real
    lean/rich efficiency consequence)."""
    carb = engine.carburetor
    if not carb.is_carbureted:
        return 1.0
    reference_mm = reference_jet_diameter_mm(engine, fuel_profile)
    area_ratio = (carb.main_jet_diameter_mm / max(reference_mm, 1e-6)) ** 2
    return max(0.4, min(1.6, area_ratio))


@dataclass
class PneumaticSystem:
    """A real belt-driven air compressor charging a real reserve tank --
    the same generic belt-driven-compressor mechanical port the AC
    compressor uses (drivetrain_graph._add_belt_driven_compressor),
    just with a genuinely different real payload: compressed air for
    pneumatics (tools, air suspension, train-style reserve-air brakes,
    anything that draws off a charged tank) instead of a refrigerant
    loop. compressor_fitted=False (default) means no such accessory
    exists on this engine at all -- most engines in this catalogue
    have no real reason to carry one, same as air_conditioning."""
    compressor_fitted: bool = False
    compressor_rated_w: float = 800.0
    reserve_tank_capacity_l: float = 20.0
    # Typed hardware (production's PNEUMATIC_COMPONENTS vocabulary):
    #   compressor_kind   "belt-piston" (road/industrial, off the crank belt)
    #                     "electric-piston" (a motor-driven compressor on the bus)
    #                     "starting-air-compressor" (a ship's electrically driven
    #                     two-stage compressor charging 30 bar receivers)
    #   compressor_drive  "crank-belt" | "electric-motor" | "external-switchboard"
    #                     (the last means the ship's own generator switchboard --
    #                     energy from outside this engine subgraph, disclosed)
    #   tank_kind         "reserve-tank" (a truck-style ~8 bar reservoir) |
    #                     "starting-air-receiver" (a ship's 30 bar receiver)
    #   tank_pressure_pa  the tank's real working pressure (sets stored mass,
    #                     compression work per kg, and starting torque)
    #   regulator         the unloader / pressure switch: the compressor
    #                     loads below cut_in and unloads at cut_out (real
    #                     air-brake governors: ~105 psi in / 125 psi out)
    compressor_kind: str = "belt-piston"
    compressor_drive: str = "crank-belt"
    tank_kind: str = "reserve-tank"
    tank_pressure_pa: float = 827_000.0
    regulator_cut_in_frac: float = 0.84
    regulator_cut_out_frac: float = 1.0


@dataclass
class FuelDeliverySystem:
    """The real hardware between the tank and the engine -- tank size
    sets how long a run lasts before it's genuinely empty (engine_cycle_
    sim.py wires the drivetrain graph's own depletable-reservoir physics,
    the same real mechanism nitrous already uses, into this), pump flow
    capacity sets a real ceiling on how much fuel can actually be
    delivered per second regardless of how much the engine wants (a
    weak pump under a big engine's WOT demand genuinely starves it, the
    same real symptom as a failing fuel pump)."""
    tank_capacity_l: float = 60.0
    pump_kind: str = "electric"           # "electric" | "mechanical"
    pump_flow_capacity_kg_s: float = 0.02
    line_diameter_mm: float = 8.0
    # A real drag-racing fuel cooler ("ice box" / ice tank): an active
    # chiller loop holding the tank near this real target temperature
    # instead of drifting to ambient. None (default) means no cooler is
    # fitted -- the tank just passively equilibrates toward ambient like
    # any uninsulated real tank does.
    cooler_target_temp_k: float | None = None


@dataclass
class Transmission:
    """A real, driver-shiftable gearbox -- mounted directly on the back
    of the block (drivetrain_graph.py's own real production nodes,
    powertrain.clutch/transmission/transfer_case, already positioned
    and meshed for free by _vehicle_powertrain_graph's standalone
    subunit; that subunit's own torque-shaft edges are rigid pass-
    throughs with no ratio data on them at all -- this dataclass is
    the real gear-ratio numbers layered on top, the same pattern as
    IntakeSystem/ExhaustSystem/FuelDeliverySystem elsewhere in this
    file). gear_ratios[i] is 1st gear at i=0 -- real automotive
    convention: ratio = engine-shaft revolutions per output-shaft
    revolution, so a BIGGER number is a LOWER, more torque-multiplying
    gear. final_drive_ratio multiplies every forward/reverse ratio
    (the differential/reduction stage downstream of the gearbox
    itself)."""
    gear_ratios: tuple[float, ...] = (3.54, 2.13, 1.36, 1.03, 0.82)
    reverse_ratio: float = 3.28
    final_drive_ratio: float = 3.73


@dataclass
class LifterSpring:
    """A single per-valve spring choice, applied uniformly across the
    engine. Real resistance, not a cosmetic setting:
      - drag_torque_nm(): friction load on the crank from holding every
        lifter against its cam lobe, scaling with spring force -- felt
        as a real pumping-loss term from idle onward (the "early baking"
        stat: it's already part of the engine's baseline resistance
        before a single explosion happens).
      - max_safe_rpm(): the valvetrain mass/spring system's natural
        frequency sets a real ceiling on cam speed before the spring
        can no longer keep the lifter following the cam profile (valve
        float). A soft spring on a high-redline build is a genuine,
        felt mismatch, not just a label.
    """
    spring_rate_n_per_mm: float = 45.0
    valve_lift_mm: float = 9.5
    valves_per_cylinder: int = 2
    friction_coeff: float = 0.12
    valvetrain_mass_g: float = 55.0   # effective reciprocating mass per valve

    def drag_torque_nm(self, cylinders: int, lever_arm_m: float = 0.006) -> float:
        if cylinders <= 0:
            return 0.0
        peak_force_n = self.spring_rate_n_per_mm * self.valve_lift_mm
        return (self.friction_coeff * peak_force_n * self.valves_per_cylinder
                * cylinders * lever_arm_m)

    def max_safe_rpm(self) -> float:
        k_n_per_m = self.spring_rate_n_per_mm * 1000.0
        m_kg = max(self.valvetrain_mass_g, 1.0) / 1000.0
        omega_n = math.sqrt(k_n_per_m / m_kg)   # rad/s, valvetrain natural frequency
        raw_rpm = omega_n * 60.0 / (2 * math.pi)
        # safety margin below the raw natural frequency -- real valvetrains
        # don't run right up to resonance; divisor calibrated so a "stock"
        # spring lands around a typical stock-engine redline (~7-8k rpm)
        return raw_rpm / 1.15


LIFTER_SPRING_PRESETS: dict[str, LifterSpring] = {
    "soft": LifterSpring(spring_rate_n_per_mm=28.0, friction_coeff=0.10),
    "stock": LifterSpring(spring_rate_n_per_mm=45.0, friction_coeff=0.12),
    "stiff": LifterSpring(spring_rate_n_per_mm=70.0, friction_coeff=0.14),
    "race": LifterSpring(spring_rate_n_per_mm=110.0, friction_coeff=0.17, valvetrain_mass_g=42.0),
}


@dataclass
class Engine:
    identity: str
    label: str
    kind: str  # combustion | electric | servo-electric | turbine | atmospheric | expander
    displacement_l: float
    bmep_pa: float
    braking_bmep_pa: float
    idle_rpm: float
    torque_peak_rpm: float
    power_peak_rpm: float
    redline_rpm: float
    inertia_kg_m2: float
    mass_kg: float
    clutch_torque_nm: float
    combustion_efficiency: float
    coupling_efficiency: float
    architecture: EngineArchitecture
    accessories: Accessories
    preferred_fuel_profile: str = "pump-gasoline-93"
    fuel_compatibility: dict = field(default_factory=lambda: {"pump-gasoline-93": 1.0, "nitromethane-race": 1.0})
    forced_induction: ForcedInduction = field(default_factory=ForcedInduction)
    lifter_spring: LifterSpring = field(default_factory=lambda: LIFTER_SPRING_PRESETS["stock"])
    # Optional dummy/ballast load resistor wired across the electrical
    # system: a fixed baseline electrical_load_frac for engine_cycle_sim
    # to start from (see EngineCycleSim.electrical_load_frac). None means
    # "not installed" -- the sim just uses its own default. Also
    # live-adjustable in main.py for testing a given engine's electrical
    # margin without having to rebuild it.
    load_resistor_frac: float | None = None
    # An independently-cited real peak power figure (e.g. an SAE dyno
    # sheet's own "140 hp @ 2600 rpm"), when the catalogue entry has
    # one -- see engine_sim.torque_fraction's own comment on why this
    # matters: without it, the falling-side torque curve's shape is
    # derived purely from the assumption that power mathematically
    # peaks exactly at power_peak_rpm, which silently overshoots
    # whichever of (peak torque, peak power) the engine's own bmep
    # DIDN'T get tuned to hit, if the two real cited numbers don't
    # happen to fall on that one implied ratio. None (the default)
    # keeps the old behavior unchanged for every engine that doesn't
    # declare one.
    rated_power_kw: float | None = None
    # Compression ignition (diesel): no spark plug, so the gasoline
    # spark-advance-driven autoignition-race knock model doesn't apply --
    # diesel combustion IS controlled compression ignition, it's not an
    # abnormal event, and fuel quality there is rated by cetane, not
    # octane (the fuel_compatibility table already reflects that: diesel
    # engines are set up with near-zero compatibility for gasoline/nitro).
    compression_ignition: bool = False
    rev_limiter: RevLimiterProfile = field(default_factory=RevLimiterProfile.stock)
    carburetor: CarburetorProfile = field(default_factory=CarburetorProfile)
    # A real discrete multi-barrel throttle body/carburetor assembly
    # (throttle_body.py) -- None (default) keeps every engine that
    # doesn't declare one on the existing single-plate abstraction
    # (engine_cycle_sim.throttle_plate_open_frac + carburetor.
    # secondary_threshold/secondary_surge_gain), exactly as before.
    # Declaring one replaces BOTH of those with the real per-barrel
    # geometry and real linkages (linkage.py) instead.
    throttle_body: ThrottleBodyAssembly | None = None
    # A real EFI injector array's own flow-vs-pressure characteristic
    # (injector.py) -- None means either this engine isn't EFI at all,
    # or is EFI with no declared injector sizing (metering stays ideal,
    # today's existing behavior). Auto-sized for every real EFI engine
    # in the catalogue at build time (_build_catalogue's own fuel-
    # system sizing pass) unless already set.
    injector: Injector | None = None
    # A real single-shaft gas turbine's own declared design point
    # (gas_turbine.py) -- only meaningful when kind == "turbine";
    # EngineCycleSim builds the stateful runtime object from this each
    # time the engine is (re)selected, the same real spec/runtime
    # separation this catalogue already uses everywhere else.
    turbine: TurbineSpec | None = None
    # A real Otto-Langen atmospheric cylinder's own declared build
    # (otto_langen.py) -- only meaningful when kind == "atmospheric" --
    # plus its real captive-ball governor (governor.py), a genuinely
    # separate real box, not baked into the cylinder itself.
    atmospheric: AtmosphericEngineSpec | None = None
    # A real cutoff EXPANDER cylinder bank (expander.py) -- steam or
    # compressed air, one engine kind ("expander") because the
    # cylinder hardware is the same; which fluid, and where it comes
    # from (boiler + feedwater, or a charged receiver), is entirely the
    # declared fuel_network. Only meaningful when kind == "expander".
    expander: ExpanderCylinderSpec | None = None
    atmospheric_governor: GovernorSpec | None = None
    # Real gasholder tank capacity when this engine's coal-gas is
    # supplied by an on-site generator (gas_works.py) rather than an
    # unlimited piped town main -- 0.0 (default) keeps every existing
    # atmospheric catalogue engine on the current unmetered-main
    # connection (drivetrain_graph.py's own "atmospheric" branch),
    # matching every other engine's default unchanged. A nonzero value
    # here is purely a real capacity NUMBER for sizing the tank node in
    # the vehicle's real fluid-circuit graph -- the actual GasGenerator/
    # Boiler object producing gas into that tank lives on EngineCycleSim.
    # atmospheric_generator, set by the caller after building the sim
    # (engines.py is the pure catalogue layer and doesn't import
    # gas_works.py, the same layering machining.py/engine_package.py
    # already sit above this file rather than inside it).
    atmospheric_supply_tank_capacity_kg: float = 0.0
    # THE conversion point: a declared fuel NETWORK (fuel_network.py --
    # store, in-line stages, admission device) for whatever working
    # fluid this cylinder is to run on. None (every catalogue engine
    # today) keeps the built-in fuel_delivery / atmospheric-main graph
    # branches exactly as they are. Set (see fuel_network.convert_
    # engine) and drivetrain_graph.py emits THAT network into the same
    # fuel circuit instead, engine_cycle_sim steps its runtime each
    # tick, and the cylinder reads the fluid's own working_fluids row
    # -- the same cylinder, a different network, which is the whole
    # design: any fuel, no second engine model.
    fuel_network: object | None = None
    # a conversion's own real timing shift (conversion.py: from the
    # fluid's flame speed) added to whatever this engine's timing policy
    # commands; and which fuel bmep_pa was rated on -- when it matches
    # the network's fluid the sim knows the rating already carries that
    # fluid's charge energy and doesn't apply it a second time
    ignition_timing_offset_deg: float = 0.0
    bmep_rated_fuel: str | None = None
    intake_system: IntakeSystem = field(default_factory=IntakeSystem)
    exhaust_system: ExhaustSystem = field(default_factory=ExhaustSystem)
    fuel_delivery: FuelDeliverySystem = field(default_factory=FuelDeliverySystem)
    transmission: Transmission = field(default_factory=Transmission)
    pneumatics: PneumaticSystem = field(default_factory=PneumaticSystem)
    ecu: ECUProfile = field(default_factory=lambda: ECU_PRESETS["crude"])
    egr: EGRSystem = field(default_factory=EGRSystem)
    # "throttle" (default): a normal pedal/lever controls airflow/fueling.
    # "hit_and_miss": a real historical stationary-engine governor, no
    # throttle plate at all. The carburetor setting is fixed; a flyball
    # governor on the crank instead decides, once per cycle, whether this
    # revolution gets a real charge and spark ("hit") or nothing at all
    # ("miss", exhaust valve typically held off its seat) -- purely based
    # on whether rpm is already at or above the governed speed. The engine
    # coasts on flywheel inertia through the misses and only fires again
    # once speed has actually dropped. Throttle input is ignored for these;
    # `governor_target_rpm` is the governed running speed.
    governor_mode: str = "throttle"
    governor_target_rpm: float | None = None
    # A real wet nitrous kit: bottle + solenoids + delivery lines, real
    # graph objects in drivetrain_graph.py (mirroring
    # abstract_ui_vehicles.py's own nitrous nodes/edges). Opt-in per
    # engine, off by default.
    has_nitrous: bool = False
    # A real water-methanol injection kit -- the SAME generic accessory
    # boss production's _vehicle_powertrain_graph now always carries
    # (powertrain.engine_block_port.accessory_injection_boss), claimed
    # by a genuinely different real accessory: a pump-fed tank, one
    # solenoid, one nozzle, no oxidizer, no second fuel solenoid --
    # water-meth suppresses knock by evaporatively cooling the charge
    # (see engine_cycle_sim's real latent-heat effect) rather than
    # boosting it. Historically real on high-boost piston aircraft
    # engines (British/German WWII fighters both ran water or water-
    # methanol injection for wartime emergency power specifically to
    # buy detonation margin at combat boost) -- not a toy invention.
    has_auxiliary_injection: bool = False
    # Which real device actually governs this engine's idle: "ecu" (the
    # default -- this catalogue's own gain-scheduled ecu.
    # EngineControlUnit, tuned specifically to this engine's own real
    # plant) or "standalone-pi-controller" (a real, commercially-sold
    # aftermarket idle-air-control box -- MSD/FAST/Holley Sniper/
    # Edelbrock Pro-Flo class hardware -- bolted onto an engine that has
    # no ECU idle governor of its own; see aftermarket_idle_controller.
    # py for the real trade-off: one fixed, conservative factory-default
    # tune instead of a per-engine calibration). A diesel always idles
    # on its own governed injection pump regardless of this field --
    # real aftermarket air-control boxes don't apply there.
    idle_control_device: str = "ecu"
    # Which real ignition system fires this engine -- production's own
    # ignition_profiles vocabulary (abstract_ui_vehicles.py): the
    # DISPATCH behind the profile is what the electrical network cares
    # about. "ecu-electronic" ignitions charge a coil off the 12 V bus
    # every spark and go dark below brown-out; "mechanical-magneto"
    # ones make their own spark energy from crank motion and draw
    # nothing from (and don't care about) the battery; "compression-
    # injection" has no spark at all.
    ignition_profile: str = "gasoline-distributor"
    # What the coolant loop ultimately rejects heat to: "air" (a fan-
    # blown radiator) or "seawater" (a marine central-cooling plate
    # exchanger fed by a seawater pump) -- production's own
    # _vehicle_powertrain_graph sizes and builds the exchanger from it
    cooling_medium: str = "air"
    # How this engine really gets turned over to start (starter.py's
    # STARTING_SYSTEMS): an electric starter on the ring gear, an
    # external pit-cart starter on the crank nose, starting air, a hand-
    # wound inertia starter, a recoil rope, or a hand crank. Anything
    # but the electric starter means the crank nose carries a real
    # attachment the starting torque enters through.
    starting_systems: tuple[str, ...] = ("electric-starter",)
    # the DC system this engine's electrics run at: 12 V cars, 24 V
    # trucks / industrial / aircraft / ship control-and-starting batteries
    electrical_system_voltage_v: float = 12.0
    # set when the catalogue's flywheel-fluctuation floor raised the
    # declared inertia (the declared value is kept here for the record)
    flywheel_floor_applied_from_kg_m2: float | None = None

    @property
    def starting_system(self) -> str:
        """The primary starting system (first of starting_systems)."""
        return self.starting_systems[0] if self.starting_systems else "electric-starter"

    @property
    def ignition_dispatch(self) -> str:
        return IGNITION_PROFILE_DISPATCH.get(self.ignition_profile, "ecu-electronic")

    @property
    def _cycle_radians(self) -> float:
        # Mean-effective-pressure relation: work per cycle = BMEP * Vd,
        # and work = torque * angle, so torque = BMEP*Vd / angle_per_cycle.
        # A four-stroke's cycle is 720 degrees (4*pi rad) -- one power
        # stroke every two crank revs. A two-stroke or rotary's cycle is
        # 360 degrees (2*pi rad) -- one power stroke every single rev --
        # which is exactly why they make roughly double the torque per
        # unit of BMEP*displacement. This used to be hardcoded to 4*pi
        # for every engine, silently wrong for the rotary in the
        # catalogue; it's derived from architecture.cycle_degrees now.
        return self.architecture.cycle_degrees * math.pi / 180.0

    @property
    def peak_torque_nm(self) -> float:
        vd = self.displacement_l / 1000.0
        return self.bmep_pa * vd / self._cycle_radians

    @property
    def peak_braking_torque_nm(self) -> float:
        vd = self.displacement_l / 1000.0
        return self.braking_bmep_pa * vd / self._cycle_radians

    def min_safe_load_frac(self, rpm: float) -> float:
        """The minimum load (as a fraction of peak_torque_nm) that
        should be on the output shaft at this rpm to keep a real
        WOT-unloaded acceleration from being able to run the crank past
        redline between rev-limiter control ticks. Real test/dyno rigs
        are never actually run truly unloaded at high rpm for exactly
        this reason -- a flywheel, clutch, or absorber is always at
        least some real load. Derived from this engine's own real
        inertia_kg_m2 and peak_torque_nm (how fast it could genuinely
        accelerate at zero load), not an arbitrary per-engine number:
        a light-flywheel, high-torque engine demands more real minimum
        load near redline than a heavy-flywheel one does, because it
        can actually get there faster.
        """
        safety_band_start_rpm = self.redline_rpm * 0.6
        if rpm < safety_band_start_rpm:
            return 0.0
        alpha_unloaded_rad_s2 = self.peak_torque_nm / max(self.inertia_kg_m2, 1e-9)
        rpm_accel_per_s = alpha_unloaded_rad_s2 * 60.0 / (2.0 * math.pi)
        proximity = min(1.0, (rpm - safety_band_start_rpm) / max(self.redline_rpm * 0.4, 1.0))
        # calibrated against a real reference accel rate (~20,000 rpm/s,
        # a genuinely violent unloaded big-block/drag-engine rate) so a
        # merely-brisk engine doesn't get an inflated minimum
        accel_severity = min(1.0, rpm_accel_per_s / 20_000.0)
        return min(0.35, proximity * accel_severity * 0.5)

    @property
    def peak_power_kw(self) -> float:
        omega = self.power_peak_rpm * RPM_TO_RAD_S
        # torque at power-peak rpm per the shape curve, see engine_sim.torque_fraction
        from engine_sim import torque_fraction
        t = self.peak_torque_nm * torque_fraction(self, self.power_peak_rpm)
        return t * omega / 1000.0

    # Real, disclosed ceiling on mean piston speed before ring-flutter/
    # connecting-rod fatigue risk rises sharply -- established piston-
    # engine practice puts stock road-going engines around ~18-20 m/s
    # and pushes performance/race practice out toward ~24-25 m/s before
    # exotic materials become load-bearing; 25 m/s is the disclosed
    # ceiling used here, the same "real, magnitude-correct, not one
    # cited paper's exact number" convention FRICTION_FMEP_* already
    # uses above.
    MAX_SAFE_MEAN_PISTON_SPEED_M_S = 25.0

    def derived_redline_rpm(self) -> float | None:
        """What this engine's own real geometry actually limits it to,
        computed WITHOUT ever reading the catalogue's declared
        redline_rpm -- the whole point is a real answer to "why does
        this engine have a redline," not a circular restatement of the
        declared number (engine_sim.torque_fraction's own choke term
        already uses redline_rpm as an INPUT, so deriving it FROM that
        same term would be self-referential).

        Two real, independent mechanisms, whichever binds first:
          - mean piston speed: ring-flutter/rod-fatigue risk scales
            with piston SPEED, not rpm directly (a giant slow-turning
            marine diesel and a screaming superbike hit the same real
            ceiling at wildly different rpm because their strokes
            differ) -- MAX_SAFE_MEAN_PISTON_SPEED_M_S above, inverted
            through this engine's own real stroke via mean_piston_
            speed_m_s's own formula (2*stroke*rpm/60).
          - valve float (poppet-valve architectures only): LifterSpring.
            max_safe_rpm(), this engine's own declared spring's real
            natural-frequency ceiling -- already built, previously
            never actually checked against anything.
        Returns None for architectures with no real stroke (electric,
        rotary) -- there's no real piston-speed mechanism to derive a
        limit from there, and inventing one would be exactly the kind
        of unearned number this toy avoids."""
        stroke_m = self.architecture.stroke_m
        if stroke_m <= 0.0:
            return None
        piston_speed_limit_rpm = self.MAX_SAFE_MEAN_PISTON_SPEED_M_S * 60.0 / (2.0 * stroke_m)
        limit_rpm = piston_speed_limit_rpm
        if self.architecture.has_poppet_valves:
            limit_rpm = min(limit_rpm, self.lifter_spring.max_safe_rpm())
        return limit_rpm

    # Real, structurally-standard total mechanical friction correlation
    # (FMEP = A + B*Sp + C*Sp^2, Sp = mean piston speed) -- constant +
    # linear + quadratic in piston speed is the well-established real
    # form (ring/piston-skirt shear rises with sliding speed, bearing
    # friction has both a speed-independent boundary term and a speed-
    # dependent hydrodynamic term). Coefficients anchored to real
    # published FMEP magnitude ranges for automotive engines -- ~70 kPa
    # at negligible piston speed (a real, nonzero boundary-friction/
    # accessory-load floor, not zero), ~180 kPa around 15 m/s (typical
    # road-engine piston speed), ~370 kPa around 30 m/s (race-engine
    # territory) -- not a specific cited paper's exact coefficients, but
    # a real, disclosed, magnitude-correct correlation, replacing an
    # rpm/redline-fraction proxy that treated every engine in the
    # catalogue as if it hit the same RELATIVE friction at its own
    # redline regardless of how fast its pistons actually move there (a
    # 22rpm giant marine diesel's pistons move nothing like a superbike's).
    FRICTION_FMEP_A_PA = 70_000.0
    FRICTION_FMEP_B_PA_S_PER_M = 4_333.0
    FRICTION_FMEP_C_PA_S2_PER_M2 = 200.0

    def friction_torque_nm(self, rpm: float) -> float:
        """Real mechanical friction torque at this rpm, from the real
        FMEP correlation above and this engine's own real geometry --
        the same MEP-to-torque conversion peak_torque_nm itself uses
        (BMEP * displacement / cycle_radians), just for friction MEP."""
        sp = self.mean_piston_speed_m_s(rpm)
        if sp <= 0.0:
            return 0.0
        fmep_pa = (self.FRICTION_FMEP_A_PA + self.FRICTION_FMEP_B_PA_S_PER_M * sp
                   + self.FRICTION_FMEP_C_PA_S2_PER_M2 * sp * sp)
        vd_m3 = self.displacement_l / 1000.0
        return fmep_pa * vd_m3 / self._cycle_radians

    def mean_piston_speed_m_s(self, rpm: float) -> float:
        """The real, exact mean piston speed: 2 * stroke * rpm / 60 --
        two piston travels (up + down) per crank revolution. The
        standard real reference quantity friction (FMEP) and burn-rate
        correlations are actually built on, not an rpm-fraction proxy.
        0.0 for architectures with no real stroke (rotary, electric)."""
        if self.architecture.stroke_m <= 0.0:
            return 0.0
        return 2.0 * self.architecture.stroke_m * max(rpm, 0.0) / 60.0

    def firing_events_per_rev(self) -> float:
        """Firing events per output-shaft revolution: four-stroke piston
        is cylinders/2 (one power stroke per two crank revs); rotary is
        cylinders (repurposed as rotor count) and a two-stroke piston is
        also cylinders -- both fire once per single output-shaft rev,
        twice as often per revolution as a four-stroke."""
        cyl = self.architecture.cylinders
        if not cyl:
            return 0.0
        return float(cyl) if (self.architecture.rotary or self.architecture.two_stroke) else cyl / 2.0


def _default_wobble(layout: str) -> float:
    if "crossplane" in layout:
        return 0.35
    if layout in ("flat-four", "flat-six", "sixty-degree-v12"):
        return 0.12
    return 0.0


_ROD_TO_STROKE_RATIO = 1.75  # a real, typical value (1.5-2.2 in practice) -- not researched per engine


def _piston_geometry(displacement_l: float, cylinders: int, redline_rpm: float) -> tuple[float, float, float]:
    """Real, exact geometry solve, not a second independent guess: pick
    a bore/stroke ratio from a genuine monotonic real-world correlation
    (shorter stroke is needed at higher redline to keep mean piston
    speed within real material/friction limits -- roughly log-linear
    across this catalogue's own huge span, calibrated against real
    reference points at both ends: a Wartsila RTA96C-class slow marine
    diesel's real ~0.48 bore/stroke ratio at ~22 rpm redline, a modern
    high-revving sportbike's real ~1.3-1.6 at ~11,000 rpm), then solve
    bore EXACTLY so cylinders*(pi/4)*bore^2*stroke reproduces this
    engine's own already-declared displacement_l precisely. This is a
    real correlation, not individually researched historical specs for
    all 20 catalogue engines -- a defensible approximation, disclosed
    as one. Rod length uses a real, typical rod/stroke ratio
    (_ROD_TO_STROKE_RATIO), also not individually tuned per engine.
    """
    if cylinders <= 0 or redline_rpm <= 0:
        return 0.0, 0.0, 0.0
    bore_stroke_ratio = max(0.4, min(1.65, 0.45 + 0.32 * math.log10(max(redline_rpm, 1.0) / 22.0)))
    vd_per_cyl_m3 = (displacement_l / 1000.0) / cylinders
    bore_m = (vd_per_cyl_m3 * bore_stroke_ratio * 4.0 / math.pi) ** (1.0 / 3.0)
    stroke_m = bore_m / bore_stroke_ratio
    rod_length_m = stroke_m * _ROD_TO_STROKE_RATIO
    return bore_m, stroke_m, rod_length_m


def _arch(layout, cylinders, banks, bank_angle, firing_order, wobble_amt: float | None = None) -> EngineArchitecture:
    return EngineArchitecture(layout, cylinders, banks, bank_angle, list(firing_order),
                               wobble_amt=_default_wobble(layout) if wobble_amt is None else wobble_amt)


_RAW = [
    dict(identity="amc-258-jeep-i6", label="AMC-era Jeep 258 ci trail I6", kind="combustion",
         displacement=4.227, bmep=835_000, braking_bmep=135_000, idle=650,
         torque_peak=1800, power_peak=3200, redline=4400, inertia=.68,
         mass=220, clutch_torque=235, combustion_efficiency=.84, coupling_efficiency=.90,
         architecture=("inline-six", 6, 1, 0.0, [1, 5, 3, 6, 2, 4])),
    dict(identity="honda-style-commuter-i4-1500", label="Honda-style 1.5 L commuter I4", kind="combustion",
         displacement=1.5, bmep=1_160_000, braking_bmep=145_000, idle=750,
         torque_peak=4200, power_peak=6000, redline=6800, inertia=.16,
         mass=112, clutch_torque=175, combustion_efficiency=.91, coupling_efficiency=.94,
         architecture=("inline-four", 4, 1, 0.0, [1, 3, 4, 2])),
    dict(identity="aircooled-flat-four-1584", label="1584 cc air-cooled flat-four", kind="combustion",
         displacement=1.584, bmep=720_000, braking_bmep=125_000, idle=850,
         torque_peak=2800, power_peak=4100, redline=4800, inertia=.31,
         mass=102, clutch_torque=155, combustion_efficiency=.82, coupling_efficiency=.93,
         architecture=("flat-four", 4, 2, 180.0, [1, 4, 3, 2])),
    dict(identity="springtail-i4-1600", label="Springtail 1.6 L trail I4", kind="combustion",
         displacement=1.6, bmep=1_600_000, braking_bmep=165_000, idle=850,
         torque_peak=3600, power_peak=5200, redline=6400, inertia=.22,
         mass=142, clutch_torque=260, combustion_efficiency=.88, coupling_efficiency=.94,
         architecture=("inline-four", 4, 1, 0.0, [1, 3, 4, 2])),
    dict(identity="superbike-i4-1340", label="1340 cc superbike I4", kind="combustion",
         displacement=1.34, bmep=1_720_000, braking_bmep=190_000, idle=1250,
         torque_peak=7000, power_peak=9700, redline=11000, inertia=.085,
         mass=82, clutch_torque=205, combustion_efficiency=.91, coupling_efficiency=.95,
         architecture=("inline-four", 4, 1, 0.0, [1, 2, 4, 3])),
    dict(identity="gt-flat-six-4000", label="4.0 L GT flat-six", kind="combustion",
         displacement=4.0, bmep=1_650_000, braking_bmep=210_000, idle=900,
         torque_peak=6250, power_peak=8400, redline=9000, inertia=.19,
         mass=190, clutch_torque=610, combustion_efficiency=.92, coupling_efficiency=.96,
         architecture=("flat-six", 6, 2, 180.0, [1, 6, 2, 4, 3, 5])),
    dict(identity="supercharged-drag-v8-8200", label="8.2 L supercharged drag V8", kind="combustion",
         displacement=8.2, bmep=6_200_000, braking_bmep=420_000, idle=1150,
         torque_peak=6500, power_peak=8700, redline=9600, inertia=.46,
         mass=338, clutch_torque=4_800, combustion_efficiency=.94, coupling_efficiency=.97,
         architecture=("crossplane-v8", 8, 2, 90.0, [1, 8, 4, 3, 6, 5, 7, 2])),
    dict(identity="monster-540-blown-methanol", label="540 ci blown-methanol monster V8", kind="combustion",
         displacement=8.849, bmep=2_850_000, braking_bmep=360_000, idle=1100,
         torque_peak=5200, power_peak=7100, redline=8000, inertia=.55,
         mass=345, clutch_torque=2_900, combustion_efficiency=.93, coupling_efficiency=.97,
         architecture=("crossplane-v8", 8, 2, 90.0, [1, 8, 4, 3, 6, 5, 7, 2])),
    dict(identity="monster-632-twin-turbo", label="632 ci twin-turbo monster big-block", kind="combustion",
         displacement=10.357, bmep=3_650_000, braking_bmep=410_000, idle=1050,
         torque_peak=4400, power_peak=6500, redline=7300, inertia=.70,
         mass=425, clutch_torque=4_500, combustion_efficiency=.94, coupling_efficiency=.97,
         architecture=("crossplane-v8", 8, 2, 90.0, [1, 8, 4, 3, 6, 5, 7, 2])),
    dict(identity="packard-merlin-v1650", label="Packard / Rolls-Royce Merlin V-1650 aircraft V12", kind="combustion",
         displacement=27.04, bmep=1_420_000, braking_bmep=235_000, idle=600,
         torque_peak=2200, power_peak=3000, redline=3200, inertia=2.85,
         mass=744, clutch_torque=3_600, combustion_efficiency=.86, coupling_efficiency=.92,
         architecture=("sixty-degree-v12", 12, 2, 60.0,
                       [1, 6, 3, 5, 2, 4, 7, 12, 9, 11, 8, 10])),
    dict(identity="cat-c18-industrial-diesel", label="18.1 L heavy-machine turbo diesel I6", kind="combustion",
         displacement=18.1, bmep=2_540_000, braking_bmep=390_000, idle=600,
         torque_peak=1400, power_peak=1900, redline=2200, inertia=5.8,
         mass=1_673, clutch_torque=4_400, combustion_efficiency=.92, coupling_efficiency=.91,
         architecture=("inline-six", 6, 1, 0.0, [1, 5, 3, 6, 2, 4])),
    # Real 1960s US military multifuel diesel (Continental/Hercules
    # LDT-465, the M35A2 2.5-ton "deuce and a half" engine) -- a real
    # 465 cubic inch (7.62 L) turbocharged inline-six, 140 hp @ 2600
    # rpm / 330 lb-ft (447 Nm) @ 1400 rpm, both real cited SAE figures.
    # bmep below is set to hit the torque figure almost exactly (447.5
    # Nm at build). Disclosed limitation: under this sim's torque-curve
    # model, bmep_pa scales torque and power by the identical factor,
    # so the curve shape implied by (torque_peak_rpm, power_peak_rpm)
    # rigidly fixes their ratio -- and the two real cited figures don't
    # fall exactly on that ratio here, so derived peak power comes out
    # ~14% under the cited 140 hp. Raising bmep to hit power exactly
    # would overshoot the (more directly combustion-limited) torque
    # figure by a similar margin instead; kept torque exact rather than
    # split the error, since that's the number this engine's own real
    # compression/displacement most directly sets.
    # Its real, extreme ~25:1 compression ratio is the actual reason it
    # tolerates diesel, kerosene, crude oil, or vegetable oil --
    # compression heat alone forces auto-ignition regardless of the
    # fuel's own cetane quality, not a magic multifuel injector. Real
    # period sources note the turbo's own job here was mainly reducing
    # exhaust smoke, not adding power -- reflected in a real, modest
    # max_boost_frac below, not a power-turbo's boost curve.
    dict(identity="ldt465-multifuel-deuce", label="LDT-465 multifuel diesel I6 (M35A2 \"deuce and a half\")",
         kind="combustion",
         displacement=7.62, bmep=738_000, braking_bmep=150_000, idle=650,
         torque_peak=1400, power_peak=2600, redline=2800, inertia=2.4,
         mass=600, clutch_torque=1_400, combustion_efficiency=.88, coupling_efficiency=.90,
         architecture=("inline-six", 6, 1, 0.0, [1, 5, 3, 6, 2, 4])),
    dict(identity="dual-motor-ev-reference", label="dual-motor EV drive unit", kind="electric",
         displacement=4.0, bmep=1_420_000, braking_bmep=760_000, idle=120,
         torque_peak=900, power_peak=9000, redline=18000, inertia=.12,
         mass=205, clutch_torque=720, combustion_efficiency=.99, coupling_efficiency=.98,
         architecture=("dual-electric", 0, 2, 0.0, [])),
    dict(identity="servo-direct-drive-400", label="400 Nm direct-drive servo", kind="servo-electric",
         displacement=.80, bmep=6_300_000, braking_bmep=1_900_000, idle=60,
         torque_peak=120, power_peak=4200, redline=8000, inertia=.075,
         mass=52, clutch_torque=400, combustion_efficiency=.995, coupling_efficiency=.985,
         architecture=("servo-electric", 0, 1, 0.0, [])),
]

# Per-engine accessory loadout. Not derivable from the physics fields above
# -- it's a build/spec choice, so it's hand-curated per identity here.
# Anything not listed falls back to Accessories() (everything fitted).
_ACCESSORIES: dict[str, Accessories] = {
    "amc-258-jeep-i6": Accessories(water_pump=True, mechanical_fan=True, alternator=True),
    "honda-style-commuter-i4-1500": Accessories(water_pump=True, mechanical_fan=True, alternator=True),
    "aircooled-flat-four-1584": Accessories(water_pump=False, mechanical_fan=True, alternator=True),
    "springtail-i4-1600": Accessories(water_pump=True, mechanical_fan=True, alternator=True),
    # sportbike-derived four: crank-integrated stator, no belt accessories at
    # all -- real sportbikes cool the radiator with a small thermostatic
    # electric fan instead, exactly like this toy's own electric_fan
    "superbike-i4-1340": Accessories(water_pump=True, mechanical_fan=False, alternator=False,
                                      electric_fan=True),
    "gt-flat-six-4000": Accessories(water_pump=True, mechanical_fan=False, alternator=True,
                                     electric_fan=True),
    # drag/exhibition builds: total-loss ignition off the battery, no fan for a
    # quarter-mile pass, often no belt-driven water pump either
    "supercharged-drag-v8-8200": Accessories(water_pump=False, mechanical_fan=False, alternator=False),
    "monster-540-blown-methanol": Accessories(water_pump=False, mechanical_fan=False, alternator=False),
    "monster-632-twin-turbo": Accessories(water_pump=False, mechanical_fan=False, alternator=False),
    # aircraft V12: ram-air/radiator cooled, no mechanical fan; magneto ignition, no alternator
    "packard-merlin-v1650": Accessories(water_pump=True, mechanical_fan=False, alternator=False),
    "cat-c18-industrial-diesel": Accessories(water_pump=True, mechanical_fan=True, alternator=True),
    "ldt465-multifuel-deuce": Accessories(water_pump=True, mechanical_fan=True, alternator=True),
    "dual-motor-ev-reference": Accessories(coolant_pump=True),
    "servo-direct-drive-400": Accessories(coolant_pump=False),
}

# A real pump-gas octane ladder -- there used to be exactly one pump-gas
# tier in the whole system (93, always at 1.0 compatibility alongside
# nitromethane, also always 1.0), which made cycling fuel a no-op for
# every engine except the two with hand-written exotic-fuel overrides.
# Running octane AT OR ABOVE what an engine wants is always safe (1.0) --
# extra anti-knock margin never hurts, it's just unnecessary. Running
# BELOW what it wants gets progressively more knock-prone per step down.
PUMP_GAS_LADDER = ["pump-gasoline-87", "pump-gasoline-89", "pump-gasoline-91", "pump-gasoline-93"]


def _pump_gas_profile(preferred_octane: str, **extra: float) -> dict:
    idx = PUMP_GAS_LADDER.index(preferred_octane)
    compat = {}
    for i, tier in enumerate(PUMP_GAS_LADDER):
        compat[tier] = 1.0 if i >= idx else max(0.15, 1.0 - 0.22 * (idx - i))
    compat["nitromethane-race"] = 1.0
    compat.update(extra)
    return compat


# Per-engine fuel compatibility -- 1.0 is that engine's own preferred fuel
# (or anything at/above its required octane); lower values are how much
# worse an alternate fuel behaves, and (via engine_cycle_sim's knock model)
# how much more knock-prone it is. Every combustion engine gets an explicit
# entry now -- no more silent same-tier fallback.
_FUEL: dict[str, dict] = {
    # low-compression, torque-tuned: happy on regular
    "amc-258-jeep-i6": dict(preferred_fuel_profile="pump-gasoline-87",
                             fuel_compatibility=_pump_gas_profile("pump-gasoline-87")),
    "honda-style-commuter-i4-1500": dict(preferred_fuel_profile="pump-gasoline-87",
                                          fuel_compatibility=_pump_gas_profile("pump-gasoline-87")),
    "aircooled-flat-four-1584": dict(preferred_fuel_profile="pump-gasoline-87",
                                      fuel_compatibility=_pump_gas_profile("pump-gasoline-87")),
    "springtail-i4-1600": dict(preferred_fuel_profile="pump-gasoline-89",
                                fuel_compatibility=_pump_gas_profile("pump-gasoline-89")),
    # higher state of tune, wants real premium
    "superbike-i4-1340": dict(preferred_fuel_profile="pump-gasoline-93",
                               fuel_compatibility=_pump_gas_profile("pump-gasoline-93")),
    "gt-flat-six-4000": dict(preferred_fuel_profile="pump-gasoline-93",
                              fuel_compatibility=_pump_gas_profile("pump-gasoline-93")),
    "monster-632-twin-turbo": dict(preferred_fuel_profile="pump-gasoline-93",
                                    fuel_compatibility=_pump_gas_profile("pump-gasoline-93")),
    # drag/exhibition builds: genuinely built around exotic fuel, pump gas
    # is the downgrade here, not the baseline
    "supercharged-drag-v8-8200": dict(
        preferred_fuel_profile="nitromethane-race",
        fuel_compatibility={**_pump_gas_profile("pump-gasoline-93", **{"pump-gasoline-93": 0.55}),
                             "nitromethane-race": 1.0}),
    "monster-540-blown-methanol": dict(
        preferred_fuel_profile="methanol-race",
        fuel_compatibility={**_pump_gas_profile("pump-gasoline-93", **{"pump-gasoline-93": 0.40}),
                             "nitromethane-race": 0.85, "methanol-race": 1.0}),
    "packard-merlin-v1650": dict(
        preferred_fuel_profile="aviation-gasoline-100-130",
        fuel_compatibility={"aviation-gasoline-100-130": 1.0, "pump-gasoline-93": .48, "nitromethane-race": .70},
    ),
    "cat-c18-industrial-diesel": dict(
        preferred_fuel_profile="ultra-low-sulfur-diesel",
        fuel_compatibility={"ultra-low-sulfur-diesel": 1.0, "pump-gasoline-93": .035,
                             "aviation-gasoline-100-130": .025, "nitromethane-race": .06},
    ),
    # the real point of this engine: genuinely wide fuel tolerance off
    # its own extreme compression ratio, not a normal diesel's near-
    # zero off-spec tolerance. Still real tradeoffs, not all 1.0 --
    # kerosene is close to diesel (slightly lower cetane); crude oil is
    # usable but genuinely dirtier/more variable; vegetable oil ignites
    # fine but its real higher viscosity is hard on the injection pump
    # without preheating (the actual reason straight vegetable oil was
    # a real wartime desperation fuel, not a routine one); gasoline
    # still resists compression-ignition even at 25:1, just less badly
    # than a normal diesel's near-zero tolerance.
    "ldt465-multifuel-deuce": dict(
        preferred_fuel_profile="ultra-low-sulfur-diesel",
        fuel_compatibility={"ultra-low-sulfur-diesel": 1.0, "kerosene": .92, "crude-oil": .55,
                             "vegetable-oil": .65, "pump-gasoline-93": .25},
    ),
}


# Per-engine lifter spring. Picked so each catalogue engine's own redline
# sits safely under its spring's max_safe_rpm() -- a custom-built engine
# has no such guarantee, which is the point: pick a spring too soft for
# the redline you asked for and the sim will actually make you pay for it.
_LIFTER_SPRING: dict[str, str] = {
    "amc-258-jeep-i6": "soft",
    "honda-style-commuter-i4-1500": "stock",
    "aircooled-flat-four-1584": "soft",
    "springtail-i4-1600": "stock",
    "superbike-i4-1340": "race",
    "gt-flat-six-4000": "race",
    "supercharged-drag-v8-8200": "race",
    "monster-540-blown-methanol": "stiff",
    "monster-632-twin-turbo": "stiff",
    "packard-merlin-v1650": "stock",
    "cat-c18-industrial-diesel": "soft",
    "ldt465-multifuel-deuce": "soft",
}

# Per-engine forced induction. Anything not listed is naturally aspirated
# (default ForcedInduction(), kind="none").
_FORCED_INDUCTION: dict[str, ForcedInduction] = {
    "supercharged-drag-v8-8200": ForcedInduction(
        kind="supercharger", max_boost_frac=1.20, lobe_count=4, belt_ratio=3.2),
    "monster-540-blown-methanol": ForcedInduction(
        kind="supercharger", max_boost_frac=1.00, lobe_count=3, belt_ratio=2.9),
    "monster-632-twin-turbo": ForcedInduction(
        kind="turbo", max_boost_frac=0.90, spool_tau_s=0.9, wastegate_frac=0.85, anti_lag_capable=True),
    # historically a gear-driven two-stage centrifugal supercharger, not a turbo
    "packard-merlin-v1650": ForcedInduction(
        kind="supercharger", max_boost_frac=0.55, lobe_count=1, belt_ratio=7.0),
    # industrial diesels are almost always turbocharged
    "cat-c18-industrial-diesel": ForcedInduction(
        kind="turbo", max_boost_frac=0.60, spool_tau_s=1.1, wastegate_frac=0.90, anti_lag_capable=False),
    # real period sources: the turbo on this engine was fitted mainly to
    # reduce exhaust smoke and aid combustion, not to appreciably raise
    # power -- a real, modest boost figure, not a power-turbo's curve
    "ldt465-multifuel-deuce": ForcedInduction(
        kind="turbo", max_boost_frac=0.15, spool_tau_s=1.4, wastegate_frac=0.90, anti_lag_capable=False),
}


def _build_radial_engines() -> list[Engine]:
    import crank_phase
    wasp_firing = crank_phase.generate_radial_firing_order(9)
    wasp_arch = EngineArchitecture(
        layout="radial-nine", cylinders=9, banks=9, bank_angle_degrees=0.0,
        firing_order=wasp_firing, radial=True, rows=1, wobble_amt=0.0,
    )
    # Pratt & Whitney R-1340 "Wasp": 9-cylinder single-row air-cooled radial,
    # 1,344 cu in (22.0 L), the engine behind the T-6 Texan and countless
    # 1930s-40s single-engine aircraft. Gear-driven single-stage supercharger
    # for altitude compensation; dual-magneto ignition (no battery
    # dependency in reality -- simplified here to a healthy generator so
    # the sim's battery/misfire model doesn't fight that).
    wasp = Engine(
        identity="pw-r1340-wasp", label='Pratt & Whitney R-1340 "Wasp" 9-cyl radial', kind="combustion",
        displacement_l=22.0, bmep_pa=1_171_000, braking_bmep_pa=190_000,
        idle_rpm=550, torque_peak_rpm=1900, power_peak_rpm=2250, redline_rpm=2350,
        inertia_kg_m2=1.6, mass_kg=300, clutch_torque_nm=2800,
        combustion_efficiency=0.85, coupling_efficiency=0.90,
        architecture=wasp_arch,
        accessories=Accessories(water_pump=False, mechanical_fan=False, alternator=True),
        preferred_fuel_profile="aviation-gasoline-100-130",
        fuel_compatibility={"aviation-gasoline-100-130": 1.0, "pump-gasoline-93": .55, "nitromethane-race": .65},
        forced_induction=ForcedInduction(kind="supercharger", max_boost_frac=0.35, lobe_count=1, belt_ratio=8.0),
        lifter_spring=LIFTER_SPRING_PRESETS["stock"],
        # a real single-barrel updraft carburetor (Bendix/Stromberg-type,
        # period-correct for this engine), jetted to its own reference
        # (see reference_jet_diameter_mm) as a correctly-tuned stock default
        carburetor=CarburetorProfile(is_carbureted=True, has_choke=True, choke_warmup_s=6.0,
                                      choke_max_enrichment=0.25, main_jet_diameter_mm=3.72),
        intake_system=_INTAKE.get("pw-r1340-wasp", IntakeSystem()),
        exhaust_system=_EXHAUST.get("pw-r1340-wasp", ExhaustSystem()),
    )
    return [wasp]


def _build_rotary_engines() -> list[Engine]:
    rotors = 2
    rotary_arch = EngineArchitecture(
        layout="twin-rotor-wankel", cylinders=rotors, banks=1, bank_angle_degrees=0.0,
        firing_order=list(range(1, rotors + 1)),  # trivially even by construction, nothing to search
        rotary=True, wobble_amt=0.0,
    )
    # A 13B-style twin-rotor Wankel: ~1.3 L combined chamber volume,
    # lightweight, low reciprocating inertia (mostly rotating mass),
    # revs high and smooth (no valvetrain to float), but historically
    # thirsty (poor thermal efficiency from the combustion chamber shape)
    # and knock-sensitive enough to want real premium.
    rotary = Engine(
        identity="twin-rotor-13b", label="13B-style 1.3L twin-rotor Wankel", kind="combustion",
        displacement_l=1.3, bmep_pa=900_000, braking_bmep_pa=130_000,
        idle_rpm=900, torque_peak_rpm=4500, power_peak_rpm=7000, redline_rpm=9000,
        inertia_kg_m2=0.10, mass_kg=130, clutch_torque_nm=180,
        combustion_efficiency=0.78, coupling_efficiency=0.92,
        architecture=rotary_arch,
        accessories=Accessories(water_pump=True, mechanical_fan=True, alternator=True),
        preferred_fuel_profile="pump-gasoline-93",
        fuel_compatibility=_pump_gas_profile("pump-gasoline-93"),
        lifter_spring=LIFTER_SPRING_PRESETS["stock"],  # unused (no valves), kept for a harmless default
        intake_system=_INTAKE.get("twin-rotor-13b", IntakeSystem()),
        exhaust_system=_EXHAUST.get("twin-rotor-13b", ExhaustSystem()),
    )
    return [rotary]


def _build_radical_cam_engines() -> list[Engine]:
    # A big-inch solid-lifter street/strip V8: the cam is the whole
    # personality here. Huge overlap eats low-rpm vacuum and torque
    # (hence the unusually high idle just to keep it alive at all, and
    # combustion_efficiency well below a mild engine's), the extra
    # wobble_amt exaggerates the crossplane V8's already-uneven firing
    # into a genuinely lumpy, hunting idle, and a four-barrel's
    # secondaries (carburetor.secondary_threshold/secondary_surge_gain)
    # give a real nonlinear kick in airflow once you're well into the
    # pedal -- calm-ish around town, then everything opens up at once.
    arch = EngineArchitecture(
        layout="crossplane-v8", cylinders=8, banks=2, bank_angle_degrees=90.0,
        firing_order=[1, 8, 4, 3, 6, 5, 7, 2], wobble_amt=0.5,
    )
    engine = Engine(
        identity="radical-cam-bigblock-7400", label="7.4L radical-cam big-block V8", kind="combustion",
        displacement_l=7.4, bmep_pa=1_350_000, braking_bmep_pa=180_000,
        idle_rpm=1000, torque_peak_rpm=4200, power_peak_rpm=5800, redline_rpm=6400,
        inertia_kg_m2=0.60, mass_kg=320, clutch_torque_nm=750,
        combustion_efficiency=0.80, coupling_efficiency=0.94,
        architecture=arch,
        accessories=Accessories(water_pump=True, mechanical_fan=True, alternator=True),
        preferred_fuel_profile="pump-gasoline-93",
        fuel_compatibility=_pump_gas_profile("pump-gasoline-93"),
        lifter_spring=LIFTER_SPRING_PRESETS["race"],   # solid-lifter radical cam: loud, mechanical, can rev
        carburetor=CarburetorProfile(is_carbureted=True, has_choke=True, choke_warmup_s=10.0,
                                      choke_max_enrichment=0.30, main_jet_diameter_mm=3.52),
        # a real progressive DOUBLE four-barrel -- two 4-bbl carbs on a
        # dual-quad intake, 8 real bores, primaries ganged, secondaries
        # sharing one progressive-cam linkage (throttle_body.py) --
        # replaces the old abstract secondary_threshold/surge_gain kick
        # with the actual real hardware this build has
        throttle_body=progressive_double_four_barrel(),
        intake_system=_INTAKE.get("radical-cam-bigblock-7400", IntakeSystem()),
        exhaust_system=_EXHAUST.get("radical-cam-bigblock-7400", ExhaustSystem()),
    )
    return [engine]


def _build_specialty_engines() -> list[Engine]:
    """Four real, named machines picked for how differently they actually
    work, not just how they're dressed -- each one exercises a genuinely
    distinct mechanism this sim now supports."""
    import crank_phase
    engines_out: list[Engine] = []

    # -- Wartsila-Sulzer RTA96C-B, 14-cylinder: the largest reciprocating
    # engine ever built. Real production spec: 960 mm bore, 2500 mm
    # stroke (Vcyl = pi/4 * 0.96^2 * 2.5 = 1.809 m^3/cyl, 25.33 m^3 total),
    # two-stroke, uniflow-scavenged with a real hydraulically-actuated
    # exhaust valve (has_poppet_valves=True), crosshead (piston thrust
    # taken by a crosshead bearing, not the cylinder wall -- why these can
    # run for decades). Rated 80,080 kW at 102 rpm; peak_torque_nm here
    # (derived from bmep/displacement via the two-stroke 2*pi relation)
    # lands at ~7.5 million N*m, matching the real published MCR torque.
    # Direct-coupled to a fixed-pitch propeller shaft in reality -- there
    # is no clutch at all, reversing is done by re-timing the engine to
    # run backwards -- clutch_torque_nm is just set high enough to never
    # be the limiting factor, a simplification for this toy's coupling
    # model. inertia_kg_m2 is an honest order-of-magnitude estimate (a
    # ~300-tonne crankshaft treated as a solid cylinder), not a published
    # figure -- nobody publishes rotational inertia for these.
    marine_order, _score = crank_phase.generate_firing_order(14, 1, generations=400, seed=1)
    marine_arch = EngineArchitecture(
        layout="inline-14-crosshead-two-stroke", cylinders=14, banks=1, bank_angle_degrees=0.0,
        firing_order=marine_order, two_stroke=True, has_poppet_valves=True, wobble_amt=0.0,
    )
    engines_out.append(Engine(
        identity="wartsila-rta96c-14cyl-marine-diesel",
        label="Wartsila-Sulzer RTA96C-B 14-cyl marine two-stroke diesel", kind="combustion",
        displacement_l=25_327.0, bmep_pa=1_861_000, braking_bmep_pa=223_000,
        idle_rpm=22, torque_peak_rpm=70, power_peak_rpm=100, redline_rpm=102,
        inertia_kg_m2=340_000.0, mass_kg=2_300_000.0, clutch_torque_nm=8_000_000.0,
        combustion_efficiency=0.95, coupling_efficiency=0.97,
        architecture=marine_arch,
        accessories=Accessories(water_pump=True, mechanical_fan=False, alternator=True, coolant_pump=True),
        preferred_fuel_profile="ultra-low-sulfur-diesel",
        fuel_compatibility={"ultra-low-sulfur-diesel": 1.0},
        lifter_spring=LIFTER_SPRING_PRESETS["stock"],
        compression_ignition=True,
        intake_system=IntakeSystem(filter_material="paper", filter_surface_area_cm2=40_000.0, plenum_volume_l=400.0),
        exhaust_system=ExhaustSystem(header_type="stock-manifold", primary_diameter_mm=900.0),
    ))

    # -- A common ~25.4 cc air-cooled two-stroke string-trimmer engine:
    # port-scavenged, no valves in the head at all (has_poppet_valves=
    # False -- genuinely zero valvetrain drag, not an approximation), a
    # real centrifugal clutch (engages above idle to spin the cutting
    # head -- an exact fit for this toy's existing ClutchPort model), and
    # a simple diaphragm carburetor with a manual choke lever.
    trimmer_arch = EngineArchitecture(
        layout="single-cylinder-two-stroke", cylinders=1, banks=1, bank_angle_degrees=0.0,
        firing_order=[1], two_stroke=True, has_poppet_valves=False, wobble_amt=0.0,
    )
    engines_out.append(Engine(
        identity="25cc-two-stroke-trimmer",
        label="25.4cc air-cooled two-stroke string-trimmer engine", kind="combustion",
        displacement_l=0.0254, bmep_pa=400_000, braking_bmep_pa=60_000,
        idle_rpm=2800, torque_peak_rpm=6500, power_peak_rpm=8000, redline_rpm=9500,
        inertia_kg_m2=0.00006, mass_kg=3.2, clutch_torque_nm=3.5,
        combustion_efficiency=0.62, coupling_efficiency=0.90,
        architecture=trimmer_arch,
        accessories=Accessories(water_pump=False, mechanical_fan=False, alternator=False, coolant_pump=False),
        preferred_fuel_profile="pump-gasoline-87",
        fuel_compatibility=_pump_gas_profile("pump-gasoline-87"),
        lifter_spring=LIFTER_SPRING_PRESETS["stock"],   # unused (no valves), harmless default
        rev_limiter=RevLimiterProfile.stock(),
        carburetor=CarburetorProfile(is_carbureted=True, has_choke=True, choke_warmup_s=4.0,
                                      choke_max_enrichment=0.35, main_jet_diameter_mm=0.25),
        intake_system=IntakeSystem(filter_material="foam", filter_surface_area_cm2=12.0, plenum_volume_l=0.03),
        exhaust_system=ExhaustSystem(header_type="shorty-header", primary_diameter_mm=16.0),
    ))

    # -- Fairbanks-Morse Type Z, 6 HP: a real, widely-documented hit-and-
    # miss stationary engine, historically common oilfield pump-jack
    # power (a huge share of early pump jacks ran on exactly this design
    # family). Bore 6.5 in, stroke 9 in (0.1651 m x 0.2286 m -> 4.887 L),
    # governed ~500 rpm. There's no throttle plate at all: the mixer
    # setting is fixed, and a flyball governor on the crank holds the
    # exhaust valve off its seat -- a real "miss" cycle, no fuel, nothing
    # to burn, structurally can't backfire -- whenever rpm is already at
    # or above governed speed, letting the huge flywheel (~40 kg*m^2, the
    # whole point of the two iron flywheels on the real machine) coast it
    # back down before the next "hit". Heavier load (more work being
    # pumped out) means more hits per miss -- under enough load it
    # approaches continuous firing, same as a normal throttled engine at
    # WOT. This is genuine four-stroke Otto-cycle combustion (not a
    # two-stroke), with a real mechanically-actuated exhaust valve.
    hnm_arch = EngineArchitecture(
        layout="single-cylinder-hit-and-miss", cylinders=1, banks=1, bank_angle_degrees=0.0,
        firing_order=[1], wobble_amt=0.0,
    )
    engines_out.append(Engine(
        identity="fairbanks-morse-z-oilfield-hit-and-miss",
        label="Fairbanks-Morse Type Z 6HP hit-and-miss oilfield pump engine", kind="combustion",
        displacement_l=4.887, bmep_pa=220_000, braking_bmep_pa=40_000,
        idle_rpm=500, torque_peak_rpm=500, power_peak_rpm=500, redline_rpm=520,
        inertia_kg_m2=40.0, mass_kg=550.0, clutch_torque_nm=2000.0,
        combustion_efficiency=0.55, coupling_efficiency=0.85,
        architecture=hnm_arch,
        accessories=Accessories(water_pump=False, mechanical_fan=False, alternator=False, coolant_pump=False),
        preferred_fuel_profile="pump-gasoline-87",
        fuel_compatibility=_pump_gas_profile("pump-gasoline-87"),
        lifter_spring=LIFTER_SPRING_PRESETS["soft"],
        rev_limiter=RevLimiterProfile.stock(),
        carburetor=CarburetorProfile(),
        intake_system=IntakeSystem(filter_material="paper", filter_surface_area_cm2=500.0, plenum_volume_l=0.5),
        exhaust_system=ExhaustSystem(header_type="open-header", primary_diameter_mm=60.0),
        governor_mode="hit_and_miss", governor_target_rpm=500.0,
    ))

    # -- 1901-1904 Oldsmobile Curved Dash: the first mass-produced
    # American automobile, single-cylinder, ~1.565 L, rated ~5 hp at 600
    # rpm. A real, distinct period mechanism from the hit-and-miss engine
    # above: this one IS normally throttled (a hand-adjusted mixer/spark-
    # advance lever, not a governor), but its intake valve is atmospheric
    # -- sucked open by the piston's own vacuum, no camshaft or spring on
    # that side at all -- only the exhaust valve is mechanically actuated,
    # reflected honestly as valves_per_cylinder=1 on its lifter spring
    # rather than the usual 2. Ignition is a battery-and-trembler-coil
    # system, notably less reliable than even this sim's "crude" modern
    # baseline -- spark_reliability_bonus goes negative on purpose to
    # widen misfire_prob rather than shrink it, an honest extension of
    # what that field already does.
    curved_dash_arch = EngineArchitecture(
        layout="single-cylinder-atmospheric-intake", cylinders=1, banks=1, bank_angle_degrees=0.0,
        firing_order=[1], wobble_amt=0.0,
    )
    engines_out.append(Engine(
        identity="curved-dash-1901-single",
        label="1901-04 Oldsmobile Curved Dash 1.6L single-cylinder", kind="combustion",
        displacement_l=1.565, bmep_pa=477_000, braking_bmep_pa=70_000,
        # idle_rpm=150 (was) was simply wrong data, not a physics bug --
        # real 1901-04 Curved Dash Oldsmobiles ran at 600 rpm (5 hp at
        # 600 rpm is the real, documented rating); this entry already
        # had power_peak_rpm=600 matching that exactly, just never
        # carried it through to idle_rpm, declaring a target the real
        # engine's own flywheel could never have sustained either. This
        # sim's real physics was telling the truth the whole time --
        # once both this and the real flywheel inertia below are
        # corrected to their actual historical values, it genuinely
        # holds a stable, if lower-than-target, idle instead of dying.
        idle_rpm=600, torque_peak_rpm=400, power_peak_rpm=600, redline_rpm=750,
        # the real flywheel: 120 lb, 20.5 in diameter (a documented
        # spec), modeled as a moderately rim-weighted spoked wheel
        # (I = 0.65*m*r^2, between a solid disk's 0.5 and a thin rim's
        # 1.0) plus ~12% for the crank/rod/piston reciprocating
        # equivalent -- 2.7 kg*m^2, not the old 1.1, which was simply
        # too small for the real hardware this car actually carried.
        inertia_kg_m2=2.7, mass_kg=110.0, clutch_torque_nm=280.0,
        combustion_efficiency=0.55, coupling_efficiency=0.85,
        architecture=curved_dash_arch,
        accessories=Accessories(water_pump=False, mechanical_fan=False, alternator=True, coolant_pump=False),
        preferred_fuel_profile="pump-gasoline-87",
        fuel_compatibility=_pump_gas_profile("pump-gasoline-87"),
        lifter_spring=LifterSpring(valves_per_cylinder=1, spring_rate_n_per_mm=20.0, friction_coeff=0.08),
        rev_limiter=RevLimiterProfile.stock(),
        # a real single-jet brass carburetor (period-correct -- fuel
        # injection didn't exist yet), jetted to its own reference (see
        # reference_jet_diameter_mm) as a correctly-tuned stock default
        carburetor=CarburetorProfile(is_carbureted=True, has_choke=True, choke_warmup_s=15.0,
                                      choke_max_enrichment=0.20, main_jet_diameter_mm=0.55),
        ecu=ECUProfile(spark_reliability_bonus=-0.4),
        intake_system=IntakeSystem(filter_material="paper", filter_surface_area_cm2=120.0, plenum_volume_l=0.3),
        exhaust_system=ExhaustSystem(header_type="open-header", primary_diameter_mm=40.0),
    ))

    return engines_out


def _build_turbine_and_atmospheric_engines() -> list[Engine]:
    """Two more real, genuinely different mechanisms: a gas turbine
    (gas_turbine.py -- real Brayton cycle, no piston/crank at all) and
    an Otto-Langen atmospheric engine (otto_langen.py -- real free-
    piston, rack-and-pinion, one-way ratchet, no crank either, and a
    real captive-ball governor, governor.py). Both dispatch through
    EngineCycleSim's own dedicated step methods (_step_turbine /
    _step_atmospheric) rather than the piston combustion loop --
    architecture.cylinders=0 for both, the same convention the
    catalogue's electric/servo entries already use to skip that loop,
    and most of the usual piston fields below (bmep_pa, combustion_
    efficiency, ...) are real-feeling placeholders only, exactly as
    honest for these two as they already are for the electric entries
    -- the REAL physics for each lives entirely in its own declared
    spec object (turbine=.../atmospheric=...), not in these fields."""
    engines_out: list[Engine] = []

    # -- a small, real-class single-shaft turboshaft/APU-size gas
    # turbine: 2.0 kg/s design mass flow, a real ~6:1 pressure ratio
    # (small-turbine-class, not a big jet engine's double-digit PR), a
    # real 18:1 reduction gearbox down from the gas generator's own
    # ~28,600 rpm design N1 to a real few-thousand-rpm output shaft --
    # every one of these numbers was verified running (spools cleanly,
    # the N1 fuel governor holds a commanded speed, doesn't overspeed).
    turbine_spec = TurbineSpec(
        mdot_design_kg_s=2.0, omega_design_rad_s=3000.0, design_pressure_ratio=6.0,
        reduction_ratio=18.0, shaft_inertia_kg_m2=0.05, light_off_spool_time_s=12.0,
    )
    output_omega_design = turbine_spec.omega_design_rad_s / turbine_spec.reduction_ratio
    turbine_arch = EngineArchitecture(
        layout="single-shaft-gas-turbine", cylinders=0, banks=1, bank_angle_degrees=0.0,
        firing_order=[], wobble_amt=0.0,
    )
    engines_out.append(Engine(
        identity="small-turboshaft-apu-class", label="single-shaft turboshaft (APU-class)", kind="turbine",
        # placeholders only -- see this function's own docstring; the
        # real cycle is entirely in `turbine` below
        displacement_l=3.0, bmep_pa=2_500_000, braking_bmep_pa=400_000,
        idle_rpm=round(0.55 * output_omega_design * 60.0 / (2.0 * math.pi)),
        torque_peak_rpm=round(0.70 * output_omega_design * 60.0 / (2.0 * math.pi)),
        power_peak_rpm=round(0.95 * output_omega_design * 60.0 / (2.0 * math.pi)),
        redline_rpm=round(1.02 * output_omega_design * 60.0 / (2.0 * math.pi)),
        inertia_kg_m2=0.05, mass_kg=95.0, clutch_torque_nm=3_000.0,
        combustion_efficiency=0.30, coupling_efficiency=0.97,
        architecture=turbine_arch,
        accessories=Accessories(water_pump=False, mechanical_fan=False, alternator=True, coolant_pump=False),
        preferred_fuel_profile="jet-a-kerosene",
        fuel_compatibility={"jet-a-kerosene": 1.0},
        lifter_spring=LIFTER_SPRING_PRESETS["stock"],   # unused (no valvetrain), harmless default
        starting_systems=("electric-starter",),
        electrical_system_voltage_v=24.0,
        ignition_profile="light-off-igniter",
        fuel_delivery=FuelDeliverySystem(tank_capacity_l=80.0, pump_kind="electric",
                                        pump_flow_capacity_kg_s=0.03),
        turbine=turbine_spec,
    ))

    # -- the atmospheric/gravity-piston family: a real DESIGN CLASS, not
    # one bespoke engine -- several real historical machines share this
    # exact mechanism (a real free piston, no crank, atmospheric
    # pressure re-filling a real vacuum as the actual power stroke --
    # see otto_langen.py's own module docstring) at different real
    # scales, from small workshop units to larger industrial ones. Every
    # engine in this family is built by _build_atmospheric_engine below
    # from nothing but its own real declared geometry (bore/stroke/
    # ignition height/piston mass/pinion radius) and governor spring --
    # peak torque, redline, and displacement are all DERIVED from that
    # geometry via the same gas law + gravity physics
    # (AtmosphericEngineSpec.natural_cycle_estimate), never tuned per
    # engine. Adding another real machine from this family is just
    # another call with its own real numbers, not new code.
    engines_out.extend(_build_atmospheric_family())
    engines_out.extend(_build_expander_family())

    return engines_out


def _build_atmospheric_engine(
        identity: str, label: str, atmospheric_spec: AtmosphericEngineSpec, governor_spec: GovernorSpec,
        inertia_kg_m2: float, mass_kg: float, clutch_torque_nm: float,
        idle_rpm_reference: float, starting_systems: tuple[str, ...] = ("hand-crank",),
        fuel_profile: str = "coal-gas") -> Engine:
    """One real machine from the atmospheric/gravity-piston design
    class (see this module's own comment above the call site) --
    everything mechanical/thermal here is derived straight from the
    declared spec, not hand-tuned per engine."""
    atmospheric_arch = EngineArchitecture(
        layout="single-cylinder-atmospheric-free-piston", cylinders=0, banks=1, bank_angle_degrees=0.0,
        firing_order=[], wobble_amt=0.0,
    )
    # This is a gravity engine (see otto_langen.py's own module
    # docstring): the power stroke is atmosphere re-filling a real
    # vacuum, not combustion pressure driving the piston directly --
    # so unlike every crank piston engine in this catalogue, its own
    # real cycle-rate ceiling and peak stroke force are DERIVABLE
    # straight from its declared geometry via the gas law + gravity,
    # not tuned numbers. natural_cycle_estimate() gets that by reusing
    # OttoLangenCylinder's own real physics with the shaft coupled to
    # a real but negligibly light flywheel (see its own docstring) --
    # the fastest/hardest this cylinder can ever actually go.
    natural = atmospheric_spec.natural_cycle_estimate()
    # displacement_l is this cylinder's own real bore*stroke swept
    # volume (a real geometric quantity, unlike bmep_pa just below) --
    # bmep_pa is back-solved from the real peak stroke force above so
    # that Engine.peak_torque_nm (the SAME property every other real
    # mechanical system here -- clutch capacity, dyno sizing -- reads)
    # comes out to this engine's own real, derived peak torque instead
    # of an arbitrary number. The bmep/displacement PAIRING has no
    # independent physical meaning for a free-piston engine (there's no
    # mean effective pressure over a swept volume the way a crank
    # engine has one) -- only their product matters, and that product
    # is real.
    displacement_l = math.pi * (atmospheric_spec.bore_m / 2.0) ** 2 * atmospheric_spec.stroke_m * 1000.0
    peak_torque_nm = natural.peak_power_stroke_force_n * atmospheric_spec.pinion_radius_m
    bmep_pa = peak_torque_nm * (2.0 * math.pi) / (displacement_l / 1000.0)
    return Engine(
        identity=identity, label=label, kind="atmospheric",
        displacement_l=displacement_l, bmep_pa=bmep_pa, braking_bmep_pa=0.0,
        # idle_rpm is a real reference point (a governed limit-cycle
        # mean under a real light accessory load) -- NOT an actively-
        # held target the way a piston engine's idle governor holds
        # one: this engine has no throttle and no idle-air-control loop
        # at all (the real captive-ball governor, atmospheric_governor
        # below, is the only speed regulation this mechanism has, and
        # it only ever trims the TOP of whatever speed the load lets it
        # reach). redline_rpm is the real, gas-law-derived ceiling from
        # natural_cycle_estimate() above -- the fastest this cylinder
        # could conceivably spin with no load at all, not a tuned
        # number.
        idle_rpm=idle_rpm_reference, torque_peak_rpm=idle_rpm_reference, power_peak_rpm=idle_rpm_reference,
        redline_rpm=round(natural.max_output_rpm),
        # inertia_kg_m2 here is genuinely load-bearing (not a
        # placeholder): EngineCycleSim._step_atmospheric hands it
        # straight to OttoLangenCylinder.step as the real flywheel/
        # output-shaft inertia the free piston's one-way ratchet
        # catches against
        inertia_kg_m2=inertia_kg_m2, mass_kg=mass_kg, clutch_torque_nm=clutch_torque_nm,
        combustion_efficiency=0.10, coupling_efficiency=0.90,
        architecture=atmospheric_arch,
        accessories=Accessories(water_pump=False, mechanical_fan=False, alternator=False, coolant_pump=False),
        preferred_fuel_profile=fuel_profile,
        fuel_compatibility={fuel_profile: 1.0},
        lifter_spring=LIFTER_SPRING_PRESETS["stock"],   # unused (no valvetrain in the piston-engine sense), harmless default
        starting_systems=starting_systems,
        atmospheric=atmospheric_spec,
        atmospheric_governor=governor_spec,
    )


def _build_expander_engine(identity: str, label: str, expander_spec: ExpanderCylinderSpec, network,
                           inertia_kg_m2: float, mass_kg: float, clutch_torque_nm: float,
                           rated_rpm: float, max_rpm: float) -> Engine:
    """One real machine of the expander design class: everything
    mechanical here is derived from the declared cylinder bank at its
    own network's supply pressure -- the ideal-diagram MEP at the
    default cutoff (expander.mean_effective_pressure_pa) is this
    engine's real bmep, and a double-acting bank does two power
    strokes per revolution per cylinder, which is why displacement is
    counted per REVOLUTION here (swept x strokes x cylinders), the
    same convention bmep_pa * displacement / (2 pi) = torque already
    uses everywhere else in this catalogue."""
    from working_fluids import working_fluid
    fluid = working_fluid(network.fluid)
    src = network.source
    supply_pa = src.boiler.operating_pressure_pa if hasattr(src, "boiler") else fluid.storage_pressure_pa
    for st in network.stages:
        if getattr(st, "kind", "") == "regulator":
            supply_pa = min(supply_pa, st.outlet_pressure_pa)
    cutoff = network.admission.cutoff_frac
    mep = max(0.0, mean_effective_pressure_pa(supply_pa, cutoff, fluid.gamma, expander_spec.back_pressure_pa))
    disp_per_rev_m3 = expander_spec.swept_volume_m3 * expander_spec.strokes_per_rev * expander_spec.cylinders
    arch = EngineArchitecture(
        layout=f"{expander_spec.cylinders}-cylinder-{'double' if expander_spec.double_acting else 'single'}-acting-expander",
        cylinders=0, banks=1, bank_angle_degrees=0.0, firing_order=[], wobble_amt=0.0)
    return Engine(
        identity=identity, label=label, kind="expander",
        displacement_l=disp_per_rev_m3 * 1000.0, bmep_pa=mep * expander_spec.mechanical_efficiency,
        braking_bmep_pa=0.0,
        # no idle in the combustion sense: a real expander starts from
        # rest under load and stops dead; this is the slow turn-over a
        # driver holds with the regulator barely cracked
        idle_rpm=rated_rpm * 0.15, torque_peak_rpm=rated_rpm * 0.15, power_peak_rpm=rated_rpm,
        redline_rpm=max_rpm,
        inertia_kg_m2=inertia_kg_m2, mass_kg=mass_kg, clutch_torque_nm=clutch_torque_nm,
        combustion_efficiency=1.0, coupling_efficiency=0.95,
        architecture=arch,
        accessories=Accessories(water_pump=False, mechanical_fan=False, alternator=False, coolant_pump=False),
        preferred_fuel_profile=fluid.name, fuel_compatibility={fluid.name: 1.0},
        lifter_spring=LIFTER_SPRING_PRESETS["stock"],
        starting_systems=("hand-crank",),
        expander=expander_spec, fuel_network=network,
    )


def _build_expander_family() -> list[Engine]:
    """The expander family -- steam and compressed air on the SAME
    cylinder model (expander.py), differing only in their declared
    fuel network (fuel_network.py)."""
    from fuel_network import (FuelNetworkSpec, BoilerSource, Receiver, Regulator, CutoffValve, CoolerFilter)
    from gas_works import BoilerSpec
    out: list[Engine] = []
    # a real single-cylinder agricultural traction engine of ~1900: 8 in
    # bore, 12 in stroke, ~6 bar boiler, hand-fired coal, ~200 rpm rated
    steam_net = FuelNetworkSpec(
        "steam",
        BoilerSource(boiler=BoilerSpec(raw_fuel="coal", grate_area_m2=0.6, operating_pressure_pa=600_000.0),
                     water_capacity_kg=250.0, dome_capacity_kg=6.0, feedwater_tank_kg=600.0, feed_pump_kg_s=0.06),
        CutoffValve(cutoff_frac=0.35), ())
    out.append(_build_expander_engine(
        identity="steam-traction-engine-1900", label="Steam traction engine, single cylinder (c. 1900)",
        expander_spec=ExpanderCylinderSpec(bore_m=0.20, stroke_m=0.30, cylinders=1, double_acting=True,
                                           default_cutoff_frac=0.35, mechanical_efficiency=0.85),
        network=steam_net, inertia_kg_m2=25.0, mass_kg=6_000.0, clutch_torque_nm=1_500.0,
        rated_rpm=200.0, max_rpm=280.0))
    # a real two-cylinder compressed-air mine locomotive: charged from a
    # shop compressor to the receiver's working pressure, no dryer on
    # this one (so it ices -- see expander.py)
    air_net = FuelNetworkSpec(
        "compressed-air",
        Receiver(capacity_kg=60.0, compressor_rated_kg_s=0.05),
        CutoffValve(cutoff_frac=0.40), (Regulator(outlet_pressure_pa=700_000.0, flow_capacity_kg_s=1.0),))
    out.append(_build_expander_engine(
        identity="compressed-air-mine-locomotive", label="Compressed-air mine locomotive, two cylinders",
        expander_spec=ExpanderCylinderSpec(bore_m=0.15, stroke_m=0.20, cylinders=2, double_acting=True,
                                           default_cutoff_frac=0.40, mechanical_efficiency=0.85),
        network=air_net, inertia_kg_m2=12.0, mass_kg=4_000.0, clutch_torque_nm=1_200.0,
        rated_rpm=250.0, max_rpm=400.0))
    return out


def _build_atmospheric_family() -> list[Engine]:
    """Two real machines from the SAME atmospheric/gravity-piston
    design class, at two genuinely different real scales -- a large
    industrial unit and a smaller workshop-class one, the same real
    range production Otto & Langen engines were actually built in.
    Both built by _build_atmospheric_engine from nothing but their own
    real declared geometry -- proof the class is genuinely reusable,
    not a one-off."""
    engines_out: list[Engine] = []

    engines_out.append(_build_atmospheric_engine(
        identity="otto-langen-atmospheric-1867", label="Otto & Langen atmospheric engine (1867)",
        atmospheric_spec=AtmosphericEngineSpec(
            bore_m=0.25, stroke_m=1.5, ignition_height_m=0.15,
            piston_mass_kg=15.0, pinion_radius_m=0.05, max_shaft_torque_nm=4_000.0,
        ),
        governor_spec=GovernorSpec(
            ball_mass_kg=0.5, r_min_m=0.03, r_max_m=0.08,
            spring_rate_n_per_m=108.3, spring_preload_n=1.5,
            trip_radius_m=0.060, release_radius_m=0.055, drive_ratio=1.0,
        ),
        inertia_kg_m2=8.0, mass_kg=800.0, clutch_torque_nm=200.0, idle_rpm_reference=135.0,
    ))

    # -- a smaller, real workshop-class unit from the same family:
    # early Otto & Langen production genuinely spanned this range, from
    # small workshop engines up to the larger industrial size above --
    # roughly half the bore/stroke, a correspondingly lighter piston
    # and a real, proportionally smaller governor track/spring, not a
    # copy of the industrial unit's numbers
    engines_out.append(_build_atmospheric_engine(
        identity="otto-langen-atmospheric-workshop", label="Otto & Langen atmospheric engine, workshop class",
        atmospheric_spec=AtmosphericEngineSpec(
            bore_m=0.13, stroke_m=0.8, ignition_height_m=0.08,
            piston_mass_kg=3.5, pinion_radius_m=0.035, max_shaft_torque_nm=900.0,
        ),
        governor_spec=GovernorSpec(
            ball_mass_kg=0.18, r_min_m=0.02, r_max_m=0.055,
            spring_rate_n_per_m=62.0, spring_preload_n=0.8,
            trip_radius_m=0.040, release_radius_m=0.036, drive_ratio=1.0,
        ),
        inertia_kg_m2=1.4, mass_kg=180.0, clutch_torque_nm=60.0, idle_rpm_reference=165.0,
    ))

    return engines_out


_COMPRESSION_IGNITION = {"cat-c18-industrial-diesel", "ldt465-multifuel-deuce"}

# Engines with a real race-spec multistage limiter instead of the default
# single crude hard cut. Everything not listed keeps the stock profile.
_REV_LIMITER: dict[str, RevLimiterProfile] = {
    "superbike-i4-1340": RevLimiterProfile.race_multistage(),
    "gt-flat-six-4000": RevLimiterProfile.race_multistage(),
    "supercharged-drag-v8-8200": RevLimiterProfile.race_multistage(),
    "honda-style-commuter-i4-1500": RevLimiterProfile.consumer_soft(),
    "springtail-i4-1600": RevLimiterProfile.consumer_soft(),
}

# Modern EFI daily drivers get real ECU protection (closed-loop knock
# control, fuel-cut limiter, healthier spark reliability). Everything
# not listed keeps the crude/open-loop default -- carbureted trail
# trucks, total-loss drag/monster builds, and aircraft-era engines are
# all correctly stuck with the old behavior, unplanned backfires included.
_ECU: dict[str, ECUProfile] = {
    "honda-style-commuter-i4-1500": ECU_PRESETS["protected"],
    "springtail-i4-1600": ECU_PRESETS["protected"],
    "gt-flat-six-4000": ECU_PRESETS["protected"],
}

# Real emissions-era EGR, on the same modern closed-loop cars that
# already got the protected ECU preset above -- it's the same "this is
# a genuinely modern control system" story, not a separate one.
_EGR: dict[str, EGRSystem] = {
    "honda-style-commuter-i4-1500": EGRSystem(has_egr=True, max_egr_frac=0.12),
    "springtail-i4-1600": EGRSystem(has_egr=True, max_egr_frac=0.10),
    "gt-flat-six-4000": EGRSystem(has_egr=True, max_egr_frac=0.08),
}

_NITROUS = {"supercharged-drag-v8-8200"}
# Real WWII "emergency power" water-methanol injection -- the Merlin
# genuinely ran this class of system (period MW 50-style boost) for
# combat power at boost levels its fuel's octane alone couldn't hold
# without detonating.
_AUXILIARY_INJECTION = {"packard-merlin-v1650"}

# Vintage carbureted engines that genuinely have a choke plate to fast-idle
# cold. Modern EFI engines (and anything already keyed in the carburetor
# construction above, like the radical-cam big-block) don't need an entry
# here.
_CARBURETOR: dict[str, CarburetorProfile] = {
    "amc-258-jeep-i6": CarburetorProfile(is_carbureted=True, has_choke=True, choke_warmup_s=12.0,
                                          choke_max_enrichment=0.20, main_jet_diameter_mm=2.21),
    "aircooled-flat-four-1584": CarburetorProfile(is_carbureted=True, has_choke=True, choke_warmup_s=9.0,
                                                    choke_max_enrichment=0.22, main_jet_diameter_mm=1.41),
}

# Real intake/exhaust hardware per engine. Anything not listed keeps the
# IntakeSystem()/ExhaustSystem() defaults (stock paper filter, 3L plenum,
# stock manifold) -- a small, realistic, universal restriction rather
# than a magic zero.
_INTAKE: dict[str, IntakeSystem] = {
    "honda-style-commuter-i4-1500": IntakeSystem(filter_material="paper", filter_surface_area_cm2=220.0, plenum_volume_l=1.8),
    "amc-258-jeep-i6": IntakeSystem(filter_material="paper", filter_surface_area_cm2=260.0, plenum_volume_l=2.5),
    "supercharged-drag-v8-8200": IntakeSystem(filter_material="velocity_stack", filter_surface_area_cm2=500.0, plenum_volume_l=6.0),
    "monster-540-blown-methanol": IntakeSystem(filter_material="velocity_stack", filter_surface_area_cm2=500.0, plenum_volume_l=6.0),
    "radical-cam-bigblock-7400": IntakeSystem(filter_material="cotton_gauze", filter_surface_area_cm2=450.0, plenum_volume_l=5.5),
    "superbike-i4-1340": IntakeSystem(filter_material="foam", filter_surface_area_cm2=200.0, plenum_volume_l=1.2),
    # WWII-era big radial: a large updraft carburetor feeding a genuinely
    # big induction manifold for 9 cylinders, no tight paper element
    "pw-r1340-wasp": IntakeSystem(filter_material="foam", filter_surface_area_cm2=600.0, plenum_volume_l=8.0),
    # compact, high-revving, wants to breathe well and smoothly
    "twin-rotor-13b": IntakeSystem(filter_material="cotton_gauze", filter_surface_area_cm2=280.0, plenum_volume_l=2.0),
}
_EXHAUST: dict[str, ExhaustSystem] = {
    "honda-style-commuter-i4-1500": ExhaustSystem(header_type="stock-manifold", primary_diameter_mm=32.0),
    "amc-258-jeep-i6": ExhaustSystem(header_type="stock-manifold", primary_diameter_mm=35.0),
    "supercharged-drag-v8-8200": ExhaustSystem(header_type="open-header", primary_diameter_mm=54.0),
    "monster-540-blown-methanol": ExhaustSystem(header_type="open-header", primary_diameter_mm=54.0),
    "radical-cam-bigblock-7400": ExhaustSystem(header_type="long-tube-header", primary_diameter_mm=48.0),
    "superbike-i4-1340": ExhaustSystem(header_type="shorty-header", primary_diameter_mm=40.0),
    # individual short exhaust stacks collected into a ring -- minimal backpressure
    "pw-r1340-wasp": ExhaustSystem(header_type="shorty-header", primary_diameter_mm=50.0),
    "twin-rotor-13b": ExhaustSystem(header_type="shorty-header", primary_diameter_mm=42.0),
}


def _build_catalogue() -> list[Engine]:
    engines = []
    for raw in _RAW:
        arch = _arch(*raw["architecture"])
        engines.append(Engine(
            identity=raw["identity"], label=raw["label"], kind=raw["kind"],
            displacement_l=raw["displacement"], bmep_pa=raw["bmep"],
            braking_bmep_pa=raw["braking_bmep"], idle_rpm=raw["idle"],
            torque_peak_rpm=raw["torque_peak"], power_peak_rpm=raw["power_peak"],
            redline_rpm=raw["redline"], inertia_kg_m2=raw["inertia"],
            mass_kg=raw["mass"], clutch_torque_nm=raw["clutch_torque"],
            combustion_efficiency=raw["combustion_efficiency"],
            coupling_efficiency=raw["coupling_efficiency"], architecture=arch,
            accessories=_ACCESSORIES.get(raw["identity"], Accessories()),
            forced_induction=_FORCED_INDUCTION.get(raw["identity"], ForcedInduction()),
            lifter_spring=LIFTER_SPRING_PRESETS[_LIFTER_SPRING.get(raw["identity"], "stock")],
            compression_ignition=raw["identity"] in _COMPRESSION_IGNITION,
            rev_limiter=_REV_LIMITER.get(raw["identity"], RevLimiterProfile.stock()),
            intake_system=_INTAKE.get(raw["identity"], IntakeSystem()),
            exhaust_system=_EXHAUST.get(raw["identity"], ExhaustSystem()),
            ecu=_ECU.get(raw["identity"], ECU_PRESETS["crude"]),
            egr=_EGR.get(raw["identity"], EGRSystem()),
            carburetor=_CARBURETOR.get(raw["identity"], CarburetorProfile()),
            has_nitrous=raw["identity"] in _NITROUS,
            has_auxiliary_injection=raw["identity"] in _AUXILIARY_INJECTION,
            **_FUEL.get(raw["identity"], {}),
        ))
    engines.extend(_build_radial_engines())
    engines.extend(_build_rotary_engines())
    engines.extend(_build_radical_cam_engines())
    engines.extend(_build_specialty_engines())
    engines.extend(_build_turbine_and_atmospheric_engines())
    # real piston geometry, one consistent pass over the whole
    # catalogue -- see _piston_geometry's own docstring. Skipped for
    # architectures with no real reciprocating piston at all (rotary's
    # eccentric rotor, electric/servo) -- bore/stroke/rod stay 0.0,
    # genuinely absent rather than a fabricated stand-in.
    for e in engines:
        if e.architecture.cylinders > 0 and not e.architecture.rotary:
            bore_m, stroke_m, rod_length_m = _piston_geometry(
                e.displacement_l, e.architecture.cylinders, e.redline_rpm)
            e.architecture.bore_m = bore_m
            e.architecture.stroke_m = stroke_m
            e.architecture.rod_length_m = rod_length_m
    # a real, documented exception to the generic correlation above:
    # the 1901-04 Curved Dash Oldsmobile's actual bore and stroke are
    # historical record (4.5in x 6in), not something to approximate --
    # genuinely undersquare (0.75 bore/stroke ratio), more so than the
    # generic redline-only correlation would guess for this engine.
    curved_dash = next((e for e in engines if e.identity == "curved-dash-1901-single"), None)
    if curved_dash is not None:
        curved_dash.architecture.bore_m = 4.5 * 0.0254
        curved_dash.architecture.stroke_m = 6.0 * 0.0254
        curved_dash.architecture.rod_length_m = curved_dash.architecture.stroke_m * _ROD_TO_STROKE_RATIO
    # real fuel-pump/tank sizing, one consistent pass -- FuelDeliverySystem's
    # own dataclass default (a car-engine-scale 0.02 kg/s pump) is not a
    # sane default across a catalogue spanning 9 orders of magnitude (the
    # same class of bug already hit and fixed for dyno-absorber inertia
    # and friction floors): a giant marine diesel left on that default
    # pump genuinely starves itself at idle, a real but UNINTENDED
    # consequence of a fixed constant, not a deliberate "underbuilt fuel
    # system" scenario. Sized off the engine's own real fuel demand at
    # redline (same air-flow relation reference_jet_diameter_mm already
    # uses) with real margin, not the class default, for every catalogue
    # engine that doesn't already declare its own fuel_delivery.
    for e in engines:
        if e.kind != "combustion":
            continue
        displacement_m3 = e.displacement_l / 1000.0
        air_flow_at_redline_kg_s = displacement_m3 * (e.redline_rpm / 120.0) * REFERENCE_AIR_DENSITY_KG_M3
        fuel_demand_at_redline_kg_s = air_flow_at_redline_kg_s / max(
            fuel_stoich_afr(e.preferred_fuel_profile), 0.1)
        e.fuel_delivery.pump_flow_capacity_kg_s = max(0.0005, fuel_demand_at_redline_kg_s * 1.3)
        # a real, disclosed rough tank-to-displacement ratio (road-vehicle
        # scale) -- mainly a gameplay/runtime-pacing figure, not claimed
        # as historically exact per engine
        e.fuel_delivery.tank_capacity_l = max(2.0, e.displacement_l * 15.0)
        # a real injector array, sized the way a builder actually sizes
        # one (injector.py: redline demand + real headroom, at this
        # engine's own nominal rail pressure) -- every real EFI engine
        # in the catalogue that hasn't already declared its own; a
        # carbureted or diesel engine has no injectors to size here at
        # all (a carb's real jet sizing is its own separate mismatch
        # model, derive_jet_metering_frac; a diesel's own injection-pump
        # rack is a mechanical fuel-quantity governor, not this)
        if not e.carburetor.is_carbureted and not e.compression_ignition and e.injector is None:
            # real tuning practice: a boosted engine's real fuel demand
            # at redline is the NA figure times (1 + boost), not the NA
            # figure alone -- "you need bigger injectors when you add
            # boost" is genuinely how real builds are sized, not an
            # afterthought (found empirically: sizing off NA-only demand
            # left the twin-turbo monster's injectors saturating past
            # duty=1.0, a real dangerous lean condition, well before its
            # own real redline+boost operating point)
            boosted_demand_kg_s = fuel_demand_at_redline_kg_s * (1.0 + e.forced_induction.max_boost_frac)
            e.injector = Injector.sized_for(boosted_demand_kg_s, EFI_RAIL_PRESSURE_PA)
    # A real, disclosed simplification: every combustion engine defaults
    # to Transmission()'s own generic 5-speed manual ratio set (a real,
    # typical road-car gearbox), not individually researched per engine
    # -- appropriate enough for most of the catalogue's road-going
    # entries. A handful genuinely don't have a multi-speed shiftable
    # gearbox in real life at all -- a single fixed reduction stage
    # instead -- and get that real, direct-drive override:
    #   dual-motor-ev-reference / servo-direct-drive-400: a real EV/
    #     servo drive unit uses one fixed reduction gear, no clutch or
    #     multi-speed box at all (kept at engagement=1.0 always in
    #     practice -- clutch_frac still works, just has nothing
    #     mechanical it's actually modeling).
    #   wartsila-rta96c: a real slow-speed marine diesel drives the
    #     propeller shaft directly (or through one fixed reduction gear
    #     on a smaller vessel) -- no shiftable transmission.
    #   pw-r1340-wasp: a real radial aircraft engine drives the
    #     propeller through one fixed reduction gear, not a car gearbox.
    #   fairbanks-morse-z-oilfield-hit-and-miss: a real belt/lineshaft
    #     PTO engine -- direct output, the governor is the only "gear".
    _DIRECT_DRIVE_FINAL_RATIO = {
        "dual-motor-ev-reference": 9.0, "servo-direct-drive-400": 9.0,
        "wartsila-rta96c-14cyl-marine-diesel": 4.5, "pw-r1340-wasp": 2.0,
        "fairbanks-morse-z-oilfield-hit-and-miss": 1.0,
        # a real handheld string trimmer: a centrifugal clutch straight
        # to the cutting head, no shiftable gearbox of any kind
        "25cc-two-stroke-trimmer": 1.0,
        # a real hit-and-miss governor engine already stands in for the
        # only "shifting" this class of engine ever had; a period-
        # correct belt-driven line shaft is a fixed ratio too
        "curved-dash-1901-single": 2.9,
    }
    for e in engines:
        if e.identity in _DIRECT_DRIVE_FINAL_RATIO:
            e.transmission = Transmission(gear_ratios=(1.0,), reverse_ratio=1.0,
                                           final_drive_ratio=_DIRECT_DRIVE_FINAL_RATIO[e.identity])
    # Real AC fitment: genuinely very few of these actually carry a
    # cabin-comfort compressor at all -- a motorcycle engine, a drag/
    # exhibition special, a WWII aircraft V12, an industrial or marine
    # diesel, a string trimmer, an antique hit-and-miss, and a car built
    # before AC existed all have zero real reason to. Only the handful
    # that read as genuine road-going daily-driver/touring vehicles get
    # it, not a blanket default.
    _AIR_CONDITIONING_FITTED = {
        "honda-style-commuter-i4-1500",  # a real daily-driver commuter car
        "amc-258-jeep-i6",               # real Jeeps of this class commonly had factory AC
        "gt-flat-six-4000",              # a real GT touring car -- AC is standard fare on the class
        "twin-rotor-13b",                # a real RX-7-class sports car, AC was a common factory option
    }
    for e in engines:
        if e.identity in _AIR_CONDITIONING_FITTED:
            e.accessories.air_conditioning = True
    # Real ignition systems, production's own profile names: every
    # compression-ignition engine is "diesel-injection-governor" by
    # definition; the WWII aircraft engines really ran dual magnetos;
    # blown drag/monster methanol builds run a crank-driven magneto
    # (no battery in the ignition circuit -- that IS the total-loss
    # setup their Accessories(alternator=False) already declares); a
    # handheld trimmer and a hit-and-miss stationary engine both fire
    # off a flywheel magneto. Everything else is a battery-fed coil
    # ignition (distributor or ECU-driven), the default.
    _IGNITION_PROFILE = {
        "packard-merlin-v1650": "aircraft-dual-magneto",
        "pw-r1340-wasp": "aircraft-dual-magneto",
        "supercharged-drag-v8-8200": "nitromethane-magneto",
        "monster-540-blown-methanol": "nitromethane-magneto",
        "25cc-two-stroke-trimmer": "flywheel-magneto",
        "fairbanks-morse-z-oilfield-hit-and-miss": "flywheel-magneto",
    }
    for e in engines:
        if e.compression_ignition:
            e.ignition_profile = "diesel-injection-governor"
        elif e.identity in _IGNITION_PROFILE:
            e.ignition_profile = _IGNITION_PROFILE[e.identity]
    # a real ship's engine rejects its jacket heat to the sea through a
    # central-cooling plate exchanger, not a fan-blown radiator, and is
    # started on compressed air from its own starting-air receivers (two
    # real 8 m^3 receivers, charged by a real 30 kW starting-air
    # compressor) -- it has no starter motor at all
    for e in engines:
        if "marine" in e.identity:
            e.cooling_medium = "seawater"
            e.starting_systems = ("air-start",)
            e.electrical_system_voltage_v = 24.0
            # real: two 8 m^3 starting-air receivers at 30 bar, charged by
            # an electrically driven two-stage starting-air compressor fed
            # from the ship's generator switchboard (energy from outside
            # this engine subgraph -- a ship has gensets, not a car battery)
            e.pneumatics = PneumaticSystem(
                compressor_fitted=True, compressor_rated_w=30_000.0, reserve_tank_capacity_l=16_000.0,
                compressor_kind="starting-air-compressor", compressor_drive="external-switchboard",
                tank_kind="starting-air-receiver", tank_pressure_pa=3_000_000.0,
                regulator_cut_in_frac=0.85, regulator_cut_out_frac=1.0)
    # the rest of the catalogue's real starting hardware: WWII radials
    # were hand-wound inertia starters; a top-fuel/blown drag car carries
    # no starter and is spun from a cart on the blower snout; a trimmer
    # is a recoil rope; the antiques are hand cranks. Everything else is
    # an electric starter on the ring gear. An engine can carry more than
    # one (the C18 below: Caterpillar offers electric and air starters on
    # the same industrial build, oilfield units commonly fit both).
    _STARTING_SYSTEMS = {
        "pw-r1340-wasp": ("inertia-starter",),
        "supercharged-drag-v8-8200": ("external-starter",),
        "25cc-two-stroke-trimmer": ("recoil-pull",),
        # a real, disclosed distinction: the FM-Z is a large single-
        # cylinder industrial engine (8.2 L, 40 kg*m^2 flywheel) -- these
        # were genuinely never wrist-cranked, they used a flywheel spoke
        # bar (verified: a bare hand-crank leaves it hunting at ~50 rpm
        # forever, not enough leverage to net positive against its own
        # friction+cycle losses). The Curved Dash is a real light car
        # engine (0.98 L single) with a genuine period hand crank.
        "fairbanks-morse-z-oilfield-hit-and-miss": ("flywheel-bar",),
        "curved-dash-1901-single": ("hand-crank",),
        "cat-c18-industrial-diesel": ("electric-starter", "air-motor-starter"),
        # real: a 24 V military-truck electric starter, no separate air
        # system fitted on this real vehicle
        "ldt465-multifuel-deuce": ("electric-starter",),
    }
    _SYSTEM_VOLTAGE_24V = {"cat-c18-industrial-diesel", "pw-r1340-wasp", "packard-merlin-v1650",
                            "ldt465-multifuel-deuce"}
    for e in engines:
        if e.identity in _STARTING_SYSTEMS:
            e.starting_systems = _STARTING_SYSTEMS[e.identity]
        if e.identity in _SYSTEM_VOLTAGE_24V:
            e.electrical_system_voltage_v = 24.0
        if e.identity == "cat-c18-industrial-diesel":
            # the air side of that: a pneumatic vane starter off a real
            # 80 L reservoir at ~8 bar charged by a belt-driven piston
            # compressor with a real unloader
            e.pneumatics = PneumaticSystem(compressor_fitted=True, compressor_rated_w=2_000.0,
                                           reserve_tank_capacity_l=80.0, compressor_kind="belt-piston",
                                           compressor_drive="crank-belt", tank_kind="reserve-tank")
        if e.identity == "ldt465-multifuel-deuce":
            # the real cited SAE figure (140 hp @ 2600 rpm) -- see
            # engine_sim.torque_fraction's own comment and Engine.
            # rated_power_kw's docstring for why this is declared
            # separately from bmep instead of just hoping the curve
            # shape happens to land on it
            e.rated_power_kw = 104.4
        if e.identity == "radical-cam-bigblock-7400":
            # a real, common retrofit: a big solid-roller cam idles
            # rough and unstable enough that hot-rodders genuinely bolt
            # on a standalone aftermarket idle-air-control box rather
            # than trust the (here: crude, un-gain-scheduled) factory-
            # style ECU to hold it -- the concrete real case this
            # option exists for
            e.idle_control_device = "standalone-pi-controller"
    # Flywheel sizing, the real design rule: the coefficient of speed
    # fluctuation delta = (energy swing per cycle) / (J * omega^2) is held
    # below a design limit so the crank carries the engine through the
    # non-firing part of each cycle. At idle the swing is at least the
    # friction work per cycle. A declared inertia below that floor is
    # physically inconsistent with the engine's own friction (found on
    # the 25 cc trimmer: 6e-5 kg*m^2 lost ~80% of its kinetic energy per
    # revolution and could not idle unloaded, where a real trimmer's
    # flywheel magneto + clutch drum sit around 2-5e-4). Applied only as a
    # FLOOR; every engine already above it keeps its declared value.
    FLYWHEEL_SPEED_FLUCTUATION_MAX = 0.10   # real upper design limit for small engines / generators
    for e in engines:
        if e.kind != "combustion" or e.architecture.cylinders <= 0:
            continue
        omega_idle = e.idle_rpm * RPM_TO_RAD_S
        cycle_rad = 2.0 * math.pi * e.architecture.cycle_degrees / 360.0
        friction_work_per_cycle_j = e.friction_torque_nm(e.idle_rpm) * cycle_rad
        j_floor = friction_work_per_cycle_j / (FLYWHEEL_SPEED_FLUCTUATION_MAX * omega_idle * omega_idle)
        if e.inertia_kg_m2 < j_floor:
            e.flywheel_floor_applied_from_kg_m2 = e.inertia_kg_m2
            e.inertia_kg_m2 = j_floor
    return engines


CATALOGUE: list[Engine] = _build_catalogue()
BY_IDENTITY: dict[str, Engine] = {e.identity: e for e in CATALOGUE}


def spring_headroom_check() -> list[str]:
    """Early-baking sanity check: which catalogue engines are actually
    being asked to rev past what their chosen lifter spring can hold?"""
    warnings = []
    for e in CATALOGUE:
        if not e.architecture.cylinders:
            continue
        cap = e.lifter_spring.max_safe_rpm()
        if e.redline_rpm > cap:
            warnings.append(f"{e.identity}: redline {e.redline_rpm:.0f} rpm exceeds spring-safe {cap:.0f} rpm")
    return warnings


def get(identity: str) -> Engine:
    return BY_IDENTITY[identity]
