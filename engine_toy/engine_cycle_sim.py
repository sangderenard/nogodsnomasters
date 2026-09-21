"""Crank-domain physical engine simulator.

Where engine_sim.EngineState is a single lumped-rpm integrator driven
directly by a throttle-to-torque curve, this is the real owner of the
engine's whole delivery contract: throttle -> manifold air pressure
(with fill/empty lag), rpm + load -> ignition timing advance,
electrical load vs. alternator output -> battery voltage -> spark
energy -> misfire probability, and (timing, air charge, fuel octane)
-> knock/ping risk. rpm itself falls out of integrating individual
per-cylinder combustion torque pulses positioned in crank-angle by the
engine's real firing order and the *current* ignition timing -- so
idle lope, a stumble under a bad ignition-timing/fuel combination, and
an actual knock event are things that emerge from the simulation
rather than hand-tuned audio wobble.

Every rule below (MAP target curve, spark-advance curve, knock-rate
formula) is a plain function attribute on EngineCycleSim -- a
"procedural kernel policy" in the same sense as mech_parts.Part -- so
it can be swapped or tuned per engine without touching the integrator.

Knock is modeled the way it actually happens: a race between the flame
front consuming the charge and the unburned end-gas ahead of it
autoigniting on its own. Each cylinder accumulates a knock integral
(Livengood-Wu-shaped, not literally calibrated to real chemistry) for
the duration of its burn window; if the integral reaches 1.0 before the
window naturally completes, the end-gas won. Because the burn window is
a fixed span of crank-degrees, it's shorter in real time at high rpm --
so less time is available for the integral to accumulate, and knock
becomes structurally rarer at high rpm without any hand-written rpm
term, exactly the "knock is a low-rpm/high-load problem" behavior real
engines show.

The rev limiter is a real fuel cut, not a clamp on the rpm number: once
rpm crosses the limiter point, spark is cut (with a hysteresis band
before it resumes), and rpm falls under genuine pumping/friction load
until it drops back below the limiter -- so it bounces/chatters at the
ceiling instead of going dead flat.

The external brake_load_nm (a dyno absorber, in spirit) doesn't act
directly on the crank as a bare subtracted scalar anymore -- it acts on
a small virtual load shaft, coupled to the crank through a
drivetrain_port.ClutchPort, matching the real game's clutch formula
exactly (abstract_ui_vehicles.py:2332-2337): a saturating tanh reaction
driven purely by relative angular speed. No backlash here -- that's a
gear/belt-lash phenomenon (see drivetrain_port.GearedCoupling, used by
the real accessory-drive-belt/geared-timing-drive edges the drivetrain
graph solver walks, drivetrain_graph.py), not something a friction
clutch has.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import math
import random as _random
import hashlib as _hashlib

import engines
from engines import (
    Engine, RPM_TO_RAD_S, derive_jet_metering_frac, fuel_stoich_afr,
    fuel_latent_heat_j_per_kg, FUEL_SPECIFIC_HEAT_J_PER_KGK, ENGINE_BAY_AMBIENT_K,
    WATER_METHANOL_LATENT_HEAT_J_PER_KG,
    convective_heat_loss_w,
)
from engine_sim import torque_fraction, electric_torque_fraction
from drivetrain_port import ClutchPort
from drivetrain_graph import DrivetrainSolver, build_drivetrain_graph
# the ECU's own program -- idle governors and their constants live in
# the ECU box now (ecu.py), not inline here; the sim owns one instance
# and delegates to it
from aftermarket_idle_controller import StandaloneIdleController
from ecu import (EngineControlUnit, MAP_IDLE_FRAC_DEFAULT, DIESEL_UNTHROTTLED_MAP_FRAC,
                 idle_map_base_frac, idle_fuel_base_frac, is_computerized)
from ignition_driver import IgnitionDriver
from dyno_controller import DynoController
from electrical_network import ElectricalNetwork, ECU_QUIESCENT_W, AC_CLUTCH_COIL_W, EFI_RAIL_PRESSURE_PA

ELECTRIC_COMPRESSOR_MOTOR_EFFICIENCY = 0.80   # real small industrial DC motor
from drivetrain_graph import (rotor_aero_load_torque_nm, COOLING_FAN_DISK_AREA_M2,
                              PNEUMATIC_RESERVE_PRESSURE_PA)
from starter import StartingSystems
import cylinder_ports


def _identity_seed(identity: str) -> int:
    """Same per-identity seeding valve_state/crankcase_state use."""
    return int(_hashlib.sha1(identity.encode()).hexdigest()[:8], 16)
from gas_turbine import SingleShaftGasTurbine
from otto_langen import (OttoLangenCylinder, P_ATM_PA as OTTO_P_ATM_PA, T_ATM_K as OTTO_T_ATM_K,
                          GAMMA_COMBUSTION_GAS, IGNITION_PRESSURE_RATIO,
                          fuel_gas_properties, fuel_mass_to_volume_fraction, charge_is_flammable)
from governor import CaptiveBallGovernor
from expander import ExpanderCylinderBank
from valve_state import ValveState
from crankcase_state import CrankcaseState
from ballistics import FluidLayer, inferred_fluid
from damage_state import (
    Puncture,
    PartDamageState,
    mesh_part_identity,
    record_penetration,
    register_graph_parts,
    sync_part_pressures,
)
import numpy as _np
from gas_works import _steam_properties
from fuel_network import (FuelNetworkRuntime, SupplyTick, charge_energy_factor, intake_flashback_risk)
from hole_emitters import HoleEmitterField
from burst import BurstField
from ordnance import OrdnanceField
import engine_geometry
import wear as wear_module
import wear_debris
import combustion_kernel

# How much of the sump passes the filter in a second, how good that
# element is, and how fast what is in the oil drops out of it where no
# filter can reach. A full-flow pump turns the charge over several times
# a minute, which is why the coarse flake is gone quickly and the fine
# fraction -- the part an oil sample reads -- is not.
OIL_TURNOVER_PER_S = 0.05
OIL_FILTER_BETA = 200.0          # beta_200: a real full-flow element
OIL_SETTLING_PER_S = 2.0e-6
BURN_WINDOW_DEG = 50.0  # no longer used to SHAPE the torque pulse -- see _burned_frac
# Real, disclosed combustion-kernel constants: a representative laminar
# flame speed for gasoline near stoichiometric/atmospheric conditions,
# and a typical simplified turbulence-intensity-to-mean-piston-speed
# ratio (both real, standard order-of-magnitude values used in reduced-
# order SI combustion models -- not individually tuned per fuel/engine).
LAMINAR_FLAME_SPEED_M_S = 0.4
FLAME_TURBULENCE_GAIN = 0.5
SPARK_KERNEL_SEED_RADIUS_M = 0.0005
MAX_SUBSTEP_DEG = 2.5
MISFIRE_BASE_PROB = 0.012
KNOCK_RATE_BASE = 50.0
REV_LIMITER_BACKFIRE_PROB = 0.35
RMS_TAU_S = 0.15   # smoothing window for torque/power RMS -- long enough to
                    # average out individual combustion pulses, short enough
                    # to still track a real throttle/load change

# Real exhaust gas state now comes straight from drivetrain_graph.py's own
# real exhaust-air fluid circuit (a genuine combustion heat-share + real
# choked-flow backpressure, stepped there, not a toy-side formula here) --
# this ambient baseline is the only piece still needed on this side.
EXHAUST_TEMP_AMBIENT_K = 293.15
RUNNER_BLOCK_CONDUCTION_FRAC = 0.4  # real, disclosed: a runner's surrounding air is conductively
                                     # pulled toward the block's own live temp, not fully to it (no solid contact)
REFERENCE_INTAKE_TEMP_K = 293.15  # the real charge-density correction's baseline
# The fixed-timestep accumulator's own tick size and catch-up bound --
# see EngineCycleSim.step()'s own docstring. 1ms matches this codebase's
# own established test/reference dt (drivetrain_graph.py's
# DrivetrainSolver._REFERENCE_DT_S); 50 steps bounds one real call's
# worst-case catch-up burst to 50ms of simulated time, the same
# generous-but-bounded margin the old single MAX_PHYSICS_DT_S cap used.
FIXED_PHYSICS_DT_S = 0.001
MAX_CATCHUP_STEPS = 50
SPEED_OF_SOUND_REF_MPS = 480.0   # at EXHAUST_TEMP_AMBIENT_K-ish hot-idle baseline


@dataclass
class IgnitionEvent:
    cylinder: int
    position_key: int         # cylinder number, for looking up 3D position elsewhere
    strength: float            # 0..~1, torque-pulse magnitude actually delivered
    knock: bool
    knock_intensity: float
    misfire: bool
    preignition: bool = False  # surface ignition off a hot spot BEFORE the spark -- see the firing loop


# Pre-ignition (surface ignition): a glowing deposit, an overheated
# exhaust valve or plug electrode lights the charge before the spark
# ever fires. Unlike knock (end-gas autoignition late in the burn) it
# starts the burn EARLY, against the rising piston -- a much larger
# pressure rise, a heavy chamber ring, lost work, and a big heat pulse
# into that piston, which is how it runs away into holed pistons. The
# per-cylinder hot-spot risk is valve_state.py's own real deposit model
# (cylinder_hotspot_risk); charge temperature and load set how easily a
# hot surface lights the mixture.
PREIGNITION_BASE_PROB = 0.35          # per spark event at hotspot_risk 1.0, full load, hot charge
PREIGNITION_STRENGTH_FRAC = 0.6       # work lost to lighting against compression
PREIGNITION_PISTON_HEAT_K = 4.0       # per event, into that cylinder's own bay temperature
ATM_PRESSURE_PA = 101_325.0
# Ring detection. An explicit integrator pushed past its own stability
# limit does not blow up gently -- it ALTERNATES: the coupling torque
# flips sign on consecutive steps while its magnitude stays pinned at
# saturation. Real physics does not do that (a damper's torque decays
# smoothly toward equilibrium), so a run of sign flips at saturation is
# a reliable signature of the numerics rather than the machine, and it
# is worth detecting rather than quietly reporting the phantom slip and
# phantom heat that come with it.
RING_WINDOW = 8                 # consecutive samples to judge on
RING_ALTERNATION_FRAC = 0.75    # this many sign flips in the window = ringing
RING_SIGNIFICANT_FRAC = 0.2     # ...at a magnitude that is real, not noise
RING_MAX_ESCALATION = 16        # how far the reactive substepping may go
RING_ESCALATION_HOLD = 400      # substeps to hold it before relaxing again
# The anti-stall booster's own real trip calibration. The trip point is
# BELOW the governed idle (this catches an engine being dragged through
# its idle, and must not be true at a healthy idle); the sag detector
# watches a band just above it and fires on rate; release needs a real
# recovery above idle so it cannot chatter.
IDLE_ASSIST_TRIP_FRAC_OF_IDLE = 0.88
IDLE_ASSIST_SAG_WATCH_FRAC = 1.05
IDLE_ASSIST_SAG_RPM_PER_S = 180.0      # a load that is genuinely pulling the engine down
IDLE_ASSIST_RELEASE_FRAC_OF_IDLE = 1.02
# The oil pump's pickup sits just off the floor of the pan. Below this
# fraction of the pan's own charge the pickup starts breaking surface --
# on a moving vehicle it does so long before the pan is dry, which is
# why an oil-pressure light comes on in a corner and not at zero. What
# it draws instead is air, and the existing cavitation rule
# (node_effects.OilPumpBehaviour) is what turns that into lost delivery.
OIL_PICKUP_UNCOVERS_FRAC = 0.35
OIL_PICKUP_DRY_FRAC = 0.05
FUEL_BURST_CAP_KG_DEFAULT = 0.05      # burst.py: the fuel vapour that takes part in a deflagration
# fire.py products into the bay air (air_volumes): a pool fire's plume
# and its real, sooty carbon-monoxide yield
AIR_PER_FUEL_MASS = 14.7              # stoichiometric air/fuel by mass
POOL_FIRE_PLUME_K = 1200.0
POOL_FIRE_CO_YIELD_KG_PER_KG = 0.05   # a ventilation-limited hydrocarbon pool fire is a serious CO source
# the shrapnel cascade and vessel rupture (conservative thresholds)
CASCADE_MIN_J = 25.0                  # a fragment below this dents paint at most
CASCADE_MAX_RAYS = 12                 # the most energetic fragments of one burst that get a ray
VESSEL_RUPTURE_MIN_STORED_J = 5_000.0  # a vessel holding less than this just vents through its hole
VESSEL_RUPTURE_MIN_J = 300.0           # ... and only if the hit spent at least this in its wall
EFFECTS_EVERY_TICKS = 4               # node_effects.assess cadence


# Real production throttle-cable-to-plate-linkage travel curve
# (abstract_ui_vehicles.py's own travel_table for actuator.throttle.lever
# -- this toy's standalone _vehicle_powertrain_graph subunit doesn't
# include that cable/lever assembly, so the real numbers are reused
# directly here rather than re-guessed). Pedal position doesn't move
# the butterfly plate linearly -- this real curve does: slow initial
# response, then a steep middle section, easing off near wide open.
THROTTLE_LINKAGE_TRAVEL_TABLE = ((0.0, 0.0), (0.18, 0.006), (0.50, 0.019), (0.78, 0.030), (1.0, 0.038))
BUTTERFLY_MAX_ANGLE_DEG = 78.0  # real, typical throttle-plate max opening before the shaft/stop


def _interp_table(x: float, table: tuple[tuple[float, float], ...]) -> float:
    if x <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            span = x1 - x0
            return y0 if span <= 1e-12 else y0 + (y1 - y0) * (x - x0) / span
    return table[-1][1]


def throttle_plate_open_frac(throttle: float) -> float:
    """Real butterfly-valve flow area as a fraction of the bore's full
    open area, given pedal position -- two real, distinct nonlinear
    stages compose here, not one linear pedal-to-MAP slope:
      1. cable/linkage travel vs pedal position (THROTTLE_LINKAGE_
         TRAVEL_TABLE above), interpolated.
      2. a circular butterfly plate's actual projected open area as it
         rotates across a circular bore: area_frac = 1 - cos(angle) --
         the real geometric relation for a disc pivoting in a round
         duct, not linear in angle. A plate barely cracked open lets
         very little through; the last several degrees before wide-
         open add comparatively little extra area (the real reason
         idle/tip-in is so sensitive and WOT feels comparatively flat)."""
    t = max(0.0, min(1.0, throttle))
    travel_m = _interp_table(t, THROTTLE_LINKAGE_TRAVEL_TABLE)
    linkage_frac = travel_m / THROTTLE_LINKAGE_TRAVEL_TABLE[-1][1]
    angle_rad = math.radians(linkage_frac * BUTTERFLY_MAX_ANGLE_DEG)
    max_angle_rad = math.radians(BUTTERFLY_MAX_ANGLE_DEG)
    return (1.0 - math.cos(angle_rad)) / max(1.0 - math.cos(max_angle_rad), 1e-6)


def throttle_plate_angle_deg(throttle: float) -> float:
    t = max(0.0, min(1.0, throttle))
    travel_m = _interp_table(t, THROTTLE_LINKAGE_TRAVEL_TABLE)
    return (travel_m / THROTTLE_LINKAGE_TRAVEL_TABLE[-1][1]) * BUTTERFLY_MAX_ANGLE_DEG


def default_map_target_policy(throttle: float, rpm: float, idle_rpm: float, engine: Engine,
                              intake_demand_kg_s: float = 0.0) -> float:
    if engine.compression_ignition:
        # diesels are unthrottled -- there's no throttle plate restricting
        # intake air at all, power is controlled by fuel quantity, not air.
        # MAP stays near atmospheric regardless of pedal position (boost,
        # if any, still adds on top via the turbo/supercharger path).
        return DIESEL_UNTHROTTLED_MAP_FRAC
    idle_floor = MAP_IDLE_FRAC_DEFAULT
    if throttle < 0.08:
        # idle-air-control governor: driver's foot is off the pedal, so
        # airflow is closed-loop regulated to hold idle rpm instead of
        # following throttle position directly. Authority runs to the
        # real physical ceiling (fully open idle-air path), not an
        # arbitrary idle_floor+0.15 cap -- that cap was implicitly
        # tuned against the old flat, generous torque_fraction curve;
        # with the real, more honest (and often lower) low-rpm
        # combustion-quality curve now in engine_sim.py, a governor
        # capped that tight can't command enough air to recover a
        # sagging engine and lets it stall out instead, exactly the
        # kind of artificial ceiling silently overriding a real closed-
        # loop controller this session has been finding and removing.
        error = (idle_rpm - rpm) / max(idle_rpm, 1.0)
        return max(0.02, min(1.0, idle_floor + 1.1 * error))
    if engine.throttle_body is not None:
        # a real discrete multi-barrel assembly (throttle_body.py):
        # its own real per-barrel geometry and linkages already ARE
        # the real open-area-vs-pedal curve, primaries and any
        # progressive secondaries included -- nothing more to bolt on
        open_frac = engine.throttle_body.open_area_frac(throttle)
        base = idle_floor + open_frac * (1.0 - idle_floor)
    else:
        base = idle_floor + throttle_plate_open_frac(throttle) * (1.0 - idle_floor)
        carb = engine.carburetor
        if carb.secondary_threshold < 1.0 and throttle > carb.secondary_threshold:
            # secondaries opening: a real nonlinear kick in airflow past
            # the threshold, not just more of the same linear response
            # (the abstract stand-in for an engine with no discrete
            # throttle_body of its own declared)
            past = (throttle - carb.secondary_threshold) / (1.0 - carb.secondary_threshold)
            base += past * carb.secondary_surge_gain
    # filter restriction: a real orifice pressure-drop, which scales with
    # the SQUARE of actual mass flow through it -- driven by the real
    # flow rate this engine is actually pulling (intake_demand_kg_s,
    # displacement*rpm*density), not an rpm-fraction proxy standing in
    # for flow. An unfiltered open manifold (a velocity stack, no filter
    # at all -- real for some of this catalogue's race engines) would
    # genuinely see raw atmospheric with none of this; everything else
    # has a real filter/throttle restriction between it and atmosphere.
    reference_flow_kg_s = (engine.displacement_l / 1000.0) * (engine.redline_rpm / 120.0) * 1.2
    flow_frac = min(1.2, intake_demand_kg_s / max(reference_flow_kg_s, 1e-6))
    restriction_loss = engine.intake_system.restriction_frac * (flow_frac ** 2) * throttle
    base -= restriction_loss
    return max(0.05, min(1.0, base))


def default_ignition_timing_policy(rpm_frac: float, map_frac: float) -> float:
    # a fixed, generic curve -- the fallback for anything with no real
    # piston/bore to derive an MBT curve from at all (rotary, electric);
    # see mbt_ignition_timing_policy for the real, per-engine derivation
    # every combustion engine with a real bore actually uses now
    return max(5.0, min(38.0, 8.0 + 24.0 * rpm_frac - 22.0 * map_frac))


MBT_50PCT_BURN_TARGET_ATDC_DEG = 12.0  # real, typical: best-torque peak cylinder pressure lands ~12deg ATDC
VACUUM_ADVANCE_GAIN_DEG = 10.0  # real, disclosed stand-in for load-dependent dilution/turbulence this flame model doesn't carry
IGNITION_TIMING_MIN_DEG = 5.0
IGNITION_TIMING_MAX_DEG = 45.0
# Separate, realistic-magnitude flame-speed constants for TIMING only --
# NOT the same LAMINAR_FLAME_SPEED_M_S/FLAME_TURBULENCE_GAIN the torque
# kernel above uses. Those were tuned purely for PULSE SHAPE: the
# torque calibration (pulse_calib) self-normalizes to the right average
# torque regardless of how many crank-degrees the burn actually takes,
# so their absolute duration was never validated against a real crank-
# angle budget. Ignition timing genuinely needs that real absolute
# duration (real 0-50%-mass-burned windows run ~15-25 real crank
# degrees for a typical SI engine), so it gets its own, realistically-
# calibrated turbulence gain instead of silently inheriting a duration
# that was only ever correct in the shape-normalized sense.
IGNITION_LAMINAR_FLAME_SPEED_M_S = 0.4
IGNITION_FLAME_TURBULENCE_GAIN = 4.0


def mbt_ignition_timing_policy(engine: Engine, rpm: float, map_frac: float) -> float:
    """Real MBT (minimum-advance-for-best-torque) base timing, derived
    from THIS engine's own flame-kernel burn physics -- the same
    LAMINAR_FLAME_SPEED_M_S/FLAME_TURBULENCE_GAIN/bore_m the combustion
    substep loop actually integrates torque from -- instead of one
    fixed rpm/map-fraction curve shared by every engine regardless of
    bore or state of tune.

    Real principle: best torque happens when roughly half the charge
    mass has burned a fixed real crank angle after TDC
    (MBT_50PCT_BURN_TARGET_ATDC_DEG) -- ignition has to start early
    enough that the flame reaches 50% burned mass (the same real
    sphere-volume scaling the combustion loop uses: radius =
    bore_half*0.5**(1/3), since mass fraction ~ radius^3) exactly there.
    Solving for how many crank degrees of head start that takes, using
    the real turbulent flame speed AT THIS RPM, is what naturally
    increases advance with rpm -- a real derived mechanism, not a
    hand-picked centrifugal-advance slope. A bigger bore or a slower-
    burning engine genuinely needs more advance; this now actually
    reflects that per engine instead of guessing the same number for
    all of them.

    Low MAP genuinely burns slower in reality (more diluted, less
    in-cylinder turbulence at low charge density than this simplified
    piston-speed-only flame model captures) -- VACUUM_ADVANCE_GAIN_DEG
    is the one remaining disclosed correction standing in for that,
    not re-derived from first principles here.

    Falls back to the old generic curve for anything with no real bore
    to derive flame-kernel geometry from at all (rotary, electric)."""
    bore_half_m = engine.architecture.bore_m / 2.0
    if bore_half_m <= 0.0 or rpm <= 0.0:
        rpm_frac = rpm / max(engine.redline_rpm, 1.0)
        return default_ignition_timing_policy(rpm_frac, map_frac)
    turbulent_flame_speed = (IGNITION_LAMINAR_FLAME_SPEED_M_S
                              + IGNITION_FLAME_TURBULENCE_GAIN * engine.mean_piston_speed_m_s(rpm))
    radius_at_50pct_m = bore_half_m * (0.5 ** (1.0 / 3.0))
    time_to_50pct_s = radius_at_50pct_m / max(turbulent_flame_speed, 1e-6)
    deg_to_50pct = time_to_50pct_s * rpm * 6.0   # 6 real crank-degrees per second per rpm
    advance = deg_to_50pct - MBT_50PCT_BURN_TARGET_ATDC_DEG + VACUUM_ADVANCE_GAIN_DEG * (1.0 - map_frac)
    return max(IGNITION_TIMING_MIN_DEG, min(IGNITION_TIMING_MAX_DEG, advance))


KNOCK_TEMP_REFERENCE_K = 293.15  # baseline charge temp the rest of this formula was tuned against
KNOCK_TEMP_SCALE_K = 25.0        # real, disclosed e-folding sensitivity -- end-gas autoignition kinetics
                                  # are genuinely roughly exponential in charge temperature (Arrhenius-
                                  # shaped), not linear; this is the approximation's one free constant


IN_CYLINDER_WALL_FILM_W_M2K = 100.0     # bulk end-gas-to-wall coefficient during compression (disclosed; real cranking data: ~250 K lost at 200 rpm, ~nothing at idle)
AIR_CV_J_PER_KGK = 718.0
DEFAULT_COMPRESSION_RATIO = 9.0         # when an engine declares none
_SLOW_COMPRESSION_CACHE: dict[str, tuple[float, float, float]] = {}


def slow_compression_knock_factor(engine: Engine, rpm: float) -> float:
    """The real reason a cranking engine doesn't knock: the end gas is
    compressed slowly and gives its heat to the liner before the spark
    (a first-order wall film: tau = m_gas*c_v / (h*A) on the real charge
    mass and wetted area of THIS cylinder), so its temperature is far
    below the adiabatic T0*CR^(gamma-1) the fast-compression case keeps.
    The knock rate is Arrhenius-like in end-gas temperature
    (KNOCK_TEMP_SCALE_K), so the factor is exp(-dT/scale) where dT is
    the EXTRA cooling relative to this engine's IDLE speed -- unity at
    idle and above (the calibrated knock behaviour of a running engine
    is untouched; real engines knock most at 1500-3000 rpm and this must
    not dilute that), collapsing toward zero as a cranking compression
    stretches to seconds."""
    arch = engine.architecture
    if arch.cylinders <= 0 or rpm <= 0.0:
        return 1.0
    cached = _SLOW_COMPRESSION_CACHE.get(engine.identity)
    if cached is None:
        vd_cyl = engine.displacement_l / 1000.0 / arch.cylinders
        if arch.bore_m > 0.0 and arch.stroke_m > 0.0:
            wetted = math.pi * arch.bore_m * arch.stroke_m + math.pi * arch.bore_m ** 2 / 4.0
        else:
            wetted = 4.0 * vd_cyl ** (2.0 / 3.0)
        tau_wall = 1.2 * vd_cyl * AIR_CV_J_PER_KGK / (IN_CYLINDER_WALL_FILM_W_M2K * max(wetted, 1e-6))
        cr = float(getattr(engine, "compression_ratio", 0.0) or DEFAULT_COMPRESSION_RATIO)
        t_rise = KNOCK_TEMP_REFERENCE_K * (cr ** 0.4 - 1.0)   # adiabatic end-gas rise above the charge temperature
        cached = (tau_wall, t_rise, max(engine.idle_rpm, 1.0))
        _SLOW_COMPRESSION_CACHE[engine.identity] = cached
    tau_wall, t_rise, rpm_ref = cached

    def cooling(r: float) -> float:
        t_comp = 0.5 / (r / 60.0)   # half a revolution of compression, in seconds
        return t_rise * (1.0 - math.exp(-t_comp / tau_wall))

    extra = cooling(rpm) - cooling(rpm_ref)
    return math.exp(-max(0.0, extra) / KNOCK_TEMP_SCALE_K)


def default_knock_rate_policy(map_frac: float, timing_deg: float, octane_factor: float,
                               charge_temp_k: float = KNOCK_TEMP_REFERENCE_K) -> float:
    """Rate (1/s) at which the end-gas races toward autoignition; knock
    happens when this integrated over the burn window reaches 1.0.
    Rises sharply with charge pressure (MAP, the dominant term -- this
    is what makes low-rpm lugging under load dangerous even though
    ignition timing is retarded there, not advanced), with spark
    advance (additive, not a hard gate -- retarded timing lowers risk
    but never crushes it to zero on its own), and with charge
    TEMPERATURE (the real, direct link this toy's whole staged-intake-
    temperature chain -- runner heat-soak, plenum compression heating,
    fuel evaporative cooling, a fitted intercooler/charge cooler --
    was missing until now: a hotter charge is measurably closer to its
    own autoignition point regardless of what pressure/timing say,
    which is the actual physical reason charge cooling fights knock,
    not just a knock-margin fudge applied elsewhere). Falls sharply
    with the fuel's octane/knock resistance."""
    pressure_term = max(0.0, map_frac) ** 2.5
    timing_term = 0.5 + max(0.0, timing_deg) / 40.0
    temp_term = math.exp((charge_temp_k - KNOCK_TEMP_REFERENCE_K) / KNOCK_TEMP_SCALE_K)
    resistance = max(0.15, octane_factor) ** 2.2
    return KNOCK_RATE_BASE * pressure_term * timing_term * temp_term / resistance


@dataclass
class BackfireEvent:
    kind: str          # "planned" (anti-lag) | "unplanned" (misfire igniting downstream)
    strength: float


@dataclass
class EngineCycleState:
    rpm: float = 0.0
    crank_angle_deg: float = 0.0
    manifold_pressure_frac: float = MAP_IDLE_FRAC_DEFAULT
    ignition_timing_deg: float = 10.0
    battery_voltage: float = 12.6       # the bus voltage (electrical_network.py)
    alternator_output_frac: float = 0.0  # alternator current / its rated current
    spark_energy_frac: float = 1.0
    battery_soc_frac: float = 1.0
    alternator_current_a: float = 0.0
    electrical_load_w: float = 0.0
    bus_regulating: bool = False        # alternator holding the regulator setpoint
    cranking: bool = False              # the starting system is turning the crank
    starter_note: str = ""
    starter_current_a: float = 0.0
    knock_flag: bool = False
    knock_intensity: float = 0.0
    misfire_flag: bool = False
    ignition_cut: bool = False
    valve_float_flag: bool = False
    valve_float_risk: float = 0.0
    # per-cylinder valvetrain state (valve_state.py): each cylinder's
    # own breathing factor, float speed and combined valve factor from
    # its springs' and rockers' machine error and present condition
    cylinder_breathing_frac: list = field(default_factory=list)
    cylinder_float_rpm: list = field(default_factory=list)
    cylinder_valve_factor: list = field(default_factory=list)
    cylinder_carbon_frac: list = field(default_factory=list)
    cylinder_hotspot_risk: list = field(default_factory=list)
    # crankcase participation (crankcase_state.py)
    cylinder_oil_film_mg: list = field(default_factory=list)
    oil_consumption_ml_per_h: float = 0.0
    blowby_l_per_min: float = 0.0
    crankcase_pressure_kpa: float = 0.0
    # what is in the oil, as a used-oil report states it
    oil_soot_frac: float = 0.0
    oil_debris_captured_g: float = 0.0
    sump_oil_l: float = 0.0
    oil_fuel_dilution_frac: float = 0.0
    current_torque_nm: float = 0.0
    power_kw: float = 0.0
    torque_rms_nm: float = 0.0
    power_rms_kw: float = 0.0
    stalled: bool = False
    turbo_spool_frac: float = 0.0
    boost_frac: float = 0.0
    wastegate_flutter: bool = False
    surge_flag: bool = False
    backfire_flag: bool = False
    backfire_kind: str | None = None
    rev_limiter_active: bool = False
    brake_clutch_locked: bool = False
    supercharger_drag_nm: float = 0.0
    # The REAL, measured instantaneous firing/ignition rate -- a
    # general, every-cylinder-kind quantity (see EngineCycleSim.
    # _record_ignition, called by both the piston combustion loop and
    # _step_atmospheric whenever a real ignition happens), not bespoke
    # to any one engine kind. 1/(time since the last real ignition),
    # decaying honestly toward 0 if nothing has fired in a while rather
    # than holding a stale rate. For a crank engine this is normally
    # very close to rpm*firing_events_per_rev, but it's the actually-
    # measured rate (so it naturally reflects misfires, rev-limiter
    # cuts, and governor misses); for a free-piston engine (kind==
    # "atmospheric") whose cycle time is set by its own gas-law physics
    # rather than the flywheel's current speed, that rpm-based formula
    # doesn't apply at all, and this is the only real firing-rate
    # signal there is.
    real_fire_hz: float = 0.0
    accessory_drag_nm: float = 0.0
    ac_compressor_load_w: float = 0.0
    cam_phase_lag_deg: float = 0.0
    exhaust_temp_k: float = 293.15          # manifold/primary-side gas temp, before any pipe travel
    exhaust_tailpipe_temp_k: float = 293.15  # after the real per-segment convective cooling chain
    exhaust_pressure_frac: float = 0.0
    coolant_temp_k: float = 293.15
    coolant_flow_lpm: float = 0.0
    oil_flow_lpm: float = 0.0
    nitrous_boost_frac: float = 0.0
    nitrous_fill_frac: float = 0.0
    pneumatic_idle_assist_boost_frac: float = 0.0
    pneumatic_idle_assist_delivered_kg_s: float = 0.0
    # emissions.py / air_volumes.py: what leaves the pipe, where it goes,
    # and what the engine is breathing back
    mixture_phi: float = 1.0
    co_tailpipe_g_s: float = 0.0
    hc_tailpipe_g_s: float = 0.0
    nox_tailpipe_g_s: float = 0.0
    co_engine_out_g_s: float = 0.0
    preignition_flag: bool = False
    preignition_intensity: float = 0.0
    preignition_count: int = 0
    # parts that have burst (burst.py): gone from the mesh, their fluid
    # gone from their circuit, an open end left on it
    absent_parts: set = field(default_factory=set)
    # node_effects.py verdicts published for the dashboard/audio
    fire_heat_release_w: float = 0.0
    fires_burning: int = 0
    coolant_lost_l: float = 0.0
    # couplings.py: what the rig is actually driving the load through
    coupling_kind: str = "dry-friction"
    coupling_locked: bool = False
    coupling_slipping: bool = False
    coupling_capacity_nm: float = 0.0
    coupling_heat_w: float = 0.0
    # the solver's own honesty check on the dyno junction
    junction_ringing: bool = False
    junction_ring_events: int = 0
    junction_substeps: int = 1
    # plant.py readings
    plant_dewpoint_k: float = 273.15
    plant_gunk_kg: float = 0.0
    hydraulic_oil_temp_k: float = 293.15
    oil_pickup_air_frac: float = 0.0      # how much air the pump is drawing off an uncovered pickup
    smoke_factor: float = 0.0
    exhaust_open_frac: float = 0.0
    engine_dead: str | None = None
    spark_lost: bool = False
    brakes_lost: bool = False
    startable: bool = True
    mounts_lost: int = 0
    # per-firing-slot record of what each cylinder ACTUALLY did on its
    # last pass (engine_sound.py schedules its event kernels off these
    # on its own crank clock -- no combustion sound for a slot that did
    # not burn): [strength 0..1 (0 = cut/no fuel), misfire 0/1,
    # knock_intensity 0..1, preignition 0/1], indexed by firing-order
    # slot; slot_angles_deg is each slot's own firing angle in the cycle
    slot_records: list = field(default_factory=list)
    slot_angles_deg: list = field(default_factory=list)
    # a hit-and-miss governor holding the exhaust valve off its seat:
    # the cylinder never compresses, so its exhaust port passes nothing
    exhaust_valve_held_open: bool = False
    # count of real ignitions so far (any kind) -- an event stream for
    # a listener that has no crank slots to clock from (a free piston)
    fire_event_count: int = 0
    compression_brake_active: bool = False
    compression_brake_torque_nm: float = 0.0
    catalyst_brick_temp_k: float = 293.15
    catalyst_co_efficiency: float = 0.0
    catalyst_nox_efficiency: float = 0.0
    co_emitted_indoors_kg: float = 0.0
    bay_air_temp_k: float = 293.15
    bay_co_ppm: float = 0.0
    bay_o2_frac: float = 0.2095
    room_co_ppm: float = 0.0
    room_o2_frac: float = 0.2095
    room_temp_k: float = 293.15
    intake_source: str = "bay"
    intake_o2_factor: float = 1.0
    occupant_cohb_pct: float = 0.0
    fuel_fill_frac: float = 1.0
    # atmospheric (gas-fuelled) engines only -- the real charge the
    # mixer actually admitted this tick and where it came from (see
    # _step_atmospheric / otto_langen.FUEL_GAS_PROPERTIES): fuel-gas
    # VOLUME fraction of the charge, its equivalence ratio against
    # the fuel's own stoichiometric requirement, how much of the
    # demanded gas the supply actually delivered (the shortfall is
    # room air), the gasholder's own contents purity (volume basis),
    # and whether those contents currently sit INSIDE the fuel's
    # flammable band (a real explosion hazard: a holder part-way
    # through purging its air, not a running one full of pure gas)
    charge_gas_volume_fraction: float = 0.0
    charge_equivalence_ratio: float = 0.0
    fuel_availability_frac: float = 1.0
    gasholder_composition_frac: float = 1.0
    gasholder_flammable_hazard: bool = False
    # a declared fuel network's own live readings (fuel_network.py);
    # inert defaults for every network-less engine
    fuel_supply_pressure_pa: float = 101_325.0
    fuel_network_warnings: tuple = ()
    boiler_water_frac: float = 1.0
    boiler_failed: bool = False
    intake_flashback_risk: float = 0.0
    # expander engines (expander.py): the cylinder bank's own live state
    expander_cylinder_temp_k: float = 293.15
    expander_ice_frac: float = 0.0
    expander_drain_cocks_open: bool = True
    expander_chest_pressure_pa: float = 101_325.0
    expander_mep_pa: float = 0.0
    fuel_temp_k: float = 293.15
    injector_duty_frac: float = 0.0     # 0 = no declared injector array on this engine; >1.0 (pre-clamp) means genuinely too small for the demand
    throttle_plate_angle_deg: float = 0.0
    dyno_rpm: float = 0.0             # the real virtual dyno-absorber drum's own speed (load shaft), not the crank's
    dyno_absorbed_kw: float = 0.0     # real power the brake is actually absorbing right now: brake_load_nm * dyno_omega
    dyno_kinetic_energy_j: float = 0.0  # real rotational KE currently stored in the spinning drum, 0.5*I*omega^2
    dyno_torque_nm: float = 0.0       # real instantaneous torque actually reaching the drum this tick (post-gear)
    dyno_pull_state: str | None = None  # None | "settle" | "shift" | "pulling" -- see EngineCycleSim.start_wot_dyno_pull
    dyno_pull_complete_flag: bool = False  # one-tick pulse when a pull finishes -- for event-log consumers
    dyno_pull_peak_torque_nm: float = 0.0
    dyno_pull_peak_torque_rpm: float = 0.0
    dyno_pull_peak_power_kw: float = 0.0
    dyno_pull_peak_power_rpm: float = 0.0
    intake_charge_temp_k: float = 293.15    # final, at the cylinder -- after every real stage below
    intake_runner_temp_k: float = 293.15    # after the real runner heat-soak stage, before the plenum
    oil_pressure_pa: float = 101_325.0
    oil_temp_k: float = 293.15
    intake_flow_demand_frac: float = 0.0  # real demand/flow_capacity_kg_s -- >1 means choked right now
    exhaust_flow_demand_frac: float = 0.0
    # per-cylinder block-metal temperature, index i == cylinder number i+1
    # (arch.firing_order's own numbering) -- see EngineCycleSim.step()'s
    # own real per-cylinder heat balance for how this is actually driven
    cylinder_block_temps_k: list[float] = field(default_factory=list)
    # Real per-component durability ledger -- see wear.py. component
    # name -> damage fraction, 0.0 (new) to 1.0 (worn out); empty for
    # engine kinds with no declared wearing parts (electric). Reset to
    # a fresh 0.0 ledger by EngineCycleSim.set_engine (a new engine is
    # a new, unworn part), never by a plain sim reset.
    wear: dict[str, float] = field(default_factory=dict)
    # Every graph node/edge has a record, while procedural mesh parts
    # are added on first damage. Punctures survive stop/start and are
    # cleared only when set_engine installs a different physical engine.
    part_damage: dict[str, PartDamageState] = field(default_factory=dict)


# Chance an unburned charge dumped into the exhaust by a misfire actually
# ignites downstream (a real backfire) rather than just fizzling out --
# rises sharply with how much boost/heat is in the exhaust system.
BACKFIRE_BASE_PROB = 0.12
BACKFIRE_BOOST_GAIN = 0.55
ANTI_LAG_LIFT_THRESHOLD = -1.2   # d(throttle)/dt more negative than this = a lift-off event
ANTI_LAG_COOLDOWN_S = 0.6
SURGE_BOOST_THRESHOLD = 0.3
SURGE_DECAY_S = 0.35
# dyno brake/throttle rpm-target gains moved to dyno_controller.py
# idle governor gains/gating (IDLE_MAP_*, IDLE_FUEL_*, STALL_*) moved to
# ecu.py -- the ECU's own program, see EngineControlUnit
FUEL_RACK_TAU_S = 0.15  # real injection-pump rack/flyweight response lag, matches MAP_TAU_BASE_S's reference value
# Quick-shift timing: a real driver (or an auto-clutch paddle system)
# doesn't ride the clutch through a shift -- pedal out fast, swap the
# dog/synchro engagement, pedal back in fast. These are real, typical
# durations for a deliberate quick manual shift (a lazy street shift
# takes longer; a race shift is faster still), not a hand-picked curve
# shape -- the sequencer below is a genuine two-stage clutch_frac ramp,
# same real mechanism a driver's own foot performs.
SHIFT_CLUTCH_OUT_S = 0.12
SHIFT_CLUTCH_IN_S = 0.22
DYNO_PULL_SETTLE_S = 1.0        # real dwell in neutral before shifting, letting rpm settle to idle first
DYNO_PULL_MAX_DURATION_S = 20.0  # safety timeout if an engine's own real dynamics never quite reach redline
DYNO_PULL_SHIFT_RPM_FRAC = 0.92  # WOT through each real ratio before the final redline pull
# Real dyno software bins/smooths raw torque samples before plotting a
# curve -- an instantaneous single-substep reading is dominated by
# combustion-pulse ripple (worse for a low cylinder count), the same
# real reason a real dyno graph is never as spiky as the raw sensor
# trace. A short exponential average, not a curve shape -- big enough
# to average over several firing pulses, small enough to still track
# the real underlying rpm-dependent torque/power curve as the pull runs.
DYNO_TORQUE_SMOOTH_TAU_S = 0.15
# a real exhaust brake valve is never perfectly sealed -- drivetrain_
# graph.py's own floor (max(0.05, 1-exhaust_brake_frac)) already covers
# the absolute minimum leak-through; this is the real, disclosed
# fraction of the exhaust port's flow capacity the valve actually
# closes when engaged (a real closed diesel exhaust brake commonly
# still passes on the order of 20% of full flow)
EXHAUST_BRAKE_MAX_RESTRICTION_FRAC = 0.80
# compression-release brake: the retarding mean effective pressure of a
# real Jake at rated speed (~0.8-1.0 MPa -- half or more of a diesel's
# firing BMEP, which is why it absorbs 60-70 % of rated power), scaled
# with speed (more charge trapped, higher compression pressure) and
# interlocked to zero fuel and above idle like the real control
COMPRESSION_RELEASE_MEP_PA = 900_000.0
COMPRESSION_RELEASE_MIN_RPM_FRAC_OF_IDLE = 1.2
# The bench-test drum (drivetrain_graph.py's dyno_absorber node) is
# deliberately light -- 3x the engine's OWN crank inertia, see that
# node's own comment -- so a resistive brake-load test isn't fighting
# the drum's own mass. A real INERTIA-dyno pull needs the opposite: a
# properly-sized flywheel-equivalent mass is the entire reason T=I*alpha
# gives a clean, resolvable torque reading at all (a real Dynojet
# roller is a genuinely heavy drum, not a feather-light bench fixture).
# Temporarily swap to a realistic road-load-equivalent inertia for the
# duration of a pull only, restoring the light bench value afterward.
DYNO_PULL_INERTIA_KG_M2_PER_NM = 0.15


@dataclass
class EngineCycleSim:
    engine: Engine
    # None uses whatever installation fuel system the engine specification
    # explicitly carries.  A station/vehicle supplies its own real reservoir
    # records here so the engine compiler does not invent an onboard tank.
    external_fuel_supply: dict | None = None
    throttle: float = 0.0
    brake_load_nm: float = 0.0
    clutch_frac: float = 1.0   # 0=pedal floored/disengaged, 1=fully released/locked -- ClutchPort.engagement
    # None = seed from the engine's identity (reproducible per engine);
    # set it to run the same engine down a different random path
    seed: int | None = None
    gear_index: int = 0        # 0=neutral (every engine starts in N), negative=reverse, 1..N=forward gear (engine.transmission.gear_ratios)
    electrical_load_frac: float = 0.30
    # A load commanded by another graph controller (PTO pump, service
    # generator, etc.) is known before rpm sags.  This channel informs the
    # ECU's existing idle-load feedforward; the actual reaction torque still
    # enters through the drivetrain, so the load is not applied twice.
    known_accessory_shaft_load_w: float = 0.0
    fuel_choice: str | None = None   # None -> engine.preferred_fuel_profile
    rev_limiter_enabled: bool = True  # off = watch what it's actually there to prevent
    anti_lag_enabled: bool = False    # turbo + forced_induction.anti_lag_capable only
    nitrous_active: bool = False       # engine.has_nitrous only -- opens the real solenoid
    auxiliary_injection_active: bool = True   # engine.has_auxiliary_injection only -- a real WMI kit is normally armed whenever the engine runs, not driver-toggled like nitrous
    garage_mode: str = "outdoors"                 # air_volumes.AirStack: "outdoors" | "open" | "closed" | "sealed"
    exhaust_extractor_fitted: bool = False         # the big orange duct over the tailpipe, vented outside
    cell_fan_m3_s: float = 0.0                     # a dyno-cell ventilation fan on the room
    pneumatic_idle_assist_enabled: bool = True   # engine.pneumatics.idle_assist_fitted only -- a real toggleable anti-stall switch, armed by default once fitted; the ACTUAL dump also needs rpm to actually be under the trip point
    ac_enabled: bool = False           # engine.accessories.air_conditioning only -- engages the real belt compressor
    # a real exhaust brake valve (see EXHAUST_BRAKE_MAX_RESTRICTION_FRAC
    # and drivetrain_graph.step's exhaust_brake_frac) -- off by default,
    # matching a real vehicle: it's a driver-engaged system, not
    # something that's just always on. Now that there's a real gearbox
    # to downshift for engine braking too, "off" genuinely means a
    # freer coast, the way letting off the exhaust brake actually does.
    # Valvetrain/bearing friction drag is real mechanical loss and stays
    # on regardless -- that's not what this toggle ever controlled.
    engine_brake_enabled: bool = False
    compression_brake_enabled: bool = False        # engine.compression_release_brake only -- the Jake switch
    brake_target_rpm: float | None = None  # None = manual brake_load_nm control (E/D).
                                             # Set = a real PI dyno controller drives
                                             # brake_load_nm itself to hold this rpm --
                                             # idle-protect is just this with target=idle_rpm
    throttle_target_rpm: float | None = None  # None = manual throttle control. Set = a real
                                                # PI governor drives throttle itself to hold this
                                                # rpm -- independent of brake_target_rpm: either,
                                                # neither, or both can be active at once (a real
                                                # dyno commonly runs both a throttle servo AND a
                                                # load absorber together)

    map_target_policy: Callable[[float, float, float, Engine], float] = default_map_target_policy
    ignition_timing_policy: Callable[[Engine, float, float], float] = mbt_ignition_timing_policy
    knock_rate_policy: Callable[[float, float, float, float], float] = default_knock_rate_policy

    state: EngineCycleState = field(default_factory=EngineCycleState)
    pending_events: list[IgnitionEvent] = field(default_factory=list)
    pending_backfires: list[BackfireEvent] = field(default_factory=list)
    # Persistent effect subsystems are declared members of the managed
    # object, not dynamically typed attributes.  Besides documenting the
    # actual lifetime boundary, these concrete identities let whole-program
    # lowering pursue each subsystem's authored ``step`` implementation.
    hole_emitters: HoleEmitterField = field(init=False, repr=False)
    bursts: BurstField = field(init=False, repr=False)
    ordnance: OrdnanceField = field(init=False, repr=False)
    # A real on-site fuel-gas generator (gas_works.GasGenerator) or
    # boiler (gas_works.Boiler) feeding this engine's real gasholder
    # tank instead of an unlimited piped main -- None (default) keeps
    # every engine, including every atmospheric one, on today's
    # unmetered behavior. The caller sets this AFTER building the sim
    # (engine_cycle_sim.py doesn't import gas_works.py, same layering
    # reason engines.py doesn't either -- see Engine.atmospheric_
    # supply_tank_capacity_kg's own docstring), and is responsible for
    # declaring a matching nonzero atmospheric_supply_tank_capacity_kg
    # on the Engine itself so drivetrain_graph.py actually builds the
    # real gasholder tank node/circuit this generator fills.
    atmospheric_generator: object | None = None

    def __post_init__(self) -> None:
        # THE SIM'S OWN RANDOM STREAM. The stochastic combustion events
        # (misfire, pre-ignition, backfire, a soft limiter's cut) used
        # the process-global `random`, which made two runs of the same
        # engine from the same state disagree by a couple of percent and
        # made an exact regression -- or a parity check against a
        # compiled core -- impossible to write. Seeded per engine, the
        # same convention valve_state and crankcase_state already use
        # (_seed(identity)), so the randomness is still real and still
        # per-engine, but reproducible. Pass `seed` to vary it
        # deliberately; two sims with the same seed now agree exactly.
        self._rng = _random.Random(
            _identity_seed(self.engine.identity) if self.seed is None else int(self.seed))
        # THE WEAR LEDGER HAS TO EXIST FROM THE START. `state.wear` was
        # only ever created by `set_engine`, which a sim built straight
        # from an engine never calls -- so a freshly constructed sim
        # carried an EMPTY ledger, `step_wear` returned immediately on
        # it every tick, and nothing in the entire durability model ever
        # moved unless the operator happened to switch engines in the
        # UI and back. Everything downstream of wear (oil debris, ring
        # leak, blow-by, oil analysis) was reading a ledger that could
        # not change.
        if not self.state.wear:
            self.state.wear = wear_module.new_wear_state(self.engine)
        self._omega = 0.0
        self._total_crank_deg = 0.0
        self._map_frac_na = idle_map_base_frac(self.engine)  # ECU's per-engine idle calibration
        self._fuel_rack_frac = idle_fuel_base_frac(self.engine)  # ECU's per-engine idle rack calibration
        self._prev_throttle = 0.0
        self._surge_timer = 0.0
        self._antilag_cooldown = 0.0
        self._firing_angle_deg: dict[int, float] = {}
        self._slot_of_cyl: dict[int, int] = {}
        self.hole_emitters = HoleEmitterField()
        # the emitters do not own fluid state: they ask the sim for the
        # real pressure and the real contents, and hand back every gram
        # they take, so the HUD, the pump and the leak are one machine
        self._spray_needs_resolve = True
        self.hole_emitters.pressure_of = self._circuit_pressure_pa
        self.hole_emitters.remaining_of = self._circuit_remaining_l
        self.hole_emitters.on_loss = self._deplete_circuit
        self.bursts = BurstField()
        self.damage_events: list[dict] = []
        self._fuel_exposure_s: dict[str, float] = {}
        # node_effects.py: what the hurt nodes do to the engine, assessed
        # every few ticks from the damage/emitter/absence records
        from node_effects import Effects
        self._effects = Effects()
        self._node_conditions: dict = {}
        self._effects_tick = 0
        self._circuit_base: dict = {}
        # shrapnel cascade (burst.py -> the projectile engine): the app or
        # snapshot supplies a RayMesh factory for the live geometry; with
        # none, fragments fly but hole nothing
        self.ray_mesh_factory = None
        self.cascade_log: list[str] = []
        self._cascade_depth = 0
        # ordnance.py: placed charges and their fuzes
        from fire import FireField
        from fittings import FittingField
        from plant import AuxiliaryPlantRuntime
        # plant.py: the auxiliary skid (air treatment, refrigerant loop,
        # hydraulic tank, controls, accessory bank) -- None on an engine
        # that declares no such plant, which is most of them
        self.plant = AuxiliaryPlantRuntime.build(self.engine)
        # the machine's own hydraulic lever: how much of the pump's flow
        # the actuators are taking, and how hard they are pushing. Set
        # by the operator, the same way brake_load_nm is. Flow 0 with
        # load 1 is a lever held against its stop -- the whole pump
        # output over the relief valve as heat.
        self.hydraulic_flow_frac = 0.0
        self.hydraulic_load_frac = 0.0
        # equipment.py: anything bolted on that runs off the engine's
        # own fluids. None until something is fitted, which is most
        # engines most of the time.
        self.equipment = None
        self.ordnance = OrdnanceField()
        # automatic_transmission.py: a hydraulic machine, built only for
        # an engine that declares one -- a manual box is a gear case
        # (gear_cases.py) and has none of this
        self.automatic = self._build_automatic()
        # fire.py: burning as a lasting state, not an event
        self.fires = FireField()
        # fittings.py: a hole with something screwed into it is a port
        self.fittings = FittingField()
        self._last_fire_total_deg: dict[int, float] = {}
        self._last_strength: dict[int, float] = {}
        self._knock_accum: dict[int, float] = {}
        self._knocked_this_burn: dict[int, bool] = {}
        self._active_event: dict[int, IgnitionEvent] = {}
        self._flame_radius_m: dict[int, float] = {}
        self._burned_frac: dict[int, float] = {}
        # the two real electrical boxes this sim delegates its control
        # programs to -- same identities as the production vehicle graph
        # (electrical.ecu / electrical.ignition_driver)
        self.ecu = EngineControlUnit()
        self.ignition = IgnitionDriver()
        # a real, optional aftermarket idle-air-control box -- inert
        # unless engine.idle_control_device actually selects it (see
        # aftermarket_idle_controller.py); always constructed so the
        # step loop never has to branch on whether it exists, only on
        # which device this engine's build actually uses
        self.standalone_idle = StandaloneIdleController()
        # the test cell's own controller (brake/throttle rpm targets --
        # not vehicle hardware) and the real 12 V bus the boxes hang on
        self.dyno = DynoController()
        self.electrical = ElectricalNetwork(self.engine)
        # the real starting system (starter.py) and the crank-nose
        # attachment any external torque source the game bolts on feeds
        self.starter = StartingSystems(self.engine)
        # the turbine/Otto-Langen engines' own real, stateful runtime
        # objects (gas_turbine.py / otto_langen.py + its real captive-
        # ball governor) -- built directly from the engine's own
        # declared spec, the same real spec/runtime split this
        # catalogue already uses everywhere else (e.g. ForcedInduction,
        # StartingSystems). None for every other engine kind.
        self._turbine: SingleShaftGasTurbine | None = (
            self.engine.turbine.build() if self.engine.turbine is not None else None)
        self._atmospheric_cyl: OttoLangenCylinder | None = (
            self.engine.atmospheric.build() if self.engine.atmospheric is not None else None)
        self._atmo_governor: CaptiveBallGovernor | None = (
            self.engine.atmospheric_governor.build() if self.engine.atmospheric_governor is not None else None)
        self._expander: ExpanderCylinderBank | None = (
            self.engine.expander.build() if self.engine.expander is not None else None)
        self._time_since_last_ignition_s = 0.0
        self.external_crank_torque_nm = 0.0
        self._crank_assist_nm = 0.0
        self._electric_fan_cmd = 0.0
        self._block_thermal_cache: tuple[float, float, float, float] | None = None
        self._load_omega = 0.0
        self._torque_ms = 0.0   # mean-square accumulators for RMS
        self._power_ms = 0.0
        self.ecu.reset()
        self.standalone_idle.reset()
        self._quick_shift_state: str | None = None   # None | "clutch_out" | "clutch_in"
        self._quick_shift_target_gear = 1
        self._quick_shift_timer_s = 0.0
        self._last_dyno_torque_nm = 0.0
        self._dyno_pull_state: str | None = None   # None | "settle" | "shift" | "pulling"
        self._dyno_pull_timer_s = 0.0
        self._dyno_pull_saved: tuple[float, float, int] | None = None  # (throttle, brake_load_nm, gear_index)
        self._dyno_pull_torque_ema = 0.0
        self._dyno_pull_power_ema = 0.0
        self._dyno_pull_saved_drum_inertia: float | None = None
        self._time_since_start_s = 0.0
        self._brake_junction = self._build_brake_junction()
        self._recompute_firing_angles()
        self._drivetrain = self._build_drivetrain()
        # This is owned engine state, including before the first physics
        # tick.  The dt system checkpoints a registered engine before it
        # advances it, so the declared state span must be readable from the
        # freshly constructed state as well as from a warmed engine.
        self._drivetrain_out: dict = {}
        # WHERE the compressed air is, as distinct from how much there
        # is: the circuit keeps the one authoritative stored mass, this
        # says which of the real vessels is holding it (air_vessels.py).
        self.air_vessels = self._build_air_vessels()
        self._waste_heat_kw = 0.0
        self._intake_supply_pressure_pa = 101_325.0
        self._intake_demand_kg_s = 0.0
        self._fuel_demand_kg_s = 0.0
        #: how full the cylinder actually gets, relative to the
        #: displacement demand at this MAP -- the breathing terms
        #: the demand omits. 1.0 until the first tick solves it.
        self._charge_fill_frac = 1.0
        self._fuel_cooling_kw = 0.0
        self._auxiliary_injection_cooling_kw = 0.0
        self._fuel_starvation_frac = 1.0
        # how fast rpm is really moving, for the anti-stall sag detector
        self._rpm_rate_per_s = 0.0
        self._idle_assist_latched = False
        self._ring_hist: list = []
        self._ring_escalation = 1
        self._ring_escalation_ticks = 0
        self._exhaust_demand_kg_s = 0.0
        self._physics_accum_s = 0.0
        # THE DECLARED FLAT SPAN. `compile_contract.py` says the engine is
        # opaque because "its state is a graph of Python objects rather
        # than typed spans", and that going native "needs a real piece of
        # work on the class rather than on this file". This is that span:
        # one contiguous float64 array in the `engine_abi` layout, which
        # `engine_state.pack`/`unpack` fill and read. Declaring it in
        # `program_abi.records` is what stops every method on this class
        # being a wall.
        self.state_span = None
        if self.engine.load_resistor_frac is not None:
            self.electrical_load_frac = self.engine.load_resistor_frac
        self.stop()

    def _build_drivetrain(self) -> DrivetrainSolver:
        # the real internal solver: camshaft+bearings+timing drive,
        # no-belt accessory takeoff+alternator CVT, and (if present)
        # supercharger belt all live as nodes/edges in one graph
        # document (drivetrain_graph.py), walked generically by
        # constraint kind -- not hand-picked per-mechanism classes
        # a declared fuel network (Engine.fuel_network) gets its runtime
        # rebuilt alongside the graph it was emitted into -- same
        # lifecycle (init/start/stop), never a stale generator or boiler
        # hanging off a fresh circuit
        spec = getattr(self.engine, "fuel_network", None)
        self._fuel_network: FuelNetworkRuntime | None = FuelNetworkRuntime.build(spec) if spec is not None else None
        # the valvetrain's own state persists across stop/start (springs
        # do not un-sag on a restart); it is rebuilt only for a new engine
        vs = getattr(self, "_valve_state", None)
        if vs is None or vs.identity != self.engine.identity:
            self._valve_state = ValveState.from_engine(self.engine)
        cs = getattr(self, "_crankcase_state", None)
        if cs is None or cs.identity != self.engine.identity:
            self._crankcase_state = CrankcaseState.from_engine(self.engine)
            # a different engine is a different engine: its oil has none
            # of the last one's history in it
            self._oil_debris = wear_debris.DebrisLedger()
            self._wear_vector_last = None
        self._cyl_valve_factor = _np.ones(self._valve_state.n_cyl)
        # a network-only swap (fuel_network.convert_engine) keeps the
        # catalogue rating and applies the fluid's charge energy live; a
        # full conversion (conversion.apply_conversion) already re-rated
        # bmep on the fluid -- bmep_rated_fuel says which
        self._charge_energy_factor = (
            charge_energy_factor(spec.fluid)
            if spec is not None and spec.working_fluid.combustible
            and getattr(self.engine, "bmep_rated_fuel", None) != spec.fluid else 1.0)
        self._flashback_accum = 0.0
        self._last_net_tick: SupplyTick | None = None
        graph = build_drivetrain_graph(
            self.engine, external_fuel_supply=self.external_fuel_supply)
        dyno_node = next(n for n in graph["nodes"] if n["identity"] == "dyno_absorber")
        self._dyno_mass_kg = dyno_node["mass_kg"]
        self._dyno_inertia_kg_m2 = dyno_node["inertia_kg_m2"]
        self._dyno_drum_radius_m = dyno_node["drum_radius_m"]
        # the idle-assist dump port's own real area (0 -- no flow ever --
        # when the engine doesn't carry one), and the real trip point:
        # the engine's own declared idle_assist_trip_rpm if it gave one,
        # else a real default a bit above idle so the dump actually
        # catches a sag BEFORE the engine reaches idle, not after
        pn = self.engine.pneumatics
        if pn.compressor_fitted and pn.idle_assist_fitted:
            port_radius_m = max(0.0, pn.idle_assist_port_diameter_mm) / 2000.0
            self._pneumatic_idle_assist_area_m2 = math.pi * port_radius_m * port_radius_m
            # The trip point sits BELOW the governed idle, not above it.
            # This is an ANTI-STALL device: it is there to catch an
            # engine that a sudden load is dragging down THROUGH its
            # idle, not to blow air into an engine that is idling
            # perfectly well. A trip above idle means the condition is
            # true at normal idle and the reservoir simply empties for
            # no reason -- which is exactly what it did.
            self._pneumatic_idle_assist_trip_rpm = (
                pn.idle_assist_trip_rpm if pn.idle_assist_trip_rpm is not None
                else self.engine.idle_rpm * IDLE_ASSIST_TRIP_FRAC_OF_IDLE)
        else:
            self._pneumatic_idle_assist_area_m2 = 0.0
            self._pneumatic_idle_assist_trip_rpm = 0.0
        drivetrain = DrivetrainSolver(graph)
        # the splash emitters: the crank's dipper on every declared
        # oil-splash-path (dressing.py) -- rebuilt with the graph
        self.hole_emitters.emitters = [e for e in self.hole_emitters.emitters if e.kind != "splash"]
        self.hole_emitters.add_splash_from_graph(graph)
        # any port whose cap/plug is not fitted is a real hole from here on
        self.hole_emitters.sync_open_ports(graph)
        register_graph_parts(self.state.part_damage, graph)
        sync_part_pressures(self.state.part_damage, drivetrain.fluid_circuits)
        return drivetrain

    def _sync_part_damage_pressures(self) -> None:
        """Re-point every tracked part at its fluid circuit.

        Topology, not state: this is only needed when the graph changes
        or a part appears that was not in it. It used to run every tick
        to keep a copied pressure fresh; parts now read their pressure
        through the binding (damage_state.PartDamageState.pressure_pa),
        so there is nothing to refresh per tick and nothing that can go
        stale between ticks."""
        sync_part_pressures(self.state.part_damage, self._drivetrain.fluid_circuits)

    def apply_penetration(self, penetration) -> list[tuple[str, Puncture]]:
        """Persist the holes from one ray penetration in the live engine,
        put an emitter (hole_emitters.HoleEmitter) on every boundary hole
        so what the part contains actually passes through it from now on,
        and queue each impact for the damage sound synth."""
        self._sync_part_damage_pressures()
        recorded = record_penetration(
            self.state.part_damage,
            self._drivetrain.graph,
            self._drivetrain.fluid_circuits,
            penetration,
        )
        self.hole_emitters.add_from_punctures(recorded, self._drivetrain.graph, self._drivetrain.fluid_circuits)
        # the geometry just changed: whatever a jet used to land on may
        # now have a hole through it, so every spray is asked again
        for _em in self.hole_emitters.emitters:
            _em.spray_resolved = False
        by_part = {ident: p for ident, p in recorded}
        # a round through a placed charge: its filling's own impact
        # sensitivity decides whether that starts a firing sequence
        if self.ordnance.charges and recorded:
            graph_ = self._drivetrain.graph
            for c in self.ordnance.charges:
                if c.fired:
                    continue
                cpos = c.where(graph_)
                for ident, p in recorded:
                    hit = _np.asarray(p.entry_position_m, dtype=float)
                    if float(_np.linalg.norm(hit - cpos)) <= 0.15 and c.disturb(p.energy_spent_j):
                        self.ordnance.log.append(
                            f"{c.identity} struck by a {p.energy_spent_j:.0f} J hit near {ident.split('.')[-1]}: fuze running")
        # a pressurised gas vessel holed hard enough lets go (conservative:
        # a real volume, real pressure, a real hole with energy behind it)
        import burst as _burst
        for ident, p in recorded:
            if ident in self.state.absent_parts or self._cascade_depth >= 3:
                continue
            node = next((n for n in self._drivetrain.graph["nodes"] if n["identity"] == ident), None)
            if node is None:
                continue
            # a real declared pressure vessel ("high-pressure-canister":
            # the pneumatic reserve/brake reservoirs, a nitrous bottle, a
            # WMI tank, a gasholder) holed with real energy in its wall
            # lets go -- how much it then releases is its own charge and
            # pressure (burst.vessel_energy_j), never a constant
            fill = self._vessel_fill_frac(ident, node)
            e_stored = _burst.vessel_energy_j(node, fill) if node.get("kind") == "high-pressure-canister" else 0.0
            if e_stored >= VESSEL_RUPTURE_MIN_STORED_J and p.energy_spent_j >= VESSEL_RUPTURE_MIN_J:
                p_pa = float(node.get("bottle_pressure_pa", 0.0) or node.get("working_pressure_pa", 0.0) or 0.0)
                self.cascade_log.append(
                    f"vessel {ident.split('.')[-1]} ({_burst.vessel_contents_kg(node, fill) * 1000:.0f} g at "
                    f"{p_pa / 1e5:.0f} bar, {e_stored / 1000:.1f} kJ stored) ruptured by a {p.energy_spent_j:.0f} J hit")
                self.burst_part(ident)

        for mesh_part, impact in penetration.impacts:
            identity = mesh_part_identity(mesh_part, self._drivetrain.graph)
            st = self.state.part_damage.get(identity)
            p = by_part.get(identity)
            self.damage_events.append({
                "kind": "impact", "part": identity, "mode": str(getattr(impact.mode, "value", impact.mode)),
                "material": (p.material if p is not None else str(mesh_part.split("_")[-1])),
                "wall_m": float(impact.effective_thickness_m), "energy_spent_j": float(impact.energy_spent_j),
                "pressure_pa": float(st.pressure_pa) if st is not None else 101_325.0,
                "speed_after_m_s": float(impact.speed_after_m_s),
                "calibre_m": float(getattr(impact.projectile_after, "diameter_m", 0.0076) or 0.0076),
            })
        return recorded

    # ---- the one place a circuit's real pressure and contents are read ----
    def _circuit_by_id(self, circuit_id):
        from hole_emitters import circuit_identity as _cid
        if circuit_id is None:
            return None
        for c in self._drivetrain.fluid_circuits:
            if _cid(c) == circuit_id:
                return c
        return None

    def _plant_reservoir(self, circuit_id):
        """The auxiliary plant's own fluid stores are NOT graph fluid
        circuits -- they live on `plant` (hydraulics.py / refrigeration.
        py) -- so the three accessors below have to know about them, or
        a hole in the hydraulic tank finds no contents and quietly does
        nothing. Returns (kind, object) or (None, None)."""
        p = getattr(self, "plant", None)
        if p is None:
            return None, None
        if circuit_id == "hydraulic":
            return "hydraulic", getattr(p, "hydraulics", None)
        if circuit_id == "refrigerant":
            return "refrigerant", getattr(p, "loop", None)
        if circuit_id == "nitrogen":
            return "nitrogen", getattr(p, "hydraulics", None)
        return None, None

    def _circuit_pressure_pa(self, circuit_id, circuit=None) -> float:
        """What is really behind a hole in that circuit.

        A DEPLETABLE vessel (an air receiver, a bottle, a fuel tank)
        does not carry its pressure in `pressure_pa` at all -- its state
        is `fill_level_frac` against `bottle_capacity_kg`, and in a
        fixed volume the pressure follows the mass directly, so a
        half-empty receiver really is at half its working pressure. A
        vented liquid store has no pressure of its own; a pumped or
        solved circuit carries its own solved value."""
        kind, obj = self._plant_reservoir(circuit_id)
        if obj is not None:
            if kind == "hydraulic":
                # a holed reservoir leaks at its blanket pressure, and a
                # holed line at the working pressure behind it
                return max(float(getattr(obj, "blanket_pressure_pa", ATM_PRESSURE_PA)), ATM_PRESSURE_PA)
            if kind == "refrigerant":
                # the high side is what a holed condenser or drier vents
                return max(float(getattr(obj, "discharge_pa", 0.0)), ATM_PRESSURE_PA)
            if kind == "nitrogen":
                bottle = float(getattr(obj, "nitrogen_bottle_pressure_pa", 0.0) or 0.0)
                return max(ATM_PRESSURE_PA, bottle * max(0.0, float(getattr(obj, "nitrogen_fill_frac", 0.0))))
        c = circuit if circuit is not None else self._circuit_by_id(circuit_id)
        if c is None:
            return ATM_PRESSURE_PA
        if float(getattr(c, "bottle_capacity_kg", 0.0) or 0.0) > 0.0:
            working = float(getattr(c, "working_pressure_pa", 0.0) or 0.0)
            if working <= 0.0:
                return max(float(c.pressure_pa), ATM_PRESSURE_PA)
            return max(ATM_PRESSURE_PA, working * max(0.0, min(1.0, c.fill_level_frac)))
        return max(float(c.pressure_pa), ATM_PRESSURE_PA)

    def _circuit_remaining_l(self, circuit_id, circuit=None) -> float:
        """How much is actually left, read from the real reservoir the
        dashboard shows -- never from a leak ledger."""
        from hole_emitters import DENSITY_KG_M3
        kind, obj = self._plant_reservoir(circuit_id)
        if obj is not None:
            if kind == "hydraulic":
                return float(getattr(obj, "oil_l", 0.0))
            if kind == "refrigerant":
                return 0.9 * max(0.0, float(getattr(obj, "charge_frac", 0.0)))   # a car-sized charge, litres-equivalent
            if kind == "nitrogen":
                return 1e9 if float(getattr(obj, "nitrogen_fill_frac", 0.0)) > 0.0 else 0.0
        c = circuit if circuit is not None else self._circuit_by_id(circuit_id)
        if circuit_id == "oil":
            cs = getattr(self, "_crankcase_state", None)
            if cs is not None:
                return float(cs.oil_kg) / 0.87
        cap = float(getattr(c, "bottle_capacity_kg", 0.0) or 0.0) if c is not None else 0.0
        if cap > 0.0:
            rho = DENSITY_KG_M3.get(circuit_id, 745.0)
            return cap * max(0.0, min(1.0, c.fill_level_frac)) / rho * 1000.0
        if circuit_id == "coolant" and c is not None:
            return max(0.0, float(c.volume_l) - self.state.coolant_lost_l)
        return float(c.volume_l) if c is not None else 0.0

    def _deplete_circuit(self, circuit_id, amount: float) -> None:
        """Take that much out of the REAL reservoir -- litres for a
        liquid, kilograms for a gas. This is the only path by which a
        leak, a burst or a fitted port removes fluid, so the tank bar,
        the sump reading, the receiver pressure and the engine's own
        starvation all move together."""
        from hole_emitters import DENSITY_KG_M3
        if amount <= 0.0:
            return
        kind, obj = self._plant_reservoir(circuit_id)
        if obj is not None:
            if kind == "hydraulic":
                obj.oil_l = max(0.0, obj.oil_l - amount)
                return
            if kind == "refrigerant":
                obj.charge_frac = max(0.0, obj.charge_frac - amount / 0.9)
                return
            if kind == "nitrogen":
                obj.nitrogen_fill_frac = max(0.0, obj.nitrogen_fill_frac
                                             - amount / max(obj.nitrogen_capacity_kg, 1e-9))
                return
        c = self._circuit_by_id(circuit_id)
        if circuit_id == "oil":
            cs = getattr(self, "_crankcase_state", None)
            if cs is not None:
                cs.oil_kg = max(0.0, cs.oil_kg - amount * 0.87)
                self.state.sump_oil_l = cs.oil_kg / 0.87
                return
        if circuit_id == "coolant":
            self.state.coolant_lost_l += amount
            if c is not None:
                c.volume_l = max(0.05, c.volume_l - amount)   # less coolant really is less thermal mass
            return
        if c is not None and float(getattr(c, "bottle_capacity_kg", 0.0) or 0.0) > 0.0:
            liquid = circuit_id in ("fuel", "water")
            kg = amount / 1000.0 * DENSITY_KG_M3.get(circuit_id, 745.0) if liquid else amount
            c.fill_level_frac = max(0.0, c.fill_level_frac - kg / max(c.bottle_capacity_kg, 1e-6))

    def _vessel_fill_frac(self, identity: str, node: dict) -> float:
        """How full that vessel actually is right now: its circuit's own
        live fill_level_frac when it has one, else its declared level."""
        for c in self._drivetrain.fluid_circuits:
            if identity in c.nodes and float(getattr(c, "bottle_capacity_kg", 0.0) or 0.0) > 0.0:
                return float(c.fill_level_frac)
        return float(node.get("fill_level_frac", 1.0) or 1.0)

    def drain_damage_events(self) -> list[dict]:
        events = self.damage_events
        self.damage_events = []
        return events

    def ignite_within(self, centre, radius_m: float, source: str) -> list[str]:
        """A real ignition source (an ordnance fireball, a burning part)
        at a point: every flammable thing within reach lights. What is
        flammable is what is actually THERE -- a fuel emitter's spray,
        or a fuel-carrying part -- so a shot tank that is merely pouring
        catches only when something lights it, never on its own."""
        import burst as _b
        from hole_emitters import fluid_key as _fk
        c = _np.asarray(centre, dtype=float)
        lit: list[str] = []
        # a spray of fuel in the fireball burns back to its source
        for em in self.hole_emitters.emitters:
            if em.fluid != "fuel" or em.regime == "none" or em.part in self.state.absent_parts:
                continue
            pts = em.particles(8, 0.3)
            if len(pts) and float(_np.min(_np.linalg.norm(pts - c[None, :], axis=1))) <= radius_m:
                lit.append(em.part)
        # a fuel-carrying part inside it
        for n in self._drivetrain.graph["nodes"]:
            ident = n["identity"]
            if ident in self.state.absent_parts or ident in lit:
                continue
            pos = n.get("reference_position")
            if pos is None or _b.contents_key(n) != "fuel":
                continue
            if float(_np.linalg.norm(_np.asarray(pos, dtype=float) - c)) <= radius_m:
                lit.append(ident)
        for ident in dict.fromkeys(lit):
            self.cascade_log.append(f"{source} ignited {ident.split('.')[-1]}")
            self.burst_part(ident, ignited=True)
        return list(dict.fromkeys(lit))

    def burst_part(self, identity: str, energy_j: float | None = None, ignited: bool = False) -> "PartBurst | None":
        """An explosive failure of one graph node: it becomes absent, its
        material and contents fly (burst.make_burst -- ranged by the
        density of what they fly into), its fluid leaves its circuit
        and an open end stays on that circuit as a hole emitter, and
        the damage synth gets the blast."""
        import burst as burst_module
        from hole_emitters import HoleEmitter, LIQUID_OF_CIRCUIT, circuit_identity as _cid
        graph = self._drivetrain.graph
        node = next((n for n in graph["nodes"] if n["identity"] == identity), None)
        if node is None or identity in self.state.absent_parts:
            return None
        fluid = node.get("fluid")
        vol_l = float(node.get("fluid_volume_l", 0.0) or 0.0)
        if energy_j is None:
            # what it holds decides: a fuel volume deflagrates, a gas
            # vessel releases its stored pV work, anything else is a
            # bare mechanical let-go
            from hole_emitters import fluid_key as _fkey
            import burst as _burst_e
            stored = (_burst_e.vessel_energy_j(node, self._vessel_fill_frac(identity, node))
                      if node.get("kind") == "high-pressure-canister" else 0.0)
            if stored > 0.0:
                energy_j = stored          # a real vessel: its own declared charge and pressure
            elif _burst_e.contents_key(node) == "fuel" and ignited:
                self._light_fire(identity, node)
                # fuel only ever burns when something LIT it: an ordnance
                # fireball, or its own spray reaching a part above the
                # fuel's autoignition temperature. A tank that is merely
                # shot open splits and pours -- petrol does not detonate
                # because a bullet went through it.
                energy_j = _burst_e.fuel_vapour_energy_j(node, self._vessel_fill_frac(identity, node))
            else:
                c = next((c for c in self._drivetrain.fluid_circuits if identity in c.nodes), None)
                if c is not None and c.kind_class == "compressible-gas" and c.pressure_pa > 150_000.0:
                    p = float(c.pressure_pa); v = max(vol_l, 0.1) / 1000.0
                    energy_j = p * v * math.log(p / 101_325.0)
                else:
                    energy_j = 1500.0     # a bare mechanical let-go: it splits, it does not burn
        pts = [n["reference_position"] for n in graph["nodes"] if not n.get("chassis_side") and n.get("reference_position") is not None]
        floor_y = min(p[1] for p in pts) - 0.25 if pts else -0.5
        b = burst_module.make_burst(graph, identity, float(energy_j), self.bursts.rng, floor_y)
        self.bursts.bursts.append(b)
        self.state.absent_parts.add(identity)
        # its fluid is gone from the circuit; the circuit is open here
        cid = next((_cid(c) for c in self._drivetrain.fluid_circuits if identity in c.nodes), None)
        if cid is not None:
            spilled_l = vol_l if vol_l > 0.0 else self._circuit_remaining_l(cid)
            if spilled_l > 0.0:
                self.hole_emitters.lost_l[cid] = self.hole_emitters.lost_l.get(cid, 0.0) + spilled_l
                self._deplete_circuit(cid, spilled_l)     # the real reservoir loses it, not a ledger
            liquid = LIQUID_OF_CIRCUIT.get(cid) or (_fkey(fluid) if _fkey(fluid) != "gas" else None)
            r_open = float(next((e.get("radius", 0.008) for e in graph["edges"] if e["a"] == identity or e["b"] == identity), 0.008))
            self.hole_emitters.emitters.append(HoleEmitter(
                identity=f"{identity}.burst_open_end", part=identity, circuit=cid, fluid=liquid or "gas",
                position=tuple(float(v) for v in node["reference_position"]), direction=(0.0, -1.0, 0.0),
                radius_m=max(0.004, r_open), through=False))
        # AND WHATEVER IT SIMPLY HELD. A part can contain a real volume
        # without being on any pumped circuit -- a gear case, a cooler,
        # a filter housing -- and when the part is gone that volume is
        # not "still in the system", it is on the floor. This is the
        # rule that makes it burst in place rather than quietly vanish
        # with the part.
        self._spill_contained_fluid(identity, node)
        self.damage_events.append({"kind": "blast", "part": identity, "energy_j": float(energy_j),
                                   "volume_m3": max(0.002, vol_l / 1000.0 + 0.01)})
        self._cascade_fragments(b, node)
        return b

    def _spill_contained_fluid(self, identity: str, node: dict | None = None) -> float:
        """A destroyed part dumps what it was holding, where it stood.

        Circuit fluid is already handled by the circuit itself. This is
        for the volume a part contains on its own -- the gear oil in a
        differential, the litre in a cooler, what is in a filter
        housing. It leaves as a real emitter at the part's own position
        with its own contents, so it pours, pools, sounds and colours
        itself like any other spill, and it stops when the part is
        empty because nothing is pumping it."""
        from hole_emitters import HoleEmitter
        graph = self._drivetrain.graph
        if node is None:
            node = next((n for n in graph["nodes"] if n["identity"] == identity), None)
        if node is None:
            return 0.0
        litres = float(node.get("fluid_volume_l", 0.0) or 0.0)
        fluid = node.get("fluid")
        if litres <= 0.0 or not fluid:
            return 0.0
        if any(em.identity == f"{identity}.contents" for em in self.hole_emitters.emitters):
            return 0.0                                   # already spilling
        from fluids import fluid_key as _fk
        key = _fk(fluid)
        if key is None or key == "gas":
            return 0.0
        pos = node.get("reference_position") or (0.0, 0.0, 0.0)
        half = node.get("body_half_extent_m") or (0.05, 0.05, 0.05)
        # a case that has been destroyed is open across its whole
        # bottom, not through a bullet hole: the opening is sized from
        # the part itself
        r_open = max(0.01, 0.35 * min(float(half[0]), float(half[2])))
        self.hole_emitters.emitters.append(HoleEmitter(
            identity=f"{identity}.contents", part=identity, circuit=None, fluid=key,
            position=tuple(float(v) for v in pos), direction=(0.0, -1.0, 0.0),
            radius_m=r_open, through=False, kind="contained",
            contained_l=litres, contained_head_m=max(0.05, float(half[1]) * 2.0)))
        self.damage_events.append({"kind": "spill", "part": identity,
                                   "fluid": key, "litres": litres})
        return litres

    def _cascade_fragments(self, b, node) -> None:
        """The shrapnel through the projectile engine: the casing
        fragments with real energy (>= CASCADE_MIN_J, the most energetic
        CASCADE_MAX_RAYS of them) each become a tumbling projectile
        (ballistics.ProjectileState: its own mass, size, speed, a random
        yaw and tumble rate, the casing material's hardness) fired along
        its launch direction through RayMesh.penetrate -> apply_
        penetration -- holes, emitters, sounds, and possibly the next
        rupture (depth-limited). Conservative: fragments are blunt,
        tumbling, and mostly below the energy that holes a casting."""
        if self.ray_mesh_factory is None or self._cascade_depth >= 3:
            return
        from ballistics import ProjectileState
        from engine_rays import Ray
        import burst as burst_module
        solid = b.clouds[0] if b.clouds else None
        if solid is None or not len(solid.pos):
            return
        e = 0.5 * solid.mass * _np.sum(solid.vel ** 2, axis=1)
        order = _np.argsort(-e)
        picks = [i for i in order[:CASCADE_MAX_RAYS] if e[i] >= CASCADE_MIN_J]
        if not picks:
            return
        hardness = 1.0e9 if "iron" in str(node.get("material", "")) or "steel" in str(node.get("material", "")) else 0.5e9
        rm = self.ray_mesh_factory()
        rng = self.bursts.rng
        self._cascade_depth += 1
        try:
            for i in picks:
                v = solid.vel[i]; speed = float(_np.linalg.norm(v))
                d = v / max(speed, 1e-9)
                r = float(solid.radius[i])
                ps = ProjectileState(mass_kg=float(solid.mass[i]), diameter_m=2.0 * r, speed_m_s=speed,
                                     direction=tuple(float(x) for x in d), length_m=2.5 * r,
                                     yaw_rad=float(rng.uniform(0.2, math.pi / 2)), tumble_rad_s=float(rng.uniform(30.0, 200.0)),
                                     hardness_pa=hardness, integrity=0.8)
                start = solid.pos[i] + d * 0.02
                pen = rm.penetrate(Ray.from_points(start, start + d), ps.energy_j, 2.0 * r, projectile=ps,
                                   fluid_by_part=self.ballistic_fluid_for_part)
                rec = self.apply_penetration(pen)
                if rec:
                    self.cascade_log.append(
                        f"fragment {solid.mass[i] * 1000:.0f} g at {speed:.0f} m/s ({e[i]:.0f} J) from {b.part.split('.')[-1]}: "
                        + "; ".join(f"{ident.split('.')[-1]} {q.damage_mode}" for ident, q in rec))
        finally:
            self._cascade_depth -= 1

    def _step_oil_pickup(self) -> None:
        """A falling sump uncovers the pickup, and the pump draws air.
        The air is put into the oil circuit's own contents (the same
        FluidMix a hole's ingest would fill), so the SAME cavitation
        rule that handles a holed suction line handles a dry sump --
        one mechanism, not two."""
        cs = getattr(self, "_crankcase_state", None)
        c = self._circuit_by_id("oil")
        if cs is None or c is None:
            return
        level = float(cs.oil_kg) / max(float(cs.oil_capacity_kg), 1e-6)
        if level >= OIL_PICKUP_UNCOVERS_FRAC:
            self.state.oil_pickup_air_frac = 0.0
            return
        span = max(OIL_PICKUP_UNCOVERS_FRAC - OIL_PICKUP_DRY_FRAC, 1e-6)
        air = max(0.0, min(1.0, (OIL_PICKUP_UNCOVERS_FRAC - level) / span))
        self.state.oil_pickup_air_frac = air
        mix = self.hole_emitters.contents("oil", c, "engine-oil")
        total = mix.total_kg
        if total > 0.0:
            # hold the blend at the fraction of air the pickup is really
            # drawing, rather than accumulating it forever
            want_air = total * air
            have_air = mix.kg.get("air", 0.0)
            if want_air > have_air:
                mix.add("air", want_air - have_air)
            elif have_air > want_air:
                mix.kg["air"] = want_air
        # and the delivered pressure falls with it: a pump passing air
        # cannot hold its relief pressure
        self.state.oil_pressure_pa = ATM_PRESSURE_PA + (self.state.oil_pressure_pa - ATM_PRESSURE_PA) * (1.0 - air)

    def _apply_node_effects(self) -> None:
        """node_effects.assess -> the levers the sim already integrates:
        per-cylinder charge (in the firing loop via _effects), fuel and
        mixture (idem), exhaust restriction (the solver call), boost (the
        turbo step), the coolant/oil circuits' own exchange and pump
        sizing (scaled from their untouched base), a dead engine (the
        crank seizes: ignition off, a hard decel)."""
        import node_effects
        self._node_conditions, self._effects = node_effects.assess(self)
        ef = self._effects
        # A PART THAT HAS LOST ITS STRUCTURE DUMPS WHAT IT HELD. This
        # catches the destruction that is not an explosion -- a case
        # beaten open by repeated hits, a housing corroded until its
        # remaining wall gave up -- which otherwise left its oil
        # nowhere, neither in the part nor on the floor.
        for ident, cond in self._node_conditions.items():
            if cond.structure_lost:
                self._spill_contained_fluid(ident)
        from hole_emitters import circuit_identity as _cid
        for c in self._drivetrain.fluid_circuits:
            cid = _cid(c)
            base = self._circuit_base.setdefault(cid, (float(c.active_heat_exchange_w_per_k), float(c.pump_design_flow_lpm)))
            if cid == "coolant":
                c.active_heat_exchange_w_per_k = base[0] * ef.coolant_exchange_factor
                c.pump_design_flow_lpm = base[1] * ef.coolant_flow_factor
            elif cid == "oil":
                c.pump_design_flow_lpm = base[1] * ef.oil_flow_factor
        self.state.smoke_factor = ef.smoke_factor
        self.state.exhaust_open_frac = ef.exhaust_open_frac
        if ef.engine_dead and not self.state.engine_dead:
            self.state.engine_dead = ef.engine_dead
            self.state.ignition_cut = True
        if ef.spark_lost and not self.engine.compression_ignition:
            # a compression-ignition engine has no spark to lose; a spark
            # engine with its ignition shot away simply stops firing
            self.state.ignition_cut = True
        self.state.spark_lost = ef.spark_lost
        self.state.brakes_lost = ef.brakes_lost
        self.state.startable = ef.startable
        self.state.mounts_lost = ef.mounts_lost

    def _light_fire(self, identity: str, node: dict) -> None:
        """A real pool fire from what that part was actually holding."""
        import burst as _b
        fuel_kg = _b.vessel_contents_kg(node, self._vessel_fill_frac(identity, node))
        if fuel_kg <= 0.0:
            fuel_kg = float(node.get("fluid_volume_l", 0.0) or 0.0) / 1000.0 * 745.0
        pos = node.get("reference_position") or (0.0, 0.0, 0.0)
        f = self.fires.light(identity, pos, _b.contents_key(node) or "fuel", max(fuel_kg, 0.05), "ignition")
        if f is not None:
            self.cascade_log.append(f"{identity.split('.')[-1]} alight: {f.heat_release_w / 1000:.0f} kW pool fire")

    def _step_fire(self, dt: float) -> None:
        """Burn, spread by radiation, breathe the bay's own oxygen, and
        heat what is around. Nothing here starts a fire -- only a real
        ignition does (ignite_within / an ordnance fireball / a spray on
        a hot part)."""
        air = getattr(self, "_air", None)
        # a fire breathes the volume it is actually IN: a spill under the
        # vehicle pools on the garage floor, not inside the engine bay,
        # and a 60-litre tank fire in a one-cubic-metre bay would be a
        # fiction. Each fire is assigned the smallest volume that
        # contains it -- the bay if it is inside the engine's own
        # envelope, else the room, else outdoors.
        volumes = self._fire_volumes(air)
        burned = 0.0
        for f in self.fires.fires:
            vol = volumes.get(f.identity)
            o2 = float(getattr(vol, "o2_frac", 0.2095)) if vol is not None else 0.2095
            used = f.step(dt, o2)
            burned += used
            if used > 0.0 and vol is not None and dt > 0.0:
                fuel_kg_s = used / dt
                vol.add_exhaust(fuel_kg_s * (1.0 + AIR_PER_FUEL_MASS), POOL_FIRE_PLUME_K,
                                fuel_kg_s * POOL_FIRE_CO_YIELD_KG_PER_KG, 0.0)
        self.fires.total_burned_kg += burned
        self.state.fire_heat_release_w = self.fires.heat_release_w
        self.state.fires_burning = len(self.fires.burning)
        if not self.fires.burning:
            return
        # spread: anything flammable this fire is radiating enough at
        import burst as _b
        cand = []
        for n in self._drivetrain.graph["nodes"]:
            ident = n["identity"]
            if ident in self.state.absent_parts or _b.contents_key(n) != "fuel":
                continue
            if any(f.identity == ident for f in self.fires.fires):
                continue
            pos = n.get("reference_position")
            if pos is not None:
                cand.append((ident, pos))
        for ident, pos, flux in self.fires.spread_targets(cand):
            self.cascade_log.append(f"fire spread to {ident.split('.')[-1]} ({flux / 1000:.0f} kW/m2 radiant)")
            self.burst_part(ident, ignited=True)


    def _fire_volumes(self, air) -> dict:
        """Which air volume each fire breathes: the engine bay when the
        fire sits inside the engine's own envelope, otherwise the room
        (or nothing at all, outdoors)."""
        if air is None:
            return {}
        pts = [n["reference_position"] for n in self._drivetrain.graph["nodes"]
               if not n.get("chassis_side") and n.get("reference_position") is not None]
        out = {}
        if pts:
            lo = [min(p[i] for p in pts) - 0.15 for i in range(3)]
            hi = [max(p[i] for p in pts) + 0.15 for i in range(3)]
        else:
            lo = hi = None
        room = getattr(air, "garage", None)
        for f in self.fires.fires:
            inside = lo is not None and all(lo[i] <= f.position[i] <= hi[i] for i in range(3))
            out[f.identity] = air.bay if inside else room
        return out

    def _plant_wants_cooling(self) -> bool:
        """True while the plant's refrigerant loop has a load calling and
        its master switch is on -- what actually pulls the clutch in."""
        p = getattr(self, "plant", None)
        if p is None or not p.controls.main_chiller:
            return False
        return any(l.calling and l.demand_w > 0.0 for l in p.loop.loads)

    def _plant_leak_losses(self, dt: float) -> None:
        """What the plant's own holed hardware is losing.

        The refrigerant loop is the one that matters most: a holed
        chiller, condenser or drier vents its charge to atmosphere, and
        the loop's own LOW-PRESSURE CUTOUT is what then stops the
        compressor -- which is the real protection, and the real reason
        a shot air-conditioning system simply stops working rather than
        destroying its compressor."""
        p = self.plant
        if p is None:
            return
        vented = 0.0
        for em in self.hole_emitters.emitters:
            if em.circuit == "refrigerant" and em.regime not in ("none", "fitted"):
                vented += em.mass_flow_kg_s * dt
        if vented > 0.0:
            # a car-sized loop holds well under a kilogram, so a real
            # hole empties it in seconds
            charge_kg = 0.9
            p.loop.charge_frac = max(0.0, p.loop.charge_frac - vented / charge_kg)
        # the hydraulic reservoir loses what its own holes pass
        h = getattr(p, "hydraulics", None)
        if h is not None:
            lost_l = 0.0
            for em in self.hole_emitters.emitters:
                if em.circuit == "hydraulic" and em.regime not in ("none", "fitted", "splash"):
                    lost_l += em.mass_flow_kg_s / 870.0 * 1000.0 * dt
            if lost_l > 0.0:
                h.oil_l = max(0.0, h.oil_l - lost_l)
        # and a holed nitrogen bottle simply empties
        if h is not None and h.nitrogen_fitted:
            n2 = sum(em.mass_flow_kg_s for em in self.hole_emitters.emitters
                     if em.circuit == "nitrogen" and em.regime not in ("none", "fitted"))
            if n2 > 0.0:
                h.nitrogen_fill_frac = max(0.0, h.nitrogen_fill_frac
                                           - n2 * dt / max(h.nitrogen_capacity_kg, 1e-9))

    def fit_loadout(self, identity: str) -> list[str]:
        """Bolt a named equipment package onto this engine.

        The rig then drives the engine's own hydraulic levers from what
        the equipment is really doing, and takes bench supply for
        whatever this engine cannot provide -- which is the whole point
        of not having to build the parts onto the engine by hand."""
        from equipment import EquipmentRig
        self.equipment = EquipmentRig.from_loadout(identity)
        import loadouts
        sup = loadouts.supply_from_engine(self)
        if sup is None:
            return [f"  {identity} fitted; this engine has no hydraulic plant, so the bench "
                    f"will supply all of it"]
        return loadouts.get(identity).check_against(sup["flow_l_min"], sup["pressure_pa"])

    def remove_loadout(self) -> None:
        """Take the equipment off, and stop driving the levers it was
        driving -- otherwise the plant keeps loading the crank for work
        nothing is doing any more."""
        self.equipment = None
        self.hydraulic_flow_frac = 0.0
        self.hydraulic_load_frac = 0.0

    def _step_equipment(self, dt: float) -> None:
        if self.equipment is not None:
            self.equipment.step(dt, self)

    def _build_automatic(self):
        """The automatic gearbox this engine declares, if it declares
        one."""
        tr = getattr(self.engine, "transmission", None)
        if tr is None or str(getattr(tr, "kind", "manual")) != "automatic":
            return None
        from automatic_transmission import AutomaticTransmission
        return AutomaticTransmission(
            fluid_l=float(getattr(tr, "fluid_l", 9.5)),
            gear_ratios=tuple(tr.gear_ratios), final_drive_ratio=float(tr.final_drive_ratio),
            lock_up_capable=bool(getattr(tr, "lock_up_capable", True)))

    def _step_automatic(self, dt: float) -> None:
        """The converter and the packs, against what the crank and the
        load are really doing this tick."""
        at = getattr(self, "automatic", None)
        if at is None:
            return
        ambient = 293.15
        air = getattr(self, "_air", None)
        if air is not None and getattr(air, "bay", None) is not None:
            ambient = float(getattr(air.bay, "temp_k", 293.15) or 293.15)
        at.step(dt, float(self._omega), float(self._load_omega), float(self.throttle), ambient)
        # what a hole took out of it is what it has lost: the emitter is
        # the only place fluid leaves, here as everywhere else
        for em in self.hole_emitters.emitters:
            if em.fluid == "transmission-fluid" and em.mass_flow_kg_s > 0.0 and em.regime not in ("none", "fitted"):
                at.lose(em.mass_flow_kg_s / 850.0 * 1000.0 * dt)

    def _build_air_vessels(self):
        """The real pneumatic vessels this build actually carries."""
        try:
            from air_vessels import AirVesselSet
            from hole_emitters import circuit_identity as _cid
        except Exception:
            return None
        pneu = next((c for c in self._drivetrain.fluid_circuits
                     if _cid(c) == "pneumatic-reserve"), None)
        if pneu is None or pneu.bottle_capacity_kg <= 0.0:
            return None
        return AirVesselSet.from_graph(self._drivetrain.graph, pneu)

    def _step_air_vessels(self, dt: float) -> None:
        """Distribute the circuit's own stored mass across the vessels.

        The circuit stays authoritative -- this never creates or
        destroys air, it only moves the one real total between the wet
        tank, the reserve and the protected brake reservoirs."""
        vs = getattr(self, "air_vessels", None)
        if vs is None:
            return
        from hole_emitters import circuit_identity as _cid
        pneu = next((c for c in self._drivetrain.fluid_circuits
                     if _cid(c) == "pneumatic-reserve"), None)
        if pneu is None:
            return
        vs.protection_pressure_pa = float(getattr(pneu, "protection_pressure_pa", 0.0) or 0.0)
        vs.step(dt, float(pneu.fill_level_frac) * float(pneu.bottle_capacity_kg))

    def _step_plant(self, dt: float) -> None:
        """Feed the auxiliary plant what the engine is really doing: the
        compressor's own delivered mass flow, the AC compressor's real
        shaft speed, the bus it charges from, and the airflow its
        condenser is actually getting."""
        from hole_emitters import circuit_identity as _cid
        pneu = next((c for c in self._drivetrain.fluid_circuits if _cid(c) == "pneumatic-reserve"), None)
        flow = float(getattr(pneu, "delivered_flow_kg_s", 0.0) or 0.0) if pneu is not None else 0.0
        ac_omega = float(self._drivetrain.omega.get("ac_compressor", 0.0))
        ambient_k = 293.15
        air = getattr(self, "_air", None)
        if air is not None and getattr(air, "bay", None) is not None:
            ambient_k = float(getattr(air.bay, "temp_k", 293.15) or 293.15)
        # the condenser sits in the cooling stack: it gets whatever air
        # the fan and road speed are really moving through it
        fan_frac = 1.0 if self.engine.accessories.mechanical_fan else 0.0
        fan_frac = max(fan_frac, float(self._drivetrain_out.get("cooling_fan_flow_m3_s", 0.0)) > 0.0)
        reading = getattr(self.electrical, "reading", None)
        bus_v = float(getattr(reading, "voltage_v", 12.6) or 12.6)
        charging = float(getattr(reading, "battery_current_a", 0.0) or 0.0) >= -0.5 and self.state.rpm > 200.0
        self.plant.step(dt, compressor_flow_kg_s=flow, ac_omega_rad_s=ac_omega,
                        intake_k=float(self.state.intake_charge_temp_k or 293.15),
                        tank_pressure_pa=float(self.engine.pneumatics.tank_pressure_pa),
                        ambient_k=ambient_k, bus_voltage_v=bus_v, engine_charging=bool(charging),
                        condenser_airflow=float(fan_frac), crank_rpm=float(self.state.rpm),
                        coolant_temp_k=float(self.state.coolant_temp_k))
        h = self.plant.hydraulics
        if h is not None:
            h.commanded_flow_frac = float(self.hydraulic_flow_frac)
            h.load_frac = float(self.hydraulic_load_frac)
            # oil over its own flash point is a real fire waiting for a
            # source: a hydraulic tank that hot, holed and spraying is
            # exactly how machine fires start. It is NOT self-igniting --
            # the same rule as everything else here: something has to
            # light it (fire.py / ignite_within), and if anything already
            # is burning nearby, this is what catches next.
            if h.burning and self.fires.burning:
                tank = next((n for n in self._drivetrain.graph["nodes"]
                             if n["identity"] == "plant.hydraulic_tank"), None)
                if tank is not None and "plant.hydraulic_tank" not in self.state.absent_parts:
                    if not any(f.identity == "plant.hydraulic_tank" for f in self.fires.fires):
                        self.cascade_log.append(
                            f"hydraulic oil at {h.temp_k - 273.15:.0f} C is over its flash point next to a fire")
                        self.fires.light("plant.hydraulic_tank", tank["reference_position"], "engine-oil",
                                         h.oil_l / 1000.0 * 870.0, "hydraulic oil over flash point")
        self.state.plant_dewpoint_k = self.plant.delivered_dewpoint_k
        self.state.plant_gunk_kg = self.plant.treatment.gunk.total_kg
        if self.plant.hydraulics is not None:
            self.state.hydraulic_oil_temp_k = self.plant.hydraulics.temp_k

    def _step_fittings(self, dt: float) -> None:
        """Anything screwed into a hole gets its flow this tick."""
        if not self.fittings.ports:
            return
        from hole_emitters import circuit_identity as _cid, DENSITY_KG_M3
        pressures = {_cid(c): self._circuit_pressure_pa(_cid(c), c) for c in self._drivetrain.fluid_circuits}
        by_id = {em.identity: em for em in self.hole_emitters.emitters}
        self.fittings.step(dt, by_id, {"pressures": pressures, "fires": self.fires,
                                       "coolant_temp_k": self.state.coolant_temp_k})
        # what a fitted port takes really leaves the circuit
        for port in self.fittings.ports:
            if port.flow_kg_s > 0.0 and not port.fed_from_outside:
                em = by_id.get(port.emitter_identity)
                if em is not None and em.circuit:
                    if em.fluid == "gas":
                        self._deplete_circuit(em.circuit, port.flow_kg_s * dt)
                    else:
                        rho = DENSITY_KG_M3.get(em.fluid, 900.0)
                        self._deplete_circuit(em.circuit, port.flow_kg_s * dt / rho * 1000.0)

    def _resolve_spray_targets(self) -> None:
        """Ask the geometry where each jet actually lands.

        Oil thrown inside an engine is not lost -- it hits a web, a
        skirt, the inside of a cover, and runs back to the sump. The only
        oil that goes is the oil that gets OUT. Rather than assume either
        way, every emitter's cone is cast against the same ray mesh a
        projectile uses and the answer read off what it hits (see
        HoleEmitterField.resolve_spray).

        Cheap by construction: a handful of rays per emitter, once, and
        again only when the geometry changes under it -- a round through
        a cover turns an emitter that was spraying onto the inside of
        that cover into one spraying at the sky, and the oil it throws
        genuinely does start leaving the engine."""
        if self.ray_mesh_factory is None:
            return
        if not any(not em.spray_resolved for em in self.hole_emitters.emitters):
            return
        try:
            self.hole_emitters.resolve_spray(self.ray_mesh_factory())
        except Exception:
            # never let a spray question stop the engine running; an
            # unresolved emitter keeps its honest default of "it left"
            pass

    def _step_oil_contamination(self, dt: float, cs, fuel_kg_s: float) -> None:
        """What the engine is putting into its own oil, this tick.

        Three things that were each modelled and none of which were
        joined up:

          * the wear ledger knows how far gone every part is, so the
            CHANGE in that is metal off those parts and into the sump
            (wear_debris: a closed mass balance, not a guessed rate)
          * blow-by carries combustion soot down past the rings, and on
            a diesel that is a hundred times the mass of the metal
          * the rings' own wear opens them up, which is what blow-by IS,
            so a tired ring pack blackens its oil faster for the real
            reason rather than by a separate rule

        The filter then takes what it can reach of it, and what it
        cannot reach stays in the oil -- which is what an oil sample
        reads, and why one taken after a change reads clean on an engine
        that is still destroying itself."""
        led = self._oil_debris
        before = self._wear_vector_last
        now = wear_debris.damage_vector(self.state.wear)
        if before is not None:
            led.debit(before, now, float(getattr(self.engine, "displacement_l", 4.0) or 4.0))
        self._wear_vector_last = now
        # the rings are as open as they are worn
        cs.apply_ring_wear(float(self.state.wear.get("piston_rings", 0.0) or 0.0))
        # soot, by blow-by, at this fuel's own declared sootiness
        # THE SAME CLOCK FOR EVERYTHING THAT ACCUMULATES WITH DUTY.
        # `wear.WEAR_TIME_ACCELERATION` exists so an engine can be aged
        # inside a play session; if it sped up metal and not soot, an
        # accelerated engine would wear its rings out with clean oil and
        # every ratio between the two would be wrong. Soot, filtration
        # and settling all run on it as well, so what changes is how
        # fast the clock turns and not what the engine does.
        duty = max(0.0, wear_module.WEAR_TIME_ACCELERATION) * dt
        family = combustion_kernel.family_for(self.engine, self.fuel_choice)
        led.debit_soot(wear_debris.soot_into_oil_g(
            max(0.0, float(fuel_kg_s)) * duty, family,
            float(_np.mean(cs.ring_leak))))
        # the charge passing the element, and a little of what is in it
        # dropping out where no filter reaches
        led.through_filter(min(1.0, duty * OIL_TURNOVER_PER_S), OIL_FILTER_BETA)
        led.settle(duty * OIL_SETTLING_PER_S)
        self.state.oil_soot_frac = led.soot_frac_of_oil(cs.oil_kg)
        self.state.oil_debris_captured_g = led.captured_g

    def _check_burst_triggers(self, dt: float) -> None:
        """The real fire: fuel spraying from a punctured fuel part whose
        stream reaches an exhaust part above the fuel's autoignition
        temperature lights after a short exposure, and the fuel part
        deflagrates."""
        if not self.hole_emitters.emitters:
            return
        import burst as burst_module
        hot = float(self.state.exhaust_temp_k) >= burst_module.FUEL_AUTOIGNITION_K["fuel"]
        graph = self._drivetrain.graph
        hot_nodes = None
        for em in self.hole_emitters.emitters:
            if em.fluid != "fuel" or em.regime not in ("spray", "pour") or em.part in self.state.absent_parts:
                continue
            if not hot:
                self._fuel_exposure_s[em.part] = 0.0
                continue
            if hot_nodes is None:
                hot_nodes = _np.array([n["reference_position"] for n in graph["nodes"]
                                      if "exhaust" in n["identity"] and n.get("reference_position") is not None] or [[1e9, 1e9, 1e9]])
            pts = em.particles(8, 0.3)
            d = float(_np.min(_np.linalg.norm(pts[:, None, :] - hot_nodes[None, :, :], axis=2)))
            if d <= 0.12:
                self._fuel_exposure_s[em.part] = self._fuel_exposure_s.get(em.part, 0.0) + dt
                if self._fuel_exposure_s[em.part] >= burst_module.FUEL_EXPOSURE_S:
                    self.cascade_log.append(
                        f"fuel spray from {em.part.split('.')[-1]} on {self.state.exhaust_temp_k:.0f} K exhaust: ignited")
                    self.burst_part(em.part, ignited=True)
                    self._fuel_exposure_s[em.part] = 0.0
            else:
                self._fuel_exposure_s[em.part] = max(0.0, self._fuel_exposure_s.get(em.part, 0.0) - dt)

    def raise_blast(self, energy_j: float, part: str = "", volume_m3: float = 0.01) -> None:
        """An explosive failure event for the damage synth (a fuel volume
        lighting, a vessel letting go) -- the sim's failure logic calls
        this; nothing here invents one."""
        self.damage_events.append({"kind": "blast", "part": part, "energy_j": float(energy_j), "volume_m3": float(volume_m3)})

    def ballistic_fluid_for_part(self, mesh_part: str, material: str) -> FluidLayer | None:
        """Return the live contained fluid seen behind a struck part wall."""
        identity = mesh_part_identity(mesh_part, self._drivetrain.graph)
        state = self.state.part_damage.get(identity)
        if state is None and "water_jacket" in mesh_part:
            # a jacket piece the graph has no node for: give it a record
            # and bind that ONE record, rather than re-binding every part
            state = PartDamageState(identity=mesh_part)
            self.state.part_damage[mesh_part] = state
            sync_part_pressures({mesh_part: state}, self._drivetrain.fluid_circuits)
        pressure_pa = state.pressure_pa if state is not None else 101_325.0
        return inferred_fluid(mesh_part, material, pressure_pa)

    def _ensure_air_stack(self):
        """The bay/room air chain (air_volumes.AirStack), sized off the
        engine's own real graph envelope, rebuilt when the engine or the
        room mode changes."""
        import air_volumes
        key = (self.engine.identity, self.garage_mode)
        if getattr(self, "_air_key", None) != key:
            ext = (0.9, 0.7, 0.8)
            try:
                pts = [n["reference_position"] for n in self._drivetrain_graph_nodes() if not n.get("chassis_side")]
                lo = [min(p[i] for p in pts) for i in range(3)]; hi = [max(p[i] for p in pts) for i in range(3)]
                ext = tuple(max(0.3, hi[i] - lo[i]) for i in range(3))
            except Exception:
                pass
            bay_m3 = (ext[0] + 0.4) * (ext[1] + 0.3) * (ext[2] + 0.4)
            self._air = air_volumes.AirStack.build(bay_m3, garage_mode=self.garage_mode)
            self._air_key = key
        self._air.set_garage_mode(self.garage_mode) if self._air.garage_mode != self.garage_mode else None
        self._air.extractor_fitted = self.exhaust_extractor_fitted
        self._air.cell_fan_m3_s = self.cell_fan_m3_s
        return self._air

    def _drivetrain_graph_nodes(self):
        return self._drivetrain.graph.get("nodes", [])

    def _step_air_and_emissions(self, dt: float) -> None:
        """Engine-out -> catalyst -> the volume the pipe ends in ->
        the occupant; and what the intake breathes back next tick."""
        import emissions
        eng = self.engine
        air = self._ensure_air_stack()
        if not hasattr(self, "_catalyst_state"):
            self._catalyst_state = emissions.CatalystState()
            self._occupant = emissions.Occupant()
        ci = getattr(eng, "compression_ignition", False) or eng.identity in getattr(engines, "_COMPRESSION_IGNITION", set())
        running = not self.state.stalled
        exhaust_kg_s = (self._exhaust_demand_kg_s if running else 0.0)
        fuel_kg_s = (self._fuel_demand_kg_s * min(self.state.mixture_phi, 2.0) if running else 0.0)
        rates = emissions.engine_out(exhaust_kg_s, fuel_kg_s, self.state.mixture_phi,
                                     self.state.manifold_pressure_frac, ci, misfire_frac=0.0,
                                     crevice_factor=eng.chamber.hc_factor)
        self.state.co_engine_out_g_s = rates.co_kg_s * 1000.0
        cat_spec = None
        if eng.exhaust_system.catalyst_fitted and eng.exhaust_system.layout_has_catalyst:
            cat_spec = self.catalytic_converter
        # the brick sees the collector-outlet gas: segment temps after the
        # collector, before the cat (the chain the sim already computes)
        inlet_k = self.state.exhaust_temp_k
        segs = eng.exhaust_system.segments
        temps = eng.exhaust_system.segment_outlet_temps_k(self.state.exhaust_temp_k, exhaust_kg_s, EXHAUST_TEMP_AMBIENT_K)
        for s, t in zip(segs, temps):
            if s.kind == "collector":
                inlet_k = t
        co, hc, nox = self._catalyst_state.step(dt, cat_spec, inlet_k, exhaust_kg_s, self.state.mixture_phi, rates)
        self.state.catalyst_brick_temp_k = self._catalyst_state.brick_temp_k
        self.state.catalyst_co_efficiency = self._catalyst_state.co_efficiency
        self.state.catalyst_nox_efficiency = self._catalyst_state.nox_efficiency
        self.state.co_tailpipe_g_s = co * 1000.0
        self.state.hc_tailpipe_g_s = hc * 1000.0
        self.state.nox_tailpipe_g_s = nox * 1000.0
        # where the pipe ends
        dest = "bay" if eng.exhaust_system.open_ended_in_bay else "tailpipe"
        indoors = air.exhaust_to(dest, exhaust_kg_s, self.state.exhaust_tailpipe_temp_k, co, rates.o2_frac_in_exhaust)
        self.state.co_emitted_indoors_kg += indoors * dt
        # heat into the bay: the block/radiator share of waste heat (the
        # exhaust share left through the pipe above) -- a disclosed 0.55
        # of the total, the coolant+oil+convection fraction this sim's
        # own circuit heat shares carry
        air.bay.heat_in_w += max(self._waste_heat_kw, 0.0) * 1000.0 * 0.55 * (1.0 if running else 0.0)
        air.ventilate_bay(self._drivetrain_out.get("cooling_fan_flow_m3_s", 0.0))
        air.ventilate_garage()
        # the intake draw, from wherever the filter actually is
        source_name = getattr(eng.intake_system, "air_source", "engine-bay")
        if source_name == "engine-bay":
            source = air.bay
            self.state.intake_source = "bay"
        else:
            source = air.outside
            self.state.intake_source = "room" if source is not None else "outside"
        draw = (self._intake_demand_kg_s / 1.2) if running else 0.0
        if source is not None:
            source.draw_m3_s += draw
            self._intake_source_temp_k = source.temp_k
            self.state.intake_o2_factor = max(0.0, min(1.0, source.o2_frac / 0.2095))
        else:
            self._intake_source_temp_k = EXHAUST_TEMP_AMBIENT_K
            self.state.intake_o2_factor = 1.0
        air.step(dt)
        self.state.bay_air_temp_k = air.bay.temp_k
        self.state.bay_co_ppm = air.bay.co_ppm
        self.state.bay_o2_frac = air.bay.o2_frac
        if air.garage is not None:
            self.state.room_co_ppm = air.garage.co_ppm
            self.state.room_o2_frac = air.garage.o2_frac
            self.state.room_temp_k = air.garage.temp_k
            self._occupant.step(dt, air.garage.co_ppm)
        else:
            self.state.room_co_ppm = 0.0; self.state.room_o2_frac = 0.2095; self.state.room_temp_k = EXHAUST_TEMP_AMBIENT_K
            self._occupant.step(dt, 0.0)
        self.state.occupant_cohb_pct = self._occupant.cohb_pct

    @property
    def catalytic_converter(self):
        """The real cat this engine's declared exhaust carries (None when
        the layout has none or it has been pulled)."""
        from engines import CatalyticConverter
        eng = self.engine
        if not (eng.exhaust_system.layout_has_catalyst and eng.exhaust_system.catalyst_fitted):
            return None
        ci = eng.identity in getattr(engines, "_COMPRESSION_IGNITION", set())
        era = 1990 if "1990" in eng.identity else 2000
        return CatalyticConverter.for_engine(eng.displacement_l, ci, era_year=era)

    @property
    def occupant(self):
        return getattr(self, "_occupant", None)

    def _pneumatic_idle_assist_active(self) -> bool:
        """The real trip decision.

        This is an ANTI-STALL booster, and what it is looking for is a
        SAG -- an engine being dragged down by a load it cannot take --
        not simply a low rpm. A static "below this speed" window is
        wrong twice over: it is true at normal idle (so the reservoir
        empties for nothing), and it fires too LATE to help, because by
        the time a big diesel has actually reached the threshold the
        stall is already unavoidable. So there are two ways in, and one
        way out:

          TRIP if rpm has fallen below the trip point (which sits below
          the governed idle), OR if rpm is near idle and FALLING faster
          than the sag rate -- catching it on the way down, which is
          the whole point of the device.

          RELEASE once rpm has recovered above idle again, with real
          hysteresis, so it does not chatter on and off across a
          threshold.

        Plus the interlocks: the hardware fitted, the toggle on, the
        engine actually running (a stalled or cranking engine has no
        sag to catch, and dumping air into a dead one does nothing).
        """
        if self._pneumatic_idle_assist_area_m2 <= 0.0 or not self.pneumatic_idle_assist_enabled:
            self._idle_assist_latched = False
            return False
        rpm = self.state.rpm
        idle = max(self.engine.idle_rpm, 1.0)
        if self.state.stalled or rpm <= idle * 0.4:
            # cranking, dying or dead: not a sag
            self._idle_assist_latched = False
            return False
        if self._idle_assist_latched:
            # stay in until it has genuinely recovered
            if rpm >= idle * IDLE_ASSIST_RELEASE_FRAC_OF_IDLE:
                self._idle_assist_latched = False
        else:
            below = rpm < self._pneumatic_idle_assist_trip_rpm
            sagging = (rpm < idle * IDLE_ASSIST_SAG_WATCH_FRAC
                       and self._rpm_rate_per_s < -IDLE_ASSIST_SAG_RPM_PER_S)
            if below or sagging:
                self._idle_assist_latched = True
        return self._idle_assist_latched

    def _note_junction_ring(self, torque_nm: float, capacity_nm: float) -> None:
        """Watch the junction torque for the alternating-sign signature
        of an integrator past its stability limit."""
        hist = self._ring_hist
        hist.append(torque_nm)
        if len(hist) > RING_WINDOW:
            del hist[0]
        if len(hist) < RING_WINDOW or capacity_nm <= 0.0:
            return
        flips = sum(1 for a, b in zip(hist, hist[1:]) if a * b < 0.0)
        # Alternation alone is the signature; it does NOT have to be
        # pinned at capacity to be ringing. Requiring saturation as well
        # (the first version did) missed the commonest case: a junction
        # flipping between, say, -75 % and +100 % of capacity every
        # single step, which is just as unphysical and just as wrong.
        # The only thing to exclude is alternation in the noise, so the
        # magnitude has to be a real fraction of what the coupling can
        # hold.
        mean_mag = sum(abs(t) for t in hist) / len(hist)
        ringing = (flips >= (RING_WINDOW - 1) * RING_ALTERNATION_FRAC
                   and mean_mag >= capacity_nm * RING_SIGNIFICANT_FRAC)
        self.state.junction_ringing = bool(ringing)
        if ringing:
            self.state.junction_ring_events += 1
            # REACTIVE, TRANSIENT-ONLY SUBSTEPPING. The a-priori plan
            # (_clutch_substep_plan) sizes the step from the junction's
            # own natural frequency, which is the right first answer but
            # cannot know about everything else feeding the same shaft --
            # a governor cutting fuel, a rev limiter chopping in and out,
            # a load that has no equilibrium to settle into. When the
            # torque actually alternates anyway, escalate: subdivide this
            # junction harder for a short while and let it decay. The
            # escalation decays on its own, so it costs nothing once the
            # transient has passed.
            self._ring_escalation = min(RING_MAX_ESCALATION, max(2, self._ring_escalation * 2))
            self._ring_escalation_ticks = RING_ESCALATION_HOLD

    def _coupling_apply_pa(self) -> float:
        """Where this coupling's apply pressure comes from."""
        spec = getattr(self, "coupling_spec", None)
        if spec is None:
            return 0.0
        if spec.apply_source == "engine-hydraulics":
            # the PILOT circuit: a regulated supply live whenever the
            # pump turns, not the implement circuit's working pressure
            plant = getattr(self, "plant", None)
            h = getattr(plant, "hydraulics", None) if plant is not None else None
            return float(h.pilot_pressure_pa) if (h is not None and h.pilot_available) else 0.0
        return spec.apply_pressure_pa      # a rig pump, or a clamping spring

    def _coupled_junction_torque(self, dt: float, omega_drive: float, omega_load: float) -> float:
        """The torque the rig's coupling actually passes to the load.

        ONE capacity, owned by the coupling. For a FRICTION coupling the
        solver's own ClutchPort spring/damper computes the torque and
        saturates at exactly the capacity the coupling's apply pressure
        buys -- the port is the numerics, the coupling is the hardware,
        and neither caps what the other already capped. For a
        HYDRODYNAMIC coupling the port's tanh is simply the wrong law
        (a fluid coupling goes with the square of the speed difference),
        so the coupling computes the torque itself and the port is not
        used at all.
        """
        c = getattr(self, "coupling", None)
        if c is None:
            return self._brake_junction.step(dt, omega_drive, omega_load)
        c.engagement = self.clutch_frac
        # A SPEED-DEPENDENT COUPLING NEEDS THE SPEED. `Coupling.step` sets
        # this itself, but the friction path below never calls it -- the
        # port does the integrating and only `set_apply` is consulted --
        # so a centrifugal clutch would be asked for its capacity with a
        # drive speed of zero and answer "disengaged" forever.
        c.drive_omega_rad_s = abs(omega_drive)
        apply_pa = self._coupling_apply_pa()
        if c.kind in ("fluid-coupling", "torque-converter"):
            torque = c.step(dt, omega_drive, omega_load, 0.0, apply_pa)
        else:
            cap = max(c.set_apply(apply_pa), 1e-3)
            self._brake_junction.max_torque_nm = cap
            self._brake_junction.stiffness_nm_per_rad_s = cap / c.transition_slip_rad_s
            torque = self._brake_junction.step(dt, omega_drive, omega_load)
            c.observe(torque, omega_drive - omega_load)
            self._note_junction_ring(torque, cap)
        # a wet clutch runs in the machine's own oil, so its slip heat
        # goes into that oil -- assigned unconditionally, because a
        # clutch that has stopped slipping is making none
        if getattr(self.coupling_spec, "apply_source", "") == "engine-hydraulics":
            plant = getattr(self, "plant", None)
            h = getattr(plant, "hydraulics", None) if plant is not None else None
            if h is not None:
                h.coupling_heat_w = c.heat_w
        return torque

    def _build_brake_junction(self) -> ClutchPort:
        """The rig's coupling to the load.

        A test cell does not couple every engine through the same
        clutch: it uses what the engine is actually built to drive
        through (`couplings.recommended_for`), and on a heavy engine
        that is a wet multi-plate applied off the engine's OWN hydraulic
        reservoir -- which is why a big diesel can hold full brake
        torque here without the slip a dry plate would give. The
        ClutchPort below stays as the numerical spring/damper the solver
        integrates; the coupling decides how much torque it is allowed
        to pass and whether it is locked.
        """
        import couplings
        peak = max(self.engine.peak_torque_nm, 1.0)
        self.coupling_spec = couplings.recommended_for(self.engine)
        self.coupling = self.coupling_spec.build(peak)
        # NOT `or`. A centrifugal clutch at rest reports a capacity of
        # exactly 0.0 -- which is the truth, and is falsy, so `or` read it
        # as "no opinion" and substituted three times peak torque. The
        # clutch that cannot stall its engine was being built as one that
        # could hold anything.
        resting = self.coupling.torque_capacity_nm()
        if self.coupling.kind == "centrifugal":
            cap = max(self.coupling.rated_torque_nm, peak * 0.2)
        else:
            cap = max(resting if resting > 0.0 else peak * 3.0, peak * 0.2)
        # the port's "stiffness" is the SLOPE of the tanh that stands in
        # for friction, so it is capacity over the coupling's own real
        # transition width -- not a number picked off peak torque
        return ClutchPort(stiffness_nm_per_rad_s=cap / self.coupling.transition_slip_rad_s,
                          max_torque_nm=cap)

    def shift_up(self) -> None:
        self._request_quick_shift(self.gear_index + 1)

    def shift_down(self) -> None:
        self._request_quick_shift(self.gear_index - 1)

    def _request_quick_shift(self, target_gear: int) -> None:
        """One button, the whole sequence: a real driver doesn't ride
        the clutch through a shift, they're off it fast, swap gears,
        back on it fast. Clamps to the real range this engine's own
        transmission actually has (see engines.Transmission) -- can't
        request a gear that doesn't exist. Re-triggering mid-sequence
        (a fast double-tap) just retargets the in-flight shift rather
        than queuing or ignoring it, the same real feel as a quick
        driver correction."""
        trans = self.engine.transmission
        target_gear = max(-1, min(len(trans.gear_ratios), target_gear))
        if target_gear == self.gear_index and self._quick_shift_state is None:
            return
        self._quick_shift_target_gear = target_gear
        self._quick_shift_state = "clutch_out"

    def _update_quick_shift(self, dt: float) -> None:
        if self._quick_shift_state is None:
            return
        if self._quick_shift_state == "clutch_out":
            self.clutch_frac = max(0.0, self.clutch_frac - dt / SHIFT_CLUTCH_OUT_S)
            if self.clutch_frac <= 0.0:
                self.gear_index = self._quick_shift_target_gear
                self._quick_shift_state = "clutch_in"
        elif self._quick_shift_state == "clutch_in":
            self.clutch_frac = min(1.0, self.clutch_frac + dt / SHIFT_CLUTCH_IN_S)
            if self.clutch_frac >= 1.0:
                self._quick_shift_state = None

    def start_wot_dyno_pull(self) -> None:
        """One button: neutral -> settle -> first gear, then make a real
        quick-shift through every transmission ratio at WOT and run the final
        gear to redline. brake_load_nm goes to 0.0 for the whole pull -- a real
        inertia-dyno pull measures power from the drum's own known
        inertia accelerating (power = torque_at_drum * drum_omega, both
        already real/live via dyno_torque_nm/dyno_rpm), not against a
        resistive brake load, which is the same real technique a
        Dynojet-style chassis dyno actually uses. Captured at the
        TRANSMISSION EXIT (the dyno drum, post-clutch-and-gearing) --
        the same real quantities the DYNO dashboard line already
        surfaces, not the crank's own raw numbers. Ignored if a pull
        (or a quick-shift) is already in progress."""
        if self._dyno_pull_state is not None or self._quick_shift_state is not None:
            return
        self._dyno_pull_saved = (self.throttle, self.brake_load_nm, self.gear_index)
        self.throttle = 0.0
        self.brake_load_nm = 0.0
        self.gear_index = 0   # straight to neutral -- no clutch choreography needed to get there
        self.state.dyno_pull_peak_torque_nm = 0.0
        self.state.dyno_pull_peak_torque_rpm = 0.0
        self.state.dyno_pull_peak_power_kw = 0.0
        self.state.dyno_pull_peak_power_rpm = 0.0
        self._dyno_pull_timer_s = 0.0
        self._dyno_pull_torque_ema = 0.0
        self._dyno_pull_power_ema = 0.0
        self._dyno_pull_saved_drum_inertia = self._dyno_inertia_kg_m2
        self._dyno_inertia_kg_m2 = max(
            self._dyno_inertia_kg_m2, self.engine.peak_torque_nm * DYNO_PULL_INERTIA_KG_M2_PER_NM)
        self._dyno_pull_state = "settle"
        self.state.dyno_pull_state = "settle"

    def _update_dyno_pull(self, dt: float) -> None:
        self.state.dyno_pull_complete_flag = False
        if self._dyno_pull_state is None:
            return
        self._dyno_pull_timer_s += dt
        if self._dyno_pull_state == "settle":
            if self._dyno_pull_timer_s >= DYNO_PULL_SETTLE_S:
                self._request_quick_shift(1)
                self._dyno_pull_state = "shift"
                self.state.dyno_pull_state = "shift"
        elif self._dyno_pull_state == "shift":
            if self._quick_shift_state is None:
                self.throttle = 1.0
                self._dyno_pull_timer_s = 0.0
                self._dyno_pull_state = "pulling"
                self.state.dyno_pull_state = "pulling"
        elif self._dyno_pull_state == "pulling":
            # only track peaks once the clutch is genuinely LOCKED
            # (self._brake_junction.locked, the same real property the
            # dashboard's brake_clutch_locked already reads) -- right
            # after a shift re-engages, the clutch is still slipping and
            # its own real relative-speed-driven reaction torque can
            # transiently spike well past anything the engine's actual
            # combustion output ever produces (ClutchPort's max_torque_nm
            # ceiling, times the gear ratio); recording that spike as
            # "peak torque" would be capturing clutch-engagement
            # transient, not a real point on this engine's own power
            # curve.
            dyno_power_kw = self.state.dyno_torque_nm * self._load_omega / 1000.0
            smooth_alpha = min(1.0, dt / DYNO_TORQUE_SMOOTH_TAU_S)
            self._dyno_pull_torque_ema += (self.state.dyno_torque_nm - self._dyno_pull_torque_ema) * smooth_alpha
            self._dyno_pull_power_ema += (dyno_power_kw - self._dyno_pull_power_ema) * smooth_alpha
            # skip the brief post-shift clutch-engagement transient
            # window -- a real, genuine slip-torque spike right as the
            # clutch relocks, not a point on this engine's own steady
            # combustion torque curve. DYNO_PULL_SETTLE_S worth of real
            # time is plenty for that transient to fully decay.
            if self._dyno_pull_timer_s >= DYNO_PULL_SETTLE_S:
                if self._dyno_pull_torque_ema > self.state.dyno_pull_peak_torque_nm:
                    self.state.dyno_pull_peak_torque_nm = self._dyno_pull_torque_ema
                    self.state.dyno_pull_peak_torque_rpm = self.state.rpm
                if self._dyno_pull_power_ema > self.state.dyno_pull_peak_power_kw:
                    self.state.dyno_pull_peak_power_kw = self._dyno_pull_power_ema
                    self.state.dyno_pull_peak_power_rpm = self.state.rpm
            top_gear = len(self.engine.transmission.gear_ratios)
            target_fraction = (0.98 if self.gear_index >= top_gear
                               else DYNO_PULL_SHIFT_RPM_FRAC)
            redline_reached = self.state.rpm >= self.engine.redline_rpm * target_fraction
            timed_out = self._dyno_pull_timer_s >= DYNO_PULL_MAX_DURATION_S
            if redline_reached or timed_out:
                if self.gear_index < top_gear:
                    self.throttle = 0.15
                    self._request_quick_shift(self.gear_index + 1)
                    self._dyno_pull_state = "shift"
                    self.state.dyno_pull_state = "shift"
                    return
                _saved_throttle, saved_brake, _saved_gear = self._dyno_pull_saved or (0.0, 0.0, self.gear_index)
                self.throttle = 0.0
                self.brake_load_nm = saved_brake
                # a finished pull hands the engine back in NEUTRAL, the
                # way a cell operator lets it drop to idle unloaded --
                # not silently left in the pull's last gear against a
                # spinning drum
                self.gear_index = 0
                self._dyno_pull_saved = None
                if self._dyno_pull_saved_drum_inertia is not None:
                    self._dyno_inertia_kg_m2 = self._dyno_pull_saved_drum_inertia
                    self._dyno_pull_saved_drum_inertia = None
                self._dyno_pull_state = None
                self.state.dyno_pull_state = None
                self.state.dyno_pull_complete_flag = True

    def _idle_coupled_inertia_kg_m2(self) -> float:
        """What the ECU's neutral switch (and its calibration) tells it
        about the inertia its idle loop is actually governing: the crank
        alone in N, the crank plus the load reflected through the
        current ratio when a gear is engaged and the clutch is out.
        Feeds ecu.idle_pi_gains_frac's pole placement."""
        j = self.engine.inertia_kg_m2
        ratio = self._current_gear_ratio()
        if self.gear_index != 0 and ratio > 0.0 and self.clutch_frac > 0.5:
            j += self._dyno_inertia_kg_m2 / (ratio * ratio)
        return j

    def _current_gear_ratio(self) -> float:
        """engine-shaft revolutions per output-shaft revolution for
        whatever gear_index is currently selected, already folded
        together with the final drive -- 0.0 means no path to the load
        at all (neutral), matching a real neutral-clutch-in free-rev."""
        trans = self.engine.transmission
        if self.gear_index == 0:
            return 0.0
        if self.gear_index < 0:
            return trans.reverse_ratio * trans.final_drive_ratio
        idx = self.gear_index - 1
        if idx >= len(trans.gear_ratios):
            return 0.0
        return trans.gear_ratios[idx] * trans.final_drive_ratio

    def _recompute_firing_angles(self) -> None:
        arch = self.engine.architecture
        # the architecture's own firing schedule -- evenly spaced for an
        # ordinary crank, genuinely unequal for an odd-fire one
        slot_angles = arch.slot_angles_deg()
        # real per-cylinder swept volume, for the parametric volume-
        # pressure exchange each firing event draws against the real
        # intake-air circuit with (drivetrain_graph.draw_cylinder_charge)
        self._cylinder_volume_m3 = (self.engine.displacement_l / 1000.0) / max(arch.cylinders, 1)
        self.ignition.reset(len(self.engine.rev_limiter.stages))
        self._firing_angle_deg = {cyl: slot_angles[slot] for slot, cyl in enumerate(arch.firing_order)}
        self._slot_of_cyl = {cyl: slot for slot, cyl in enumerate(arch.firing_order)}
        self.state.slot_angles_deg = list(slot_angles)
        self.state.slot_records = [[0.0, 0.0, 0.0, 0.0] for _ in arch.firing_order]
        self._last_fire_total_deg = {cyl: -1e9 for cyl in arch.firing_order}
        self._last_strength = {cyl: 0.0 for cyl in arch.firing_order}
        self._knock_accum = {cyl: 0.0 for cyl in arch.firing_order}
        self._knocked_this_burn = {cyl: False for cyl in arch.firing_order}
        self._active_event = {}
        self._flame_radius_m = {}
        self._burned_frac = {}

    def reset_damage(self) -> None:
        """Put the machine back to undamaged, everywhere at once.

        Damage is not one list: it is punctures on the mesh pieces,
        emitters on the holes they made, missing parts, burst debris,
        fires, contamination, spilled contents and the fluid those
        spills took out of the real reservoirs. Undoing it in one place
        rather than several is the point -- a half-reset machine, still
        carrying an emitter for a hole that no longer exists, is worse
        than one that was never reset."""
        self.state.part_damage = {}
        self.state.absent_parts = set()
        self.state.coolant_lost_l = 0.0
        # NOTE: these are the fields, not the reports. `fouling()` and
        # `summary()` are DERIVED from `mix`; assigning over one of them
        # replaces a method with a dict and the next dashboard frame
        # dies calling it. The real state is what gets cleared here.
        self.hole_emitters.emitters = []
        self.hole_emitters.lost_l = {}
        self.hole_emitters.mix = {}
        self.hole_emitters.ingest_kg_s = {}
        self.bursts.bursts = []
        self.fires.fires = []
        self.fires.total_burned_kg = 0.0
        self.ordnance.charges = []
        self.ordnance.log = []
        self.fittings.ports = []
        self.damage_events = []
        self.cascade_log = []
        self._fuel_exposure_s = {}
        self._node_conditions = {}
        from node_effects import Effects
        self._effects = Effects()
        # the fluids themselves go back to full: a reset machine that is
        # still empty of oil has not been reset
        for c in self._drivetrain.fluid_circuits:
            c.fill_level_frac = 1.0
        cs = getattr(self, "_crankcase_state", None)
        if cs is not None:
            cs.oil_kg = cs.oil_capacity_kg
        p = getattr(self, "plant", None)
        if p is not None:
            p.loop.charge_frac = 1.0
            h = getattr(p, "hydraulics", None)
            if h is not None:
                h.oil_l = h.tank_capacity_l
                h.nitrogen_fill_frac = 1.0
        # and the splash emitters, which are hardware rather than damage
        self.hole_emitters.add_splash_from_graph(self._drivetrain.graph)
        self.hole_emitters.sync_open_ports(self._drivetrain.graph)
        register_graph_parts(self.state.part_damage, self._drivetrain.graph)

    def set_engine(self, engine: Engine) -> None:
        self.engine = engine
        self.state.wear = wear_module.new_wear_state(engine)
        self.state.part_damage = {}
        self.fuel_choice = None
        self._recompute_firing_angles()
        if engine.load_resistor_frac is not None:
            self.electrical_load_frac = engine.load_resistor_frac
        self.throttle = 0.0
        self.brake_load_nm = 0.0
        self.brake_target_rpm = None
        self.throttle_target_rpm = None
        self.dyno.reset()
        self.electrical = ElectricalNetwork(engine)
        self.starter = StartingSystems(engine)
        self._turbine = engine.turbine.build() if engine.turbine is not None else None
        self._atmospheric_cyl = engine.atmospheric.build() if engine.atmospheric is not None else None
        self._atmo_governor = engine.atmospheric_governor.build() if engine.atmospheric_governor is not None else None
        self._expander = engine.expander.build() if engine.expander is not None else None
        self._time_since_last_ignition_s = 0.0
        self.external_crank_torque_nm = 0.0
        self._crank_assist_nm = 0.0
        self._electric_fan_cmd = 0.0
        self._block_thermal_cache = None
        self.ecu.reset()
        self.standalone_idle.reset()
        self._quick_shift_state: str | None = None   # None | "clutch_out" | "clutch_in"
        self._quick_shift_target_gear = 1
        self._quick_shift_timer_s = 0.0
        self._last_dyno_torque_nm = 0.0
        self._dyno_pull_state: str | None = None   # None | "settle" | "shift" | "pulling"
        self._dyno_pull_timer_s = 0.0
        self._dyno_pull_saved: tuple[float, float, int] | None = None  # (throttle, brake_load_nm, gear_index)
        self._dyno_pull_torque_ema = 0.0
        self._dyno_pull_power_ema = 0.0
        self._dyno_pull_saved_drum_inertia: float | None = None
        self._time_since_start_s = 0.0
        self._load_omega = 0.0
        self._brake_junction = self._build_brake_junction()
        self._drivetrain = self._build_drivetrain()
        # a different engine is a different machine: the holes, missing
        # parts, fires and spills of the last one do not belong to it
        from plant import AuxiliaryPlantRuntime
        self.plant = AuxiliaryPlantRuntime.build(self.engine)
        self.air_vessels = self._build_air_vessels()
        self.automatic = self._build_automatic()
        self.equipment = None
        self.hydraulic_flow_frac = 0.0
        self.hydraulic_load_frac = 0.0
        self.reset_damage()
        self._waste_heat_kw = 0.0
        self._intake_supply_pressure_pa = 101_325.0
        self._intake_demand_kg_s = 0.0
        self._fuel_demand_kg_s = 0.0
        #: how full the cylinder actually gets, relative to the
        #: displacement demand at this MAP -- the breathing terms
        #: the demand omits. 1.0 until the first tick solves it.
        self._charge_fill_frac = 1.0
        self._fuel_cooling_kw = 0.0
        self._auxiliary_injection_cooling_kw = 0.0
        self._fuel_starvation_frac = 1.0
        self._exhaust_demand_kg_s = 0.0
        self._physics_accum_s = 0.0
        # a real engine swap doesn't hand you back a warm, already-
        # idling machine -- it's cold, and getting it running takes
        # this engine's own real starting system (starter.py), the same
        # way switching to it the first time would. self.start() (the
        # instant teleport-to-idle) stays available as an explicit
        # warm-start for regressions/tooling, just not the default here.
        self.stop()

    def stop(self) -> None:
        self._omega = 0.0
        self.starter.reset()
        self._crank_assist_nm = 0.0
        self.state.cranking = False
        self._map_frac_na = idle_map_base_frac(self.engine)  # ECU's per-engine idle calibration
        self._fuel_rack_frac = idle_fuel_base_frac(self.engine)  # ECU's per-engine idle rack calibration
        self.state.rpm = 0.0
        self.state.stalled = False
        self.state.manifold_pressure_frac = MAP_IDLE_FRAC_DEFAULT
        self.state.battery_voltage = self.electrical.nominal_voltage_v
        if self._turbine is not None:
            self._turbine.reset()
        if self._atmospheric_cyl is not None:
            # a real dead-cold reset: piston back at ignition height, at
            # rest, awaiting a fresh charge -- rebuilt from the engine's
            # own declared spec rather than __post_init__ (which only
            # re-derives x_m/v_m_s/working_pressure_pa, not phase/
            # locked/cycle_count, so it can't fully clear a cylinder
            # that stopped mid-stroke)
            self._atmospheric_cyl = self.engine.atmospheric.build()
        if self._atmo_governor is not None:
            self._atmo_governor.reset()
        if self._expander is not None:
            # a cold, undamaged bank -- the same "swap the wrecked part"
            # convention the Otto-Langen cylinder uses on stop()
            self._expander = self.engine.expander.build()
        self._time_since_last_ignition_s = 0.0
        self.state.real_fire_hz = 0.0

    # -- the declared span: what crosses the boundary ------------------
    def sync_state_span(self):
        """Pack the whole live state into the declared flat span."""
        from engine_abi import engine_graph_abi
        from engine_state import pack
        self.state_span = pack(self, engine_graph_abi(self.engine),
                               self.state_span)
        return self.state_span

    def load_state_span(self, span=None) -> None:
        """Write a declared flat span back into the live state."""
        from engine_abi import engine_graph_abi
        from engine_state import unpack
        source = self.state_span if span is None else span
        if source is not None:
            unpack(self, engine_graph_abi(self.engine), source)

    def state_span_length(self) -> int:
        from engine_abi import engine_graph_abi
        return int(engine_graph_abi(self.engine).state_stride)

    def start(self) -> None:
        """!!! TEST-CELL WARM START -- NOT A REAL START. !!!

        THIS TELEPORTS THE ENGINE TO IDLE. It does not crank, it does
        not draw a single amp off the battery, it does not care whether
        there is fuel in the tank or a starter bolted to the block, and
        it cannot fail. A piston engine goes from stopped to idling in
        one assignment.

        IT EXISTS ONLY so regressions can reach a running engine in one
        line without paying for a start sequence they are not testing.
        Anything that is ABOUT starting, or that reports on it, or that
        a player would experience, must use engage_starter() instead --
        that is the real path, through starter.py, with seven real
        starting systems behind it (electric, cart, air, wound-flywheel
        inertia, recoil rope, hand crank, flywheel bar), real bus sag,
        real receiver air, real duty limits, a real catch, and a real
        ring gear you can grind to pieces by keying a running engine.

        Using this where a real start belongs does not raise, warn, or
        look wrong in any output. It just quietly makes every starting
        system in the project irrelevant, and makes a dead battery and a
        full one indistinguishable. It has already caused one wrong
        conclusion in this repo -- a profiler reported there was
        "nothing to measure" at startup, because it called this.
        """
        self._omega = self.engine.idle_rpm * RPM_TO_RAD_S
        self.state.rpm = self.engine.idle_rpm
        self.state.stalled = False
        self.state.ignition_cut = False
        self._time_since_start_s = 0.0
        if self._turbine is not None:
            # A TURBINE DOES NOT TELEPORT TO IDLE. Every other engine
            # kind here gets an "already running, warm" start, which is a
            # fair simplification for a piston engine that makes torque
            # on its first firing stroke. A gas turbine's start IS its
            # characteristic behaviour: a starter motors the spool up to
            # light-off speed with the combustor dark, the igniters fire,
            # and only then does the engine accelerate itself to idle --
            # tens of seconds, all of it audible and none of it optional.
            # Teleporting past that removed the one thing that makes a
            # turbine feel like a turbine, and left light_off_spool_time_s
            # declared and unused on every turbine in the catalogue.
            #
            # So: start from rest and let the existing machinery run the
            # sequence. `step` already refuses to fire below
            # LIGHT_OFF_OMEGA_FRAC and already motors the rotor from the
            # engine's own declared starting_systems; nothing new is
            # needed except not skipping it.
            # A TURBINE USES ITS REAL STARTER. Every other engine kind
            # here gets an "already running, warm" teleport, which is a
            # fair simplification for a piston engine that makes torque on
            # its first firing stroke. A gas turbine's start IS its
            # characteristic behaviour, and the machinery for it was
            # already all present and simply not reached: `step` refuses
            # to fire below LIGHT_OFF_OMEGA_FRAC, `_step_turbine` already
            # feeds `self._crank_assist_nm * reduction` to the gas
            # generator, and `engage_starter` already drives that from the
            # engine's own declared starting_systems. The only thing
            # missing was that `start()` skipped past all of it, which
            # also left light_off_spool_time_s declared and unused.
            #
            # So: from rest, with the starter engaged, and let the real
            # sequence run -- motor the spool dark, light off, and
            # accelerate on its own combustion up to idle.
            self._turbine.omega_rad_s = 0.0
            self._omega = 0.0
            self.state.rpm = 0.0
            self.engage_starter()
        if self._atmospheric_cyl is not None:
            # the free piston's own natural resting state -- parked at
            # ignition height, at rest, ready for the next real charge;
            # the flywheel is simply spun up to this catalogue entry's
            # governed idle speed directly, same "already running, warm"
            # convention as every other engine kind here. Rebuilt fresh
            # (see stop()) rather than __post_init__, which can't fully
            # clear a cylinder that was mid-stroke.
            self._atmospheric_cyl = self.engine.atmospheric.build()
        # the dyno's clutch is always engaged in this sim (no separate
        # clutch pedal) -- starting the load shaft at rest while the
        # crank starts at idle speed is a bogus, unrequested "dump the
        # clutch onto a dead-stopped drum" transient, not a real dyno
        # procedure. Start matched to idle instead; real slip still
        # happens the moment throttle or brake_load_nm actually changes.
        self._load_omega = self._omega
        self._drivetrain = self._build_drivetrain()
        self._waste_heat_kw = 0.0
        self._intake_supply_pressure_pa = 101_325.0
        self._intake_demand_kg_s = 0.0
        self._fuel_demand_kg_s = 0.0
        #: how full the cylinder actually gets, relative to the
        #: displacement demand at this MAP -- the breathing terms
        #: the demand omits. 1.0 until the first tick solves it.
        self._charge_fill_frac = 1.0
        self._fuel_cooling_kw = 0.0
        self._auxiliary_injection_cooling_kw = 0.0
        self._fuel_starvation_frac = 1.0
        self._exhaust_demand_kg_s = 0.0
        self._physics_accum_s = 0.0

    def engage_starter(self, kind: str | None = None) -> None:
        """The operator's start action, whatever this engine's real
        starting system is (starter.py; `kind` picks one when the engine
        carries several): the crank turns over from wherever it is under
        that system's real torque, draws its real energy, and the starter
        drops out when the engine catches. The old start() teleport-to-
        idle remains as the test cell's warm start for regressions."""
        self.state.stalled = False
        self.state.ignition_cut = False
        if self._omega <= 0.0:
            self._time_since_start_s = 0.0
        self.starter.engage(kind)

    def restart_if_stalled(self) -> None:
        if self.state.stalled:
            self.engage_starter()
        # real key-back-to-"run" behavior: re-enables the ignition
        # system regardless of whether a full start() was needed, e.g.
        # the engine is still coasting down from a kill and catches
        # again rather than requiring the starter
        self.state.ignition_cut = False

    @property
    def rpm(self) -> float:
        return self.state.rpm

    @property
    def stalled(self) -> bool:
        return self.state.stalled

    @stalled.setter
    def stalled(self, value: bool) -> None:
        self.state.stalled = value
        if value:
            self._omega = 0.0
            self.state.rpm = 0.0

    @property
    def current_torque_nm(self) -> float:
        return self.state.current_torque_nm

    @property
    def power_kw(self) -> float:
        return self.state.power_kw

    def _combustion_efficiency_now(self) -> float:
        """How completely the charge burns AT THIS MIXTURE.

        The engine's declared combustion_efficiency is a calibrated
        figure for that engine at stoichiometric, and it stays exactly
        that: this returns it unchanged at phi = 1.0, so no engine in
        the catalogue moves at its own calibration point.

        What it adds is the MIXTURE DEPENDENCE, which the sim had none
        of -- combustion efficiency was a constant no matter how rich
        the engine ran. Past stoichiometric there is not enough oxygen
        to burn all the fuel and at most 1/phi of it can react, which is
        real stoichiometry and the dominant term by far: at phi 1.25 a
        fifth of the fuel has nothing to react with. That is also why
        enrichment makes MORE power (it burns all the available AIR,
        which is the actually limited quantity) while making specific
        consumption worse, and the sim could not previously express
        either half of that.

        combustion_efficiency.completeness supplies the ratio; see its
        module docstring for the crevice and wall-quench terms, and for
        why the derived absolute value is NOT substituted here -- it
        disagrees with the declaration by 78% on the hit-and-miss, whose
        real losses are mixture preparation and are not modelled."""
        declared = float(getattr(self.engine, "combustion_efficiency", 0.85) or 0.85)
        try:
            import combustion_efficiency as _ce
            phi = float(getattr(self.state, "mixture_phi", 1.0) or 1.0)
            here = _ce.for_engine(self.engine, phi).efficiency
            stoich = _ce.for_engine(self.engine, 1.0).efficiency
            if stoich > 1e-9:
                return declared * max(0.2, min(1.0, here / stoich))
        except Exception:
            pass
        return declared

    def _fuel_circuit(self):
        return next((c for c in self._drivetrain.fluid_circuits
                     if c.kind_class == "high-pressure-liquid-supply" and any("fuel" in nid for nid in c.nodes)),
                    None)

    def _step_fuel_network(self, dt: float, demand_kg_s: float) -> SupplyTick | None:
        """Advance the declared fuel network one tick (None if this
        engine has none) and push its live flow ceiling onto the real
        fuel circuit before the drivetrain steps that circuit's
        reservoir. Everything else the tick carries (production,
        composition, lock-off, coolant draw) is handed straight into
        the drivetrain step by the caller."""
        rt = self._fuel_network
        if rt is None:
            return None
        circuit = self._fuel_circuit()
        fill = circuit.fill_level_frac if circuit is not None else 1.0
        tick = rt.step(dt, self.state.rpm, demand_kg_s, fill, engine_running=not self.state.stalled,
                       coolant_temp_k=self.state.coolant_temp_k, fuel_temp_k=self.state.fuel_temp_k)
        if circuit is not None:
            circuit.flow_capacity_kg_s = tick.flow_capacity_kg_s
        self.state.fuel_supply_pressure_pa = tick.supply_pressure_pa
        self.state.fuel_network_warnings = tick.warnings
        self.state.boiler_water_frac = tick.boiler_water_frac
        self.state.boiler_failed = tick.boiler_failed
        self._last_net_tick = tick
        return tick

    def _octane_factor(self) -> float:
        fuel = self.fuel_choice or self.engine.preferred_fuel_profile
        return self.engine.fuel_compatibility.get(fuel, 1.0)

    # --- block metal heat equation, parametric ---------------------------
    BLOCK_MASS_FRACTION_OF_ENGINE = 0.45      # real: block + heads are ~45% of a dressed engine's mass
    BLOCK_METAL_SPECIFIC_HEAT_J_KGK = 460.0   # cast iron (production's FLUID_LINE_MATERIALS "cast-iron-casting")
    BLOCK_METAL_CONDUCTIVITY_W_MK = 50.0      # cast iron
    BLOCK_CONDUCTION_AREA_FRAC = 0.30         # solid metal fraction of the block cross-section (block_dynamics uses the same)
    JACKET_FILM_W_M2K = 1500.0                # forced water in a jacket (real 1000-3000)
    FINNED_AIR_FILM_W_M2K = 60.0              # air-cooled finned cylinder to bay air, fin-area-weighted

    def _block_thermal_params(self) -> tuple[float, float, float, float]:
        """(wetted area per cylinder m^2, film coefficient W/m^2K, metal
        heat capacity per cylinder J/K, bay-to-bay conductance W/K) --
        all from this engine's own bore/stroke, mass, block section and
        bore spacing, so a 1.5 L four and a 0.96 m-bore marine cylinder
        each get their real numbers instead of one car-scale constant
        (the old fixed 300 W/m^2K over a 4*V^(2/3) area left the marine
        block 250 K above its coolant)."""
        if self._block_thermal_cache is not None:
            return self._block_thermal_cache
        eng = self.engine
        arch = eng.architecture
        n_cyl = max(arch.cylinders, 1)
        vd_per_cyl_m3 = eng.displacement_l / 1000.0 / n_cyl
        if arch.bore_m > 0.0 and arch.stroke_m > 0.0:
            # real liner wall plus the head's fire-deck face
            wetted_m2 = math.pi * arch.bore_m * arch.stroke_m + math.pi * arch.bore_m ** 2 / 4.0
        else:
            wetted_m2 = 4.0 * vd_per_cyl_m3 ** (2.0 / 3.0)
        film = self.JACKET_FILM_W_M2K if eng.accessories.water_pump else self.FINNED_AIR_FILM_W_M2K
        heat_capacity = (eng.mass_kg * self.BLOCK_MASS_FRACTION_OF_ENGINE / n_cyl) * self.BLOCK_METAL_SPECIFIC_HEAT_J_KGK
        # k*A/L between neighbouring bays: A = solid fraction of the block's
        # real cross-section, L = real bore spacing (distinct crank stations)
        half_yz = engine_geometry.block_half_yz_m(eng)
        section_m2 = self.BLOCK_CONDUCTION_AREA_FRAC * (2.0 * half_yz) ** 2
        xs = sorted({round(site.position[0], 4) for site in engine_geometry.cylinder_sites(eng)})
        spacing = min((xs[i + 1] - xs[i] for i in range(len(xs) - 1)), default=0.0)
        lateral = self.BLOCK_METAL_CONDUCTIVITY_W_MK * section_m2 / spacing if spacing > 1e-6 else 0.0
        self._block_thermal_cache = (wetted_m2, film, heat_capacity, lateral)
        return self._block_thermal_cache

    def _update_dyno_controllers(self, dt: float) -> None:
        """The test cell's brake/throttle rpm-target loops live in
        dyno_controller.DynoController (not vehicle hardware) -- this
        just hands them the cell's readings and applies their commands."""
        eng = self.engine
        self.brake_load_nm = self.dyno.brake_load_nm(eng, self.brake_target_rpm, self.state.rpm,
                                                     self.brake_load_nm, dt)
        self.throttle = self.dyno.throttle(eng, self.throttle_target_rpm, self.state.rpm, self.throttle, dt)

    # the idle governors (_idle_map_target / _idle_fuel_quantity /
    # _idle_load_feedforward_frac) moved to ecu.EngineControlUnit --
    # this sim delegates to self.ecu, it no longer holds that program

    def _step_forced_induction(self, dt: float, rpm_frac: float, dthrottle_dt: float) -> None:
        fi = self.engine.forced_induction
        st = self.state
        st.wastegate_flutter = False

        if fi.kind == "none":
            st.turbo_spool_frac = 0.0
            st.boost_frac = 0.0
            st.surge_flag = False
            return

        if fi.kind == "supercharger":
            # a genuine belt drive off the crank, real stiffness/damping/
            # backlash -- stepped as part of the drivetrain graph solver
            # (drivetrain_graph.py) just above, not an instant rpm formula.
            # Boost lags the belt's own dynamics instead of tracking
            # throttle/rpm the instant they change.
            # A positive-displacement blower is a pump: it delivers its
            # own swept volume per rotor rev at its INLET density, the
            # engine swallows its own swept volume per two crank revs at
            # MANIFOLD density, and mass conservation across the two
            # gives the steady pressure ratio directly:
            #   P_man / P_inlet = (D_blower * n_rotor) / (D_engine/2 * n_crank)
            # -- a pure ratio of displacements times the ACTUAL rotor/
            # crank speed ratio the belt is really delivering. rpm cancels
            # (both sides scale with crank speed), which is exactly why a
            # real roots/screw blower makes roughly the same boost across
            # the rev range, and the throttle plays NO part in it: the
            # declared max_boost_frac IS the displacement ratio (blower
            # size + belt overdrive) expressed as PR-1 at nominal belt
            # ratio. The throttle only sets the blower's inlet pressure
            # (draw-through, butterflies on the injector hat above the
            # blower) -- applied multiplicatively in step(), not here.
            target_blower_omega = self._omega * fi.belt_ratio
            blower_omega = self._drivetrain.omega.get("supercharger_rotor", 0.0)
            blower_speed_frac = max(0.0, min(1.5, blower_omega / max(target_blower_omega, 1.0)))
            st.turbo_spool_frac = min(1.0, blower_speed_frac)
            if fi.blower_type == "centrifugal":
                # an impeller's pressure rise goes with tip speed squared:
                # boost climbs with the square of the ratio to its rated
                # speed (rated = crank at power_peak_rpm through the gears)
                rated_omega = max(self.engine.power_peak_rpm, 1.0) * RPM_TO_RAD_S * fi.belt_ratio
                rated_frac = max(0.0, min(1.5, blower_omega / rated_omega))
                st.boost_frac = fi.max_boost_frac * rated_frac * rated_frac
                # a centrifugal stage surges like a turbo's when the
                # throttle shuts on it at speed
                st.surge_flag = self.throttle < 0.08 and rated_frac > 0.6
            else:
                st.boost_frac = fi.max_boost_frac * blower_speed_frac
                st.surge_flag = False
            self._surge_timer = 0.0
        else:  # turbo
            # Real exhaust gas energy: a turbine extracts real work from
            # mass flow times gas enthalpy (proportional to how far above
            # ambient the exhaust actually is), a restriction of its own
            # that exists at every flow rate -- NOT the same thing as the
            # exhaust pipe/port's own choked-flow backpressure (that's a
            # separate real limit, rarely exceeded below redline, and
            # stays as the real cap on exhaust_pressure_frac itself).
            # Calibrated against this same engine's own real redline
            # exhaust flow at a real plausible hot-exhaust reference
            # temperature, not an arbitrary constant.
            exhaust_demand_kg_s = self._exhaust_demand_kg_s
            exhaust_temp_above_ambient_k = max(0.0, self.state.exhaust_temp_k - EXHAUST_TEMP_AMBIENT_K)
            displacement_m3 = self.engine.displacement_l / 1000.0
            reference_exhaust_flow_kg_s = displacement_m3 * (self.engine.redline_rpm / 120.0) * 1.2 * 1.068
            reference_gas_energy = max(1e-6, reference_exhaust_flow_kg_s * 900.0)
            gas_energy = exhaust_demand_kg_s * exhaust_temp_above_ambient_k
            flow_energy = min(1.0, gas_energy / reference_gas_energy)
            # still blended with throttle: that flow needs combustion
            # heat behind it to be worth anything, which a closed
            # throttle genuinely starves regardless of pumped volume
            exhaust_energy = max(0.0, flow_energy * (0.25 + 0.75 * self.throttle))
            target_spool = min(1.0, exhaust_energy)
            st.turbo_spool_frac += (target_spool - st.turbo_spool_frac) * min(1.0, dt / max(fi.spool_tau_s, 1e-3))
            st.turbo_spool_frac = max(0.0, min(1.0, st.turbo_spool_frac))

            effective_spool = st.turbo_spool_frac
            if effective_spool > fi.wastegate_frac:
                st.wastegate_flutter = True
                effective_spool = fi.wastegate_frac + (effective_spool - fi.wastegate_frac) * 0.15
            st.boost_frac = fi.max_boost_frac * effective_spool

            # compressor surge: throttle slammed shut while the turbo is still
            # spinning fast -- flow demand collapses faster than shaft speed
            if dthrottle_dt < ANTI_LAG_LIFT_THRESHOLD and st.boost_frac > SURGE_BOOST_THRESHOLD:
                self._surge_timer = SURGE_DECAY_S
            self._surge_timer = max(0.0, self._surge_timer - dt)
            st.surge_flag = self._surge_timer > 0.0

            self._antilag_cooldown = max(0.0, self._antilag_cooldown - dt)
            if (self.anti_lag_enabled and fi.anti_lag_capable and self._antilag_cooldown <= 0.0
                    and dthrottle_dt < ANTI_LAG_LIFT_THRESHOLD and st.boost_frac > 0.15):
                self._antilag_cooldown = ANTI_LAG_COOLDOWN_S
                for _ in range(self._rng.randint(2, 4)):
                    self.pending_backfires.append(BackfireEvent(kind="planned", strength=0.6 + 0.4 * st.boost_frac))
        # boost is folded into manifold_pressure_frac by the caller (step()),
        # added fresh each step on top of the lagged NA baseline -- this is
        # what raises knock risk for forced-induction engines, since knock
        # risk already scales with manifold_pressure_frac

    def step(self, dt: float) -> None:
        """Advance the sim by dt, real wall-clock seconds -- via a fixed-
        timestep accumulator, the standard real-time-sim answer to
        "real wall-clock dt is never actually stable": raw frame time
        (GC pauses, OS scheduling, the audio synth thread contending for
        the GIL, a keyboard poll, anything) is never fed to the physics
        directly. It's only ever added to an accumulator; the physics
        itself always advances in identical, precomputed FIXED_PHYSICS_DT_S
        chunks, however many of them the accumulator can currently
        afford, with the leftover remainder carried over to next call --
        real elapsed time is honestly consumed, never dropped, but the
        physics itself never sees a variable, unpredictable dt and is
        fully deterministic given the same sequence of throttle/brake
        inputs, independent of real-time jitter. The accumulator itself
        is capped (MAX_CATCHUP_STEPS) so a genuinely long real stall (the
        custom-engine builder's blocking console prompt, say) catches up
        over a bounded number of steps instead of replaying minutes of
        physics in one burst."""
        self._physics_accum_s = min(
            self._physics_accum_s + dt, FIXED_PHYSICS_DT_S * MAX_CATCHUP_STEPS)
        while self._physics_accum_s >= FIXED_PHYSICS_DT_S:
            _rpm_before = self.state.rpm
            self._step_once(FIXED_PHYSICS_DT_S)
            # the real rate of change of rpm, smoothed just enough to be
            # a usable signal -- what the anti-stall sag detector reads
            raw = (self.state.rpm - _rpm_before) / FIXED_PHYSICS_DT_S
            self._rpm_rate_per_s += (raw - self._rpm_rate_per_s) * min(1.0, FIXED_PHYSICS_DT_S / 0.08)
            self._physics_accum_s -= FIXED_PHYSICS_DT_S
            if self.hole_emitters.emitters:
                self._resolve_spray_targets()
                self.hole_emitters.step(FIXED_PHYSICS_DT_S, self._drivetrain.fluid_circuits, self.state.exhaust_temp_k,
                                        crank_omega_rad_s=self._omega)
                self._check_burst_triggers(FIXED_PHYSICS_DT_S)
            if self.bursts.bursts:
                self.bursts.step(FIXED_PHYSICS_DT_S, self._drivetrain.graph)
            if self.ordnance.charges:
                self.ordnance.step(FIXED_PHYSICS_DT_S, self)
            self._step_equipment(FIXED_PHYSICS_DT_S)
            self._step_air_vessels(FIXED_PHYSICS_DT_S)
            if self.automatic is not None:
                self._step_automatic(FIXED_PHYSICS_DT_S)
            if self.plant is not None:
                self._plant_leak_losses(FIXED_PHYSICS_DT_S)
                self._step_plant(FIXED_PHYSICS_DT_S)
            if self.fires.fires:
                self._step_fire(FIXED_PHYSICS_DT_S)
            if self.fittings.ports:
                self._step_fittings(FIXED_PHYSICS_DT_S)
            self._step_oil_pickup()
            self._effects_tick += 1
            if (self.hole_emitters.emitters or self.state.absent_parts or any(
                    st.impacts for st in self.state.part_damage.values())) and self._effects_tick % EFFECTS_EVERY_TICKS == 0:
                self._apply_node_effects()

    def _step_once(self, dt: float) -> None:
        eng = self.engine
        if self.state.engine_dead:
            # the crank is seized/free of its block: no torque, a hard decel
            self._omega *= math.exp(-dt * 6.0)
            if self._omega < 0.5:
                self._omega = 0.0
            self.state.rpm = self._omega / RPM_TO_RAD_S
            self.state.current_torque_nm = 0.0
            self.state.power_kw = 0.0
            self.state.ignition_cut = True
            return
        if (self.state.stalled and not self.starter.engaged) or dt <= 0.0:
            self.state.rpm = 0.0
            self.state.current_torque_nm = 0.0
            self.state.power_kw = 0.0
            self.state.cranking = False
            return
        # the starting system's real torque on the crank this tick, and
        # what it consumed -- one-tick-lagged readings (bus voltage,
        # receiver pressure) like every other cross-system value here
        out = getattr(self, "_drivetrain_out", {})
        # before the first solver call the receiver is whatever it was
        # built as (full) -- not "empty"
        starter_reading = self.starter.step(
            dt, self._omega, self.state.rpm, self.state.battery_voltage,
            out.get("pneumatic_reserve_fill_frac", 1.0),
            out.get("pneumatic_reserve_pressure_pa", PNEUMATIC_RESERVE_PRESSURE_PA))
        self._crank_assist_nm = starter_reading.assist_torque_nm + self.external_crank_torque_nm
        self.state.cranking = self.starter.engaged
        self.state.starter_note = starter_reading.note
        self.state.starter_current_a = starter_reading.bus_current_a

        arch = eng.architecture
        if not arch.cylinders:
            if eng.kind == "turbine":
                self._step_turbine(dt)
            elif eng.kind == "atmospheric":
                self._step_atmospheric(dt)
            elif eng.kind == "expander":
                self._step_expander(dt)
            else:
                self._step_electric(dt)
            return

        self._update_dyno_controllers(dt)
        self._update_quick_shift(dt)
        self._update_dyno_pull(dt)
        self._time_since_start_s += dt
        rpm_frac = max(self.state.rpm, 0.0) / max(eng.redline_rpm, 1.0)
        self.state.throttle_plate_angle_deg = throttle_plate_angle_deg(self.throttle)
        if self.throttle < 0.08 and not eng.compression_ignition:
            # the real PI idle-air-control governor -- whichever real
            # device this engine's own build actually has (see
            # engines.Engine.idle_control_device): the ECU by default,
            # or a real standalone aftermarket box on an engine that
            # doesn't trust/have ECU-governed idle
            if eng.idle_control_device == "standalone-pi-controller":
                self.ecu.release_map_governor()
                map_target = self.standalone_idle.idle_map_target(eng, self.state.rpm, dt)
            else:
                self.standalone_idle.reset()
                map_target = self.ecu.idle_map_target(
                    eng, self.state.rpm,
                    self.state.ac_compressor_load_w
                    + self.electrical.reading.alternator_shaft_load_w
                    + self.known_accessory_shaft_load_w, dt,
                    coupled_inertia_kg_m2=self._idle_coupled_inertia_kg_m2())
        else:
            self.ecu.release_map_governor()
            self.standalone_idle.reset()
            map_target = self.map_target_policy(self.throttle, self.state.rpm, eng.idle_rpm, eng,
                                                intake_demand_kg_s=self._intake_demand_kg_s)
        if eng.governor_mode == "hit_and_miss":
            # no throttle plate at all -- the mixer/carburetor setting is
            # fixed, wide open; a flyball governor decides per-revolution
            # whether to fire at all, not how much charge to admit when
            # it does. Pedal input is ignored on purpose.
            map_target = 0.92
        carb = eng.carburetor
        if carb.has_choke and self._time_since_start_s < carb.choke_warmup_s:
            time_frac = 1.0 - self._time_since_start_s / carb.choke_warmup_s
            # a real automatic choke's fast-idle cam backs off as rpm
            # actually rises, not just on a timer -- otherwise an engine
            # that revs up quickly keeps getting choke enrichment it no
            # longer needs, which is exactly what feeds rpm climbing
            # further, which feeds more time before the timer alone would
            # have backed off: a real runaway-idle failure mode, not a
            # cosmetic one
            rpm_excess = max(0.0, (self.state.rpm - eng.idle_rpm) / max(eng.idle_rpm, 1.0))
            rpm_release = max(0.0, 1.0 - rpm_excess)
            choke_frac = time_frac * rpm_release
            map_target = min(1.0, map_target + carb.choke_max_enrichment * choke_frac)
        self._map_frac_na += (map_target - self._map_frac_na) * min(1.0, dt / eng.intake_system.map_tau_s)

        backfires_before = len(self.pending_backfires)
        dthrottle_dt = (self.throttle - self._prev_throttle) / max(dt, 1e-6)
        # the real internal drivetrain solver: camshaft+bearings+timing
        # drive, no-belt accessory takeoff+alternator CVT, and any
        # supercharger belt -- stepped once here so its outputs (crank
        # reaction torque, cam phase lag) are ready before combustion
        # the dyno absorber lives in this same graph (drivetrain_graph.py)
        # as a real harness part now, but its actual coupling physics
        # still run in the finer-grained combustion substep loop below
        # (it needs to react to individual firing pulses, not just the
        # per-dt-call rate this solver runs at) -- so both dyno edges are
        # skipped here to avoid double-applying the reaction
        net_tick = self._step_fuel_network(dt, self._fuel_demand_kg_s)
        self._drivetrain_out = self._drivetrain.step(
            dt, self._omega, self.electrical.reading.alternator_shaft_load_w,
            # a vaporizer's real draw on the coolant comes straight off
            # the heat the jacket would otherwise have to reject
            waste_heat_kw=self._waste_heat_kw - (net_tick.coolant_heat_draw_w / 1000.0 if net_tick else 0.0),
            fuel_production_kg_s=net_tick.production_kg_s if net_tick else 0.0,
            fuel_production_composition_frac=net_tick.production_composition_frac if net_tick else 1.0,
            fuel_valve_open=net_tick.valve_open if net_tick else True,
            electric_fan_on_frac=self._electric_fan_cmd,
            nitrous_active=self.nitrous_active and eng.has_nitrous,
            auxiliary_injection_active=self.auxiliary_injection_active and eng.has_auxiliary_injection,
            # last tick's own solved MAP -- the same one-tick lag every
            # other cross-system reading in this loop already uses; a
            # real progressive WMI controller reads this off the
            # engine's own boost signal
            regulator_map_frac=self.state.manifold_pressure_frac,
            intake_demand_kg_s=self._intake_demand_kg_s,
            intake_supply_pressure_pa=self._intake_supply_pressure_pa,
            exhaust_demand_kg_s=self._exhaust_demand_kg_s,
            fuel_demand_kg_s=self._fuel_demand_kg_s,
            exhaust_brake_frac=EXHAUST_BRAKE_MAX_RESTRICTION_FRAC if self.engine_brake_enabled else 0.0,
            exhaust_system_restriction_frac=eng.exhaust_system.static_backpressure_frac * self._effects.exhaust_restriction_factor,
            # a fuel heater (SVO conversion) is the same real "condition
            # the line's fuel toward a target" mechanism a drag cooler is
            fuel_cooler_target_k=(net_tick.fuel_conditioning_target_k
                                  if net_tick is not None and net_tick.fuel_conditioning_target_k is not None
                                  else eng.fuel_delivery.cooler_target_temp_k),
            # a real magnetic clutch coil, either energized or not -- its
            # own tanh saturation (drivetrain_graph.py's friction-clutch-
            # shaft dispatch, the same real mechanism the dyno junction
            # and the driver's transmission clutch already use) is what
            # gives smooth, self-limiting engagement, not a hand-rolled
            # ramp layered on top of the wrong coupling type
            # the AC compressor turns when the cabin asks for it OR the
            # auxiliary plant's own chillers do -- it is one machine
            # driving all three evaporators, so anything calling on the
            # loop is what engages the clutch
            ac_active=((self.ac_enabled or self._plant_wants_cooling())
                       and eng.accessories.air_conditioning),
            disabled_edges=frozenset({"dyno_friction_clutch", "dyno_roller_contact"}),
            starting_air_kg_s=self.starter.reading.air_kg_s,
            pneumatic_idle_assist_active=self._pneumatic_idle_assist_active(),
            # a fouled air system does not pass what a clean one does:
            # sludge in the ports and stuck spools are a real restriction
            # on every pneumatic consumer, the idle-assist dump included
            pneumatic_idle_assist_port_area_m2=(self._pneumatic_idle_assist_area_m2
                                                * self._effects.air_flow_factor))
        self._step_forced_induction(dt, rpm_frac, dthrottle_dt)
        self._prev_throttle = self.throttle
        # real nitrous effect: extra oxidizer (the wet kit's fuel
        # solenoid keeps the mixture correct as it adds it) shows up as
        # more effective charge density, the same MAP-additive path
        # boost already uses -- scaled off the real delivered flow the
        # fluid graph's bottle circuit just computed, not a flat number
        nitrous_flow_kg_s = self._drivetrain_out.get("nitrous_delivered_kg_s", 0.0)
        self.state.nitrous_boost_frac = min(0.35, nitrous_flow_kg_s * 8.0)
        self.state.nitrous_fill_frac = self._drivetrain_out.get("nitrous_fill_frac", 0.0)
        # real compressed-air idle assist: unlike nitrous (a liquid
        # oxidizer whose MAP-equivalent gain is a modeling convenience),
        # this really is gas dumped straight into the manifold, so its
        # boost-frac is the genuinely physical ratio of what the port
        # just delivered to what the engine itself is actually demanding
        # to breathe this tick -- not a flat, empirically-tuned constant.
        # Capped at 0.5 atm-equivalent: a real anti-stall dump is sized
        # to bridge a momentary sag, not run the engine on stored air.
        pneumatic_flow_kg_s = self._drivetrain_out.get("pneumatic_idle_assist_delivered_kg_s", 0.0)
        self.state.pneumatic_idle_assist_delivered_kg_s = pneumatic_flow_kg_s
        self.state.pneumatic_idle_assist_boost_frac = (
            min(0.5, pneumatic_flow_kg_s / max(self._intake_demand_kg_s, 1e-4)) if pneumatic_flow_kg_s > 0.0 else 0.0)
        # real fuel-supply starvation: how much of last tick's actual
        # combustion demand the pump/tank could really deliver (1.0 =
        # kept up fully; drops as the tank empties or demand outruns
        # the pump's real flow_capacity_kg_s ceiling) -- applied as a
        # genuine multiplier on fuel_quantity_frac below, so an empty
        # tank starves combustion for real instead of running forever
        self.state.fuel_fill_frac = self._drivetrain_out.get("fuel_fill_frac", 1.0)
        self.state.fuel_temp_k = self._drivetrain_out.get("fuel_temp_k", REFERENCE_INTAKE_TEMP_K)
        fuel_delivered_kg_s = self._drivetrain_out.get("fuel_delivered_kg_s", 0.0)
        self._fuel_starvation_frac = (
            min(1.0, fuel_delivered_kg_s / self._fuel_demand_kg_s) if self._fuel_demand_kg_s > 1e-9 else 1.0)
        # boost is a fresh addition on top of the naturally-aspirated
        # baseline each step, never accumulated step-over-step -- this is
        # the UPSTREAM supply pressure (what the supercharger/nitrous can
        # push), fed to next tick's drivetrain solver call as
        # intake_supply_pressure_pa (same one-tick lag as waste_heat_kw,
        # negligible against the intake circuit's own real fill lag)
        if eng.forced_induction.kind == "supercharger":
            # draw-through blower: the throttle sits UPSTREAM of the
            # rotors, so what it sets is the blower's inlet pressure
            # (_map_frac_na, the throttled/filtered supply), and the
            # blower's own pressure ratio (1 + boost_frac, set purely by
            # displacement ratio x belt speed in _step_forced_induction)
            # multiplies THAT. Not additive: an additive boost would put a
            # closed-throttle blown engine above atmospheric at idle,
            # which is not how a blower behind a shut butterfly behaves --
            # it pulls a vacuum on its own inlet instead.
            raw_map_frac = (self._map_frac_na * (1.0 + self.state.boost_frac)
                           + self.state.nitrous_boost_frac + self.state.pneumatic_idle_assist_boost_frac)
        else:
            raw_map_frac = (self._map_frac_na + self.state.boost_frac
                           + self.state.nitrous_boost_frac + self.state.pneumatic_idle_assist_boost_frac)
        self._intake_supply_pressure_pa = 101_325.0 * raw_map_frac
        # real engine breathing demand at the current rpm -- displacement
        # swept once per two crank revs (four-stroke) -- times the real
        # charge density this tick's own MAP represents (ideal gas law:
        # density scales with pressure at ~fixed temp), NOT a flat
        # atmospheric-density constant. This is the missing half of the
        # real choke ceiling: without it, a boosted engine's demand never
        # numerically reflects how much MORE mass it's actually asking
        # the intake tract to pass at 2x atmospheric pressure than at 1x,
        # so flow_capacity_kg_s (abstract_ui_vehicles.py's
        # powertrain.intake_plenum_port) never actually saturated boost
        # at all -- it just looked capped by coincidence before, and MAP
        # genuinely piled up unbounded (2.19x atmospheric, power swinging
        # 400kW-3900kW between ticks, never converging) once a separate
        # fix removed that coincidence. Reads last tick's own MAP (the
        # same one-tick lag already used for waste_heat_kw/supply
        # pressure above) since this tick's hasn't been solved yet.
        displacement_m3 = eng.displacement_l / 1000.0
        charge_density_frac = max(0.2, self.state.manifold_pressure_frac)
        self._intake_demand_kg_s = displacement_m3 * (max(self.state.rpm, 0.0) / 120.0) * 1.2 * charge_density_frac
        # real exhaust mass flow: the same real air demand above plus the
        # real fuel mass combustion actually adds (mass is conserved --
        # exhaust gas leaving is intake air plus fuel burned, roughly a
        # stoichiometric ~14.7:1 AFR's worth, ~6.8% on top), fed to next
        # tick's drivetrain solver as exhaust_demand_kg_s, compared
        # against the real exhaust-primary flow_capacity_kg_s ceiling the
        # same way intake demand is compared against the intake ceiling
        self._exhaust_demand_kg_s = self._intake_demand_kg_s * 1.068
        # real fuel mass demand: intake air demand divided by whatever
        # fuel's ACTUAL stoichiometric ratio is (engines.fuel_stoich_afr),
        # not the fixed 14.7 gasoline assumption the exhaust-mass estimate
        # above uses -- methanol/nitro genuinely draw far more fuel mass
        # per kg of air, which is exactly what makes the fuel tank/pump
        # circuit below (drivetrain_graph's real depletable-reservoir
        # physics) drain faster on those fuels, a real consequence, not
        # a coincidence of the exhaust estimate's own gasoline constant.
        stoich_afr = fuel_stoich_afr(self.fuel_choice or eng.preferred_fuel_profile)
        # METERED ON THE AIR THE CYLINDER ACTUALLY GETS, not the air the
        # engine asks the tract for.
        #
        # _intake_demand_kg_s above is a DEMAND -- displacement rate at
        # this tick's MAP -- and that is exactly right for what it is
        # used for, comparing against the tract's choke ceiling. It is
        # the wrong number to fuel against, because it omits every
        # breathing term between the manifold and the cylinder:
        # runner resonance, the head's own breathing ceiling, the
        # chamber's wall-loss factor, and the charge's temperature
        # density. Those are exactly the factors the per-cylinder
        # strength calculation applies below, and the pressure term is
        # deliberately NOT repeated here because the demand already
        # carries it.
        #
        # Fuelling on demand over-fuels every engine by its own
        # breathing shortfall -- about 16% on the catalogue's AMC 258 --
        # which showed up as a brake thermal efficiency of 16-18% and a
        # BSFC near 500 g/kWh against a real 25-30% and 280-330. A real
        # engine meters on measured or speed-density-estimated CHARGE,
        # never on unfulfilled demand.
        #
        # One tick of lag, the same lag waste_heat_kw and the supply
        # pressure above already take, because the charge fraction is
        # solved per cylinder later in this same tick.
        self._fuel_demand_kg_s = (self._intake_demand_kg_s
                                  * max(0.05, min(1.5, self._charge_fill_frac))
                                  / max(stoich_afr, 0.1))
        # real evaporative + sensible charge cooling from the fuel
        # itself: a latent-heat term that's ALWAYS present (liquid fuel
        # boiling off in the charge absorbs real heat regardless of fuel
        # temperature -- this was previously entirely unmodeled), plus
        # an extra sensible term when the fuel is actively colder than
        # ambient (a real drag-cooler's whole point). Uses LAST tick's
        # solved fuel_temp_k (the same one-tick lag intake_supply_
        # pressure_pa/waste_heat_kw already use) since this tick's fuel
        # circuit hasn't been solved yet when this fed INTO that solve.
        fuel_profile = self.fuel_choice or eng.preferred_fuel_profile
        latent_heat_w = self._fuel_demand_kg_s * fuel_latent_heat_j_per_kg(fuel_profile)
        sensible_cooling_w = self._fuel_demand_kg_s * FUEL_SPECIFIC_HEAT_J_PER_KGK * max(
            0.0, REFERENCE_INTAKE_TEMP_K - self.state.fuel_temp_k)
        self._fuel_cooling_kw = (latent_heat_w + sensible_cooling_w) / 1000.0
        # the same real evaporative-cooling mechanism, off whatever real
        # accessory is actually manifolded into the generic injection
        # boss (abstract_ui_vehicles.py's powertrain.engine_block_port.
        # accessory_injection_boss): a water-meth kit's whole point is
        # charge cooling, and its latent heat per kg (engines.
        # WATER_METHANOL_LATENT_HEAT_J_PER_KG) is far higher than
        # fuel's, so a modest delivered mass buys real, substantial
        # knock margin
        auxiliary_injection_kg_s = self._drivetrain_out.get("auxiliary_injection_delivered_kg_s", 0.0)
        self._auxiliary_injection_cooling_kw = (
            auxiliary_injection_kg_s * WATER_METHANOL_LATENT_HEAT_J_PER_KG) / 1000.0
        # manifold_pressure_frac itself now comes straight from that real
        # circuit's own solved pressure state, not a formula computed here
        self.state.manifold_pressure_frac = self._drivetrain_out.get(
            "intake_manifold_pressure_pa", 101_325.0) / 101_325.0

        # --- the 12 V bus: every box's real draw, solved as a circuit ---
        # (electrical_network.py) -- alternator speed from the belt the
        # drivetrain solver just stepped, loads from what each box is
        # really doing right now
        computerized = is_computerized(eng)
        ac_engaged = self.ac_enabled and eng.accessories.air_conditioning
        fan_load_w = 0.0
        fan_spec = self._drivetrain._fan_specs.get("powertrain.cooling_fan_electric")
        if fan_spec is not None:
            fan_omega = self._drivetrain.omega.get("powertrain.cooling_fan_electric", 0.0)
            fan_load_w = self.electrical.electric_fan_load_w(
                rotor_aero_load_torque_nm(fan_omega, fan_spec["flow_coeff"], COOLING_FAN_DISK_AREA_M2), fan_omega)
        loads_w = (self.electrical.ignition_load_w(self.state.rpm, self.state.ignition_cut)
                   + self.electrical.fuel_pump_load_w(fuel_delivered_kg_s)
                   + fan_load_w
                   + (AC_CLUTCH_COIL_W if ac_engaged else 0.0)
                   + self.electrical.load_resistor_w(self.electrical_load_frac)
                   # the starter motor's real draw while cranking: the
                   # heaviest load the bus ever sees, and what sags it
                   + self.starter.reading.bus_current_a * max(self.state.battery_voltage, 1.0)
                   # a bus-fed electric air compressor when its unloader
                   # has it loaded (a switchboard-fed one is outside this bus)
                   + (self._drivetrain_out.get("pneumatic_compressor_running_w", 0.0) / ELECTRIC_COMPRESSOR_MOTOR_EFFICIENCY
                      if self._drivetrain_out.get("pneumatic_compressor_drive") == "electric-motor" else 0.0))
        bus = self.electrical.step(dt, self._drivetrain.omega.get("electrical.alternator", 0.0), loads_w,
                                   ecu_powered_draw_w=ECU_QUIESCENT_W if computerized else 0.0)
        # a real computer goes dark below brown-out; a mechanical/magneto
        # engine has no computer to lose
        self.ecu.powered = self.electrical.powered or not computerized
        self.state.battery_voltage = bus.voltage_v
        self.state.battery_soc_frac = bus.battery_soc_frac
        self.state.alternator_current_a = bus.alternator_current_a
        self.state.electrical_load_w = bus.load_w
        self.state.bus_regulating = bus.regulating
        self.state.alternator_output_frac = (bus.alternator_current_a / self.electrical.alternator.rated_current_a
                                             if self.electrical.alternator.rated_w > 0.0 else 0.0)
        # the ignition box's spark energy off what actually feeds its coil
        self.state.spark_energy_frac = self.ignition.spark_energy_frac(eng, bus.voltage_v)
        # the ECU's fan thermostat program, for next tick's solver call
        self._electric_fan_cmd = self.ecu.electric_fan_command(self._drivetrain.coolant_temp_k())

        base_timing_deg = (self.ignition_timing_policy(eng, self.state.rpm, self.state.manifold_pressure_frac)
                           + getattr(eng, "ignition_timing_offset_deg", 0.0))
        ecu = eng.ecu
        # real timing-chain lag (from the drivetrain graph solver) reads
        # straight through to spark timing on an older distributor-driven
        # ignition that rides the same chain -- a small, genuine retard
        cam_lag_deg = abs(self._drivetrain_out.get("cam_phase_lag_rad", 0.0)) * (180.0 / math.pi) * 0.5
        self.state.cam_phase_lag_deg = cam_lag_deg
        # the ECU's spark timing program: MBT base less its knock-retard
        # pull-back (reacting to LAST tick's knock) less the chain lag
        self.state.ignition_timing_deg = self.ecu.spark_timing_deg(
            eng, base_timing_deg, self.state.knock_flag, self.state.knock_intensity, cam_lag_deg, dt)

        # fuel conversion physics (fuel_network.py): knock compatibility
        # is already in fuel_compatibility (derived, see convert_engine)
        # and is what this knock-resistance term reads; the fluid's
        # charge-energy ratio is applied where the admitted charge's
        # own strength is assembled (fuel_quantity_frac below)
        octane_factor = self._octane_factor()
        if self._fuel_network is not None:
            risk = intake_flashback_risk(self._fuel_network.fluid.name, self.state.intake_charge_temp_k,
                                         self._fuel_network.spec.admission)
            self.state.intake_flashback_risk = risk
            # expected flashbacks accumulate per admission event; each
            # whole one is a real unplanned intake backfire
            self._flashback_accum += risk * (self.state.rpm / 60.0) * (eng.architecture.cylinders / 2.0) * dt
            if self._flashback_accum >= 1.0:
                self._flashback_accum -= 1.0
                self.pending_backfires.append(BackfireEvent(kind="intake-flashback", strength=min(1.0, 0.5 + risk)))
        misfire_prob = (MISFIRE_BASE_PROB + 0.6 * (1.0 - self.state.spark_energy_frac)) * (1.0 - ecu.spark_reliability_bonus)

        # EGR: inert exhaust gas displaces some of the fresh charge, so it
        # costs a real, proportional slice of combustion torque -- and that
        # same displaced fraction is what lowers peak combustion temperature
        # and fights knock, independent of spark timing.
        egr_active_frac = eng.egr.active_frac(self.throttle)
        # closed-loop AFR power enrichment: extra fuel dumped purely to cool
        # the charge as throttle nears WOT, right when knock risk peaks --
        # not a torque effect, a knock-resistance one.
        afr_cooling_frac = self.ecu.power_enrichment_frac(eng, self.throttle)
        knock_dilution_frac = max(0.0, min(0.9, egr_active_frac + afr_cooling_frac))
        # a virtual dyno-absorber shaft's inertia, scaled off the engine's
        # own -- a fixed absolute floor here (the old 0.02 kg*m^2) was
        # calibrated for normal-car-engine scale and became a genuine bug
        # once the catalogue spans 9 orders of magnitude (a 0.00006 kg*m^2
        # trimmer crank up to a 340,000 kg*m^2 ship crank): a fixed floor
        # 300x the trimmer's own inertia meant the load shaft could never
        # spin up, so the clutch junction stayed pinned near max resistance
        # and stalled a 1.6 N*m engine outright. 1e-6 here is just a
        # divide-by-zero guard, not a physically meaningful minimum.
        # the dyno absorber's real declared inertia (a real cast-iron
        # roller drum's mass/radius, see drivetrain_graph.py's
        # dyno_absorber node), not a scaled guess off the engine's own
        load_inertia_kg_m2 = self._dyno_inertia_kg_m2
        # diesel: MAP no longer tracks throttle (unthrottled air), so the
        # pedal has to reach the engine some other way -- directly as fuel
        # quantity, the way a real injection pump governor actually works,
        # complete with its own idle governor acting on fuel instead of air.
        # Gasoline/rotary already get their throttle response through MAP.
        if eng.compression_ignition:
            if self.throttle < 0.08:
                target_fuel_frac = self.ecu.idle_fuel_quantity(
                    eng, self.state.rpm,
                    self.state.ac_compressor_load_w
                    + self.electrical.reading.alternator_shaft_load_w
                    + self.known_accessory_shaft_load_w, dt,
                    coupled_inertia_kg_m2=self._idle_coupled_inertia_kg_m2())
            else:
                self.ecu.release_fuel_governor()
                target_fuel_frac = max(0.12, min(1.0, self.throttle))
            # a real injection-pump rack/flyweight assembly has finite
            # mechanical response speed, same real reason MAP itself gets
            # a first-order lag (map_tau_s) instead of jumping straight
            # to its target -- without this, a very slow-firing engine
            # (few combustion events per second at idle) sees the
            # governor's raw per-tick output directly, undamped, and the
            # sparse discrete torque impulses excite a real control-loop
            # oscillation the lag naturally damps out (verified against
            # the catalogue's giant 22rpm marine diesel, the extreme case).
            self._fuel_rack_frac += (target_fuel_frac - self._fuel_rack_frac) * min(1.0, dt / FUEL_RACK_TAU_S)
            fuel_quantity_frac = self._fuel_rack_frac
        else:
            # electronic injection metering is always correct (1.0); a
            # carbureted engine's real single jet either matches its
            # own reference sizing or doesn't (engines.derive_jet_
            # metering_frac) -- genuinely lean or rich, not silently
            # perfect, and it changes with whatever fuel is actually
            # loaded (same jet, different fuel, different mixture)
            fuel_quantity_frac = derive_jet_metering_frac(eng, self.fuel_choice or eng.preferred_fuel_profile)
            gaseous_admission = (self._fuel_network is not None
                                 and self._fuel_network.spec.admission.kind in ("mixer", "gas-injector"))
            if gaseous_admission:
                # a fuel-network conversion to a gaseous fuel replaces the
                # liquid jet / liquid injector with its own mixer or gas
                # injector, sized for THAT gas (fuel_network.Mixer's own
                # fraction) -- sizing a liquid jet against a gas's density
                # is exactly the wrong-physics result this bypasses
                # ...and the charge it admits carries this fluid's real
                # fixed-volume energy relative to the gasoline the
                # catalogue's bmep was rated on (fuel_network.charge_
                # energy_factor: ~0.90 natural gas, ~0.83 hydrogen,
                # ~0.72 wood gas), a derived multiplier, not a tuned one
                fuel_quantity_frac = self._charge_energy_factor
                self.state.injector_duty_frac = 0.0
            elif eng.injector is not None:
                # a real, declared injector array (injector.py): its own
                # sqrt(pressure) flow capacity at this engine's nominal
                # rail pressure caps what it can actually deliver --
                # distinct from (and stacking with) pump/tank starvation
                # below, a genuinely different real failure mode
                # (undersized injectors vs. a starved supply)
                duty = eng.injector.duty_for_target_flow(self._fuel_demand_kg_s, EFI_RAIL_PRESSURE_PA)
                self.state.injector_duty_frac = duty
                if duty > 1.0:
                    fuel_quantity_frac /= duty
            else:
                self.state.injector_duty_frac = 0.0
        # real fuel-supply starvation applies uniformly, carbureted/EFI/
        # diesel alike -- an empty tank or an outrun pump doesn't care
        # which metering scheme is downstream of it
        fuel_quantity_frac *= self._fuel_starvation_frac
        # the real mixture this tick, for emissions: metering relative
        # to stoich (a carb's jet, an EFI's correct 1.0), plus any power
        # enrichment; a diesel's rack is load, and it always burns lean
        if eng.identity in getattr(engines, "_COMPRESSION_IGNITION", set()):
            self.state.mixture_phi = 0.2 + 0.7 * max(0.0, min(1.0, fuel_quantity_frac))
        else:
            self.state.mixture_phi = max(0.3, fuel_quantity_frac / max(self._fuel_starvation_frac, 1e-6)) * (1.0 + afr_cooling_frac)
        # oxygen actually in what it is breathing (air_volumes): a bay or
        # a sealed room going oxygen-poor is real power loss
        fuel_quantity_frac *= max(0.0, self.state.intake_o2_factor) ** 0.5
        # node_effects: a starved/aerated fuel supply, a leaned or oiled charge
        fuel_quantity_frac *= self._effects.fuel_supply_factor * self._effects.mixture_factor
        if computerized and not self.ecu.powered:
            # an EFI engine's injectors are the computer's outputs -- no
            # computer, no fuel. A carbureted or diesel engine keeps
            # metering mechanically with a dead bus.
            fuel_quantity_frac = 0.0

        if arch.rotary or not arch.has_poppet_valves:
            # port-timed (rotary), or a small port-scavenged two-stroke
            # with no valves in the head at all -- either way there's no
            # lifter spring to drag or float. Apex-seal wear (rotary) is a
            # real failure mode but a different mechanism than this
            # model covers.
            valvetrain_drag_nm = 0.0
            self.state.valve_float_flag = False
            self.state.valve_float_risk = 0.0
            float_penalty = 1.0
        else:
            spring = eng.lifter_spring
            valvetrain_drag_nm = spring.drag_torque_nm(
                arch.cylinders, valves_per_cylinder=cylinder_ports.valves_per_cylinder(self.engine))
            spring_safe_rpm = spring.max_safe_rpm()
            float_risk = self.state.rpm / max(spring_safe_rpm, 1.0)
            self.state.valve_float_flag = float_risk > 1.0
            self.state.valve_float_risk = max(0.0, float_risk - 1.0)
            # no artificial floor -- a real floating valve doesn't stop
            # getting worse past some fixed point and settle into a
            # permanent 65%-torque plateau; it keeps degrading toward
            # genuinely losing the cylinder entirely (this used to floor
            # at 0.35 and go completely flat past valve_float_risk~1.5,
            # so once real float risk actually got that high, valve
            # float itself stopped contributing anything further to the
            # engine's own slow ongoing rpm climb -- the same shape of
            # bug as the broken choke formula, the flat 1.12x-redline
            # rpm cap, and the frozen friction-growth cap, all fixed
            # alongside this one)
            # per-cylinder, from the valvetrain's own state arrays: every
            # cylinder floats at ITS lowest valve's float speed, breathes
            # through ITS intake lift, and leaks at ITS seats -- one
            # vectorized pass per tick, applied per cylinder below in
            # base_strength; the engine-wide figures report the worst
            # cylinder. The springs age here too (sag with cycles and
            # block temperature, exhaust seats recede).
            vs = self._valve_state
            # crankcase participation first: oil film, blow-by, dilution per
            # cylinder from last tick's combustion strengths; its oil burn is
            # a carbon source for the valves
            cs = self._crankcase_state
            strengths = _np.array([self._last_strength.get(c, 0.0) for c in range(1, cs.n_cyl + 1)])
            cs.step(dt, self.state.rpm, strengths, fuel_quantity_frac, self.state.coolant_temp_k,
                    fires_per_s=self.state.rpm / 60.0 * eng.architecture.firing_events_per_rev
                    if hasattr(eng.architecture, "firing_events_per_rev") else self.state.rpm / 120.0 * cs.n_cyl,
                    deposit_kg_s=self.hole_emitters.splash_deposit_kg_s(cs.n_cyl))
            self.state.cylinder_oil_film_mg = (cs.film_kg * 1e6).tolist()
            self.state.oil_consumption_ml_per_h = float(cs.burn_kg_s.sum()) * 3600.0 / 0.87 * 1000.0
            self.state.blowby_l_per_min = float(cs.blowby_kg_s.sum()) / 1.2 * 60000.0
            self.state.crankcase_pressure_kpa = cs.crankcase_pressure_pa / 1000.0
            self.state.sump_oil_l = cs.oil_kg / 0.87
            self.state.oil_fuel_dilution_frac = cs.dilution_frac
            self._step_oil_contamination(dt, cs, fuel_delivered_kg_s)
            vs.age(dt, self.state.rpm, self.state.cylinder_block_temps_k,
                   exhaust_temp_k=self.state.exhaust_temp_k, load_frac=self.throttle,
                   richness=fuel_quantity_frac, direct_injection=eng.compression_ignition,
                   oil_burn_frac=cs.oil_burn_frac())
            pc = vs.per_cylinder()
            self._cyl_valve_factor = (vs.float_penalty(self.state.rpm, pc["float_rpm"])
                                      * pc["breathing_frac"] * (1.0 - pc["leak_frac"]))
            worst_float = float(_np.min(pc["float_rpm"])) if len(pc["float_rpm"]) else spring_safe_rpm
            float_risk = self.state.rpm / max(min(spring_safe_rpm, worst_float), 1.0)
            self.state.valve_float_flag = float_risk > 1.0
            self.state.valve_float_risk = max(0.0, float_risk - 1.0)
            self.state.cylinder_breathing_frac = pc["breathing_frac"].tolist()
            self.state.cylinder_float_rpm = pc["float_rpm"].tolist()
            self.state.cylinder_valve_factor = self._cyl_valve_factor.tolist()
            self.state.cylinder_carbon_frac = pc["carbon_frac"].tolist()
            self.state.cylinder_hotspot_risk = pc["hotspot_risk"].tolist()
            float_penalty = 1.0   # applied per cylinder via _cyl_valve_factor

        deg_per_s = self._omega * 180.0 / math.pi
        total_deg_this_step = deg_per_s * dt
        n_sub = max(1, math.ceil(abs(total_deg_this_step) / MAX_SUBSTEP_DEG))
        sub_dt = dt / n_sub

        any_knock = False
        max_knock_intensity = 0.0
        any_misfire = False
        any_preignition = False
        max_preignition = 0.0

        # intake/exhaust harmonics: how many times per crank REVOLUTION a
        # runner/pipe actually gets excited -- a 4-stroke cylinder fires
        # once per 720 degrees (half as often per rev as a 2-stroke's
        # 360), so this is the real quantity both IntakeSystem's Helmholtz
        # tuned_rpm and ExhaustSystem's quarter-wave tuned_rpm are built
        # around, not a fixed assumption of "4-stroke" baked into either.
        firing_events_per_rev = arch.cylinders * 360.0 / arch.cycle_degrees

        for _ in range(n_sub):
            deg_per_s = self._omega * 180.0 / math.pi
            dtheta = deg_per_s * sub_dt
            prev_pos = self._total_crank_deg % arch.cycle_degrees
            self._total_crank_deg += dtheta
            new_pos = self._total_crank_deg % arch.cycle_degrees

            live_rpm = self._omega / RPM_TO_RAD_S
            # the rev limiter is the ignition box's decision, made at
            # crank-event rate -- ignition_driver.IgnitionDriver owns the
            # latched stages / soft taper and hands back this substep's
            # per-spark cut probability
            active_severity = self.ignition.cut_severity(eng, live_rpm, self.rev_limiter_enabled)

            governor_skip = (eng.governor_mode == "hit_and_miss"
                              and live_rpm >= (eng.governor_target_rpm or eng.redline_rpm))
            self.state.exhaust_valve_held_open = bool(governor_skip)

            torque_nm = 0.0
            any_real_ignition_this_substep = False
            for cyl, firing_angle in self._firing_angle_deg.items():
                spark_angle = (firing_angle - self.state.ignition_timing_deg) % arch.cycle_degrees
                if self._crossed(prev_pos, new_pos, spark_angle, arch.cycle_degrees):
                    limiter_cut = active_severity > 0.0 and self._rng.random() < active_severity
                    misfire = (not limiter_cut) and (not governor_skip) and self._rng.random() < misfire_prob
                    # ignition_cut: the key/kill-switch turning the
                    # ignition system off, not a stall -- every cylinder
                    # simply gets no spark, same real mechanism a misfire
                    # already models, so the crank coasts down under its
                    # own pumping/friction/dyno load instead of being
                    # teleported to a dead stop
                    cut = misfire or limiter_cut or governor_skip or self.state.ignition_cut
                    # surface ignition before the spark: a hot spot (this
                    # cylinder's own real deposit/valve risk) lights a hot,
                    # dense charge early -- see PREIGNITION_* above
                    hotspot = (self.state.cylinder_hotspot_risk[cyl - 1]
                               if 0 < cyl <= len(self.state.cylinder_hotspot_risk) else 0.0)
                    preignition = False
                    if (not cut) and hotspot > 0.0 and not eng.compression_ignition:
                        heat_factor = max(0.0, min(1.5, 0.3 + (self.state.intake_charge_temp_k - 300.0) / 120.0))
                        p_pre = PREIGNITION_BASE_PROB * hotspot * (0.2 + 0.8 * self.state.manifold_pressure_frac) * heat_factor
                        preignition = self._rng.random() < p_pre

                    # the real cylinder cycle integrating the slower
                    # fluid transfer: this event draws its own charge out
                    # of the real intake-air circuit's current pressure
                    # (a parametric two-volume equilibrium exchange, not
                    # a flat copy of one shared manifold_pressure_frac
                    # for every cylinder identically), and genuinely
                    # depletes that circuit a little each time -- many
                    # fast firing events between slow fluid-circuit steps
                    # really do pull a shared plenum down, same real
                    # effect a bank of cylinders drawing on one intake
                    # has
                    cylinder_charge_frac = self._drivetrain.draw_cylinder_charge(self._cylinder_volume_m3)
                    # real intake-runner ram effect: a genuine Helmholtz
                    # resonance (engines.IntakeSystem.resonance_gain) layered
                    # on top of the solved flow-capacity charge above --
                    # the actual reason a tuned NA intake can exceed 1.0 VE
                    # in a real rpm band, not present at all off that band.
                    cylinder_charge_frac *= eng.intake_system.resonance_gain(
                        live_rpm, firing_events_per_rev, self.state.intake_charge_temp_k)
                    # the head's own breathing ceiling (valve size the
                    # included angle allows) and its wall-loss penalty
                    cylinder_charge_frac *= eng.chamber.breathing_factor * eng.chamber.efficiency_factor
                    # real ideal-gas density correction: the same MAP can
                    # hold a lighter (hotter) or denser (cooler) charge --
                    # a boosted engine's charge is compression-heated by
                    # the turbo/supercharger itself (drivetrain_graph.py's
                    # real adiabatic-compression relation on the intake
                    # circuit), only partly rejected by the plenum's own
                    # real passive cooling capacity. This is the actual
                    # physical value an intercooler provides -- a cooler
                    # charge at the same MAP is a genuinely denser one.
                    charge_density_frac = REFERENCE_INTAKE_TEMP_K / max(self.state.intake_charge_temp_k, 200.0)
                    cyl_valve = float(self._cyl_valve_factor[cyl - 1]) if 0 < cyl <= len(self._cyl_valve_factor) else 1.0
                    cyl_valve *= self._effects.cylinder_factor.get(cyl, 1.0)      # node_effects: a holed/dead bore
                    # the BREATHING half of the charge, for next tick's
                    # fuel metering: resonance and the chamber factors
                    # and the temperature density, with the pressure
                    # term divided back out because _intake_demand_kg_s
                    # already carries it and must not carry it twice
                    _map = max(0.05, float(self.state.manifold_pressure_frac))
                    self._charge_fill_frac = (cylinder_charge_frac * charge_density_frac) / _map
                    base_strength = (torque_fraction(eng, self.state.rpm) * cylinder_charge_frac
                                      * charge_density_frac
                                      * fuel_quantity_frac * self._combustion_efficiency_now() * float_penalty
                                      * cyl_valve
                                      * (1.0 - egr_active_frac))
                    strength = 0.0 if cut else base_strength
                    if preignition:
                        strength *= PREIGNITION_STRENGTH_FRAC
                        pre_intensity = min(1.0, 0.6 + 0.4 * hotspot)
                        any_knock = True
                        max_knock_intensity = max(max_knock_intensity, pre_intensity)
                        any_preignition = True
                        max_preignition = max(max_preignition, pre_intensity)
                        self.state.preignition_count += 1
                        if 0 < cyl <= len(self.state.cylinder_block_temps_k):
                            self.state.cylinder_block_temps_k[cyl - 1] += PREIGNITION_PISTON_HEAT_K * pre_intensity

                    if cut:
                        if governor_skip or (limiter_cut and ecu.limiter_fuel_cut):
                            # governor "miss": the flyball governor holds the
                            # exhaust valve off its seat (or blocks the fuel
                            # pump on later designs) for the whole revolution
                            # -- nothing is admitted to burn, same structural
                            # reason a fuel-cut limiter can't backfire either.
                            backfire_prob = 0.0
                        else:
                            backfire_prob = BACKFIRE_BASE_PROB + BACKFIRE_BOOST_GAIN * self.state.boost_frac
                            if limiter_cut:
                                backfire_prob = max(backfire_prob, REV_LIMITER_BACKFIRE_PROB * active_severity)
                        if self._rng.random() < backfire_prob:
                            self.pending_backfires.append(
                                BackfireEvent(kind="unplanned", strength=0.4 + 0.6 * base_strength))

                    self._last_fire_total_deg[cyl] = self._total_crank_deg
                    self._last_strength[cyl] = strength
                    slot = self._slot_of_cyl.get(cyl)
                    if slot is not None and slot < len(self.state.slot_records):
                        self.state.slot_records[slot] = [
                            float(strength), 1.0 if misfire else 0.0,
                            float(min(1.0, 0.6 + 0.4 * hotspot)) if preignition else 0.0,
                            1.0 if preignition else 0.0]
                    if not cut:
                        self.state.fire_event_count += 1
                    self._knock_accum[cyl] = 0.0
                    self._knocked_this_burn[cyl] = False
                    # a real spark kernel, seeded at ignition -- see the
                    # burn-window loop below for how it actually grows
                    self._flame_radius_m[cyl] = SPARK_KERNEL_SEED_RADIUS_M
                    self._burned_frac[cyl] = 0.0

                    # not published to pending_events yet -- held until its
                    # burn window closes and knock is fully resolved (see
                    # the finalize branch below)
                    ev = IgnitionEvent(cylinder=cyl, position_key=cyl, strength=strength,
                                        knock=preignition, knock_intensity=(min(1.0, 0.6 + 0.4 * hotspot) if preignition else 0.0),
                                        misfire=misfire, preignition=preignition)
                    if preignition:
                        self._knocked_this_burn[cyl] = True   # the early burn already IS the violent one; no second knock on top
                    self._active_event[cyl] = ev
                    any_misfire = any_misfire or misfire
                    any_real_ignition_this_substep = any_real_ignition_this_substep or not cut

            # a real, GENERAL per-tick firing-rate measurement (see
            # EngineCycleState.real_fire_hz's own docstring) -- this
            # engine's own real ignition events feed it the exact same
            # way an atmospheric engine's do (_step_atmospheric), not a
            # bespoke per-kind formula
            self._record_ignition(sub_dt, any_real_ignition_this_substep)
            # real per-component durability accrual -- see wear.py.
            # knock isn't finalized for THIS substep's own cylinder
            # until its burn window closes later in this same loop, so
            # this uses the most recently resolved knock_flag (at most
            # one substep of real lag) rather than restructuring the
            # burn-window resolution above just to feed this
            wear_module.step_wear(self.state.wear, eng, sub_dt, live_rpm,
                                   any_real_ignition_this_substep, self.state.knock_flag)

            # calibrate so the time-average of n_cyl combustion events (once
            # per cylinder per cycle) reproduces eng.peak_torque_nm * strength
            # -- self-normalizing now: since burned_frac is constructed to
            # integrate to exactly 1.0 over WHATEVER angular extent the real
            # flame actually takes (no longer a fixed BURN_WINDOW_DEG), the
            # calibration no longer needs to know that extent in advance.
            n_cyl = max(len(self._firing_angle_deg), 1)
            pulse_calib = arch.cycle_degrees / n_cyl
            bore_half_m = max(eng.architecture.bore_m / 2.0, 1e-6)
            # diesel is compression ignition -- there's no spark to advance
            # into knock territory, so the gasoline autoignition-race model
            # doesn't apply; a rate of 0 means the accumulator never moves
            knock_rate = 0.0 if eng.compression_ignition else self.knock_rate_policy(
                self.state.manifold_pressure_frac * (1.0 - knock_dilution_frac),
                self.state.ignition_timing_deg, octane_factor, self.state.intake_charge_temp_k)
            # the end gas cools to the liner during a SLOW compression
            # (real: at cranking speed there is no knock), see
            # slow_compression_knock_factor -- relative to this engine's
            # own torque-peak speed so the calibrated rate there is untouched
            knock_rate *= slow_compression_knock_factor(eng, live_rpm)
            # chamber geometry: a long flame path (side plug) parks the
            # end gas longer; squish turbulence burns it first
            knock_rate *= eng.chamber.knock_factor
            for cyl, last_fire in self._last_fire_total_deg.items():
                since = self._total_crank_deg - last_fire
                burned = self._burned_frac.get(cyl, 1.0)
                if since >= 0.0 and burned < 1.0:
                    # a real, integrated flame kernel -- not a pre-shaped
                    # pulse. Turbulent flame speed from THIS INSTANT's real
                    # mean piston speed (genuinely time-varying as rpm
                    # itself evolves during the burn), grown substep by
                    # substep; burned mass fraction from real sphere-
                    # volume scaling (radius^3), clipped at the real
                    # cylinder bore -- engine_toy's own architecture.bore_m,
                    # not assumed. The torque this substep is the real,
                    # emergent RATE of mass burned (d_burned/d_theta), not
                    # a lookup -- the total impulse over the whole event
                    # still calibrates to peak_torque_nm*strength (pulse_calib
                    # above), but how that's actually delivered over crank
                    # angle now comes from the flame's own real growth.
                    if self._last_strength[cyl] > 0.0 and not self._knocked_this_burn[cyl]:
                        self._knock_accum[cyl] += knock_rate * sub_dt
                        if self._knock_accum[cyl] >= 1.0:
                            self._knocked_this_burn[cyl] = True
                            intensity = max(0.0, min(1.0, 1.0 - burned))
                            self._last_strength[cyl] *= (1.0 - 0.25 * intensity)
                            ev = self._active_event.get(cyl)
                            if ev is not None:
                                ev.knock = True
                                ev.knock_intensity = intensity
                            slot = self._slot_of_cyl.get(cyl)
                            if slot is not None and slot < len(self.state.slot_records):
                                self.state.slot_records[slot][2] = max(self.state.slot_records[slot][2], float(intensity))
                            any_knock = True
                            max_knock_intensity = max(max_knock_intensity, intensity)
                    turbulent_flame_speed = (LAMINAR_FLAME_SPEED_M_S + FLAME_TURBULENCE_GAIN * eng.mean_piston_speed_m_s(live_rpm)) \
                        * eng.chamber.burn_speed_factor
                    # the front has to cross the chamber's REAL farthest
                    # path (a side plug: nearly a whole bore), not half a bore
                    path_m = bore_half_m * eng.chamber.flame_path_rel
                    self._flame_radius_m[cyl] = min(
                        path_m, self._flame_radius_m.get(cyl, SPARK_KERNEL_SEED_RADIUS_M) + turbulent_flame_speed * sub_dt)
                    new_burned = min(1.0, (self._flame_radius_m[cyl] / path_m) ** 3)
                    d_burned = max(0.0, new_burned - burned)
                    self._burned_frac[cyl] = new_burned
                    strength = self._last_strength[cyl]
                    if dtheta > 1e-9:
                        torque_nm += eng.peak_torque_nm * strength * pulse_calib * (d_burned / dtheta)
                elif burned >= 1.0 and cyl in self._active_event:
                    # burn window closed -- knock (if any) is fully resolved,
                    # only now is it safe to publish this event: consumers
                    # calling drain_events() before this would otherwise see
                    # a stale knock=False even when the cylinder did knock
                    # partway through its burn
                    self.pending_events.append(self._active_event.pop(cyl))

            # the real exhaust circuit's own solved backpressure (from
            # the drivetrain solver's real per-cylinder exhaust flow
            # capacity vs demand), not the static hardware-only formula --
            # tapered by this pipe's own real scavenging assist near ITS
            # tuned rpm (engines.ExhaustSystem.scavenging_assist_frac):
            # the reflected low-pressure wave off the open tailpipe end
            # helping pump the cylinder, applied as a ratio onto the real
            # solved value rather than replacing it with a static estimate
            live_backpressure_frac = self.state.exhaust_pressure_frac * (
                1.0 - eng.exhaust_system.scavenging_assist_frac(live_rpm, firing_events_per_rev))
            # Two real, DISTINCT mechanisms, not one blended estimate:
            #  - throttle_pumping_loss_nm: ordinary closed-throttle
            #    intake-vacuum pumping loss. Every real engine has this
            #    -- it's not something a driver can switch off, it's
            #    just what a nearly-closed butterfly plate does to the
            #    intake stroke -- so it's unconditional, independent of
            #    engine_brake_enabled.
            #  - brake_component: the actual driver-operated exhaust
            #    brake. engine_brake_enabled closes a real valve
            #    (EXHAUST_BRAKE_MAX_RESTRICTION_FRAC, drivetrain_graph.
            #    step's exhaust_brake_frac, threaded in above) that
            #    genuinely shrinks the exhaust port's effective flow
            #    capacity -- the same real choked-flow physics an
            #    undersized exhaust already uses builds real
            #    backpressure from that, which is what live_
            #    backpressure_frac carries here. Braking torque scales
            #    directly off that real pressure excess (each exhaust
            #    stroke doing real extra work against it), not an
            #    abstract number -- this IS the actual mechanism now.
            throttle_pumping_loss_nm = eng.peak_braking_torque_nm * torque_fraction(eng, self.state.rpm) * \
                (1.0 - self.state.manifold_pressure_frac) * 0.5
            brake_component = eng.peak_braking_torque_nm * torque_fraction(eng, self.state.rpm) * live_backpressure_frac
            # the compression-release brake: real retarding torque from
            # dumping each cylinder's compressed charge at TDC, T = MEP *
            # V / cycle_rad, only with the fuel off and above idle
            compression_release_nm = 0.0
            if (eng.compression_release_brake and self.compression_brake_enabled and self.throttle < 0.05
                    and live_rpm > eng.idle_rpm * COMPRESSION_RELEASE_MIN_RPM_FRAC_OF_IDLE):
                cycle_rad = math.radians(arch.cycle_degrees)
                speed_frac = max(0.35, min(1.0, live_rpm / max(eng.power_peak_rpm, 1.0)))
                compression_release_nm = COMPRESSION_RELEASE_MEP_PA * (eng.displacement_l / 1000.0) / cycle_rad * speed_frac
            brake_component += compression_release_nm
            self.state.compression_brake_active = compression_release_nm > 0.0
            self.state.compression_brake_torque_nm = compression_release_nm
            # bearing/seal/oil-pump drag: real, always present, independent of
            # the compression/pumping braking the engine-brake toggle controls
            # -- piston engines already get a floor from valvetrain_drag_nm, but
            # a rotary has none of that (no poppet valves), so without this it's
            # genuinely frictionless with the engine brake off and free-revs
            # forever even at closed throttle
            # real FMEP-based friction (engines.py's Engine.friction_torque_nm),
            # driven by this engine's own real mean piston speed -- not
            # an rpm/redline-fraction proxy, which treated every engine
            # as if it hit the same RELATIVE friction at its own redline
            # regardless of how fast its pistons actually move there (a
            # 22rpm giant marine diesel's pistons move nothing like a
            # superbike's). Uncapped -- real bearing/seal drag keeps
            # rising with speed, it doesn't flatline past some arbitrary
            # rpm ratio.
            #
            # Rotary has no stroke_m at all (no real piston to have a
            # speed) -- kept on the old rpm/redline-fraction proxy just
            # for that one case, explicitly, rather than silently
            # zeroing its friction (which would reintroduce the exact
            # "frictionless rotary free-revs forever" bug this term was
            # originally added to prevent). A real rotor-tip-speed-based
            # correlation is the honest fix there, not built yet.
            if arch.rotary:
                baseline_friction_nm = eng.peak_torque_nm * (0.025 + 0.02 * (live_rpm / max(eng.redline_rpm, 1.0)))
            else:
                baseline_friction_nm = eng.friction_torque_nm(live_rpm)
            # the drivetrain graph solver's own reaction torque -- camshaft
            # bearing drag, timing-chain mesh reaction, accessory-shaft/
            # alternator-CVT reaction, and any supercharger belt reaction,
            # all summed by the solver itself as it walks its own edges
            drivetrain_reaction_nm = self._drivetrain_out.get("crank_reaction_torque_nm", 0.0)
            self.state.accessory_drag_nm = drivetrain_reaction_nm
            pumping_loss = (throttle_pumping_loss_nm + brake_component
                             + valvetrain_drag_nm + baseline_friction_nm + drivetrain_reaction_nm)

            # the commanded brake_load_nm doesn't act on the crank directly
            # -- it's a resistive torque on a small virtual load shaft (a
            # dyno absorber), coupled to the crank through a real clutch
            # port (relative-speed-driven saturating reaction, matching
            # the real game's clutch_torque formula exactly). The SAME
            # port now also serves as the real driver-operated clutch
            # pedal (ClutchPort.engagement, 0=floored/disengaged,
            # 1=fully released/locked) and, when a gear is actually
            # selected, reflects the load shaft through the transmission's
            # real gear ratio -- an ideal (lossless, rigid) gear relation:
            # the load's speed reflects into the crank's own frame scaled
            # UP by the ratio (a low gear spins the crank much faster
            # than the output for the same output speed), and by energy
            # conservation the torque that reaches the load reflects back
            # DOWN scaled the other way... no -- torque multiplies BY the
            # ratio (that's the actual point of a low gear: 3.5x crank
            # torque reaches the output in 1st gear), while speed divides
            # by it. Neutral (effective_ratio 0) means no path to the
            # load at all -- the engine free-revs against the clutch
            # alone, exactly like a real neutral-clutch-in.
            self._brake_junction.engagement = self.clutch_frac
            effective_ratio = self._current_gear_ratio()
            # THE JUNCTION GETS ITS OWN STABLE STEP. The combustion loop
            # subdivides on CRANK ANGLE (MAX_SUBSTEP_DEG), which has
            # nothing to do with how fast the clutch/dyno pair actually
            # is: behind a low gear a light dyno drum reflects into the
            # crank's frame as a tiny, very fast inertia, and stepping
            # that at the crank-angle substep integrates it well past
            # its own stability limit. The turbine, electric and
            # atmospheric paths already asked `_clutch_substep_plan` for
            # the right step (see its docstring -- the same bug was
            # found there once); the piston loop never did, and rang.
            #
            # This costs nothing when the junction is comfortably
            # stable: n_sub comes back as 1 and it is one pass.
            n_j, j_dt = (self._clutch_substep_plan(sub_dt, eng.inertia_kg_m2)
                         if effective_ratio > 0.0 else (1, sub_dt))
            # ...and whatever the ring detector has asked for on top
            if self._ring_escalation_ticks > 0:
                self._ring_escalation_ticks -= 1
                n_j = min(DrivetrainSolver.SUBSTEP_CAP, n_j * self._ring_escalation)
                j_dt = sub_dt / n_j
            elif self._ring_escalation > 1:
                self._ring_escalation = max(1, self._ring_escalation // 2)
            self.state.junction_substeps = n_j
            junction_torque = 0.0
            net_torque = 0.0
            dyno_torque_nm = 0.0
            for _ in range(n_j):
                if effective_ratio > 0.0:
                    junction_torque = self._coupled_junction_torque(
                        j_dt, self._omega, self._load_omega * effective_ratio)
                else:
                    junction_torque = 0.0
                # plus whatever is turning the crank from outside: the
                # starting system's torque (starter.py) and any
                # attachment the game feeds through the crank nose
                net_torque = torque_nm - pumping_loss - junction_torque + self._crank_assist_nm
                self._omega = max(0.0, self._omega + net_torque / max(eng.inertia_kg_m2, 1e-9) * j_dt)
                if effective_ratio > 0.0:
                    dyno_torque_nm = junction_torque * effective_ratio
                    domega_load = (dyno_torque_nm - self.brake_load_nm) / load_inertia_kg_m2 * j_dt
                else:
                    dyno_torque_nm = 0.0
                    domega_load = -self.brake_load_nm / load_inertia_kg_m2 * j_dt
                self._load_omega = max(0.0, self._load_omega + domega_load)
            self._last_dyno_torque_nm = dyno_torque_nm

            self.state.current_torque_nm = net_torque + pumping_loss + junction_torque
            instant_power_kw = self.state.current_torque_nm * self._omega / 1000.0
            rms_alpha = min(1.0, sub_dt / RMS_TAU_S)
            self._torque_ms += (self.state.current_torque_nm ** 2 - self._torque_ms) * rms_alpha
            self._power_ms += (instant_power_kw ** 2 - self._power_ms) * rms_alpha

        self.state.torque_rms_nm = math.sqrt(max(0.0, self._torque_ms))
        self.state.power_rms_kw = math.sqrt(max(0.0, self._power_ms))
        self.state.dyno_rpm = self._load_omega / RPM_TO_RAD_S
        self.state.dyno_absorbed_kw = self.brake_load_nm * self._load_omega / 1000.0
        self.state.dyno_kinetic_energy_j = 0.5 * self._dyno_inertia_kg_m2 * self._load_omega * self._load_omega
        self.state.dyno_torque_nm = self._last_dyno_torque_nm
        self.state.brake_clutch_locked = self._brake_junction.locked
        c = getattr(self, "coupling", None)
        if c is not None:
            self.state.coupling_kind = c.kind
            self.state.coupling_locked = bool(c.locked)
            self.state.coupling_slipping = bool(c.slipping)
            self.state.coupling_capacity_nm = float(c.capacity_nm)
            self.state.coupling_heat_w = float(c.heat_w)
        self.state.rev_limiter_active = self.ignition.any_stage_active or (eng.rev_limiter.soft_taper and active_severity > 0.0)
        self.state.knock_flag = any_knock
        self.state.knock_intensity = max_knock_intensity
        self.state.misfire_flag = any_misfire
        self.state.preignition_flag = any_preignition
        self.state.preignition_intensity = max_preignition

        new_backfires = self.pending_backfires[backfires_before:]
        self.state.backfire_flag = len(new_backfires) > 0
        self.state.backfire_kind = new_backfires[-1].kind if new_backfires else None

        new_rpm = self._omega / RPM_TO_RAD_S
        # "stopped" only when nothing is turning it: a hand crank on a
        # 2.7 kg*m^2 flywheel legitimately spends many ticks under
        # 0.5 rpm on its way up, and zeroing it there froze the crank
        if new_rpm <= 0.5 and self._crank_assist_nm <= 0.0:
            self.state.rpm = 0.0
            self._omega = 0.0
            if self.brake_load_nm > eng.peak_torque_nm * 0.35:
                self.state.stalled = True
        else:
            # a real, honest value -- not min(new_rpm, redline*1.12), an
            # arbitrary reporting ceiling that used to hide here. It
            # capped only the DISPLAYED/consumed rpm, not the crank's
            # real self._omega (already uncapped, real torque/inertia
            # integration) -- but state.rpm feeds torque_fraction,
            # valve_float_risk, knock risk, ignition timing, and MAP,
            # so once the true crank speed exceeded 1.12x redline this
            # silently froze every one of them at that arbitrary point
            # forever, regardless of what the real physics said,
            # defeating the rev-limiter-disable feature outright: an
            # engine could never be observed exceeding 1.12x redline no
            # matter how much real torque was actually available.
            self.state.rpm = new_rpm
        self.state.crank_angle_deg = self._total_crank_deg % arch.cycle_degrees
        self.state.power_kw = self.state.current_torque_nm * self._omega / 1000.0

        # real exhaust gas state: waste heat is fuel energy that didn't
        # become shaft power (power_out/efficiency - power_out), driving
        # a genuine evolving temperature and rpm-driven pressure, not a
        # static resistance number
        # power_rms_kw (already RMS-smoothed over individual combustion
        # pulses), not the instantaneous power_kw -- that oscillates
        # between the firing pulses and is near zero most single ticks,
        # which was silently starving the coolant/exhaust thermal model
        # of almost all its actual heat input
        waste_heat_kw = self.state.power_rms_kw * (1.0 / max(eng.combustion_efficiency, 0.05) - 1.0)
        # fed into next tick's drivetrain solver call (see above) -- a
        # one-dt lag against a multi-second coolant thermal time constant
        # is negligible, and keeps the solver called once per tick instead
        # of needing torque/power finalized before it can run at all
        self._waste_heat_kw = waste_heat_kw
        # both now read straight from the real exhaust-gas circuit the
        # drivetrain solver itself just stepped (drivetrain_graph.py's
        # real combustion heat-share + real choked-flow backpressure),
        # not a toy-side formula -- the circuit's own real fill-time-
        # constant dynamics already provide the lag these used to fake
        # with a separate local tau
        self.state.exhaust_temp_k = self._drivetrain_out.get("exhaust_temp_k", EXHAUST_TEMP_AMBIENT_K)
        # the real exhaust NETWORK's own job: this manifold-side gas
        # temperature is only the starting point -- every real pipe
        # segment downstream (primary, collector, cat, muffler,
        # tailpipe) sheds real convective heat off its own real surface
        # area as the gas actually travels through it (engines.
        # ExhaustSystem.segment_outlet_temps_k), each segment's outlet
        # feeding the next one's inlet. A stock boxed-in system's
        # tailpipe genuinely runs cooler than an open header's -- not a
        # per-header-type fudge, the real consequence of more real pipe
        # surface area to shed heat through.
        exhaust_segment_temps_k = eng.exhaust_system.segment_outlet_temps_k(
            self.state.exhaust_temp_k, self._exhaust_demand_kg_s, EXHAUST_TEMP_AMBIENT_K)
        self.state.exhaust_tailpipe_temp_k = (
            exhaust_segment_temps_k[-1] if exhaust_segment_temps_k else self.state.exhaust_temp_k)
        self.state.exhaust_pressure_frac = self._drivetrain_out.get("exhaust_backpressure_frac", 0.0)
        self.state.coolant_temp_k = self._drivetrain_out.get("coolant_temp_k", EXHAUST_TEMP_AMBIENT_K)
        self._step_air_and_emissions(dt)
        self.state.coolant_flow_lpm = self._drivetrain_out.get("coolant_flow_lpm", 0.0)

        # Real per-cylinder block-metal temperature: a simple lumped
        # thermal-mass balance per cylinder bay, not a fabricated
        # gradient. Heat in is THIS cylinder's own real share of the
        # engine's total waste heat, split by its own actual firing
        # strength this cycle (self._last_strength, the same real
        # per-cylinder combustion signal engine_cycle_sim already
        # tracks for knock/misfire) -- a cylinder firing harder
        # genuinely runs its own bay hotter, not an even split. Heat out
        # is the real Newton's-law convective loss to the coolant jacket
        # (engines.convective_heat_loss_w, the same primitive the
        # exhaust/fuel thermal chains already use), off a real disclosed
        # per-cylinder wetted surface area derived from this engine's
        # own displacement-per-cylinder. Lateral conduction to each
        # immediate neighbor bay (real: cast-iron/aluminum block metal
        # conducts along its own length) is what actually lets a
        # localized hot cylinder visibly warm the ones next to it
        # instead of every bay evolving in total isolation.
        n_cyl = max(eng.architecture.cylinders, 1)
        if len(self.state.cylinder_block_temps_k) != n_cyl:
            self.state.cylinder_block_temps_k = [self.state.coolant_temp_k] * n_cyl
        total_strength = sum(self._last_strength.values())
        if total_strength > 1e-6 and eng.architecture.cylinders:
            # the metal's own heat equation, parametric in THIS engine's
            # real geometry and material (see _block_thermal_params):
            # wetted liner+head area, jacket film coefficient, per-bay
            # metal heat capacity, and bay-to-bay conduction k*A/L
            (block_surface_area_per_cyl_m2, BLOCK_COOLANT_HTC_W_PER_M2K,
             BLOCK_THERMAL_MASS_J_PER_K_PER_CYL, BLOCK_LATERAL_CONDUCTION_W_PER_K) = self._block_thermal_params()
            waste_heat_w = self._waste_heat_kw * 1000.0
            old_temps = self.state.cylinder_block_temps_k
            new_temps = list(old_temps)
            # Indexed by real PHYSICAL cylinder number (1..n_cyl, the
            # same numbering engine_geometry.cylinder_sites positions
            # the mesh with), not firing-order sequence -- iterating
            # firing_order directly would store cylinder 4's heat at
            # array slot 1 for a 1-4-2-6-3-5 V8, misaligned with every
            # mesh segment this drives.
            for idx in range(n_cyl):
                cyl = idx + 1
                strength_frac = self._last_strength.get(cyl, 0.0) / total_strength
                heat_in_w = waste_heat_w * strength_frac
                heat_out_w = convective_heat_loss_w(
                    block_surface_area_per_cyl_m2, old_temps[idx], self.state.coolant_temp_k,
                    htc_w_per_m2k=BLOCK_COOLANT_HTC_W_PER_M2K)
                lateral_w = 0.0
                if idx > 0:
                    lateral_w += BLOCK_LATERAL_CONDUCTION_W_PER_K * (old_temps[idx - 1] - old_temps[idx])
                if idx < n_cyl - 1:
                    lateral_w += BLOCK_LATERAL_CONDUCTION_W_PER_K * (old_temps[idx + 1] - old_temps[idx])
                net_w = heat_in_w - heat_out_w + lateral_w
                new_temps[idx] = max(self.state.coolant_temp_k - 5.0,
                                      old_temps[idx] + net_w * dt / BLOCK_THERMAL_MASS_J_PER_K_PER_CYL)
            self.state.cylinder_block_temps_k = new_temps
        # real evaporative + sensible charge cooling from the fuel
        # itself (self._fuel_cooling_kw, computed above from the real
        # fuel's own latent heat + how much colder than ambient it is),
        # applied as the real flow-through energy balance a charge-
        # cooling calculation actually is: cooling power divided by the
        # fresh air mass flow's own real heat capacity. Clamped to a
        # real plausible upper bound (60K) since air mass flow genuinely
        # can be near zero at idle/cranking, where this ratio isn't
        # meaningful -- not a claim that charge cooling stops existing
        # then, just that this instantaneous-flow estimate breaks down
        # at near-zero flow the same way any per-flow intensive figure
        # does.
        AIR_SPECIFIC_HEAT_J_PER_KGK = 1005.0
        # STAGE 1: the runner heat-soak exhaust already models
        # (segment_outlet_temps_k) but intake never had -- air genuinely
        # picks up real heat convecting down the runner toward the
        # engine bay it's routed through, via the runner's own real
        # surface area. That surrounding air temperature isn't a fixed
        # constant -- it's conductively coupled to the block's OWN real,
        # LIVE current state (self.state.coolant_temp_k when a real
        # coolant circuit exists, oil_temp_k as the real block-proximity
        # proxy on an air-cooled engine that has neither) -- a cold-
        # started engine's runner sits next to a still-cold block and
        # barely warms the charge at all; a fully-warmed or overheating
        # block genuinely cooks it. RUNNER_BLOCK_CONDUCTION_FRAC is the
        # one real, disclosed constant here: the runner's surrounding
        # air doesn't reach full block temperature (there's a real air
        # gap, not solid contact), just a real, partial pull toward it.
        if eng.accessories.water_pump:
            block_temp_k = self.state.coolant_temp_k
        elif not eng.architecture.two_stroke:
            block_temp_k = self.state.oil_temp_k
        else:
            block_temp_k = ENGINE_BAY_AMBIENT_K
        # the air the filter actually draws (air_volumes): hot under-hood
        # air for a bay filter, the room/outside for a cold-air box
        inlet_air_k = getattr(self, "_intake_source_temp_k", EXHAUST_TEMP_AMBIENT_K)
        runner_surround_temp_k = inlet_air_k + (
            block_temp_k - inlet_air_k) * RUNNER_BLOCK_CONDUCTION_FRAC
        runner_temp_k = eng.intake_system.runner_outlet_temp_k(
            inlet_air_k, self._intake_demand_kg_s, runner_surround_temp_k)
        self.state.intake_runner_temp_k = runner_temp_k
        plenum_temp_k = (self._drivetrain_out.get("intake_charge_temp_k", EXHAUST_TEMP_AMBIENT_K)
                          + (runner_temp_k - EXHAUST_TEMP_AMBIENT_K))
        # STAGE 2: real evaporative + sensible charge cooling from the
        # fuel itself (self._fuel_cooling_kw, computed above from the
        # real fuel's own latent heat + how much colder than ambient it
        # is), applied as the real flow-through energy balance a
        # charge-cooling calculation actually is: cooling power divided
        # by the fresh air mass flow's own real heat capacity. Clamped
        # to a real plausible upper bound (120K) since air mass flow
        # genuinely can be near zero at idle/cranking, where this ratio
        # isn't meaningful -- not a claim that charge cooling stops
        # existing then, just that this instantaneous-flow estimate
        # breaks down at near-zero flow the same way any per-flow
        # intensive figure does. Real methanol/nitro drag engines
        # genuinely report extreme charge cooling (manifold icing is a
        # known, real symptom), well beyond a gasoline engine's much
        # more modest real evaporative cooling.
        charge_cooling_delta_k = min(120.0, ((self._fuel_cooling_kw + self._auxiliary_injection_cooling_kw) * 1000.0) / max(
            self._intake_demand_kg_s * AIR_SPECIFIC_HEAT_J_PER_KGK, 1e-3))
        self.state.intake_charge_temp_k = max(150.0, plenum_temp_k - charge_cooling_delta_k)
        self.state.oil_pressure_pa = self._drivetrain_out.get("oil_pressure_pa", 101_325.0)
        # real demand-vs-capacity: how much the intake/exhaust ports are
        # actually being asked to pass right now against their own real
        # flow_capacity_kg_s ceiling -- >1.0 means genuinely choked this
        # tick, the actual real-time cause behind any MAP droop/backpressure
        # rather than just the pressure symptom those already report
        intake_capacity = self._drivetrain_out.get("intake_flow_capacity_kg_s", 0.0)
        self.state.intake_flow_demand_frac = (
            self._intake_demand_kg_s / intake_capacity) if intake_capacity > 0.0 else 0.0
        exhaust_capacity = self._drivetrain_out.get("exhaust_flow_capacity_kg_s", 0.0)
        self.state.exhaust_flow_demand_frac = (
            self._exhaust_demand_kg_s / exhaust_capacity) if exhaust_capacity > 0.0 else 0.0
        self.state.oil_temp_k = self._drivetrain_out.get("oil_temp_k", EXHAUST_TEMP_AMBIENT_K)
        self.state.oil_flow_lpm = self._drivetrain_out.get("oil_flow_lpm", 0.0)
        self.state.ac_compressor_load_w = self._drivetrain_out.get("ac_compressor_load_w", 0.0)

    def _step_electric(self, dt: float) -> None:
        eng = self.engine
        # this engine kind's real "starting system" isn't a cranking
        # motor spinning up a combustion engine -- it's a contactor
        # closing and the inverter energizing, a real event with
        # essentially no mechanical delay (a direct-drive electric
        # motor makes full torque from a dead stop, no warm-up phase to
        # wait through). engage_starter() still exists as the one real
        # user action that takes this unit from off to ready; it just
        # catches instantly here instead of motoring a crank over.
        if self.starter.engaged:
            self.starter.release()
            self.state.cranking = False
            self.state.stalled = False
        self._update_dyno_controllers(dt)
        # the real quick-shift/dyno-pull sequencers (clutch-out -> gear
        # change -> clutch-in; the WOT inertia-dyno pull) only actually
        # advance if stepped every tick -- previously only the piston
        # combustion loop did this, so pressing shift or starting a pull
        # on this engine kind set the request up and then silently never
        # progressed it (found via a real report: shifting "not working"
        # on the Otto-Langen surfaced the exact same missing call there)
        self._update_quick_shift(dt)
        self._update_dyno_pull(dt)
        rpm_frac = max(self.state.rpm, 0.0) / max(eng.redline_rpm, 1.0)
        self.state.manifold_pressure_frac = self.throttle
        self.state.ignition_timing_deg = 0.0
        # an electric drive unit's traction bus is production's own
        # energy.storage pack, not this 12 V accessory network -- not
        # modeled here; the accessory bus just reads nominal
        self.state.battery_voltage = 12.6
        self.state.spark_energy_frac = 1.0
        self.state.alternator_output_frac = 1.0
        self.state.battery_soc_frac = 1.0
        self.state.knock_flag = False
        self.state.knock_intensity = 0.0
        self.state.misfire_flag = False
        self.state.valve_float_flag = False
        self.state.valve_float_risk = 0.0
        self.state.turbo_spool_frac = 0.0
        self.state.boost_frac = 0.0
        self.state.wastegate_flutter = False
        self.state.surge_flag = False
        self.state.backfire_flag = False
        self.state.backfire_kind = None

        # NOT torque_fraction -- that curve models real combustion
        # quality ramping with rpm (piston-engine-specific physics,
        # zero at rpm==0 by design). An electric motor's real curve is
        # the opposite shape: peak torque from a dead stop, roll-off
        # only above a real base speed -- see electric_torque_fraction's
        # own docstring. Reusing the combustion curve here meant this
        # engine kind was mathematically unable to move from a genuine
        # standstill (zero torque forever at rpm==0), invisible only
        # because the old start() always teleported straight to idle_
        # rpm and never actually left it there at rest.
        frac = electric_torque_fraction(eng, self.state.rpm)
        torque_nm = eng.peak_torque_nm * frac * self.throttle
        # engine_brake_enabled doubles as regen-braking on/off here -- real
        # EVs let you toggle one-pedal/regen deceleration the same way;
        # off is a genuine freewheel/coast with no motor drag at all
        regen_loss = eng.peak_braking_torque_nm * frac * (1.0 - self.throttle) * 0.15
        pumping_loss = regen_loss if self.engine_brake_enabled else 0.0
        net_torque = torque_nm - pumping_loss - self.brake_load_nm
        domega = net_torque / max(eng.inertia_kg_m2, 1e-9) * dt
        self._omega = max(0.0, self._omega + domega)
        new_rpm = self._omega / RPM_TO_RAD_S
        if new_rpm <= 0.5:
            self.state.rpm = 0.0
            self._omega = 0.0
        else:
            # a real, honest value -- not min(new_rpm, redline*1.12), an
            # arbitrary reporting ceiling that used to hide here. It
            # capped only the DISPLAYED/consumed rpm, not the crank's
            # real self._omega (already uncapped, real torque/inertia
            # integration) -- but state.rpm feeds torque_fraction,
            # valve_float_risk, knock risk, ignition timing, and MAP,
            # so once the true crank speed exceeded 1.12x redline this
            # silently froze every one of them at that arbitrary point
            # forever, regardless of what the real physics said,
            # defeating the rev-limiter-disable feature outright: an
            # engine could never be observed exceeding 1.12x redline no
            # matter how much real torque was actually available.
            self.state.rpm = new_rpm
        self.state.current_torque_nm = net_torque + pumping_loss + self.brake_load_nm
        self.state.power_kw = self.state.current_torque_nm * self._omega / 1000.0
        rms_alpha = min(1.0, dt / RMS_TAU_S)
        self._torque_ms += (self.state.current_torque_nm ** 2 - self._torque_ms) * rms_alpha
        self._power_ms += (self.state.power_kw ** 2 - self._power_ms) * rms_alpha
        self.state.torque_rms_nm = math.sqrt(max(0.0, self._torque_ms))
        self.state.power_rms_kw = math.sqrt(max(0.0, self._power_ms))

    def _record_ignition(self, dt: float, fired: bool) -> None:
        """A real, general per-engine-kind firing-rate measurement --
        EVERY cylinder/kind that has a real discrete combustion event
        feeds this the same way (the piston combustion loop and
        _step_atmospheric both call it), not a bespoke per-kind field.
        See EngineCycleState.real_fire_hz's own docstring for why this
        exists instead of deriving a rate from rpm *
        firing_events_per_rev -- that relation is only real for a
        crank engine in the first place, and even there this is the
        actually-measured rate (naturally reflecting misfires/rev-
        limiter cuts/governor misses), not an assumed one."""
        self._time_since_last_ignition_s += dt
        if fired:
            self.state.real_fire_hz = 1.0 / max(self._time_since_last_ignition_s, 1e-6)
            self._time_since_last_ignition_s = 0.0
        else:
            self.state.real_fire_hz = min(
                self.state.real_fire_hz, 1.0 / max(self._time_since_last_ignition_s, 1e-6))

    def _clutch_substep_plan(self, dt: float, engine_side_inertia_kg_m2: float) -> tuple[int, float]:
        """The exact same real numerical-stability requirement
        DrivetrainSolver computes for its own stiff spring-integrated
        graph edges (drivetrain_graph.py's STABILITY_MARGIN /
        _compute_stable_sub_dt): the dyno/gearbox clutch junction
        (drivetrain_port.ClutchPort) is that same kind of stiff spring,
        just stepped directly here instead of as a graph edge, and it
        needs the identical treatment. Stepping it at the full outer
        dt (found empirically, on the turbine's own dyno-pull probe)
        made the dyno reading freeze at a nonphysical, wildly inflated
        constant torque instead of settling -- the classic sign of an
        explicit-Euler spring integrated past its own stability limit.
        Neutral (ratio<=0) never touches the junction at all, so it
        needs none of this."""
        ratio = self._current_gear_ratio()
        if ratio <= 0.0:
            return 1, dt
        # the load's own real inertia, reflected into the SAME angular
        # frame the junction compares against (self._load_omega*ratio,
        # not self._load_omega itself) -- reflecting an inertia through
        # a speed-multiplying ratio divides it by ratio^2, the standard
        # real kinetic-energy-preserving reflection (this is why a
        # light dyno drum behind a low first gear is the stiffest,
        # fastest-natural-frequency case: it looks tiny and fast in the
        # engine's own frame)
        reflected_load_inertia = self._dyno_inertia_kg_m2 / max(ratio * ratio, 1e-9)
        reduced_inertia = max(min(engine_side_inertia_kg_m2, reflected_load_inertia), 1e-9)
        omega_n = math.sqrt(self._brake_junction.stiffness_nm_per_rad_s / reduced_inertia)
        if omega_n <= 0.0:
            return 1, dt
        stable_sub_dt = DrivetrainSolver.STABILITY_MARGIN / omega_n
        n_sub = min(DrivetrainSolver.SUBSTEP_CAP, max(1, math.ceil(dt / stable_sub_dt)))
        return n_sub, dt / n_sub

    def _step_turbine(self, dt: float) -> None:
        """A real single-shaft gas turbine (gas_turbine.py) -- no
        crank, no piston loop, but the SAME real rig every other engine
        in this catalogue sits on: the drivetrain graph's own fuel
        tank/pump/rail circuit (widened to admit `kind == "turbine"` --
        see drivetrain_graph.py), and the SAME clutch-pedal/gearbox/
        dyno-absorber chain (self._brake_junction / self._load_omega /
        self._current_gear_ratio()) the piston combustion loop uses,
        just stepped once per tick instead of in a fine substep loop --
        nothing here has individual firing pulses fast enough to need
        that resolution. self._omega is the OUTPUT shaft (post-
        reduction-gearbox) -- the same real "flywheel" speed every
        other engine's clutch/gearbox/dyno chain already assumes."""
        eng = self.engine
        ts = eng.turbine
        turb = self._turbine
        self._update_dyno_controllers(dt)
        # see _step_electric's own comment: these sequencers only
        # advance if stepped every tick
        self._update_quick_shift(dt)
        self._update_dyno_pull(dt)
        reduction = max(ts.reduction_ratio, 1e-6)
        load_inertia_kg_m2 = self._dyno_inertia_kg_m2

        # the real fuel-fired belt-driven alternator (accessories.
        # alternator=True) and the real onboard jet-fuel tank/pump/rail
        # -- both live in the SAME drivetrain graph every other engine
        # uses, stepped with last tick's own real fuel flow as the
        # circuit's demand (the same one-tick-lag convention every
        # other cross-system reading in this sim already uses). The
        # dyno's own graph edges are disabled here exactly like the
        # piston loop disables them -- the real coupling is solved
        # below, off this same graph's own dyno_absorber mass/inertia,
        # not duplicated inside the graph solver too.
        self._drivetrain_out = self._drivetrain.step(
            dt, self._omega, self.electrical.reading.alternator_shaft_load_w,
            fuel_demand_kg_s=turb.fuel_flow_kg_s,
            disabled_edges=frozenset({"dyno_friction_clutch", "dyno_roller_contact"}))
        accessory_reaction_nm = self._drivetrain_out.get("crank_reaction_torque_nm", 0.0)

        # the SAME real clutch/gearbox/dyno-absorber chain the piston
        # loop uses: brake_load_nm never touches this shaft directly --
        # it's a resistive torque on the dyno's own load shaft, coupled
        # here through the real clutch port (ClutchPort, the driver's
        # pedal doubling as the dyno's own coupling) and, when a gear is
        # selected, the transmission's real ratio
        self._brake_junction.engagement = self.clutch_frac
        effective_ratio = self._current_gear_ratio()
        # the SAME real starter torque already computed this tick
        # (self._crank_assist_nm, from starter.py off this engine's own
        # declared starting_systems) -- reused rather than inventing a
        # second, unaccounted motoring torque: whatever energy the
        # starter/battery accounting already charged for is exactly
        # what accelerates this shaft, on the gas-generator side of the
        # gearbox (a real starter motors the gas generator, not the
        # slow output shaft)
        starter_assist_gg_nm = self._crank_assist_nm * reduction

        # the clutch junction is a stiff spring (ClutchPort) and needs
        # the same finely-subdivided stepping the piston combustion
        # loop already gives it -- see _clutch_substep_plan's own
        # docstring for what stepping it at the full outer dt does
        n_sub, sub_dt = self._clutch_substep_plan(dt, ts.shaft_inertia_kg_m2 / (reduction * reduction))
        junction_torque = 0.0
        dyno_torque_nm = 0.0
        for _ in range(n_sub):
            if effective_ratio > 0.0:
                junction_torque = self._coupled_junction_torque(
                    sub_dt, self._omega, self._load_omega * effective_ratio)
            else:
                junction_torque = 0.0
            output_load_nm = junction_torque + accessory_reaction_nm
            # the output-shaft load (dyno clutch junction + accessory
            # reaction) reflected onto the faster gas-generator shaft --
            # a real gearbox torque transform, power conserved:
            # T_gg * omega_gg = T_out * omega_out
            load_gg_nm = output_load_nm / reduction
            turb.step(sub_dt, self.throttle, load_torque_nm=load_gg_nm,
                      starter_assist_torque_nm=starter_assist_gg_nm)
            # Turbines have no crank, but their gas-generator shaft still
            # has a real angular position.  Keep it on the existing public
            # phase channel so baked rotor geometry is driven by the cycle
            # solve instead of remaining frozen at frame zero.
            self._total_crank_deg += (
                turb.omega_rad_s * 180.0 / math.pi * sub_dt)
            # NOT snapped to zero below some display rpm threshold -- the
            # only real mechanical floor this rotor has is that it can't
            # spin backward (a compressor/turbine wheel doesn't run in
            # reverse), which max(0.0, ...) alone already enforces
            self._omega = max(0.0, turb.omega_rad_s / reduction)

            if effective_ratio > 0.0:
                dyno_torque_nm = junction_torque * effective_ratio
                domega_load = (dyno_torque_nm - self.brake_load_nm) / load_inertia_kg_m2 * sub_dt
            else:
                dyno_torque_nm = 0.0
                domega_load = -self.brake_load_nm / load_inertia_kg_m2 * sub_dt
            # the dyno drum's own real mechanical floor -- a passive
            # absorber only removes energy, it can't drive its own drum
            # backward
            self._load_omega = max(0.0, self._load_omega + domega_load)

        self.state.rpm = self._omega / RPM_TO_RAD_S
        self.state.crank_angle_deg = self._total_crank_deg % 360.0
        self.state.dyno_rpm = self._load_omega / RPM_TO_RAD_S
        self.state.dyno_absorbed_kw = self.brake_load_nm * self._load_omega / 1000.0
        self.state.dyno_kinetic_energy_j = 0.5 * self._dyno_inertia_kg_m2 * self._load_omega * self._load_omega
        self.state.dyno_torque_nm = dyno_torque_nm
        self.state.brake_clutch_locked = self._brake_junction.locked

        # the real cockpit gauges this engine actually has, reusing the
        # existing HUD-compatible fields other engine kinds already
        # populate for the same physical quantity (spool-up fraction,
        # pressure rise, exhaust gas temperature)
        self.state.turbo_spool_frac = turb.n1_frac
        self.state.boost_frac = max(0.0, turb.pressure_ratio - 1.0) * self._effects.boost_factor
        self.state.exhaust_temp_k = turb.egt_k
        self.state.manifold_pressure_frac = min(1.0, turb.n1_frac)
        self.state.ignition_timing_deg = 0.0

        # the real 12/24 V accessory bus -- the SAME real circuit solve
        # (electrical_network.ElectricalNetwork.step) the piston loop
        # uses, not a nominal-voltage stand-in. The engine's own
        # ignition_profile ("light-off-igniter", see engines.py's
        # IGNITION_PROFILE_DISPATCH) already makes ignition_load_w
        # correctly return 0 -- a turbine's igniter is a one-shot
        # start-only device, not a continuous per-rev coil draw -- and
        # fuel_pump_load_w is real (this engine's own declared electric
        # pump). ecu_powered_draw_w=0.0: a disclosed simplification,
        # this APU-class turbine's fuel control is hydromechanical
        # (this engine's own real N1 governor lives in gas_turbine.py,
        # not on this 12/24 V bus), not a FADEC.
        loads_w = (self.electrical.ignition_load_w(self.state.rpm, False)
                   + self.electrical.fuel_pump_load_w(turb.fuel_flow_kg_s)
                   + self.electrical.load_resistor_w(self.electrical_load_frac)
                   + self.starter.reading.bus_current_a * max(self.state.battery_voltage, 1.0))
        alternator_omega = self._drivetrain.omega.get("electrical.alternator", 0.0)
        bus = self.electrical.step(dt, alternator_omega, loads_w, ecu_powered_draw_w=0.0)
        self.state.battery_voltage = bus.voltage_v
        self.state.battery_soc_frac = bus.battery_soc_frac
        self.state.alternator_current_a = bus.alternator_current_a
        self.state.electrical_load_w = bus.load_w
        self.state.bus_regulating = bus.regulating
        self.state.alternator_output_frac = (
            bus.alternator_current_a / self.electrical.alternator.rated_current_a
            if self.electrical.alternator.rated_w > 0.0 else 0.0)
        self.state.spark_energy_frac = 1.0
        self.state.knock_flag = False
        self.state.knock_intensity = 0.0
        self.state.misfire_flag = False
        self.state.valve_float_flag = False
        self.state.valve_float_risk = 0.0
        self.state.wastegate_flutter = False
        self.state.surge_flag = False
        self.state.backfire_flag = False
        self.state.backfire_kind = None
        # the real drivetrain graph's own fuel circuit (tank/pump/rail,
        # the same real depletable-reservoir mechanism every liquid-
        # fueled engine in this catalogue already uses) -- not a
        # separate hand-rolled density/burn tracker
        self.state.fuel_fill_frac = self._drivetrain_out.get("fuel_fill_frac", 1.0)
        self.state.fuel_temp_k = self._drivetrain_out.get("fuel_temp_k", REFERENCE_INTAKE_TEMP_K)

        # the real recomputed gas-generator torque balance this tick
        # (turbine_work_j_per_kg/compressor_work_j_per_kg/mdot_kg_s are
        # the exact same published fields step() just derived and used
        # internally), reflected through the gearbox to report the real
        # OUTPUT shaft torque/power -- not the gas-generator's own
        # internal net accelerating torque, which is a different number
        omega_for_torque = max(turb.omega_rad_s, 50.0)
        gas_power_w = turb.mdot_kg_s * (turb.turbine_work_j_per_kg - turb.compressor_work_j_per_kg)
        friction_nm = math.copysign(turb.bearing_friction_nm, turb.omega_rad_s) if turb.omega_rad_s != 0.0 else 0.0
        gg_available_nm = gas_power_w / omega_for_torque - friction_nm
        self.state.current_torque_nm = gg_available_nm * reduction
        self.state.power_kw = self.state.current_torque_nm * self._omega / 1000.0
        rms_alpha = min(1.0, dt / RMS_TAU_S)
        self._torque_ms += (self.state.current_torque_nm ** 2 - self._torque_ms) * rms_alpha
        self._power_ms += (self.state.power_kw ** 2 - self._power_ms) * rms_alpha
        self.state.torque_rms_nm = math.sqrt(max(0.0, self._torque_ms))
        self.state.power_rms_kw = math.sqrt(max(0.0, self._power_ms))

    def prime_gasholder(self, fill_level_frac: float, composition_frac: float) -> None:
        """Set the real gasholder's starting contents (atmospheric
        engines with a declared gasholder only). The default is a
        running plant: full of pure gas. A cold NEW plant is
        (0.0, 0.0): bell seated, the residual space under it holding
        plain air -- which the generator then has to physically purge
        through the flammable band (state.gasholder_flammable_hazard)
        before the engine can draw a charge that fires at all."""
        for c in self._drivetrain.fluid_circuits:
            if (c.kind_class == "high-pressure-liquid-supply" and c.bottle_capacity_kg > 0.0
                    and any("fuel" in nid for nid in c.nodes)):
                c.fill_level_frac = max(0.0, min(1.0, fill_level_frac))
                c.composition_frac = max(0.0, min(1.0, composition_frac))
                return
        raise ValueError(f"{self.engine.identity}: no depletable fuel source to prime "
                         "(no gasholder capacity and no fuel network with a vessel)")

    @property
    def expander_drain_cocks_open(self) -> bool:
        return self._expander.drain_cocks_open if self._expander is not None else False

    @expander_drain_cocks_open.setter
    def expander_drain_cocks_open(self, open_: bool) -> None:
        if self._expander is None:
            raise ValueError(f"{self.engine.identity} has no expander cylinder bank")
        self._expander.drain_cocks_open = bool(open_)

    def _step_expander(self, dt: float) -> None:
        """A cutoff expander bank (expander.py) on whatever its declared
        fuel network supplies -- a boiler's dome or a charged receiver
        (fuel_network.py). No combustion, no idle: the throttle is the
        regulator/throttle valve setting chest pressure between
        atmospheric and the line's supply, and the engine starts from
        rest under load and stops dead without it. Rigged to the SAME
        clutch/gearbox/dyno chain every other engine kind sits on."""
        eng = self.engine
        bank = self._expander
        rt = self._fuel_network
        fluid = rt.fluid
        self._update_dyno_controllers(dt)
        self._update_quick_shift(dt)
        self._update_dyno_pull(dt)
        load_inertia_kg_m2 = self._dyno_inertia_kg_m2

        # last tick's actual draw is this tick's demand on the line --
        # the same one-tick lag every cross-system reading here uses
        demand_kg_s = bank.last_mass_kg_s
        net_tick = self._step_fuel_network(dt, demand_kg_s)
        self._drivetrain_out = self._drivetrain.step(
            dt, self._omega, 0.0,
            fuel_demand_kg_s=demand_kg_s, fuel_production_kg_s=net_tick.production_kg_s,
            fuel_production_composition_frac=net_tick.production_composition_frac,
            fuel_valve_open=net_tick.valve_open,
            disabled_edges=frozenset({"dyno_friction_clutch", "dyno_roller_contact"}))
        accessory_reaction_nm = self._drivetrain_out.get("crank_reaction_torque_nm", 0.0)
        availability = self._drivetrain_out.get("fuel_availability_frac", 1.0)
        supply_pa = net_tick.supply_pressure_pa
        # the throttle valve: chest pressure between atmospheric and what
        # the line actually holds right now -- the network's own supply
        # pressure already follows the dome's/receiver's fill (an
        # emptying vessel sags continuously), so no separate
        # availability term is layered on here
        chest_pa = OTTO_P_ATM_PA + (supply_pa - OTTO_P_ATM_PA) * max(0.0, min(1.0, self.throttle))
        # a real choke on the line (a regulator's or port's rated flow,
        # fuel_network's flow ceiling): when the bank wants more than
        # the choke passes, chest pressure falls until the draw matches
        # -- mass through a choke scales with pressure, so the chest
        # sags in proportion. This is what actually caps an unloaded
        # expander's speed on a real line, not a separate rev limit.
        line_cap = net_tick.flow_capacity_kg_s
        if line_cap > 0.0 and demand_kg_s > line_cap:
            chest_pa = OTTO_P_ATM_PA + (chest_pa - OTTO_P_ATM_PA) * (line_cap / demand_kg_s)
        is_steam = fluid.name == "steam"
        if is_steam:
            supply_t_k, _latent = _steam_properties(max(chest_pa, OTTO_P_ATM_PA))
        else:
            supply_t_k = OTTO_T_ATM_K
        supply_rho = chest_pa / (fluid.gas_constant_j_per_kgk * max(supply_t_k, 1.0))
        cutoff = rt.spec.admission.cutoff_frac
        direction = -1.0 if self.gear_index < 0 else 1.0
        # a real dryer/separator on the line (fuel_network.CoolerFilter)
        # is what keeps an air engine's exhaust port from icing
        dry_supply = any(getattr(st, "kind", "") == "cooler-filter" for st in rt.spec.stages)

        self._brake_junction.engagement = self.clutch_frac
        effective_ratio = self._current_gear_ratio()
        n_sub, sub_dt = self._clutch_substep_plan(dt, eng.inertia_kg_m2)
        junction_torque = 0.0
        dyno_torque_nm = 0.0
        shaft_load_nm = 0.0
        engine_torque_nm = 0.0
        for _ in range(n_sub):
            if effective_ratio > 0.0:
                junction_torque = self._brake_junction.step(sub_dt, self._omega, self._load_omega * effective_ratio)
            else:
                junction_torque = 0.0
            shaft_load_nm = junction_torque + accessory_reaction_nm
            engine_torque_nm, _mass = bank.step(sub_dt, self._omega, chest_pa, supply_t_k, supply_rho,
                                                fluid.gamma, is_steam, cutoff_frac=cutoff, direction=direction,
                                                dry_supply=dry_supply)
            domega = (engine_torque_nm - shaft_load_nm) / eng.inertia_kg_m2 * sub_dt
            # an expander can be driven backward by its load only until
            # it stalls -- and it genuinely starts from zero, so no idle
            # floor and no rpm clamp beyond "not driven negative by a
            # load it can't hold" (the reversing link is `direction`)
            self._omega = max(0.0, self._omega + domega)
            if effective_ratio > 0.0:
                dyno_torque_nm = junction_torque * effective_ratio
                domega_load = (dyno_torque_nm - self.brake_load_nm) / load_inertia_kg_m2 * sub_dt
            else:
                dyno_torque_nm = 0.0
                domega_load = -self.brake_load_nm / load_inertia_kg_m2 * sub_dt
            self._load_omega = max(0.0, self._load_omega + domega_load)

        self.state.rpm = self._omega / RPM_TO_RAD_S
        self.state.dyno_rpm = self._load_omega / RPM_TO_RAD_S
        self.state.dyno_absorbed_kw = self.brake_load_nm * self._load_omega / 1000.0
        self.state.dyno_kinetic_energy_j = 0.5 * self._dyno_inertia_kg_m2 * self._load_omega * self._load_omega
        self.state.dyno_torque_nm = dyno_torque_nm
        self.state.brake_clutch_locked = self._brake_junction.locked
        self.state.current_torque_nm = engine_torque_nm
        self.state.power_kw = engine_torque_nm * self._omega / 1000.0
        rms_alpha = min(1.0, dt / RMS_TAU_S)
        self._torque_ms += (self.state.current_torque_nm ** 2 - self._torque_ms) * rms_alpha
        self._power_ms += (self.state.power_kw ** 2 - self._power_ms) * rms_alpha
        self.state.torque_rms_nm = math.sqrt(max(0.0, self._torque_ms))
        self.state.power_rms_kw = math.sqrt(max(0.0, self._power_ms))
        self.state.manifold_pressure_frac = chest_pa / max(supply_pa, OTTO_P_ATM_PA)
        self.state.expander_chest_pressure_pa = chest_pa
        self.state.expander_mep_pa = bank.last_mep_pa
        self.state.expander_cylinder_temp_k = bank.cylinder_metal_temp_k
        self.state.expander_ice_frac = bank.ice_frac
        self.state.expander_drain_cocks_open = bank.drain_cocks_open
        self.state.exhaust_temp_k = bank.last_exhaust_temp_k
        self.state.exhaust_tailpipe_temp_k = bank.last_exhaust_temp_k
        self.state.fuel_fill_frac = self._drivetrain_out.get("fuel_fill_frac", 1.0)
        self.state.fuel_availability_frac = availability
        self.state.ignition_timing_deg = 0.0
        self.state.battery_voltage = self.electrical.nominal_voltage_v
        self.state.spark_energy_frac = 1.0
        self.state.alternator_output_frac = 0.0
        self.state.battery_soc_frac = 1.0
        for flag in ("knock_flag", "misfire_flag", "valve_float_flag", "wastegate_flutter", "surge_flag",
                     "backfire_flag"):
            setattr(self.state, flag, False)
        self.state.knock_intensity = 0.0
        self.state.valve_float_risk = 0.0
        self.state.turbo_spool_frac = 0.0
        self.state.boost_frac = 0.0
        self.state.backfire_kind = None
        # a cracked cylinder (water hammer) or a failed boiler is a real,
        # permanently-down state -- the same "stalled, needs real
        # intervention" every other kind reports
        if bank.destroyed or net_tick.boiler_failed:
            self.state.stalled = True

    def _step_atmospheric(self, dt: float) -> None:
        """The real Otto-Langen free-piston cylinder (otto_langen.py),
        governed by its own real captive-ball governor (governor.py) --
        no crank, no throttle plate: the mixer setting is fixed wide
        open (the real historical mechanism) and the governor alone
        decides, once per cycle, whether the next charge fires at all.
        Rigged to the SAME real clutch/gearbox/dyno-absorber chain
        every other engine in this catalogue sits on (self._brake_
        junction / self._load_omega / self._current_gear_ratio()) --
        brake_load_nm never touches the flywheel directly, same real
        reasoning as the piston loop's own comment on that junction."""
        eng = self.engine
        cyl = self._atmospheric_cyl
        gov = self._atmo_governor
        self._update_dyno_controllers(dt)
        # see _step_electric's own comment: these sequencers only
        # advance if stepped every tick -- this is the actual bug
        # behind "shifting isn't working" on this engine
        self._update_quick_shift(dt)
        self._update_dyno_pull(dt)
        load_inertia_kg_m2 = self._dyno_inertia_kg_m2

        # the same drivetrain graph every other engine steps -- no
        # accessories fitted on this real 1867-era build (Accessories
        # all False for it) so accessory_reaction_nm comes back ~0, but
        # it's the same real call path, not skipped as a special case.
        #
        # fuel_demand_kg_s: this cylinder's own real per-cycle charge
        # mass (its declared ignition-height volume at real atmospheric
        # density -- air's own R as a disclosed stand-in for the gas/
        # air mixture's exact composition, same "magnitude-correct, not
        # one exact assay" convention this toy already uses elsewhere)
        # times how often it's actually firing right now (real_fire_hz).
        # fuel_production_kg_s: whatever this engine's own on-site
        # generator (see atmospheric_generator's own docstring) is
        # making this tick, using this SAME tick's live rpm as its real
        # belt-speed signal -- both 0.0 with no generator declared, so
        # this changes nothing for a piped-main engine.
        atmo_spec = eng.atmospheric
        fuel_profile = atmo_spec.fuel_profile
        gas_props = fuel_gas_properties(fuel_profile)
        mixer_gas_frac = atmo_spec.mixer_gas_volume_fraction_resolved
        if self._fuel_network is not None and self._fuel_network.mixer_gas_volume_fraction() is not None:
            mixer_gas_frac = self._fuel_network.mixer_gas_volume_fraction()
        charge_volume_m3 = cyl.piston_area_m2 * atmo_spec.ignition_height_m
        # Only the GAS port's share of each charge is drawn from the gas
        # line -- the slide valve's real fixed port ratio (mixer_gas_
        # frac, see otto_langen.MIXER_DESIGN_EQUIVALENCE_RATIO); the
        # rest is room air the piston's own vacuum pulls in directly and
        # no reservoir has to supply. Demand is that gas volume at the
        # fuel's own real density, times how often the engine is
        # actually firing right now (real_fire_hz). This is roughly a
        # tenth of the whole-charge mass the earlier wiring demanded.
        fuel_demand_kg_s = (charge_volume_m3 * mixer_gas_frac * gas_props["density_kg_m3"]
                            * self.state.real_fire_hz)
        fuel_production_kg_s = 0.0
        production_composition_frac = 1.0
        fuel_valve_open = True
        net_tick = self._step_fuel_network(dt, fuel_demand_kg_s)
        if net_tick is not None:
            fuel_production_kg_s = net_tick.production_kg_s
            production_composition_frac = net_tick.production_composition_frac
            fuel_valve_open = net_tick.valve_open
        elif self.atmospheric_generator is not None:
            if eng.atmospheric_supply_tank_capacity_kg <= 0.0:
                raise ValueError(
                    f"{eng.identity}: an on-site generator needs a real gasholder to feed "
                    "(Engine.atmospheric_supply_tank_capacity_kg > 0) -- a bare unmetered main "
                    "and a generator are two different real supplies, not one")
            fuel_production_kg_s = self.atmospheric_generator.gas_output_kg_s(dt, engine_rpm=self.state.rpm)
            # the generator's own outlet plumbing purging its startup
            # air (GasQualityBath): what flows out of it INTO the holder
            # is its production -- the holder's bell, not the engine,
            # is what the generator sees downstream, so the engine's
            # own draw can never pull air backward through the
            # generator here (that backfill term is real only for a
            # generator sucked on directly, with no holder between)
            if hasattr(self.atmospheric_generator, "step_quality"):
                production_composition_frac = self.atmospheric_generator.step_quality(dt, fuel_production_kg_s)
        self._drivetrain_out = self._drivetrain.step(
            dt, self._omega, 0.0,
            fuel_demand_kg_s=fuel_demand_kg_s, fuel_production_kg_s=fuel_production_kg_s,
            fuel_production_composition_frac=production_composition_frac,
            fuel_valve_open=fuel_valve_open,
            disabled_edges=frozenset({"dyno_friction_clutch", "dyno_roller_contact"}))
        accessory_reaction_nm = self._drivetrain_out.get("crank_reaction_torque_nm", 0.0)
        # The real charge this mixer just admitted -- composed from
        # real physics, not a fill-fraction proxy: the mixer's fixed
        # gas share, times what fraction of the gas LINE's contents is
        # genuinely fuel gas (the holder's own composition, converted
        # from the mass basis the reservoir balances in to the volume
        # basis port areas and flammability limits are stated in),
        # times how much of the demanded gas was actually delivered
        # (a starved holder's shortfall is room air -- the cylinder
        # fills to atmospheric regardless). The pressure rise then
        # follows from the fuel's own real flammability limits and
        # heat release (otto_langen.charge_ignition_pressure_ratio):
        # too lean and the igniter lights NOTHING -- a real misfire,
        # gated into `fire` below alongside the governor's own skip.
        # This is "idle via adjusting the combustion" for an engine
        # with no throttle plate, and the actual real safety argument
        # for the stroke margin: the holder can never deliver more
        # than pure gas (composition 1.0), so the design mixer is the
        # physical ceiling on ignition strength. An unmetered town
        # main is pure gas fully available -- exactly the spec's own
        # cited ratio, unchanged.
        availability = self._drivetrain_out.get("fuel_availability_frac", 1.0)
        holder_composition_mass = self._drivetrain_out.get("fuel_composition_frac", 1.0)
        holder_composition_vol = fuel_mass_to_volume_fraction(holder_composition_mass, gas_props["density_kg_m3"])
        charge_gas_frac = mixer_gas_frac * holder_composition_vol * availability
        cyl.ignition_pressure_ratio = atmo_spec.charge_ignition_pressure_ratio(charge_gas_frac)
        charge_flammable = charge_is_flammable(fuel_profile, charge_gas_frac)
        holder_fill_frac = self._drivetrain_out.get("fuel_fill_frac", 1.0)
        self.state.charge_gas_volume_fraction = charge_gas_frac
        self.state.charge_equivalence_ratio = charge_gas_frac / max(gas_props["stoich_vol_frac"], 1e-9)
        self.state.fuel_availability_frac = availability
        self.state.gasholder_composition_frac = holder_composition_vol
        self.state.gasholder_flammable_hazard = (
            holder_fill_frac > 0.0 and charge_is_flammable(fuel_profile, holder_composition_vol))

        self._brake_junction.engagement = self.clutch_frac
        effective_ratio = self._current_gear_ratio()

        # the same stiff-spring clutch-junction stability requirement
        # the turbine's own _step_turbine now respects (see
        # _clutch_substep_plan) -- this flywheel's own inertia is
        # exactly eng.inertia_kg_m2, the same real number cyl.step()
        # already integrates the ratchet catch against
        n_sub, sub_dt = self._clutch_substep_plan(dt, eng.inertia_kg_m2)
        junction_torque = 0.0
        dyno_torque_nm = 0.0
        shaft_load_nm = 0.0
        old_omega = self._omega
        for _ in range(n_sub):
            if effective_ratio > 0.0:
                junction_torque = self._brake_junction.step(sub_dt, self._omega, self._load_omega * effective_ratio)
            else:
                junction_torque = 0.0
            shaft_load_nm = junction_torque + accessory_reaction_nm

            fire = True
            if gov is not None:
                fire = not gov.step(sub_dt, self._omega)
            # a charge outside its fuel's flammable band doesn't light
            # no matter what the governor decided -- the governor still
            # steps (it's a real mechanism on the shaft either way)
            fire = fire and charge_flammable

            prev_phase = cyl.phase
            new_omega = cyl.step(sub_dt, self._omega, eng.inertia_kg_m2, shaft_load_nm, fire=fire)
            # a real ignition just happened -- the SAME general
            # pending_events queue the piston combustion loop publishes
            # to (see _record_ignition's own docstring: this is a
            # every-kind mechanism, not bespoke to this engine)
            just_ignited = prev_phase == "awaiting_ignition" and cyl.phase == "free_flight"
            if just_ignited and fire:
                self.state.fire_event_count += 1
            if just_ignited:
                self.pending_events.append(IgnitionEvent(
                    cylinder=0, position_key=0, strength=1.0,
                    knock=False, knock_intensity=0.0, misfire=not fire))
            self._record_ignition(sub_dt, just_ignited)
            # NOT clamped to zero below some rpm floor the way piston/
            # electric engines are (a real "stalled, nothing left to
            # restart it" state): a cold Otto-Langen flywheel genuinely
            # builds up real speed from a series of individually tiny
            # ratchet catches (see otto_langen.py's own module
            # docstring), each one only a fraction of a rpm -- snapping
            # that back to exactly zero every tick would never let it
            # accumulate and permanently freeze the cold-start bootstrap
            # (found empirically: it did exactly that). max(0.0, ...)
            # alone is the real mechanical floor -- the ratchet is
            # one-way, this flywheel genuinely cannot be driven backward
            # by anything on this shaft.
            self._omega = max(0.0, new_omega)

            if effective_ratio > 0.0:
                dyno_torque_nm = junction_torque * effective_ratio
                domega_load = (dyno_torque_nm - self.brake_load_nm) / load_inertia_kg_m2 * sub_dt
            else:
                dyno_torque_nm = 0.0
                domega_load = -self.brake_load_nm / load_inertia_kg_m2 * sub_dt
            self._load_omega = max(0.0, self._load_omega + domega_load)

        self.state.rpm = self._omega / RPM_TO_RAD_S
        self.state.dyno_rpm = self._load_omega / RPM_TO_RAD_S
        self.state.dyno_absorbed_kw = self.brake_load_nm * self._load_omega / 1000.0
        self.state.dyno_kinetic_energy_j = 0.5 * self._dyno_inertia_kg_m2 * self._load_omega * self._load_omega
        self.state.dyno_torque_nm = dyno_torque_nm
        self.state.brake_clutch_locked = self._brake_junction.locked

        # the real instantaneous shaft torque this tick -- backed out of
        # the shaft's own actual omega change (cyl.step already owns the
        # full ratchet-catch/freewheel solve internally) plus whatever
        # load was already subtracted going in, the same real
        # torque=inertia*domega/dt+load relation used everywhere else
        shaft_torque_nm = (self._omega - old_omega) / max(dt, 1e-9) * eng.inertia_kg_m2 + shaft_load_nm
        self.state.current_torque_nm = shaft_torque_nm
        self.state.power_kw = shaft_torque_nm * self._omega / 1000.0
        rms_alpha = min(1.0, dt / RMS_TAU_S)
        self._torque_ms += (self.state.current_torque_nm ** 2 - self._torque_ms) * rms_alpha
        self._power_ms += (self.state.power_kw ** 2 - self._power_ms) * rms_alpha
        self.state.torque_rms_nm = math.sqrt(max(0.0, self._torque_ms))
        self.state.power_rms_kw = math.sqrt(max(0.0, self._power_ms))

        # always atmospheric -- there's no throttle plate at all on this
        # real mechanism (see this method's own docstring); a real,
        # non-restrictive admission every cycle
        self.state.manifold_pressure_frac = 1.0
        self.state.ignition_timing_deg = 0.0
        # the real working-gas temperature, off the SAME isentrope
        # (P*V^gamma = const) otto_langen.py's own pressure relation
        # already tracks: ignition is a real, near-constant-volume
        # event, so T1 = T_atm * IGNITION_PRESSURE_RATIO by the ideal
        # gas law at fixed V, and T follows the gas's own real P
        # isentropically from there (T/T1 = (P/P1)^((gamma-1)/gamma)) --
        # this engine's own real "exhaust gas temperature," not a
        # separate exhaust-system model the way a piston engine has one
        t1_k = OTTO_T_ATM_K * IGNITION_PRESSURE_RATIO
        p1_pa = OTTO_P_ATM_PA * IGNITION_PRESSURE_RATIO
        pressure_ratio_from_ignite = max(cyl.working_pressure_pa, 1.0) / p1_pa
        self.state.exhaust_temp_k = t1_k * pressure_ratio_from_ignite ** (
            (GAMMA_COMBUSTION_GAS - 1.0) / GAMMA_COMBUSTION_GAS)
        # no alternator, no battery, no starter draw on this real
        # 1867-era build (Accessories.alternator=False, starting_
        # systems=("hand-crank",) -- a genuinely dark electrical bus,
        # not an oversight)
        self.state.battery_voltage = self.electrical.nominal_voltage_v
        self.state.spark_energy_frac = 1.0
        self.state.alternator_output_frac = 0.0
        self.state.battery_soc_frac = 1.0
        self.state.knock_flag = False
        self.state.knock_intensity = 0.0
        # a real, documented consequence, not a survivable bounce (see
        # otto_langen.STROKE_STOP_DESTRUCTIVE_SPEED_M_S's own comment,
        # researched rather than invented: full-size Otto-Langen
        # engines were genuinely destroyed by a hard top-of-stroke
        # impact) -- once the cylinder reports it, this engine is
        # permanently down, the same real "stalled, needs real
        # intervention" state every other engine kind uses, not a
        # quieter flag nobody would notice
        if cyl.destroyed:
            self.state.stalled = True
        self.state.misfire_flag = not fire
        self.state.valve_float_flag = False
        self.state.valve_float_risk = 0.0
        self.state.turbo_spool_frac = 0.0
        self.state.boost_frac = 0.0
        self.state.wastegate_flutter = False
        self.state.surge_flag = False
        self.state.backfire_flag = False
        self.state.backfire_kind = None
        # the SAME real drivetrain-graph fuel circuit every liquid-fuel
        # engine reports from: 1.0 forever on an unmetered town main
        # (no vessel to deplete), the gasholder's real fill otherwise
        self.state.fuel_fill_frac = holder_fill_frac

    @staticmethod
    def _crossed(prev: float, new: float, target: float, period: float) -> bool:
        if new >= prev:
            return prev <= target < new
        return target >= prev or target < new

    def drain_events(self) -> list[IgnitionEvent]:
        events = self.pending_events
        self.pending_events = []
        return events

    def drain_backfires(self) -> list[BackfireEvent]:
        events = self.pending_backfires
        self.pending_backfires = []
        return events
