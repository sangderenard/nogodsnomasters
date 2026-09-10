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
import random

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
from gas_turbine import SingleShaftGasTurbine
from otto_langen import (OttoLangenCylinder, P_ATM_PA as OTTO_P_ATM_PA, T_ATM_K as OTTO_T_ATM_K,
                          GAMMA_COMBUSTION_GAS, IGNITION_PRESSURE_RATIO,
                          fuel_gas_properties, fuel_mass_to_volume_fraction, charge_is_flammable)
from governor import CaptiveBallGovernor
from expander import ExpanderCylinderBank
from valve_state import ValveState
from crankcase_state import CrankcaseState
import numpy as _np
from gas_works import _steam_properties
from fuel_network import (FuelNetworkRuntime, SupplyTick, charge_energy_factor, intake_flashback_risk)
import engine_geometry
import wear as wear_module
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
    throttle: float = 0.0
    brake_load_nm: float = 0.0
    clutch_frac: float = 1.0   # 0=pedal floored/disengaged, 1=fully released/locked -- ClutchPort.engagement
    gear_index: int = 0        # 0=neutral (every engine starts in N), negative=reverse, 1..N=forward gear (engine.transmission.gear_ratios)
    electrical_load_frac: float = 0.30
    fuel_choice: str | None = None   # None -> engine.preferred_fuel_profile
    rev_limiter_enabled: bool = True  # off = watch what it's actually there to prevent
    anti_lag_enabled: bool = False    # turbo + forced_induction.anti_lag_capable only
    nitrous_active: bool = False       # engine.has_nitrous only -- opens the real solenoid
    auxiliary_injection_active: bool = True   # engine.has_auxiliary_injection only -- a real WMI kit is normally armed whenever the engine runs, not driver-toggled like nitrous
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
        self._omega = 0.0
        self._total_crank_deg = 0.0
        self._map_frac_na = idle_map_base_frac(self.engine)  # ECU's per-engine idle calibration
        self._fuel_rack_frac = idle_fuel_base_frac(self.engine)  # ECU's per-engine idle rack calibration
        self._prev_throttle = 0.0
        self._surge_timer = 0.0
        self._antilag_cooldown = 0.0
        self._firing_angle_deg: dict[int, float] = {}
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
        self._waste_heat_kw = 0.0
        self._intake_supply_pressure_pa = 101_325.0
        self._intake_demand_kg_s = 0.0
        self._fuel_demand_kg_s = 0.0
        self._fuel_cooling_kw = 0.0
        self._auxiliary_injection_cooling_kw = 0.0
        self._fuel_starvation_frac = 1.0
        self._exhaust_demand_kg_s = 0.0
        self._physics_accum_s = 0.0
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
        graph = build_drivetrain_graph(self.engine)
        dyno_node = next(n for n in graph["nodes"] if n["identity"] == "dyno_absorber")
        self._dyno_mass_kg = dyno_node["mass_kg"]
        self._dyno_inertia_kg_m2 = dyno_node["inertia_kg_m2"]
        self._dyno_drum_radius_m = dyno_node["drum_radius_m"]
        return DrivetrainSolver(graph)

    def _build_brake_junction(self) -> ClutchPort:
        peak = max(self.engine.peak_torque_nm, 1.0)
        return ClutchPort(stiffness_nm_per_rad_s=peak * 8.0, max_torque_nm=peak * 3.0)

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
        """One button: neutral -> settle -> real quick-shift into the
        transmission's own top ("H") gear -> wide open throttle, run to
        redline. brake_load_nm goes to 0.0 for the whole pull -- a real
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
                top_gear = len(self.engine.transmission.gear_ratios)
                self._request_quick_shift(top_gear)
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
            redline_reached = self.state.rpm >= self.engine.redline_rpm * 0.98
            timed_out = self._dyno_pull_timer_s >= DYNO_PULL_MAX_DURATION_S
            if redline_reached or timed_out:
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
        n = len(arch.firing_order)
        step_deg = arch.cycle_degrees / max(n, 1)
        # real per-cylinder swept volume, for the parametric volume-
        # pressure exchange each firing event draws against the real
        # intake-air circuit with (drivetrain_graph.draw_cylinder_charge)
        self._cylinder_volume_m3 = (self.engine.displacement_l / 1000.0) / max(arch.cylinders, 1)
        self.ignition.reset(len(self.engine.rev_limiter.stages))
        self._firing_angle_deg = {cyl: slot * step_deg for slot, cyl in enumerate(arch.firing_order)}
        self._last_fire_total_deg = {cyl: -1e9 for cyl in arch.firing_order}
        self._last_strength = {cyl: 0.0 for cyl in arch.firing_order}
        self._knock_accum = {cyl: 0.0 for cyl in arch.firing_order}
        self._knocked_this_burn = {cyl: False for cyl in arch.firing_order}
        self._active_event = {}
        self._flame_radius_m = {}
        self._burned_frac = {}

    def set_engine(self, engine: Engine) -> None:
        self.engine = engine
        self.state.wear = wear_module.new_wear_state(engine)
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
        self._waste_heat_kw = 0.0
        self._intake_supply_pressure_pa = 101_325.0
        self._intake_demand_kg_s = 0.0
        self._fuel_demand_kg_s = 0.0
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

    def start(self) -> None:
        self._omega = self.engine.idle_rpm * RPM_TO_RAD_S
        self.state.rpm = self.engine.idle_rpm
        self.state.stalled = False
        self.state.ignition_cut = False
        self._time_since_start_s = 0.0
        if self._turbine is not None:
            # a real gas generator spinning at the idle N1 this catalogue
            # entry's own idle_rpm was derived from (see engines.py's
            # _build_turbine_and_atmospheric_engines) -- the "already
            # running, warm" teleport this method already gives every
            # other engine kind, not a cold-start light-off sequence
            reduction = max(self.engine.turbine.reduction_ratio, 1e-6)
            self._turbine.omega_rad_s = self._omega * reduction
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
                for _ in range(random.randint(2, 4)):
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
            self._step_once(FIXED_PHYSICS_DT_S)
            self._physics_accum_s -= FIXED_PHYSICS_DT_S

    def _step_once(self, dt: float) -> None:
        eng = self.engine
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
                    self.state.ac_compressor_load_w + self.electrical.reading.alternator_shaft_load_w, dt,
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
            ac_active=self.ac_enabled and eng.accessories.air_conditioning,
            disabled_edges=frozenset({"dyno_friction_clutch", "dyno_roller_contact"}),
            starting_air_kg_s=self.starter.reading.air_kg_s)
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
            raw_map_frac = self._map_frac_na * (1.0 + self.state.boost_frac) + self.state.nitrous_boost_frac
        else:
            raw_map_frac = self._map_frac_na + self.state.boost_frac + self.state.nitrous_boost_frac
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
        self._fuel_demand_kg_s = self._intake_demand_kg_s / max(stoich_afr, 0.1)
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
                    self.state.ac_compressor_load_w + self.electrical.reading.alternator_shaft_load_w, dt,
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
            valvetrain_drag_nm = spring.drag_torque_nm(arch.cylinders)
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
                    if hasattr(eng.architecture, "firing_events_per_rev") else self.state.rpm / 120.0 * cs.n_cyl)
            self.state.cylinder_oil_film_mg = (cs.film_kg * 1e6).tolist()
            self.state.oil_consumption_ml_per_h = float(cs.burn_kg_s.sum()) * 3600.0 / 0.87 * 1000.0
            self.state.blowby_l_per_min = float(cs.blowby_kg_s.sum()) / 1.2 * 60000.0
            self.state.crankcase_pressure_kpa = cs.crankcase_pressure_pa / 1000.0
            self.state.sump_oil_l = cs.oil_kg / 0.87
            self.state.oil_fuel_dilution_frac = cs.dilution_frac
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

            torque_nm = 0.0
            any_real_ignition_this_substep = False
            for cyl, firing_angle in self._firing_angle_deg.items():
                spark_angle = (firing_angle - self.state.ignition_timing_deg) % arch.cycle_degrees
                if self._crossed(prev_pos, new_pos, spark_angle, arch.cycle_degrees):
                    limiter_cut = active_severity > 0.0 and random.random() < active_severity
                    misfire = (not limiter_cut) and (not governor_skip) and random.random() < misfire_prob
                    # ignition_cut: the key/kill-switch turning the
                    # ignition system off, not a stall -- every cylinder
                    # simply gets no spark, same real mechanism a misfire
                    # already models, so the crank coasts down under its
                    # own pumping/friction/dyno load instead of being
                    # teleported to a dead stop
                    cut = misfire or limiter_cut or governor_skip or self.state.ignition_cut

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
                    base_strength = (torque_fraction(eng, self.state.rpm) * cylinder_charge_frac
                                      * charge_density_frac
                                      * fuel_quantity_frac * eng.combustion_efficiency * float_penalty
                                      * cyl_valve
                                      * (1.0 - egr_active_frac))
                    strength = 0.0 if cut else base_strength

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
                        if random.random() < backfire_prob:
                            self.pending_backfires.append(
                                BackfireEvent(kind="unplanned", strength=0.4 + 0.6 * base_strength))

                    self._last_fire_total_deg[cyl] = self._total_crank_deg
                    self._last_strength[cyl] = strength
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
                                        knock=False, knock_intensity=0.0, misfire=misfire)
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
                            any_knock = True
                            max_knock_intensity = max(max_knock_intensity, intensity)
                    turbulent_flame_speed = LAMINAR_FLAME_SPEED_M_S + FLAME_TURBULENCE_GAIN * eng.mean_piston_speed_m_s(live_rpm)
                    self._flame_radius_m[cyl] = min(
                        bore_half_m, self._flame_radius_m.get(cyl, SPARK_KERNEL_SEED_RADIUS_M) + turbulent_flame_speed * sub_dt)
                    new_burned = min(1.0, (self._flame_radius_m[cyl] / bore_half_m) ** 3)
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
            if effective_ratio > 0.0:
                junction_torque = self._brake_junction.step(
                    sub_dt, self._omega, self._load_omega * effective_ratio)
            else:
                junction_torque = 0.0
            # plus whatever is turning the crank from outside: the
            # starting system's torque (starter.py) and any attachment
            # the game feeds through the crank nose
            net_torque = torque_nm - pumping_loss - junction_torque + self._crank_assist_nm
            domega = net_torque / max(eng.inertia_kg_m2, 1e-9) * sub_dt
            self._omega = max(0.0, self._omega + domega)

            if effective_ratio > 0.0:
                dyno_torque_nm = junction_torque * effective_ratio
                domega_load = (dyno_torque_nm - self.brake_load_nm) / load_inertia_kg_m2 * sub_dt
            else:
                dyno_torque_nm = 0.0
                domega_load = -self.brake_load_nm / load_inertia_kg_m2 * sub_dt
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
        self.state.rev_limiter_active = self.ignition.any_stage_active or (eng.rev_limiter.soft_taper and active_severity > 0.0)
        self.state.knock_flag = any_knock
        self.state.knock_intensity = max_knock_intensity
        self.state.misfire_flag = any_misfire

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
        runner_surround_temp_k = EXHAUST_TEMP_AMBIENT_K + (
            block_temp_k - EXHAUST_TEMP_AMBIENT_K) * RUNNER_BLOCK_CONDUCTION_FRAC
        runner_temp_k = eng.intake_system.runner_outlet_temp_k(
            EXHAUST_TEMP_AMBIENT_K, self._intake_demand_kg_s, runner_surround_temp_k)
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
                junction_torque = self._brake_junction.step(sub_dt, self._omega, self._load_omega * effective_ratio)
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
        self.state.boost_frac = max(0.0, turb.pressure_ratio - 1.0)
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
