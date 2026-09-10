"""The engine control unit as its own real box -- the ECU's own program,
not physics tangled into EngineCycleSim.

Everything here used to live inline in engine_cycle_sim.py's
EngineCycleSim (three private methods and a block of module constants).
It is moved here unchanged in behavior so that the idle program is one
self-contained, tunable unit with a clean sensor-in / command-out
boundary: the ECU reads live readings it is handed (rpm, a known
accessory load) plus the engine's own declared profile, holds its OWN
controller state (integrators, which governor is currently engaged),
and returns a commanded MAP target or fuel quantity. EngineCycleSim
owns one instance and delegates; it never reaches into the ECU's
integrators itself.

This is deliberately the same identity the real production vehicle
graph already carries for this box -- "electrical.ecu", kind
"vehicle-computer" (turing's abstract_ui_vehicles.py, the electrical
harness section) -- so the engine_toy ECU node is the real one, not a
toy-invented sibling. The logical gating in here (what engages when,
how hard) is exactly the part meant to be tuned freely from now on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from engines import Engine, RPM_TO_RAD_S
from engine_sim import torque_fraction

# Real, typical idle manifold pressure as a fraction of atmospheric --
# the commanded starting point every idle correction is applied on top
# of, and what the intake circuit is seeded at on start() so a fresh
# start doesn't manufacture a pressure-drop transient.
MAP_IDLE_FRAC_DEFAULT = 0.30

# --- idle-speed control loop design (both governors) --------------------
# The gains are not constants: they're placed from the engine's own
# plant. Around idle the crank is an inertia driven by a torque that is
# (near enough) proportional to the actuator -- manifold pressure for a
# throttled engine, fuel quantity for a diesel:
#     J * d(omega)/dt = K_t * u - T_load
# A PI on that plant gives a second-order closed loop, and choosing its
# natural frequency and damping fixes the gains exactly:
#     Kp = 2*zeta*w_n*J / K_t      Ki = w_n^2*J / K_t
# J is the COUPLED inertia the ECU's neutral switch tells it about (crank
# alone in N; crank plus the reflected driveline in gear -- a real ECU
# gain-schedules on P/N vs D for precisely this reason), K_t this
# engine's own torque per unit actuator at idle. That is what makes a
# 0.00006 kg*m^2 trimmer and a 0.46 kg*m^2 blown V8 each get the loop a
# real calibration engineer would give them, instead of one flat gain
# that hunted the small ones and stalled the big ones the moment the
# dyno drum's inertia stopped damping the loop in gear.
IDLE_LOOP_NATURAL_FREQ_RAD_S = 2.0   # real ISC bandwidth ~0.3 Hz: safely below the manifold-fill lag
IDLE_LOOP_DAMPING = 1.0              # critically damped: no overshoot into a stall from below
# The loop must be placed well ABOVE the plant's own open-loop unstable
# pole (slope_b / J, see IdlePlant) or the manifold lag lets the sag win
# -- the 25 cc trimmer's pole sits at 2.1 /s, every other engine's under
# 0.9 /s, and a flat 2 rad/s loop could not hold it. Real ISC bandwidth
# is scheduled per engine; the ceiling is the firing rate (the loop can't
# sample faster than the crank delivers torque pulses).
IDLE_LOOP_UNSTABLE_POLE_MARGIN = 3.0
IDLE_LOOP_MAX_FRAC_OF_FIRING_RATE = 0.25
IDLE_MAP_INTEGRAL_CLAMP_FRAC = 2.0
# Real "anti-stall"/idle-recovery behavior: a real ECU doesn't correct a
# deep rpm sag with the same gain it uses for ordinary idle hunting --
# it opens up harder specifically as stall risk rises. Below this
# fraction of idle_rpm, the proportional gain ramps up smoothly (never
# a step, so this can't itself introduce a new discontinuity-driven
# oscillation) toward STALL_RESCUE_KP_MULTIPLIER right at the point rpm
# hits zero. Found empirically real: a uniform-gain governor left rpm
# recovering from a deep misfire-driven sag no faster than it recovers
# from ordinary hunting, which is not how a real idle-air-control
# system behaves once actually close to stalling.
STALL_RISK_RPM_FRAC = 0.70
STALL_RESCUE_KP_MULTIPLIER = 2.5

# --- diesel injection-pump idle (fuel quantity) governor ---------------
IDLE_FUEL_QUANTITY_BASE_FRAC = 0.12  # the old diesel formula's own base -- kept, see idle_fuel_quantity
IDLE_FUEL_INTEGRAL_CLAMP_FRAC = 2.0


@dataclass(frozen=True)
class IdlePlant:
    """The ECU's engine model at the idle operating point -- the
    linearization of the SAME torque balance engine_cycle_sim's
    combustion loop integrates every substep:

        T_net(w, u) = P*tf(w)*u  -  B*tf(w)*(1-u)*0.5  -  V  -  F(w)

    (P peak torque, tf the real torque-vs-rpm shape, B peak braking
    torque so the second term is the closed-throttle intake pumping loss
    the sim charges, V valvetrain drag, F the FMEP friction). For a
    diesel the actuator u is fuel quantity and the pumping term is the
    small constant an unthrottled manifold leaves. Three numbers fall
    out of it, all per engine, none hand-tuned:

      base_u   the actuator that balances T_net = 0 at idle (the ECU's
               base-airflow / base-rack table entry)
      k_t      dT/du   crank torque per unit actuator (the loop gain)
      slope_b  dT/dw   the plant's own speed slope at idle -- POSITIVE
               on every engine here (torque rises with rpm faster than
               friction does below the torque peak), i.e. an open-loop
               UNSTABLE pole at +slope_b/J. The governor exists to hold
               it; a design that ignored it (pure-inertia pole placement)
               left the trimmer's loop too soft to catch a sag.
    """
    base_u: float
    k_t: float
    slope_b: float
    u_lo: float
    u_hi: float


_IDLE_PLANT_CACHE: dict[str, IdlePlant] = {}


def _friction_nm(engine: Engine, rpm: float) -> float:
    # the sim's own friction law, rotary proxy included (engine_cycle_sim's pumping_loss block)
    if engine.architecture.rotary:
        return engine.peak_torque_nm * (0.025 + 0.02 * (rpm / max(engine.redline_rpm, 1.0)))
    return engine.friction_torque_nm(rpm)


def idle_plant(engine: Engine) -> IdlePlant:
    cached = _IDLE_PLANT_CACHE.get(engine.identity)
    if cached is not None:
        return cached
    arch = engine.architecture
    rpm0 = max(engine.idle_rpm, 1.0)
    d_rpm = max(1.0, rpm0 * 0.05)
    tf0 = torque_fraction(engine, rpm0)
    tf_slope = (torque_fraction(engine, rpm0 + d_rpm) - torque_fraction(engine, rpm0 - d_rpm)) / (2.0 * d_rpm)
    f_slope = (_friction_nm(engine, rpm0 + d_rpm) - _friction_nm(engine, rpm0 - d_rpm)) / (2.0 * d_rpm)
    p = engine.peak_torque_nm
    b = engine.peak_braking_torque_nm
    v = engine.lifter_spring.drag_torque_nm(arch.cylinders) if arch.has_poppet_valves else 0.0
    f0 = _friction_nm(engine, rpm0)
    if engine.compression_ignition:
        # unthrottled: charge fixed at DIESEL_UNTHROTTLED_MAP_FRAC, fuel is the actuator
        k_t = p * tf0 * DIESEL_UNTHROTTLED_MAP_FRAC
        pump0 = 0.5 * b * tf0 * (1.0 - DIESEL_UNTHROTTLED_MAP_FRAC)
        base = (pump0 + v + f0) / max(k_t, 1e-9)
        u_lo, u_hi = FUEL_IDLE_FLOOR_FRAC, FUEL_IDLE_CEILING_FRAC
        base = max(u_lo, min(u_hi, base))
        torque_speed_slope = (p * DIESEL_UNTHROTTLED_MAP_FRAC * base - pump0 / max(tf0, 1e-9)) * tf_slope
    else:
        # dT/du includes the pumping loss the extra manifold pressure removes
        k_t = p * tf0 + 0.5 * b * tf0
        base = (0.5 * b * tf0 + v + f0) / max(k_t, 1e-9)
        u_lo, u_hi = MAP_IDLE_FLOOR_FRAC, MAP_IDLE_CEILING_FRAC
        base = max(u_lo, min(0.9, base))
        torque_speed_slope = (p * base - 0.5 * b * (1.0 - base)) * tf_slope
    # per rad/s of crank speed, positive = destabilizing
    slope_b = (torque_speed_slope - f_slope) / RPM_TO_RAD_S
    plant = IdlePlant(base_u=base, k_t=max(k_t, 1e-9), slope_b=slope_b, u_lo=u_lo, u_hi=u_hi)
    _IDLE_PLANT_CACHE[engine.identity] = plant
    return plant


def idle_actuator_torque_gain_nm(engine: Engine) -> float:
    """K_t: crank torque per unit of the idle actuator at idle speed."""
    return idle_plant(engine).k_t


def idle_pi_gains_frac(engine: Engine, coupled_inertia_kg_m2: float) -> tuple[float, float]:
    """(kp, ki) in this ECU's working units -- actuator fraction per unit
    FRACTIONAL rpm error, and per unit of its time integral -- from pole
    placement on the linearized plant J*s - b (see IdlePlant):
        J s^2 + (Kp K_t - b) s + Ki K_t = s^2 + 2 zeta w_n s + w_n^2
        Kp = (2 zeta w_n J + b) / K_t        Ki = w_n^2 J / K_t
    Converting the physical gains (per rad/s of error) to fractional-
    error units multiplies by idle speed in rad/s."""
    j = max(coupled_inertia_kg_m2, 1e-9)
    plant = idle_plant(engine)
    w_n = idle_loop_natural_freq_rad_s(engine, j)
    kp_phys = (2.0 * IDLE_LOOP_DAMPING * w_n * j + plant.slope_b) / plant.k_t
    ki_phys = w_n ** 2 * j / plant.k_t
    idle_omega = max(engine.idle_rpm, 1.0) * RPM_TO_RAD_S
    return max(0.0, kp_phys) * idle_omega, ki_phys * idle_omega


def idle_loop_natural_freq_rad_s(engine: Engine, coupled_inertia_kg_m2: float) -> float:
    """The scheduled loop bandwidth: the nominal ISC figure, raised to
    dominate this engine's own unstable pole by a margin, capped at a
    fraction of its firing rate."""
    plant = idle_plant(engine)
    unstable_pole = max(0.0, plant.slope_b / max(coupled_inertia_kg_m2, 1e-9))
    w_n = max(IDLE_LOOP_NATURAL_FREQ_RAD_S, IDLE_LOOP_UNSTABLE_POLE_MARGIN * unstable_pole)
    arch = engine.architecture
    firing_rate_hz = arch.cylinders * (engine.idle_rpm / 60.0) * (360.0 / max(arch.cycle_degrees, 1.0))
    return min(w_n, max(IDLE_LOOP_NATURAL_FREQ_RAD_S, 2.0 * math.pi * firing_rate_hz * IDLE_LOOP_MAX_FRAC_OF_FIRING_RATE))


def blower_pressure_ratio(engine: Engine) -> float:
    """The plant gain between what the ECU commands (throttle-side
    inlet pressure) and the manifold pressure the engine actually
    breathes. On a draw-through blown engine the blower multiplies
    whatever the throttle lets into its inlet by its own displacement-
    ratio pressure ratio (engine_cycle_sim folds it as inlet x
    (1 + max_boost_frac), the belt-delivered PR at nominal ratio); a
    naturally-aspirated or turbo engine has no such multiplier at the
    throttle. The ECU's idle calibration is in MANIFOLD terms (that is
    what sets torque) and is divided by this to become an inlet
    command -- without that the blown engines were seeded at 2x the
    manifold pressure idle needs, and the same PI gains that hold an
    NA engine steady drove them into a 2.2x-hotter loop that hunted
    800-2800 rpm (found on the drag V8 / monster 540 / Wasp the moment
    the blower became a real pump instead of a throttle-scaled term)."""
    fi = engine.forced_induction
    return 1.0 + fi.max_boost_frac if fi.kind == "supercharger" else 1.0


MAP_IDLE_FLOOR_FRAC = 0.02      # closed idle-air path: the deepest vacuum the throttle body can hold
MAP_IDLE_CEILING_FRAC = 1.0
FUEL_IDLE_FLOOR_FRAC = 0.05     # injection-pump rack stops
FUEL_IDLE_CEILING_FRAC = 0.35


def idle_base_manifold_frac(engine: Engine) -> float:
    """The ECU's base idle position, in MANIFOLD terms, derived instead
    of tabled: the actuator that balances the engine's own idle torque
    balance (friction + valvetrain drag + closed-throttle pumping, see
    IdlePlant). A real ECU carries this as a calibrated base-airflow
    table; the closed loop only ever TRIMS around it. One flat 0.30 atm
    for every engine was the whole reason a blown 8.2 L V8 revved to
    7400 rpm unloaded on 'idle' air while a 25 cc two-stroke starved on
    the same number."""
    if engine.architecture.cylinders <= 0 or engine.compression_ignition:
        return MAP_IDLE_FRAC_DEFAULT
    return idle_plant(engine).base_u


DIESEL_UNTHROTTLED_MAP_FRAC = 0.92   # no throttle plate: manifold sits just under atmospheric regardless of pedal


def idle_map_base_frac(engine: Engine) -> float:
    """The ECU's calibrated idle base INLET command for THIS engine:
    the derived base manifold pressure divided by the blower's own
    pressure ratio (see blower_pressure_ratio). A diesel has no air
    actuator at all -- its manifold is the unthrottled value and its
    derived base lives on the fuel rack (idle_fuel_base_frac)."""
    if engine.compression_ignition:
        return DIESEL_UNTHROTTLED_MAP_FRAC
    return idle_base_manifold_frac(engine) / blower_pressure_ratio(engine)


def idle_fuel_base_frac(engine: Engine) -> float:
    """The diesel analogue: base rack position from the same plant model."""
    if engine.architecture.cylinders <= 0 or not engine.compression_ignition:
        return IDLE_FUEL_QUANTITY_BASE_FRAC
    return idle_plant(engine).base_u


def pi_with_antiwindup(integral: float, error: float, kp: float, ki: float, base: float,
                       feedforward: float, lo: float, hi: float, dt: float) -> tuple[float, float]:
    """One PI step with two real anti-windup measures: (1) conditional
    integration -- the integrator is frozen while the command sits on a
    stop and the error would only push it further into that stop (the
    textbook fix for the limit cycle a wound-up integrator produces
    when it has to unwind through its whole range before the actuator
    moves again); (2) the integral term itself is bounded to what the
    actuator can still deliver around base, not an arbitrary constant.
    Returns (command, new_integral)."""
    unsat = base + feedforward + kp * error + ki * integral
    saturated_high = unsat >= hi and error > 0.0
    saturated_low = unsat <= lo and error < 0.0
    if ki > 0.0 and not (saturated_high or saturated_low):
        integral += error * dt
        i_lo = (lo - base) / ki
        i_hi = (hi - base) / ki
        integral = max(i_lo, min(i_hi, integral))
    cmd = base + feedforward + kp * error + ki * integral
    return max(lo, min(hi, cmd)), integral


# --- electric radiator fan thermostat program ---------------------------
ELECTRIC_FAN_ON_TEMP_K = 358.15   # real ~85 degC thermostatic switch-on point
ELECTRIC_FAN_BAND_K = 10.0        # real PWM/relay proportional band above it


def is_computerized(engine: Engine) -> bool:
    """Does this engine actually have an engine computer on the bus?
    A carbureted/distributor or magneto engine has no ECU to brown out
    -- its idle 'governor' is a mechanical idle screw / IAC solenoid or
    a flyball, and it keeps running with a dead battery as long as the
    spark source does. Disclosed rule: any closed-loop feature in the
    ECUProfile, or an ECU-driven ignition on a fuel-injected engine,
    means a real computer is fitted."""
    ecu = engine.ecu
    return (ecu.closed_loop_knock_control or ecu.closed_loop_afr or ecu.limiter_fuel_cut
            or (engine.ignition_dispatch == "ecu-electronic" and not engine.carburetor.is_carbureted
                and not engine.compression_ignition))


class EngineControlUnit:
    """One ECU: owns the idle governors' state and the gating between
    them, the knock-retard program, the WOT power-enrichment program,
    and the electric-fan thermostat. Construct once per sim; reset() on
    engine start. `powered` is set from the electrical bus each tick --
    below brown-out a real computer stops running its programs."""

    def __init__(self) -> None:
        self.powered = True
        self.reset()

    def reset(self) -> None:
        self._idle_map_integral = 0.0
        self._idle_governor_was_active = False
        self._idle_fuel_integral = 0.0
        self._idle_fuel_governor_was_active = False
        self._knock_retard_deg = 0.0

    # -- spark timing program --------------------------------------------
    @property
    def knock_retard_deg(self) -> float:
        return self._knock_retard_deg

    def spark_timing_deg(self, engine: Engine, base_timing_deg: float, knock_flag: bool,
                         knock_intensity: float, cam_lag_deg: float, dt: float) -> float:
        """Commanded spark advance: the engine's own MBT base curve, less
        the knock-retard program's current pull-back (closed-loop knock
        control reacting to LAST tick's knock -- a real knock sensor
        can't know a cylinder knocked until after it happened), less any
        real timing-chain lag an older distributor-driven ignition rides
        on. Recovers advance once knock clears."""
        ecu = engine.ecu
        if ecu.closed_loop_knock_control and self.powered:
            if knock_flag:
                retard_rate = ecu.knock_retard_deg_per_s * (0.5 + 0.5 * knock_intensity)
                self._knock_retard_deg = min(ecu.max_knock_retard_deg, self._knock_retard_deg + retard_rate * dt)
            else:
                self._knock_retard_deg = max(0.0, self._knock_retard_deg - ecu.knock_recover_deg_per_s * dt)
        return max(5.0, base_timing_deg - self._knock_retard_deg - cam_lag_deg)

    # -- mixture program -------------------------------------------------
    def power_enrichment_frac(self, engine: Engine, throttle: float) -> float:
        """Closed-loop AFR power enrichment: extra fuel dumped purely to
        cool the charge as throttle nears WOT, right when knock risk
        peaks -- not a torque effect, a knock-resistance one. A crude
        open-loop carb/ECU has no such program."""
        ecu = engine.ecu
        if not (ecu.closed_loop_afr and self.powered):
            return 0.0
        wot_ramp = max(0.0, min(1.0, (throttle - 0.75) / 0.25))
        return ecu.power_enrichment_richness * wot_ramp

    # -- cooling program -------------------------------------------------
    def electric_fan_command(self, coolant_temp_k: float) -> float:
        """Fan duty 0..1 off the real coolant thermostat switch point,
        the same way a real relay/PWM fan controller runs it. No power,
        no fan -- a real electric fan is a relay the computer closes."""
        if not self.powered:
            return 0.0
        return max(0.0, min(1.0, (coolant_temp_k - ELECTRIC_FAN_ON_TEMP_K) / ELECTRIC_FAN_BAND_K))

    # -- gating: the caller tells the ECU when the driver's foot is back
    #    on the pedal, so the governor knows its next engagement is a
    #    fresh one (integrator re-zeroed) rather than a continuation
    def release_map_governor(self) -> None:
        self._idle_governor_was_active = False

    def release_fuel_governor(self) -> None:
        self._idle_fuel_governor_was_active = False

    def idle_load_feedforward_frac(self, engine: Engine, ac_compressor_load_w: float) -> float:
        """Real "idle-up": a real ECU doesn't wait for rpm to actually
        sag before reacting to a KNOWN new load (AC compressor
        engaging, the alternator's electrical load -- both are real
        signals a modern ECU reads) -- it adds air/fuel proportional to
        that load right away, a feedforward term on top of the closed-
        loop correction. The caller hands in the SUM of known accessory
        shaft loads (the parameter name is historical).
        Without this, engaging a real accessory load at idle genuinely
        can stall a small engine before the PI loop's own reactive
        correction ever catches up -- verified: the AC compressor's own
        real belt torque (ac_compressor_load_w, one-tick-lagged same as
        every other cross-system reading) was doing exactly that.
        Expressed as a fraction of this engine's own real peak power
        capability at idle, not a flat number -- a big engine barely
        notices the same load a small commuter engine would stall on."""
        idle_power_capacity_w = max(engine.peak_torque_nm * engine.idle_rpm * RPM_TO_RAD_S, 1.0)
        return ac_compressor_load_w / idle_power_capacity_w

    def idle_map_target(self, engine: Engine, rpm: float, ac_compressor_load_w: float, dt: float,
                        coupled_inertia_kg_m2: float | None = None) -> float:
        """The gasoline/rotary idle-air-control governor, called instead
        of map_target_policy whenever the driver's foot is off the
        pedal. engine.ecu.idle_governor_mode selects the control law
        (engines.ECUProfile's own field, shared with the diesel fuel-
        quantity governor below): "p" alone can never fully close a
        steady disturbance (friction/pumping loss never goes away, so a
        P-only governor always settles short of idle_rpm by whatever
        gap its own proportional term needs to balance that load) --
        every catalogue engine's idle test showed exactly this, real,
        stable, but permanently undershooting. "pi" adds integral
        action, the same real control-theory fix already used for
        brake_target_rpm/throttle_target_rpm in engine_cycle_sim.py."""
        if not self._idle_governor_was_active:
            self._idle_map_integral = 0.0
            self._idle_governor_was_active = True
        error = (engine.idle_rpm - rpm) / max(engine.idle_rpm, 1.0)
        # smooth stall-risk gain schedule: 1.0x at/above STALL_RISK_RPM_FRAC
        # of idle_rpm, ramping linearly to STALL_RESCUE_KP_MULTIPLIER at
        # rpm=0 -- a real rescue response, not a harder version of the
        # same steady-state hunting correction
        rpm_frac = rpm / max(engine.idle_rpm, 1.0)
        risk = max(0.0, min(1.0, (STALL_RISK_RPM_FRAC - rpm_frac) / max(STALL_RISK_RPM_FRAC, 1e-6)))
        kp_base, ki_base = idle_pi_gains_frac(
            engine, engine.inertia_kg_m2 if coupled_inertia_kg_m2 is None else coupled_inertia_kg_m2)
        kp = kp_base * (1.0 + risk * (STALL_RESCUE_KP_MULTIPLIER - 1.0))
        ki = ki_base if engine.ecu.idle_governor_mode == "pi" else 0.0
        # the whole loop is calibrated in MANIFOLD pressure (what sets
        # torque), clamped there, and only then turned into the inlet
        # command by the blower's plant gain -- so a blown engine's
        # closed-throttle floor is the same 0.02 atm in the manifold an
        # NA engine gets, not 0.02 x PR
        manifold_cmd, self._idle_map_integral = pi_with_antiwindup(
            self._idle_map_integral, error, kp, ki, idle_base_manifold_frac(engine),
            self.idle_load_feedforward_frac(engine, ac_compressor_load_w),
            MAP_IDLE_FLOOR_FRAC, MAP_IDLE_CEILING_FRAC, dt)
        return manifold_cmd / blower_pressure_ratio(engine)

    def idle_fuel_quantity(self, engine: Engine, rpm: float, ac_compressor_load_w: float, dt: float,
                           coupled_inertia_kg_m2: float | None = None) -> float:
        """The diesel analogue of idle_map_target: compression-ignition
        engines aren't throttled on air (MAP tracks nothing near idle),
        so the idle governor acts directly on injected fuel quantity via
        a real injection-pump-style governor instead. Same
        engine.ecu.idle_governor_mode switch, same real "p alone
        permanently undershoots, pi closes it" reasoning -- one shared
        field drives both idle governors instead of the diesel path
        being silently stuck p-only while gasoline got fixed."""
        if not self._idle_fuel_governor_was_active:
            self._idle_fuel_integral = 0.0
            self._idle_fuel_governor_was_active = True
        error = (engine.idle_rpm - rpm) / max(engine.idle_rpm, 1.0)
        kp, ki_base = idle_pi_gains_frac(
            engine, engine.inertia_kg_m2 if coupled_inertia_kg_m2 is None else coupled_inertia_kg_m2)
        ki = ki_base if engine.ecu.idle_governor_mode == "pi" else 0.0
        cmd, self._idle_fuel_integral = pi_with_antiwindup(
            self._idle_fuel_integral, error, kp, ki, idle_fuel_base_frac(engine),
            self.idle_load_feedforward_frac(engine, ac_compressor_load_w),
            FUEL_IDLE_FLOOR_FRAC, FUEL_IDLE_CEILING_FRAC, dt)
        return cmd
