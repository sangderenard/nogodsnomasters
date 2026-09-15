"""A simple single-shaft gas turbine (Brayton cycle) -- the real
mechanism, not a lookup table: a compressor and a turbine on one common
shaft, a combustor between them, real isentropic gas relations
throughout. Reuses the exact same real compression-work relation
already in this codebase (drivetrain_graph.py's supercharger/turbo
intake-circuit compression work: cp*T1*(PR^((gamma-1)/gamma) - 1),
real air cp=1005 J/kgK, gamma=1.4) instead of inventing a separate
formula for what is physically the same process.

Real cycle, one shaft:
  1. COMPRESSOR draws ambient air, raises it to a real pressure ratio
     and temperature (isentropic compression, real efficiency loss).
  2. COMBUSTOR adds heat at ~constant pressure, raising the gas to
     turbine inlet temperature T3 -- the real control input a gas
     turbine actually has (a fuel valve setting how hot the gas
     entering the turbine gets), standing in here for full combustion
     chemistry the same honest way every other simplified stage in
     this catalogue names its one control simplification.
  3. TURBINE expands that hot gas back down, extracting real work
     (isentropic expansion, real efficiency loss) -- turbine work
     first has to cover whatever the compressor on the SAME shaft
     needs (the real "gas generator" torque balance every single-
     shaft turbine has to satisfy); only the surplus is net shaft
     power available to a load.
  4. EXHAUST -- the expanded gas leaves; nothing is recirculated.

Three disclosed simplifications (the real mechanism stands, only the
compressor/turbine's own detailed aerodynamic maps are stood in for):
  - Pressure ratio vs. shaft speed follows a real centrifugal-
    compressor affinity law, PR-1 proportional to (omega/omega_design)^2
    (tip-speed-squared scaling, the standard first-order real relation
    for a centrifugal stage) -- not a measured compressor map.
  - Mass flow vs. shaft speed follows the matching real first-order
    corrected-flow affinity law, mdot proportional to omega -- not a
    measured flow map either.
  - Turbine inlet temperature T3 is a direct control input (the
    "throttle"), not derived from a metered real fuel flow and a real
    flame/dilution model -- REAL fuel consumption is still derived FROM
    T3 (the combustor's own real energy balance, Q = mdot*cp*(T3-T2),
    divided by the fuel's real heating value), so the number you get
    for fuel burn is real, just not fed by a from-scratch combustion
    model the way the catalogue's piston engines are.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from regulator import ClosedLoopRegulator

# real air properties (same constants already used in drivetrain_graph.py's
# real compression-work relation)
CP_AIR_J_PER_KGK = 1005.0
GAMMA_AIR = 1.4
ISENTROPIC_EXPONENT = (GAMMA_AIR - 1.0) / GAMMA_AIR   # 0.2857
AMBIENT_T_K = 288.15
AMBIENT_P_PA = 101_325.0

# real jet-fuel-class heating value (kerosene, Jet-A/JP-8 class -- the
# real fuel this class of engine actually burns)
TURBINE_FUEL_HEATING_VALUE_J_PER_KG = 43_000_000.0

# real N1-limiting fuel governor: every actual gas turbine has one
# (a FADEC or a hydromechanical fuel control unit) for exactly the
# reason this module found empirically without it -- an open-loop fuel
# valve lets the compressor run away toward its own real overspeed/
# burst limit, since turbine work keeps growing with T3 faster than
# compressor work grows with speed. The governor's real job is to hold
# a COMMANDED N1 (what "throttle" actually means on a real turbine),
# not to set T3 directly.
MAX_T3_K = 1_400.0                 # real class-typical turbine inlet temperature limit
IDLE_N1_FRAC = 0.55                # real: turbines idle at a much higher N1 than a piston engine's idle rpm fraction -- below this the compressor is near surge/stall
MAX_N1_FRAC = 1.02                 # real: a small, deliberate overspeed margin above 100% design


@dataclass(frozen=True)
class TurbineSpec:
    """The real, DECLARED design point for one turbine engine -- what
    goes in the catalogue (engines.Engine.turbine) -- kept separate
    from SingleShaftGasTurbine itself, which also carries live runtime
    state (omega, T3, the fuel governor's own integrator). `build()`
    constructs a fresh runtime object from this spec, the same real
    separation this catalogue already uses everywhere else (a
    dataclass of declared numbers vs. the stateful sim object built
    from it)."""
    mdot_design_kg_s: float
    omega_design_rad_s: float
    design_pressure_ratio: float
    reduction_ratio: float          # real gearbox ratio: gas-generator omega / output shaft omega
    compressor_efficiency: float = 0.82
    turbine_efficiency: float = 0.88
    shaft_inertia_kg_m2: float = 0.05
    bearing_friction_nm: float = 2.0
    #: DROOP. A hydromechanical fuel control unit is a PROPORTIONAL
    #: device: flyweights sense N1, a spring opposes them, and the fuel
    #: valve sits wherever the two balance. Speed therefore SAGS under
    #: load and recovers only when the operator opens the throttle --
    #: typically 3-5% no-load to full-load on a turboshaft. That droop is
    #: not a defect to be tuned out: it is what lets two units share a
    #: load, and it is what the operator feels.
    #:
    #: Modelled as PI instead, integral action drives steady-state error
    #: to exactly zero and the engine holds one speed no matter what is
    #: done to it. Measured before this change: 7189 rpm at 0 Nm, at
    #: 4000 Nm and at 9000 Nm -- identical to four figures. See
    #: governor.CaptiveBallGovernor for the pattern to follow: simulate
    #: the mechanism and let droop fall out of the spring.
    governor_droop_frac: float = 0.04
    #: ACCELERATION SCHEDULE. A real FCU deliberately rate-limits fuel on
    #: acceleration, because dumping fuel into a compressor that has not
    #: caught up walks it into surge and cooks the turbine. It is why a
    #: turbine takes seconds to answer a throttle and a piston engine
    #: does not. Decel is NOT limited the same way -- taking fuel away is
    #: always safe, which is why one spools down faster than it spools up.
    accel_schedule_k_per_s: float = 260.0
    light_off_spool_time_s: float = 12.0   # real: how long a starter motors this rotor up to light-off speed

    def rated_shaft_torque_nm(self, max_t3_k: float = MAX_T3_K,
                              ambient_k: float = AMBIENT_T_K) -> float:
        """Peak torque this cycle can actually put on its OUTPUT shaft.

        Derived from the declared cycle rather than from the placeholder
        bmep/displacement an Engine row carries for a turbine -- those
        fields exist only so the piston-shaped parts of the catalogue
        have something to read, and sizing anything off them means
        sizing it off a number chosen to look plausible.

        Net specific work is turbine work less compressor work at the
        limiting turbine inlet temperature; times mass flow gives shaft
        power; divided by the output shaft's own speed gives torque.
        """
        ideal_c = CP_AIR_J_PER_KGK * ambient_k * (
            self.design_pressure_ratio ** ISENTROPIC_EXPONENT - 1.0)
        compressor_w = ideal_c / max(self.compressor_efficiency, 1e-3)
        turbine_w = (CP_AIR_J_PER_KGK * max_t3_k
                     * (1.0 - self.design_pressure_ratio ** -ISENTROPIC_EXPONENT)
                     * self.turbine_efficiency)
        net_w_per_kg = max(turbine_w - compressor_w, 0.0)
        power_w = net_w_per_kg * self.mdot_design_kg_s
        output_omega = self.omega_design_rad_s / max(self.reduction_ratio, 1e-6)
        return power_w / max(output_omega, 1e-6)

    def build(self) -> "SingleShaftGasTurbine":
        return SingleShaftGasTurbine(
            mdot_design_kg_s=self.mdot_design_kg_s, omega_design_rad_s=self.omega_design_rad_s,
            design_pressure_ratio=self.design_pressure_ratio,
            compressor_efficiency=self.compressor_efficiency, turbine_efficiency=self.turbine_efficiency,
            shaft_inertia_kg_m2=self.shaft_inertia_kg_m2, bearing_friction_nm=self.bearing_friction_nm,
            governor_droop_frac=self.governor_droop_frac,
            accel_schedule_k_per_s=self.accel_schedule_k_per_s)


@dataclass
class SingleShaftGasTurbine:
    """One real gas-generator spool: compressor + turbine, rigidly
    coupled on one shaft (no separate free power turbine -- a real
    turbojet-core/APU-class machine, the simplest real configuration).
    `mdot_design_kg_s` / `omega_design_rad_s` / `design_pressure_ratio`
    are this engine's own real design point (what its compressor
    affinity laws are anchored to); `shaft_inertia_kg_m2` is the real
    rotating assembly's own inertia (turbines spin light and fast, not
    heavy and slow like a piston crank -- genuinely tiny compared to
    this catalogue's reciprocating engines)."""
    mdot_design_kg_s: float
    omega_design_rad_s: float
    design_pressure_ratio: float
    compressor_efficiency: float = 0.82
    turbine_efficiency: float = 0.88
    shaft_inertia_kg_m2: float = 0.05
    bearing_friction_nm: float = 2.0
    #: see TurbineSpec for what these are and why
    governor_droop_frac: float = 0.04
    accel_schedule_k_per_s: float = 260.0

    omega_rad_s: float = field(default=0.0, init=False)
    t3_k: float = field(default=AMBIENT_T_K, init=False)
    _t3_command_k: float = field(default=AMBIENT_T_K, init=False)
    # the real fuel governor, built directly from the generic installable
    # regulator (regulator.py) -- not a bespoke inline PI loop: a real
    # fuel-control-unit's own job is exactly "hold N1 at a commanded
    # target by adjusting how much heat this line adds," which is
    # precisely what a ClosedLoopRegulator installed on the fuel line is
    fuel_governor: ClosedLoopRegulator = field(init=False)

    def __post_init__(self) -> None:
        # real, disclosed gains: K of commanded T3 per unit of
        # fractional N1 error, placed by the same practical tuning
        # approach a real hydromechanical fuel control unit's own
        # calibration gets -- fast enough to hold N1 against a load
        # step, gentle enough not to chase the compressor's own
        # omega^2 nonlinearity into oscillation
        # PROPORTIONAL ONLY: ki = 0, so droop exists by construction
        # rather than by tuning. The gain is DERIVED from the declared
        # droop rather than picked -- full fuel authority corresponds to
        # exactly `governor_droop_frac` of N1 error, so a 4% droop engine
        # really does sit 4% low at full load.
        self.fuel_governor = ClosedLoopRegulator(
            kp=(MAX_T3_K - AMBIENT_T_K) / max(self.governor_droop_frac, 1e-3),
            ki=0.0, output_min=AMBIENT_T_K, output_max=MAX_T3_K)
    fuel_flow_kg_s: float = field(default=0.0, init=False)
    pressure_ratio: float = field(default=1.0, init=False)
    mdot_kg_s: float = field(default=0.0, init=False)
    compressor_work_j_per_kg: float = field(default=0.0, init=False)
    turbine_work_j_per_kg: float = field(default=0.0, init=False)
    egt_k: float = field(default=AMBIENT_T_K, init=False)   # exhaust gas temperature -- the real cockpit gauge

    def reset(self) -> None:
        self.omega_rad_s = 0.0
        self.t3_k = AMBIENT_T_K
        self.fuel_flow_kg_s = 0.0
        self._t3_command_k = AMBIENT_T_K
        self.fuel_governor.reset()

    def step(self, dt: float, throttle: float, load_torque_nm: float = 0.0,
             starter_assist_torque_nm: float = 0.0) -> float:
        """`throttle` (0..1) commands turbine inlet temperature between
        ambient (no combustion) and a real ~1400 K class limit -- the
        real control a gas turbine's fuel governor actually has.
        Returns the net shaft torque this tick (for a caller that wants
        it; the shaft's own omega is already integrated internally,
        same ownership pattern as otto_langen.OttoLangenCylinder owning
        its own piston<->shaft solve)."""
        omega_frac = max(0.0, self.omega_rad_s) / max(self.omega_design_rad_s, 1e-6)
        # real centrifugal affinity laws: PR-1 and mdot both scale off
        # shaft speed, anchored at the declared design point
        self.pressure_ratio = 1.0 + (self.design_pressure_ratio - 1.0) * omega_frac * omega_frac
        self.mdot_kg_s = self.mdot_design_kg_s * omega_frac

        # a cold/stopped shaft moves no air and can't sustain combustion
        # (a real igniter needs airflow through it) -- a real minimum
        # light-off speed, below which the throttle command can't reach
        # the combustor at all (this is what the starter is actually
        # for: motoring the shaft up to light-off speed first)
        LIGHT_OFF_OMEGA_FRAC = 0.08
        can_fire = omega_frac >= LIGHT_OFF_OMEGA_FRAC
        if can_fire:
            # real N1 governor, installed on the fuel line as a generic
            # ClosedLoopRegulator (regulator.py): throttle commands a
            # target N1 between a real idle floor and a real max-
            # continuous ceiling, and the regulator closes that error by
            # commanding T3 -- this is what actually keeps the shaft at
            # a stable, commanded speed instead of running away (found
            # empirically: raw open-loop T3 spun this same compressor/
            # turbine pair past 300% N1 in under a minute, exactly the
            # real overspeed failure a fuel governor exists to prevent)
            target_n1 = IDLE_N1_FRAC + max(0.0, min(1.0, throttle)) * (MAX_N1_FRAC - IDLE_N1_FRAC)
            self._t3_command_k = self.fuel_governor.regulate(dt, target_n1, self.n1_frac, base=AMBIENT_T_K)
        else:
            self.fuel_governor.reset()
            self._t3_command_k = AMBIENT_T_K
        # real thermal lag: the combustor/turbine metal + gas thermal
        # mass doesn't track a fuel valve step instantly
        # The schedule and the lag are two different real things: the
        # schedule is a deliberate control limit on how fast fuel may be
        # ADDED, the lag is metal and gas not following a valve step. They
        # must be applied to the same quantity ONCE, not composed --
        # capping the lag's TARGET and then lagging toward the cap
        # multiplies the two limits together. Measured doing exactly that:
        # a declared 260 K/s schedule became an effective 3.25 K/s, and
        # the engine took minutes to reach a temperature it should reach
        # in seconds. So the lag computes the rise, and the schedule caps
        # that rise.
        T3_TAU_S = 0.4
        rise = (self._t3_command_k - self.t3_k) * min(1.0, dt / T3_TAU_S)
        if rise > 0.0 and self.accel_schedule_k_per_s > 0.0:
            rise = min(rise, self.accel_schedule_k_per_s * dt)
        self.t3_k += rise

        # compressor: real isentropic work (the exact relation already
        # used elsewhere in this codebase for supercharger/turbo
        # compression), efficiency loss as real extra shaft work for
        # the same pressure rise
        ideal_w_c = CP_AIR_J_PER_KGK * AMBIENT_T_K * (self.pressure_ratio ** ISENTROPIC_EXPONENT - 1.0)
        self.compressor_work_j_per_kg = ideal_w_c / max(self.compressor_efficiency, 1e-3)
        t2_k = AMBIENT_T_K + self.compressor_work_j_per_kg / CP_AIR_J_PER_KGK

        # combustor: real constant-pressure energy balance -- the ACTUAL
        # fuel flow this tick's T3 command requires, off the real fuel
        # heating value, not a throttle-proportional guess
        if self.mdot_kg_s > 1e-6 and self.t3_k > t2_k:
            q_j_per_kg = CP_AIR_J_PER_KGK * (self.t3_k - t2_k)
            self.fuel_flow_kg_s = self.mdot_kg_s * q_j_per_kg / TURBINE_FUEL_HEATING_VALUE_J_PER_KG
        else:
            self.fuel_flow_kg_s = 0.0

        # turbine: real isentropic expansion back through the same
        # pressure ratio (a direct-drive single shaft with no separate
        # power turbine -- the turbine has to pass through the same
        # pressure drop the compressor built, real for this real
        # simplest-case machine), efficiency loss as real work NOT
        # recovered for the same pressure drop
        ideal_w_t = CP_AIR_J_PER_KGK * self.t3_k * (1.0 - self.pressure_ratio ** (-ISENTROPIC_EXPONENT))
        self.turbine_work_j_per_kg = ideal_w_t * self.turbine_efficiency
        self.egt_k = self.t3_k - self.turbine_work_j_per_kg / CP_AIR_J_PER_KGK

        # the real gas-generator torque balance: turbine work first pays
        # for the compressor on the same shaft; only the surplus (or
        # deficit, while spooling from a cold start) accelerates/
        # decelerates the real shaft inertia against friction and
        # whatever external load (a generator, a prop, a dyno) is on it
        net_specific_work_j_per_kg = self.turbine_work_j_per_kg - self.compressor_work_j_per_kg
        gas_power_w = self.mdot_kg_s * net_specific_work_j_per_kg
        omega_for_torque = max(self.omega_rad_s, 50.0)   # avoid a divide-by-zero at true standstill
        gas_torque_nm = gas_power_w / omega_for_torque
        friction_nm = math.copysign(self.bearing_friction_nm, self.omega_rad_s) if self.omega_rad_s != 0.0 else 0.0
        net_torque_nm = gas_torque_nm - friction_nm - load_torque_nm + starter_assist_torque_nm

        self.omega_rad_s = max(0.0, self.omega_rad_s + net_torque_nm / self.shaft_inertia_kg_m2 * dt)
        return net_torque_nm

    @property
    def rpm(self) -> float:
        return self.omega_rad_s * 60.0 / (2.0 * math.pi)

    @property
    def n1_frac(self) -> float:
        """The real cockpit "N1" gauge: shaft speed as a fraction of
        design (100% N1), not raw rpm."""
        return self.omega_rad_s / max(self.omega_design_rad_s, 1e-6)
