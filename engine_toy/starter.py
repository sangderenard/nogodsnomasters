"""Starting systems as real torque sources on the crank.

The engine never "teleports" to idle any more: something has to turn the
crank against cold friction and compression until it fires, and that
something is a real box with real limits, feeding torque in through a
real attachment:

  electric-starter   a series-wound DC motor sized from THIS engine's own
                     cold-crank demand, driving the flywheel ring gear
                     through a Bendix pinion whose ratio comes from the
                     block's real size; it draws real current off the
                     12 V bus (the bus sags, the coil's spark energy drops
                     -- the real hard-start mechanism) and overruns when
                     the engine catches
  external-starter   the same motor on a pit cart at 24 V, on the crank
                     nose / blower snout (drag cars carry no starter)
  air-start          a slow-speed marine diesel: starting air from its own
                     receiver admitted to the cylinders on their working
                     stroke -- torque is pressure x piston area x crank
                     throw over the admission window, and every admission
                     really costs receiver air
  inertia-starter    a hand-wound flywheel (WWII radials): human power
                     stored for a while, then dumped into the crank
                     through a friction clutch and a big reduction
  recoil-pull        one pull of a rope on a ratchet drum (trimmers)
  hand-crank         a person on a small removable crank handle (light
                     antiques): short lever, one-hand grip force
  flywheel-bar       a person on a pry/spoke bar through the flywheel rim
                     (large single-cylinder stationary/industrial
                     engines -- real technique for engines too big to
                     wrist-crank): a much longer lever and a two-handed,
                     body-weight pull, applied as real short bumps timed
                     to roll the flywheel past compression rather than
                     continuous cranking -- a genuinely different real
                     starting aid from hand-crank, not the same one
                     scaled up. Verified real necessity: this engine
                     class's own friction plus one firing cycle's real
                     energy balance does not net positive against a bare
                     75 W wrist crank (found on the catalogue's own 8.2 L
                     single -- it hunts at ~50 rpm forever without more
                     leverage, exactly why these were never wrist-started
                     in practice).

Every mode ends up as one number the crank integrates -- an assist torque
-- plus whatever the source consumed (bus current, receiver air, human
effort). Anything the game wants to bolt on later feeds the same
attachment: EngineCycleSim.external_crank_torque_nm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from engines import Engine, RPM_TO_RAD_S
from drivetrain_port import ClutchPort
from electrical_network import (COLD_CRANK_MEP_PA, CRANKING_RPM, STARTER_DRIVE_EFFICIENCY,
                                NOMINAL_BUS_VOLTAGE_V)
import engine_geometry

# --- electric starter (real series-wound automotive starter) -----------
STARTER_DESIGN_EFFICIENCY = 0.60        # real loaded efficiency at the cranking point
PINION_PITCH_DIAMETER_M = 0.028         # real 9-10 tooth Bendix pinion
PINION_RATIO_MIN, PINION_RATIO_MAX = 6.0, 20.0
STARTER_MAX_CRANK_S = 15.0              # real thermal duty limit before you let it cool
EXTERNAL_STARTER_SUPPLY_V = 24.0        # pit-cart external starter supply
EXTERNAL_STARTER_REDUCTION = 20.0       # real geared external starter on a crank-nose hex
# --- air start ----------------------------------------------------------
AIR_START_ADMISSION_WINDOW_FRAC = 0.30  # real: air admitted over ~1/3 of the working stroke
AIR_START_MEAN_CRANK_FACTOR = 0.80      # mean of sin(theta) over that window
AIR_START_MIN_PRESSURE_PA = 400_000.0   # below this the air-start valves can't turn the engine (toy receiver tops at 8.3 bar)
# --- human effort (real ergonomics) --------------------------------------
HUMAN_HAND_CRANK_POWER_W = 75.0          # sustained arm cranking
HUMAN_HAND_CRANK_BURST_POWER_W = 300.0   # real anaerobic burst a person puts into a crank handle
HUMAN_BURST_DURATION_S = 12.0            # ... for about this long before it decays to the sustained figure
HUMAN_HAND_CRANK_TORQUE_MAX_NM = 60.0    # ~200 N one-hand grip on a 0.3 m crank handle
# flywheel-bar: real values for a two-handed pry bar through the rim/
# spokes of a large industrial engine's own flywheel -- a longer lever
# (~0.7 m) at real two-handed body-weight force (~350 N sustained pull,
# ~700 N on a hard bump), delivered as short repeated bumps (a real
# technique: rock it back off compression, then heave it through and
# past TDC) rather than continuous rotation -- modeled the same way as
# the recoil pull: real energy per bump, a real duty cycle between them.
FLYWHEEL_BAR_LEVER_M = 0.7
FLYWHEEL_BAR_SUSTAINED_FORCE_N = 350.0
FLYWHEEL_BAR_BUMP_FORCE_N = 700.0
FLYWHEEL_BAR_BUMP_ARC_RAD = 0.9          # real arc a person can sweep a bar through in one heave
FLYWHEEL_BAR_BUMP_DURATION_S = 0.5
FLYWHEEL_BAR_RESET_S = 1.0               # real time to reposition for the next bump
RECOIL_PULL_ENERGY_J = 40.0              # ~50 N over ~0.8 m of rope
RECOIL_PULL_DURATION_S = 0.6
RECOIL_ROPE_CRANK_TURNS = 4.0            # crank revolutions per full pull through the drum
INERTIA_STARTER_WIND_S = 20.0            # real hand-wind time before engagement
INERTIA_STARTER_MAX_ENERGY_J = 8_000.0   # real hand inertia starter flywheel energy
INERTIA_STARTER_FLYWHEEL_OMEGA_RAD_S = 1250.0   # ~12000 rpm flywheel at full wind
INERTIA_STARTER_REDUCTION = 100.0
# --- catch / drop-out ----------------------------------------------------
CATCH_RPM_FRAC_OF_IDLE = 0.6
CATCH_HOLD_S = 0.25

# --- air vane motor starter (trucks / industrial / oilfield diesels) ------
AIR_MOTOR_VOLUMETRIC_EFFICIENCY = 0.85     # real vane motor leakage
AIR_MOTOR_MECHANICAL_EFFICIENCY = 0.80

STARTING_SYSTEMS = ("electric-starter", "external-starter", "air-start", "air-motor-starter",
                    "inertia-starter", "recoil-pull", "hand-crank", "flywheel-bar")


@dataclass
class AirMotorStarter:
    """A pneumatic vane motor on a Bendix pinion, one-way at the ring
    gear: torque is gauge pressure times displacement per radian (T =
    p * D / 2*pi), sized so it delivers the cold-crank torque at the
    receiver's working pressure; air consumption is displacement times
    motor speed times the air density at receiver pressure -- a real
    draw off the same receiver the compressor fills."""
    pinion_ratio: float
    displacement_m3_per_rev: float

    @classmethod
    def for_engine(cls, engine: Engine) -> "AirMotorStarter":
        ring_gear_dia_m = 2.0 * engine_geometry.block_half_yz_m(engine)
        ratio = max(PINION_RATIO_MIN, min(PINION_RATIO_MAX, ring_gear_dia_m / PINION_PITCH_DIAMETER_M))
        t_motor = cold_cranking_torque_nm(engine) / (ratio * STARTER_DRIVE_EFFICIENCY)
        # sized at the engine's OWN receiver working pressure (engines.PneumaticSystem)
        p_gauge = engine.pneumatics.tank_pressure_pa - 101_325.0
        displacement = t_motor * 2.0 * math.pi / (p_gauge * AIR_MOTOR_MECHANICAL_EFFICIENCY)
        return cls(pinion_ratio=ratio, displacement_m3_per_rev=displacement)

    def crank_torque_and_air(self, crank_omega: float, receiver_pressure_pa: float) -> tuple[float, float]:
        p_gauge = max(0.0, receiver_pressure_pa - 101_325.0)
        if p_gauge <= 0.0:
            return 0.0, 0.0
        t_motor = p_gauge * self.displacement_m3_per_rev / (2.0 * math.pi) * AIR_MOTOR_MECHANICAL_EFFICIENCY
        omega_m = max(0.0, crank_omega) * self.pinion_ratio
        rho = 1.2 * receiver_pressure_pa / 101_325.0
        # even at stall the vanes leak; the flow term is the swept volume
        # at motor speed plus that leakage fraction of its design flow
        design_omega_m = CRANKING_RPM * RPM_TO_RAD_S * self.pinion_ratio
        swept_m3_s = self.displacement_m3_per_rev * (omega_m + (1.0 - AIR_MOTOR_VOLUMETRIC_EFFICIENCY) * design_omega_m) / (2.0 * math.pi)
        return t_motor * self.pinion_ratio * STARTER_DRIVE_EFFICIENCY, swept_m3_s * rho


def cold_cranking_torque_nm(engine: Engine) -> float:
    arch = engine.architecture
    cycle_rad = 2.0 * math.pi * arch.cycle_degrees / 360.0 if arch.cylinders else 4.0 * math.pi
    return COLD_CRANK_MEP_PA * (engine.displacement_l / 1000.0) / cycle_rad


@dataclass
class ElectricStarter:
    """Series DC motor: V = I*R + K*I*w (back-EMF proportional to
    current AND speed for a series field), T = K*I^2. R and K are set by
    the design point -- deliver the engine's cold-crank torque at
    cranking speed at nominal voltage at STARTER_DESIGN_EFFICIENCY --
    so a bigger engine gets a bigger, hungrier motor by construction."""
    pinion_ratio: float
    resistance_ohm: float
    k_series: float
    rated_power_w: float
    design_current_a: float
    supply_v: float | None = None    # None = the vehicle's own bus; a number = an external cart

    @classmethod
    def for_engine(cls, engine: Engine, external: bool = False) -> "ElectricStarter":
        if external:
            ratio = EXTERNAL_STARTER_REDUCTION
            v_nom = EXTERNAL_STARTER_SUPPLY_V
        else:
            ring_gear_dia_m = 2.0 * engine_geometry.block_half_yz_m(engine)
            ratio = max(PINION_RATIO_MIN, min(PINION_RATIO_MAX, ring_gear_dia_m / PINION_PITCH_DIAMETER_M))
            # wound for this engine's own system voltage (a 24 V starter on a 24 V truck)
            v_nom = NOMINAL_BUS_VOLTAGE_V * engine.electrical_system_voltage_v / 12.0
        omega_crank_d = CRANKING_RPM * RPM_TO_RAD_S
        omega_m_d = omega_crank_d * ratio
        t_m_d = cold_cranking_torque_nm(engine) / (ratio * STARTER_DRIVE_EFFICIENCY)
        i_d = max(1.0, t_m_d * omega_m_d / (STARTER_DESIGN_EFFICIENCY * v_nom))
        r = (1.0 - STARTER_DESIGN_EFFICIENCY) * v_nom / i_d
        k = STARTER_DESIGN_EFFICIENCY * v_nom / (i_d * omega_m_d)
        return cls(pinion_ratio=ratio, resistance_ohm=r, k_series=k, rated_power_w=t_m_d * omega_m_d,
                   design_current_a=i_d, supply_v=(v_nom if external else None))

    def crank_torque_and_current(self, crank_omega: float, bus_v: float) -> tuple[float, float]:
        v = self.supply_v if self.supply_v is not None else bus_v
        if v <= 0.5:
            return 0.0, 0.0
        omega_m = max(0.0, crank_omega) * self.pinion_ratio
        i = v / (self.resistance_ohm + self.k_series * omega_m)
        t_m = self.k_series * i * i
        return t_m * self.pinion_ratio * STARTER_DRIVE_EFFICIENCY, i


@dataclass
class StarterReading:
    assist_torque_nm: float = 0.0
    bus_current_a: float = 0.0
    air_kg_s: float = 0.0
    note: str = ""


class StartingSystems:
    """Every starting system an engine really carries (engines.Engine.
    starting_systems -- a Cat C18 can have an electric starter AND an
    air starter), presented as one: engage(kind=None) fires the primary
    (or a named one), readings sum, and any engaged one means cranking."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.systems = [StartingSystem(engine, kind) for kind in (engine.starting_systems or ("electric-starter",))]
        self.reading = StarterReading()

    @property
    def mode(self) -> str:
        return " + ".join(s.mode for s in self.systems)

    @property
    def engaged(self) -> bool:
        return any(s.engaged for s in self.systems)

    @property
    def motor(self) -> ElectricStarter | None:
        return next((s.motor for s in self.systems if s.motor is not None), None)

    @property
    def air_motor(self) -> AirMotorStarter | None:
        return next((s.air_motor for s in self.systems if s.air_motor is not None), None)

    def reset(self) -> None:
        for s in self.systems:
            s.reset()
        self.reading = StarterReading()

    def engage(self, kind: str | None = None) -> None:
        target = next((s for s in self.systems if kind is None or s.mode == kind), self.systems[0])
        target.engage()

    def release(self) -> None:
        for s in self.systems:
            s.release()

    def step(self, dt: float, crank_omega: float, rpm: float, bus_v: float,
             air_receiver_fill_frac: float, air_receiver_pressure_pa: float) -> StarterReading:
        total = StarterReading()
        notes = []
        for s in self.systems:
            r = s.step(dt, crank_omega, rpm, bus_v, air_receiver_fill_frac, air_receiver_pressure_pa)
            total.assist_torque_nm += r.assist_torque_nm
            total.bus_current_a += r.bus_current_a
            total.air_kg_s += r.air_kg_s
            if r.note:
                notes.append(f"{s.mode}: {r.note}" if len(self.systems) > 1 else r.note)
        total.note = "; ".join(notes)
        self.reading = total
        return total


class StartingSystem:
    """One engine's starting system, whichever kind it really has."""

    def __init__(self, engine: Engine, mode: str | None = None) -> None:
        self.engine = engine
        self.mode = mode or engine.starting_system
        self.motor: ElectricStarter | None = None
        self.air_motor: AirMotorStarter | None = None
        if self.mode in ("electric-starter", "external-starter"):
            self.motor = ElectricStarter.for_engine(engine, external=(self.mode == "external-starter"))
        elif self.mode == "air-motor-starter":
            self.air_motor = AirMotorStarter.for_engine(engine)
        self._inertia_j = 2.0 * INERTIA_STARTER_MAX_ENERGY_J / INERTIA_STARTER_FLYWHEEL_OMEGA_RAD_S ** 2
        t_max = 2.0 * cold_cranking_torque_nm(engine)
        self._inertia_clutch = ClutchPort(stiffness_nm_per_rad_s=t_max * 8.0, max_torque_nm=t_max)
        self.reset()

    def reset(self) -> None:
        self.engaged = False
        self.timer_s = 0.0
        self.catch_timer_s = 0.0
        self.phase = ""
        self._flywheel_omega = 0.0
        self._flywheel_energy_j = 0.0
        self._rope_left_s = 0.0
        self._bar_bump_left_s = 0.0
        self._bar_reset_left_s = 0.0
        self.reading = StarterReading()

    @property
    def cranking(self) -> bool:
        return self.engaged

    def engage(self) -> None:
        """The operator's action: key to START, hit the cart button,
        open the starting-air valve, start winding, yank the rope, or
        take the crank handle."""
        self.engaged = True
        self.timer_s = 0.0
        self.catch_timer_s = 0.0
        if self.mode == "inertia-starter":
            self.phase = "winding"
            self._flywheel_energy_j = 0.0
            self._flywheel_omega = 0.0
        elif self.mode == "recoil-pull":
            self.phase = "pulling"
            self._rope_left_s = RECOIL_PULL_DURATION_S
        elif self.mode == "flywheel-bar":
            self.phase = "bumping"
            self._bar_bump_left_s = FLYWHEEL_BAR_BUMP_DURATION_S
            self._bar_reset_left_s = 0.0
        else:
            self.phase = "cranking"

    def release(self) -> None:
        self.engaged = False
        self.phase = ""
        self.reading = StarterReading()

    def catch_rpm(self) -> float:
        idle = self.engine.idle_rpm
        if self.mode == "inertia-starter":
            return min(CATCH_RPM_FRAC_OF_IDLE * idle, 3.0 * self._crank_rpm_at_full_wind())
        return CATCH_RPM_FRAC_OF_IDLE * idle

    def _crank_rpm_at_full_wind(self) -> float:
        return INERTIA_STARTER_FLYWHEEL_OMEGA_RAD_S / INERTIA_STARTER_REDUCTION / RPM_TO_RAD_S

    def step(self, dt: float, crank_omega: float, rpm: float, bus_v: float,
             air_receiver_fill_frac: float, air_receiver_pressure_pa: float) -> StarterReading:
        """Advance the starter one tick and return what it does to the
        crank and what it consumed. Handles catch (engine running on its
        own above catch_rpm for CATCH_HOLD_S) and the duty limit."""
        if not self.engaged:
            self.reading = StarterReading()
            return self.reading
        eng = self.engine
        self.timer_s += dt
        assist = 0.0
        current = 0.0
        air = 0.0
        note = self.phase
        if self.mode in ("electric-starter", "external-starter") and self.motor is not None:
            assist, current = self.motor.crank_torque_and_current(crank_omega, bus_v)
            if self.motor.supply_v is not None:
                current = 0.0   # cart-fed: nothing off the vehicle bus
            note = "cart starter" if self.motor.supply_v else "cranking"
            if self.timer_s > STARTER_MAX_CRANK_S:
                self.release(); self.reading = StarterReading(note="starter duty limit"); return self.reading
        elif self.mode == "air-motor-starter" and self.air_motor is not None:
            if air_receiver_fill_frac > 0.0:
                assist, air = self.air_motor.crank_torque_and_air(crank_omega, air_receiver_pressure_pa)
                note = f"air motor, receiver {air_receiver_fill_frac * 100:.0f}%"
            else:
                self.release(); self.reading = StarterReading(note="receiver empty"); return self.reading
        elif self.mode == "air-start":
            p_gauge = max(0.0, air_receiver_pressure_pa - 101_325.0)
            if air_receiver_fill_frac > 0.0 and air_receiver_pressure_pa >= AIR_START_MIN_PRESSURE_PA:
                arch = eng.architecture
                piston_area = math.pi * arch.bore_m ** 2 / 4.0
                assist = (p_gauge * piston_area * (arch.stroke_m / 2.0) * arch.cylinders
                          * AIR_START_ADMISSION_WINDOW_FRAC * AIR_START_MEAN_CRANK_FACTOR)
                rho = 1.2 * air_receiver_pressure_pa / 101_325.0
                vd_cyl = eng.displacement_l / 1000.0 / max(arch.cylinders, 1)
                admissions_per_s = arch.cylinders * (max(rpm, 0.0) / 60.0) * (360.0 / max(arch.cycle_degrees, 1.0))
                # at standstill the open valves still blow down into the
                # cylinders that happen to be in their window
                admissions_per_s = max(admissions_per_s, 0.5)
                air = AIR_START_ADMISSION_WINDOW_FRAC * vd_cyl * rho * admissions_per_s
                note = f"air start, receiver {air_receiver_fill_frac * 100:.0f}%"
            else:
                note = "air start: receiver pressure too low"
                if air_receiver_fill_frac <= 0.0:
                    self.release(); self.reading = StarterReading(note="receiver empty"); return self.reading
        elif self.mode == "inertia-starter":
            if self.phase == "winding":
                self._flywheel_energy_j = min(INERTIA_STARTER_MAX_ENERGY_J,
                                              self._flywheel_energy_j + HUMAN_HAND_CRANK_POWER_W * dt)
                self._flywheel_omega = math.sqrt(2.0 * self._flywheel_energy_j / self._inertia_j)
                note = f"winding {self._flywheel_energy_j / (HUMAN_HAND_CRANK_POWER_W * INERTIA_STARTER_WIND_S) * 100:.0f}%"
                if self.timer_s >= INERTIA_STARTER_WIND_S:
                    self.phase = "engaged"
                return self._finish(assist, current, air, note)
            # engaged: friction clutch between the geared-down flywheel and the crank
            drive_omega = self._flywheel_omega / INERTIA_STARTER_REDUCTION
            assist = max(0.0, self._inertia_clutch.step(dt, drive_omega, crank_omega))
            # the flywheel pays for that torque (through the reduction)
            self._flywheel_omega = max(0.0, self._flywheel_omega
                                       - assist / INERTIA_STARTER_REDUCTION / self._inertia_j * dt)
            self._flywheel_energy_j = 0.5 * self._inertia_j * self._flywheel_omega ** 2
            note = f"inertia starter engaged, flywheel {self._flywheel_energy_j / INERTIA_STARTER_MAX_ENERGY_J * 100:.0f}%"
            if self._flywheel_omega < 1.0:
                self.release(); self.reading = StarterReading(note="inertia starter run down"); return self.reading
        elif self.mode == "recoil-pull":
            rope_omega = 2.0 * math.pi * RECOIL_ROPE_CRANK_TURNS / RECOIL_PULL_DURATION_S
            if self._rope_left_s > 0.0:
                self._rope_left_s -= dt
                # the pawl only drives while the rope is faster than the crank
                if crank_omega < rope_omega:
                    assist = RECOIL_PULL_ENERGY_J / (2.0 * math.pi * RECOIL_ROPE_CRANK_TURNS)
                note = "pulling"
            else:
                self.release(); self.reading = StarterReading(note="rope rewound"); return self.reading
        elif self.mode == "flywheel-bar":
            # a pry bar against the flywheel rim is direct contact, not a
            # one-way ratchet -- unlike the recoil rope's pawl, there's no
            # overrun to test at real cranking speeds (the bar's own
            # push speed, FLYWHEEL_BAR_BUMP_ARC_RAD/DURATION, is well
            # above anything this flywheel reaches under hand power
            # alone). Real cycle: heave the bar through its arc, let go
            # and reposition, heave again -- decaying bump force with the
            # same real fatigue envelope as the hand-crank burst.
            burst = math.exp(-self.timer_s / HUMAN_BURST_DURATION_S)
            bump_force_n = FLYWHEEL_BAR_SUSTAINED_FORCE_N + (
                FLYWHEEL_BAR_BUMP_FORCE_N - FLYWHEEL_BAR_SUSTAINED_FORCE_N) * burst
            if self._bar_bump_left_s > 0.0:
                self._bar_bump_left_s -= dt
                assist = bump_force_n * FLYWHEEL_BAR_LEVER_M
                note = "bumping"
            else:
                self._bar_reset_left_s -= dt
                note = "repositioning"
                if self._bar_reset_left_s <= 0.0:
                    self._bar_bump_left_s = FLYWHEEL_BAR_BUMP_DURATION_S
                    self._bar_reset_left_s = FLYWHEEL_BAR_RESET_S
            if self.timer_s > 90.0:
                self.release(); self.reading = StarterReading(note="operator exhausted"); return self.reading
        elif self.mode == "hand-crank":
            # burst power first (anaerobic), decaying exponentially to the
            # sustainable figure -- real cranking ergonomics
            burst = HUMAN_HAND_CRANK_BURST_POWER_W * math.exp(-self.timer_s / HUMAN_BURST_DURATION_S)
            power_w = max(HUMAN_HAND_CRANK_POWER_W, burst)
            assist = min(HUMAN_HAND_CRANK_TORQUE_MAX_NM, power_w / max(crank_omega, 1.0))
            note = "hand cranking (burst)" if burst > HUMAN_HAND_CRANK_POWER_W * 1.5 else "hand cranking"
            if self.timer_s > 60.0:
                self.release(); self.reading = StarterReading(note="operator exhausted"); return self.reading
        # catch: the engine is running on its own
        if rpm >= self.catch_rpm():
            self.catch_timer_s += dt
            if self.catch_timer_s >= CATCH_HOLD_S:
                self.release(); self.reading = StarterReading(note="caught"); return self.reading
        else:
            self.catch_timer_s = 0.0
        return self._finish(assist, current, air, note)

    def _finish(self, assist: float, current: float, air: float, note: str) -> StarterReading:
        self.reading = StarterReading(assist_torque_nm=assist, bus_current_a=current, air_kg_s=air, note=note)
        return self.reading
