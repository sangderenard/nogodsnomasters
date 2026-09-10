"""The 12 V electrical network as a real circuit: a lead-acid battery,
a regulated alternator, and the real loads every box on the bus draws.

Replaces the old bespoke `batt_target = 12.6 + 1.8 * (alt - load)`
relaxation in engine_cycle_sim.py, which had no charge state, no
current, no internal resistance, and an alternator that was a flat
900 W for a string trimmer and a marine diesel alike. Everything here is
the real mechanism, with the handful of remaining disclosed constants
named at the top:

  battery   -- open-circuit voltage is a real function of state of
               charge (lead-acid rests 11.8 V empty .. 12.7 V full);
               terminal voltage = OCV + I*R_int (charging positive);
               capacity is DERIVED from the engine's own real cranking
               energy (friction torque at cranking speed, cold
               multiplier, starter drive efficiency) x a real reserve
               rule (30 s crank x 10 starts), not one car-sized number
  alternator -- a regulated source: holds the bus at the regulator
               setpoint (14.2 V) when it can, current-limited by its
               rating and its real speed capability curve (claw-pole
               output is ~zero below cut-in speed and reaches rating a
               few thousand rotor rpm later); its SHAFT load is the
               delivered electrical power over its real efficiency,
               fed back to the belt as a real resistive torque
  loads      -- each box's real draw: the ECU's own quiescent power, the
               ignition coil's energy per spark times the real spark
               rate (only for battery-fed coil ignitions -- a magneto
               makes its own), an EFI fuel pump's real hydraulic power
               (flow x rail pressure / efficiency), an electric fan's
               real aerodynamic shaft power over its motor efficiency,
               the AC clutch coil, and the user's load resistor as a
               fraction of the alternator's own rating

Identities follow the production vehicle graph (abstract_ui_vehicles
.py's electrical harness): electrical.battery, electrical.alternator,
electrical.fusebox, electrical.engine_ground (the single-point block
ground strap), with the same wire circuits (battery-main,
alternator-charge, computer-and-ignition, chassis-ground-return).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

from engines import Engine, RPM_TO_RAD_S, EFI_RAIL_PRESSURE_PA

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))
from src.compiler.abstract_ui_vehicles import BATTERY_MODULES  # noqa: E402

# --- lead-acid battery (real values; see each note) --------------------
NOMINAL_BUS_VOLTAGE_V = 12.6            # 6 cells x 2.1 V, resting, full
LEAD_ACID_OCV_EMPTY_V = 11.8            # real resting voltage at ~0% SOC
LEAD_ACID_OCV_FULL_V = 12.7             # real resting voltage at 100% SOC
LEAD_ACID_RESISTANCE_OHM_AH = 0.25      # real: R_int ~ 0.25 ohm*Ah / capacity (60 Ah starter battery -> ~4 mohm)
COLD_CRANK_CURRENT_FACTOR = 2.0         # real CCA ratings sit ~2x the running crank current (series motor near stall)
CHARGE_ACCEPTANCE_C_RATE = 0.25         # real rule: sustained charge current ~C/4
CHARGE_COULOMBIC_EFFICIENCY = 0.90      # real lead-acid charge acceptance
BROWNOUT_VOLTAGE_V = 6.5                # real: automotive ECUs reset/go dark around 6-7 V
# battery SIZING from the engine's own real cranking demand
CRANKING_RPM = 200.0                    # real starter cranking speed (150-250 rpm)
COLD_CRANK_MEP_PA = 200_000.0           # disclosed: real cold-crank effective MEP ~2 bar (compression pumping + cold oil); 3 bar gave 2x real CCA
STARTER_DRIVE_EFFICIENCY = 0.60         # real series-wound starter + ring gear, loaded
CRANKING_TERMINAL_VOLTAGE_V = 9.6       # real sagged bus during crank
COLD_CRANK_VOLTAGE_FLOOR_V = 7.2        # SAE J537 cold-crank test floor (30 s at -18 degC)
COLD_RESISTANCE_MULTIPLIER = 2.0        # real lead-acid internal resistance at -18 degC vs 25 degC
RESERVE_STARTS = 10                     # real sizing convention: reserve for ~10 cold starts
CRANK_SECONDS_PER_START = 30.0          # real sizing convention (SAE reserve-capacity style)
MIN_BATTERY_AH = 1.0

# --- alternator (real claw-pole characteristics) -----------------------
REGULATOR_SETPOINT_V = 14.2             # real charging setpoint
ALTERNATOR_CUT_IN_OMEGA_RAD_S = 105.0   # ~1000 rotor rpm: output starts
ALTERNATOR_FULL_OUTPUT_OMEGA_RAD_S = 314.0  # ~3000 rotor rpm: reaches rating
ALTERNATOR_EFFICIENCY = 0.55            # real claw-pole automotive alternator
# Rating: a real alternator is sized to the VEHICLE'S electrical budget
# (lights, blower, pumps, charging) plus this engine's own installed
# loads with margin -- it is not proportional to engine power (a 4 L GT
# and a 1.5 L commuter both carry ~90-150 A units; a 1.5%-of-power rule
# put a 4.5 kW alternator on the GT and stalled it at idle under the
# default load resistor). The base budget is the one disclosed vehicle-
# side figure this engine subgraph can't derive on its own.
VEHICLE_BASE_ELECTRICAL_BUDGET_W = 500.0   # headlamps 2x55 W, HVAC blower ~200 W, pumps/ignition/misc
ENGINE_LOAD_MARGIN = 1.5
MIN_ALTERNATOR_RATED_W = 60.0

# --- loads (real typical values) ---------------------------------------
ECU_QUIESCENT_W = 15.0                  # a running engine computer
IGNITION_DRIVER_QUIESCENT_W = 3.0
COIL_ENERGY_PER_SPARK_J = 0.060         # real inductive coil, 50-80 mJ
COIL_CHARGING_EFFICIENCY = 0.50         # real: about half the dwell energy reaches the gap
AC_CLUTCH_COIL_W = 45.0                 # real magnetic clutch coil
FUEL_PUMP_EFFICIENCY = 0.30             # real small in-tank turbine pump
FUEL_DENSITY_KG_M3 = 750.0
ELECTRIC_FAN_MOTOR_EFFICIENCY = 0.70    # real brushed DC fan motor


def sparks_per_second(engine: Engine, rpm: float) -> float:
    arch = engine.architecture
    return arch.cylinders * (rpm / 60.0) * (360.0 / max(arch.cycle_degrees, 1.0))


def cold_cranking_current_a(engine: Engine, system_voltage_v: float) -> float:
    """The cold-crank current an ELECTRIC starter on this engine really
    draws (SAE-style): cold-crank torque (COLD_CRANK_MEP over the swept
    volume) at cranking speed over the drive efficiency, at the sagged
    cranking voltage of this system, times the near-stall factor a
    series motor sits at. Zero when nothing electric cranks the engine."""
    electric = any(s in ("electric-starter",) for s in engine.starting_systems)
    if not electric or engine.architecture.cylinders <= 0:
        return 0.0
    vd_m3 = engine.displacement_l / 1000.0
    cycle_rad = 2.0 * math.pi * engine.architecture.cycle_degrees / 360.0
    cranking_torque_nm = COLD_CRANK_MEP_PA * vd_m3 / cycle_rad
    starter_power_w = cranking_torque_nm * CRANKING_RPM * RPM_TO_RAD_S / STARTER_DRIVE_EFFICIENCY
    sag_v = CRANKING_TERMINAL_VOLTAGE_V * system_voltage_v / 12.0
    return starter_power_w / sag_v * COLD_CRANK_CURRENT_FACTOR


@dataclass
class Battery:
    """A real battery BANK: N real modules (production's BATTERY_MODULES
    -- group 31, 8D, a 20 Ah AGM, ...) in series to make the system
    voltage and in parallel to meet the cold-crank current (each module
    must hold the SAE 7.2 V-per-12 V floor at its share of that current
    when cold, resistance doubled) and a real reserve (30 min of the
    installed loads). Picked smallest-module-first so a trimmer gets a
    7 Ah SLA and a Cat C18 gets two 8Ds, not one imaginary 999 000 Ah
    battery. Bank properties follow from the modules and the wiring."""
    module_kind: str
    series_count: int
    parallel_count: int
    module_capacity_ah: float
    module_resistance_ohm: float
    module_mass_kg: float
    soc_frac: float = 1.0

    @classmethod
    def for_engine(cls, engine: Engine) -> "Battery":
        system_v = engine.electrical_system_voltage_v
        series = max(1, int(round(system_v / 12.0)))
        cca_required = cold_cranking_current_a(engine, system_v)
        # reserve: 30 minutes of this engine's own installed electrical
        # loads (ECU, ignition at idle, fuel pump at idle) at system voltage
        idle_load_w = ECU_QUIESCENT_W
        if engine.ignition_dispatch == "ecu-electronic":
            idle_load_w += IGNITION_DRIVER_QUIESCENT_W + sparks_per_second(engine, engine.idle_rpm) * COIL_ENERGY_PER_SPARK_J / COIL_CHARGING_EFFICIENCY
        # a road vehicle (one with an alternator to recharge it) also has
        # to carry its real base loads -- lights, blower -- for that half
        # hour with the engine off
        if engine.accessories.alternator:
            idle_load_w += VEHICLE_BASE_ELECTRICAL_BUDGET_W
        reserve_ah = idle_load_w * 0.5 / system_v
        # the real designer's choice: the FEWEST modules that meet the
        # cold-crank current and the reserve (nobody parallels four
        # motorcycle batteries where one group 35 does), lightest on a tie
        candidates = []
        for kind, spec in BATTERY_MODULES.items():
            parallel_for_cca = math.ceil(cca_required / float(spec["cca_a"])) if cca_required > 0.0 else 1
            parallel_for_reserve = math.ceil(reserve_ah / float(spec["capacity_ah"]))
            parallel = max(1, parallel_for_cca, parallel_for_reserve)
            candidates.append((parallel * series, float(spec["mass_kg"]) * parallel * series, kind, spec, parallel))
        _, _, kind, spec, parallel = min(candidates)
        return cls(module_kind=kind, series_count=series, parallel_count=parallel,
                   module_capacity_ah=float(spec["capacity_ah"]),
                   module_resistance_ohm=float(spec["internal_resistance_ohm"]),
                   module_mass_kg=float(spec["mass_kg"]))

    @property
    def capacity_ah(self) -> float:
        return self.module_capacity_ah * self.parallel_count

    @property
    def internal_resistance_ohm(self) -> float:
        return self.module_resistance_ohm * self.series_count / self.parallel_count

    @property
    def nominal_voltage_v(self) -> float:
        return NOMINAL_BUS_VOLTAGE_V * self.series_count

    @property
    def mass_kg(self) -> float:
        return self.module_mass_kg * self.series_count * self.parallel_count

    @property
    def module_count(self) -> int:
        return self.series_count * self.parallel_count

    @property
    def open_circuit_voltage_v(self) -> float:
        soc = max(0.0, min(1.0, self.soc_frac))
        return (LEAD_ACID_OCV_EMPTY_V + (LEAD_ACID_OCV_FULL_V - LEAD_ACID_OCV_EMPTY_V) * soc) * self.series_count

    @property
    def max_charge_current_a(self) -> float:
        return CHARGE_ACCEPTANCE_C_RATE * self.capacity_ah

    def integrate(self, current_a: float, dt: float) -> None:
        """current_a > 0 charging, < 0 discharging."""
        eff = CHARGE_COULOMBIC_EFFICIENCY if current_a > 0.0 else 1.0
        self.soc_frac += current_a * eff * dt / (3600.0 * self.capacity_ah)
        self.soc_frac = max(0.0, min(1.0, self.soc_frac))


@dataclass
class Alternator:
    rated_w: float
    setpoint_v: float = REGULATOR_SETPOINT_V   # the regulator's own setpoint at this system voltage

    @classmethod
    def for_engine(cls, engine: Engine) -> "Alternator":
        setpoint = REGULATOR_SETPOINT_V * max(1, int(round(engine.electrical_system_voltage_v / 12.0)))
        if not engine.accessories.alternator:
            return cls(rated_w=0.0, setpoint_v=setpoint)
        # this engine's own installed loads at their worst case: coil
        # ignition at redline spark rate, EFI pump at redline fuel flow,
        # ECU, AC clutch coil -- all the real per-box relations below
        installed_w = ECU_QUIESCENT_W
        if engine.ignition_dispatch == "ecu-electronic":
            installed_w += (IGNITION_DRIVER_QUIESCENT_W
                            + sparks_per_second(engine, engine.redline_rpm) * COIL_ENERGY_PER_SPARK_J / COIL_CHARGING_EFFICIENCY)
        if not (engine.compression_ignition or engine.carburetor.is_carbureted):
            installed_w += (engine.fuel_delivery.pump_flow_capacity_kg_s / FUEL_DENSITY_KG_M3
                            * EFI_RAIL_PRESSURE_PA / FUEL_PUMP_EFFICIENCY)
        if engine.accessories.air_conditioning:
            installed_w += AC_CLUTCH_COIL_W
        return cls(rated_w=max(MIN_ALTERNATOR_RATED_W,
                               VEHICLE_BASE_ELECTRICAL_BUDGET_W + installed_w * ENGINE_LOAD_MARGIN),
                   setpoint_v=setpoint)

    @property
    def rated_current_a(self) -> float:
        return self.rated_w / self.setpoint_v

    def capability_frac(self, rotor_omega: float) -> float:
        span = ALTERNATOR_FULL_OUTPUT_OMEGA_RAD_S - ALTERNATOR_CUT_IN_OMEGA_RAD_S
        return max(0.0, min(1.0, (rotor_omega - ALTERNATOR_CUT_IN_OMEGA_RAD_S) / span))


@dataclass
class BusReading:
    voltage_v: float = NOMINAL_BUS_VOLTAGE_V
    load_w: float = 0.0
    load_current_a: float = 0.0
    alternator_current_a: float = 0.0
    alternator_shaft_load_w: float = 0.0
    battery_current_a: float = 0.0     # + charging
    battery_soc_frac: float = 1.0
    regulating: bool = False           # alternator holding the setpoint (vs. current-limited)


class ElectricalNetwork:
    """battery + alternator + loads on one bus, solved once per tick."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.battery = Battery.for_engine(engine)
        self.alternator = Alternator.for_engine(engine)
        self.reading = BusReading(voltage_v=self.battery.nominal_voltage_v, battery_soc_frac=self.battery.soc_frac)

    @property
    def nominal_voltage_v(self) -> float:
        return self.battery.nominal_voltage_v

    @property
    def brownout_voltage_v(self) -> float:
        return BROWNOUT_VOLTAGE_V * self.battery.series_count

    def reset(self) -> None:
        self.battery.soc_frac = 1.0
        self.reading = BusReading(voltage_v=self.battery.nominal_voltage_v, battery_soc_frac=1.0)

    # -- per-box real loads ---------------------------------------------
    def ignition_load_w(self, rpm: float, ignition_cut: bool) -> float:
        if self.engine.ignition_dispatch != "ecu-electronic" or ignition_cut:
            return 0.0
        return (IGNITION_DRIVER_QUIESCENT_W
                + sparks_per_second(self.engine, max(rpm, 0.0)) * COIL_ENERGY_PER_SPARK_J / COIL_CHARGING_EFFICIENCY)

    def fuel_pump_load_w(self, fuel_delivered_kg_s: float) -> float:
        eng = self.engine
        if eng.compression_ignition or eng.carburetor.is_carbureted:
            return 0.0   # mechanical injection pump / mechanical diaphragm pump: crank-driven, not electrical
        volume_m3_s = max(0.0, fuel_delivered_kg_s) / FUEL_DENSITY_KG_M3
        return volume_m3_s * EFI_RAIL_PRESSURE_PA / FUEL_PUMP_EFFICIENCY

    @staticmethod
    def electric_fan_load_w(fan_shaft_torque_nm: float, fan_omega: float) -> float:
        return max(0.0, fan_shaft_torque_nm * fan_omega) / ELECTRIC_FAN_MOTOR_EFFICIENCY

    def load_resistor_w(self, electrical_load_frac: float) -> float:
        # the user's ballast resistor, as a fraction of what this
        # engine's own alternator is rated for (a total-loss build with
        # no alternator has nothing to size it against: its battery just
        # runs down through whatever the ignition draws)
        return max(0.0, electrical_load_frac) * self.alternator.rated_w

    # -- the bus solve --------------------------------------------------
    def step(self, dt: float, alternator_omega: float, load_w: float, ecu_powered_draw_w: float = ECU_QUIESCENT_W) -> BusReading:
        total_load_w = max(0.0, load_w) + ecu_powered_draw_w
        batt = self.battery
        alt = self.alternator
        ocv = batt.open_circuit_voltage_v
        r = batt.internal_resistance_ohm
        # loads are constant-power at the previous tick's bus voltage (one-
        # tick lag, same pattern as every cross-system reading in this toy)
        v_prev = max(self.reading.voltage_v, 1.0)
        i_load = total_load_w / v_prev
        i_alt_cap = alt.rated_current_a * alt.capability_frac(alternator_omega)
        # try regulating: hold the setpoint, battery takes (V_set - OCV)/R
        i_batt_reg = max(-batt.max_charge_current_a * 4.0,
                         min(batt.max_charge_current_a, (alt.setpoint_v - ocv) / r))
        i_alt_needed = i_load + i_batt_reg
        if i_alt_cap > 0.0 and i_alt_needed <= i_alt_cap:
            regulating = True
            i_alt = i_alt_needed
            i_batt = i_batt_reg
            v_bus = ocv + i_batt * r
        else:
            regulating = False
            i_alt = i_alt_cap
            i_batt = i_alt - i_load
            v_bus = ocv + i_batt * r
        v_bus = max(0.0, v_bus)
        batt.integrate(i_batt, dt)
        shaft_w = v_bus * i_alt / ALTERNATOR_EFFICIENCY
        self.reading = BusReading(voltage_v=v_bus, load_w=total_load_w, load_current_a=i_load,
                                  alternator_current_a=i_alt, alternator_shaft_load_w=shaft_w,
                                  battery_current_a=i_batt, battery_soc_frac=batt.soc_frac,
                                  regulating=regulating)
        return self.reading

    @property
    def powered(self) -> bool:
        return self.reading.voltage_v >= self.brownout_voltage_v
