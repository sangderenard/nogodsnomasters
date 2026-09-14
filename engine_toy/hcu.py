"""Hydraulic control unit for the station support system.

The HCU is a computer, not a support solver.  It reads measurements made by
the structure and hydraulic systems and returns valve, lock, and preload
commands.  ``OutriggerSet`` remains the hydraulic plant and ``LiveStructure``
remains the full beam evolution; this module is only the program between
their sensors and actuators, in the same sense that :mod:`ecu` is the program
between engine sensors and the throttle/fuel hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from regulator import ClosedLoopRegulator


INNER = ("inner.fr", "inner.fl", "inner.rl", "inner.rr")
OUTER = ("outer.a", "outer.b", "outer.c", "outer.d")
ALL_SUPPORTS = OUTER + INNER

HOLD = "hold"
ACTIVE_LEVEL = "active-level"
FORCE_PRELOAD = "incoming-force-preload"
READY_TO_SHIP = "ready-to-ship"
WALK_CORNERS = "walk-corners"
URGENT = "urgent"
FAULT = "fault"


@dataclass(frozen=True)
class StructureSensorFrame:
    """One simultaneous HCU input sample.

    Positions come from the beam engine after its current evolution step.
    Forces and pressures come from the graph-joint and hydraulic engines.
    No field here is an HCU estimate of the structure.
    """
    structure_positions_m: dict[str, tuple[float, float, float]]
    pillar_extensions_m: dict[str, float] = field(default_factory=dict)
    corner_extensions_m: dict[str, float] = field(default_factory=dict)
    corner_angles_deg: dict[str, float] = field(default_factory=dict)
    pad_clearance_m: dict[str, float] = field(default_factory=dict)
    pad_contact: dict[str, bool] = field(default_factory=dict)
    support_force_n: dict[str, float] = field(default_factory=dict)
    cylinder_pressure_pa: dict[str, float] = field(default_factory=dict)
    length_locked: dict[str, bool] = field(default_factory=dict)
    hydraulic_supply_pressure_pa: float = 0.0
    hydraulic_required_pressure_pa: float = 0.0
    engine_running: bool = True
    bus_voltage_v: float = 48.0
    unexpected_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    platform_engine_running: dict[str, bool] = field(default_factory=dict)
    platform_starter_engaged: dict[str, bool] = field(default_factory=dict)
    ignition_low: bool = False
    ignition_high: bool = False
    kill_low_signal_sum: float = 0.0
    kill_high_signal_sum: float = 0.0
    kill_consensus_required: float = 1.0
    backup_pump_available: bool = False
    backup_battery_soc: float = 1.0
    backup_reservoir_oil_l: float = 180.0


@dataclass(frozen=True)
class HCUCommand:
    mode: str
    phase: str
    pillar_valves: dict[str, float]
    corner_telescope_valves: dict[str, float]
    corner_swing_valves: dict[str, float]
    length_locks: dict[str, bool]
    corner_preload_target_n: dict[str, float]
    moving_corner: str | None = None
    fault: str | None = None
    engine_start_request: dict[str, bool] = field(default_factory=dict)
    engine_kill_request: dict[str, bool] = field(default_factory=dict)
    backup_pump_command: float = 0.0


@dataclass(frozen=True)
class EngineAuthorityCommand:
    start: bool = False
    kill: bool = False
    retry_delay_s: float = 0.0
    attempt: int = 0


class EnginePlatformAuthority:
    """Reusable start/stop authority for one engine platform.

    One ignition rail names this platform and authorises a start.  Removing
    that authority does not stop a running engine.  Its separate kill rail is
    a summed approval signal and stops the engine only at consensus.
    """

    def __init__(self, base_retry_s: float = 5.0,
                 maximum_retry_s: float = 300.0) -> None:
        self.base_retry_s = float(base_retry_s)
        self.maximum_retry_s = float(maximum_retry_s)
        self.attempts = 0
        self.cooldown_s = 0.0
        self._attempt_delay_s = 0.0

    def step(self, dt: float, *, running: bool, starter_engaged: bool,
             ignition_live: bool, kill_signal_sum: float,
             kill_consensus_required: float) -> EngineAuthorityCommand:
        if kill_signal_sum >= kill_consensus_required:
            if running:
                self.attempts = 0
                self.cooldown_s = 0.0
                return EngineAuthorityCommand(kill=True)
            return EngineAuthorityCommand()
        if running:
            self.attempts = 0
            self.cooldown_s = 0.0
            return EngineAuthorityCommand()
        if starter_engaged:
            # The delay is between completed attempts, not consumed while a
            # starter is still doing its real crank duty.
            self.cooldown_s = max(self.cooldown_s, self._attempt_delay_s)
            return EngineAuthorityCommand(
                retry_delay_s=self.cooldown_s, attempt=self.attempts)
        self.cooldown_s = max(0.0, self.cooldown_s - max(0.0, dt))
        if not ignition_live or self.cooldown_s > 0.0:
            return EngineAuthorityCommand(
                retry_delay_s=self.cooldown_s, attempt=self.attempts)
        self.attempts += 1
        self._attempt_delay_s = min(
            self.maximum_retry_s,
            self.base_retry_s * (2.0 ** max(0, self.attempts - 1)))
        self.cooldown_s = self._attempt_delay_s
        return EngineAuthorityCommand(start=True,
                                      retry_delay_s=self.cooldown_s,
                                      attempt=self.attempts)


@dataclass
class HCUBackupHydraulics:
    """The HCU's independent, slow, high-torque electric hydraulic source."""
    battery_voltage_v: float = 48.0
    battery_capacity_ah: float = 500.0
    battery_soc: float = 1.0
    usable_depth_of_discharge: float = 0.50
    motor_rated_w: float = 5_500.0
    motor_pump_efficiency: float = 0.78
    pump_displacement_cc_rev: float = 12.0
    pump_rpm: float = 600.0
    relief_pressure_pa: float = 21_000_000.0
    reservoir_gross_l: float = 220.0
    reservoir_oil_l: float = 180.0

    @property
    def geometric_flow_l_min(self) -> float:
        return self.pump_displacement_cc_rev * self.pump_rpm / 1000.0

    @property
    def usable_battery_kwh(self) -> float:
        return (self.battery_voltage_v * self.battery_capacity_ah / 1000.0
                * self.usable_depth_of_discharge)

    def step(self, dt: float, *, command: float, demand_l_min: float,
             required_pressure_pa: float) -> dict:
        duty = max(0.0, min(1.0, float(command)))
        pressure = min(self.relief_pressure_pa,
                       max(101_325.0, float(required_pressure_pa)))
        power_limited_flow = (self.motor_rated_w * self.motor_pump_efficiency
                              * 60_000.0 / pressure)
        flow = min(max(0.0, float(demand_l_min)),
                   self.geometric_flow_l_min * duty,
                   power_limited_flow * duty)
        hydraulic_w = pressure * flow / 60_000.0
        electrical_w = hydraulic_w / max(self.motor_pump_efficiency, 1e-9)
        total_battery_j = self.battery_voltage_v * self.battery_capacity_ah * 3600.0
        minimum_soc = 1.0 - self.usable_depth_of_discharge
        self.battery_soc = max(minimum_soc,
                               self.battery_soc - electrical_w * dt
                               / max(total_battery_j, 1.0))
        available = (self.battery_soc > minimum_soc + 1e-9
                     and self.reservoir_oil_l > 20.0)
        return {"available": available, "pressure_pa": pressure,
                "delivered_l_min": flow, "hydraulic_w": hydraulic_w,
                "electrical_w": electrical_w,
                "battery_soc": self.battery_soc,
                "reservoir_oil_l": self.reservoir_oil_l}


@dataclass
class NitrogenBottlePair:
    """Two isolated 50 L / 300 bar nitrogen service bottles."""
    bottle_volume_l: float = 50.0
    bottle_count: int = 2
    pressure_pa: float = 30_000_000.0
    temperature_k: float = 293.15
    regulated_standard_flow_l_min: float = 100.0

    def top_off(self, dt: float, *, receiver_pressure_pa: float,
                receiver_gas_volume_l: float,
                target_pressure_pa: float) -> dict:
        """Transfer nitrogen through the regulator using ideal-gas inventory."""
        gas_constant = 8.314462618
        source_volume_m3 = self.bottle_volume_l * self.bottle_count / 1000.0
        receiver_volume_m3 = max(receiver_gas_volume_l, 1e-9) / 1000.0
        target = min(max(receiver_pressure_pa, target_pressure_pa),
                     self.pressure_pa)
        wanted_mol = max(0.0, (target - receiver_pressure_pa)
                         * receiver_volume_m3
                         / (gas_constant * self.temperature_k))
        standard_mol_s = (self.regulated_standard_flow_l_min / 1000.0 / 60.0
                          * 101_325.0 / (gas_constant * 293.15))
        available_mol = max(0.0, (self.pressure_pa - target)
                            * source_volume_m3
                            / (gas_constant * self.temperature_k))
        moved_mol = min(wanted_mol, available_mol,
                        standard_mol_s * max(0.0, dt))
        receiver_pressure = (receiver_pressure_pa + moved_mol
                             * gas_constant * self.temperature_k
                             / receiver_volume_m3)
        self.pressure_pa = max(101_325.0, self.pressure_pa - moved_mol
                               * gas_constant * self.temperature_k
                               / source_volume_m3)
        return {"receiver_pressure_pa": receiver_pressure,
                "bottle_pressure_pa": self.pressure_pa,
                "transferred_mol": moved_mol}


class HydraulicSwitchboard:
    """HCU proportional source-to-draw dispatch without inventing force.

    Source records contain measured pressure and available flow.  Draw
    records contain requested flow, required pressure, and priority.  The
    returned matrix is what the real proportional/check-valve bank is asked
    to pass; downstream hydraulic engines still compute pressure loss and
    actuator work.
    """

    def dispatch(self, sources: dict, draws: dict,
                 source_weights: dict[str, float] | None = None) -> dict:
        remaining = {name: max(0.0, float(value["available_flow_l_min"]))
                     for name, value in sources.items()}
        weights = {name: max(0.0, float((source_weights or {}).get(name, 1.0)))
                   for name in sources}
        contributions = {draw: {source: 0.0 for source in sources}
                         for draw in draws}
        unserved = {}
        ordered = sorted(draws.items(),
                         key=lambda item: float(item[1].get("priority", 0.0)),
                         reverse=True)
        for draw, demand in ordered:
            wanted = max(0.0, float(demand.get("requested_flow_l_min", 0.0)))
            required = max(0.0, float(demand.get("required_pressure_pa", 0.0)))
            eligible = [name for name, source in sources.items()
                        if float(source.get("pressure_pa", 0.0)) >= required
                        and remaining[name] > 0.0 and weights[name] > 0.0]
            left = wanted
            while left > 1e-9 and eligible:
                total_weight = sum(weights[name] for name in eligible)
                proposed = {name: left * weights[name] / total_weight
                            for name in eligible}
                moved = 0.0
                next_eligible = []
                for name in eligible:
                    take = min(remaining[name], proposed[name])
                    contributions[draw][name] += take
                    remaining[name] -= take
                    left -= take
                    moved += take
                    if remaining[name] > 1e-9:
                        next_eligible.append(name)
                if moved <= 1e-12:
                    break
                eligible = next_eligible
            unserved[draw] = max(0.0, left)
        return {"contributions_l_min": contributions,
                "remaining_source_l_min": remaining,
                "unserved_draw_l_min": unserved}


class HydraulicControlUnit:
    """The station HCU's persistent program and controller state."""

    PAD_HOVER_M = 0.010
    PAD_HOVER_TOLERANCE_M = 0.001
    LEVEL_TOLERANCE_M = 0.001
    FORCE_ABORT_N = 5_000.0
    BROWNOUT_V = 36.0
    PRESSURE_MARGIN_PA = 250_000.0

    def __init__(self) -> None:
        self.powered = True
        self.mode = HOLD
        self.phase = "holding"
        self.target_deck_y_m: float | None = None
        self.target_pillar_extension_m = 0.0
        self.incoming_force_n = np.zeros(3)
        self.base_preload_n = 60_000.0
        self.walk_deploy = False
        self.walk_order = list(OUTER)
        self._walk_index = 0
        self.fault: str | None = None
        self._level = {
            # At 1.0 the four 140 mm pillars ask for 222 L/min and line
            # loss dominates the load.  A quarter spool is the actual plant
            # schedule: about 55 L/min, 89 bar and 9.6 kW at design mass.
            name: ClosedLoopRegulator(4.0, 0.6, -0.25, 0.25,
                                      integral_clamp=0.30)
            for name in INNER
        }
        self._preload = {
            name: ClosedLoopRegulator(1.0 / 120_000.0,
                                      1.0 / 600_000.0,
                                      -0.30, 0.30,
                                      integral_clamp=120_000.0)
            for name in OUTER
        }
        self._engine_authority = {
            platform: EnginePlatformAuthority() for platform in ("low", "high")
        }

    def reset(self) -> None:
        self.mode, self.phase, self.fault = HOLD, "holding", None
        self.target_deck_y_m = None
        self._walk_index = 0
        for controller in (*self._level.values(), *self._preload.values()):
            controller.reset()

    def active_level(self, target_deck_y_m: float | None = None) -> None:
        self.mode, self.phase, self.fault = ACTIVE_LEVEL, "levelling", None
        self.target_deck_y_m = target_deck_y_m

    def preload_for_force(self, force_on_structure_n, *,
                          base_preload_n: float = 60_000.0,
                          target_deck_y_m: float | None = None) -> None:
        self.mode, self.phase, self.fault = FORCE_PRELOAD, "preloading", None
        self.incoming_force_n = np.asarray(force_on_structure_n, float)
        self.base_preload_n = max(0.0, float(base_preload_n))
        if target_deck_y_m is not None:
            self.target_deck_y_m = float(target_deck_y_m)

    def ready_to_ship(self, pillar_extension_m: float) -> None:
        """Raise on the four pillars, then tuck corners one at a time."""
        self.mode, self.phase, self.fault = READY_TO_SHIP, "raise-on-pillars", None
        self.target_pillar_extension_m = max(0.0, float(pillar_extension_m))
        self.walk_deploy = False
        self._walk_index = 0

    def urgent(self) -> None:
        """Maximum-response mode: start both plants and open all supports."""
        self.mode, self.phase, self.fault = URGENT, "full-oil-to-outriggers", None

    def walk_corners(self, *, deploy: bool,
                     order: tuple[str, ...] = OUTER) -> None:
        """Reconfigure corners serially while the moving pad hovers at 10 mm."""
        if set(order) != set(OUTER) or len(order) != len(OUTER):
            raise ValueError("corner walk order must contain every outer corner once")
        self.mode, self.phase, self.fault = WALK_CORNERS, "hover-pad", None
        self.walk_deploy = bool(deploy)
        self.walk_order = list(order)
        self._walk_index = 0

    @staticmethod
    def _clamp_command(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def _deck_target(self, sensors: StructureSensorFrame) -> float:
        ys = [float(sensors.structure_positions_m[name][1]) for name in INNER]
        if self.target_deck_y_m is None:
            self.target_deck_y_m = sum(ys) / len(ys)
        return self.target_deck_y_m

    def _level_commands(self, dt: float,
                        sensors: StructureSensorFrame) -> dict[str, float]:
        target = self._deck_target(sensors)
        return {name: self._level[name].regulate(
                    dt, target, float(sensors.structure_positions_m[name][1]))
                for name in INNER}

    def _preload_targets(self, sensors: StructureSensorFrame) -> dict[str, float]:
        horizontal = self.incoming_force_n[[0, 2]]
        magnitude = float(np.linalg.norm(horizontal))
        direction = horizontal / magnitude if magnitude > 1e-9 else np.zeros(2)
        targets = {}
        for name in OUTER:
            p = np.asarray(sensors.structure_positions_m[name], float)[[0, 2]]
            radius = p / max(float(np.linalg.norm(p)), 1e-9)
            # ``incoming_force_n`` is the force ON the structure.  The ground
            # support lying into that force is biased into compression; the
            # opposing support retains half the base preload, never slack.
            directional = float(np.dot(radius, direction))
            targets[name] = self.base_preload_n * (1.0 + 0.5 * directional)
        return targets

    def _safe_to_move(self, sensors: StructureSensorFrame) -> bool:
        return (sensors.engine_running
                and sensors.bus_voltage_v >= self.BROWNOUT_V
                and sensors.hydraulic_supply_pressure_pa
                    >= sensors.hydraulic_required_pressure_pa
                       + self.PRESSURE_MARGIN_PA
                and all(sensors.length_locked.get(name, False) for name in INNER))

    def _walk(self, sensors: StructureSensorFrame,
              telescope: dict[str, float], swing: dict[str, float],
              locks: dict[str, bool]) -> str | None:
        if self._walk_index >= len(self.walk_order):
            self.phase = "corners-deployed" if self.walk_deploy else "corners-tucked"
            return None
        moving = self.walk_order[self._walk_index]
        for name in OUTER:
            locks[name] = name != moving

        unexpected = float(np.linalg.norm(np.asarray(sensors.unexpected_force_n, float)))
        if unexpected >= self.FORCE_ABORT_N:
            self.phase = "emergency-seat"
            telescope[moving] = 1.0
            swing[moving] = 0.0
            if sensors.pad_contact.get(moving, False):
                locks[moving] = True
                telescope[moving] = 0.0
                self.mode = FAULT
                self.fault = (f"unexpected {unexpected:.0f} N load while {moving} "
                              "was walking; pad seated and locked")
            return moving

        clearance = float(sensors.pad_clearance_m.get(moving, 0.0))
        error = self.PAD_HOVER_M - clearance
        # Positive telescope command moves the pad down, reducing clearance.
        telescope[moving] = self._clamp_command(-6.0 * error)
        if abs(error) > self.PAD_HOVER_TOLERANCE_M:
            self.phase = "hover-pad"
            return moving

        self.phase = "swing-corner"
        swing[moving] = 0.35 if self.walk_deploy else -0.35
        angle = float(sensors.corner_angles_deg.get(moving, 0.0))
        at_angle = angle >= 33.5 if self.walk_deploy else angle <= 0.5
        if at_angle:
            self._walk_index += 1
            self.phase = "hover-pad" if self._walk_index < len(self.walk_order) \
                else ("corners-deployed" if self.walk_deploy else "corners-tucked")
        return moving

    def step(self, dt: float, sensors: StructureSensorFrame) -> HCUCommand:
        pillar = {name: 0.0 for name in INNER}
        telescope = {name: 0.0 for name in OUTER}
        swing = {name: 0.0 for name in OUTER}
        locks = {name: True for name in ALL_SUPPORTS}
        preload = {name: 0.0 for name in OUTER}
        moving = None
        backup_pump = 0.0
        engine_start = {}
        engine_kill = {}

        for platform, authority in self._engine_authority.items():
            ignition = (True if self.mode == URGENT else
                        sensors.ignition_low if platform == "low"
                        else sensors.ignition_high)
            kill_sum = (sensors.kill_low_signal_sum if platform == "low"
                        else sensors.kill_high_signal_sum)
            command = authority.step(
                dt,
                running=sensors.platform_engine_running.get(platform, False),
                starter_engaged=sensors.platform_starter_engaged.get(platform, False),
                ignition_live=ignition,
                kill_signal_sum=kill_sum,
                kill_consensus_required=sensors.kill_consensus_required)
            engine_start[platform] = command.start
            engine_kill[platform] = command.kill

        if not self.powered or sensors.bus_voltage_v < self.BROWNOUT_V:
            return HCUCommand(FAULT, "locked-on-power-loss", pillar, telescope,
                              swing, locks, preload, fault="HCU brownout",
                              engine_start_request=engine_start,
                              engine_kill_request=engine_kill,
                              backup_pump_command=backup_pump)

        if self.mode in (ACTIVE_LEVEL, FORCE_PRELOAD, WALK_CORNERS):
            pillar.update(self._level_commands(dt, sensors))

        if self.mode == FORCE_PRELOAD:
            preload = self._preload_targets(sensors)
            for name in OUTER:
                telescope[name] = self._preload[name].regulate(
                    dt, preload[name], sensors.support_force_n.get(name, 0.0))
        elif self.mode == URGENT:
            pillar = {name: 1.0 for name in INNER}
            telescope = {name: 1.0 for name in OUTER}
            locks = {name: False for name in ALL_SUPPORTS}
            backup_pump = 1.0
        elif self.mode == READY_TO_SHIP:
            extension_error = min(
                self.target_pillar_extension_m
                - sensors.pillar_extensions_m.get(name, 0.0)
                for name in INNER)
            if extension_error > self.LEVEL_TOLERANCE_M:
                if self._safe_to_move(sensors):
                    pillar = {name: 0.25 for name in INNER}
                else:
                    self.phase = "waiting-for-plant"
                    if (sensors.backup_pump_available
                            and sensors.backup_battery_soc > .50
                            and sensors.backup_reservoir_oil_l > 20.0):
                        backup_pump = 1.0
            else:
                self.phase = "hover-pad"
                moving = self._walk(sensors, telescope, swing, locks)
        elif self.mode == WALK_CORNERS:
            if self._safe_to_move(sensors):
                moving = self._walk(sensors, telescope, swing, locks)
            else:
                self.phase = "waiting-for-plant"
                if (sensors.backup_pump_available
                        and sensors.backup_battery_soc > .50
                        and sensors.backup_reservoir_oil_l > 20.0):
                    backup_pump = 1.0

        return HCUCommand(self.mode, self.phase, pillar, telescope, swing,
                          locks, preload, moving_corner=moving,
                          fault=self.fault,
                          engine_start_request=engine_start,
                          engine_kill_request=engine_kill,
                          backup_pump_command=backup_pump)


def apply_engine_authority(sim, *, start: bool, kill: bool) -> None:
    """Apply one HCU rail decision through the engine's real controls."""
    if kill:
        sim.stop()
    elif start and not sim.starter.engaged and sim.state.rpm <= 0.5:
        sim.engage_starter()


def structure_sensor_frame(structure, *, hydraulic=None,
                           ground_y_m: float | None = None,
                           **overrides) -> StructureSensorFrame:
    """Read the HCU's structural positions from a live beam-engine state."""
    xyz = structure.positions()
    index = structure.solver.index
    positions = {}
    for name in INNER:
        node = f"stand.work.{name.split('.')[-1]}"
        positions[name] = tuple(float(v) for v in xyz[index[node]])
    for name in OUTER:
        node = f"stand.leg.{name}.top"
        positions[name] = tuple(float(v) for v in xyz[index[node]])
    if ground_y_m is None:
        ground_y_m = min(float(xyz[index[f"stand.leg.{name}.pad"]][1])
                         for name in ALL_SUPPORTS)
    clearance = {
        name: max(0.0, float(xyz[index[f"stand.leg.{name}.pad"]][1])
                  - ground_y_m)
        for name in OUTER
    }
    supplied = {} if hydraulic is None else dict(hydraulic)
    supplied.update(overrides)
    return StructureSensorFrame(positions, pad_clearance_m=clearance,
                                **supplied)


def emit_hcu(g, stand, site: dict) -> dict:
    """Install the HCU, four structural sensors, and its valve manifold."""
    h = stand.half
    computer = "stand.hcu"
    manifold = "stand.hcu.valve_manifold"
    rear_right = site["corners"]["rr"]
    rear_left = site["corners"]["rl"]
    g.motion_group, g.assembly = "frame", "support-control"
    g.node(computer, (0.0, stand.deck_y - 0.22, -h + 0.13),
           "vehicle-computer", material="aluminium-cast",
           mass_in_total=True, mass_kg=14.0, in_view=True,
           half_extent_m=(0.24, 0.09, 0.16), part_role="hydraulic-control-unit",
           program="hcu.HydraulicControlUnit",
           control_authority="valves-and-positive-length-locks-only",
           powered_by="station-48v-dc")
    g.node(manifold, (0.0, stand.deck_y - 0.40, -h + 0.13),
           "hydraulic-valve-manifold", material="steel-plate",
           mass_in_total=True, mass_kg=58.0, in_view=True,
           half_extent_m=(0.46, 0.16, 0.20), part_role="support-valve-manifold",
           valve_channels=16, relief_pressure_pa=35_000_000.0,
           pressure_header_nominal_id_m=0.050,
           return_header_nominal_id_m=0.065,
           rated_flow_l_min=240.0,
           normal_operation_oil_sections="isolated",
           urgent_operation_oil_sections="common-blended-bus",
           flush_supply_port=True, flush_return_port=True,
           inline_contamination_monitor=True)
    booster = "stand.hcu.pressure_booster"
    hand_pump = "stand.hcu.emergency_hand_pump"
    battery = "stand.hcu.backup_battery"
    electric_pump = "stand.hcu.backup_electric_pump"
    reservoir = "stand.hcu.backup_reservoir"
    accumulator = "stand.hcu.accumulator"
    switchboard = "stand.hcu.solenoid_switchboard"
    coolant_manifold = "stand.hcu.coolant_manifold"
    g.node(booster, (-0.55, stand.deck_y - 0.40, -h + 0.13),
           "hydraulic-pressure-intensifier", material="steel-plate",
           mass_in_total=True, mass_kg=44.0, in_view=True,
           half_extent_m=(0.18, 0.16, 0.18),
           part_role="hcu-local-pressure-booster",
           pressure_ratio=1.6, maximum_outlet_pressure_pa=35_000_000.0,
           note="higher pressure at proportionally reduced flow; not a speed boost")
    g.node(hand_pump, (0.60, stand.deck_y - 0.40, -h + 0.13),
           "manual-hydraulic-pump", material="steel-plate",
           mass_in_total=True, mass_kg=19.0, in_view=True,
           half_extent_m=(0.18, 0.12, 0.16),
           part_role="five-day-emergency-support-pump",
           displacement_l_per_stroke=0.008, sustainable_strokes_per_min=20.0,
           sustainable_duty_fraction=0.10,
           average_flow_l_min=0.016,
           handle_force_at_89bar_n=237.0)
    g.node(battery, (0.62, stand.deck_y - 0.82, -h + 0.13),
           "lead-acid-traction-battery", material="steel-plate",
           mass_in_total=True, mass_kg=680.0, in_view=True,
           half_extent_m=(0.48, 0.27, 0.32),
           part_role="hcu-independent-48v-battery",
           nominal_voltage_v=48.0, capacity_ah=500.0,
           stored_energy_kwh=24.0, usable_energy_kwh=12.0,
           chemistry="flooded-lead-acid")
    g.node(electric_pump, (-0.58, stand.deck_y - 0.78, -h + 0.13),
           "electric-hydraulic-pump", material="steel-plate",
           mass_in_total=True, mass_kg=76.0, in_view=True,
           half_extent_m=(0.30, 0.19, 0.22),
           part_role="hcu-slow-torque-backup-pump",
           motor_voltage_v=48.0, motor_rated_w=5_500.0,
           pump_displacement_cc_rev=12.0, pump_rpm=600.0,
           rated_flow_l_min=7.2, relief_pressure_pa=21_000_000.0)
    g.node(reservoir, (0.0, stand.deck_y - 1.12, -h + 0.13),
           "hydraulic-reservoir", material="steel-plate",
           mass_in_total=True, mass_kg=225.0, in_view=True,
           half_extent_m=(0.58, 0.28, 0.34),
           part_role="hcu-independent-baffled-reservoir",
           gross_volume_l=220.0, oil_volume_l=180.0,
           fluid="hydraulic-oil", baffled=True)
    g.node(accumulator, (-1.05, stand.deck_y - 0.68, -h + 0.13),
           "high-pressure-canister", material="pressed-steel",
           mass_in_total=True, mass_kg=54.0, in_view=True,
           shape="drum", drum_axis=(0.0, 1.0, 0.0),
           drum_radius_m=.16, drum_length_m=.72,
           half_extent_m=(.16, .36, .16),
           part_role="hcu-gas-over-oil-accumulator",
           gas="nitrogen", gas_volume_l=40.0,
           precharge_pressure_pa=12_000_000.0,
           maximum_pressure_pa=35_000_000.0)
    g.node(switchboard, (0.0, stand.deck_y - 0.64, -h + 0.13),
           "electrohydraulic-solenoid-switchboard", material="aluminium-cast",
           mass_in_total=True, mass_kg=46.0, in_view=True,
           half_extent_m=(.44, .14, .20),
           part_role="hcu-programmable-source-draw-switchboard",
           proportional_inlet_channels=5, proportional_draw_channels=3,
           inlet_check_valves=True, load_sense=True,
           isolated_pump_cartridges=True,
           draw_networks=("support", "turret-motion", "recoil-service"),
           urgent_common_oil_bus=True,
           post_urgent_flush_required=True)
    g.node(coolant_manifold, (0.92, stand.deck_y - 0.64, -h + 0.13),
           "coolant-solenoid-manifold", material="aluminium-cast",
           mass_in_total=True, mass_kg=31.0, in_view=True,
           half_extent_m=(.30, .14, .18),
           part_role="hcu-coolant-manifold-and-control",
           circuits_isolated=True, channels=("low-platform", "high-platform"),
           supply_header_nominal_id_m=.038,
           return_header_nominal_id_m=.044,
           bypass_valves=True, temperature_sensors_per_channel=True,
           control_program="station_cooling.SiteCoolingRuntime")
    for ident, node, radius in ((f"{computer}.mount.rr", rear_right, .018),
                                (f"{computer}.mount.rl", rear_left, .018),
                                (f"{manifold}.mount.rr", rear_right, .024),
                                (f"{manifold}.mount.rl", rear_left, .024),
                                (f"{booster}.mount", rear_left, .022),
                                (f"{hand_pump}.mount", rear_right, .018),
                                (f"{battery}.mount", rear_right, .030),
                                (f"{battery}.mount.secondary", rear_left, .030),
                                (f"{electric_pump}.mount", rear_left, .026),
                                (f"{reservoir}.mount.rr", rear_right, .032),
                                (f"{reservoir}.mount.rl", rear_left, .032),
                                (f"{accumulator}.mount", rear_left, .025),
                                (f"{switchboard}.mount.rr", rear_right, .024),
                                (f"{switchboard}.mount.rl", rear_left, .024),
                                (f"{coolant_manifold}.mount.rr", rear_right, .020),
                                (f"{coolant_manifold}.mount.rl", rear_left, .020)):
        source = (computer if ".hcu.mount" in ident else
                  booster if ".pressure_booster." in ident else
                  hand_pump if ".emergency_hand_pump." in ident else manifold)
        if ".backup_battery." in ident:
            source = battery
        elif ".backup_electric_pump." in ident:
            source = electric_pump
        elif ".backup_reservoir." in ident:
            source = reservoir
        elif ".accumulator." in ident:
            source = accumulator
        elif ".solenoid_switchboard." in ident:
            source = switchboard
        elif ".coolant_manifold." in ident:
            source = coolant_manifold
        g.edge(ident, source, node, "rigid-distance", radius=radius,
               alloy="4130n", palette="chassis-grey", beam_solvable=True,
               part_role="equipment-mount",
               load_path="control-hardware-into-rear-work-area-beam")
    g.edge("stand.hcu.command_bus", computer, switchboard,
           "insulated-copper-wire", radius=.006, palette="service-line",
           beam_solvable=False, routed=True, circuit_identity="hcu-command")
    g.edge("stand.hcu.coolant_command_bus", computer, coolant_manifold,
           "insulated-copper-wire", radius=.004, palette="service-line",
           beam_solvable=False, routed=True,
           circuit_identity="hcu-coolant-command")
    g.edge("stand.hcu.switchboard_to_manifold", switchboard, manifold,
           "pressure-rated-hydraulic-line", radius=.032,
           palette="service-line", beam_solvable=False, routed=True,
           nominal_id_m=.050, rated_flow_l_min=240.0,
           circuit_identity="hcu-selected-pressure")
    for source, tag in ((booster, "booster"), (hand_pump, "hand")):
        g.edge(f"stand.hcu.{tag}_pressure", source, switchboard,
               "pressure-rated-hydraulic-line", radius=.022,
               palette="service-line", beam_solvable=False, routed=True,
               nominal_id_m=.032, circuit_identity="hcu-pressure")
        g.edge(f"stand.hcu.{tag}_return", manifold, source,
               "flexible-hydraulic-hose", radius=.026,
               palette="service-line", beam_solvable=False, routed=True,
               nominal_id_m=.040, circuit_identity="hcu-return")
    g.edge("stand.hcu.backup_battery_feed", battery, electric_pump,
           "insulated-copper-wire", radius=.014, palette="service-line",
           beam_solvable=False, routed=True, circuit_identity="hcu-48v-backup",
           maximum_current_a=160.0)
    g.edge("stand.hcu.backup_pump_suction", reservoir, electric_pump,
           "flexible-hydraulic-hose", radius=.026, palette="service-line",
           beam_solvable=False, routed=True, nominal_id_m=.040,
           circuit_identity="hcu-suction")
    g.edge("stand.hcu.backup_pump_pressure", electric_pump, switchboard,
           "pressure-rated-hydraulic-line", radius=.022,
           palette="service-line", beam_solvable=False, routed=True,
           nominal_id_m=.032, circuit_identity="hcu-pressure")
    g.edge("stand.hcu.manifold_return_to_reservoir", manifold, reservoir,
           "flexible-hydraulic-hose", radius=.030, palette="service-line",
           beam_solvable=False, routed=True, nominal_id_m=.050,
           circuit_identity="hcu-return")
    g.edge("stand.hcu.accumulator_pressure", accumulator, switchboard,
           "pressure-rated-hydraulic-line", radius=.022,
           palette="service-line", beam_solvable=False, routed=True,
           nominal_id_m=.032, circuit_identity="hcu-pressure")

    # Three separate nitrogen services: support hydraulics, turret motion,
    # and the recoil recuperator.  Only gas-side charging ports are joined;
    # oil never crosses between these circuits.
    recoil_port = "stand.hcu.recoil_nitrogen_charge_port"
    authored_nodes = {n["identity"] for n in g.nodes}
    recoil_carrier = next((candidate for candidate in (
        "turret.absorber.piston.spring",
        "mount.absorber.reaction.pneumatic")
        if candidate in authored_nodes), None)
    if recoil_carrier is None:
        raise ValueError("production gun has no authored recuperator gas-side body")
    carrier_pos = next(n["reference_position"] for n in g.nodes
                       if n["identity"] == recoil_carrier)
    g.node(recoil_port, (carrier_pos[0] - .10, carrier_pos[1], carrier_pos[2]),
           "nitrogen-charge-port", material="brass", mass_in_total=True,
           mass_kg=1.2, in_view=True, half_extent_m=(.035, .035, .055),
           structural_participation=False,
           solver_condensed_into=recoil_carrier, solver_condensed_mass=True,
           part_role="recoil-recuperator-gas-side-service-port",
           oil_circuit_isolated=True)
    for suffix in ("mount", "mount.secondary"):
        g.edge(f"{recoil_port}.{suffix}", recoil_port, recoil_carrier,
               "rigid-distance", radius=.008, alloy="4130n",
               palette="chassis-grey", beam_solvable=False,
               part_role="service-port-mount")
    motion_port = "stand.hcu.turret_motion_nitrogen_charge_port"
    motion_carrier = "turret.elevation_anchor"
    motion_pos = next(n["reference_position"] for n in g.nodes
                      if n["identity"] == motion_carrier)
    g.node(motion_port, (motion_pos[0] + .10, motion_pos[1], motion_pos[2]),
           "nitrogen-charge-port", material="brass", mass_in_total=True,
           mass_kg=1.2, in_view=True, half_extent_m=(.035, .035, .055),
           structural_participation=False,
           solver_condensed_into=motion_carrier, solver_condensed_mass=True,
           part_role="turret-motion-accumulator-gas-side-service-port",
           oil_circuit_isolated=True)
    for suffix in ("mount", "mount.secondary"):
        g.edge(f"{motion_port}.{suffix}", motion_port, motion_carrier,
               "rigid-distance", radius=.008, alloy="4130n",
               palette="chassis-grey", beam_solvable=False,
               part_role="service-port-mount")
    nitrogen_targets = {
        "support": (accumulator, 12_000_000.0),
        "turret_motion": (motion_port, 12_000_000.0),
        "recoil": (recoil_port, 5_000_000.0),
    }
    floor_surface_y = float(site.get(
        "floor_surface_y_m", stand.deck_y - stand.lower_room_clear_height_m))
    floor_carriers = tuple(site.get("floor_plate", {}).get("nodes", {}).values())
    floor_position = {
        node["identity"]: np.asarray(node["reference_position"], float)
        for node in g.nodes if node["identity"] in floor_carriers}
    bottle_nodes = []
    for circuit_index, (circuit, (target_node, target_pa)) in enumerate(
            nitrogen_targets.items()):
        regulator = f"stand.hcu.nitrogen.{circuit}.regulator"
        z = -1.00 + circuit_index * 1.00
        g.node(regulator, (0.0, stand.deck_y - 1.18, z),
               "nitrogen-pressure-regulator", material="brass",
               mass_in_total=True, mass_kg=4.5, in_view=True,
               half_extent_m=(.09, .08, .08),
               structural_participation=False,
               solver_condensed_into=site["corners"][
                   "rr" if circuit_index != 2 else "fr"],
               solver_condensed_mass=True,
               part_role=f"{circuit}-nitrogen-service-regulator",
               target_pressure_pa=target_pa, independent_check_valve=True,
               oil_circuit_isolated=True)
        g.edge(f"{regulator}.mount", regulator,
               site["corners"]["rr" if circuit_index != 2 else "fr"],
               "rigid-distance", radius=.010, alloy="4130n",
               palette="chassis-grey", beam_solvable=False,
               part_role="nitrogen-regulator-panel-mount")
        for side, x in (("left", -1.48), ("right", 1.48)):
            bottle = f"stand.hcu.nitrogen.{circuit}.{side}"
            here = np.asarray((x, floor_surface_y, z), float)
            carrier = min(floor_carriers,
                          key=lambda name: float(np.linalg.norm(
                              floor_position[name][[0, 2]] - here[[0, 2]])))
            g.node(bottle, (x, floor_surface_y + .75, z),
                   "high-pressure-nitrogen-bottle", material="4340qt",
                   mass_in_total=True, mass_kg=82.0, in_view=True,
                   structural_participation=False,
                   solver_condensed_into=carrier, solver_condensed_mass=True,
                   shape="drum", drum_axis=(0.0, 1.0, 0.0),
                   drum_radius_m=.12, drum_length_m=1.50,
                   half_extent_m=(.12, .75, .12),
                   part_role=f"{circuit}-nitrogen-torpedo",
                   water_volume_l=50.0, working_pressure_pa=30_000_000.0,
                   gas="nitrogen", oil_circuit_isolated=True,
                   installed_on="tank-service-floor-plate",
                   clamp_type="two-saddle-through-bolted")
            for suffix in ("upper", "lower"):
                g.edge(f"{bottle}.rack.{suffix}", bottle, carrier,
                       "rigid-distance", radius=.014, alloy="4130n",
                       palette="chassis-grey", beam_solvable=False,
                       part_role="secured-nitrogen-bottle-floor-clamp",
                       structural_participation=False,
                       clamp_position=suffix)
            g.edge(f"{bottle}.to_regulator", bottle, regulator,
                   "pressure-rated-air-line", radius=.006,
                   palette="service-line", beam_solvable=False, routed=True,
                   circuit_identity=f"nitrogen-{circuit}",
                   shutoff_valve=True, check_valve=True)
            bottle_nodes.append(bottle)
        g.edge(f"{regulator}.to_charge_port", regulator, target_node,
               "pressure-rated-air-line", radius=.006,
               palette="service-line", beam_solvable=False, routed=True,
               circuit_identity=f"nitrogen-{circuit}",
               check_valve=True, pressure_transducer=True,
               gas_side_only=True, oil_circuit_isolated=True)

    # Both powerplants feed the same HCU coordination manifold.  These are
    # deliberately large headers; small hoses here would erase the rapid
    # turbine schedule in pressure loss before it reached a valve.
    for side in ("port", "starboard"):
        pump = f"powerplant.{side}.hydraulic"
        g.edge(f"stand.hcu.header.pressure.{side}", pump, switchboard,
               "pressure-rated-hydraulic-line", radius=.032,
               palette="service-line", beam_solvable=False, routed=True,
               nominal_id_m=.050, rated_flow_l_min=240.0,
               circuit_identity="hcu-pressure")
        g.edge(f"stand.hcu.header.return.{side}", manifold, pump,
               "flexible-hydraulic-hose", radius=.040,
               palette="service-line", beam_solvable=False, routed=True,
               nominal_id_m=.065, rated_flow_l_min=240.0,
               circuit_identity="hcu-return")

    # LOW and HIGH name the two machine platforms, not redundant signal
    # channels.  Each platform has one start-authority rail and one separate
    # consensus-summed kill rail.  The current installation maps low to the
    # night/main multifuel plant and high to the heavy-combat turbine plant.
    for platform, side in (("low", "port"), ("high", "starboard")):
        engine = f"engine.{side}"
        g.edge(f"stand.hcu.ignition_{platform}", computer, engine,
               "insulated-copper-wire", radius=.003,
               palette="service-line", beam_solvable=False, routed=True,
               circuit_identity=f"hcu-ignition-{platform}",
               signal_semantics="live-rail-authorises-start-and-retry",
               machine_platform=platform)
        g.edge(f"stand.hcu.kill_{platform}", computer, engine,
               "insulated-copper-wire", radius=.003,
               palette="service-line", beam_solvable=False, routed=True,
               circuit_identity=f"hcu-kill-{platform}",
               signal_semantics="summed-system-approval-must-reach-consensus",
               machine_platform=platform)

    sensors = {}
    for name in INNER:
        tag = name.split(".")[-1]
        carrier = site["corners"][tag]
        p = next(n["reference_position"] for n in g.nodes
                 if n["identity"] == carrier)
        ident = f"stand.hcu.structure_sensor.{tag}"
        g.node(ident, (p[0], p[1] - .045, p[2]), "position-sensor",
               material="aluminium-cast", mass_in_total=True, mass_kg=.35,
               in_view=True, half_extent_m=(.035, .018, .035),
               structural_participation=False,
               solver_condensed_into=carrier, solver_condensed_mass=True,
               part_role="structure-position-and-inclination-sensor",
               resolution_m=0.0001, hcu_channel=name)
        g.edge(f"{ident}.mount", ident, carrier, "rigid-distance",
               radius=.008, alloy="4130n", palette="chassis-grey",
               beam_solvable=False, part_role="sensor-mount")
        g.edge(f"{ident}.mount.secondary", ident, carrier, "rigid-distance",
               radius=.006, alloy="4130n", palette="chassis-grey",
               beam_solvable=False, part_role="sensor-second-fastener")
        g.edge(f"{ident}.signal", ident, computer,
               "insulated-copper-wire", radius=.003, palette="service-line",
               beam_solvable=False, routed=True, circuit_identity="hcu-sensors")
        sensors[name] = ident

    # The rod transducer, pressure transducer and load pin are integral to
    # each jack; declaring them on that exact graph edge prevents a second
    # fake actuator or load path from appearing in the structure.
    edge_by_id = {e["identity"]: e for e in g.edges}
    for name in ALL_SUPPORTS:
        actuator = edge_by_id[f"stand.leg.{name}"]
        actuator.update(position_sensor="absolute-magnetostrictive",
                        pressure_transducer="both-cylinder-chambers",
                        load_sensor="top-pin-strain-gauge",
                        hcu_channel=name)
    for name in OUTER:
        edge_by_id[f"stand.deploy_ram.{name}"].update(
            position_sensor="absolute-magnetostrictive",
            hcu_channel=f"{name}.swing")
    return {"computer": computer, "manifold": manifold,
            "solenoid_switchboard": switchboard,
            "coolant_manifold": coolant_manifold,
            "pressure_booster": booster, "emergency_hand_pump": hand_pump,
            "backup_battery": battery, "backup_electric_pump": electric_pump,
            "backup_reservoir": reservoir,
            "accumulator": accumulator,
            "nitrogen_bottles": tuple(bottle_nodes),
            "nitrogen_circuits": tuple(nitrogen_targets),
            "structure_sensors": sensors}
