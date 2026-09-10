"""The ignition driver as its own real box -- the rev limiter lives here,
not in the ECU and not inline in EngineCycleSim.

Real split, and why it's a different unit from ecu.EngineControlUnit:
the idle governor is a slow closed-loop program on airflow/fuel (an ECU
job). A rev limiter is a per-spark decision made at crank-event rate
by whatever fires the coils -- historically a standalone CD ignition
box with an rpm module (the drag/monster builds in this catalogue run
exactly that: magneto or CD box with its own rev control, no engine
computer at all), and on a modern engine still the ignition-driver
stage the ECU commands, gated by crank/cam position. So it takes the
same real identity the production vehicle graph already carries for
this box -- "electrical.ignition_driver", fed by the ECU over the real
"crank-cam-timed-ignition" wire (turing's abstract_ui_vehicles.py) --
and owns only what that box really owns: which limiter stages are
latched right now and the resulting per-spark cut probability. The
spark-event loop in engine_cycle_sim.py asks it once per substep.
"""
from __future__ import annotations

from engines import Engine

NOMINAL_COIL_SUPPLY_V = 12.6   # the voltage the coil's dwell/energy is rated at


class IgnitionDriver:
    """One ignition box: latched rev-limiter stage state + per-spark
    cut severity. reset(stage_count) on engine start."""

    def __init__(self) -> None:
        self._stage_active: list[bool] = []

    def reset(self, stage_count: int) -> None:
        self._stage_active = [False] * stage_count

    @property
    def any_stage_active(self) -> bool:
        return any(self._stage_active)

    @staticmethod
    def spark_energy_frac(engine: Engine, bus_voltage_v: float) -> float:
        """Spark energy relative to nominal, from what actually feeds the
        coil. An inductive coil stores E = 1/2 L I^2 with I = V/R at the
        end of dwell, so energy goes with the SQUARE of bus voltage --
        a sagging battery hurts spark far more than the old linear
        (V-9)/(12.6-9) stand-in claimed, and a 14.2 V charging bus is
        simply saturated (clipped at nominal, no bonus). A magneto makes
        its own energy from crank motion and doesn't see the bus at all;
        compression ignition has no spark."""
        dispatch = engine.ignition_dispatch
        if dispatch == "mechanical-magneto" or dispatch == "compression-injection":
            return 1.0
        # the coil is rated at this engine's own system voltage (a 24 V
        # truck coil at 24 V, not a 12 V coil at 24 V)
        nominal_v = NOMINAL_COIL_SUPPLY_V * engine.electrical_system_voltage_v / 12.0
        return max(0.0, min(1.0, (bus_voltage_v / nominal_v) ** 2))

    def cut_severity(self, engine: Engine, live_rpm: float, enabled: bool) -> float:
        """Probability that the next spark is cut, in [0, 1]. Staged
        limiters latch each stage with real hysteresis (engage_frac /
        disengage_frac of redline) and report the harshest latched
        stage; a soft-taper limiter (a real production soft cut:
        individual pulses skipped with a probability that ramps
        smoothly with rpm) feeds the same cut path with a continuous
        severity. `enabled` False is a real, explicit disable -- the cut
        logic never runs, so you can watch what the limiter exists to
        prevent (valve float, knock, whatever an unchecked over-rev
        does) -- not a fake very-high-redline trick."""
        if not enabled:
            return 0.0
        lim = engine.rev_limiter
        severity = 0.0
        for i, stage in enumerate(lim.stages):
            if live_rpm >= engine.redline_rpm * stage.engage_frac:
                self._stage_active[i] = True
            elif live_rpm < engine.redline_rpm * stage.disengage_frac:
                self._stage_active[i] = False
            if self._stage_active[i]:
                severity = max(severity, stage.cut_severity)
        if lim.soft_taper:
            frac_over = ((live_rpm / engine.redline_rpm - lim.soft_taper_start_frac)
                         / max(1e-6, lim.soft_taper_band_frac))
            severity = max(0.0, min(1.0, frac_over)) * lim.soft_taper_max_cut
        return severity
