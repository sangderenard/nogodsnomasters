"""A real, commercially-available class of hardware: a standalone idle-
air-control box -- MSD, FAST, Holley Sniper, Edelbrock Pro-Flo and
others all sell one -- bolted onto an engine that has no ECU idle
governor of its own (a carbureted engine, or a stand-alone EFI system
with no closed-loop idle feature built in) to give it real closed-loop
idle control it otherwise wouldn't have. `engines.Engine.
idle_control_device` picks which real device actually governs THIS
engine's idle -- "ecu" (the default: this catalogue's own
ecu.EngineControlUnit, gain-scheduled per engine via a real linearized-
plant pole placement, the way a real factory ECU calibration is tuned
specifically for its own engine) or "standalone-pi-controller" (this
module: a real generic aftermarket box, sold with one fixed, conservative
factory-default PI tune since it can't know the specific engine's own
plant the way a custom ECU calibration does -- a genuine real trade-off,
not a worse copy of the same thing).

Built directly from the same installable regulator (regulator.py) the
turbine's own fuel governor uses -- one real closed-loop PI primitive,
reused wherever a real device needs one, rather than a second bespoke
implementation.

Real, disclosed scope limit: these are AIR-side products for throttled
(gasoline/rotary) engines -- a diesel's idle is governed by its
injection pump rack, a mechanical fuel-metering problem no aftermarket
air-control box addresses. A diesel engine ignores idle_control_device
and always idles on its own governed injection pump.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engines import Engine
from regulator import ClosedLoopRegulator
from ecu import MAP_IDLE_FLOOR_FRAC, MAP_IDLE_CEILING_FRAC, idle_base_manifold_frac, blower_pressure_ratio

# A real, generic factory-default tune -- conservative (soft, slow)
# compared to a per-engine ECU calibration, exactly the real trade-off
# a one-size-fits-most aftermarket box makes: it has to stay stable
# across everything from a small four to a big V8 without ever being
# individually tuned, so it can't run as aggressive a gain as a real
# factory calibration keyed to one specific engine's own plant.
STANDALONE_KP_FRAC_PER_FRAC = 0.55
STANDALONE_KI_FRAC_PER_FRAC = 0.35
STANDALONE_INTEGRAL_CLAMP = 3.0


@dataclass
class StandaloneIdleController:
    """One real aftermarket box. Same real call shape as ecu.
    EngineControlUnit.idle_map_target so engine_cycle_sim can call
    whichever device this engine's build actually has without
    branching on the caller side."""
    regulator: ClosedLoopRegulator = field(init=False)

    def __post_init__(self) -> None:
        self.regulator = ClosedLoopRegulator(
            kp=STANDALONE_KP_FRAC_PER_FRAC, ki=STANDALONE_KI_FRAC_PER_FRAC,
            output_min=MAP_IDLE_FLOOR_FRAC, output_max=MAP_IDLE_CEILING_FRAC,
            integral_clamp=STANDALONE_INTEGRAL_CLAMP)

    def reset(self) -> None:
        self.regulator.reset()

    def idle_map_target(self, engine: Engine, rpm: float, dt: float) -> float:
        # both target and feedback in fractional-of-idle-rpm units, the
        # same convention the ECU's own gains are tuned in, so error =
        # target - feedback = (idle_rpm - rpm) / idle_rpm exactly
        manifold_cmd = self.regulator.regulate(
            dt, target=1.0, feedback=rpm / max(engine.idle_rpm, 1.0),
            base=idle_base_manifold_frac(engine))
        return manifold_cmd / blower_pressure_ratio(engine)
