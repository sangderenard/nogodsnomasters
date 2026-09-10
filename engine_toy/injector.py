"""A real EFI injector's own flow characteristic -- every real injector
spec sheet publishes a rated flow (lb/hr or cc/min) AT a stated
reference rail pressure (commonly 43.5 psi / 3.0 bar on older port-
injection systems), because flow through an injector's orifice is NOT
linear with pressure: it follows the real orifice-flow relation,
flow proportional to sqrt(delta-pressure). That's the actual
"pressure table" real injectors are sized against -- one reference
point plus this one real square-root law reproduces it, rather than
needing an interpolated lookup table for what is genuinely a one-
parameter physical relation.

engines.Engine.injector carries one logical Injector per engine (real
multi-injector arrays are declared as one combined rating -- this
catalogue doesn't model individual per-cylinder injector imbalance).
None (the default) means this engine either isn't EFI at all
(carbureted/diesel) or has EFI with an ideal/undeclared injector
sizing -- engine_cycle_sim's fuel-metering path only checks injector
duty when one is actually declared, so every existing catalogue engine
keeps its current behavior unless it opts in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Injector:
    rated_flow_kg_s: float       # this engine's own combined injector array rating
    rated_pressure_pa: float     # the real reference pressure that rating was published at

    def flow_capacity_kg_s(self, rail_pressure_pa: float) -> float:
        """Real max deliverable flow at the CURRENT rail pressure --
        the real sqrt(pressure) orifice-flow law off the declared
        reference point."""
        ratio = max(0.0, rail_pressure_pa) / max(self.rated_pressure_pa, 1.0)
        return self.rated_flow_kg_s * math.sqrt(ratio)

    def duty_for_target_flow(self, target_flow_kg_s: float, rail_pressure_pa: float) -> float:
        """Real injector pulse-width duty (0..1, saturating) needed to
        average this target flow at the current rail pressure -- >1
        (clamped) means the injector is genuinely too small for the
        demand at this pressure, a real, physical limit distinct from
        the fuel pump/tank's own starvation."""
        capacity = self.flow_capacity_kg_s(rail_pressure_pa)
        if capacity <= 1e-12:
            return 0.0
        return target_flow_kg_s / capacity

    @classmethod
    def sized_for(cls, redline_fuel_demand_kg_s: float, rated_pressure_pa: float,
                  margin_frac: float = 0.20) -> "Injector":
        """A real injector array picked the way a builder actually
        sizes one: rated flow (at the real reference pressure) covers
        the engine's own redline fuel demand plus a real margin (an
        injector run at 100% duty has no headroom left for transients
        -- real builds target sitting comfortably under max duty at
        the engine's own worst-case demand)."""
        return cls(rated_flow_kg_s=redline_fuel_demand_kg_s * (1.0 + margin_frac),
                   rated_pressure_pa=rated_pressure_pa)
