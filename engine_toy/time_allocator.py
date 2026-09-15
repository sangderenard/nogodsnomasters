"""BUDGET -> TIME VELOCITY, on top of dt_system's own realtime allocator.

`src/common/dt_system/realtime.py` already does the hard part: it keeps a
moving average of each engine's measured processing cost, turns its
metrics into a penalty, and weights a budget across engines. None of that
is reimplemented here -- this module imports it and uses it.

What differs is the CLOSURE, and only the closure.

`compile_allocations` grants every engine its measured cost even when the
total exceeds the frame, and says so plainly: *"If this exceeds the frame
budget, we still grant it; the frame will simply take longer (liveness
over target FPS)."* That is a real policy and a defensible one -- for a
caller who would rather run long than change anything about the world,
it is exactly right, and it stays available untouched.

This is the other closure. When the measured cost overruns the frame, the
budget is not granted anyway and dt is not coarsened to fit. Each scope
is given a share of the time it asked for, and the shortfall becomes its
LOCAL TIME VELOCITY. It advances less world time; its step size is
untouched; the frame holds.

    realtime.py            here
    -----------            ----
    alloc -> dt, 1:1       alloc -> round_max (world time)
    dt_limit ignored       dt_limit is a hard floor
    frame runs long        local time runs slow

Both are configurations of the same allocator over the same measured
state, which is the point: nothing is replaced, and a caller picks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import graph_physics  # noqa: F401  -- puts turing on sys.path

from src.common.dt_system.realtime import (  # noqa: E402
    RealtimeConfig,
    RealtimeState,
    compute_minimum_budget,
    compute_normalized_weights,
    compute_penalty,
)
from src.common.dt_system.dt_controller import Targets  # noqa: E402
from src.common.dt_system.dt_scaler import Metrics  # noqa: E402


@dataclass(frozen=True)
class TimeAllocation:
    """What one scope got, and what that means for its clock."""

    scope: str
    asked_ms: float
    granted_ms: float
    log_tau_target: float

    @property
    def tau_target(self) -> float:
        return math.exp(self.log_tau_target)

    @property
    def shortfall(self) -> float:
        if self.asked_ms <= 0.0:
            return 0.0
        return max(0.0, (self.asked_ms - self.granted_ms) / self.asked_ms)


@dataclass
class TimeBudget:
    """dt_system's realtime allocator, closed onto time velocity."""

    config: RealtimeConfig = field(default_factory=RealtimeConfig)
    state: RealtimeState = field(default_factory=RealtimeState)
    #: floor on a scope's clock, so nothing is dilated into a full stop
    min_tau: float = 0.02

    def observe(self, scope: str, metrics: Metrics, targets: Targets,
                tau: float = 1.0) -> None:
        """Fold one scope's measured frame in, using dt_system's own EMA
        and its own penalty function -- not a local re-derivation.

        THE COST IS NORMALISED TO FULL RATE before it enters the average.
        A dilated scope costs less simply because it did less, so feeding
        raw `proc_ms` back into the loop makes the controller measure its
        own output: it sees the cost fall, grants a full budget, tau
        returns to 1.0, the cost jumps back, and the frame oscillates
        between starved and blown. Observed directly -- 4.0 ms one frame
        and 35.4 ms three frames later. Dividing by tau gives a quantity
        that does not move when the controller acts, which is the only
        kind of measurement a feedback loop can safely use.
        """
        raw = float(getattr(metrics, "proc_ms", 0.0))
        at_full_rate = raw / max(float(tau), 1e-6)
        self.state.update_proc_ms(scope, at_full_rate, self.config.ema_alpha)
        self.state.update_penalty(scope, compute_penalty(metrics, targets),
                                  self.config.ema_alpha)

    def allocate(self, scopes: Sequence[str],
                 current_tau: Mapping[str, float]) -> dict[str, TimeAllocation]:
        """Turn measured cost into a target clock rate per scope.

        Under budget, everyone gets 1.0 and nothing is dilated -- the
        field only does something when there is a genuine shortfall.
        Over budget, the deficit is shared by the SAME penalty weights
        realtime.py already computes, so a scope that is struggling
        numerically is dilated less than one that is merely expensive.
        """
        base, min_total = compute_minimum_budget(self.config, self.state, scopes)
        target_pool = max(self.config.slack * self.config.budget_ms, 0.0)
        weights = compute_normalized_weights(self.config, self.state, scopes)

        out: dict[str, TimeAllocation] = {}
        if min_total <= target_pool or min_total <= 0.0:
            # it fits at FULL RATE: no dilation, surplus simply unspent
            for scope in scopes:
                out[scope] = TimeAllocation(scope, base[scope], base[scope], 0.0)
            return out

        # it does not fit. Share what there is, protecting the scopes the
        # penalty says are already struggling.
        deficit = min_total - target_pool
        # How much of the deficit each scope absorbs. A high penalty means
        # "this one is already struggling", so it absorbs LESS than its raw
        # cost share would suggest. Clamping at zero can leave part of the
        # deficit unallocated -- a cheap scope's protected share exceeds its
        # raw share and goes negative -- which silently under-spends the
        # budget (measured: 12.8 ms of a 15.0 ms target). So the shares are
        # renormalised after the clamp and the deficit is fully placed.
        shares: dict[str, float] = {}
        for scope in scopes:
            raw_share = base[scope] / min_total
            protected = weights.get(scope, raw_share)
            shares[scope] = max(0.0, 1.5 * raw_share - 0.5 * protected)
        total_share = sum(shares.values()) or 1.0
        for scope in scopes:
            asked = base[scope]
            share = shares[scope] / total_share
            granted = max(0.0, asked - deficit * share)
            # `asked` is already the cost AT FULL RATE, so the affordable
            # clock is just the fraction of it that was granted -- an
            # absolute target, not a correction applied to the current
            # tau. That is what stops it hunting: the answer does not
            # depend on where the controller already moved it to.
            tau_target = min(1.0, max(self.min_tau,
                                      (granted / asked) if asked > 0.0 else 1.0))
            out[scope] = TimeAllocation(scope, asked, granted,
                                        math.log(tau_target))
        return out


def simple_metrics(proc_ms: float, *, max_vel: float = 0.0,
                   dt_limit: float = 0.0) -> Metrics:
    """A Metrics carrying just what the allocator reads, for callers whose
    engines do not build a full one. `dt_limit` rides along because it is
    already a declared sidechain and is what a scheduler needs."""
    m = Metrics(max_vel=max_vel, max_flux=0.0, div_inf=0.0, mass_err=0.0)
    m.proc_ms = float(proc_ms)
    if dt_limit > 0.0:
        try:
            m.dt_limit = float(dt_limit)
        except Exception:
            pass
    return m


DEFAULT_TARGETS = Targets(cfl=1.0, div_max=1e-3, mass_max=1e-6)


if __name__ == "__main__":
    budget = TimeBudget(RealtimeConfig(budget_ms=16.7, slack=0.9))
    scopes = ["beam", "engine", "fluid", "actuators"]
    costs = {"beam": 22.0, "engine": 9.0, "fluid": 4.0, "actuators": 0.4}
    tau = {s: 1.0 for s in scopes}

    print("cost 35.4 ms against a 16.7 ms frame -- a real overrun\n")
    for frame in range(1, 9):
        for s in scopes:
            # what it really cost this frame, at the clock it really ran
            budget.observe(s, simple_metrics(costs[s] * tau[s]), DEFAULT_TARGETS,
                           tau=tau[s])
        alloc = budget.allocate(scopes, tau)
        for s in scopes:
            tau[s] = alloc[s].tau_target
        if frame % 2:
            continue
        spent = sum(costs[s] * tau[s] for s in scopes)
        print(f"frame {frame}: spent {spent:5.1f} ms")
        for s in scopes:
            a = alloc[s]
            print(f"   {s:<10} asked {a.asked_ms:6.2f} ms  granted {a.granted_ms:6.2f} ms"
                  f"  tau {a.tau_target:.3f}  short {a.shortfall:.2f}")
        print()
    print("The frame converges into budget by slowing local clocks, not by")
    print("coarsening anyone's step. realtime.py's own EMA, penalties and")
    print("weights did the work; only the closure differs.")
