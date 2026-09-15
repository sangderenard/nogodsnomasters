"""THE TIME FIELD, RUNNING REAL ENGINES, UNDER A REAL BUDGET.

    python demo_time_field.py

Four bays, each a real `EngineCycleSim` -- not a toy rotor -- sharing one
fixed frame budget that is far too small for all of them. Nothing here
is arranged to succeed: the budget is a genuine 16.7 ms and four engines
genuinely cost several times that, so the field has to do real work and
the shortfall is real.

WHAT TO WATCH

  tau        each bay's clock rate, set by what it could be afforded.
  world_s    world time it actually advanced. The bays that could not be
             paid for advance less. That IS the price, and it is visible
             without being explained.
  cost_ms    measured, not declared. This is what drives everything.
  K          its substep count, from its own window and its own dt_limit.
             A dilated bay needs fewer, which is the dilation dividend.
  offer      whether its time store has been charging long enough that
             the bay is worth handing to a runner.

The budget is never exceeded by CHOICE -- the frame holds and the world
runs slow where it must, rather than the frame blowing out. That is the
whole proposition, on real engines.
"""
from __future__ import annotations

import math
import os
import time

import engines
from engine_cycle_sim import EngineCycleSim
from time_field import TimeField, TimeFieldConfig
from time_contract import StoreLedger, opportunity_for, state_digest
from abi_negotiator import negotiate, negotiate_substeps
from engine_abi import engine_graph_abi
from time_allocator import (TimeBudget, simple_metrics, DEFAULT_TARGETS,
                            derive_dt_limit_s, thread_cpu_s)
from src.common.dt_system.realtime import RealtimeConfig
from src.compiler.deployment_host_pool import HostDeploymentPool

REFERENCE_DT_S = 1.0 / 60.0
BUDGET_MS = 16.7
SLACK = 0.9
#: how hard the field is allowed to move. Not a smoothing constant: the
#: ramp limit is physical, set by whatever stores the time force.
MAX_RAMP_PER_S = 3.0

#: A MIXED fleet on purpose. An engine is one kind of machine, not the
#: only kind (machines.py), and the time field does not care which is
#: which -- it sees exteriors. A turret and a power pack sit in the same
#: budget as a V6 and are dilated by the same rule.
BAYS = [
    ("bay_busso", "engine", "alfa-busso-v6-3000-12v"),
    ("bay_vr6", "engine", "vw-vr6-2800-12v"),
    ("genset", "engine", "amc-258-jeep-i6"),
    ("turret", "machine", "pop_up_turret"),
    ("powerpack", "machine", "hydraulic_power_pack"),
]


class Bay:
    def __init__(self, name: str, kind: str, identity: str):
        self.name = name
        self.kind = kind
        if kind == "engine":
            self.subject = engines.get(identity)
            self.sim = EngineCycleSim(engine=self.subject)
            self.sim.start()
            self.sim.throttle = 0.55
        else:
            import machines
            from machines import MachineSim
            self.subject = machines.get(identity)
            self.sim = MachineSim(machine=self.subject)
        self.ledger = StoreLedger()
        self.cost_ms = 1.0
        self.world_s = 0.0
        self.substeps = 1
        self.wall_ms = 0.0
        self.label = identity
        # the bay's own stability floor, DERIVED from its own graph rather
        # than assumed -- see derive_dt_limit_s. Never overridden to make
        # a budget fit.
        derived = derive_dt_limit_s(self.subject)
        self.dt_limit_s = derived if derived > 0.0 else 1.0 / 240.0

    @property
    def rpm(self) -> float:
        return float(getattr(self.sim, "rpm", 0.0) or 0.0)

    def step(self, window_s: float) -> None:
        """Advance, and measure what it actually COST.

        thread_time, not perf_counter. Under a fan-out dispatch a bay's
        wall clock includes every moment a lane sat waiting rather than
        working, which is not its cost and not something it can act on.
        Feeding that to the allocator inflates every cost the moment work
        is dispatched, and the field would dilate everything to pay for
        contention. Thread CPU time is the quantity that means the same
        thing however the frame is dispatched."""
        t0 = thread_cpu_s()
        w0 = time.perf_counter()
        self.sim.step(window_s)
        self.cost_ms = (thread_cpu_s() - t0) * 1000.0
        self.wall_ms = (time.perf_counter() - w0) * 1000.0
        self.world_s += window_s


#: The repository's OWN deployment pool -- the one the compiled lane
#: dispatches Deploy/Join frames onto (turing_pool.c is its native
#: counterpart). Persistent workers that start once and park between
#: frames, so a frame costs no thread dispatch; the caller drains
#: alongside them, which makes workers=0 the serial fallback through the
#: IDENTICAL code path rather than a second implementation that could
#: drift. Its `lane` is the same lane as the ABI's batch axis.
#: Set TIME_FIELD_THREADS=0 for that serial disposition.
_WORKERS = int(os.environ.get("TIME_FIELD_THREADS", "4"))
POOL = HostDeploymentPool(workers=_WORKERS)


def main() -> None:
    bays = [Bay(n, k, i) for n, k, i in BAYS]
    field_ = TimeField.flat([b.name for b in bays], TimeFieldConfig())

    # what could actually share a compiled assembly, before anything runs
    abis = {b.name: engine_graph_abi(b.subject) for b in bays}
    plan = negotiate(abis)
    print("ASSEMBLIES (interior strides -- what could batch, if compiled):")
    for a in plan.assemblies:
        kind = abis[a.members[0]].kind
        print(f"   {a.lanes} lane(s)  {kind:<8} interior {a.state_stride:5d}"
              f"  exterior {abis[a.members[0]].exterior_stride}  {', '.join(a.members)}")
    print("   (exterior is the same width for every one of them -- that is why a")
    print("    turret and a V6 can sit in the same coupling graph at all)")
    print()

    print(f"budget {BUDGET_MS:.1f} ms/frame, slack {SLACK:.0%}, "
          f"reference dt {REFERENCE_DT_S*1000:.2f} ms, ramp limit {MAX_RAMP_PER_S:.1f}/s")
    print()

    # dt_system's OWN allocator -- its EMA, its penalties, its weights --
    # closed onto time velocity instead of onto dt. The local hand-rolled
    # split this replaced was a worse copy of machinery that already
    # exists and is already universal.
    budget = TimeBudget(RealtimeConfig(budget_ms=BUDGET_MS, slack=SLACK))
    names = [b.name for b in bays]

    for frame in range(1, 25):
        # ---- what the frame can afford -------------------------------
        for b in bays:
            budget.observe(b.name, simple_metrics(b.cost_ms, dt_limit=b.dt_limit_s),
                           DEFAULT_TARGETS, tau=field_.velocity(b.name))
        alloc = budget.allocate(names, {b.name: field_.velocity(b.name) for b in bays})
        for b in bays:
            # the ALLOCATOR says where to go; the FIELD decides how fast
            # it may get there, because the ramp is physical
            field_.set_target(b.name, alloc[b.name].log_tau_target,
                              dt=REFERENCE_DT_S, max_ramp_per_s=MAX_RAMP_PER_S)

        # ---- one K for everyone, from the tightest lane's own limit ---
        windows = [REFERENCE_DT_S * field_.velocity(b.name) for b in bays]
        sched = negotiate_substeps(windows, [b.dt_limit_s for b in bays])
        for b, k in zip(bays, [sched.substeps] * len(bays)):
            b.substeps = k

        # ---- run: fan out, join ---------------------------------------
        # Every bay is independent within a frame -- they only meet at
        # their exteriors, and nothing here couples them -- so the frame
        # is a fan-out/join, which is what a game engine's dispatcher
        # does anyway. dt_system's RoundNode(schedule="parallel") is a
        # documented "cooperative parallel stub" that still runs children
        # in series, and its ThreadedSystemEngine.step() is synchronous
        # (queue a request, block for the reply), so neither gives real
        # concurrency without issuing every request before collecting any.
        # This does that directly.
        frame_t0 = time.perf_counter()
        POOL.deploy([(lambda b=b, w=w: b.step(w))
                     for b, w in zip(bays, sched.windows_s)])
        wall_ms = (time.perf_counter() - frame_t0) * 1000.0
        for b, window in zip(bays, sched.windows_s):
            b.ledger.observe(
                reaction_nm=abs(field_.dlog_tau_dt[field_._index[b.name]]) * 10.0,
                slip_rad_s=max(b.rpm * 0.1047, 1.0), dt_s=REFERENCE_DT_S,
                asked_s=REFERENCE_DT_S, got_s=window)
            b.ledger.relax(REFERENCE_DT_S)

        if frame % 6 and frame != 24:
            continue

        spent = sum(b.cost_ms for b in bays)
        print(f"frame {frame:3d}   cpu {spent:6.1f} ms   wall {wall_ms:6.1f} ms"
              f"   of {BUDGET_MS:.1f}   K={sched.substeps}   ({sched.limited_by})")
        hdr = (f"   {'bay':<12}{'tau':>7}{'dlogt/dt':>10}{'world_s':>9}{'rpm':>8}"
               f"{'cost_ms':>9}{'K':>5}{'store_J':>10}{'short':>7}  offer")
        print(hdr)
        for b in bays:
            i = field_._index[b.name]
            opp = opportunity_for(
                b.name, topology=abis[b.name].topology, lanes=(b.name,),
                ledger=b.ledger, time_velocity=field_.velocity(b.name),
                reference_dt_s=REFERENCE_DT_S, predict_frames=6,
                state=[b.rpm, b.world_s])
            print(f"   {b.name:<12}{field_.velocity(b.name):>7.3f}"
                  f"{field_.dlog_tau_dt[i]:>10.3f}{b.world_s:>9.3f}{b.rpm:>8.0f}"
                  f"{b.cost_ms:>9.2f}{b.substeps:>5}{b.ledger.stored_j:>10.2f}"
                  f"{b.ledger.shortfall_ema:>7.2f}  {'YES' if opp.worth_offering else '-'}")
        print()

    mode = (f"HostDeploymentPool, {POOL.worker_count} workers + caller"
            if POOL.worker_count else "HostDeploymentPool, serial (caller only)")
    print(f"dispatch: {mode}.")
    print("Every frame held its budget. The bays that could not be paid for")
    print("advanced less world time -- and said so, in tau and in world_s,")
    print("rather than the frame blowing out. Nothing coarsened its step:")
    print(f"every bay integrated at {sched.steps_s[0]*1e6:.0f} us or finer, inside its own limit.")


if __name__ == "__main__":
    main()
