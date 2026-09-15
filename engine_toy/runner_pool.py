"""RUNNERS: who can help, what cone they can cover, and whether it is worth it.

A LOCAL THREAD AND A REMOTE PEER ARE THE SAME THING. Both are a runner
with a latency and a throughput; only the numbers differ. So the whole
mechanism can be built and tested locally against a thread pool today,
and multiplayer is the same code with a bigger ping. That is why this
module has no networking in it and does not need any to be true.

WHY A CONE, AND WHY ONLY REMOTELY

Locally there is nothing to fan out over: the owner already knows its own
throttle this frame, so a speculative fan would compute branches it
throws away. Remotely the input is NOT known -- by the time a helper's
answer arrives, the owner has moved, and the helper had to guess what it
did. So the cone is exactly the input uncertainty over the latency
window, and its width is set by how fast an input can physically move:

    reach   = input_rate_per_s * round_trip_s
    branches = ceil(2 * reach / resolution) + 1

A pedal cannot teleport, which is what makes the cone finite. Slower
inputs, or lower ping, mean a narrower cone for the same coverage.

WHAT DELIVERED SPEED BUYS

The useful measure is not "can this runner simulate fast" but "can it
deliver a cone that still covers where the owner will be, before the
owner gets there itself". Three numbers decide it, and all three are
already measured elsewhere in this system rather than declared here:

  round_trip_s        the runner's ping, measured
  substeps_per_s      its throughput, measured
  the zone's own K    from `negotiate_substeps`: ceil(window / dt_limit)

THE DILATION DIVIDEND. A dilated zone is cheaper to help, and not
hand-wavingly so: its window is shorter, so it needs FEWER substeps to
stay inside the same stability limit. Cost per frame falls with tau
directly. The zone that had to be slowed because nobody could afford it
is, for that exact reason, the cheapest one to take off someone's hands.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class InputAxis:
    """One control the owner may move while a prediction is in flight."""

    name: str
    #: how fast it can physically change -- a pedal, not a teleport
    rate_per_s: float
    #: how finely two branches must differ before they are worth separating
    resolution: float

    def reach(self, round_trip_s: float) -> float:
        return abs(self.rate_per_s) * max(round_trip_s, 0.0)

    def branches(self, round_trip_s: float) -> int:
        if self.resolution <= 0.0:
            return 1
        return int(math.ceil(2.0 * self.reach(round_trip_s) / self.resolution)) + 1


@dataclass(frozen=True)
class RunnerProfile:
    """A helper. A thread, a server, or somebody's spare machine."""

    name: str
    round_trip_s: float
    substeps_per_s: float
    #: per-branch dispatch+join cost, measured. Never zero: even a local
    #: thread costs ~90 us to hand work to and collect it from.
    dispatch_s: float = 0.0
    #: which compiled assemblies it actually has -- the negotiator's
    #: topology signatures. Without the assembly it cannot help at all,
    #: however fast it is.
    topologies: frozenset[str] = frozenset()
    #: How much of this runner's nominal throughput is actually realised
    #: once it contends for whatever it shares with the owner. 1.0 is
    #: genuinely separate hardware. A local CPython thread pool measures
    #: around 0.09 on CPU-bound work -- the GIL -- which is the honest
    #: reason a local thread is not "free capacity", stated as a measured
    #: number rather than a flag. It rises toward 1.0 once the work is in
    #: compiled code that releases the GIL, which is precisely what the
    #: native lane buys.
    contention_factor: float = 1.0

    @property
    def effective_substeps_per_s(self) -> float:
        return self.substeps_per_s * max(self.contention_factor, 0.0)

    @property
    def is_local(self) -> bool:
        return self.round_trip_s <= 1e-4


@dataclass(frozen=True)
class ConePlan:
    """What a runner would actually undertake."""

    runner: str
    round_trip_s: float
    branches: int
    frames: int
    substeps_per_frame: int
    total_substeps: int
    compute_s: float
    deadline_s: float
    reason: str

    @property
    def feasible(self) -> bool:
        return self.compute_s <= self.deadline_s and self.branches > 0

    @property
    def headroom(self) -> float:
        """How much of the deadline is left over. Below 1.0 is a miss."""
        if self.compute_s <= 0.0:
            return float("inf")
        return self.deadline_s / self.compute_s


def plan_cone(runner: RunnerProfile, *, topology: str, time_velocity: float,
              reference_dt_s: float, dt_limit_s: float,
              axes: Sequence[InputAxis], horizon_frames: int) -> ConePlan:
    """Work out whether this runner can help, and how.

    The deadline is the round trip: an answer that arrives after the
    owner needed it is not an answer. The work is the cone width times
    the frames to cover times each frame's own substep count -- and that
    substep count is where a dilated zone gets cheap.
    """
    if topology not in runner.topologies:
        return ConePlan(runner.name, runner.round_trip_s, 0, 0, 0, 0, 0.0,
                        runner.round_trip_s, f"does not have the {topology} assembly")

    branches = 1
    for axis in axes:
        branches *= max(1, axis.branches(runner.round_trip_s))

    # frames the prediction must cover: enough to still be ahead when it lands
    frames = max(1, int(math.ceil(runner.round_trip_s / reference_dt_s)),
                 int(horizon_frames))

    # THE DILATION DIVIDEND: window = reference * tau, and K = window/dt_limit
    window_s = reference_dt_s * float(time_velocity)
    substeps = max(1, int(math.ceil(window_s / dt_limit_s - 1e-12))) if dt_limit_s > 0 else 1

    total = branches * frames * substeps
    # dispatch is not free and it grows with the cone: a measured
    # per-branch cost on top of the fixed round trip. A local thread pays
    # this too -- it is a real runner and it takes the same test.
    overhead_s = runner.round_trip_s + branches * runner.dispatch_s
    compute_s = total / max(runner.effective_substeps_per_s, 1e-9) + overhead_s
    # The deadline is HOW FAR AHEAD we are predicting, not the ping. The
    # ping is spent inside that horizon, not against it -- charging the
    # round trip against a deadline that WAS the round trip made every
    # remote runner fail by construction, which is wrong: a distant
    # runner predicting nine frames out has 150 ms of horizon to deliver
    # into, and its ping is part of what it spends.
    deadline = max(frames * reference_dt_s, reference_dt_s)
    reason = ("fits" if compute_s <= deadline
              else f"needs {compute_s*1000:.1f} ms of compute for a "
                   f"{deadline*1000:.1f} ms deadline")
    return ConePlan(runner.name, runner.round_trip_s, branches, frames, substeps,
                    total, compute_s, deadline, reason)


def best_runner(runners: Iterable[RunnerProfile], **kw) -> tuple[ConePlan | None, list[ConePlan]]:
    """Every runner's honest assessment, and the best feasible one.

    Preference is for the plan with the most headroom, not the lowest
    ping: a fast machine on a bad link can still beat a slow one nearby,
    and the only way to know is to cost the cone.
    """
    plans = [plan_cone(r, **kw) for r in runners]
    feasible = [p for p in plans if p.feasible]
    feasible.sort(key=lambda p: -p.headroom)
    return (feasible[0] if feasible else None), plans


# ---------------------------------------------------------------------
# the pool: the same contract, served by threads
# ---------------------------------------------------------------------

@dataclass
class RunnerPool:
    """Our own commitment to do work for people, served locally.

    Identical interface to a remote peer, which is the point: build it
    against threads, ship it against the network, change nothing but the
    latency. It multithreads and it multiplayers for the same reason.
    """

    workers: int = 4
    _pool: ThreadPoolExecutor | None = field(default=None, repr=False)

    def __enter__(self) -> "RunnerPool":
        self._pool = ThreadPoolExecutor(max_workers=self.workers)
        return self

    def __exit__(self, *exc) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def run_cone(self, plan: ConePlan, branch_fn: Callable[[int], object],
                 *, simulated_latency_s: float = 0.0) -> list[object]:
        """Run every branch of a cone concurrently and return the fan.

        `branch_fn(i)` computes branch i. The owner selects from the fan
        once its real input is known -- which is the whole trick: the
        latency is spent covering possibilities rather than waiting.
        """
        if self._pool is None:
            raise RuntimeError("use RunnerPool as a context manager")
        if simulated_latency_s > 0.0:
            time.sleep(simulated_latency_s)
        return list(self._pool.map(branch_fn, range(plan.branches)))


#: Measured on this machine, four threads, identical serial cost (~27 ms):
#:   pure Python, GIL held      0.97x speedup -> contention factor 0.24
#:   BLAS matmul, GIL released  2.31x speedup -> contention factor 0.58
#: So compiling the loop down to native, where it releases the GIL, is
#: worth about 2.4x of LOCAL capacity on its own -- before counting any
#: of the single-thread speedup the compile also buys. It is what turns
#: the local pool from a runner that cannot help into one that can.
GIL_HELD_CONTENTION = 0.24
GIL_RELEASED_CONTENTION = 0.58


def measure_local_runner(name: str = "local-thread", *, workers: int = 4,
                         substeps_per_s: float = 900_000.0,
                         topologies: frozenset[str] = frozenset(),
                         samples: int = 40) -> RunnerProfile:
    """Time a real thread pool rather than assuming anything about it.

    Measures the dispatch+join cost of a round trip through the pool, and
    the speedup actually obtained on CPU-bound work. On CPython the
    second number lands near 1.1 for four threads, so the contention
    factor comes out around 0.09: a local pool is a real runner that
    mostly cannot help, and it says so in measured numbers.
    """
    import statistics

    def _spin(n: int) -> float:
        total = 0.0
        for i in range(n):
            total += i * 0.5
        return total

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_spin, [64] * workers))          # warm
        trips = []
        for _ in range(samples):
            t0 = time.perf_counter()
            list(pool.map(_spin, [1]))
            trips.append(time.perf_counter() - t0)
        dispatch = statistics.median(trips)
        work = 300_000
        t0 = time.perf_counter(); _spin(work); serial = time.perf_counter() - t0
        t0 = time.perf_counter(); list(pool.map(_spin, [work // workers] * workers))
        parallel = time.perf_counter() - t0
    speedup = serial / max(parallel, 1e-9)
    return RunnerProfile(
        name=name, round_trip_s=dispatch, substeps_per_s=substeps_per_s,
        dispatch_s=dispatch, topologies=topologies,
        contention_factor=max(0.0, speedup / max(workers, 1)))


def select_branch(axes: Sequence[InputAxis], round_trip_s: float,
                  centre: Sequence[float], actual: Sequence[float]) -> int:
    """Which branch the owner's ACTUAL input turned out to be.

    Returns -1 if the input escaped the cone -- which is a real outcome,
    not an error: the owner moved faster than the cone was built for and
    the prediction is simply discarded.
    """
    index = 0
    stride = 1
    for axis, c, a in zip(axes, centre, actual):
        n = max(1, axis.branches(round_trip_s))
        reach = axis.reach(round_trip_s)
        if reach <= 0.0:
            slot = 0
        else:
            frac = (float(a) - float(c) + reach) / (2.0 * reach)
            if frac < -1e-9 or frac > 1.0 + 1e-9:
                return -1
            slot = min(n - 1, max(0, int(round(frac * (n - 1)))))
        index += slot * stride
        stride *= n
    return index


if __name__ == "__main__":
    import dataclasses

    TOPOLOGY = "cyl6.banks2.4stroke.recip.cam2.circ6.node325"
    axes = [InputAxis("throttle", rate_per_s=4.0, resolution=0.25)]   # 0..1 pedal
    runners = [
        # the owner's own pool, MEASURED -- it takes the same test as
        # everyone else and is not assumed to be free
        measure_local_runner(topologies=frozenset({TOPOLOGY})),
        RunnerProfile("lan-peer",   0.0040, 600_000, 0.0004, frozenset({TOPOLOGY})),
        RunnerProfile("broadband",  0.0350, 900_000, 0.0006, frozenset({TOPOLOGY})),
        RunnerProfile("distant",    0.1400, 900_000, 0.0010, frozenset({TOPOLOGY})),
        RunnerProfile("fast-no-asm", 0.0040, 5_000_000, 0.0004, frozenset()),
    ]
    local = runners[0]
    print(f"measured local pool: dispatch {local.dispatch_s*1e6:.0f} us/branch, "
          f"contention factor {local.contention_factor:.3f} "
          f"(effective {local.effective_substeps_per_s:,.0f} substeps/s)" + chr(10))

    print("Cone a runner must cover, and whether its delivered speed gets there.")
    print("Zone at tau=1.00 (not dilated), dt_limit 1/2000 s, reference 1/60 s\n")
    hdr = f"{'runner':<14}{'ping':>8}{'branch':>8}{'frames':>8}{'K':>5}{'work':>9}{'compute':>10}{'deadline':>10}{'head':>7}  verdict"
    print(hdr); print("-" * len(hdr))
    for r in runners:
        p = plan_cone(r, topology=TOPOLOGY, time_velocity=1.0,
                      reference_dt_s=1/60, dt_limit_s=1/2000,
                      axes=axes, horizon_frames=2)
        head = "-" if not p.feasible else f"{p.headroom:.2f}x"
        print(f"{r.name:<14}{r.round_trip_s*1000:>7.1f}m{p.branches:>8}{p.frames:>8}{p.substeps_per_frame:>5}"
              f"{p.total_substeps:>9}{p.compute_s*1000:>9.2f}m{p.deadline_s*1000:>9.1f}m{head:>7}  {p.reason}")

    print("\nSame runners, but the zone is DILATED to tau=0.25 -- shorter window,")
    print("so fewer substeps to stay inside the same stability limit:\n")
    print(hdr); print("-" * len(hdr))
    for r in runners:
        p = plan_cone(r, topology=TOPOLOGY, time_velocity=0.25,
                      reference_dt_s=1/60, dt_limit_s=1/2000,
                      axes=axes, horizon_frames=2)
        head = "-" if not p.feasible else f"{p.headroom:.2f}x"
        print(f"{r.name:<14}{r.round_trip_s*1000:>7.1f}m{p.branches:>8}{p.frames:>8}{p.substeps_per_frame:>5}"
              f"{p.total_substeps:>9}{p.compute_s*1000:>9.2f}m{p.deadline_s*1000:>9.1f}m{head:>7}  {p.reason}")

    native = dataclasses.replace(
        runners[0], name="local-NATIVE",
        contention_factor=GIL_RELEASED_CONTENTION)
    p_now = plan_cone(runners[0], topology=TOPOLOGY, time_velocity=0.25,
                      reference_dt_s=1/60, dt_limit_s=1/2000,
                      axes=axes, horizon_frames=2)
    p_native = plan_cone(native, topology=TOPOLOGY, time_velocity=0.25,
                         reference_dt_s=1/60, dt_limit_s=1/2000,
                         axes=axes, horizon_frames=2)
    print("")
    print(f"local pool today  : {p_now.compute_s*1000:6.2f} ms  headroom {p_now.headroom:6.2f}x")
    print(f"local pool NATIVE : {p_native.compute_s*1000:6.2f} ms  headroom {p_native.headroom:6.2f}x"
          f"   ({p_now.compute_s/max(p_native.compute_s,1e-12):.2f}x better)")

    best, _ = best_runner(runners, topology=TOPOLOGY, time_velocity=0.25,
                          reference_dt_s=1/60, dt_limit_s=1/2000,
                          axes=axes, horizon_frames=2)
    print(f"\nbest feasible: {best.runner} ({best.headroom:.2f}x headroom, "
          f"{best.branches} branches)")

    # and run it -- threads now, network later, same contract
    with RunnerPool(workers=4) as pool:
        fan = pool.run_cone(best, lambda i: ("branch", i, 0.5 + (i - best.branches // 2) * 0.25))
    picked = select_branch(axes, best.round_trip_s, centre=[0.5], actual=[0.62])
    escaped = select_branch(axes, best.round_trip_s, centre=[0.5], actual=[0.99])
    print(f"ran {len(fan)} branches on the pool ({best.runner}, {best.round_trip_s*1000:.0f} ms);"
          f" owner's real throttle 0.62 -> branch {picked}")
    print(f"a throttle of 0.99 would have escaped the cone -> {escaped} "
          f"(discard, not an error)")
    print("")
    print("The local pool is not excluded -- it takes the same test and its own")
    print("measured numbers decide it. Under CPython the GIL is what rules it out,")
    print("and that stops being true once the work is compiled and releases it.")
