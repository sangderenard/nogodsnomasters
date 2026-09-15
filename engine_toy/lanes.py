"""LANES: a nothing class that means "these are independent".

In Python it does the obvious thing -- hands the work to the repository's
own deployment pool and waits. Under the compiler it is not executed at
all: it is RECOGNISED, and replaced by control data
(`control_source.ParallelDeployment`), after which the existing pipeline
takes over.

WHY A MARKER RATHER THAN THREADS

Emitting threads literally would be the wrong answer, because the
compiler can already make a better decision than the author can. Once a
region reaches the SSA deployment layer it is:

  1. PROVEN independent -- `deployment_ssa_binding` checks that no value
     defined in one lane is consumed by a sibling, and a region that
     fails is reported and left unbound, never silently "fixed". The
     annotation is a claim; the compiler turns it into a theorem or
     refuses it.
  2. RE-PLANNED -- `deployment_classification` decides per region whether
     it is `graphics-output`, `shader-compute`, `thread-workers` or
     `host-linear`, from the region's real extent effect. Lanes an author
     wrote as threads may come out as a compute shader, as element
     splitting across a host pool, or as the recorded linear order.

So `Lanes` grants PERMISSION to run concurrently. It does not select
threads, and nothing downstream is obliged to use them. That is the same
contract `ControlDeploymentRegion` already states for itself: "deliberately
does not select GLSL, SIMD, threads, or any other backend".

The safety property falls out of (1): a `Lanes` block whose bodies are
NOT actually independent does not miscompile. The binding pass declines
to bind it, reports why, and the recorded serial order stands.

THE SHAPE, chosen to be trivially recognisable in an AST

    with Lanes() as lanes:
        lanes.lane(step_bay, bay_a, window)
        lanes.lane(step_bay, bay_b, window)

A `With` whose context expression is a `Lanes(...)` call, whose body is a
sequence of `<name>.lane(fn, *args)` calls. One node kind to match, one
call shape to collect, and every lane is already a plain call with its
own arguments -- which is exactly the operand list
`ParallelDeployment(lanes=...)` wants. Nothing has to be inferred from
dataflow to build the region; only to PROVE it.

Loop form, for when the lanes are an iterable rather than written out:

    for item in Lanes.over(items):
        ...

which the recogniser maps onto the loop-origin deployment path
`precompile_to_ssa` already mints regions from, rather than a second
mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence


#: Schedule preferences `control_source.ParallelDeployment` accepts. Carried
#: through verbatim so the annotation and the control record cannot disagree
#: about a vocabulary.
SCHEDULE_PREFERENCES = ("asap", "alap")


@dataclass
class _Lane:
    """One unit of work, kept as the call it was written as.

    Deliberately not a closure: `fn` and `args` stay separate so the
    recogniser sees an ordinary call with ordinary operands. A lambda
    would erase exactly the structure the lowerer needs.
    """

    fn: Callable[..., Any]
    args: tuple
    kwargs: dict

    def __call__(self) -> Any:
        return self.fn(*self.args, **self.kwargs)


class Lanes:
    """Independent work, however it ends up being run.

    Used as a context manager: lanes are declared inside the block and
    dispatched when it exits. The result of lane *i* is `results[i]`.
    """

    def __init__(self, *, schedule: str = "alap", workers: int | None = None):
        preference = str(schedule).lower()
        if preference not in SCHEDULE_PREFERENCES:
            raise ValueError(
                f"schedule must be one of {SCHEDULE_PREFERENCES}, not {schedule!r}")
        self.schedule = preference
        self._workers = workers
        self._lanes: list[_Lane] = []
        self.results: tuple = ()

    # -- authoring -----------------------------------------------------
    def lane(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Declare one independent lane. Nothing runs yet."""
        self._lanes.append(_Lane(fn, args, kwargs))

    def __len__(self) -> int:
        return len(self._lanes)

    # -- the Python disposition ----------------------------------------
    def __enter__(self) -> "Lanes":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None:
            return False
        self.run()
        return False

    def run(self) -> tuple:
        """Dispatch every lane and wait.

        Uses the repository's own pool, so the serial disposition
        (`workers=0`) is the identical claim loop rather than a second
        implementation, and a lane failure carries its lane index.
        """
        if not self._lanes:
            self.results = ()
            return self.results
        pool = _shared_pool(self._workers)
        self.results = tuple(pool.deploy(list(self._lanes)))
        return self.results

    # -- the loop form -------------------------------------------------
    @staticmethod
    def over(items: Iterable[Any], *, schedule: str = "alap") -> "LaneIterable":
        """Mark a loop's iterations as independent lanes.

        Transparent in Python -- iterating yields the same items in the
        same order, so the loop runs exactly as written. The marker is
        what the recogniser sees; it maps onto the loop-origin deployment
        path the lowerer already mints regions from.
        """
        return LaneIterable(tuple(items), schedule)


@dataclass(frozen=True)
class LaneIterable:
    """A pass-through iterable that says its iterations are independent."""

    items: tuple
    schedule: str = "alap"

    def __iter__(self) -> Iterator[Any]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


# ---------------------------------------------------------------------
# the pool, shared, because workers should start once
# ---------------------------------------------------------------------

_POOLS: dict[int, Any] = {}


def _shared_pool(workers: int | None):
    """One pool per worker count, kept alive.

    Workers that start once and park between frames are the whole point
    of the deployment pool; building a new one per block would put the
    startup cost back.
    """
    import graph_physics  # noqa: F401 -- puts turing on sys.path
    from src.compiler.deployment_host_pool import HostDeploymentPool

    key = -1 if workers is None else int(workers)
    pool = _POOLS.get(key)
    if pool is None:
        pool = HostDeploymentPool(workers=workers)
        _POOLS[key] = pool
    return pool


def close_pools() -> None:
    """Release every shared pool. For tests and shutdown."""
    while _POOLS:
        _, pool = _POOLS.popitem()
        try:
            pool.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# what the recogniser matches -- stated here so the front end and the
# authored form cannot drift apart
# ---------------------------------------------------------------------

#: The context-manager form: `with Lanes(...) as x:` containing
#: `x.lane(fn, *args)` calls. Maps to ParallelDeployment(lanes=...) with
#: one lane per call, in written order.
RECOGNISED_BLOCK = "With(context=Call(Lanes), body=[Call(<name>.lane, ...)])"

#: The loop form: `for target in Lanes.over(items):`. Maps onto the
#: loop-origin deployment region rather than a second mechanism.
RECOGNISED_LOOP = "For(iter=Call(Lanes.over, items))"


if __name__ == "__main__":
    import time

    def work(label: str, spin: int) -> str:
        total = 0.0
        for i in range(spin):
            total += i * 0.5
        return f"{label}:{total:.0f}"

    with Lanes() as lanes:
        lanes.lane(work, "a", 200_000)
        lanes.lane(work, "b", 200_000)
        lanes.lane(work, "c", 200_000)
    print(f"{len(lanes)} lanes -> {lanes.results}")

    # the loop form is a pass-through: the loop runs exactly as written
    seen = [item for item in Lanes.over((1, 2, 3))]
    print(f"loop form yields {seen} unchanged")

    print()
    print("In Python this dispatched on HostDeploymentPool.")
    print("Under the compiler it is not executed: it is recognised and")
    print("replaced by ParallelDeployment, then PROVEN independent and")
    print("RE-PLANNED by deployment_classification -- which may choose")
    print("shader-compute or host-linear instead of threads at all.")
    close_pools()
