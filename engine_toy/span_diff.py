"""Where two runs of an engine part company, named by span and by frame.

    python span_diff.py

WHAT THIS IS FOR. A plugin author changes something -- a manifold, a
spring rate, a new accessory on the crank -- and the engine behaves
differently. "Differently" is not actionable. This says WHICH declared
state first diverged, on WHICH frame, and by how much, out of 3410 slots
across 46 spans.

It works because `engine_state.py` made the whole live sim a flat span
with an exact restore. Two runs are then two traces of the same layout,
and a diff over them is exact rather than statistical: it names
`edge_relative_angle_rad` or `cyl_knock_accum`, not "the drivetrain".

THE FIRST FRAME THAT DIVERGES IS THE ONE THAT MATTERS. Everything after
it is consequence. Chaos guarantees that a run which parts company at
frame 12 will look wholly different by frame 400, and reporting the
largest difference at the end tells you only that the engine is
sensitive. So the report is ordered by WHEN, not by HOW MUCH.

IT DIFFERS FROM THE COST SIDE, WHICH NOW EXISTS. This names WHAT
changed; `time_trials/profile_frame.py` names what it COST and what it
did to the world's step floor. They are deliberately separate readings:
this one is exact and about state, that one is statistical and about the
host. The note here used to say cost attribution "needs the laws
compiled" -- they are compiled now, and the reading it was waiting for
is the one in that file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

import engines
from engine_abi import engine_graph_abi
from engine_cycle_sim import EngineCycleSim
from engine_state import pack


@dataclass
class Trace:
    """One run, recorded as a span per frame."""

    label: str
    stride: int
    spans: tuple
    frames: list = field(default_factory=list)

    def record(self, sim, abi) -> None:
        self.frames.append(pack(sim, abi).copy())


def run(label: str, identity: str, frames: int = 120, dt: float = 1.0 / 240.0,
        *, throttle: float = 0.6,
        prepare: Callable[[EngineCycleSim], None] | None = None) -> Trace:
    """Run one engine and trace its declared state every frame.

    `prepare` is where a variant goes: anything that changes the machine
    before it runs, which is what a plugin is.
    """
    sim = EngineCycleSim(engine=engines.get(identity))
    sim.start()
    sim.throttle = throttle
    if prepare is not None:
        prepare(sim)
    abi = engine_graph_abi(sim.engine)
    trace = Trace(label=label, stride=abi.state_stride, spans=abi.spans)
    for _ in range(frames):
        sim.step(dt)
        trace.record(sim, abi)
    return trace


@dataclass(frozen=True)
class Divergence:
    frame: int
    span: str
    slot: int
    unit: str
    left: float
    right: float

    @property
    def absolute(self) -> float:
        return abs(self.left - self.right)

    @property
    def relative(self) -> float:
        scale = max(abs(self.left), abs(self.right))
        return self.absolute / scale if scale > 0 else float("inf")


def compare(left: Trace, right: Trace, *, tolerance: float = 0.0
            ) -> tuple[list[Divergence], int | None]:
    """Every span that ever differs, ordered by the frame it first did.

    Returns the list and the first diverging frame, or None if the two
    runs are identical for as long as both were traced.
    """
    if left.stride != right.stride:
        raise ValueError(
            f"different strides: {left.stride} vs {right.stride}. These are "
            f"not the same topology, so there is no slot-by-slot comparison "
            f"to make -- a change that moves the stride is a change of "
            f"machine, and `abi_negotiator` will already refuse to batch "
            f"them together."
        )
    seen: dict[str, Divergence] = {}
    first_frame: int | None = None
    for frame, (a, b) in enumerate(zip(left.frames, right.frames)):
        both_nan = np.isnan(a) & np.isnan(b)
        delta = np.abs(np.where(both_nan, 0.0, a - b))
        delta = np.nan_to_num(delta, nan=np.inf)
        hits = np.nonzero(delta > tolerance)[0]
        if hits.size == 0:
            continue
        if first_frame is None:
            first_frame = frame
        for span in left.spans:
            inside = hits[(hits >= span.slot)
                          & (hits < span.slot + span.count)]
            if inside.size == 0 or span.name in seen:
                continue
            slot = int(inside[0])
            seen[span.name] = Divergence(
                frame=frame, span=span.name, slot=slot, unit=span.unit,
                left=float(a[slot]), right=float(b[slot]))
    ordered = sorted(seen.values(), key=lambda d: (d.frame, d.span))
    return ordered, first_frame


def report(left: Trace, right: Trace, *, tolerance: float = 0.0,
           limit: int = 14) -> str:
    divergences, first = compare(left, right, tolerance=tolerance)
    lines = [f"{left.label} vs {right.label}: {left.stride} slots, "
             f"{len(left.frames)} frames"]
    if not divergences:
        lines.append("   identical in every declared slot, every frame")
        return "\n".join(lines)
    lines.append(f"   first parted company on frame {first} of "
                 f"{len(left.frames)}")
    lines.append(f"   {len(divergences)} of {len(left.spans)} spans differ, "
                 f"earliest first")
    lines.append(f"   {'frame':>6}  {'span':<28}{'slot':>6}  "
                 f"{'left':>14}{'right':>14}  {'rel':>9}")
    for divergence in divergences[:limit]:
        lines.append(
            f"   {divergence.frame:>6}  {divergence.span:<28}"
            f"{divergence.slot:>6}  {divergence.left:>14.6g}"
            f"{divergence.right:>14.6g}  {divergence.relative:>9.2e}")
    if len(divergences) > limit:
        lines.append(f"   ... and {len(divergences) - limit} more")
    return "\n".join(lines)


def main() -> None:
    engine = "mazda-b6ze-miata-1990"

    print("=" * 74)
    print("A run against itself -- the floor. Anything here is noise in the")
    print("method, not a finding about a change.")
    print("=" * 74)
    print(report(run("run A", engine), run("run B", engine)))

    print()
    print("=" * 74)
    print("The same engine carrying one extra accessory: what a plugin is.")
    print("=" * 74)

    def loaded(sim):
        sim.electrical_load_frac = 0.55

    print(report(run("stock", engine),
                 run("+electrical load", engine, prepare=loaded)))

    print()
    print("=" * 74)
    print("A change to the driver rather than to the machine.")
    print("=" * 74)
    print(report(run("throttle 0.60", engine, throttle=0.60),
                 run("throttle 0.62", engine, throttle=0.62)))


if __name__ == "__main__":
    main()
