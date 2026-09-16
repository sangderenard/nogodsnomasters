"""WHERE THE TIME GOES, AND WHAT IT DOES TO THE FLOOR.

    python time_trials/profile_frame.py
    python time_trials/profile_frame.py --frames 400 --window 1

THE QUESTION THIS ANSWERS. A person builds a car -- a manifold, a
gearbox, an accessory hung off the crank -- and their car is slow in a
way that has nothing to do with how fast it drives. Somewhere in the
machine they assembled is a subsystem that cannot advance a second of
world time in a second of wall time, and everyone sharing that world is
either waiting for them or going on without them. "The sim is slow" is
not actionable. THIS subsystem, THIS many milliseconds, and THIS is what
it does to the step everyone else has to take: that is actionable.

TAU IS MEASURED, NEVER SET. Every ledger here is world seconds advanced
over wall seconds spent. Nobody chooses it; it is what the machine did.
A subsystem at tau 0.3 spent three seconds of the host's time to move
the world one -- and a subsystem's tau is a fact about the HOST, while
the floor below is a fact about the WORLD.

THE TWO COLUMNS ARE DIFFERENT KINDS OF THING, and conflating them is the
mistake this report exists to prevent:

  COST is wall time. It says whether the host can keep up. Fix it with
  faster code, and nothing about the physics changes.

  FLOOR is world time. It is the largest step a subsystem's own
  mathematics can be advanced by without coming apart -- the contact
  law's radial mode, an engine's crank-angle grain. Fix it only by
  changing the physics, and no amount of compiling touches it.

A subsystem can be cheap and still set a punishing floor, and then
everyone substeps at its grain and pays for it in COST. That coupling --
one subsystem's mathematics setting everyone's bill -- is the thing a
plugin author cannot see without this.

MEASURED ON THE STOCK PAIR, and the coupling is not hypothetical: both
engines cost 3.98 ms a frame at tau 0.25 while both bodies run at tau
1.3-1.5, and the engines are ALSO the floor, at 0.0624 ms against a 1 ms
frame. Seventeen substeps of crank angle is why the engine costs what it
costs. The bodies, on the same host, keep ahead of the clock.

WHAT CHANGED TO MAKE THIS POSSIBLE. `span_diff.py` says of itself: "It
diffs state, not cost. Attributing wall time to a span ... needs the laws
compiled, since at present an eager step dominates anything a plugin
could do." Both laws are now native kernels, which took the body from
161 ms a call to 0.023 ms, and the eager step no longer drowns the
measurement. What is left is a real profile of a real machine.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE), str(_HERE.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from craft_graph import EngineBatch, build_graphs, FLOOR_S     # noqa: E402
from time_allocator import derive_dt_limit_s                   # noqa: E402


def profile(batch, fleet, *, frames: int, dt: float) -> dict:
    """Run the real loop and keep the ledgers the subsystems keep.

    The loop is the game's loop, unchanged. A profiler that runs a
    different loop from the one being profiled reports on a program
    nobody executes.
    """
    for engine in (r.engine for r in batch.registrations):
        engine.reset_ledger()
    fleet.reset_ledger()
    laws = {"vehicle body": fleet.vehicle, "wheel contact": fleet.contact}

    names = list(fleet.craft)
    throttle = {name: 0.9 for name in names}
    started = time.perf_counter()
    for _ in range(frames):
        batch.step(dt)
        fleet.step(dt, shaft_omega={n: batch.exterior(n)[0] for n in names},
                   throttle=throttle)
    wall = time.perf_counter() - started
    return {"wall_s": wall, "world_s": frames * dt, "frames": frames,
            "dt": dt, "laws": laws}


def report(batch, fleet, run: dict) -> str:
    wall, world = run["wall_s"], run["world_s"]
    frames, dt = run["frames"], run["dt"]
    lines = []
    lines.append("=" * 78)
    lines.append(f"{frames} frames of {dt * 1e3:.3f} ms  ->  "
                 f"{world:.3f} s of world in {wall:.3f} s of wall")
    lines.append(f"tau for the whole machine: {world / wall:.4f}   "
                 + ("ahead of the clock" if world >= wall
                    else f"DILATED -- {wall / world:.1f} s of host per "
                         f"second of world"))
    lines.append("=" * 78)

    lines.append("")
    lines.append("COST -- wall time, a fact about the host")
    lines.append(f"  {'subsystem':<26}{'ms/frame':>10}{'% frame':>9}"
                 f"{'calls':>8}{'tau':>9}")
    rows = []
    for registration in batch.registrations:
        engine = registration.engine
        rows.append((registration.name, engine.wall_s, engine.calls,
                     engine.tau))
    rows.append(("all bodies (one batch)", fleet.wall_s, fleet.calls, fleet.tau))
    accounted = sum(row[1] for row in rows)
    for name, seconds, calls, tau in sorted(rows, key=lambda r: -r[1]):
        lines.append(f"  {name:<26}{seconds / frames * 1e3:>10.4f}"
                     f"{seconds / wall * 100:>8.1f}%{calls:>8}{tau:>9.3f}")
    lines.append(f"  {'-- unaccounted (dt system,':<26}"
                 f"{(wall - accounted) / frames * 1e3:>10.4f}"
                 f"{(wall - accounted) / wall * 100:>8.1f}%")
    lines.append("     loop, bookkeeping) --")

    lines.append("")
    lines.append("     of which, the compiled laws (shared by every craft):")
    for label, law in run["laws"].items():
        lines.append(f"       {label:<22}{law.wall_s / frames * 1e3:>10.4f}"
                     f"{law.wall_s / wall * 100:>8.1f}%{law.calls:>8}"
                     f"     {law.per_call_ms * 1e3:.1f} us/call")

    lines.append("")
    lines.append("FLOOR -- world time, a fact about the world")
    lines.append("  the largest step each subsystem's own mathematics allows.")
    lines.append("  The smallest of these is the world's floor: everyone")
    lines.append("  substeps to it, and pays for it above.")
    lines.append(f"  {'subsystem':<26}{'floor ms':>10}{'substeps at dt':>16}")
    floors = []
    for name, sim in batch.sims.items():
        limit = derive_dt_limit_s(sim.engine)
        if limit:
            floors.append((f"{name}.engine", limit))
    for name, craft in fleet.craft.items():
        if craft.limit_s < float("inf"):
            floors.append((f"{name}.contact", craft.limit_s))
    for name, limit in sorted(floors, key=lambda f: f[1]):
        need = max(1, int(-(-dt // limit)))
        lines.append(f"  {name:<26}{limit * 1e3:>10.4f}{need:>16}")
    if floors:
        tightest = min(floors, key=lambda f: f[1])
        lines.append("")
        lines.append(f"  the world's floor is {tightest[0]} at "
                     f"{tightest[1] * 1e3:.4f} ms")

    lines.append("")
    lines.append("READ IT THIS WAY: the top of COST is what to make faster.")
    lines.append("The top of FLOOR is what to make gentler. A fix aimed at")
    lines.append("the wrong column does nothing -- compiling a subsystem")
    lines.append("never widens its floor, and gentling one never makes its")
    lines.append("code faster.")
    if floors and rows:
        tight = tightest[0].rsplit(".", 1)[0]
        dear = max(rows, key=lambda r: r[1])[0].rsplit(".", 1)[0]
        if tight == dear:
            need = max(1, int(-(-dt // tightest[1])))
            lines.append("")
            lines.append(f"  HERE THEY ARE THE SAME SUBSYSTEM. {tight} is both")
            lines.append("  the dearest and the tightest, and those are not two")
            lines.append(f"  findings: its floor of {tightest[1] * 1e3:.4f} ms "
                         f"forces {need} substeps into")
            lines.append(f"  every {dt * 1e3:.3f} ms frame, and its cost is that "
                         f"count times the")
            lines.append("  cost of one substep. Compiling it divides the second")
            lines.append("  factor. Only its physics touches the first.")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--window", type=int, default=1,
                        help="ticks of FIXED_PHYSICS_DT_S per frame")
    args = parser.parse_args(argv)

    craft = {"roadster": "mazda-b6ze-miata-1990",
             "coupe": "vw-vr6-2800-12v"}
    started = time.perf_counter()
    batch = EngineBatch.build(craft)
    fleet, n_in, n_contact, n_carry = build_graphs(list(craft))
    print(f"built in {time.perf_counter() - started:.1f}s  "
          f"({n_in} body inputs, {n_contact} contact inputs, "
          f"{n_carry} carried)")

    dt = FLOOR_S * args.window
    # A settling pass first: the first frames pay for page faults, lazy
    # imports and a cold branch predictor, and reporting those as the
    # cost of a subsystem is a lie about the steady state.
    profile(batch, fleet, frames=20, dt=dt)
    run = profile(batch, fleet, frames=args.frames, dt=dt)
    print(report(batch, fleet, run))


if __name__ == "__main__":
    main()
