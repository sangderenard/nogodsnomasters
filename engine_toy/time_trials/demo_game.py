"""A demo game built on the engine, the vehicle graph and the tire.

    python time_trials/demo_game.py --frames 120

The game, and only the game: it owns the loop, the controls and the
reporting. What a craft IS lives in `craft_graph.py`, and what the game
LOOKS like lives in `watch.py`. They were one file until it was pointed
out that a car definition and a demo loop are not the same thing.

Compiling this file is the point of the split, and what stops it is now
known by name rather than by category. Lowering `EngineCycleSim.step`
directly refuses like this:

    CompilationSubdivisionRequired: a loop's body regions are scheduled
    but the loop itself could not compile, which would otherwise silently
    run the body once with no iteration ...
      blockers=('opaque-state-effect',)
      self.hole_emitters.step(...)  self.bursts.step(...)
      self.ordnance.step(...)

`opaque-state-effect` is not a diagnosis, it is the DEFAULT: the
classifier in `topological_reducer.py` recognises a loop-body mutation
only when the state is a built-in container and the method is one of
seven. Every `.step()` ever written falls through it.

Counted across that whole lowering: 60 state effects, of which 34 are
`sequence_mutation`, 6 are `mapping_mutation`, and 20 are the catchall --
seven receivers, four operator names. The engine's own flat span
(`engine_state.py`, a stride per topology rather than one number) is what
a fix is expected to be built on, because a copy of declared state around
an unknown call gives the effect the output value the loop recurrence
needs.
"""
from __future__ import annotations

import argparse
import math
import time

from craft_graph import (EngineBatch, build_graphs, FLOOR_S, WHEEL_NAMES)


def frame(batch, fleet, dt, throttle, shaft_omega):
    """One tick of the game, with the objects arriving as PARAMETERS.

    A record declaration alone does not clear `opaque-state-effect`: a
    binding is `{function, parameter, record}`, so the compiler learns an
    object's layout through a parameter it was handed, not through a local
    a constructor returned. `main` builds `batch` and `graph` as locals,
    so there was nothing to bind them to and the refusal stood however
    many records were declared.

    This is the same work, shaped so it can be bound.
    """
    batch.step(dt)
    fleet.step(dt, shaft_omega=shaft_omega, throttle=throttle)
    return fleet.craft


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--throttle", type=float, default=0.9)
    parser.add_argument("--window", type=int, default=1,
                        help="ticks of FIXED_PHYSICS_DT_S per step. Nothing "
                             "substeps for you: at 16 the contact law is "
                             "past its own radial mode and returns no "
                             "ground at all.")
    args = parser.parse_args(argv)

    craft = {"roadster": "mazda-b6ze-miata-1990",
             "coupe": "vw-vr6-2800-12v"}

    started = time.perf_counter()
    batch = EngineBatch.build(craft)
    fleet, n_in, n_contact, n_carry = build_graphs(list(craft))
    print(f"built in {time.perf_counter() - started:.1f}s")
    print(f"   engine batch : {len(batch.sims)} sims in one dt round")
    for name, limit in batch.dt_limits().items():
        print(f"      {name:<10} {batch.sims[name].engine.identity:<24}"
              f" dt_limit {limit * 1e6:8.1f} us")
    print(f"   vehicle graph: {n_in} inputs, {n_contact} contact inputs, "
          f"{n_carry} carried states, one per craft")
    print(f"   static tire deflection "
          f"{fleet.craft[list(craft)[0]].static_deflection_m * 1000:.2f} mm\n")

    dt = FLOOR_S * args.window
    names = list(craft)
    throttle = {name: args.throttle for name in names}
    for frame in range(args.frames):
        batch.step(dt)
        fleet.step(dt, shaft_omega={n: batch.exterior(n)[0] for n in names},
                   throttle=throttle)
        if frame % max(args.frames // 6, 1) == 0:
            for name in names:
                row = fleet.last(name)
                print(f"   f{frame:4d} {name:<9} rpm "
                      f"{float(batch.sims[name].rpm or 0.0):7.1f}"
                      f"  clutch {row['clutch_torque_nm']:8.1f} Nm"
                      f"  wheel_w {row['wheel_omega']:8.2f}"
                      f"  Fy {row['contact_force_y']:9.1f} N"
                      f"  patch {row['patch_cm2']:6.1f} cm2"
                      f"  x {row['x']:8.3f} m")
    print(f"\nworld {args.frames * dt:.3f} s")
    for name in names:
        row = fleet.last(name)
        print(f"   {name:<10} x {row['x']:9.3f} m  z {row['z']:8.3f} m"
              f"  speed {row['speed_x']:7.3f} m/s"
              f"  rpm {float(batch.sims[name].rpm or 0.0):7.1f}")


if __name__ == "__main__":
    main()
