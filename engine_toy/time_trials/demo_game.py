"""A demo game built on the engine, the vehicle graph and the tire.

    python time_trials/demo_game.py --frames 120

The game, and only the game: it owns the loop, the controls and the
reporting. What a craft IS lives in `craft_graph.py`, and what the game
LOOKS like lives in `watch.py`. They were one file until it was pointed
out that a car definition and a demo loop are not the same thing.

Compiling this file is the point of the split. Handed to
`lower_ast_source_to_ssa` under engine_toy's contract, everything here
lowers except two calls, and the compiler says which ones:

    CompilationSubdivisionRequired: a loop's body regions are scheduled
    but the loop itself could not compile ...
      blockers=('opaque-state-effect',)  batch.step(dt)
      blockers=('opaque-state-effect',)  graph.step(dt, shaft_omega=...)

Both are methods on objects whose state is not declared in
`program_abi.records`. For the engine that declaration is now writable --
`engine_state.py` carries the whole live sim as 3156 flat slots with an
exact restore -- and it is the work between that message and a compiled
game.
"""
from __future__ import annotations

import argparse
import math
import time

from craft_graph import (EngineBatch, build_graphs, FLOOR_S, WHEEL_NAMES)


def frame(batch, graph, dt, throttle, shaft_omega):
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
    graph.step(dt, shaft_omega=shaft_omega, throttle=throttle, steer=0.0)
    return graph.last


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
    graphs, n_in, n_contact, n_carry = build_graphs(list(craft))
    print(f"built in {time.perf_counter() - started:.1f}s")
    print(f"   engine batch : {len(batch.sims)} sims in one dt round")
    for name, limit in batch.dt_limits().items():
        print(f"      {name:<10} {batch.sims[name].engine.identity:<24}"
              f" dt_limit {limit * 1e6:8.1f} us")
    print(f"   vehicle graph: {n_in} inputs, {n_contact} contact inputs, "
          f"{n_carry} carried states, one per craft")
    print(f"   static tire deflection "
          f"{graphs[list(craft)[0]].static_deflection_m * 1000:.2f} mm\n")

    dt = FLOOR_S * args.window
    for frame in range(args.frames):
        batch.step(dt)
        for name, graph in graphs.items():
            omega, _torque = batch.exterior(name)
            graph.step(dt, shaft_omega=omega, throttle=args.throttle, steer=0.0)
        if frame % max(args.frames // 6, 1) == 0:
            for name, graph in graphs.items():
                row = graph.last
                print(f"   f{frame:4d} {name:<9} rpm "
                      f"{float(batch.sims[name].rpm or 0.0):7.1f}"
                      f"  clutch {row['clutch_torque_nm']:8.1f} Nm"
                      f"  wheel_w {row['wheel_omega']:8.2f}"
                      f"  Fy {row['contact_force_y']:9.1f} N"
                      f"  patch {row['patch_cm2']:6.1f} cm2"
                      f"  x {row['x']:8.3f} m")
    print(f"\nworld {args.frames * dt:.3f} s")
    for name, graph in graphs.items():
        row = graph.last
        print(f"   {name:<10} x {row['x']:9.3f} m  z {row['z']:8.3f} m"
              f"  speed {row['speed_x']:7.3f} m/s"
              f"  rpm {float(batch.sims[name].rpm or 0.0):7.1f}")


if __name__ == "__main__":
    main()
