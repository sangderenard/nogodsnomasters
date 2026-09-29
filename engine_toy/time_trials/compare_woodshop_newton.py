"""Compare eager and native Woodshop Newton boundaries value by value.

Unlike the whole-world profiler, this rig stops before floor and pair contact
resolution.  It compares every lane of every Newton column, the adaptive-time
state, and all telemetry, reporting both absolute and IEEE-754 ULP distance.
The native build enables the C emitter's post-instruction full-value trace.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle
import struct

import numpy as np

from woodshop import WoodshopSimulation


def _ordered_bits(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return (~bits & 0xFFFF_FFFF_FFFF_FFFF) if bits >> 63 else (
        bits | 0x8000_0000_0000_0000
    )


def _ulp_distance(left: float, right: float) -> int | None:
    if math.isnan(left) or math.isnan(right):
        return None
    if left == right:
        return 0
    return abs(_ordered_bits(left) - _ordered_bits(right))


def _comparison(step: int, field: str, lane: int | None,
                eager: float, native: float) -> dict:
    absolute = abs(native - eager) if all(map(math.isfinite, (eager, native))) else (
        0.0 if eager == native else math.inf
    )
    return {
        "step": step,
        "field": field,
        "lane": lane,
        "eager": eager,
        "native": native,
        "absolute": absolute,
        "scaled": absolute / max(1.0, abs(eager), abs(native)),
        "ulp": _ulp_distance(eager, native),
    }


def _initial_state(sim: WoodshopSimulation) -> dict[str, list[float]]:
    return {
        identity: [
            *(float(value) for value in sim.items[identity].center_xyz()),
            *(float(value) for value in
              sim.items[identity].linear_momentum_kg_m_s),
        ]
        for identity in sorted(sim.items)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module-snapshot", type=Path)
    args = parser.parse_args()

    eager = WoodshopSimulation()
    native = WoodshopSimulation()
    initial_eager = _initial_state(eager)
    initial_native = _initial_state(native)
    if initial_eager != initial_native:
        raise RuntimeError("fresh eager/native Woodshop states are not identical")

    artifact = native.enable_native_newton(
        args.build, optimization="O2", trace=True, trace_full_values=True,
    )
    if args.module_snapshot is not None:
        args.module_snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.module_snapshot.write_bytes(pickle.dumps(
            native.world_rules._native_newton_system.module
        ))
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TURING_TRACE_FILE"] = str(args.trace.resolve())
    comparisons: list[dict] = []
    step_summaries = []
    for step in range(args.steps):
        eager_columns = eager.world_rules._advance_newton_dt_system(args.dt)
        native_columns = native.world_rules._advance_newton_dt_system(args.dt)
        if eager_columns.keys() != native_columns.keys():
            raise RuntimeError("eager/native Newton column sets differ")
        before = len(comparisons)
        for name in eager_columns:
            eager_values = np.asarray(eager_columns[name]).reshape(-1)
            native_values = np.asarray(native_columns[name]).reshape(-1)
            if eager_values.shape != native_values.shape:
                raise RuntimeError(f"column {name!r} shapes differ")
            comparisons.extend(
                _comparison(step, f"column.{name}", lane, float(left), float(right))
                for lane, (left, right) in enumerate(
                    zip(eager_values, native_values)
                )
            )
        comparisons.append(_comparison(
            step, "dt_next", None,
            float(eager.world_rules._newton_dt_next),
            float(native.world_rules._newton_dt_next),
        ))
        for lane, (left, right) in enumerate(zip(
            eager.world_rules.newton_dt_state.telemetry,
            native.world_rules.newton_dt_state.telemetry,
        )):
            comparisons.append(_comparison(
                step, "telemetry", lane, float(left), float(right),
            ))
        current = comparisons[before:]
        nonzero = [row for row in current if row["absolute"] != 0.0]
        step_summaries.append({
            "step": step,
            "comparisons": len(current),
            "nonzero": len(nonzero),
            "max_absolute": max(
                (row["absolute"] for row in current), default=0.0,
            ),
            "max_ulp": max(
                (row["ulp"] for row in current if row["ulp"] is not None),
                default=0,
            ),
        })

    differing = [row for row in comparisons if row["absolute"] != 0.0]
    result = {
        "artifact": str(artifact),
        "trace": str(args.trace.resolve()),
        "configuration": {"steps": args.steps, "dt": args.dt},
        "initial_state_exact": True,
        "step_summaries": step_summaries,
        "comparison_count": len(comparisons),
        "differing_count": len(differing),
        "first_difference": differing[0] if differing else None,
        "largest_absolute": max(
            differing, key=lambda row: row["absolute"], default=None,
        ),
        "largest_ulp": max(
            (row for row in differing if row["ulp"] is not None),
            key=lambda row: row["ulp"], default=None,
        ),
        "differences": differing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps({
        key: result[key] for key in (
            "step_summaries", "comparison_count", "differing_count",
            "first_difference", "largest_absolute", "largest_ulp",
        )
    }), flush=True)
    print(f"TRACE {args.trace.resolve()}", flush=True)
    print(f"OUTPUT {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
