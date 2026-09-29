"""Time the Woodshop Newton entry with the complete hot loop in C.

Python prepares one real Woodshop material-buffer table and builds the
compiler's standalone host.  The executable then reloads that state, performs
warmup calls, and times repeated calls to the linked Newton entry without
Python or ctypes anywhere inside the measured interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
TURING_ROOT = ENGINE_ROOT.parent / "turing"
for import_root in (ENGINE_ROOT, TURING_ROOT, TURING_ROOT / "examples"):
    sys.path.insert(0, str(import_root))

from profile_woodshop_native import (
    _load_cached_native_system,
    _native_source_fingerprint,
)
from woodshop import WoodshopSimulation


def _read_outputs(path: Path, artifact, feeds) -> dict[int, np.ndarray]:
    payload = path.read_bytes()
    offset = 0
    outputs: dict[int, np.ndarray] = {}
    for value_id, dtype in zip(artifact.buffer_order, artifact.buffer_dtypes):
        initial = np.atleast_1d(np.asarray(feeds[int(value_id)]))
        wanted = np.dtype({
            "float64": np.float64,
            "int64": np.int64,
            "int32": np.int32,
            "bool": np.bool_,
        }[str(dtype)])
        byte_count = int(initial.size) * wanted.itemsize
        outputs[int(value_id)] = np.frombuffer(
            payload[offset:offset + byte_count], dtype=wanted,
        ).copy().reshape(initial.shape)
        offset += byte_count
    if offset != len(payload):
        raise RuntimeError(
            f"standalone output has {len(payload) - offset} trailing bytes"
        )
    return outputs


def _fields(stdout: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in stdout.split() if "=" in token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--parity-frames", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rebuild-native", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.parity_frames < 1:
        parser.error("frames/parity-frames must be positive and warmup non-negative")

    args.build.mkdir(parents=True, exist_ok=True)
    cache = args.build / "native-system.pkl"
    fingerprint = _native_source_fingerprint()
    system = None if args.rebuild_native else _load_cached_native_system(
        cache, fingerprint,
    )
    cache_reused = system is not None

    simulation = WoodshopSimulation()
    if system is None:
        simulation.enable_native_newton(args.build / "module", optimization="O2")
        system = simulation.world_rules._native_newton_system
        cache.write_bytes(pickle.dumps({
            "fingerprint": fingerprint,
            "system": system,
        }))
    else:
        simulation.world_rules._ensure_newton_dt_system()
        simulation.world_rules._attach_native_newton_system(system)

    # One ordinary call establishes the exact public shapes and values at the
    # same boundary Woodshop uses.  It is setup, not part of either timing.
    simulation.world_rules._advance_newton_dt_system(args.dt)
    prepared = simulation.world_rules.newton_native_execution
    feeds = {
        int(value_id): value.copy()
        for value_id, value in prepared.buffers.items()
    }
    standalone = system.artifact.compile_standalone(
        args.build / "standalone", feeds, optimization="O2", link="static",
    )

    # Compare a few evolving calls against the same compiled entry reached
    # through ctypes.  This is outside the benchmark and requires exact bytes.
    reference = system.artifact.prepare_execution({
        value_id: value.copy() for value_id, value in feeds.items()
    })
    for _ in range(args.parity_frames):
        reference.run()
    parity_process = standalone.run(frames=args.parity_frames)
    native_outputs = _read_outputs(
        standalone.final_outputs_path, system.artifact, feeds,
    )
    differing = []
    for value_id in system.artifact.buffer_order:
        left = reference.buffers[int(value_id)]
        right = native_outputs[int(value_id)]
        if not np.array_equal(left, right, equal_nan=True):
            differing.append(int(value_id))
    if differing:
        raise RuntimeError(
            f"standalone and ctypes buffers differ after parity run: {differing}"
        )

    measured = standalone.run(frames=args.frames, warmup=args.warmup)
    values = _fields(measured.stdout)
    result = {
        "scope": "linked Woodshop Newton dt-system entry; native C loop only",
        "configuration": {
            "frames": args.frames,
            "warmup": args.warmup,
            "parity_frames": args.parity_frames,
            "dt": args.dt,
            "cache_reused": cache_reused,
            "executable": str(standalone.executable_path),
        },
        "timing": {
            "elapsed_s": float(values["elapsed_s"]),
            "mean_ms": float(values["ns_per_frame"]) / 1.0e6,
            "fps": float(values["fps"]),
        },
        "parity": {
            "exact": True,
            "buffers": len(system.artifact.buffer_order),
            "ctypes_frames": args.parity_frames,
            "standalone_stdout": parity_process.stdout.strip(),
        },
        "measured_stdout": measured.stdout.strip(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps(result), flush=True)
    print(f"OUTPUT {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
