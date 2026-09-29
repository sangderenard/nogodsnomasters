"""Long paired profile of eager and native Woodshop advances.

Compilation and warm-up are excluded from timing.  The measured loop records
ordinary wall time without profiler instrumentation and compares every
authoritative Newton-written object coordinate and momentum after each frame.
A subsequent, separately reported cProfile sample attributes Python cost
without contaminating the primary averages.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import pstats
import statistics
import time

import woodshop as woodshop_module
from woodshop import WoodshopSimulation
from llvm_dt_system import lowered_system


def _native_source_fingerprint() -> str:
    """Identify every local Python source that can affect native lowering."""
    digest = hashlib.sha256()
    turing_root = Path(lowered_system.__code__.co_filename).resolve().parents[1]
    sources = [
        Path(woodshop_module.__file__).resolve(),
        (turing_root / "examples" / "llvm_dt_system.py").resolve(),
        *sorted((turing_root / "src").rglob("*.py")),
    ]
    for source in sources:
        digest.update(str(source).encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    digest.update(b"optimization=O2;piece_mode=link")
    return digest.hexdigest()


def _load_cached_native_system(cache: Path, fingerprint: str):
    if not cache.is_file():
        return None
    try:
        payload = pickle.loads(cache.read_bytes())
        system = payload["system"]
        library = Path(system.artifact.library_path)
        if payload.get("fingerprint") != fingerprint or not library.is_file():
            return None
        return system
    except (KeyError, OSError, pickle.PickleError, AttributeError, EOFError):
        return None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(samples: list[float]) -> dict[str, float]:
    mean = statistics.fmean(samples)
    return {
        "frames": len(samples),
        "total_s": sum(samples),
        "mean_ms": mean * 1.0e3,
        "median_ms": statistics.median(samples) * 1.0e3,
        "p95_ms": _percentile(samples, 0.95) * 1.0e3,
        "stddev_ms": statistics.pstdev(samples) * 1.0e3,
        "min_ms": min(samples) * 1.0e3,
        "max_ms": max(samples) * 1.0e3,
        "fps_from_mean": 1.0 / mean,
    }


def _snapshot(sim: WoodshopSimulation) -> dict[str, float]:
    result: dict[str, float] = {}
    for identity in sorted(sim.items):
        item = sim.items[identity]
        for axis, value in zip("xyz", item.center_xyz()):
            result[f"item.{identity}.position_{axis}"] = float(value)
        for axis, value in zip("xyz", item.linear_momentum_kg_m_s):
            result[f"item.{identity}.momentum_{axis}"] = float(value)
    result["newton.dt_next"] = float(sim.world_rules._newton_dt_next)
    telemetry = sim.world_rules.newton_dt_state.telemetry
    for index, value in enumerate(telemetry):
        result[f"newton.telemetry_{index}"] = float(value)
    return result


class DeviationTracker:
    def __init__(self) -> None:
        self.comparisons = 0
        self.max_abs = 0.0
        self.max_scaled = 0.0
        self.max_abs_location: dict | None = None
        self.max_scaled_location: dict | None = None
        self.first_nonzero: dict | None = None
        self.nonfinite: list[dict] = []

    def compare(self, frame: int, eager: dict[str, float],
                native: dict[str, float]) -> None:
        if eager.keys() != native.keys():
            raise RuntimeError("eager/native snapshot keys differ")
        for key in eager:
            reference = eager[key]
            candidate = native[key]
            self.comparisons += 1
            if not (math.isfinite(reference) and math.isfinite(candidate)):
                if reference == candidate:
                    continue
                self.nonfinite.append({
                    "frame": frame,
                    "field": key,
                    "eager": reference,
                    "native": candidate,
                })
                continue
            absolute = abs(candidate - reference)
            scaled = absolute / max(1.0, abs(reference), abs(candidate))
            detail = {
                "frame": frame,
                "field": key,
                "eager": reference,
                "native": candidate,
                "absolute": absolute,
                "scaled": scaled,
            }
            if absolute != 0.0 and self.first_nonzero is None:
                self.first_nonzero = detail
            if absolute > self.max_abs:
                self.max_abs = absolute
                self.max_abs_location = detail
            if scaled > self.max_scaled:
                self.max_scaled = scaled
                self.max_scaled_location = detail

    def summary(self) -> dict:
        return {
            "scalar_comparisons": self.comparisons,
            "max_absolute": self.max_abs,
            "max_scaled": self.max_scaled,
            "max_absolute_location": self.max_abs_location,
            "max_scaled_location": self.max_scaled_location,
            "first_nonzero": self.first_nonzero,
            "nonfinite_mismatches": self.nonfinite,
        }


def _profile_text(profile: cProfile.Profile, rows: int = 30) -> str:
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
        "cumulative"
    ).print_stats(rows)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--profile-frames", type=int, default=30)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rebuild-native", action="store_true",
        help="ignore a compatible saved NativeSystem and compile again",
    )
    args = parser.parse_args()

    eager = WoodshopSimulation()
    native = WoodshopSimulation()
    cache = args.build / "native-system.pkl"
    fingerprint = _native_source_fingerprint()
    system = (
        None if args.rebuild_native
        else _load_cached_native_system(cache, fingerprint)
    )
    cache_reused = system is not None
    if system is not None:
        native.world_rules._ensure_newton_dt_system()
        native.world_rules._attach_native_newton_system(system)
        artifact = Path(system.artifact.library_path)
        compile_s = 0.0
    else:
        compile_started = time.perf_counter()
        artifact = native.enable_native_newton(args.build, optimization="O2")
        compile_s = time.perf_counter() - compile_started
        system = native.world_rules._native_newton_system
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps({
            "fingerprint": fingerprint,
            "system": system,
        }))
    print(f"ARTIFACT {artifact}", flush=True)
    print(f"COMPILE_S {compile_s:.6f}", flush=True)
    print(f"CACHE_REUSED {cache_reused}", flush=True)

    deviation = DeviationTracker()
    for frame in range(-args.warmup, 0):
        eager.advance(args.dt)
        native.advance(args.dt)
        deviation.compare(frame, _snapshot(eager), _snapshot(native))
    print(f"WARMUP_COMPLETE {args.warmup}", flush=True)

    eager_times: list[float] = []
    native_times: list[float] = []
    for frame in range(args.frames):
        order = ((eager, eager_times), (native, native_times))
        if frame % 2:
            order = tuple(reversed(order))
        for simulation, samples in order:
            started = time.perf_counter()
            simulation.advance(args.dt)
            samples.append(time.perf_counter() - started)
        deviation.compare(frame, _snapshot(eager), _snapshot(native))
        if (frame + 1) % 50 == 0:
            print(f"MEASURED {frame + 1}/{args.frames}", flush=True)

    eager_profile = cProfile.Profile()
    native_profile = cProfile.Profile()
    for offset in range(args.profile_frames):
        frame = args.frames + offset
        if offset % 2:
            native_profile.runcall(native.advance, args.dt)
            eager_profile.runcall(eager.advance, args.dt)
        else:
            eager_profile.runcall(eager.advance, args.dt)
            native_profile.runcall(native.advance, args.dt)
        deviation.compare(frame, _snapshot(eager), _snapshot(native))

    result = {
        "configuration": {
            "dt": args.dt,
            "warmup_frames": args.warmup,
            "measured_frames": args.frames,
            "profile_frames": args.profile_frames,
            "artifact": str(artifact),
            "compile_s": compile_s,
            "native_system_cache": str(cache.resolve()),
            "native_system_cache_reused": cache_reused,
        },
        "timing": {
            "eager": _timing_summary(eager_times),
            "native": _timing_summary(native_times),
        },
        "deviation": deviation.summary(),
        "final": {
            "eager": _snapshot(eager),
            "native": _snapshot(native),
        },
        "cprofile": {
            "eager": _profile_text(eager_profile),
            "native": _profile_text(native_profile),
        },
    }
    result["timing"]["native_speedup_percent"] = (
        1.0 - result["timing"]["native"]["mean_ms"]
        / result["timing"]["eager"]["mean_ms"]
    ) * 100.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps({
        "timing": result["timing"],
        "deviation": result["deviation"],
    }), flush=True)
    print(f"OUTPUT {args.output}", flush=True)


if __name__ == "__main__":
    main()
