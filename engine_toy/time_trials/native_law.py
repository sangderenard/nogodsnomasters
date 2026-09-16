"""A compiled law with its buffers exposed, so a call marshals nothing.

    law = NativeLaw.build(compile_wheel_contact_ssa(),
                          "abstract_ui_wheel_contact")
    law.write(vector_in_argument_order)
    law.run()
    law.read(into)

WHY BUFFERS AND NOT ARGUMENTS. Handing a law its inputs as freshly built
`AbstractTensor` columns rebuilds 40 or 341 objects per call and reads 7
or 146 back out. `prepare_artifact_execution` instead hands back the
kernel's OWN buffers: writing an input is writing the memory the kernel
will read, and reading an output is reading the memory it wrote.

They are separate 0-d arrays, not one arena -- checked, two input buffers
do not share memory and the array you feed is not the array you get back
-- so a call is a Python loop over the ports plus the kernel. That loop
is still the whole marshalling cost, and it is small:

    wheel contact, 40 in / 7 out    12.51 ms -> 0.0240 ms     521x

end to end, every write and read included, with all seven outputs
bit-identical to the eager lane (max abs diff 0.000e+00,
`chassis_force_y` 12225.706161 both ways).

WHY THE EMISSION IS CACHED HERE TOO. The repository's symbolic cache
works -- measured, a second run HITS on both laws. What it hands back is
a `SymbolicEquationCompilation`, and loading that is 19.6 s for the
contact law and 95.2 s for the body: 97 MB of cloudpickle, all of it
sympy-derived IR that is consumed once by `emit_ssa_function_to_llvm`
and then dead. That 115 s WAS the startup. Nothing was being recomputed,
which is why looking for a cache miss found nothing to fix.

So this caches one layer further down: the emitted artifact -- IR text
and a few integer tables, small -- plus the four name maps a call needs.
On a hit the compilation is never built and never loaded; startup is
reading a small pickle and one `zig cc`.

THE PORT ORDER IS THE CONTRACT. `argument_names` and `output_names` fix
it, and `parameter_names` / `output_ids` map each name to the value whose
buffer it is. Both were checked to be exactly 1:1 with the argument list
-- no argument without a buffer, no buffer without an argument -- so a
vector in argument order addresses the kernel completely.
"""
from __future__ import annotations

import inspect
import pickle
import warnings
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.compiler.ssa_llvm_backend import (
    emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)

BAKED = Path(__file__).resolve().parents[1] / "baked"
EMISSIONS = BAKED / "emissions"


def _emission_key(producer, entry: str, key_sources) -> str | None:
    """A name for this emission that costs nothing to compute.

    Deliberately the SAME two facts the repository's own symbolic cache
    keys on -- the digest of the producer's source files and the digest
    of the lowering pipeline -- so an edit that invalidates that cache
    invalidates this one too, and neither can serve a stale program the
    other has already given up on.

    `_producer_record` is documented as a "construction-free identity":
    it reads source files and nothing else, so asking whether the answer
    is on disk does not cost what producing the answer would.
    """
    if not callable(producer):
        return None
    try:
        from src.compiler.symbolic_equation_compiler import (
            _producer_record, _pipeline_implementation)
        # UNWRAP FIRST. Every `compile_*_ssa` is `lru_cache`-wrapped, and a
        # `functools._lru_cache_wrapper` has no source file, so
        # `_producer_record` raises TypeError on it. Caught and turned into
        # a None key, that was a PERMANENT silent miss: the cache was never
        # written, every launch paid the full 115 s, and nothing said why.
        # Exactly the failure this file's own docstring warns about in
        # another form -- an absence of complaints proving nothing.
        _record, source_digest = _producer_record(
            inspect.unwrap(producer), tuple(key_sources))
        return f"{entry}.{source_digest[:16]}.{_pipeline_implementation()[:16]}"
    except Exception as error:
        warnings.warn(
            f"native law {entry}: no cache key ({type(error).__name__}: "
            f"{error}). The emission will be rebuilt on every launch.",
            RuntimeWarning, stacklevel=2)
        return None


def _emission_path(producer, entry: str, key_sources) -> Path | None:
    key = _emission_key(producer, entry, key_sources)
    return None if key is None else EMISSIONS / f"{key}.pkl"


def _load_emission(producer, entry: str, key_sources):
    path = _emission_path(producer, entry, key_sources)
    if path is None or not path.exists():
        return None
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except Exception:
        return None
    if not isinstance(payload, dict) or "artifact" not in payload:
        return None
    return payload


def _store_emission(producer, entry: str, key_sources, payload) -> None:
    path = _emission_path(producer, entry, key_sources)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".partial")
    try:
        with scratch.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        scratch.replace(path)          # atomic: never a half-written hit
    except Exception:
        scratch.unlink(missing_ok=True)


@dataclass
class NativeLaw:
    """One compiled law, addressed by the names it was authored with."""

    name: str
    execution: Any
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_buffers: list = field(default_factory=list)
    output_buffers: list = field(default_factory=list)
    ir_lines: int = 0
    input_slot: dict = field(default_factory=dict)
    output_slot: dict = field(default_factory=dict)
    #: what this law has cost. A profiler that has to be switched on
    #: measures a different program from the one that runs, so the ledger
    #: is always kept: one `perf_counter` pair against a 24 us call is
    #: under a percent, and the number is the whole point.
    calls: int = 0
    wall_s: float = 0.0

    @classmethod
    def build(cls, producer: Any, entry: str, key_sources=()) -> "NativeLaw":
        """The compiled law, from disk when the sources have not changed.

        `producer` is the repository's own `compile_*_ssa` function, NOT
        its result -- passing the result would build the thing this is
        here to avoid building. It is called only on a miss.
        """
        cached = _load_emission(producer, entry, key_sources)
        if cached is None:
            compilation = producer() if callable(producer) else producer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                artifact = emit_ssa_function_to_llvm(compilation.module, entry)
            if artifact.shortfalls:
                raise RuntimeError(f"{entry}: shortfalls {artifact.shortfalls}")
            metadata = compilation.function.metadata
            cached = {
                "artifact": artifact,
                "inputs": tuple(metadata["argument_names"]),
                "outputs": tuple(metadata["output_names"]),
                "parameters": dict(metadata["parameter_names"]),
                "output_ids": dict(compilation.output_ids),
            }
            _store_emission(producer, entry, key_sources, cached)
        artifact = cached["artifact"]
        ir = str(getattr(artifact, "llvm_ir", "") or "")
        lines = len(ir.splitlines())
        if lines < 15:
            # An empty body emits clean and reports zero shortfalls, so an
            # absence of complaints proves nothing on its own: a bound
            # record can clear a refusal without materialising anything.
            # See COMPILER_INTERPRETATION_RULES.
            raise RuntimeError(
                f"{entry}: emitted {lines} lines of IR, which is an empty "
                f"body reported as success")
        directory = BAKED / f"law_{entry}"
        directory.mkdir(parents=True, exist_ok=True)
        native = compile_artifact(artifact, directory=directory)

        inputs = tuple(cached["inputs"])
        outputs = tuple(cached["outputs"])
        parameters = dict(cached["parameters"])
        output_ids = dict(cached["output_ids"])
        missing = [n for n in inputs if n not in parameters]
        if missing:
            raise RuntimeError(f"{entry}: arguments with no buffer: {missing}")
        execution = prepare_artifact_execution(
            native, {parameters[n]: np.array(0.0, dtype=np.float64)
                     for n in inputs})
        return cls(
            name=entry, execution=execution, input_names=inputs,
            output_names=outputs, ir_lines=lines,
            input_buffers=[execution.buffers[parameters[n]] for n in inputs],
            output_buffers=[execution.buffers[output_ids[n]] for n in outputs],
            input_slot={n: i for i, n in enumerate(inputs)},
            output_slot={n: i for i, n in enumerate(outputs)})

    # -- the whole interface -------------------------------------------
    def write(self, vector) -> None:
        """Every input, in argument order.

        All of them, every call: two craft share one compiled law and one
        set of buffers, so whatever the other one last wrote is still
        sitting there.
        """
        for buffer, value in zip(self.input_buffers, vector):
            buffer[...] = value

    def run(self) -> None:
        self.execution.run()

    def read(self, into=None):
        out = np.empty(len(self.output_buffers)) if into is None else into
        for slot, buffer in enumerate(self.output_buffers):
            out[slot] = buffer[()]
        return out

    def call(self, vector, into=None):
        started = perf_counter()
        self.write(vector)
        self.run()
        out = self.read(into)
        self.wall_s += perf_counter() - started
        self.calls += 1
        return out

    def reset_ledger(self) -> None:
        self.calls = 0
        self.wall_s = 0.0

    @property
    def per_call_ms(self) -> float:
        return self.wall_s / self.calls * 1e3 if self.calls else 0.0
