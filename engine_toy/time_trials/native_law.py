"""A law's batch kernel, prepared once, driven through its own buffers.

    law = NativeLaw.build(compile_wheel_contact_ssa,
                          "abstract_ui_wheel_contact", batch=8)
    law.column("tire_radial_compression")[:] = depths   # 8 wheels at once
    law.run()
    forces = law.out("chassis_force_y")

THE LOWERING IS NOT MINE. `src/compiler/native_law_kernels.law_kernel`
does it: it takes the law's AbstractTensor stage -- the stage the
compiler materialises from the law's SSA -- and lowers THAT through
`lower_ast_source_to_ssa` under a contract that declares every input as
a batch span. sympy -> AbstractTensor -> SSA, which is the sanctioned
route, and it caches the kernel and its DLL on disk itself.

An earlier version of this file emitted LLVM straight from the symbolic
module, skipping the AbstractTensor stage entirely. It worked and it was
bit-exact, and it was still a second lowering path maintained beside the
repository's own. This one has no lowering in it at all.

WHAT IS LEFT HERE IS THE BUFFER DISCIPLINE. `LawKernel.__call__` runs
`prepare_artifact_execution` on every call, which reallocates the whole
ABI each time. Preparing once and writing into the buffers it already
holds is the entire contribution, and it is worth measuring. Eight
contact rows, one frame's worth for two craft:

    per-row, 8 calls of batch 1             0.1143 ms
    LawKernel.__call__ at batch 8           0.2909 ms
    batch 8, prepared once  (this)          0.0577 ms

five times the repository's call path and twice a per-row loop over a
hand-emitted kernel, with `chassis_force_y` 12225.706161 in all three
and a difference of 0.000e+00.

At batch > 1 the arrays fed to `prepare_artifact_execution` ARE the
buffers it returns -- checked, `buffers[id] is column` -- so writing a
column is writing the kernel's memory and there is no copy anywhere on
the path.

WHY THE KERNEL IS CACHED AGAIN HERE. `law_kernel` already caches, but
its key is `symbolic_abstract_tensor_source(compilation, ...)` -- you
must be holding the compilation to ask whether its kernel is on disk.
Loading that compilation is 19.6 s for the contact law and 95.2 s for
the body, of cloudpickle alone, and it is the whole of a cold start.
So the record here is keyed on the two things that can be known from
FILES: the digest of the producer's source and the digest of the
lowering pipeline -- the same two facts the repository's symbolic cache
keys on, so an edit that invalidates one invalidates the other and
neither can serve what the other has given up on.

Measured end to end on the game's two laws: 128.1 s of startup down to
11.3 s, and 97 MB of cached payload down to 392 KB.
"""
from __future__ import annotations

import inspect
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

BAKED = Path(__file__).resolve().parents[1] / "baked"
KERNELS = BAKED / "kernels"
BACKEND = "llvm"


def _kernel_key(producer, entry: str, batch: int, key_sources) -> str | None:
    """A name for this kernel that costs nothing to compute.

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
        # `_producer_record` raises TypeError on it. Caught quietly, that
        # was a PERMANENT silent miss: nothing written, full price paid
        # every launch, and no complaint anywhere.
        _record, source_digest = _producer_record(
            inspect.unwrap(producer), tuple(key_sources))
        return (f"{entry}.b{int(batch)}.{source_digest[:16]}"
                f".{_pipeline_implementation()[:16]}")
    except Exception as error:
        warnings.warn(
            f"native law {entry}: no cache key ({type(error).__name__}: "
            f"{error}). Its kernel will be rebuilt on every launch.",
            RuntimeWarning, stacklevel=2)
        return None


def _load_kernel(key: str | None):
    """The stored kernel, with its library pointed at the DLL beside it."""
    if key is None:
        return None
    record = KERNELS / key / "kernel.pkl"
    if not record.is_file():
        return None
    try:
        with record.open("rb") as stream:
            kernel = pickle.load(stream)
        libraries = sorted((KERNELS / key).glob("*.dll"))
        if not libraries:
            return None
        kernel.artifact.library_path = libraries[0]
        kernel.artifact._entry = None
        return kernel
    except Exception:
        return None          # a stale or foreign record is rebuilt, never used


def _store_kernel(key: str | None, kernel) -> None:
    if key is None:
        return
    from src.compiler.ssa_llvm_backend import compile_artifact
    directory = KERNELS / key
    directory.mkdir(parents=True, exist_ok=True)
    try:
        # Build a DLL this cache owns. `law_kernel` leaves one in the
        # system temp directory, and a cache whose payload lives in temp
        # is a cache that evaporates.
        compile_artifact(kernel.artifact, directory=directory,
                         optimization="O2")
        entry = kernel.artifact._entry
        library = kernel.artifact.library_path
        kernel.artifact._entry = None
        kernel.artifact.library_path = None
        scratch = directory / "kernel.partial"
        with scratch.open("wb") as stream:
            pickle.dump(kernel, stream, protocol=pickle.HIGHEST_PROTOCOL)
        scratch.replace(directory / "kernel.pkl")  # never a half-written hit
        kernel.artifact.library_path = library
        kernel.artifact._entry = entry
    except Exception as error:
        warnings.warn(f"native law: kernel not cached ({type(error).__name__}:"
                      f" {error})", RuntimeWarning, stacklevel=2)


@dataclass
class NativeLaw:
    """One law's batch kernel, addressed by the names it was authored with."""

    name: str
    batch: int
    execution: Any
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    columns: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    input_slot: dict = field(default_factory=dict)
    output_slot: dict = field(default_factory=dict)
    #: what this law has cost. A profiler that has to be switched on
    #: measures a different program from the one that runs, so the ledger
    #: is always kept: one `perf_counter` pair against a 58 us call is
    #: well under a percent, and the number is the whole point.
    calls: int = 0
    wall_s: float = 0.0

    @classmethod
    def build(cls, producer: Any, entry: str, *, batch: int = 1,
              key_sources=()) -> "NativeLaw":
        """The law's kernel, from disk when the sources have not changed.

        `producer` is the repository's own `compile_*_ssa` function, NOT
        its result: holding the result means having paid the load this
        exists to avoid.
        """
        from src.compiler.ssa_llvm_backend import prepare_artifact_execution

        os.environ.setdefault("TURING_LAW_NATIVE", BACKEND)
        key = _kernel_key(producer, entry, batch, key_sources)
        kernel = _load_kernel(key)
        if kernel is None:
            from src.compiler.native_law_kernels import law_kernel
            compilation = producer() if callable(producer) else producer
            kernel = law_kernel(compilation, entry, int(batch), BACKEND)
            _store_kernel(key, kernel)

        inputs = tuple(kernel.argument_names)
        columns = {name: np.zeros(int(batch), dtype=np.float64)
                   for name in inputs}
        execution = prepare_artifact_execution(kernel.artifact, {
            value_id: columns[name]
            for value_id, name in zip(kernel.argument_ids, inputs)})
        # THE FED ARRAY MUST BE THE BUFFER. Where it is not, take the
        # buffer instead: otherwise every write here lands somewhere the
        # kernel will never read, and the law runs on zeros while
        # reporting no error at all.
        for value_id, name in zip(kernel.argument_ids, inputs):
            if execution.buffers[value_id] is not columns[name]:
                columns[name] = execution.buffers[value_id]

        outputs = {name: execution.buffers[value_id]
                   for name, value_id in kernel.output_ids.items()}
        # Outputs the law reduces to a constant have no buffer, because a
        # literal is never a region output. They are still outputs, and a
        # reader must not have to know which kind it is asking for.
        for name, value in getattr(kernel, "constant_outputs", {}).items():
            outputs[name] = np.full(int(batch), float(value), dtype=np.float64)
        order = tuple(outputs)
        return cls(name=entry, batch=int(batch), execution=execution,
                   input_names=inputs, output_names=order,
                   columns=columns, outputs=outputs,
                   input_slot={n: i for i, n in enumerate(inputs)},
                   output_slot={n: i for i, n in enumerate(order)})

    # -- the whole interface -------------------------------------------
    def column(self, name: str):
        """The kernel's own memory for one input, one slot per batch row."""
        return self.columns.get(name)

    def out(self, name: str):
        return self.outputs.get(name)

    def put(self, name: str, value, row: int | None = None) -> None:
        buffer = self.columns.get(name)
        if buffer is None:
            return
        if row is None:
            buffer[:] = value
        else:
            buffer[row] = value

    def get(self, name: str, row: int = 0) -> float:
        buffer = self.outputs.get(name)
        return 0.0 if buffer is None else float(buffer[row])

    def run(self) -> None:
        started = perf_counter()
        self.execution.run()
        self.wall_s += perf_counter() - started
        self.calls += 1

    def reset_ledger(self) -> None:
        self.calls = 0
        self.wall_s = 0.0

    @property
    def per_call_ms(self) -> float:
        return self.wall_s / self.calls * 1e3 if self.calls else 0.0

    def missing(self, names) -> tuple[str, ...]:
        """Names this law does not take -- worth knowing loudly.

        Writing an input that does not exist is silent otherwise, and a
        law quietly ignoring half of what it was told is the failure this
        whole build kept running into.
        """
        return tuple(n for n in names if n not in self.columns)


# ---------------------------------------------------------------------
# the fallback, for a law the batched route refuses
# ---------------------------------------------------------------------

def _emission(producer, entry: str, key_sources):
    """The law emitted straight from its symbolic module, one row wide.

    THIS IS THE FALLBACK AND IT IS NOT THE PREFERRED ROUTE. It skips the
    AbstractTensor stage, which is where the repository wants adjustment
    to happen, and it has no batch axis. It exists because the batched
    route currently REFUSES the vehicle body, and a craft that cannot
    move is worse than a craft that moves the slow way.
    """
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
    key = _kernel_key(producer, entry, 0, key_sources)
    record = None if key is None else KERNELS / key / "emission.pkl"
    if record is not None and record.is_file():
        try:
            with record.open("rb") as stream:
                return pickle.load(stream)
        except Exception:
            pass
    compilation = producer() if callable(producer) else producer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifact = emit_ssa_function_to_llvm(compilation.module, entry)
    if artifact.shortfalls:
        raise RuntimeError(f"{entry}: shortfalls {artifact.shortfalls}")
    metadata = compilation.function.metadata
    payload = {
        "artifact": artifact,
        "inputs": tuple(metadata["argument_names"]),
        "outputs": tuple(metadata["output_names"]),
        "parameters": dict(metadata["parameter_names"]),
        "output_ids": dict(compilation.output_ids),
    }
    if record is not None:
        record.parent.mkdir(parents=True, exist_ok=True)
        try:
            scratch = record.with_suffix(".partial")
            with scratch.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            scratch.replace(record)
        except Exception:
            pass
    return payload


@dataclass
class FannedLaw:
    """One row-at-a-time kernel, presented as if it had a batch axis.

    Same interface as `NativeLaw` -- a column per input, a column per
    output, one `run()` -- so the fleet above does not have to know which
    kind of law it is holding. Underneath, `run()` loops the rows through
    a single set of scalar buffers.

    THE COST IS THE MARSHALLING, not the kernel. 341 writes and 146 reads
    per row, measured at 113 us a call against a kernel that runs in 23
    us. A real batch axis removes all of it, which is exactly what is
    blocked by the aggregate defect recorded in `native_law_kernels`.
    """

    name: str
    batch: int
    execution: Any
    input_names: tuple
    output_names: tuple
    columns: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    input_buffers: list = field(default_factory=list)
    output_buffers: list = field(default_factory=list)
    calls: int = 0
    wall_s: float = 0.0
    _in_pairs: list = field(default_factory=list)
    _out_pairs: list = field(default_factory=list)

    @classmethod
    def build(cls, producer, entry: str, *, batch: int = 1,
              key_sources=()) -> "FannedLaw":
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, prepare_artifact_execution)
        payload = _emission(producer, entry, key_sources)
        artifact = payload["artifact"]
        directory = BAKED / f"law_{entry}"
        directory.mkdir(parents=True, exist_ok=True)
        native = compile_artifact(artifact, directory=directory)
        inputs = tuple(payload["inputs"])
        outputs = tuple(payload["outputs"])
        parameters = dict(payload["parameters"])
        output_ids = dict(payload["output_ids"])
        execution = prepare_artifact_execution(
            native, {parameters[n]: np.array(0.0, dtype=np.float64)
                     for n in inputs if n in parameters})
        law = cls(name=entry, batch=int(batch), execution=execution,
                  input_names=inputs, output_names=outputs)
        law.input_buffers = [execution.buffers[parameters[n]] for n in inputs]
        law.output_buffers = [execution.buffers.get(output_ids.get(n))
                              for n in outputs]
        law.columns = {n: np.zeros(int(batch)) for n in inputs}
        law.outputs = {n: np.zeros(int(batch)) for n in outputs}
        law._in_pairs = [(law.input_buffers[i], law.columns[n])
                         for i, n in enumerate(inputs)]
        law._out_pairs = [(law.output_buffers[i], law.outputs[n])
                          for i, n in enumerate(outputs)
                          if law.output_buffers[i] is not None]
        return law

    def column(self, name): return self.columns.get(name)

    def out(self, name): return self.outputs.get(name)

    def put(self, name, value, row=None) -> None:
        buffer = self.columns.get(name)
        if buffer is None:
            return
        if row is None:
            buffer[:] = value
        else:
            buffer[row] = value

    def get(self, name, row: int = 0) -> float:
        buffer = self.outputs.get(name)
        return 0.0 if buffer is None else float(buffer[row])

    def run(self) -> None:
        """Every row through the one set of scalar buffers.

        The pairs are built ONCE. Resolving `self.columns[name][row]` per
        port per row -- a dict lookup and a numpy scalar index, 341 times
        a row -- cost 374 us a call against the 113 us the same work took
        through a flat list.
        """
        started = perf_counter()
        for row in range(self.batch):
            for buffer, column in self._in_pairs:
                buffer[...] = column[row]
            self.execution.run()
            for buffer, column in self._out_pairs:
                column[row] = buffer[()]
        self.wall_s += perf_counter() - started
        self.calls += 1

    def reset_ledger(self) -> None:
        self.calls = 0
        self.wall_s = 0.0

    @property
    def per_call_ms(self) -> float:
        return self.wall_s / self.calls * 1e3 if self.calls else 0.0


def build_law(producer, entry: str, *, batch: int = 1, key_sources=()):
    """The batched kernel where the repository can give one; else fanned.

    NEVER SILENT. A law that falls back is a law running the slow way
    forever, and the whole point of the profiler beside this file is that
    nobody should have to guess why their machine is dilated.
    """
    key = _kernel_key(producer, entry, batch, key_sources)
    refusal = KERNELS / key / "refused.txt" if key else None
    if refusal is not None and refusal.is_file():
        # A REFUSAL IS A RESULT AND IT IS CACHED LIKE ONE. Re-attempting
        # the batched lowering of the vehicle body costs four minutes of
        # every launch to learn what the last launch already learned.
        # Keyed the same way, so any edit to the law or the pipeline
        # tries it again.
        warnings.warn(
            f"native law {entry}: batched route refused earlier and the "
            f"refusal still stands -- {refusal.read_text(encoding='utf-8')[:200]}",
            RuntimeWarning, stacklevel=2)
        return FannedLaw.build(producer, entry, batch=batch,
                               key_sources=key_sources)
    try:
        return NativeLaw.build(producer, entry, batch=batch,
                               key_sources=key_sources)
    except Exception as error:
        if refusal is not None:
            refusal.parent.mkdir(parents=True, exist_ok=True)
            refusal.write_text(f"{type(error).__name__}: {error}",
                               encoding="utf-8")
        warnings.warn(
            f"native law {entry}: the batched route refused it "
            f"({type(error).__name__}: {str(error)[:160]}). Falling back to "
            f"row-at-a-time emission, which costs the marshalling a batch "
            f"axis would remove.", RuntimeWarning, stacklevel=2)
        return FannedLaw.build(producer, entry, batch=batch,
                               key_sources=key_sources)
