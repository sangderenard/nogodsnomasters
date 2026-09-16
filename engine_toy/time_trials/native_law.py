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

THE PORT ORDER IS THE CONTRACT. `argument_names` and `output_names` fix
it, and `parameter_names` / `output_ids` map each name to the value whose
buffer it is. Both were checked to be exactly 1:1 with the argument list
-- no argument without a buffer, no buffer without an argument -- so a
vector in argument order addresses the kernel completely.
"""
from __future__ import annotations

import warnings
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.compiler.ssa_llvm_backend import (
    emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)

BAKED = Path(__file__).resolve().parents[1] / "baked"


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
    def build(cls, compilation: Any, entry: str) -> "NativeLaw":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            artifact = emit_ssa_function_to_llvm(compilation.module, entry)
        if artifact.shortfalls:
            raise RuntimeError(f"{entry}: shortfalls {artifact.shortfalls}")
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

        metadata = compilation.function.metadata
        inputs = tuple(metadata["argument_names"])
        outputs = tuple(metadata["output_names"])
        parameters = dict(metadata["parameter_names"])
        output_ids = dict(compilation.output_ids)
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
