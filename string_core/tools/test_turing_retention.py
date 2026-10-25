"""Compile and execute Token Orchard's vectorized retention selection."""

import argparse
from pathlib import Path
import os
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
TURING_ROOT = WORKSPACE_ROOT / "turing"
BUILD_ROOT = PROJECT_ROOT / "build" / "turing_retention_gate"
BUILD_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(BUILD_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TURING_ROOT))

from token_orchard.retention import retention_program_abi
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)


ENTRYPOINT = "retention_stale_mask"
ENTRY_SOURCE = (
    PROJECT_ROOT / "token_orchard" / "semantic_tensor.py"
).read_text(encoding="utf-8")


def lowered_retention():
    contract = ExtractionContract(
        TURING_ROOT / "extraction_contracts" / "program_extraction.yaml"
    ).with_sources(
        (
            (
                "token_orchard.contract",
                PROJECT_ROOT / "token_orchard" / "contract.py",
            ),
            (
                "token_orchard.semantic_tensor",
                PROJECT_ROOT / "token_orchard" / "semantic_tensor.py",
            ),
        )
    ).with_program_abi(retention_program_abi())
    module, unused_outputs, exports = lower_ast_source_to_ssa(
        ENTRY_SOURCE,
        ENTRYPOINT,
        name="token_orchard_retention_v0",
        extraction_contract=contract,
        progress=lambda unused_message: None,
        progress_bars=False,
        compile_profile=False,
    )
    export = exports[0]
    function = module.functions[export]
    parameters = dict(function.metadata["parameter_names"])
    returned = next(
        instruction.args
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    return module, export, parameters, int(returned[0].id)


def feeds(parameters):
    return {
        parameters["last_notice_tick"]: np.array([2, 7, 9, 1], dtype=np.int64),
        parameters["protected"]: np.array([0, 0, 0, 1], dtype=np.int64),
        parameters["current_tick"]: np.array([10], dtype=np.int64),
        parameters["idle_ticks"]: np.array([3], dtype=np.int64),
    }


def assert_result(execution, output, backend):
    got = execution.buffers[output].reshape(-1).tolist()
    expected = [1, 1, 0, 0]
    if got != expected:
        raise AssertionError(f"{backend} got {got}, expected {expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("c", "llvm", "all"),
        default="all",
        help="native backend to verify",
    )
    args = parser.parse_args()
    module, export, parameters, output = lowered_retention()

    if args.backend in {"c", "all"}:
        c_artifact = emit_ssa_to_c(module, export)
        if not c_artifact.complete:
            raise AssertionError(c_artifact.shortfalls)
        c_artifact.compile(BUILD_ROOT / "c")
        assert_result(
            c_artifact.prepare_execution(feeds(parameters)).run(), output, "c"
        )
        print("TOKEN ORCHARD RETENTION C PASS: vector mask 4/4")

    if args.backend in {"llvm", "all"}:
        llvm_emission = emit_ssa_function_to_llvm(module, export)
        if llvm_emission.shortfalls:
            raise AssertionError(llvm_emission.shortfalls)
        llvm_artifact = compile_artifact(
            llvm_emission, directory=BUILD_ROOT / "llvm"
        )
        assert_result(
            prepare_artifact_execution(llvm_artifact, feeds(parameters)).run(),
            output,
            "llvm",
        )
        print("TOKEN ORCHARD RETENTION LLVM PASS: vector mask 4/4")


if __name__ == "__main__":
    main()
