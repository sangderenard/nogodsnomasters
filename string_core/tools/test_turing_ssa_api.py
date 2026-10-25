"""Compile and execute Token Orchard's first runtime-sized SSA API batch."""

import argparse
from pathlib import Path
import os
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
TURING_ROOT = WORKSPACE_ROOT / "turing"
BUILD_ROOT = PROJECT_ROOT / "build" / "turing_ssa_api_gate"
BUILD_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(BUILD_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TURING_ROOT))

from token_orchard.contract import AGGREGATE_BASE
from token_orchard.ssa_contract import SSA_ENTRYPOINT, ssa_program_abi
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)


ENTRY_SOURCE = (
    PROJECT_ROOT / "token_orchard" / "ssa_api.py"
).read_text(encoding="utf-8")


def lowered_api():
    contract = ExtractionContract(
        TURING_ROOT / "extraction_contracts" / "program_extraction.yaml"
    ).with_sources(
        (
            ("token_orchard.contract", PROJECT_ROOT / "token_orchard" / "contract.py"),
            ("token_orchard.ssa_api", PROJECT_ROOT / "token_orchard" / "ssa_api.py"),
        )
    ).with_program_abi(ssa_program_abi())
    module, unused_outputs, exports = lower_ast_source_to_ssa(
        ENTRY_SOURCE,
        SSA_ENTRYPOINT,
        name="token_orchard_ssa_api_v0",
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
    parameter_abi = function.metadata["parameter_value_abi"]
    expected_abi = {
        item["parameter"]: item for item in ssa_program_abi()["values"]
    }
    for name, expected in expected_abi.items():
        actual = parameter_abi[name]
        for field in ("storage", "dtype", "rank", "python_type"):
            if actual[field] != expected[field]:
                raise AssertionError((name, field, actual, expected))
    result_accounting = dict(returned[0].accounting or {})
    if (
        result_accounting.get("program_abi_storage") != "span"
        or result_accounting.get("program_abi_rank") != 1
    ):
        raise AssertionError(result_accounting)
    return module, export, parameters, int(returned[0].id)


def feeds(parameters):
    return {
        parameters["tokens"]: np.array(
            [0, 1, 3, AGGREGATE_BASE, AGGREGATE_BASE + 1, AGGREGATE_BASE + 2],
            dtype=np.int64,
        ),
        parameters["aggregate_count"]: np.array([2], dtype=np.int64),
    }


def assert_result(execution, output, backend):
    got = execution.buffers[output].reshape(-1).tolist()
    expected = [0, 1, 2, 3, 3, 0]
    if got != expected:
        raise AssertionError(f"{backend} got {got}, expected {expected}")


def assert_runtime_abi(runtime_abi, backend):
    if runtime_abi["schema"] != "turing-runtime-abi-v1":
        raise AssertionError(runtime_abi)
    buffers = runtime_abi["buffers"]
    tokens = next(item for item in buffers if item["role"] == "input" and item["rank"] == 1)
    scalar = next(item for item in buffers if item["role"] == "input" and item["rank"] == 0)
    output = next(item for item in buffers if item["role"] == "output")
    if (tokens["dtype"], tokens["shape"], tokens["extents_from"]) != (
        "int64", None, "fed"
    ):
        raise AssertionError((backend, tokens))
    if (scalar["dtype"], scalar["shape"]) != ("int64", []):
        raise AssertionError((backend, scalar))
    if (output["dtype"], output["rank"], output["shape"]) != (
        "int64", 1, None
    ):
        raise AssertionError((backend, output))
    if output["extents_from"] != tokens["index"]:
        raise AssertionError((backend, output, tokens))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("c", "llvm", "all"),
        default="all",
        help="native backend to verify",
    )
    args = parser.parse_args()
    module, export, parameters, output = lowered_api()
    if args.backend in {"c", "all"}:
        c_artifact = emit_ssa_to_c(module, export)
        if not c_artifact.complete:
            raise AssertionError(c_artifact.shortfalls)
        c_artifact.compile(BUILD_ROOT / "c")
        assert_result(
            c_artifact.prepare_execution(feeds(parameters)).run(), output, "c"
        )
        assert_runtime_abi(c_artifact.runtime_abi(), "c")
        print("TOKEN ORCHARD SSA API V0 C PASS: runtime span 6/6")
        print("RUNTIME ABI: turing-runtime-abi-v1; int64 span -> int64 span")

    if args.backend in {"llvm", "all"}:
        llvm_emission = emit_ssa_function_to_llvm(module, export)
        if llvm_emission.shortfalls:
            raise AssertionError(llvm_emission.shortfalls)
        llvm_artifact = compile_artifact(
            llvm_emission, directory=BUILD_ROOT / "llvm"
        )
        assert_result(
            prepare_artifact_execution(
                llvm_artifact, feeds(parameters)
            ).run(),
            output,
            "llvm",
        )
        assert_runtime_abi(llvm_artifact.runtime_abi(), "llvm")
        print("TOKEN ORCHARD SSA API V0 LLVM PASS: runtime span 6/6")


if __name__ == "__main__":
    main()
