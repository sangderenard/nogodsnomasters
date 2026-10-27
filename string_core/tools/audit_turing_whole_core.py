"""Compile and execute Token Orchard's full manager acceptance workload.

The submitted entry is the real tracked ``repeated_manager`` acceptance
workload. Turing pursues the complete reachable Token Orchard source graph,
then the exact resulting repository-SSA module is emitted and executed through
each native backend. No reduced kernel substitutes for the manager workload.
"""

from pathlib import Path
import os
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
TURING_ROOT = WORKSPACE_ROOT / "turing"
BUILD_ROOT = PROJECT_ROOT / "build" / "turing_whole_core_audit"

BUILD_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(BUILD_ROOT)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TURING_ROOT))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)


ENTRY_SOURCE = (
    PROJECT_ROOT / "acceptance" / "manager_workload.py"
).read_text(encoding="utf-8")

ENTRYPOINT = "repeated_manager"

CASES = (
    (0, 0),
    (1, 7000175),
    (17, 119003791),
    (300, 2100321600),
    (3000, 21027516000),
)


def feeds(parameters, rounds):
    return {
        parameters["rounds"]: np.array([rounds], dtype=np.int64),
    }


def assert_result(execution, output, expected, backend, rounds):
    got = int(execution.buffers[output].reshape(-1)[0])
    if got != expected:
        raise AssertionError(
            f"{backend} rounds={rounds} got {got}, expected {expected}"
        )


def run_c(module, export, parameters, output):
    artifact = emit_ssa_to_c(module, export)
    if not artifact.complete:
        raise AssertionError(artifact.shortfalls)

    artifact.compile(BUILD_ROOT / "c")

    for rounds, expected in CASES:
        execution = artifact.prepare_execution(
            feeds(parameters, rounds)
        ).run()
        assert_result(execution, output, expected, "c", rounds)

    print("TOKEN ORCHARD WHOLE-CORE C PASS: 5/5 golden manager cases")


def run_llvm(module, export, parameters, output):
    emitted = emit_ssa_function_to_llvm(module, export)
    if emitted.shortfalls:
        raise AssertionError(emitted.shortfalls)

    artifact = compile_artifact(
        emitted,
        directory=BUILD_ROOT / "llvm",
    )

    for rounds, expected in CASES:
        execution = prepare_artifact_execution(
            artifact,
            feeds(parameters, rounds),
        ).run()
        assert_result(execution, output, expected, "llvm", rounds)

    print("TOKEN ORCHARD WHOLE-CORE LLVM PASS: 5/5 golden manager cases")


def main():
    contract = (
        ExtractionContract(
            TURING_ROOT / "extraction_contracts" / "program_extraction.yaml"
        )
        .with_sources(
            (
                (
                    "token_orchard.contract",
                    PROJECT_ROOT / "token_orchard" / "contract.py",
                ),
                (
                    "token_orchard.kernel",
                    PROJECT_ROOT / "token_orchard" / "kernel.py",
                ),
                (
                    "token_orchard.core",
                    PROJECT_ROOT / "token_orchard" / "core.py",
                ),
            )
        )
        .with_execution_file(
            TURING_ROOT
            / "extraction_contracts"
            / "vehicle_full_native_execution.yaml"
        )
    )

    module, unused_outputs, exports = lower_ast_source_to_ssa(
        ENTRY_SOURCE,
        ENTRYPOINT,
        name="token_orchard_manager_acceptance",
        extraction_contract=contract,
        progress=print,
        progress_bars=True,
        compile_profile=True,
    )

    print(
        "TOKEN ORCHARD WHOLE-CORE LOWER PASS:",
        f"functions={len(module.functions)}",
        f"exports={tuple(exports)}",
    )

    if len(exports) != 1:
        raise AssertionError(
            f"expected exactly one exported manager entry, got {exports!r}"
        )

    export = exports[0]
    function = module.functions[export]
    parameters = dict(function.metadata["parameter_names"])

    returns = [
        instruction.args
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    ]

    if len(returns) != 1 or len(returns[0]) != 1:
        raise AssertionError(
            "repeated_manager must have exactly one scalar return: "
            f"{returns!r}"
        )

    output = int(returns[0][0].id)

    run_c(module, export, parameters, output)
    run_llvm(module, export, parameters, output)

    print("TOKEN ORCHARD WHOLE-CORE NATIVE ACCEPTANCE PASS")


if __name__ == "__main__":
    main()