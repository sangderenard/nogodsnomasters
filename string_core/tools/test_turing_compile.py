"""Lower, compile, and execute Token Orchard's shared fingerprint kernel."""

from pathlib import Path
import os
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
TURING_ROOT = WORKSPACE_ROOT / "turing"
BUILD_ROOT = PROJECT_ROOT / "build" / "turing_compile_gate"
BUILD_ROOT.mkdir(parents=True, exist_ok=True)
# Turing initializes repository caches during import, so establish their
# ignored lifetime before importing the compiler modules.
os.chdir(BUILD_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TURING_ROOT))

from token_orchard.kernel import fingerprint_atom_step
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)


CASES = (
    (0, 0, 0),
    (0xCBF29CE4, 0x84222325, 0x41),
    (0xFFFFFFFF, 0xFFFFFFFF, 0x10FFFF),
    (0x340D631B, 0x7BDDDCDA, 0x1F9E0),
)


def lowered_kernel():
    source = (PROJECT_ROOT / "token_orchard" / "kernel.py").read_text(
        encoding="utf-8"
    )
    module, unused_outputs, exports = lower_ast_source_to_ssa(
        source,
        "fingerprint_atom_step",
        name="token_orchard_fingerprint_atom_step",
        extraction_contract=str(
            TURING_ROOT / "extraction_contracts" / "program_extraction.yaml"
        ),
        progress=lambda unused_message: None,
        progress_bars=False,
        compile_profile=False,
    )
    findings = CorrelationTable.build(module).findings(module)
    if findings:
        raise AssertionError(findings)
    export = exports[0]
    function = module.functions[export]
    parameters = dict(function.metadata["parameter_names"])
    returns = [
        instruction.args
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    ]
    if len(returns) != 1 or len(returns[0]) != 2:
        raise AssertionError("fingerprint kernel must return exactly two limbs")
    return module, export, parameters, tuple(int(value.id) for value in returns[0])


def feeds(parameters, case):
    return {
        parameters[name]: np.array([value], dtype=np.int64)
        for name, value in zip(("high", "low", "atom"), case)
    }


def assert_execution(execution, outputs, expected, backend):
    got = tuple(
        int(execution.buffers[value_id].reshape(-1)[0])
        for value_id in outputs
    )
    if got != expected:
        raise AssertionError(f"{backend} got {got}, expected {expected}")


def run_c(module, export, parameters, outputs, directory):
    artifact = emit_ssa_to_c(module, export)
    if not artifact.complete:
        raise AssertionError(artifact.shortfalls)
    artifact.compile(directory / "c")
    for case in CASES:
        execution = artifact.prepare_execution(feeds(parameters, case)).run()
        assert_execution(
            execution, outputs, fingerprint_atom_step(*case), "c"
        )
    print("C COMPILE+EXECUTE PASS: 4/4 exact limb pairs")


def run_llvm(module, export, parameters, outputs, directory):
    emitted = emit_ssa_function_to_llvm(module, export)
    if emitted.shortfalls:
        raise AssertionError(emitted.shortfalls)
    artifact = compile_artifact(emitted, directory=directory / "llvm")
    for case in CASES:
        execution = prepare_artifact_execution(
            artifact, feeds(parameters, case)
        ).run()
        assert_execution(
            execution, outputs, fingerprint_atom_step(*case), "llvm"
        )
    print("LLVM COMPILE+EXECUTE PASS: 4/4 exact limb pairs")


def main():
    # Keep native products under the project's ignored build tree.  On
    # Windows the C artifact deliberately owns its loaded DLL for the life of
    # the process, so a TemporaryDirectory would attempt to unlink a live
    # module after an otherwise successful gate.  Turing's repository caches
    # are working-directory relative, so compilation runs here as well.
    directory = BUILD_ROOT
    module, export, parameters, outputs = lowered_kernel()
    run_c(module, export, parameters, outputs, directory)
    run_llvm(module, export, parameters, outputs, directory)
    print("TOKEN ORCHARD TURING COMPILE GATE PASS")


if __name__ == "__main__":
    main()
