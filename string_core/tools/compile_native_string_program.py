"""Compile the tracked Token Orchard consumer as one full-native product."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "acceptance" / "native_string_program.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turing-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    turing_root = args.turing_root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(turing_root))
    sys.path.insert(0, str(PROJECT_ROOT))

    from token_orchard.api import String
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    contract = ExtractionContract(
        turing_root / "extraction_contracts" / "program_extraction.yaml"
    ).with_sources((
        ("token_orchard.api", PROJECT_ROOT / "token_orchard" / "api.py"),
        ("token_orchard.__init__", PROJECT_ROOT / "token_orchard" / "__init__.py"),
    ))
    execution = contract.execution
    if not (
        execution.host_runtime == "native"
        and execution.native_lowering == "required"
        and execution.dispatch_unit == "whole_program"
        and execution.unlowered_behavior == "reject"
        and execution.require_full_native
        and execution.python_callbacks == "reject"
    ):
        raise RuntimeError("the selected extraction contract permits Python execution")

    print(f"source={SOURCE}", flush=True)
    print(f"compiler={turing_root}", flush=True)
    print(f"execution={execution.receipt()}", flush=True)
    module, unused_outputs, exports = lower_ast_source_to_ssa(
        SOURCE.read_text(encoding="utf-8"),
        "root",
        name="token_orchard_native_string_program",
        extraction_contract=contract,
        python_bindings={"String": String},
        progress=lambda message: print(message, flush=True),
        progress_bars=False,
    )
    if len(exports) != 1:
        raise RuntimeError(f"expected one program entry, got {exports!r}")
    artifact = emit_ssa_to_c(module, exports[0])
    if not artifact.complete:
        raise RuntimeError(f"native C emission refused: {artifact.shortfalls}")
    artifact.compile(output / "c")
    print(f"FULL-NATIVE STRING PROGRAM COMPILED: {artifact}", flush=True)


if __name__ == "__main__":
    main()
