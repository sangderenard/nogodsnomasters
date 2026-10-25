"""Attempt holistic Turing lowering of Token Orchard's live semantic core.

This is deliberately an audit, not a second implementation.  The submitted
entry constructs the real ``Vocabulary`` and calls its real ``intern_batch``;
Turing pursues the declared Token Orchard source files from that boundary.
The audit is allowed to fail loudly while a compiler frontier remains.
"""

from pathlib import Path
import os
import sys


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


ENTRY_SOURCE = """
from token_orchard.core import Vocabulary


def token_orchard_whole_core_entry(
    domain: int, values: list[int], offsets: list[int]
):
    vocabulary = Vocabulary()
    return vocabulary.intern_batch(domain, values, offsets)
"""


def main():
    contract = ExtractionContract(
        TURING_ROOT / "extraction_contracts" / "program_extraction.yaml"
    ).with_sources(
        (
            ("token_orchard.contract", PROJECT_ROOT / "token_orchard" / "contract.py"),
            ("token_orchard.kernel", PROJECT_ROOT / "token_orchard" / "kernel.py"),
            ("token_orchard.core", PROJECT_ROOT / "token_orchard" / "core.py"),
        )
    )
    module, unused_outputs, exports = lower_ast_source_to_ssa(
        ENTRY_SOURCE,
        "token_orchard_whole_core_entry",
        name="token_orchard_whole_core",
        extraction_contract=contract,
        progress=print,
        progress_bars=False,
        compile_profile=True,
    )
    print(
        "TOKEN ORCHARD WHOLE-CORE LOWER PASS:",
        f"functions={len(module.functions)}",
        f"exports={tuple(exports)}",
    )


if __name__ == "__main__":
    main()
