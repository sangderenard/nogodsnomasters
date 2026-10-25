"""Read-only audit of exact Turing seams available for Token Orchard capture."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TURING_ROOT = PROJECT_ROOT.parent / "turing"


EXPECTED_FUNCTIONS = {
    TURING_ROOT / "src" / "compiler" / "fortran_c_shell.py": {
        "_utf8_encode_aliases",
        "_bytes_join_source_transforms",
    },
    TURING_ROOT / "src" / "compiler" / "resident_sequence_materialization.py": {
        "_source_glyph_sequence_materializations",
        "_string_token_glyph_materializations",
    },
    TURING_ROOT / "src" / "common" / "tensors" / "topological_reducer.py": {
        "_relink_settled_glyph_formatters",
    },
}


def main():
    for path, expected in EXPECTED_FUNCTIONS.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        missing = expected - found
        if missing:
            raise SystemExit(f"missing Turing string seams in {path}: {sorted(missing)}")

    lifted_test = (
        TURING_ROOT / "tests" / "test_lifted_machine_ssa_execution.py"
    ).read_text(encoding="utf-8")
    symbols = (
        "_PyUnicode_ToLowercase",
        "_PyUnicode_IsAlpha",
        "_PyUnicode_ToDecimalDigit",
    )
    missing_symbols = [symbol for symbol in symbols if symbol not in lifted_test]
    if missing_symbols:
        raise SystemExit(f"missing lifted CPython oracles: {missing_symbols}")

    print("TOKEN ORCHARD PYTHON CAPTURE AUDIT PASS")
    print("exact Turing seams: glyphs, StringTable, strip, UTF-8 encode, bytes join")
    print("lifted CPython scalar oracles: lowercase, alpha, decimal digit")
    print("no Turing files modified")


if __name__ == "__main__":
    main()
