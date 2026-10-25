"""Audit Turing's current boundary for the Token Orchard ``String`` shim.

This is a focused extraction diagnostic.  Success today means reproducing the
known missing compiler boundary: Turing recognizes and pursues the exact
``token_orchard.api.String.bytes`` source, enters the Python service veneer,
and then rejects ``TokenService`` because that implementation is deliberately
not declared as compiler source.  Once Turing owns the component boundary,
this audit must be revised to require a receipted Token Orchard call without
pursuing ``_one_token`` or the Python queue/lock implementation.
"""

import ast
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
TURING_ROOT = WORKSPACE_ROOT / "turing"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TURING_ROOT))

from token_orchard.api import String
from src.compiler.extraction_contract import ExtractionContract
from src.transmogrifier.graph.graph_express2 import (
    _expand_unresolved_ast_parents,
)


ENTRY_SOURCE = """
def root(payload):
    return String.bytes(payload)
"""


def main():
    contract = ExtractionContract(
        TURING_ROOT / "extraction_contracts" / "program_extraction.yaml"
    ).with_sources(
        (
            (
                "token_orchard.api",
                PROJECT_ROOT / "token_orchard" / "api.py",
            ),
            (
                "token_orchard.__init__",
                PROJECT_ROOT / "token_orchard" / "__init__.py",
            ),
        )
    )
    tree, links, unresolved, unused_bindings = _expand_unresolved_ast_parents(
        ast.parse(ENTRY_SOURCE),
        {"String": String},
        include=contract,
        pursuit_roots=("root",),
    )
    discovered = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    linked = {definition.name for definition, unused_call in links}
    call_rows = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", getattr(node.func, "attr", ""))
        receipt = getattr(node, "_extraction_contract", {})
        call_rows.append((name, receipt))

    required_pursuit = {"bytes", "_one_token", "_byte_values"}
    if not required_pursuit <= discovered | linked:
        raise SystemExit(
            "unexpected extraction result: exact String.bytes pursuit was not visible"
        )
    token_service_receipts = [
        receipt for name, receipt in call_rows if name == "TokenService"
    ]
    if not token_service_receipts:
        raise SystemExit(
            "unexpected extraction result: known TokenService frontier was not visible"
        )
    frontier = token_service_receipts[0]
    if (
        frontier.get("identity") != "token_orchard.service.TokenService"
        or frontier.get("classification") != "unknown"
        or frontier.get("action") != "reject"
        or frontier.get("parameters", {}).get("reason")
        != "provenance_not_declared"
    ):
        raise SystemExit(f"unexpected TokenService frontier receipt: {frontier}")

    print("TOKEN ORCHARD STRING SHIM AUDIT: reproduced current missing boundary")
    print("pursued:", ", ".join(sorted(required_pursuit)))
    print("frontier: TokenService -> provenance_not_declared")
    print(
        "required repair: stop at exact token_orchard.api.String.bytes identity "
        "and emit a receipted Token Orchard component call"
    )


if __name__ == "__main__":
    main()
