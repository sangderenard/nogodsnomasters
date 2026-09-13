"""THE EXTRACTION CONTRACT engine_toy compiles under.

WHAT A CONTRACT IS FOR, because not passing one looks free and is not.
`lower_ast_source_to_ssa` used to accept `extraction_contract=None` and
run with the machine-decompilation gate off. Nothing announced that.
What you got instead was a lowering with no declared ABI for anything
crossing its boundary, so every receiver the compiler could not see
came back `opaque-state-effect` -- and those failures read exactly like
defects in the program being compiled. A whole afternoon can go into
reading them that way.

The base sheet already says the two things that matter most:

    dependency_search: reachable        the compiler pursues the
                                        program's own dependencies. It
                                        does not need to be handed them,
                                        and narrowing a target to avoid
                                        an unresolved receiver is
                                        working around the compiler
                                        rather than with it.
    unlowered_behavior: execute_in_python
                                        a compartment that will not
                                        lower stays resident in Python.
                                        With a contract that is a
                                        boundary; without one it is a
                                        hard refusal.

So most of what looked like the engine sim being uncompilable was the
absence of this file.

WHAT A RECORD DECLARATION ADDS. `program_abi.records` states the
storage, dtype, rank, shape and mutability of each field of a class
that crosses the boundary. Declared, those fields can be spans the
native side writes through; undeclared, the object is opaque and every
method on it is a wall. `EngineCycleSim` is not declared here yet --
its state is a graph of Python objects rather than typed spans, and
inventing an ABI for it would be asserting a layout the class does not
have. Under `execute_in_python` it does not need one to make progress;
it needs one to go fully native, and that is a real piece of work on
the class rather than on this file.
"""
from __future__ import annotations

from pathlib import Path

import graph_physics  # puts turing on sys.path

_TURING = Path(graph_physics._TURING_ROOT)
_SHEETS = _TURING / "extraction_contracts"

#: the repository's own extraction policy
PROGRAM_EXTRACTION = _SHEETS / "program_extraction.yaml"
#: the overlay that demands everything lower natively
FULL_NATIVE_EXECUTION = _SHEETS / "vehicle_full_native_execution.yaml"


def contract(*, full_native: bool = False):
    """The contract engine_toy lowers under.

    `full_native=True` applies the overlay that refuses a Python-
    resident boundary. Use it to ASK whether a law is fully native --
    never to compile something for the first time, because then every
    unlowered compartment is a failure instead of a boundary and the
    report tells you nothing about which ones were close.
    """
    from src.compiler.extraction_contract import ExtractionContract

    policy = ExtractionContract(PROGRAM_EXTRACTION)
    if full_native:
        policy = policy.with_execution_file(FULL_NATIVE_EXECUTION)
    return policy


def declared_records() -> tuple[str, ...]:
    """Which records the base sheet already declares an ABI for."""
    return tuple(contract().program_abi.receipt().get("records", {}))


if __name__ == "__main__":
    c = contract()
    print(f"sheet     {PROGRAM_EXTRACTION}")
    print(f"records   {', '.join(declared_records()) or '(none)'}")
    receipt = c.execution.receipt() if hasattr(c, "execution") else {}
    for key in ("host_runtime", "dependency_search", "native_lowering",
                "unlowered_behavior", "require_full_native"):
        if key in receipt:
            print(f"  {key:20s} {receipt[key]}")
