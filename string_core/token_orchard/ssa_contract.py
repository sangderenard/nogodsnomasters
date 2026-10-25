"""Versioned Turing ProgramABI contract for Token Orchard's SSA surface."""

from token_orchard.contract import SSA_API_VERSION


SSA_ENTRYPOINT = "token_kind_batch"
ABSTRACT_TENSOR_TYPE = "src.common.tensors.abstraction.AbstractTensor"


def ssa_program_abi():
    """Return a fresh ProgramABI declaration for SSA API version zero."""
    if SSA_API_VERSION != 0:
        raise ValueError("no ProgramABI declaration for this SSA API version")
    return {
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": SSA_ENTRYPOINT,
                "parameter": "tokens",
                "storage": "span",
                "dtype": "int64",
                "rank": 1,
                "python_type": ABSTRACT_TENSOR_TYPE,
            },
            {
                "function": SSA_ENTRYPOINT,
                "parameter": "aggregate_count",
                "storage": "scalar",
                "dtype": "int64",
                "rank": 0,
                "shape": [],
                "python_type": "builtins.int",
            },
        ],
    }
