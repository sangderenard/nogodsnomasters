"""Compiler-facing identity and reachability contract for ``String`` calls."""

from __future__ import annotations

from .contract import SEMANTIC_DEPTH_NONE


STRING_NAMESPACE_IDENTITY = "token_orchard.api.String"
STRING_SHIM_ABI_VERSION = 0

# Exact source identities are the authority.  The compiler must not infer a
# string operation from a method spelling on an unrelated class.
STRING_METHOD_CONTRACTS = {
    "bytes": {
        "identity": f"{STRING_NAMESPACE_IDENTITY}.bytes",
        "opcode": "intern_bytes",
        "semantic_depth": SEMANTIC_DEPTH_NONE,
        "arguments": (("payload", "uint8", 1),),
        "results": (("token", "uint64", 0),),
    },
    "bytes_batch": {
        "identity": f"{STRING_NAMESPACE_IDENTITY}.bytes_batch",
        "opcode": "intern_bytes_batch",
        "semantic_depth": SEMANTIC_DEPTH_NONE,
        "arguments": (
            ("values", "uint8", 1),
            ("offsets", "uint64", 1),
        ),
        "results": (("tokens", "uint64", 1),),
    },
    "decode_receipt": {
        "identity": f"{STRING_NAMESPACE_IDENTITY}.decode_receipt",
        "opcode": "decode_receipt",
        "semantic_depth": SEMANTIC_DEPTH_NONE,
        "arguments": (
            ("payload", "uint8", 1),
            ("encoding", "uint16", 0),
        ),
        "results": (
            ("status", "uint16", 0),
            ("encoding", "uint16", 0),
            ("byte_token", "uint64", 0),
            ("text_token", "uint64", 0),
        ),
    },
    "decode": {
        "identity": f"{STRING_NAMESPACE_IDENTITY}.decode",
        "opcode": "decode_or_status",
        "semantic_depth": SEMANTIC_DEPTH_NONE,
        "arguments": (
            ("payload", "uint8", 1),
            ("encoding", "uint16", 0),
        ),
        "results": (("text_token", "uint64", 0),),
    },
}


def required_semantic_depth(method_names):
    """Return the maximum optional phase reachable from exact shim methods."""
    depth = SEMANTIC_DEPTH_NONE
    for name in method_names:
        contract = STRING_METHOD_CONTRACTS.get(name)
        if contract is None:
            raise ValueError(f"unknown String namespace method {name!r}")
        depth = max(depth, int(contract["semantic_depth"]))
    return depth


def string_shim_receipt(method_names):
    """Build the deterministic compiler/linker receipt for used methods."""
    names = tuple(dict.fromkeys(method_names))
    methods = tuple(STRING_METHOD_CONTRACTS[name] for name in names)
    return {
        "abi_version": STRING_SHIM_ABI_VERSION,
        "namespace_identity": STRING_NAMESPACE_IDENTITY,
        "semantic_depth": required_semantic_depth(names),
        "methods": methods,
    }
