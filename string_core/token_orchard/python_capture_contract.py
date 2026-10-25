"""Exact Python-source capture contract for automatic Token Orchard linking.

This is compiler metadata, not monkeypatching. Live Python keeps CPython's
ordinary behavior. Turing may replace a reachable operation only after its
receiver/value class and full source form prove one entry below.
"""

from __future__ import annotations

from .contract import SEMANTIC_DEPTH_NONE


PYTHON_CAPTURE_ABI_VERSION = 0
TOKEN_ORCHARD_COMPONENT = "token_orchard"

CAPTURE_CORE_READY = 1
CAPTURE_NEEDS_BATCH_OPCODE = 2
CAPTURE_NEEDS_EXACT_TRANSFORM = 3


PYTHON_STRING_CAPTURES = {
    "source.text_literal": {
        "proof": "source_glyph_sequence + builtins.str value class",
        "opcode": "intern_text_constant",
        "state": CAPTURE_CORE_READY,
    },
    "source.bytes_literal": {
        "proof": "literal bytes value + builtins.bytes value class",
        "opcode": "intern_bytes_constant",
        "state": CAPTURE_CORE_READY,
    },
    "operator.add.text": {
        "proof": "both operands are text-domain token values",
        "opcode": "concat",
        "state": CAPTURE_CORE_READY,
    },
    "operator.add.bytes": {
        "proof": "both operands are byte-domain token values",
        "opcode": "concat",
        "state": CAPTURE_CORE_READY,
    },
    "builtins.len.text_or_bytes": {
        "proof": "argument is a Token Orchard token with proven domain",
        "opcode": "length",
        "state": CAPTURE_CORE_READY,
    },
    "operator.getitem.index": {
        "proof": "token receiver + scalar integer index",
        "opcode": "atom_at",
        "state": CAPTURE_CORE_READY,
    },
    "operator.getitem.slice": {
        "proof": "token receiver + bounded unit-step scalar slice",
        "opcode": "slice",
        "state": CAPTURE_CORE_READY,
    },
    "operator.eq.text_or_bytes": {
        "proof": "both tokens share vocabulary identity and domain",
        "opcode": "token_equal",
        "state": CAPTURE_CORE_READY,
    },
    "operator.contains.text_or_bytes": {
        "proof": "same-domain haystack and needle tokens",
        "opcode": "exact_contains_batch",
        "state": CAPTURE_NEEDS_BATCH_OPCODE,
    },
    "builtins.str_or_bytes.find": {
        "proof": "same-domain receiver and needle tokens + scalar bounds",
        "opcode": "exact_find_batch",
        "state": CAPTURE_NEEDS_BATCH_OPCODE,
    },
    "builtins.str_or_bytes.startswith": {
        "proof": "same-domain receiver and prefix tokens + scalar bounds",
        "opcode": "prefix_equal_batch",
        "state": CAPTURE_NEEDS_BATCH_OPCODE,
    },
    "builtins.str_or_bytes.endswith": {
        "proof": "same-domain receiver and suffix tokens + scalar bounds",
        "opcode": "suffix_equal_batch",
        "state": CAPTURE_NEEDS_BATCH_OPCODE,
    },
    "builtins.str_or_bytes.count": {
        "proof": "same-domain receiver and needle tokens + scalar bounds",
        "opcode": "exact_count_batch",
        "state": CAPTURE_NEEDS_BATCH_OPCODE,
    },
    "builtins.str.encode.explicit": {
        "proof": "text token receiver + explicit supported encoding literal",
        "opcode": "encode_receipt",
        "state": CAPTURE_NEEDS_BATCH_OPCODE,
    },
    "builtins.bytes.decode.explicit": {
        "proof": "byte token receiver + explicit supported encoding literal",
        "opcode": "decode_receipt",
        "state": CAPTURE_CORE_READY,
    },
    "builtins.str.strip": {
        "proof": "exact builtins.str receiver and exact strip argument form",
        "opcode": "strip_transform_receipt",
        "state": CAPTURE_NEEDS_EXACT_TRANSFORM,
    },
    "builtins.str.join": {
        "proof": "exact text separator and flat token batch",
        "opcode": "join_batch",
        "state": CAPTURE_NEEDS_BATCH_OPCODE,
    },
    "builtins.bytes.join": {
        "proof": "exact byte separator and flat token batch",
        "opcode": "join_batch",
        "state": CAPTURE_NEEDS_BATCH_OPCODE,
    },
    "builtins.str.replace": {
        "proof": "exact text receiver, old/new tokens, and count semantics",
        "opcode": "replace_transform_receipt",
        "state": CAPTURE_NEEDS_EXACT_TRANSFORM,
    },
    "builtins.str.lower": {
        "proof": "exact builtins.str receiver and versioned Unicode tables",
        "opcode": "lower_transform_receipt",
        "state": CAPTURE_NEEDS_EXACT_TRANSFORM,
    },
    "builtins.str.isalpha": {
        "proof": "exact builtins.str receiver and versioned Unicode tables",
        "opcode": "unicode_predicate_batch",
        "state": CAPTURE_NEEDS_BATCH_OPCODE,
    },
    "builtins.str.isdecimal": {
        "proof": "exact builtins.str receiver and versioned Unicode tables",
        "opcode": "unicode_predicate_batch",
        "state": CAPTURE_NEEDS_BATCH_OPCODE,
    },
}


# These are authored compiler facts already present in Turing. They are the
# join points for the component capture; they are not reimplemented here.
TURING_EXISTING_STRING_SEAMS = {
    "source_glyph_sequence": "literal and JoinedStr glyph recipes",
    "string_token": "resident StringTable token projection",
    "str.strip": "exact StringTable glyph projection transform",
    "utf8_encode_alias": "explicit str.encode('utf-8') ABI transform",
    "bytes_join_source_transform": "b''.join(list(parameter)) byte view",
    "cpython_unicode_exports": (
        "lifted _PyUnicode scalar helpers are differential oracles only"
    ),
}


def python_capture_receipt(source_forms):
    """Return the deterministic component-link receipt for exact forms."""
    forms = tuple(dict.fromkeys(source_forms))
    captures = []
    for source_form in forms:
        capture = PYTHON_STRING_CAPTURES.get(source_form)
        if capture is None:
            raise ValueError(f"unknown Python string capture {source_form!r}")
        captures.append({"source_form": source_form, **capture})
    return {
        "abi_version": PYTHON_CAPTURE_ABI_VERSION,
        "required_component": TOKEN_ORCHARD_COMPONENT if captures else None,
        "semantic_depth": SEMANTIC_DEPTH_NONE,
        "captures": tuple(captures),
    }
