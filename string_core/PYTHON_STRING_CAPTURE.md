# Automatic Python source capture for Token Orchard

## Answer

Do this at compiler ingestion, not by monkeypatching Python and not by
installing a replacement `str` class. Live Python continues to execute normal
CPython objects. When Turing compiles a reachable source operation, its
existing receiver/value-class and concordance evidence may emit a Token
Orchard component receipt. That receipt automatically links one shared
vocabulary and rewrites the operation to an integer token opcode.

The authoritative machine-readable table is
`token_orchard/python_capture_contract.py`.

## Existing Turing seams

Turing already owns useful exact facts:

- `source_glyph_sequence` for literals, f-strings, and formatted components;
- resident `StringTable` token-to-glyph projections;
- exact `str.strip()` StringTable projections used by glyph recipes;
- `_utf8_encode_aliases` for explicit `str.encode('utf-8')` parameter views;
- `_bytes_join_source_transforms` for `b''.join(list(parameter))` byte views.
- machine-code lifting tests for CPython's `_PyUnicode_ToLowercase`,
  `_PyUnicode_IsAlpha`, and `_PyUnicode_ToDecimalDigit` exports.

These should be joined to Token Orchard. They should not be copied into a
parallel string recognizer.

The lifted CPython exports are valuable differential oracles, not automatic
permission to call libpython from a native Token Orchard product. In
particular, a scalar `_PyUnicode_ToLowercase` result does not by itself prove
the complete multi-scalar behavior of `str.lower()`. Explicit lower/isalpha/
isdecimal capture remains blocked until the versioned Unicode-table and batch
operation contracts are complete.

## First automatic capture set

The first safe set is text/byte literals, same-domain concatenation, length,
integer indexing, unit-step slicing, equality inside one vocabulary, and
explicit encoding/decoding. All become token operations. A reachable capture
requires component `token_orchard` at semantic depth zero, so it brings no NN
state unless a separate reachable semantic operation requests it.

`str.strip`, `str.replace`, and joins are intentionally not declared ready
merely because Turing can identify them. Token Orchard still needs exact,
batched transform opcodes and transformation receipts for those semantics.
The next read-only integer surface is exact containment, `find`, `startswith`,
`endswith`, and non-overlapping `count`, all with same-domain tokens and
explicit scalar bounds. They are declared pending batch opcodes rather than
pretending that host-string fallback is part of the native core. Implement
these before broader transforms: they need no Unicode policy and exercise the
same exact expansion/index substrate needed by join and replace.
Default-codec encode/decode, locale-sensitive behavior, normalization, case
folding, subclass overrides, and an unrelated method merely named `strip` are
not eligible.

## Required Turing integration

1. Read the incumbent exact receiver/value-class cells and source-operation
   identities. Never match a spelling alone.
2. Emit `python_capture_receipt(...)` data onto the source-linked call or
   operator record.
3. Add Token Orchard once when any reachable receipt names the component.
4. Lower ready captures to the existing Token Orchard integer ABI and one
   vocabulary/writer state. Do not compile the Python queue veneer.
5. Refuse a capture whose proof is incomplete or whose opcode is not ready;
   ordinary Python semantics remain the fallback only when the build target
   explicitly supports a CPython runtime.
6. Preserve observable Python behavior: encoding errors, slice bounds,
   subclass dispatch, and mutation/effect order. Optimization is downstream
   of semantic identity.

This work belongs on the Turing team's current integration branch and must be
rebased and gated with its active PR sequence. Token Orchard does not modify
Turing from the coordination repository.
