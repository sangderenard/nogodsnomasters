# Token Orchard Phase 0 contract

**Contract version:** 0

**Schema version:** 0

**Batch ABI version:** 0

This receipt fixes the executable semantics for the first reference core.

## Exact domains

Token Orchard has two disjoint domains:

- `DOMAIN_TEXT = 1`: finite sequences of Unicode scalar values;
- `DOMAIN_BYTES = 2`: finite sequences of unsigned octets.

Text identity never depends on a byte encoding. Byte identity never implies
that a sequence is valid text. Encoding is an explicit, strict bridge that
returns both token identities in a receipt. No replacement, normalization,
case folding, or locale conversion is implicit.

## Integer widths

- token IDs, epochs, offsets, lengths, and batch values use unsigned 64-bit
  semantics;
- text atoms are valid unsigned 21-bit Unicode scalar values excluding the
  surrogate range;
- byte atoms are unsigned 8-bit values;
- fingerprints are two unsigned 64-bit words;
- Python integers are range-checked where they cross these contracts.

Overflow returns `STATUS_LIMIT_EXCEEDED` and does not mutate the vocabulary.

## Token allocation

`0` is invalid. `1` is the text-empty token and `2` is the byte-empty token.
Text atom tokens occupy an arithmetic range ordered by scalar value with the
surrogate hole removed. Byte atom tokens occupy the following 256 values.
Aggregate IDs begin at the next value and increase monotonically. Empty and
atom tokens exist at epoch zero without allocating a million Python objects.
Aggregate IDs are never recycled.

This arithmetic atom scheme provides deterministic IDs and constant-time atom
conversion while keeping authoritative aggregate storage compact.

## Vocabulary and epochs

A vocabulary is one mutation history containing both domains. Aggregate mint
number `n` advances the epoch by one. Existing token identity and expansion
never change. A bare token ID is local to a vocabulary; portable use will also
require the future vocabulary/snapshot receipt.

## Authoritative arrays

Aggregate rows are aligned across these structure-of-arrays fields:

```text
domain
atom_length
fingerprint_hi
fingerprint_lo
exact_begin
exact_count
cover_begin
cover_count
created_epoch
exact_atoms
cover_tokens
```

The Python reference materializes these as contiguous standard-library
`array` buffers (`B`, `I`, and `Q` element codes), not lists of boxed integer
objects. They expose the buffer protocol for zero-copy typed views. Portable
snapshot byte order will be fixed independently of the host-native array byte
order in Phase 2.

Fingerprint and prefix indexes are derived and rebuildable. Exact atoms prove
identity. Covers prove graph structure but do not define identity.

## Fingerprint

Version zero uses two deterministic 64-bit FNV-1a streams with distinct offset
bases over the domain and four little-endian bytes per atom. The executable
kernel represents each lane as two `u32` limbs, so multiplication never exceeds
safe signed-`i64` arithmetic in compiled targets. Length participates in the
candidate-index key. Tests may select constant-zero fingerprint mode. Every
candidate match is verified against exact atoms, so fingerprint selection
cannot alter semantics.

## Cover order

An initial cover may use only epoch-zero atom tokens and aggregates minted
before the new node. Among exact covers, choose the lexicographic minimum of:

```text
(piece_count, atom_piece_count, token_id_vector)
```

The atom cover guarantees existence. Derived indexes must not affect this
ordering.

## Batch ordering

Batch input is a flat values array plus `N + 1` offsets. Offset structure is
validated before mutation. Items execute from index zero through `N - 1` and
observe successful earlier items in the same batch. An invalid item returns an
item status without mutation and does not prevent later valid items. A malformed
batch envelope rejects the whole batch before mutation.

## Version-zero limits

Defaults are configurable downward for tests:

- maximum atoms in one sequence: `1_048_576`;
- maximum items in one batch: `65_536`;
- maximum aggregate tokens: `4_294_967_296`;
- maximum token ID: `2^64 - 1`.

## Status codes

```text
0  OK
1  INVALID_DOMAIN
2  INVALID_ATOM
3  INVALID_TOKEN
4  INVALID_OFFSETS
5  INVALID_SLICE
6  LIMIT_EXCEEDED
7  INVALID_ENCODING
8  UNSUPPORTED_ENCODING
9  INVALID_SNAPSHOT
10 UNSUPPORTED_SCHEMA
11 QUEUE_FULL
12 SERVICE_CLOSED
13 SERVICE_TIMEOUT
14 SERVICE_FAILURE
```

Normal failure is returned as integers, not exceptions.
