# Token Orchard agent contract

This file governs work inside `C:\dev\Powershell\string_core`.

Read `README.md`, `TOKEN_ORCHARD_ARCHITECTURE.md`, and, when beginning the
implementation continuation, `CONTINUATION.md`. Before Turing integration,
also read `C:\dev\Powershell\turing\AGENTS.md` and the current opening receipt
in `turing\TEST_BASELINE_AND_HAZARDS.md`.

## Authority and scope

- The architecture report is the semantic contract. If code and the report
  differ, stop and reconcile them explicitly; do not silently weaken a law.
- Token Orchard owns its identity and string semantics. Numerical batch work
  uses Turing's existing `AbstractTensor` substrate as-is; do not invent a
  second tensor façade. Native products must retain an integer ABI and need
  not retain the Python/Turing development dependency.
- Do not modify Turing, Nodus, or Spectral Analyzer during a Token Orchard
  milestone unless the task explicitly includes that integration.
- Do not initialize nested Git metadata. The root repository owns this folder.

## Semantic prohibitions

- No implicit Unicode normalization, case folding, locale conversion, or
  malformed-input replacement. Transformations are explicit and receipted.
- A hash is an index hint, never proof of identity. Exact atom comparison is
  mandatory before reusing a token.
- Token identity is not its decomposition. Covers and mipmaps may improve
  without changing a token ID.
- A token ID is meaningful only with its vocabulary and epoch or snapshot.
- Neural models may propose covers, caches, or transforms; they may not mint
  authoritative identities or redefine exact expansion.
- The hot core does not use host strings. Encoding and decoding are adapters.

## Engineering constraints

- Begin with a legible restricted-Python reference, then compile that same
  logic. Do not maintain a hand-translated native algorithm beside it.
- Use structure-of-arrays storage and flat values plus offsets for ragged
  batches. Avoid object-per-node authoritative storage.
- Mutation has one deterministic serialization point. Readers consume
  immutable snapshots or published epochs.
- Keep the public batch ABI integer-only and versioned.
- Build tests from laws before optimizing. Benchmarks are not correctness
  tests and microbenchmarks are not end-to-end throughput claims.

Turing's sanctioned public source-compiler entry is
`src.compiler.fortran_c_shell.lower_ast_source_to_ssa`. Use it as documented.
Do not select an internal function or build a parallel compiler path.

Follow the root and `speaktome` guidance, preserve unrelated dirty work, do
not install packages manually, and record material findings in the shared
experience reports. Generated products belong in the ignored paths declared
by this directory's `.gitignore`.
