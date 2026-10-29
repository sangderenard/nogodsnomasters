# Token Orchard

Compile and execute the token-derived `Linear -> Tanh -> Linear` Adam demo:

```powershell
python demos/profiled_text_learning.py --native
```

Token Orchard is the planned graph-native string core for the
`nogodsnomasters` ecosystem. After ingress, character sequences are not
repeatedly manipulated as host-language strings. They are interned as exact
sequences of integer atoms, composed from existing tokens when possible,
minted as tokens in their own right, and subsequently handled through integer
IDs, offsets, graph edges, and immutable memory lookups.

This directory is a deliberate tracked exception in the root coordination
repository, like `engine_toy/`. It is not a nested Git repository and it is not
owned by Turing, Nodus, or Spectral Analyzer.

## Current status

Phase 0 and the scalar Phase 1 reference are implemented. The current core has
disjoint Unicode-scalar and arbitrary-byte domains, arithmetic atom tokens,
collision-proof aggregate interning, deterministic prior-token covers, aligned
typed contiguous structure-of-arrays storage, flat batch ingestion, integer
statuses, strict receipted encoding bridges, deterministic validated snapshots,
and a bounded single-writer queue prototype. Its live two-lane fingerprint
step is now a compiler-safe `u32`-limb kernel, lowered from its actual source
through Turing's sanctioned entry and compile-and-execute verified on C and
LLVM. Queue drains greedily fuse compatible requests into a flat COO-style
values-plus-offsets ABI. One bounded staging pass pads that fused batch, and
the shared kernel runs across the complete `AbstractTensor` batch with masked
lanes for parallel atom validation and fingerprinting. The scalar path is only
a differential oracle, not a compilation target. Eager NumPy dispatch now
slightly beats that oracle at the largest recorded uniform batch, although
whole-core native lowering has now been attempted against the live
`Vocabulary.intern_batch` graph. Turing pursues the complete reachable source
but correctly refuses its implicit Python-array growth: native arena
replacement requires explicit typed pointer cells, capacity, and a heap owner.
The audit is retained as a failing architectural gate while that state ABI is
built. The first versioned SSA-linkable query is also live: the authored
`token_kind_batch(AbstractTensor, int)` source carries a rank-one `int64` span
contract, lowers through the canonical entry, compiles to C and LLVM, publishes
`turing-runtime-abi-v1`, and executes six mixed token cases exactly. Its
ProgramABI declaration is centralized in `token_orchard/ssa_contract.py`.
An optional continuous embedding attachment can now consume newly minted cover
edges and integer-only live-program observations on its own bounded queue. It
maintains a versioned dense token-row map with a fixed deterministic position
on the n-dimensional unit sphere, a shared `AbstractNN` residual network, and
a distinct meaning vector equal to their sum. The established
`Linear -> Tanh -> Linear` network and `Adam` optimizer learn from ordered COO
cover targets and typed context targets; target assembly and inference remain
whole-batch AbstractTensor operations. Reversing a cover changes its target.
The observation ABI represents source, runtime, decompiler, and glyph graph
facts without sending host strings into the core.

Semantic compilation is depth-selectable. The declared prefix is position,
ordered composition, live context, training, immutable publication, and batch
query. `semantic_state_abi(depth, dimension)` proves which state exists at a
chosen depth; no model or Adam state exists below training. The first three
preparation stages are standalone AbstractTensor roots with a depth-pruned
Turing ProgramABI declaration. Their eager laws are green, but they do not yet
claim native artifacts; native VJP/Adam packaging remains behind the recorded
Turing provenance gate.
Queue loss or epoch gaps mark it desynchronized; failures cannot change or
fail authoritative interning. This is an advisory learning surface and
positional-to-meaning adapter, not textual authority, a finished general
reasoner, a mip publisher, or a native-performance claim.
Immutable reader publication, cancellation, and mipmaps remain later phases.
The fingerprint compile gate is only a toolchain proof; it is not intended to
become a separately dispatched runtime subsystem.

## Live Python namespace

`from token_orchard import String` imports a non-instantiable namespace, not a
string type. `String.bytes(payload)`, `String.bytes_batch(payloads)`, and the
explicit `String.decode_receipt(payload, encoding)` accept bytes-like ingress
and return integer token IDs or integer receipts. They share one process-global
self-growing vocabulary and submit real flat batches to its bounded writer
thread. No per-string Python object is created, and host `str` is rejected.

The live Python singleton is an ergonomic boundary only. Its exact source
identity and [compiler handoff](STRING_COMPILER_HANDOFF.md) require Turing to
replace recognized namespace calls with the corresponding linked Token Orchard
SSA operation, retaining only the maximum optional semantic depth reached by
the program. The Python locks, queue implementation, daemon lifecycle, and
exception veneer are never compilation targets.

## Profiled learning demonstration

Run `python demos/profiled_text_learning.py`. The demonstration interns two
exact scalar sequences representing a text edit, waits for the background
graph learner, submits a typed transform observation, performs AbstractNN
backward and Adam updates, and reports queue, epoch, loss, optimizer, and
`cProfile` receipts. It is intentionally marked `native_training: False`: the
current Turing native Adam gate still rejects a generated `forward_vjp`
without source-cell provenance. The exact reproduction and compiler-team
acceptance prompt are in
[NATIVE_NN_DEMO_COMPILER_HANDOFF.md](NATIVE_NN_DEMO_COMPILER_HANDOFF.md).

## Automatic Python source capture

[PYTHON_STRING_CAPTURE.md](PYTHON_STRING_CAPTURE.md) defines exact compiler
capture of ordinary Python string/bytes source forms. It never monkeypatches
Python or introduces a string object. A proven reachable form automatically
requires the Token Orchard component and lowers to token operations. The
machine-readable authority is `token_orchard/python_capture_contract.py`;
strip, replace, join, and Unicode casing remain blocked until their exact
batched transform opcodes exist.

## Retention without identity reuse

Optional lifecycle state records batched token notices against an explicit
logical tick and uses `retention_stale_mask` as one AbstractTensor operation to
select inactive, unprotected aggregate rows. A stale row is only eligible for
a future vocabulary-generation rotation. It is never deleted from a live
epoch, and its integer ID is never recycled. Once immutable readers release
their epoch pins, survivors can be reinterned under a new vocabulary identity
with an explicit remap; learned state then follows an explicit cold, retained,
or conditioned-out policy. Wall-clock time and silent unlearning are excluded.

- [Architecture and engineering report](TOKEN_ORCHARD_ARCHITECTURE.md)
- [Executable Phase 0 contract](PHASE_0_CONTRACT.md)
- [Implementation receipt](IMPLEMENTATION_RECEIPT.md)
- [Continuation prompt](CONTINUATION.md)
- [Compiler integration handoff](STRING_COMPILER_HANDOFF.md)
- [Automatic Python capture](PYTHON_STRING_CAPTURE.md)
- [Native NN compiler handoff](NATIVE_NN_DEMO_COMPILER_HANDOFF.md)
- [Local agent contract](AGENTS.md)

Run the standard-library law suite from this directory:

```powershell
python -m unittest discover -s tests -v
```

Compile and execute the live fingerprint kernel through Turing's C and LLVM
backends with:

```powershell
python tools/test_turing_compile.py
```

Compile and execute the version-zero integer SSA query API through C and LLVM,
while checking its authored annotations, ProgramABI, derived output extent,
and exported runtime ABI, with:

```powershell
python tools/test_turing_ssa_api.py --backend all
```

Compile and execute the vectorized retention eligibility mask through C and
LLVM with:

```powershell
python tools/test_turing_retention.py
```

Turing PR #923 repaired the runtime-sized signed-integer LLVM arithmetic lane;
both gates now execute the exact authored AbstractTensor code through C and
LLVM. Token Orchard's logical identifiers remain unsigned by contract, while
the currently proven native ProgramABI is deliberately `int64`: unsupported
unsigned declarations must be refused, not silently type-erased.

The partial manager acceptance and live workbench have been removed. Complete
String Core Python/native equivalence and timing remain unverified.

Run the explicitly non-native reference baseline with:

```powershell
python -m benchmarks.reference_baseline
```

Measure the tensor preflight across installed AbstractTensor backends with:

```powershell
python -m benchmarks.tensor_preflight_baseline
```

## One-sentence contract

Given an exact atom sequence, return its stable token in a vocabulary; retain
an exact, collision-proof expansion; expose steady-state work as batched
integer operations; and permit multiple exact decompositions and render or
training mipmaps without changing identity.

Turing, Nodus, and Spectral Analyzer contain valuable existing string, token,
glyph, and asset machinery. They are evidence and behavioral oracles for this
project, not runtime dependencies and not code to be casually replaced.
