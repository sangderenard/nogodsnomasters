# Token Orchard implementation receipt

**Date:** 2026-10-24

**Implemented scope:** Phase 0, scalar Phase 1, selected Phase 2 snapshot work,
and a narrow Phase 4 queue prototype

## Delivered

- Exact, disjoint Unicode-scalar and arbitrary-byte token domains.
- Separate canonical empty tokens and arithmetic atom-token ranges.
- Monotonic `u64` aggregate IDs without preallocating scalar objects.
- Exact candidate comparison after fingerprint lookup.
- Shared signed-`i64`-safe, two-lane FNV-1a kernel using `u32` limbs.
- Constant-fingerprint mode proving collision independence.
- Deterministic minimum-piece, minimum-atom, token-ID-tied covers using only
  prior tokens.
- Contiguous typed structure-of-arrays storage through Python's buffer protocol.
- Flat values-plus-offsets batch interning with declared ordered semantics.
- Length, atom-at, concatenation, and scalar-offset slicing.
- Strict UTF-8, UTF-16LE/BE, UTF-32LE/BE, and Latin-1 boundary adapters.
- Receipts linking byte tokens and text tokens; malformed encoded bytes remain
  preserved without replacement text.
- Deterministic little-endian snapshots with SHA-256 integrity, complete
  structural revalidation, index rebuilding, and deterministic continuation.
- Request-count-bounded multi-producer admission to one writer thread, grouped
  drains, request IDs, queue-full status, clean shutdown, and post-close state
  access.
- Greedy FIFO fusion of consecutive same-domain requests under explicit item
  and atom budgets, with exact receipt-boundary reconstruction.
- One padded AbstractTensor execution over each fused COO-style batch for
  parallel atom predicates and masked fingerprint lanes, reusing the exact
  scalar/native limb functions without per-length Python buckets.
- Span-based scalar oracle internals, so differential execution does not copy
  each item slice before validation, hashing, exact comparison, cover search,
  and ordered commit.
- Fixed-extent result and fusion buffers where batch surveys already prove
  their sizes; no append/extend growth in those paths.
- A version-zero SSA query surface whose canonical authored function uses the
  repository's `TYPE_CHECKING`-guarded `AbstractTensor` annotation convention.
- A centralized Turing ProgramABI contract for `token_kind_batch`: rank-one
  runtime-sized `int64` input span, scalar `int64` aggregate count, and derived
  rank-one `int64` output span.
- C compile-and-execute proof for that SSA API, including validation of the
  normalized compiler contract and exported `turing-runtime-abi-v1` receipt.
- A detachable continuous embedding learner fed only after ordered mint
  commits, through flat parent/cover/offset graph deltas.
- Versioned dense token-row snapshots and batched AbstractTensor gather,
  scatter-sum, cover-target assembly, AbstractNN inference, and Adam training
  on a separate bounded worker queue. Fixed n-dimensional unit-sphere
  positions, shared AbstractNN residuals, and resulting meaning vectors are
  published as distinct fields, together with optimizer-step and latest-loss
  receipts.
- Ordered COO cover targets carry edge ordinal and cover length, so reversed
  covers no longer collapse to the same mean. A separate version-zero,
  integer-only observation ABI batches adjacency, containment, role, typed
  edge, definition/reference, operation, transform, glyph, outcome, and
  temporal facts with context, position, extent, polarity, and fixed-point
  weight.
- Optional semantic compilation is a strict depth prefix: position, ordered
  composition, context, training, publication, query. The state ABI adds model
  parameters and Adam moments only at training depth. The first three
  AbstractTensor preparation roots are implemented and eager tested with a
  depth-pruned Turing ProgramABI; no native artifact is claimed past the
  existing compiler gate.
- Explicit desynchronization on queue loss or epoch gaps, with tests proving a
  failed advisory sink cannot fail or change authoritative interning.
- A non-instantiable live Python `String` namespace. Bytes-like ingress flows
  through one lazy process-global `TokenService`; results are integer tokens or
  integer status/encoding/token receipts. Host text and instance construction
  are rejected. Exact method identities, opcodes, and semantic depths are
  versioned for compiler ingestion without compiling Python queue machinery.
- A generational retention contract with flat integer notice batches,
  aggregate-row-aligned lifecycle state, three explicit semantic carry
  policies, and a vectorized AbstractTensor stale-row mask. It deliberately
  does not delete from a live vocabulary or recycle a token ID; actual arena
  reclamation waits for immutable epoch pins and a new vocabulary identity.
- A focused Turing source-pursuit audit for the shim. It proves the compiler
  currently pursues `_byte_values`, `_one_token`, and `String.bytes`, then
  reaches `token_orchard.service.TokenService` with
  `default:unknown / provenance_not_declared` instead of a component receipt.
- An exact Python source-capture manifest. Reachable literals, concatenation,
  length, indexing, slicing, equality, and explicit encoding bridges can
  require one Token Orchard component. Existing Turing glyph/StringTable,
  strip, UTF-8 encode, and byte-join seams are recorded; unimplemented
  transforms are refused rather than inferred from names.
- A profiled text-edit learning demonstration using the real writer and
  learner queues, AbstractNN reverse, and Adam. Its receipt explicitly says
  native training is unavailable while the generated `forward_vjp` lacks
  source-cell provenance.

## Verification

Executed from `C:\dev\Powershell\string_core`:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q token_orchard tests demos benchmarks tools
python tools/test_turing_compile.py
python tools/test_turing_ssa_api.py --backend all
python tools/test_turing_retention.py --backend all
python tools/audit_turing_string_shim.py
python tools/audit_turing_python_string_seams.py
python demos/profiled_text_learning.py
python -m benchmarks.reference_baseline
python -m benchmarks.tensor_preflight_baseline
```

Result: **72 tests passed** in the final run; bytecode compilation succeeded.
The retention mask's exact AbstractTensor source and integer-only ProgramABI
also compiled through Turing's C and LLVM backends and executed **4/4 rows**
exactly in each.
The canonical Turing source entry lowered the exact live
`fingerprint_atom_step` implementation. Turing's C and LLVM backends each
compiled it and executed **4/4 exact limb-pair cases**, including zero, maximum
`u32` limbs, the maximum Unicode scalar, and a supplementary-plane scalar.
The gate exited successfully and retains generated products only beneath the
ignored `build/turing_compile_gate/` directory.

The first actual SSA-linkable Token Orchard query now compiles independently
of the scalar fingerprint proof. `token_kind_batch` is submitted from its real
source, not through an import-only surrogate. Its normal-Python signature uses
a postponed `AbstractTensor` annotation imported only under `TYPE_CHECKING`;
Turing consumes that exact spelling as a tensor-span declaration. The separate
`ssa_program_abi()` contract fixes physical widths and rank. The gate verifies
that the compiler-normalized parameter ABI agrees with that declaration, that
the returned SSA value retains rank-one span accounting, and that each emitted
runtime ABI derives the output extent from the input span. C and LLVM each
compiled and executed `[0, 1, 3, aggregate+0, aggregate+1, aggregate+2]` as
`[invalid, empty, atom, aggregate, aggregate, invalid]` exactly.

Turing PR #923 repaired the runtime-sized signed-integer LLVM path. The query
and retention gates now prove the same authored code in C and LLVM. The native
ABI remains explicitly signed `int64`; logical unsigned Token Orchard IDs do
not permit a compiler to erase signedness silently. Unsigned native
declarations remain a distinct compiler frontier and must be refused until
implemented.

The exact AbstractNN capture mechanism used by the new adapter was checked
against Turing's `test_captured_training_step.py`: all four graph/Adam capture
laws pass. Native graph-reverse packaging is currently stale relative to
Turing's source-provenance concordance and fails before LLVM emission:

```text
function_address: no source cells for
'engine_dendrite_learner__forward_vjp' -- the caller named no source definition
```

This is not specific to Adam: the plain parametric multiply VJP, independent
parameter-group schedule, and both scalar-loss adjoint variants fail at the
same `lower_training_motion_to_repository_ssa` declaration. That is retained
as a compiler-team handoff rather than patched or bypassed in Token Orchard.
The older whole-program XOR native test is explicitly skipped by Turing
because its capture is not execution-bounded. Therefore the adapter is proven
to train through AbstractNN/Adam eagerly, and those components have a captured
training graph, but this exact adapter does not yet claim a native artifact.

A second audit now submits a thin top-level wrapper around the real
`Vocabulary.intern_batch`, declares the exact Token Orchard sources to the
extraction contract, and asks the same sanctioned Turing entry to lower the
whole reachable graph. During this audit, compiler-oriented but
semantics-preserving changes made the integer ABI explicit, removed optional
tuple state, replaced batch-output growth with fixed-size indexed stores, and
made cover selection a total scalar transition. All 43 laws stayed green.
The audit advances through source pursuit, specialization, class-surface ABI
settlement, and deployment planning before one remaining refusal:

```text
dynamic-growth-unavailable: dynamic sequence growth requires typed scalar
native pointer cells, an explicit heap owner, and support for arena replacement
```

That refusal describes the persistent vocabulary arrays grown by aggregate
minting. It is the correct next boundary: define capacity-bearing owned native
arenas and pass their counts/capacities through the program ABI. The audit is
retained at `tools/audit_turing_whole_core.py` and deliberately exits nonzero
until holistic lowering succeeds; it does not substitute a reduced algorithm.

The padded AbstractTensor preflight was remeasured after removing per-length
Python buckets. At 16,384 strings of 16 atoms, the observed eager ratios against
the scalar oracle were **1.04x NumPy**, **0.91x Nodus**, and **0.79x Torch**.
This is a single-run crossover receipt, not a native-performance claim; the
compiled graph remains the intended execution mode.

An attempted fully tensor-resident COO-to-padding conversion was rejected after
measurement. The existing `AbstractTensor.searchsorted` is correctly defined in
portable base operations as broadcast comparison plus reduction; using it to
derive every atom's row therefore changed packing from O(atoms) to
O(items*atoms), reaching roughly 53 seconds for the largest NumPy case. The
implemented boundary keeps one O(atoms) topology-only host pack. The eventual
zero-copy integer ABI should carry COO row/column coordinates directly rather
than asking the tensor graph to rediscover them from offsets.

The optional learner is intentionally a surface implementation, not a trained
general reasoner. For each committed graph delta or live observation batch,
Python performs only queue and categorical row-map/topology work.
AbstractTensor deterministically constructs and row-normalizes token
coordinates onto the n-dimensional unit sphere, then gathers and scatter-sums
ordered child or typed context meaning rows into targets. Turing's
ordinary `Linear(D,H) -> Tanh -> Linear(H,D)` AbstractNN predicts residuals and
its established Adam implementation trains the shared parameters. The final
layer is zero-initialized, and every optimizer step is followed by one batched
forward over all rows so cached embeddings share one parameter epoch. Its
version-three schema / policy four keeps \(q_t\), \(r_t=F_\theta(q_t)\), and
\(z_t=q_t+r_t\) distinct. Token numbers seed fixed positions but are not
compared as distances or used as training targets. Observation cursors make
loss explicit, and a position-only learner proves the network and optimizer
are not constructed below training depth. Negative sampling beyond explicit
negative facts, a context-conditioned projector, held-out evaluation, reverse
child updates, durable optimizer persistence, snapshot resynchronization, and
native packaging remain future work. Model output has no path to mint IDs or
modify exact expansions and covers.

The final deterministic reference benchmark used Python 3.11.7 on Windows 10
build 26300, with 2,000 unique 16-atom printable-ASCII sequences:

```text
items                  2,000
atoms                 32,000
elapsed             1.7567 s
items/s             1,138.5
atoms/s            18,216.4
traced peak bytes  1,253,368
```

This is explicitly a restricted-Python reference baseline, not a native or
end-to-end service-performance claim. An earlier boxed-list run peaked at
1,582,280 traced bytes; typed arrays reduced that measured peak by about 22%.
Timing differences between short runs are not treated as significant. The
current limb kernel deliberately exchanges Python scalar speed for bounded,
portable compiler arithmetic; optimization belongs after whole-law parity.

One failed command is retained for reproducibility:

```powershell
python benchmarks/reference_baseline.py
```

It failed with `ModuleNotFoundError: token_orchard` because direct script
execution roots imports in `benchmarks/`. The supported invocation is the
module command shown above; `benchmarks/__init__.py` makes that explicit.

## Known limits

- The shared fingerprint atom step, batched token-kind query, and vectorized
  retention mask have C/LLVM native parity with exported runtime-sized SSA
  ABIs. The compiler now
  ingests the live vocabulary, exact candidate proof, graph cover, and ordered
  batch path together, but native emission is blocked at persistent arena
  growth; snapshots, encoding bridges, and the queue remain Python-only.
- Fortran parity has not yet been attempted; C and LLVM are the measured lanes.
- Typed buffers and batches are now mixed: queue validation and fingerprinting
  have a real AbstractTensor lane, while exact comparison, cover search, and
  commits remain scalar. No SIMD/GPU speed claim is made.
- Eager AbstractTensor results are backend- and batch-size-sensitive. The
  padded whole-batch form crossed the scalar oracle only for NumPy at the
  largest recorded batch; graph/kernel fusion inside whole-core compilation is
  still the performance target. A separately dispatched fingerprint
  accelerator is explicitly not the target.
- The derived prefix index groups by first atom and scans that bucket. It is
  deterministic but not yet the compact trie/automaton intended for scale.
- Snapshot vocabulary identity is not yet specified beyond exact content and
  digest; cross-vocabulary token references still require a later contract.
- The service reports grouped completion epochs but does not yet publish
  immutable pinned reader views. Cancellation and idempotent retries remain
  open.
- Full batch opcodes for every read/query operation are not implemented.
- The flat ingress ABI carries values and offsets, but not precomputed COO row
  and column coordinates. Padding therefore has one linear host topology pack;
  all atom predicates and fingerprint arithmetic remain AbstractTensor work.
- Queue slots are bounded, but queued payload atoms do not yet reserve and
  release a global credit budget; strict total-memory boundedness remains open.
- Encoding adapters intentionally use host codecs at the boundary. No host
  string is used by the authoritative core.
- Exact mipmaps, learned proposals, embedding persistence/resynchronization,
  glyph attachments, and neighboring-repository adapters remain future phases.
- Observation batches train online and are not replayed. The current learner
  has no held-out evaluator or anti-forgetting schedule, so its vectors are an
  evolving advisory index rather than evidence of durable learned reasoning.
- Turing's eager autograd tape is process-global. Token Orchard serializes its
  own model creation, training, and refresh sessions, but independent eager
  trainers in the same process do not yet share that lock. Native captured
  training removes this hosting limitation once the compiler provenance
  regression above is repaired.

## Interpretation

This is an executable semantic foundation with a scalar cross-backend seam and
a first real batched SSA/C query boundary, not the finished native string
machine. Its strongest result is that
text, arbitrary bytes, encoding transforms, aggregate identity, graph covers,
storage layout, persistence, and queued mutation meet one tested integer
contract—and the fingerprint step used by that contract crosses Python, C,
and LLVM without a second implementation. That same limb code now also runs
as vector lanes through AbstractTensor; performance honesty requires fusing
those operations during whole-core lowering rather than promoting the partial
gate as an independent high-throughput service.
