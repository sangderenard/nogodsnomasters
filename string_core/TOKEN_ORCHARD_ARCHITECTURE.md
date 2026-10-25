# Token Orchard

## A graph-native integer string core, batched service, and compiler target

**Architecture receipt:** 2026-10-24

**Status:** Python reference implemented; live fingerprint kernel verified in C and LLVM

**Repository ownership:** tracked directly by `nogodsnomasters`

**Working name:** Token Orchard

---

## Abstract

Token Orchard treats a character sequence as a permanent fact in a vocabulary,
not as a transient heap object to be scanned, copied, hashed, and compared by
every consumer. At the boundary, encoded text becomes an exact sequence of
Unicode scalar-value atoms. The core interns that sequence, proves whether it
already exists, records an exact graph expansion, and returns a compact integer
token. Steady-state activity then operates on token IDs, integer arrays,
offsets, graph edges, and immutable memory lookups.

A newly observed sequence is first expressed using already known tokens. The
core then mints one token for the whole sequence. Its semantic identity is the
exact atom sequence; its chosen decomposition is an optimization. Later epochs
may attach better decompositions at character, grapheme, subword, word,
phrase, rendering, or neural resolutions without changing identity. These are
the string equivalent of mipmaps: exact alternate views over the same fact.

The project is deliberately small and independent. A restricted-Python
reference defines the laws and is itself the source intended for native
compilation. A dedicated single-writer service may absorb batches from a queue
while readers use immutable snapshots. Turing may eventually bake a closed
vocabulary and only the required faculties into compiled output. Nodus and
Spectral Analyzer are not made owners of the core; their existing token, glyph,
and asset systems are evidence, interoperability targets, and behavioral
oracles.

The objective is not merely faster strings. It is to make character sequences
native to graph computation, tensor batching, deterministic compilation, and
language-model training without surrendering exact textual identity.

Arbitrary bytes form a second, disjoint exact domain. A byte sequence does not
become text merely because it resembles an encoding, and text identity does
not depend on UTF-8 or any other serialization. Strict, versioned encoding
receipts connect byte tokens to text tokens without destroying malformed or
unknown binary input.

---

## 1. Thesis and boundary

Traditional string programming repeatedly rediscovers where a sequence begins
and ends, how long it is, whether another sequence is equal, what pieces it
contains, and how it should be shaped, rendered, embedded, or batched. Token
Orchard moves that discovery to ingress. Once a sequence is known, equality is
normally integer equality, length is a gather, concatenation is interning an
edge sequence, and rendering or neural data attaches to the same graph node.

The core has three inseparable identities:

1. **Semantic store:** exact sequences have collision-proof identity.
2. **Graph machine:** tokens expand through tokens down to scalar atoms.
3. **Batch service:** work enters and leaves as flat integer tensors.

A hash table without the graph is merely an interner. A graph without exact
identity is unsafe. A Python object API without the integer batch ABI cannot
become the intended compiler and tensor substrate.

In scope are exact interning, deterministic minting, decomposition through
existing tokens, immutable snapshots, a batched service loop, exact alternate
covers, native compilation, static vocabulary baking, and stable attachment
points for glyph, search, render, and neural metadata.

The first core is not a text editor, rope, font shaper, renderer, distributed
database, or replacement for Nodus paths. It never makes token numbers portable
without a vocabulary receipt and never lets a neural model decide textual
truth.

The public Python spelling `String` is a non-instantiable namespace, not a
string value class. Calling `String.bytes(...)` or an explicit encoding method
submits ingress to the one process/program vocabulary and returns integer token
IDs. Compiler ingestion recognizes the exact namespace and method identities,
then links the corresponding Token Orchard SSA faculty. An unrelated class or
method spelling is not special. There is no `String` instance ABI.

---

## 2. Mathematical object

Let \(A\) be the valid Unicode scalar values:

\[
A=\{0,\ldots,0x10FFFF\}\setminus\{0xD800,\ldots,0xDFFF\}.
\]

Let \(A^*\) be their finite sequences, including the empty sequence. A
vocabulary at epoch \(e\) is a partial bijection

\[
I_e:S_e\leftrightarrow T_e,
\]

where \(S_e\subset A^*\) and \(T_e\subset\mathbb N_{64}\). Its inverse
\(E_e:T_e\to S_e\) is exact expansion. Therefore

\[
E_e(I_e(s))=s
\]

and, for resident sequences,

\[
I_e(x)=I_e(y)\iff x=y.
\]

That biconditional is the heart of the system. Fingerprints, normalization,
glyph resemblance, neural similarity, and identical rendering do not weaken it.

Let \(B=\{0,\ldots,255\}\). The same vocabulary also carries a disjoint
partial bijection over \(B^*\). Text and byte domains have separate empty and
atom tokens. Even when a one-byte encoding maps byte 65 to scalar 65, those
tokens are not equal.

Every scalar has a canonical atom token. The empty sequence has a canonical
empty token. Atoms are not bytes, UTF-16 code units, grapheme clusters, or font
glyphs. Those are explicit derived views.

A bare `TokenId` is a compact local handle. Portable identity is at least

\[
(VocabularyId,TokenId),
\]

and reproducible frozen-state identity additionally carries
`SnapshotDigest` and `SchemaVersion`. Token IDs are monotonic unsigned 64-bit
values minted by one writer and never recycled. Merging vocabularies produces
an explicit remap.

### No implicit normalization

Canonically equivalent Unicode sequences remain distinct unless an explicit
transformation is requested. Normalization, case folding, locale conversion,
malformed-input replacement, and transliteration return new tokens with
versioned receipts. Compilation, provenance, and training must know when
information changed.

### Encoding is a receipted relation

For encoding policy \(q\), successful strict decoding records

\[
(q,t_{bytes},t_{text}),\qquad decode_q(E(t_{bytes}))=E(t_{text}).
\]

If decoding fails, the byte token remains valid and exact while the text token
is absent. There is no silent replacement character. Encoding performs the
inverse relation when defined. The current reference supports strict UTF-8,
UTF-16LE/BE, UTF-32LE/BE, and Latin-1 adapters.

---

## 3. Identity and collision proof

The core computes a wide fingerprint \(H(s)\), initially at least 128 bits,
and uses \((H(s),|s|)\) to retrieve candidates. It then compares exact atoms
before returning an existing token:

1. validate the atom sequence;
2. compute fingerprint and length;
3. retrieve candidates;
4. compare every candidate's exact expansion;
5. reuse the exactly equal token or mint a new one.

A fingerprint is a candidate generator, never proof. Tests must replace it
with a constant to force total collision. If distinct sequences then alias,
the system never possessed exact identity.

---

## 4. The token graph

Every non-atom token has one or more exact covers by resident tokens. A cover
of sequence \(s\) is

\[
C=(t_0,t_1,\ldots,t_{k-1})
\]

such that

\[
E(t_0)\Vert E(t_1)\Vert\cdots\Vert E(t_{k-1})=s.
\]

The atom cover always exists. A newly minted node's initial cover may use only
tokens present before the mint, so authoritative edges point backward in
creation order and the expansion graph is acyclic.

The reference cover is selected by the lexicographic minimum

\[
C_e(s)=\arg\min_C
\left(|C|,\sum_i levelPenalty(t_i),(t_0,\ldots,t_{k-1})\right).
\]

The final token vector gives a total deterministic tie-break. A prefix index
enumerates matches and dynamic programming finds the minimum cover. Optimized
search must return the same result for the same snapshot and policy version.

Crucially, identity is \(s\), not its cover. A later epoch may find

\[
C_{e+1}(s)\ne C_e(s),\qquad E(C_{e+1}(s))=E(C_e(s))=s,
\]

without changing the token. This is what keeps compiler constants, caches,
render attachments, and embeddings stable.

Reverse parent indexes, occurrence positions, prefixes, suffixes, transforms,
imports, and consumer attachments may be derived. Exact expansion and identity
remain authoritative.

---

## 5. String mipmaps

A mip is an alternate exact cover optimized for a named consumer and scale:
scalar atoms, short fragments, grapheme clusters, learned subwords, words,
phrases, shaped glyph runs, or render assets at several resolutions.

For node \(t\), mip \(m\) is

\[
M_m(t)=(u_0,\ldots,u_j),\qquad
E(u_0)\Vert\cdots\Vert E(u_j)=E(t).
\]

Each record names its policy, policy version, source snapshot, cost, and
validation status. Grapheme rules and learned tokenization evolve; versioned
mips let them change while textual identity does not.

A consumer-specific cost may be

\[
J_m(C)=\alpha_m|C|+\beta_m MissCost(C)+
\gamma_m BoundaryPenalty(C)+\delta_m TransferBytes(C).
\]

Heuristics and neural models may propose candidates. The exact validator alone
publishes them.

---

## 6. Authoritative storage

Storage is structure-of-arrays, never one authoritative object per token.

| Array | Element | Meaning |
|---|---:|---|
| `token_kind` | `u8` | empty, atom, aggregate, reserved |
| `atom_value` | `u32` | scalar for atoms, sentinel otherwise |
| `atom_length` | `u64` | expanded scalar count |
| `fingerprint_hi/lo` | `u64` | fingerprint words |
| `cover_begin` | `u64` | offset in `cover_tokens` |
| `cover_count` | `u32` | canonical-cover length |
| `created_epoch` | `u64` | mint epoch |
| `cover_tokens` | `u64` | flattened cover edges |
| `mip_begin/count` | `u64/u32` | attached mip descriptors |

The reference implementation uses contiguous standard-library typed arrays,
so the authoritative fields expose the buffer protocol and do not box every
integer as a Python object. Native and tensor consumers can take typed views;
portable serialization still declares byte order explicitly.

Hot collision checks may use flattened atom pages or checkpoints rather than
recursive graph walks. Such pages are materialized views whose digests agree
with the graph.

Readers observe immutable epoch \(e\). The writer constructs \(e+1\), validates
indexes and arrays, then atomically publishes its descriptor. Readers pinned to
\(e\) finish there; reclamation waits. Portable snapshots are deterministic and
little-endian, with magic, schema, vocabulary identity, epoch, aligned section
descriptors, policy versions, section digests, and whole-snapshot digest. They
contain no process pointer or Python pickle.

### Reclamation is generational

Token IDs remain monotonic and exact expansion remains stable for the entire
lifetime of one vocabulary identity. Consequently, inactivity cannot turn a
live token into a tombstone and cannot permit its number to name new content.
Retention instead uses aggregate-row-aligned `last_notice_tick`,
`notice_count`, and `protected` arrays. Notices are flat integer batches;
staleness is selected in parallel from an explicit logical tick. Wall-clock
sampling is neither deterministic enough nor part of the ABI.

A reclaiming writer waits until no reader pins the old published epochs, then
constructs a new vocabulary identity from retained exact sequences and emits
an old-token/new-token remap. Only then may the old arenas, derived glyph
sets, mips, and embedding rows be released. Reappearance is a reinternment in
the new vocabulary, never reuse inside the old one.

The semantic carry policy is receipted independently: cold initialization,
retention of the mapped learned row, or conditioned removal followed by
retraining/replay. “Conditioned out” is not approximated by zeroing a vector:
the learner must possess sufficient observation/checkpoint provenance to
remove the retired node's contribution. Until that exists, the policy is a
declared unsupported phase rather than a silent heuristic.

---

## 7. Integer batch ABI

The public boundary is versioned and integer-only. Encoding adapters produce
scalar values before submission.

```text
request_opcode : u16[R]    response_id     : u64[R]
request_flags  : u16[R]    status          : u16[R]
request_id     : u64[R]    result_begin    : u64[R]
arg_begin      : u64[R]    result_count    : u32[R]
arg_count      : u32[R]    result_values   : u64[M]
arg_values     : u64[N]    published_epoch : u64[R]
```

Initial operations intern atom or token sequences, concatenate, return length
or an atom, slice by scalar offsets, compare, find one expansion in another,
fetch a cover or mip, and pin or release an epoch. Each opcode states whether
it may mint, what request order it observes, and how errors are encoded. No
exception object crosses the ABI.

The first executable SSA slice of this contract is version zero's
`token_kind_batch(tokens: AbstractTensor, aggregate_count: int)`. Its separate
ProgramABI declaration fixes `tokens` as a runtime-sized rank-one `int64` span
and `aggregate_count` as a scalar `int64`; the output is a derived rank-one
`int64` span. The annotation is postponed and its import is guarded by
`TYPE_CHECKING`, following Turing's authored-parameter convention: Python runs
the same function with a real AbstractTensor value, while compiler ingestion
uses the syntax as the span contract. Both native artifacts publish
`turing-runtime-abi-v1` and derives output extents from the input buffer.

Turing PR #923 made runtime-sized signed-integer elementwise arithmetic
contract-correct in LLVM. The exact authored AbstractTensor implementation now
passes the C and LLVM gates without a second native implementation. The proven
physical ProgramABI remains signed `int64`; Token Orchard's logical unsigned-ID
contract does not authorize silent native type erasure, so an unsupported
unsigned declaration must remain a compiler refusal until that lane exists.

Domain is an explicit integer argument. Identically valued text scalars and
bytes cannot alias. Encoding adapters live at the boundary and return integer
receipts naming the encoding, byte token, text token, and status.

---

## 8. Queue and concurrency

Token minting has one deterministic serialization point:

```text
producers -> bounded MPSC queue -> single writer/batcher
                                      |
                                      v
                              immutable epoch
                                      |
readers  <-------- pinned snapshots --+
```

The writer drains a bounded request or atom budget, executes declared batch
order, deduplicates identical requests, mints monotonically, and publishes one
epoch. Read-only work can bypass the mutation queue against a pinned snapshot.

The implemented writer greedily fuses consecutive same-domain requests under
request, item, and atom limits. It preserves FIFO item order and splits the
result back into the original receipt boundaries. Compatible items are then
bucketed by sequence length: validation predicates and both fingerprint lanes
run across each bucket through `AbstractTensor`, while exact comparison, cover
selection, and mint commits remain serialized. This is the intended split
between parallel numerical screening and order-dependent graph semantics.

The queue has bounded memory, explicit backpressure, idempotent request IDs,
deterministic ordering, cancellation only before execution, size limits, no
partial visibility, and shutdown that reports every accepted request. “Own
thread” is an operating mode, not a semantic dependency: embedded compiled
programs may call the same executor synchronously.

---

## 9. Tensorization and training

Ragged data is flat values plus offsets:

\[
sequence_i=V[O_i:O_{i+1}].
\]

Fingerprint accumulation, length gathers, candidate buckets, equality
screening, cover costs, and read-only traversal can be batched. Mutation order
and exact final comparison remain authoritative. GPU transfer is justified only
when batch size repays it, and every optimized path must match the scalar
reference for the same snapshot and policy.

The first live tensor lane uses the same annotated limb functions as the
scalar reference and native compile gate; it is not a second hash algorithm.
Measurements on eager NumPy, Torch, and Nodus backends currently show dispatch
and materialization costs outweighing the vector work even at 16,384 rows of
16 atoms. Therefore “tensorized” is a representation and execution fact, not
yet a speed claim. The next compiler gate is the whole reachable string core as
one compilation product: tensor screening, exact lookup, indexing, cover work,
ordered commits, publication, and batch results. The fingerprint-only native
gate remains a diagnostic oracle, not a runtime partition. Backend fusion and
vectorization belong inside whole-program lowering, where the compiler can
remove intermediate materialization and the linker can retain only faculties
reachable from the consuming program.

Token IDs are categorical labels, not ordinal magnitudes. Training uses a
versioned dense row map

\[
R_e:T_e\to\{0,\ldots,|T_e|-1\}
\]

and records vocabulary, snapshot, schema, token, row, and mip policy. Useful
objectives include parent-from-cover, cover-from-parent, mip choice under cost,
cache warming, and context contrast. Model output is always a proposal subject
to deterministic exact validation.

The first executable learning attachment uses a deliberately narrow online
objective. After an ordered writer commit, the service may copy the newly
minted rows into a flat delta:

```text
begin_epoch, end_epoch
parent_tokens[P]
cover_values[E], cover_offsets[P+1]
```

A separate bounded worker maintains the versioned dense row map. Every token
has a fixed deterministic positional encoding \(q_t\) on the
\((D-1)\)-sphere. Turing's established `AbstractNN` supplies a shared residual
network \(F_\theta:\mathbb R^D\to\mathbb R^D\), concretely
`Linear(D,H) -> Tanh -> Linear(H,D)`. Position and meaning remain separate:

\[
\lVert q_t\rVert_2=1,
\qquad r_t=F_\theta(q_t),
\qquad z_t=q_t+r_t.
\]

The output layer begins at zero, so the initial residual is exactly zero. For
parent \(p\) with canonical cover \(C(p)\), a flat AbstractTensor
gather/scatter batch retains each COO edge ordinal and parent cover length.
Let \(a(i,k)\in[0.5,1.5)^D\) be a fixed versioned channel modulation derived
with bounded integer arithmetic from edge ordinal \(i\), cover length \(k\),
and channel. The batch builds

\[
\mu(p)=\frac{1}{|C(p)|}\sum_i a(i,|C(p)|)\odot z_{C(p)_i},
\qquad y_p=\mu(p)-q_p,
\qquad
\mathcal L(\theta)=\frac1P\sum_p\lVert F_\theta(q_p)-y_p\rVert_2^2.
\]

Thus `ab` and `ba` do not collapse to the same training target. This is an
ordered compositional signal, not a claim that the modulation itself is a
complete syntax model.

The repository's `Adam` updates \(\theta\); no local optimizer equation is
maintained. After each step, one batched network forward refreshes residuals
and meaning vectors for every resident row, so cached rows cannot represent an
older parameter state.

All parents in the delta update together. Python manages only queue admission,
the categorical token-to-row map, and flat edge topology; vector arithmetic is
AbstractTensor work. The writer never waits for training. A full queue, an
epoch gap, or a sink failure marks the attachment desynchronized and cannot
alter the authoritative commit or its receipt.

The worker also accepts a version-zero integer-only observation batch:

```text
begin_sequence
kind[E], subject[E], object[E], context[E]
position[E], extent[E], flags[E], weight_q16[E]
```

Kinds cover adjacency, containment, program roles, typed edges,
definition/reference relations, operations, transforms, glyph relations,
outcomes, and temporal relations. Subject is the token being learned, object
is the related token, context optionally names scope/program/type/operation,
and position is a signed ordinal or displacement. Relation, position, extent,
polarity, and weight modulate object/context meaning vectors before one COO
scatter produces per-subject targets. Contiguous observation sequence numbers
make loss and replay gaps explicit; a gap desynchronizes the attachment.

This policy is a plumbing and batching proof, not a claim that the present
ordered and context objectives are sufficient semantic embeddings or that
this small residual MLP is the final shared projector. Negative sampling
beyond explicit negative facts, a context-conditioned projector, held-out
evaluation, durable optimizer persistence, snapshot resynchronization, and
learned mip proposals remain separate gates.
Observation batches are presently consumed online without replay; held-out
prediction and an anti-forgetting schedule are required before semantic
quality claims.

### Optional semantic compilation depth

The semantic product is a strict build-time prefix, never a runtime tagged
union:

```text
0 exact string core only
1 fixed spherical positions
2 ordered COO composition
3 integer live-context reduction
4 AbstractNN forward + VJP + Adam state
5 immutable embedding epoch publication
6 batched top-k/token proposal query
```

Each depth reserves one named compilation root and adds only its required state.
`semantic_state_abi(depth, dimension)` is the inspectable state receipt: Adam
moments and parameters first appear at depth four, published vectors at five,
and query buffers at six. The preparation roots for depths one through three
are implemented as ordinary AbstractTensor source functions. Depth four uses
Turing's existing network, graph reverse, and Adam mechanisms; it must not gain
a local optimizer or backward implementation. Native packaging of that phase
waits on the recorded generated-VJP source-provenance repair. Publication and
query are contracted next phases, not falsely claimed native products.
`semantic_program_abi(depth)` supplies Turing's physical span/scalar contracts
for implemented roots only and refuses unimplemented training depth or above.

These observations are intentionally source-neutral. A compiler, live
program, decompiler, glyph manufacturer, or training ingestor can emit the
same integer facts. Learned state can support retrieval, prediction, mip
proposals, cache scheduling, source navigation, and later programmatic
reasoning, but it never changes token identity or exact expansion.

---

## 10. Restricted Python as executable truth

The first implementation uses compiler-friendly Python: integer arithmetic,
flat numeric arrays, counted loops, explicit branches and statuses, no
reflection, no host-string work inside core algorithms, no object graph as
authoritative storage, and no dictionary iteration order as semantics.

It is not throwaway pseudocode. The same algorithm is intended to lower through
Turing's sanctioned public entry:

`src.compiler.fortran_c_shell.lower_ast_source_to_ssa`

Compilation starts only after the scalar laws pass. Backends are compared with
the Python oracle. An unsupported construct is understood or recorded as a
compiler gap; it does not justify a parallel hand-written implementation.

The first compiled slice is not a demonstration duplicate: `core.py` calls the
same `fingerprint_atom_step` in `kernel.py` that the compile gate reads and
lowers. Two `u32` limbs per FNV-1a lane bound every intermediate below signed
`i64`; C and LLVM both compile and return exact Python-oracle limbs for the
boundary corpus. This proves that shared arithmetic slice, not yet the whole
vocabulary or service.

---

## 11. Compiler bake-in

For program \(P\), a compiler can:

1. collect identifiers, literals, field names, diagnostics, and labels;
2. intern them in deterministic source order;
3. compute the transitive cover closure;
4. serialize the minimal initial snapshot;
5. replace static strings with token constants;
6. determine reachable faculties;
7. emit only required arrays and operations;
8. link the dynamic writer only if runtime interning is reachable.

A closed program may need only equality, length, and concatenation. It should
not carry normalization tables, substring search, glyph metadata, or a service
thread. Token Orchard is a compiler target, not a mandatory runtime monolith.

Ordinary Python syntax may select this component without constructing a
`String` namespace call, but only through exact compiler evidence. Text and
byte literals, same-domain addition, length, indexing, slicing, equality, and
explicit encoding bridges can emit a component receipt when receiver/value
class and operation identity are proven. Turing's incumbent source glyph,
StringTable projection, explicit UTF-8 encode, and byte-join facts are the join
surface; there is no parallel spelling recognizer and no live monkeypatch.

Recognizing syntax does not make an operation implemented. Strip, replace,
join, casing, classification, and other transforms require exact batched
opcodes, Unicode/policy versions, and transformation receipts before capture
may replace Python semantics. Lifted CPython Unicode helpers are differential
oracles, not a native libpython dependency and not semantic proof for a wider
method.

The eventual C-compatible boundary exposes opaque vocabulary and snapshot
handles plus batches over explicitly owned buffers. It specifies widths,
endianness, alignment, lifetime, statuses, thread safety, versions, limits,
pin/release, and merge remaps. It avoids per-token allocation across the ABI.

---

## 12. Evidence and coexistence

The neighboring repositories demonstrate parts of this shape:

- Turing has whole-string tokens, character-token sequences, resident sequence
  materialization, integer string operations, and reversible integer encoding
  for training data.
- Nodus converts text through code points, shaping clusters, glyphs, graph
  edges, and token IDs; it demonstrates graph-shaped text and the importance of
  explicit deduplication.
- Spectral Analyzer manufactures token render assets, catalogs stable identity,
  composes cached token strings, and falls back from whole sequences through
  words to characters.

These are evidence and later differential oracles, not implementation
dependencies. Integration begins with adapters and measurements, never an
unrequested rewrite of a working neighboring system. The detailed evidence is
in `speaktome/AGENTS/experience_reports/1792849246_AUDIT_Turing_Nodus_Spectral_Glyph_Token_Architecture.md`.

---

## 13. Safety and exact laws

Interning is potentially permanent allocation. The service needs sequence,
batch, vocabulary, byte, and caller budgets; offset validation; explicit ID
exhaustion; recursion-free or bounded expansion; import cycle rejection;
snapshot validation; and no hidden malformed-encoding policy. Empty, enormous,
repetitive, shared-prefix, collision-heavy, and cover-ambiguous inputs are
ordinary tests.

For valid \(x,y,z\), the executable laws are:

1. `expand(intern(x)) == x`.
2. Repeated `intern(x)` returns the same token.
3. `intern(x) == intern(y)` iff `x == y`.
4. `expand(concat(tx, ty)) == x || y`.
5. `length(intern(x)) == len(x)` in scalar atoms.
6. A valid slice expands to the exact scalar subsequence.
7. Every cover and mip expands exactly to its node.
8. Unrelated later mints never change old IDs or expansions.
9. Snapshot serialization preserves authoritative identity and arrays.
10. Collapsing the fingerprint cannot change semantic results.
11. Batch execution equals its declared scalar sequence.

Determinism tests repeat histories in fresh processes, vary producer scheduling
while fixing accepted order, reload and continue snapshots, compare compiled
backends, and verify caches cannot change reference covers. Concurrency tests
cover same-sequence races, old readers during publication, backpressure,
shutdown accounting, cancellation, and snapshot reclamation.

Performance receipts report distributions and memory: atoms and requests per
second, unique mints, hit/miss latency by batch, collision verification, cover
search by vocabulary density, publish pause, resident bytes, reader throughput,
and reference/backend parity. Every result names hardware, build, schema,
policy, snapshot digest, input distribution, warm-up, samples, and cache state.

---

## 14. Development sequence

### Phase 0 — Make the contract executable

Freeze integer widths, empty and atom rules, statuses, batch ordering, snapshot
header, limits, and law tests. No optimization precedes this receipt.

**Implemented in the Python reference.**

### Phase 1 — Scalar reference

Implement scalar interning, forced-collision verification, deterministic
existing-token covers, aggregate minting, expansion, length, concatenation, and
slicing in restricted Python. The implemented reference also includes the
parallel exact byte domain and strict encoding receipts. **Implemented.**

### Phase 2 — Arrays and batches

Move authoritative state to flat arrays; implement opcodes, scalar/batch parity,
deterministic snapshots, reload, and continuation.

**Partially implemented:** typed flat arrays, ordered batch interning, and
validated deterministic snapshots are present. The full opcode surface is not.

### Phase 3 — Native compilation

Lower the reference through the sanctioned Turing entry. Establish Python, C,
LLVM, and Fortran parity where supported. Record gaps instead of forking laws.

**Started:** the live fingerprint atom-step kernel has exact Python/C/LLVM
compile-and-execute parity. The live whole-core graph is now a durable lowering
audit and reaches deployment planning. Its single current refusal is dynamic
persistent arena growth without explicit native pointer ownership/capacity.
The next implementation is therefore a capacity-bearing vocabulary-state ABI,
not another isolated kernel. Whole-core emission and Fortran parity remain
open.

### Phase 4 — Queue service

Add the bounded producer queue, one writer, epoch publication, pinned readers,
backpressure, shutdown, and concurrency stress.

**Prototype implemented:** bounded admission, one writer, grouped drains,
greedy compatible-request fusion into flat COO-style buffers, one padded
AbstractTensor validation/fingerprinting batch with masked lanes,
request receipts, backpressure, and clean shutdown are tested. Immutable pinned
reader publication, whole-core compilation/linkage, and cancellation remain
open.

### Phase 5 — Mipmap substrate

Add versioned alternate covers and exact validation. Start with scalar,
canonical, and one deterministic derived level before learned or render mips.

### Phase 6 — Compiler bake-in

Collect static strings, bake a minimal snapshot, substitute constants,
eliminate unreachable faculties, and link mutation only when necessary.

### Phase 7 — Training and adapters

Add dense row maps and training receipts. Build differential adapters for
Turing, Nodus, and Spectral Analyzer only under explicit integration tasks.

---

## 15. First milestone acceptance gate

The scalar milestone is complete only when:

- valid scalar sequences round-trip exactly;
- concurrent identical requests mint one token;
- distinct sequences survive forced total hash collision;
- each aggregate cover uses only prior tokens and expands exactly;
- IDs and expansions survive deterministic snapshot reload;
- scalar and batch paths return identical statuses and values;
- invalid scalars, offsets, sizes, and exhaustion return defined errors;
- no host-string operation occurs after ingress;
- memory and throughput baselines are recorded without inflated claims.

Native compilation, threading, mipmaps, and integration are separate gates.

### Decisions fixed by this receipt

- atoms are Unicode scalar values;
- arbitrary bytes occupy a separate exact domain;
- encoding is an explicit strict relation between byte and text tokens;
- normalization is explicit;
- handles are monotonic `u64` and never recycled;
- portable references carry vocabulary and snapshot identity;
- fingerprints never prove equality;
- one writer orders mutation and readers see immutable epochs;
- identity is independent of covers and mips;
- the ABI is integer-only and versioned;
- the Python reference is intended to compile;
- neural output is advisory;
- neighboring repositories are not runtime dependencies.

The fingerprint algorithm, reserved ID ranges, snapshot alignment, prefix-index
structure, admission thresholds, materialization thresholds, and precise native
ownership ABI remain Phase 0 engineering decisions. Measurement may choose
among them but may not change the laws above.

---

## Closing statement

The smallest honest description of Token Orchard is: **a machine that remembers
every exact character sequence once, gives that fact an integer name, and lets
the rest of the system compute on the name and its graph.**

Atoms preserve truth. Tokens provide identity. Covers provide structure.
Mipmaps provide scale. Arrays provide locality. Batches provide throughput.
Snapshots provide concurrency and reproducibility. Compilation removes unused
machinery. Neural systems learn from the graph without becoming its authority.

If those separations survive implementation, compiler constants, graph labels,
glyph manufacture, render caches, tensor batches, training corpora, and services
can share one exact substrate without sharing one monolithic application. The
groundwork is now explicit. The next task is to make the laws executable.
