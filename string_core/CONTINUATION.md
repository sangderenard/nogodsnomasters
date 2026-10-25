# Token Orchard continuation prompt

Copy the prompt below into the next implementation task. It begins from the
implemented reference rather than repeating Phase 0/1.

---

Continue **Token Orchard** in `C:\dev\Powershell\string_core`, tracked by the
root `nogodsnomasters` repository.

Read the root and `speaktome` agent guidance, relevant experience reports, and
then every Markdown file in `string_core/`, especially `AGENTS.md`,
`PHASE_0_CONTRACT.md`, `TOKEN_ORCHARD_ARCHITECTURE.md`, and
`IMPLEMENTATION_RECEIPT.md`. Inspect the implementation and tests before
changing them. Preserve unrelated dirty work.

Do not redo Phase 0 or the scalar Phase 1 reference. The existing baseline has:

- disjoint exact Unicode-scalar and byte domains;
- arithmetic empty/atom tokens and monotonic aggregate IDs;
- collision-proof exact interning under forced total hash collision;
- deterministic covers using prior tokens;
- contiguous typed structure-of-arrays buffers;
- flat ordered batch interning;
- strict encoding receipts;
- deterministic little-endian digested snapshots with structural validation;
- a bounded single-writer queue prototype with backpressure and shutdown;
- a live compiler-safe FNV-1a limb kernel called by the reference core;
- exact compile-and-execute parity for that kernel through Turing C and LLVM;
- greedy compatible-request fusion into a COO-style flat ABI, followed by one
  padded AbstractTensor batch for masked atom validation and fingerprinting
  using the same live kernel;
- eager crossover measurements showing a small NumPy win only at the largest
  recorded batch, so whole-core compiler lowering remains mandatory;
- a durable holistic audit that pursues the live `Vocabulary.intern_batch`
  graph through Turing and now has one isolated refusal: persistent dynamic
  arena growth lacks explicit native pointer ownership and capacity;
- a version-zero `token_kind_batch` SSA API with an authored AbstractTensor
  annotation, centralized ProgramABI, exact C/LLVM compile/execute proof, and
  exported runtime-sized `int64 span -> int64 span` ABI;
- Turing PR #923's signed-integer LLVM repair, proven by both the runtime-sized
  query API and the vectorized retention mask; unsupported unsigned native
  declarations remain refusals rather than silently erased types;
- a retained Turing native graph-reverse diagnostic: generated training-motion
  functions currently reach `FunctionTable.declare` without source-cell
  provenance required by `function_address` concordance. Plain parametric VJP,
  parameter-group training, scalar-loss adjoints, and the compiled perforated
  Adam learner all fail at that same boundary; AbstractNN's four captured
  training-step laws remain green;
- an optional bounded background embedding learner consuming flat mint deltas,
  maintaining fixed unit-hypersphere positions separately from learned
  residuals and meaning vectors, training Turing's established
  `Linear -> Tanh -> Linear` AbstractNN with its Adam optimizer from batched
  ordered COO cover targets and typed live-context targets, publishing a
  versioned dense row map, and becoming explicitly desynchronized on queue or
  observation-range loss;
- a version-zero integer observation ABI for source/runtime/decompiler/glyph
  facts, plus a strict six-stage optional semantic compilation prefix and an
  inspectable state ABI which excludes network/Adam state below training;
- standalone AbstractTensor preparation roots for spherical position, ordered
  composition, and typed observation reduction, with a depth-pruned Turing
  ProgramABI; these are eager-verified but do not yet claim native artifacts;
- a non-instantiable `String` namespace for live Python bytes ingress. It
  returns only integer tokens/receipts and routes all mutation through one lazy
  process-global bounded writer service; its exact compiler identity and
  depth-selection acceptance contract are in `STRING_COMPILER_HANDOFF.md`;
- an optional integer retention-notice ABI and an AbstractTensor stale-row
  mask. Reclamation is explicitly a future vocabulary-generation rotation
  after epoch pins clear; IDs are never recycled, and semantic carry is an
  explicit cold/retained/conditioned policy;
- a focused extraction audit proving current Turing pursues the Python shim
  into `TokenService` and rejects it as undeclared instead of recognizing the
  exact Token Orchard component boundary;
- an exact Python source-capture manifest joined to Turing's existing glyph,
  StringTable, explicit UTF-8 encode, and byte-join facts. Ready token
  operations are separated from strip/replace/join/casing work that still
  needs exact batched transform opcodes;
- a profiled live text-edit demonstration exercising both queues, ordered and
  context training, backward, and Adam. It truthfully reports the still-red
  native training capability and names the generated-VJP provenance blocker;
- 72 passing standard-library tests after the namespace, retention, capture,
  and learning milestones.

The next coherent milestone is **immutable epoch publication and the complete
integer operation surface**:

1. define the capacity-bearing native vocabulary-state ABI (typed arena
   pointers, used counts, capacities, ownership, and growth/error policy), then
   clear `tools/audit_turing_whole_core.py` and lower the entire reachable core
   as one Turing compilation product: tensor screening, exact candidate
   verification, indexes, cover construction, ordered mutation, epoch
   publication, and response assembly. Retain the fingerprint-only C/LLVM gate
   solely as a differential diagnostic, never as a separately dispatched
   service boundary;
2. separate mutable writer state from immutable AbstractTensor reader snapshot
   handles, then pin/release epochs and prove old readers remain stable;
3. add integer batch opcodes for expand, length, atom-at, concat, slice,
   compare, and exact find—not only batch interning; extend the existing
   versioned SSA contract rather than creating an unrelated calling surface;
4. add atom-credit admission/release for a true queued-memory bound, then add
   cancellation-before-execution and idempotent request-ID behavior;
5. replace the first-atom candidate scan with a measured compact prefix index
   while preserving the Phase 0 cover order exactly;
6. retain contiguous buffers and measure allocations, resident bytes, and
   cold/warm throughput;
7. extend snapshot receipts with a stable vocabulary identity without making
   token IDs content hashes.

The learner is only a first attachment surface. Before treating it as a durable
continuous service, add snapshot-based full resynchronization, persisted model
and optimizer receipts, bounded atom/edge/observation credits, negative
sampling and held-out prediction measurements. Then implement immutable
embedding publication and batched top-k/token proposal queries as depths five
and six. Neural output remains advisory and may propose mips or caches only
after deterministic exact validation; it never mints identities.
The current learner begins at epoch zero. An existing vocabulary must first
offer `graph_delta_since_row(0)` and wait for that epoch before live attachment;
persisted partial bootstrap receipts are not yet claimed.

When revising ingress, add optional validated COO row/column coordinates beside
values and offsets so padding can be a direct tensor scatter. Do not derive
those coordinates with the current base-operation `AbstractTensor.searchsorted`:
that implementation is a broadcast comparison matrix and measured
O(items*atoms) here. Until the ABI carries coordinates, retain the single
O(atoms) topology-only pack.

Treat vectorization honestly. The queue now presents one padded AbstractTensor
batch, not Python length buckets or per-string calls. Do not bypass
AbstractTensor with a NumPy side implementation and do not promote scalar
convenience methods into compilation targets: compile the batched shared core,
let the compiler fuse its internal graph, and differential-test every backend
against the scalar oracle, including constant fingerprint mode. Hashes never
prove identity.

Do not modify Turing, Nodus, or Spectral Analyzer unless the user explicitly
opens that integration task. When Turing compilation is authorized, first read
its local `AGENTS.md` and current hazard receipt and use only
`src.compiler.fortran_c_shell.lower_ast_source_to_ssa`. Retain the PR #906
testing discipline, confirm the current `FUZZ_EXT` range, and treat cloud Linux
as authoritative.

Run at minimum:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q token_orchard tests benchmarks tools
python tools/test_turing_compile.py
python tools/test_turing_ssa_api.py --backend all
python tools/test_turing_retention.py --backend all
python tools/audit_turing_whole_core.py  # expected failure until arena ABI lands
python -m benchmarks.reference_baseline
python -m benchmarks.tensor_preflight_baseline
```

Record commands, measurements, failures, and exact limitations in both the
project receipt and a compliant shared experience report. Do not call a queue
epoch immutable until a pinned-reader test proves it. Call only the fingerprint
step native today: expand that claim solely as additional live code paths cross
compiler/backend parity.
