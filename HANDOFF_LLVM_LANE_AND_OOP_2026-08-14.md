# Handoff — LLVM lane, OOP interchange, print capture, growth display
**Date:** 2026-08-14
**Repos touched:** `turing/`, `nodus/` (both uncommitted; see Footprint)

This document is written for someone resuming cold. It states what is
*proven by a run*, what is *built but unproven*, and what is *known broken*,
separately and without blurring. Where I got something wrong during the
session, that is recorded too — several of those mistakes are the most useful
part of this document, because they are patterns that will recur.

---

## 1. The governing frame

Two lines of work ran in this session, sharing one principle the user states
repeatedly and enforced hard:

> **Coordinate existing machinery. Do not author parallel machinery.**

The compiler is already a staged rig (`opportunistic_pipeline.COMPILER_KINDS`:
`source → process_graph → annotated_graph → deployment_plan → dual_ir →
folded_dual_ir → ssa_module → target_source → executable`). Lighting up a
capability means *registering a provider for a segment*, not writing a new
compiler. Every time I drifted from that, the user caught it, and every time
the correction was right.

Standing rules established this session, all of which still apply:

- **No tape.** `lower_abstract_tensor_tape_to_llvm_ssa` and the whole
  captured-tape lane are OUT for this work. (Ruled twice, emphatically.)
- **No FusedProgram bake.** The dual IR entry is precompile-only,
  whole-program, parameters symbolic.
- **No JIT.** The lane ends at a compiled native artifact
  (`executable` kind). MCJIT/`OptimizingJITProgram` is the tape lane's idiom.
- **No new emitters where tables belong.** The deliverable for a backend is
  *a table of deterministic likeness*, matching the Fortran sibling.
- **No float-only paths.** `_double` kernels were always temporary; dtype and
  extents come from descriptors. **All backends must cover their types,
  including Fortran.**
- **Do not avoid abstract_nn.** If capture struggles on it, *show the
  struggle*; never substitute a simpler program to make a test pass.
- **Ask when unsure rather than inventing.**

---

## 2. AOT → SSA → LLVM lane (task #36)

### 2.1 The path, confirmed by probe

```
compile_ast_aot(source, entrypoint, feeds,
                precompile_only=True, bake_mode="whole_program",
                mutable_parameters=<all params>)          # THE dual-IR entry
  → compilation.compiled_shell_program / .shell_control_program
    / .region_programs / .hierarchy_plan / .identity_table
    / .function_outputs / .function_parameters
→ lower_precompile_and_control_to_ssa(..., tensor_ssa_reference=
      c_backend_repository_ssa_reference())               # repository SSA
→ ssa_llvm_backend.emit_ssa_function_to_llvm(module, "numerical_region_0")
→ ssa_llvm_backend.compile_artifact(...)                  # zig cc, AOT
```

**Measured on a full 29-parameter MLP training step (forward + backward +
Adam, 127 SSA instructions):**

| stage | result |
|---|---|
| AOT capture | 1.1 s |
| SSA lowering | 1.1 s (6.4 s with the tensor reference wired) |
| lowering shortfalls | **0** (from 108) |
| emission shortfalls | **0** |
| emitted region | 109 kernel calls, 331 extent reads, 144 buffers, 1366 IR lines |
| native artifact | **NOT YET PRODUCED** — see 2.4 |

The compiler pre-carves numeric regions: `numerical_region_0` is a pure
call sequence into the *authored* kernels (`binary_double ×82`,
`matmul_double ×8`, `unary_double ×8`, `transpose_double ×5`,
`sum_double ×3`, `binary_scalar_double ×2`), and `planned_control` is only
`Call/Const/GetElementPtr/Load/Ret` carrying **all 19 outputs**. The dual-IR
split is the emitter's work plan; I did not invent it.

> **Retraction worth recording:** early in the session I reported that the
> output projection dropped 12 of 19 outputs and that `sum(dim=0)` was lost.
> Both were artifacts of entering through the *wrong* entry
> (`lower_ast_source_to_ssa` directly). Through `compile_ast_aot` they do not
> occur. Do not chase them.

### 2.2 `src/compiler/ssa_llvm_backend.py` (new)

Structured exactly like `ssa_fortran_backend.py`: tables first, emitter
second, compilation as a separate optional step.

- `_BINARY` / `_UNARY` — 40 scalar entries, SSA opcode → LLVM instruction
  template (`{0}`, `{1}`, `{out}`). Mirrors the Fortran table's key set
  including the unsigned `ULt`/`ULe` pair.
- `_TENSOR` — 56 entries, SSA tensor operation → **authored kernel symbol**.
  Bodies and signatures are *not* restated; they live in
  `c_backend_llvm_ssa`. `mean → sum_double` with the scalar-divide noted.
- `_SHAPE_ONLY` — view operations, aliased in every target.
- Accessors: `supported_scalar_operations()`, `supported_tensor_operations()`,
  `scalar_likeness()`, `tensor_likeness()`.
- `emit_ssa_function_to_llvm(module, fn, *, entry_name, text_sink=False)` —
  in-order fold over the compiler's schedule. Kernel calling conventions are
  **parsed from the authored `define` lines**, never re-catalogued. Anything
  off-table becomes a named `LLVMEmissionShortfall`.
- `compile_artifact(artifact, directory=...)` — `python -m ziglang cc -shared`
  (the C backend's own toolchain resolution), links the text sink when needed.

ABI of the emitted entry:
`void <name>(double** buffers, int32_t* extents)` — one pointer per SSA
tensor value in `artifact.buffer_order`, one int32 per extent read in
`artifact.extent_order`. **The executor companion that measures extents and
allocates buffers does not exist yet.** That is the next concrete build.

### 2.3 Typed dynamic extents in `tensor_ssa_lowering.py` (shared layer)

This is the substantive compiler work of the session, and it benefits
**every** backend, not just LLVM. Under `precompile_only` the whole-program
capture keeps parameters symbolic, so static shapes are absent by design.
Previously that produced 108 refusals. Now extents become ordinary SSA.

Fixes, in the order they were forced by measurement (108 → 13 → 0):

1. **Scalar `Const` payload** (`_constant_payload`): the whole-object compiler
   spells scalar and vector constants with the same `values` key; `tuple()`
   on a scalar raised. Now uses the module's own `_as_sequence` idiom.
   *(The user observed the deeper fix is a canonical payload class defining
   `__iter__` at the producer — offered, not built, to avoid a wide
   producer/consumer change mid-flight.)*
2. **Whole-descriptor dynamic crossing** (`ensure_dynamic`): the descriptor's
   validator requires shape **and** rank **and** element-count together; I
   first minted them piecemeal and it raised. All three are minted at once
   and cached per tensor.
3. **Extent inheritance** (`register_tensor(extent_ids_from=...)`): an
   elementwise result inheriting a dynamic source's state must inherit its
   extent value ids, else it is dynamic-without-extents and raises.
4. **Single-authority kernel linking** (`precompile_to_ssa.py` ~5496): when a
   tensor reference is supplied, *its* module is the kernel import. A second
   independent `import_llvm_to_repository_ssa` minted different `Function`
   identities and collided at the linker (`binary_value`).
5. **Dynamic matmul dims** (`dim_extent`), **flat mean** (`sum_double` +
   scalar `Div`), **dynamic axis reductions** (runtime shape/rank operands),
   **dynamic transpose** (documented 0/1-swap rank-2 contract).
6. **Binary conform through `broadcast_double`** with runtime extents. This
   closed a genuine **silent-wrongness landmine**: the old code passed
   mismatched operands straight through when either shape was unknown.
7. **Tensorhood gate** for `Div`: with symbolic shapes, an operand already
   registered in the tensor table is proof of tensorhood.

### 2.4 Where the lane actually stops

`zig cc` rejects the emitted module:

```
error: multiple definition of local value named 'buffer.addr.0'
```

Two `%buffer.addr.0` definitions at lines 403 and 712 of the emitted `.ll`.
I had just changed buffer loads from a collected preamble to inline-at-first-
use (correctly — the user's point that the compiler's schedule must be
respected was right), and **the duplicate is almost certainly the authored
kernel texts being spliced in with their own local names, or my emitter
emitting into a second region without resetting `buffer_index`.** This was
not diagnosed before the session ended. It is a small, local bug in
`emit_ssa_function_to_llvm`, and the reproduction is one command (2.5).

**This is why the LLVM lane has no time yet.** Everything upstream of it
is at zero shortfalls; the blocker is one duplicate-symbol defect in my own
emission.

### 2.5 Reproduction

`tests/test_ssa_llvm_backend.py` runs the whole lane on
`examples/xor_project/train_xor.py` (the real abstract_nn program — Model,
Linear, Adam, the actual loop). Three tests: lowers without shortfalls,
emits without shortfalls, compiles to a native artifact.

**Status: the test has never completed a run.** Every invocation was either
killed by the user or produced empty output. On the real abstract_nn program
the capture is slow enough that I never saw its census. The user's
observation — *"it's compiling like it's swallowing the python ecosystem"* —
is the live hypothesis and is the reason print capture (§4) was started.

Scratchpad probes that DID complete are the synthetic training step
(`probe_step_ssa.py`, `measure_lowering.py`) — those produced every number in
§2.1.

---

## 3. OOP interchange layer (task #30) — proven earlier in the session

Both crossings are proven end to end and covered by a passing pytest
(`tests/test_oop_interchange.py`, 4 passed):

- **Object crossing:** `abstract_nn/core.py` → `build_from_ast` ingestion
  capture (class capture is ingestion-time, no pursuit needed) →
  `oop_schema.class_schema_from_map_ir_object` → SCHEMA V1 text → nodus
  `generic_object_test <file>` document mode: *"class 'Linear' (python) — 5
  field(s), 5 method(s) — lives as a C++ object"*, per-instance state and
  dispatch verified. Method arity now crosses (parameters captured, `self`
  dropped).
- **Operation crossing:** `shallow_interpretation.method_to_graph_ir` —
  semantic lowering + return-cone slice + `process_graph_to_nodus_graph_ir`.
  `ReLU6.forward` → 18 nodes → executed in C++ → `0 0 0 0.5 3 6`, matching
  numpy within 1e-9. `GELU.forward` (helper pursuit + class constants) → 15
  nodes, exact. `MSELoss.forward` (reductions) → 6 nodes → `0.3125`, exact.
- **nodus reductions**: `tensor_reduce_sum_all` / `tensor_reduce_mean_all`
  added to `tensor_math` (beside `tensor_reduce_sum_axis_f32`), covering
  every numeric dtype with NumPy-parity output rules; the tool bodies are
  thin calls over them. *(First attempt authored the loops inside the tool
  registration — the user caught it; relocated.)*

---

## 4. Print capture (task #37) — started, not finished

**Done:** `node_special_cases.interpret_special_case` — the single ingestion
switch — now classifies:

- `print(...)` → `SpecialCase("stream_publish", {"stream": "text",
  "argument_count": n})`
- `float/int/bool/str/len` → named cast vocabulary ops

Verified: `print(x)` and the casts return the right `SpecialCase`; an
ordinary call still defers with `None`.

**Done:** the universal sink —
`accelerator_backends/c_backend/turing_stream_buffer.{c,h}`: one growable
buffer per stream id, `turing_stream_publish` (the exact signature every
existing emitter already calls), a `_double` variant, `turing_stream_text`
to drain (**caller may ignore it entirely**), overflow counted rather than
fatal. Compiles clean.

**Done:** capability flag — `emit_ssa_function_to_llvm(text_sink=bool)`.
With a sink, publications become calls and the artifact links the buffer.
Without one, publications are **elided**; the numeric SSA is identical
either way, because a publish is never load-bearing.

**NOT done:** the planner side. Nothing yet turns the ingestion
`stream_publish` node into a `StreamPublishBlock` (`control_source.py:244`),
which is where backpressure/reservation actually lives and which the C
renderer already emits (`control_source.py:634`). Until that link exists,
`print` is classified but not routed.

---

## 5. Growth display (§ the visualizer, and where I struggled)

**This is the part where I performed worst, and the record matters.**

The user asked for an optional live visual of the graph growing. I built
`src/compiler/graph_growth_display.py` — first as a naive points-and-lines
shader with my own spiral layout. It worked (and acquired a real GL context
on this machine through the registered providers), but it was a *lesser copy*
of something the repo already had.

The user then said: do it like the spring physics demo with the rainbow trail
visualizer we already have for process graphs. **Finding it took me far too
many searches**, and my first two answers were wrong:

1. I found `src/rendering/opengl_render/inspiration/simplegraphspring5.py` —
   which *is* the thing the user remembered: `class ProcessGraphHelper(
   ProcessGraph, GraphObject)`, spring targets from level/type/role
   (`BETA_LEVEL/TYPE/ROLE`), `particles.Visualizer` for rainbow trails,
   240 Hz threaded physics behind a double buffer. I was ready to revive it.
2. It is **unrevivable as-is**: four sibling modules are simply absent from
   the repo (`bound_spring`, `membrane_portal`, `binding_memranes_sympy`,
   plus flat non-package imports). `particles.py` survives but has one
   stranded import (`graph_express2_tests`).
3. The user said *"there are many versions but there is one already working,
   already runnable"* — and they were right, and I had not found it. It is
   **`src/computational_world/`**: `install_bound_spring` /
   `append_bound_spring` / `advance_bound_spring` over
   `ComputationalWorldState`, a `ComputationalWorld(AbstractTensorStateMachine)`,
   with three test files including `test_bound_spring_aot.py`.

**The lesson:** the `inspiration/` folder is a graveyard of prior versions;
the living implementation was in a plainly-named package I did not search
because its name (`computational_world`) does not contain "spring", "render",
or "visual". Search by *capability* (`install_bound_spring`), not by theme.

**Current state of the display:** rewritten to drive `computational_world`.
Attaches via `graph_accessor().subscribe(...)` (the accessor's own docstring
calls it "lock-aware read access for live compiler/visualization observers").
Verified: a 6-node graph produced 6 spring nodes and 6 installed edges with
physics ticking and the display still active.

One more mistake worth recording: my first version hand-assigned
`state.spring_edge_index` directly, leaving every other per-edge array at
zero rows → `shape mismatch: (2,) vs (0,)`. That is precisely the
"bolt-on around scheduled machinery" pattern the user warns about.
`install_bound_spring` owns all those arrays; `grow()` now goes through it
exclusively, carrying settled positions forward so growth still reads as
accretion.

**Not done:** the rainbow ghost-trail layer (`particles.Visualizer`, needs
its one stranded import fixed) and the GL-context shell launcher for
translation that the user named as the next piece.

---

## 6. Backend defects found and fixed (all verified by rerun)

These were found by running ordinary abstract_nn code and are independent of
the compiler work:

| defect | fix | verification |
|---|---|---|
| E17: nodus arena refuses `mul` on comparison masks — **the default backend broke ordinary abstract_nn code** | promote bool→uint8 for compute, cast back (array path); `np.result_type` before the boundary (scalar path) | `mask*mask`, `mask*scalar`, full `ReLU6.forward` on the live arena all match numpy |
| `pure_backend.long_()` arity + `to_dtype_` dual calling convention | receiver-style defaults on all five casts; `to_dtype_` recognises both conventions; widened the dtype spelling set | `ReLU6.forward` on pure matches numpy |
| **`_v2_valuewise` divisibility lift computed silently wrong values** under interior-singleton broadcast | real NumPy-rule broadcasting (`_broadcast_result_shape` + `_broadcast_flat_index`) | mask shapes and values element-exact vs numpy |
| same defect in `_v3_valuewise` (`where`) | same helpers | — |
| `core.py` `shape` property-vs-method schism (broke every `MaxPool2d` user) | one `_shape_of` helper, 15 sites | CNN trains |
| **`train_step` destroyed conv gradients** — module-wrapped layers stash `W._grad` during the tape walk and the next loop overwrote it with the tape's `None` | harvest the stash instead of clobbering | CNN loss 0.666→0.572 in 40 epochs, conv grads verified live |
| `train_step`/`Adam` assumed every layer is Linear-shaped (4 sites) | walk each layer's actual parameters; `None`-grad params skip cleanly | pooling/flatten layers no longer desync |
| debug print flooding every training step (`whiteboard_runtime`) | removed | — |

---

## 7. Race results (the reason for the timing work)

`examples/pattern_project/train_patterns.py` — MLP on a checkerboard, CNN on
striped textures, both through `abstract_nn`'s own `train_loop`/`Adam`.
CLI takes backend names (`main(argv)`); no args races all available.

| network | numpy | nodus arena |
|---|---|---|
| MLP (800 epochs) | 69.9 ms/epoch | 96.2 ms/epoch (1.38×) |
| CNN (120 epochs) | 119.8 ms/epoch | 163.7 ms/epoch (1.37×) |

Loss curves identical on both (MLP 0.250→0.151; CNN 1.035→0.325).
**The LLVM entry is the third racer and has no time yet** (§2.4).

Rejected candidates, for the record: `CTensorOperations` (precompiled cffi
kernels — not compiling the Python) and the tape lane (ruled out).

---

## 8. Footprint (all uncommitted)

**turing — modified:** `abstract_nn/{core,optimizer,train}.py`,
`abstraction_methods/elementwise.py`, `accelerator_backends/nodus_backend.py`,
`autoautograd/whiteboard_runtime.py`, `numpy_backend.py`, `pure_backend.py`,
`compiler/{ast_process_graph,fortran_c_shell,kernel_ir_lowering,
precompile_to_ssa,tensor_ssa_lowering}.py`,
`transmogrifier/graph/{graph_express2,node_special_cases}.py`,
`tests/test_kernel_ir_lowering.py`

**turing — new:** `compiler/{ssa_llvm_backend,shallow_interpretation,
oop_schema,graph_growth_display}.py`,
`accelerator_backends/c_backend/turing_stream_buffer.{c,h}`,
`tests/{test_oop_interchange,test_ssa_llvm_backend}.py`,
`examples/{xor_project,pattern_project}/`

**nodus — modified:** `CMakeLists.txt`, `include/canonical_ops.h`,
`include/common/tensors/abstraction/tensor_math.h`, `include/tool_registry.h`,
`ops/*`, `src/common/tensors/abstraction/{abstract_tensor_graph_ir,
tensor_math}.cpp`, `src/kernel*`, `tests/*`; **new:**
`tests/tensor_graph_execute_test.cpp`, `docs/TIERS.md`

**Test edits reverted on instruction:** I modified
`tests/test_llvm_repository_ssa.py` to match the new dynamic-extent contract
and was told to stop hand-rolling tests. **That test now fails**
(`test_unknown_tensor_extent_is_not_silently_compiled_as_one_element`) because
the behaviour it guards has legitimately changed — unknown extents are no
longer *refused*, they become runtime SSA. **Someone must decide how that
guard should be re-expressed.** It is a real, open disagreement between the
old contract and the new capability, not a regression.

Also failing, and untouched by design: 6 tests in
`tests/test_c_backend_llvm_ssa.py` covering the **tape** lowering path.

---

## 9. Frontier — what to do next, in order

1. **Fix the duplicate `%buffer.addr.0`** in `emit_ssa_function_to_llvm`
   (§2.4). One bug between here and a compiled artifact.
2. **Build the executor companion**: measure extents by propagation over the
   same instruction stream, allocate buffers per `artifact.buffer_order`,
   call the entry, read outputs. Then one-step numeric parity vs numpy.
3. **Time it** in `train_patterns.py` as the third racer.
4. **Get `tests/test_ssa_llvm_backend.py` to actually complete** on the real
   abstract_nn program — and if capture is swallowing the Python ecosystem,
   that is the finding, not an obstacle to route around.
5. **Finish print routing**: ingestion `stream_publish` → planner
   `StreamPublishBlock`.
6. **Resolve the extent-guard test** (§8) with the user.
7. **Fortran types**: the same descriptor-driven dtype coverage the user
   ruled for LLVM applies to Fortran; `_BINARY`/`_UNARY` there are
   double-only in the same way.
8. **Trail layer + GL shell launcher** for the display (§5).

---

## 10. How I failed, so it is not repeated

- **Searched by theme, not capability.** Cost many turns on the visualizer
  and produced a wrong "it's unrevivable" conclusion before finding
  `computational_world`. Grep for the *function that must exist*.
- **Bolted onto scheduled machinery** twice: a buffer-load preamble that
  reordered the compiler's schedule, and a direct write to
  `spring_edge_index` around `install_bound_spring`. Both times the symptom
  looked like a dependency-ordering problem and was actually me.
- **Wrote code where a table was asked for**, then an emitter where
  coordination was asked for. Read the deliverable noun in the instruction.
- **Substituted an easier program** (an AT-only XOR) when the real
  abstract_nn program was slow to capture. Reverted on instruction; the file
  is deleted. Show the struggle instead.
- **Hand-edited a guard test** to match new behaviour. Left reverted and
  flagged for the user instead.
- **Claimed properties "by construction"** (DLL boundary, own reduction)
  before a run confirmed them. They are argued in §2.1 but the real
  abstract_nn census still has not been seen.

---

## 11. Visualization continuation — 2026-08-14

The reduced `compiler/graph_growth_display.py` renderer described in §5 is no
longer the implementation. It is now only a compatibility entry into the
already-working visible surface in `rendering/precompiled_graph.py`, which
uses `MultiNetworkFluxSpring` and `LiveVizGLPoints` directly.

The continuation added:

- lexical source class/scope provenance carried from AST ingestion through
  evolution events and cross-IR handoffs;
- a live top-K census reporting attributed subgraph size, relative size,
  nodes/second, SCC-safe depth/height, and per-stage counts;
- a separate cool-colored urgency ring outside culprit nodes. This is strictly
  renderer state: tests prove applying it does not alter spring positions or
  velocities, and no hierarchy coordinate is fed into the physics;
- an emergency compiler-thread clamp, defaulting to depth 512, height 512, and
  50,000 nodes per attributed branch. It aborts with the culprit and stage
  census. Raising it requires explicit CLI `--growth-limit-boost` or explicit
  limit flags;
- full-evolution source launching through
  `python -m src.rendering.precompiled_graph_demo --source <file> --entrypoint
  <function>`.

Verification: 18 focused tests pass; the real pygame/OpenGL window opened and
closed cleanly during a live compiler run; an intentionally tiny CLI branch
ceiling stopped compilation and named `scope spectral_route`. A bounded XOR
`abstract_nn` launch did not run long enough to produce its first useful class
census, so that measurement remains open and must not be replaced with a
simpler program.
## 2026-08-14 exact live compiler-event visualization correction

The visualization contract is now explicitly compiler-coupled, not a sequence
of periodically sampled whole-graph snapshots. The full compilation launcher
subscribes to `EvolutionMetaGraph` before compilation starts and applies each
event to the visible FluxSpring graph in sequence:

- `component-spawn` / `component-update`: reveal that node mutation;
- `component-link` / `component-handoff`: reveal that edge mutation;
- no event coalescing or catch-up skipping;
- a bounded queue pauses the compiler when rendering falls behind, while the
  already-visible spring system continues taking physics steps every frame.

`--event-trace` prints the same sequence number, kind, and target currently
being applied to the window. `--release-hz` controls event reveal cadence, not
snapshot cadence, and `--max-event-backlog` controls compiler backpressure.

The emergency clamp and namespace restart cascade wrap this stream. A failed
attempt stays visible; a changed precise boundary rule starts a fresh compile
whose events append into the same live ledger. They do not replace or batch the
node-by-node / edge-by-edge visualization.

Verification on 2026-08-14: 31 focused tests passed, including explicit tests
that a full event queue blocks the compiler and that an edge remains absent
until its exact link event is projected. The suite also locks down a startup
race between replayed and concurrently emitted events. A real pygame/OpenGL
run traced sequence `#0`, `#1`, `#2` in order while FluxSpring and growth
telemetry ran.

### XOR training-file verification and compiler isolation

The real `examples/xor_project/train_xor.py --entrypoint train` run revealed
that a compiler thread could monopolize the Python GIL and starve both queued
events and FluxSpring. Full compilation now runs in a spawned process. Its
authoritative events cross a bounded multiprocessing queue into
`EvolutionMetaGraph.ingest_event`; sequence numbers are retained, and a full
visual backlog still backpressures the compiler process.

The same run now exposes pre-ingestion AOT phases and dependency discovery as
real compiler-run nodes/edges, then transitions into ProcessGraph events. A
60-second observed run reached 48 attributed `train` nodes, 47 edges, and
depth/height 47. The trace surfaced `Model`, `Adam`, `MSELoss`, nested methods,
unresolved builtins, and several explicitly reported unbounded upward searches.

A deliberate safeguard run using branch/depth/height ceilings 15/12/12 stopped
at 16 nodes and retained a 16-node/15-edge analyzable graph. Expansion rate
decayed to zero while physics kept running, and a receipt pointed to
`boundary_namespaces/python/train`. A renderer-only failure found during this
test—Nodus rejecting FFT `complex128` in spectral inertia—was removed by
keeping integrator-local FFT physics on its NumPy histories.

### Exhaustive program-extraction contract

Program extraction is now governed by
`turing/extraction_contracts/program_extraction.yaml`, separately from OOP
spoof namespaces. Every callable is classified as authored Python, repository
Python, third-party Python, stdlib Python, builtin, native extension, dynamic
library, or unknown; the sheet must provide a default for every class.

Actions are explicit: ingest Python source, lower as an intrinsic, retain an
existing Python host call, use a native extension/DLL in place, opt into bounded
machine-code decompilation, or reject. Decompilation requires an exact rule,
`explicit_opt_in: true`, and function/byte/dependency-depth limits. The PE
worklist now enforces those limits and retains policy-limit dependency edges.
Python pursuit likewise enforces file, aggregate-byte, work-item, and call
dependency-depth ceilings.

Choices are attached to call nodes and stored as ProcessGraph receipts. The
contract fingerprint invalidates AOT checkpoints and the live launcher accepts
`--extraction-contract`. A governed XOR run still pursued `from_list_like`,
`Model`, `set_seed`, `Adam`, `MSELoss`, `Linear`, and their methods, while
`range`, `float`, and `print` no longer entered source/decompile pursuit. Live
labels now report `contract depth N/512` instead of claiming an unbounded
search. Focused verification: 37 tests passed.
