# Boundary Fluency Campaign — turing ↔ nodus

**Date:** 2026-08-13
**Companion:** `BOUNDARY_ERROR_REGISTER_2026-08-13.md` (the observed failure modes every rung must guard against)
**Prior evidence ledger:** `TRANSLATION_MATRIX_SIGNOFF_2026-08-12.md`

## Purpose

Resolve the turing↔nodus boundary into demonstrated fluency by iterating through
examples of increasing complexity, each translated end-to-end into a **nodus
headless runnable** and verified against an independent reference. Every rung
produces a substitution certificate (inputs battery, tolerances, reference
identity) so admission is a stored fact, not an assurance.

The governing invariant: AbstractTensor's base operator set plus its reference
implementation define semantics once; any backend that provably matches the
contract can be swapped per-operator with autograd preserved. Each rung of this
ladder is a substitution proof at a larger granularity.

The governing mode rule: numeric interiors compile native (recursively only
when provably boundary-free); host-boundary calls remain preserved channels;
tables that need live Python get a Python instance to live in, isolated by KPN
edge discipline (no shared memory).

## The Ladder

Rungs are cumulative: each reuses the machinery the previous rung proved, and
each names the error-register entries (E-numbers) it specifically guards
against. A rung is DONE only when its headless runnable passes its verification
battery with the certificate recorded.

### Rung 0 — Existence (DONE 2026-08-13)
- **Example:** zero modules, zero rows, zero tensors; create → step ×11 → destroy.
- **Runnable:** `minimal_existence_test.py` over `nodus_headless_bridge`.
- **Proves:** engine lifecycle is sound at rest; failures above this rung are
  in translation or configuration, not the engine's existence.
- **Guards:** E1 (paused scheduler), E9 (stale workspace autoload contaminating
  a "blank" context — observed live in this rung's log).

### Rung 1 — Single vocabulary operator, both engines (DONE 2026-08-13)
- **Example:** every eligible canonical op (40), as one generated tool per
  backend group (inmemory, eigen) = 80 standalone tool DLLs.
- **Runnable:** `nodus/tests/vocab_tool_dll_test.cpp` (registered ctest
  `vocab_tool_dll_test`). Result: **80/80 PASS**, cross-engine agreement and
  independent `<cmath>` reference.
- **Proves:** the vocabulary actualizer pathway; handle currency
  (`VT_ABSTRACT_TENSOR`) across DLL boundaries; one shared arena via
  `nodus_tensor_core`; the substitution gate at op granularity.
- **Guards:** E2 (silent null `create()`), E5 (cross-image arena identity),
  E4 (stack currency mismatch), E6 (build-config macro overrides).

### Rung 2 — Compiled scalar section import (DONE 2026-08-13)
- **Example:** `blend_signal(a,b,scale) = m - m³/3, m = a*scale + b` — pure
  arithmetic, boundary-clean by construction. Compiled by turing
  (`compile_section_to_dll`, 10 s) to
  `nodus/module_library/test_build/artifact_import_fixture/blend_signal.dll`
  + `blend_signal.api.yaml`.
- **Runnable:** `artifact_import_test` (registered ctest). Result: **PASS**,
  `got 0.575189` exact, negative control (perturbed inputs rejected) and
  substitution certificate written. First run FAILED twice, productively:
  the failures minted register entries E14 (input misread as output after a
  silent artifact-load failure) and E15 (dependency search flags inert
  without a fully-qualified backslash path; runtime deps must be declared in
  the contract — the producer knows them at compile time).
- **Proves:** the api-v1 contract is sufficient for automatic launch wiring;
  a turing-compiled artifact becomes a nodus tool with zero hand-written glue.
- **Guards:** E7 (boundary-dirty input hanging the compiler — this rung's twin
  fixture demonstrated it), E11 (entry-point parameter-order divergence: the
  fixture's control entry orders `t0,t2,t1` while its region entry orders
  `t0,t1,t2`; wrappers must follow the *selected entry's* declared order),
  E12 (dtype: artifact layer is float64; vocabulary tools are float32 —
  conversions must be explicit, never assumed).

### Rung 3 — Array section with extents
- **Example:** an elementwise-over-arrays section (e.g. saturating mix over
  n-element vectors) whose contract carries `shape`/`extent` parameters and
  by-reference arrays.
- **Runnable:** extend `artifact_import_test` with the array fixture; wrapper
  generation must resolve extent parameters from the arrays that name them.
- **Proves:** the importer's v1 shortfalls (extents, arrays) close; payload
  mapping (`nodus_tensor_map`) replaces element reads at the call boundary.
- **Guards:** E11, E12; new hazard: extent/array binding mistakes produce
  access violations, not wrong numbers (the contract states passing precisely
  because of this — honor it mechanically).

### Rung 4 — Multi-op fused section vs its own composition
- **Example:** a section turing fuses from several canonical ops (the
  `dispatch_region_to_fused_program` hook), imported as ONE tool; the same
  program also spelled as a canvas row program of vocabulary tools.
- **Runnable:** headless canvas executes both forms on identical inputs; the
  fused import must agree with the primitive composition AND the reference.
- **Proves:** fusion is semantics-preserving across the boundary; the
  granularity dial (one-op tools ↔ fused sections) is sound; the certificate
  covers a *section*, not just an op.
- **Guards:** E3 (headless scheduling assumptions — this is the first rung
  executing a REAL canvas headlessly since the thread_manager fixes), E8
  (deferred-logger reordering when diagnosing), E10 (moving crash sites =
  suspect memory discipline, not the nearest log line).

### Rung 5 — Structured-class operations (REDUCE first)
- **Example:** `sum` / `mean` over a vector — the first ops whose KernelIR
  expression uses a structured class (REDUCE) rather than an elementwise slot.
- **Runnable:** two admitted witnesses headlessly compared: the imported
  C/LLVM artifact (`sum_double`, `mean_dim` — already real in turing) and the
  SPIR-V/KernelIR realization; both against the reference.
- **Progress 2026-08-13:** the device-side realization EXISTS and validates —
  `kernel_spirv.cpp` now assembles REDUCE (single-invocation structured loop:
  Phi/LoopMerge/back-edge, per the class's documented v1 form) and GENERATE
  (pure function of gid); both modules pass
  `spirv-val --target-env vulkan1.1`; CONTRACT/REMAP/ORDER refuse by name
  pending assembly; KIRTEXT parses all five class spellings. `mean` is
  expressible today as REDUCE(add) + an elementwise 1/n scale.
- **Producer half PROVEN 2026-08-13:** `kernel_ir_lowering.py` emits REDUCE
  from repository SSA — a reduction crosses as *itself* (one instruction
  naming its combining operation, source passed whole) rather than atomized
  into the elementwise vocabulary. Turing's verbatim KIRTEXT parses in nodus
  and assembles to a `spirv-val`-clean module. Remaining for this rung:
  **execution** of the two witnesses (this kernel vs turing's `sum_double`)
  against the reference.

> **Why full KernelIR coverage of the tensor surface is the whole game:** the
> fundamental operators carry composition schemas. Completing the fundamental
> set does not merely make those operations differentiable — the extended
> surface (linear algebra, FFT, …) is *composed* from the fundamental set and
> therefore **inherits autograd by construction**. So breadth is not ~300
> independent implementations; it is one basis plus its composition schemas.
> This is precisely why the structured classes matter: they are the basis
> elements the elementwise vocabulary could never span.
- **Proves:** the matrix establishment (66/66 lowerable) carries to execution;
  reductions cross the boundary in both the import lane and the IR lane.
- **Guards:** E12; new hazard: reduction identity/empty-extent edge cases —
  battery must include n=1 and the neutral element.

### Rung 6 — Dual IR pipeline, node-for-node
- **Example:** a small multi-region program (3–4 fused sections with a fan-out)
  carved by turing's fusion planner; each region imported as a tool; the dual
  IR topology copied node-for-node into a CANVAS/KPN document (opaque-region
  module flavor: Input rows → one imported plugin row → Output row).
- **Runnable:** `nodus_headless_validate`-style harness: load canvas, push
  tokens, run to quiescence, pull result, compare against turing executing the
  same program natively.
- **Proves:** "dual IR can be almost node-for-node copied for making a KPN
  pipeline" — at import granularity; typed edges carry the inter-section
  contract; scheduler ordering honors the region DAG.
- **Guards:** E1, E2, E3 all at once — this is the rung where the entire
  earlier headless debugging campaign becomes regression armor; also E13
  (registry id aliasing: canvas ROWS name bare ids; loaded tools must answer
  to them).

### Rung 7 — Preserved boundary: a table with a Python instance
- **Example:** a pipeline where one section is host-boundary (e.g. an op whose
  implementation is live Python calling its own LLVM-compiled pieces), given
  a Python instance to live in (subinterpreter per table).
- **Runnable:** headless canvas where the Python-hosted table exchanges tokens
  over ordinary typed edges with native tables; no shared memory; the instance
  reaches nodus tensors only through `tensor_abi`.
- **Proves:** cross-environment operators, elegantly: ownership rule honored
  (nodus provisions the context), KPN isolation preserved, python references
  to compiled sections still work from inside the instance.
- **Guards:** E7 (this is the *correct* resolution of what E7's failure mode
  punishes), E5 (the instance must bind the host's `nodus_tensor_core`, not
  its own copy).

### Rung 8 — Autograd round trip
- **Example:** a differentiable program whose forward runs through imported /
  composed nodus tools and whose gradient is produced by turing's tape against
  the same canonical ops.
- **Runnable:** headless forward in nodus; gradient checked against turing's
  autograd on the reference backend AND against finite differences.
- **Proves:** the point of the whole matrix — substitution preserves autograd:
  swap any admitted backend under the tape and gradients remain exactly right.
- **Guards:** everything above; this rung certifies the invariant itself.

## Discipline

- **One rung at a time; regressions run every rung below.** The ctest registry
  is the ladder's memory.
- **Negative controls are part of fluency** (the `wasm_fidelity` lesson): each
  harness must demonstrate it *catches* a deliberately wrong witness before
  its PASS is trusted.
- **Certificates accumulate** beside artifacts/manifests; the NODUSPKG
  `capability_tags` plus certificate are an operator set's passport.
- **Every failure goes to the register** (`BOUNDARY_ERROR_REGISTER`), with the
  rung that exposed it and the mechanical guard adopted.
