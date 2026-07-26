# Handoff: `nodus_tensor_core` substrate extraction

> **Status update — 2026-07-26:** The linker blocker described below is
> resolved. `src/common/thread_pool.cpp` is now part of `nodus_tensor_core`,
> excluded from the two canvas amalgams, and
> `build/Release/nodus_tensor_core.dll` exists. Canonical-op generation,
> canonical-op/Turing agreement, the Tensor Calculator bridge, edge/backend
> coordination, memory manifest, tensor registry, pool torture, and Kpath
> raster tests pass.
>
> One focused regression was found after extraction:
> `tensor_backend_smoke_test` asserted that destroying and recreating a tensor
> must return the identical virtual address. The default pool explicitly does
> not cache handles and delegates reuse to the backend allocator, so address
> identity is not contractual. The test has been corrected to retain its
> actual allocation/map/persistence/wrap checks. A complete Release build now
> confirms the corrected smoke test and the standalone Tensor Calculator test.
>
> The diagnosis and failed-link transcript below are retained as an execution
> record, not as current instructions. The current Nodus tree also has a
> TensorMath-backed persistent calculator, canonical AbstractTensor ToolIR
> registrations, and versioned ingestion of Turing's equal-shape
> `FusedProgram` transport.

**Date:** 2026-07-25
**Title:** Extracting nodus's tensor-subsystem singletons into a single shared library

Format borrowed from `speaktome/AGENTS/experience_reports/template_doc_report.md` (that
repo's agent-ecosystem documentation pattern), placed here at the top level per the
existing `NODUS_PLUCK_HANDOFF.md` precedent, since this work is itself the direct
prerequisite for the nodus↔turing/pluck effort that doc coordinates.

**If you are picking this up:** the extraction is green; do not repeat the
linker-fix steps in the historical sections below. Full research context lives
in [research/12_substrate_blocker.md](research/12_substrate_blocker.md).

## Overview

`research/12_substrate_blocker.md` documented a latent-but-fatal defect: several of
nodus's tensor-subsystem singletons (`tensor_registry`'s backend map, the
`InMemoryBackend` instance, the default `AbstractTensorPool`, `GP_MemBackend`'s CPU
handle map) are `static`s in `.cpp` files that get compiled into **both**
`canvas_tables` (shared DLL) and `canvas_tables_static`. A host linking the static lib
while loading `canvas_tables.dll` plugins ends up with two independent copies of each —
so a tensor handle created on one side is invisible/meaningless on the other. This is the
same defect class (silent 0/null failure) that broke the kpath plugin tool earlier this
session (see `research/06_memory_substrate.md`), and it's foundational to two things the
project owner cares about a lot: the turing↔nodus translation unit, and universal
browser-inspectability (both require tensor state to mean the same thing across module/
process/surface boundaries — see `research/13_ui_as_structure_and_compartmentalization.md`).

The user's direction after discussing the GUI/rasterizer-coupling question (which turned
out to be orthogonal — verified firsthand that `src/common/tensors/` has no upward
dependency on the GUI, one named bridge file aside) was: **option 2a** — extract just the
stateful compute leaf into one single-shared-instance library, leave the GUI/runtime
amalgam (`canvas_tables`/`canvas_tables_static`) linking it as before, so the coupling the
owner wants (canvas ⊃ table ⊃ stage = subgraphing, rope-sim-style attach-to-object
pattern, etc.) is untouched.

## Steps taken

1. Added a new CMake target **`nodus_tensor_core`** (`SHARED`, `WINDOWS_EXPORT_ALL_SYMBOLS
   ON`, C++20) in `nodus/CMakeLists.txt`, declared *before* `canvas_tables`/
   `canvas_tables_static` so both can link it. `canvas_tables` and `canvas_tables_static`
   now both `target_link_libraries(... PUBLIC nodus_tensor_core)`.
2. Excluded the moved files from the `auto_sources.cmake` glob that feeds
   `canvas_tables`/`canvas_tables_static` (same pattern already used there for
   `src/runtime/` and `src/headless/`), so each file compiles into exactly one binary.
3. Iteratively discovered the true dependency closure by attempting an **isolated build of
   `nodus_tensor_core` alone** (`cmake --build build --config Release --target
   nodus_tensor_core`) and reading each link error — did not guess the file list up front.
   Current source list (all confirmed Torch/Eigen/PNG/HarfBuzz/SDL-free by grepping
   `#include`s before adding each):
   - `src/common/tensors/abstraction/tensor_registry.cpp` (the backend-registry statics —
     the original target of the fix)
   - `src/common/tensors/abstraction/in_memory_backend.cpp` (the `InMemoryBackend`
     singleton instance)
   - `src/common/tensors/abstraction/abstract_tensor.cpp` (found firsthand: `AbstractTensor
     ::create` routes through **its own** `static AbstractTensorPool` singleton,
     `default_tensor_pool()` — a fourth per-copy singleton not in the original Doc 12
     writeup; had to move this file too, not just reference it)
   - `src/common/tensors/abstraction/abstract_tensor_pool.cpp` (no statics of its own, but
     instantiated by the above; must live in the same binary)
   - `src/mem_backend_host.cpp` (the `GP_MemBackend` handle maps, incl. the CPU
     self-describing-handle fix from earlier this session)
   - `src/spirv_translation.cpp` (pulled in because `mem_backend_host.cpp`'s
     `gp_mem_backend_translate_to_spirv`/`cpu_dispatch_kernel` call
     `SpirvTranslator::translate_kernel_to_spirv` directly — confirmed via grep this is the
     only definition, no ODR risk)
   - `src/common/tensors/abstraction/tensor_math.cpp` (~9,300 lines — added after
     discovering `in_memory_backend.cpp`'s `set_item` and `abstract_tensor.cpp`'s `gather`
     call *into* `tensor_math.cpp` — `tensor_op_pool()`/`submit_row_jobs()`/
     `tensor_transfer()`. The coupling is two-way, not the one-way "tensor_math depends on
     the backend" picture in the original research docs. Moving it is also a **net build
     time win**: it currently compiles twice, once per amalgam; now once.)

## Historical linker failure (resolved)

The isolated build of `nodus_tensor_core` (with the 7 files above) originally
failed with **4
unresolved externals**, all from `tensor_math.obj`, all pointing at one missing file:

```
tensor_math.obj : error LNK2019: unresolved external symbol
  nodus::JobBatch::wait(void)
  nodus::ThreadPool::ThreadPool(ThreadPool::Options const&)
  nodus::ThreadPool::~ThreadPool(void)
  nodus::ThreadPool::submit_batch(ThreadPool::Job const*, unsigned int)
  (all referenced from tensor_math.cpp's dyadic_mt_bitmask_algo<...>)
```

This was traced to `src/common/thread_pool.cpp` /
`include/common/thread_pool.h`. That source is now included in
`nodus_tensor_core` and excluded from the duplicate amalgam compilation; the
isolated target and its downstream tests link successfully.

## Lessons learned

- **Don't front-load the file list — discover it by isolated-build-and-read-the-linker.**
  Every one of the 3 extra files beyond the original Doc 12 guess (`abstract_tensor.cpp`,
  `tensor_math.cpp`, `thread_pool.cpp`) was found this way, not by static analysis. Static
  analysis (grep/read) missed the `default_tensor_pool()` singleton and the two-way
  `tensor_math` coupling entirely.
- **Build the new small target in isolation before touching the big amalgam.**
  That sequence exposed the dependency closure cleanly; the downstream
  `canvas_tables` builds and regression targets were run afterward.
- A background shell command from earlier in this session (`b6fi16bxq`, an old kpath
  rebuild, already superseded/irrelevant) came back with `status: stopped` / "no
  completion record" on a later check — a reminder that this environment's background
  tasks can be silently dropped across session boundaries, so **don't trust an
  in-progress build's fate without re-checking its log file directly**, which is what this
  handoff's "Observed behaviour" section is grounded in (the actual log content, not a
  remembered exit code).

## Historical resolution checklist

1. **Completed:** add `src/common/thread_pool.cpp` to `nodus_tensor_core`'s source list in
   `nodus/CMakeLists.txt`, and add the matching exclude regex to
   `nodus/cmake/auto_sources.cmake` (mirror the existing entries — `thread_pool` lives at
   `src/common/thread_pool.cpp`, not under `abstraction/`, so it needs its own regex line,
   not a suffix added to the existing `tensors/abstraction/(...)\.cpp$` one).
2. **Completed:** rebuild `nodus_tensor_core` in isolation and resolve its
   dependency closure.
3. **Completed:** build `canvas_tables` and
   `canvas_tables_static` against it (`cmake --build build --config Release --target
   canvas_tables canvas_tables_static`), then the full regression suite already used to
   validate the earlier mem-backend fix: `test_mem_backend_manifest`,
   `test_edge_backend_coordination`, `test_repo_package`, `test_tool_dll`,
   `test_tensor_backend_smoke`, `tensor_registry_test`, `abstract_tensor_pool_torture`,
   plus `test_kpath_raster_tool` (the one that motivated all of this).
4. **Still useful as a narrower regression:** add the exact plugin-DLL
   `AbstractTensor` creation → static-host read scenario if no existing plugin
   test names that boundary explicitly. Current handle-transfer, registry,
   tool-DLL, and package tests cover the surrounding substrate, but should not
   be mislabeled as this exact case.
5. **Current frontier:** the deferred FS-backend map split remains a separate
   latent concern. Program interchange is no longer wholly parked:
   ProcessGraph → Nodus GraphIR/ToolIR and equal-shape FusedProgram → prepared
   Tensor Calculator paths are both implemented and tested.

## Prompt History (verbatim instructions that shaped this work, most recent first)

- *"let's shift back to working on the relationship between Turing and Nodus... did we get
  stuck on step 1? where are we now"* — followed by a sharp correction: *"you were saying a
  lot of stuff that looked like 'I'm going to invent some shit bubbled on top of things I
  don't look into the nature of' so I didn't know everything I wanted was already done and
  all you need to do is fill out translation tables and data contracts"* — this correctly
  called out that I was about to propose new high-level-IR infrastructure without having
  exhaustively checked whether nodus already had it (it partially does, under names like
  `tool_ir.h`/`graph_ir.h`/`abstract_op_graph.h`/`GraphRuntime` — not yet fully read/
  reconciled against turing's FusedProgramIR).
- *"we're sidetracked from what I need now more than anything, none of this matters as much
  as making sure the coding works to faithfully pass tensor data, not programs, yet."* —
  this is the instruction that scoped the current work down to exactly the
  `nodus_tensor_core` extraction and explicitly parked the program-IR/`graph_ir.h`
  investigation for later.
- *"in speaktome there is an agent ecosystem where you can document work, make a remark of
  it in the top level repo and then use it to note where you left off please"* — the
  instruction this document is fulfilling.
