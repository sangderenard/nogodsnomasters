# Boundary Error Register — turing ↔ nodus

**Date:** 2026-08-13
**Companion:** `BOUNDARY_FLUENCY_CAMPAIGN_2026-08-13.md` (rungs cite these by E-number)

Every entry below was **observed live** in this working session (or directly
demonstrated by an experiment run in it), not hypothesized. Each records the
symptom as it actually presented, the real cause, and the mechanical guard now
in place or required. The register exists because the symptoms are almost all
*silent or misleading* — the boundary's failure modes rarely announce
themselves.

---

**E1 — Scheduler silently disabled by default.**
Symptom: headless `run_to_quiescence` returns "quiescent" after one tick,
nothing executes, no error anywhere.
Cause: `thread_mgr_paused` defaults to true; only the GUI's canvas-head init
ever cleared it.
Guard: `nodus_headless_create()` unpauses explicitly (with the contract stated
in a comment). Rung harnesses must assert *work happened*, never just
"quiescent."

**E2 — Tool instantiation silently null.**
Symptom: rows exist, plugin ids are correct, execution runs — and every tool
row no-ops. `tool_registry_global().create(id)` returns null.
Cause: MSVC extracts static-library objects lazily; a self-registering object
(static registrar) nothing references never links into slim binaries. The
registration pathway compiled, passed its own unit test, and reached no real
binary.
Guard: registration must ride an explicitly-called path (the vocabulary/import
actualizers + package ingest), never a static-initializer side effect alone.
Harnesses assert non-null creation per id before trusting any execution.

**E3 — GUI-only state assumed headlessly.**
Symptom: zero rows execute for any module despite correct semantic row lists.
Cause: the row-execution loop bound came from the per-module *rendering*
table (populated only by a GUI sync function; 0 headlessly) instead of the
GraphRuntime-owned `io_rows`. Same family: `gp_table_get_row` consulted inside
the row loop for pointer-mode markers — harmlessly failing today, same trap
shape.
Guard: scheduling reads GraphRuntime-owned state only (fix in place, commented
at the site). Any new engine read in an execution path must answer: "who
populates this headlessly?"

**E4 — Stack currency mismatch no-ops safely (and silently).**
Symptom: a tool executes and does nothing; values seem to vanish.
Cause: typed raw stack — tools pop only their declared type. Scalar doubles
pushed to a tensor-pointer tool (or handles to a pointer-convention tool)
simply refuse the pop. Three currencies exist: scalar doubles, in-image
`AbstractTensor*` (registrar convention), cross-image handles
(`VT_ABSTRACT_TENSOR`, the generated-tool convention).
Guard: generated/imported tools speak handles exclusively; harnesses push the
currency the tool under test declares; mixed-family composition requires an
explicit conversion row, never coincidence.

**E5 — Cross-image singleton/arena split-brain.**
Symptom: handles valid in the host are unknown in the plugin (or vice versa);
pointer-identity checks fail mysteriously.
Cause: linking the static lib into one image and the DLL into another gives
two registries/arenas. The repo's own answer is `nodus_tensor_core` (shared,
export-all, deliberately the single home of tensor singletons).
Guard: every host and every plugin that touches tensors links
`nodus_tensor_core`; nothing tensor-bearing links `canvas_tables_static` while
its plugins link the DLL. The vocab checker's design (host uses
LoadLibrary+create_tool, links tensor core only) is the template.

**E6 — Build-config macros override in-source defaults.**
Symptom: an edited `#ifndef X / #define X 1` diagnostic default has no effect;
instrumentation appears dead.
Cause: CMake injects `-DX=0` on the compiler command line (`option()` +
`target_compile_definitions`), which beats any in-source default.
Guard: check `CMakeLists.txt` for the macro before trusting an in-source
default; enable via `-DX=ON` reconfigure. (The `NODUS_DEBUG_PLUGIN_DEPLOY`
detour cost a full build-run cycle.)

**E7 — Boundary-dirty input hangs recursive compilation.**
Symptom: automatic section compile runs indefinitely, no output directory, no
error.
Cause: a conditional `__import__("math")` (host-boundary call) inside the
numeric path of a fixture handed to the down-to-the-bone mode — the compiler
pursues call edges that have no compilable bottom.
Guard: mode selection is explicit: recursive mode only for provably
boundary-free sections; anything touching `HOST_BOUNDARY_OPERATORS` or
undeclared imports compiles boundary-respecting (interiors native, boundary
calls preserved). The boundary-clean twin of the same fixture compiled in 10 s.

**E8 — Deferred console logger falsifies event order.**
Symptom: log tails "show" what happened before a crash/exit; conclusions drawn
from adjacency are wrong. Host "PASS" printed before the engine's own lines.
Cause: `console_logger.h` redirects printf/fprintf into a background-flushed
queue (≤200 ms lag, final flush at exit); interleaving with direct stdout is
not chronological.
Guard: never infer ordering across the two streams; for order-sensitive
diagnosis use direct `stderr` writes or timestamps; treat "last line before
death" as unreliable near process end.

**E9 — Ambient state contaminates "fresh" contexts.**
Symptom: a brand-new context contains modules/rows nobody added (observed live
in the Rung-0 log: a stale two-module workspace loaded into the blank
context).
Cause: `gp_canvas_create` auto-loads `canvas_workspace.txt` (cwd-relative,
autosaved every 2 s by any GUI run); separately, canvas load overlays
`module_library/serialized/module_<N>.gpmod` blobs keyed only by module index
— unscoped, cross-canvas shared state.
Guard: headless harnesses run in controlled cwds and load their canvas file
explicitly after `canvas_clear_workspace` (the loader does clear before
rebuild — verified); the index-keyed library overlay remains an open
architectural hazard: do not rely on module indices as identity across
canvases.

**E10 — Moving crash sites mean memory corruption, not local bugs.**
Symptom: SIGSEGV at a different location each run; clean completion under a
debugger.
Cause: undefined behavior corrupting the heap upstream; the crash surfaces
wherever the corruption is next touched; debugger heap layout masks it.
Guard: when the crash site moves, stop reading tails and reach for
ASan/AppVerifier-class tooling; printf-bisection cannot localize this class.

**E11 — Entry points of one artifact disagree on parameter order.**
Symptom: (latent — caught by reading the contract, would otherwise be wrong
numbers or an access violation).
Cause: the fixture's control entry declares `t0, t2, t1` (a, b, scale) while
its numerical-region entry declares `t0, t1, t2` — declaration order is the
call order *per entry point*, and they legitimately differ.
Guard: wrappers marshal in the **selected entry's** declared order, never a
global assumption; the api-v1 `passing` field is honored mechanically ("a
caller that gets this backwards gets an access violation, not a wrong
number").

**E12 — dtype fault lines between layers.**
Symptom: no-ops or garbage when mixing layers.
Cause: the C tensor-op layer is double-only; generated vocabulary tools'
eigen bodies are float32-first; KernelIR's `ScalarType` is f32-only today
(f64 is a *named shortfall* in the lowering, not a silent narrowing).
Guard: every wrapper/harness validates dtype via `nodus_tensor_describe`
before payload access; conversions are explicit rows/instructions; type
simulation (the mandate's fluent-types requirement) closes these gaps
deliberately, never implicitly.

**E13 — Loaded-tool ids vs catalog ids (RESOLVED 2026-08-13).**
Symptom: canvas ROWS name bare ids (`abstract_tensor.tanh`);
`PluginLoader::load_module` registers timestamp-suffixed unique ids so
multiple loads coexist — bare references instantiated nothing.
Cause: two id spaces with no declared aliasing.
Resolution: first-class alias layer in `ToolRegistry`
(`register_alias`/`resolve`), wired into `load_module`. Policy, pinned by
`registry_alias_test` clause by clause: aliases resolve one hop to real
entries only; a real entry always beats an alias of the same name (in-image
reference implementations keep priority over loaded witnesses); re-registering
re-points (latest load wins; earlier versions stay reachable by unique id);
unregistering a target drops its aliases — no dangling names that quietly
stop meaning anything.

**E14 — A tool that bails leaves its inputs, and a naive harness reads its
own input back as "output".**
Symptom: Rung 2's first FAIL presented as `got 1.4` — exactly the `scale`
input. Looked like an ABI/order bug (E11); was not.
Cause: the wrapper's artifact load failed silently (see E15), its null-entry
guard returned without consuming the stack, and the harness popped the
topmost *input* handle as the "result."
Guard: harnesses assert the output handle differs from every input handle
(identity check now in `artifact_import_test`); generated wrappers report
load/symbol failure to stderr once instead of no-oping forever. Composes with
the register's first pattern: silence plus a plausible-looking value is the
worst combination.

**E15 — Dependency search flags are silently inert without a fully qualified
backslash path; runtime dependencies belong in the contract.**
Symptom: artifact `LoadLibrary` error 126 persisted even with the Fortran
runtime DLLs placed beside the artifact and
`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` requested.
Cause: the LOAD_LIBRARY_SEARCH_* flags only take effect for a path Windows
considers fully qualified — forward slashes defeat the check, and the flags
degrade silently to default search (which looks in the *application's*
directory, not the artifact's). Python's loader failed on the same
dependencies for its own reason (restricted DLL directories since 3.8).
Guard (CLOSED 2026-08-13): generated wrappers emit native backslash paths;
runtime dependencies are **declared in the api contract's metadata at compile
time** (`fortran_c_shell.py` now packs `runtime_dependencies` from the
toolchain's own bin directory — the producer knows) and the generated wrapper
**registers each declared dependency's directory with the loader
(`AddDllDirectory`)** rather than preloading files: a preloaded libgfortran
still needs the loader to find libquadmath for itself, while a registered
directory serves every import that lives there, order-independently. Proven
by re-running Rung 2 with the beside-copied DLLs deleted: the declared
contract was the only resolution path, and it passed.

**E16 — A dangling build target makes every later build a no-op, silently.**
Symptom: source edits appeared to have no effect; deleted test sections kept
printing; three consecutive "successful" builds reran stale binaries.
Cause: a `.cpp` was deleted while its `add_executable` target remained, so the
CMake **generate** step failed ("No SOURCES given to target"). The build then
reported nothing useful and MSBuild reused existing binaries. Filtering build
output for `error C[0-9]`/`error LNK` hid it — the failure was a *CMake* error,
matching neither pattern.
Guard: when a test's behavior contradicts its source, suspect the build before
the code; check the configure step's own output, not just compiler diagnostics.
Deleting a source file means deleting its target in the same change.

**E17 — The live nodus arena backend refuses `mul` on comparison masks.**
Symptom (2026-08-14): plain `ReLU6().forward(x)` on a default-constructed
tensor raises `NodusArenaError: binary mul failed: unsupported`
(`nodus_arena.py:158`, reached from `nodus_backend.py:244`).
Cause (partially diagnosed): `AbstractTensor.get_tensor()` now defaults to
the live nodus arena backend, and the masks produced by `greater`/`less`
reach the arena's binary `mul` support check, which answers "unsupported".
WHICH operand property triggers the rejection has not been diagnosed — the
mask-dtype theory is plausible but unverified. So ordinary piecewise
abstract_nn code (`mask * value`, the idiom every activation in
`abstract_nn/activations.py` uses) fails on the DEFAULT backend. The same
idiom also fails on the PURE backend for an unrelated reason
(`PurePythonTensorOperations.long_()` arity mismatch inside
`cast_bool_like` — its own defect, tracked separately); of the backends
tried, only numpy computed it.
Guard: reference computations in cross-ecosystem tests must pin an explicit
working backend (`get_tensor(..., cls=...)`); diagnose the arena's actual
rejection reason before changing either side.
FIXED (2026-08-14): diagnosis completed — comparisons deliberately return
`np.bool_` masks and NumPy's promotion keeps bool×bool at bool, a dtype the
arena's numeric executor refuses. `nodus_backend` now computes bool-dtype
binary ops in uint8 and casts the pull back (array path), and promotes via
`np.result_type` before the boundary (scalar path) — NumPy-parity preserved.
Verified: `mask*mask`, `mask*scalar`, and full `ReLU6.forward` on the live
arena backend match numpy exactly. The pure-backend `long_`/`to_dtype_`
receiver-convention defects (the "unrelated reason" above) are fixed the same
day: all dtype casts accept receiver-style calls, and `to_dtype_` recognises
both calling conventions; `ReLU6.forward` on pure now matches numpy.

---

## Reading this register

Three patterns account for nearly everything above:

1. **Silence is the default failure mode** (E1, E2, E3, E4, E9): the engine's
   defensive style prefers no-op over crash, which preserves uptime and hides
   causes. Harnesses must therefore assert positive evidence of work, never
   absence of error.
2. **Identity must be engineered, not assumed** (E5, E11, E13, E9's blob
   overlay): images, arenas, parameter orders, and id spaces each need an
   explicit, single point of truth.
3. **Diagnosis tools have their own failure modes** (E6, E8, E10): the
   instrument can lie — know when logs reorder, when macros are overridden,
   and when a moving crash site disqualifies local reasoning.
