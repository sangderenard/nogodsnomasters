# Handoff: nodus ↔ spectral-analyzer ("pluck") integration

Written 2026-07-25 for whichever agent picks this up next from this coordinating folder
(`C:\dev\Powershell`). This is the "start here" briefing — read this before touching either repo.

## Where things are

- **nodus** — `C:\dev\Powershell\nodus`. C++ node-graph editor/runtime, git repo, branch
  `nogodsnomasters`.
- **spectral-analyzer** ("pluck") — `C:\dev\Powershell\spectral-analyzer`. **Was** at
  `C:\Users\alber\Downloads\spectral-analyzer` (wrong location); being moved here via `robocopy
  /MOVE` as of this writing. If that folder isn't at the path above yet, or looks incomplete, the
  move may still be finishing or may have needed a retry — check for `C:\Users\alber\Downloads\
  spectral-analyzer` still existing (partial move) before assuming something's wrong. If the move
  ever needs a fallback: check spectral-analyzer's `.gitignore` first (a large fraction of its
  ~40GB / ~30k files is almost certainly generated render output / build cache that doesn't need
  to move at all, only the tracked source does), or commit+push+clone fresh into this folder.
- **Live plan** — `C:\Users\alber\.claude\plans\eager-leaping-wall.md`. This is the actual approved,
  in-progress implementation plan. Read it in full; this document is a summary, that one is the
  spec.
- **Session memory** — `C:\Users\alber\.claude\projects\C--dev-Powershell-nodus\memory\`. Durable
  notes from the session that produced this plan: `MEMORY.md` is the index. Worth reading
  `project_nodus_pluck_architecture.md`, `feedback_repo_boundary_discipline.md`,
  `feedback_kpath_vs_canvas_rasterizer.md`, and `project_nodus_tool_library_plan.md` specifically —
  they carry reasoning that isn't fully repeated below.

## The relationship between the two repos (get this right, it's load-bearing)

**Not a feature-pick merge.** Nodus is a development environment and its runtime — meant to be
general enough to work as an OS. Pluck (spectral-analyzer) is a real-time, gamified 3D
world/experience (walk around, camera/optical stations, a fabricator, a playable guitar) built on
top of that environment, with genuinely good, complex rendering (five separate ray tracers, a real
parametric lens/camera compiler, a 48-shader GLSL/BDPT/VCM pipeline). Pluck's rendering is **not**
being replaced or ported into nodus. The plan is narrower: pluck's *scheduling backbone* (today:
`globals_renderer.py`'s dispatcher + a fixed T1–T5 ray pipeline) is meant to eventually be
expressed as nodus `GraphRuntime`/`ThreadManager` graph nodes, so nodus's computation engine
becomes the backbone underneath pluck — while pluck keeps its own identity as the packaged
experience layered on top. Two repos, two purposes, staying decoupled.

A prior integration attempt (three "Nodus ITool" DLLs under a now-deleted
`spectral-analyzer/csrc/nodus_optical_tools/`) was built and **reverted** — documented in
spectral-analyzer's own `NODUS_TOOL_BRIEF.md` — because it used private per-tool FIFOs instead of
real shared graph edges, never called the actual `edge_publish`/`edge_consume` host ABI. Read that
file before attempting anything that rhymes with it.

## What this specific plan is building

Nodus currently has almost no tools (3 hand-registered builtins total), and its own `kpath` vector-
renderer/machine-pathing substrate has only one narrow function exposed as a tool. The immediate
goal is a clean, general way for **any repo** (spectral-analyzer first, but designed for a much
larger future ecosystem of repos wanting to plug into nodus) to contribute tools to nodus without
nodus's core accumulating bespoke per-repo code.

Key design points (full detail in the plan file):

1. **Nodus already has two tool-creation tiers**: graph-collapse (author a subgraph out of a small
   closed set of primitives — `ModuleToolKind` — actualize it into a compiled tool; exists because
   that closed primitive set is what makes a graph *translatable*, via a real but currently mostly-
   scaffolded multi-backend compiler: `src/kernel_isa.h`'s `nodus::spirv::KernelIR`, a minimal
   ~20-op portable ISA explicitly built for SPIR-V lowering, plus `translation_matrix.h`'s
   runtime-pluggable backend registry) and manual hand-written tools in nodus's own repo.
2. **This plan adds a third tier: repo ingestion.** An external repo carries its own
   `nodus_package/` folder (tools + type/edge "contracts"), and nodus gains a new manifest format
   (`GP_RepoPackage`, sibling to the existing `GP_ModuleLibrary`) plus an ingestion routine that
   loads the whole thing as one named, listable, unloadable-as-a-unit package — not N anonymous
   DLLs with no sense of provenance.
3. **Capability tags, not an IR-compliance gate.** A tool wrapping already-compiled opaque native
   code (like anything in spectral-analyzer's `csrc/`) can never be SPIR-V-translatable, and that's
   fine — it just has to be honestly tagged (`CAP_ISA` / `CAP_BINARY` / `CAP_BACKWARD`, a small
   closed numeric enum for now, not strings — no general string-interning utility exists in nodus
   yet) rather than silently assumed portable.
4. **Cross-repo boundary discipline, flipped by which repo is compiling**: whichever repo's package
   source is being built vendors the *other* repo's tiny stable ABI-contract headers and links
   normally against its own repo's own artifacts; dynamic runtime `LoadLibrary` is reserved only
   for the one seam where the two repos' compiled outputs would otherwise have to know about each
   other's build order. For spectral-analyzer's `nodus_package/tools/*.cpp`: link
   `base_rasterizer_c` normally (same repo), vendor a few of nodus's headers
   (`tool_api.h`/`value_types.h`, likely genuinely header-only — worth confirming their transitive
   includes are too) — no requirement that either repo be built before the other.

## Progress so far (check the plan file's task list / these files for current state)

- `include/tool_api.h` — `CAP_ISA`/`CAP_BINARY`/`CAP_BACKWARD` enum added; `RegisterTypesFn`
  typedef and `NODUS_PLUGIN_REGISTER_TYPES_NAME` convention added. **Done.**
- `src/plugin_loader.cpp` — `PluginLoader::load_module` now looks up an optional `register_types`
  export and calls it once at load time, before any tool instance is created. **Done.**
- Not yet started: `include/repo_package.h`/`src/repo_package.cpp` (the `GP_RepoPackage` manifest +
  read/write + ingestion routine), CMake wiring, verification tests, the kpath-rasterizer plugin
  tool (`plugins/kpath_raster_tool/`), and spectral-analyzer's actual `nodus_package/` with the
  four `base_rasterizer_c` wrapper tools.

## Also worth knowing (unrelated but live in the same session)

- Earlier this session, kpath's scatter-rasterization path (`rasterize_program_scatter_with_spatial_kernel`,
  `src/common/tensors/abstraction/kpath/kpath_raster.cpp`) was restored from a 6-month stub and a
  real bug in `build_axpby_plan` (`src/common/tensors/abstraction/tensor_math.cpp`) was fixed —
  it was never mapping backend memory before use, a null-pointer crash. Both are verified working;
  `kpath_relgeo_scene_demo.exe` now produces correct PNGs via both the kernel-transform and scatter
  paths. This is *why* kpath's rasterizer was picked as the first same-repo plugin-tool target —
  it's freshly proven-working, not theoretical.
- `tests/tensor_backend_smoke_test.cpp` has a real, currently-unfixed race: it asserts pointer
  identity across a free-then-reallocate of the same size class, but the allocator's async cleaner
  thread (`src/common/tensors/abstraction/in_memory_backend.cpp`) can legitimately coalesce the
  freed span first, breaking that assumption. User said not to worry about it for now
  ("if that test is failing but our test passed I don't necessarily care") — noted here in case it
  matters later, not urgent.
