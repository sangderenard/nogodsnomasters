# Claude catch-up review, 2026-11-10

Read-only. No builds, tests, commits, restarts or pushes were run for this
review. Hashes identify what was inspected, not an integration gate.

## 1. Where Claude left off

| Session (app) | Last active | Left state |
| --- | --- | --- |
| Game clone with WebAssembly (Fable 5.2) | 2026-11-10 14:51 | Meowdoku service work. Live-view fix `4b1032b6` is in turing main. The log-page worker (server log ring, `/api/log`, glyph-mesh log page, reset-to-defaults) died on the weekly rate limit; its branch `worktree-agent-abce6c236a4a86060` (3 commits, 2026-11-07) is NOT merged into turing main. Resumable after 5 pm Chicago today. |
| Provenance bootstrap continuation | 2026-10-21 | Tree clean at turing `913e4c2c`, pushed. Physical record posted at producers, audit zero on 13 probe modes. Open: the loop induction dtype must be declared before regions are planned (region signature is built before `declare_loop_scope` runs); no bootstrap run since `d5b43897`. |
| Push the bootstrap past its stops (cloud thread, parked 2026-10-29) | 2026-10-29 | `turing/docs/CONTINUATION_2026-10-29_BOOTSTRAP_STOPS.md`. Bootstrap stops at unit 076 `identity_spill._reindex_in_place`: `Any`-typed tuple rows as dict keys. Needs a decision (union-of-tuple-shapes key representation). Also open: whole tuple as loop variable, `.items()` pair projection misapplied, codec (pickle replacement) compile blocked on raise handling (PR 1166 abandoned). |
| Turing review and wait / Turing continuation document | 2026-10-08 to 10-17 | Convergence docs 12 to 32; superseded by the above. |

Standing Claude-lineage receipts on turing main after that: 2026-11-01
bootstrap reconciliation (standalone-raise ingestion repaired, llvmlite dep),
2026-11-06 dt-system controller lowered to NumPy (Opus 5.5), 2026-11-06 `math`
float-function dtype repair (atan2 quotient now lowers to Wasm).

## 2. Repo evolution since mid-October

Pushed state: turing, nodus, spectral-analyzer, speaktome and the root branch
`codex/live-string-core-demo` all have zero unpushed commits. The unpushed
material is UNCOMMITTED:

| Repo | Uncommitted |
| --- | --- |
| turing | 51 files, +5850/-925: compiler files (precompile_to_ssa, fortran_c_shell, identity_concordance, glsl_*, ssa_wasm/webgpu backends, symbolic_equation_compiler, process_graph_autograd, topological_reducer) plus Meowdoku studio/marble files. Concurrent agent is live in here today (hazard entries stamped 2026-11-10). |
| nodus | contract_negotiator.h, contract_negotiator_tool.h, contract_socket.h (untracked), thread_manager.cpp cycle fix, headless ABI edge token calls, artifact_importer success-flag fix, tests. Two docs dated 2026-11-09. |
| spectral-analyzer | FirstHitLoan capture/resume seam, OBJ/MTL intake, light shape physics; FIRST_HIT_CLIENT_CONTINUATION.md untracked. |
| speaktome | 14 untracked experience reports (Nov 8-10). |
| root | string_core trap148 tools, scratch_vjp_probe.py, site/demos/living-data-map-play (Springtail restoration), two PR bodies. |

Timeline:

- Oct 16-24: honorary catalogue (Helmholtz light), Token Orchard / String Core
  in codex lanes; turing Windows LLVM row export repair; Nodus graph-definition
  v1, Turing row tools (C/LLVM/Fortran) adopted as Nodus rows, SPIR-V row,
  Vulkan SSBO FIFO proof. Nodus continuation handoff 2026-10-23 declares
  sending ProcessGraph into Nodus a category error: Turing lowers to SSA and
  the product becomes a row.
- Oct 25-27: three human commits in string_core recording agents refusing to
  remove identified problems. Dense codex hazard stream on Token Orchard,
  manager frames, native type audits.
- Oct 28-31: Claude threads on int intervals, u64 formals, exceptions; codex
  compressor integration; threading / dual-IR / synchronization SSA docs
  (`turing/docs/compiler_convergence/*2026-10-31*`).
- Nov 1-5: bootstrap reconciliation report; compressor runs 53-71 (codex);
  Meowdoku web (3D tiles, Python state machine, 2026-11-05).
- Nov 6-8: dt controller NumPy lowering; Meowdoku game explosion (bullet hell,
  mobroad, purrline, wordle, Marble Madness, dice, fountain, studio) via
  worktree-agent merges; service backbone audit; Marble Studio lighting rig.
- Nov 9-10: Nodus/Turing tensor boundary qualification; native job pool /
  contract negotiator / socket; FirstHitLoan client-job bridge released as
  opt-in; dozens of studio hazard entries per hour from the concurrent agent.

## 3. State against the three priorities

### Priority 1: compiler bootstraps, lowers the dt system, eats Meowdoku

What is real:
- Whole-program bootstrap tools exist (`tools/bootstrap_compiler_whole_program.py`,
  `whole_program_compiler_bootstrap.py`, `bootstrap_compiler_concordance.py`).
  Last measured frontier: stop at unit 076, `Any` tuple dict keys. Nothing
  natively verified from the bootstrap yet (research doc 2026-10-21 still true).
- dt system: `llvm_dt_system` controller now runs as a NumPy source-to-source
  product by default (`fused_program_python_backend.lower_functions_to_numpy`),
  3000-tick bit-identical, 1.8 to 0.66 ms/step. This is a Python-dialect
  product, not native. The dt system itself (`src/common/dt_system`) has not
  been lowered; pieces remain linked native calls.
- Meowdoku: 87 Python files, 39k lines. Compiled today: fountain laws (WGSL
  `fast` + Wasm `develop`), entity laws (`entity_motion.py`, `policy_wasm.py`,
  `split_lowering.py` JS control + Wasm regions), `damped_axis` lowered to C
  through the Nodus row compiler with zero concordance disagreements (DLL
  compile blocked by network policy in that environment, not here).
- Known compiler defects on the Meowdoku path: contact WebGPU build splits
  into 12 regions because generated arithmetic routes through lexical
  Load/Store nodes with no direct numerical edges (hazard 2026-11-10 07:50);
  Wasm lane lacks sin/cos/exp/FloorDiv spellings; C lane lacks atan2;
  `math.hypot` refused by final concordance.

Decisions waiting on you (nobody may pick these):
1. Representation of a union of tuple shapes used as a dict key (bootstrap stop 076).
2. Whether the row-handle record may extend to loop targets (whole tuple as loop var).
3. Owner for raise handling (codec compile).
4. Loop induction dtype: post early in `lower_control_sections_to_ssa`, or unlock `identity_concordance.py` so `declare_loop_scope` owns it.

### Priority 2: Marble Studio

Owned by you and the concurrent agent. Turing main at `f998b80f` (Nov 8) plus
a large dirty tree being edited today. I did not touch it and will not. Only
cross-cutting note: that tree also holds compiler-file edits (list above), so
a compiler PR from a fresh worktree is the safe shape, never an edit in place.

### Priority 3: Nodus bridging and client/server work sharing

What is real:
- Binaries described into rows: `nodus/tools/turing_row_tool_compiler.py`
  (request JSON: row_entry, columns, backend, extraction_contract) produces
  C/LLVM/Fortran products with `NODUS BUFFER ABI V2`; `artifact_importer.cpp`
  generates the wrapper that keeps AbstractTensor handles at the ports;
  graph-definition v1 round-trips definition -> native graph -> definition.
  Missing: the automated step from an arbitrary program's binaries plus an
  API description to a row request. Today every request is hand-authored
  JSON; no API-description ingester exists (grep found only hspir_outline.md).
- Messenger loop / work sharing: `contract_negotiator.h` (retained unaccepted
  contracts, offer per market turn, loan generation, at-most-once result),
  `contract_socket.h` (72-byte v1 loopback protocol), two-module ThreadManager
  graph test passed under direct MSVC compile. Spectral `FirstHitLoan` +
  `client_ray_relay.py` + Nodus FIFO: 256 rays bit-exact through browser Wasm,
  but 13.7 s cooperative vs 0.012 s server-only. Correctness proven, no speedup.
- Why it is not efficient yet (your framing holds): each compiled standalone
  piece is its own boundary (mailbox 4.6 s, native step 5.2 s, client 6 s,
  overlapping); the browser side is per-game hand-wired (`compute.js`,
  `share.js`, `client_ray_participation.js`) with no unified site-building
  class; and ThreadManager still ignores typed transfer return values around
  thread_manager.cpp:894 and :1429 (absent input may fabricate zero; full
  output may drop a result). The scheduler cycle fix and these are
  uncommitted and their CMake build never ran (blocked by network policy).

## 4. Hygiene items (no action taken)

- turing: `worktree-agent-abce6c236a4a86060` (log page) and
  `worktree-agent-acae7257c8e4ec853` unmerged.
- turing: stray file `NUL` and `.claude/` untracked at repo root.
- root: `nogodsnomasters` is 6 behind origin; `wip/2026-10-03-inflight` 1 ahead (unverified checkpoint).
- Memory note: `project-llvm-dt-system-is-its-own-lowered-source` still applies; the NumPy product cache lives under `build/dt_numpy`.
