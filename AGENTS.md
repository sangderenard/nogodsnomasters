# `C:\dev\Powershell` — top-level workspace map

**This is not a project. It's a junk drawer.** `C:\dev\Powershell` is a personal
scratch workspace where dozens of unrelated projects live side by side in
whatever state they were last left in — active, abandoned, half-migrated,
duplicated, or zipped up as a backup. There is no single product, no shared
build system, and no shared purpose across the subfolders below. Treat each
one as its own independent repo that happens to share a parent directory,
not as parts of a monorepo.

## Git status of this directory

This folder is a root-only coordination repository on branch
`nogodsnomasters`, with `origin` at
`https://github.com/sangderenard/nogodsnomasters.git`. The repository tracks
only files located directly at this level, while `.gitignore` treats every
child directory as opaque. The unrelated Electrofluid repository and remote
belong to `pcb/`.

Do not force-add ignored directories or weaken the root-only ignore rule. Child
projects retain their own Git histories and local-only files; this repository
documents how they relate but does not own or back up their contents.

## Where agent behavior guidance actually lives

This file is a map, not a style guide. If you're an agent looking for rules
on how to behave, test, install dependencies, or navigate this tree, **go to
[`speaktome/AGENTS.md`](speaktome/AGENTS.md)** and its companions:
- [`speaktome/AGENTS_FILESYSTEM_MAP.md`](speaktome/AGENTS_FILESYSTEM_MAP.md)
- [`speaktome/AGENTS_TESTING_ADVICE.md`](speaktome/AGENTS_TESTING_ADVICE.md)
- [`speaktome/AGENTS_DO_NOT_PIP_MANUALLY.md`](speaktome/AGENTS_DO_NOT_PIP_MANUALLY.md)

A few other subfolders (`amp/`, `ampgit/`, `geometry/`, `turing/`, `codex/`,
`codex-personal/`) also carry their own `AGENTS.md` — when working inside
one of those, its local file governs that folder, not this one.

## Subfolder / root-file profile

| Path | What it is |
|---|---|
| `.claude/` | Claude Code local settings for this directory. |
| `.git/` | Git metadata for this folder's repo (see remote note above). |
| `NODUS_PLUCK_HANDOFF.md` | Cross-project coordination doc for the active `nodus` ↔ `spectral-analyzer` integration effort — the one file at this level that's actually meant to tie two subfolders together. |
| `NODUS_TENSOR_CORE_EXTRACTION_HANDOFF.md` | Resolved `nodus_tensor_core` substrate-extraction record (2026-07-25, green update 2026-07-26). The historical linker failure is retained for archaeology; its opening status block describes the current tensor/calculator boundary. |
| `research/` | Cross-repo research library on abstract-tensor→any-language translation (turing ↔ nodus), the tensor-subsystem substrate, and the UI-as-structure architecture rationale. Start at `research/README.md`. |
| `_quarantine/` | Noise, orphaned archives, third-party tool checkouts, and superseded duplicates moved out of the way (2026-07-25). Not part of the ecosystem — see the section below. Nothing depends on it; it can be deleted once you're sure you don't want any of it back. |
| `amp/` | Python/C audio graph runtime ("AudioGraph" control-history driven). Has its own `AGENTS.md` with a strict no-ad-hoc-smoke-test policy. |
| `fftfree/` | Header-only C++ FFT library (Eigen-based, Cooley–Tukey/Stockham, C API). |
| `geometry/` | "Guardian Geometry Engine" — C++ vectorized geometry/DEC (Discrete Exterior Calculus) library. Has its own `AGENTS.md`. |
| `lfsavoider/` | PowerShell/bash tooling to strip Git LFS from a repo and archive large binaries to GCS instead. |
| `meta-nn/` | Python "graph-shaped meta network" training/orchestration workspace. |
| `nodus/` | C++ graph editor/runtime: canvas and table ABIs, plugins/repository ingestion, headless surfaces, KernelIR, and the shared tensor substrate. Actively developed and integrated with `spectral-analyzer`; see `NODUS_PLUCK_HANDOFF.md`. |
| `pcb/` | Python PCB layout / fluid-sim GUI application. |
| `pigeon/` | Python research project: "Bicameral Network for Provenance-Preserving... Neural and Logical Graph Refinement." |
| `socio_karma_module/` | Small Python social-simulation module (acts/graph/audio) plus one audio asset. |
| `socratic/` | Python "socratic_precept_graph" — spaCy/sentence-transformers concept-graph extraction from conversational text. |
| `speaktome/` | Multi-project Python repo (beam search controllers, `laplace`/DEC utilities, `tensorprinting`, etc.) sharing one venv. **Home of this tree's actual agent-behavior guidance** — see above. |
| `spectral-analyzer/` | "Pluck" — graph-driven audio/instrument simulation plus GPU-accelerated optics, cameras, and real-time 3D rendering. Its original CQT analyzer remains one subsystem. Being integrated with `nodus`; see `NODUS_PLUCK_HANDOFF.md`. |
| `transmogrifier/` | Legacy standalone snapshot of graph/tensor and `BindingMembrane` experiments. Active Transmogrifier code is integrated into `turing/src/transmogrifier/`; see `transmogrifier/README.md`. |
| `turing/` | Python compiler/runtime research spanning managed scientific time, abstract tensors, SSA and graph compilation, analog tape execution, cellular simulation, Riemannian blocks, and the integrated Transmogrifier package. Has its own `AGENTS.md`. |
| `wheelhouse/` | Self-contained template for building a Git-LFS-free pip wheelhouse; companion to `lfsavoider/`. |

If you land in this directory and aren't sure which subfolder you should be
in, ask — the sprawl here is real and getting the wrong one is easy.

## What's in `_quarantine/` (moved 2026-07-25, reversible)

None of these are ecosystem subrepos; all were untracked by this repo. Moved
aside to cut ~3 GB+ of duplicate/noise clutter from the top level. To restore
any of them, just `mv _quarantine/<name> .` back.

| Path | Why it was quarantined |
|---|---|
| `New folder/` | Empty, stray Windows-created folder. |
| `ampgit/` | Near-duplicate of `amp/` (same `AGENTS.md`, same filenames) — a second working copy (~445 MB). |
| `codex/` | Clone of OpenAI's Codex CLI (third-party tool); its remote points at `codex-personal.git` — a redundant checkout of `codex-personal/`. |
| `codex-personal/` | Your fork of OpenAI's Codex CLI — a third-party coding-agent tool used *for* dev, not part of the ecosystem. |
| `codex-rs/` | 4 KB orphaned fragment — just a `core/` subdir, no `.git`; a stray partial copy of the Codex Rust tree. |
| `debug_sandbox.rs` | Scratch Rust file calling `codex_core`/`codex_common` — a Codex debugging harness, wired to nothing here. |
| `emdashtoken.py` | One-off script scanning the `cl100k_base` tiktoken vocab for em-dashes. |
| `fftfree-old/` | Superseded "-old" duplicate of `fftfree/` (identical README) — ~2.5 GB. |
| `geometry.zip` | Redundant zip backup of the live `geometry/` repo. |
| `src.zip` | Orphaned archive of an unrelated `src/` python tree — no matching unpacked dir. |
| `turing_abstract_nn_renders/` | A single `input_prediction.gif` render output from `turing`, not a project. |
