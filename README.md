# C:\dev\Powershell workspace

This directory is a collection of independent repositories and experiments, not a
monorepo. The parent Git repository exists only to back up root-level orientation
and cross-repository coordination documents.

## Parent Git boundary

The root [`.gitignore`](.gitignore) ignores every immediate child directory.
Consequently, even `git add -A` at this level can stage only eligible root files;
it cannot recurse into `nodus/`, `turing/`, `speaktome/`, `spectral-analyzer/`,
`research/`, `_quarantine/`, or any other child tree.

This boundary prevents the parent repository from accidentally absorbing,
replacing, or publishing local-only files in child workspaces. It does **not**
back up those files: each child repository or local data directory still needs
its own backup policy.

This coordination repository is published as
[`sangderenard/nogodsnomasters`](https://github.com/sangderenard/nogodsnomasters)
on branch `nogodsnomasters`. The unrelated Electrofluid remote belongs to the
independent [`pcb/`](pcb/) repository.

## Current documentation priorities

| Path | Role | Start here |
|---|---|---|
| [`speaktome/`](speaktome/) | Multi-project Python environment: beam-search work, tensor utilities, DEC/Laplace, and agent tooling | [`speaktome/README.md`](speaktome/README.md) |
| [`spectral-analyzer/`](spectral-analyzer/) | “Pluck”: graph-driven audio, instrument, acoustic, optical, camera, and rendering research | [`spectral-analyzer/README.md`](spectral-analyzer/README.md) |
| [`turing/`](turing/) | Python compiler/runtime research: managed time, tensor abstraction, SSA, analog tape execution, cellular simulation, and Transmogrifier | [`turing/README.md`](turing/README.md) |
| [`nodus/`](nodus/) | C++ graph runtime/editor, plugin ABI, tensor substrate, KernelIR, and headless/browser-facing surfaces | [`nodus/README.md`](nodus/README.md) |
| [`research/`](research/) | Cross-repository architectural research for the Turing ↔ Nodus tensor and translation boundary | [`research/README.md`](research/README.md) |

The standalone [`transmogrifier/`](transmogrifier/) directory is a legacy source
snapshot. Active Transmogrifier development lives in
[`turing/src/transmogrifier/`](turing/src/transmogrifier/).

## Active cross-repository work

- [`NODUS_PLUCK_HANDOFF.md`](NODUS_PLUCK_HANDOFF.md) records the Nodus ↔ Pluck
  integration boundary.
- [`NODUS_TENSOR_CORE_EXTRACTION_HANDOFF.md`](NODUS_TENSOR_CORE_EXTRACTION_HANDOFF.md)
  records the now-green tensor-substrate extraction and its historical linker
  diagnosis.
- [`research/`](research/) contains orientation and architectural analysis. It is
  not, by itself, an implementation specification.

## Working safely here

1. Choose a subdirectory before building or editing.
2. Read that repository's `AGENTS.md` when present.
3. Run Git commands inside the intended repository and inspect status before staging.
4. Treat `_quarantine/` as reversible storage, not as part of the active ecosystem.

For the detailed directory inventory and agent routing rules, see
[`AGENTS.md`](AGENTS.md).
