# SpeakToMe project atlas

**Initial evidence snapshot:** 2026-11-11 02:55:47 UTC.
**Purpose:** map one project history without destroying component ownership.
**Companions:** [history](SPEAKTOME_HISTORY.md), [forensics](SPEAKTOME_FORENSICS.md),
[exact repository snapshot](SPEAKTOME_REPOSITORIES.json).

## 1. Four different maps occupy the same workspace

1. **Purpose map:** the owner's originating SpeakToMe ambition and the systems
   developed to support it. This is the reason the superrepo is one document.
2. **Source lineage map:** files imported, copied, rewritten, extracted, returned,
   or preserved as snapshots. This needs file and commit evidence.
3. **Runtime map:** which component actually invokes another, exchanges data,
   schedules work, or supplies a service. A design diagram alone cannot prove it.
4. **Git/legal map:** which repository tracks a path, which ref holds a change,
   what notices govern it, and whether publication is permitted.

These maps overlap but are not interchangeable. A subsystem can belong to the
project's purpose without being tracked by its outer repository. A copied
implementation can acquire an independent development history. A proposed
integration can be meaningful before it is implemented. A common origin does
not silently grant a common license.

## 2. Responsibilities and boundaries

### SpeakToMe: what the machinery is for

The original retained overview describes a full beam lattice, with human and
artificial agents operating inside the branching token space. Controllers,
scorers, graph movement, beam retirement, model distribution inspection, and
visualization serve that exploration. The October 2026 probability-geometry
thesis distinguishes transition probability from change in predictive state,
behavioral divergence from lexical fan-out, and uncertainty from geometry.

Sources: `speaktome/AGENTS/PROJECT_OVERVIEW.md`,
`speaktome/holographic_signal_proposal.md`,
`speaktome/SPEAK_TO_ME_PROBABILITY_GEOMETRY.md`,
`speaktome/VISION_FORWARD_BACKWARD_DIFFUSION.md`.

The codebase also preserves Laplace/DEC, tensor, CRT, printing, time-sync,
DataProcessor, and other experiments. Their inclusion is historically meaningful;
it is not proof that all of them are required dependencies of every search run.
Read `speaktome/AGENTS/CODEBASE_REGISTRY.md` for its local package organization.

**Own here:** language-model/search semantics and application-facing contracts
assigned to this repository. **Do not duplicate here:** Turing's compiler and
managed-time machinery, Nodus's graph ownership, or the machine implementation.

### Turing: numerical, executable, and scientific structure

Turing has its own extensive research program: tensor abstraction, numerical
precision and backend dispatch, source/SSA compilation, graph and memory
representations, managed scientific time, analog tape execution, Riemannian
blocks, cellular/physical simulation, and integrated Transmogrifier work.
It is not merely a decoder helper or the source of one browser demonstration.

The source compiler's sanctioned entry in the root guidance is
`src.compiler.fortran_c_shell.lower_ast_source_to_ssa`. Follow current Turing
instructions and hazard receipts rather than substituting a different compiler
entry or inferring success from a convenient inner function. The existing
managed-time system owns save/restore and transactional advancement. Engines
declare their state and participate in that system instead of introducing a
parallel snapshot/runtime convention.

Sources: `turing/README.md`, `turing/MODULES.md`, `turing/AGENTS.md`,
`turing/TEST_BASELINE_AND_HAZARDS.md`, `turing/docs/hazards/`,
`turing/docs/managed_time_runtime.md`, `turing/src/transmogrifier/`.

**Historical connection:** tensor files were imported from SpeakToMe in 2025;
the developed implementation returned in 2026. **Current boundary:** particular
compiler/runtime interfaces must be checked at the consumer and producer heads.
It is not enough that both sides contain a class named `AbstractTensor`.

### Nodus: native graph authority and composition

Nodus supplies a C++ graph editor/runtime, typed ports and rows, graph identity,
plugins and repository ingestion, memory/tensor handles, headless surfaces,
portable kernel IR, and inspection. Its historical library name `canvas_tables`
describes an origin, not the limits of the current system.

The July 2026 ProcessGraph import and TensorMath work provide concrete evidence
of Turing/Nodus contact. November contract-port and compile-host work adds a
further service boundary. This does not justify claiming every graph can already
be compiled, shipped, and run correctly through every backend.

Sources: `nodus/README.md`, `nodus/include/`, `nodus/src/`, `nodus/docs/`,
`PROCESS_GRAPH_TRANSLATION_HANDOFF.md`, `NODUS_BRIDGING_PROPOSAL_2026-11-10.md`,
`NODUS_MEOWDOKU_SERVICE_BACKBONE_AUDIT_2026-11-07.md`.

**Own here:** graph/tool identity, execution and host contracts. **Keep separate:**
search growth semantics, physical laws, and the compiler's source semantics.

### Pluck / spectral-analyzer: sensory and physical expression

The retained repository begins as a Constant-Q spectral analyzer. The current
README describes graph-driven audio, strings and instruments, coupled acoustics,
optical transport, cameras, sensors, and real-time rendering. Its original name
must not reduce the documented system to a spectrogram.

A useful distinction is between sensory presentation of a computation and the
computation's ownership. A Pluck camera can observe a world without becoming its
scheduler; a graph-driven instrument can consume signals without owning language
search. Current integrations must preserve those interfaces.

Sources: `spectral-analyzer/README.md`, `ARCHITECTURE.md`,
`VITRUVIAN_REPOSITORY_ATLAS.md`, `RAY_TRACER_INVENTORY.md` within that repository,
and outer `NODUS_PLUCK_HANDOFF.md`. The November Spectral marble-preview commits
are examples of specific integration, not evidence of a complete product.

### Geometry: continuous/discrete translation

Guardian Geometry Engine describes vectorized C++ geometry and DEC, delayed
computation, parametric domains, and graph relationships. Alpha/Kakarot and
SpeakToMe contain related geometry/YoungMan/Laplace history. Keep direct file
lineage separate from conceptual correspondence: a similar parametric or metric
idea is not proof that an implementation was copied.

Sources: `geometry/README.md`, `geometry/AGENTS.md`,
`GEOMETRY_NODUS_FUTURE.md`, `ALPHA_KAKAROT_LINEAGE_AUDIT.md`.

### engine_toy: real machines, not disposable demonstration code

This outer repository deliberately tracks `engine_toy/`. Source contains machine,
part, line, and port identities and a MachineSystem integrated with Turing's
existing dt and tensor systems. Physical laws, state contracts, engines, tools,
and viewers are substantive code. The word `toy` is not permission to replace
that implementation with a simpler approximation.

Separate the real physics program, its compile contract, a generated native
artifact, an inferred/surrogate model, and a browser viewer. A faster surrogate
or a compiled individual law is not proof that the full physical simulator was
lowered and validated. Historical frontier reports should retain their exact
commit/date rather than being turned into today's blocker without rerunning.

Sources: `engine_toy/`, `ENGINE_TOY_ARCHITECTURE_NOTES.md`,
`ENGINE_EQUATION_ANNEX.md`, `HONORARY_ENGINE_EQUATION_CATALOGUE.md`,
`HIERARCHICAL_PHYSICS_SIMULATOR_PROPOSITION.md`,
`engine_toy/HANDOFF_ENGINE_LOWERING_2026-09-16.md`.

### Token Orchard and the outer artifact system

`string_core/` is deliberately tracked here as Token Orchard: graph-native
integer string-core, service, and compiler-target work. Consult its README rather
than classifying it from a folder name.

`site/`, the root `index.html`, `server.go`, and `server_test.go` belong to the
outer repository's compiler-page/artifact system. Versioned bundles preserve
sources, emitted backends, WASM, mathematical models, and build receipts.
`PUBLISHING_BUNDLES_TO_ROOT.md` governs that workflow. A generated artifact is
useful evidence of a producer and a particular build; it is not by itself an
adequate landing page for all of SpeakToMe or all of Turing.

## 3. Git inventory at the captured heads

All 15 captured databases reported non-shallow histories. "Captured commits"
means commits reachable from the exact captured refs plus the canonical HEAD,
not every object ever created. See the forensic exclusions before using counts.

| Local canonical path | HEAD | HEAD ancestry | Captured commits | Tracked paths at HEAD |
|---|---|---:|---:|---:|
| `.` | `937de5dd4947` | 312 | 648 | 2,677 |
| `speaktome` | `10ed5f35ff59` | 1,021 | 1,038 | 4,308 |
| `turing` | `a90b9f5c2215` | 5,356 | 5,729 | 5,714 |
| `nodus` | `99fc81a09463` | 252 | 276 | 478 |
| `spectral-analyzer` | `edddb335630a` | 199 | 212 | 1,029 |
| `geometry` | `97128724fb47` | 95 | 100 | 172 |
| `amp` | `2b61525dc988` | 447 | 457 | 262 |
| `fftfree` | `642dcf442664` | 71 | 90 | 43 |
| `meta-nn` | `93fc1dc91472` | 76 | 76 | 191 |
| `pigeon` | `c8f4854c243b` | 12 | 12 | 13 |
| `socratic` | `6054198bc20f` | 24 | 24 | 142 |
| `tensor-calculator` | `34f9fec626ea` | 2 | 2 | 6 |
| `lfsavoider` | `c74122180973` | 28 | 31 | 28 |
| `wheelhouse` | `c34733fb0b56` | 19 | 19 | 12 |
| `pcb` | `b6c64e6e7c04` | 5 | 7 | 48 |

The captured canonical branches are not a guarantee that all work is merged.
For example, the SpeakToMe REST/SDK work is on a separate worktree branch at the
time of this pass. Its existence does not change what the canonical snapshot
contains. The JSON records all captured ref targets so a reader can distinguish
that state without guessing from a directory name.

### Other independent repositories and their relation

| Path | Observed role | Relationship evidence and caution |
|---|---|---|
| `amp/` | Python/C AudioGraph runtime and control-history work | Audio-engine branch; local AGENTS imposes its own testing constraints. Direct extraction from SpeakToMe not established by this pass. |
| `fftfree/` | Eigen-based FFT core, C API, dispatch and worker-budget controls | Computational component; Pluck has a vendoring receipt. Check exact source revision and license before distribution. |
| `meta-nn/` | Graph-shaped training and orchestration, explicit stages and gates | Its README calls it extracted; the exact originating commit/path remains to be reconstructed. |
| `pigeon/` | Proposed provenance-preserving bicameral logical/neural graphs | Relevant conceptual research; README proposal language is not a measured implementation result. |
| `socratic/` | Experimental concept/precept extraction and graph visualization | NLP/semantic-graph branch. No `origin` remote configured in the captured local database. |
| `tensor-calculator/` | Persistent native tensor/instruction prototype | Explicitly superseded. Its README says active calculation belongs to Nodus TensorMath and Turing's own FusedProgram lowering. Do not resurrect it as shared substrate. |
| `lfsavoider/` | LFS-removal/archive tooling | Repository-management support, not the application's compiler or graph engine. Never run destructive migration tools merely to investigate history. |
| `wheelhouse/` | Package-wheel preparation template | Environment/distribution support with its own submodule and notice obligations. On indefinite hiatus by the owner's decision of 2026-11-11. |
| `pcb/` | Electrofluid PCB/fluid GUI | Independent Git owner; historically confused with the outer remote. Relationship to overall ideas requires evidence, not a claim that this is the parent repository. |

### Tracked components, untracked material, and aliases

- `site/`, `engine_toy/`, `string_core/`: owned directly by the root Git tree.
- `transmogrifier/`: legacy standalone source snapshot. Its README directs new
  work to `turing/src/transmogrifier/`; unique unintegrated experiments may remain.
- `research/`: detailed cross-repository studies, currently ignored by the parent.
  Its README's old statement about an unrelated root remote is stale. It remains
  useful evidence, but a reader of the public root repository cannot assume those
  files have been published with it.
- `socio_karma_module/`: local Python/audio material, no immediate `.git` found.
- `bike_ble/`: local Bluetooth trainer/heart-rate API, no immediate `.git` found.
  The name and subject do not establish a source-transfer relationship to older
  DataProcessor/workout experiments. No hardware was contacted for this audit.
- `wasm-gallery/` and `wi/`: local artifact trees, not independent Git databases
  in this inventory. Provenance must be established at file/bundle level.
- `srv/`, `artifacts/`, `build/`, caches and temporary probes: operational or
  generated/local material. They are not automatically source, public content,
  or safe to commit. Do not inspect or publish secrets to explain architecture.
- `archives/`, `_quarantine/`, and `C:\Alpha`: archaeological leads rather than
  disposable noise. This pass did not inventory all of their contents.
- `turing-*`, `win-main*`, `.dt-*-baseline`, and similar paths: many are linked
  worktrees. The JSON groups aliases using Git's common-directory identity.
  Names that resemble worktrees but have no immediate `.git` are not assumed to
  be functioning worktrees.
- `speaktome-rest-20261111`, `.nodus-python-graph`,
  `spectral-marble-trace-20261107`: linked worktrees in the captured databases.

A directory's current absence of `.git` does not prove it never had history.
A linked worktree's separate working files do not make it a separate repository.
Nested third-party submodules have their own histories and were not counted as
independent first-party projects in this ledger.

## 4. Read the runtime composition without inventing a monolith

The July 25 branching-search design allocates responsibilities approximately as:

```text
SpeakToMe: search states, distributions, causal growth and publication
    consumes numerical/geometry facilities from Turing / AbstractTensor
    exposes graph material through Nodus identity, ports and inspection
    may be perceived/manipulated through Pluck sensory and physical systems

Turing compiler/runtime <-> Nodus graph/tool boundaries
            |
            +-> existing physical engines, including outer engine_toy
            +-> generated artifacts and thin viewing/control surfaces
```

This is a responsibility map, not a verified universal runtime call graph.
Specific interfaces have implementation evidence; the full composition still
requires concrete state, ABI, scheduling, transport, and parity checks.

November service extraction similarly needs a bounded description. REST handler
ownership can move into SpeakToMe while Turing retains authentication, deployment,
storage, and compiler responsibilities and Nodus retains graph/contract execution.
That move does not create a new simulator or prove live machine control. Read the
actual paired revisions and deployment receipt before declaring the change live.

## 5. Public presentation is not system ownership

The direction being explored is a distinct explanatory site per repository,
cross-linked as one navigable SpeakToMe body of work, with both interactive
navigation and ordinary static links. Turing's breadth merits its own presentation.
The SpeakToMe front door can include the product of the development/game-engine
work without reducing every other repository to a game demo.

This is owner direction recorded during the documentation work, not an implemented
site network or a settled page design. Meowdoku is a working name for part of that
work, not an established public product identity. Preserve serious scientific and
compiler documentation outside interactive experiences so it remains readable,
searchable, accessible, and usable by agents. See
[TURING_LANDING_PAGE_AUDIT.md](TURING_LANDING_PAGE_AUDIT.md).

## Licensing is a separate boundary

The owner described the common aspiration as "spiritually they're all MIT from
speaktome". That records intent and history. It does not replace the actual
notices, establish ownership of third-party material, or enact a license change.

During this same session, the owner subsequently authorized first-party MIT
licensing across the five central repositories. This is now recorded locally in
the following commits, with explicit third-party exclusions and LICENSING.md:

- Outer: `a84c7435f4f76dc560585329360cc984ca44721e`.
- Turing: `c34e50d08fe7aa64bc51e983cf52b143f47665c4`.
- Nodus: `af74ee5ebc58b1a55ee286da40a3024db3ad15d2`.
- spectral-analyzer: `25f6ae5262ef3b1236aeccab05d4036eeaa09bc7`.
- SpeakToMe REST branch: `9f4da00575368baffd4ee8dbbd94fa9ed3b73c5f`.

Read the current [outer licensing scope](LICENSING.md) and each component's own
LICENSE/LICENSING.md. Local commit presence is not proof of publication. Other
first-party repositories outside this five-repository change still require
separate review.

**Historical audit state, superseded for current first-party grants by the above
commits:** the parallel audit found the following at the inspected revisions:

| Component/revision | Observed notice | What must not be inferred |
|---|---|---|
| SpeakToMe `10ed5f3` (also REST worktree before SDK cleanup) | Root MIT, copyright 2025 SpeakToMe Authors; pyproject declares MIT | Not a grant for every bundled file or every other repository |
| Outer repo `937de5d` | Copyright 2026 sangderenard; all rights reserved, inspection/discussion permission and separate written permission for other uses | Not MIT or a general open-source permission grant |
| Turing `fd9a9903` | No tracked top-level LICENSE/COPYING/NOTICE found; isolated MIT headers and third-party material exist | No universal license inferred from those headers |
| Nodus `d3814f0` | No tracked top-level LICENSE/COPYING/NOTICE found; `stb_easy_font.h` includes its own alternatives | No blanket license inferred from vendored stb |
| spectral-analyzer `edddb335` | No tracked top-level LICENSE/COPYING/NOTICE found; fftfree vendoring receipt defers license selection | No distribution permission inferred from a vendoring receipt |

These are scoped observations from a separate inventory, not a compatibility
opinion. Newer commits must be checked before publishing a release. Other
components and nested dependencies need their own notice review.

SpeakToMe's third-party submodules include Apache-2.0, MPL-2.0, MIT, zlib-style,
BSD, GPL-scoped modules, and other terms at their pinned revisions. Those terms
remain relevant even when first-party code is MIT. The historical Garmin SDK
has separate restrictive terms. A local cleanup commit `2c7adc2078adc55361cc6a74424a011e47f05411`
untracks its SDK directory and ignores it while preserving disk files; it does
not purge earlier Git history, remove every Garmin-derived file elsewhere, or
prove publication of that branch. Check the license worker's receipt and actual
remote ref before relying on the cleanup.

No license file was edited by the documentation lane; the separate authorized
licensing lane made the changes listed above. Do not change notices,
strip attribution, or broaden permissions as a side effect of making the
project narrative coherent.

## 6. Agent routing and integration checklist

1. Identify the owning repository with `git rev-parse --show-toplevel` and inspect
   status, current branch, and common-directory/worktree relationship.
2. Read local AGENTS and current hazards/receipts. The root's guidance directs
   agents to SpeakToMe's guestbook for cross-project history, not authority to
   override another repository's instructions.
3. Find the existing subsystem and its sanctioned entry. Check both producer
   and consumer before changing a cross-repository contract.
4. Name the evidence level: design, source implementation, emitted artifact,
   built binary, focused passing test, complete passing integration, or deployment.
5. Keep changes in their proper repositories; stage only intended files/hunks.
   Do not absorb ignored children into the outer repository.
6. Preserve historical failures and superseded attempts with dates. Link a newer
   receipt rather than rewriting the old one as if it had always been correct.
7. Record the new work and exact relevant prompts under SpeakToMe's guestbook
   conventions. Do not include credentials or unrelated personal information.
8. When publication is authorized, verify the exact remote commit separately.
   A local commit, upstream ref, cached remote-tracking branch, and deployed
   service are four different states.
