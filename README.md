# SpeakToMe: the project and its superrepo

SpeakToMe began as an instrument for exploring a language model's branching
probability space instead of immediately collapsing it to one answer. The
compiler, tensor systems, graph runtime, managed scientific time, geometry,
physical machines, and sensory demonstrations developed around that ambition
are substantial systems in their own right. They are also the machinery that
makes the larger SpeakToMe project possible.

This outer repository, **nogodsnomasters**, is the project's historical and
architectural document, integration record, and the owner of selected actual
implementations. It is not merely a folder of unrelated demonstrations.
**Shared purpose does not mean shared Git ownership:** the component repositories
retain their own histories, instructions, licensing, and release boundaries.

The owner clarified this interpretation on 2026-11-11 UTC. The documentation
below connects that intent to recoverable Git history and source evidence. It
does not claim that the complete integrated product is already operational.

## Read the project at three depths

1. **Orientation:** this README and [the project atlas](SPEAKTOME_PROJECT_ATLAS.md).
   Learn what each system owns and how it relates to SpeakToMe.
2. **History and intent:** [the narrative timeline](SPEAKTOME_HISTORY.md).
   Follow origins, extractions, return paths, experiments, and convergence.
3. **Audit the evidence:** [forensic method and claim register](SPEAKTOME_FORENSICS.md),
   [repository snapshot](SPEAKTOME_REPOSITORIES.json), and
   [complete captured commit ledger](SPEAKTOME_COMMIT_TIMELINE.csv).
   The initial ledger contains **8,721 repository/commit records across 15 Git
   databases**, with full object IDs, parents, author/committer timestamps, and
   membership in each captured canonical HEAD. Coverage limits are explicit.

This is an initial, reproducible historical foundation, not a claim that every
source file, branch, abandoned experiment, or earlier archive has been audited.
The continuation plan is part of [the forensic record](SPEAKTOME_FORENSICS.md).

## The originating purpose and the developed systems

| System | Its own responsibility | Relationship to the larger project |
|---|---|---|
| [SpeakToMe](https://github.com/sangderenard/speaktome) | Search semantics, beam/FluxGraph exploration, model distributions, user/agent interaction | Originating application and research purpose; retained history and cross-project guestbook |
| [Turing](https://github.com/sangderenard/Turing) | AbstractTensor, compiler/SSA and backend lowering, managed time, integrated Transmogrifier, scientific execution | Developed numerical and executable machinery; a broad project deserving its own explanation |
| [Nodus](https://github.com/sangderenard/nodus) | Native graph identity, typed ports, graph execution, host/plugin contracts, inspection | Graph composition and execution boundary; not a replacement language-model search policy |
| [Pluck / spectral-analyzer](https://github.com/sangderenard/spectral-analyzer) | Audio, instruments, optics, cameras, sensors, rendering | Physical and sensory systems through which computation can be perceived and manipulated |
| [Guardian Geometry Engine](https://github.com/sangderenard/geometry) | Vectorized geometry, discrete exterior calculus, parametric and discrete representations | Geometric research branch and historical substrate; integration must be proved per boundary |
| [engine_toy](engine_toy/) | Machines, parts, lines, ports, engines, physical laws and simulations | Real machine implementation owned by this outer repository, consuming existing Turing systems |
| [Token Orchard](string_core/README.md) | Graph-native integer string core and compiler-target work | Another substantive outer-repository component, not an ignored scratch directory |
| [Published artifacts](site/) | Versioned program bundles, compiler outputs, inspection pages | Evidence and delivery surfaces for the systems that generated them, not the whole project's identity |

The strongest direct lineage is **SpeakToMe tensors -> Turing's imported tensor
package -> developed AbstractTensor -> SpeakToMe again**. The history document
shows matching Git blobs and exact commits. Other relationships have different
levels of evidence: direct import, implemented interface, preserved design, or
owner-stated intent. They are not flattened into one invented inheritance tree.

## A short history

- **Before the retained SpeakToMe history:** DataProcessor, Alpha, and Kakarot
  preserve older interpolation, geometry, process-flow, and simulation ideas.
  Existing archaeological reports describe them; their dates and transfer paths
  are not all independently established by this pass.
- **June 2025:** SpeakToMe's earliest retained commit already describes humans
  and agents operating inside a full branching token lattice. Tensor abstractions,
  beam controllers, visualization, DEC/Laplace, and agent records coexist there.
- **August 2025:** Turing imports the tensor package from SpeakToMe, explicitly
  documenting its origin. Its compiler, tensor, simulation, and graph research
  subsequently develop well beyond that imported state.
- **July 2026:** the evolved tensor implementation returns to SpeakToMe;
  bidirectional FluxGraph work develops a persistent search region. The outer
  repository is established and records how SpeakToMe, Turing, Nodus, geometry,
  and Pluck compose.
- **August-November 2026:** compiler and runtime work, machine implementations,
  generated pages, graph boundaries, sensory demonstrations, and service work
  accumulate across their proper repositories. Some integration boundaries are
  implemented; others remain proposals or partially verified work.
- **November 2026:** the owner explicitly identifies the whole superrepo as the
  document of SpeakToMe. This corrects the old orientation that called these
  projects unrelated, without erasing their independent engineering identities.

## Git ownership and default branches

The root [`.gitignore`](.gitignore) ignores immediate child directories **except
explicitly tracked children**: `site/`, `engine_toy/`, and `string_core/`.
Root-level documents, the loopback server, and these exceptions belong here.
Other children remain opaque to the parent. Do not force-add them or assume a
root commit backs them up. Even here, inspect the index before staging: the
tracked exceptions make a blanket `git add -A` unsafe during concurrent work.

The customary default branch is `nogodsnomasters` for this outer repository,
SpeakToMe, Nodus, and spectral-analyzer; Turing uses `main`. A local worktree may
be on a differently named branch. Verify its current branch and the actual
remote target before publishing. A cached `origin/HEAD` is not live proof of the
remote default or of publication. See the snapshot for the captured state.

There are many `turing-*` directories. Most are linked worktrees sharing Turing's
Git database, not distinct projects. The atlas distinguishes repositories,
worktrees, snapshots, tracked components, and local-only material.

## Status, evidence, and licensing

- A commit message records an author's description, not a successful runtime
  test. A generated shader, emitted source, built binary, working demo, and
  verified end-to-end product are different achievements.
- Historical test receipts retain their commit and date. They do not certify
  today's HEAD. No compiler, simulation, deployment, or hardware test was run
  for this documentation pass.
- The owner separately authorized first-party MIT grants in the five central
  repositories; third-party terms remain independent. Read
  this repository's [LICENSE](LICENSE) and [licensing scope](LICENSING.md),
  each component's actual notices, and
  the licensing boundary in [the atlas](SPEAKTOME_PROJECT_ATLAS.md#licensing-is-a-separate-boundary).
  SpeakToMe's root MIT license is not a blanket license for all bundled content
  or the entire superrepo.
- Turing's repository documentation, Turing's public demonstration page, and
  this repository's compiler gallery are different surfaces. The initial
  [landing-page audit](TURING_LANDING_PAGE_AUDIT.md) records that distinction.
  It does not implement or authorize a redesign.

## For agents

Read [AGENTS.md](AGENTS.md), then the destination repository's local guidance.
Start with the existing implementation, its current receipt, and its history.
Do not construct a smaller substitute and describe it as completion of the
existing system. Preserve independent Git boundaries, concurrent edits,
historical evidence, and the distinction between intent and measured behavior.

## Local compiler-page server

The root repository also owns the small loopback-only server for the published
compiler inspection page:

```powershell
go run .
```

Open `http://localhost:8787`. The page can upload a trusted `.py` file to the
local Turing compiler and browse every page bundle already prepared beneath
`site/programs/`. Generation is deliberately restricted to loopback clients;
compiling Python source is a trusted local operation, not a public web service.

Generated programs use one standard, content-versioned layout:

```text
site/programs/<slug>/versions/<version>/
  bundle.json
  index.html
  source/python_source/<upload>.py
  source/<backend>/<artifact>
  wasm/<module>.wasm
  math/sympy-process-model.json
  build/compiler.log
```

The Turing page-builder commands default to this repository root, even when
launched from inside `turing/`. They reject `turing/` itself as a gallery
destination so generated pages cannot silently split across both repositories.

The server derives the gallery by walking valid `bundle.json` files in this
tree. There is no hand-maintained gallery index. A source may provide a literal
`TURING_PAGE` dictionary for `entrypoint`, `title`, `slug`, probe `feeds`, feed
expressions, dimensions, and compiler loop options; the upload form can override
the common identity and probe fields.

For the detailed directory inventory and agent routing rules, see
[`AGENTS.md`](AGENTS.md).
