# SpeakToMe: history, intent, and the development of its machinery

**Initial reconstruction:** 2026-11-11 UTC.
**Pinned evidence:** [repository snapshot](SPEAKTOME_REPOSITORIES.json).
**Full captured timeline:** [8,721 commit records](SPEAKTOME_COMMIT_TIMELINE.csv).
**Evidence discipline:** [method and claim register](SPEAKTOME_FORENSICS.md).

## 1. The correction that determines how to read the record

The owner's current explanation is not that an assortment of demos has recently
been renamed SpeakToMe. SpeakToMe was the originating purpose. Engines and
subsystems developed far enough to demonstrate themselves without yet realizing
that originating application. The outer repository gives those histories,
implementations, and relationships a common document.

The owner's statements are preserved verbatim below. They establish intent,
not a retrospective claim that every proposed connection already worked. Git
and source establish a narrower but substantial body of concrete continuity.

Earlier root documentation called the workspace an unrelated "junk drawer" and
said the parent owned only coordination files. Both descriptions were misleading
as current orientation: the owner identifies a common purpose, and the root
actually tracks site, engine_toy, and string_core. Independent histories and the
ignore boundary remain real. This reconstruction corrects the interpretation
without flattening those histories into a fictitious monorepo.

## 2. Before the retained SpeakToMe repository

The July 25 archaeological reports reconstruct earlier DataProcessor, Alpha,
and Kakarot work: time/media interpolation, arbitrary named axes, editable
process chains, asynchronous worker generations, cached refinement, parametric
geometry, YoungMan isosurface extraction, scheduling, simulated worlds, and
neural surrogates. They preserve questions that recur throughout the project:

- What lies behind the current sample or representation?
- How can resolution follow error, purpose, and available resources?
- How can long-running work remain inspectable, cancellable, and coherent?
- How do geometry, computation, and interaction describe the same process?

Sources: [pre-Kakarot lineage](PRE_KAKAROT_TO_SPEAKTOME_LINEAGE.md) and
[Alpha/Kakarot audit](ALPHA_KAKAROT_LINEAGE_AUDIT.md). The latter names `C:\Alpha`
as its archaeological source and distinguishes verified source, strong
correspondence, and proposal.

**Limit:** those reports were read as historical analyses; the complete Alpha
snapshots were not independently re-audited. Names such as "Level 8" are not
release dates. File timestamps are not proof of authorship or chronology. This
is not a proven single chain of Git ancestors.

## 3. June 2025: the purpose is already explicit

SpeakToMe's earliest retained root is `a00b59ecf438dad40687e51531f5dd6b35cacab9`,
authored June 7, 2025 with its original offset retained in the ledger. Its project
overview exposes the **full beam lattice**, with humans and artificial agents
operating **inside the branching token space**. The holographic proposal concerns
full probability distributions and viable trajectories, not only a final string.

This root already contains inherited material. It is the earliest retained commit
in this capture, not the birthday of the idea or proof that no earlier repository
existed. Reset/import history must remain visible.

The June record develops tensor abstraction across controllers and geometry,
multiple backends, accelerator management, compilation ideas, beam tools,
DEC/Laplace, visualization, and agent conventions. Many changes are small
iterations and merges. Their volume measures recorded activity, not correctness.

The guestbook preserves detailed observations and user prompts, including work
later performed in other repositories. Read it as dated evidence, not timeless
instructions or a substitute for current source.

## 4. August 2025: direct extraction into Turing

Turing's retained history starts August 3. On August 18,
`62d826b7c8f6959003b8e199e97e4d681d4d8874`, "early import of the tensors module",
adds `src/tensors`. Its README says:

> This package provides an abstract tensor interface and backend implementations.
> It was moved out of the main `speaktome` project to serve as a standalone
> codebase.

Git-object comparisons reproduce byte-identical backend files:

| SpeakToMe at `a7fa45c0` | Turing at `62d826b7` | Shared blob |
|---|---|---|
| `tensors/numpy_backend.py` | `src/tensors/numpy_backend.py` | `045d35696f0d9fe86e160619b07333ef52325b7f` |
| `tensors/torch_backend.py` | `src/tensors/torch_backend.py` | `f6663d0f52535b2b24317242fa3217dd97d2abf5` |
| `tensors/jax_backend.py` | `src/tensors/jax_backend.py` | `fb3eb3aa1c06fe949db2d9166ebe0bca69469c15` |

The SpeakToMe revision is dated June 13, before the Turing import. Matching blobs
prove identical bytes. The README and dates support direction. They do not prove
every later Turing module came from SpeakToMe or establish ownership of third-party
algorithms used by the backends.

Turing develops its own compiler, SSA, analog-tape, graph, simulation, and tensor
program. Integrated Transmogrifier carries executable-graph, scheduling, memory,
and binding experiments there. The standalone `transmogrifier/` snapshot remains
an archaeological comparison, not the current development destination. Exact
transfer history for every Transmogrifier file remains an audit task.

## 5. Specialization is not separation of purpose

The retained roots of geometry, Pigeon, amp, fftfree, Socratic, Nodus, and meta-nn
span June 2025 through January 2026. These are repository dates, not necessarily
idea-origin dates. Their READMEs and source address distinct problems: DEC and
vector geometry; provenance-rich neural/logical graphs; control-history audio;
FFT execution; concept graphs; native graph authority; and explicit training
pipelines.

Nodus's early history is canvas/table oriented. January entries introduce
AbstractTensor and relational geometry. Its current library name therefore
understates its scope. Similar vocabulary does not prove file transfer, and
this audit does not invent extraction commits for every sibling.

Pluck's spectral-analyzer root is April 7, 2026. The initial subject names
CQT/filterbank analysis, GPU inverse CQT, and asynchronous synthesis. Later source
and documentation span instruments, physical simulation, optics, sensors, and
rendering. The repository name remains an origin marker rather than the whole
system.

## 6. July 2026: machinery returns and the outer document forms

July 14 SpeakToMe `226924420fe735e37189bf45b1f73f99f3c371b6` vendors Turing's
developed AbstractTensor implementation back and migrates LookaheadController.
The inspected diff includes an expanded tensor tree and controller changes.
Follow-up `983997b2d9` migrates BeamSearch/Scorer and drops a compatibility shim.
The direction is a round trip:

```text
SpeakToMe tensor experiments
  -> extracted Turing tensor package
  -> independent development and generalization
  -> evolved machinery reintegrated into SpeakToMe
```

Also July 14, `40f7e1991d8f9bde2f93667e6bc6465b91fb3d14` adds a 350-line
FluxGraph implementation, a demo, and 174 lines of tests. Its subject, "a graph
that never commits", describes a persistent bidirectional search object with
flux/pressure dynamics. This pass inspected changed paths and counts; it did not
rerun the tests. Code, design, and measured behavior remain separate claims.

The outer repository starts July 25 at `d6dafbd`. Its early sequence separates
the Electrofluid remote, records Alpha/Kakarot and pre-Kakarot history, describes
geometry's Nodus role, and writes
[SPEAKTOME_BRANCHING_SEARCH_GEOMETRY.md](SPEAKTOME_BRANCHING_SEARCH_GEOMETRY.md).
That document allocates search semantics to SpeakToMe, numerical/metric/sampling
work to Turing, graph identity and scheduling to Nodus, and sensory interaction
to Pluck. It is architectural evidence, not proof that the entire composition
ran then.

July 26 Nodus `fc3c6f9` adds ProcessGraph reception as tensor tool graphs, with
implementation and regression-test changes. Later tensor/IR commits establish
further concrete contact. The short-lived tensor-calculator prototype is
explicitly superseded: it duplicated operator semantics already owned by Nodus
TensorMath. Retaining that branch prevents future agents from treating an
attractive archaeological artifact as active architecture.

## 7. August to October: compiler, machines, and demonstrations

The root's August record publishes generated compiler pages, managed-columnar
fluid worlds, and runtime demonstrations. Turing expands compiler lowering,
state/ABI handling, backend semantics, graph execution, and AbstractUI. A homepage
generated by a compiler is evidence about the producer and artifact, not an
adequate explanation of every subsystem or the original SpeakToMe purpose.

September 10 outer commit `bf1af14aaad5b8661c87322c22800cd6e4b23a94` explicitly
tracks engine_toy as a real versioned project. Its ignore exception and added
implementation contradict "only coordination documents". Machine, vehicle,
craft, and managed-time work remains substantive even when shown in a demo.

Historical lowering receipts preserve hard boundaries. An engine compile
failure, native-dt repair, or measured surrogate result needs its exact revision.
Do not retell these as either "nothing worked" or "the engine is finished".
Physics correctness, full-native compilation, repeated stateful execution,
frontend quality, and performance require separate evidence.

October 2 SpeakToMe `0189af26b13bef294b4c5cf7a860e5746d46c46b` adds the
probability-geometry thesis. It clarifies the originating purpose using
conditional distributions, uncertainty, divergence, behavioral reconvergence,
and geometric exploration. Its text marks proposed mechanisms and separates
literature-supported findings from project design. It does not establish that
every mechanism is implemented or validated.

October also records canonical Turing integration, extensive type/SSA/numerical
work, and Token Orchard integration in the root. The ledger retains the large
commit record; this narrative is selective and does not claim every diff was
read line by line.

## 8. November: convergence and the danger of stale status

The November record includes Meowdoku/Marble Studio, Nodus service-backbone work,
Spectral rendering, typed physical objects, compiler services, authentication and
messaging, graph ports, repository maps, and further compiler development. These
are not all one completed deployment. Working names do not define the full
intellectual identity of the systems underneath.

SpeakToMe's `1794364200_DOC_Convergence_Continuation_Report.md` records the
owner's distinction: the engines are solid and need other solid engine parts;
the demo site is the problem. The recorded direction is a thin page control
shell, a finished Pythonic product, and compiler/server participation at the
boundary between compilable and noncompilable parts. This is a dated receipt,
not proof that every later change achieved the plan.

Status changed even during this audit. Captured Turing `a90b9f5c2215` adds
`divider_nodus_bridge.py`, tests, a handoff, and a hazard receipt, describing one
divider piece sent at a time to the Nodus compile host. Older "the divider has no
worker" claims need historical scope. This pass did not execute the new bridge
or prove live deployment. Source emission is not a built native product merely
because a job is labelled complete.

Separate-worktree REST extraction assigns shared HTTP handlers to SpeakToMe
while retaining compiler, authentication, storage, deployment, graph, and machine
responsibilities in their proper systems. It must be evaluated as paired source
changes and a deployment, not as a simulator implementation. Local commits and
remote publication are different states.

## 9. Owner intent, verbatim

These statements were supplied during the 2026-11-11 UTC documentation session.
Spelling is preserved. They are primary evidence of interpretation and aspiration,
not source-code validation or legal conclusions.

> hey speaktome is where this started, those were demonstrations of the systems that made them
>
> everything kind of got as far as it could go and never realized speaktome because they were not speaktome they were the engines and subsystems speaktome needed. there is a complete history and liage traceable through git and the remaining files in speaktome that shows that the other subrepos came from this one, this one that usees nogodsnomasters  as it's main, was made complete by forming the outer nogodsnomasters repo

> we're going to be developing the readme and licenses for the superrepo et all, explainign to agents how it all maps out, in exhausting details, of the histories intensions ideas in their massive timeline requiring digital forensics. the entire suprerepo is a document of the speaktome project

> you know what we'll make a site for each eventually if you're allowed to make that many sites for your repos and they'll link across to each other as part of how speaktome will allow navigation of their system as an interactive total readme that also is that in static site links, and the speaktome landing will have the uh product of the development site of meowdoku, a private working name because it's infringement to call it that or something, that will be the game engine, a hidden repo in the same space as i don't know what, doing serious work, hidden inside

> I think it's correct to say I didn't know how each repo would stand or get finished but spiritually they're all MIT from speaktome

The site network and private/public presentation are directions to develop,
not existing guarantees. The working-name concern is not a finding of infringement.
The MIT aspiration does not grant rights in third-party material; first-party
license changes require their own decision and commits.

## 10. Selected dated anchors

**S:** source/document content inspected or directly compared in this pass.
**C:** commit identity/date/subject verified; behavior not independently tested.
**D:** inspected design/historical document, not proof of working integration.

Dates use the authored date as stored; the CSV retains its offset and committer
time. Links identify objects, not a promise that unpublished local objects are
already available on GitHub.

| Authored date | Repository / commit | Grade | Why it matters |
|---|---|---|---|
| 2025-06-07 | speaktome [`a00b59ecf4`](https://github.com/sangderenard/speaktome/commit/a00b59ecf438dad40687e51531f5dd6b35cacab9) | S | Earliest retained root; full-beam-lattice purpose. |
| 2025-06-08 | speaktome [`74533e6d3d`](https://github.com/sangderenard/speaktome/commit/74533e6d3d4ac190e69b7a360892e13fb6346800) | C | Tensor abstraction enters PyGGraphController. |
| 2025-06-08 | speaktome [`70979ed169`](https://github.com/sangderenard/speaktome/commit/70979ed1697a7c508d9b7c027cc1243645446131) | C | Tensor abstraction enters geometry modules. |
| 2025-06-13 | speaktome [`a7fa45c0cb`](https://github.com/sangderenard/speaktome/commit/a7fa45c0cb14f3418b00b33cc8822ac66a2cb6d3) | S | Earlier backend blobs later imported into Turing. |
| 2025-06-13 | lfsavoider [`9175ac51c4`](https://github.com/sangderenard/lfsavoider/commit/9175ac51c4060d5f84e67a48ecd3adfbf8f418a5) | C | LFS-avoider retained root. |
| 2025-06-13 | wheelhouse [`87ee40b1a1`](https://github.com/sangderenard/wheelhouse/commit/87ee40b1a14dc5a6959692684a9c9cd69b146c73) | C | Retained root explicitly records a reset. |
| 2025-06-22 | geometry [`a0829d74e6`](https://github.com/sangderenard/geometry/commit/a0829d74e675d9262e845d1267cd40c2b2e760af) | C | Geometry retained root. |
| 2025-08-02 | pigeon [`4b6031649c`](https://github.com/sangderenard/pigeon/commit/4b6031649cb1e5134b1e6b3351fb55527a9065e0) | C | Pigeon retained root. |
| 2025-08-03 | turing [`0638f94ac3`](https://github.com/sangderenard/Turing/commit/0638f94ac3063177562512499dd730c34908d7f6) | C | Turing retained root. |
| 2025-08-03 | turing [`7490c55d01`](https://github.com/sangderenard/Turing/commit/7490c55d0103b0b6668bf6d14c0edb9ba015ebd9) | C | Loop-aware SSA builder. |
| 2025-08-05 | turing [`1dde49dc37`](https://github.com/sangderenard/Turing/commit/1dde49dc3778d4d098aae6fc00c97986cd2c67e8) | C | Tape-compiler integration preparation. |
| 2025-08-18 | turing [`62d826b7c8`](https://github.com/sangderenard/Turing/commit/62d826b7c8f6959003b8e199e97e4d681d4d8874) | S | Direct tensor import; explicit README and matching blobs. |
| 2025-09-23 | pcb [`57d5a076a8`](https://github.com/sangderenard/electrofluid/commit/57d5a076a83ca3b40b58c57dfc8a231bd415fb49) | C | Electrofluid retained root. |
| 2025-10-19 | amp [`2644b112ce`](https://github.com/sangderenard/amp/commit/2644b112ceeb58ebba414cbc588f54aea4ed2eb2) | C | amp retained root. |
| 2025-10-28 | fftfree [`60d44eabc4`](https://github.com/sangderenard/fftfree/commit/60d44eabc4e7ebe7cf0e8a8cb7292179245aae06) | C | fftfree retained root. |
| 2025-12-14 | socratic `266c17f96d` (local) | C | Socratic retained root, no origin in snapshot. |
| 2025-12-18 | nodus [`54cd81e57c`](https://github.com/sangderenard/nodus/commit/54cd81e57c6f93b4b65e53ffca4cb597b903615e) | C | Nodus retained root. |
| 2026-01-07 | meta-nn [`9b5c7e1258`](https://github.com/sangderenard/meta-nn/commit/9b5c7e1258aa8e2a660fd4a0f014491fb76a3680) | C | Retained root says resuming work, not necessarily the origin. |
| 2026-01-08 | nodus [`d751bafbf6`](https://github.com/sangderenard/nodus/commit/d751bafbf6ae76792c57d7dc7fccb8152164403e) | C | Preparation to integrate AbstractTensor. |
| 2026-01-09 | nodus [`83300f3278`](https://github.com/sangderenard/nodus/commit/83300f3278dc201de6d0c16034170e0e95361416) | C | AbstractTensor and spatial-wave beginnings. |
| 2026-04-07 | spectral-analyzer [`ac35ecd1c3`](https://github.com/sangderenard/spectral-analyzer/commit/ac35ecd1c3b985e0ba5abd4abdadce540c4aff4d) | C | CQT, GPU iCQT and asynchronous synthesis retained root. |
| 2026-07-14 | speaktome [`226924420f`](https://github.com/sangderenard/speaktome/commit/226924420fe735e37189bf45b1f73f99f3c371b6) | S | Developed Turing tensor implementation returns. |
| 2026-07-14 | speaktome [`983997b2d9`](https://github.com/sangderenard/speaktome/commit/983997b2d9ce0a1a6a9a551f89b86f5e0f7014c3) | C | BeamSearch/Scorer migrate to AbstractTensor API. |
| 2026-07-14 | speaktome [`40f7e1991d`](https://github.com/sangderenard/speaktome/commit/40f7e1991d8f9bde2f93667e6bc6465b91fb3d14) | S | FluxGraph implementation, demo and tests. |
| 2026-07-25 | nogodsnomasters [`d6dafbd787`](https://github.com/sangderenard/nogodsnomasters/commit/d6dafbd7875140add099cd58f5ffc14a08f32f7f) | S | Outer coordination repository established. |
| 2026-07-25 | nogodsnomasters [`be79d02a03`](https://github.com/sangderenard/nogodsnomasters/commit/be79d02a032742e28fd6f6b0cb4ec6fa768a48c7) | C | Electrofluid remote detached from root. |
| 2026-07-25 | nogodsnomasters [`5e39e64df2`](https://github.com/sangderenard/nogodsnomasters/commit/5e39e64df229f130a35c8eef4dd02fc5998afdc4) | D | Alpha/Kakarot lineage documented. |
| 2026-07-25 | nogodsnomasters [`6e6f4a77a0`](https://github.com/sangderenard/nogodsnomasters/commit/6e6f4a77a0dfdd670b4c5a5f1a5cc82ec2cb24ac) | D | Geometry/Nodus future documented. |
| 2026-07-25 | nogodsnomasters [`4bb19867ad`](https://github.com/sangderenard/nogodsnomasters/commit/4bb19867ad7c43cd17f18b20a6f1d46a124d8cb7) | D | Branching-search composition documented. |
| 2026-07-25 | nogodsnomasters [`4b2209418a`](https://github.com/sangderenard/nogodsnomasters/commit/4b2209418a1af2f771a2152b891508ea495cd7c2) | C | Historical protective source-available notice. |
| 2026-07-25 | nogodsnomasters [`238c54210f`](https://github.com/sangderenard/nogodsnomasters/commit/238c54210fa6e3aeabc2c91436c4b7bddbe697e8) | D | Pre-Kakarot lineage recorded. |
| 2026-07-26 | tensor-calculator [`9fd125c198`](https://github.com/sangderenard/tensor-calculator/commit/9fd125c1981b7fd99ba2ac4e951ae6908868e724) | C | Persistent calculator prototype, later superseded. |
| 2026-07-26 | nodus [`fc3c6f9a93`](https://github.com/sangderenard/nodus/commit/fc3c6f9a936a64ead86937a6a11beac28f86d165) | S | ProcessGraph tensor-tool-graph implementation and tests. |
| 2026-07-26 | nogodsnomasters [`4bf0fc6c4a`](https://github.com/sangderenard/nogodsnomasters/commit/4bf0fc6c4acd2d65108d46b75f9e4c4df16318c4) | D | ProcessGraph translation frontier documented. |
| 2026-07-26 | nodus [`d943fc6e38`](https://github.com/sangderenard/nodus/commit/d943fc6e38165fc89694271665cf903b73a9568c) | C | Calculator programs routed through TensorMath. |
| 2026-07-31 | nodus [`41168f3c65`](https://github.com/sangderenard/nodus/commit/41168f3c651ce82b50b932d224e1fa4f62c834b3) | C | Canonical operators served over tensor ABI. |
| 2026-07-31 | turing [`5f687330e1`](https://github.com/sangderenard/Turing/commit/5f687330e1fa83c721f3714cfeb4c0334bcb50b2) | C | Compiler output installed as repository homepage. |
| 2026-08-01 | nogodsnomasters [`a819fa5916`](https://github.com/sangderenard/nogodsnomasters/commit/a819fa59166aacd1ed3611b8566124c2f13db3ce) | C | Generated program gallery and local runner. |
| 2026-08-24 | nogodsnomasters [`74aeedc09a`](https://github.com/sangderenard/nogodsnomasters/commit/74aeedc09aca0c51f0cf29286adeb3e6872ca6e0) | C | Managed-time fluid demo publication recorded. |
| 2026-08-26 | turing [`0fb3244b5a`](https://github.com/sangderenard/Turing/commit/0fb3244b5a307a5e9f1c6b4713486f23f6417416) | C | AbstractUI compiler and vehicle physics. |
| 2026-08-27 | turing [`353b16c3e2`](https://github.com/sangderenard/Turing/commit/353b16c3e2d2293376b58622f317fb4aa5f25db2) | S | Captured gh-pages tip is living-data-map/vehicle demo. |
| 2026-09-10 | nogodsnomasters [`bf1af14aaa`](https://github.com/sangderenard/nogodsnomasters/commit/bf1af14aaad5b8661c87322c22800cd6e4b23a94) | S | engine_toy deliberately tracked as real implementation. |
| 2026-10-02 | speaktome [`0189af26b1`](https://github.com/sangderenard/speaktome/commit/0189af26b13bef294b4c5cf7a860e5746d46c46b) | S | Probability-geometry thesis added. |
| 2026-10-11 | nogodsnomasters [`9fc96c969c`](https://github.com/sangderenard/nogodsnomasters/commit/9fc96c969cc513dd994846a9f09a4e3664099573) | C | Agents directed to canonical Turing compiler. |
| 2026-10-24 | nogodsnomasters [`59b5b77495`](https://github.com/sangderenard/nogodsnomasters/commit/59b5b774951bcef0d8e5070805e96e0866d75dde) | C | Token Orchard core and research synchronized. |
| 2026-10-29 | turing [`e410d7438e`](https://github.com/sangderenard/Turing/commit/e410d7438ec94ad394e642e7bde596e9512d977e) | C | Float64 transcendentals routed through signal cores. |
| 2026-11-07 | spectral-analyzer [`ad79bd58fb`](https://github.com/sangderenard/spectral-analyzer/commit/ad79bd58fb519f7e17f06af6b2bbbcc63c5ca24b) | C | Authored checker rendered through thin-lens sensor. |
| 2026-11-07 | spectral-analyzer [`9355b98e9d`](https://github.com/sangderenard/spectral-analyzer/commit/9355b98e9d9b7688677c3cccf0e189470212ceb1) | C | Spectral sensor/Turing JPEG-writer orbit-frame contact. |
| 2026-11-07 | nodus [`6f3183e49d`](https://github.com/sangderenard/nodus/commit/6f3183e49d74d2392d889bfde740bafb72a8f680) | C | Service-backbone audit. |
| 2026-11-07 | nogodsnomasters [`ea0be04080`](https://github.com/sangderenard/nogodsnomasters/commit/ea0be040804ed5c093ad728458700037dfd8b4b2) | D | Nodus/Meowdoku service audit. |
| 2026-11-10 | nodus [`01f7d656cc`](https://github.com/sangderenard/nodus/commit/01f7d656cc578610bc204ac19a21e2b42dd37ff5) | C | Persisted ports/FIFO and loopback contract ports. |
| 2026-11-10 | nodus [`d3814f0165`](https://github.com/sangderenard/nodus/commit/d3814f01650960cc075b0c4be1ab72198888038e) | C | Nonblocking one-job compile host. |
| 2026-11-10 | turing [`526cae74d5`](https://github.com/sangderenard/Turing/commit/526cae74d5c50c042a1968f1a74e145ae5869fc6) | C | Repository-world scanner/extractors and compile job. |
| 2026-11-10 | speaktome [`6c8a5196d8`](https://github.com/sangderenard/speaktome/commit/6c8a5196d8f04d1052aa537a55b524834451bcff) | D | Cross-project convergence continuation report. |
| 2026-11-10 | turing [`a90b9f5c22`](https://github.com/sangderenard/Turing/commit/a90b9f5c2215cd017b7f7ceb532c47f06da8c6b4) | S | Divider-to-Nodus bridge source and tests; not executed here. |

## 11. Quantitative overview

Counts include merges and local branch/checkpoint history. They are not features,
lines of code, productivity measures, or completion scores. A missing month means
no captured authored timestamp falls there, not proof no work occurred. Resets
and imports can conceal older work. Months use stored author dates.

| Month | All records | SpeakToMe | Turing | Nodus | Pluck | Outer | Other |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-06 | 1,029 | 887 | 0 | 0 | 0 | 0 | 142 |
| 2025-07 | 8 | 0 | 0 | 0 | 0 | 0 | 8 |
| 2025-08 | 1,321 | 0 | 1,310 | 0 | 0 | 0 | 11 |
| 2025-09 | 613 | 0 | 606 | 0 | 0 | 0 | 7 |
| 2025-10 | 364 | 0 | 0 | 0 | 0 | 0 | 364 |
| 2025-11 | 172 | 0 | 0 | 0 | 0 | 0 | 172 |
| 2025-12 | 197 | 0 | 0 | 166 | 0 | 0 | 31 |
| 2026-01 | 25 | 0 | 0 | 24 | 0 | 0 | 1 |
| 2026-03 | 107 | 0 | 0 | 33 | 0 | 0 | 74 |
| 2026-04 | 28 | 0 | 0 | 0 | 28 | 0 | 0 |
| 2026-05 | 80 | 0 | 0 | 0 | 80 | 0 | 0 |
| 2026-06 | 29 | 0 | 0 | 0 | 29 | 0 | 0 |
| 2026-07 | 189 | 17 | 86 | 14 | 45 | 20 | 7 |
| 2026-08 | 722 | 30 | 598 | 6 | 0 | 88 | 0 |
| 2026-09 | 471 | 54 | 180 | 0 | 1 | 236 | 0 |
| 2026-10 | 2,185 | 15 | 2,106 | 20 | 2 | 41 | 1 |
| 2026-11 | 1,181 | 35 | 843 | 13 | 27 | 263 | 0 |

## 12. Established findings and remaining work

Strong evidence establishes an early SpeakToMe purpose, direct tensor extraction
into Turing, the return of developed machinery, concrete graph/compiler contacts,
real outer-owned implementations, and current owner-defined project unity.

This pass does not exhaustively establish every sibling extraction, all earlier
Alpha/DataProcessor chronology, every third-party origin, the current first
refusal of large compiler targets, fully working end-to-end SpeakToMe, or live
publication/deployment of every change. The next pass should build per-component
source-transfer dossiers rather than fill gaps with a cleaner story.

**Later decision during this session:** the owner authorized first-party MIT
licensing changes across SpeakToMe, Turing, Nodus, spectral-analyzer, and the outer
repository. That is a subsequent concrete decision, not merely the earlier
aspiration quoted above. See each repository's `LICENSE` and `LICENSING.md` and
the final licensing receipt for the enacted scope. Third-party terms remain
independent. The captured Git ledger predates those changes and is intentionally
not rewritten to imply they existed at the snapshot time.
