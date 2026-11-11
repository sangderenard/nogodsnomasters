# SpeakToMe forensic record and continuation protocol

**Version:** initial documentation pass, 2026-11-11 UTC.
**Evidence captured:** 2026-11-11 02:55:47 UTC; each repository's refs were read
sequentially, not in one cross-repository atomic transaction.
**Related:** [history](SPEAKTOME_HISTORY.md), [atlas](SPEAKTOME_PROJECT_ATLAS.md),
[repository metadata](SPEAKTOME_REPOSITORIES.json),
[commit ledger](SPEAKTOME_COMMIT_TIMELINE.csv).

## 1. Findings before method

1. SpeakToMe's earliest retained source describes the application purpose before
   the current compiler, Nodus, and outer repository existed in their retained
   histories.
2. Turing's 2025 tensor import explicitly names SpeakToMe and preserves matching
   backend blobs. The developed tensor tree returns to SpeakToMe in 2026.
3. The outer repository's source ownership includes real implementations and
   published artifacts; it is not limited to coordination documents.
4. Concrete compiler/graph/service interfaces coexist with proposals, historical
   failures, standalone snapshots, and demonstrations. Project unity does not
   establish universal integration or completion.
5. A repository-wide history needs both readable interpretation and a large
   auditable ledger. This pass supplies the foundation, not an assertion that
   every file or old archive has already been forensically examined.

## 2. Evidence classes

- **Owner statement:** establishes intent, preference, or a decision. Quote and
  date it. It cannot independently establish runtime behavior or third-party rights.
- **Git metadata:** full commit ID, parents, refs and timestamps. A subject proves
  what was recorded, not that its assertion is correct.
- **Source evidence:** inspected file contents at a named object. Working-tree
  contents without a pinned revision need a separate observation date/status.
- **Identity evidence:** matching Git blobs prove equal bytes at the compared
  paths/revisions. Read surrounding changes to establish transfer direction.
- **Design evidence:** a proposal, thesis, architectural map or owner's plan.
  Preserve its status instead of promoting it to implementation.
- **Runtime evidence:** actual commands, environment, revisions and receipts.
  A historical test pass is not a present test pass.
- **Publication evidence:** an exact remote ref or deployed artifact verified
  after the action. Local commit and remote-tracking cache are not enough.

Use these categories in each new dossier. Preserve contradictory evidence until
it is explained; do not remove a failed experiment to make the story cleaner.

## 3. How this inventory was assembled

The working scope was `C:\dev\Powershell`, the existing outer repository and
its immediate child Git checkouts. The investigation:

1. Read root AGENTS, SpeakToMe's referenced agent/environment/testing guidance,
   the guestbook conventions, existing lineage documents and current READMEs.
2. Checked Git status before edits and preserved pre-existing untracked material.
3. Enumerated immediate directories with `.git` entries. Used
   `git rev-parse --path-format=absolute --git-common-dir` to group linked
   worktrees instead of counting them as separate projects.
4. Selected the 15 canonical repository paths recorded in the JSON. Captured
   each canonical HEAD and exact `for-each-ref` name/object pairs.
5. Walked the union of captured ref object IDs and canonical HEAD via
   `git log --stdin`; independently recorded canonical HEAD ancestry.
6. Stored full commit ID, author timestamp, committer timestamp, parents,
   membership in captured HEAD ancestry and subject. No author email addresses
   or credentials were exported.
7. Sorted the CSV by parsed author timestamp, then repository and commit ID.
   This is a chronological index, not a replacement for the commit DAG.
8. Recorded tracked-path counts, top-level names, branch, configured origin,
   cached origin/HEAD, shallow state, roots, and worktree aliases in the JSON.
9. Inspected selected primary files/diffs and reproduced the three tensor blob
   comparisons. Read existing historical analyses with their provenance intact.
10. Authored the README, agent orientation, atlas, history, this method, and the
    landing-page audit. No compilation, deployment or physical-device run was
    needed or performed for this documentation change.

No fetch was used to refresh the snapshot. No transport substitution was used
to overcome an access denial. Ref state can therefore be stale relative to the
remote, while the local evidence remains exactly identifiable.

## 4. Captured scope and exclusions

The ledger has **8,721 unique `(repository, commit)` records** across **15 Git
databases**. All report non-shallow histories. These are commits reachable from
captured refs plus canonical HEADs, including local branches, remote-tracking
branches, tags where present, merges and checkpoints. Identical commit IDs in
different repositories remain separate provenance records.

Not included or not exhaustively audited:

- reflog-only/unreachable/dangling objects and deleted branches;
- detached worktree HEADs not reachable from captured refs or canonical HEAD;
- repositories deeper than the immediate-child scan, including nested submodules;
- full contents of `_quarantine/`, `archives/`, `repo-cache/`, `C:\Alpha`, or
  earlier backups and reset histories;
- all ignored/untracked source, generated artifacts, package caches or local state;
- every commit body/diff, every implementation file, or every third-party notice;
- fresh remote branch/default/Pages configuration, live services, or build status;
- credentials, account databases, private keys and operational secrets.

Non-shallow does not mean complete historical survival: a reset, copied tree,
missing remote branch, or archived repository can still leave gaps. Presence of
a root commit is a lower bound on recoverable history, not an origin-date proof.

The JSON's earliest-author/latest-committer summaries and the CSV's chronological
sort compare parsed offset-aware timestamps. The original timestamp strings and
offsets are preserved. Parent relationships, not wall clocks, establish ancestry.

## 5. Claim register

| ID | Claim | Evidence | Boundary |
|---|---|---|---|
| H01 | SpeakToMe's branching-space purpose predates the retained Turing/Nodus histories | `speaktome:a00b59e`, project overview and holographic proposal | Earliest retained source, not idea birthday |
| H02 | A tensor package was extracted into Turing from SpeakToMe | `turing:62d826b7:src/tensors/README.md`; three exact backend blobs | Direct evidence for those files/package, not every Turing subsystem |
| H03 | Developed tensors returned from Turing to SpeakToMe | `speaktome:2269244` diff and migration | Does not certify every backend or controller |
| H04 | FluxGraph's persistent-search object was implemented | `speaktome:40f7e19`, implementation/demo/tests | Tests not rerun in this pass |
| H05 | The outer composition allocates distinct component responsibilities | `nogodsnomasters:4bb1986`, branching-search design | Design, not complete runtime proof |
| H06 | Nodus has concrete ProcessGraph import work | `nodus:fc3c6f9`, implementation and test diff | Particular interface, not all cross-runtime behavior |
| H07 | Root owns engine_toy as real source | `nogodsnomasters:bf1af14`, .gitignore and source additions | Independent children remain opaque |
| H08 | Earlier source-available terms differed from MIT | `nogodsnomasters:4b22094` and pre-change LICENSE | Historical state; later authorized license commits supersede first-party current terms |
| H09 | Tensor-calculator is superseded, not shared substrate | `tensor-calculator:34f9fec`, README | Preserved prototype remains useful history |
| H10 | Turing's cached Pages branch serves a large living-data-map demo | `turing:353b16c3:index.html` | Local cached branch, not verified live Pages source |
| H11 | New divider/Nodus bridge work exists at capture | `turing:a90b9f5c`, bridge, tests and handoff | No new execution/deployment assertion |
| H12 | One project document with independently useful subsystems is the owner's intended framing | Exact current statements in history | Intent; does not rewrite Git/legal ownership |
| H13 | Early DataProcessor/Alpha/Kakarot concepts recur later | Existing July lineage audits | Secondary reconstruction until those sources are independently revisited |
| H14 | Every sibling is a direct source extraction from SpeakToMe | Not established exhaustively | Owner's broader lineage claim guides continued research, not invented commit edges |

## 6. Reproduce the strongest transfer evidence

Run inside the existing checkouts, without fetching or modifying them:

```powershell
git -C speaktome rev-parse a7fa45c0:tensors/numpy_backend.py
git -C turing rev-parse 62d826b7:src/tensors/numpy_backend.py
git -C speaktome rev-parse a7fa45c0:tensors/torch_backend.py
git -C turing rev-parse 62d826b7:src/tensors/torch_backend.py
git -C speaktome rev-parse a7fa45c0:tensors/jax_backend.py
git -C turing rev-parse 62d826b7:src/tensors/jax_backend.py
git -C turing show 62d826b7:src/tensors/README.md
git -C speaktome show --stat 2269244
git -C speaktome show --stat 40f7e19
```

For each repository, the snapshot's exact OIDs are the inputs for regeneration.
Pass those OIDs through standard input to `git log --stdin` rather than placing
hundreds of refs on a Windows command line. Use the format
`%H%x1f%aI%x1f%cI%x1f%P%x1f%s`, then write a properly quoted UTF-8 CSV with the
schema above. For HEAD membership, use `git rev-list <captured-head>`, not today's
HEAD. Preserve the original snapshot as evidence; a refresh should state its
new time and describe changed scope rather than silently relabel old results.

Source-transfer dossiers should add `git log --follow -- <path>`, old/new
`ls-tree` records, matching blobs, renames, and surrounding README/import changes.
A line-similarity result alone is weaker than an explicit transfer plus identity.
Do not normalize files and then call matching normalized output byte-identical.

## 7. What changed after the snapshot

The concurrent licensing work landed locally after capture:

| Repository | Licensing commit |
|---|---|
| Outer nogodsnomasters | `a84c7435f4f76dc560585329360cc984ca44721e` |
| Turing | `c34e50d08fe7aa64bc51e983cf52b143f47665c4` |
| Nodus | `af74ee5ebc58b1a55ee286da40a3024db3ad15d2` |
| spectral-analyzer | `25f6ae5262ef3b1236aeccab05d4036eeaa09bc7` |
| SpeakToMe REST worktree | `9f4da00575368baffd4ee8dbbd94fa9ed3b73c5f` |

These enact the owner's first-party MIT direction with component-specific
LICENSING.md notices and retained third-party obligations. The documentation
worker did not edit LICENSE files. These newer commits are intentionally not
backfilled into the earlier frozen ledger. Their presence locally does not
prove remote publication.

SpeakToMe SDK cleanup `2c7adc2078adc55361cc6a74424a011e47f05411` precedes its
licensing commit on that worktree: 1,973 tracked SDK paths removed from the index,
local files preserved and hash-verified by the responsible lane. This is not
Git-history removal, not cleanup of every other branch, and not proof all
Garmin-related files elsewhere have been removed.

## 8. Ranked continuation queue

1. **SpeakToMe package extraction dossiers.** Inventory original/current package
   paths, compare historical blobs against Turing, geometry, amp, meta-nn,
   Pigeon, Socratic and other candidate destinations. Prove or qualify each edge.
2. **Transmogrifier dossier.** Compare the standalone snapshot against Turing's
   imported/integrated tree; list unique experiments, direct copies, rewrites,
   and missing history. Do not delete the snapshot while determining uniqueness.
3. **Pre-Git/Alpha archaeology.** Inventory archives by content hash and manifest,
   locate contemporaneous documents, reconstruct partial ordering, and clearly
   label inferred dates. Do not use directory names as a substitute for evidence.
4. **Idea/decision index.** Index guestbook prompt history and design files by
   concept: branching probability, geometry, managed time, graph authority,
   compilation, physical laws, sensory interfaces, productive worlds, permanence.
   Preserve exact quotations and distinguish intent from measured behavior.
5. **Boundary dossiers.** For each implemented edge, identify producer/consumer
   commits, ABI/state/transport contracts, tests, latest refusal, and deployment
   status. Avoid restarting mature systems under a new interface name.
6. **Artifact genealogy.** Map each published bundle/demo to its source commit,
   generator, binary hashes, input state, and build/test receipt. Keep failed
   and superseded artifacts identifiable rather than presenting all as current.
7. **Licensing provenance.** Follow the separate license inventory for remaining
   components and nested dependencies. First-party MIT does not erase copyleft,
   attribution, SDK, model, asset, or generated-output obligations.
8. **Presentation/site plan.** Verify actual hosting refs/configuration and decide
   public/private boundaries, stable component URLs, static navigation and
   accessible interactive layers. The current audit is not a deployment approval.

For each continuation deliver a dated, hash-anchored dossier, list what was not
examined, and update this index without erasing the earlier evidence state.

## 9. Initial evidence-file checksums

These SHA-256 checksums identify the frozen UTF-8, LF-normalized Git file
contents, not later Markdown edits. A Windows checkout may use CRLF; normalize
line endings or hash `git show <revision>:<path>` bytes when comparing.

- `SPEAKTOME_REPOSITORIES.json`: `0ea4cbaa6d050c1976ae5c148113d0d07a41b78bae438de23f1227698f264ece`
- `SPEAKTOME_COMMIT_TIMELINE.csv`: `83fa1f8c7405dd342fb148f996878bccf0b2fc2c05d1135ff48c723c36fef9f9`

## 10. Guestbook receipt

The paired SpeakToMe report is `AGENTS/experience_reports/1794366679_AUDIT_SpeakToMe_Superrepo_History.md`, with continuation stub `todo/1794366679_speaktome_history_continuation.stub.md`. These files belong to SpeakToMe, not the outer Git tree.
