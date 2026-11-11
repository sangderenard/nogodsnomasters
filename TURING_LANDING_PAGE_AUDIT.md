# Turing landing-page audit and presentation direction

**Scope:** source inspection, 2026-11-11 UTC. No page redesign, publish,
deployment, hosting change, or new site was performed.

## 1. Three surfaces that must not be conflated

| Surface | Verified source | What it currently represents |
|---|---|---|
| Turing repository README | `turing/README.md` at captured main `a90b9f5c2215` | Broad compiler/runtime introduction followed quickly by historical Riemannian/tape-machine material |
| Turing's locally cached Pages branch | [`353b16c3e2d2:index.html`](https://github.com/sangderenard/Turing/blob/353b16c3e2d2293376b58622f317fb4aa5f25db2/index.html) | A single 7,736,401-byte HTML file titled `AbstractUI · living data map`; cached tip authored 2026-08-27 |
| Outer nogodsnomasters compiler-page/gallery source | Root `index.html` at `937de5dd4947`, plus `site/` and loopback Go server | 404,321-byte generated inspection/gallery HTML titled `index`, with local trusted-source compilation controls |

The cached Turing Pages tree contains only `index.html`. Its recent subjects
concern vehicle respawns, off-road playground behavior, Safari fallback, resident
vehicle graph, and crawler hardware. Source contains vehicle/shader and mobile
control machinery. This is a specific generated demonstration, not a site-wide
map of Turing's intellectual and technical scope.

No current GitHub Pages settings or live URL were queried in this pass. Therefore
this identifies the historical/cached page precisely without claiming that it
is today's deployed page or that a particular public URL is configured. The
repository README and page are also different from the broader development
service and its authenticated functions.

## 2. Why the identity needs rethinking

Turing spans AbstractTensor, precision and backend dispatch, source/SSA
compilation, graph/IR systems, managed scientific time, simulation, analog tape,
Riemannian models, and Transmogrifier. Leading with one large living-data-map
application makes the demonstration stand in for the machinery that produced it.
That is the same interpretive error the owner corrected at superrepo scale.

The README's first paragraph recognizes this breadth, but its next major section
is a historical research idea and then a tape-machine walkthrough. Those are
important preserved topics; their present ordering is not a complete onboarding
path for compiler, tensor, managed-time, or integration users.

The existing demonstration should retain a clear historical/artifact identity.
It should not be erased, quietly declared complete, or rewritten as a replacement
engine merely to give the landing page a stronger impression.

## 3. Owner direction, distinguished from implementation

The latest direction is eventually one site per repository, cross-linked as a
navigable SpeakToMe body of work. Ordinary static links and an interactive "total
README" should both provide access. Turing's independent breadth warrants its
own site rather than only a subsection or embedded binary on another landing.

The SpeakToMe landing can include the product of the development/game-engine
work. Meowdoku is a private working name, not an established public identity.
The owner's concern about naming is not a verified infringement finding. Desired
private repository/work boundaries are not automatically provided by static
hosting or a game interface. Decide and verify those boundaries explicitly.

The verbatim direction is preserved in [the history](SPEAKTOME_HISTORY.md#9-owner-intent-verbatim).
This document records the direction and an initial editorial proposal; it does
not assert settled layout, hosting capacity, access controls, domains, public
release scope, or final naming.

## 4. Proposed Turing information architecture

This is a proposal for the next design conversation, not a new implementation:

1. **What Turing is.** A concise description in its own terms, with a short
   relationship link to SpeakToMe and the larger project history.
2. **Choose a system.** Compiler/SSA, tensors/numerics, managed time, graphs and
   Transmogrifier, physical/cellular simulation, analog tape, and geometric/neural
   research. Each entry links to source, local contracts and current evidence.
3. **Use and inspect.** Supported setup and canonical entry points, concrete
   examples, and the difference between emitted source, built binaries, and
   verified runtime behavior. Do not manufacture universal installation promises.
4. **Demonstrations with provenance.** Each demo names its producer, input/source
   revision, build/runtime requirements, current limitations and archive status.
   The living data map belongs here as a specific artifact.
5. **History and open frontiers.** The tensor extraction/return lineage, major
   design decisions, meaningful unsuccessful attempts, and dated current receipts.
6. **Cross-project navigation.** SpeakToMe, Nodus, Pluck, outer machines and other
   component sites, using ordinary links even when an interactive world is available.

Static documentation should remain usable without a GPU, large WASM download,
interactive controls, authentication, or knowledge of game metaphors. Interactive
navigation can enrich that stable information rather than hide it.

## 5. Conditions before implementation/publication

- Verify the actual Turing Pages source and current live artifact, not only the
  cached `origin/gh-pages` ref.
- Read current page-generation/publishing instructions and preserve the existing
  producer/artifact contract rather than hand-editing generated output blindly.
- Settle which pages and artifacts are public, private, archived, or service-only.
  Hiding a link is not access control.
- Keep Turing's site, root compiler gallery, and SpeakToMe front door as separately
  identified surfaces with deliberate links and ownership.
- Use current first-party LICENSE/LICENSING.md and applicable dependency/model/
  asset notices before embedding or distributing binaries.
- Check navigation, keyboard/mobile access, no-GPU/static behavior, demo load cost,
  source links, broken links, and security boundaries on the actual chosen build.
- Obtain the needed publication authorization and verify the resulting remote
  ref and deployed artifact before calling a new site live.

No site code changed in this pass. The output is a grounded source inventory and
presentation scope for subsequent work, not another demonstration project.
