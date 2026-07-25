# SpeakToMe branching search as geometry

**Status:** architectural direction and proposed experiment, not a claim of
present manifold integration

**Date:** 2026-07-25

## Summary

SpeakToMe's recent bidirectional language-search work produces a geometric
object worth studying in its own right. It does not merely keep the best few
sentences. It grows alternatives forward and backward from an anchor, preserves
their causal relationships, lets branches compete for compute through a live
flux/pressure system, and can re-root the search without throwing the old
topology away.

The result is often described as a branching tree. That is the right first
picture, but the current `FluxGraph` is already richer:

- forward growth appends tokens or complete words;
- backward growth prepends them through `ImplicitBackpathScorer`;
- a node can have several simultaneous causal parents through the seed layer;
- forward and reverse edge channels can have different conductance;
- alternatives remain present after a current best path is read;
- branch pressure, flow, maturity, auxin, and model evidence change with time;
- re-rooting changes the point of view while preserving the graph.

Accordingly, the most honest mathematical description is an **evolving,
directed, weighted, stratified search graph**. Tree-like regions are its local
branching pieces. Joins, shared parents, seed-layer crossings, and re-rooting
make the whole object more general than one immutable rooted tree.

The research goal is to find useful local geometric models of that object and
sample them with the geometry / AbstractTensor lineage. The geometry must help
us perceive and explore the model's “wide truth”: not just which path wins, but
how probability, semantic variation, uncertainty, and viable alternatives are
shaped around a region of language.

## What exists now

The lineage is visible in two stages inside `speaktome/`.

`VISION_FORWARD_BACKWARD_DIFFUSION.md` describes forward and backward beams as
independent workers covering overlapping regions. Its central concern is the
shape of model belief: a narrow peak, a broad plateau, or disagreement between
the two directions. “Diffusion” there is a structural analogy, not a denoising
diffusion model.

`speaktome/core/flux_graph.py` develops that idea into a persistent search
organism:

1. An anchor sequence establishes the initial seed layer.
2. Ordinary causal next-token scoring grows forward.
3. prepend-and-rescore probes grow backward.
4. Optional tries resolve BPE fragments into word-level edges before exposing
   them to the graph's physics.
5. A bounded compute budget expands selected live tips.
6. Model evidence rewards useful growth, while a separate fluid-like system
   transports solvent and named solubles and determines live pressure.
7. Digestion, auxin-like suppression, starvation, branch maturity, and
   conductance influence which parts continue growing.
8. A sufficiently strong node can become the new anchor without deleting the
   prior graph.

This is deeper parallelism than batched token scoring alone. There is potential
parallel work:

- among candidate tokens within one expansion;
- among subword branches while constructing a word;
- among many active graph tips;
- between forward and backward growth;
- among independently transported substances and edge channels;
- among graph regions settling, being scored, inspected, or rendered;
- across tensor backends through `AbstractTensor`.

That nested concurrency is part of the geometry. It creates many simultaneously
valid local histories instead of one serial decoding trace.

## The geometry already implicit in the search

The search state supplies several different structures that must not be
collapsed into one arbitrary visualization coordinate.

| Search quantity | Geometric interpretation |
|---|---|
| node identity | a sampled state or point with durable provenance |
| parent/child relations | causal topology and admissible paths |
| signed level | position along the global reading axis |
| forward/backward direction | orientation, not merely screen placement |
| local log evidence | transition cost or local potential |
| cumulative/path evidence | path action, with length normalization |
| pressure and transported material | a time-varying measure/control field |
| edge conductance | directional permeability or transport weight |
| maturity | persistent structural reinforcement |
| auxin and growth interest | competing fields on the branching complex |
| active, expanded, or burned state | an evolving boundary/frontier |
| hidden state or logits | candidate coordinates for a semantic chart |

The causal topology is exact data. A transformer embedding is only one possible
coordinate chart on it. A two- or three-dimensional display projection is only
a view of a chart. Neither is allowed to replace node IDs, directions, or
parent relationships.

### Why this is not automatically a smooth manifold

A branch point has several outgoing neighborhoods. A join can have several
causal parents. Burned regions and active tips form boundaries. Discrete words
occupy unequal token spans. Re-rooting changes the preferred coordinate origin.
Transformer representations can fold semantically similar paths together even
when their causal histories differ.

Those are singular and stratified features, not defects to smooth away. A
manifold approximation may be appropriate inside a sufficiently coherent
region—for example, a family of nearby continuations represented by hidden
states—but the global object should retain its graph topology.

A practical formulation is therefore:

```text
exact directed search complex
    + node and edge fields
    + local semantic charts
    + chart-specific metrics and measures
    = a sampleable stratified metric-measure object
```

It may eventually call for graph geometry, manifold learning, discrete exterior
calculus, or a directed/Finsler-like metric rather than one globally symmetric
Riemannian metric. Forward and backward conditional costs are genuinely
different, so symmetrizing them too early would erase useful information.

## What “sampling the manifold” should mean

Sampling should not mean projecting every node to three dimensions and choosing
pretty points. It should select graph states or paths for a declared purpose
while preserving enough provenance to recover the language and its causal
history.

Useful sampling objectives include:

- **belief coverage:** represent broad, near-equal alternatives rather than
  repeating one high-probability basin;
- **frontier coverage:** distribute compute among geometrically distinct live
  tips;
- **meeting analysis:** locate regions where forward and backward readings
  converge or disagree;
- **curvature/adaptation:** sample densely where local distributions or semantic
  directions change sharply;
- **flux importance:** follow pressure or transported useful material without
  confusing that control signal with probability;
- **path diversity:** choose complete readable paths that are far apart under a
  semantic metric but remain plausible;
- **inspection and play:** expose representative branches to Nodus or Pluck
  without flattening the graph into a single answer;
- **training data:** retain unusual but coherent regions whose geometry differs
  from the dominant basin.

Candidate policies to compare include ordinary top-k, uniform live-tip
sampling, probability-mass sampling, flux-weighted sampling, farthest-point or
geodesic coverage, diffusion/spectral sampling, and adaptive sampling near
high-disagreement regions.

## A proposed snapshot contract

The first integration should be a read-only snapshot adapter, not a rewrite of
`FluxGraph`. A snapshot should contain:

```text
SearchGeometrySnapshot
    node_ids
    tokens / resolved words
    directions and signed levels
    complete parent and child adjacency
    anchor, frontier, expanded, and burned masks
    local, cumulative, and rolled-up evidence
    pressure, solubles, maturity, auxin, and growth interests
    forward and reverse edge conductances
    optional hidden-state, logit, or attention features
    model/tokenizer/configuration identity
    graph tick and snapshot provenance
```

The adapter should publish consistent snapshots at tick boundaries, following
the existing `published_snapshot` discipline. Sampling results must return
stable node or path identities alongside tensor coordinates.

Feature extraction should be pluggable. No one feature space is the geometry:

- final hidden states can describe local semantic representation;
- normalized logits can describe predictive belief;
- Jensen–Shannon or another distribution distance can compare predictions;
- graph distance preserves causal separation;
- negative log transition probability supplies a directed path cost;
- flow and conductance describe the current search physiology;
- attention summaries may provide another learned local chart.

The metric should likewise be declared and versioned. A first metric can be a
simple weighted combination for experimentation, but each component should
remain observable so that a visually compelling projection cannot conceal a
bad distance definition.

## Where AbstractTensor belongs

SpeakToMe already uses the AbstractTensor lineage so search logic is not welded
to PyTorch. Turing extends that lineage with backend-agnostic numerical work,
including grid domains, transforms, metric-aware Laplace machinery, and the
recent YoungMan demonstration.

AbstractTensor is the natural execution substrate for:

- batched node features and masks;
- local pairwise distances;
- sparse or neighborhood adjacency;
- metric tensors or learned local quadratic forms;
- graph Laplacians and diffusion operators;
- eigenspaces or low-dimensional chart coordinates;
- weighted, stratified, and stochastic sample selection;
- comparative tables that run on NumPy, Torch, or another supported backend.

It should not own language-search semantics. SpeakToMe remains responsible for
model scoring, word formation, causal paths, expansion, and the meaning of its
physiological fields.

The continuous/deferred geometry work matters because the correct metric may
depend on the eventual query. A visualization, a compute scheduler, a
forward/backward agreement study, and a training-data sampler may need different
coordinates and resolutions over the same frozen graph. We should preserve a
parametric chart/metric program until the requested nodes, neighborhood, and
sampling purpose are known, then resolve only what is needed.

## Project boundaries

The likely long-term ownership is:

- **SpeakToMe:** transformer interaction, bidirectional growth, causal topology,
  word/token semantics, search physiology, and snapshot publication.
- **Turing / AbstractTensor:** backend-agnostic metrics, local charts, manifold
  learning, diffusion/Laplacian analysis, and numerical sampling experiments.
- **geometry:** eventually, portable declarations for domains, transforms,
  metrics, topology, refinement, and accelerated native kernels. Its current
  repository is still mostly an architectural scaffold and should not yet be
  made a hard dependency.
- **Nodus:** durable graph identity, typed ports, scheduling, capability
  negotiation, inspection, and composition of the eventual services.
- **Pluck / spectral-analyzer:** interactive OpenGL and other sensory views of
  the living search geometry, potentially making its branches, fields, and
  sampling choices playable.

This keeps the language graph authoritative while allowing several geometric
interpretations to coexist.

## Smallest useful experiment

The first experiment should be numeric, reproducible, and modest:

1. Run a small real-model `FluxGraph` from a fixed anchor and random seed.
2. Freeze complete snapshots at selected ticks.
3. Record node identity, topology, direction, level, evidence, flux fields, and
   a transformer feature vector for each live node.
4. Build local neighborhoods that include causal edges and separately computed
   semantic neighbors.
5. Compare a symmetric baseline metric—cosine hidden-state distance plus
   predictive-distribution distance—while retaining directed transition costs
   as a separate field.
6. Use AbstractTensor to construct a weighted graph Laplacian or diffusion
   operator and derive a small spectral chart.
7. Compare top-k, uniform-frontier, flux-weighted, and geometry-aware samples.
8. Report pandas tables for branch coverage, path diversity, probability mass,
   forward/backward disagreement, and compute cost.
9. Return every selected point as a node/path ID and decode it back to readable
   text as a provenance check.

The decisive success is not a beautiful projection. It is evidence that a
geometry-aware sampler covers meaningful alternatives or identifies
forward/backward structure better than conventional beam selection at a
comparable compute budget.

After the numeric result is trustworthy, Pluck's ordinary OpenGL renderer can
show the causal skeleton, chart coordinates, and changing fields as separate
layers. The viewer must make projection distortion visible and allow selection
to resolve back to exact graph paths.

## Questions the experiment must answer

1. Which transformer representation gives stable local neighborhoods as the
   graph continues to grow?
2. Should forward and backward regions use separate charts or a shared chart
   with explicit orientation?
3. Where do causal neighbors and semantic neighbors disagree?
4. Does a symmetric metric lose essential conditional directionality?
5. Is pressure a useful sampling measure, a scheduler signal only, or both?
6. How should multiply-parented nodes contribute to path probability and graph
   volume?
7. Can diffusion coordinates identify broad belief plateaus and meeting regions
   that top-k misses?
8. Which quantities survive re-rooting invariantly, and which must be understood
   relative to the current anchor?
9. Can sampled representatives be decoded into coherent paths without hiding
   important alternate ancestry?

## Guardrails

- Do not call the complete FluxGraph a smooth manifold.
- Do not identify screen distance with semantic, causal, or probabilistic
  distance.
- Do not overwrite exact graph topology with a nearest-neighbor graph.
- Do not equate pressure with probability; the implementation explicitly keeps
  live pressure separate from accumulated language score.
- Do not silently symmetrize forward/backward transition costs.
- Do not force every consumer to materialize every hidden state or pairwise
  distance.
- Do not make the unfinished standalone `geometry/` repository a dependency
  merely to establish the vocabulary.
- Do not let a sampler return anonymous vectors; all results need node/path and
  snapshot provenance.
- Do not optimize away wide alternatives before the experiment can measure
  whether they contain useful structure.

## The larger point

Ordinary beam search asks which few strings should survive the next pruning
step. SpeakToMe's bidirectional FluxGraph asks what kind of living region of
language appears when commitment is deferred and many possible histories remain
active together.

Geometry gives us a way to ask the next question: how can we measure and sample
that region without reducing it back to a winner list?

The answer will probably not be one manifold or one metric. It will be an exact
branching search complex, several late-bound local charts, and sampling policies
chosen for explicit purposes. That is precisely why the geometry and
AbstractTensor work belong in this lineage.
