# Self-repairing semantic translation network

Status: architectural intention, recorded 2026-08-05. This document describes
the intended system, not a claim that every component is implemented today.

Related current-state records:

- [`PROCESS_GRAPH_TRANSLATION_HANDOFF.md`](PROCESS_GRAPH_TRANSLATION_HANDOFF.md)
  describes the verified ProcessGraph translation frontier.
- [`research/README.md`](research/README.md) indexes the cross-repository
  abstract-tensor-to-any-language research.

## Governing intention

The translation system should become more than a collection of converters. It
should operate as a **self-repairing semantic transport network**.

Treat the translation matrix as logically complete even when its physical
implementation is sparse. For every pair of source and target semantic domains,
the cell

\[
T_{A\rightarrow B}: \mathcal{S}_A \rightarrow \mathcal{S}_B
\]

is an obligation. A cell may be discharged by:

- a direct translator;
- a composed path through intermediate languages or representations;
- a lowering into the machine-code proxy and a lifting back out;
- a synthesized specialization;
- or an explicit proof that conversion is impossible under the requested
  constraints.

An empty cell is therefore not merely "unsupported." It is a failed proof
obligation from which the system should produce a useful failure witness and,
when possible, synthesize the smallest missing semantic bridge.

Completeness does not mean that every cell contains handwritten code. It means
that every demanded route does one of three things:

1. produces a verified, commuting translation;
2. synthesizes and installs a missing bridge with an explicit validity domain;
3. returns a precise impossibility witness.

## The constrained repair agent

The repair agent is a Chinese-room participant in a deliberately bounded
construction problem. It need not understand the entire programming ecosystem
in a human architectural sense. It receives:

1. the source fragment in language or domain `A`;
2. the target contract for language or domain `B`;
3. the available operators and translation rules;
4. attempted graph paths and their failure witnesses;
5. the required observable behavior;
6. the machine-code proxy's forward and reverse representations;
7. tests, invariants, cost limits, and permitted approximations.

Its task is not simply "write an `A`-to-`B` compiler." Its task is:

> Produce the smallest admissible transformation that closes this particular
> semantic gap, then generalize it only as far as the evidence permits.

This constraint is essential. It discourages the invention of a large, fragile
translator when the actual missing bridge is one operator, calling convention,
memory rule, control-flow construct, or narrowly scoped runtime primitive.

The agent may choose a route `A -> Z -> B`; it should not be required to write
directly in either endpoint language. Suitable intermediates may include:

- the normalized operator IR;
- SSA;
- continuation-passing IR;
- a memory-explicit IR;
- SPIR-V;
- LLVM IR;
- C or Fortran;
- WebAssembly;
- a generated state machine;
- or a narrowly scoped runtime primitive.

An intermediate is admissible only when it exposes the missing semantics. It
must not merely relocate or conceal the failure. Lowering coroutines through a
continuation IR is principled; hiding them in an opaque, unauditable runtime blob
may reproduce behavior but weakens the matrix's semantic account.

## Structured translation gaps

When no valid route exists, the graph should emit a structured conflict rather
than a generic compiler error. For example:

```yaml
TranslationGap:
  source_domain: Python
  target_domain: WebAssembly
  source_operator: generator_yield
  required_semantics:
    - suspend execution
    - preserve local state
    - resume with incoming value
    - propagate exceptions
  available_target_operators:
    - call
    - return
    - indirect_call
    - linear_memory
    - mutable_globals
  rejected_paths:
    - path: Python -> SSA -> WASM
      failure: no continuation-state lowering
    - path: Python -> C -> WASM
      failure: runtime dependency exceeds policy
  machine_proxy_evidence:
    - state save before suspension
    - dispatch on resume address
    - exception edge to caller
  constraints:
    - static deployment
    - deterministic
    - no external interpreter
```

This object turns open-ended code generation into a bounded construction
problem. It should identify the earliest localized semantic obstruction, retain
the reasons rejected routes failed, and carry enough evidence for either repair
or a defensible impossibility result.

## Diffusion through the translation graph

Candidate routes should be proposed by diffusing demand outward from both
endpoints:

\[
A \rightsquigarrow \cdots \qquad \cdots \leftsquigarrow B
\]

The search should not optimize hop count alone. Every translation edge carries
a multidimensional cost:

\[
C(e) =
w_s E_{\text{semantic loss}} +
w_r E_{\text{runtime burden}} +
w_p E_{\text{performance}} +
w_v E_{\text{verification}} +
w_c E_{\text{complexity}}
\]

The graph should seek a low-energy meeting region under the request's actual
constraints. A longer route such as

```text
Python
  -> normalized AST
  -> effect-explicit SSA
  -> continuation IR
  -> structured control-flow IR
  -> WebAssembly
```

may be preferable to a shorter `Python -> JavaScript -> WebAssembly wrapper`
route when determinism, static deployment, reversibility, auditability, or
native numerical performance matters.

Diffusion proposes candidate corridors. Static contracts, formal checks where
available, and execution testing select the route that is actually installed or
used.

## The bidirectional machine-code proxy

Machine code should serve as a common observational layer:

\[
A \rightarrow M \leftarrow B
\]

More importantly, the native source behavior and the translated target behavior
can be compared after lowering:

\[
A \rightarrow M_A
\]

\[
\widehat{A\rightarrow B}(A) \rightarrow M_B
\]

The comparison should cover consequential behavior such as:

- control-flow decisions;
- memory reads and writes;
- call effects;
- arithmetic state;
- exceptions and traps;
- externally visible I/O;
- selected timing or resource characteristics when required by the contract.

Raw binary comparison is too brittle because register allocation, instruction
selection, layout, and optimization differ without changing semantics. Instead,
both binaries should be reduced to consequential state transitions:

\[
\operatorname{BehaviorGraph}(M_A)
\approx
\operatorname{BehaviorGraph}(M_B)
\]

Decision-node reduction makes the machine proxy an execution-derived
specification rather than merely a terminal target.

The reverse direction is equally important. If machine behavior can be lifted
into SSA, ProcessGraph, or a decision-state representation, machine code becomes
a semantic relay station:

```text
source construct
    -> reference executable
    -> machine decision graph
    -> abstract operation pattern
    -> target implementation
```

This is example-driven compiler synthesis with a strong intermediate witness.

## What a completed matrix entry contains

A successful repair installs more than generated code. It installs a qualified
translation package:

```yaml
TranslationRule:
  source_pattern: ...
  semantic_preconditions: ...
  target_expansion: ...
  introduced_runtime_state: ...
  preserved_invariants: ...
  unsupported_cases: ...
  proof_tests: ...
  differential_tests: ...
  cost_model: ...
  provenance: ...
  confidence: ...
  generalization_boundary: ...
```

The matrix must distinguish among:

- a universally valid operator translation;
- a translation valid only under bounded integer arithmetic;
- a specialization valid only for static shapes;
- a declared heuristic approximation;
- and a one-off patch for a single program.

Without these validity domains, the matrix can appear complete while silently
accumulating false equivalences.

## Independent validation is mandatory

The central danger is validating one translation with another translation from
the same flawed lineage. If the agent generates `A -> B` and both sides are then
lowered through the same incorrect operator table, their machine forms can agree
perfectly while preserving the same error.

Every installed rule should use at least one sufficiently independent oracle
path. Prefer agreement among several differently constructed witnesses:

1. native execution of the original;
2. target execution of the generated version;
3. machine decision-graph comparison;
4. property and metamorphic tests;
5. an independent interpreter or reference implementation;
6. round-trip transformations where they have semantic meaning;
7. symbolic or SMT checks over bounded regions;
8. adversarial differential generation.

Agreement among independent lineages is stronger evidence than agreement within
one translation lineage. Provenance must make shared dependencies visible so
apparently separate witnesses are not mistakenly counted as independent.

## Repair and installation control loop

The intended operating loop is:

1. Receive a requested conversion `A -> B` and its constraints.
2. Search the existing translation graph.
3. Rank complete paths by semantic and computational cost.
4. Attempt the best admissible path.
5. Run static contract checks.
6. Execute differential tests.
7. Lower both reference and candidate behavior into the machine proxy.
8. Compare their consequential state graphs.
9. Localize the first divergence.
10. Emit a structured `TranslationGap`.
11. Give the gap, operators, witnesses, and constraints to the repair agent.
12. Let the agent synthesize a rule, intermediate, or runtime mechanism.
13. Re-run independent verification.
14. Minimize the successful rule.
15. Generalize it against generated and adversarial cases.
16. Install it in the matrix with its validity domain, provenance, confidence,
    and cost model.

First-divergence localization is a particularly valuable repair signal. Instead
of reporting that a 20,000-line translation is wrong, the system should be able
to say:

> At decision node 417, the reference preserves alias identity while the
> candidate copies the value.

That converts a broad failure into a narrow semantic obligation.

## Diagram completion as the definition of success

The system consists of cooperating roles:

- the graph proposes semantic routes;
- the matrix records obligations and installed validity domains;
- compiler and operator tables supply known transformations;
- the bidirectional machine proxy supplies behavioral evidence;
- the test system attacks proposed equivalences;
- the repair agent manufactures missing morphisms when no valid route closes;
- impossibility witnesses preserve honest boundaries where repair cannot meet
  the constraints.

In categorical terms, the system attempts to complete a partially defined
diagram of translations and tests whether relevant paths commute:

\[
T_{B\rightarrow D} \circ T_{A\rightarrow B}
\overset{?}{\sim}
T_{C\rightarrow D} \circ T_{A\rightarrow C}
\]

The machine proxy contributes the observational equivalence relation `~`, but
does not replace independent oracles.

The resulting definition of completeness is operational and falsifiable. A
translation network is "round" not when every cell contains a bespoke compiler,
but when every demanded route can produce a verified commuting translation,
synthesize a bounded missing bridge, or precisely explain why the requested
square cannot commute.

The repair agent is therefore not asked to understand programming in the
abstract. It is handed a broken square in a semantic diagram and instructed:

> Make this square commute efficiently, or prove why it cannot.
