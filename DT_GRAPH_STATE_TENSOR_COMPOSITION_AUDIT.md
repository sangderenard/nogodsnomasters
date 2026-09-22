# dt_graph audit: from arbitrary physics equations to a composed state tensor

**Scope.** This audits `turing/src/common/dt_system` (dt_graph, its scheduler,
its registries) against one question: given an arbitrary set of law
equations — such as [engine_toy/honorary_engine_equation_catalogue.py](/C:/dev/Powershell.worktrees/markdown-equations-to-python-classes/engine_toy/honorary_engine_equation_catalogue.py)
and its source, [HONORARY_ENGINE_EQUATION_CATALOGUE.md](/C:/dev/Powershell.worktrees/markdown-equations-to-python-classes/HONORARY_ENGINE_EQUATION_CATALOGUE.md) —
how would we mechanically construct (a) each law's own state-tensor
composition and (b) one total, dependency-resolved state-tensor composition
across the whole set, the way dt_graph composes a step across engines?

Every file/line reference below was read directly, not paraphrased from
memory. Where dt_graph does **not** yet do something the question asks for,
that is stated as a gap, not implied to already exist.

**Correction, added after the first pass (see §3).** Sections 1-2 below
describe `dt_graph.py`'s `GraphBuilder`/`MetaLoopRunner`/`ILPScheduler`
machinery accurately, but that machinery is **not the production lane** —
`llvm_dt_system.py`'s own docstring says so directly: "no `GraphBuilder`/
`MetaLoopRunner` (that layer is optional composition on top and has never
been lowered...)". The actual, running union-of-state mechanism, used by
both the atmosphere/raincloud chamber and the newer Faraday/EM law set, is
`llvm_dt_system.py`'s `column_names_of`/`state_source`/`participant_registry`
— §3 covers it. §6-7 were also corrected: the general answer to a coupling
"cycle" is not graph condensation, it is how the boundary-transfer term
between two laws is written (see §7's rewrite).

## 1. What dt_graph actually is

[dt_graph.py](/C:/dev/Powershell/turing/src/common/dt_system/dt_graph.py) is a
graph *of scheduling*, not a graph of physical state dependency. Its node
types:

- `StateNode` — a reference to one simulator's state object (opaque `Any`).
- `AdvanceNode` — a leaf that calls `advance(state, dt) -> (ok, Metrics, state)`
  and writes the result back to its `StateNode`.
- `ControllerNode` — an `STController` + `Targets` + `dx`.
- `EngineNode` — adapts an `EngineRegistration` into an `AdvanceNode`, wiring
  `sync_engine_from_table` → `step_with_state` → `publish_engine_to_table`
  ([dt_graph.py:89-133](/C:/dev/Powershell/turing/src/common/dt_system/dt_graph.py)).
- `RoundNode` — a superstep: a `plan`, a `controller`, `children`
  (`AdvanceNode`s or nested `RoundNode`s), and a `schedule` string:
  `"sequential" | "interleave" | "parallel"`
  ([dt_graph.py:139-158](/C:/dev/Powershell/turing/src/common/dt_system/dt_graph.py)).

`MetaLoopRunner.set_process_graph` turns a `RoundNode` tree into an execution
order via `DtToProcessAdapter` + `ILPScheduler`
([dt_graph.py:344-368](/C:/dev/Powershell/turing/src/common/dt_system/dt_graph.py)),
then `run_round` walks that order once per frame.

**`EngineRegistration`** ([engine_api.py:277-290](/C:/dev/Powershell/turing/src/common/dt_system/engine_api.py))
carries `name, engine, targets, dx, distribution, ctrl, localize,
solver_config` — **no `owned_state`/`read_state`/"columns" field**. The
"columns" language in this repo's own `AGENTS.md` ("Engines register their
columns as parameters; the dt system saves and restores them") refers to
each engine implementing its own `sync_from_state`/`publish_to_state` hooks
against a shared **`StateTable`**
([state_table.py:22-39, 232-266](/C:/dev/Powershell/turing/src/common/dt_system/state_table.py)),
which is a plain `Dict[Key, Any]` plus some mesh identity maps
(`group_to_vertices`, `vertex_to_groups`, ...). There is no declared,
machine-checkable list of which state names an engine owns versus reads —
only two hardcoded adapters (`_sync_demo_state`, `_sync_fluid_state`) and an
escape hatch for an engine to supply its own hook. **Nothing here builds a
per-engine or total "state tensor."**

## 2. Where "dependency" and "cycle" actually mean something today

The real scheduling logic lives in
[dt_process_adapter.py](/C:/dev/Powershell/turing/src/common/dt_system/dt_process_adapter.py)
and [transmogrifier/ilpscheduler.py](/C:/dev/Powershell/turing/src/transmogrifier/ilpscheduler.py).

`DtToProcessAdapter._emit_round` builds one edge per **authored schedule
adjacency** — for `schedule="sequential"`, it links the previous child
group's tail to the next child group's head; for `"parallel"`, groups get no
edges at all ([dt_process_adapter.py:75-107](/C:/dev/Powershell/turing/src/common/dt_system/dt_process_adapter.py)).
**This graph is built from the human-authored `RoundNode.schedule` string,
never from which state fields an `AdvanceNode` reads or writes.** Two
engines that both mutate the same physical quantity but are marked
`"parallel"` get no edge; the graph cannot know they conflict, because it was
never told what either of them touches.

`ILPScheduler.compute_asap_levels` does contain real cycle **detection** — a
white/gray/black DFS that raises `ValueError("Cycle detected during
scheduling; node in cycle: {n}")` the moment it revisits a gray node
([ilpscheduler.py:38-54](/C:/dev/Powershell/turing/src/transmogrifier/ilpscheduler.py)).
That is *detect-and-refuse*, not *resolve*: today's dt_graph never resolves a
cycle, it forbids the human from authoring one. "Dependency-resolved cycles
like what dt graph does" does not exist yet as stated — it is the gap this
audit is scoping, not a mechanism to imitate.

Also noted in passing, not part of this audit's scope to fix: `compute_levels`
has an unconditional `return return_value` before its second `match order:`
block ([ilpscheduler.py:14-25](/C:/dev/Powershell/turing/src/transmogrifier/ilpscheduler.py)),
so the `order="processing"` (reversed-level) branch is dead code. Any design
built on top of `compute_levels` should not assume `order` currently does
anything.

## 3. The actual production lane: `llvm_dt_system.py`, not `dt_graph.py`

[`llvm_dt_system.py`](/C:/dev/Powershell/turing/examples/llvm_dt_system.py)
says outright, in its own module docstring, that the layer §1-2 just
described is not what runs:

> one `state` object..., one `STController` kept alive across frames, and
> one `advance(state, dt)` function... Nothing else of the dt system is
> needed -- no `DtCompatibleEngine`, no `StateTable`, no
> `GraphBuilder`/`MetaLoopRunner` (that layer is optional composition on
> top and has never been lowered; see llvm_dt_system.py.bak for the
> discovery).

`chamber_raincloud_demo.py` (the atmosphere/raincloud chamber scenario)
repeats the same rule even more bluntly: *"The simulation is produced by
the sanctioned lane... Never the AbstractTensor interpreter. Ever."* This is
the actual union/composition mechanism, and it is real and mechanical,
which corrects what §1 says about `EngineRegistration` having no
owned/read declaration — that is true of `EngineRegistration`, but the
compiled units this lane actually uses declare exactly that:

- **`column_names_of(pieces)`** ([llvm_dt_system.py:46-53](/C:/dev/Powershell/turing/examples/llvm_dt_system.py))
  walks a list of already-compiled `LLVMPiece` objects — each carrying its
  own `argument_names` (what it reads) and `output_names` (what it writes,
  by a `_next` suffix convention) from compilation — and returns **the union
  of every column any piece reads, in first-appearance order**, skipping
  `dt`. This is the union-negotiation the earlier draft of this audit
  claimed didn't exist; it does, and it works off declarations each piece
  already carries from compilation, not a new declaration this audit would
  need to invent.
- **`state_source(columns)`** ([llvm_dt_system.py:55-83](/C:/dev/Powershell/turing/examples/llvm_dt_system.py))
  spells out a real `PieceState` class from that union, one span field per
  column, plus the `StepSpans`/`Metrics` publication fields from §4 below —
  generated as source text and `exec`'d, so the same text that runs in
  Python is what is hard-compiled.
- **`participant_registry(pieces)`** ([llvm_dt_system.py:223-234](/C:/dev/Powershell/turing/examples/llvm_dt_system.py))
  declares each piece as a `ParticipantRegistry` participant "once, in
  causal order" — and that order is **the order the caller's `pieces` list
  was given in**, not anything computed from a dependency analysis.
- **`dt_system(piece_files, columns, ...)`** ([llvm_dt_system.py:253-283](/C:/dev/Powershell/turing/examples/llvm_dt_system.py))
  is the real entry point: load pieces → union their columns via the above
  → build `PieceState` → step via `run_superstep`.

Two law modules are already written in the shape this lane consumes —
`symbolic_chamber_solvers.py` (the atmosphere/raincloud chamber: air,
species, droplet, surface and pool laws) and `symbolic_em_solvers.py` (a
newer Faraday/EM law set, not yet wired to its own demo/join driver the way
the chamber is). Both declare `LAWS`, `LAW_PUBLICATIONS` and `SCHEDULE`,
compiled per-law via `tools/compile_symbolic_source.py`, whose `schedule`
argument feeds `compile_sympy_equations`'s *internal* ASAP/ALAP ordering of
one law's own sub-expression graph — a different, narrower use of
`ILPScheduler` than §1-2, and not a cross-law ordering at all.

**What is still true even in the real lane:** the union of columns is
mechanical, but the *order pieces run in* — hence which piece's write is
visible to which piece's read within one step — is exactly what the
caller's `piece_files` list says it is. Nothing here computes that order
from a dependency graph either. That is addressed directly in §7, because
it turns out not to need one.

## 4. The part of dt_system that already IS the right shape to imitate

Three sibling files build exactly the kind of composed, presence-masked
tensor the audit's question is asking for — just scoped to **diagnostics**,
not physical state:

- [`time_contracts.py`](/C:/dev/Powershell/turing/src/common/dt_system/time_contracts.py):
  `ParticipantRegistry.declare(name) -> int` assigns a dense monotonic id at
  build time, in causal (declaration) order
  ([time_contracts.py:58-86](/C:/dev/Powershell/turing/src/common/dt_system/time_contracts.py)).
  `HOLD/BIND/DILATE/SUBCYCLE` are a small declared *contract* each participant
  states about how its own time constant should be treated.
- [`error_channels.py`](/C:/dev/Powershell/turing/src/common/dt_system/error_channels.py):
  the same dense-id pattern for diagnostic channels, with an explicit design
  argument for why (string-keyed dicts hash on the step path and fnv1a-64
  keys do not survive a float64 mantissa) ([error_channels.py:1-45](/C:/dev/Powershell/turing/src/common/dt_system/error_channels.py)).
- [`participants.py`](/C:/dev/Powershell/turing/src/common/dt_system/participants.py):
  `Publication` is exactly "one law's individual composition" — optional
  `channels`, `dt_limit`, `tau_s`, `contract`, per-participant `limits`, with
  every field's *absence* tracked as its own span rather than defaulted to
  zero ([participants.py:87-106](/C:/dev/Powershell/turing/src/common/dt_system/participants.py)).
  `StepSpans.of(registry, published)` is exactly "the total composition": it
  walks `registry.declared()` in causal order and produces dense arrays —
  `pub_tau, pub_tau_present, pub_contract, pub_dt_limit,
  pub_dt_limit_present` shaped `(P,)`, and `pub_values, pub_present,
  pub_limits, pub_limits_present` shaped `(P*C,)`
  ([participants.py:109-196](/C:/dev/Powershell/turing/src/common/dt_system/participants.py)).
  `Metrics` carries these fields directly and `STController`/`dt_controller.py`
  consume them by index, never by name, on the step path
  ([dt_controller.py:134, 231-236, 301-337](/C:/dev/Powershell/turing/src/common/dt_system/dt_controller.py)).

This is the load-bearing insight for the audit: **dt_system already has a
correct answer to "how do you compose many participants' individual
contributions into one total tensor without losing who-published-what" —
it is just never been pointed at physical state.** Building a
`PhysicsStateRegistry`/`PhysicsStateSpans` pair on this exact pattern, rather
than inventing a new one, is the path of least resistance and the one this
repo's own `AGENTS.md` would insist on ("use the existing system").

## 5. What the equation catalogue already gives us for free

[honorary_engine_equation_catalogue.py](/C:/dev/Powershell.worktrees/markdown-equations-to-python-classes/engine_toy/honorary_engine_equation_catalogue.py)
has 612 `eq_*` sympy `Eq`/relational objects across 22 honorary-engine
prefixes, discoverable via its own `_discover_equations()`
([honorary_engine_equation_catalogue.py](/C:/dev/Powershell.worktrees/markdown-equations-to-python-classes/engine_toy/honorary_engine_equation_catalogue.py)).
Two things it does **not** yet carry, that a law needs before it can be
composed the way section 4 above composes participants:

1. **owned vs. read split.** Every `eq_XX_n` is `sp.Eq(lhs, rhs)` (or an
   inequality). For the overwhelming majority, `lhs` is the quantity the law
   defines (its column to *own*) and every other free symbol on either side
   is something it *reads*. This is mechanical: `lhs.free_symbols` minus
   bound/dummy symbols (`sp.Sum`/`sp.Integral` index variables) is `owned`;
   `(lhs.free_symbols | rhs.free_symbols) - owned - {universal constants}` is
   `read`. It fails only where a law states a pure constraint with no clear
   LHS-as-definition (`eq_N5_1 = g_n >= 0`, `eq_G7_2 = L_xi >= 0`) — those
   need a human tag (`"constraint"`, owns nothing) rather than the heuristic.
2. **canonical identity.** The symbol registry section of the same file
   (`KNOWN_COLLISIONS`, `describe_symbol`, `raw_token_report`) is precisely
   the missing piece that keeps step 1 honest: without it, Timoshenko's `M`
   (bending moment) and De Laval's `M` (Mach number) would both mechanically
   resolve to the *same* free symbol `Symbol('M')`, and a naive dependency
   builder would wire a law that reads Mach number to "depend on" whichever
   engine last wrote a bending moment. `describe_symbol(token, engine=...)`
   already returns the disambiguated identity when curated, and an honest
   "not curated yet" list otherwise — exactly the gate a dependency-graph
   builder must pass every symbol through before turning it into an edge.

## 6. Constructing each law's individual state-tensor composition

For one law `eq_XX_n`, with `owned`/`read` extracted per §5.1 and each symbol
resolved per §5.2:

```
LawComposition(
    law_id="XX_n",
    engine="Timoshenko",              # from ENGINE_PREFIXES
    owns=(CanonicalField("bending_moment", "M_t"),),
    reads=(CanonicalField("shear_area", "kappa_s*G_t*A_t", ...), ...),
)
```

This is structurally identical to a `Publication` (§4): `owns` is what this
law claims to produce this step (like `Publication.channels`), `reads` is
what it consumes. The only addition beyond `Publication`'s fields is that
canonical-identity resolution step, because law names collide across
engines in a way participant names in the existing system never do (a
`ParticipantRegistry` name is chosen once, by a human, precisely to avoid
this; a sympy display string was chosen for LaTeX fidelity and was never
meant to be collision-free).

## 7. Cross-system coupling does not need a dependency graph

**This section was rewritten.** The first pass proposed detecting cycles in
a law → law graph and condensing strongly connected components into
simultaneous-solve groups (`networkx.condensation` + a fixed-point/Newton
`RoundNode`). That is unnecessary machinery for the general case, for a
reason already visible in code this audit had read but not connected:

**Order-sensitivity between two coupled laws is a property of how their
boundary-transfer term is written, not an inherent property of a
dependency graph.** `participants.py` already says as much: the participant
axis is in *causal order*, and "participant `i` may consume what `i-1`
published this same step" — which is a **choice**, not a discovered fact.
A law's coupling term either:

- reads "whatever the other side has published so far this step" — same-step
  coupling, order matters, and the author chose that on purpose, or
- reads "the other side's *previous* step" — a one-step lag, completely
  ordinary in split/multiphysics schemes — and order stops mattering,
  because nothing this step depends on anything else this step.

Neither case needs a graph algorithm to discover; the modeler decides it
where the boundary term is written. This is also, concretely, why
`dt_negotiation_helpers.py.bak` was rejected (`turing/examples`, header:
*"DEPRECATED -- DO NOT USE, DO NOT IMPORT, DO NOT REVIVE"*): it tried to
solve ordering/claims generically **inside the dt system**, and its own
postmortem says the right place for that decision is *"at a molecular
causal boundary of the sim... not in the dt system."* A genuine algebraic
loop (two quantities mutually and instantaneously determined, e.g. a stiff
equation-of-state/momentum pair that truly cannot be lagged) is a narrower,
real case — but it is the exception a modeler flags explicitly, not the
default a scheduler must discover and condense for every coupling.

**The two-tier channel split already answers "how do coupled systems'
limits reconcile."** dt_system already separates two different kinds of
limit, and this split is the actual mechanism, not a graph:

- **Personal boundaries**: `Publication.channels`/`Publication.limits`
  (§4) let each system publish what *it* measured against limits declared
  for *it*, landing in `StepSpans.pub_values`/`pub_present`/`pub_limits` —
  one system's own business, judged against its own bar.
- **Outer, cross-system deltas**: `Metrics.error_channels`/
  `Targets.error_limits` (the aggregate channels in `error_channels.py`,
  e.g. `mass_err`, `div_inf`) are where a *boundary-transfer* consistency
  measure belongs — "how much did what system A said left differ from what
  system B said arrived" is exactly the shape of an aggregate error channel
  with its own limit, checked by the same `STController` that already
  grows/shrinks/rejects `dt` on every other channel.

So constructing the total composition for an arbitrary set of laws needs
§4's union (declare every field once, causal order) and §5's owned/read
split (to know what each law is a candidate personal-channel publisher
for) — but *not* a cycle-resolution pass. A cross-system delta that needs
watching is declared as its own outer error channel, exactly the way
`mass_err`/`div_inf` already are, and the boundary term that produces it is
written to read a lagged or same-step value on purpose.

## 8. Concrete next steps, in build order

1. Write the owned/read extractor over `_discover_equations()`'s output
   (§5.1) — pure sympy, no dt_system dependency yet.
2. Route every extracted symbol through `describe_symbol`/`raw_token_report`
   (§5.2) and produce a report of every *uncurated* collision the current
   `KNOWN_COLLISIONS` table doesn't cover — this will be large (§ "Symbol
   Identity Registry" in the catalogue file already says as much), and is
   the actual bottleneck: the union in §3/§6 is only as trustworthy as the
   identity resolution feeding it.
3. Follow `llvm_dt_system.py`'s own shape (§3), not a new adapter: for a
   chosen, deliberately small subset of laws first (e.g.
   `ENGINE_SETS['chamber_witness']`, already defined in the catalogue), get
   each law to the same `argument_names`/`output_names` shape a compiled
   `LLVMPiece` already has, so `column_names_of` can union them the same
   way it unions the chamber/EM law pieces today. This is a much smaller
   lift than a new scheduler: it reuses `column_names_of`,
   `participant_registry` and `StepSpans` outright rather than building
   parallel machinery.
4. Where a law's coupling term needs a specific choice (same-step vs.
   lagged, per §7), make that choice explicitly in the term itself, and
   where a boundary needs watching, publish it as its own aggregate error
   channel (`error_channels.py`) with its own limit — not as a new
   scheduling primitive.
5. Only reach for `dt_graph.py`/`GraphBuilder`/`MetaLoopRunner` if a
   genuine, narrow algebraic loop (§7) is found that truly cannot be
   lagged — and even then, treat it as the rare, explicitly-flagged case
   it is, not the default assumption for every coupling.

No code changes were made for this audit; it is analysis only, per the
request to "start analyzing and auditing."
