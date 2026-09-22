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

## 3. The part of dt_system that already IS the right shape to imitate

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

## 4. What the equation catalogue already gives us for free

[honorary_engine_equation_catalogue.py](/C:/dev/Powershell.worktrees/markdown-equations-to-python-classes/engine_toy/honorary_engine_equation_catalogue.py)
has 612 `eq_*` sympy `Eq`/relational objects across 22 honorary-engine
prefixes, discoverable via its own `_discover_equations()`
([honorary_engine_equation_catalogue.py](/C:/dev/Powershell.worktrees/markdown-equations-to-python-classes/engine_toy/honorary_engine_equation_catalogue.py)).
Two things it does **not** yet carry, that a law needs before it can be
composed the way section 3 above composes participants:

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

## 5. Constructing each law's individual state-tensor composition

For one law `eq_XX_n`, with `owned`/`read` extracted per §4.1 and each symbol
resolved per §4.2:

```
LawComposition(
    law_id="XX_n",
    engine="Timoshenko",              # from ENGINE_PREFIXES
    owns=(CanonicalField("bending_moment", "M_t"),),
    reads=(CanonicalField("shear_area", "kappa_s*G_t*A_t", ...), ...),
)
```

This is structurally identical to a `Publication` (§3): `owns` is what this
law claims to produce this step (like `Publication.channels`), `reads` is
what it consumes. The only addition beyond `Publication`'s fields is that
canonical-identity resolution step, because law names collide across
engines in a way participant names in the existing system never do (a
`ParticipantRegistry` name is chosen once, by a human, precisely to avoid
this; a sympy display string was chosen for LaTeX fidelity and was never
meant to be collision-free).

## 6. Constructing the total, dependency-resolved composition

1. **Build a bipartite law → field graph**: for every `LawComposition`, add
   edge `field → law` for each `owns` entry and `law → field` for each
   `reads` entry.
2. **Project to a law → law graph**: `law_B` depends on `law_A` if `law_A`
   owns a field `law_B` reads. Multiple laws owning the same field is itself
   a finding to surface, not silently allow — it means two engines both
   claim to be the owner Noether's NO3 says must be singular
   ("energy must be summed over disjoint physical stores... not counted
   again under another engine name").
3. **Detect cycles the way `ILPScheduler.compute_asap_levels` already does**
   (§2) — reuse its gray/black DFS almost verbatim — but where it raises,
   **condense** instead: `networkx.strongly_connected_components` /
   `networkx.condensation` (networkx is already a dependency of this exact
   subsystem, imported directly in `dt_process_adapter.py`) collapses each
   SCC into one node. The condensation of any directed graph is guaranteed
   acyclic, so `compute_asap_levels`'s DFS can then run unmodified on the
   condensed graph to get real ASAP levels.
4. **Decide what a condensed SCC means physically**, using a per-field
   contract in the same spirit as `time_contracts.py`'s `HOLD/BIND/DILATE/
   SUBCYCLE` (§3): a cycle is either
   - a genuine **algebraic loop** (e.g. an equation of state and a momentum
     balance that mutually determine pressure and density at the same
     instant) — solved as one coupled system inside a single `RoundNode`
     with `schedule="parallel"` and a fixed-point/Newton `ControllerNode`
     wrapping the whole SCC, or
   - a **laggable coupling** — one edge is marked to read the *previous*
     published value instead of the current one (this is exactly what
     `StepSpans`'s causal order already lets a later participant do to an
     earlier one's publication within one step; laggable-across-cycle is the
     same idea applied across a step boundary) — which breaks the cycle
     explicitly, at a stated order-of-accuracy cost that should be recorded,
     not silently assumed.
5. **Map the condensed, leveled graph onto dt_graph's existing node types**:
   canonical field → `StateNode`; law → `AdvanceNode`; an SCC needing
   simultaneous solve → a `RoundNode` (`schedule="parallel"`); the levels
   from step 3 → the sequential ordering between `RoundNode`s at the parent
   level. This is not a new execution engine — it is a `PhysicsToProcessAdapter`
   sibling to `DtToProcessAdapter`, built from data dependencies instead of
   an authored schedule string, feeding the same `MetaLoopRunner`.
6. **Compose the total state tensor with `StepSpans`'s exact discipline**
   (§3): a `PhysicsStateRegistry` declares every canonical field once, in
   the topological order step 3-4 produced; a `PhysicsStateSpans` (built the
   same way `StepSpans.of` is) holds `values`/`present` for every field, so
   "this law did not publish this field this step" and "this law published
   zero" remain distinguishable the same way dt_system already insists on
   for diagnostics.

## 7. Concrete next steps, in build order

1. Write the owned/read extractor over `_discover_equations()`'s output
   (§4.1) — pure sympy, no dt_system dependency yet.
2. Route every extracted symbol through `describe_symbol`/`raw_token_report`
   (§4.2) and produce a report of every *uncurated* collision the current
   `KNOWN_COLLISIONS` table doesn't cover — this will be large (§ "Symbol
   Identity Registry" in the catalogue file already says as much), and is
   the actual bottleneck: the dependency graph is only as trustworthy as the
   identity resolution feeding it.
3. Build the law → law graph and run condensation + `compute_asap_levels` on
   a small, deliberately chosen subset first (e.g. `ENGINE_SETS['chamber_witness']`,
   already defined in the catalogue) rather than all 612 equations at once,
   since real algebraic loops (Gibbs ↔ Navier-Stokes EOS, Fourier ↔ Gibbs
   energy-temperature closure) are concentrated in a handful of engines and
   are easiest to validate there.
4. Only then attempt the `PhysicsToProcessAdapter`/`PhysicsStateRegistry`
   wiring into `MetaLoopRunner` — reusing `dt_process_adapter.py`'s adapter
   shape and `participants.py`'s span-building code rather than parallel
   implementations of either, per this repo's own standing instruction to
   use what already exists.

No code changes were made for this audit; it is analysis only, per the
request to "start analyzing and auditing."
