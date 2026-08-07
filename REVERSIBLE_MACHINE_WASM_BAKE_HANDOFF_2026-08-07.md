# Reversible Machine WASM Bake: Transition Document
**Date:** 2026-08-07
**Scope:** `turing/` compiler — `topological_reducer.py`, `graph_express2.py`, `loop_composer.py`, `glsl_deployment_strategy.py`, `c_primitive_program.py`, `fused_ir.py`, `site_bundle.py`, `machine_fork_exploration.py`, `machine_execution.py` (branch `codex/recursive-reduction-bridge`)
**Continues from:** `CLASS_STATE_ABI_TRANSITION_2026-08-07.md` (class-shaped state ABI work)

This document exists so a fresh session can pick this up with zero prior context. Everything below is verified against the real driver script, not inferred — every claimed fix was confirmed by re-running the actual build and observing the specific error change or disappear.

**Primary test target throughout:** `turing/examples/build_machine_fork_exploration_page.py` (untracked, not part of the repo — recreate it if missing; see Section 6), which compiles `machine_fork_exploration.explore_forking_paths` — the real reversible-AMD64-machine auto-forking explorer, using genuine `MachinePathForest`/`MachineExecutionOrchestrator` objects — through `build_program_bundle` with `bake_mode="whole_program"` (a fully self-contained WASM artifact, no host coordinator left at runtime).

Run it with:
```bash
cd turing && PYTHONPATH=. python examples/build_machine_fork_exploration_page.py
```

---

## 1. Where this session started

The class-shaped-state work in the prior transition doc got `explore_forking_paths` far enough to reach a new, distinct class of failure: `random_source: Callable[[], float] = random.random` couldn't be synthesized correctly, and beyond that, real bugs in ingestion/SSA-reduction that only a complex real program (not a toy) surfaces. Each was found, root-caused against real code, and fixed — see Section 2. This is the pattern for the whole session: **find the real bug by reading the actual code the error points at, fix that exact thing, verify by re-running the real build, commit.** Several early attempts in this session were source-level workarounds or guesses instead of real fixes; the user caught each one and required the actual root cause. That correction shaped the rest of the session and is worth preserving as the standing instruction for whoever continues this.

---

## 2. What was fixed, in the order it was found (all committed, branch `codex/recursive-reduction-bridge`)

### 2.1. `random_source` must be algorithmic, not frozen (`a73733c`)

**The ask:** any reference to obtaining a random value must go through a real `AbstractTensor` random generator — not a source-level default restored verbatim, and not a value frozen from one discovery trace. A compiled/AOT program can't reach stdlib `random`'s opaque global state at runtime; it can only execute what the compiler itself lowered.

**The fix:** `site_bundle.py`'s `_synthesize_state_defaults` now detects a parameter whose real default is a `random.<name>` attribute reference (structural AST match against the module's own `import random` aliases, with a fixed-depth check — see `_stdlib_random_module_aliases`/`_stdlib_random_attribute`) and redirects the guard to `AbstractTensor.random.<name>` (a real, complete, deterministic Xoroshiro128** generator, `src/common/tensors/abstraction_methods/random.py`) instead of the invalid `Callable[[], float]()` synthesis. Critically, the guard binds the **bound method itself**, not a called value — `random_source() ` inside the loop stays a genuine per-call operation.

**Verified:** trace showed `random_source` resolved to `method:<bound method Random.random of ...>`, confirmed via the real build's diagnostic dump.

### 2.2. Three ingestion bugs surfaced getting the real program through discovery (also `a73733c`)

1. **`root_head_id` probe shape** — a bare `0` was broadcast into a size-4 numeric probe array by `_probe_value`'s generic scalar rule. Fixed the driver script to use the existing `{"literal": 0}` probe form (the correct existing mechanism for "this is a literal, not a numeric-kernel array").
2. **`_RuntimeAnnAssignNormalizer` stripped class-level `AnnAssign`** (`graph_express2.py`) — it walked the *entire* tree, including class bodies, converting every `AnnAssign` to `Assign`. A class body's own `AnnAssign` is its field schema (name/type/default) — the class-table builder (`topological_reducer.py`'s `class_field_defaults`) reads it directly off the untouched `ClassDef` AST. Any locally-defined dataclass silently lost its entire field list before the class table ever saw it. Fixed: `visit_ClassDef` now only recurses into method bodies, not the class's own direct-child statements.
3. **`ProcessGraph.build_graph`'s `GetAttr` special-case never descended into `node.value`** — harmless for a bare-name receiver (resolved elsewhere, on demand, by lexical lookup) but fatal for a compound receiver (`forest.heads.get`, a `GetAttr` whose own `.value` is another `GetAttr`): the inner node never got a graph node at all, so nothing downstream could ever wire its own receiver. Fixed: the special-case branch now recurses into `node.value` (role `"value"`) before returning, matching what the neighboring docstring already (incorrectly) claimed it did.

### 2.3. `Try`/`except` — completed an already-half-built SSA reduction, not new machinery (`f707ee7`)

**Do not re-litigate this one; it was investigated at real depth and the design is correct.** `topological_reducer.py`'s `reduce_statement` already had a working SSA reduction for `ast.If` (builds a `Phi` node when the two branches assign different values to the same name) and a *separate*, less complete one for `ast.Try` — the Try reduction merged branch environments but only kept a name when both branches produced the *identical* value; when they genuinely differed (the ordinary shape of `try: x = f() except E: x = None`), the name was silently dropped, leaving later references to resolve as an opaque external input.

**The fix:** when Try branch values differ, bind the name to the `Try` node's own id instead of dropping it. No new node type needed — `evaluate_node`'s existing `ast.Try` handling (`glsl_deployment_strategy.py`) already re-runs body/handlers and returns whichever arm applies, the exact same on-demand resolution a `Phi` gives an `If`. With the value now resolvable, `loop_composer.py`'s blanket `"ast.Try disqualifies a loop"` rule stopped reflecting reality for this shape, so `ast.Try` was removed from its forbidden set (`Raise`/`With`/`AsyncWith`/`Await` remain forbidden — none of those have an equivalent reduction).

**A rejected shortcut, explicitly, for the record:** an earlier attempt in this session "fixed" the same symptom by editing `machine_fork_exploration.py`'s source to remove the `try/except` entirely (moving the underlying `VocabularyDecodeError` catch into `machine_execution.py`, matching a pattern used at other call sites). The user correctly rejected this — it made this *one* program's *one* try/except go away without giving the compiler any actual capability; a different `try/except` anywhere else would hit the identical blanket refusal. It was fully reverted before the real fix above was written. If you find a similar "make the symptom disappear by editing the source program" fix anywhere, revert it and find the compiler bug instead.

### 2.4. `Meta.shape_source_ids` — real shape provenance, not a bigger version of 2.3 (`4f78bb5`)

**Context, precisely:** `Meta` (`fused_ir.py`) is per-value metadata in the numeric IR (`FusedProgram.meta`). `shape` has always been a concrete tuple captured once, from whatever a discovery trace happened to observe — correct for that trace, not necessarily any other real run. `Meta` already had exactly this "usually absent, sometimes a real reference" pattern for a different field: `source_id` (a view's buffer-owning value, chased by `resolve_view_source`). `shape` never got the same treatment.

**The fix:** added `shape_source_ids: tuple[int | None, ...] | None` to `Meta` — `None` per dimension for a genuine compile-time constant, a real ProcessGraph node id when that dimension's extent is computed at runtime. Wired at `_observe_process_graph_node` (`glsl_deployment_strategy.py`), which already correlates a tape primitive with its ProcessGraph node: when that node is a creation operator (`CREATION_OPERATORS` from `operator_catalog.py`) whose size argument isn't provably static (`_source_static_value`, the existing helper — not a shallow type guess, which was tried and was wrong on the first attempt), the result's `shape_source_ids` now records the size argument's real node id.

**Verified directly**, not assumed: a probe compiling `AbstractTensor.zeros((len(items),))` shows `shape_source_ids=(4,)` (the `len(items)` call's node); the same call with a literal `(3,)` shows `shape_source_ids=None`. Both checked in the same probe run.

### 2.5. The origin was metadata, not a dependency — fixed for real (`87cbf5a`, `64388e4`)

**The gap, precisely (found via direct user pushback, correctly):** `Meta.shape_source_ids` alone was inert. Dependency searching in this compiler only ever walks real edges — ProcessGraph parent/child links, or `OpStep.input_ids` in the flat numeric IR. `shape_source_ids` was a side-channel note on `Meta`, not an edge; nothing was ever going to find it. Deeper still: `_wrap_creation_fn` (`abstraction.py`) records every creation call to the tape with `inputs=[]`, unconditionally — the tape only tracks tensor-to-tensor flow (it's the autodiff mechanism), so a plain Python `size` tuple was *never* going to survive as a tracked dependency there, regardless of anything in the compiler. It got baked into `OpStep.attrs` as a frozen literal, with zero input edges — confirmed directly by printing the actual `OpStep`.

**The fix:** `_compile_single_native_node` (`c_primitive_program.py`) now promotes a non-static size to a genuine operand — added to `input_ids`/`feeds` directly under its real ProcessGraph node id (already resolved; no transient-tape-identity remapping needed, unlike a real tensor operand). Verified: `ordered_feed_ids` (the existing, generic, unmodified feed-dependency walker) now finds it automatically, with zero special-casing added anywhere in that walk.

**The follow-up correction (`64388e4`):** the first version of this fix left `attrs['size']`/`attrs['shape']` in place as a "harmless fallback" alongside the real operand. The user correctly identified this as not harmless — a stale, one-trace-only literal sitting next to the real dependency is a silent-wrong-answer trap for any future consumer that doesn't yet know to look for `dynamic_shape_input_id`. Fixed: both keys are popped once the operand is promoted (plus a second, later `kernel_kind == "fill"` branch in the same function that was unconditionally re-setting `attrs["shape"]` right after — found only by re-running the probe and seeing the literal reappear). A consumer not yet built for `dynamic_shape_input_id` now fails loudly on the missing key instead of silently compiling a fixed-size buffer that's wrong for every run but the one that was traced.

**Still not done:** no backend actually reads `dynamic_shape_input_id` to allocate/grow memory at runtime. The dependency is real and correctly tracked now; nothing consumes it to change what gets compiled. This is real, scoped, future work — not started.

### 2.6. `.extend(generator)` — a redirect to already-solved machinery, not new capability (`baaf8e5`)

**The size question was already answered elsewhere; this session initially got this one wrong.** `pending.extend(child.head_id for child in children)`'s state effect classified as `opaque` because `topological_reducer.py`'s state-effect classifier only recognized `.append(single_arg)`. The first instinct — "widen the same heuristic to `.extend`" — was correctly rejected: `.extend`'s argument is a generator, whose element count isn't known until runtime, and naively publishing the generator object itself as if it were one scalar item would be a silent wrong answer, not a fix.

**What actually resolved it:** `LoopComposer` already has a real, separate mechanism for exactly this — a generator expression used as a comprehension's materializer (`loop_composer.py:1960-1993`, the `"generator expression is itself the materializer"` case). The generator's collected result is already a tracked value by the time `.extend` sees it; `.extend(generator)` only needed to be recognized as "concat this already-materialized collection," not as a new dynamic-count-tracking feature.

**The fix:** `state_effects` classification (`topological_reducer.py`) now also marks `.extend(x)` as `indexed_publication` when `x` is a `GeneratorExp`/`ListComp` with exactly one argument — same mode as `.append`, distinguished downstream by `effect.operator`.

**Verified against the real build:** the `opaque-state-effect` blocker disappeared from all three flagged loops.

### 2.7. `remove_loops` defaulted to `True`, forcing DISPATCH regardless of everything above (also `baaf8e5`)

**The last blocker in this whole chain, and the smallest fix.** Even with 2.3–2.6 done, `explore_forking_paths`' loops still showed `strategy=dispatch` — unconditionally checked in `loop_composer.py`'s blockers list, independent of state-effect classification. Root cause: `site_bundle.py`'s `remove_loops` config defaults to `True` (correct for the common flat-numeric-kernel case, where a loop has no meaningful identity worth preserving structurally), which sets `native_while=False`, which forces every `while` loop to `DISPATCH` strategy no matter what.

**The fix:** declared `TURING_PAGE = {"remove_loops": False}` at module level in `machine_fork_exploration.py` — the established per-source-file config mechanism (`ast.literal_eval`'d, same one `bake_mode`/`schedule_preference` already use). This program's two `while` loops *are* the exploration itself, not incidental iteration.

**Verified against the real build: the entire WASM whole-program control-lowering refusal is gone.** No more `strategy=dispatch`, `opaque-state-effect`, or `Try` on any loop in this program.

---

## 3. Current blocker (not yet investigated)

The build now fails past the entire control-lowering stage, on a new and unrelated error:

```
ValueError: compiled state feedback does not match the Python ABI; missing inputs=['policy', 'random_source']; missing outputs=['policy', 'random_source']
```

Raised in `site_bundle.py:2297` (`build_program_bundle`). This is a separate validation from everything in Section 2 — an ABI/`state_feedback` consistency check that `policy` and `random_source` don't satisfy. Not yet traced. Given the pattern established this session, the right first step is reading the actual check at `site_bundle.py:2297` and the surrounding `state_feedback` construction, not guessing.

Plausible starting hypothesis (unverified): `policy` and `random_source` are both parameters synthesized by `_synthesize_state_defaults` (Section 2.1's mechanism) — `policy` gets a real `ForkExplorationPolicy()` construction, `random_source` gets the `AbstractTensor.random.random` bound-method redirect from 2.1. Neither is a `state_feedback` parameter in the ordinary sense (state that should thread from one call's output back into the next call's input) — they're one-shot configuration/utility values. The ABI check may be incorrectly expecting every synthesized parameter to also be a `state_feedback` round-trip, which wouldn't be correct for these two. Verify against the real code before acting on this.

---

## 4. Known, deliberately deferred, real gaps (do not re-derive; act only if asked)

- **`head.status = ...` — object-field mutation through a parameter, several hops deep.** `head = forest.heads.get(head_id)`; `head.status = ...` mutates a field on an object reached by dereferencing into the function's own parameter (`forest`). Every loop-state mechanism in this compiler (`loop_carried_bindings`, `state_effects`, `iteration_outputs`) is built around lexically-scoped *local variable* rebinding. None of them has any concept of "this loop mutates a field on an object owned outside the loop." This is not a narrow whitelist gap like 2.6 was — there is no existing code path that even attempts to classify this kind of mutation. Fully uninvestigated; do not assume it's solved by anything in Section 2.
- **Backend codegen for `dynamic_shape_input_id`** (end of Section 2.5): the dependency is real; nothing reads it to actually allocate or grow memory at runtime instead of a frozen buffer size.
- **`pending.pop()` / `pending.pop(0)`** — return values are used (assigned), so they never enter the discarded-call `state_effect_calls` collector at all. Whatever happens to `pending`'s size here is tracked only through ordinary carried-binding rebinding of the name; the shrink itself is invisible to any state-effect mechanism. Hasn't surfaced as a blocker, but that means nothing has asked the question yet, not that it's correctly modeled.
- **`.extend`'s classification is narrow by design** — only a `GeneratorExp`/`ListComp` with exactly one call argument. A `list.extend(some_other_list_variable)` (not a comprehension literal) is not covered and would still classify `opaque`. Widen only if a real program needs it.

---

## 5. Working principles that mattered this session (preserve these)

- **Read the actual code the error names before proposing a fix.** Every wrong turn this session (the `Try` source-edit shortcut, the first shallow `Meta` type-check, leaving the stale `attrs` literal, the initial "just widen `.extend`" instinct) came from reasoning about architecture without first reading the exact function involved. Every correct fix came from reading the real code, then usually finding the fix was smaller and more precedented than initially feared.
- **Verify with a real probe or the real build, not a plausibility argument.** Every fix in Section 2 has a specific, quoted verification (a trace value, a test count, a diff in the build's error message) — not "this should work."
- **A source-level edit that makes one program's symptom disappear without giving the compiler new capability is not a fix.** See 2.3's rejected shortcut.
- **When something looks like it needs a big new subsystem, check for an existing analogous mechanism first.** `source_id` on `Meta` predicted `shape_source_ids`'s shape exactly (2.4). The comprehension materializer predicted `.extend`'s fix exactly (2.6). This compiler has been built with real, generalizable patterns already in place more often than it's been missing them.

---

## 6. Reproducing the environment

`turing/examples/build_machine_fork_exploration_page.py` is untracked (deliberately, per the prior transition doc) — if it's missing, recreate it:

```python
from types import SimpleNamespace

from src.compiler import machine_fork_exploration as mfe
from src.compiler.amd64_machine_semantics import default_effect_handlers
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator, MachineExecutionState, ReversibleMachineExecutor,
)
from src.compiler.machine_path_forest import MachinePathForest
from src.compiler.site_bundle import build_program_bundle

entry = 0x2000
program = SimpleNamespace(
    image=SimpleNamespace(image_base=entry, entrypoint_rva=0),
    functions=(SimpleNamespace(report=SimpleNamespace(instructions=())),),
)
orchestrator = MachineExecutionOrchestrator(program, effect_handlers=default_effect_handlers())
reversible = ReversibleMachineExecutor.create(orchestrator, MachineExecutionState(pc=entry))
forest = MachinePathForest(reversible, maximum_heads=64)

build_program_bundle(
    open(mfe.__file__, encoding="utf-8").read(), "C:/dev/Powershell",
    source_filename="machine_fork_exploration.py",
    entrypoint="explore_forking_paths",
    python_package="src.compiler",
    probes={"forest": forest, "root_head_id": {"literal": 0}},
)
```

Run with `PYTHONPATH=.` set to the `turing/` directory root.

Relevant test suites for the files touched this session:
```bash
PYTHONPATH=. python -m pytest \
  tests/test_abstract_tensor_topological_reducer.py \
  tests/test_control_source.py \
  tests/test_loop_composer.py \
  tests/test_machine_fork_exploration.py \
  tests/test_c_primitive_program.py \
  tests/test_machine_execution_aot_compile.py -q
```
All 93 pass as of this document.
