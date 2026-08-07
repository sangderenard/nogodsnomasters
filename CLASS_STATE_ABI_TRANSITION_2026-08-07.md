# Class-Shaped State ABI: Transition Document
**Date:** 2026-08-07
**Scope:** `turing/` compiler — `site_bundle.py`, `topological_reducer.py`, `graph_express2.py`, `glsl_deployment_strategy.py` (read-only, bug located, not yet fixed)
**Continues from:** `CMD_BINARY_EXECUTOR_COMPILATION_HANDOFF.md` (the reversible AMD64 machine executor / no-second-interpreter mission)

This document exists because the client stopped surfacing intermediate tool actions to the user mid-session. Everything below is written so a fresh session (human or agent) can pick this up with zero prior context and know exactly what changed, why, what's proven, and what's still broken.

---

## 1. The problem this session actually solved

The immediate, concrete test case used throughout:

```python
class Counter:
    def __init__(self):
        self.value = 0

def step(counter: Counter):
    if counter is None:
        counter = Counter()
    counter.value = counter.value + 1
    return counter
```

Compiling `step` through `build_program_bundle` (the real page-generation entrypoint, `turing/src/compiler/site_bundle.py`) is meant to produce a WASM page where each call increments a persisted `Counter.value` across calls (`state_feedback` threads the previous call's output back into the next call's input — the same mechanism already used for ordinary numeric-kernel programs).

This is **not a toy**: it's the minimal reproduction of "a page whose state is a class instance," which is a real, generalizable requirement — any `.dream`/Python source with class-shaped state (not just flat numeric kernels) needs this to work. The broader mission (reversible AMD64 machine executor) has exactly this shape: `MachineExecutionState`, `MachinePathForest`, etc. are all class instances that need to survive across calls.

### Why this matters beyond the toy example

Earlier in this session (see `examples/build_machine_fork_exploration_page.py`, untracked, real driver script), the same class of problem blocked compiling `machine_fork_exploration.explore_forking_paths`, which takes a real `MachinePathForest` object. That work is **paused, not abandoned** — it needs everything below to land first, plus a separate fix for `MachinePathForest` (and any live object generally) not being JSON-serializable for `_content_version`'s content-addressed build caching (a distinct, deliberately-deferred problem — see Section 6).

---

## 2. What was fixed, in the order it was found

### 2.1. `identity_table` never named a class field's write (topological_reducer.py)

**File:** `turing/src/common/tensors/topological_reducer.py`, function `bind_target`, the `ast.Attribute` target branch (~line 1618-1685).

**The bug:** `bind_target` has two sibling branches. The `ast.Name` branch (`x = ...`) does:
```python
identity_bindings.setdefault(target.id, []).append(value)
```
— this is the line that makes `x` show up in `identity_table` at all. The `ast.Attribute` branch (`counter.value = ...`) builds a completely correct `SetAttr` graph node (correctly wired to `(receiver, "object")` and `(value, "value")`, with `attribute="value"` recorded) — but never called the equivalent `identity_bindings.setdefault(...)`. The field write was real, structurally correct graph data; it just never got a name, so nothing downstream could select it by name.

**The fix:** added, right after the `SetAttr` node and its `_replace_inputs` wiring, before `return`:
```python
if isinstance(target.value, ast.Name):
    identity_bindings.setdefault(
        f"{target.value.id}.{target.attr}", []
    ).append(node_id)
```
Only fires when the receiver is a plain name (`counter.value = ...`), not a chained/subscripted receiver — deliberately narrow, matches the one case needed, doesn't guess at anything more general.

**Verified:** `identity_table` for the test source now contains `'counter.value': (10,)` — one unambiguous node id — alongside the pre-existing `'counter': (0, 5, 6)` (the whole-object mutation history, still present, still ambiguous, now simply unused for ABI purposes).

**Test suite:** `test_abstract_tensor_topological_reducer.py`, `test_machine_execution_aot_compile.py`, `test_site_bundle.py` — 53 passed, only the same 2 pre-existing Fortran-toolchain failures (unrelated linker issue, `strndup` undefined — a `msys64`/`mingw` environment problem, not caused by any change this session).

### 2.2. `state_feedback` / WASM output naming still used the whole-object name (site_bundle.py)

**File:** `turing/src/compiler/site_bundle.py`, inside `build_program_bundle`, right before `program = project_public_numerical_program(aot)` (~line 2018).

**The bug:** `_synthesize_state_defaults` (existing mechanism from earlier this session — see Section 5) marks a class-shaped, unconfigured, type-annotated parameter for synthesis and sets `state_feedback = {"counter": "counter"}`. But `counter` never resolves to one scalar terminal — its `identity_table` entry is a 3-node ambiguous mutation history, not a value. `project_public_numerical_program` (in `aot_compile.py`, deliberately **not modified** — see Section 3 for why) selects outputs by exact name match against `compilation.function_outputs`, one scalar node per name.

**The fix:** after AOT compile, before projection, swap each synthesized parameter's whole-object name for its real field names (now discoverable per fix 2.1) in both `aot.function_outputs` and `contract.state_feedback`:
```python
if synthesized:
    updated_outputs: list[str] = []
    updated_state_feedback = dict(contract.state_feedback)
    for name in aot.function_outputs:
        if name not in synthesized:
            updated_outputs.append(name)
            continue
        fields = tuple(
            candidate for candidate in aot.identity_table
            if candidate.startswith(f"{name}.")
        )
        if not fields:
            raise ValueError(
                f"synthesized state parameter {name!r} has no "
                "discovered fields to publish as compiled outputs"
            )
        updated_outputs.extend(fields)
        updated_state_feedback.pop(name, None)
        for field in fields:
            updated_state_feedback[field] = field
    aot = replace(aot, function_outputs=tuple(updated_outputs))
    contract = replace(contract, state_feedback=updated_state_feedback)
```

This reuses `project_public_numerical_program` completely unmodified — it was never wrong, it just never had a name to work with. `dataclasses.replace(aot, function_outputs=...)` feeds it the corrected name set; `AOTCompilation` is a frozen dataclass so this is safe.

**Verified (isolated, not yet end-to-end — see Section 4 for what's still broken):** `identity_table` correctly resolves `'counter.value'` to a single node once fed the corrected `function_outputs`.

### 2.3. Locally-defined classes were never recognized as classes at all (graph_express2.py + topological_reducer.py) — the deepest, most consequential fix

This is the one that took the longest to find and matters most beyond this one test case.

**The bug, traced in three layers:**

1. `graph.G.graph["class_table"]` (consulted by `evaluate_node`'s `ast.Call` handling in `glsl_deployment_strategy.py` to route a call through the real, AST-grounded `_CompiledStructuralObject` construction path) is only built at the very end of `reduce_abstract_tensor_topology` (`topological_reducer.py:2799`), from data that itself requires the whole graph to already be reduced.
2. The much earlier `ast.Name`-resolution code that actually decides what a bare name like `Counter` *means* when used as a callee (`topological_reducer.py`, inside `_normalize_lexical_values`, ~line 1141-1209) never consulted anything about classes at all — it checks `function_table.reference(name)` for ordinary functions, and if that fails, falls through to `static_bindings`/`python_bindings` resolution, and ultimately to treating the name as a generic **external** input.
3. Confirmed empirically (debug print, since reverted) that `Counter` — a class defined in the *same module being compiled* — was resolving through `static_python_bindings`, which is populated by `_expand_python_static_bindings` (`aot_compile.py:188`): a transitive walk of `__code__.co_names`/`__globals__` across every function/class reachable from whatever `python_bindings` the caller supplied. `site_bundle.py` passes `python_bindings=globals()` (its own module globals). Somewhere transitively reachable from that is `glsl_backend.py:48`, `from collections import Counter` — a bare, unaliased import completely unrelated to the user's source. Because the user's own `Counter` was never claimed by anything upstream, this unrelated internal binding silently won the name lookup. `counter = Counter()` silently became `counter = collections.Counter()` — a real object, with no `.value` attribute, hence the `AttributeError: 'Counter' object has no attribute 'value'` seen throughout debugging (note: the *message* said `'Counter' object` because `type(collections.Counter()).__name__ == 'Counter'` too — collections.Counter's own class is *also* named `Counter`, which made this exceptionally confusing to diagnose from the error text alone).

**This is architecturally significant, not just a naming collision.** The user's explicit standing rule from earlier this session — "you can't use python bindings, you cannot make a program depending on python" — was being violated by the *architecture*, not by any one bad line: `static_python_bindings` is supposed to be a narrow escape hatch for genuine externals, but because the *real*, AST-grounded resolution path (class_table) was never populated early enough to claim the name first, the escape hatch became the only thing that ever had an answer. The user's framing, verbatim: *"we will allow all external ports, including python, but those will all be special cases over the default which is to have no outside world other than system."* The default path must always get first refusal; `python_bindings` only fires when the default path has nothing.

**The fix — two coordinated changes:**

**(a) `graph_express2.py`, inside `ProcessGraph.build_from_ast`, immediately after `self.G.graph["map_ir"] = _map_ir_from_ast(tree)` (~line 1862):**
```python
self.G.graph["class_definitions"] = frozenset(
    str(item["class_name"])
    for item in self.G.graph["map_ir"].get("objects", ())
)
```
`map_ir["objects"]` was *already* being built correctly at this exact point — `_map_ir_from_ast` walks the tree for every `ast.ClassDef` and builds a full schema (fields, methods, permissions) via `_class_schema_from_ast`. This data was correct the whole time; it just wasn't published anywhere the earlier Name/Call resolution code could see it. This one line publishes the fact "these names are locally-defined classes" as an ingestion-time, AST-derived fact — before any reduction pass runs, before any python_bindings fallback is ever consulted. This is deliberately the *only* new fact recorded; no new class-schema-building logic was written, because `_map_ir_from_ast` already does that correctly.

**(b) `topological_reducer.py`, inside the `ast.Call` handling in `_normalize_lexical_values`'s `resolve_expression` (~line 1255):**
```python
if isinstance(expression, ast.Call):
    node_id = id(expression)
    if (
        isinstance(expression.func, ast.Name)
        and expression.func.id
        in (graph.G.graph.get("class_definitions") or ())
        and node_id in graph.G
    ):
        graph.G.nodes[node_id].setdefault(
            "attributes", {},
        )["class_ref"] = expression.func.id
    callee = resolve_expression(expression.func)
    if isinstance(callee, int) and callee in graph.G:
        ...
```
(`node_id = id(expression)` was hoisted above the pre-existing `callee = resolve_expression(...)` call so this check can run first — `resolve_expression` was left completely unchanged otherwise.)

This directly sets `class_ref` on the Call node using the early ingestion-time fact — no route through `_StaticPythonReference`, no route through `static_reference_node`, no touching of `python_bindings` machinery at all. `evaluate_node`'s existing `if class_ref is not None:` branch (`glsl_deployment_strategy.py:8372`, completely unmodified) then correctly builds a real `_CompiledStructuralObject`, runs the class's own `__init__` through the compiler's own discovery/capture machinery, and produces a genuinely compiler-owned, structurally-tracked instance — never touching live CPython class objects at all for a class the source itself defines.

**Verified:** re-running the same isolated AOT probe that previously produced `AttributeError: 'Counter' object has no attribute 'value'` (i.e., the `collections.Counter` collision) now produces a *different* error — `AttributeError: 'Counter' object has no attribute 'value'` **again**, but this time confirmed (via the same debug-print technique, since removed) to be a genuine `_CompiledStructuralObject`-related failure, not the `collections.Counter` shadowing bug. This is a **different, real bug**, described in Section 4. The shadowing bug itself — confirmed via debug print showing `class_ref='Counter'` now set correctly on the Call node before removing that print — is fixed.

**Test suite:** not yet re-run after fix 2.3 specifically (see Section 7, first item). Fixes 2.1 and 2.2 were verified against the suite; fix 2.3 has only been verified via the isolated debug-print check described above and needs the full suite run before it can be called confirmed-safe.

---

## 3. Explicit boundaries respected this session (do not cross without re-confirming with the user)

- **Never touched `project_public_numerical_program`, `emit_wasm_module`, `_CompiledStructuralObject`, or the `class_ref is not None:` branch in `evaluate_node`.** All three were already correct; every fix this session was about getting the right *name*/*fact* to them earlier, never about changing their logic. This was an explicit, repeated user correction pattern this session: don't patch machinery that's already right, find why it never got asked.
- **No `exec()`, no `eval()`.** All fixes are pure AST/graph structural changes. `_synthesize_state_defaults` (Section 5) uses `ast.parse`/`ast.unparse` only.
- **No new `python_bindings`-style live-Python dependency.** Fix 2.3 in particular is specifically about *removing* reliance on a live-Python fallback for something that should never have needed one.
- **`static_python_bindings`/`_expand_python_static_bindings` (`aot_compile.py:188`) was diagnosed but deliberately not modified.** It has a real, independent bug — its transitive `co_names`/`__globals__` walk is unscoped and can still shadow a user's own name with anything transitively reachable through the compiler's own internal imports, for any name fix 2.3's `class_definitions` mechanism doesn't already claim first (e.g., a name that isn't a class at all, or a class the source doesn't define locally but the compiler's own internals happen to also use). This is real and worth fixing but is a separate, broader hardening task — narrowing `_expand_python_static_bindings`'s scope, or making it error/skip on a name the *source itself* binds, rather than silently deferring to it. **Not done this session. Flagged for follow-up.**

---

## 4. What's still broken — the current, real, next blocker

Running the real `build_program_bundle` end-to-end (not a standalone script — the actual function) on the `Counter` test source now fails at:

```
ValueError: compiled state feedback does not match the Python ABI;
missing inputs=['counter.value']; missing outputs=['counter.value']
```

This is a **different** failure from before fix 2.3 (which was `AttributeError` during discovery). This is now failing later, at WASM ABI assembly (`site_bundle.py`, the `missing_inputs`/`missing_outputs` check right after `emit_wasm_module`, ~line 2024-2039).

**Diagnosis in progress, not complete.** The suspected mechanism (confirmed by isolated inspection, not yet confirmed inside the real pipeline — this is exactly the distinction the user pushed back on mid-session, "why are you using custom python," so treat the following as a *lead*, not a confirmed root cause):

`project_public_numerical_program` (`aot_compile.py:360`), after selecting `outputs = {'counter.value': <node>}`, needs to find exactly one region program (from `compilation.region_programs`) or the top-level fallback (`compilation.compiled_shell_program.program`) whose live node set contains that output's node id. Under `precompile_only=True` (the flag `build_program_bundle` actually uses — confirmed, not assumed), an isolated check on a *plain numeric* function (`def double(x): return x * 2`) showed `region_programs` empty (`len == 0`) and the fallback's own `.steps`/`.outputs` also empty, called directly via `compile_ast_aot(..., precompile_only=True)`. Yet the real `build_program_bundle` **does** successfully publish working, fidelity-verified WASM for `double` (`wasm-fidelity.json`, `passed: true, case_count: 3`) — so either:
- (a) something between `compile_ast_aot` returning and `project_public_numerical_program` being called populates real steps that the isolated check didn't replicate, or
- (b) `double`'s success doesn't actually exercise this path meaningfully (it has no `state_feedback`, so the `missing_inputs`/`missing_outputs` check that's currently failing for `counter` is simply *never reached* for `double` — meaning `double`'s WASM might also have an empty/degenerate `program.outputs` internally, and its correctness is coming from somewhere else in the emission path that doesn't require named outputs to be populated the way `state_feedback` validation requires).

**This was where debugging was interrupted** (temporary print statements added to `site_bundle.py` to inspect `program.outputs`/`region_programs` inside the real function were reverted per the user's direction to stop using ad-hoc scripts and instrument the real path instead — the instrumented run was made but the output landed inside `compile_log` (internal `StringIO`, both stdout *and* stderr redirected by `build_program_bundle` itself for its own diagnostics), not the terminal, so **no data was actually captured yet**. The print was reverted without ever seeing its output.

**Next concrete step:** re-add equivalent instrumentation, but read it correctly this time — either via `progress_sink` (the real, intended channel for exactly this kind of visibility, per `build_program_bundle`'s own docstring: *"a caller that wants terminal visibility during the build passes a printing sink"*), or by inspecting `compile_log` after the exception. Do **not** go back to a standalone script that reimplements the call — use `progress_sink=print` (or equivalent) on the real `build_program_bundle` call, which is the mechanism that already exists for exactly this need and was sitting unused throughout this debugging stretch.

---

## 5. Context carried in from before this document (prior session, summarized)

These were already fixed and are stable going into this session; not re-verified today but no reason to suspect regression (none of today's changes touch this area):

- **`graph.python_package`**: `ProcessGraph.build_from_ast(..., resolve_unresolved_parents=True)` already has real import resolution (`_import_ast_bindings` → real `importlib.import_module`); it silently failed because `graph.python_package` was never set. Fixed by setting it before `build_from_ast` in `site_bundle.py`. This replaced an earlier, incorrect `python_bindings=<externally supplied globals>` parameter that was fully removed.
- **`_synthesize_state_defaults`** (`site_bundle.py`, placed just before `_content_version`): pure AST rewrite. For a parameter that is both (a) absent from `contract.feeds`/`contract.constant_map` entirely and (b) has a written type annotation, inserts `if x is None: x = T()` using the parameter's own annotation. This is the "detect not instantiated, instantiate inside" mechanism, and is what makes `counter: Counter` with no supplied value get a real, compiler-traced construction rather than a numeric-kernel-style probe guess. Verified not to affect ordinary numeric-kernel parameters (test suite baseline: 2 pre-existing failures, unaffected by this mechanism).
- **`_feed_value`**: returns literal `None` for a synthesized parameter (so discovery actually sees Python `None` and takes the `is None` branch), otherwise ordinary `_probe_value` numeric probing.

---

## 6. Deliberately deferred, unrelated to the above

`MachinePathForest` (and any live, non-JSON-serializable object generally) cannot be passed as an explicit `probes` value to `build_program_bundle`, because `_content_version` (`site_bundle.py:1092`) `json.dumps`'s `contract.feeds` directly for content-addressed build caching, and a live object isn't JSON-serializable. This is a **separate** problem from everything in this document — "detect not instantiated, construct inside" (Section 5) solves the case where state has a real zero-arg-derivable constructor; it does not solve passing an *already-constructed* complex object in from outside. Not touched this session. Relevant when resuming the `machine_fork_exploration.explore_forking_paths`/`MachinePathForest` driver script (`examples/build_machine_fork_exploration_page.py`, untracked, real, currently unrunnable until both this document's Section 4 blocker and this JSON-serialization problem are resolved).

---

## 7. Concrete next steps, in order

1. Re-run the full relevant test suite (`test_abstract_tensor_topological_reducer.py`, `test_machine_execution_aot_compile.py`, `test_site_bundle.py`, and ideally anything exercising class ingestion generally — `test_wasm_class_coordinator.py`, `test_wasm_class_modules.py`, `test_process_graph_shell.py` were found to reference the class-module machinery but not run this session) to confirm fix 2.3 (the `class_definitions`/`class_ref` change) hasn't regressed anything. This is the most structurally significant change of the three and the least test-verified so far.
2. Resume the Section 4 blocker using `progress_sink` on the real `build_program_bundle` call (not a standalone script) to see what `project_public_numerical_program` actually produces for `counter.value` inside the real pipeline.
3. Once `Counter` builds end-to-end, verify in a browser: load the published page, confirm `counter.value` actually increments across calls (not just that the WASM module compiles) — per this project's standing rule that UI/browser-observable changes need actual browser verification, not just a successful compile.
4. Separately, consider hardening `_expand_python_static_bindings` (`aot_compile.py:188`) per Section 3 — narrowing its transitive walk so it can never silently shadow a name the source itself binds, independent of whether `class_definitions` already covers the class case.
5. Resume `machine_fork_exploration`/`MachinePathForest` (Section 6) only after the above are solid — it depends on both this document's fixes and the separate JSON-serialization problem.

---

## 8. Files touched this session (turing repo)

- `src/common/tensors/topological_reducer.py` — fix 2.1 (identity binding for attribute writes) and fix 2.3b (`class_ref` on locally-defined-class calls).
- `src/compiler/site_bundle.py` — fix 2.2 (field-name substitution for synthesized state outputs). Temporary debug prints added and reverted during this session; file should be clean of debug output at commit time.
- `src/transmogrifier/graph/graph_express2.py` — fix 2.3a (`class_definitions` published at ingestion time).
- `examples/build_machine_fork_exploration_page.py` (untracked, new) — the real driver script for the paused `MachinePathForest` work (Section 6).

**Not touched by this session, but present as uncommitted changes in the turing working tree — flagged, not swept into any commit without separate confirmation:**
- `src/common/tensors/autoautograd/fluxspring/design intent.txt`
- `src/common/tensors/autoautograd/whiteboard_runtime.py`

These predate this session's work and are unrelated to everything above. Committing them together with this session's changes would misattribute unrelated work; they're being left uncommitted pending explicit confirmation of what they are.
