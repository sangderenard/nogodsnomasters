# Reference Operators and the Field-State ABI: Transition Document
**Date:** 2026-08-07
**Scope:** `turing/` compiler — `topological_reducer.py`, `graph_express2.py`, `node_special_cases.py`, `glsl_deployment_strategy.py`, `site_bundle.py`, plus an unrelated Fortran-toolchain fix (`fortran_toolchain.py`, `ssa_fortran_backend.py`, `fortran_c_shell.py`)
**Continues from:** `CLASS_STATE_ABI_TRANSITION_2026-08-07.md` (fixes 2.1–2.3, the `Counter`/`counter.value` state-feedback mission)

Written so a fresh session can pick this up with zero prior context. This session picked up exactly where the prior transition doc's Section 4 blocker was left, root-caused it fully, and landed one full architectural direction change plus several concrete fixes along the way.

---

## 1. Fixed and verified this session

### 1.1. Fortran `strndup` link failure (unrelated to everything else below)

**Symptom:** `test_program_bundle_compiles_fortran_and_records_output_fidelity` and `test_backend_targets_restricts_published_source_tabs` both failed with:
```
libgfortran.a(string.o): undefined reference to `strndup'
```

**Root cause, isolated experimentally (not guessed):** the installed mingw-w64 toolchain is `gcc 16.1.0` against CRT `mingw-w64-x86_64-crt-git 13.0.0.r57` — a version mismatch. A pure-compute Fortran module links fine with `-static`; anything touching libgfortran's string/formatted-I/O runtime (`write`, `print`, `trim`, even a plain unformatted `write(unit) x` with zero characters in the program) pulls in `libgfortran.a(string.o)`, whose `_gfortrani_fc_strdup` calls `strndup`, which this CRT snapshot doesn't define. Confirmed via direct `gfortran -shared ... -static` experiments outside the compiler, isolating exactly which construct triggers it.

**User's framing, correct:** this is not a data-model problem ("eliminate strings via token encoding" was considered and explicitly rejected) — any generated module that does *any* I/O pulls this in regardless of whether the program's own data touches strings at all. `-static` itself was confirmed necessary and deliberate (user develops at the hard/standalone case because it simultaneously proves the easy case) and was not weakened.

**Fix:** a minimal `strndup` shim, embedded as generated C source and passed to both link sites.
- `src/compiler/fortran_toolchain.py` — added `_CRT_SHIM_SOURCE` and `standalone_runtime_shim_sources(compiler, directory, standalone)`, which writes the shim to `<workdir>/turing_crt_shim.c` and returns its path (empty tuple when not on a standalone GNU-Fortran/Windows link, so it's a no-op everywhere else).
- `src/compiler/ssa_fortran_backend.py` — `compile_module`'s link command now includes the shim source.
- `src/compiler/fortran_c_shell.py` — same, in the native Fortran/C-shell executable link command.

**Verified:** both previously-failing tests pass (confirmed via direct pytest run, not inferred).

### 1.2. `_source_static_literal`'s `ast.Dict` branch mis-paired keys and values

**File:** `src/compiler/glsl_deployment_strategy.py`, `_source_static_literal`.

**Bug:** parents for a dict literal's keys and values arrive from ingestion *grouped* (`role='keys'` × N, then `role='values'` × N), not interleaved. The old code did `dict(zip(resolved[::2], resolved[1::2]))` on the flat parent list, which — for a grouped list — pairs key→key and value→value across the midpoint. Confirmed empirically by dumping real parent roles for `{"a": 1, "b": 2, "c": 3}`: `resolved = ['a','b','c',1,2,3]` → old code produced `{'a':'b', 'c':1, 2:3}`, garbage. This was the origin of a real crash seen earlier in the session (`TypeError: '<' not supported between instances of 'NoneType' and 'str'` inside a cache-key `sorted()` call, downstream of this corrupt dict).

**Fix:** pair parents by role (`keys`/`values`) instead of by position, and explicitly reject `{**unpack}` (a `None` key slot) as not source-static rather than silently mis-pairing around it.

**Verified:** confirmed via a standalone probe dumping real ingested parent roles before and after the fix; the corrupt-dict crash is gone.

### 1.3. `AbstractTensor.reshape` (and its whole aliased family) missing from every method table

**Files:** `src/transmogrifier/graph/graph_express2.py` (new `_attach_external_methods`, wired into `ProcessGraph.build_from_ast`'s `retain=` path).

**Bug:** `AbstractTensor.reshape = _reshape_methods.reshape` (and `view`, `flatten`, `transpose`, `permute`, `unsqueeze`, `squeeze`, `swapaxes`, `repeat`, `repeat_interleave` — `abstraction.py:3461` on) binds a real method whose `def` lives in a different module (`abstraction_methods/reshape.py`). `retain=`'s class ingestion (`_source_ast_definition`) only reads the class's own source text, so this whole aliased family was invisible to `_class_schema_from_ast`'s `methods` list — and therefore to `map_ir["graphs"]`, and therefore to `build_map_dependency_regions`'s `bindings` — even though `len(methods) > 100` still passed (the live Python method table has them; the AST-derived schema didn't).

**Fix:** `_attach_external_methods(retained_class, definition)` — after pulling in a retained class's own `ClassDef`, additionally parses the *defining module's* source, finds `ClassName.method = some_name` module-level assignments, resolves each RHS's own source definition via the same `_source_ast_definition` used elsewhere, renames it to the bound attribute name, and appends it into the retained class's body as an ordinary method. Only a plain name/dotted-attribute RHS is treated as a method alias; a literal or call result is left as an ordinary class attribute.

**Verified:** `test_aot_ingests_and_retains_the_abstract_tensor_class_object` passes (it was failing before this fix, specifically on `AbstractTensor.reshape` missing from `dependency_regions.bindings`; `tensor`/`sum` — defined directly in the class body — already passed).

---

## 2. The core architectural finding this session (not yet fully implemented)

### 2.1. Why `counter.value` never worked: the real root cause, corrected twice

The prior transition doc's Section 4 blocker (`missing inputs=['counter.value']; missing outputs=['counter.value']`) was diagnosed as a naming/ABI-shape problem. **That diagnosis was wrong.** The actual chain, confirmed with a `progress_sink`/direct-write probe against the real `build_program_bundle` (not a reimplemented standalone script — the transition doc's own prescribed next step):

```
bake_mode='whole_program'
region_programs=0
shell_control_program=ControlProgram   region_indices=()
function_outputs=('counter.value',)    <- correct, from fix 2.2
identity_table_keys=['counter','counter.value']   <- correct, from fix 2.1
state_feedback={'counter.value': 'counter.value'} <- correct
program_type=FusedProgram
program_outputs={}   program_feeds=set()   feed_origins=None   <- EMPTY
```

Every front-end piece (identity binding, output naming, state_feedback) was already correct. The compiled program itself was **empty** — nothing lowered `counter.value = counter.value + 1` into a kernel at all. `missing inputs/outputs` was accurate but late: the real question is why nothing was ever compiled.

**Root cause, traced to one line:** `evaluate_node`'s `SetAttr` case (`glsl_deployment_strategy.py`, ~line 9456, unmodified this session — found, not touched) does:
```python
setattr(receiver, (data.get("attributes") or {})["attribute"], value)
```
This is a live CPython `setattr` during discovery — it computes the right *value* and it's visible to the interpreter, but it never registers a graph step. Compare the `GetAttr`-equivalent read branch: `result = getattr(receiver, expression.attr)` — same story, a bare Python value with no persisted identity.

**The deeper question — "why does anything arrive here that isn't a callee/method":** the graph already has a real, general mechanism for a node to reference *another* node it depends on: `callee_ref`, `method_ref`, `class_ref` — all real node-id references a walker (`build_map_dependency_regions`, the region-dispatch fusion planner, etc.) can dereference and recurse into. `SetAttr`/`GetAttr` had **no equivalent**. Their only per-node data was `attributes["attribute"] = "value"` — a bare *string label*, not a reference to anything. There was structurally nothing for any walker to recurse into. That is the actual, general root cause, not a bug in any one function — every symptom this session traced (empty region_programs, the missing-inputs/outputs error, `_source_static_literal`'s missing `Attribute` case) is a different downstream consequence of the same gap.

### 2.2. The fix landed this session: `field_ref`

Symmetric with `callee_ref`/`method_ref`, on both the producer and consumer side:

**Producer side (ingestion), `src/common/tensors/topological_reducer.py`:**
- `bind_target`'s `ast.Attribute` (SetAttr) branch: when the receiver is a plain name (`counter.value = ...`), mints or reuses (via `input_value`, the same memoized-by-name mechanism ordinary parameters already use) a real `Input` node named `f"{receiver}.{attr}"`, and stamps `attributes["field_ref"]` on the `SetAttr` node pointing at it.
- `resolve_expression`'s `ast.Attribute` (read/Load) branch: same mint-or-reuse (shared identity, since `input_value` memoizes by name), stamped as `field_ref` on the `GetAttr` node.
- Because `input_value` also feeds `identity_bindings`, `identity_table["counter.value"]` now naturally contains `(InputNodeId, SetAttrNodeId, ...)` in the order encountered — the `Input` node satisfies the *input* side genuinely, and the existing `identity_table[name][-1]` convention (fix 2.2) still correctly picks the last write for the *output* side. Nothing about fix 2.1/2.2 needed to change.

**`node_special_cases.py`:** `ast.Attribute` in `Load` context is now normalized to canonical type `"GetAttr"` at ingestion (a new case in `interpret_special_case`), the same way a tensor-bearing `Call` is flagged without collapsing. `Store`-context `Attribute` (assignment targets) is explicitly excluded — that's `bind_target`'s job, already correct, not to be fought over the same AST node identity.

**Consumer side, `src/compiler/glsl_deployment_strategy.py`:**
- The generic `Input`-node resolution loop inside `_coordinate_scheduled_capture_impl` (~line 6159) already had `has_deferred_local_definition` — "a Python local, not an invocation requirement" — but gated to `binding_kind == "external"` only. Extended to also cover `binding_kind == "field"`: a field this function itself assigns (via its own later `SetAttr`, now linked by the same `identity_table` entry `field_ref` participates in) is not something its caller must supply either.
- A second, separate copy of the same "missing ProcessGraph input" raise inside `evaluate_node` itself (~line 7545, reached when an `Input` node is resolved lazily mid-execution rather than pre-populated) had no equivalent fallback at all. Added one: for `binding_kind == "field"`, look for a later, non-`Input`-typed identity under the same name in `graph.G.graph["identity_table"]`, and if found, evaluate *that* instead of raising.

**NOT changed:** `build_map_dependency_regions` (`shell_reference_tables.py`) was *not* given a `field_ref` case, despite earlier framing in this session suggesting it should be. That function's `pending` worklist holds `function_table` references (cross-function/method call targets); a field's `Input` node id is not a function reference, and forcing it into that exact list would be actively wrong (it would call `function_table.entry(some_input_node_id)`, which isn't a function). This was caught and corrected mid-session rather than implemented incorrectly. `field_ref`'s actual consumer is the *contiguity/dependency-selection* mechanism described in Section 3, not this cross-function walker.

**Verified, precisely:** re-running the real `build_program_bundle` Counter probe after each piece landed showed the error change in sequence exactly as expected — `KeyError: missing ProcessGraph input 'counter.value' in step` (first fix) → cleared; `KeyError: missing ProcessGraph input 'self.value' in __init__` (second fix, a *different* function's copy of the same class of bug) → cleared; final state: discovery completes end-to-end with **zero errors**, and the pipeline reaches the (deliberately added, see Section 2.3) `RuntimeError` refusal cleanly, with `region_programs` still `0`. This is honest, expected non-progress on the *emission* side and real, confirmed progress on the *discovery/walkability* side — two genuinely distinct problems, not one.

### 2.3. A deliberate refusal, added and then recognized as pointed at the wrong permanent fix

`site_bundle.py`, right before `project_public_numerical_program(aot)`: when `synthesized` (class-shaped state parameters) is non-empty and no navigable control regions exist (`region_programs` empty), **raise** rather than silently flatten and publish a page whose state is unnavigable. Mirrors the existing whole-program-bake refusal immediately above it in the same function.

```
RuntimeError: WebAssembly emission refused a final fused reduction for
class-shaped state ['counter']: flattening would erase the object structure
['counter.value'] is named after, and no navigable control regions were
planned to emit instead (region_programs=0, region_indices=())
```

This is **correct as a safety gate** (never silently publish a broken/unnavigable page) but the session's later architectural discussion concluded it should never actually *fire* for a case like `Counter` once Section 3 is implemented — a tensor-free field-update body should route to a structural lowering path instead of ever reaching the flatten decision at all. The refusal stays as the correct fallback for a genuinely pathological case; it is not the intended steady-state outcome for ordinary class-shaped state.

---

## 3. The next architectural piece — designed this session, not yet implemented

### 3.1. The governing principle, stated by the user and worth preserving verbatim in spirit

The flatten-into-one-`FusedProgram` reduction is a **tensor-kernel optimization** — it exists to make elementwise operations parallelizable across shape. It was never a general "how do we compile" step. Applying it to field/scalar bookkeeping (`counter.value = counter.value + 1`) was a category error: there's no shape to vectorize across, so reduction has nothing to optimize and everything to lose (all per-object structure). Standing rule recorded in memory (`no-final-fused-reduction-multi-function.md`): a final fused reduction is only admissible for a single function/tensor kernel.

The tape (`glsl_deployment_strategy.py`'s discovery-time capture, keyed by `tensor_identity(result)`) was always "a convenience" for cases where runtime behavior is genuinely ambiguous from source alone (which backend, dtype, shape a dynamically-dispatched tensor op resolves to). Plain scalar/reference arithmetic has no such ambiguity — `topological_reducer.py` already builds a fully correct, statically-derived graph for it. Routing it through the tape (or promoting every plain value to `AbstractTensor` to make it "visible") would import overhead this class of computation never needed. The user was explicit that both of those options cross a line the project doesn't want to cross.

### 3.2. The qualification test — built and verified this session

**File:** `src/transmogrifier/graph/node_special_cases.py`, new function `graph_has_tensor_operation(graph, node_ids=None)`.

Walks the given node set (default: the whole graph) checking `attributes.get("tensor") is not None` — the ingestion-time stamp `build_graph` already writes from `tensor_operation_name` (an AST-derived fact: does this `Call`'s callee name a canonical `AbstractTensor` operation). No execution, no tape — purely static.

**Verified** against two known cases via a standalone probe (ingestion + reduction only, no WASM):
```
Counter/step:   graph_has_tensor_operation = False
tensor kernel (value.reshape((-1,)).sum()): graph_has_tensor_operation = True
```
Both classify correctly.

### 3.3. Where the filter slips in — found, not yet modified

`src/compiler/process_graph_fusion.py`:

- **`dispatch_region_to_fused_program(graph, region)`** (~line 1154): the actual "make a fused program from this section of operators" entry point. Currently unconditionally builds `Meta(shape=..., dtype=..., device=...)` for every node in the region — the tensor assumption. This function executes on whatever `region.node_ids` it's handed; it does not decide the span.
- **`plan_process_graph_dispatches(graph, profile)`** (~line 1049): the actual span-decision point. `fusible = {node_id for node_id in graph.G if _operation(graph, node_id) in profile.fusible_ops}`, then `nx.weakly_connected_components(graph.G.subgraph(fusible))` — **this is already exactly the contiguity mechanism the user described**: components naturally split wherever the fusible-node set isn't adjacent. No new grouping machinery is needed, only a second, parallel fusible-set feeding the same mechanism.
- **`profile.fusible_ops`**, concretely, is `frozenset(ELEMENTWISE_UNARY | ELEMENTWISE_BINARY)` (`program_order.py:73`, `torch_process_graph.py:164`) — pure arithmetic op names. `SetAttr`/`GetAttr` are not and never were members, so a field-touching chain can never join *any* region under the current selection, regardless of tensor content. This is confirmed as the literal, precise mechanism behind `region_programs=0`, not a downstream symptom of something else.
- Even if membership were fixed, the region is separately rejected by a profitability score (`max(0, len(nodes)-1)*launch_cost + internal_edges*intermediate_traffic_cost - binding_count*binding_cost`, reject if `<= 0`) — a GPU-launch/memory-locality tradeoff heuristic. **The user explicitly decided this class of analysis is out of scope for the computational reducer as applied to reference operators**: there is no "worth it" question for correctness-preserving state threading; a reference-operator chain is either present (correct) or not.

### 3.4. The settled design for the next session to implement

Naming, per the user: **"reference operators"** — `SetAttr`, `GetAttr` (plus `Input`/`Constant` as the leaves a reference resolves to/from), as the conceptual sibling of the existing elementwise set, because what unifies them is establishing/dereferencing an identity, not performing arithmetic.

Concretely, not yet written:
1. A second fusible-set alongside `ELEMENTWISE_UNARY | ELEMENTWISE_BINARY` — a `REFERENCE_OPERATORS` set (`SetAttr`, `GetAttr`, and whatever leaf types are needed) — feeding the *same* `weakly_connected_components` contiguity selection in `plan_process_graph_dispatches`, but **with no profitability-score gate**: every connected component in this induced subgraph becomes a region unconditionally.
2. `dispatch_region_to_fused_program`, per selected region, branches on `graph_has_tensor_operation(graph, region.node_ids)`:
   - Tensor-bearing → existing code path, completely unchanged (Meta-assuming, elementwise/reduction op handling).
   - Tensor-free → a new, direct structural emission: one `OpStep` per graph node, in the graph's own topological order, no algebraic fusion, no `Meta` shape/dtype/device assumption. "Fully realized" falls out for free — a pure structural transcription can't simplify anything because it never runs the code, only walks the already-correct graph `topological_reducer.py` built.
3. Once this exists, `Counter`'s `step` should produce a real, non-empty `region_programs` entry, and the Section 2.3 refusal should stop firing for this class of case — it routes to structural lowering instead of ever reaching the flatten decision.

**Not yet done:** the actual code for (1) and (2). This is genuinely new code, not a patch — scoped and designed, but the next session's first task, not carried over as "in progress."

---

## 4. Explicit boundaries respected this session

- Never touched `project_public_numerical_program`, `emit_wasm_module`, `dispatch_region_to_fused_program`'s existing tensor path, or `_CompiledStructuralObject`'s core shape — consistent with the standing rule from the prior transition doc (don't patch machinery that's already right).
- No `exec()`/`eval()`. `_attach_external_methods` uses `inspect.getsource`/`ast.parse` only, same discipline as `_source_ast_definition` it reuses.
- No new `python_bindings`-style live-Python dependency introduced.
- `build_map_dependency_regions` was investigated as a candidate for `field_ref` consumption and deliberately **not** modified — see Section 2.2's "NOT changed" note; forcing it in would have been wrong, not just unnecessary.
- The `_source_static_literal` `Attribute`-case gap (Section 1.2 is a different bug in the same function; this is a *second*, still-open gap: no case for `Attribute`/`GetAttr` at all, falls through to `raise ValueError`) was identified but **not fixed** — it's used only by `_callsite_specialized_shell_type`'s specialization cache-key builder, an optimization shortcut unrelated to the `region_programs`/state-feedback mission. Flagged for separate follow-up, not bundled in.

---

## 5. Files touched this session (turing repo)

- `src/common/tensors/topological_reducer.py` — `field_ref` stamping on `SetAttr` (`bind_target`) and `GetAttr` (`resolve_expression`).
- `src/compiler/glsl_deployment_strategy.py` — `_source_static_literal`'s dict-pairing fix; `_coordinate_scheduled_capture_impl`'s deferred-local-definition check extended to `binding_kind == "field"`; a second, separate `evaluate_node`-internal fallback for the same `binding_kind == "field"` case; the `RuntimeError` refusal in spirit (actually landed in `site_bundle.py`, see below).
- `src/compiler/site_bundle.py` — the class-shaped-state flattening refusal (Section 2.3).
- `src/transmogrifier/graph/graph_express2.py` — `_attach_external_methods` (Section 1.3).
- `src/transmogrifier/graph/node_special_cases.py` — `ast.Attribute`→`GetAttr` normalization case; `graph_has_tensor_operation` (Section 3.2).
- `src/compiler/fortran_toolchain.py`, `src/compiler/ssa_fortran_backend.py`, `src/compiler/fortran_c_shell.py` — the `strndup` shim (Section 1.1, unrelated to everything else).

**Not touched by this session, present as uncommitted changes, predate this session (carried forward from the prior transition doc, still unexplained, left uncommitted pending explicit confirmation):**
- `src/common/tensors/autoautograd/fluxspring/design intent.txt`
- `src/common/tensors/autoautograd/whiteboard_runtime.py`

---

## 6. Concrete next steps, in order (superseded by Section 7 — read that first)

Section 3.4's plan below (implementing `REFERENCE_OPERATORS` in `process_graph_fusion.py`) was **attempted and abandoned**: that module's `dispatch_region_to_fused_program`/`plan_process_graph_dispatches` turned out not to be reachable from `build_program_bundle`'s actual AOT path at all (it belongs to a separate JIT/order-based route — `program_order.py`/`torch_process_graph.py`). The real entrance was found and used instead; see Section 7.

1. ~~Implement Section 3.4...~~ — wrong file; superseded.
2. ~~Re-run the real `Counter` probe...~~ — done, repeatedly; see Section 7 for the actual sequence of fixes and the final, precise blocker found.
3. Not yet reached — no successful build yet to verify in a browser.
4. Still open, still deliberately deferred, still not blocking.
5. Still correctly gated on the above landing first.

---

## 7. What actually happened after Section 6 was written (same session, direct continuation)

The real entrance, found by tracing where `region_programs` actually comes from for the AOT path (not the JIT path Section 3.4 targeted): `_observe_process_graph_node` (`glsl_deployment_strategy.py`, module-level, called by GraphDeepCompiler-generated numeric-kernel source at the exact point each operation executes) and the consolidation point `target.captured_region_programs = dict(zip(target.compiled_region_indices, target.compiled_tapes))` (~line 12596), which is what `aot.region_programs` is ultimately built from.

### 7.1. The accounting mechanism, built and landed

- `ProcessGraphGLSLDeployment.__init__` gained `self.reference_operator_sequence = []` — a plain, ordered list. Not a strategy, not a region planner: append-only, in exact execution order, because these are sequential memory operations with real causality that reordering or fusing would break (the user's framing, verbatim, driving this design).
- `evaluate_node`'s `SetAttr` and `GetAttr` branches append their own node id to it as they execute.
- `_observe_process_graph_node` — the tape/tensor-primitive-correlation observer — gained an unconditional branch: **any** non-`AbstractTensor` result it observes gets appended to `shell.reference_operator_sequence` too, impartially, not scoped to field access specifically. This is deliberate, per explicit user correction: "make your system impartially document the operations in order for things that aren't fully tensor" — no special-casing, no scoping decision. The tape's own tensor-correlation logic (everything below this branch) is untouched and still exists for genuine tensor ambiguity.
- At the `captured_region_programs` consolidation point, `reference_operator_sequence` (deduplicated, order preserved) gets packaged into one `FusedProgram`/`CapturedFusedProgram` — one `OpStep` per recorded node, transcribed directly from the already-correct `ProcessGraph` structure (`op`/`type`, `parents`, `attributes`, plus `field_ref` pulled in explicitly since it's stored as an attribute, not a `parents` edge). No fusion, no algebraic simplification, no launch-cost scoring — a direct structural transcription.
- `OpStep`/`FusedProgram`/`CapturedFusedProgram` had to be added to the generated `ProcessGraphGLSLDeployment` class's `exec()` namespace dict explicitly — that generated class runs in its own namespace, separate from this module's own top-level imports.

### 7.2. Two real, non-obvious bugs found and fixed via direct instrumentation (not inference)

**`field_ref` going stale across node-id relabeling.** `field_ref` (Section 3.4/earlier work — the `Input`-node reference `SetAttr`/`GetAttr` carry) is a bare node-id value stored inside a node's `attributes` dict, not a `parents`/edge relationship. `topological_reducer.py`'s reduction pass relabels every node id to a small monotonic sequence (`mapping = {node_id: value_id for value_id, node_id in enumerate(ordered)}`, `nx.relabel_nodes`) — `identity_table` and several other attribute-embedded references (`loop_carried_bindings`, etc.) are manually remapped through `mapping` right after, in the same loop; `field_ref` was not. Fixed by adding it to that same per-node remap loop (~`topological_reducer.py:2541`), dropping it if its target didn't survive reduction rather than leaving a dangling id. Verified: the field's `Input` node went from resolving to nothing (`type=None, attributes=None`) to resolving correctly (`type='Input', binding_name='counter.value', binding_kind='field'`).

**The identity-aliasing check intercepting plain arithmetic.** `_observe_process_graph_node`'s existing aliasing logic (`if result is parent_value: ... return result`, treating the node as an SSA alias rather than a new computation) ran *before* the new non-tensor recording branch. CPython interns small integers, so `0 + 1` producing `1` satisfies `result is parent_value` against the literal `1` operand by cache coincidence — a pure implementation detail, meaningful only for tensors (where identity means shared storage), not for plain values. This silently made the `Add` node between a field's `GetAttr` and `SetAttr` invisible — it looked like an alias, not a computation, so it was dropped, and its result surfaced as a phantom external feed instead of an internal step. Fixed by moving the non-tensor check ahead of the aliasing logic. Found by direct instrumentation after three other candidate call sites were checked and ruled out (`evaluate_node`'s "structural binary operator" `isinstance(expression, (ast.BinOp, ast.AugAssign))` branch, `evaluate_reduced_control_expression`, the `AugAssign` scalar-source-update shortcut) — none of them were where `counter.value + 1` actually executes; it runs through GraphDeepCompiler-generated source calling `_observe_process_graph_node` directly, not through `evaluate_node`'s interpreted branches at all.

**A known-unfixed defensive patch, not root-caused:** `field_ref`'s remap fix can still, in principle, produce an id belonging to a *different* shell's numbering than the one currently consolidating `reference_operator_sequence` — a `KeyError: 18` was observed once (a stale/foreign node id, not in `target.process_graph.G`). Fixed defensively (`if node_id in target.process_graph.G`, filtering rather than crashing) but the actual source of the mismatch was not traced. Revisit if state ever looks incomplete/wrong rather than absent.

### 7.3. The corrected ABI gate

The Section 2.3 refusal (`site_bundle.py`) originally required `aot.shell_control_program.region_indices` to be non-empty. Traced precisely: that field names *planned GPU-shader-dispatch regions* specifically — a tensor-shaped concept, populated by shader-region planning (`function(**bound_inputs)`, `deployment_outputs`/`deployment_store_nodes` — all genuinely GPU-dispatch machinery) *before* discovery even runs. Class-shaped state with no tensor content is never planned as one of those regions and structurally never will be; requiring it was forcing tensor-shaped machinery onto content that was never tensor-shaped — the exact category error the refusal exists to prevent elsewhere. Relaxed to `navigable_regions = bool(aot.region_programs)` — the real signal, holding whatever was actually captured regardless of shape.

### 7.4. Verified end state, and the final, precise remaining blocker

Confirmed via direct instrumentation of the actual object reaching `emit_wasm_module` (not inference): the full causal chain is captured correctly.

```
program.feeds   = {3, 13, 15}
program.outputs = {'counter.value': 17}
program.steps   = [('GetAttr', [13, 3], 14), ('Add', [14, 15], 16), ('setattr', [13, 16, 3], 17)]
program.extras  = {'capture_feed_origins': {3: {'binding_name': 'counter.value'}, 13: {'binding_name': 'counter'}}}
```

Node 3 is the field's `Input` node (correctly named `counter.value`), node 13 is `counter` itself (a `Phi` node from the `if counter is None:` branch), node 14/16/17 are the read/arithmetic/write steps. `project_public_numerical_program` selects this exact program and correctly rebuilds `capture_feed_origins` onto its returned copy (`aot_compile.py:414-420`) — verified intact on the object that reaches `emit_wasm_module`.

**The build still fails**, with the same-shaped error as when this document was first written: `missing inputs=['counter.value']`. The cause is now exact and final, not a mystery: `fused_program_wasm_backend.py`'s `feed_names()` does `if not candidate.isidentifier(): candidate = f"feed{index}"`. `"counter.value".isidentifier()` is `False` — a dot is not valid in a Python identifier — so the correctly-captured, correctly-named feed is silently renamed to `"feed0"` at the exact last step before the WASM parameter list is built, and the ABI check (which looks for the literal string `"counter.value"`) correctly reports it missing.

**Do not loosen `isidentifier()` to fix this.** That was this session's own conclusion, reached and then re-confirmed by hitting the actual mechanism: `emit_wasm_module`'s `run(count, feed0, ..., out0, ...)` convention is the single-function flat ABI — the same one Section 3's whole architectural discussion concluded field-shaped state must never be forced through, because it structurally cannot represent per-object identity. Patching the name check would "fix" this one symptom by re-entrenching exactly the mistake the reference-operator work exists to route around.

**The actual remaining task:** this program needs to reach `wasm_class_coordinator.py`'s field-slot/byte-address ABI instead — "one memory, an ordered inventory of callable methods, and field slots containing byte addresses in that memory," already built, already tested (the segmented Mandelbrot deployment is its first client), explicitly designed to be "the receiving side of future OOP lowering." That connection — routing a tensor-free, class-shaped `FusedProgram` like the one now correctly captured here into the coordinator's inventory/field-slot construction instead of into `emit_wasm_module` — does not exist yet. It is a real, scoped integration task, not a design question; the hard parts (capturing the right operations, in the right order, with the right identities) are done and verified.

### 7.5. Explicit state of the repository at end of session

Three commits landed this session beyond what Sections 1-6 describe, all on `codex/recursive-reduction-bridge`:
1. `field_ref` stale-across-relabeling fix + the corrected `navigable_regions` ABI gate (Section 7.2/7.3).
2. The `reference_operator_sequence` accounting mechanism itself (Section 7.1) — this was actually landed *before* commit 1 chronologically but is described after it here since 7.1 is the mechanism and 7.2/7.3 are bugs found while testing it; check `git log` on this branch for exact order if it matters.
3. The aliasing-check reorder fix + defensive `field_ref` foreign-id filtering (Section 7.2, second bug).

**Not run this session:** the broader test suite, against any of commits in Section 7. Only the single `Counter`/`build_program_bundle` probe was used throughout (repeatedly, as the error signature changed). This is a real, stated risk, not an oversight — the observer hook in particular (`_observe_process_graph_node`'s new branch) fires for *any* non-tensor value anywhere in the compiler, not just `Counter`'s fields, and could affect other, unrelated compiled programs' region capture in ways not yet checked. **Run the full relevant suite before trusting these commits are safe**, the same way Section 1's fixes were verified against it.
