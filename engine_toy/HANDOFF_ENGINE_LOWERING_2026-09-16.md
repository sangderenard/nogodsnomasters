# Handoff: compiling the engine, and what is actually in the way

## What this work is

`engine_toy/time_trials` is a small driving demo built entirely on the
repository's own laws, and its purpose is diagnostic rather than
recreational. Two craft run on turing's `abstract_ui_vehicle_step` and
`abstract_ui_wheel_contact`, each with a real `EngineCycleSim` in its
engine slot, and the point of it is to make time dilation across
subsystems something you can observe and attribute rather than infer.
`time_trials/profile_frame.py` is the instrument: it reports, per
subsystem, the wall seconds spent against the world seconds advanced,
and separately the largest step each subsystem's own mathematics
tolerates.

Keep those two columns distinct. Wall time is a statement about the host
and compiling changes it. The step floor is a statement about the world
and only the physics changes it. A measure aimed at the wrong column
accomplishes nothing, and most of the confusion in this area comes from
treating a floor as though it were waste.

## Where the runtime stands

Both body laws execute as native kernels. The craft are no longer
independent objects that each call a law; they are rows in one set of
columns, with the vehicle law at a batch of one row per craft and the
contact law at one row per wheel per craft. A tick is therefore two
kernel calls regardless of how many craft exist. The craft advance
together and take the tightest stability limit in the fleet, because
they share a world and therefore share its floor.

Measured over two hundred frames at a one millisecond world step, two
craft:

    coupe.engine       4.17 ms/frame   40.6%   tau 0.240   floor 0.0624 ms
    roadster.engine    3.99 ms/frame   38.9%   tau 0.250   floor 0.0820 ms
    all bodies         1.74 ms/frame   16.9%   tau 0.575
      vehicle body     0.886 ms        295 us/call   (fallback path)
      wheel contact    0.016 ms        7.8 us/call   (batched kernel)

The contact law is batched and costs 7.8 microseconds a call. The
vehicle body is not, and costs 295 microseconds against a kernel that
runs in roughly 23, because it falls back to a row-at-a-time path that
marshals 341 inputs and 146 outputs per row. Closing that gap is
compiler work described below, not physics work.

The engines account for about eighty percent of a frame, and this is not
waste. The drivetrain junction takes roughly 139 substeps per step
because its own stability demands them: suppressing the escalation that
enforces this drops the cost to 2.42 milliseconds but produces 617 ring
events in 200 steps against 4 with it enabled. Compiling the engine
would divide the cost of one substep and would not reduce their number.

Startup is about eleven seconds when caches are warm. Be aware that the
symbolic cache does hit, and that a hit is itself expensive: loading the
cached compilation costs roughly twenty seconds for the contact law and
ninety-five for the body, all of it deserialisation. `native_law.py`
therefore caches one layer further down, storing the emitted kernel
keyed on the producer's source digest and the lowering pipeline digest,
so a warm start never loads the compilation at all. Editing any turing
file that feeds those digests invalidates both caches; a single cosmetic
change measured at 10.5 seconds of startup becoming 585.

## What prevents the engine from lowering

Lowering `EngineCycleSim.step` reaches deployment planning and then
refuses, naming three calls inside one loop: `self.hole_emitters.step`,
`self.bursts.step`, and `self.ordnance.step`. The blocker reported for
each is `opaque-state-effect`.

That phrase is misleading and has been amended in the compiler, but
understand the substance. `opaque` is the default classification, not a
detection. The classifier in `src/common/tensors/topological_reducer.py`
recognises a loop-body mutation only when the state's aggregate kind is a
known container and the operator appears in a fixed vocabulary of seven
names, now held as data in `src/compiler/loop_ir.py` as
`SEQUENCE_MUTATION_OPERATORS`, `MAPPING_MUTATION_OPERATORS` and their
union. Every other mutation falls through to the default. A method named
`step` was never going to be recognised, and neither would any domain
operator a program defines for itself.

The loop is refused on the mode alone, at
`src/compiler/loop_composer.py` around line 4732. This matters, because
an opaque effect still carries its state output. The compiler has
confirmed this directly: each refused effect now reports
`has_state_output: True`, together with `receiver_class: None` and
`method_ref: None`. Nothing is missing that needs computing. What is
missing is a model of the transition, and the specific fact absent is
the receiver's class identity at the point where effects are classified.

Counted across that whole lowering there are nineteen distinct state
effects: ten sequence mutations, two mapping mutations, and seven in the
default. The seven come from `vol.add_exhaust`, `at.lose`, `turb.step`,
`self.hole_emitters.step`, `self.bursts.step`, `self.ordnance.step`, and
`self.ordnance.log.append`. Three of those receivers are locals bound
from instance state, for instance `turb = self._turbine`, which means any
table keyed on parameter name cannot reach them; it must key on resolved
class identity.

There is a related asymmetry worth knowing before designing anything. A
parameter declared through `program_abi.values` materialises on the
function lowering path and is consumed at `fortran_c_shell.py` around
line 15136. A parameter declared through `program_abi.records` is
attached to the graph around line 13753 and is read by nothing on that
path; the code that turns record fields into ABI inputs exists only
inside `_class_surface_ssa_program`. A record therefore resolves, lifts
the refusal, materialises nothing, and yields an empty body reported with
zero shortfalls. A complaint now reports that case rather than leaving it
silent.

## The other blocker, which is separate -- now closed

The vehicle body would not lower through the batched route because
emission refused a call that selected 144 aggregate positions while the
callee produced 143. This is fixed in turing at `31329aa9`, and the
mechanism was traced on a five-line source in seconds rather than argued
from the body:

    def tick(p0, p1, p2, p3):
        z = 0
        t0 = p0 * p1
        t1 = t0 + z
        return t0, t1, z

The planner is correct: a canonical literal is control-owned, so every
region's rematerialised copy stays private and the region publishes the
computed outputs only. The control lowerer's `finish` in
`src/compiler/precompile_to_ssa.py` then found no value for the returned
literal -- no region publishes it, nothing had materialised it -- and
dropped it from the Ret silently. The call-frame projection pass in
`fortran_c_shell.py` around line 16400 later saw the authored output
still missing, found the callee's private `Const` by id, and appended
that id to the region call's declared `output_ids` after the callee's
Ret was already fixed. The planner sorts its outputs and the pass
appends, which is exactly why the body's declared list ascended through
1933..4332 and ended on 349, the literal `t109 = 0`'s own id. Only two
sites in the tree write `output_ids` on a region call; the other, in
`ssa_call_input_adapters.py`, renames an id already present and cannot
lengthen the list.

The earlier reproductions emitted clean because a literal that is only
returned takes a different path (it was already a provisional formal and
the late literal recovery turned it into a `Const`). The literal has to be
consumed inside a region as well. Measured while pinning this: literal
occurrences do not pool -- `z = 0` and an inline `0` are two canonical
ids, and a region owns only the copy it uses. So the body's `t109` must be
consumed by name inside a region of its stage; that is the only way the
projection pass can find id 349 in a callee.

`finish` now resolves an authored output whose identity is a canonical
literal through `external_value`, the provisional formal that
`_materialize_control_constants` turns into the function's own `Const` --
the rule a folded `flag = True; break` already followed on its break
edge. The region call then declares exactly what the callee returns.
Because the entry now exposes that output as a rank-0 buffer,
`native_law_kernels._lower_law` serves any literal-bound output as a
constant column whether or not `named_outputs` lists it; a per-row reader
would otherwise index past a one-cell buffer. Pinned in
`tests/test_control_returns_owned_literal.py` for all three shapes.

What is not yet verified is the body lowering itself. That needs the
batched route run on the real law, and the compiler edit has invalidated
the kernel cache keyed on the lowering pipeline digest, so the next start
of `time_trials` rebuilds it. Expect the shortfall to be gone; if a
different shortfall appears, it is a different defect and this document
should say so rather than absorb it.

## The plan

The aggregate mismatch is closed above; the bisection took the axis
"literal consumed inside the region", not output count or region count.

The class identity gap is half closed; see the continuation section at
the end. The half that remains is stated there as a measured fact with a
one-second reproduction, and it is not the field read.

Use `compile_probe.CompileProbe` for every compile rather than writing a
new instrument each time. It records phases with timings, every state
effect with its assigned mode, both ABI answers side by side, cache loads
with hit or miss and seconds, warnings, and the complete refusal payload.
It deduplicates effects, which matters: each effect is constructed twice,
once before its output identifier is assigned and once after, and
counting them naively doubles the inventory and invents conclusions.

## Constraints

Do not alter wall-time negotiation. Do not change the vehicle law's axis
conventions; the third component of the wheel gyroscopic reaction is
deliberately zero because the tuple is consumed as roll and yaw, and this
area has previously been an energy injector. Do not run the vehicle build
or a full lowering without being asked. Prefer targeted tests and
seconds-long reproductions over the full suite. New compiler-created
identifiers come from `src/compiler/monotonic_ids.py` through
`GLOBAL_MONOTONIC_IDS.mint()`; do not reintroduce local counters or
maximum-scan allocation. Never lower with `extraction_contract=None`,
which disables the machine decompilation gate; `compile_contract.contract()`
now defaults to refusing a Python-resident remainder, and its permissive
form exists to enumerate boundaries rather than to declare success.

Both repositories currently have other agents committing. Commit promptly
rather than leaving edits in the working tree, and expect turing edits to
invalidate the caches described above.

## Continuation, 2026-09-16 later in the day

Everything below was measured on reproductions that run in seconds, and
the two compiler changes are committed in turing with pinning tests.

### What was established about the class identity gap

The classifier's behaviour when the receiver class IS known was measured
first, on the same loop shape with the mutating method on `self`
(`self.bump(dt)` inside `for i in range(4)`): the call resolves a
`method_ref`, becomes a source-linked call, the classifier's own rule in
`topological_reducer.py` (around line 4280: "a source-linked method is
already an ordinary SSA call ... the callee's own GetAttr/SetAttr and
calls carry its effects") skips it, no state effect is recorded, and the
loop lowers with `bump` linked as its own function. So a resolved receiver
class is sufficient; nothing else is missing for this shape.

For a field that holds an object, the read `self.field` carried no class.
The reducer's field-read resolution (around line 2145) uses the enclosing
class only to look up the field's aggregate kind, and no table mapped a
field to the class it holds. Turing `d2b33c82` adds that table
(`(owner, field) -> class`, from `self.f = Cls()` in the owner's methods
and from `f: Cls` annotations, published on the class table as
`field_classes`), stamps `result_class_ref` on the field read, and lets
the deployment side's `receiver_class` walk read `result_class_ref`. It
is deliberately not `class_ref`: a dozen readers in
`glsl_deployment_strategy.py` treat that attribute's presence as a
construction. A field assigned two different classes is contested and
stays unresolved. Pinned in `tests/test_field_object_receiver_class.py`
for the direct read, the engine's `turb = self._turbine` alias spelling,
and the contested negative; the ten existing reducer and loop-composer
tests touching class references and opaque effects still pass.

### What the engine probe said afterwards, and what it costs

`EngineCycleSim.step` under the engine contract through
`compile_probe.CompileProbe`: 1829.7 seconds, still refused, the same
three effects, each with `receiver_class: None`. The probe's phase list
accounts for about fourteen seconds of that; the rest falls after the
last progress message, inside deployment preparation before the refusal
is raised. This makes the engine probe unusable as an iteration tool.
Do not iterate on it. The reproduction below answers the same question in
under a second.

### The remaining gap, stated as a fact

`HoleEmitterField`, `BurstField` and `OrdnanceField` are imported into
`engine_cycle_sim.py` from other modules. Measured with a two-module
source (`from field_mod import Field` and `self.field = Field()`): after
`reduce_abstract_tensor_topology`, the graph's class table has exactly one
key, `Sim`. `Field` is not in `class_definitions`, so `field_classes` is
empty and the field read carries nothing. The reducer fills
`class_definitions` from `ClassDef` nodes in the graph (around line
5106), and an imported class's `ClassDef` is never one of them.

The same source with the instance constructed locally in `step`
(`f = Field()` then `f.step(dt)` in the loop) refuses identically. So this
is not about field reads at all for imported classes: a method call on an
instance of an imported class has no `method_ref` even when the
construction is on the previous line. The static-reference path in the
reducer (`static_reference_node`, around line 1545) marks a constructor
call with `class_ref` only when the class is already in the class table,
which is the same circularity from the other side.

Where to look next, in order:

1. How the pursuit ingests an imported class. The reducer's comment at
   line 1565 says "imported dataclasses enter the function table as
   structural class references, but their defining ClassDef belongs to a
   different source unit", and `_python_source_identity` tagging at line
   5062 exists for classes from other modules, so there is a route by
   which an external class's methods reach the function table with a
   `method_owner`. The deployment side's `specialized_method_reference`
   matches `entry.graph.G.graph["method_owner"]` against a receiver's
   Python type, but only for `specializations`, which the engine contract
   does not supply for these fields.
2. Whether `class_table` should be populated for pursued external classes
   from the function table's `method_owner` facts, so that the class name
   resolved from a constructor (static reference `class_ref`) or from
   `field_classes` finds methods. If it should, the field-class table
   must also learn the class name from a constructor call whose callee is
   a static reference rather than a local `ClassDef`; the table currently
   requires `func.id in class_definitions`.
3. Reproduce with `imported_class.py` in this session's scratchpad shape:
   a two-file source, a spy on `reduce_abstract_tensor_topology` printing
   the class table and the attribute node's attributes. It runs in under
   a second and shows the exact table state.

Also unresolved and separate: `vol = volumes.get(...)` (a mapping value),
`at = getattr(self, "automatic", None)`, and
`self._turbine = engine.turbine.build() if ... else None` (a conditional
method result). None of these is a constructor assignment; each needs
its own resolution route and none was attempted.

### State of the trees

Turing: `31329aa9` (returned literal is the control function's own
output), `d2b33c82` (field-held class resolution). Both compiler edits
invalidate the batched kernel cache; the first `time_trials` start after
them rebuilds the contact and body kernels, and the body lowering itself
has not been rerun. The expected outcome is that the aggregate mismatch
is gone; a different shortfall would be a different defect.

## Continuation, later still: the dot operator, and the contract

### The edge that was missing, and where

Source pursuit at ingestion (`graph_express2._expand_unresolved_ast_parents`)
resolves a call's target by bindings, name to Python object: `self.field.step`
is `bindings["self"]`, then `getattr(owner, "field")`, then the class's own
`self.field = Cls()` from source when that fails (`_class_field_reference`),
then `getattr(Cls, "step")`. The pursued method's class is admitted as a
`ClassDef` in the module, which is how a class enters the reducer's
`class_definitions` and class table. That is the `.` edge that populates the
class system.

For a class supplied as a Python object, `self` is bound to the class. For a
class that exists only in the submitted text -- `Sim` in the repro,
`EngineCycleSim` in the engine -- `self` was bound to nothing (measured:
`self=None` while `Field` resolved fine), so the chain broke at its first
link, every `self.<field>.<method>()` was reported `dynamic_or_primitive`,
and the field's class never entered the table. Turing `e40c6df9` binds
`self`/`cls` of a source-text class's methods to the `ast.ClassDef` and
navigates a field on it through the class body: `self.x = Cls()`,
`self.x: T = ...`, class-level `x: T`, and `setattr(self, "x", ...)` with a
literal name. Method names resolve to nothing on this path; the reducer
links those through the class table as before. The AST-valued binding is
kept out of the `_python_bindings` the reducer reads as static values, and
the contract's occurrence decision is never asked about an AST node.
Pinned in `tests/test_field_object_receiver_class.py` (the resolver step in
isolation, the declared source lowering with the imported method linked,
and the undeclared source named by the refusal). The ast-parent file
passes except `test_ingestion_leaves_source_less_parent_unresolved`, which
was already stale: `builtins.len` has carried a declarative identity
program since `f79e3448`, and such calls are skipped before the unresolved
list is written.

### What the resolved call then hit: the contract's boundary

With the receiver resolved the reason became `declared_boundary`. The
sheet's roots are relative to the sheet, i.e. the turing tree, and
engine_toy sits beside it. Measured under the engine's own contract:
`ordnance.OrdnanceField.step` and `burst.BurstField.step` classify
`unknown` and are rejected `provenance_not_declared`. So once the `.` step
works, the contract itself refuses to pursue the engine's field classes.
Before the resolver fix that question was never reached.

The contract had no way to be told about source outside its roots. It now
has one, and deliberately not a directory root: `with_sources` (turing,
second commit after `e40c6df9`) admits ONE module per entry, by name and by
file path, and only when both match; siblings stay unknown, and what a
declared module imports is not admitted by the entry. `compile_contract.py`
lists ordnance, burst and hole_emitters. The loop refusal reads the
decision the rejected call node already carries and says: the identity,
the origin file, the classification, the rule, the reason, and that the
one file should be declared if the program is meant to use it, otherwise
the program is using something it should not.

Measured with the ingestion tally (`_expand_unresolved_ast_parents` on the
engine source, about sixty seconds):

    directory root:   40 definitions admitted; 13 submitted source,
                      ordnance 12, hole_emitters 8, burst 3,
                      ballistics 2, engine_rays 2
    selective list:   36 admitted; the same minus ballistics/engine_rays,
                      which now surface by name as rejected `unknown`:
                      ballistics.material_profile, ballistics.ProjectileState,
                      engine_rays.Ray.from_points

Also visible in the tally, and worth deciding on: the engine step path
reaches `numpy.asarray`, `numpy.zeros` (rejected under full-native as a
native extension without a declared ABI) and `numpy.random` Generator
methods (`normal`, `uniform`, `lognormal`, classified unknown). Whether the
engine should be reaching those from the compiled step is a program
question, not a compiler one.

### What is still unverified

The engine lowering itself under the selective contract. The probe run
during this work used the directory-root contract and takes about thirty
minutes; its verdict, when it lands, says whether the loop lowers once the
three field classes are ingested, not what the selective contract does
with ballistics and engine_rays. Expect the next refusal, if any, to name
one of those by file.
