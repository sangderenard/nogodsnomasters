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

## The other blocker, which is separate

The vehicle body will not lower through the batched route. Emission
refuses because the call selects 144 aggregate positions while the callee
produces 143. The region call is emitted at
`src/compiler/precompile_to_ssa.py` around line 2930 carrying
`output_ids` but no `output_slots`. The branch in
`src/compiler/ssa_llvm_backend.py` around line 2874 resolves exactly this
mismatch by mapping declared positions onto callee output indices, and it
is reachable only when that attribute is present. The site that populates
it correctly is `fortran_c_shell.py` around line 27570, whose own comment
warns that caller and callee identifiers occupy independent numbering
domains and must not be matched by value.

Do not assume the cause is the one constant in the law's return. That was
my inference and it does not survive testing: five small reproductions
containing a literal, a pass-through parameter, a shared temporary and
combinations of them all emit cleanly with no mismatch. Whatever produces
the mismatch requires something those lack, most plausibly output count
or the presence of several planned regions.

## The plan

Work the aggregate mismatch first, because it is the one that unblocks a
measurable regression, and work it by bisection rather than by argument.
Grow a synthetic case toward the real one along output count and region
count. Each iteration costs seconds where the real build costs half an
hour, and the amended diagnostic will name what is missing once it
reproduces. If it does not reproduce even at the body's shape, that is
itself informative and points at the class-surface path instead.

Then take the class identity gap. The question is now specific: a
receiver acquires `class_ref` somewhere, and `self.ordnance` does not get
one even though `self.ordnance = OrdnanceField()` appears in the same
file the compiler is reading. Find where that resolution happens and why
it does not reach these receivers. Only after that is it worth deciding
whether the remedy is to let declared records inform the classifier or to
resolve class references through attribute chains; the evidence currently
favours declaration over copying, since the outputs already exist.

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
