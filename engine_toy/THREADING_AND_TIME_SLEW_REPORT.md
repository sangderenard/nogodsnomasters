# Threading, the function port, and the time-slew field as the graph engines feel it

Report, 2026-09-15. Third in sequence after
`TIME_FIELD_AND_COMPILE_REPORT.md` (performance, determinism, the compile
route) and `TIME_FIELD_DESIGN.md` (the field itself).

This one covers three things: the single missing port that stops threaded
programs compiling, what remains to finish the threading plan, and what
the finished system actually is from inside — the geometry a graph engine
experiences when time is a field rather than a constant.

---

## Part 0 — verified state

What is measured and committed, so the rest can be read against it:

| | |
|---|---|
| sim throughput | 217 → 26.6 ms/frame, **8.2x**, 0.63x realtime for one engine |
| determinism | seeded per identity; was ~2% run-to-run, now bit-identical |
| `dt_limit` | derived per subject from its own stiffest coupling (496 µs Busso, 457 µs AMC, **12.3 µs** marine diesel) |
| batch scaling | per-bay cost flat 26.0–27.1 ms across 1–8 lanes; exactly linear |
| threading recognition | `threading.Thread/Condition/Event` → dispatcher ops, verified |
| thread disposition | now `pooled_job` / `persistent`, committed to turing |
| compile of threaded source | **fails**, one specific error, see Part 1 |

---

## Part 1 — the function port

### The failure, exactly

Four programs through `lower_ast_source_to_ssa` under the repository
contract:

```
plain arithmetic                 LOWERED
loop, independent iterations     LOWERED
threaded, one worker             FortranEmissionError: dispatcher arguments
                                 lack SSA identities: thread_create at node 6
threaded, N workers over a list  ...same, at node 11
```

Everything up to emission works. `python-threading-to-dispatcher` mints
`thread_create` / `thread_start` / `thread_join`; `graph_express2` requeues
the target's body as a source dependency so the thread's code *is*
compiled; the ops arrive at emission carrying their pooled-job
disposition. Then:

```python
if len(arguments) != len(expression.args) or len(keywords) != len(expression.keywords):
    raise FortranEmissionError(
        f"dispatcher arguments lack SSA identities: {dispatch['operation']} at node {node_id}")
```

`threading.Thread(target=work, args=(a,))` carries `target=work` — a
function. A function has no SSA value id and never will. `args=(a,)` is
fine; only `target` fails.

```python
@dataclass(frozen=True)
class DispatchBlock:
    argument_value_ids: tuple[int, ...] = ()
    keyword_argument_value_ids: tuple[tuple[str, int], ...] = ()
```

Every port is an SSA value. **The dispatcher vocabulary can say what
values a job takes but not what code it runs** — even though the compiler
has already resolved and compiled that code. That is the entire defect.

### The registry already exists

No new registry is needed. `turing/src/transmogrifier/function_table.py`:

- `FunctionTable` — *"Addressable, recursion-aware function registry
  shared by compiler IRs."*
- `FunctionReference(address: int)` — *"Opaque address of one entry in a
  FunctionTable."*

**A function reference is already an integer address**, the same shape as
an SSA value id. So `DispatchBlock` does not need a new kind of port; it
needs a port of the shape it already has, pointing at a different table.
`function_ref=int(reference.address)` is the established spelling.

Recursion-awareness matters here rather than being incidental:
`DECLARED / RESOLVING / RESOLVED / EXTERNAL` is exactly what lets the
table hold a reference to a function still being lowered — and a thread
target frequently is not resolved when its `Thread(...)` call is reached.

Tables are built by `process_graph_function_linking.link_process_graph_
functions`, *"the source-neutral function-table operation needed when
different front ends meet at ProcessGraph"*: declare, assign
`callee.function_table`, stamp `function_ref` on the graph,
`table.resolve_graph(reference, callee)`.

**One hazard, already documented at `aot_compile:1475`:** `declare()`
binds under the bare unqualified name, last writer wins. Lookups must use
`reference_by_source_node(node_id)`, never `reference(name)`. That comment
exists because bare-name lookup was a real bug.

### The change

1. `DispatchBlock` gains `entry_reference: int | None = None` — a
   `FunctionReference.address`, not a name.
2. At the point `graph_express2` already identifies the entry expression
   (line 1787), declare it into `graph.function_table` and record the
   reference on the node. That site currently detects the entry and
   discards the fact; it only uses it to requeue the body.
3. Emission routes the `target` keyword to `entry_reference` and excludes
   it from the arity check. `args=(a,)` stays an ordinary value port.

Three edits, no new concepts, no new table.

### Where the invariant belongs

The arity check is in the worst possible place. Emission is the *last*
moment to discover this, which is why a missing function port surfaces as
"lack SSA identities" at a node number, thousands of lines from the cause.

The invariant should be stated at graph build:

> every dispatcher argument resolves to either an SSA value or a function
> reference

There it can name the argument, and the target is still in hand.

### Why this generalises

`Thread(target=...)` is not a special case, it is the first instance.
**Any argument whose resolved identity is a callable definition rather
than a value is an entry reference.** Callbacks, `key=`, `map(f, xs)`,
any higher-order call — all hit this identical wall today, and all are
fixed by the same port. The extraction receipt already carries `identity`
and `action`, so the test is available wherever it is wanted.

---

## Part 2 — finishing threading

### Done

`python_special_cases.py` translates ordinary Python threading into
dispatcher operations under the rule `python-threading-to-dispatcher`.
Verified end to end: `threading.Thread(target=…, args=…)` followed by
`.start()` and `.join()` lowers to `thread_create`, `thread_start`,
`thread_join`.

Committed to turing (`e6c2fe37`): those three now carry

```
dispatch_disposition = "pooled_job"
pool_residency       = "persistent"
```

on both the live record and the extraction contract's parameters. The
reason, stated at the rule rather than left to each backend: **a `Thread`
in authored source is a request for concurrent progress, not for an
operating-system thread.** Taken literally it means spawn-and-destroy per
unit of work, the one disposition the runtime never wants —
`turing_pool` exists precisely because workers should start once and park
between frames ("already up and waiting for jobs"). A backend reading
`dispatch_disposition` cannot choose to spawn, and the zero-worker case
stays a *configuration* of the same claim loop rather than a second
implementation that could drift.

`Condition` and `Event` deliberately untouched: synchronisation, not work
submission.

### Remaining

**The function port** (Part 1). Nothing threaded compiles without it.

**`concurrent.futures`.** `_THREADING_CONSTRUCTORS` covers
`threading.Thread/Condition/Event` only. `ThreadPoolExecutor`,
`.submit`, `.map` and `Future.result` are the more common spelling in
modern code and translate to nothing. They are also a *better* fit for
the pooled-job disposition than raw `Thread` is, since they already say
"pool".

**`with X() as y:` never binds.** Bindings are recorded only in
`visit_Assign`. A `withitem`'s `optional_vars` is not an assignment, so
`with threading.Condition() as gate:` yields `condition_create` and then
nothing — `gate.wait()` is invisible. (The existing `visit_With` handles
`with c:` for an already-bound name, which is a different form.) Ordinary
threading language that does not translate.

**Handles that travel through containers.** `workers.append(worker)`
then `for worker in workers: worker.join()` loses the binding, so the
join is not recognised. Direct `worker.join()` works. This is the same
class of gap as the `with` form: `bindings` is a flat name map with no
notion of a handle flowing through a structure.

### What it feeds

Once a threaded region compiles, it is not executed as written. It enters
a chain that is already complete and already proves its own claims:

```
ordinary threading in source
  -> python-threading-to-dispatcher        (recognition)
  -> dispatch ops, pooled_job/persistent   (disposition)
  -> Deploy/Join region                    (structure)
  -> deployment_ssa_binding                (PROOF)
  -> deployment_classification             (RE-PLAN)
  -> turing_pool / HostDeploymentPool      (execution)
```

The proof step is worth restating because it changes what an annotation
means: *"lanes are independent iff no value defined in one lane is
consumed by a sibling; a region that fails the check is reported and left
unbound — never silently 'fixed'."* An author writing threads makes a
claim; the compiler converts it to a theorem or refuses it and keeps the
serial order. Threads cannot miscompile into a race.

The re-plan step changes what threads *are*: `deployment_classification`
decides per region between `graphics-output`, `shader-compute`,
`thread-workers` and `host-linear` from the region's real extent effect.
Work an author wrote as threads may come out as a compute shader or as
element splitting. **Authored threading is permission, not prescription.**

And loops need none of this. A loop with no carried bindings, not
backpressured, whose state effects are all indexed publications is
*already* discovered by `loop_composer` and minted as a
`parallel_candidate` region with no annotation at all. Derivation beats
declaration; explicit threading is for the heterogeneous case analysis
cannot see.

---

## Part 3 — the finished product: time-slew geometry as a graph engine experiences it

The parts above are plumbing. This is what they are for.

### What an engine is, from inside

A graph engine — a V6, a beam bank, a voxel fluid, a turret — is a
region of a coupling graph with its own clock. It has an **interior**
(crank, cylinders, circuits, timing: stiff, phase-locked, necessarily one
time zone) and an **exterior** (shaft, ports, what it is bolted to: where
couplings live, and therefore the only place a time gradient can exist).

Measured, those are very different sizes: interiors of 386, 367, 21, 12
scalars for a Busso, an AMC six, a turret and a power pack — and an
exterior of **6 for every one of them.** A small uniform boundary is why
a turret and a V6 can sit in the same coupling graph at all, and why an
interior can be handed to a runner: it is a pure function of its own
state plus that boundary.

### The geometry

**The field is a scalar potential.** One number per node, `log τ`. A
joint's time ratio is a *difference* across it.

**Level is unobservable.** Shifting the whole field changes no ratio
(verified). Inside a zone every law is already written against that
zone's own clock, so the physics is exact and nothing needs to know its
own rate. An engine cannot measure its own time velocity, only its
neighbour's relative to itself. Time genuinely drops out — locally.

**Circulation is zero by construction.** Because ratios are differences
of a potential, going around any closed loop multiplies to exactly 1.
Verified on a real machine graph: 306 nodes, 259 coupling edges, 57
components, **10 independent cycles**, ratio product around a real cycle
`1.000000000000000` with a randomised field. This is not enforced; it
cannot fail. Ten cycle constraints simply do not exist.

**Nesting adds.** Scopes compose by summing logs, which is exactly what
nested `RoundNode` windows do multiplicatively. A beam at 0.4 inside a
chassis at 0.5 is effective 0.200, siblings untouched. The dt graph's
tree *is* the field's tree.

**Slew is the only thing with consequences.** A constant gradient is
free — an ideal open differential passes any rate difference losslessly,
forever, felt by nobody, because its constraint is on *angles* and work
is torque×angle, both scalars with no clock in them. Verified: the
coupled ledger closes to exactly `0.000e+00 J` at 1:1, 2:1, 10:1 and
both-dilated. It is the **derivative** that is a force: the adaptor's
freedom must angularly accelerate, at a torque of its own inertia times
`d/dt[(ω₁-ω₂)/2]`.

So the field is dynamical — `log τ` *and* `d(log τ)/dt`, position and
velocity — and the second is what the machine feels.

### What a graph engine actually experiences

**Running slow feels like nothing.** An engine at τ=0.3 has no internal
symptom. Its combustion, its oil pressure, its valve timing are all
exactly right on its own clock. It is not degraded; it is *elsewhere in
time*. The only observable is at its boundary, against a neighbour.

**The price is real and external.** In the same wall-clock frame it
advances less world time — measured directly, a dilated bay covering
0.239 s while a free one covers 0.400 s. It produces less, burns less
fuel, wears less, and falls behind anything running at full rate. The
cost of being expensive is paid in world time, which is the only currency
that cannot be gamed.

**Slewing feels like a clutch.** Changing a zone's rate transmits torque
through whatever joins it. Measured on the rig: a differential asked to
ramp supplies 0.051 N·m, a torque converter 0.127 N·m (more spider
inertia, bigger reaction for the same ramp). Settle the field and both
return to exactly 0.000. Load-shedding is not free; it is an event the
machine feels, and how fast it may happen is set by hardware inertia, not
by a scheduler constant. **Slamming the time field is dumping the
clutch.**

**A joint without freedom shears.** Rigid hub, splined shaft, gear mesh:
no rate-difference degree of freedom, so any slew demands unbounded
torque. Stall, then shatter — the same equation at infinite stiffness,
not a rule imposed from outside. The field's gradient is bounded
*physically*, by what mechanisms are present, and the system is
self-auditing: you do not enumerate which joints need adaptors, you run
it and the ones that do not have them break, loudly, exactly where one is
missing.

**Phase is stricter than rate.** A differential lets accumulated angle
drift without limit — correct for wheels, catastrophic for valve timing.
So a timing drive admits no gradient at *any* stiffness, which is why an
engine's crank, cams and valvetrain are necessarily one zone and the
boundary is at the coupling. The parts that can carry a time gradient are
already sitting exactly where you would want to split a zone.

### Where dispatch and time meet

They are the same axis, which is the point of Parts 1 and 2.

`turing_lane_fn(context, **lane**, chunk, chunks_per_lane)` in the C
runtime; `HostDeploymentPool.deploy(lanes)` on the host; the engine index
as the outer extent in the ABI; the batch a compiled assembly runs.
One lane, four names.

A batch needs a uniform substep **count**; each kind needs its own step
**size**. Both are satisfiable because they constrain different things —
K is shared, the window is per scope. Verified across the real
population, each inside its own floor:

| kind | window | step | its limit |
|---|---|---|---|
| beam | 3.33 ms | 49.8 µs | 50 µs |
| rigid body | 8.33 ms | 124.4 µs | 1000 µs |
| voxel fluid | 16.67 ms | 248.8 µs | 250 µs |
| engine | 16.67 ms | 248.8 µs | 500 µs |

And the argument for the whole design in one number: **the beam sets K
for everybody.** At τ=1.0 its 50 µs floor over a full 16.67 ms window
forces K=334. Dilating *only the beam* to 0.2 gives K=67 — five times
cheaper for the entire frame, with no step coarsened anywhere. A global
adaptive dt cannot do that. It either runs everyone at the beam's step or
breaks the beam.

### The invariant the whole thing rests on

**Stability is a hard floor; time is the soft variable.** The budget
decides how much world time a region advances. It never decides how
finely that region integrates. This is the opposite of
adaptive-dt-under-budget, which buys framerate by discarding the
stability limit and then fails exactly when the simulation becomes
interesting.

Everything else follows. Because steps are never coarsened, a dilated
lane is still *correct*, just behind. Because it is behind rather than
wrong, falling short of budget is survivable and legible instead of
catastrophic. Because the shortfall is a physical quantity on the graph,
it can be seen, rendered, priced, and sold to a runner.

### What it composes into

**Implicit mod profiling.** Cost is measured, not declared, so a mod
cannot claim to be cheap — its own clock tells on it. Local: bad code
slows its own neighbourhood, never everyone's frame. Diegetic: the
profile is something you watch happen to your machine, with an obvious
target (τ → 1.0). And enforced by the conservative law — the time-price
only binds if the physics cannot be cheated, which is why the symbolic →
tensor → SSA path matters as a *sandbox* and not only as a speed lane.

**Contracts and offloading.** The time store charges when the field keeps
being ramped, which is chronic under-allocation stated in joules — a
signal the machine already produces. Acceptance has two independent
gates, because conservation catches fraud but **not** a resampling error:
two LSD rigs, one converting rates correctly and one not, both closed
their energy ledgers to machine epsilon (`-3.2e-14 J`) while reaching
materially different speeds. A helper must therefore also declare the
frame-converted rates it used.

**A cone, but only remotely.** Locally there is nothing to fan over — the
owner knows its own throttle. Remotely the input is unknown for a round
trip, and the cone is exactly that uncertainty, finite because a pedal
cannot teleport: `branches = ceil(2·rate·rtt/resolution)+1`. 2 branches
at 4 ms, 6 across 9 frames at 140 ms. And the dilated zone is both the
one most needing help and the cheapest to help, because its shorter
window needs fewer substeps for the same stability limit.

**One mechanism for threads and for peers.** A local worker and a remote
machine are the same runner with different numbers. Which is why the
dispatch work and the time work are one project: the pool that takes a
compiled lane is the pool that takes an offloaded zone, and the lane
index is the same integer throughout.

---

## The shape of done

A world where every scientific simulation is compiled symbolic math that
passed through AbstractTensor to declare its batch shape; where those
compiled assemblies are dispatched as lanes onto hot pools that never
spawn; where each region carries its own clock on a gauge-invariant field
whose slew is bounded by real hardware inertia; where a region nobody can
afford runs slow instead of breaking the frame, says so in joules, and
can be bought by whoever has the hardware — and where an author writes
ordinary loops and ordinary threads, and the compiler proves what is
independent, decides what runs where, and never takes their word for it.

The distance to it, from here, is one integer port on a dataclass.
