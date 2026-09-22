# Addendum: the simulator graph, engagement, population scaling, time slip, and the kernel-fusion trade-off

Companion to
[HIERARCHICAL_PHYSICS_SIMULATOR_PROPOSITION.md](HIERARCHICAL_PHYSICS_SIMULATOR_PROPOSITION.md)
and [ENGINE_EQUATION_ANNEX.md](ENGINE_EQUATION_ANNEX.md). Those two
documents named the fourteen honorary engines and the equations each
one owns or should own. This addendum answers a different question:
once those engines actually run side by side, what *kind of object*
holds them together, and what does the repository already have that
answers that — grounded, as before, in `turing/` (a sibling
repository, not tracked here, cited with plain code spans rather than
links).

## 1. A domain boundary is a graph edge

Every "engine A hands engine B a boundary quantity" statement in the
main proposition — Faraday publishing forces to Newton and heat to
Fourier (§3), Navier–Stokes asking Gibbs for an equation-of-state
closure (§4), Bjerknes and Navier–Stokes "exchang[ing] the same
physical boundary quantities" at a promotion seam (§5), the thermal
mesh "connect[ing] all of them through energy exchange" in the
chamber (§20) — is not a metaphor for coupling. It is literally an
edge in a graph whose nodes are dt-managed simulators and whose edges
are the shared quantities crossing between them. §19's central rule
("which physical descriptions are currently attached to this object's
state") is the node side of that graph; this section is the edge side.

This is not a new abstraction invented for this addendum. It is the
existing shape of `turing/src/common/dt_system/dt_graph.py`: a
`RoundNode` holds `children: List[Union[AdvanceNode, RoundNode]]`, each
`AdvanceNode` wraps one engine's `advance(state, dt)`, and
`MetaLoopRunner.set_process_graph` converts that tree into an actual
`ProcessGraph` (via `DtToProcessAdapter`) and schedules it with
`transmogrifier.ilpscheduler.ILPScheduler` — the same
process-graph/dependency-graph compiler-and-scheduler infrastructure
the main proposition's §16 already names as Turing's canonical
lowering path, here reused for scheduling *simulators* instead of
scheduling *SSA instructions*. `ILPScheduler.compute_levels(method,
order)` supports `asap`/`alap`/`max_local_slack` scheduling methods
over `dependency`/`processing` orderings — a real dependency-graph
scheduler, not a description of one.

`turing/examples/chamber_dt_join.py` is the small, already-running
instance of exactly this pattern, one level below the full
`RoundNode`/`ProcessGraph` machinery: it is the file where the user's
own mid-sentence realization landed — "atmosphere, I think you can
already see the dt system graph" — because `CompiledLaw.from_equations`
(a Bjerknes-family atmosphere law) and a Faraday-family field law both
become the same `LawEngine` object, both step through the same
`advance(state, dt)` call, and both are composable children of
whatever graph structure calls them. The chamber does not fuse those
two laws into one kernel; it runs them as two separately compiled
kernels under one shared clock, which is the coupling-graph pattern
already working in miniature, not merely a possible future extension
of it.

## 2. Engagement and disengagement: promotion and reduction as graph rewriting

§13 of the main proposition ("physical promotion") already describes
a region moving between fidelity levels — bulk Bjerknes state
promoted to a resolved Navier–Stokes region, a triangle's thermal
state promoted toward Gibbs/Bragg/Hamilton — and states that "the
deeper model can be instantiated as a Nodus subgraph behind the same
exterior ports," reduced back when it becomes unnecessary.

Read against §1 above, promotion and reduction are graph rewrites, not
state transformations happening inside one fixed node:

- **Engaging** a deeper engine means adding a node to the simulator
  graph (a new `AdvanceNode`/`RoundNode`, or in Nodus terms a new
  instantiated subgraph, per §17 of the main proposition) and rewiring
  the boundary edges that used to terminate at the cheaper model onto
  the new node instead — the exterior ports do not change, only what
  answers them.
- **Disengaging** is the reverse rewrite: the deeper node's current
  state is reduced/baked back into the cheaper model's state
  representation, the deeper node is removed, and the boundary edges
  are reattached to the cheaper node.
- A dt-managed engine already has the per-step vocabulary this
  requires without any new mechanism: `DtCompatibleEngine.step_with_state`
  (`turing/src/common/dt_system/engine_api.py`) returns `(ok, metrics,
  state)` every step, and nothing in the `RoundNode`/`AdvanceNode`
  contract requires a node's presence in the graph to be permanent —
  a node can be absent from one `set_process_graph` call and present
  in the next, which is engagement/disengagement expressed as an
  ordinary graph-membership change, not a new state machine.

This is the same recursive composition property §17 already names
("modules compose into programs; programs compile into modules; those
modules compose again"), applied to *when* a module is in the graph
at all rather than only to how it is built.

## 3. Statistical population scaling: one simulator standing in for many

A dt-managed simulator node need not represent exactly one physical
instance. The same engine object can be asked to represent an assumed
*population* of statistically similar instances, reporting aggregate
or distributional behavior rather than one trajectory — a single
Newton particle-integrator node standing in for an ensemble of
identical particles weighted by a multiplicity factor; a single
Bjerknes droplet-population node (already close to this in spirit:
`turing/examples/symbolic_chamber_solvers.py`'s `aerosol_step` already
carries a number density `N` and a mean diameter `D_mean` rather than
one droplet's trajectory) standing in for a cloud of droplets; a
single Bragg lattice-defect node reporting a defect *density* rather
than tracking individual dislocations.

This is not a new mechanism to invent from nothing — it is the
reserved **Boltzmann** engine's entire reason for existing, stated
plainly in the main proposition: Boltzmann "naturally owns the
mesoscopic regime ... where you care about particle distribution
functions but don't want either explicit molecules or continuum CFD."
A population-scaled node is precisely a node whose published state is
a distribution function (or its low-order moments — density, mean
velocity, temperature) rather than a single instance's state, and the
**Chapman–Enskog** bridge named in the equation annex's Boltzmann
section (§13 there) is exactly the formal justification for when that
population-level description may be reduced further into a
continuum Navier–Stokes node, or refined back into explicit discrete
instances — the same promotion/reduction rewrite from §2 above, with
"how many instances one node speaks for" as the quantity being
promoted or reduced, alongside fidelity.

No file surveyed implements this scaling mechanism today; it is named
here as the natural role a population-scaled dt node would play,
consistent with the annex's honest "not yet authored" labeling for
Boltzmann's own equations.

## 4. Time slip: simulators are allowed to disagree about what time it is

Two simulator nodes coupled through a shared boundary do not have to
share one clock, and the repository already tracks the consequence of
that explicitly rather than assuming it away. `DtCompatibleEngine`
(`turing/src/common/dt_system/engine_api.py`) carries three fields per
engine:

```
world_time: float = 0.0
observer_time: float = 0.0
causal_ceiling_dt: float = float('inf')
```

`step_with_state` computes `actual_dt = min(requested_dt, ceiling)` and
`slip = requested_dt - actual_dt` every step. A nonzero `slip` when
not running in realtime mode is a *hard* signal — the step is refused
(`ok=False`) and the slip amount is written into the returned
`Metrics`' error channels rather than silently absorbed. `world_time`
and `observer_time` are advanced independently (each only advances if
the engine itself did not already move it), which is precisely two
separately tracked clocks for the same node: the time the simulated
world has reached, and the time an external observer of that
simulation has reached — already separable fields, not a single
scalar assumption.

Composed across a graph (§1 above), this is what lets two coupled
simulators run at different rates without one silently overrunning
the other's causal horizon: `causal_ceiling_dt` is exactly the
mechanism by which one node can tell the scheduler "I cannot advance
further than this without violating my own stability or causality,"
and the refused/slipped step is the tracked record of *how far behind*
one simulator has fallen relative to what was asked of it — the "time
slip tracked between simulators" is not a proposed feature, it is
already the return contract of every compliant engine's own step
function, waiting to be read at the graph level rather than only at
the single-engine level.

## 5. The kernel-fusion trade-off: one law is technically eligible, and technically the wrong default

SymPy provides common-subexpression elimination (`sympy.cse`) as an
ordinary library facility, and nothing about the compiler pipeline
described in the main proposition's §14/§16 (`compile_sympy_equations`
→ ProcessGraph → SSA → compiled backend) forbids handing it the union
of every engine's equations at once. This is a real, available option,
not a hypothetical one: a bundle author could, today, concatenate
every `LAWS` entry from every engine surveyed in the equation annex
into one simultaneous SymPy system, run CSE across the whole union to
collapse shared subexpressions (shared thermodynamic terms between
Gibbs and Navier–Stokes, shared field terms between Faraday and
Bragg's carrier transport, and so on), and compile the result to one
kernel.

Doing so has one unavoidable consequence: **the whole system inherits
one shared timestep.** A single fused kernel has exactly one `dt`
argument; there is no seam left inside it at which
`dt_controller.STController`/`run_superstep` could apply a different
stability bound to, say, the EM half-step (`dt_limit = dx/(c sqrt
3)`, nanosecond-scale for a lab-sized cell) and the atmosphere
half-step (`dt_limit` governed by the much slower Hertz–Knudsen and
Köhler timescales in the same file) independently. The fused kernel
must run at the smaller of every constituent law's `dt_limit`, all
the time, for every law, whether or not that law's own physics needed
it that step.

The other cost is practical rather than physical: fusing the full
twelve-plus-engine roster into one SSA program before it can be
compiled once is a combinatorial expansion of exactly the kind
`turing`'s own compiler documentation warns about elsewhere in the
tree — the more equations enter one simultaneous compilation unit,
the longer that unit's own compile pass takes, and a bundle this large
fused into one kernel is the "twenty-hour compile" version of the
question, paid once per change to *any* constituent law, since the
whole fused unit must be recompiled together.

**Keeping the engines as separate compiled kernels, composed through
the graph in §1 rather than fused into one kernel, avoids both costs
at once**: each engine keeps its own `dt_limit`/`Targets`/`STController`
authority (the main proposition's §15, unmodified), each engine
recompiles independently when only it changes, and the scheduler
(`ILPScheduler`, §1 above) is precisely the tool that already exists
to say "here are all the separately compiled kernels, and here is the
dependency graph that tells you what order and what nesting to run
them in" — without requiring them to become one kernel to be
coordinated. Fusion remains available for the rare case where a
sub-bundle is known to share one timestep validly (two laws on the
same voxel grid with genuinely coincident stability limits, the way
`voxel_air_step` and `voxel_species_step` already share a cell today)
and the compile cost of that smaller fused unit is acceptable; it is
not the default posture the roster as a whole should adopt.

## 6. Summary: what already exists versus what this addendum names

| Idea | Status |
|---|---|
| Domain boundary as a graph edge | **Already the shape of** `dt_graph.py`'s `RoundNode`/`AdvanceNode` tree and its `ProcessGraph` conversion; named here as the correct reading of the main proposition's boundary language. |
| Engage/disengage as graph rewriting | **Composition of two already-real mechanisms** (§13's promotion/reduction, §17's Nodus subgraph instantiation) into one explicit statement; no new code exists for it yet. |
| Statistical population scaling | **Not yet implemented anywhere surveyed.** Named here as the concrete job the reserved Boltzmann engine already implies. |
| Time slip between simulators | **Already tracked**, per engine, via `world_time`/`observer_time`/`causal_ceiling_dt`/`slip` in `engine_api.py`; not yet read or reported at the cross-simulator graph level. |
| One-kernel CSE fusion of the whole roster | **Technically possible today**, deliberately not recommended as the default, for the timestep-locking and compile-time reasons in §5. |
| Separate kernels composed by a dependency-graph scheduler | **Already real and running**, at small scale, in `chamber_dt_join.py`; already real at full generality in `dt_graph.py`'s `MetaLoopRunner`/`ILPScheduler`. This is the recommended posture for the full engine roster. |
