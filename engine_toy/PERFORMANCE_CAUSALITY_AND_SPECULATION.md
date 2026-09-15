# Performance and causality, and what it costs to run ahead

Design record, 2026-09-15. Fifth in sequence after `TIME_FIELD_DESIGN.md`,
`TIME_FIELD_AND_COMPILE_REPORT.md`, `THREADING_AND_TIME_SLEW_REPORT.md` and
`NEGATIVE_DRIFT_AND_DIFFUSION_REPORT.md`.

Written because a day of building a racer on the dt system produced one
category error over and over, in four different disguises, and the error
is worth stating once in ink so it stops being rediscovered. Everything
in Parts 1 to 3 is measured. Part 4 onward is design, and says so.

---

## Part 0 -- the error, stated once

**A measurement about the host was being used as a causal quantity.**

There are two clocks and they were being read off one number:

*Wall time* is what the machine spent. It is an observation about this
computer on this run. Nothing inside the physics can see it, and it is
only ever known *after* the work is done.

*World time* is what the simulation advanced. It is owned by each sim's
own stability floor. It is the causal axis.

Every scheduler written during that day tried to make a within-frame
decision out of a between-frame observation -- the cost of frame N cannot
inform frame N, only frame N+1 -- and each needed banking, prediction and
caps to paper over the impossibility. They were all wrong in kind rather
than in detail.

The repository already had the correct construction. It was not used.

---

## Part 1 -- the two pictures, and the single arrow between them

### Performance

Wall milliseconds, spent by the host thread, measured per advance.
Observational and always lagging. It has exactly one legitimate output:
**it informs the window a zone is offered on a later frame.** It never
sets a `dt`, never subdivides anything, and never appears inside a law.
Its caps are budgets, and overrunning one is a scheduling fact and
nothing more.

### Causality

World seconds. The window a zone is *offered* is a budget decision. The
subdivision within it is owned by stability and by nobody else. Steps never coarsen -- a lane that needs finer steps gets a
shorter window, less world time per frame, not a worse integration. Its
caps are not budgets: they are places where world time is silently
dropped, which is the part of this document that matters most.

### Tau is a result, not a lever

**This was got wrong for a whole day and the rest of the document depends
on the correction.** Tau was being treated as a control signal: the
allocator observes cost, sets a tau target, the window becomes
`reference_dt * tau`. That puts load on one end of tau and makes it an
input.

It is an output. A zone is offered a window. Its substeppers do what
their own stability requires, its caps bind or do not, and some amount of
world time comes out the other side. **Tau is that amount over the
reference.** Nobody sets it. It is read.

*The inverted-feedback problem dissolves.* Feeding measured cost into an
allocation appeared to hand an expensive engine a larger `dt` and so more
cost. There was never a loop to invert: the budget decides the window,
and whatever tau results is a fact about what happened.

*Asked-versus-got is the definition, not hygiene.* It is the measurement
that constitutes tau, which makes the caps in Part 3 graver than they
first read. When `MAX_CATCHUP_STEPS` binds and an engine advances 50 ms
instead of what it was asked for, that IS a tau change with no record of
it. Every silent cap is the field losing track of the truth.

*Tau is used at boundaries, not in scheduling.* Resampling a product from
a zone whose realised rate differed, and the adaptor reaction when a
realised rate is changing. Neither is a scheduler.

The ramp limit moves with it. It was being applied as a clamp on the way
in. It states what a joint can *stand*, so it belongs on the reaction as
a diagnostic: if a measured rate changes faster than the adaptor can
support, the joint shears. A consequence, not a governor.

**A naming hazard in the live code.** `TimeBudget.allocate` returns
`log_tau_target` and `demo_time_field` feeds it straight into
`field_.set_target`. What the allocator produces is *how much world time
is being offered*; the tau that comes out is whatever the zone managed
with it. Two quantities wearing one word is what kept the confusion alive.

---

## Part 1b -- the world time floor, and the only two ways to differ

Substeps come off until what is left is irreducible: **one substep of the
most expensive sim.** It cannot be subdivided, cannot be made cheaper and
cannot be skipped. The world time that single substep covers is the
**world time floor** -- the smallest grain the shared world advances in.
For the machines here that is `FIXED_PHYSICS_DT_S`, 1 ms, costing about
1.9 ms of wall time in the Mazda and 2.6 ms in the Wasp.

**Real time is then just real time.** Not managed, not targeted, not
budgeted. It is whatever the floor costs, repeated.

That leaves exactly two ways for anything to differ from anything else.

**Behind together.** The shared world advances at the floor and every
zone in it shares that rate. Nobody is publicly faster. This is why a
measured tau comes out *uniform* across zones -- observed here as 0.180
for all four, which reads like a scheduling bug and is not one.

**Ahead privately.** A zone may run past the shared front only in
speculation: unpublished, quarantined, and subject to contest, collapse
and conservation exactly as Parts 4 and 5 describe. To become public it
must come back and land at the shared front.

There is no third state, and that is the safety argument. Public
divergence would be one party's causality outrunning another's with no
adjudication. Making "ahead" strictly private means every disagreement is
settled before it is recorded.

---

### The arrow

One direction, one hop, nothing else crosses:

    a past frame's measured cost may choose a future frame's window

Manipulating a sim's `dt` to control wall cost is spending the causal
axis to buy performance, and the sim never agreed to it -- its `dt_limit`
is the entire statement of what it will tolerate. Choosing how much world
time to *ask for* is legitimate, because a zone cannot observe its own
rate. That is the gauge freedom, and it is the whole reason tau is fair
where a manipulated `dt` is not.

### Where the live code violated it

`MetaLoopRunner.run_round`'s realtime branch took `compile_allocations`'s
output -- a wall-clock millisecond budget -- and used it directly as
`step_dt` in seconds. A unit pun is the entire mechanism by which a host
measurement became a causal quantity. Trace the feedback and it is also
inverted for dilation: an expensive engine earns a larger allocation,
therefore a larger `dt`, therefore more world time and more cost.

---

## Part 2 -- what was measured, on the way to finding that

All figures from `engine_toy/time_trials/race.py` driving a Mazda B6ZE
and a Pratt & Whitney R-1340 through `GraphBuilder.round` and
`MetaLoopRunner`.

### The engines are not the expensive part

| | |
|---|---|
| one call of 16.67 ms world time | 25.18 ms wall |
| 42 calls of 397 us world time | 23.76 ms wall (0.9x) |
| one call of ~0 world time | ~0.00 ms -- no fixed per-call cost |

Subdivision is free. An engine costs what the world time costs,
regardless of how it is cut up. Any theory that blamed substepping for
the cost was wrong, and this measurement is why.

### Published stability floors

| sim | `dt_limit` |
|---|---|
| car.engine | 396 us (two degrees of crank) |
| plane.engine | 609 us |
| drivetrain | 1000 us |
| chassis at rest | none -- "no opinion" |

Tightest floor 396 us, so 42 substeps for a 16.67 ms round. Entirely
reasonable, and not the source of any problem.

### The scientific lane costs 800x its own physics

One round, profiled, warmed first so this is steady state:

| | |
|---|---|
| whole round | 52.6 s |
| `_RoundTransaction.copy_shallow` -> `copy.deepcopy` | 44.8 s |
| of which `Integrator.snapshot` (state, velocity, archive) | 30.8 s |
| of which `StateTable.snapshot` (deepcopy of the whole table) | 12.9 s |
| deepcopy calls | 8,400,000 |
| checkpoints taken in that one round | 2,653 |
| actual physics | ~65 ms |

`run_round` has two paths and they are not a tuning difference. The
scientific path wraps every attempt in a `_RoundTransaction`, checkpoints
with `copy_shallow()` -- which despite the name deep-copies -- and restores
on reject. Correct for a study. Ruinous for a frame.

The realtime path takes no checkpoint and never restores. **Rollback is a
capability a game declines**, because a game does not re-run a frame it
did not like. See Part 5 for the one case where that decision has teeth.

### Three defects found in the realtime lane

**One.** `compile_allocations` returns a mapping keyed by `id(adv)`.
`run_round` did `zip(self._schedule, self._realtime_allocations)`, and
iterating a mapping walks its *keys*, so every engine received a pointer
address as its millisecond allocation: 2429530572816 ms, a `step_dt` of
2.4e9 seconds. A car travelled 1.8e21 m in 120 frames. Fixed by looking
each allocation up under the identity it was keyed with.

**Two.** One engine has three identities. Cost is written under
`unique_label` (`car.engine_0`), the node is named
`advance:car.engine_0`, and `_renew_allocations` reads `id(adv)`. The
measurement is filed under one key and read under another, so the
baseline lookup always misses and `compile_allocations` falls back to
`ms_floor` plus an even split of the slack pool. Every engine gets the
same flat number. Not yet fixed -- the right key is a question about what
identity a registration has, not a patch.

**Three.** Nothing measures the cost at all. The wrappers are named
`adv_with_timing` and time nothing; they read `getattr(m, 'proc_ms', 0.0)`
off metrics that never set it. `run_round` does measure `elapsed` per
engine and drops it into `_last_timings` without feeding it back. The
loop is open at both ends.

### After the first fix

| | |
|---|---|
| 120 rounds | 1.47 s wall, 12.2 ms/round |
| frame prescription | 16.7 ms |
| world time advanced | 0.36 s |

The frame holds and the world runs behind. That is the proposition
working: not a fast simulation, a *live* one that tells you honestly how
far behind it is.

---

## Part 3 -- there are four substeppers, and three of them lie

Inside what looks like one sim there are four nested dt systems:

1. The outer round's window per zone.
2. `EngineCycleSim`'s fixed 1 ms accumulator -- `FIXED_PHYSICS_DT_S`,
   capped at `MAX_CATCHUP_STEPS = 50`.
3. `DrivetrainSolver` inside it, subdividing to `STABILITY_MARGIN /
   fastest_omega_n`, capped at `SUBSTEP_CAP = 200`.
4. The clutch junction inside the combustion loop, `_clutch_substep_plan`,
   capped at 200 again with ring escalation on top.

Each is a legitimate stability substepper and each must keep substepping -- 
an engine has to substep its own stability, and the outer system must not
duplicate that work.

**Duplicating it is a real failure mode and it was hit.**
`EngineCycleSim._step_once` already advances its own `DrivetrainSolver`.
Registering that same solver again as a separate `DrivetrainEngine`
advanced it twice per frame, with different `dt` and a stale `_omega`,
and the thermal state overflowed to `math range error`. A sim that owns a
sub-sim is one entry in the sequence, not two.

**The caps are the part that needs ink.** When a cap binds, the sim
advances *less world time than it was asked for* and tells nobody. One
`EngineCycleSim.step` call advances at most 50 ms of world time and
silently discards the remainder. That is drift being manufactured inside
a sim, one level below where the store ledger accounts for it at the zone
boundary.

**The rule this implies:** every substepper must report asked versus got.
Then a profile says where wall time went, the ledger says where world
time went, and the two pictures stay separate instead of being inferred
from one number.

---

## Part 4 -- the horizon is a light cone

*Design from here on.*

A zone may advance freely only as far as its inputs are determined. Past
that point it is not integrating, it is speculating: it must assume the
inputs, and everything it computes is conditional on the assumption.

This is already in the negative-drift report -- a zone running ahead
reaches a coupling boundary before its neighbours have produced the state
it needs, which is the same structure as the remote case, unknown input
for an interval, and takes the same cone with the boundary exchange
interval in place of round-trip time.

**There is already a real `c`.** One coupling edge per exchange interval
is the fastest anything causal can propagate. So the cone in
`runner_pool` is a light cone, and the determined horizon is its surface.
The speculation rules and the field rules are one geometry seen twice.

**The blot.** Conditional state written into the shared channel is read
by everyone else as fact. A zone that publishes a speculated future has
made its own assumption into someone else's input, and the error stops
being local -- it becomes part of the path others integrate from. The
violation is not guessing. It is letting the guess be indistinguishable
from the record.

So published and speculated state need separate channels. Today there is
one `StateTable` and everything in it is fact.

**And this is why declining rollback was a commitment.** No restore means
no way to un-blot, which is coherent only under a strict invariant:

    never advance a zone past the horizon where its inputs are determined

Dilation obeys that for free -- a zone running behind only ever reads
inputs that already exist. Running ahead does not. Neither does a step
that spans supersteps, because it will land in a future whose inputs were
assumed at launch. Those are the same problem in different clothes.

Rollback therefore should not be a global mode. It is a capability a zone
needs exactly when it crosses its horizon: scoped to the speculation, not
switched on for the cascade.

---

## Part 5 -- collapse, and who pays for it

Full rollback to the moment of departure is the wrong unit and too much
penalty. If another party turns out to occupy a region you had written
into, they did not cause your error and must not fund it. All they did
was assert a claim.

**Cost asymmetry is the governing invariant.** Rejecting a speculation
must never require the authoritative timeline to recompute anything. That
forces the quarantine above: speculative state can never have been merged,
only offered. It is also why acceptance by digest plus conservation is the
right shape and why it works at all -- the sim is seeded per identity, so
identical inputs give bit-identical output and acceptance is a comparison
rather than an act of trust.

**Checkpoints are earned, not taken.** A checkpoint commits only once the
region it covers has gone uncontested for the exchange interval. Commits
therefore lag the present by exactly the window in which someone could
still contest. No new parameter: it is the same round trip the cone width
is already built from.

**Two tiers of state.** A cheap "fair" path is maintained always, never
speculates, and is valid by construction. Banked speculative branches sit
above it. On collapse you are returned to the authoritative thread at
whichever banked state was still a possibility; if none was, you land on
the fair path. A player with a small machine gets the fair path always,
and pays no penalty for not speculating.

**Speculation is a service the speculator funds.** Its product is a
contract over a region of spacetime which becomes world map if
uncontested. A large machine buys coverage of futures, not a better
simulation, and every one of those futures still has to survive contest
and conservation to become record.

---

## Part 6 -- the field, with density as its source

Since tau is measured rather than assigned, the field is not imposed on
the world -- it is a record of what the world did. Which makes the
following an observation about structure rather than a control scheme.

`log_tau` is not *like* a lapse function. It is one: a scalar per node
setting the local clock rate against a reference, stored as a log so
gradients add, with the absolute level unobservable and zero circulation
by construction.

**Computational density is the mass.** A region thick with expensive
simulation is heavy and its clocks run slow; empty regions sit at
reference. Dilation stops being something an allocator assigns by penalty
weights and becomes something that simply happens and is then measured --
which is the same point as tau being a result, seen at the scale of a
region instead of a zone.

**And this is performance turned on its head.** Everywhere else, cost is
a problem to be concealed: drop a frame, lower a setting, interpolate,
and never tell the player. Here the machine's throughput IS its clock
rate in the world, stated openly. The simulation's performance becomes
diegetic -- part of the fiction rather than hidden behind it -- and every
mechanic in Part 7 falls out of taking that literally. Energy
differentials bleed down the density gradient into low-density space,
which is that field relaxing -- already measured in `time_diffusion`,
attenuating with radius and arriving as roughly r-squared.

**This fixes the gauge.** The one genuinely undefined freedom in the
field was what tau is measured against, since the level is unobservable.
The answer is empty space, far from any computational mass. That is a
boundary condition rather than a convention.

**The hazard, stated before the algebra gets pretty.** The clean
conservation result -- the coupled ledger closing to exactly 0.0 J at
every ratio including 10:1 -- holds precisely because what crosses a
boundary is torque times angle and neither has a clock in it. The
negative-drift report warns that ledgering reference-frame energy invents
energy on every rate change, sixteen-fold at tau=4, from nothing.

So a relativistic energy relation may be a *display* quantity or a
*scheduling* quantity, never the account. Where it earns its place is as
the demand model: rest cost as the mass term, rate-of-change cost as the
momentum term, adding in quadrature so a settled zone pays only its rest
cost. **Unverified and required before adoption:** that the quadrature
form is invariant under a shift of the level. If it is not, it is
measuring the gauge and it is wrong however well it reads.

---

## Part 7 -- what falls out as play

None of the following was designed. It is what the rules above do when
there is more than one party.

**Empty ground is fast ground.** Low density is high lapse, so a sparse
position runs near reference while a dense one wades.

**Occupying it destroys it.** March an army onto the fast patch and the
army is the mass that slows it. A force defending good ground is the
reason it stops being good. A resource that evaporates when claimed.

**Presence is denial, and it is cheap.** One unit parked in an enemy's
open country poisons it. You need not take ground, only stand in it.

**Light units are light.** A lone scout acts at nearly full rate and a
packed column crawls, and nobody wrote a movement stat to make that true.

**Nobody inside can tell.** Every law in the dense formation runs against
its own clock, so the centre of the column feels perfectly normal. Only a
commander on a sparse hilltop, or a product crossing the boundary,
observes the difference.

**They are not defending land.** An uncontested region is where
speculation commits -- where banked futures become record. So a force
defending an empty patch is defending its right to have been correct about
the future, and by massing there it contests the region and stops anything
committing.

### Why this legitimizes electromagnetic and rail weapons

The paradox above has exactly one clean escape: **contest a region without
occupying it.** Standing in the enemy's fast ground denies it to them but
makes you heavy and slow too. A weapon that can reach four kilometres
denies that region while you remain sparse and fast.

A projectile envelope is a claim on a distant region at a future time.
Anyone inside it cannot determine their own future -- they cannot know
whether they will be hit -- so they cannot commit a checkpoint there.
Range is future-denial, and it is the only form of denial that does not
cost you your own clock.

**And the electromagnetic case has the same structure as the time field's
own rule.** A static field is predictable, so it denies nothing -- 
precisely as a settled time gradient costs nothing. Only a *changing*
field denies, for the same reason only a changing gradient costs torque.
A varying emission injects an input nobody downstream can predict, and
speculation inside its radius cannot commit. The denial radius is set by
propagation over the exchange interval, which is the light cone again.

Jamming and area denial turn out to be the same operation viewed from two
sides, and both are expressed in the vocabulary the field already has.

---

## What is not decided

1. **Whether an accepted contract binds other parties' futures or only
   supplies your own.** If it binds, it is a real claim and contest must
   be adjudicated. If it only supplies, it is a cache and collapse is
   free. Everything about the difficulty of Part 5 hangs on this.
2. **What identity a registration has**, which is defect Two in Part 2.
   Three names for one engine is not a typo; it is an unanswered question
   about what the scheduler is keyed on.
3. **What the fair path must guarantee.** Presumably that it is valid
   under the worst admissible collapse, which makes it a conservative
   envelope rather than merely a cheap model.
4. **Region is on the coupling graph, not in space** -- two things that
   cannot couple cannot contest -- but the graph for a world map is not the
   graph for a drivetrain, and that correspondence is unwritten.
5. **The diffusivity `D`** remains a free parameter, as the previous
   report already recorded. Deriving it per edge from the torque the edge
   can stand is stated, not measured.
