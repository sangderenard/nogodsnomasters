# The time field: local time velocity as a scheduling and physics primitive

Design and staged implementation, 2026-09-15. Companion to
`TIME_FIELD_AND_COMPILE_REPORT.md`, which covers the performance work and
the compile route that led here.

---

## The problem it solves

The sim runs at 0.57x realtime against a 4x target. The shortfall has to
go somewhere. Two conventional options are both bad:

* **let the frame run long** — what `dt_system`'s `step_realtime_once`
  does today, deliberately: *"liveness over target FPS"*. Everything
  lags, uniformly, and nobody can tell which part is expensive.
* **coarsen dt to fit** — buys framerate by discarding the stability
  limit, and blows up exactly when the sim gets interesting.

The third option: **take a fixed portion of the realtime budget, and let
local time velocity absorb the difference.** Regions that cannot be paid
for advance less *world* time. The step size is never coarsened; only the
interval is shortened. Stability becomes a hard floor and time becomes
the soft variable — the opposite of adaptive-dt-under-budget.

Nothing here is a replacement for what `dt_system` already does. It is a
finer articulation of it, selected by configuration:

| | `realtime.py` today | this |
|---|---|---|
| allocation granularity | per engine id | per scope/node |
| what an allocation buys | `dt` directly, 1:1 | `round_max` — world time |
| stability limit | ignored, by design | hard floor, never crossed |
| overrun | frame runs long, globally | local world time shrinks |

---

## The form

### One scalar per node, stored as a log

The field is `log_tau` per node. A joint's time ratio is a **difference**
across it. Any difference field has zero circulation around a closed
loop, so cycle consistency holds **by construction** rather than by
solving a constraint system.

That matters because the coupling graph is not a tree. Measured on a real
machine graph: **306 nodes, 259 coupling edges, 57 components, 10
independent cycles.** With independent per-joint ratios those ten cycles
would each impose a closure constraint forever. As a difference field
they vanish. Verified: with a randomised field, the ratio product around
a real 4-node cycle is `1.000000000000000`.

The absolute level is **unobservable** — a gauge choice. Shifting the
whole field changes no ratio (verified). Inside a zone every law is
already written against that zone's own clock, so the physics is exact
and nothing needs to know its own rate.

### Time does not drop out in three places

1. **crossing a boundary** — a product sampled in one zone read in
   another must be resampled; it must carry the rate it was sampled at
2. **comparing across zones** — dyno figures, parity against a compiled
   core; both lie silently otherwise
3. **when the rate is changing**

### The derivative is the one with consequences

A **constant** gradient is unobservable. A **changing** one is a force:
the adaptor's freedom must angularly accelerate at `d/dt[(w1-w2)/2]`,
costing torque equal to the store's inertia times that acceleration. So
the field is dynamical — `log_tau` *and* `dlog_tau_dt`, position and
velocity.

Consequence: **the control loop's rate limit is physics, not tuning.**
Slamming the time field is dumping the clutch. How fast load-shedding may
ramp is a property of the machine.

### Hierarchy: nested scopes compose

`dt_system`'s `RoundNode` already nests — an outer round delegates its
window to an inner controller that may subdivide. Nested windows
multiply; logs add. **The RoundNode tree and the field's tree are the
same tree.** Verified: a beam at 0.4 inside a chassis at 0.5 gives
effective 0.200 exactly, siblings unaffected.

This is what makes one scalar per scope enough for a heterogeneous stack
rather than needing a scheme per kind.

---

## What can carry a gradient

Not a flag we invent — a property of the mechanism. A joint can carry a
**sustained** time gradient iff it has a **rate-difference degree of
freedom**.

| mechanism | freedom | sustained cost | notes |
|---|---|---|---|
| differential | yes | **none** | spiders absorb unbounded difference |
| torque converter | yes | heat | locking up removes the freedom |
| clutch (dry / wet-plate) | yes | heat | slips, wears, heats |
| rigid shaft / keyed hub | **no** | — | shears |
| camshaft timing drive | **no** | — | carries *phase*; no adaptor at any stiffness |

An ideal open differential is **lossless** for any rate difference:

```
equal torque T both outputs, carrier 2T, w_c = (w1+w2)/2
P_in  = 2T * (w1+w2)/2 = T(w1+w2)
P_out = T*w1 + T*w2    = T(w1+w2)
```

Verified in `time_rig.py`: the coupled ledger closes to
**exactly `0.000e+00 J`** at 1:1, 2:1, 10:1 and both-dilated. The
differential's constraint is on **angles**, and work is torque×angle —
both scalars with no clock in them. So an ideal differential needs *no
time awareness at all*. That is why it "just works".

Everything without a freedom **shears**: a rigid joint has effectively
infinite store inertia, so any derivative demands unbounded torque. The
failure is the same equation at its limit, not a rule imposed from
outside — which is what bounds the field **physically** instead of by a
tuning constant.

**Phase is separate from rate.** A differential lets accumulated angle
drift without limit — correct for wheels, catastrophic for valve timing.
So an engine's crank, cams and valvetrain are necessarily **one zone**,
and the boundary is at the coupling, which is exactly where the slipping
parts already sit.

---

## Two perspective scales

* **Interior** — crank, cylinders, circuits, valve timing, shafts. Stiff,
  phase-locked, necessarily one time zone. What a compiled assembly
  privately owns; what batches by topology.
* **Exterior** — output shaft, ports, what it is bolted to. Where
  couplings live, therefore the only place a gradient can exist.

Measured strides: Busso interior **386**, AMC **367**, hydraulic power
pack **12**, pop-up turret **21** — and the exterior is **6 for every one
of them**. A small uniform boundary is why a turret and a V6 can sit in
the same coupling graph at all.

This also says why offloading is possible: an interior is a pure function
of its own state plus its exterior boundary, so a runner needs the
interior span and the exterior inputs and nothing else.

**An engine is one kind of machine, not the only kind.** `machines.py`
already builds the same node/edge documents for a power pack, a scissor
lift or a turret, so the ABI derives from the graph and gates only the
genuinely engine-specific spans on an architecture being present.

---

## Heterogeneous scheduling

A batch needs a uniform substep **count**; each kind needs its own step
**size**. Both are satisfiable because they constrain different things —
K is shared, the window is per scope.

One K across the real population, each inside its own floor:

| kind | window | step | its limit |
|---|---|---|---|
| beam | 3.33 ms | 49.8 µs | 50 µs |
| rigid body | 8.33 ms | 124.4 µs | 1000 µs |
| voxel fluid | 16.67 ms | 248.8 µs | 250 µs |
| actuators | 16.67 ms | 248.8 µs | 2000 µs |
| circuits | 16.67 ms | 248.8 µs | 5000 µs |
| engine | 16.67 ms | 248.8 µs | 500 µs |

**The argument for the whole design, in one number:** the beam sets K for
everybody. At `tau=1.0` its 50 µs floor over a full 16.67 ms window
forces **K=334**. Dilating *only the beam* to 0.2 brings it to **67** —
five times cheaper for the entire frame, without coarsening a single step
anywhere. A global adaptive dt cannot do that; it either runs everyone at
the beam's step or breaks the beam.

---

## Implicit mod profiling

The game is for programmers to abuse the flexibility of and then pay the
price in mostly conservative symbolic physical law. Establishing how to
navigate time parameters through the dt system's matrices makes **mod
profiling implicit**.

It cannot be gamed, because cost is **measured, not declared**. A mod
cannot claim to be cheap; its own clock tells on it.

* **Local** — bad code slows its own neighbourhood, not everyone's frame.
* **Diegetic** — the profile is something you watch happen to your
  machine; optimising is gameplay, with an obvious target (tau -> 1.0).
* **Enforced by the conservative law** — the time-price only binds if the
  physics cannot be cheated. The symbolic -> AbstractTensor -> SSA path is
  where a mod's law is checked structurally rather than trusted. **The
  pipeline is the sandbox.**

Two things must hold or it becomes dishonest:

1. **Cost attribution must follow the same graph the field does.** If an
   expensive mod's cost lands on a neighbour's node, the profiling lies.
2. **The price must be a price.** If slow local time is ever
   advantageous, people will make things expensive deliberately. It
   closes if consequences (wear, heat soak, fuel burn, ATF oxidation) are
   booked per **world** second rather than per frame.

---

## Contracts and offloading

The time force store is already measured for a physical reason, and it
doubles as the honest signal that a system needs help: a store that keeps
charging means the field keeps being ramped, which is chronic
under-allocation stated in joules.

* **Leading** edge: allocation shortfall (asked vs got).
* **Lagging** confirmation: the store's persistence.

Both are required, or every transient raises a contract that is stale
before it is accepted.

### Acceptance has two independent gates

1. **Fraud** — the energy ledger must close. Costs a subtraction, not a
   simulation. A cheating helper is caught (measured: 240 J open).
2. **Correctness** — every cross-rate joint must declare the
   frame-converted rates it used, checked against our own field.

The second gate exists because of a result from `time_rig.py`:
**conservation does not catch a resampling error.** Two LSD rigs — one
resampling correctly, one not — both closed their energy ledgers to
machine epsilon (`-3.2e-14 J`) while reaching materially different speeds
(12.371/3.144 vs 12.956/3.057). An energy gate accepts both. A helper
running a different time zone is exactly the party most likely to make
that mistake, so it must show its work.

Determinism is what makes any of this possible: the sim is now seeded per
identity, so the same state and inputs give bit-identical output.
Against a 2%-noisy sim no prediction could ever be validated.

### The cone

Locally there is nothing to fan over — the owner already knows its own
throttle. **Remotely there is**, because by the time an answer arrives the
owner has moved. The cone is the input uncertainty over the latency
window, and it is finite because a pedal cannot teleport:

```
reach    = input_rate_per_s * round_trip_s
branches = ceil(2 * reach / resolution) + 1
```

Measured plans (zone at tau=1.0, `dt_limit` 1/2000 s):

| runner | ping | branches | frames | verdict |
|---|---|---|---|---|
| lan peer | 4 ms | 2 | 2 | fits |
| broadband | 35 ms | 3 | 3 | fits |
| distant | 140 ms | 6 | 9 | fits |
| fast, no assembly | 4 ms | — | — | **rejected** |

The deadline is the **prediction horizon**, not the ping — the ping is
spent inside the horizon. Charging it against a deadline that *was* the
ping made every remote runner fail by construction.

**A local thread is a runner and takes the same test.** Measured on this
machine: **93 µs** dispatch+join for one branch, 250 µs for eight, and a
contention factor of **0.24** under the GIL (4 threads, 0.97x speedup).

**The native dividend, measured**, same serial cost (~27 ms):

| workload | 4 threads | speedup | contention factor |
|---|---|---|---|
| pure Python (GIL held) | 113.5 ms | 0.97x | **0.24** |
| BLAS matmul (GIL released) | 46.8 ms | 2.31x | **0.58** |

Compiling to native is worth **~2.4x of local capacity on its own**,
before any single-thread speedup. It is what turns the local pool from a
runner that cannot help into one that can.

**Caveat found in the same run:** at small cone sizes dispatch dominates,
so the 2.4x throughput gain shows up as only 1.30x end-to-end. The native
win needs *fat* dispatches — bigger horizons, more lanes per assembly —
not many small ones.

**And a local thread is not additional capacity.** It competes for cores
the owner is already short of. Without modelling that, a chooser picks
the local thread every time and never actually offloads.

### The scaling property

A local thread and a remote peer are the same object with different
numbers. So the mechanism is built and tested locally against a thread
pool today, and multiplayer is the same code with a bigger ping. It
multithreads and it multiplayers for the same reason.

---

## The modules

All runnable and self-testing (`python <module>.py`). Staged in
`engine_toy` so they could be exercised against real graphs before moving
into `dt_system`, which is universal.

| module | what it is |
|---|---|
| `time_field.py` | the field: log tau + derivative, adaptors, stores, hierarchy |
| `engine_abi.py` | interior/exterior boundary for an engine **or** a machine |
| `abi_negotiator.py` | topology buckets, stated rejections, one K per batch |
| `time_rig.py` | the one-glance table; coupled differential ledger |
| `time_contract.py` | store-as-signal, two acceptance gates |
| `runner_pool.py` | cone math, runner assessment, local pool |
| `demo_time_field.py` | mixed fleet of real engines and machines under a budget |

---

## What is not done

* **Not in `dt_system`.** It is staged in `engine_toy`. Moving it is what
  makes it real, and `dt_system` is universal — nothing there should be
  removed, only configured.
* **`demo_time_field.py` hand-rolls the budget split** when
  `realtime.py`'s `compile_allocations` already does it properly, with
  EMA costs and penalty weights. The local copy is worse and should go.
* **Concurrency is not exercised.** `RoundNode.schedule='parallel'`
  exists and the heterogeneous test is sequential.
* **`dt_limit`s in the heterogeneous table are representative figures**
  chosen per kind, not each engine's published value. The real ones come
  from `Metrics.dt_limit`, which already exists and is already plumbed.
* **Nothing is wired into the live sim.** This is form and testbed.
