# engine_toy: three V6s, an 8.3x sim, and the road to a compiled engine loop

Session report, 2026-09-15.

Four things happened, and they turned out to be one thing. A request for a V6
exposed parameters the model could not express; unhardcoding those exposed how
much of the sim recomputes what has not changed; fixing that exposed that the
sim was never deterministic; and being unable to verify a compiled core against
a non-deterministic oracle is what forced the compile route to be understood
properly rather than assumed.

---

## 1. The engines

The catalogue had I4, I6, flat-four, flat-six, V8, V12, radial, rotary and a
spread of singles. It had **no V6 at all**. Three were added, chosen so each
one makes the model do something it could not previously do.

### `buick-231-oddfire-v6-1975` — 1975 Buick 231 "Fireball"

GM cut two cylinders off the 90-degree 300/340 V8 and kept the V8's crankshaft,
so six rods share three unsplit crankpins on a 90-degree block. The firing
interval alternates **90/150 degrees** instead of an even 120. That is the
famous shake, and it is the only engine in the catalogue that fires unevenly by
crank design. (The 1977-on 231 got a split-pin crank; this is the 1975-76
engine, and the direct ancestor of the Grand National turbo six.)

231 ci, 3.800 x 3.400 in, 8.0:1, Rochester 2GC two-barrel, SAE net 110 hp @
4000 and 175 lb-ft @ 2000. Turbo-Hydramatic 350 with a converter that has no
lock-up clutch at all.

### `vw-vr6-2800-12v` — 1992 VW VR6 2.8 12v

A vee so narrow (15 degrees) that both banks share **one cylinder head and one
head gasket**. The bores stagger along a single block deck rather than sitting
in two castings, so the engine is as long and as wide as an inline-four with
six cylinders in it — which is the whole point, since it was designed to fit
across a Golf-platform bay. Its timing chains run off the **flywheel end**,
under the bellhousing, not behind a front cover.

2792 cc, 81.0 x 90.3 mm, 10.0:1, SOHC, 12 valves, 174 PS @ 5800 / 235 Nm @ 4200.

It fires evenly — a 120-degree crank in a 15-degree vee is an even-fire engine
— so its low wobble is the ordinary V6 rocking couple, not an odd-fire interval.
Its uneven induction note comes from genuinely unequal intake runner lengths,
because one bank sits ahead of the other.

### `alfa-busso-v6-3000-12v` — 1990 Alfa Romeo 164 3.0 12v

A 60-degree V6 on a split-pin crank: the smoothest vee a six can be built at,
even-firing and naturally balanced, needing no balance shaft to be civilised.
Belt-driven single overhead cam per bank — **two camshafts, one toothed belt** —
and the long equal-length intake runners over the vee are the engine's
signature, acoustically and visually.

2959 cc, 93.0 x 72.6 mm (very oversquare, 1.28), 9.5:1, 192 hp @ 5600 / 255 Nm
@ 4400, in the transverse 164.

### Verified

All three build, idle, and pull: Buick 4346, VR6 5588, Busso 6312 rpm at WOT,
each sensible against its own redline. Geometric displacement from the declared
real bore/stroke reproduces declared displacement to four decimal places.

Roster positions **8, 9, 10** — between the GT flat-six and the drag V8.
`[` / `]` cycle engines in the pygame build.

---

## 2. What had to be unhardcoded

None of these three could be described honestly by the model as it stood. Each
field below defaults to the previous behaviour, so no existing engine changed.

| field | what was wrong |
|---|---|
| `EngineArchitecture.valvetrain` | `derive_valvetrain` called *every* two-valve head under 7000 rpm "pushrod". Both new SOHC engines came out wrong. Its own docstring already called itself a stand-in "until EngineArchitecture declares it". |
| `EngineArchitecture.valves_per_cylinder` | Lived only on `LifterSpring`, and every shared spring preset is a two-valve spring. Declaring a four-valve head meant hand-building a spring to change a number that is not a spring property. Now resolved in one place, `cylinder_ports.valves_per_cylinder`, with `drag_torque_nm` taking the real count. |
| `EngineArchitecture.camshaft_count` | The shared powertrain subunit built exactly **one** `powertrain.camshaft` for every engine ever, so a V6 with two heads carried one cam between them. The Busso now gets two, laid one per bank, each with its own bearings and drive. |
| `EngineArchitecture.timing_drive` / `timing_drive_at` | The drive was hardcoded to a chain at the front face. The Busso's belt now gets its real compliance (drive stiffness 315127 -> 53553; a chain is unchanged), and the VR6's drive is built at the rear of the block. |
| `EngineArchitecture.firing_intervals_deg` | See section 3. |
| `Transmission.lock_up_capable` | Hardcoded `True` in `AutomaticTransmission` and never plumbed, so a 1975 converter that never locks would have. The Camry's A140E really does lock up, so it was right by accident. |

Three mesh/effects matchers keyed on the literal string `timing_chain` were
widened so a belt or gear run is not silently unstyled and unmodelled.

**A rule learned mid-session and worth keeping:** cache graph *structure*, never
part *parameters*. `ExhaustSystem.segments` was cached and then reverted — all
parts are technically mutable, and there is no need to collapse parameters
until a parameterised build proves too slow.

---

## 3. The odd-fire crank is real, not a wobble factor

`wobble_amt` exists for "a crank-throw design choice, not derivable from
cylinder count/bank angle alone", and the first pass used it to *stand in* for
the Buick's odd firing. That was a disclosed simplification and it was not good
enough.

`firing_intervals_deg` now declares the real 90/150 alternation, and
`EngineArchitecture.slot_angles_deg()` is **the single place firing timing
resolves** — crank throws, the combustion scheduler and the acoustic model all
read it, so an odd-fire engine cannot come out uneven in one and even in
another.

Verified through the whole chain:

- Throws land at 0/90, 120/210, 240/330 — three pin positions 120 degrees
  apart with each shared pair 90 degrees apart. That is the real shared-pin
  geometry, and it fell out of the declaration rather than being entered.
- The offline acoustic bake at 2000 rpm produces events 7.5 / 12.5 / 7.5 /
  12.5 ms apart — exactly 90/150 at that speed.
- Every even-fire engine in the catalogue produces bit-identical angles to the
  old even spacing, checked across all 32.

`wobble_amt` went back to an ordinary V6's 0.10, because synthetic half-order
content on top of a genuinely uneven schedule would count it twice.

---

## 4. Mesh restage

`restage_static_mesh` claimed to cost "a buffer write" and actually deleted and
rebuilt the entire `GLMesh`, re-deriving atlas UVs and the thermal id column
that its own docstring said were "still valid precisely because the triangles
did not change".

`GLMesh` now caches its interleaved rows and gains `update_geometry`, which
writes only positions, normals and (optionally) material ids, and **refuses**
rather than writing past a buffer when topology has changed. A latent bug came
out with it: restaging after a burst part previously resurrected the part,
because `set_absent_parts` uploads a subset and restage uploaded the whole mesh.
The subset is now tracked and honoured by both paths.

Verified against a real 14,688-triangle mesh: first restage uploads, subsequent
restages are two buffer writes and zero uploads, a burst subset stages only its
kept triangles, and a topology change falls back safely.

---

## 5. Performance: it was all one bug

The console build was not realtime. Profiling found **one disease** with many
presentations: *recomputing something fixed by topology, every tick.*

| fix | what it was |
|---|---|
| `damage_state.sync_part_pressures` -> index | O(parts x circuits x edges) per tick. **90% of the entire sim.** |
| part->circuit **binding**, pressure read live | six floats broadcast into 345 objects every tick, able to go stale between writes |
| `hole_emitters.circuit_identity` | a `sorted()[0]` — a sort to find a minimum — **9,341 times a second**, for an answer fixed by the plumbing |
| `damage_state._circuit_identity` | the same sort |
| `node_effects.assess` -> causer-pays work set | scanned all 570 graph nodes to conclude an undamaged engine is undamaged |
| `net_torque` dict rebuilt per substep | rehashed ~570 identity strings per substep to arrive at the same layout |
| supercharger node set-comp, splash `any()` | fixed topology, rebuilt per call |

### The architectural fix: damage is the causer's bill

The last of these is the one worth keeping as a principle. Assessing damage was
a standing tax on every part in the machine; it is now a bill paid by whatever
caused it. Four things can damage a part, and **each one names its own parts**:
a hole or impact names the record's part, a burst names the absent part, a leak
names the parts on its circuit, a neglected air system names the parts its gunk
reaches. Neighbours of something structurally gone join the set once, and only
once, after `struct_gone` is known.

Measured: an undamaged engine now builds **0** node conditions instead of 570.
Five damaged parts build five. Cost scales with damage — 0 parts 29.2 ms, 5
parts 30.1 ms, 40 parts 33.3 ms — instead of being paid in full, always.

### Numbers

| stage | ms/step @ dt=1/500 | realtime |
|---|---|---|
| start | 28.96 | 0.069x |
| circuit-lookup index | 6.64 | 0.30x |
| bind, don't broadcast | 4.87 | 0.41x |
| sorts removed | 4.13 | 0.48x |
| causer-pays assess | 3.45 | 0.58x |
| per-substep rebuilds | **3.23** | **0.62x** |

**8.3x cumulative.** At the app's real timestep (dt=1/60) that is 29.2 ms
against a 16.67 ms budget: **0.57x realtime**.

Coverage: three architectures (I6, 60-degree V6, 14-cylinder marine
two-stroke), two timesteps, undamaged / 5 / 40 damaged parts. All land
0.49-0.58x, so nothing is engine-specific and nothing hides in the damaged
path.

**Target is 4x realtime.** That is ~7x from here and it is not available in
Python. Compiling is the only route.

---

## 6. Determinism (a prerequisite, now met)

The sim was **stochastic and nobody knew**: 99-119 rpm of run-to-run spread on
identical inputs, about 2%, from the process-global `random` module in five
combustion sites (misfire, pre-ignition, backfire, soft-limiter cut).

This cost real time during the session — a "regression" was investigated and
chased that was entirely inside the noise floor. More importantly, a compiled
core **cannot be parity-checked against a 2%-noisy oracle**; the balloon tire
lane's 1.8e-14 parity is only meaningful because both sides can be driven
identically.

`EngineCycleSim` now owns `self._rng = random.Random(_identity_seed(identity))`
with a `seed: int | None = None` field — the same per-identity convention
`valve_state` and `crankcase_state` already used. Runs now reproduce exactly;
different seeds still diverge properly.

---

## 7. The compile route (this was understood wrongly, then corrected)

The first plan was to keep the stateful Python object graph and have it call
into a compiled numeric core. That is wrong, and `structure_native.py` says why
in its own docstring:

> *"a Python loop around a compiled kernel is not a compiled program."*
> The ballistics lane proved it: **42 microseconds of marshalling around a 2
> microsecond kernel**, which threw away most of what compiling was for.

The route is **AST source -> SSA -> native**, not sympy-only, and
`engine_toy/compile_contract.py` already exists for it. Its docstring names the
situation exactly:

> *"most of what looked like the engine sim being uncompilable was the absence
> of this file."*

It also names the real blocker: **`EngineCycleSim` is not declared**, because
"its state is a graph of Python objects rather than typed spans".

### The shape that works

From `structure_native.py` and turing's `vehicle_balloon_tire_native.py`:

- **The loops go INSIDE** — over substeps and over the batch index. Arrays are
  the interface. The kernel does not return between lanes.
- **The law stays symbolic; the loop/assembly is authored raw code and
  compiled.** `vehicle_balloon_tire_native.py` emits only "face scatter, bead
  reduction, persistent state integration" and "introduces no alternative
  contact or material law".
- **The ABI template is `balloon_tire_graph_abi()`**: `state` is a flat tuple
  of *named scalars* (`skin_position_0_x`, ...), `parameters` a flat dict of
  floats. `NativeBalloonTireAssembly` carries `input_names` / `output_names` /
  `state_scalar_count`. No objects cross the boundary. That is precisely
  "supply inputs and receive outputs in Python".
- Backends already built for the balloon tire: c-fixed, c-resident,
  llvm-viewer, managed, managed-native, native-c. Preference order for us:
  **LLVM > Fortran > C**.

The emitted C shows the batching directly:

```c
for (w = 0; w < TIRE_WHEELS; w++) {
    ... in[FIELD_BASE + w * wheel_input_fields] ...
    double *s = state + w * TIRE_STATE_STRIDE + 6 * v;
```

The batch axis is the outer loop with a fixed stride into flat arrays.

### Destination

Every scientific sim runs as compiled symbolic math that went through
**AbstractTensor in between to declare a batch shape**. The tensor step is what
provides batching.

### Batching: engines, with two constraints

**The batch axis is multiple engines.** Two constraints come with it, neither a
blocker, both of which must be *stated* in the ABI rather than discovered:

1. **A batch is one topology.** `TIRE_VERTICES` and `TIRE_STATE_STRIDE` are
   compile-time constants — that is *why* stride arithmetic works. The balloon
   tire gets this free (four identical wheels). Engines do not: a 60-degree V6
   and the 14-cylinder marine diesel differ in cylinder count and circuit count
   (6 circuits on the AMC, 9 on the C18). So: bucket by topology signature, one
   compiled assembly per distinct configuration, N instances batched inside it.
   `balloon_tire_graph_abi(config)` already works this way.

2. **Substep count is uniform across a batch.** The lane loop runs in lockstep.
   This is not a restriction once the time field exists — see below.

Within a single engine, the ~17 substeps per frame are a strict dependency
chain. There is no batch axis there. The axis is across engines.

---

## 8. Local time velocity

### The idea

Parameterise local time. Anything — a player, a machine, an engine — has a
sphere of influence with a time-velocity modulation. Build something so
convoluted that it is hard to process, and your time moves slower. If an engine
needs a different substep count, alter its time velocity so it keeps batching
with everyone.

### Why it resolves the batching constraint exactly

A batch needs a uniform substep **count**. It never needed a uniform substep
**size**. Run K substeps for every lane, give lane *i* its own step size *h_i*,
and lane *i* advances *K·h_i* of local time. Per-engine substep size **is**
per-engine time velocity. One float per lane, no branching, lockstep preserved.

**Critically: you never coarsen the step, you shorten the interval.** Budget
decides how much *world time* a region advances; it never decides how finely
that region integrates. Stability is a hard floor and time is the soft
variable. This is the opposite of adaptive-dt-under-budget, which buys
framerate by trading away stability exactly when the sim gets interesting.

### The field, not a set of spheres

Time is a free variable **at any joint**, adjusted to its consequent necessity,
projected into a 3D time field.

Measured on the AMC coupling graph: 306 nodes, 259 coupling edges, 57
components, **10 independent cycles**. Not a tree. So if each joint carried an
independent ratio, those 10 cycles would constrain it — go around a loop
multiplying ratios and you must land on 1, or the loop manufactures time.

The formulation that removes the problem: **store one scalar per node —
log time velocity — and let every joint ratio be a difference across it.** Any
difference field has zero circulation around a closed loop, so cycle
consistency holds *by construction*. The multiplicative constraint becomes
additive and the holonomy vanishes. What is left is a scalar potential on the
graph, and **that potential is the 3D time field** — not a projection of joint
ratios as a separate step. The field is primary; joints read gradients off it.

The projection into 3D rides machinery that already exists: the thermal system
already scatters a per-node scalar into a spatial field for the shader
(per-node kelvin -> per-fragment Planck). A per-node log-time-velocity is
structurally the identical object — renderable on proven machinery, and
spatially queryable for anything not on the graph at all.

### The joint: a differential, slipping

Two shafts joined with different local rates accumulate different angle over a
frame. That difference is **twist** — the joint absorbs the rate mismatch as
stored strain, continuously, recovering the rigid case smoothly as rates
converge. A time gradient across a joint is indistinguishable from strain in
that joint.

This also subsumes the earlier "one clock per coupling domain" rule: a rigid
joint can absorb no time gradient, so it forces equal rates. Same law, special
case at infinite stiffness. And it makes the gradient limit quantitative — the
gradient a joint may carry is set by **its own compliance and strain limit**,
not a tuning constant.

The part every car already has for this is the **differential**: two outputs
free to turn at different rates, one common torque, and a carrier that takes
the reaction. The equal-torque closure is therefore conservative, *because
there is a carrier* — and the time field is the carrier. The full spectrum maps
onto hardware that already exists:

- **locked diff** — rates forced equal; rigid joint; shared clock
- **open diff** — rate difference unbounded, torque split fixed; time free
- **LSD / viscous coupling** — bounded difference with restoring torque; what
  real joints are, and what the contract should be written against

And a **planetary set** is the general form: one linear constraint among three
rates, which is exactly what additive log-time-velocity gives.

The sharper case is the **torque converter**, which this codebase already
models in detail: it permits continuous rate mismatch (slip), transmits torque
across it, and turns the mismatch into heat — `converter_slip_rad_s`,
`slip_heat_w`, ATF temperature, oxidation at 121 C, damage that stays done. A
locked converter is a shared clock; an unlocked one is a permitted time
gradient with its cost booked as heat. That is the honest energy story: steep
gradients should **dissipate**, not silently vanish. It also gives the gradient
limit a physical criterion — you may push a gradient as hard as the joint can
shed the heat, which is the same criterion that decides whether a converter
locks up.

### Fixed budget, resampled products

Rather than running everywhere sub-realtime, take a **fixed (tuned) portion of
the realtime budget**; all products are resampled and carry their **time
angle**.

The control loop is: measured cost -> field -> world-time advance -> cost. It
will want rate limiting and damping like any feedback controller, or it hunts.
Cost attribution must be at the same granularity the field resolves.

Carrying the time angle is the *mechanism*, not annotation. A product sampled
at one rate crossing into a region at another is the differential case, and the
only way to resample it correctly is to know the angle it was produced at. Drop
it and coupled things drift silently — the exact failure that hid inside the 2%
noise floor before seeding.

**Honest boundary: the field distributes throughput, it does not create it.**
With a fixed budget we stop being *broken* — no stutter, no instability, and
the shortfall becomes a legible physical quantity allocated by importance
rather than smeared uniformly. But every region running at time velocity 1.0
still requires the raw throughput. The compile buys headroom *in the field*:
how much of the world can run at full rate at once.

---

## 9. Why this is the whole design

The game is for programmers to abuse the flexibility of, and then pay the price
in mostly conservative symbolic physical law. Establishing in the engine how to
navigate time parameters through the dt system's matrices makes **mod profiling
implicit**.

It is the first profiling scheme that cannot be gamed, because cost is
*measured*, not declared. A mod cannot claim to be cheap; its own clock tells
on it. Three properties follow:

- **Local.** Bad code does not degrade anyone else's frame. It slows its own
  neighbourhood. You cannot ruin someone else's experience by writing a
  convoluted machine — only your own local time.
- **Diegetic.** The profile is a thing you watch happen to your machine, not a
  readout in a dev tool. Optimising becomes gameplay, with an obvious target:
  push your time velocity back toward 1.0.
- **Enforced by the conservative law.** The time-price only *binds* if the
  physics cannot be cheated. If a modder could author a law that manufactures
  energy, they would buy their way out and the scheme is decorative. So the
  sympy -> AbstractTensor -> SSA path is not only how we go fast — it is where
  a mod's law can be checked structurally instead of trusted. **The pipeline is
  the sandbox.** That is the strongest argument for doing the engine loop
  properly in the compiled lane: the precedent set there is the one every mod
  inherits.

### Multiplayer and offloading

Users are responsible for rendering their own content, or pay the server to
assist. Eventually, permissioned multiplayer state sharing allows offloading
into power users with the hardware to provide state updates.

Local time velocity is then also the **negotiation unit**: a region nobody can
afford runs slow until someone with the hardware takes it. Offloading is not a
special mode — it is a change in who is paying for a region's time.

### Two things that must be nailed down to stay honest

1. **Cost attribution must follow the same graph the field does.** If an
   expensive mod's cost lands on a neighbour's node, the "profiling" is lying
   and the tax is unfair. Attribution and field resolution must share per-node
   granularity — which they do if both ride the same node spans.
2. **The price must actually be a price.** If slow local time is ever
   *advantageous* — more real seconds to react, less wear per world-second,
   fewer rounds downrange — people will make things expensive on purpose and it
   becomes an exploit. It closes if production is measured in world time. Worth
   auditing that every consequence already modelled (wear, heat soak, fuel
   burn, ATF oxidation) is booked per **world** second rather than per frame,
   or the incentive inverts quietly.

---

## 10. What is next

**`engine_graph_abi()`** — a contract, not a build:

- engine index as the **outer extent** from day one
- `state` as flat named scalars: crank angle, omega, per-cylinder charge and
  thermal, per-circuit pressure/temp/fill, valve/carbon, crankcase
- **log-time-velocity as a node span**, not a per-lane parameter — it is a node
  quantity, and a lane's effective rate is a boundary value read off the field
- **a per-node cost accumulator on the same index**, so attribution and
  actuation are the same object rather than two systems that can disagree
- the joint closure **written down**: equal-torque with the field as carrier,
  dissipative at the gradient — rather than discovered later from a drift

Then the native assembly with the loops inside, following
`vehicle_balloon_tire_native.py`. Then the ~7x.

Still open from the hunt, deliberately left:

- `ExhaustSystem.segments` stays parameterised (per the mutability rule above)
- `node_effects.condition_lines` still sorts, but it is display-only and cold
- `damage_state.mesh_part_identity` sorts all graph nodes by label length per
  call; cold today, but it is the same bug class if it ever gets hot
- The existing valve-count inconsistency: the Camry 3S-FE, Miata B6ZE and
  Merlin are commented as DOHC four-valve heads but still model 2 valves/cyl
  because they use shared two-valve spring presets. Now fixable in one line
  each, left alone because it changes their valvetrain drag.

---

## Verification record

- 32 engines build; all three new ones idle and pull to sensible speeds
- firing schedule: all 32 checked, every even-fire engine bit-identical to the
  old even spacing; odd-fire verified through throws, scheduler and acoustic bake
- 1,922 circuit-pressure lookups identical to the pre-index implementation
  across three engines
- circuit identity: cached `min` matches the old `sorted()[0]` for every circuit
- part->circuit binding verified live (tracks a depressurising crankcase vent)
  and atmospheric for unbound parts
- assess verified undamaged / damaged / burst, including neighbour conditions
- `GLMesh.update_geometry` verified for in-place write, UV preservation,
  topology refusal, burst-subset staging and fall-back
- determinism verified across three engines; seeds verified to diverge
- tests: `test_fluid_routing`, `test_station_cooling`,
  `test_automatic_transmission`, `test_engine_test_stand` — **17 passed**

## A note on this commit

`drivetrain_graph.py` and `engine_cycle_sim.py` contain concurrent work from
another session (the `rotational_friction` refactor) alongside the changes
described here. `automatic_transmission.py` is line-endings only. The untracked
`rotational_friction.py`, `engine_test_stand.py` and their tests are also that
other work, not this report's.
