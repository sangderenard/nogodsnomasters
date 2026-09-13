# Engine Toy → Game: architecture notes

Captured from a working session on `engine_toy/` (the standalone engine
simulation toy at `C:\dev\Powershell\engine_toy\`, git-ignored here and
managed separately from this repo and from `turing/`). This is the
strategic picture for how the toy's work is meant to reach the real
game, written down before it got lost across the session.

## The core reframing

The toy is **the engine designer**, not a second runtime that needs to
become part of the game's own compiled system. Its job is to let
someone build and validate an engine — real parts, real geometry, real
behavior — in isolation, then **bake** the result into a package the
game can assemble, the same way `engine_baker.py` already bakes
acoustic and vibration traces today, just extended much further.

## The engine role is "a box with connections"

From the game's point of view, an engine is a stable interface, not a
specific implementation:

- A real, sufficiently wide bus: torque in/out, heat, electrical,
  pneumatic — whatever channels a given engine actually uses. Not
  artificially minimal.
- Real, addressable internal parts (a transfer case, a starter, a
  governor, ...), each with its own real connection points and its own
  real serviceability — detectable wear/fault state ("click
  detection"), not just a black-box curve.

More than one implementation can satisfy that same interface, and they
can coexist without the game caring which one a given engine is:

1. **Run it live.** The toy's own simulation, executed directly. Not
   required, but not excluded either.
2. **Bake it.** Coefficients, splines, an N-dimensional behavior map
   (inputs like rpm/throttle/load/temperature/**damage state** →
   outputs like torque/heat/fuel burn), at whatever resolution actually
   captures the real behavior.

The toy's actual job, independent of which implementation eventually
runs: characterize the engine as *"these inputs, in this state,
produce this state evolution and these outputs."* That characterization
is the real deliverable; baking or live-running it are just two ways to
execute it.

## Damage and service are not bolted on afterward

The behavior map needs a damage/wear dimension from the start, not as
a later patch on a clean-condition model. A degraded or serviced part
has to actually change the engine's output behavior through the same
real characterization, not a separate hacked-in penalty.

## Why not the state-machine substrate (for now)

Investigated directly rather than assumed:

- `turing/src/common/abstract_tensor_state_machine.py`'s
  `AbstractTensorStateMachine` is real and wired into the compiler
  (`turing/src/compiler/state_machine_ast.py`,
  `turing/src/transmogrifier/graph/graph_express2.py`), but has exactly
  two subclasses in the whole codebase (`ComputationalWorld`,
  `ColumnarMultifluidEngine`), both single-phase (`case 0` only) today.
  It requires a `match`-over-one-scalar-field `transition()` with each
  case dispatching to exactly one `self.method()` call, plus an
  enforced `snapshot`/`restore`/`get_state` ABC contract.
- The **tire simulator** — the real precedent worth generalizing from —
  does **not** use that substrate at all. It's stepped through a
  lighter, already-generalized contract: a plain dataclass state
  (`BalloonTireManagedState` in
  `turing/src/compiler/vehicle_python_compilation.py`) with
  `dt_limit_hint()` / `copy_shallow()` / `restore()`, paired with a
  free `advance(state, dt) -> (ok, metrics)` function, run by
  `dt_controller.run_superstep` — a generic function that owns
  dt-selection, rollback, and retry for *any* state+advance pair.
- Production already has a real, mature torque-communication
  convention between drivetrain and wheel/tire:
  `delivered_axle_torques[wheel]`, `external_hub_torque_{wheel}`,
  `tire_reaction_torque_{wheel}`, real differential/traction-control
  routing, all in `abstract_ui_vehicles.py`. A baked engine package
  should target *this* real convention, not invent a new one.
- "Partial validator" was checked for and does not exist anywhere in
  the codebase or its git history — a dead lead, noted so it isn't
  chased again.
- The compiler has been fighting to successfully compile the dt system
  for 3+ days of continuous agent work as of this note. Adding more
  complexity into that same path (state-machine porting, dt-system
  integration for the engine toy) right now risks compounding an
  already-hard problem for no immediate payoff. The baking approach
  above deliberately avoids touching the dt system or the state-machine
  substrate at all.

## Refined: crate engine, not a black box

"Box with connections" (above) undersold it. The right model is a real
crate engine: a complete package that isn't fully self-sufficient by
design. Some parts ship in the crate. Some parts the buyer has to
install before it runs. That distinction is load-bearing, not a
simplification.

- **The installable unit is a prism** — a bounding volume — carrying a
  declared, finite set of real fluid and mechanical connectors on its
  boundary.
- **Internally it still resolves constituent part identities**, even
  though it presents to the game as one baked object. Damage, wear,
  and service need to reach a real sub-part (a valve spring, a
  transfer case, a starter), not just move a dial on an opaque blob.
- **Every system the toy simulates has to be classified**: built-in
  (ships inside the prism) or supplied-elsewhere (the game/vehicle
  provides it; the crate just declares it needs one). This is not
  automatic and not the same for every part.

This classification already exists, half-formed, in the toy's own
code — a sign it's the right cut, not a new invention. `drivetrain_
graph.py`'s fuel circuit comment already places the tank/pump at a
`chassis_remote` position specifically because "a real fuel tank is
never bolted to the engine itself, it sits elsewhere in the vehicle" —
already treated as supplied-elsewhere, just without a formal declared
mechanism. The fuel rail/injectors, by contrast, are engine-mounted at
`front` — already treated as built-in. The toy has been drawing this
line implicitly through node placement; what's missing is making it an
explicit, declared property of each part instead of something buried
in a position choice.

**The classification rule itself is not settled** and needs guidance
beyond the fuel-tank example before it gets generalized to coolant,
starter battery, exhaust, pneumatic lines, etc. — noted as open rather
than guessed at.

## The prism is a bar cage, and mounts are load-bearing on it

Refined further: the prism the game supplies isn't an abstract
clearance volume — it's a real bar cage structure (actual structural
members with real positions), and the engine is expected to use it for
its own mounts, not just fit inside its silhouette. This is a real,
existing concept on the toy side already: `engine_geometry.py::mount_
points()` already returns real declared attachment positions
(`front_mount`, `rear_mount`, plus `oil_pan`/`exhaust_manifold`/
valve-cover points) computed from the engine's own crank extent and
bank geometry. What's missing is the other half: correlating those
points against the cage's own real structural node positions (do
`front_mount`/`rear_mount` actually land on a real bar of the supplied
cage, at a real, attachable position, for THIS vehicle's cage), not
just checking the engine's silhouette fits inside a bounding shape.
The fit check is a mount-to-structural-node correlation, not a
volume-containment test.

## Corrected: the engine is not connected through a torque port at all — it's already a real part in the graph

First pass at this got it wrong by pattern-matching on the word
"torque" instead of checking what actually carries the engine's output.
`external_differential_wrench_torque_{axle}`/`external_hub_torque_
{wheel}` are real, but they're *additive* side inputs (a winch, a PTO,
traction-control intervention) — not how the engine itself connects.

The real answer, traced directly (`abstract_ui_vehicles.py` ~line
7271): `_vehicle_powertrain_graph` builds this real edge list —

```python
torque_edges = [
    ("engine_to_clutch", "powertrain.engine", "powertrain.clutch", "engine_torque"),
    ("clutch_to_transmission", "powertrain.clutch", "powertrain.transmission", "clutch_torque"),
    ("transmission_to_transfer_case", "powertrain.transmission", "powertrain.transfer_case", "transmission_output_torque"),
]
if include_wheel_output:
    torque_edges += [
        ("transfer_case_to_shaft", "powertrain.transfer_case", "powertrain.center_shaft", "driveline_torque"),
        ("shaft_to_front_diff", ...), ("shaft_to_rear_diff", ...),
    ]
```

The first three edges — engine through clutch, transmission, and out to
the transfer case — are built **unconditionally**. Only the edges past
the transfer case (out to wheels/differential) are gated behind
`include_wheel_output`, which the toy leaves off. So the toy's own
`build_drivetrain_graph` already contains the real `powertrain.engine →
powertrain.clutch → powertrain.transmission → powertrain.transfer_case`
chain, with the same real named torque channels (`engine_torque`,
`clutch_torque`, `transmission_output_torque`) a full vehicle build
uses — because it's the literal same function building it, not a
separate compatible copy.

**There is no connection contract to invent.** The toy's powertrain
graph is a structural *prefix* of a full vehicle's, stopping at
`powertrain.transfer_case`. An engine designed and validated in the
toy becomes real in the game by calling the same
`_vehicle_powertrain_graph` with `include_wheel_output=True` inside an
actual vehicle build, continuing the identical chain the toy already
built the first three links of. "Assembly" is running the rest of a
function the toy already runs the start of, not translating between
two representations.

## The best-case target for the baked characterization

Not just numeric coefficients or lookup splines as a fallback: the real
target is authoring the engine's physics as a system of **SymPy
equations**, reduced through sympy → AbstractTensor → compiled to LLVM
or Fortran — the same real symbolic-to-native pipeline already proven
elsewhere in this codebase (the compiled-eigh precedent, the BLAS
compiler track, the Python-material symbolic-cache path). This is not
a new idea being proposed for the engine specifically — it's already
how the tire's own real physics works: `vehicle_balloon_tire.py`'s own
docstring states its "material, gas, bead, and hard-surface contact
equations are authored once as SymPy and lowered through the
repository process graph/SSA pipeline; native and browser backends
consume that same graph." The tire's `run_superstep`/dataclass contract
(above) is the *stepping* wrapper; the *physics inside* `advance()` is
already sympy-authored and compiled, not hand-written numeric code.

So the real end state for an engine built in the toy: author its
governing equations in SymPy (torque/heat/flow relations — much of
this already exists as real, disclosed formulas in `gas_turbine.py`,
`otto_langen.py`, `governor.py`, just written directly in Python
instead of SymPy), reduce and compile them the same way, and let the
baked package carry that compiled kernel rather than (or in addition
to) a sampled N-D lookup map. This is the same reduction step the dt
system's own equations go through, so it should benefit from whatever
comes out the other side of the current 3-day dt-system compiler
effort, rather than needing separate tooling built for it.

## Implemented: `engine_toy/engine_package.py`

The first real OOP scaffolding for the crate-engine model above.
Verified against all 23 catalogue engines with no failures. Built
entirely off the same `build_drivetrain_graph` output the toy's own
simulation already uses — no new node/edge vocabulary invented.

- `PartSourcing` (`BUILT_IN` / `SUPPLIED_ELSEWHERE`) plus `part_sourcing
  (node_identity)`, driven by `_SOURCING_OVERRIDES` — currently only
  `fuel.tank`/`fuel.pump` are classified `SUPPLIED_ELSEWHERE`, the one
  settled real case. Everything else defaults `BUILT_IN` until a real
  case is worked through; the classification rule for the rest is
  still open, per the section above.
- `Prism` — the crate's real bounding volume, spanning only `BUILT_IN`
  node positions (a supplied-elsewhere part doesn't count toward the
  crate's own footprint, by definition).
- `Connector` — a real graph edge crossing the built-in/supplied
  boundary (found automatically, not hand-declared per engine).
  Verified: piston and turbine engines (both have a real fuel tank)
  correctly produce exactly one connector, `fuel.pump_to_rail`; the
  Otto-Langen (coal-gas utility main, no tank) and the EV (no fuel
  system at all) correctly produce zero, honestly.
- `EnginePackage.build(engine)` — assembles the above into one real
  package per engine.
- `EnginePackage.correlate_mounts(cage_nodes, tolerance_m=0.05)` — the
  mount-to-cage-node correlation from the bar-cage section above (not
  a bounding-volume containment test). Verified two ways: an empty
  cage honestly reports every mount unmatched (no fake positives); a
  synthetic cage placed exactly at `engine_geometry.mount_points()`'s
  real `front_mount`/`rear_mount` positions correctly matches those two
  at zero distance while correctly leaving `oil_pan`/`exhaust_manifold`/
  `valve_cover` (real accessory reference positions, not structural
  mounts) unmatched, since they're genuinely too far from the cage.

## Updated: real mount hardware, an automated mounting policy, and transmission-as-cargo (`engine_mounts.py`)

The two paragraphs above describe the FIRST cut of mount correlation —
`engine_geometry.py::mount_points()`'s own invented `front_mount`/
`rear_mount` pair. That was a toy-only approximation living beside the
graph, not IN it. The graph already declares the real thing:
`mount.engine_left`/`mount.engine_right` (plus `mount.transmission_
left/right` and `mount.transfer_case_left/right`), built by the same
production `_vehicle_powertrain_graph` subunit `drivetrain_graph.py`
already calls — just never read by anything on the toy side. Two real
bugs found and fixed while wiring this up:

- `_vehicle_powertrain_graph`'s `half_width` parameter (how far apart
  the left/right mounts sit) was never passed, defaulting to `0.0` —
  every one of those "left/right" pairs was silently landing on the
  crank centerline. Fixed by passing `engine_geometry.block_half_yz_m
  (engine)`, the toy's own one real place block half-width is computed.
- `EnginePackage.build` classified `powertrain.transmission`/
  `powertrain.transfer_case` (and their mount nodes) as `BUILT_IN` by
  `_SOURCING_OVERRIDES`'s default, meaning every crate silently shipped
  WITH a transmission baked in — the opposite of a real crate engine.
  Now `SUPPLIED_ELSEWHERE` by default, matching the fuel tank's own
  real reasoning; `EnginePackage.build(include_transmission=True)` (a
  player deliberately baking more into one crate) reclassifies them
  `BUILT_IN` for that build only.

`engine_mounts.py` (Stage 4 of the "scientific bake", directly on top
of `block_dynamics.py`'s own Stage 1/2/3: lumped mass/stiffness modal
solve + `select_mount_points`) is the automated POLICY: given a real,
closed set of install contexts (`automotive`, `marine`, `aircraft`,
`industrial_stationary`, `bar_cage`), `select_technique` resolves the
real mounting TECHNIQUE a real installer would pick FIRST — rubber
isolator, cradle/subframe (heavy engines), solid-bolted (industrial
skids, aircraft firewalls, the toy's own atmospheric/turbine engines
regardless of context), or bar-cage (the crate's own supplied
structure). `block_dynamics`'s real modal candidates then refine the
hardware GRADE within that family (a firmer isolator at a point
genuinely riding close to a resonance antinode) rather than flipping
the whole install onto a different technique per-bolt, the same way a
real installer wouldn't. `wants_torque_strap` adds a real secondary
reaction anchor (never a standalone technique — it only ever
supplements a compliant primary mount) when specific torque or forced
induction/nitrous calls for one. `block_dynamics.build_block_network`
also grew an `include_transmission`/`transmission_mass_kg` option that
hangs the transmission's own real mass off the nearest block station
through a real bellhousing-stiffness edge (same pattern as its
existing oil-pan DOF) — `transmission_mass_kg` is deliberately a
caller-supplied real number, not fabricated, since no transmission
mass exists anywhere on `Engine` (a crate engine's own `mass_kg` is
the bare engine only). `EnginePackage.correlate_mounts` now checks
`self.mounting` (the resolved, technique-aware list) instead of the
old flat `mount_points()` dict. Verified against all 26 catalogue
engines: builds clean with and without `include_transmission`, and a
synthetic cage placed exactly at the real `mount.engine_left/right`
positions matches every one of them at zero distance.

## Updated: real per-component mass distribution instead of one lumped point mass (`drivetrain_graph.py`, `block_dynamics.py`)

A real center of gravity and a real inertia tensor can't come out of
`engine.mass_kg` sitting entirely on one node at the crank centerline
-- that's a point mass, not a distribution, and any mount-stability
check built on it would be judging a fiction. `block_dynamics.py`'s
own docstring already documented what `engine.mass_kg`/"powertrain.
engine" is real supposed to represent: "the bare block+crank+heads
casting" -- accessories (alternator, camshaft, ...) and the oil pan
already carry their own separate, real masses elsewhere in the same
graph, untouched here.

Real, disclosed, order-of-magnitude split of THAT bucket across its
three real named parts (not tuned to fit anything, same "genuine
literature value" spirit as `block_dynamics.py`'s own cast-iron
modulus constant): `ENGINE_BLOCK_MASS_FRACTION = 0.50` (block casting,
onto the existing per-cylinder `powertrain.engine_block_body.cylinder_
N` segments -- collapsed to the crank centerline, same as before, this
is bounding geometry, not a claim about real block cross-section
shape), `ENGINE_HEAD_MASS_FRACTION = 0.30` (new `powertrain.cylinder_
head.cylinder_N` nodes, one per real cylinder, placed at that
cylinder's own real 3D site position -- `engine_geometry.cylinder_
sites()` already carries the real bank-angle lateral offset a V/
opposed engine's heads genuinely sit at, reused directly rather than
re-derived), `ENGINE_CRANK_FLYWHEEL_MASS_FRACTION = 0.20` (stays on
"powertrain.engine" itself, the crank's own already-real reference
node). A radial/electric engine (no real per-cylinder axial spread)
folds the head share into the single monolithic block node instead of
fabricating a position for it. All three land with `mass_in_total=
True` -- a real field the production subunit already writes with real
meaning (whole-assembly mass accounting) but the toy never read.

`block_dynamics.build_block_network` no longer re-derives an even
mass split off "powertrain.engine"'s (now much smaller) total; each
station reads its own real segment mass plus its co-located head's
real mass directly off the graph. Verified two ways across all 26
catalogue engines: block+head+crank shares sum EXACTLY to `engine.
mass_kg` (zero drift, every engine); the full mesh/GL build pipeline
(`build_engine_mesh`) and the live sim (`EngineCycleSim.step`) both
run unaffected, since the new head nodes carry no visual geometry of
their own and `vehicle_mesh.py`'s `body_half_extent_m` reader already
has a real fallback for a node that doesn't declare one.

## Updated: a universal stability gate, an optional TRA decoupling check, and a subframe fallback (`engine_mounts.py`, `engine_mass_properties.py`)

Real industry practice splits mount validation into two tiers, and
this follows both rather than inventing a bespoke check: (1) a cheap,
near-universal geometric invariant every real mount layout satisfies
regardless of technique — at least 3 non-collinear support points
whose horizontal (X-Z) footprint contains the assembly's real center
of gravity (2-point mounts can never resist roll/tip-over no matter
how stiff); (2) the real, standard automotive NVH method for anything
deeper — Torque Roll Axis (TRA) rigid-body decoupling, a 6-DOF (3
translation + 3 rotation about the real center of gravity) generalized
eigenproblem built from each mount's own real position and stiffness,
run only for compliant technique families (rubber isolator/cradle;
skipped for solid/bar-cage/aircraft, matching real practice — NVH
refinement isn't the point of a rigid install).

`engine_mass_properties.py` is the real foundation this needed: total
mass, center of gravity, and a real 3x3 inertia tensor, computed as a
real point-mass distribution over every `mass_in_total` node in the
graph (the block/head/crank split above, plus whatever else the
production subunit already marks that way) — not a separate hand-
placed estimate.

`assign_mounting` now returns a `MountingPlan` (mounts + technique +
`StabilityResult` + `RigidBodyProperties` + optional `TRAResult`)
instead of a bare list. When the direct real mount points (`mount.
engine_left/right`, optionally the driveline's own) fail the
stability gate, technique falls back to `CRADLE_SUBFRAME` and a real
subframe is generated: a rectangle sized directly off the assembly's
own real mass-bearing X-extent and Z-spread, which GUARANTEES CG
containment by construction rather than checking afterward and hoping.
`EnginePackage.correlate_mounts` and its `.mounting` field updated to
the new `MountingPlan` shape (`.mounting.mounts`).

**A real, disclosed finding from actually running this, not assumed:**
every catalogue engine falls back to the subframe today, for a real,
understood reason, not a bug in the check. `mount.engine_left/right`
sit at the crank's own single X position (the production subunit's own
declared geometry) — for anything longer than a single-cylinder
engine, the block/head casting's own real mass extends well fore/aft
of that one station, so the true center of gravity routinely lands
outside a support footprint anchored at one X value, even with the
driveline's own mount points added on the aft side. The subframe
fallback is the check doing its actual job on the currently-declared
mount geometry — a real, disclosed limitation of `mount.engine_left/
right`'s own single-station placement (a production-subunit concern,
out of this toy's own scope to relocate), not evidence the gate itself
is wrong. Verified across all 26 catalogue engines and 6 install-
context combinations: builds clean, `BAR_CAGE` correctly never
triggers the subframe fallback (a supplied cage's own stability stays
the vehicle/game's responsibility, honestly reported unmatched rather
than silently overridden).

## Implemented: the crate as one steppable, animated object (`engine_crate.py`)

The vehicle-graph/engine contract, cleaned up: the vehicle leases the
engine a PRISM; the engine answers with its own real connections to
the frame (mount points -- real graph nodes that sit inside that same
prism or exactly on its boundary by construction, since the prism is
measured from the same `BUILT_IN` node set the mounts are drawn from);
the engine's own animated frames live inside that same prism (built
off the identical graph the prism was measured from, so mesh and
mount geometry can never diverge); and ONE step function advances real
state and hands back the real forces crossing the prism's own
boundary. `EngineCrate` is exactly that, built from three pieces that
already existed rather than a new parallel system -- `EnginePackage`
(prism + mounting), `EngineAnimation` (engine_mesh.py's baked moving-
part frames), and `EngineCycleSim` (the live crank-domain sim).
`EngineCrate.step(dt)`:

1. Advances the real sim by `dt` (a no-op while stalled -- starting is
   the sim's own real starter-engagement action).
2. Indexes the already-baked animation frame for the resulting crank
   angle (`EngineAnimation.frame_index` -- never re-derives geometry
   at run time, same "bake once, index during playback" contract the
   pygame frontend already relies on).
3. Returns the real forces crossing the prism's own boundary at each
   mount point this tick, via the new `engine_mounts.mount_loads`: a
   real static wrench (the assembly's own real weight through its own
   real center of gravity, plus the crank's own real reaction torque
   about the crank axis -- Newton's third law, using `EngineCycleSim`'s
   own live `current_torque_nm`) resolved through each mount's real
   position and stiffness the same way `evaluate_tra` assembles its
   own rigid-body system, so the two share one real derivation instead
   of two that could disagree. For a compliant technique the
   distribution is stiffness-weighted (a stiffer mount takes
   proportionally more load, the standard method for an elastically-
   mounted rigid body); a rigid technique splits evenly across points,
   a disclosed simplification for what a fully rigid multi-point
   support is otherwise a genuinely indeterminate problem.

Verified two ways before wiring anything up: a symmetric 4-point
subframe under pure gravity splits the real weight EXACTLY evenly
(245.25 N each on a 100 kg body); two points under a pure roll torque
produce an exactly equal and opposite force couple matching the
applied moment by hand calculation. Then verified end to end across
all 26 catalogue engines (build + step + read mount_loads back): the
summed mount forces match the assembly's own real weight to the
reported precision every time, zero failures.

## Fixed: mount.engine_left/right rederived as a real 4-point set, root-causing the subframe-every-time finding

The prior section's own disclosed finding -- every catalogue engine
fell back to the subframe -- was traced to its real, single root
cause and fixed rather than left standing: `mount.engine_left`/`mount.
engine_right` (production subunit's own declared geometry) sit at ONE
real axial station, so 2 points there can never form a real support
polygon at all (a line, not a polygon -- `check_stability`'s own
reasoning), and for anything longer than one cylinder the block/head
mass genuinely extends fore/aft of that single station regardless.

Nothing external currently dictates the spacing these have to match
(no real supplied cage exists yet), so `drivetrain_graph.py` now
rederives them as a real 4-point set spanning the crank's own real
front/rear extent: `mount.engine_front_left/right` and `mount.engine_
rear_left/right`, at real insets from `crank_shaft.front`/`.rear` (the
timing-cover end vs the flywheel end -- already this engine's own
INTRINSIC axial references, independent of how a vehicle later
orients the whole crate; a transverse install just rotates this same
real geometry about the vertical axis before bolting it in, it doesn't
change the engine's own casting attachment points). This stays
correctly frame-orientation-neutral: everything is derived and
checked in the engine's own local axes, never a specific vehicle bay
width or orientation assumption.

Verified across all 26 catalogue engines: 22 now pass the universal
stability gate DIRECTLY with these 4 real points, no subframe needed
(up from 0 of 26 before this fix). The remaining 4 (`honda-style-
commuter-i4-1500`, `aircooled-flat-four-1584`, `servo-direct-drive-
400`, `25cc-two-stroke-trimmer`) still fall back to the subframe for a
real, DIFFERENT, and correctly-caught reason: their own real center of
gravity has a lateral (Z) offset (an off-center accessory/battery
mass) slightly beyond the production subunit's own declared lateral
mount spread (`half_width * 0.58`) -- the subframe fallback earning
its keep exactly as designed for "genuinely strange" cases, not a
residual bug. Full pipeline (EnginePackage/correlate_mounts/EngineCrate/
mesh build/live sim) re-verified clean end to end after the change.

## Fixed: the intake as declared hardware — units, barrels, planes, placement (`engines.IntakeSystem`, `dressing.py`)

Two symptoms the user saw in the live view — a box wired to only one
of two throttle bodies, and blue runners converging on a chamber that
wasn't under its own throttle body — traced to ONE conflation: the
planner treated *barrel count* as *number of spatially separate inlet
units*. A two-barrel carburetor is one casting with two bores a few cm
apart over one divided plenum; the code built two throttle bodies 35 cm
apart at each firing group's centroid. The same conflation made a
"unified four-barrel" impossible to express. A second, independent
system (the legacy "manifolds sit relative to the real heads" pass in
`drivetrain_graph.py`) then rewrote only `powertrain.intake_plenum` —
never `_2` — dragging one chamber to the engine's center while its own
throttle body stayed put.

**What was already right:** `distribute_ports` groups by crank throw
angle; for the Jeep six that yields barrels alternating 0,1,0,1,0,1
across the 1-5-3-6-2-4 order — the genuine dual-plane split, kept.

**The model now, all declared on `IntakeSystem` with real defaults
derived once (catalogue can override):**
- `inlet_units` — spatially distinct carbs/throttle bodies (single carb
  1, dual quads 2, ITBs one per port). Barrels per unit come from the
  existing `ThrottleBodyAssembly`; an 8-barrel declaration is two
  4-barrel units, never one casting with eight bores.
- `plenum_planes` — firing-order halves of a DIVIDED plenum under one
  unit (dual-plane = 2), co-located; each runner is tagged with its
  plane and joins its own half of the same chamber.
- `plenum_placement` — `valley` (between the banks of a V/flat only),
  `inboard` (above the head on a straight engine), `piped` (fed by a
  real pipe from a compressor outlet — a turbo, or a blower that
  genuinely plumbs to and from the manifold rather than sitting on the
  block). `auto` derives from bank count and forced induction.

`derive_intake_hardware` resolves these to an `IntakeHardware` record;
`plan_intake` builds one chamber + one throttle body per UNIT and draws
that unit's barrels side by side on it (`build_throttle_parts`). The
legacy repositioner is now gated by an OWNERSHIP rule — it only touches
a plenum dressing did not place (`plenum_style` marker) — replacing an
earlier, too-blunt chamber-count guess. Verified: Jeep six = one
centered 2-barrel, dual-plane, inboard; `radical-cam-bigblock-7400` =
two 4-barrel units in the valley (double quad); `monster-632-twin-turbo`
= single throttle at the compressor point, piped to the chamber.

**Default-plugged bungs** (same `engine-block-port` vocabulary as
`assembly_ports.emit_ports_graph`, drawn as the same `port_` stubs as
head oil ports): a MAP/vacuum tap and an IAT boss on every chamber, a
ported-vacuum tap at every throttle base, and an injector bung on every
runner of an engine with no port injection — so injection, sensors, or
vacuum lines attach later by plumbing a real declared point, not by
inventing a casting.

Also in this batch: `mount.engine_*` nodes and their `six-axis-
compliant-mount` edges now render (the `powertrain-mount` kind was
skipped outright; the edges were left dangling when the nodes were
rederived — caught by a dangling-edge regression that now runs on all
26 engines, zero failures); air filter / throttle body / plenum each
have their own material; a separate throttle-plate animation
(`build_throttle_animation`, keyed by throttle position, composited
alongside the crank-angle frames — the live plate angle comes from
`EngineCycleState.throttle_plate_angle_deg`); the mesh reduction stage
is explicit and optional (`EngineGLView.detail`/`spring_style`,
`main_pygame.MESH_DETAIL`); viewport moved up and right.

**Flagged, not decided:** `supercharged-drag-v8-8200` (ITBs on a blown
engine) still carries the production subunit's generic `intake_plenum`/
`throttle_body` nodes untouched, because the ITB path skips the chamber
loop. A real blown drag engine runs a hat/plenum over the blower; the
right representation there is a hardware decision, not a guess.

## Parts catalogue, stage 1: the data bugs a real-parts audit exposed

A per-engine audit (what each of the 26 engines declares vs. what a
real one of that type carries) ranked 20 part families to add. Before
adding anything, the audit surfaced wrong hardware already present,
fixed here first:
- **Total-loss two-strokes carried wet-sump ports** (drain plug,
  dipstick, main gallery, head oil fill, pan rim faces) because
  `assembly_ports.part_ports` emitted them unconditionally. It now takes
  `wet_sump`; `drivetrain_graph` and `head_mesh` both pass
  `not two_stroke`, so the graph and the drawn stubs agree.
- **Non-piston kinds carried a piston core**: electric, servo-electric,
  atmospheric and expander engines had a camshaft, a piston intake
  plenum, a PCV port and a wet-sump pan/pump; the turbine had the first
  three. Purged by kind. The driveline chain stays for every kind (it is
  the toy's real dyno coupling); the turbine keeps its oil system (a
  real APU has one).
- **A 1910 hit-and-miss engine had an EFI rail** — `CarburetorProfile()`
  defaults to injected. Fairbanks-Morse is now `is_carbureted=True` (a
  mixer/vaporizer is a carburetor in this vocabulary) and gets its bowl.
- **"Twin turbo" had one turbo.** `ForcedInduction.turbo_count` (default
  1; 2 on the 632) clones the production turbo node with its own oil
  feed/drain and exhaust take-off, mirrored to the other bank, and
  splits the exhaust heat path across the units. The old
  `not two_stroke` gate on `has_turbo` is gone (a large marine
  two-stroke cannot run without its turbos; what it lacks is a wet
  sump, which production already gates separately).

**Flagged, not decided:** the Wärtsilä declares no forced induction at
all, so it still has no turbo nodes. Giving it its real 3–4 turbos means
declaring a real scavenge boost that changes its physics — a hardware
value to set deliberately, not fudged in with zero boost.

## Parts catalogue, stage 2: the universal crank-engine bolt-ons (`engine_parts.py`)

Every part here is emitted from data the engine ALREADY declares --
never from a per-engine list -- and hangs off an existing node at a
real mounting point. Combustion engines only in this stage; the
non-piston kinds get their own real parts in later stages.

- **Exhaust downstream.** `ExhaustSystem.segments` was already the real
  ordered cat/muffler/tailpipe chain the acoustic model reads; it is
  now geometry too, extended from each real collector: a short drop
  along the collector's own `outlet_direction` (now stored on the node
  by `emit_header_graph`), then every segment rearward along the crank
  axis, cans as boxes and pipe as drums, plus a plugged O2 bung. The
  production collector -> exhaust_manifold junction edge is untouched
  (the exhaust circuit, the turbo heat path and the audio key on it);
  the chain's edges carry no flow capacity, so summed circuit capacity
  is unchanged.
- **Crank-end hardware.** `crank_mesh` already DREW the pulley/damper,
  flange and flywheel; it now exposes `crank_end_fittings(layout)` so
  the graph declares `harmonic_balancer`, `flywheel` and
  `flywheel.ring_gear` (or `flywheel_front` on a twin-flywheel single)
  at exactly the drawn positions. Those nodes carry `drawn_by`, which
  `vehicle_mesh` honours by not drawing a second body over the real one.
- **Starter hardware** chosen by the real `STARTING_SYSTEMS`
  `engages`/`drive` strings production already writes onto
  `starter_drive`: ring-gear engines get a series-DC motor + solenoid
  (or a pneumatic vane motor, no solenoid) meshed to the ring gear;
  recoil-pull gets a rope drum on the nose; hand-crank / starter-cart /
  inertia starter get their real nose fitting; flywheel-bar and
  air-start need nothing beyond the flywheel/heads already there.
- **Belt drive**: a pulley per accessory on the ring plus a
  tensioner/idler; **timing drive**: cover flush on the block's real
  front face (`_block_front_face_x`, read off the block segments' own
  half-extents -- a fixed offset from crank_x_min landed inside a long
  casting and floated ahead of a short one) with crank/cam sprockets
  and the run between them.
- **Coolant plumbing** (liquid-cooled only): expansion bottle off the
  radiator, heater core with supply/return hoses. **Emissions**: PCV
  valve on the existing PCV port; EGR valve + tube when `EGRSystem.
  has_egr`. **Bellhousing** spanning crank rear -> clutch -> gearbox.
- New material families: pulley, damper, starter, coolant gear, EGR,
  PCV, housing. Hub/bolted-joint edges are joints, not pipes, and stay
  out of the view; hoses, the EGR tube and the timing run draw.

Verified over all 26 engines (graph + dangling-edge check, mesh build)
and by eye at full detail from both ends of the Jeep six.

Three corrections from looking at those frames, all real hardware:
- **Distributor end is a declared rule, not a constant.** It sat at the
  rear crank station for every engine; a single-bank engine drives it
  off the FRONT of the cam (the Jeep 258, Ford 300, most inline fours),
  a V engine's cam-in-block drive is at the REAR of the valley. Now
  front for one bank, rear for two or more (`dressing.py`).
- **The starter lies outside the case.** Its body was placed 1.25
  flywheel radii from the flywheel centre, i.e. inside the crankcase
  envelope; it now bolts to the bellhousing flank, outboard, low, with
  the pinion reaching in to the ring gear.
- **Muffler and tailpipe are chassis plumbing.** They stay in the graph
  (the exhaust circuit and the acoustic model read them) but carry
  `chassis_side=True`: excluded from the engine view (a metre of
  tailpipe was dominating the camera fit) and classified
  `SUPPLIED_ELSEWHERE` by `EnginePackage` for the same real reason the
  fuel tank is -- via a node attribute, not another identity-prefix
  rule. The close-coupled cat stays engine-side.
- Colours: the cotton-gauze cleaner is red; distributor, coils and plug
  leads are the red aftermarket-ignition look; magneto and glow-plug
  bus keep their dark material.

## Parts catalogue, stage 3: forced induction and mechanical injection (`engine_parts.py`, `dressing.py`)

- **The blown engine is not ITBs on the ports.** `derive_dressing` now
  yields a `"hat"` throttle style for a supercharger with open stacks:
  one unit -- the injector hat (bugcatcher) -- whose barrels are the
  hat's butterflies (a declared `ThrottleBodyAssembly` if any, else the
  classic pair), one open plenum, because the blower case IS the
  manifold. Dressing still places a single chamber and throttle body;
  `_emit_supercharger` then builds the real stack around the existing
  `supercharger_rotor`: manifold plate on the plenum, case sized by the
  declared rotor pack (lobe count -> rotor diameter class, boost
  fraction -> case length, disclosed proportions of the block's own
  half-width), the production rotor node becomes rotor 1 with a timed
  rotor 2 beside it, drive snout forward to the belt plane with the
  blower pulley on it and a belt run to the damper, the burst panel /
  restraint over the case, and the hat moved on top of the case (its
  own bungs riding with it) with the scoop on the hat.
- **Mechanical injection** when the build declares a mechanical pump:
  `FuelDeliverySystem.pump_kind="mechanical"` on the two blown alcohol/
  nitro engines (every catalogue engine used to read electric). A
  mechanical pump is engine-mounted, so `drivetrain_graph` places it on
  the block's front flank rather than at the chassis with an electric
  pump. `_emit_mechanical_injection` adds the barrel valve on the hat's
  linkage, fed from the pump and feeding the port rail, and hat nozzles
  (two per butterfly) as connected fuel ports; port nozzles are the
  injector bosses the rail already feeds.
- **Each turbo as real hardware.** Around every `powertrain.
  turbocharger[_N]` point mass: compressor housing (cold, inboard) and
  turbine housing (hot, outboard) on the cartridge, wastegate on the
  turbine, blow-off on the compressor, an up-pipe from the nearest real
  collector, a downpipe rearward; one air-to-air charge cooler ahead of
  the engine with hot-side pipes from every compressor and a cold-side
  pipe to the plenum.
- The tank and an electric pump carry `chassis_side` (with muffler and
  tailpipe); `vehicle_mesh` skips such nodes in the engine view and
  `EnginePackage` classifies them supplied-elsewhere -- one attribute,
  one rule, for everything hung from the body.
- Ordering: `emit_universal_parts` now runs LAST in
  `build_drivetrain_graph`, after the accessory ring and the
  supercharger rotor -- the blower case is built around that rotor, and
  a turbine takes its up-pipe off a real collector.

Verified over all 26 engines (graph + dangling-edge check, mesh build,
crate builds in three modes) and by eye at full detail on the blown
drag V8 and the twin-turbo 632.

Still open in this family: the Wärtsilä's turbos (needs a declared
scavenge boost), the Merlin's two-stage/two-speed blower and aftercooler
circuit, the Wasp's supercharger in its rear accessory case.

## Parts catalogue, stage 4: aircraft, industrial and air-cooled families

- **Lubrication is declared.** `Engine.lubrication` ("auto" | wet-sump |
  dry-sump | splash-bath | drip | total-loss), applied per identity in
  the same finalize loop that assigns `ignition_profile`; the Merlin,
  the Wasp and the GT flat-six are real dry-sump engines the race-fuel
  rule alone never caught, the trimmer is total-loss. `derive_dressing`
  honours it. A dry sump now also gets its oil cooler on the return to
  the tank and a belt drive from the damper to the scavenge pump.
- **Dual ignition.** On `aircraft-dual-magneto` engines `cylinder_ports`
  adds a second plug boss across the bore, and dressing emits two
  magnetos on the rear accessory end (where an aircraft's accessory
  drive is), plug 1 of every cylinder on magneto 1, plug 2 on magneto 2
  -- the real certified redundancy. Merlin: 12 + 12 plugs.
- **Aircraft drive.** No clutch, gearbox, transfer case or bellhousing
  on an aircraft engine: `_emit_aircraft_drive` purges the car chain
  (mounts and all) and adds the propeller reduction gearbox on the
  block's front face, the prop shaft and hub on the crank axis, and the
  constant-speed governor on the gearcase. The dyno couples to
  `powertrain.engine` directly, so the solver sees no change.
- **Governors and pumps.** A hit-and-miss engine's flyball governor
  with its latch-out linkage; a compression-ignition engine's
  cam-driven injection pump on the block flank feeding the rail, the
  lift pump now feeding the injection pump's gallery instead of the
  rail.
- **Air cooling.** The production cooling stack is gated on a water
  pump, so an air-cooled engine had NO cooling hardware; a boxer/inline
  now gets the fan, its housing and a tin over each bank ducting the
  blast across the fins, and a radial gets a baffle per cylinder (from
  the real cylinder sites -- radials fold their heads into one block
  node) and a ring of cowl flaps behind the row.
- **Wet-sump gate completed**: the per-head deck oil feed/return faces
  were still emitted on the total-loss two-stroke; they are gated now.

Verified over all 26 engines (graph + dangling-edge check, mesh build,
crate builds in three modes) and by eye at full detail on the Merlin,
the air-cooled flat-four and the Cat C18.

**Flagged, not decided:** `_emit_supercharger` builds a Roots-style
case in the valley with the throttle on top for ANY supercharger. That
is right for the two blown drag engines and wrong for both aircraft
engines, whose blowers are gear-driven centrifugal stages at the rear
(the Merlin's two-stage, two-speed, with its own aftercooler circuit;
the Wasp's single-stage in the rear accessory case) fed by a pressure
carburettor. `ForcedInduction` has no supercharger TYPE field
(roots / screw / centrifugal) to key on -- a declaration to add, not a
guess to make here. Also still open: the Wärtsilä's turbos (needs a
declared scavenge boost) and whether the charge cooler is chassis-side.

## Parts catalogue, stage 5: the non-piston kinds, and a transverse pair

- **Turbine** (`_emit_turbine_gas_path`): inlet plenum + screen,
  centrifugal impeller + diffuser, annular combustor with six nozzles
  and the light-off igniter + exciter, NGVs + turbine wheel, power
  turbine, exhaust duct, the power-turbine reduction the spec already
  declares, accessory gearbox with starter-generator, bleed valve, FCU
  fed from the rail -- all sized from `TurbineSpec` (design mass flow
  sets the annulus at a disclosed inlet velocity, pressure ratio the
  compressor length).
- **Electric / servo** (`_emit_electric_drive_unit`): motor housing,
  stator, rotor pack (on the shaft), resolver, inverter on top with
  three phase busbars and its coolant plate, single-speed gearset +
  differential (EV) or planetary gearhead + output flange (servo), HV
  contactor box with pre-charge, DC-DC converter to the 12 V fusebox.
- **Expander** (`_emit_expander_plant`): steam -- boiler barrel,
  firebox, steam dome, safety valve, water gauge, regulator feeding the
  existing source/admission chain, chimney with blast nozzle, feedwater
  injector from a chassis-side water tank; compressed air -- two
  reservoirs, charging valve, check valve, gauge, inter-stage reheater.
  Valve gear gets a node on the eccentric `cylinder_ports` already
  draws (`drawn_by`), plus the reverser lever.
- **Atmospheric** (`_emit_atmospheric_mechanism`): rack, pinion and
  freewheel (nodes on drawn geometry), slide valve on its eccentric,
  pilot burner at the flame port on the gas line, the captive-ball
  governor `governor.py` already simulates.
- **Small-engine kit** (`_emit_small_engine_kit`): the trimmer's
  flywheel magneto (coil against a magnet ring cast into the flywheel,
  whose vanes are the fan), spark-arrestor muffler, ON-engine tank with
  primer bulb (a trimmer's tank is bolted to the engine -- `chassis_
  side=False` overrides the car default), diaphragm carburettor with its
  crankcase pulse line; the hit-and-miss engine's water hopper (node on
  the drawn hopper) and igniter trip lever.
- **A 1990 pair, declared like their neighbours**: Toyota Camry 3S-FE
  (2.0 DOHC 16v, transverse) and Mazda Miata B6ZE (1.6 DOHC 16v,
  longitudinal, wasted-spark coil packs -- no distributor), from real
  published figures (bmep from torque and displacement).
- **`Engine.installation`** ("longitudinal" | "transverse") is declared
  hardware. Transverse replaces the production car chain (gearbox,
  transfer case, bypass, their mounts) with a TRANSAXLE off the clutch
  housing -- final drive and differential low in the case, halfshafts
  out both ends PARALLEL to the crank -- and the two mounts a real
  three-point pendulum layout puts on it (upper case mount, lower
  torque rod), in `engine_mounts`' roles and `EnginePackage`'s
  supplied-elsewhere set. All still in the engine's own local frame; the
  vehicle turns the crate on installation.
- Heater core and expansion bottle are body-mounted: `chassis_side`.

Verified over all 28 engines (graph + dangling-edge check, mesh build,
crate builds in three modes) and by eye at full detail on both new
engines.

## Engine-only view bbox leak, and the automated mount/frame system

- **Fixed a real camera-fit bug**: the transaxle/final-drive/differential/
  halfshafts (the Camry's real transverse driveline) were never added
  to `drivetrain_graph.ENGINE_VIEW_EXCLUDE_PREFIXES`, so the engine-only
  mesh view dragged in the halfshafts reaching toward the wheels and
  blew out the camera's auto-fit into a tiny, dark frame. Same real cut
  as the transmission/transfer-case exclusion already there. `bay_view.
  py` had its OWN separate, independently-stale exclude list with the
  same gap -- fixed there too.
- **A second, more general leak**: `vehicle_mesh.py` skipped drawing a
  `chassis_side` NODE's own box, but never checked an EDGE's endpoints
  for `chassis_side` -- so a coolant line into the (correctly hidden)
  expansion bottle still reached out to the bottle's real chassis-side
  position as a long floating tube. Added `_chassis_side_ids()` and gated
  both edge loops in `build_drivetrain_solid_parts` on it. This is the
  actual, general fix; a same-vintage narrower workaround for muffler/
  tailpipe already existed in `engine_mesh.wanted_in_view` by name
  substring and was left alone (harmless, now redundant with the new
  general check for that specific pair).
- **The real automated mount/frame system, actually wired up**:
  `engine_mounts.assign_mounting` (the CG-in-hull/TRA/subframe policy
  already built) and `EnginePackage.correlate_mounts` existed but were
  never called from anywhere real. `bay_view.py` -- the standalone bay-
  packing viewer that already runs the production `fit_vehicle_chassis_
  to_power_unit` fitter -- now calls `assign_mounting` directly in
  `_rebuild()` and draws its real mount points (yellow) against the
  real fitted chassis/firewall, transformed by the SAME rotation/
  translation the engine mesh itself uses. No bespoke per-engine mount
  set was authored; cycling engines or toggling transverse just re-runs
  the same real policy.
- **A real, generic torque-output-to-chassis stub**, also drawn (magenta):
  a transverse install's real halfshaft nodes when they exist; otherwise
  the most-downstream real driveline node that does (`transfer_case`,
  else `direct_drive_bypass`, else bare `transmission`) -- always a real
  existing graph node, never invented geometry, and always exactly one
  marker regardless of install style, per the real "it just needs to be
  a torque node that can function as output to the chassis" requirement.

## Turbo/supercharger mounting: swept ducts instead of fixed positions

- **`duct_routing.py`** (new, dependency-free): `duct_waypoints(a, b,
  rise, forward, forward_amount)` -- a real two-bend "over the top" path
  between two points -- and `lay_routed_pipe(...)` -- lays waypoint
  nodes + chained segment edges along any such path. Pulled out of the
  turbo up-pipe code specifically so it has zero import risk with either
  `dressing.py` or `engine_parts.py` and can be reused by both.
- **Turbo mounting is now derived, not read from a fixed production
  stub**: `_turbo_swoop_points()` builds a real up/forward/back sweep
  off the feeding collector -- lift off the header, arc forward (and up)
  toward the front of the engine, arrive at the turbine inlet from
  above/behind -- and the turbo's own point-mass position IS that
  sweep's arrival point (`t["reference_position"]` is overwritten from
  it), not an independently-guessed spot. The up-pipe itself is now a
  real routed multi-segment duct (`duct_routing.lay_routed_pipe`)
  instead of one straight tube.
- **Fixed a real desync this surfaced**: the turbo's own oil feed/drain
  child ports (`powertrain.turbocharger.oil_feed`/`.oil_drain`, hung by
  the PRODUCTION graph at the turbo's original stub position) never
  moved when the swoop relocated the turbo -- so the housings would
  swoop away while their oil lines stayed pinned to the old spot. Fixed
  by shifting every child node under the turbo's own identity prefix by
  the same delta the swoop applied.
- **`ForcedInduction.turbo_layout`** ("parallel" default | "serial"):
  parallel keeps the existing one-turbo-per-bank z-mirrored cloning;
  serial (compound/staged) skips the mirror entirely -- every unit stays
  on the same side, and `_emit_turbo_hardware` chains each later stage's
  hot feed off the PREVIOUS stage's own turbine outlet instead of the
  collector directly, with `stage` progressively pushing each unit
  further along the same swoop (more forward, higher). Verified on a
  cloned `monster-632-twin-turbo` in both layouts: no dangling edges,
  turbo_2 sits further forward/higher than turbo_1 in serial, both stay
  z-mirrored in parallel.
- **`ForcedInduction.blower_mount`** ("on-block" default | "remote") +
  `blower_position_local`: on-block is the existing valley/manifold-
  plate stack, untouched. Remote drops the manifold plate entirely,
  places the case at a declared or derived position (front-low-to-one-
  side default), and connects the case-to-plenum leg with a real routed
  duct instead of the short direct edge on-block uses -- "pass duct to
  duct" for a blower that doesn't sit on the intake. The throttle-body-
  onto-case ("hat") relocation code needed no changes at all: it was
  already derived off `case_c`, which now just lives somewhere else.
  Verified on a cloned `supercharged-drag-v8-8200` in both modes.
- **`IntakeSystem.air_source`** ("engine-bay" default | "remote-box" |
  "underside-snorkel") + `air_box_position_local` /
  `snorkel_inlet_position_local`: independent of forced induction and
  of `plenum_placement` (that's the chamber under the throttle; this is
  where the FILTER breathes from). `_emit_remote_air_intake` leaves a
  real stub (`powertrain.air_filter.hose_end`) exactly where the filter
  used to sit -- so every existing downstream edge keeps working
  unchanged regardless of throttle-body count -- relocates the real
  `powertrain.air_filter` node to the box position, and ducts between
  them. A snorkel adds one more real node (`powertrain.snorkel_inlet`)
  and a plain straight riser (deliberately not the over-the-top duct
  shape -- a snorkel riser doesn't need to dodge anything). Runs after
  turbo/supercharger emission so it always starts from wherever the
  filter's own position actually settled (on-block "scoop" repositioning
  included). `engine_mesh.py`'s view whitelist and the air-filter
  material rule both extended to cover the new node/edge names.
- All of the above verified against the full 28-engine catalogue
  (graph + dangling-edge check, mesh build, crate build, EnginePackage
  build) with no regressions, in addition to the synthetic parallel/
  serial and on-block/remote clones above.
- Not yet done: a visual render of the swoop came back with the turbos
  present and correctly positioned (verified directly via node
  positions and edge integrity) but not clearly visible in-frame --
  likely camera occlusion/exposure at the azimuths tried, not a data
  bug. Worth a proper look with a deliberately chosen camera angle next
  time renders are being iterated on, but not chased further this pass.

## Compressed-air idle assist: a second real consumer on the air system

- **Real precedent**: Knorr-Bremse's Pneumatic Booster System -- the
  truck's own brake-reservoir air dumped into the intake manifold
  through a ring of large angled nozzles by high-speed solenoids, to
  kill turbo lag / prevent stall under a sudden load. A LARGE-bore
  manifold-port class, not a nitrous-style jet: 20 mm default.
- **Declared hardware** (`PneumaticSystem`): `idle_assist_fitted`,
  `idle_assist_port_diameter_mm` (20), `idle_assist_trip_rpm` (None ->
  `idle_rpm * 1.15` at wiring), plus the downstream air system when
  `brake_system_fitted`: `brake_reservoir_capacity_l` (60, primary +
  secondary) and `protection_valve_pressure_pa` (5.5 bar, real air-
  brake pressure-protection practice). Only the C18 carries it today,
  but nothing is keyed by identity -- any engine declaring a reserve
  tank + the port gets the whole thing.
- **Graph** (`drivetrain_graph.py`): wet tank -> `pneumatic_protection_
  valve` -> `pneumatic_brake_reservoir_primary/secondary` -> `pneumatic_
  treadle_valve` -> `pneumatic_brake_chambers` (reservoirs/treadle/
  chambers `chassis_side`), and `pneumatic_isolation_valve` on the
  protected side. `engine_parts._emit_pneumatic_idle_assist` puts the
  real `powertrain.intake_plenum.compressed_air_port` (a bung, drawn by
  the existing bung renderer) on the plenum and runs the `compressed-
  air-line` from the isolation valve (or straight off the wet tank when
  no brake system is fitted). Every one of these edges is "compressed-
  air-line", so they all join the ONE existing pneumatic-reserve fluid
  circuit: compressor charging, the air-motor starter's draw, the brake
  reservoirs' stored mass, and the dump genuinely share one lumped
  fill.
- **Physics**: `choked_orifice_mass_flow_kg_s` -- the standard choked
  compressible-orifice relation for air (Cd 0.65), evaluated at the
  tank's own current pressure (the same linear fill->pressure relation
  `pneumatic_reserve_pressure_pa` already reports). New `FluidCircuit`
  fields `idle_assist_delivered_kg_s` (kept SEPARATE from `delivered_
  flow_kg_s`, which already means compressor charge rate on this
  circuit) and `protection_pressure_pa`. The dump is only fed while
  tank pressure > protection pressure -- the valve's one real job.
  Verified: 0.47 kg/s at a full 8 bar 80 L + 60 L system, tapering with
  pressure, zero at/below 5.5 bar.
- **Sim side** (`engine_cycle_sim.py`): `pneumatic_idle_assist_enabled`
  (live toggle, armed by default once fitted -- `I` in main_pygame.py);
  the trip gate is armed AND `idle_rpm * 0.4 < rpm < trip_rpm` AND not
  stalled. The lower floor is a real interlock, not decoration: without
  it the dump emptied the reservoir into a 0 rpm engine before the
  starter was engaged. Delivered mass becomes a REAL boost fraction --
  the ratio of dumped flow to the engine's own breathing demand this
  tick (unlike nitrous's tuned constant, this is literally gas into the
  manifold), capped at 0.5 atm-equivalent, added into `raw_map_frac`
  beside boost/nitrous. New state: `pneumatic_idle_assist_boost_frac`,
  `pneumatic_idle_assist_delivered_kg_s`.
- **Verified end to end** on the C18: idles ~699 (trip 690); a 1.2x
  peak-torque brake load sags it to 690, the dump fires (0.36 kg/s,
  boost hits the 0.5 cap), rpm recovers to 694-700, the dump shuts
  off. Real consequence kept: at these rates the system is a seconds-
  long burst device, exactly as PBS is -- the tank, not the port, is
  the limit.
- Only the piston-engine `_drivetrain.step` call site passes the new
  args -- same as `starting_air_kg_s` already; no non-piston engine
  declares an air system.

## Compressed-air idle assist (industrial engines)

- `PneumaticSystem.idle_assist_fitted / idle_assist_port_diameter_mm /
  idle_assist_trip_rpm`: a real second consumer on the SAME reserve
  tank the compressor charges (and the C18's air-motor starter draws
  from) -- a fixed-diameter dump port straight into the intake plenum
  (`engine_parts._emit_pneumatic_idle_assist`, a real bung + a
  `compressed-air-line` edge that joins the existing pneumatic
  circuit). Declared on the C18 only, since it's the one engine with a
  real reserve tank that isn't a ship's starting-air receiver.
- Flow is real choked compressible orifice flow (`drivetrain_graph.
  choked_orifice_mass_flow_kg_s`, air, gamma 1.4) at whatever pressure
  the tank actually holds, drained from the circuit's own fill
  fraction every tick, read back as `pneumatic_idle_assist_delivered_
  kg_s` and turned into MAP the physical way: the ratio of what the
  port delivered to what the engine itself is demanding to breathe
  (capped at 0.5 atm), not a tuned constant like nitrous.
- The trip decision lives on the sim (`EngineCycleSim.pneumatic_idle_
  assist_enabled`, the `I` key in main_pygame): armed AND below the
  trip rpm (declared, else idle*1.15) AND actually turning.

## Live thermal self-emission, and the combustion kernel

- **The window bug**: `EngineGLView._draw` set `glViewport` to its own
  620x620 FBO and never restored it, so every later composite (the
  engine quad, the legend) rasterised into the window's bottom-left
  corner whatever x/y the compositor was given. It now saves and
  restores the caller's viewport.
- **Blackbody glow is live, per part, per tick** -- diagnostic, not a
  bake. `engine_mesh.blackbody_emission_rgb` is Planck's law sampled
  at three visible wavelengths; the colour and the relative brightness
  across temperature are real, the ONE disclosed calibration is
  `BLACKBODY_SCENE_GAIN` (red-band radiance at 1000 K = 0.08 in the
  rig's non-physical light units: dull red at 1000 K, plain orange at
  1150, yellow into white past ~1300). `EngineMesh.part_groups` keeps
  each part's real thermal circuit (`block_cyl_N`, `exhaust`,
  `coolant`, `oil`, `intake`; pistons now belong to their cylinder's
  group, the turbo hot side to exhaust). `engine_gl_view.
  _ThermalMaterialDB` gives every (material, group) pair present its
  own material row and rewrites those rows' emission from
  `thermal_groups_from_state(sim.state)` each tick -- so a bay whose
  coolant stopped flowing climbs through cherry red to white exactly
  as its own `cylinder_block_temps_k` does. Hot part clusters are also
  real point emitters (per group, per engine side) lighting their
  neighbours, sqrt-compressed so a header doesn't flood the bay.
- **`combustion_kernel.py`**: each cylinder's burn as light, baked in
  VOLUME: per burn phase the kernel is the real chamber between the
  crown (the same crank kinematics cylinder_ports places the piston
  with, at the crank angle that phase occurs -- TDC alignment with
  the sim's own `_firing_angle_deg` verified: theta == 0 at fire on
  every cylinder) and the head face, growing from an igniter core to
  the full bore by the cube root of burnt fraction. Colour is COMPOSED:
  Planck at the fuel family's flame temperature weighted by its soot
  luminosity, plus a clean flame's blue chemiluminescence
  (`CombustionVisual`, per fuel family, overridable per engine as
  `Engine.combustion_visual`). Residue is declared per family (diesel
  black soot 0.55, two-stroke blue-white oil smoke, methanol near
  nothing, hydrogen none) and puffs from the cylinder's REAL exhaust
  PortSpec during its exhaust stroke. Which frame each cylinder shows
  comes from the sim's own firing bookkeeping every tick
  (`combustion_state_from_sim`: degrees since `_last_fire_total_deg`,
  the real `_last_strength` -- a cut, a misfire, a knock-weakened burn
  all show as themselves). Every live flame is a point emitter: the
  burn lights the crown and bore from inside (verified by pixel diff:
  ~93k pixels change when one cylinder fires, the crankcase/pan lit
  warm). At TDC the kernel is honestly a ~7 mm disc -- the LIGHT it
  throws is the visible thing, as in a real engine.
- Next (agreed, not started): a per-part UV temperature texture with
  its own surface-conduction kernel, using the lumped group
  temperatures above as its boundary conditions -- the shader already
  reads a per-material emissive UV layer (`uEmitUv`, texstack
  emit_uv_layer/emit_gain); the meshes just need a real atlas (tubes ->
  angle x length, cuboids -> six faces, prisms) instead of zero UVs.

## The cat as a real part, what leaves the pipe, and the air the engine lives in

- **`engines.CatalyticConverter`**: substrate ~0.8x displacement,
  a real precious-metal loading (three-way: Pt/Pd/Rh; a diesel
  oxidation cat: Pt-heavy, no Rh; pre-1996 bricks Pt-rich) with a
  dated, disclosed scrap value -- the "free platinum". `ExhaustSystem.
  catalyst_fitted` pulls it (the `M` key): its segment leaves the pipe,
  so its restriction, its sound damping (`brightness_frac`), its
  conversion and its scrap value all go together. Backpressure was
  never actually applied from the declared exhaust system before --
  the circuit's choke capacity came only off the primaries -- so
  `static_backpressure_frac` (now a series SUM over a fixed reference
  count, not a live mean) is passed into `DrivetrainSolver.step` as
  `exhaust_system_restriction_frac` and narrows the exhaust circuit's
  real flow capacity. Pulling the cat is a real, small power gain; it
  also lowers every engine's baseline slightly now that the cans count.
- **`emissions.py`**: engine-out CO/HC/NOx from the live mixture
  (`state.mixture_phi`: metering relative to stoich plus power
  enrichment; a diesel's rack as a lean phi), load and exhaust mass
  flow -- textbook Heywood-shaped trends, not tuned numbers. `Catalyst
  State`: a brick with real thermal inertia that must light off (~520
  K, sigmoid) and only converts inside the lambda window (rich starves
  oxidation, lean starves reduction), fed by the collector-outlet gas
  temperature the segment chain already computes, warmed by its own
  oxidation exotherm. `Occupant`: carboxyhaemoglobin by the Stewart
  relation (%COHb/min = 3.317e-5 * ppm^1.036 * RMV), 320 min elimination
  half-life, the standard symptom ladder, and a conservative
  minutes-to-lethal at the current ppm.
- **`air_volumes.py`**: ambient -> garage/dyno cell (optional: `J`
  cycles outdoors / door open / closed / sealed) -> engine bay, each a
  well-mixed volume with temperature, CO and oxygen. The bay is
  ventilated by the cooling fan's own real delivered flow (`cooling_
  fan_flow_m3_s`, every fan's flow coefficient x live shaft speed, new
  solver output) through the grille plus ram air with speed; the room
  by its door state, leaks, and a cell fan. The engine dumps its block/
  radiator heat share into the bay, exhausts into wherever the pipe
  ENDS (the room for a tailpipe, the bay itself for an open header),
  and draws its intake from wherever the FILTER is: the bay for an
  under-hood filter, the room/outside for a cold-air box or snorkel --
  which is now the real reason `IntakeSystem.air_source` matters
  (hot bay air = denser-charge loss; oxygen-poor air = real power loss,
  `intake_o2_factor` on the charge). The exhaust extractor (`Q`) is the
  motorsport answer for a closed cell -- the big orange high-temp duct
  over the tailpipe -- modelled as a capture fraction vented outside;
  a cell fan (`cell_fan_m3_s`) covers what it misses.
- Dashboard: CAT / EXHAUST / AIR / ROOM lines in `toy_shared.
  dashboard_lines`.

## Animation schedule and the flame lights

- The combustion kernel's point emitters are OFF by default
  (`EngineGLView.flame_lights_enabled`): a tiny light fully enclosed in
  a bore is exactly what a Phong point light does worst (it shines
  straight through the walls that should trap it), and it made that
  failure obvious. The kernel's own surface self-emission stays.
- `animation_divisions="dense"` (the default now, main_pygame too):
  `engine_mesh.significant_crank_angles` builds the baked crank
  schedule from the engine's own ignition schedule -- the firing
  interval (720 over the number of distinct firing angles in the
  layout) divided into the fewest equal steps no coarser than 5
  degrees, so the grid is COMMENSURATE with the firing order: every
  cylinder's dead centres, valve open / peak / close (valvetrain_parts'
  own 0.32-of-cycle lift trace off the throw phase) and firing TDC land
  on the same relative frame phase (a V8's 90-degree interval -> 5
  degrees, 144 frames; a 14-cylinder's 51.43 -> 4.675, 154). Any event
  further than half a step from a grid frame is added explicitly; on a
  commensurate grid that is none of them. Bake cost is the existing
  incremental baker (150-1000 ms/frame by engine size), memory 0.3-1.7
  MB/frame -- fine for a user baking their own engine once.

## Audio: smooth inputs, continuance, and real event sounds

- **Threading**: synthesis already ran on its own `audio-synth` thread
  with the PortAudio callback only copying from a ring buffer. The
  catch-up artefact was upstream of that: every block read the LAST
  60 Hz tick's raw values, so pitch moved in tick-rate stairs and a
  late tick was followed by a jump. `AudioStreamer._smoothed` now slews
  every continuous control (rpm 45 ms, fire-rate 45 ms, throttle 60,
  load 80, boost 90) toward the latest snapshot per block; a stale
  snapshot simply holds (state continuance -- a paused or slow sim keeps
  sounding as it last was); a new engine re-seeds instead of gliding.
  Inside the synth the firing frequency is ramped PER SAMPLE across a
  block (phase = cumsum of a per-sample increment), so pitch is
  continuous through the block, not stepped at its edge (measured:
  block-edge sample jumps now smaller than in-block ones). Lookahead
  0.35 s.
- **Knock ring = the chamber's own modes, not a chord.** The fixed
  3400/5200 Hz pair is gone. `engine_sound.knock_ring_modes_hz`: Draper
  modes f = c*rho/(pi*B), rho = 1.841, 3.054, 3.832, 4.201, with the
  bore from the architecture (or displacement/cylinders at square) and
  c from the hot charge (900 K + load); the Camry rings ~4.7-6 kHz
  first mode, a 128 mm big-block ~3.4 kHz, the Waertsilae's 1.15 m bore
  ~370 Hz. Intake Helmholtz and exhaust quarter-wave are real too but
  belong to the induction roar and the firing-note/backfire, not knock.
- **Backfire by kind.** An exhaust event (`unplanned`/`planned`) rings
  the exhaust pipe's own quarter-wave `tuned_frequency_hz` at the live
  gas temperature with its odd harmonics (a closed-open pipe), the
  crack scaled by `brightness_frac` (a muffler eats the crack, leaves
  the boom). An `intake-flashback` is not an exhaust event: it rings
  the intake's Helmholtz (`IntakeSystem.tuned_frequency_hz`) as a low
  whoof with a low-passed gust, in the BAY channel, no crack (measured:
  bay energy at the Helmholtz x4.4, header untouched).
- **Misfire = one missing pulse.** The header note is a periodic pulse
  train, so a misfire removes exactly one firing period with a raised-
  cosine window (95 % deep) instead of an 85 % dip over an arbitrary
  20 ms, plus a soft low chuff for the unburnt charge going out the
  exhaust valve.
- **Pre-ignition is now a real sim event** (`IgnitionEvent.preignition`,
  `state.preignition_flag/intensity/count`): surface ignition before
  the spark off a real hot spot -- valve_state's own exhaust-valve
  carbon (`cylinder_hotspot_risk`) -- scaled by load and charge
  temperature (`PREIGNITION_*` constants). It starts the burn early
  against compression: 60 % of the work, a violent chamber ring
  (counted as knock, no second knock on top), and a real heat pulse
  into that cylinder's own bay temperature -- the runaway path to a
  holed piston is there (12 events in 4 s at 90 % throttle with 0.9
  carbon, +8-16 K per bay). Sound: the bore modes again but the lowest
  dominant, longer (18 ms), with a real low structure thud. Logged as
  "PRE-IGNITION (hot spot) -- piston heating".
- Not yet: compression-release (Jake) braking as its own part with
  per-cylinder exhaust-valve actuation (an "advanced port type"
  alongside VVT / deactivation / decompression levers); a declared
  `CombustionChamber` (kind, valve angle, plug count/offset, squish
  area/clearance, S/V, crevice, crown dome) feeding knock rate, burn
  window, heat split, breathing cap, HC, ring modes and the head mesh.

## Compression-release brake: an advanced port type

- `cylinder_ports.PortSpec.actuation` is the hook for anything beyond a
  plain cam-driven valve: "cam", "compression-release", "none" (a wall
  port or boss). `Engine.compression_release_brake` (declared on the
  C18 and the LDT-465, real Jacobs-equipped units) marks every exhaust
  valve port "compression-release"; `engine_parts._emit_compression_
  brake` then emits the real parts -- a slave-piston actuator over each
  such valve and one housing per bank spanning them, oil-fed off the
  pump, its own material and view entry.
- Torque: T = MEP * V / cycle_rad with a real ~0.9 MPa retarding MEP
  scaled with speed, interlocked to zero fuel and above 1.2x idle
  (`COMPRESSION_RELEASE_*`), added to the existing brake_component; the
  `B` key now cycles off -> exhaust brake -> + compression release ->
  off. Sound: the bark -- no combustion, a bright staccato high-passed
  release burst gated once per firing period over the closed-throttle
  note. Verified: C18 1296 Nm retarding, 2200 -> 690 rpm in 2 s
  against 1850 without.
- The same `actuation` field is where VVT, cylinder deactivation and a
  hand-start decompression lever go next.

## Plosives: the exhaust event and the detonation front, live and baked

- The airborne exhaust event is not the combustion ring (the block
  carries that) -- it is the exhaust valve cracking open on a still-
  pressurised cylinder: a one-sided pressure step (the PLOSIVE, a
  single ~1.4 ms quarter-wave click) followed by the turbulent jet
  past the seat. Both paths now have it. `mech_parts.Source` grew
  `plosive_tau_s/plosive_amp` and `noise_*` fields; the crank
  combustion part emits a `cylN-blowdown` airborne source at each
  cylinder's EVO (fire + 180 deg four-stroke / +100 two-stroke), and
  `bake_points` renders click + seeded high-passed noise so the bake
  loop stays seamless. The live synth gates the same two things off
  the per-sample firing phase, so they stay locked to the note.
- **The jet's envelope is the real valve motion**: `mech_parts.
  blowdown_envelope(theta)` = the valvetrain's own half-sine lift over
  0.32 of the cycle (the curtain area) times the cylinder pressure
  collapsing over ~55 crank degrees. Because a valve stays open far
  longer than one firing interval, the live synth sums this cylinder's
  envelope with the previous cylinder's still-decaying one, one
  interval apart -- the same overlap a real multi-cylinder exhaust has.
- Detonation gets its own plosive: at knock onset the front hitting
  the wall is a pressure step BEFORE the ring -- a first-sample click
  plus ~1 ms of high-passed burst, then the bore modes (first-ms rms
  measured 2.3x the plain note).
- Verified on the I4, V8, two-stroke and single; bake finite and
  seamless; catalogue regression (with the Jake armed everywhere) ALL
  28 OK.

## The cylinder head's interior: `CombustionChamber`

- Seven declared numbers plus a kind (`engines.CombustionChamber`,
  presets for hemi / pent-roof / wedge / bathtub / heron / flathead /
  open, per-engine kinds in `_CHAMBER`, overridable field by field):
  included valve angle, plug count and offset, squish area and
  clearance, surface-to-volume, crevice volume, crown dome. Everything
  derived is relative to a central-plug hemi = 1.0.
- What each drives, now wired: `flame_path_rel` (the front crosses the
  chamber's REAL farthest path -- a flathead's side plug nearly a whole
  bore) and `burn_speed_factor` (squish turbulence) in the burn window;
  `knock_factor` on the knock rate (a flathead 2.1x, a big-block wedge
  1.6x, a four-valve pent-roof 0.93x); `efficiency_factor` (wall loss
  by S/V) and `breathing_factor` (valve size the included angle allows)
  on the cylinder charge; `hc_factor` (crevices) into emissions.engine_
  out; `ring_mode_factor` on the knock ring (a hemi rings lower than
  the pancake estimate); and the head mesh cuts the real roof profile
  per cylinder from the declared kind (`head_mesh._chamber_profile`,
  drawn off the graph's serialised `combustion_chamber` so the renderer
  needs no Engine object) -- a dome, a ridge, a slant, a trough, or
  nothing for a flat Heron/flathead deck.
- Not modelled: multi-plug timing offsets, the dome's effect on the
  combustion-kernel visual origin, chamber-specific swirl/tumble.

## Thermal emission on the GPU: kelvins in an SSBO, Planck in the shader

- The lumped/hot-spot glow no longer goes through Python-written
  material rows or a texture. `base_material.frag.glsl` (the shared
  spectral-analyzer shader) gained `ThermalChunk` (SSBO binding 16: one
  float of kelvin per thermal id), `uThermalEmission / uThermalGain /
  uThermalRefK / uThermalFloorK`, and a three-band Planck evaluated
  per fragment, indexed by the vertex's existing `aGroupId` (flat, so
  per triangle -- "where triangle could be fine"). Off by default, so
  the analyzer's other users see identical output. `BaseGLRenderer.
  set_thermal_emission(...)` / `set_thermal_kelvins(array)` (a
  glBufferSubData once allocated).
- `EngineGLView.thermal_mode = "ssbo"` (default): every part with a
  thermal group gets its own thermal id at upload (`_thermal_id_
  column`); a tick is `_upload_thermal_kelvins` -- one float per id =
  its group temperature plus, for the hot-spot parts (`_hotspot_parts`:
  each exhaust port's bore top and chamber roof, each head casting, a
  turbine housing), a real excess scaled by that cylinder's live burn.
  Measured 0.19-0.31 ms per tick on the I6 / V8 / 14-cylinder (was
  6-31 ms in the per-tile Python kernel). `thermal_mode = "rows"` keeps
  the older per-(material, group) emission rows; the material rows now
  carry only base look plus the explicit flame/smoke overrides.
- `thermal_texture_enabled` (default False): the `thermal_atlas.py`
  per-part temperature TEXTURE with its surface-conduction kernel and
  the renderer's emissive UV layer -- kept as an opt-in close-up mode
  (a Python kernel plus a texture-array upload per few ticks); the
  atlas' own per-tile loops were vectorised over a stacked
  (tiles, 16, 16) array but it is still the wrong layer for every-tick
  use. Per-VERTEX gradients on the SSBO path would need a non-flat id
  or a per-vertex kelvin buffer -- a later step.

## Blower type: Roots / twin-screw / centrifugal

- `ForcedInduction.blower_type` ("roots" default, "twin-screw",
  "centrifugal"), `impeller_blades`, `stages`. The Merlin is declared
  as its real two-stage gear-driven centrifugal (14 blades, 7:1), the
  Wasp a single-stage (12 blades, 8:1); the drag V8s keep their Roots
  cases.
- Physics: a positive-displacement blower's boost tracks shaft speed
  linearly (unchanged); a centrifugal's pressure rise goes with tip
  speed SQUARED, so boost = max * (speed / rated)^2 with rated = crank
  at power_peak_rpm through the gears -- the Merlin makes 0.024 at 630
  rpm and 0.58 at 3090 (verified against the law to 3 decimals), and
  it surges like a turbo when the throttle shuts on it at speed.
- Parts (`engine_parts._emit_centrifugal_blower`): never a case on the
  manifold -- one volute drum per stage at the REAR of the block off a
  gear housing on the crank rear (the wheelcase blower a Merlin and a
  radial actually carry), the production rotor node becoming the
  first-stage impeller, extra impellers on the same shaft for later
  stages, the throttle/carburettor moved to the blower INLET (a
  Merlin's updraft carb genuinely feeds the supercharger, draw-through)
  and a routed outlet duct to the plenum. No burst panel, snout or
  belt pulley (those are Roots hardware).
- Sound: the whine is the blade-pass order (drive ratio x blades: the
  Merlin's 98th order, ~4 kHz at 2500 rpm) instead of the lobe order
  (the drag V8's 13th, ~530 Hz), louder with boost, with the turbo's
  surge flutter when surging.

## Audio: rpm-invariant events, edge-triggered flags, the knock rattle

- **The explosion does not change pitch with rpm; it just comes faster.**
  Each blowdown's spectrum is fixed geometry, so the header harmonic
  stack is now weighted at each harmonic's ABSOLUTE frequency by the
  exhaust's fixed response -- the pipe's odd quarter-wave modes (a
  Lorentzian boost at each) under a muffler's fixed low-pass corner
  (1.2 kHz boxed, ~6 kHz open header) -- instead of by harmonic index,
  so where the energy sits no longer slides up with rpm (spectral
  centroid measured flat at 1500 / 3000 / 6000 rpm). The plosive click
  is now ~1.4 ms of real time at any rpm; the blowdown jet still runs
  its real crank-degree length (which does shorten in seconds).
- **The buzzer was a retrigger bug**: a knock/misfire/pre-ignition flag
  holds for one 16.7 ms physics tick and every 11.6 ms audio block was
  re-arming a short ring while it held. All three are rising-edge
  triggered now; knock additionally re-fires at most once per firing
  event while the flag holds (probability with intensity), each hit
  with fresh random bore-mode phases, 3 ms decay, half-mixed with a
  matching high-passed noise burst -- sparse metallic pings, marbles
  on iron, never a tone. There is no separate "knock sensor" sound;
  that buzz WAS the ring.
- **The Waertsilae's turbochargers are real now**: production gates
  its turbo node on a wet sump, which a crosshead two-stroke doesn't
  have, so `build_drivetrain_graph` emits the same node/edge set
  itself (four units, `turbo_count=4`, ~3.5 bar absolute scavenge:
  `max_boost_frac=2.5`, 4 s spool, no wastegate) -- four volutes with
  swept up-pipes and a charge cooler; boost spools to 2.3 in 20 s at
  ~100 rpm. The `_FORCED_INDUCTION` table now also applies to engines
  built directly rather than through the catalogue dicts. Gap: the
  engine has no lube circuit at all in the graph (`_WET_SUMP_ONLY`
  purges the pump for every two-stroke, and dressing's dry-sump needs
  that pump), so the turbo bearing lines are not emitted yet -- a real
  crosshead lube system (electric pumps, a separate tank, coolers) is
  the next item.

## Lube kinds as ports, and a pitch that follows the crank

- `assembly_ports.lube_kind_for(engine)`: "wet-sump" (pan, dipstick,
  galleries), "dry-sump" (galleries, feed/return faces, fill, a
  `crankcase.scavenge_drain` port -- no pan, no dipstick: the tank has
  those), "total-loss" (a breather and nothing else). `part_ports`
  takes `lube=` (the old `wet_sump=` still works); drivetrain_graph
  and head_mesh pass the engine's own kind. For a dry-sump engine that
  production gave no pump (a crosshead two-stroke: production keys its
  whole oil circuit on a wet sump) drivetrain_graph emits the SAME
  pump node/edge set production does -- engine-driven attached pump,
  gallery feed with its relief valve, heat-exchange share -- with the
  suction off the scavenge drain until dressing's dry-sump block re-
  points it at the reserve tank it adds (dressing's own scavenge line
  now pulls from the drain port when there is no pan). engine_parts
  wires every turbo's bearing feed/drain to that pump/tank when
  production didn't. The Waertsilae now carries pump, tank, scavenge
  pump, cooler and eight turbo oil lines and runs 3.7 L/min of oil;
  the Merlin/Wasp/GT dry sumps gained their scavenge drain; wet-sump
  and total-loss engines are unchanged.
- ~~Pitch is the crank's~~ -- superseded the same day, see the next
  section: pitch is nobody's but the spinning parts'.

## Sound from parts: pitch only where something spins, events only where something happens

The user's rule, verbatim: "PITCH IS NOT TO BE DETERMINED BY ANYTHING
BUT RESONANCE, RPM DOES NOT CHANGE THE PITCH OF DETONATION ONLY THE
FREQUENCY OF ITS OCCURRENCE ... pitch plays into several parts but not
detonation ... every one of them needs to be matched down to the
tiniest part that can be said to cause it." The additive harmonic stack
at the firing frequency (and the crank-rate variant that replaced it
for an afternoon) is gone.

- `sound_parts.py` -- the catalogue of what makes sound, from the
  graph. `build_pitched_catalogue(engine, graph)` walks the rotation-
  carrying edges (`constraint` accessory-drive-belt / geared-timing-
  drive / rigid-keyed-hub ...) outward from the crank multiplying each
  edge's real `ratio`, and gives every rotating-mass node its pass
  count: fan blades (5/7), pump vanes (6/8), gerotor lobes (4/6),
  alternator pole-pairs (6, x3 phases), blower lobes x2 rotors or
  impeller blades, cam sprocket teeth (40 at 0.5), a reduction pinion,
  a gear scavenge stage. Each becomes a `mech_parts.RotaryTonePart`
  (the same class the offline baker uses) with order = ratio x count;
  belt-driven ones get two strands 4 cents apart. A turbo is pitched
  off ITS shaft (spool x a rated speed from the wheel's inertia: 150k
  rpm on a car unit, 17k on the Waertsilae's), never the crank.
  `describe()` is the audit: "electrical.alternator: 6 pole-pairs x
  ratio 2.6 = order 15.6".
- Fixed event kernels (`sound_parts.pipe_response_kernel` etc.): the
  exhaust pipe's odd quarter-wave modes at f0 from the live gas
  temperature with the pipe's own Q (open header ~6, muffled ~2.5); the
  block ring (320 Hz at a 96.5 mm bore, 1/sqrt(bore)); a valve-seat
  clack; an injector-plunger tick; the 1.4 ms plosive; the chamber's
  Draper bore modes in sine and quadrature (so each knock hit gets its
  own random phase as a two-weight mix).
- `engine_sound.py` keeps its OWN crank clock, integrated per sample
  from the smoothed rpm, and fires each cylinder's events when it
  crosses that slot's angle. What it fires is decided by the sim's
  per-slot record of what that cylinder actually did on its last pass
  (`EngineCycleState.slot_records`: strength, misfire, knock intensity,
  pre-ignition; `slot_angles_deg`; `exhaust_valve_held_open` for the
  hit-and-miss governor; `fire_event_count` for the free piston): no
  burn, no combustion sound. The exhaust port's blowdown amplitude is
  the port's own pressure -- a burn, a motored compression (0.12 x
  MAP), a held-open valve (nothing), a compression-release crack at TDC
  (1.1 x MAP, its own 2 ms crack + 6 ms release). Valve seats close
  twice per cylinder per cycle; the intake roar is the sum of every
  open port's raised-cosine window; the blowdown's turbulent jet is one
  noise stream shaped by (event train * the real valve-lift envelope at
  the current crank speed) -- the only place rpm enters an event: how
  many seconds its crank degrees take.
- Vectorised, as asked: every event class is an impulse train
  convolved with its kernel through cached FFTs (`_Convolver`,
  overlap-added by `_Tail`), every pitched voice is a row of one
  `_SineBank` matrix op with per-sample frequency ramps. Measured
  (`audio_check.py`): header spectral centroid ~3 kHz and moving < 10 %
  when the event rate doubles (it was 8 kHz and rpm-bound before the
  two-pole muffle); all-cut header 0.05 rms vs 0.29 fired; held-open
  0.006; ~3 ms per 11.6 ms block warm (the first engine in a process
  pays 50-100 ms of warm-up once). `LiveAudioState.extras` carries the
  slot records, held-open flag, fire count, MAP and the graph to the
  synth thread; `throttle_program.py`'s offline callers still work
  (every slot burning at the throttle's charge, a fallback catalogue
  from the declared accessories).

## The damage system's own sounds, and emitters on the holes

- `damage_sound.py` (`DamageSoundSynth`): every `ImpactResult` the ray
  applied is queued by `EngineCycleSim.apply_penetration` into
  `sim.damage_events` (drained through `LiveAudioState.take_damage`,
  mixed in `AudioStreamer._run`, bay + a quarter into the header). The
  struck material and wall thickness pick the ring (a cover clangs at
  1.8 kHz for 40 ms, a case bells at 900 Hz for 140 ms); the damage
  mode adds the event: dent = thud, crater/spall = crack + click,
  puncture = click + ring + a pop if the part was pressurised, ricochet
  = click + the zing (a descending Strouhal chirp of the tumbling
  projectile). `blast(energy, volume)`: a low boom sized by the energy
  at the bay's own mode, a crack, debris rattle. Successive holes of
  one shot land ~1 ms apart.
- `hole_emitters.py`: a puncture is geometry; an emitter is the physics
  on it. `HoleEmitterField` on `sim.hole_emitters`, one emitter per
  boundary hole (entry, and exit when it went through), stepped every
  physics tick against the live `fluid_circuits`. Regimes decided from
  the hole and the pressure: spray (Bernoulli, v = sqrt(2 dp/rho),
  capped by what the pump delivers -- above that the line pressure
  collapses to what drives exactly the pump's flow through the total
  hole area), pour (Torricelli on the gravity head, shrinking with the
  level), drip (below 2 ml/s, Tate's-law drops counted per tick), gas
  (choked orifice, the pneumatic system's own relation), and INGEST --
  the negative emitter: a volume below outside pressure (a plenum at
  part throttle, a suction line) draws bay air in through the same
  orifice law reversed, plus any other emitter's jet whose particles
  pass within 0.25 m (captured by the hole's solid angle). Every
  circuit's contents are a `FluidMix` (mass per fluid: nominal, air,
  oil, coolant, fuel, particulate); leaks remove proportionally, ingest
  adds, gas circuits are flow-through. `lost_l` and `fouling()` are the
  hooks for consequences (oil starvation, coolant loss, a leaned
  charge) -- exposed on the dashboard (LEAK/LOST/FOUL lines), NOT yet
  closed back into the thermal/mixture steps.
- Flow character, per the user: every stream judges itself laminar or
  turbulent by the fluid's own numbers -- the liquid Reynolds number
  (rho v d / mu, viscosity per fluid: warm oil 0.045 Pa s, coolant
  0.0012) for laminar/turbulent, the GAS Weber number (rho_air v^2 d /
  sigma, the Ohnesorge-Reitz map) for column / wavy / droplets / mist.
  A coolant pour at 1 m/s is a turbulent Rayleigh column; oil off a
  pump line at 3.5 m/s is a laminar wavy stream shedding drops; a
  dipper at 13 m/s makes mist. The character sets the sound (a laminar
  rope is nearly silent, a mist hisses) and the cloud's cone.
- Splash emitters: the `oil-splash-path` edges dressing.py declares
  (pan/trough -> every bore bottom) are no longer drawn as pipes
  (vehicle_mesh skips them); `HoleEmitterField.add_splash_from_graph`
  puts a dipper emitter at the bottom of each cylinder's own crank
  throw (layout: crank centre, throw radius), aimed at its bore bottom,
  flinging the swept slice of the bath per revolution at the throw's
  tip speed (nothing below a quarter level). They emit TO A PLACE: the
  arc ends at the bore bottom, and the retained film fraction is what
  `crankcase_state.step(deposit_kg_s=...)` now takes per cylinder in
  place of its uniform crank-speed law (calibrated to the same 0.35
  mg/s at 2000 rpm).
- Particles: `HoleEmitter.droplets(rng, n)` samples a cloud (random
  age along the arc, random direction in the character's cone, gravity
  for liquids, an entraining plume for gas, converging streamlines for
  ingest); `EngineGLView.set_emitter_particles` builds all of them as
  one motion-blurred streak soup (`_streak_soup`, vectorised) in the
  fluid's own `leak_*` material -- a few hundred droplets per stream,
  6000 per frame budget, one draw. main_pygame re-uploads every tick.
- `burst.py`: part burst and absence. `sim.burst_part(identity,
  energy_j=None)` -- energy from what it holds (fuel: the vapour cloud
  x LHV x a 2 % open-air deflagration fraction; a gas vessel: its p V
  ln(p/p_atm); else a bare 1.5 kJ let-go). The casing becomes 12-90
  fragments (log-normal masses, 60 % of the mass) and the contents a
  droplet cloud; speeds from the Gurney-style equal share of the
  energy fraction (35 % solids, 15 % fluid); directions sampled over
  the sphere and RE-WEIGHTED BY THE DENSITY OF THE SURROUNDING SPACE
  (`medium_density_at`: bay air 1.2, a neighbouring fluid volume its
  fluid's density, a casting its material's -- weight rho_air/rho,
  launch speed x sqrt of the same). Then every fragment flies under
  gravity with quadratic drag in whatever it is inside of at each
  step; a casting wall stops it, the bay floor settles it. Measured: a
  1 kg iron part at 3 kJ throws 77 m/s chips tens of metres in open
  air, 3 m/s and under a metre submerged in oil; with a casting wall
  5 cm away no fragment launches into it. The part is ABSENT from then
  on (`state.absent_parts`; `EngineGLView.set_absent_parts` re-uploads
  the static mesh without its triangles), its fluid leaves the circuit,
  and an open end stays on that circuit as a hole emitter. The real
  trigger: fuel spraying from a punctured fuel part whose stream
  reaches an exhaust part above the fuel's autoignition (520 K) for
  0.6 s lights, and the fuel part deflagrates (`_check_burst_triggers`).
- `snapshot.py`: state -> picture, no controls. `python snapshot.py
  <engine> --rpm 5200 --throttle 1 --run 4 --shoot <node|edge> --burst
  <node> --after 0.5 --azimuth 1.1 --out x.png` builds the real sim,
  drives it there, shoots/bursts through the app's own path, runs
  `--after` seconds past the last event, renders one frame of the same
  view the app shows (thermal, live combustion, cutouts, droplets,
  absence) and prints the state (rpm, slot strengths, LEAK/LOST/FOUL,
  bursts, emitters with their character). Regression `regress10.py`
  (scratch): all 28 engines through graph -> sim -> streamer render with
  extras -> catalogue -> a shot -> emitters, ALL 28 OK.

## Cartridges, a firing squad, and shrapnel that cascades

- `calibres.py` -- real cartridges as real projectiles for
  `ballistics.ProjectileState`: mass, diameter, muzzle speed, length
  and the CORE's hardness, which is what actually decides whether a
  round crosses a casting (a soft lead .45 mushrooms and stops; an M61
  AP core at 2.3 GPa keeps going). .22 LR, 9 mm, .45 ACP, .357, .500
  S&W, 12 ga slug, 5.56, 5.8x42, 7.62 NATO, 7.62x39, .308 AP, .338
  Lapua, .50 BMG and .50 AP, with aliases ("colt 45", "browning 50",
  "582"). Speed falls with range on a disclosed drag law.
- `FiringSquad` is an itinerary: an ordered list of `Shot`s, each a
  calibre, a target (a graph node OR an edge -- a pipe is a target
  too), how many rounds, the group's spread, where the shooter stands
  and how long the engine runs before that volley. `execute` fires
  every round through the same path a right-click uses
  (`RayMesh.penetrate` -> `apply_penetration`), re-deriving the mesh
  after any dwell because the crank has moved. `snapshot.py --squad
  "calibre:target:rounds:spread[:dwell[:dx,dy,dz]]"`, repeatable.
- SHRAPNEL CASCADES. `EngineCycleSim._cascade_fragments`: when a part
  bursts, its most energetic casing fragments (>= 25 J, the top 12)
  each become a real tumbling projectile -- its own mass and size from
  the fragment, a random yaw and tumble rate, the casing material's
  hardness -- and are fired through the projectile engine at whatever
  is in their way. Depth-limited to three levels so a cascade
  terminates. Measured on the C18: bursting the 8.3 bar reserve tank
  threw 490 fragments at up to 288 m/s, and a 267 g piece at 96 m/s
  dented the charge cooler.
- VESSELS RUPTURE FROM THEIR OWN DECLARED NUMBERS. Every pressure
  vessel in the graph is already a `high-pressure-canister` node
  carrying `capacity_kg` and its own working/bottle pressure, so
  `burst.vessel_energy_j` reads them instead of inventing anything:
  compressed gas releases its isothermal expansion work m R T
  ln(p/p_atm); a liquefied charge (nitrous at 62 bar is saturated
  liquid N2O) is a BLEVE at a disclosed work fraction of its flash
  enthalpy; and a merely pressurised LIQUID stores p^2 V / 2K --
  joules, not kilojoules, which is why a shot water-methanol tank
  splits and pours rather than exploding. Real figures that fall out:
  a 2.3 kg nitrous bottle 87 kJ, the C18's reserve tank 138 kJ, its
  brake reservoirs 52 kJ each, the Waertsilae's starting-air receiver
  21.6 MJ, and every vented fuel tank 0.

## Explosive ordnance: put a charge on it, do not hunt for the explodey part

`ordnance.py`. A `Charge` is a real filling (TNT, C4/PE4, Semtex,
PETN, RDX, ANFO, black powder, det cord -- each with its TNT
equivalence, Gurney velocity and detonation velocity), a mass, an
optional casing, a place (a node, an edge, or a point) and a FUZE:
`command`, `timer`, `delay`, or `contact` -- and a contact fuze on a
loose charge is set off by a round through it, against the filling's
own impact sensitivity (C4 needs 200 J and shrugs off a hit; black
powder needs 5).

What a detonation does is read, not tabulated:
- BLAST from Sadovsky's free-air relation at scaled distance, normally
  reflected, capped at the filling's own Chapman-Jouguet pressure
  (rho D^2 / 4) because the free-air curve is not valid inside the
  charge and nothing pushes harder than that.
- The impulse lands on the part's WALL, not its whole mass: a thin
  shell presents rho x its declared `shell_wall_m` per square metre
  however heavy the assembly is, and fails on the classic impulsive
  criterion (skin velocity i_s/(rho t) against sqrt(2 U / rho)). That
  is why 25 g of C4 opens a throttle body, 100 g takes the plenum with
  it, 250 g reaches the cylinders, and the cast-iron block itself
  holds through all of them.
- FRAGMENTS from a cased charge leave at the Gurney velocity
  sqrt(2E) (M/C + 0.5)^(-1/2), in a Mott-style spread of masses, and
  go straight into the cascade above.
- A FIREBALL of radius 3 W^(1/3) metres, which is a real ignition
  source (below), and which is drawn as its own expanding cloud of
  luminous products leaving at a third of the detonation velocity.

## Fire only when something lights it

The rule, from the user: "burning needs to only happen if there's an
ignition like an ordnance explosion or a spray particle hits a part at
ignition temp, a shot gas tank just pours out." So:

- `burst_part(..., ignited=False)` is now the default. A fuel part
  that is merely shot open splits and pours -- petrol does not
  detonate because a bullet went through it.
- The only two ignition sources are real: an ordnance fireball
  (`ignite_within`), and a fuel spray that wets a part hotter than the
  fuel's autoignition temperature for 0.6 s (the pre-existing
  `_check_burst_triggers`).
- What an ignited tank actually bangs with is its ULLAGE, not its
  contents: the flammable vapour standing in the empty part of the
  tank at ~3.5 MJ/m3, of which a disclosed fraction is blast. A
  brim-full tank makes no bang at all (no vapour space) and simply
  burns -- which is the real reason full tanks are safer than empty
  ones.
- `fire.py` makes burning a STATE. A `PoolFire` has a real area fed by
  a real spill, burns at its fuel's own mass flux (petrol 0.055
  kg/m2 s), releases m_dot x LHV x 0.7, radiates at chi Q / 4 pi r^2
  and lights anything flammable it puts 10 kW/m2 on, breathes the air
  volume it is actually IN (a spill under the vehicle pools in the
  room, not inside the one-cubic-metre engine bay) and pushes its
  products back through the same `air_volumes.add_exhaust` path the
  exhaust uses -- hot, oxygen-free and CO-rich. Measured: a 60 L tank
  fire outdoors is a 19.8 MW pool over 11.8 m2 needing 1521 L/min to
  knock down; the same fire in a closed garage eats the room's oxygen
  in five seconds, puts itself out, and takes the engine's air with it
  (rpm 2308 -> 987 as it breathes the same depleted room).

## Fittings: a hole with something screwed into it is a port

`fittings.py`, from the user's observation that the emitters are
general enough to be useful rather than only destructive. A `Fitting`
(a real coupling: bore, minor-loss K, pressure rating -- garden hose,
45/70 mm fire hose, camlock, gas hose, air quick-coupler, JIC,
sanitary) turns an emitter into a connection, in either direction:

- SUPPLY: the engine's own fluid is the source. Tap the receiver for
  shop air, the coolant for hot water, the gallery for (filthy)
  hydraulics.
- DRAW: the fitting brings its own pressure (a street main, a bottle),
  and the port feeds inward -- watering a lawn, or a standpipe feeding
  a fire monitor.

An `Appliance` declares what it needs and reports what it really got:
`Burner` (kW from mass flow x LHV), `AbsorptionFridge` (a propane
fridge: a tiny burner at a real COP of 0.3), `Sprinkler` (L/min and mm
applied), `FireMonitor` (applies water to the biggest burning
`PoolFire` using its real critical-flow figure), `AirTool` (isothermal
expansion work), `HeatExchanger` (flow x cp x the temperature drop it
can actually give up). Fed the wrong fluid an appliance says so rather
than working. Over-pressure a fitting past its rating and it blows off
and the port is a bare hole again. Verified: a 70 mm line at 8 bar
delivers 2657 L/min against the 1521 L/min a 20 MW pool fire needs,
and knocks it down progressively; a quarter-inch coupler on the C18's
receiver runs an air tool at 172 W while the receiver's fill visibly
falls.

## What each node does when it is hurt

`node_effects.py` answers, for every graph node, the four questions:
(a) are its contents FOULED, (b) is it UNSEALED, (c) has it lost
STRUCTURAL function, (d) is a node it depends on MISSING -- and a
`NodeBehaviour` per part class decides whether that part cares and
what follows, as multiplicative levers on quantities the engine
already integrates (`Effects`: per-cylinder charge, intake flow,
mixture, exhaust restriction and openness, coolant exchange and flow,
oil flow and pressure, fuel supply, alternator, boost) plus real
verdicts (engine dead, no spark, no air brakes, cannot be restarted,
mounts broken). Structural loss is its own budget: accumulated impact
energy against mass x the material's toughness per kilogram.

Behaviours: oil pump (cavitates on air, holed loses pressure, no drive
or suction means no delivery), coolant (an oil film insulates the
exchanger, air locks the pump), intake (oil = smoke and rich, ingested
air = a lean vacuum leak, a missing throttle = unmetered), exhaust
(holed is partly open and louder into the bay), fuel (aerated or
watered starves), cylinder/head (holed loses compression on THAT bore,
gone is a dead cylinder), block (structure gone is a dead engine),
turbo (gone, or its oil feed gone, is no boost), alternator, ignition
(no magneto/coil is no spark -- a compression-ignition engine does not
care), timing (no camshaft or injection pump is a dead engine),
pneumatic (the reserve gone is no air brakes and no air start), mounts,
starter, fan, driveline. Anything else honestly reports that nothing
depends on it.

Verified end to end: 150 g of C4 on the C18's injection pump leaves it
stone dead (rpm 2124 -> 0) with the fuel rail cut off and the oil pump
with no delivery; the same on the Jeep's camshaft kills four cylinders
and drops it from 2256 to 981 rpm; ten 7.62 rounds into the block hole
exactly one cylinder and cost exactly that cylinder's compression.

## One stateful machine: the parts, the circuits and the HUD agree

A bug the user caught -- four gasoline leaks running at 11 g/s while
the fuel pump was neither working harder nor running out -- turned out
to be two separate failures of the same principle, and both are fixed:

1. **Mesh pieces are not graph nodes, and only one mapping may decide
   which part was hit.** The castings are drawn as many named pieces
   (`sump`, `cyl3_water_jacket`, `crankcase_bulkhead_4`) that are not
   nodes of their own. `damage_state.owning_part` is now the single
   declaration of which real part owns each piece, and
   `mesh_part_identity` uses it, so a hole in `cyl3_water_jacket` is a
   hole in cylinder 3 EVERYWHERE: in the circuit membership that
   decides what leaks out of it, in its node condition, in the
   dashboard and in what a burst takes with it. `node_effects` had
   grown a private copy of that table; it is gone.
2. **The emitters must not own fluid state.** They kept a private
   `lost_l` ledger while the tank bar, the sump reading and the
   receiver pressure went on as if nothing had happened. Now
   `HoleEmitterField` asks the sim for the real driving pressure
   (`_circuit_pressure_pa`) and the real contents
   (`_circuit_remaining_l`) and hands back every gram through
   `_deplete_circuit`, which is the ONLY path by which a leak, a
   burst or a fitted port removes fluid.

That second fix also exposed a real modelling gap: a depletable vessel
does not carry its pressure in `pressure_pa` at all -- its state is
`fill_level_frac` against `bottle_capacity_kg`, and in a fixed volume
pressure follows mass, so a half-empty receiver really is at half its
working pressure. Reading it correctly is what made the C18's reserve
report 7.59 bar instead of 1.01 and made shop air work at all.

And the chain now completes: a .50 BMG through the oil pan drains the
real sump 4.65 -> 0.22 L in eight seconds and stops when it is empty;
below a third of the pan's charge the pickup uncovers
(`_step_oil_pickup`) and the pump draws air -- fed into the oil
circuit's own `FluidMix` so that the EXISTING cavitation rule handles
a dry sump and a holed suction line by one mechanism -- gallery
pressure falls 420 -> 290 -> 101 kPa, and `node_effects` reports the
pump cavitating on 100 % air with no delivery.

## The auxiliary plant: air treatment, refrigeration, hydraulics, and the gunk

A machine engine carries a skid of equipment beside it, and the reason
to model it is that it is what makes air and oil QUALITY matter. The
whole thing is declared by `engines.AuxiliaryPlant` (fitted=False on
every road engine; the CAT C18 carries the full set), emitted as real
graph hardware by `plant_parts.py`, and stepped by `plant.
AuxiliaryPlantRuntime` from what the engine is really doing.

### The air train (`air_treatment.py`)

An ordered chain of `Stage`s in the order the user specified, each
doing one real thing to one real stream:

    compressor -> AFTERCOOLER (electric fan) -> CHILLER (refrigerant,
    off the AC loop) -> WATER SEPARATOR -> COALESCING FILTER ->
    PARTICULATE FILTER -> REHEATER -> WET TANK -> pneumatic manifold ->
    the isolated reserve set

The physics is ordinary psychrometrics and nothing is invented:
Magnus/Tetens saturation pressure, humidity ratio w = 0.622 p_v/(p-p_v),
a real compression discharge temperature (isentropic rise over the
machine's own efficiency), condensation of everything above w_sat when
a stage cools the stream, and the PRESSURE DEWPOINT as the one number
that says whether the system is dry. Contamination is carried per
kilogram of dry air: water from the intake's own humidity, oil from the
compressor's real carryover (a lubricated recip ~25 mg/m3, a screw ~3,
oil-free 0), dust from ambient past the intake filter.

Two things were wrong on the first pass and both are now right:

- **The reheat is RECUPERATIVE, not electric.** A real refrigerated
  dryer runs the hot wet incoming air and the cold dry outgoing air
  through one air-to-air exchanger: the hot side is pre-cooled (which
  shrinks the chiller's duty) and the cold side is reheated by exactly
  that heat. It is a wall between two pipes, not a heat pump -- heat
  flows downhill, no work in. The heat pump is the refrigerant loop.
  Wiring it dropped the aftercooler duty 5.0 -> 2.1 kW and the electric
  load from 100 A to 4.5 A; an electric element remains only as a trim.
  Reheat removes NO water: it lowers the RELATIVE humidity of air that
  is already dry, so nothing re-condenses in the tank or the lines.
- **A loading filter does not quietly pass more and more.** A coalescer
  holds its rated efficiency and costs rising pressure drop until it
  FLOODS, and then re-entrains. And it DRAINS what it coalesces
  continuously, which is why a real element lasts thousands of hours
  instead of filling up in a fortnight. Only solids load it.

`GunkLedger` is what got past and is now living in the system: oil and
water emulsify into sludge, dust bound in oil becomes abrasive, free
water corrodes a steel tank, and the ledger holds what the low points
RETAIN rather than integrating every gram that ever passed. Measured
over long runs at 12 g/s and quarry air:

| state | dewpoint | oil delivered | consequence |
|---|---|---|---|
| treated and serviced, 2000 h | 3 C | 13.6 ug/kg | everything ~1.00 |
| dust filter never changed, 4000 h | 3 C | 13.6 ug/kg | tools x0.93 |
| no filters, drains shut, chiller off, 500 h | 42 C | 13570 ug/kg | valves x0.52, seals x0.00, vessel x0.30, tank FULL and carrying water over |

`gunk_effects` feeds `node_effects.Effects` as `air_flow_factor` /
`air_seal_factor` / `air_tool_factor`, and the flow factor really does
restrict the idle-assist dump port -- so a neglected air system shows
up as an engine that will not catch its own idle.

### The refrigerant loop (`refrigeration.py`)

One AC compressor feeding three evaporators: the cabin, the compressed-
air chiller, and the hydraulic oil chiller. R134a saturation from a
two-point Clausius-Clapeyron fit, condensing temperature from ambient
plus the condenser's approach (which gets far worse without airflow),
COP = Carnot x a real 0.45 machine efficiency, capacity = shaft power
x COP split across whatever is calling.

The CLUTCH control is the real one, in the order the real switches
sit: the master switch, then something calling, then the LOW-pressure
cutout (a loop that has lost its charge never engages -- this is what
protects the compressor), then the HIGH-pressure cutout with its own
hysteresis (a stopped condenser fan trips it). The loads genuinely
compete: with the hydraulic oil at 62 C calling 10.4 kW, the air
chiller got 0.19 of the 0.81 kW it wanted and the pressure dewpoint
rose accordingly -- which is exactly why a machine's air goes wet on a
hot day under load.

### The hydraulic circuit (`hydraulics.py`)

Three real facts, modelled rather than approximated:

1. **The relief valve is a heater.** Whatever the actuators do not turn
   into work comes back as heat, and the worst case is a lever held
   against its stop: full pressure at full flow, nothing moving, every
   watt dumped. Measured: 37.8 kW of relief heat against a 240 W/K
   oil cooler flat out at 32.9 kW -> 167 C, oil destroyed, seals gone
   in 20 minutes. With the cooler fan dead too it passes its FLASH
   POINT at 360 C, and `fire.py` will light it from any real source.
2. **The tank breathes, and that is how water gets in.** Oil leaves as
   a cylinder extends and the tank draws humid air IN. Nobody spills
   water into a hydraulic tank; it arrives one breath at a time.
3. **Temperature sets viscosity and oil life.** Walther/ASTM D341
   log-log viscosity (a VG46 oil reads 46.0 cSt at 40 C by definition
   and 6.7 at 100 C), a thin film above ~100 C, cavitation when it is
   too thick to draw, and oxidation halving the oil's life per 10 K
   above 60 C.

Cleanliness is reported as a real ISO 4406 code; wear debris is
generated by thin films and dirt and removed by a beta-200 return
filter per pass.

**Dry-air blanket vs desiccant breather.** The user asked whether
feeding the reservoir from the plant's own dried air has precedent: it
does, and it is better. Every transport aircraft pressurises its
hydraulic reservoirs with engine bleed air -- nothing gets in, and the
pump always has positive inlet pressure so it cannot cavitate;
industrial power units and power-station lube systems blanket with
nitrogen or instrument-quality dried air for the same reason. A
desiccant is a consumable that fails SILENTLY once spent; a blanket
cannot saturate. But it is only as dry as its supply, and that
coupling is the point. One 8 h shift, same duty, only the reservoir
arrangement differing:

| arrangement | water in the oil |
|---|---|
| plain breather | 5297 ppm |
| desiccant breather (spent partway) | 2104 ppm |
| dry-air blanket, dryer working | 451 ppm |
| dry-air blanket, chiller switched OFF | 4928 ppm |

A blanketed tank also tolerates far thicker oil before the pump
starves (the real NPSH benefit). Note the floor: a REFRIGERATED dryer
cannot go below about +3 C dewpoint without icing, which is why the
blanket case still shows free water; a desiccant/adsorption dryer
(-40 C dewpoint) is the next stage up and is not built yet.

### Manifolds, controls and power

`plant_parts.py` emits the pneumatic manifold, the hydronic manifold, a
control panel carrying every real switch (main chiller, air dryer,
reheater, separator drain, wet-tank drain, hydraulic chiller, reserve
isolation), the LP/HP pressure switches, the drain solenoids, and an
ACCESSORY BATTERY BANK behind a CHARGE ISOLATOR. The isolator is a
real voltage-sensitive relay scaled off the system's own nominal
(closes above 1.05x nominal, drops out at nominal), so the plant's
fans, heater and solenoids run off their own bank and cannot flatten
what has to crank the engine; a flat bank stops the electric half of
the treatment train, and the air quality collapses with it.

The C18 graph goes from 302 to 358 nodes and 305 to 344 edges with all
of this on it, and every piece is a real part: it renders, it can be
shot, it leaks through hole_emitters, it bursts, and node_effects has a
`PlantBehaviour` for it.

## Ultra-dry air, a redundant blanket, warming, and an actuator in the bay

The second pass over the plant, all of it from questions worth
answering rather than features worth adding.

### Recuperation is not a heat pump (`air_treatment.py`)

Asked what "recuperating" meant. It is a wall between two pipes: the
hot wet incoming air and the cold dry outgoing air pass either side of
one air-to-air exchanger, and heat crosses it the only way heat ever
goes on its own -- downhill. No compressor, no work. The hot side is
pre-cooled (less for the chiller to do) and the cold side is reheated
by exactly that heat (relative humidity down, so nothing re-condenses
downstream). The heat originally came from the compressor: squeezing
air heats it, the aftercooler throws most of it away, the recuperator
recycles a slice of what is left. THE HEAT PUMP is the refrigerant
loop -- that one moves heat uphill, from 3 C air into 34 C ambient,
which is why it needs a compressor and real shaft power.

### The desiccant bed as a shared component (`desiccant.py`)

Both systems need one, on different duties, so the bed is one class
with real capacity and real DURABILITY:

- CAPACITY ~0.15-0.22 kg of water per kg of desiccant; finite, and
  what a cartridge's life actually is.
- DEWPOINT held at the rated number while there is unused bed ahead of
  the mass-transfer zone, then BREAKTHROUGH -- a desiccant dryer does
  not drift, it holds its figure and then stops working.
- REGENERATION by purging the offline tower with a slice of its own
  product: a real and unavoidable ~15 % of flow, which is why a
  desiccant dryer is expensive to run. Each cycle attrits the beads.
- POISONING: OIL KILLS DESICCANT PERMANENTLY, and no regeneration
  recovers it. Which is exactly why the coalescing filter belongs
  UPSTREAM of the towers -- a flooded coalescer does not merely pass
  oil on, it destroys the expensive thing behind it. Measured: 40 h
  with the coalescer removed leaves the beds 7 % poisoned and 4.8 %
  attrited over 240 cycles.

`TwinTowerDryer` is what takes the air train past the refrigerated
floor. A refrigerated chiller CANNOT go below about +3 C pressure
dewpoint without icing its own coil -- that is a hard physical limit,
and it is why the blanket case still showed free water. Adsorption
takes it to -40 C: measured 3.0 C -> -40.0 C, and the water reaching
the hydraulic oil fell from 451 ppm to 7 ppm.

### A three-way blanket manifold with real fallbacks (`hydraulics.py`)

The reservoir's gas inlet is a selector with four levels, and the rule
governing it is that the tank must NEVER be sealed with no way to
equalise (it would pull a vacuum and collapse, or suck in through
whatever seal is weakest):

    plant air  -> nitrogen bottle -> emergency desiccated vacuum break
               -> plain breather

Plant air is the everyday supply and is refused when it is not dry
enough (`blanket_max_dewpoint_k`): feeding a sealed tank WET air is
worse than letting it breathe. Nitrogen is the clean backup and the
only one that is dry AND inert -- no oxygen over the oil at all. The
vacuum break is a check valve cracking at a small negative pressure
and admitting air through a small desiccant cartridge, sized for
occasional use; if it is working hard, something upstream has failed.

### How much a reservoir actually breathes -- a modelling error found

The first version had the tank breathing the PUMP FLOW, which is wrong
by orders of magnitude: that oil comes straight back. A reservoir's
level swings by the NET ROD VOLUME of its cylinders (bore side minus
rod side), once per work cycle. Corrected to `cylinder_swing_l` x
`work_cycles_per_min`, and the sizing lesson it then produced is real
and useful:

| blanket source | 8 h of an 18 L swing at 2 cycles/min |
|---|---|
| plant air | 21.7 kg of air used (free, unlimited from the compressor) |
| nitrogen | 1.19 kg -- a 1.2 kg bottle is worth exactly one shift |
| emergency cartridge | spent in about an hour |
| BLADDER-separated reservoir | no gas exchange at all, ever |

So a consumable blanket gas cannot keep up with a machine that cycles
its cylinders all day. The real answer for that duty is a bladder or
diaphragm reservoir (`bladder_separated`), which seals the gas side
off entirely and simply changes shape; aircraft do the equivalent with
a piston driven by system pressure. The nitrogen bottle is correctly a
one-shift outage backup, not a permanent supply.

### Warming a cold system, and why not just an element

Asked whether a tank heater was needed "or something gentler". Cold
oil is a real problem -- below about -10 C a VG46 is past 2000 cSt and
the pump cannot draw its own charge -- but an immersion heater is the
crude answer and the one that goes wrong. What matters is WATT
DENSITY, not total watts: a high-density element boils the oil film at
its own surface, cokes it on as varnish, and that varnish insulates
the element so it runs hotter and cokes faster. Real lube/hydraulic
heaters are specified at ~1.2-1.5 W/cm2 with a thermostat AND a
low-level cutout, because an element running uncovered destroys itself
and can start a fire. All of that is modelled (`heater_coking`,
`heater_burned_out`, the level cutout). Three ways to warm it,
gentlest first, and the element is last:

1. the engine's own jacket coolant through an oil/coolant exchanger --
   free heat, no hot spot, cannot coke
2. the relief valve, which is already a 37 kW heater
3. a low-watt-density element, for when the engine is cold too

### The actuator in the bay: a hydraulic fan drive

Asked whether anything could use an actuator in the engine bay. The
best real one is a hydraulic fan drive -- big machines drive the
cooling fan through a hydraulic motor precisely so fan speed has
nothing to do with engine speed (flat out at idle on a hot day, idling
at full rpm on a cold one, and reversible to blow chaff out of the
core). It closes a real loop with everything else: the fan drive is a
consumer on the same circuit (fan power goes with the CUBE of speed),
every watt it burns is more heat into the same oil, but the air it
moves cools the oil cooler AND the refrigerant condenser, dropping the
loop's head pressure and giving the chillers back capacity. It
thermostats itself off its own oil. Measured: 807 rpm / 37 % airflow
taking 0.59 kW, holding the oil at 49 C instead of 134 C; and with it
idle the condenser sits at 58 C / 1596 kPa instead of 34 C / 854 kPa.

## Two sizing questions that were really wiring bugs

**"Do we need to up the C18's AC compressor?"** No -- it needed to be
USED. The graph declares a 4000 W compressor (the clutch is sized at
26.7 Nm against `REFERENCE_COMPRESSOR_OMEGA_RAD_S`), but
`refrigeration.RefrigerantLoop` had invented its own
`displacement_w_per_rad_s = 3.0` and was drawing 0.69 kW at 2200 rpm --
a SIX-FOLD disagreement between two parts of the same machine, and the
same class of bug as the emitters keeping a private fluid ledger. The
rating is now declared on the compressor node by
`_add_belt_driven_compressor` and read by
`RefrigerantLoop.from_compressor_node`, so there is one declaration of
how big the compressor is. Result on the same hot-oil case: 6.08 kW
shaft x COP 3.39 = **20.59 kW of cooling** where it had been 2.57, the
oil holding at 60 C with the chiller not even calling.

The second half of that fix was the DEMAND: the oil chiller had been
asking for the whole cooling load (a proportional 600 W/K, capped at
15 kW). A refrigerant chiller is a TRIM duty -- the oil-to-air cooler
carries the load, and the chiller holds a precise temperature when
ambient is too warm for air alone. `chiller_demand_w` now asks only for
what the cooler cannot already reject, capped at a real trim capacity.

**"The C18 idle-assist trip is too close to idle -- it just drains."**
Correct, and it was worse than it looked. The trip derived as
`idle_rpm * 1.15` = 690 rpm on an engine idling at 600, and the test
was a static window `0.4*idle < rpm < trip` -- which is TRUE AT A
HEALTHY IDLE, so the reservoir emptied for nothing. It was also the
wrong kind of test: this is an anti-stall booster (the real Knorr-
Bremse PBS), and what it looks for is a SAG, not a low rpm. A static
threshold also fires too LATE to help, because by the time a big
diesel has reached it the stall is already unavoidable. Rewritten as a
real controller:

- the trip point sits BELOW the governed idle (`idle * 0.88` = 528 rpm)
- it ALSO trips on rate: near idle and falling faster than 180 rpm/s,
  catching the engine on the way down
- it LATCHES and releases only on genuine recovery (`idle * 1.02`), so
  it cannot chatter across a threshold
- the cranking/stalled floor is kept

Verified by direct test of every path: healthy idle does not fire, a
slow drift does not fire, a fast sag near idle does, below-trip does,
and the latch holds from 540 rpm through 605 and releases at 615.

Worth noting from the same test: the C18's governor is strong enough
that even 12 000 Nm of dyno brake only drags it to 585 rpm at
-124 rpm/s, so the assist correctly never fires in that scenario --
a governor is exactly what stops a stall, and the booster is for
transients faster than the governor and turbo can answer.

## The dyno couples through the engine's own recommended hardware

The rig used one generic friction clutch for every engine, so every
engine was tested through something that slips. A real test cell does
not do that -- it uses what the engine is built to drive through, fed
from the engine's own circuits. `couplings.py` now declares four real
kinds and the distinction between them is the whole point:

- "dry-friction" -- an ordinary dry plate. Hydraulically ACTUATED is
  still a dry plate: the hydraulics only push the release bearing, so
  it slips exactly as before. This is what a workshop usually means by
  "a hydraulic clutch" and it does not solve anything.
- "wet-multi-plate" -- plates in oil, hydraulically APPLIED. Capacity
  is the real product of apply pressure x piston area x friction faces
  x mean radius x mu, so it is modulated by pressure and HOLDS without
  slip once capacity exceeds the torque asked of it.
- "fluid-coupling" -- a Foettinger coupling, torque with the SQUARE of
  the speed difference, always 2-5 % slip, can never lock on its own.
- "torque-converter" -- a fluid coupling with a stator, multiplying
  torque at stall.

What actually removes the slip is the LOCK-UP CLUTCH bridging a fluid
coupling or converter once the two sides are close. `recommended_for`
picks from what the engine IS: a heavy diesel gets a wet multi-plate
sized to hold 1.6x its own peak torque, a turbine a locking converter
(it cannot be clamped to a stationary load), a free-piston engine a
fluid coupling, a road engine its dry plate. The C18 comes out with a
wet plate of 5854 Nm capacity against its 3658 Nm peak.

**Where the apply pressure comes from, and a real error corrected.**
First version applied the clutch off the hydraulic circuit's WORKING
pressure, which reads zero whenever nobody is using the implements --
so the clutch had no capacity at all. Real machines never do that:
clutches, brakes and valve pilots run off a separate regulated PILOT
supply (25-40 bar through a reducing valve or its own small pump) that
is live the whole time the pump turns. `HydraulicCircuit.
pilot_pressure_pa` / `pilot_available` is that supply, and it is what
applies the coupling -- the rig running off the engine's own reservoir.

A wet clutch shares that oil, so its slip heat goes INTO it
(`coupling_heat_w` -> `heat_w`): a coupling held slipping is a real way
to cook a hydraulic system, and the oil cooler has to be sized for it.
Measured 25-50 kW of clutch heat at a steady 40-60 rpm of slip.

**The stall test.** With the C18 in gear through its own wet plate and
its own pilot supply, ramping the dyno brake: rpm held at ~2150 to
9600 Nm, then was dragged down 2087 -> 1963 -> 1771 -> 1578 -> 1419 ->
1117 -> 415 rpm. The anti-stall booster stayed SILENT the whole way
down until the engine went under its trip point, then fired for 0.65 s
and pulled the air reserve 95.1 % -> 77.6 %. That is the fixed device
doing exactly its job: nothing at a healthy idle, everything when the
engine is genuinely going under.

Also found in passing: the C18's gearbox starts in NEUTRAL
(`_current_gear_ratio()` is 0 at `gear_index` 0), so a dyno brake does
nothing at all until it is put in gear -- worth knowing before reading
any dyno result as a null.

**Open:** `ClutchPort` and `Coupling` both cap torque, so they double
up -- the ClutchPort's spring/damper always demands its cap while
slipping and the coupling then re-caps it. The coupling should own the
capacity and the port should just be the numerical spring. Works, but
it is two mechanisms where there should be one.

## Practical next step (not yet started)

The classification rule for everything besides the fuel tank/pump/
transmission is still open — coolant, starter battery, exhaust,
pneumatic lines, and so on each need a real answer, not a guess,
before `_SOURCING_OVERRIDES` grows further. The mount-correlation tool
also has no real cage source yet (the game doesn't supply one today)
— it's verified correct against a synthetic cage, not yet wired to
anything real. `engine_mounts.py`'s install-context selection is also
still a caller-declared string, not inferred from the engine's own
label/application (deliberately — guessing an install context from
free text would be exactly the kind of unguided assumption this
codebase avoids elsewhere).

## Implemented: any fuel is a different fuel network (`working_fluids.py`, `fuel_network.py`, `expander.py`)

The conversion question ("what hardware does it take to run this
cylinder on compressed air / hydrogen / natural gas / propane / coal or
wood gas / steam?") is answered by three modules that sit on the
existing fluid-circuit system rather than beside it:

- `working_fluids.py` -- THE registry of working fluids: every liquid
  fuel the catalogue already had, the gaseous fuels (coal gas, wood
  gas, natural gas, propane, hydrogen, petroleum vapour) and the
  non-combusting expander fluids (steam, compressed air), in one
  property vocabulary (LHV, stoich AFR, octane, density, latent heat,
  flammability limits and stoichiometric volume fraction, laminar
  flame speed, minimum ignition energy, storage class and pressure,
  gamma/R). `engines.py`'s and `otto_langen.py`'s fuel tables are
  views of it now.
- `fuel_network.py` -- a declared chain of real devices between the
  store and the admission point: sources (vented tank, pressurized
  bottle, gasholder with an on-site generator, utility main, boiler
  with feedwater, compressed-air receiver), in-line stages (staged
  regulator, coolant-heated vaporizer, cooler/filter, flame arrestor,
  purge valve, lock-off solenoid) and admission (mixer, gas injector,
  the engine's own carburetor/injectors, cutoff valve). It emits into
  the SAME drivetrain-graph "fuel" circuit every tank already lives in
  (so fill, composition, availability and flow ceilings come from the
  one reservoir primitive), and a runtime steps the stages each tick.
  `validate()` refuses a network missing what a real install of that
  fluid needs (hydrogen without an arrestor, LPG without a vaporizer,
  steam without a boiler). `convert_engine(engine, spec)` is the whole
  conversion: same cylinder, new network, knock compatibility derived
  from octane, charge energy derived from fixed-volume stoichiometry
  (natural gas ~0.90 of gasoline, hydrogen ~0.83, wood gas ~0.72).
- `expander.py` -- the cutoff expander cylinder bank steam and
  compressed air share (one engine kind, "expander"): ideal indicator
  diagram MEP, double-acting, with the two real fluid-specific
  failures -- steam water hammer (cold cylinder, drain cocks shut)
  and compressed-air exhaust icing (no dryer on the line). Two
  catalogue engines: a c. 1900 steam traction engine (boiler-limited
  to ~27 kW, which is the right order for a ~35 hp machine) and a
  two-cylinder compressed-air mine locomotive.

What this bought: a carbureted Jeep six runs on CNG, propane (starts
weak on a cold vaporizer and comes good as the coolant warms), or
hydrogen (through its arrestor, with an intake-flashback risk that
rises with intake temperature) without touching the piston loop;
the Otto-Langen runs from a generator-fed holder through a purge
valve; and a fuel-network-declared boiler runs the traction engine
with real feedwater bookkeeping.

## Implemented: conversion contracts (`engine_toy/conversion.py`) and the demo's shift-F

`plan_conversion(engine, fluid)` writes the engineering contract for
running THIS chamber on THAT fluid: ignition mode (spark / compression
/ expander), the compression-ratio target with both real routes to it
(piston/head clearance change as cc and mm of skim, or a stroke
change in mm), the timing shift from the fluid's laminar flame speed,
the fuel conditioning (heater for fry oil / crude, vaporizer for LPG),
the hardware (injection pump + glow plugs, spark system, hardened
seats, arrestor), and the predicted torque/power factors from the
same first-principles rating the builder uses (`derive_gross_bmep_pa`,
with the gas-displaces-air correction). `apply_conversion` produces
the converted Engine; `convert` does both. The knock-limited CR is the
inverse of the rating's own knock relation; a catalogue diesel left on
the generic 10:1 default is taken as a real 17.5:1 chamber.

In `main.py`, shift-F cycles the current cylinder through every
registered working fluid, applies the conversion, logs the contract,
and the dashboard shows the live network (supply pressure,
availability, fill, fuel temperature, warnings). Verified: the
commuter four on vegetable oil (17.5:1, compression ignition, heater-
limited until warm), the 1901 curved-dash on natural gas and hydrogen,
the industrial diesel on natural gas as a spark conversion, a Jeep six
as a single-acting steam or air expander; all 19 registered fluids
build a contract against the commuter engine.

Known rig property: expander engines have no governor, so with a
constant-torque dyno load they run up to wherever line-choke-limited
power meets the load; use the throttle-target governor (T) or a load
sized to the engine.

## Implemented: per-kind cylinder port layouts and meshes (`engine_toy/cylinder_ports.py`)

Four real cylinder kinds, each with its own real set of holes and its
own mesh, laid out from the same cylinder sites and the same
network/conversion declarations everything else reads:

- spark-piston: intake and exhaust valve ports in the head, a spark-
  plug boss, and a port-injector boss when the admission is a liquid
  or gas injector (a carburetor/mixer feeds the port from the runner,
  so no boss on the cylinder).
- compression-piston: intake, exhaust, a direct-injector boss on the
  bore axis, a glow-plug boss.
- atmospheric (Otto-Langen): an open-top bore with the rack and its
  guide above, and the slide valve's gas, air, flame-transfer and
  exhaust openings at the foot.
- expander: a valve chest alongside the bore with admission/exhaust
  passages to the head end, the same pair plus a rod gland at the
  crank end when double-acting, a drain cock at each working end and a
  lubricator boss.

Port COUNT follows the declared valvetrain: `lifter_spring.valves_per_
cylinder` splits into intake and exhaust throats (2 -> 1+1, 4 -> 2+2,
5 -> 3+2); a loop-scavenged two-stroke (`has_poppet_valves=False`)
has no head valves at all -- two transfer ports and a wall exhaust
port uncovered by the piston; a uniflow two-stroke (poppet valves,
two-stroke) has a ring of four scavenge ports in the wall and its
valves all exhaust. The mesh also carries the running gear -- piston
at the slider-crank position for any crank angle, con rod, crankpin
and webs on a main journal (throw phase from the firing order), a
crosshead and piston rod on a double-acting expander, the rack
pinion on the Otto-Langen -- and an outer wall that is a stack of
cooling fins when the engine is air-cooled (no water pump, no coolant
pump) or a plain water jacket otherwise.

`drivetrain_graph.build_drivetrain_graph` emits every port as an
"engine-block-port" node (port_kind, outward port_direction,
port_radius_m, fluid_role), aligning onto the production-authored
intake/exhaust valve nodes a piston engine already has, and stores the
serialized layout on the graph document ("cylinder_layout").
`vehicle_mesh.build_drivetrain_solid_parts` builds the cylinder bodies
and port stubs from that, tagged block_cyl_N so the live renderer
colours them by per-cylinder block temperature. A converted engine
(conversion.py) gets the layout its new kind and admission imply --
the commuter on fry oil grows direct injectors and glow plugs and
loses its plugs; on hydrogen it gains gas-injector bosses.

## Implemented: crank train and crankcase (`engine_toy/crank_mesh.py`)

The crankshaft and the crankcase are built once per cylinder layout,
not per cylinder. Cylinders are grouped into throw stations by their
crank-axis position (a V pair shares a crankpin; a radial row shares
one throw); main journals sit between stations and at both ends at
the station pitch, with a nose and pulley ahead of the front main and
a flange and flywheel behind the rear one; each throw has two webs,
a crankpin and counterweights at that station's phase for any crank
angle. The crankcase is a crank tunnel sized to the throw swing, one
segment per station carrying that cylinder's block temperature, a
bulkhead disc at every main, a skirt from the tunnel to each bore's
foot along that bore's own axis (which is what makes a V or flat case
a V or flat case), front cover and rear-seal bosses, and a wet sump in
the oil thermal group -- except on a loop-scavenged two-stroke, whose
crankcase is a sealed pump chamber with its own intake boss and no
sump. Expanders get a bedplate with a pedestal at each main and an
open crank. When a graph carries a cylinder layout, vehicle_mesh no
longer draws the generic block and oil-pan boxes.

## Implemented: heads, cam cases, valve covers, valley covers, Otto freewheel, expander frames (`engine_toy/head_mesh.py`, additions to `cylinder_ports.py`/`crank_mesh.py`)

- Heads are one casting per BANK (cylinders grouped by bore-axis
  direction), oriented along the bank axis, so V/flat heads tilt with
  their banks. The cam case follows the valvetrain: pushrod (cam in
  the block beside the crank, pushrods and rockers per cylinder,
  rocker cover), sohc (one cam in a cam box on the head), dohc (two
  cams); valve covers over each. The valvetrain is DERIVED until the
  catalogue declares `EngineArchitecture.valvetrain` (3+ valves ->
  dohc; two-valve over 7000 rpm -> sohc; otherwise pushrod) and is
  disclosed as such.
- Multi-bank engines get a valley cover between the bank skirts (a
  flat engine gets a case top cover instead), in the oil group.
- V/W banks breathe in from the valley: each tilted bank's intake
  ports face the engine centre plane, exhaust outboard.
- The Otto-Langen's "crank" is drawn as what it is: rack pinion ->
  freewheel drum -> flywheel shaft in two bearings on the column top,
  big flywheel outboard.
- Expanders hang their open-bottomed cylinders from a real frame:
  two frame plates (hornplates) rising from the bedplate to the
  cylinder's crank-end cover, crosshead guide bars between them.
- The cylinder layout is now emitted BEFORE the runners/primaries are
  routed, and the intake plenum / exhaust manifold nodes sit relative
  to the real heads -- the "placement issues" were manifolds placed
  at the old block-face guess, well below the new heads.
- `mesh_primitives.capped_tube_mesh`: closed drums (flywheel, pulley,
  bulkheads, fins, piston) instead of open hoops.

Not done (parked): a free-body/constraint integration of the piston
positions with the crank pinned at its bearing and animations per
kind -- the kinematic slider-crank at any crank angle is in place, so
an animation can already be driven from the sim's crank angle.

## Implemented: procedural exhaust headers (`engine_toy/exhaust_header.py`)

`plan_exhaust_header(layout, header_type)` routes every exhaust port
of the cylinder layout as a real polyline: a short stub along the
port's own direction, a mandrel-radius quarter bend (1.5 D
centreline), a run along a rail outboard and below the head parallel
to the crank, and a final bend down into the group's collector.
Grouping is the typical rule -- one bank up to four cylinders into
one collector, five to eight split front/rear into two (an inline
six's tri-Y), each bank of a V or flat its own -- and a
"stock-manifold" header_type gives a log: stubs into one shared rail
with a single outlet. `emit_header_graph` writes the plan into the
drivetrain graph as exhaust-flow-path edges through header waypoints
and an exhaust-collector node per group, re-pointing the production
`.exhaust_primary` edge so its identity survives, then one edge from
each collector to the existing downpipe junction. Primary lengths are
reported per group (the typical layout is not equal-length; a tuned
header would be a different planner, not a different graph).

## Implemented: the detail pass on singles, radial, rotary, expanders, Otto-Langen

- Singles: a 1-into-1 is a stub, one bend and a short outlet, no rail;
  a hit-and-miss layout gets an open water HOPPER on the jacket and a
  flywheel on both ends of the crank (`cooling="hopper"`,
  `twin_flywheels`), read from the catalogue layout label.
- Radial: the crankcase is a drum sized to the cylinder ring with a
  front cover and rear accessory case; the exhaust is a collector RING
  behind the cylinders that each primary bends rearward into, with
  one outlet at the bottom.
- Rotary: a new cylinder kind. Housings are drums along the eccentric
  shaft scaled off real 13B proportions (R 105 mm, e 15 mm, 80 mm
  width, 654 cc/rotor) by displacement per rotor; each housing has
  side intake ports in the end plate, a peripheral exhaust port and
  leading/trailing plug bosses in the rim; the rotor is a triangular
  prism riding the eccentric at a third of the shaft angle; the two
  rotors' exhausts join one outlet. No heads, cams or crank train
  apply (`mesh_primitives.prism_mesh` was added for the rotor).
- Expanders and the Otto-Langen: their exhaust passages/port are
  routed too -- stub, bend, and an outlet clear of the body (a blast
  pipe upward on an expander). The graph gets an exhaust-manifold
  junction for kinds the production graph never authored one for.
- The Otto-Langen flywheel is sized at a third of the column height.

## Implemented: removable valvetrain parts with state, and head oil ports (`engine_toy/valvetrain_parts.py`)

Every poppet valve in the cylinder layout is a set of individual
parts: valve (stem + head, lifted by a first-order cam phase),
spring, retainer, and a bucket tappet (overhead cam) or a rocker plus
pushrod (cam in block). Each is a `RemovablePart` with an axis-aligned
box, what it attaches to, the direction and travel it comes out
along, and what must come off first; `removal_blockers` sweeps the
box along that direction and names anything in the way, which is the
assembly rule a game needs (cover, then retainer, then spring, then
valve). Springs carry real state (`SpringState`): free length, rate
from the engine's own LifterSpring, installed height, permanent sag
(a disclosed thermally-accelerated relaxation via `age()`), and a
shim; seat load, open load and coil-bind margin follow from spring
arithmetic, so a tired spring loses seat pressure and a too-thick shim
binds. Bosses (plugs, injectors, drain cocks) are removable parts
too, with a torque state.

Meshes: `build_drivetrain_solid_parts(graph, covers_off=True)` omits
the valve covers and draws the valve gear (helix springs, retainers,
buckets/rockers/pushrods). Removal clearance sweeps from just past a
part's own far face, so parts merely touching at rest are not
blockers; the valve's box is its stem (the head is at the seat).

## Implemented: casting ports and the port-to-port mating solver (`engine_toy/assembly_ports.py`)

Corrected from the first cut: the head's own port is the FILL (a
line port on top, open until a cap or line goes on it); drains are
the pan's job. Each casting declares ports from the layout -- heads:
fill, deck-face oil feed, two deck-face oil returns, coolant
passages on jacketed engines; crankcase: the matching deck holes,
main gallery, breather, dipstick, pump pickup, pan rim; pan: rim,
pickup, drain plug. `mate_ports` pairs mating (gasket-face) ports
across different parts that are compatible, face each other and sit
within 12 mm, in n log n (bucketed by kind, sorted along the crank,
neighbour scan), and reports seals, OPEN mating ports and line
ports. Seals go into the drivetrain graph as zero-length
"port-face-seal" edges in the fluid's circuit; ports as
engine-block-port nodes with `mating`/`connected`. `transplant()`
sets a donor head's ports on another block (with an offset along
the crank) and reports what mates and what is left open -- the
mechanism for both catastrophic parts mixing and hot-rod
interchange (`deck_offset_m` models decking/spacing the donor head).

Removability is a MATING-SURFACE spec, not a geometric proof: every
`RemovablePart` carries a `MatingFace` (face, part it seats on,
normal, seal, fasteners) and `must_remove_first`; the swept-box
check survives only as an advisory (`assembly_manifest(...,
advisory_clearance=True)`). Crankcase deck ports are per bank
(`crankcase.deckN.*`) so V/flat engines seal both heads.

## Implemented: vectorized valve state with machine error and carbon (`engine_toy/valve_state.py`)

Every valve is one row in flat numpy arrays (cylinder, intake/
exhaust, spring rate, free length, installed height, nominal lift,
rocker ratio, lash, seat concentricity, sag, shim, seat recession,
carbon). The build draws each valve's machine error within real shop
tolerances from a seed of the engine identity, so cylinders differ
and the differences are stable. Derived per valve in one vectorized
pass: seat load, effective lift, open load, float speed (seat
preload scaled, deposit mass lowering it), seat leak (recession,
runout, weak seat, deposit), hot-spot risk. Reduced per cylinder
with bincount/minimum.at: breathing (product of intake lift ratios),
float speed (lowest valve), leak, carbon, hot spot. ~180 us per call
on a six.

The piston loop multiplies each cylinder's combustion strength by its
own breathing x (1 - leak) x float penalty, so every cylinder fires
uniquely; the engine-wide float figures now report the worst
cylinder. Aging runs per tick: sag from cycles and block temperature,
exhaust-seat recession, and carbon that grows under cold, rich,
light-load running (intake valves only when direct-injected; port/
carb intakes are fuel-washed) and burns off above ~620 K under load.
Service: replace_springs, shim_valve, set_lash, decarbonize. The
state persists across stop/start and feeds the parts manifest
(removable_parts(valve_state=...)). Exposed on EngineCycleState as
cylinder_breathing_frac / cylinder_float_rpm / cylinder_valve_factor /
cylinder_carbon_frac / cylinder_hotspot_risk (hot-spot is not yet
wired into the knock model).

## Implemented: engine dressing (`engine_toy/dressing.py`)

`derive_dressing(engine)` reads what the catalogue declares and picks
real installations: lubrication (wet sump with spin-on filter; dry
sump with reserve tank, scavenge pump and a shallow pan on race
fuels; a splash-bath trough with no pump on an antique single; drip
on expanders), air cleaner (paper/foam/gauze element, oil-bath on
antiques, open stacks on velocity-stack intakes), fuel filter
(inline, sediment bowl, diesel water separator), rail (port EFI rail
feeding the port-injector bosses; common rail feeding the direct-
injector bosses), ignition wiring (distributor + coil with a lead to
every plug boss; coil-on-plug packs; magneto on the crank nose driven
at half speed; glow-plug bus; the Otto-Langen's flame port) and the
intake side.

The intake is a DISTRIBUTOR, the mirror of the exhaust collector: N
intake ports gathered into M inlet chambers, ports assigned to
chambers by firing-order slot so every inlet sees evenly spaced
pulses (the dual-plane rule). M comes from a declared
ThrottleBodyAssembly's barrel count, else the typical build (single
carb / two-barrel / four-barrel / double quad on race V8s / one
throttle body / individual throttle bodies for velocity stacks); a
diesel gets one unthrottled inlet. Each runner stubs out of its port,
bends up to the rail level, runs along the bank to its chamber and
bends in (the same algorithm as the headers); on a straight engine
the runners therefore meet in the middle rather than at an end. The
inlet (barrel / throttle body / open elbow) sits on each chamber, one
air cleaner spans the inlets, or each ITB carries its own stack. A
stock log exhaust on a straight engine now dumps from its centre.
All of it goes into the drivetrain graph on the existing production
nodes (plenum, throttle body, fuel rail/bowl, oil pump/pan, ignition
driver), which are moved to where the dressing puts them; drum-
shaped parts declare drum_axis/radius/length and vehicle_mesh draws
them as capped drums. The graph carries a `dressing` report.

Solver honesty: a splash-bath engine has NO oil pump, so the dressing
removes the production pump node and everything that drove or fed it.
The drivetrain solver now tolerates that honestly instead of needing a
placeholder: a rotational edge with a missing endpoint carries no
torque and is flagged (`_EdgeState.missing_endpoint`), and an oil
circuit with no pump node but a splash edge takes its supply from the
crank's own speed (a dipper, at a disclosed 0.4 of a gear pump's
equivalent), which is also why such engines show low oil pressure.

Caveats: the derivations follow the catalogue -- the hit-and-miss
single is not declared carbureted, so it gets a throttle body; the
Merlin is not declared carbureted, so it gets one throttle body
rather than its twin-choke carb. Declaring those fixes the dressing
without touching the code.

## Implemented: crankcase participation (`engine_toy/crankcase_state.py`)

Every crankcase splash-lubricates its bores, so the dressing joins
the oil source (sump, or the trough on a pump-less engine) to every
cylinder's bore bottom with an "oil-splash-path" edge -- a real,
visible path the circuit solver does not flow as a pipe; the
participation is computed per cylinder in flat arrays: an oil film on
the bore deposited by splash (crank speed x sump level), scraped by
the rings and burned past them (film x combustion strength x ring
leak) as the engine's oil consumption and as a carbon source for the
valves (valve_state.age's oil_burn_frac); blow-by gas per fire into
the case, vented through the breather (case pressure, seal loss when
it cannot vent); fuel dilution of the oil on cold rich running that
boils off hot. Ring seal carries seeded machine error per cylinder.
The film and case-gas updates are closed-form exponential
relaxations, so a 2 ms sim tick and an hour of accelerated wear land
on the same physics (verified: 3600 x 1 s == 1 x 3600 s). State on
EngineCycleState: cylinder_oil_film_mg, oil_consumption_ml_per_h,
blowby_l_per_min, crankcase_pressure_kpa, sump_oil_l,
oil_fuel_dilution_frac; service: top_up, change_oil.

Rendering: demo_animate.py draws the dressed engines with
translucent castings and a legend, and animates a chosen engine over
N frames of a 4*pi crank cycle (pistons, rods, crank, valves, lobes
move) to a GIF.

## Implemented: the engine as a mesh with materials, the live pygame view, and rays (`engine_mesh.py`, `mesh_visualizer.py`, `engine_rays.py`)

`engine_mesh.build_engine_mesh(graph, crank_angle, covers_off)` lifts
the dressed design into (static, moving) meshes: vertices, normals,
triangles, and a material id per triangle from one material table.
Materials mirror the spectral analyzer's records (PBR base: albedo,
roughness, opacity; Phong: ambient, spec_strength, shininess), so
`material_table()` can be uploaded to that renderer's SSBOs
unchanged; `export_obj_mtl` writes OBJ + MTL (bake_snapshot now does
this per engine). Moving parts are baked into an `EngineAnimation`
at caller-chosen crank divisions (an integer count over 720 degrees
or an explicit angle list); the game plays frames scaled with rpm by
nearest baked angle and never re-derives geometry; baking is
incremental (`start_animation`/`bake_next`, filling the cycle evenly)
so switching engines never stalls the view. `mesh_primitives.DETAIL`
is the one tessellation knob (the live view bakes at 0.5); springs
render as plain cylinders at the compressed height in the game view
(`spring_style="cylinder"`) and as helices for close-ups/exports.

`mesh_visualizer.MeshVisualizer` draws that mesh on its background
thread with the analyzer's Phong model on the CPU (per-triangle
ambient + diffuse + Blinn specular from the material record,
thermal tint on temperature-carrying castings, translucent parts
composited back-to-front, opaque back faces culled), picks the baked
frame for the sim's live crank angle, overlays the stats (rpm, power,
temperatures, per-cylinder valve factor / float rpm / carbon, oil,
blow-by, case pressure, supply) and exposes a material legend the
pygame app draws under the view. Keys: O covers on/off. Live cost on
a six: ~8k triangles at ~5 fps (pygame's per-polygon draws are the
bottleneck; a vectorised rasteriser is the next step if it matters).

`engine_rays.RayMesh` is the ray interface over the final mesh:
`hits`/`traversals` (entry/exit per part, in order, with thickness
crossed), `pick` (a click), and `penetrate` (a projectile spends
energy per traversal as thickness x calibre area x the material's
toughness, holes what it gets through, stops in what absorbs the
rest; `holes_as_open_ports` hands the holes to the assembly layer as
open ports). `screen_to_ray` inverts the live view's projection, so
a left click on the view picks the part and a right click fires a
7.62 mm / 3 kJ test round along the same ray (main_pygame).

## Standing backlog (for continuity)

From earlier in the project:
- Full pneumatic-circuit generalization (7 real primitives: Volume,
  Restriction, Check valve, Controlled valve, Regulator, Compressor,
  Actuator) — only a slice built (typed tanks, an unloader, the WMI
  flow regulator).
- Stage 4 thermal steady-state solve — conductance Laplacian ready,
  never run.
- Muffler audio DSP; block thermal model coupled to oil/coolant
  circuits; four-corner battery/reservoir derivation.
- Duplicate `GearedCoupling` class in `drivetrain_port.py`.

From this session:
- Rigid pinion clutch primitive (generalize the Otto-Langen ratchet
  catch out of `otto_langen.py` into a reusable `drivetrain_port.py`
  class).
- The Otto-Langen's own mesh geometry (still the same generic fallback
  block an electric motor gets).
- Pneumatic wiring for the turbine/Otto-Langen (e.g. connecting the
  air-motor starter).
- Turbine accessory-drive specificity (still a generic minimal
  accessory set).
- Turbine thrust output + an airspeed dial on the rig (needs a real
  declared nozzle area).
- A fluid coupling (torque converter) as the turbine's default dyno-rig
  coupling instead of the rigid clutch that just slips.
- Newcomen-type atmospheric steam engine + a general boiler/steam
  primitive (parked, not abandoned).
- Richer intake/exhaust pipe-harmonic modeling in the audio synth.
- The dashboard's BUS/electrical line is still piston-only in
  `toy_shared.py`, even though the turbine now has a real circuit
  solve underneath it.

Two background investigations were spawned earlier in the session and
had not reported back as of this note: fixing `idle_regression.py`'s
wrong pass criterion for the Otto-Langen, and investigating whether a
Newcomen engine can honestly reuse the atmospheric engine class's
physics.


## Fluids as a registry, gear cases as a primitive (2026-09-11)

`fluids.py` holds one row per fluid: density, viscosity, surface tension, the
circuit identities that carry it, the labels that name it, its leak colour, and
whether it burns and at what temperature. Everything that used to be five
parallel dictionaries in `hole_emitters.py` is derived from those rows, so
**adding a fluid is adding a row** — transmission fluid and gear oil both went
in that way with no edits to the leak, spray, burst, fire or sound paths.

Lookup is EXACT KEY FIRST. Prose keywords are a fallback for legacy labels only,
longest-match, so "transmission oil" is ATF rather than engine oil.

`gear_cases.py` is the primitive for anything that is a sealed case of gears
running in gear oil: manual gearbox, transaxle, transfer case, differential,
final drive, reduction box. They share one failure surface (lose the oil, block
the breather, cook the film) so they are one type with a kind.

An **automatic transmission is deliberately NOT one of them** — its fluid is a
working fluid, not a lubricant. `automatic_transmission.py` models it as what it
is: a crank-driven pump (no engine, no line pressure, which is why you cannot
push-start one), line pressure applying the clutch packs, a torque converter
going as the square of the speed difference with stator multiplication at stall,
lock-up removing the slip, and all of that slip heat landing in the same fluid,
whose oxidation damage above ~121 C is permanent.

### Parts DECLARE what they are

Identification is by declared attribute (`gear_case`, `automatic_part`,
`part_role`, `fluid`), never by matching substrings of a node identity. One exact
table in `drivetrain_graph.py` declares roles on nodes adopted from the
production subunit; everything built in that file declares its own role where it
is created.

The substring version was not merely ugly, it was wrong: a breather node called
`powertrain.transfer_case_breather` matched the very rule that created it, and
the machine grew a second transfer case out of its own vent fitting.

## Where the compressed air is (`air_vessels.py`)

The pneumatic circuit keeps ONE authoritative stored mass. `air_vessels.py` says
only WHERE that mass is sitting — wet tank, reserve, primary and secondary brake
reservoirs — reading the vessels and the pipe topology off the graph, walking
the chain OUTWARD FROM THE COMPRESSOR (a nearest-neighbour walk gets the
direction wrong when two vessels are equidistant, and decided the reserve was fed
by a brake reservoir, backwards through the protection valve).

It is not a second ledger: `verify_conservation` is float noise, and the split
exists to express the one thing the lump cannot — the PRESSURE-PROTECTION VALVE.
Measured: service side drained to 1.5 %, brake reservoirs held 39 % and 87 %.

## Burst in place, and reset

A destroyed part dumps what it held. `_spill_contained_fluid` creates a
contained-volume emitter (`HoleEmitter.contained_l`, no circuit behind it) at the
part's own position, so a gear case emptying 2.2 L pours under its own head and
stops when empty. Fired from `burst_part` AND from structure loss, so beaten-open
and corroded-through cases spill too.

`reset_damage()` undoes all of it in one place — punctures, emitters, absences,
bursts, fires, charges, fittings, spills AND the fluid those spills took out of
the real reservoirs — and `set_engine` calls it, because a different engine is a
different machine.

## Colours

Refrigeration and everything it chills is icy pale blue (`#9fe0f2`); compressed
air is a deep saturated blue (`#1b3fb8`); hydraulic hardware is the dark red of
its own dyed fluid (`#8c2f3f`). Hydraulic oil (`#d84f66`) and ATF (`#f2607f`)
leak as two neighbouring reds — that conflation is real in a workshop and worth
keeping. Gear oil is the dark brown one.


## The pivot: parts that are not engines (2026-09-11)

Three new modules turn this from an engine simulator into a parts rig.

### `actuators.py` — parametric, real, catalogue-shaped

Linear (double-acting, single-acting-spring, double-rod, telescopic, rodless),
rotary through a limited swing (rack-and-pinion, vane, helical-spline),
continuous (gerotor/orbital oil motors, vane air motors), a closed-loop
`Positioner`, and `GasOverOilStrut`. Every number is computed from bore, rod,
stroke, vane width or displacement.

**FINE POSITIONING is set by the fluid, not the valve.** The cylinder and its
load are a spring-mass system whose stiffness is the trapped fluid's bulk
modulus. Oil is ~1.4 GPa; air's effective modulus is gamma times absolute
pressure, ~0.14 MPa. Four orders of magnitude. Measured on the same 63/35
cylinder with a 50 kg load: oil resonates at 210 Hz, air at 5.6 Hz. Static
error settles to the sensor on either; what cannot be controlled away is
DISTURBANCE DEFLECTION, and a 2 % load step moves the oil axis 60 um and the
air axis 2444 um.

**GAS OVER OIL IS NOT MAD** — it is the oleo-pneumatic strut under every
transport aircraft, Citroen's hydropneumatic suspension, every gas-charged
monotube damper and every accumulator. Gas is the spring because gas
compresses; oil is the damper because oil does not.

### `bench.py` — supply for unsatisfied inlets

`PartRig` runs parts that have no subframe, no mounts and possibly no shaft
output. `unsatisfied_inlets()` finds ports nothing is connected to and
`connect_bench_supplies()` fits a `HydraulicPowerUnit` or `AirSupply` to answer
them, saying so. That is what removes the need to build a cylinder onto the
C18 to see it work.

Oil is a closed loop and comes back to tank; air is used once and exhausted,
and the rig reports both.

### `loadouts.py` — named equipment packages

excavator-arm, loader, tipper, air-tools, landing-gear, servo-press. The point
is `check_against`, and FLOW AND PRESSURE FAIL DIFFERENTLY: short of flow, a
machine does everything slowly; short of pressure, a function will not move at
all. Against the C18's 131 L/min at 210 bar, the excavator arm runs at 25 % of
duty speed and the tipper is fully satisfied.

### Errors found and fixed while building this

1. **Starvation must not cut pressure.** A pump short of flow still makes every
   bar the load asks for; the actuator just moves slower. The first version
   scaled pressure by the flow shortfall, so an undersized pack could not lift
   a load it can certainly lift.
2. **Report DELIVERED force, not surplus.** A cylinder lifting its rated load
   exerts exactly that load. Reporting `gross - friction - load` made a
   correctly working cylinder read as exerting nothing.
3. **Required pressure must beat BREAKAWAY,** not running friction, or a
   stationary rod can never start and sits forever at 97 % of its load.
4. **Buckling length is the extended cylinder, pin to pin,** not the rod alone.
   A 2 m-stroke 100/56 ram buckles at 54 kN against a 165 kN push — buckling is
   a long-cylinder failure and the first version hid it.
5. **A strut's gas volume must exceed what its stroke sweeps.** The first
   defaults swept 2.54 L into 2.0 L of gas and reported a spring force of eight
   million kN. Now floored, and `check()` says so out loud.
6. **An orifice a little too small damps enormously too hard** (square law): a
   6 mm orifice on a 90 mm piston gave 1144 kN at 2 m/s instead of 39 kN.
7. **A `Positioner` wraps a cylinder rather than being one,** so the press
   reported needing no oil at all — a zero that looks like a pass.


## Equipment on a running engine (`equipment.py`), 2026-09-11

`EquipmentRig` is what joins the actuators to the engine. Two rules:

**One stateful machine.** Fitted equipment computes its own demand and DRIVES
`hydraulic_flow_frac` / `hydraulic_load_frac` -- the levers the plant already
integrates -- so pump torque on the crank, relief heating and oil temperature
are consequences of what the equipment really does. Air tools draw out of the
real compressed-air circuit; measured, the C18's air fell 93.7 % -> 89.1 % with
the vessel split still conserving to 4e-16 kg.

**Complementary supply for unsatisfied inlets.** What the engine provides and
what the bench had to add are reported SEPARATELY every tick. The excavator arm
on the C18: engine 127 L/min, bench 361 L/min. That difference is the statement
"this engine cannot run this equipment on its own", which is the whole reason
not to build the parts onto the C18.

### A build-order bug worth remembering

The gear-case and automatic declarations originally ran in the MIDDLE of
`build_drivetrain_graph`, before the transverse conversion replaces the
longitudinal driveline. Consequences on the Camry: a breather for a transfer
case that had since been deleted, an automatic's converter and valve body
attached to a transmission node that no longer existed, and no gear case on the
transaxle because it had not been created yet. **Fittings must be hung on the
machine that is actually being built, which means after it is built.** The block
now runs last, and the automatic attaches to whichever node is the gearbox
(transmission longitudinally, transaxle transversely).

The Camry is now a real automatic (Toyota A140E ratios, 7.2 L of ATF), so that
path is reachable from the catalogue instead of being unreachable code.

## The dashboard actually fits now

`dashboard_lines(..., max_lines=N)` trims to what the caller can draw, and
`main_pygame` computes N from the window (19 px per line from y=14, so 40).
Priority order is deliberate: fitted EQUIPMENT first (what you are operating),
then TANKS (what you are monitoring), then reference readings, which are what
the trim takes. The damage section is separately capped at 14 lines, sorted
worst-leak-first.

The head lost ten lines by dropping text that only ever said nothing was wrong:
redline basis when within limit, valve factors at build tolerance, "anti-lag not
fitted", the lifter spring when nowhere near float, and three status lines
merged into one. Longest untrimmed dashboard across the catalogue: 48 lines,
down from 61.


## Machines you can load like engines (`machines.py`), 2026-09-11

A `Machine` builds the SAME graph document an engine builds -- same node and
edge fields, same declared roles -- so the mesh, ray, damage, leak, fluid and
actuator systems all work on it without knowing what it is. No subframe, no
mounts, no shaft output.

Shipped: `hydraulic-power-pack`, `scissor-lift`, `log-splitter`, `pop-up-turret`.

**View membership is now DECLARED.** `wanted_in_view` is a keyword allowlist
written for engine hardware and was never going to contain "scissor arm". Nodes
set `in_view` and `engine_mesh.declared_in_view` honours it, instead of growing
the list forever.

**The bake had to be told there is no crank.** `significant_crank_angles`
returned 144 frames for a machine with no cylinder layout -- 144 identical
copies of one frame. It returns `[0.0]` now; engines still get 144.

**Two-stage pump** (`bench.HydraulicPowerUnit.second_stage_cc_rev`): both
sections deliver for a fast low-pressure approach, and past the unloading
valve's setting the big section dumps to tank. Measured on the log splitter:
12.8 L/min at 8 bar light, 6.1 L/min at 240 bar with the wedge in the log, ram
171 kN. No controls, no operator input -- which is why every splitter has one.

**Supply must agree with the declared reservoir.** A 12 L scissor lift had a
54 L tank because the power pack kept its own default; `sync_supply_to_parts`
reads the declared reservoir part.

## The hydraulic pop-up turret

Built from the pieces already here, each doing the job it is actually good at.

- **Hoist**: a 125/70-850 cylinder. The reason the machine has a 20 L
  ACCUMULATOR rather than a bigger pump: the pop needs ~300 L/min for two
  seconds and almost nothing afterwards. Measured: full extension in **2.24 s**
  on 15 L of stored oil, with an 11 kW pack making 18.9 L/min to refill it --
  so a second pop in quick succession is slower, which is real.
- **Traverse**: rack-and-pinion on the slew ring, constant torque through 360.
- **Elevation**: an ordinary cylinder against a load that changes sign past the
  balance point, which is what the equilibrator is for.
- **Recoil**: an oleo-pneumatic strut -- the same `GasOverOilStrut` as landing
  gear, because it is the same device solving the same problem. Gas is the
  recuperator, oil through an orifice is the brake.

`size_recoil_for(weapon, recoiling_mass, stroke)` is real gun-mount arithmetic:
the brake must absorb the free-recoil energy inside the stroke (mean force =
energy / stroke), and the orifice follows from that force at the recoil
velocity. Because orifice loss is a SQUARE law, a system sized for one charge is
badly wrong for another, which is exactly why real guns with variable charges
use a metering pin rather than a fixed hole.

Measured, Bofors 40 mm L/70 on a 255 kg recoiling mass:

| | |
|---|---|
| impulse | 1348 N.s |
| free recoil | 5.29 m/s, 3565 J |
| unbraked, arriving in 2 ms | 674 kN into the mount |
| braked | 300 mm stroke, stopped in 141 ms, peak 21 kN |

A 32x reduction, not bottomed out. The propellant gas is a third of that
impulse (it leaves at ~1.5x projectile speed), which is why a muzzle brake works
on the gas rather than on the shell.

## OPEN: the GL renderer will not draw a machine

`snapshot.render_machine` exits with status 0, writes no PNG, and raises
nothing -- even wrapped in try/except, even with the bake reduced to one frame.
The non-GL path is fully verified (mesh 1296 verts for the turret, RayMesh
traversals, punctures, emitters, bursts), so this is the renderer rather than
the machine. Not diagnosed.

Machine materials (`armour-plate`, `gun-steel`) also fall through to "other" in
the material table, so shots report the right physics with a generic label.


## The production-vocabulary turret (`turret_production.py`), 2026-09-11

The architecture the user set: the production game physics owns motion; the
engine sim is an INTERIOR, COORDINATED sim that prepares the values at the
boundary. Nothing in this module prescribes where a part goes.

### What the production graph already had

Reading `abstract_ui_vehicles._vehicle_mechanical_graph` first was the
important step: **there is already a turret in it** -- `turret.mount.*` on an
`actuated-damped-clutch-gimbal-base`, `gimbal-yaw-bearing`,
`gimbal-pitch-bearing`, and recoil as a `point-impulse-wrench-coupling` with
`load_path="individual-shot-recoil-r-cross-impulse"`. There is also a
`linear-hydraulic-actuator` family (`force-limited-alignment-strain-relief-v1`)
whose state begins `commanded_rest_length_m` -- a member whose REST LENGTH is
commanded, with series relief so it is backdrivable rather than a rigid
position source. All of that is reused rather than reinvented.

The contract: **nodes carry wrenches; edges are constraints or power-transfer
laws.** `edge()` derives rest length from the real node positions, section from
the radius, an `elastic-plastic-member-with-shear-fracture` damage law with
yield forces at area x 250 MPa x 0.72, and six-axis Kelvin-Voigt bushings.

### What we had to supply

The production turret is a small body-pin mount with no hoist and no
hydraulics, so everything else is ours: a triangulated WELL (perimeter rails,
both face diagonals, wall braces -- a four-bar rectangle is a mechanism, only a
triangulated one is a structure), a triangulated CARRIAGE, three prismatic GUIDE
STRUTS so the assembly rises straight instead of swinging on the ram, the HOIST
as a `linear-hydraulic-actuator` member, a distributed SLEW RACE, and a TRUNNION
STANDARD.

23 nodes, 56 members, 1936 kg authored.

### The structural check earned its place twice

`ProductionGraph.check()` flags any free node with fewer than three load paths.
It caught two real defects that would have been silent:

1. **The slew ring as a single point.** A turret hangs its whole rotating mass
   and its firing moment on the bearing, and one central connection cannot carry
   a moment at all -- it is a ball joint. Now authored as four race pads with a
   `gimbal-yaw-bearing` each, tied into a rigid ring.
2. **The trunnion left hanging** when the traverse drive was rerouted to push
   between the carriage and the race ring (where a real slew pinion pushes). The
   pitch node needed its pedestal plus two braces out to the race -- the triangle
   a real trunnion standard makes.

### The boundary, and what is NOT published

`hoist_rest_length_from_fluid` is the whole interface. The interior sim knows
the fluid: what the pack and accumulator deliver this tick and at what pressure.
Volume in a known bore is a length, so it publishes `commanded_rest_length_m`
and the axial force the fluid behind the member can support.

It does NOT publish a position. A position would be this sim asserting an
outcome it has no right to assert -- the solver may put the carriage somewhere
else entirely because something is sitting on the hatch or a guide has jammed.

Measured: full command in 2.04 s on accumulator flow, 309 L/min at 250 bar,
307 kN of available axial force.


## The chassis editor and the segment palette (2026-09-11)

The user's correction, and it reframes the whole direction: they do not want me
designing turrets that are convenient to program. They want **the power to make
linkages, draw them into a 3D space, and specify each edge type**, using the
real production engine as the object model and a chassis editor to build objects
from the game's segment types, expanded by our work.

So objects stop being builder functions. An object is a list of nodes and a list
of segments, each segment naming a type from the palette.

### `segment_types.py` -- the palette

Extracted from the production graph rather than invented: **75 distinct edge
constraint kinds** are authored by `_vehicle_mechanical_graph`. The catalogue
carries 66 of them plus ours, grouped into eight families with the parameters
each kind takes and defaults an editor can offer:

| family | what the physics does with it |
|---|---|
| structural | a load path with a rest length; yields and fractures |
| joint | frees some degrees of freedom, bushing carries the rest |
| spring | stores bounded energy, opposes relative velocity |
| actuator | REST LENGTH commanded from outside, series compliance |
| drive | transmits torque rather than force |
| routed | carries fluid or current; no damage law, holds nothing up |
| breakable | fails first at a declared load, where the designer chose |
| impulse | a discrete event's load path -- a shot, an impact |

Three entries are ours (`origin="engine-toy"`): the prismatic guide strut, the
oleo-pneumatic strut, and the commanded-rest-length hoist ram. **Expanding the
palette is the normal case** -- adding a row is all it takes, and the editor
picks it up with no further work.

### `chassis_editor.py`

Click empty space to place a node on the work plane, click two nodes to join
them with the current segment type, right-click to delete. TAB cycles family,
brackets cycle type within it, PgUp/PgDn moves the work plane, V validates, E
exports the production graph, S/L save and load.

The work plane is the answer to the obvious problem: a mouse gives two numbers
and a position needs three. Raise the plane, click, lower it, click again.

Picking is done in WORLD space against the pick ray rather than in screen space
against projected points -- it needs no camera matrix and behaves the same at
any orientation.

The view gained `_elevation` and `_zoom` (defaults reproduce the old fixed 0.40
rise and fitted distance exactly), because the editor needs to look at a thing
from a chosen angle and the camera only had azimuth.

Verified headless: 8 nodes and 18 members drawn programmatically validate,
export with real rest lengths and an 81 kN member yield, and round-trip through
the document format byte-identically. The viewport renders nodes as bodies and
segments as tubes with the hydraulic ram visibly heavier than the rigid members,
and `pick_ray` resolves.


## The editor belongs to the validator rig (`rig_editor.py`)

The user pointed at "the rig" and "the validator", and reading
`abstract_ui_validator_rig.py` settled several open design questions at once.
Its own first sentence is the governing one:

    "The rig is a world object, not a page, modal, scheduler, or second physics
     runtime."

So the editor does not get its own viewport, clock or physics. It is a mode of
the rig's construction stage, and the object being edited is held in the world
where you can walk around it.

**What the rig already provides, and we do not have to build:**

- A **STAGE** beside the rig box (`form.stage_identity`, its own offset and half
  extent) -- the workspace an object under construction sits on.
- **CLAMPS.** `place_validator_support` installs a body/world support "through
  the same actions as a player", and is careful that "placement does not imply a
  second physics runtime or prescribe the body's motion". Bound through
  `bind_placed_rig_point` it is a SOFT hold: per-axis stiffness and damping with
  a maximum force, not a weld. The real numbers from
  `tools/run_vehicle_native_assembly.py` are 80/120/80 kN/m, 0.8/1.2/0.8 kN.s/m,
  capped at 60 kN -- stiffer vertically, because mostly it is holding weight.
  Give a clamp a target velocity and it DRAGS the held body, which is how
  clamped work gets moved around without anyone teleporting it.
- A **MATERIAL LEDGER**: parts go `projected` -> `awaiting-material` ->
  `installed`, drawn first as faint construction-line projection and then with
  real shaded material. An editor document maps straight onto it.
- A **TICK ENVELOPE**: `assigned-ticks`, `[tick, dt, subdt, substeps]`, dispatch
  order `validator_rig, validator, initial_vehicle`, `inactive_until_called`,
  and "all-consumers-receive-the-same-tick-envelope".

`RigEditSession` is the translation: an editor document becomes construction
parts grouped by segment family (not one line item per member -- the ledger is
what a player feeds material into, and a hundred items is a spreadsheet, not a
build), clamps hold chosen nodes, and `tick()` prepares values and integrates
nothing.

Verified: a drawn 10-node / 19-segment well costs 29 material units across four
family parts, three clamps hold it, dragging one sets a target velocity rather
than a position, and a tick at dt = 1/240 prepares three clamp commands.


## Scope, fire control and the baked solution (2026-09-11)

### The scope (`scope.py`)

Three facts carry the whole thing. The sight sits above the bore, so zeroing is
a SOLVED launch angle and the trajectory crosses the line of sight twice. Angle
is the only unit that travels, so adjustments are angular and one MOA is kept at
its exact 0.29089 mrad rather than the "inch at a hundred yards" rounding. Drift
is not a sideways push -- it falls out of drag acting on velocity relative to
the air, which the trajectory law already integrates.

Verified against published .308 data with a 100 m zero: 12.2 cm at 200 m,
100.9 cm at 400 m, 306.8 cm at 600 m, and 107 cm of drift at 600 m in a 4.5 m/s
crosswind. Time of flight 0.95 s at 600 m.

The reticle knows the focal-plane trap: one mil-dot covers 40 cm at 400 m on a
first focal plane scope and 66.7 cm on a second focal plane one used at 6x when
it is only true at 10x.

### The bug that made everything miss

Scoped shots hit at 200 m and missed beyond it. The arithmetic checked out --
the two trace conventions differ by exactly the zero angle over range (23.9,
47.9, 71.8 cm at 200/400/600) -- so it looked vertical. Tracing the actual
penetration ray showed it passing the target plane 4.5 cm high and **89 cm to
the side**: `windage_mrad` is already the NEGATIVE of the drift, so subtracting
it aimed further downwind and doubled the miss. One sign. After the fix the
scoped gun hits at 200, 400, 600 and 800 m while iron sights miss at all four.

### The bake (`fire_control.SolutionTable`)

Each cartridge is traced ONCE across a range grid and stored. A solution is then
a lookup: **7 us** against seconds for a full trace. Wind is one extra trace at a
reference crosswind giving a drift-per-m/s coefficient, because drift is very
nearly linear in crosswind at fixed range.

Baked: 7.62 NATO (15.8 s) and the turret's 40 mm L/70 (22.5 s). The 40 mm went
into `calibres.py` proper rather than getting a parallel path -- 444 kJ at the
muzzle, 886 m/s at 500 m, 1.13 s to 1000 m.

### The solve is iterative, which is why it takes time

A firing solution is a FIXED POINT: where the target will be depends on the
flight time, which depends on how far away that is. So the time penalty is a
real iteration count rather than a progress bar, and showing the iterates is
showing the computer work.

### `TurretFireControl` -- two clocks, two reticles

COMPUTER TIME is one solver iteration per cycle; a faster computer shortens it
and a faster turret does not. SLEW TIME is the mount physically getting there at
whatever its hydraulics allow; a faster computer does nothing for it. The
SOLUTION reticle (amber) is where the computer believes the gun should point and
settles as the iteration converges; the AIM reticle (white) is where the barrel
is and chases at the mount's rate. The gap between them IS the penalty.

Line of sight is the same RayMesh a shot uses, so occlusion cannot disagree with
what a bullet would do, and a hostile behind a wall is not a target.

Measured on the pop-up turret, slew rates read from its own actuators (60 deg/s
traverse, 15.6 deg/s elevation): acquired a hostile at 1499 m, ignored a neutral
truck, solution converged in 8 iterations (0.4 s), on target at 0.62 s.


## The three-mechanism cannon + independent MG (2026-09-11/12)

User's design, and it is the correct architecture in the well-studied sense:
coarse-coarse-fine staged aiming, exactly how real precision mounts separate
"get roughly there fast" from "hold precisely."

`turret_production.build_gimbal_cannon_station(bore_mm=40.0)`:

1. **RING, multi-pinion.** Three parallel `gimbal-yaw-bearing` members between
   carriage and race (electric x2 + hydraulic), each declaring its own drive
   technology -- the hardware `slew_drive.SlewDrive` already models the
   behaviour for (load sharing, redundancy, mode selection).
2. **TRUNNION at the barrel's own C.O.G.**, now PARAMETRIC: barrel mass/length
   derive from `calibres.cannon(bore_mm)` + `ballistics.mass_of`, so the
   pivot re-balances automatically at every bore. Verified 0.00 mm horizontal
   offset / 0.0 N.m residual weight moment at 30, 40, 57, 76, 100 mm, all
   passing the structural check.
3. **SIX-ACTUATOR FINE-AIM PLATFORM** at the breech -- a literal Stewart
   platform reusing `armature.py`'s proven pose solver rather than a fourth
   solver. Short-stroke actuators (+-25 mm). The recoil slide now rides ON
   this platform (not the cradle), so recoil travel is relative to the stage
   doing the aiming. Verified: commanded a 0.5 deg twitch, achieved 0.502 deg
   in 2 iterations, largest single-leg travel 0.594 mm of 25 mm available --
   enormous headroom for real sub-degree correction.

**Independent machine gun**: its own yaw+pitch gimbal on the carriage,
structurally separate from the cannon's ring, [-15, 80] deg elevation vs the
cannon's [-8, 42], own trunnion brace.

### Refactor: shared base extracted

`_build_well_carriage_and_race()` factored out of the single-gun
`build_pop_up_turret` so `build_gimbal_cannon_station` cannot drift from it
the way `machines.py`'s turret already had (see below). Old builder reverified
unchanged (26 nodes, 63 edges, PASS).

### Two real bugs found building this

1. **`GasOverOilStrut` had a `step()` that takes velocity as INPUT, not
   dynamics.** Re-stamping the same velocity every tick doesn't decelerate
   anything. Added `dynamics_step(dt, mass_kg)`: integrates from current
   velocity under the strut's own spring+damper force, semi-implicit.
2. **Sign bug in `fire_recoil`:** subtracted the impulse instead of adding
   it (module convention: positive velocity = increasing compression). Every
   shot clamped to 0 mm and "partial recovery" was unmeasurable until fixed.
   After the fix, at the gun's real 330 rd/min cyclic rate, round 2 lands
   with the slide still at 277 mm of 320 mm stroke -- genuine partial
   recovery, not forced by an artificial rate.

### OPEN: two turret definitions still exist

`machines.py`'s `pop_up_turret()` (what the running demo/station/fire-control
actually use) and `turret_production.py`'s graphs (structural check, real
materials, chassis-editor palette) are still independent files. The recoil
DYNAMICS fix is live in the demo (generic `machine.recoil`/`recoiling_mass_kg`
fields). The new three-mechanism geometry is NOT yet wired into what renders
on screen -- reconciling them is the next real task, not done here.

## The canonical turret is on screen, and the bore question was re-answered

`turret_production.build_gimbal_cannon_station` is now what the demo
loads (`machines.gimbal_cannon_station`, roster name
`gimbal-cannon-station`). The two turret definitions are reconciled:
`Machine.production_graph` carries the authored structure and the
machine layer adds only the plant that drives it. The Linkage
kinematic stand-in is not used by it at all.

**Bodies declare what carries them.** `ProductionGraph.node` stamps
`motion_group` on everything it authors (frame / carriage / traverse /
elevation / fine / mg-yaw / mg-pitch, with `recoils=True` for the tube).
`pose_frames` composes the chain from those declarations and reads its
own pivots out of the document, so a station placed on a pole four
metres away poses correctly -- the yaw axis comes from `carriage.hub`,
not from the origin, which was a real bug.

**The mesh is built once and its vertices are moved.** Meshing costs
~100 ms; `articulation.ArticulatedMesh` poses it in ~4 ms, exactly, with
no baked quantisation -- which matters because quantising the aim
destroys the one thing the fine stage exists for. Members spanning two
motion groups are stretched rather than transformed rigidly, so the
elevation ram visibly extends. The posed mesh and the posed document
agree to 0.0000 mm.

**Captive pinion on a ring is a parametric option**
(`slew_drive.CaptivePinionRing`, authored by
`turret_production.author_captive_pinion_ring`, segment types
`captive-pinion-mesh` and `pinion-carrier-bearing`). Three declared
arrangements, measured at 12 kN tangential on a 1.1 m ring:

| arrangement | shaft bending | carrier | ratio |
|---|---|---|---|
| single ring, cantilevered | 121 MPa | 12.8 kN | 7.7:1 @ 97% |
| ring on top, same body | 25 MPa | 12.0 kN | 7.7:1 @ 97% |
| ring on top, grounded to frame | 25 MPa | 12.0 kN | 69:1 @ 75% |

The second ring opposes the separating force and straddles the shaft;
grounding it instead makes the pinion a planet and two teeth of
difference buys the whole ratio. Nothing picks between them by name.

### Bore vs forces, retested on the balanced mount

Two defects surfaced doing it, both now fixed:

1. **The slide sizing was authored twice.** The good arithmetic was in
   `machines.size_recoil_for`; the turret builder had its own scaling
   rules that sized the orifice off the GUN's bore against a recoil
   piston fixed at 95 mm. Square law, so small bores made 418 kN in 4 mm
   and large bores coasted into the stop undamped. Now one law,
   `actuators.size_recoil_slide`, and the recoil piston scales with the
   gun.
2. **A fixed orifice cannot hold a stroke reserve.** Force goes with
   v-squared, so it is all spike and then nothing; sizing such a brake to
   stop in 80% of travel is impossible at any diameter. `GasOverOilStrut`
   gained a declared `metering_pin`, which is what every real gun firing
   more than one charge has, and the reserve became real.

With those fixed (trunnion at the barrel's CG, tube sliding on the fine
platform, brake sized to 80% of stroke behind a metering pin):

| bore | barrel | stroke used | peak | weakest member | verdict |
|---|---|---|---|---|---|
| 20 mm | 18 kg | 44% | 4.3 kN | recoil_slide | 53x |
| 30 mm | 26 kg | 71% | 26.9 kN | recoil_slide | 8.4x |
| 40 mm | 47 kg | 76% | 83.1 kN | recoil_slide | 2.7x |
| 57 mm | 136 kg | 77% | 193 kN | trunnion_brace | 1.7x |
| 76 mm | 322 kg | 78% | 333 kN | trunnion_brace | 0.98x -- marginal |
| 100 mm | 734 kg | 90% | 375 kN | trunnion_brace | 0.87x -- overloaded |
| 120 mm | 1268 kg | 96% | 476 kN | trunnion_brace | 0.68x -- overloaded |

**The failure mode moved.** It used to be "recoil runs out of travel"
at 57 mm and up; now nothing bottoms out through 120 mm and the limit is
a structural member -- `turret.trunnion_brace`, whose radius does not
scale with the gun while everything around it does. 57 mm is comfortable
where it used to be impossible. 76 mm is on the line.

### Open

- **Barrel sleeves** (user's idea, not built): a common cradle, trunnion
  and slide with a sleeve taking interchangeable tubes is the right
  shape for this, and the table above is why -- the mount's limit is a
  fixed-radius brace, so standardising the mount and varying only the
  tube is exactly the decomposition the numbers already imply. Sizing
  the brace for the largest sleeved bore makes the whole family work.
- **Reload** has no mechanism at all yet: no magazine, no feed, no
  rounds-remaining. `mass-and-volume-limited-magazine` exists as a node
  type and nothing uses it.
- The station renders very dark next to the engine -- material or
  lighting, not geometry; the structure is all there and correct.

## The fine stage is not a firing load path — the firing lock

Measuring the shot through the six fine-aim legs said so loudly. Resolved
onto legs that lie nearly flat, one shot put **93 kN through a 20 mm
cylinder at 40 mm bore — 296 MPa of oil in a 25 MPa actuator** — and
375 kN at 76 mm. That is not a leg that needs to be bigger. With the shot
bypassing them the same legs carry **0.7 to 7.4 MPa**, so they were
right-sized all along and were merely being asked to do somebody else's
job.

`turret.firing_lock.{l,r}` (segment type `firing-lock-clamp`) ties the
cradle straight to the trunnion. Sequence is release → fine aim → clamp →
fire; the legs sit in parallel with the clamp carrying nothing.

**It is a FRICTION clamp, and that is not an implementation detail.**
Anything form-fitting — a cone, a taper, a detent, a tooth — centres
itself as it closes, which drags the gun back to nominal and throws away
the very correction the fine stage just made. Only a clamp that holds
wherever it is shut can be closed after aiming. Real hardware: hydraulic
rail brakes, shrink-disc shaft clamps.

**Sized from the standard breech, not the fitted barrel**
(`standard_breech_bore_mm`, default 76). Two clamps of 606 mm x 42 mm at
40 MPa and mu=0.15 hold 480 kN, which is 1.5x the worst shot the standard
breech can fire. A 40 mm barrel and a 76 mm barrel get the *same* lock,
same trunnion, same braces — which is the whole structural return on a
standard breech, and the reason the barrel becomes the only variable.

**Cycle cost**, 35 ms to engage and 35 ms to release:

| bore | rate | period | lock cycle | left for aiming |
|---|---|---|---|---|
| 30 mm | 381/min | 157 ms | 70 ms | 47 ms |
| 40 mm | 330/min | 182 ms | 70 ms | 72 ms |
| 57 mm | 276/min | 217 ms | 70 ms | 107 ms |
| 76 mm | 239/min | 251 ms | 70 ms | 141 ms |

Fine aim every round is affordable at every bore in the family. Holding
the clamp shut through a burst and re-aiming between bursts is the
cheaper doctrine and is what the sim should offer as a choice.

## The preloaded thrust seat — which supersedes the firing lock

The lock above works, but it is the wrong shape of answer. It solves the
problem by SWITCHING, and pays 70 ms a round for it. Asking "what if the
struts had massive preloaded springs" produced the better one.

**Recoil is axial; aiming is angular. They are orthogonal.** So there is
no need to alternate between carrying the shot and being able to aim: a
SPHERICAL seat is rigid along the bore and free in rotation at the same
time. `turret.thrust_seat` (fine.hub -> turret.pitch) takes the shot
straight into the trunnion, around the fine stage, permanently.

**Load divides by stiffness, and there is no contest.** A 20 mm leg's
stiffness is its oil column, beta*A/L = **2.8 MN/m** -- so soft that one
40 mm shot would squeeze it 28 mm and a 76 mm shot 113 mm, through an
actuator with 25 mm of travel. The steel seat in parallel is
**58 905 MN/m**. The six legs take **0.03%** of the shot: ninety grams'
worth of force, 0.0 MPa of their 25. The leg being soft is not the
problem; it is the entire mechanism.

**A series spring inside the leg would do almost nothing** -- worth
recording because it is the intuitive place to put one. The oil is
already the soft element: a 10 MN/m stack in series with 2.8 MN/m of oil
gives 2.2 MN/m, moving the load share from 0.03% to 0.023%. The spring
must act in PARALLEL, holding the gun into the seat.

**What the preload is actually fighting is the recuperator, not the
shot.** Recoil pushes the gun INTO the seat. The gas spring shoving the
tube back to battery is what can lift it off. `turret.seat_preload.{0,1,2}`
(`belleville-preload-stack`), sized at 1.4x that: 19.4 kN total at 40 mm,
71.2 kN at 76 mm, 125 kN at 100 mm -- three ordinary disc-spring stacks.
They also take the lash out of every joint they cross, which is pointing
error the coarse drives cannot control away.

Seat bearing stress is 18 MPa of an allowable 120, so the 75 mm seat is
generously oversized and could shrink.

**Aiming authority is unharmed.** About the seat centre, 1 degree of
pitch costs 6.0 mm of a 25 mm leg, so the fine stage keeps roughly +-4
degrees of authority -- far more than the fraction of a degree it exists
to deliver.

**The real consequence: the platform is now redundantly constrained.**
The six legs span all 6 rigid-body DOF (Jacobian rank 6); the seat
removes 3. That leaves 3 redundancies, so the legs share load in
translation and will fight each other there. Preload makes this benign --
it is how preloaded machine-tool structures are built deliberately -- but
**the pose solver must be restricted to rotations**, or it will happily
command translations that do nothing but redistribute preload.

`firing_lock` is now a parameter defaulting to **False**. The clamp is
kept as a declared option for travel, for a gun left laid on a bearing,
and as a second path if a preload stack is lost -- not as what makes
firing possible.
