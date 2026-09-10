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
