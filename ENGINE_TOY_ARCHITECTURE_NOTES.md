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

## Practical next step (not yet started)

The classification rule for everything besides the fuel tank/pump is
still open — coolant, starter battery, exhaust, pneumatic lines, and
so on each need a real answer, not a guess, before `_SOURCING_OVERRIDES`
grows. The mount-correlation tool also has no real cage source yet
(the game doesn't supply one today) — it's verified correct against a
synthetic cage, not yet wired to anything real.

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
