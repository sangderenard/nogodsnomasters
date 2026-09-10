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

## Confirmed: the real connection point, and it's already engine-shaped

Checked directly in `abstract_ui_vehicles.py` rather than assumed.
`external_hub_torque_{wheel}` (line 635 default, used at line 2762) is
real but wheel-level — meant for per-wheel interventions (an in-hub
motor, traction-control-style corrections), not a powertrain.

The actual engine-shaped port is one axle level up:
`external_differential_wrench_torque_{axle}` (front/rear), paired with
`external_differential_inertia_{axle}` and
`differential_wrench_shaft_omega_{axle}` (lines 637-642 defaults,
real physics at lines 2782-2804). Per tick, the differential shaft's
own integration is:

```
shaft_input_torque = drive_torque * axle_drive_fraction + center_torque
                      + external_differential_wrench_torque_{axle}
shaft_inertia = differential_brake_rotor_inertia + external_differential_inertia_{axle}
free_shaft_omega = shaft_omega + dt * (shaft_input_torque - shaft_output_torque) / shaft_inertia
```

So the real contract, already live in production, is exactly three
named scalars per axle:
- **in**: `external_differential_wrench_torque_{axle}` — the engine's
  real torque output this tick, Nm.
- **in**: `external_differential_inertia_{axle}` — the engine's own
  rotating inertia reflected at this shaft, kg·m².
- **out** (read back next tick, the same one-tick-lag convention
  `engine_cycle_sim.py` already uses everywhere): `differential_wrench_
  shaft_omega_{axle}` — the shaft's real current speed, what the
  engine's own torque curve needs as input.

This is structurally the *same shape* as the toy's own dyno-rig
junction (`EngineCycleSim._brake_junction`/`_load_omega`, an engine
reading a load shaft's speed and supplying torque back against it) —
just the toy's dyno drum swapped for the real vehicle's differential
shaft. A baked engine package doesn't need a new contract invented for
it; it needs to supply a torque(shaft_omega, internal_state) relation
shaped to feed exactly these three names.

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

## Practical next step (not yet started)

Look at the exact real shape `external_hub_torque_{wheel}` and its
neighbors expect — units, whether it's a per-tick value or a declared
curve, timing — before extending `engine_baker.py` to produce a torque
package aimed at that same contract. Get the target contract right
before building toward it.

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
