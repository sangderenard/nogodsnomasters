# Continuation report

Written at the end of a long session on the gimbal cannon station. Two
purposes: list what was asked for and is still undone, and state the
architectural requirement that should govern how the rest of it gets
built.

---

## 0. The requirement that governs everything below

**Everything must be able to run in the genuine WebAssembly engine
style.** Not "could be ported later" — built that way now, because the
cost of retrofitting is the whole reason the backlog below is shaped
the way it is.

The game engine is `turing/src/compiler/abstract_ui_*`. Its contract for
a piece of physics is established and demonstrated by
`abstract_ui_physics.symbolic_world_physics_wasm_plugin`:

```
a law authored as simultaneous SymPy equations
    -> compile_sympy_equations       SymPy -> ProcessGraph -> SSA
    -> emit_ssa_function_to_wasm     SSA -> WebAssembly, directly
    -> WorldWasmPlugin               published through the world ABI,
                                     with a declared capability
```

The equations are "the only numerical authority". The host supplies
parameters through the published arena, so they stay live-editable
without recompiling. **The host schedules it** — the law does not own
its own stepping, which is what makes a component optional and
reschedulable.

### What is actually through that door today

Exactly one thing:

```
identity   abstract-ui/plugins/structural-beam-dynamics
capability physics   source sympy
wasm       1058 bytes, sha256:2a832663a3a33bbc...
operations 95
inputs     12   outputs 6   parameters 12 live-editable
```

That is `beam_theory.symbolic_beam_dynamics_equations`, published by
`structure_plugin.py`. Getting it through required three fixes in the
compiler, all of which are landed and all of which benefit every future
law:

- **`x**4` did not lower at all.** The WASM emitter spelled `x**2` as a
  multiply and stopped. A whole-number power *is* repeated
  multiplication — not a reduction, not an identity, not a policy
  choice — so it now unrolls to 8 under every contract. Without it
  `π/4(ro⁴ − ri⁴)`, the most ordinary expression in beam theory, could
  not reach the target.
- **`Le` and `Select` had no spelling**, so any law containing a
  `Piecewise` — steel softening with temperature, a contact switching
  on, a limit saturating — came back "no direct scalar WASM spelling".
  The opcodes were sitting in `wasm_binary`'s table unused.
  `Select(mask, when_true, when_false)`, same argument order as the
  LLVM lane, so one SSA program means one thing on both targets.
- **The sqrt family needs `work_contract="deploy"`.** `develop` refuses
  it deliberately and a natural frequency is a square root, so under
  `develop` this law can never lower. Declared explicitly rather than
  worked around.

### What is NOT through that door — the real gap

Everything else. All of it is Python that happens to produce numbers:

| system | where it lives | lane today |
|---|---|---|
| frame/stiffness solve | `frame_solver.py` | numpy, host-side |
| ring joint negotiation | `ring_joints.py` | numpy, host-side |
| annulus / plate / boxed ring | `surfaces.py` | pure Python geometry |
| internal channels | `channels.py` | pure Python |
| actuators, oleo, MR damper | `actuators.py` | pure Python |
| interior ballistics | `interior_ballistics.py` | SymPy -> **LLVM only** |
| member bank (beam, batched) | `structure_native.py` | SymPy -> **LLVM only** |
| traction/clutch state | `surfaces.traction_state` | pure Python |

The two LLVM ones are the closest — they are already SymPy-authored and
already lower through the compiler, so they need a WASM emit and a
plugin wrapper rather than a rewrite. **Do those first**; they are the
cheap proof that the lane generalises.

`SSAReferenceEvaluator` runs the same SSA in Python/numpy, which is the
honest test lane: a law that agrees between the reference evaluator and
the WASM artifact is a law that has actually been compiled rather than
merely transcribed. Nothing currently uses it. **Wire it as the parity
check** — the repo already learned this lesson in `frame_parity.py` for
the vehicle lane.

### And the scheduler

`abstract_ui_dynamics` has `DynamicsSpace` / `DynamicsLane` /
`PhysicsStage`, with stages carrying inputs, outputs, a selection, a
dispatch (`compute-shader`) and a `status` of `selected-unbound`. That
is the authoritative scheduling the components are supposed to be bound
into. **Nothing built this session is registered as a stage.**

This is the specific thing I got wrong twice and it should not be got
wrong a third time:

- `live_scene.LiveStructure` steps the modal oscillator in its own
  Python loop, on its own substep count.
- I first invented a stability margin (0.15), then took one from
  `drivetrain_graph.DrivetrainSolver.STABILITY_MARGIN` (0.03) — which is
  the **engine engine's** integrator, a separate thing for engine items,
  and not the authority for a game-engine law either.

The correct shape, confirmed against `engine_cycle_sim._clutch_substep_plan`:
**one outer dt is handed to every subsystem, and each subdivides that dt
by its own stiffest term.** Not independent clocks. The subdivision is
the subsystem's business; the dt is not.

---

## 1. Never actually seen running

The single most important item. Multiple windowed demos exist and the
user has not successfully watched any of them:

- `live_scene.py` — MachineSim ticking the production graph, structure
  solved per frame, mouse orbit. Opened once, ran, closed.
- `drum_reference.py --live` — the joint on its own, orbitable.
- `drum_drive_trial.py --live` — clutches worked through six states.
- `slew_trial.py --live` — the whip test.
- `firing_frame_view.py --live` — the firing trial.

**The export path and the window path diverged and I kept running the
export path.** `slew_trial.py` with no flag writes an animated PNG
headlessly; only `--live` opens a window. Every one of these should
default to the window and take `--export` for the file, not the other
way round.

One real bug was found and fixed here and is worth remembering:
`EngineGLView.render()` draws into an offscreen FBO and returns a numpy
copy — correct for writing a PNG, useless for a window, because
`display.flip()` then presents a default framebuffer nothing ever drew
to. The live path is `render_gpu()` + `gl_compositor.Compositor.
draw_texture`. A second one: `_paint` registering materials **after**
`EngineGLView` is constructed leaves the renderer's table stale and
every part draws blank. Paint first, construct second.

## 2. The slew trial, unfinished

`slew_trial.py` is written and has never completed a run. It anchors the
outriggers, adds the 1400 kg counterweight on the drum's back rim, and
whips ±60° at full drive torque, solving at every instant with:

```
TANGENTIAL   m * alpha * r
CENTRIPETAL  m * omega^2 * r
WEIGHT       m * g
```

Known before it ran: turning mass 16 080 kg, centre of mass 1.240 m off
the traverse axis, 196 kN·m of standing offset. That residue is the
point — a balanced turntable would tell you nothing a static solve
could not.

Reversals are the load case, not top speed: at the ends of the sweep the
drive pushes one way while the momentum still goes the other.

## 3. The drive is not connected to the turret

The drum joint is finished as a **reference article** and is not wired
into the machine:

- `turret.traverse.mesh_lower.*` and `mesh_upper.*` still mesh
  `turret.yaw.*` and `turret.traverse.upper_ring.*` — the **old** ring.
- The six donuts drive nothing in `turret_production`.
- Both tracks (outer hanging from the top annulus, inner standing on the
  bottom) are on the same boxed cell in the turret build, so in that
  build a donut has nothing to react against. In `drum_drive_trial` the
  bottom annulus is locked to world, which is why it works there.

This is the topology question that was raised and deferred. It has to be
answered before the drive means anything in the machine.

## 4. Numbers that do not close

**Cooling.** 251.5 kW of rotor losses into oil that can carry 106.3 kW
at a 30 K rise. Short by 145 kW.

**Supply vs geometry.** The oil flow was sized from a declared 45 kW per
unit; the torques were computed from swept volume at full pressure. The
two do not describe the same machine: 129 L/min through 6063 cc/rev is
**21 rpm**, not the 600 the loss figure assumed. Either the rotors make
26.7 kN·m at ~21 rpm (plausible for a turret, where slew is slow and
torque is the point) or the supply has to grow. Unresolved.

**Air line velocity.** 51.6 m/s falls out of an equal-dynamic-head rule
against 10 bar air. Consistent, but high for air — noise and erosion
usually cap it near 20–30 m/s.

**The 110 kN frame limit** used throughout the recoil sizing is still an
invented number, never derived from the cradle or trunnion sections.

## 5. Articulations: declared, released, solved — not wrenched

`articulation_audit.py` grades every joint DECLARED → RELEASED → LAWED →
WRENCHED. At last run: **142 articulating members, all declared, all
released, all in the frame solve, 0 wrenched.**

`ring_joints.negotiate()` now writes 30 wrenches (8 track pads, 8
rollers, 8 hold-downs, 6 meshes) with 100% closure by construction.
**Every actuator is still unwrenched** — the laws exist in `actuators.py`
and nothing writes a result back onto the graph, so the beam solve sees
geometry and sections and never a force any actuator is developing.

`turret.equilibrator` still has no `spring_rate_n_per_m` at all. With
the trunnion now on the balance point its job is nearly nil, which is
worth deciding rather than defaulting.

## 6. Compiler defect, routed around not fixed

Reproducible in four lines:

```python
def f(m, n, a, q):
    for s in range(m):
        for i in range(n):
            q[i] = q[i] + a[i]
```

→ *"carried update value has no producer inside the loop body"*. An
array stored in an **inner** loop and carried across an **outer** one.
A single loop storing into the same array is fine.

`structure_native.py` routes around it by keeping only the member loop
inside (807 members) and moving the substep count out, which costs
almost nothing here. It sits in the store-chain identity area that is
under active work, so widening it was deliberately avoided.

**There is a wrong comment in `structure_native.py` that must be
deleted**: it claims name-shadowing (`length_m = length_m[_m]`) was the
cause. It was not — the shadowed line is still in the generated source
and compiles fine. The nested loop was the whole cause.

## 7. Standing backlog from earlier in the session

- **Muzzle brake physics** — "a modest muzzle brake optimised for
  diffusion of evidence". Never built. Under the tower-defence framing
  this is the highest-value unbuilt piece, because a brake is precisely
  the part that decides what the blast signature looks like.
- **Bore evacuator cycle** — plenum and ports declared, nothing steps
  the fill-and-blow.
- **Belt feeder** — "shells will travel on a belt feeder in and out, no
  more free ejection". The single `turret.case_ejection_column` clear
  volume is still the only route.
- **Mantlet** — not in the graph at all.
- **Cooler blankets** for the 120 mm (no jacket at maximum ordnance).
- **The 120 mm load underperforms**: 587 m/s and 114 MPa against a real
  120's 1580 m/s and 500 MPa.
- **Bore film coefficient** puts 37% of shot energy into the barrel wall
  against 15–25% for real guns.
- **No gas-leakage term**, so declared barrel "tightness" has no channel
  through which to affect velocity.
- **C18 vehicle** — water tanks, pumps, mobile EM outpost.
- **Firing on the move** / hull motion driving the mount.
- **`turret_demo.py`** still builds `machines.get("gimbal-cannon-station")`
  rather than the production graph. `MachineSim` now accepts a prebuilt
  `graph=`, so the change is small and has not been made.
- **`seat.*` nodes** in `drum_reference` lack `part_role` (test rig only).

## 8. What is finished and should not be re-litigated

For the next session's benefit, so it does not get rebuilt:

- **`joints.py`** is the one joint registry. 41 member constraints, each
  a `MemberConstraint` with a six-freedom RIGID/FREE/COMPLIANT transform,
  tokenised at declaration. `joints.T.single_axis_slider` resolves at
  import; a typo raises `AttributeError`. Seven scattered string sets
  collapsed into it.
- **Tokens are for vectorising, not for comparing.** Measured: token vs
  string in a Python loop is **0.90×** — CPython interns literals so `==`
  is already a pointer compare. The win is `graph_columns.py`: one int32
  column and a mask over the token space, **10×**, and it is the only
  form that can reach the native lane at all.
- **`rigid` excuses a member from resonance**, checked before
  `beam_solvable`. 595 body-internal edges. `frame_solver` assembles them
  as rigid links matched on **stiffness per unit length** — matching on
  section alone gave a condition number that returned twelve kilometres
  of deflection.
- **The trunnion finds its own balance point** by iteration
  (`balanced_station`). It is a fixed point, not a correction: the cradle
  and equilibrator anchor are placed *from* the pin, so the cg chases it.
  41.8 kN·m of dead moment → 0.034.
- **The drum joint** — boxed ring, two captive races at full cell depth,
  metre-deep annulus, cast 6061 body with a 4340 rim, drive volume
  pocketed into both plates with the mating bands left solid, six
  circular motors in a ring formation on a cast cradle with thrust races,
  rotary-union hubs with internal channels, dry outer face and
  direct-drive lock-up on the inner. `check: PASS`, 0.086 mm under 50 kN.
- **`surfaces.py`** — `Annulus`, `BoxedRing`, `Plate`, `BearingRace`
  (straight or arc of any sweep, one type).
- **`channels.py`** — ports grouped as an internal channel: one node with
  N openings, cored into a named body. Not a pipe.
- **`mesh_primitives.ring_mesh`** existed all along and nothing was wired
  to it; `vehicle_mesh` now has a `shape == "annulus"` branch.

---

## Suggested order

1. **Make the windows the default.** Nothing else can be verified until
   the user can watch it.
2. **Finish the slew trial** and render it.
3. **Second law through the WASM door** — `interior_ballistics` or the
   member bank, whichever is smaller, plus a reference-evaluator parity
   check. Proves the lane generalises.
4. **Register a `PhysicsStage`** for the beam plugin in a `DynamicsLane`
   so the scheduler owns the stepping, and delete `LiveStructure`'s own
   loop.
5. **Connect the drive to the turret** and answer the track topology.
6. Then the backlog in §7, muzzle brake first.
