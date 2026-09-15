# Negative drift, and what a diffusing time change does to the things in it

Report, 2026-09-15. Fourth in sequence after `TIME_FIELD_DESIGN.md`,
`TIME_FIELD_AND_COMPILE_REPORT.md` and `THREADING_AND_TIME_SLEW_REPORT.md`.

Two questions, both answered by measurement on the existing modules rather
than by assertion, plus the compiler consequences of the answers. Everything
below is reproducible with `python time_diffusion.py`.

---

## Part 1 — negative drift is already representable, and its price is linear

**It needs no new mathematics.** The field is `log_tau`, a signed scalar, and
a ratio is a difference across a joint. Nothing clamps it at zero, and
`set_target` rate-limits symmetrically (`max(-cap, min(cap, want))`). A zone
running faster than reference is the same difference with the opposite sign.
It is also only meaningful relative to a neighbour, because the absolute level
is a gauge choice: if everything speeds up, nothing has changed.

**It is conservative.** The coupled ledger publishes 1:1, 2:1, 10:1 and
both-dilated, all at or below reference. The same rig above reference:

| case | tau_a | tau_b | ledger error |
|---|---|---|---|
| reference | 1.0 | 1.0 | 0.0000e+00 J |
| b at 2x | 1.0 | 2.0 | 0.0000e+00 J |
| b at 4x | 1.0 | 4.0 | 0.0000e+00 J |
| a fast, b slow | 2.0 | 0.5 | 0.0000e+00 J |
| both fast | 3.0 | 1.5 | 0.0000e+00 J |
| b at 10x | 1.0 | 10.0 | 0.0000e+00 J |

Exactly zero in every case, for the same reason it closes below reference: the
differential's constraint is on angles, and torque times angle has no clock in
it.

**The price is substeps, exactly linear.** The window a lane advances is
`tau * frame`; the step it may take is fixed by its own stability floor. So

| tau | K | beam step | window asked | window got | stable |
|---|---|---|---|---|---|
| 0.2 | 67 | 49.75 µs | 3.33 ms | 3.33 ms | yes |
| 1.0 | 334 | 49.90 µs | 16.67 ms | 16.67 ms | yes |
| 2.0 | 667 | 49.98 µs | 33.33 ms | 33.33 ms | yes |
| 4.0 | 1334 | 49.98 µs | 66.67 ms | 66.67 ms | yes |
| 12.0 | 4000 | 50.00 µs | 200.00 ms | 200.00 ms | yes |
| 20.0 | 4096 | 50.00 µs | 333.33 ms | **204.80 ms** | yes |

Dilation is cheap for exactly the reason speed-up is expensive, and it is the
same arithmetic read in both directions.

**The ceiling is derived, not declared.** `tau_max = K_max · dt_limit /
window`. With the beam's 50 µs floor, a 16.67 ms frame and a 4096-substep
budget that is **12.29**, and the measured transition is exactly there: 12.0
still fits at K=4000, and 12.3 is where K saturates and the window begins to
be shortened.

**The failure mode past the ceiling is the one the design already has.** The
step never coarsens — it stays at 50.00 µs and stability is respected at every
row. The window shortens instead. So asking to run faster than the budget
affords returns *less world time*, exactly as dilation does: correct, but
behind. The invariant "stability is a hard floor, time is the soft variable"
holds in both directions without amendment.

**So: yes, allow it.** The one thing worth adding is that a zone running ahead
reaches a coupling boundary before its neighbours have produced the state it
needs, which is the same structure as the remote case — unknown input for an
interval — and takes the same cone, with the boundary exchange interval in
place of round-trip time.

---

## Part 2 — kinetic transfer between time scales: there isn't any

The question assumes a transfer that does not exist, and the thing that does
exist has a different shape and is measurable.

### Energy is a per-zone account

A zone's laws are written against its own clock, so its own-clock rate does not
change when its time velocity does. Seen from the reference frame the same
rotor turns at `tau * omega`, so reference-frame kinetic energy scales as
`tau**2`:

| tau | own-clock KE | reference KE | ratio |
|---|---|---|---|
| 0.25 | 22500 J | 1406.25 J | 0.0625 |
| 1.0 | 22500 J | 22500 J | 1.0 |
| 4.0 | 22500 J | 360000 J | 16.0 |

The own-clock column is the account. The reference column is a rendering of the
same state. **Ledgering the reference column would invent energy on every rate
change** — sixteen-fold here, from nothing — which is precisely the error the
gauge freedom warns about. A zone cannot measure its own time velocity, so it
cannot be charged for it.

What crosses a boundary is work: torque times angle, both clock-free. That is
why the ledger closes at every ratio including the new ones above.

### The only coupling of the field to mechanics

One term, already in the field: while the gradient is *changing*, the adaptor's
freedom must accelerate, at a reaction of its own inertia times
`d/dt[(w1-w2)/2]`. A steady gradient costs nothing. There is no other channel.
So "kinetic transfer at a point" means, exactly, **the angular impulse that
point receives through its adaptors while the field is moving.**

### What radius actually does

A change driven into one node and allowed to diffuse along a coupling chain
(`log_tau` relaxing toward its neighbours, differential adaptors at 0.02 kg·m²,
the rig's own figure, source ramp 4.0/s, D = 40):

| radius | peak reaction | time to peak | angular impulse |
|---|---|---|---|
| 1 | 12.000 N·m | 0.0 ms | 1.496 N·m·s |
| 2 | 3.765 N·m | 29.0 ms | 1.041 N·m·s |
| 3 | 2.299 N·m | 80.0 ms | 0.742 N·m·s |
| 4 | 1.648 N·m | 152.0 ms | 0.554 N·m·s |
| 5 | 1.230 N·m | 182.0 ms | 0.423 N·m·s |
| 6 | 0.870 N·m | 194.0 ms | 0.312 N·m·s |
| 7 | 0.560 N·m | 208.0 ms | 0.207 N·m·s |
| 8 | 0.277 N·m | 219.0 ms | 0.103 N·m·s |

Three things are visible and none of them were put in by hand.

**It is delayed, and the delay grows faster than linearly.** 29, 80, 152 ms for
radii 2, 3, 4 — second differences roughly constant, the signature of a
diffusive `r²` arrival. The tail flattens because an eight-node chain reaches
equilibrium and its far end reflects; the early radii are the honest part.
The observed delay is larger than the pure kernel's `r²/4D` because the source
is itself ramp-limited, so what propagates is the ramp convolved with the
kernel. That is physical, not an artefact: how fast the source may move is set
by its store's inertia.

**It is attenuated.** Peak reaction falls 12.0 → 0.28 N·m and delivered impulse
falls 1.50 → 0.10 N·m·s across eight joints. A distant point feels a smaller
and later nudge, never a different kind of event.

**It ends.** Once the profile settles the reaction is exactly 0.000000 N·m at
every joint while the gradient it settled into is large. Only the derivative
was ever a force.

### The rules this supports

1. **No kinetic energy crosses a time-scale boundary.** Energy is per zone, on
   that zone's own clock. Reference-frame energy is display.
2. **The only transfer is angular impulse through adaptors, and only while the
   field is moving.** A settled gradient of any size is free.
3. **What a point at radius `r` receives is the diffusion kernel convolved with
   the source ramp**: attenuated with distance, delayed as roughly `r²`, and
   bounded by the weakest adaptor on the path rather than by a constant.
4. **A diffusing region is the sub-graph of joints that have rate freedom.** A
   joint without it returns an infinite reaction, which is the system saying a
   gradient cannot exist there. A rigid coupling is therefore not an edge with
   very high stiffness — it is an *identification*, and the two sides are one
   zone. Defining diffusion on the freedom sub-graph makes the shear
   impossible by construction rather than by a limit.
5. **Two propagation speeds, not one.** The field spreads diffusively on the
   coupling graph; the mechanical consequence of each adaptor reaction spreads
   at the structure's own wave speed. They are different channels and should
   not be conflated in a scheduler.

What is *not* established here: the diffusivity `D` is a free parameter in this
measurement. Rule 3 says it should be derived per edge from the torque that
edge can stand — `D_edge` bounded by `T_max / (I_edge · |w · dgrad/dt|)` — which
makes the spread rate a property of the hardware, like the ramp limit already
is. That derivation is stated, not measured.

---

## Part 3 — what this asks of the compiler

The two edges meet here, and the meeting is not decorative: the field work
needs compiler properties that are currently defects.

**Carried state decides loop retention, and the field is about to add some.**
Making the field dynamical means carrying `log_tau` *and* `dlog_tau_dt` per
zone. Measured today in turing: a loop with **two** carried bindings is
retained as a coordinated recurrence — no nesting required — and a retained
loop cannot index a Python list of tensors by its loop variable, so the
compiler's own destructuring temporaries become formals nothing produces. The
balloon tire hits this today with `station_r[segment]`. A per-zone rate plus
its derivative is exactly the second carried value that moves ordinary dt
loops across that threshold. Widening the unroll rule would be a fix that
expires the moment the field gains its velocity; lowering the index in the
retained case is the one that survives. Pinned as
`turing/tests/test_retained_loop_list_index.py`.

**A job that cannot name its arguments cannot carry a lane's state.** The
threading report's function port is real, but measured at the failure the
resolved side is `('target',)` against an authored `['target', 'args']`:
`target` already resolves as a `StaticReference`, and it is **`args`** that has
no graph edge at all — the tuple's contents never enter the graph. Giving
`target` a function-reference port and excluding it from the arity check would
leave zero resolved keywords against two authored ones; relaxing the check
further would emit a job submission with no data ports, a worker that silently
receives nothing. The data ports come first. Pinned as
`turing/tests/test_dispatch_argument_ports.py`.

**Both are the same defect.** A binding the compiler has already resolved,
which the graph vocabulary has no port to carry, discovered at the far end. The
closure capture in `COMPILER_INTERPRETATION_RULES.md` section 6a is the third
instance: it acquires its caller-side binding only during call linking, so the
callee is specialized undescribed. The prescription is identical in all three —
mint the port where the binding is minted, while the thing being named is still
in hand.

**And the resampling gate applies to negative drift unchanged.** Conservation
does not catch a resampling error: the two LSD rigs both closed to machine
epsilon while reaching materially different speeds. Nothing above changes that,
and a zone running *ahead* has strictly more opportunity to resample a
neighbour's boundary product wrongly, because its window covers more world time
per exchange. A helper must declare the frame-converted rates it used, above
reference as well as below.
