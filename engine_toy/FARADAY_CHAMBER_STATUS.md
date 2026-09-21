# Faraday chamber / EM applicator lane — status and continuation

Companion to `FARADAY_CHAMBER_CONCEPTION.md`, which set the intent. This
records what exists, what each claim was checked against, what the code
deliberately refuses to answer, and what to do next.

**Nothing in this lane is committed.** Every file below is untracked,
including `circuit_graph.py` and `em_materials.py`, which were already
untracked before this work began. A commit is the first decision.

Verify the whole lane with:

```bash
python -m pytest tests/test_waveguides.py tests/test_cavities.py tests/test_chambers.py tests/test_emitters.py tests/test_applicators.py tests/test_circuit_graph.py tests/test_em_materials.py tests/test_transformers.py -q
```

218 pass, about 17 seconds.

---

## 1. The one idea

The field solver and the electrical system are **the same operator**, not
two systems joined by a bridge. `circuit_graph.py` states it in its own
header: `d0^T Y d0` is `laplace0` with branch admittance standing where
the Hodge star stands on a mesh. The nodal admittance matrix and the DEC
field operator differ only in which star they carry.

Everything here follows from taking that literally:

| tier | engine analogue | EM |
|---|---|---|
| deeply articulated | `engine_cycle_sim`, per crank angle | DEC field solve |
| the bake | per-state mode LUT | `circuit_graph.PortNetwork` |
| the consumer | `drivetrain_graph` | `circuit_graph` |

One difference matters. The engine's mode LUT is a *sampled* bake. Kron
reduction is **exact** for a linear network, so the EM bake discards the
interior and loses nothing at the ports. Where the geometry is
parametric — a waveguide, a small aperture, a rectangular cavity — the
articulated tier never runs at all, because the reduction has a closed
form. The DEC solve is reserved for shapes closed form cannot express,
which is the same division the optics stack next door makes.

**Modal expansion IS Kron reduction.** Not an analogy: the eliminated
coordinates are mode amplitudes instead of node potentials, and each
contributes a rank-one term to the port matrix. Consequently the cavity
needed no bespoke reduction — only to be written as a circuit.

---

## 2. Module inventory

| module | lines | tests | what it is |
|---|---|---|---|
| `waveguides.py` | 517 | 49 | guides as parametric geometry; penetrations as two-ports; Bethe apertures |
| `cavities.py` | 487 | 24 | cavity modes as LCR resonators; loop ports; reduction to `PortNetwork` |
| `emitters.py` | 271 | 16 | `RadiationLoad`, `RFSource`, `Magnetron` |
| `chambers.py` | 242 | 10 | the assembly: driven cavity + leaks + power ledger |
| `applicators.py` | 319 | 23 | `DielectricLoad`, `Susceptor`, `SublimationDuty`, `ApplicatorState` |
| `circuit_graph.py` | — | 57 | **modified**: coupled-group inversion (see §6) |
| `em_materials.py` | — | 18 | **modified**: water EM states (see §5) |

Dependency direction is one-way and deliberate: `applicators` →
`cavities` → `circuit_graph`, with `chambers` assembling. The specific
depends on the generic; nothing generic knows about EM.

---

## 3. What was checked, and against what

Every claim below is pinned by something computed a *different* way. The
pattern is deliberate — a self-consistent chain proves nothing.

| claim | independent source | result |
|---|---|---|
| TE₁₀ cutoff, WR-975/284/90/28 | published datasheets | 0.605 / 2.078 / 6.557 / 21.08 GHz, <0.1% |
| "32 dB per diameter" | trade rule of thumb | **derived**: `(20/ln10)·1.8412·2 = 31.98` |
| guide two-port | S₂₁ recovered from Y by inversion | `= exp(-γl)` to 1e-9 |
| Bethe aperture | σ ∝ a⁶ | doubling radius → **64.000000000×** |
| cavity Q (copper, 9.9583 GHz) | Pozar closed form | 7931.16 |
| same | numerical integration of the exposed field | 7931.16 |
| reduced port impedance | textbook coupled resonator `ω₀M²Q/L` | agrees to 1e-9 |
| lineshape | Lorentzian at the declared Q | half-power to 4 digits |
| chamber power ledger | conservation, each column read separately | balances to 1e-15 |
| shielding increment, bore 6→12 mm | standalone two-port attenuation | both **47.97 dB** |
| dielectric Q, matched filled cavity | textbook `1/tanδ` | exact to 5e-10 |
| small-loop radiation resistance | tabulated `31171(A/λ²)²` | 0.07% |
| oven mode count below 2.45 GHz | Weyl density formula | 63 counted, 68.6 predicted |

The two strongest are the coupled-resonator impedance and the shielding
increment. Both reach their answer without passing through the matrix
being tested, so if a reduction silently stopped being exact, those are
what would notice.

---

## 4. Findings worth carrying forward

**The 32 dB rule is an asymptote, not a constant.** At half the cutoff
the decay has already lost 13%. Quoting it as a constant over-predicts
shielding exactly where a designer is relying on it.

**Below cutoff a guide is reactive, not lossy.** The wave impedance is
purely imaginary — inductive for TE, capacitive for TM. A sub-cutoff
penetration shields by *reflection*. Writing it as a resistance gives the
right decibels and lies about where the energy went. A test fails if the
real part ever becomes non-zero.

**The tunnel and the aperture are different physics and must not be
fused.** For a 1 mm hole in a 1 mm wall at 2.45 GHz the tunnel accounts
for 16.0 dB and the aperture for 62.8 dB. A zero-length line attenuates
nothing, so the guide model alone calls a thin-screen hole nearly
transparent — under-predicting by 47 dB, quietly. `port_network` with
zero length is refused and names `CircularAperture`.

**A chamber shields by more than its bore.** Coupling at both ends
counts; only the assembly sees it. Measured ~40 dB more than the bore
alone.

**Water's state dominates everything.** At 2.45 GHz:

| state | ε′ | ε″ |
|---|---|---|
| liquid 20 °C | 80.1 | 12.58 |
| liquid 80 °C | 60 | 3.30 |
| ice −12 °C | 3.2 | 0.00288 |
| vapour | 1.0006 | 1e-5 |
| brine ~5 S/m | 78 | 49.1 |

Ice absorbs **4367× less than liquid at equal field**. Hot water absorbs
*less* than cold, so the coupling is mildly self-limiting rather than
runaway.

**But selectivity inverts in a driven resonant cavity — this is the most
important non-obvious result here.** A lossy load spoils Q, the field
collapses, and it absorbs almost nothing; a low-loss load lets the cavity
ring up and ends up taking most of the power. Measured in the same oven:

| load | absorbed |
|---|---|
| ice, 1 L | 574 W |
| liquid water, 200 mL | 0.49 W |

**Never quote 4367× as a power ratio.** It is a ratio at equal field. A
selective process must be designed around which channel dominates the
cavity's total loss.

**A real oven does not run on TE₁₀₁.** For a 0.30 × 0.20 × 0.25 m box,
TE₁₀₁ is at 0.780 GHz. At 2.45 GHz there are 63 modes below and 6 within
a loaded linewidth — TE₁₀₄, TE₄₁₂/TM₄₁₂, TE₃₁₃/TM₃₁₃, TM₂₃₀. See §7.

**Cryogens cannot be microwave-heated.** Nonpolar, essentially lossless,
and worse as they cool. An applicator around a dewar warms a susceptor,
or water ice and adsorbed layers, or drives a mode for something to sit
in. Never the LN₂.

**Terminology.** There is no "radiated dewar". The structure coupling
power into a load is an **applicator**; the deliberately lossy body added
to heat something that will not absorb is a **susceptor**; warming a cold
surface to drive off condensate is **regeneration**.

---

## 5. What the code refuses to answer

These are the declared limits. Each is a predicate a caller can ask, not
a sentence in a docstring — that was a standing instruction and it is
honoured throughout.

| limit | predicate | why |
|---|---|---|
| Bethe validity | `CircularAperture.is_valid_at` | needs ka < 0.5 **and** screen thinner than the hole |
| single-mode tunnel | `CircularGuide.dominant_mode_holds` | wall must be worth ≥3 decay lengths of the *second* mode |
| load is a perturbation | `DielectricLoad.is_perturbation` | 200 mL water in a 15 L cavity gives pull > 1 — the estimate reporting its own collapse |
| small-loop radiator | `RadiationLoad.is_small_at` | past a fraction of a wavelength a loop is not a dipole |
| exact cavity Q | raises for non-TE₁₀₁ | closed form is mode-specific; `approximate_unloaded_q` (`2V/δS`) is labelled an estimate |
| modal expansion | raises for non-TE₁₀₁ | machinery is general, only TE₁₀₁'s **field pattern** is written out |
| zero-length guide | raises | names `CircularAperture` |
| exactly at cutoff | raises | two-port is singular there |

Nothing in this lane is a fitted LUT. The Bessel roots are mathematical
constants; the water table is *measured anchors* at 2.45 GHz, declared as
such because no single closed form covers both liquid and ice — a single
Debye fits liquid but under-predicts ice by four orders, since ice's
relaxation sits near kHz and its microwave loss is a different mechanism.

---

## 6. Bugs found and fixed

**`waveguides.wavenumber_1_m` dropped the dielectric fill.** `k` was taken
from vacuum `c` while `cutoff_hz` already carried the fill, so β was off
by √εr in any filled guide. The cutoff still came out right because both
scale together at β = 0 — so the one test a careless suite writes passes.
*Check the slope, not the zero.* Fixed by carrying the medium's phase
speed on the mode; regression test asserts the slope.

**`circuit_graph.branch_admittance` inverted each coupling separately.**
Correct only when a branch has exactly one coupling. A cavity mode driven
by two probes has two, and so does any three-winding transformer — the
shared branch had its own inductance counted once per coupling. Matrix
inversion does not distribute over the pieces. Fixed by grouping coupled
branches (union-find) and inverting each group's `jωL` once. A test shows
the two answers genuinely differ, so this is a correctness fix.

That change also surfaced a constraint no pairwise check can see: `|k| ≤ 1`
per pair is necessary but not sufficient. For two ports on one mode the
real condition is **`k₁² + k₂² < 1`** — two loops cannot each capture 80%
of a mode's flux. Now enforced as positive-definiteness of the whole
inductance matrix.

**Cross-module discrepancy, unresolved by design.** `Magnetron.supply_notes`
checked the tube against the oven transformer already in the repo and
flagged it: that MOT's filament winding delivers 2.89 V (2 turns off an
83-turn primary on a 120 V leg) against a 3.3 V common magnetron nominal —
14% low. Neither module is wrong alone; they were declared apart and never
compared. Recorded in a test rather than smoothed over. **Which side moves
is a decision, not a fix.**

---

## 7. Continuation

### The gate

**Only TE₁₀₁'s field pattern is written out.** Everything below is blocked
on this, and a domestic oven needs it because TE₁₀₁ is not a mode it uses.

What is *not* blocked, and is already done: the multimode **assembly**.
The circuit takes N modes and the coupled-group inversion handles N
couplings on one branch. That was the enabling piece and it is finished.

### Priority order

1. **General TE_mnp / TM_mnp patterns.** Both field vectors for arbitrary
   indices; then mode inductance, per-mode Q and port coupling by
   quadrature over those patterns. The design is settled: validate the
   quadrature against the TE₁₀₁ closed forms already in `cavities.py`,
   exactly as the Q was validated three ways. These are build-time
   quantities that get baked, so quadrature cost never reaches runtime.
   *Roughly the size of `cavities.py`. Gates 2, and most of 3 and 4.*
2. **Multimode oven.** Nearly free once (1) lands.
3. **Magnetron pulling and pushing.** A real tube's frequency moves with
   the load it sees and with anode current. With 6 modes inside 12 MHz,
   pulling walks it between modes — which is *why* an oven's hot spots
   move. Makes the solve implicit (frequency ↔ load impedance), so it
   needs a fixed point. *Moderate.*
4. **Stirrer and turntable.** Time-varying coupling; the real product is
   the time-averaged heating pattern. *Moderate, mostly bookkeeping.*
5. **Door choke.** The quarter-wave trap — why a microwave door works
   despite a visible gap. Closed form, still missing. *Small.*
6. **Thermal feedback.** Load heats, ε changes, ice melts and the melt
   absorbs 4000× harder. Where the interesting behaviour lives;
   `thermal_domains.ThermalAssembly` is already there to take it.
   *Moderate.*
7. **`(species, state) → EMMaterial` binding.** `chemistry_boundary.species_state`
   already resolves a species and phase to a declared chemical state;
   nothing carries that state to its electromagnetic properties. Would let
   a vessel's declared contents pick their own EM properties instead of the
   caller naming a material. *Small-to-moderate; crosses into the organics
   registry, so read how it declares states first.*
8. **`PartPort` → penetration join.** Pure data: `voxel_ports.project_circular_port`
   lands the port, `cabinet.wall_thickness_m` supplies the wall. No new
   physics. *Small.*
9. **Bare-aperture coupling to a mode.** Currently `Penetration` requires a
   declared pickup. A bare hole's coupling comes from aperture
   polarizability, and that identification has **not** been derived here.
   Do not fake it.

### A judgement call on scope

For a **single-mode applicator** — a resonant test cavity, a cryogenic
applicator, a spectroscopy or MOT cavity — what exists is already correct
and validated, and items 1–4 buy nothing. They buy the domestic oven
specifically. If the oven is a demonstrator rather than the target, items
5, 6 and 7 give far more per unit effort.

---

## 8. Decisions waiting on a person

1. **Commit this lane.** Nothing is tracked.
2. **The filament discrepancy** (§6) — winding or nominal.
3. **Oven as target or demonstrator** — sets whether §7 item 1 is worth
   its cost.
4. **Whether the chemistry binding (item 7) should lead**, so the species
   side drives the material table rather than following it.

---

## 9. Do not rebuild

Already present and working: `rectifiers.HalfWaveDoubler`,
`HighVoltageDiode`, `HighVoltageCapacitor` (the magnetron HV chain);
`transformers.IronCoreTransformer` with its filament winding and magnetic
shunts; `thermal_domains.ThermalAssembly`; `cryogenics.VacuumJacketedVessel`
and `InsulationMedia`; `dewar_production.atmospheric_port`, which already
declares a `PartPort` with a radius and a direction — and by the
conception doc, that is already a waveguide.
