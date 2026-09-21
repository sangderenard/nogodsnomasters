# The Faraday chamber: a field simulator built the way an engine is built

Conception, 2026-09-21. Nothing here is built yet. This is the research
pass and the design it implies, written against what is already in the
tree so that the build is composition and not invention.

---

## 0. The ask, and the precedent it names

The dewar is the reference. `symbolic_atmosphere.py` authors ONE law for
"a bounded gas space holding a condensable species near its own phase
boundary", and `dewar_production.py` is the first **ideally controlled
deployment** of it: a box whose walls, vacuum annulus, cold tip, open top
and drain are all declared as ordinary production-graph nodes, so the
atmosphere law runs in a place where every boundary is known exactly.
The raincloud (`turing/examples/chamber_sim.py`) is the same law in a
place where the boundaries are weather.

The Faraday chamber is that move again, for fields:

| | atmosphere | field |
|---|---|---|
| the law | `symbolic_atmosphere.py` | `symbolic_field.py` (new) |
| ideal deployment | `dewar_production.py` | `faraday_chamber_production.py` (new) |
| the messy deployment | raincloud in a room | an engine bay, a station, weather |
| the engine | `ChamberSim(DtCompatibleEngine)` | `FieldSystem(DtCompatibleEngine)` (new) |
| the geometry layer | `thermal_domains.py` | `field_domains.py` (new) |

And the recursion the ask names explicitly -- *"one step in a dt system
while also itself being a dt system"* -- is `RoundNode` +
`MetaLoopRunner` in `turing/src/common/dt_system/dt_graph.py`. That is
already how `ChamberSim` sits under `run_superstep`, and how
`ThermalSystem` sits under whatever steps it. A `FieldSystem` that
implements `DtCompatibleEngine` (`step`, `preferred_dt`,
`causal_ceiling_dt`, `snapshot`, `restore`, published `Metrics`) is
nestable at any depth by construction. No new scheduling machinery.

---

## 1. What is already here (the inventory that makes this cheap)

**The medium is already a solved object.**
`turing/src/common/tensors/abstract_convolution/laplace_nd.py`:
`BuildLaplace3D` takes a `metric_tensor_func`, a `density_func` and a
`tension_func` -- the last two resolved per-cell by `_material_field`
(laplace_nd.py:663) -- plus a `wave_speed` and per-face boundary
conditions. That is a spatially varying wave operator on a curvilinear
grid, already written, already tested against the DEC assembly and the
continuous Laplace-Beltrami operator. `BuildGraphLaplace` is the same
thing on an arbitrary graph with Neumann masks.

**The DEC apparatus is complete as of 2026-09-21.** `THEORETIC.md:52-54`
specifies `D0` and signed `D1`; `fluxspring/fs_dec.py` computes the curl
(`D1 @ g`) and the curl-adjoint and validates `D1 @ D0 == 0`;
`HodgeStarBuilder.build_full_hodge_star` produces `hodge_0/1/2`. The
missing quarter -- **no site built a nonzero `D1`** -- is now filled:
`laplace_nd.build_face_incidence` produces the signed edge-to-face
incidence, `build_d_operators(..., faces=)` returns it, and
`tests/test_dec_d_operators.py` pins it against the hand-written torus
reference. The vector decision's bill is paid; see section 2a.

**The modal step already exists.**
`turing/src/common/tensors/riemann/demo_driven_chladni_plate.py::advance_modal_plate`
is a driven, damped, vectorised second-order step *in the eigenbasis of
LaplaceND*. That is a driven cavity, with the plate's name on it.

**The spectral/phasic split already exists.**
`abstract_convolution/spectral_propagator.py::propagate` interpolates
between the purely decaying heat kernel and the phase-carrying wave
kernel on the same eigenbasis, with `jump_length` giving a real travel
distance in hops. Complex multiplies are expanded into real ops by hand
there, with the reason stated -- the field lane inherits that decision
rather than re-litigating it.

**Meshes to occupancy already exists.** `sdf_geometry.MeshSdfKernel.compile`
voxelises a triangle mesh (`_voxelize_inside_x`, watertightness check,
capsule subtraction for punctures). `engine_rays.part_sdf` already caches
one per part. Any part from any engine can become a material field in the
chamber grid with the call that is already written.

**Ports to voxel faces already exists.** `voxel_ports.project_circular_port`
projects one physical circular aperture onto boundary faces, conserving
total area across cells. Written for the dewar's open top; a cage
penetration is the same object.

**Apertures from damage already exist.** The dewar declares
`damage_opens_to` / `damage_crosses_volume` on its jacket panels. A
through-puncture is already an aperture with real geometry, owned by
damage. The shielding law consumes it the same way the atmosphere does.

**The box already exists.** `cabinet.py` is the generic enclosure -- N
racks, optional top/bottom plant compartments, door kinds with windows
and accessories, `wall_ports`, welded-plate vs screwed-sheet
construction with real seam packs. A Faraday chamber is a `CabinetSpec`
with two new construction kinds and an EM material declaration. It is
not a new box.

**Thermal coupling already exists.** `thermal_domains.ThermalSystem`
accepts `source_w` as a dict or a callable per assembly. Dielectric and
eddy heating is a `source_w` entry. Nothing new is needed on the receiving
side.

**The electrical side already exists.** `electrical_distribution.py`
declares services, cables with one real conductor per role, breakers,
outlets -- and keeps neutral, DC return and protective earth distinct,
which is exactly the distinction a common-mode current lives on.
`dc_power.Conductor` carries the geometry. A cable is already a loop with
an area and a length; that is an antenna the moment anything asks.

**What does NOT exist:** any permittivity, permeability, conductivity or
field quantity anywhere in `engine_toy`. The nine grep hits for
"permeability" are sand moulds and air filters; "shielding" is welding
gas. This is greenfield with a very well-furnished workshop.

---

## 2. The law: what is actually being solved

**Decided (2026-09-21): full vector E/H, with the modal lane as the
authority and the time-domain lane as the fallback for moving
geometry.** What follows is written to that decision. Section 10 records
what it costs and the one question it opens.

### 2a. The state is a pair of differential forms, not three vector grids

The vector choice does not mean "three copies of the scalar field". It
means the state lives where the repository's own DEC apparatus already
puts it:

```
e   E-field as a 1-form on EDGES   (the line integral E.dl per edge)
b   B-field as a 2-form on FACES   (the flux per face)
```

and Maxwell is then two lines with no vector calculus in them at all:

```
db/dt  = -d1 e                          Faraday
*_eps de/dt = d1^T *_(1/mu) b - *_sigma e - j_src      Ampere
```

`d1` is the signed edge-to-face incidence; `d1^T` is its adjoint. That
pair *is* curl and curl-adjoint. This is the Whitney-form / Yee-on-a-
complex formulation, and it is the standard one.

**Why this is the repository's own idiom and not an import.**
`src/common/tensors/THEORETIC.md:52-54` already specifies `D0` (node to
edge) and `D1` (edge to face, signed) as the DEC operators.
`autoautograd/fluxspring/fs_dec.py` already *uses* them: `D1 @ g` at
line 82 is literally the curl of a 1-form, `D1.T() @ (...)` at line 151
is the curl-adjoint backpressure, and `fs_dec.py:30-34` and
`fs_io.py:263-267` already *validate* `D1 @ D0 == 0`.
`laplace_nd.HodgeStarBuilder.build_full_hodge_star` already produces
`hodge_0`, `hodge_1` and `hodge_2` with real vertex volumes, edge dual
areas and face areas.

**The one thing that was genuinely missing -- built 2026-09-21.**
`TransformHub.build_d_operators` used to return `{"d0": ..., "d1":
None}` with the comment "d1 placeholder", and every `DECSpec` site
passes `D1=[]`, a 0-row matrix that satisfies `D1@D0 = 0` trivially
(`fs_build_specs.py:88`). The contract, the validator, the curl, the
curl-adjoint and the Hodge stars were all written and waiting for a
face list.

What landed in `laplace_nd.py`:

- `build_face_incidence(edge_index, faces)` -- signed `(F, E)` `d1`.
  Each row is the boundary CHAIN of an oriented ring, so signs
  accumulate rather than overwrite.
- `build_d_operators(edge_index, profile, faces=None)` returns it;
  `faces=None` still returns `d1=None`, so every existing caller sees
  what it saw before.
- `FaceMapGenerator` now returns **oriented rings**. It previously
  returned `tuple(sorted(face))` -- a vertex multiset with the ring's
  closing vertex still duplicated inside it -- from which no
  orientation and therefore no sign could be recovered. One square with
  four edges produced four such "faces", each five entries long.
- `ring_key` deduplicates under rotation and reversal, which is what
  sorting was standing in for.
- Rings that revisit a vertex are rejected (a figure of eight is not a
  loop), and chorded rings are rejected as composites (two squares'
  outline is the sum of the two squares, not a third face).
- `is_planar` compares normal DIRECTIONS, as its own comment always
  said it did, and no longer crashes on a bare `(3,)` vector.

Validated in `tests/test_dec_d_operators.py` (25 tests): exact equality
with the hand-written torus `d1` in `test_dec_fft_continuous.py`;
`d1 @ d0 == 0` on four lattices and a tetrahedron; `rank(d1) == E - V +
C` in every case; the faces of a cubic lattice are its unit squares and
nothing else.

### 2b. The modal (Hamiltonian) lane -- the authority

Diagonalise once, per geometry:

```
(d1^T *_(1/mu) d1) e = w^2 *_eps e
```

a generalized symmetric-definite eigenproblem straight into the compiled
`eigh` lane -- the one that went from 521 s to 0.26 s. `*_eps` and
`*_(1/mu)` are diagonal Hodge stars, so **the material IS the operator's
own metric**: transformation optics says an `(eps, mu)` distribution and
a coordinate metric are the same object, and here they are the same two
diagonal matrices. A graded absorber, an anisotropic laminate, a ferrite
loading are entries in `*_eps` and `*_(1/mu)`. Nothing is bolted on.

- **Hamiltonian, literally.** `(e, b)` is the canonical pair and
  `H = 1/2 (e^T *_eps e + b^T *_(1/mu) b)` is the field energy. The
  E/H leapfrog stagger is not *like* a symplectic integrator, it *is*
  the canonical one -- which is why the scalar lane's framing survives
  the vector choice and gets sharper. In the eigenbasis each mode is one
  oscillator `(q_m, p_m)` at `w_m`, exactly the step
  `advance_modal_plate` already performs.
- **Energy is a first integral**, so the `energy_j` channel in
  `DT_CHANNEL_NAMES` is the law's own conserved quantity, published the
  way `ThermalSystem` already publishes it, and its drift IS the error
  signal for the dt controller.
- **`dt_limit` is closed form**: `dt < 2/w_max`. Every chamber law in
  `symbolic_chamber_solvers.py` publishes a closed-form stability bound
  from its own Jacobian; this one has the simplest such bound there is.
- **div B = 0 is structural, not enforced.** `d1 d0 = 0` -- the identity
  `fs_dec` already validates -- makes the divergence a conserved
  identity of the operator. Nothing cleans it up per tick. This is the
  same "true by construction rather than by solving a constraint system"
  argument `TIME_FIELD_DESIGN.md` makes for the log-tau difference
  field, and it is the main reason the DEC formulation is worth the
  `D1` builder.
- **Spurious modes are identified, not filtered.** `ker(d1) = im(d0)`
  exactly on a de Rham complex: the zero-frequency junk is precisely the
  discrete gradients, and `d0` -- which *is* built -- names them. They
  are deflated by a matrix that already exists, rather than suppressed
  by a tolerance.
- **Continuous frequency, for free.** Drive is `sin(2*pi*f*t)` at any
  real `f`; response is a sum over modes of a Lorentzian in `f`. No
  bins, no FFT, no frame-rate-quantised spectrum. This is what
  "continuous frequency appreciation" has to mean: the *modes* are
  discrete because the *cavity* is, the *drive* is continuous because
  the *world* is. (The sines come from the trig cores in the
  signal-math lane, not libm.)
- **Resonance is emergent, not tabled.** `f_101 = (c/2)*sqrt(1/a^2 +
  1/d^2)` falls out of the eigenvalues; a 1 m cube rings at 212 MHz
  whether or not anybody wrote that down. Load it with an engine block
  and it changes, because the block changes `*_eps`.
- **Mode budget is knowable in advance.** Weyl's law for EM modes,
  `N(f) ~ 8*pi*V*f^3/(3*c^3)`, already counts both polarisations: a
  1 m^3 chamber has ~310 modes below 1 GHz and ~39 below 500 MHz. (The
  scalar lane would have been half that -- the vector choice costs a
  factor of two in modes, not a factor of three in fields.)
- **Polarisation is now real, and this is what the decision buys.** A
  PEC wall is `n x E = 0`, which on this state is simply "the tangential
  edge degrees of freedom on the cage are zero" -- a Dirichlet condition
  on 1-forms. An aperture *removes* that constraint on the edges
  crossing it, so **a slot couples to the polarisation aligned across
  it and not to the other one.** Slot orientation matters, which is the
  single most characteristic fact about real shielding and the thing the
  scalar lane could never have said.
- **Loss is declared and derived.** A finite-conductivity wall is a
  surface-impedance (Robin) term on those same edges, with `sigma` from
  `EM_MATERIALS`. That is where `Q = 1/(2*zeta)` comes from -- the
  wall's real surface resistance (section 3), not a tuning knob.

The modal lane's limitation is real and should be stated: the basis is
valid only while the geometry is. A door opening, a part moving, a wall
being punctured invalidates it. Mitigation is the raincloud's own --
recompute on a topology event, never per tick -- and the eigenvectors
are a cacheable artifact keyed on the geometry digest, exactly like
`__lawcache__` and exactly like the `artifacts/llvm_pieces` manifest.

### 2c. The time-domain lane -- for when the geometry moves

The same `(e, b)` state and the same two Maxwell lines, stepped directly
by leapfrog at `dt <= dx/(c*sqrt(3))` instead of diagonalised. This is
the lane for a moving part, a switching transient, a spark's broadband
edge, and a puncture appearing mid-run. It costs a substep at roughly
0.6 ns per metre of cell, which is why it is the fallback.

The two lanes share the state, the operators, the materials and the
published `Metrics`. A chamber may run either; the dt system cannot tell
them apart, which is the point. The fallback is a *switch*, not a second
implementation: same `d1`, same Hodge stars, one skips the
eigendecomposition.

---

## 3. Shielding: where the weave and the slab acquire consequence

The user's handles -- weave, thickness, material -- have to *do*
something, and the textbook says exactly what. Every formula below is
named, standard, and cheap:

**Skin depth.** `d = sqrt(1/(pi*f*mu*sigma))`. Copper at 1 MHz: 66 um.
Stainless at 1 MHz: 0.42 mm. This single number is why a 0.8 mm
stainless sheet (cabinet.py's default `sheet_thickness_m`) is a good
shield at 10 MHz and a poor one at 10 kHz.

**Absorption loss.** `A = 8.686*t/d` dB. Thickness earns its dB linearly
in `t` and as `sqrt(f)`.

**Reflection loss** (Schelkunoff, plane wave):
`R ~ 168 - 10*log10(f*mu_r/sigma_r)` dB. This is why mu-metal is a
low-frequency magnetic shield and copper is not, and why the choice of
material is a real choice with a real frequency-dependent answer rather
than a quality slider.

**Aperture leakage.** `SE = 20*log10(lambda/(2*L))` for the largest
aperture dimension `L`, going to zero at `L = lambda/2`. A shield is its
worst hole. This is the formula that makes a *weave* a physical object: a
mesh of pitch `p` is an array of apertures of size `p`, with an array
factor for `n` apertures (`-10*log10(n)`) and a correction for wire
diameter and contact resistance at the crossings. Weaving tighter buys
dB; a single bad bond throws them away. Both are true in the model for
the same reason they are true in a lab.

**Waveguide below cutoff.** A circular penetration of radius `a` and
length `l` is a waveguide with `f_c = 1.841*c/(2*pi*a)`; below cutoff it
attenuates at ~32 dB per diameter of length. A honeycomb vent is that
formula with `n` cells. **And this is the free part**: every fluid line,
cable gland, drive shaft and breather that a machine already declares as
a `PartPort` with a radius and a direction *is already* such a
waveguide. `voxel_ports.project_circular_port` already knows where it
lands on the boundary. The "rich waveguide development" the ask wants is
mostly a matter of asking geometry that already exists a question nobody
has asked it yet.

**Damage is a hole.** A through-puncture from `damage_opens_to` has a
capsule SDF (`CapsuleSdf.from_puncture`) and therefore a diameter. It
enters the same aperture term. Shooting the cage degrades the shield, in
the correct band, with no special case.

These formulas are an *audit* lane: they give a shielding-effectiveness
curve in dB against frequency from the declaration alone, in
microseconds, with no field solve. The field solve then either agrees or
reveals a resonance the closed form cannot see. Two weightings, both
honest, answering different questions -- the pattern `system_spectrum.py`
already argues for.

---

## 4. The declaration layer (parts declare; nothing is string-matched)

Following the rule the fluids work established the hard way: parts
DECLARE their role. New node attributes, in the dewar's own idiom:

```
em_domain           = "cavity" | "shield" | "conductor" | "dielectric"
                    | "absorber" | "none"
em_material         = key into EM_MATERIALS
shield_construction = "solid-slab" | "woven-mesh" | "perforated" | "foil"
weave_pitch_m       , wire_diameter_m , open_area_fraction
bond_resistance_ohm                  (per seam; the thing that ruins cages)
em_aperture         = {"radius_m":..., "length_m":..., "role":"vent"|"gland"}
field_emission      = {"mechanism":..., "from_state":...}   see section 5
field_susceptible   = {"loop_area_m2":..., "victim":...}    see section 7
```

`EM_MATERIALS` is a new row table in the shape of `milspec.MATERIALS` and
`thermal_domains.THERMAL_MATERIALS` -- `key, label, sigma_s_m, eps_r,
mu_r, loss_tangent, spec, note` -- keyed on the SAME material keys
already written on graph nodes (`stainless-steel`, `copper`,
`steel-plate`, `6061t6`, ...), so most of the tree acquires EM properties
by already having declared what it is made of. Adding a material is
adding a row; that is the standard `fluids.py` set.

The cage itself is `cabinet.CabinetSpec` with `construction` extended by
`woven-mesh` / `perforated`, plus the weave fields. Racks, door, window,
glove ports, plant compartments, feet: unchanged, and all of them
suddenly have EM consequence (a glazed door is an aperture the size of
the door unless the glass is meshed or coated -- which is why real screen
rooms have that exact window).

---

## 5. The parts inside are the emitters, and their spectrum is their state

This is the design's centre of gravity, and it has a precedent in this
tree that should govern it: **the sound system takes pitch only from
spinning parts.** Nothing invents a noise; a part that turns makes a tone
at the rate it turns. The field layer is the same discipline applied to
the electrical and combustion state that the sim already advances:

| emitter | mechanism | drawn from state that already exists |
|---|---|---|
| ignition coil / spark | broadband impulse, ns-scale edge | `ignition_driver.py`, firing sequence, spark energy |
| brushed / commutated motor | commutation transient at `rpm*segments` | `dc_power`, the compressor motor's own rpm |
| alternator, generator | slot-passing harmonics at `rpm*poles/2` | `station_powerplant.py` |
| inverter / drive | PWM carrier and sidebands | declared drive on the service |
| contactor, breaker, relay | one switching event, `di/dt` into `L` | `electrical_distribution.CircuitBreaker` |
| welding arc | continuous broadband, arc-stability driven | `welding.py` already models arc quality |
| induction furnace | the strongest emitter in the tree, at its own coil frequency | `foundry.py` |
| every cable | radiated `E ~ A*dI/dt`, conducted common-mode | `BuildingCable` geometry, already declared |

The emission amplitude is proportional to `dI/dt` on a declared conductor
with a declared loop area -- both of which the electrical layer already
carries. So a part does not get an "emission" number: it gets a
*mechanism*, and the number is read off the machine's own running state,
tick by tick. Throttle up the motor and the comb of lines moves.
Misfire, and the spectrum says so before the sound does.

**And this is how a part becomes a controlled source.** The ask --
"making the very parts inside the faraday cage act as controlled field
emission" -- is then not a special mode. It is running a part at a
commanded state and letting its declared mechanism emit. A bench motor at
a swept rpm is a swept comb. That is a *calibration instrument* built out
of the machine catalogue, which is the same trick the dyno pulls by
coupling through the engine's own coupling.

---

## 6. The imposing system: background conditions, tuneable exposure

A separate object wrapping the chamber, NOT a source inside it. The
frame drives the domain's outer boundary; the cage attenuates; the
interior sees what gets through. Declared kinds, all standard hardware:

- **Helmholtz pair** -- uniform low-frequency `B`,
  `B = (4/5)^(3/2) * mu0*n*I/R` at the centre; the honest way to impose a
  DC or mains-frequency magnetic background.
- **Solenoid / coil set** -- axial field, graded along the axis.
- **TEM cell / GTEM** -- a defined `E`-field volume up to about 1 GHz;
  the standard way to state "this part sat in 10 V/m".
- **Stripline** -- cheap planar `E` for a small part.
- **Plane-wave illumination** -- an incident `(k, E0, polarisation)`
  boundary term for the full-domain case.

Exposure profiles are named against real immunity standards so the
tuning is legible rather than arbitrary: IEC 61000-4-3 radiated immunity
(80 MHz to 1 GHz sweep at 1/3/10 V/m, 80 % AM at 1 kHz), 61000-4-6
conducted, 61000-4-8 power-frequency magnetic, 61000-4-2 ESD as a single
impulse, and a free-form `(waveform, f, amplitude, polarisation, sweep)`
for anything else. "Justified but not accurate" is exactly right here:
the point is that a knob has a name, a unit and a standard behind it, not
that the chamber is calibrated.

The frame's own power draw is a real load on the electrical service, and
its coils are real masses on real mounts. It is a machine like any other.

---

## 7. What the field DOES back -- the couplings that make it a simulation

A field nobody feels is a screensaver. The return paths, all into systems
that already exist:

1. **Induced voltage** on any declared conductor loop: `V = -A*dB/dt`,
   and common-mode current on cable shields. Straight into
   `electrical_distribution`'s existing per-role conductors.
2. **Victim behaviour**: ECU sensor error, ignition timing jitter, HCU
   valve chatter, dyno signal noise -- each declared by the victim as
   `field_susceptible`, each already having a state to perturb.
3. **Heating**: eddy-current loss in conductors, dielectric loss
   `P = w*eps0*eps_r*tand*|E|^2` in insulators. Published as `source_w`
   into `ThermalSystem`, which already accepts exactly that. The engine in
   the cage gets warm because of the field, on the thermal system that is
   already running.
4. **Force**: Lorentz force on current-carrying members,
   magnetostriction hum at 2f (the transformer buzz), coil-to-coil
   attraction in the imposing frame. Into the existing structural and
   mount lane.
5. **Damage**: dielectric breakdown when local `|E|` exceeds a declared
   strength, insulation puncture, arc-over at a poor bond -- into
   `damage_state` / `burst`, where an arc-over is already a thing that can
   start a fire.
6. **Sound**: 2f magnetostriction and corona are legitimate entries in
   `sound_parts` -- and unlike rpm-derived tones they persist at zero rpm,
   which is a good test of whether the sound layer's rule ("pitch only
   from spinning parts") is a rule about *rotation* or about *derivation*.
7. **The time field.** `PERFORMANCE_CAUSALITY_AND_SPECULATION.md`, "Why
   this legitimizes electromagnetic and rail weapons", already argues that
   a *static* field denies nothing and a *changing* field denies
   speculation, with the denial radius set by propagation over the
   exchange interval. The field layer supplies the `dE/dt` that argument
   is written in terms of. This coupling is already conceived; it is
   waiting for a field to exist.

---

## 8. Shape of the code (seven modules, each with a template already in tree)

```
em_materials.py               rows: sigma, eps_r, mu_r, tan d, per material key
                              template: milspec.MATERIALS
aperture_coupling.py          closed-form SE: skin depth, absorption,
                              reflection, aperture, mesh array, waveguide
                              below cutoff.  Pure functions, no state.
                              template: hydraulic_losses.py
[the D1 builder]              DONE 2026-09-21, in turing core:
                              laplace_nd.build_face_incidence, reached
                              through build_d_operators(faces=).  General,
                              not chamber-local.
field_domains.py              CavityFieldDomain / ShellFieldDomain /
                              PathFieldDomain, FieldAssembly, FieldSystem
                              (DtCompatibleEngine).  Geometry is immutable;
                              fields are runtime state.
                              template: thermal_domains.py, one-to-one
symbolic_field.py             the law, in SymPy, lowered through
                              compile_sympy_equations like every other law
                              template: symbolic_atmosphere.py, symbolic_parts.py
field_sources.py              emission mechanisms; each reads machine state
                              template: sound_parts.py plus hole_emitters.py
exposure_frame.py             Helmholtz / TEM / GTEM / stripline / plane wave
                              plus named standard profiles
                              template: dyno_profile.py
faraday_chamber_production.py the cage as a CabinetSpec plus weave/slab, ports,
                              the imposing frame and its power
                              template: dewar_production.py
```

Nothing in that list is a stepper, a runner, a scheduler, a state table
or a snapshot convention. Those exist.

---

## 9. Staging -- the dewar's own order, because it worked

1. **`em_materials.py` plus `aperture_coupling.py` alone.** Pure tables
   and closed forms. Produces a shielding-effectiveness curve for a
   declared cage in microseconds. Testable against handbook numbers on
   day one, with zero solver risk. *This is the whole first increment and
   it is already useful* -- it answers "what does this cage do" without a
   field.
2. **`faraday_chamber_production.build()`** -- the cage, its ports, its
   door, its bonds, its imposing frame, its power. Tested exactly as
   `test_dewar_production.py` tests: the graph is one graph, the circuits
   are the circuits, every object declares its domain.
3. ~~**The `D1` builder, alone.**~~ **DONE 2026-09-21.**
   `laplace_nd.build_face_incidence` + oriented faces, 25 tests in
   `tests/test_dec_d_operators.py`, validated against the tree's own
   hand-written torus reference and against `rank(d1) == E - V + C`.
4. **`field_domains.py`** -- Hodge stars from declared materials, part
   occupancy from `MeshSdfKernel`, PEC edges from the cage's own walls.
   Test: the gradient modes come out at exactly zero frequency and get
   deflated by `d0`; the empty 1 m cube's first non-gradient eigenvalue
   is 212 MHz; `div b` is conserved to round-off with no cleanup pass.
5. **`FieldSystem`** as a `DtCompatibleEngine`, modal lane, publishing
   `energy_j`, `power_w` and a closed-form `dt_limit`. Test: energy
   conserved to the symplectic step's own order with no drive and no
   loss; `Q` matches `1/(2*zeta)` under drive; a slot passes the
   polarisation across it and rejects the other one, which is the
   vector lane earning its cost.
6. **Nest it.** Chamber under a `RoundNode` alongside the thermal system
   -- the recursion the ask asks for, demonstrated on the smallest
   possible pair.
7. **`field_sources.py`** -- one emitter first: the spark, because it is
   the most broadband and the most obviously *derived* from state.
8. **`exposure_frame.py`** -- one profile first: 61000-4-3 at 3 V/m.
9. **The couplings of section 7**, one at a time, thermal first, because
   the receiving side is already written and takes a dict.

Each step is verifiable against something outside the simulation. That
is the property the dewar had and the reason it landed.

---

## 10. Decided, and still open

**Decided 2026-09-21:**

- **Vector E/H from the start.** Cost: one `D1` builder (section 2a).
  Benefit: polarisation, slot orientation, `div B = 0` structurally, and
  a canonical Hamiltonian pair instead of an analogy to one. The
  operator's other three quarters are already written and validated.
- **Modal lane is the authority; time-domain is the fallback** for
  moving geometry, switching transients and mid-run punctures. Shared
  state, shared operators, shared materials, shared `Metrics` -- the
  fallback is a switch, not a second implementation.

**Still open:**

1. **`hodge_2` is wrong for any face that is not a triangle.**
   `build_full_hodge_star` reads `face[0], face[1], face[2]` and takes
   that triangle's area as the whole face's. Measured on a lattice of
   0.5 m cells: it reports **0.125 m^2** for a face whose area is
   **0.25 m^2** -- exactly half, the first of the two triangles. For the
   modal lane that number is `*_(1/mu)`, so it is not cosmetic. The fix
   is a fan triangulation over the oriented ring, which the rings now
   support, but it changes what the 2-form Hodge star MEANS and so was
   left alone. Fan-triangulate, or is there a reason it is the way it is?
2. **`AbstractTensor.cross` is broken for bare `(3,)` vectors.** It
   takes its component axis with `_take_along_dim`, which collapses a
   1-D input to a scalar that then has no `.stack`. Both call sites here
   were patched by presenting `(1, 3)` instead; the core op was not
   touched, because its blast radius across the tree is unknown. Fix
   `cross` properly, or leave the reshape idiom at call sites?
3. **Where does the law live?** `symbolic_atmosphere.py` sits in
   `engine_toy` and lowers through turing's compiler; the chamber laws
   sit in `turing/examples`. Which side of that line is the field law on?
4. **One chamber or a chamber registry?** `ThermalSystem` takes N
   assemblies and publishes one row. Should `FieldSystem` batch multiple
   chambers the same way from the start, or is a chamber singular?
5. **`EM_MATERIALS` keyed how?** On the existing material strings, which
   makes the whole tree EM-aware for free but silently defaults anything
   unlisted -- or as a required explicit declaration on a node, which is
   louder and means the cage cannot be built by accident?
6. **Does the cage's PEC set come from the mesh or the spec?** A
   `CabinetSpec` names its walls; `MeshSdfKernel` voxelises them. The
   edge set that gets `n x E = 0` can be derived from either, and they
   will not agree exactly at a seam or a hinge -- which is also where a
   real cage leaks. Which one is authoritative?
