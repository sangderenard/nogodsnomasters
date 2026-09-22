# Proposition: a general hierarchical physics simulator

Recorded 2026-09-22. This is an intention document, not a build log.
Nothing described here changes existing behavior in `engine_toy/` or
`turing/`; it names an architecture and a naming convention that future
work should converge toward, so the convention exists before the code
is asked to follow it.

For the equation-level roster behind each engine named below — what is
already authored as a SymPy or plain-Python law somewhere in the
repository tree today, and what canonical equations a complete
treatment of that field still requires — see
[ENGINE_EQUATION_ANNEX.md](ENGINE_EQUATION_ANNEX.md).

## Purpose

The simulator should converge the existing engine work into a small
family of physically distinct, independently extensible engines.

The objective is not to create one monolithic solver, and not to run
maximal physical detail everywhere. The objective is to give every
physical object a common ontology through which increasingly detailed
models may be requested without replacing the simpler models that
already work.

A triangle, volume, particle, edge, port, circuit, molecule, or machine
remains the same physical object as its representation is promoted.

The governing principle is:

**one physical fact, one authoritative state, many consumers.**

A puncture is one geometric puncture. Fluid flow, thermal exchange,
electromagnetic leakage, structural weakening, chemistry, and
rendering consume that same fact.

A phase change is one material-state transition. Thermal, structural,
electromagnetic, optical, and fluid solvers observe its consequences
rather than each inventing their own transition.

The package therefore distinguishes:

1. **fundamental state engines**, which own genuinely different
   categories of physical state;
2. **constitutive/law providers**, which calculate forces, fluxes,
   properties, rates, or closures;
3. **reduced models**, which cheaply represent a deeper engine over a
   declared domain of validity;
4. **deployment geometry**, which says where the laws act;
5. **`dt_system`**, which exclusively owns numerical advancement;
6. **Turing**, which converts auditable laws and source programs into
   portable compiled kernels;
7. **Nodus**, which composes, schedules, hosts, persists, and
   distributes those kernels.

## The honorary naming convention

Twelve fundamental engines carry the names of the scientists whose
work they most directly continue. This is the authoritative mapping
from honorary name to functional role:

| Honorary name | Functional name |
|---|---|
| **Lavoisier** | Chemistry / reaction & species engine |
| **Fourier** | Thermal conduction / heat-transfer engine |
| **Navier–Stokes** | Continuum fluid transport engine |
| **Bjerknes** | Atmosphere / aerosol / dispersed-phase engine |
| **Gibbs** | Thermodynamics / phase-state engine |
| **Bragg** | Solid-state / lattice / semiconductor engine |
| **Newton** | Classical force / particle / rigid-body engine |
| **Timoshenko** | Beam / frame / reduced structural engine |
| **Faraday** | Electromagnetic field / circuit engine |
| **Hamilton** | Quantum-state / electronic-structure engine |
| **Curie** | Nuclear / radiological engine |
| **Einstein** | Relativistic spacetime engine |

Two further names are reserved for layers that may eventually deserve
independent identity, rather than living as an aspect of a neighboring
engine:

| Honorary name | Functional name | Why it's reserved rather than assigned |
|---|---|---|
| **Boltzmann** | Statistical mechanics / kinetic particle-distribution engine | Naturally owns the mesoscopic regime between Newton, Navier–Stokes, and Bjerknes: particle distribution functions without either explicit molecules or continuum CFD. |
| **Noether** | Conservation / invariant / physical-ledger verification engine | A cross-cutting auditor for mass, charge, energy, and momentum conservation across every other engine, rather than a physical domain of its own. |

Sections 1–10 below detail the twelve assigned engines. Sections 11–13
cover chemistry and thermal physics as shared consumers of the deeper
engines' state, and physical promotion as the mechanism that ties all
of it together. Sections 14–19 cover the numerical, compilation,
composition, and hosting layers that are not physics engines at all.
Sections 20–21 state the reference integration environment and the one
architectural rule the whole proposition reduces to.

Several of these layers are not purely aspirational: `turing/` (a
sibling repository to this one, not tracked here) already runs working
fragments of this architecture — a Faraday-family field law, a
Bjerknes-family atmosphere law, and a dt system that already advances
both SymPy-compiled and LLVM-compiled steps through one contract.
Where that is true, the relevant section says so and names the file,
using plain code spans rather than links, since those paths live
outside this repository's worktree.

---

## 1. Newton engine — classical matter in motion

Newton is the common classical mechanics substrate.

It owns classical mechanical state:

- position;
- orientation;
- linear momentum or velocity;
- angular momentum or angular velocity;
- mass;
- inertia;
- applied generalized force;
- applied generalized torque;
- constraints and contact state where appropriate.

Newton should **not own a hardcoded integration algorithm**. Its
mathematical assertion is essentially:

**state + generalized forces + mass/inertia → state derivative.**

`dt_system` chooses and executes an appropriate integrator. This
allows the same Newton law to be advanced using semi-implicit Euler,
Verlet/leapfrog, Runge–Kutta, Newmark, implicit methods, symplectic
methods, adaptive schemes, or later specialist integrators without
changing what a force means.

### Existing lineage

The existing BoundSpring/FluxSpring work is an important predecessor.
It already demonstrates several principles worth preserving:

- forces are distinct from advancement;
- edge forces accumulate onto node state;
- incidence structure can replace backend-specific scatter mutation;
- mass converts accumulated force to acceleration;
- timestep authority can live outside the force law;
- the engine reports stability requirements rather than secretly
  subdividing time.

The existing `IntegratorEngine` is also a prototype of the separation
between force accumulation and motion advancement.

### Force providers

Many phenomena should therefore **not become engines**. They become
Newton force providers:

- Hooke springs;
- damping;
- contact;
- gravity;
- drag;
- buoyancy;
- pressure loads;
- Lorentz forces;
- radiation pressure;
- molecular potentials;
- bond forces;
- electrostatic forces;
- constraint reactions.

### Molecular motion

Ordinary molecular dynamics can initially be:

**particles + potential energy → force → Newton → dt_system
integrator.**

For a potential `U`:

```
F = -grad(U)
```

That is sufficient for a very large class of molecular geometry and
dynamics. Bond stretch, angle bend, torsion, Lennard-Jones
interaction, Coulomb interaction and other classical force fields do
not require another fundamental motion engine.

---

## 2. Timoshenko engine — reduced structural mechanics

Timoshenko is the structural reduction engine for members, frames,
shells where later appropriate, modal structures, and machine-scale
elastic assemblies.

It is not the fundamental description of matter. It consumes
constitutive properties and geometry and provides a cheap continuum
structural representation.

Existing repository work in
[`engine_toy/beam_theory.py`](engine_toy/beam_theory.py),
[`engine_toy/frame_solver.py`](engine_toy/frame_solver.py), and
[`engine_toy/structure_plugin.py`](engine_toy/structure_plugin.py)
already establishes the important distinction:

**Timoshenko, not Euler–Bernoulli.**

Shear flexibility, torsion, section properties, joints and
six-degree-of-freedom structural evolution are already present.

Timoshenko should consume material properties from deeper material
engines rather than owning permanent material constants. For example:

```
Bragg/Gibbs  ->  E, G, nu, alpha, rho, yield response, damping, damage state
             ->  Timoshenko
             ->  deformation, vibration, structural loads and reactions
```

A beam can therefore become softer because its actual material state
changed rather than because the beam solver contains a private
temperature heuristic.

---

## 3. Faraday engine — electromagnetic fields and networks

Faraday owns classical electromagnetic state.

The preferred field representation already identified is discrete
exterior calculus:

- electric field as a 1-form on edges;
- magnetic flux as a 2-form on faces.

The fundamental field equations can therefore remain topology-native,
in the same notation already used in
[`engine_toy/FARADAY_CHAMBER_CONCEPTION.md`](engine_toy/FARADAY_CHAMBER_CONCEPTION.md):

```
db/dt       = -d1 e                                    Faraday
*_eps de/dt = d1^T *_(1/mu) b - *_sigma e - j_src       Ampere
```

The same operator family naturally joins field simulation and
electrical-network simulation:

```
d0^T Y d0
```

and a DEC Laplacian differ principally in the constitutive star/
operator attached to the topology.

This permits:

- circuits;
- antennas;
- cavities;
- waveguides;
- shielding;
- dielectric heating;
- eddy currents;
- electromagnetic forces;
- radiation;
- coupling into structures and materials

to inhabit one mathematical family.

Faraday publishes forces to Newton and heat to thermal systems. It
consumes electromagnetic constitutive state from Bragg/Gibbs rather
than keeping a separate permanent material table where deeper state is
available.

**Already real.** This is not only a conception. `turing/examples/
symbolic_em_solvers.py` already authors `maxwell_faraday_step` and
`maxwell_ampere_step` as exactly this Yee-lattice pair — E on edges, H
on faces, leapfrogged half a step apart — plus `wave_step`,
`schrodinger_step`, `charged_species_step`, and `poisson_relax_step` in
the same file, sharing the same publication contract (`dt_limit`,
`energy_j`, `power_w`, and the other dt-system metric names). Every law
in that file publishes its own Courant limit and energy/power channels
rather than leaving stability or energy bookkeeping to a caller.

---

## 4. Navier–Stokes engine — continuum fluid transport

Navier–Stokes owns resolved continuum transport.

It should become the deeper fluid authority for regions where
pressure, momentum, compressibility, viscosity, species transport,
jets, boundary layers or detailed passage through apertures matter.

Its conserved state can include:

- mass density;
- momentum density;
- total energy density;
- species densities;
- optional dispersed-phase fields.

Schematically:

```
rho, rho*u, rho*E, rho*Y_1 ... rho*Y_n
```

But Navier–Stokes should **not own thermodynamic truth**. It asks
other engines for closure:

- equation of state;
- viscosity;
- thermal conductivity;
- diffusion coefficients;
- surface tension;
- latent terms;
- phase information.

Thus:

**Navier–Stokes owns conservation and transport.**

**Gibbs owns thermodynamic closure.**

---

## 5. Bjerknes engine — atmosphere and dispersed matter

Bjerknes is the reduced atmospheric and environmental engine.

It exists because resolving the entire atmosphere using CFD is usually
absurdly expensive.

The current symbolic atmosphere work in
[`engine_toy/symbolic_atmosphere.py`](engine_toy/symbolic_atmosphere.py)
and [`engine_toy/climate_separator.py`](engine_toy/climate_separator.py)
already provides an excellent foundation:

- Clausius–Clapeyron;
- Antoine saturation;
- relative humidity and supersaturation;
- dew point;
- Köhler droplet activation;
- Hertz–Knudsen phase flux;
- latent-heat exchange;
- droplet settling;
- aerosol/particle behavior;
- optical extinction;
- Rayleigh/Mie transition;
- solid/liquid/vapour fractions.

Bjerknes therefore owns useful reduced atmospheric phenomena such as:

- bulk gas volumes;
- humidity;
- fog/cloud state;
- droplets;
- precipitation;
- aerosols;
- smoke;
- environmental species transport.

A region may remain Bjerknes-level until something demands deeper
fluid resolution. For example:

**room atmosphere: Bjerknes**

while:

**high-speed leak through a valve: Navier–Stokes.**

The two representations exchange the same physical boundary
quantities. Promotion increases resolution; it does not create a new
reality.

**Already real, twice, on purpose.** The same law family already runs
in `turing/examples/symbolic_atmosphere_model.py` — Clausius–Clapeyron,
the Antoine equation, relative humidity and dew point, the Köhler
supersaturation curve, Hertz–Knudsen mass flux, and the Beer–Lambert
extinction signal a renderer needs for fog and cloud. That module and
`engine_toy/symbolic_atmosphere.py` are deliberately the same law
authored twice for two different boundary situations — a dewar/climate
chamber versus open weather — rather than one generalizing the other.
`turing/examples/chamber_dt_join.py` is where this law is actually
joined to the dt system today, alongside the Faraday-family field laws,
inside the chamber examples (`chamber_raincloud_*.py`,
`chamber_2d_pygame.py`).

---

## 6. Gibbs engine — thermodynamic state

Gibbs owns the question:

**What thermodynamic state is this matter in, and what changes are
thermodynamically allowed or favored?**

Its domain includes:

- temperature;
- pressure;
- composition;
- chemical potentials;
- Gibbs/Helmholtz free energies;
- equilibrium;
- metastability;
- phase fractions;
- phase transitions;
- latent heat;
- mixing;
- reaction affinity;
- equations of state.

Gibbs can thicken enormously without requiring Bragg. A gas mixture,
cryogenic liquid, aqueous solution or phase-equilibrium calculation
can depend heavily on Gibbs while requiring no crystal mechanics
whatsoever.

The existing chemistry compendium should couple closely to Gibbs
without allowing every producer to become a competing reaction solver.

---

## 7. Bragg engine — solid-state and material microstate

Bragg owns the question:

**What internal solid-state structure does this piece of matter
possess, and what properties follow from that structure and its
history?**

Its state can independently grow to contain:

- crystal lattice;
- crystal orientation;
- grains;
- grain boundaries;
- vacancies;
- interstitials;
- dislocations;
- defect populations;
- residual strain;
- phase morphology;
- plastic state;
- hardening;
- fracture/damage microstate;
- dopants;
- charge carriers;
- trap states;
- semiconductor bands or reduced band models;
- polarization;
- magnetization;
- phonon-related properties.

Bragg and Gibbs overlap physically but remain ontologically separate.
Gibbs might predict which phase is favored. Bragg records the actual
resulting microstructure and its history.

This distinction allows semiconductor physics to become extremely
detailed without dragging all bulk thermodynamic machinery with it,
while a complex fluid thermodynamic problem can use Gibbs without
requiring lattice physics.

Bragg publishes constitutive properties to other engines:

- elastic tensor;
- yield response;
- conductivity;
- permittivity;
- permeability;
- thermal conductivity;
- thermal expansion;
- mobility;
- carrier response;
- optical properties.

---

## 8. Hamilton engine — quantum state

Hamilton owns physical state that cannot be faithfully represented
solely by classical position and momentum.

Its fundamental abstraction is:

**quantum state + Hamiltonian → evolution/eigenstates/observables.**

Possible fidelity lanes include:

- analytic solutions;
- finite Hamiltonian matrices;
- tight-binding/Hückel-like models;
- quantum oscillators;
- electronic-structure approximations;
- Hartree–Fock;
- density-functional methods;
- time-dependent evolution;
- reduced spectroscopy models.

Hamilton should not imply that every atom in the game continuously
receives an electronic wavefunction. Instead, quantum regions are
promoted when their questions require them.

### Molecular geometry

Molecular geometry naturally becomes a Hamilton–Newton coupling. For
nuclear coordinates `R`, Hamilton supplies an electronic energy
surface `E_e(R)` and therefore forces:

```
F_i = -d(E_e)/d(R_i)
```

Newton advances the nuclei. This provides a path from cheap classical
molecular force fields through increasingly sophisticated electronic
structure without replacing the surrounding simulation architecture.

Expensive quantum work can also bake reduced artifacts:

- equilibrium geometry;
- force-field coefficients;
- polarizability;
- vibrational modes;
- reaction barriers;
- transition energies;
- dielectric response;
- effective potentials.

---

## 9. Curie engine — nuclear state and radiation

Curie owns nuclear identity and nuclear transformations.

Its domain can include:

- isotopes;
- nuclear levels;
- decay chains;
- half-lives;
- branching;
- radioactive emissions;
- nuclear reaction state;
- deposited decay heat;
- radiation-source terms;
- activation/transmutation where eventually justified.

Curie produces particles/radiation/energy which other engines consume.
Curie is therefore distinct from chemistry: changing chemical bonds
does not change the nucleus; Curie begins exactly where that
assumption ceases to hold.

---

## 10. Einstein engine — relativistic spacetime

Einstein owns relativistic spacetime/kinematics when classical
treatment is no longer sufficient. Ordinary machinery should not pay
its computational cost.

Einstein becomes relevant where questions require:

- relativistic momentum/energy;
- proper time;
- Lorentz transformation;
- high-energy particle motion;
- relativistic electromagnetic coupling;
- eventually curved spacetime if there is ever a reason to model it.

Newton remains the ordinary motion authority. Einstein is a promoted
physical description, not the default game-physics layer.

---

## 11. Chemistry (Lavoisier) — chemical identity and reaction authority

Chemistry should remain a master authority for:

- species identity;
- elemental composition;
- charge;
- reaction topology;
- equilibrium laws;
- kinetic laws;
- reaction networks.

The existing engine/chemistry boundary in
[`engine_toy/chemistry_boundary.py`](engine_toy/chemistry_boundary.py)
(exercised by
[`engine_toy/tests/test_chemistry_boundary.py`](engine_toy/tests/test_chemistry_boundary.py))
already expresses the correct architectural rule:

**a machine emits species flux; it does not invent another chemistry
solver.**

Boundaries must conserve elements, charge and mass before handing
material to chemistry.

Chemistry in turn couples to:

- Gibbs for thermodynamic driving state;
- Bjerknes/Navier–Stokes for transport;
- Hamilton where electronic structure is required;
- Bragg for chemical modification of solids;
- Curie only when nuclear identity changes.

---

## 12. Thermal physics (Fourier) — a shared continuum consumer

The existing mesh thermal machinery in
[`engine_toy/thermal_domains.py`](engine_toy/thermal_domains.py),
[`engine_toy/thermal_atlas.py`](engine_toy/thermal_atlas.py), and
[`engine_toy/thermal_storage.py`](engine_toy/thermal_storage.py)
should not be discarded merely because Gibbs and Bragg become deeper.
It is exactly the sort of efficient continuum solver the architecture
needs.

The change is that its material properties become **resolvable**. A
triangle can begin with:

**material key → tabulated cp, k, rho, emissivity.**

If deeper state is requested:

**Gibbs → phase/composition state**

and/or

**Bragg → structural/material microstate**

then the same thermal triangle receives:

**cp(state), k(state), rho(state), emissivity(state), latent terms …**

The heat equation remains the heat equation. Only the quality of its
constitutive answer improves. This pattern should apply throughout the
simulator.

---

## 13. Physical promotion: any triangle can become deep

Every geometric primitive should be capable of participating in
progressively deeper physical descriptions.

A triangle might begin as:

**geometry + material + thickness + boundary role.**

It can then acquire:

**continuum thermal state**
→ **Gibbs thermodynamic state**
→ **Bragg microstructural state**
→ **atomistic state**
→ **Hamilton quantum region**

when required.

Likewise a fluid volume can move:

**bulk Bjerknes state**
→ **resolved Navier–Stokes region**
→ **explicit particles/molecules**
→ **Hamilton treatment locally**

without changing its physical identity at the exterior boundary.

The simulator therefore uses **promotion and reduction**, not
mutually exclusive physics engines. The deeper model can be
instantiated as a Nodus subgraph behind the same exterior ports. When
it becomes unnecessary, its behavior can be reduced/baked back into a
cheaper representation.

---

## 14. SymPy laws as scientific authority

Where practical, governing laws should be authored as explicit SymPy
equation sets. These equations are not comments describing an
implementation. They are the executable mathematical authority.

The established pipeline is:

**simultaneous SymPy equations → ProcessGraph → SSA → compiled
backend → Nodus-hosted kernel.**

A proposed optimization or implementation change can therefore be
judged against something stronger than a regression test. For each
law, preserve:

- equation set;
- physical assumptions;
- dimensions/units;
- domain predicates;
- conservation laws;
- stability bound;
- limiting behavior;
- state ABI;
- independent reference calculations;
- compiled artifact provenance.

A self-consistent implementation is insufficient evidence. Validation
should deliberately reach the same physical quantity by an independent
path wherever feasible.

---

## 15. `dt_system` owns advancement

No fundamental engine invents its own world time. Each engine exposes:

- state;
- derivative/update law;
- admissible timestep or stiffness information;
- errors/invariants;
- snapshot/restore;
- accepted advancement metrics.

`dt_system` selects the numerical method and manages:

- integrator choice;
- microstepping;
- adaptive stepping;
- rollback/retry;
- stability;
- nested time domains;
- scientific versus realtime advancement.

This separation is essential. A molecular calculation may prefer
velocity Verlet. A structural assembly may use Newmark. A Hamiltonian
field may use symplectic/leapfrog evolution. A stiff chemical system
may need an implicit method. They can all live under the same
managed-time contract.

**Already real.** This is not a proposed contract; it is the one
`turing/src/common/dt_system/` already enforces. An engine's obligation
is the `DtCompatibleEngine` shim in `dt_system/engine_api.py`
(`step`/`step_with_state`, plus optional `preferred_dt`); its
advancement request/outcome pair is `SuperstepPlan`/`SuperstepResult`
in `dt_system/dt.py`; its errors and stability numbers are the
`Metrics` record in `dt_system/dt_scaler.py`
(`max_vel`, `max_flux`, `div_inf`, `mass_err`, `dt_limit`, `energy_j`,
`power_w`); and the caller-declared tolerances are `Targets` in
`dt_system/dt_controller.py` (`cfl`, `div_max`, `mass_max`, plus the
energy-exchange-fraction and shadow-growth-amplification pins). The
controller that ties them together, `STController` /
`run_superstep`, is the one call every lowered law in the tree already
goes through — `vehicle_python_compilation.py`, `native_voxel_fluid.py`,
`symbolic_fluid_dt.py`, `symbolic_fluid_native_runtime.py`, and this
proposition's own `chamber_dt_join.py` all call it the same way. No
engine anywhere in that list picks its own integrator or its own clock.

---

## 16. Turing and Nodus

Turing is the **toolchain and bootstrap environment**, not the
runtime. Its job is to ingest source, normalize semantics, build the
canonical graph/SSA, verify it, specialize it, emit target artifacts,
and package those artifacts together with enough source/provenance/
schema information to reproduce or inspect them later.

Nodus is the **living host**. It owns the persistent graph,
instantiates compiled kernels as processes, wires their ports,
schedules them, stores their state, moves products through FIFOs,
decides placement, and eventually distributes work across server and
client workers.

The lifecycle is:

```
source
  -> Turing frontend(s)
  -> canonical graph/SSA
  -> validation/reference execution
  -> specialization/backend compilation
  -> packaged kernel + ABI + source/provenance
  -> Nodus import
  -> graph placement/wiring
  -> persistent execution
```

At that point Turing can almost disappear from the running game. It is
analogous to a compiler toolchain installed on the build machine:
indispensable for creating and updating executable machinery, but not
something every simulation tick has to know about.

This also resolves an architectural tension rather than papering over
it: Nodus does not need to become a compiler, and Turing does not need
to become a distributed world runtime. Each gets to become very good
at its actual job.

### The packaged unit is a computational object, not a binary

What Turing hands to Nodus should be the durable unit of software in
the world — not merely a `.dll` or `.wasm`, but something closer to a
computational object containing:

- the compiled kernel artifact;
- canonical source/graph identity;
- ABI/state schema;
- input/output contracts;
- required capabilities;
- available backend variants;
- a deterministic digest;
- version/provenance;
- ideally a reference-test vector.

That matters because Nodus can then treat a compiled engine, tire,
optical solver, turret controller, or chemistry law uniformly. It does
not care whether the original author wrote Python, symbolic SymPy, C,
or eventually some other language — by the time Nodus sees it, it is a
process with declared state and ports. Keeping the source alongside
the artifact also means the running world does not become a graveyard
of opaque binaries: a hosted kernel can still be inspected, recompiled
for another architecture, upgraded, audited, or migrated to a
different backend.

So the language-indifference this proposition keeps returning to pays
off only at this one boundary:

**Turing cares what language the program came from. Nodus does not.**
Nodus only needs to know what the computation *is*, what state it
owns, what it consumes and publishes, and what execution capabilities
its artifact requires.

**Already real, for the Turing half.** `turing/examples/
llvm_dt_system.py` already runs the left-hand side of this lifecycle
end to end for the LLVM/C/Fortran backend family: a set of authored
Python "pieces" is compiled to `LLVMPiece` artifacts by
`src.compiler.native_law_kernels`, `state_source`/`piece_source` spell
a canonical `PieceState` and `advance_pieces` for exactly that piece
set, and the result runs under the same `run_superstep` used
everywhere else in the tree. `turing/examples/chamber_dt_join.py`
shows the same idea one layer up: `CompiledLaw.from_equations` wraps a
SymPy law compiled through `compile_sympy_equations`, and
`CompiledLaw.from_piece` wraps an already-compiled `LLVMPiece`, and
both become the same `LawEngine` object with the same `advance(state,
dt)` call — which is exactly "same declared operation semantics,
multiple valid implementations" (§18), demonstrated today, one rung
below where Nodus is eventually meant to sit. Nodus itself — the
persistent C++ graph host — is the half of this lifecycle that does
not exist yet.

---

## 17. Nodus as a composition language, not merely a host

Once Turing can turn arbitrary source languages into safe,
well-described computational modules, Nodus is not merely the host for
those compiled kernels — it is also the **composition language and
runtime for assembling trusted modules into larger executable
machines**.

The division of labor becomes:

Turing is responsible for taking arbitrary source languages and
turning them into safe, well-described computational modules.

Nodus is responsible for taking those modules and saying: *these
pieces may be connected this way, with these ports, these state
boundaries, these scheduling rules, these capabilities, and these
execution constraints.*

Nodus's own language therefore does not compete with Python, C,
Fortran, or any other authoring language — it sits **above** them.
Python might describe an engine law; C might describe a
highly-optimized solver; SymPy might describe a constitutive relation;
GLSL might describe a field kernel. Turing reduces each of those into
a portable module artifact. Nodus's language then composes, for
example:

```
engine
  - cooling loop
  - battery
  - turret hydraulic system
  - fire control
  - ballistic integrator
```

into one larger computational object.

Because Nodus understands module boundaries rather than arbitrary
language internals, it can enforce things a normal programming
language cannot easily enforce globally:

- which modules may mutate which state;
- which ports can connect;
- tensor/type compatibility;
- causality/readiness;
- scheduling policy;
- resource and execution capabilities;
- whether something may run remotely;
- whether a module is deterministic/replayable;
- what state must be persisted;
- whether a connection crosses a trust boundary.

Compiled units can themselves become modules again, which gives a
recursive property: **modules compose into programs; programs compile
into modules; those modules compose again.** That makes Nodus much
closer to a computational construction language than a conventional
runtime — and it is why the compiler is the critical nexus: Turing is
what allows arbitrary external software to become a valid Nodus
building block, and once that conversion is reliable, Nodus already
supplies the safe higher-order language for combining those blocks
into the actual world.

---

## 18. A pragmatic execution hierarchy, not one purity path

Nodus being C++ means it can host the mature native ecosystem
directly — Eigen for lightweight numerical kernels, Torch/libtorch
where tensor/autograd machinery is useful, a portable custom tensor
implementation when full control and portability matter, plus whatever
compiled artifacts Turing emits. The architecture therefore tolerates
multiple implementation classes rather than forcing everything through
one:

```
native C++ modules       -> Eigen / libtorch / custom tensor kernels
portable authored modules -> Turing frontend
                           -> canonical graph/SSA
                           -> C / LLVM / WASM / GLSL / etc.
Nodus composition          -> connects all of them under one
                              module/port/scheduling model
```

That is stronger than trying to make Turing replace the C++ ecosystem.
Turing only has to solve the cases where language indifference,
portability, introspection, or backend mobility are actually needed.
If a subsystem is already best expressed as a native C++/Eigen kernel,
Nodus can just host it. The custom tensor implementation becomes the
bridge layer: it can exist natively in Nodus while having semantics
simple enough that Turing's Python side can reproduce or compile
against it, giving a common computational vocabulary without demanding
every execution path use the same implementation.

The key invariant is:

**same declared operation semantics, multiple valid implementations.**

`matmul`, scatter/gather, FFT-ish operators, reductions, and field
stencils can each have an Eigen implementation, a libtorch
implementation, a custom-portable implementation, and a
Turing-generated implementation, while Nodus sees one capability/
operation contract. "Language indifferent" therefore does not mean one
language replacing all the others — it means one graph/composition
system capable of treating implementations from different languages as
equivalent computational components whenever they satisfy the same
contract. `CompiledLaw.from_equations` versus `CompiledLaw.from_piece`
in `turing/examples/chamber_dt_join.py` (§16) is the same invariant
already working at the dt-system layer, beneath where Nodus will sit.

---

## 19. Self-improving compilation

Because Nodus can observe graph execution, it can identify expensive
or awkward paths:

- external Python nodes;
- repeated subgraphs;
- serialization boundaries;
- expensive network calls;
- unfused kernels.

Those can become compilation tasks. A safe optimization cycle is:

**observe → identify candidate → compile/reduce → execute original
and candidate → compare against mathematical/reference authority →
benchmark → retain candidate as an alternate implementation → promote
only after verification.**

The existing implementation remains the oracle until the replacement
proves equivalence. Thus the computational network can gradually
harden frequently used pathways into efficient local compiled
machinery.

---

## 20. The controlled chamber as the reference integration environment

The Faraday chamber/dewar lineage
([`engine_toy/FARADAY_CHAMBER_CONCEPTION.md`](engine_toy/FARADAY_CHAMBER_CONCEPTION.md),
[`engine_toy/FARADAY_CHAMBER_STATUS.md`](engine_toy/FARADAY_CHAMBER_STATUS.md))
provides an excellent experimental deployment for the whole package.
Within one controlled volume:

**Faraday** models fields, circuits, cavities and EM heating.

**Navier–Stokes** models resolved flow through ports and detailed
internal flow.

**Bjerknes** models atmosphere, vapour, droplets, aerosol and
precipitation.

**Gibbs** models composition, phase and thermodynamic closure.

**Bragg** models the evolving solid-state material of the walls,
contents and devices.

**Newton** moves particles, droplets and mechanical objects.

**Timoshenko** handles reduced structural deformation and vibration of
appropriate members.

**Hamilton** can be invoked locally for electronic/material/molecular
questions.

**Curie** can provide nuclear/radiation processes where relevant.

**Einstein** exists when relativistic physics genuinely enters the
problem.

The thermal mesh connects all of them through energy exchange. The
chamber therefore becomes not merely another simulation but a
**reference convergence experiment** for the whole simulator
architecture.

**Already partly real.** `turing/examples/chamber_dt_join.py` joins
Faraday-family field laws and the Bjerknes-family atmosphere law to
`run_superstep` today, and `turing/examples/chamber_raincloud_*.py` and
`chamber_2d_pygame.py` exercise that join as running demos. What is
not yet real is the rest of the roster in one volume at once — Gibbs,
Bragg, Timoshenko, Hamilton, Curie, and Einstein are not yet part of
that same joined chamber. The convergence experiment is started, not
finished.

---

## 21. Central architectural rule

The simulator should never ask:

> Which engine owns this object?

It should ask:

> Which physical descriptions are currently attached to this object's
> state?

A steel triangle can simultaneously be:

- geometry to the renderer;
- thermal area to the heat solver;
- a boundary face to Navier–Stokes;
- a PEC/lossy boundary to Faraday;
- a structural surface/member participant;
- Gibbs matter with thermodynamic state;
- Bragg matter with evolving microstructure.

Those are not copies of the triangle. They are different questions
asked of **the same physical object**.

That is the proposed foundation for the ideal simulator package.

---

## 22. Addendum: how Turing's compiled kernels actually meet Nodus's table/stack/graph architecture

§§16–18 stated the Turing/Nodus split and the "same declared operation
semantics, multiple valid implementations" invariant at the level of
intent. This addendum grounds that split in the concrete mechanisms
already built on the Nodus side (`nodus/`, a sibling repository, not
tracked here), and names the exact seam where a Turing-compiled kernel
crosses into it.

### Turing's half: a compiled kernel plus its calling contract

Turing does not merely emit a `.dll`/`.so`/`.wasm` and stop. Every
auto-compiled artifact is emitted beside a machine-readable calling
contract — `turing/src/compiler/compiled_program_api.py`'s
`turing-compiled-program-api-v1` YAML, generated from the same
`Function` objects the code generator itself used, so it states each
entry point's C type, passing convention (value/reference), argument
role (input/output/extent), and shape authoritatively. A caller never
has to read generated source and guess an ABI; it reads the contract
that was emitted as a description of what was generated, not a second
source of truth that could disagree.

### Nodus's object contract: `ToolIR`/`ITool`, ports, and a stack

Every composable unit Nodus can run — hand-written, generated, or
imported — implements one object contract, `ITool` (`nodus/include/
tool_api.h`) or its data-only equivalent `ToolIR` (`nodus/include/
tool_ir.h`, a `BuiltinTool` wraps one). That contract declares:

- **ports**, via `port_count()`/`port_spec()`, each one an
  `Argument`, `Return`, or `Internal` `ToolPortSpec` — the same
  argument/return/internal distinction Turing's calling contract
  already records for a compiled entry point;
- **execution**, via `execute_stack(ToolStackContext&)` — values flow
  through a `ToolStackFrame` popped and pushed with
  `tool_stack_pop`/`tool_stack_push` (and block/int variants), so a
  tool's actual computation is "pop what my argument ports declared,
  push what my return ports declared," independent of how the tool's
  own body is implemented;
- **table presentation**, via `render_table(TableRenderArgs&)`
  (`caps() & ToolCaps::Table`), so the same object that computes a
  result can also draw its own row in the host's table view;
- **persistence**, via `serialize`/`deserialize`.

A tool is therefore always the same kind of object to Nodus — a set of
typed ports plus a stack-executable body plus optional presentation —
regardless of whether a human wrote it by hand, a module-composition
tool generated it, or an external artifact importer generated it.

### Three ways an object reaches Nodus, one contract

`nodus/docs/BUILD_YOUR_OWN_PLUGIN.md`, `nodus/include/module_library.h`
+ `nodus/docs/module_library_build.md`, and `nodus/include/
repo_package.h` are three loading tiers for exactly the same `ITool`
contract, not three different contracts:

1. **Hand-written plugin.** A `.cpp` implements `ITool` directly and
   exports `create_tool()`/`destroy_tool()` (`BUILD_YOUR_OWN_PLUGIN.md`).
   The author writes the ports and the stack body themselves.
2. **Module-library actualization.** A serialized `.gpmod` composition
   is actualized into generated C++ tool sources
   (`gp_module_library_actualize_from_file`,
   `module_library_actualizer.cpp`), compiled into DLLs under
   `module_library/build/` (kept deliberately separate from the repo's
   own build), and loaded by the same `PluginLoader` a hand-written
   plugin uses. The tool object is generated, not authored, but it is
   still exactly an `ITool`.
3. **Repo package ingestion.** `gp_repo_package_ingest_from_file`
   (`repo_package.h`/`.cpp`) imports a whole external repository's
   *named, listable, unloadable-as-a-unit* bundle of tools and type
   contracts (`GP_RepoPackage`, the NODUSPKG format) — and each tool
   entry may be **either** a prebuilt DLL **or** hand-written source.
   For source, `repo_package.cpp`'s `build_contract_dll` writes a
   scratch `CMakeLists.txt`, configures and builds it as a shared
   library against the host's own `canvas_tables` library, and loads
   the resulting DLL — so "provide the source, Nodus compiles and
   composes it" is not aspirational, it is the literal behavior of
   ingesting a package whose tool entry names a `.cpp` file rather than
   a `.dll`.

### The literal bridge: `artifact_importer` turns a Turing kernel into a Nodus tool

`nodus/include/artifact_importer.h` is the seam named above, made
concrete. `gp_artifact_import` takes exactly the two things Turing
produces for a compiled entry point — the artifact
(`.dll`/section library) and its `turing-compiled-program-api-v1` YAML
descriptor — and, using the YAML's authoritative
`c_type`/`passing`/`role`/`shape` fields (never by parsing generated
source), emits:

- a generated `ITool` wrapper (`source/tools/imported/<tool>.cpp`)
  that pops `VT_ABSTRACT_TENSOR` handles for the entry point's input
  parameters, maps them through the `nodus_tensor_*` transport,
  resolves any extent parameters from the arrays that name them, calls
  the artifact's entry symbol with its exact declared signature, and
  pushes the output tensor's handle;
- a `<tool>.nodus_package.txt` NODUSPKG manifest whose `TOOL` entry
  names that wrapper source and tags the artifact's source language as
  a capability (backend-group labeling for imported operator sets),
  ready for `gp_repo_package_ingest_from_file` — i.e. a Turing-compiled
  kernel re-enters Nodus through the exact same package-ingestion path
  described above, as an ordinary tool object with declared ports.

Its v1 boundaries are stated rather than silently narrowed: float64
tensors only, exactly one output parameter, a static or same-extent-
inferable output shape, and entry points with parameters only (no
card-program/arena artifacts) — each unmet case is a reported
shortfall (`GP_ArtifactImportReport`), never a quiet approximation.

### Composition above the object: the graph

Above individual tool objects, `nodus/include/common/tensors/
abstraction/graph_ir.h` supplies `GraphEdit`/`GraphEditBuilder` — the
structural layer that wires tool ports together into a graph, the
"table/stack/graph" architecture's third term. `nodus/docs/
process_graph_interop.md` names the two paths a Turing `ProcessGraph`
actually takes into that graph, and states plainly that they are kept
separate on purpose:

```
ProcessGraph -> GraphIR -> registered AbstractTensor ToolIR nodes and ports
SSA / FusedProgram -> KernelIR or Nodus in-memory calculator execution
```

The first preserves composition and UI structure — a ProcessGraph node
becomes a graph node with ports, inspectable and rewireable like any
other tool. The second is a fusion-suited execution packet: an SSA
program lowers toward `KernelIR`'s Tier-0 portable kernel ISA
(`nodus/docs/TIERS.md`) — a deliberately small, universally-
implementable instruction set that every backend emitter (SPIR-V,
GLSL, CPU, Eigen, Metal, ONNX, torch, Vulkan, Fortran) understands
directly, with anything non-primitive expressed as a Tier-1 *recipe*
over Tier-0 rather than a new opcode, so one definition reaches every
backend. Tier-2, the `AbstractTensor` layer itself, is where the ~300-
operation tensor surface and autograd actually live, and needs Tier-0
only when an operation must run as a device kernel.

### Closing the loop

This is §17's recursive property, made literal rather than aspirational:
a Turing-compiled law becomes a Nodus `ITool` object via
`artifact_importer`; that object composes with hand-written and
module-library-generated tools under one `ITool` contract because all
three are ingested through the same `repo_package`/`PluginLoader`
machinery; the same machinery can compile *source* on the spot when no
prebuilt artifact exists yet; and the resulting graph of tool objects
is itself an ordinary `GraphEdit` structure that can be inspected,
saved, and recomposed — the same object grammar all the way up.
Nothing here required Nodus to understand Python, SymPy, or whatever
language a given law was authored in. It only ever needed to
understand ports, a stack, and a graph.
