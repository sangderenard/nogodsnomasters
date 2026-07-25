# Alpha / Kakarot lineage audit

**Audit date:** 2026-07-25

**Primary archaeological source:** `C:\Alpha`

**Current descendants examined:** `speaktome`, `transmogrifier`, `turing`,
`spectral-analyzer` / Pluck, and the root coordination material

## Executive finding

Kakarot is not a missing project that should be copied intact into the current
workspace. It is the earliest place found where several ideas that now live in
different repositories were still one rough system:

- geometry-parametric isosurface extraction (`CompositeGeometry`, `Shuttle`,
  `YoungManAlgorithm`);
- object authority, locking, and messages (`ThreadManager`);
- bounded, budgeted state correction between networked simulators
  (`StateNegotiationContract`, `DebtArbitor`, `PillbugNetwork`);
- neural surrogates trained against local physics projections
  (`TrainingCoordinator`);
- and a game in which operating and progressively understanding a simulation
  contributes real computational work.

That combination is historically important. Its useful descendants, however,
are now more mature because they are separated:

- **SpeakToMe** preserved and developed the YoungMan and Laplace experiments.
- **Transmogrifier** turned implicit execution order into explicit graph
  scheduling and resource-interference analysis.
- **Turing** generalized Laplace/metric work to `AbstractTensor`, added a neural
  Riemannian block, absorbed Transmogrifier, and developed transactional managed
  time.
- **Nodus** is becoming the graph authority and execution/surface boundary.
- **Pluck** in `spectral-analyzer` is becoming the inhabitable world: a player,
  materials, fabrication, unfinished stations, simulators, cameras, and
  graph-driven physical/optical instruments.

The correct recovery is therefore **selective reconstruction behind current
interfaces**, not resurrection. In particular, Turing contains the
metric-Laplace and neural machinery needed by a future YoungMan successor, but
it does **not** yet contain YoungMan's geometry/topology compiler or its
isosurface extraction contract.

## Confidence notation

- **Verified:** present in the source inspected for this audit.
- **Strong correspondence:** not proven by commit history, but the structures
  and responsibilities align closely enough to guide integration.
- **Proposal:** an architectural or gameplay recommendation, not a claim about
  existing code.

`C:\Alpha` is a collection of nested snapshots rather than a clean history.
Names such as `Level 8`, `Level 16`, and `Level 17` are useful archaeological
strata, not reliable semantic versions or game levels.

## The recovered lineage

```text
Alpha / Kakarot
├─ CompositeGeometry + Shuttle + YoungManAlgorithm
│  └─ SpeakToMe notebook lineage
│     ├─ preserved Torch YoungMan implementation
│     └─ Laplace / DEC geometry experiments
│        └─ Turing AbstractTensor Laplace + learned Riemannian block
│
├─ ThreadManager locks + rotating object tokens + mailboxes
│  ├─ Transmogrifier dependency, phase, and interference scheduling
│  │  └─ integrated at turing/src/transmogrifier
│  └─ Nodus graph handles, edges, and execution authority
│
├─ StateNegotiationContract + networked local simulators
│  ├─ Turing managed-time windows, rollback, and retry
│  └─ Nodus/Pluck KPN and simulator coordination
│
└─ gamified simulator / learning-machine plot
   └─ Pluck's player, fabrication, construction, stations, and instruments
```

This is a conceptual and source-lineage map, not a claim that every arrow has a
continuous Git history.

## 1. Kakarot geometry and YoungManAlgorithm

### Best recovered implementation

The strongest working contextual source found is:

`C:\Alpha\Alpha\import_uuid\tests\original tests\Level 16\Development\Kakarot\Primitives\isosurface4.py`

Its runnable context includes:

- `Primitives\ziggy.py` — the visual YoungMan demo;
- `Primitives\audio.py` — an audio-oriented demo/context;
- `Primitives\compositegeometry.py`;
- `Primitives\triangulator.py`;
- `Primitives\adaptivegraphnetwork.py`;
- the `geometry_maps` cache used to avoid rebuilding topology data.

The demo was exercised during this archaeology. It loaded the cached
tetrahedral geometry and reached its OpenGL path. A small optional
`torch_geometric` compatibility stub was needed in this old environment.
That establishes the Alpha copy as more than an isolated class fragment: it is
the recovered runnable context the user remembered.

SpeakToMe contains a clear edit lineage:

| Snapshot | Approximate class size | Significance |
|---|---:|---|
| `training/notebook/173299...YoungManAlgorithm.py` | 960 lines / 27 methods | Earlier form |
| `training/notebook/173307...YoungManAlgorithm.py` | 908 lines / 28 methods | Intermediate form |
| `training/notebook/1733534643_YoungManAlgorithm.py` | 1,411 lines / 34 methods | Strongest standalone SpeakToMe form found |
| `training/notebook/1733612620_CompositeGeometry_Shape_Shuttle_YoungManAlgorithm.py` | same YoungMan class | Best bundled geometry context in SpeakToMe |

The YoungMan class in the final bundled SpeakToMe snapshot is byte-identical to
the strongest standalone SpeakToMe class. No `AbstractTensor` YoungMan port was
found.

### What makes the algorithm beautiful

YoungMan's depth of parallelism is structural, not cosmetic. It lays out work
at several nested scales:

1. a sampled spatial domain;
2. grid sites in that domain;
3. one or more jittered parametric geometry instances per site;
4. vertices and edges within every geometry;
5. batched scalar-field queries at transformed vertices;
6. vertex bitmasks and active-edge selection;
7. interpolated edge crossings;
8. candidate triangles selected from precomputed maps;
9. centroid/refinement and force/torque accumulation;
10. batched render buffers.

The expensive combinatorics are compiled into reusable maps:

- vertex offsets and counts;
- edge pairs and edge lengths;
- active-vertex and triangle maps;
- force and torque maps;
- triangle masks and centroids;
- tile sizes and buffer layouts.

That makes the algorithm closer to a **topology compiler plus a tensor query
runtime** than to ordinary marching cubes. `CompositeGeometry` supplies
parametric cell families (square, cube, tetrahedron, octahedron, icosahedron,
and network-configured forms). `Shuttle` combines weft and warp geometries and
supports asymmetric/probabilistic interpolation. YoungMan then evaluates a
field across huge batches of those local geometric programs.

### Why it did not follow Laplace3D automatically

Laplace3D is dominated by numerical tensor operations: metric evaluation,
finite differences/operators, eigendecomposition, and convolution. Those are
natural targets for an `AbstractTensor` backend.

YoungMan also contains:

- discrete topology and integer lookup tables;
- variable geometry cardinalities;
- cache construction;
- bitmask interpretation;
- rendering-buffer ownership;
- history, motion, force, and torque concerns;
- assumptions about Torch/CUDA and OpenGL allocation.

Porting the class mechanically would reproduce these couplings behind a new
tensor spelling. The missing prerequisite is a stable geometry contract, not
merely more `AbstractTensor` methods.

### Kakarot geometry applicability

| Kakarot responsibility | Current home or candidate | Assessment |
|---|---|---|
| Parametric cell definitions | Turing geometry package or a small backend-neutral geometry schema | Recover |
| `CompositeGeometry` composition | Turing/Nodus structured graph representation | Redesign around explicit typed data |
| `Shuttle` weft/warp construction | Geometry compiler transform | Preserve concept and reference outputs |
| Topology-map compilation | Pure backend-neutral compiler with integer tensors | Extract first |
| Batched scalar-field evaluation | `AbstractTensor`-capable Turing runtime | Strong fit |
| Metric-aware sampling/refinement | Turing Riemannian/Laplace machinery | Strong fit, but not implemented |
| Scheduling the work graph | Turing Transmogrifier / Nodus executor | Strong fit |
| Render-buffer allocation | Pluck renderer | Keep outside the geometry core |
| Motion, force, torque history | Managed-time simulation layer | Keep outside the topology compiler |

**Proposal:** define a `ParametricCellComplex` (name provisional) whose output
is backend-neutral topology and whose inputs are geometry parameters. Then
implement YoungMan as stages over that contract. Integer topology should remain
integer topology; only field evaluation, interpolation, refinement, and
metric-aware operators need tensor-backend abstraction.

## 2. Turing has the metric-Laplace neural architecture

The user's recollection is correct.

### Verified pieces

- `turing/src/common/tensors/riemann/manifold.py`
  - defines `ManifoldPackage`;
  - builds `BuildLaplace3D`;
  - obtains the metric from `transform.metric_tensor_func`;
  - uses the abstract tensor linear-algebra path when appropriate.
- `turing/src/common/tensors/abstract_convolution/metric_steered_conv3d.py`
  - defines `MetricSteeredConv3DWrapper`;
  - combines `BuildLaplace3D` with metric-steered separable 3-D convolution.
- `turing/src/common/tensors/riemann/grid_block.py`
  - defines `RiemannGridBlock`;
  - combines the geometry package, learned casting/assignment, FiLM-like
    modulation, and metric-steered convolution.
- `turing/docs/riemann_grid_block.md`
  - documents the intended learned Riemannian computation.

This is the architectural setup that made the user's inference plausible:
Turing possesses learned metric geometry and an abstract Laplace path.
What is absent is the discrete parametric cell system that tells YoungMan what
vertices, edges, crossings, and triangles mean.

### Relationship to later DEC work

SpeakToMe's later Laplace archives include `CompositeGeometryDEC`,
`HodgeStarBuilder`, `FaceMapGenerator`, `MetaNetwork`, `Shape`, `Shuttle`, and
`SphereGraph`. Alpha Level 17 likewise contains `CompositeGeometryDEC` and
metric-tensor Laplace experiments. This is important evidence that the
YoungMan geometry question was already branching toward discrete exterior
calculus.

It is not evidence that the question was settled. The audit found several
partially overlapping geometry models, exactly matching the user's memory that
the special geometric basis had not been accepted as final.

## 3. Lock scheduling: from object mutexes to explicit execution

### Kakarot's `ThreadManager`

**Verified:** `Primitives\threadmanager.py` is a central registry containing:

- one lock per registered object;
- one mailbox per object;
- a token checked before lock acquisition;
- token rotation on release;
- message send/receive operations;
- a manager-wide lock protecting the registry.

The durable idea is an object-scoped capability and an authority boundary:
“who may mutate this object now, and how does another actor request work?”

The implementation should not be revived:

- it can hold the global manager lock while waiting on an object lock, creating
  convoy/deadlock risk;
- its claimed timestamp/blockchain security is not supplied by the actual UUID
  construction;
- rotating UUIDs are not a distributed-consensus protocol;
- execution dependencies remain implicit in imperative lock acquisition.

### Transmogrifier's improvement

**Verified:** the active code is now inside
`turing/src/transmogrifier/`; the standalone `transmogrifier` directory is a
legacy snapshot.

`ILPScheduler` and `graph_express2.py` expose:

- ASAP and ALAP levels;
- maximum-local-slack scheduling;
- harmonic phase-lock constraints;
- node lifespans and storage demand;
- interference scheduling;
- explicit time slices (`run_at`);
- recombinatoric transformation levels.

This is a strong architectural successor to Kakarot's locks. Instead of asking
threads to discover safety by contending for mutexes, the graph makes
dependencies, concurrency, phase relationships, and resource overlap visible.

### Current coordination direction

Nodus and Pluck documentation adds:

- typed sparse graph authority;
- Kahn-ready parallel scheduling for acyclic graphs;
- ASAP/ALAP slack ordering for cyclic graphs;
- explicit graph-to-runtime boundaries.

**Proposal:** preserve Kakarot's capability concept at user/plugin/handle
boundaries, but use graph scheduling and transactional state for computation.
Do not use token possession as a substitute for dependency analysis.

## 4. Network resolution of physics simulators

Kakarot's networking directory is rough but unusually revealing.

### Verified model

- Every `RandomWalkSimulator` has position-like state, velocity, acceleration,
  temperature, emissivity, and its own `TrainingCoordinator`.
- A six-cycle coordinator:
  1. projects selected axes through `ChipmunkSlice`;
  2. asks a neural wrapper to predict the result;
  3. compares projection and prediction and turns error into heat/budget;
  4. prepares training material from state history;
  5. trains locally;
  6. submits weights, gradients, and losses to a `HeadTrainer`.
- `StateNegotiationContract` bounds corrections by per-axis force limits and
  bounds, prices them with exchange rates, and rejects work above a debt limit.
- `DebtArbitor` performs analogous arbitration per edge.
- `ContractRenegotiator` releases constrained axes as debt approaches its
  threshold.
- `PillbugNetwork` and graph utilities arrange simulators as graph nodes and
  propagate/correct state.
- `StateResolver` separately fills incomplete state, interpolates from prior
  states or streams, tracks confidence/authority, and validates the result.

The vocabulary of debt, radiation, temperature, and contracts is inconsistent,
but the underlying idea is valuable: **a neighboring simulation may propose a
correction, yet the receiver applies only a bounded, typed, resource-accounted
part of it, while prediction error creates learning material.**

### Applicability now

| Kakarot concept | Modern interpretation |
|---|---|
| Bounds | Declared state invariants / admissible correction envelope |
| Force limits | Per-window actuation or correction budget |
| Exchange type | Typed port / quantization contract |
| Debt | Error, energy, bandwidth, or compute budget—kept as separate named channels |
| Renegotiation | Backpressure or a graph-authority contract update |
| Local simulator | Pluck instrument/physics worker |
| Head trainer | Explicit aggregation service, if training is desired |
| Temperature/radiation | Physical state only when physically meaningful; otherwise telemetry |

Turing's managed-time runtime is the stronger descendant for physics
resolution. It provides authored event boundaries, multirate windows,
candidate advances, validation, rollback/retry, and complete state restoration.
Nodus/Pluck can supply graph topology, port typing, scheduling, and
backpressure. This combination is much safer than allowing network neighbors
to mutate one another through ad hoc correction contracts.

**Proposal:** retain the Kakarot experiment as a reference model and test
fixture. Re-express it as transactional proposals over managed-time windows:
each node advances locally, publishes a candidate boundary state and named
error channels, and the authority commits, retries, or rejects the window.

## 5. The Kakarot gameplay plot and Pluck

### Earliest recovered plot

The clearest statement is the Level 8 `patent.txt`. It describes:

- a player navigating a drone through wind and obstacles;
- progression from linear networks to CNNs and GNNs for autopilot;
- successful play contributing to coordinated physics predictors;
- distributed, highly specific learning tasks;
- tokens accumulated through participation;
- an educational game that exposes progressively more of the real machinery.

The Bitcoin, national-identity, “blockchain-secured mutex,” and tactical-drone
parts are neither technically established by the implementation nor necessary
to the enduring design. They should be treated as discarded period framing,
not requirements.

The lasting plot is better:

> The world initially appears to be a game. By playing carefully, the player
> constructs instruments, discovers that the instruments are real simulators,
> gains access to their graphs and models, and eventually participates in the
> work that keeps the world coherent.

This is the earliest recovered container broad enough to hold geometry,
physics, neural models, scheduling, fabrication, rendering, and collaboration
without presenting them as an arbitrary collection of developer tools.

### Why it applies directly to Pluck

**Verified in `spectral-analyzer`:**

- `PlayerController` has walk, camera, and station-interaction modes.
- The player collects tools and materials.
- Materials can be synthesized from recipes.
- Unfinished stations accept delivered materials.
- Construction/inventory state is persisted.
- `RoomWorkspace` contains placed fabricator and simulator stations.
- Cameras and station consoles are world-space interactables.
- Pluck's optical/physical graph is being aligned with Nodus authority and KPN
  scheduling.

Pluck therefore already has the mechanical grammar for Kakarot's plot. It does
not need a conventional skill tree pasted on top.

### Proposed progression: the machine reveals itself

1. **Phenomenon:** the player encounters sound, light, fields, surfaces, and
   unstable machines as world phenomena.
2. **Instrument:** collected materials permit construction of probes, cameras,
   analyzers, resonators, and fabrication stations.
3. **Model:** instruments reveal measurements and selectable explanatory
   models—mesh, field, metric, spectrum, state graph.
4. **Graph:** the player gains access to the real Nodus graph behind a station
   and can reroute or schedule it.
5. **Simulator:** a station exposes candidate futures, error channels, and
   rollback rather than pretending simulation is certain.
6. **Compiler:** advanced geometry tools reveal that an isosurface is a
   compiled parametric cell program; YoungMan belongs here.
7. **Network:** cooperation connects separately authoritative simulations
   through typed, budgeted proposals.

Unlock conditions should be material, observational, and causal:
build the detector, make the measurement, stabilize the process, understand the
dependency. They should not be arbitrary experience-point gates. The software
becomes “what it is” because the player's actions progressively earn truthful
views of the same underlying system.

### A concrete YoungMan station

**Proposal:** introduce a late-middle-game **field loom** or **surface loom**
station:

- the early interface shows only a field and a material surface;
- probes reveal sampled vertices;
- a geometry cartridge selects the parametric cell family;
- the loom view reveals weft/warp (`Shuttle`) composition;
- the graph view reveals batched query, bitmask, edge-crossing, and triangle
  stages;
- a metric lens activates Turing's learned/selected metric;
- scheduling telemetry shows parallel occupancy and bottlenecks;
- the resulting mesh is rendered and can become a physical/optical Pluck
  object.

This makes YoungMan's deep parallelism legible through play and gives its
recovery a product purpose beyond preserving a beautiful demo.

## 6. Other valuable Alpha archaeology

Alpha contains 318 Python files across the recovered numbered strata.
The following is a value inventory, not an endorsement of every implementation.

| Alpha area | Relevance now | Recommended disposition |
|---|---|---|
| Level 16 Kakarot `Primitives\isosurface4.py`, `ziggy.py`, geometry maps | Best runnable YoungMan context | Preserve exactly; hash caches and inputs; use as reference oracle |
| Level 16 `Heliograph` copies of YoungMan/geometry/triangulation | Alternate packaging of the same geometry effort | Diff before discarding; record unique behavior only |
| SpeakToMe-era `CompositeGeometryDEC`, Hodge/face-map work in Alpha Level 17 | Bridge from YoungMan cells to Laplace/DEC | Compare against Turing Riemann code; extract tests and mathematical assumptions |
| `laplace.py`, `laplace2.py`, metric-tensor experiments | Direct prehistory of SpeakToMe/Turing Laplace | Preserve as lineage and numerical test vectors |
| `temperaturesimulation*.py`, `springmass*.py`, `chipmunk_slice.py` | Local physics and surrogate-training fixtures | Keep as small managed-time migration cases |
| Kakarot `Networking` | Earliest graph of negotiating simulators and local learners | Preserve design vocabulary and scenarios; do not reuse protocol literally |
| `ThreadManager`, `PhysicalObject`, `CornerstoneShell`, `StateResolver` | Authority, state provenance, object physics | Mine tests/invariants; replace concurrency mechanism |
| `Devices` (`cpu`, `dvibus`, `computer_screen`, `window`) | Early device/bus/surface abstraction | Compare conceptually with Nodus ports/plugins and headless surfaces |
| `Renderer` (`tensor_renderer`, camera, lighting, test patterns) | Early batched rendering and inspection | Inventory shader/math ideas against Pluck; avoid parallel renderer revival |
| `audio.py`, `boombox.py`, `fftvisualizer.py` | Audio-to-geometry and analyzer ancestry | Compare with Pluck instruments and SpeakToMe audio paths |
| `mitsubashell.py`, camera/optical experiments | Optical rendering ancestry | Compare with Pluck camera/optical graph; preserve distinctive scene tests |
| `adaptivegraphnetwork.py` and graph utilities | Graph-shaped adaptive computation | Record algorithms and data contracts; assess against Nodus rather than copy |
| Level 18 `fields/{scalar,vector,tensor}`, `operators`, `nodes`, `system` | Early declarative field/operator/system decomposition | Valuable conceptual predecessor to Turing/Nodus; YAML schemas deserve a focused audit |
| Level 18 installers, bootloader, venv librarian, backup manager | Self-constructing workspace/runtime concept | Archaeology only unless a unique recovery feature is found |
| Level 8 patent/game narrative and large monolithic scripts | Earliest plot and motivation | Preserve as design history; explicitly reject obsolete identity/economic framing |
| Numbered-level duplication and nested `original tests` trees | Temporal evidence where Git history is absent | Build a manifest before deduplication; never bulk-copy into current repos |

### Particularly important Alpha branches for a follow-up pass

1. **Level 17 geometry/DEC:** determine whether any face-map, Hodge-star, or
   Shuttle behavior is absent from SpeakToMe and Turing.
2. **Level 18 declarative engines:** inspect the scalar/vector/tensor YAML
   contracts as possible ancestors of Nodus typed ports and Turing operators.
3. **Optical/rendering demos:** find behavioral scenes or calibration patterns
   not reproduced in Pluck.
4. **Audio geometry:** identify whether any field-to-sound or sound-to-surface
   mapping remains unique.
5. **Network experiments:** turn the Pillbug correction/training cycle into a
   small, deterministic specification before its metaphors are normalized.

## 7. Recovery plan

### Preserve first

Before reorganizing `C:\Alpha`:

1. create a file manifest with relative path, size, timestamp, and SHA-256;
2. separately hash the YoungMan geometry-map caches;
3. record Python import/runtime requirements for the working demo;
4. keep Alpha read-only during comparison;
5. archive source and small reference data separately from generated or
   third-party bulk assets.

Alpha should not be copied wholesale into the new root coordination repository.
The root repository intentionally ignores directory contents, and the current
subprojects have independent histories.

### Reconstruct YoungMan in stages

1. Freeze several deterministic Alpha/SpeakToMe field queries and their
   topology/triangle outputs.
2. Extract geometry definitions and topology-map compilation from rendering,
   dynamics, and OpenGL allocation.
3. Specify the backend-neutral parametric cell contract.
4. Implement a Torch reference path with equality tests against the recovered
   version.
5. Add `AbstractTensor` field evaluation and interpolation only after the
   topology contract is stable.
6. Attach Turing metric selection/refinement as an optional stage.
7. schedule the stage graph through Transmogrifier/Nodus facilities.
8. render through Pluck and expose it through the surface-loom station.

### Do not collapse responsibilities again

- Turing owns numerical/metric/managed-time semantics.
- Nodus owns graph structure, typed connection, and execution authority.
- Pluck owns inhabitable presentation, interaction, instruments, and rendering.
- Geometry topology should be a small explicit contract usable by all three.
- SpeakToMe and Alpha remain reference lineage until their behavior is covered
  by tests.

## 8. Source index

### Alpha

- `C:\Alpha\Alpha\import_uuid\tests\original tests\Level 16\Development\Kakarot\Primitives\isosurface4.py`
- `...\Kakarot\Primitives\ziggy.py`
- `...\Kakarot\Primitives\compositegeometry.py`
- `...\Kakarot\Primitives\threadmanager.py`
- `...\Kakarot\Primitives\stateresolver.py`
- `...\Kakarot\Networking\state_negotiation_contract.py`
- `...\Kakarot\Networking\debt_arbitor.py`
- `...\Kakarot\Networking\contract_renegotiator.py`
- `...\Kakarot\Networking\pillbug_network.py`
- `...\Kakarot\Networking\random_walk_simulator.py`
- `...\Kakarot\Networking\training_coordinator.py`
- `C:\Alpha\Alpha\import_uuid\tests\original tests\Level 8\Alpha\import_uuid\tests\original tests\patent.txt`

### Current workspace

- `speaktome/training/notebook/1733534643_YoungManAlgorithm.py`
- `speaktome/training/notebook/1733612620_CompositeGeometry_Shape_Shuttle_YoungManAlgorithm.py`
- `turing/src/common/tensors/riemann/manifold.py`
- `turing/src/common/tensors/riemann/grid_block.py`
- `turing/src/common/tensors/abstract_convolution/metric_steered_conv3d.py`
- `turing/src/transmogrifier/ilpscheduler.py`
- `turing/src/transmogrifier/graph/graph_express2.py`
- `turing/docs/managed_time_runtime.md`
- `spectral-analyzer/player_controller.py`
- `spectral-analyzer/ROOM_STATION_PLAN.md`
- `spectral-analyzer/NODUS_OPTICAL_KPN_INTEGRATION.md`

## Bottom line

Kakarot's enduring contribution is the plot and the composition:
geometry, simulation, learning, scheduling, and play were once imagined as one
machine. The present repositories contain stronger versions of most individual
parts, but not yet the recovered whole.

YoungMan is the largest genuine technical omission. Turing has the
metric-Laplace/neural half; Nodus and Transmogrifier have the graph/scheduling
half; Pluck has the world in which it can matter. Recovering YoungMan behind a
small topology contract—and letting a player construct and uncover it as a
surface loom—is the most coherent way to continue that lineage.
