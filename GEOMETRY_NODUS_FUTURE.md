# Geometry's eventual role in Nodus

**Status:** architectural direction, not a claim of present integration

**Date:** 2026-07-25

## Summary

The standalone `geometry/` repository is not presently ready to become a Nodus
dependency. It is a C/C++ scaffold for a native geometry engine whose intended
scope includes continuous parametric domains, delayed transforms, discrete
exterior calculus, metric tensors, generalized stencils, and YoungMan
isosurface extraction. Several of those capabilities exist only in its README.

Nevertheless, geometry will eventually be important to Nodus because Nodus
needs a rigorous answer to a question its graph and tensor systems should not
answer by themselves:

> What geometric space does a graph computation inhabit, and when should a
> continuous description become discrete data?

Nodus should remain the authority for graph structure, typed connections,
execution, handles, plugins, and inspectable state. A geometry subsystem should
provide domain and manifold semantics, late-evaluated coordinate transforms,
topology construction, metric-aware operators, and native geometry kernels.

The relationship should be:

```text
Nodus graph authority
    │
    ├─ declares geometry programs, dependencies, ports, and schedules
    ├─ owns durable tensor handles and cross-plugin data identity
    └─ chooses execution capabilities/backends
             │
             ▼
Geometry service / library
    ├─ continuous parametric domains
    ├─ late-evaluated transforms and implicit fields
    ├─ discrete topology and DEC operators
    ├─ metric-aware sampling and refinement
    └─ native vectorized kernels, eventually including YoungMan
             │
             ▼
Consumers
    ├─ Pluck rendering, optics, acoustics, and physical instruments
    ├─ Turing numerical and learned-metric computation
    ├─ mesh, raster, fabrication, CNC, and robotics backends
    └─ Nodus UI inspection and graph editing
```

Geometry should become a computational faculty of Nodus, not a competing graph
runtime and not the owner of Nodus state.

## Why Nodus needs geometry

Nodus already has several pieces that make a geometry layer useful:

- a C++20 graph editor and runtime;
- plugins and repository ingestion;
- typed ports and stable module/frame identities;
- tensor-backed edge data;
- a handle-based tensor abstraction and shared tensor-core work;
- portable `KernelIR` and backend capability descriptions;
- K-path geometry, raster, armature, machining, and painting concepts;
- headless and browser-facing inspection surfaces;
- integration with Pluck's physical, optical, and rendering systems.

Those facilities can transport coordinates and execute kernels, but coordinates
alone do not define geometry. A complete geometry contract must also express:

- the intrinsic parameter domain;
- boundaries, periodicity, and discontinuities;
- the map from intrinsic parameters to physical position;
- derivatives and metric tensors;
- discrete cells, faces, edges, and incidence;
- signed or implicit field evaluation;
- refinement and sampling policy;
- provenance: which domain, transform, metric, and topology produced a mesh.

Without a shared contract, every Nodus plugin can invent a different meaning
for “position,” “normal,” “distance,” “neighbor,” and “surface.” The graph will
remain connectable but not geometrically coherent.

## What the standalone geometry repository actually is

The repository at `geometry/` began after the original Kakarot and SpeakToMe
Python geometry experiments. Its documentation explicitly proposes:

- `parametric_domain`;
- `parametric_transform`;
- metric tensors and Laplace-Beltrami operators;
- delayed execution graphs;
- structured and unstructured DEC;
- an isoshell viewer;
- a native `youngman_algorithm` with dynamic bitmask libraries.

This makes it a clear architectural continuation of the Python lineage.
It is not a completed port.

### Implemented or partially implemented

- `ParametricDomain` supports up to sixteen axes.
- Axes carry ranges, inclusive/exclusive boundaries, periodicity,
  discontinuities, boundary conditions, and extrapolation callbacks.
- Graph, DAG, stencil, iterative-solver, and concurrency scaffolding exists.
- An Eigen-facing `DECSystem` API is declared.
- CMake defines a native static library and SIMD variants.

### Missing or largely declarative

- no `parametric_transform` implementation;
- no continuous manifold library;
- no implemented metric tensor engine;
- no implemented general meshgrid or granular-domain engine;
- no completed Eigen `DECSystem`;
- no Laplace-Beltrami implementation matching the README;
- no signed-distance or implicit-field contract;
- no YoungMan source files;
- no general late-evaluation pipeline connecting domains to topology;
- numerous placeholder graph/operator implementations.

The existing repository also mixes geometry with a broad experimental memory,
token, graph-operations, neural-network, synthesis, and assembly substrate.
Nodus now has stronger ownership of many of those responsibilities.

The useful future should therefore be recovered selectively.

## Responsibility boundary

### Nodus owns

- graph identity and topology at the application/runtime level;
- node, edge, port, and plugin identity;
- tensor-handle identity across host and plugin boundaries;
- scheduling, readiness, backpressure, and execution history;
- kernel capability tags and backend selection;
- serialization of the graph and its authored configuration;
- inspection, UI structure, and provenance surfaces;
- permission to create, modify, or publish geometry products.

### Geometry owns

- continuous domain definitions;
- coordinate charts and transforms;
- boundary, periodicity, discontinuity, and extrapolation semantics;
- metric, inverse metric, determinant, Jacobian, partials, and normals;
- implicit/signed field query contracts;
- structured and unstructured cell-complex construction;
- incidence matrices, Hodge stars, and geometric differential operators;
- spatial queries, refinement, intersection, and topology compilation;
- backend-neutral geometry-program descriptions;
- optimized CPU/GPU implementations of geometry kernels.

### Turing contributes

- the working `AbstractTensor` model;
- learned and authored metric functions;
- `GridDomain` and transform lineage;
- metric-aware Laplace construction;
- Riemannian neural blocks;
- managed-time simulation, validation, rollback, and retry;
- a reference environment for proving geometry operations backend-agnostic.

### Pluck contributes

- the inhabitable and inspectable presentation;
- ordinary OpenGL rendering;
- optical, acoustic, and physical consumers;
- cameras, instruments, fabrication, materials, and constructed stations;
- the gameplay context in which geometry becomes visible and useful.

## The core contract Nodus will eventually need

A geometry program should be data that Nodus can name, connect, inspect, and
schedule. It should not be an opaque pointer to a mesh generator.

One possible conceptual contract is:

```text
GeometryProgram
├─ domain
│  ├─ axes
│  ├─ ranges
│  ├─ boundary rules
│  ├─ periodicity
│  └─ discontinuities
├─ transform
│  ├─ intrinsic dimensions
│  ├─ physical dimensions
│  ├─ parameters
│  └─ backend/capability requirements
├─ field
│  ├─ scalar / vector / tensor result
│  ├─ signed-difference convention
│  └─ differentiability and precision requirements
├─ discretization
│  ├─ cell family
│  ├─ resolution/refinement policy
│  ├─ topology compiler
│  └─ boundary stitching policy
├─ metric
│  ├─ authored, derived, or learned
│  └─ evaluation and regularization policy
└─ outputs
   ├─ positions
   ├─ topology
   ├─ differential operators
   ├─ surface/volume attributes
   └─ provenance and error measures
```

Nodus should store or reference this program and route its tensor products.
The geometry implementation may lower it to Eigen, the Nodus tensor core,
Turing `AbstractTensor`, CPU SIMD, SPIR-V, or another capable backend.

## Late evaluation is the essential feature

The most important idea to preserve from the Python lineage is that a
continuous geometry should not be irreversibly resolved too early.

For example, YoungMan does not fundamentally need a permanently materialized
world mesh. It needs:

1. final query positions for the vertices of its selected parametric cells;
2. a signed scalar-field evaluation at those positions;
3. topology maps describing cell edges and possible surface triangles;
4. interpolation at active crossings;
5. optional metric-aware refinement at the resulting positions.

Jitter, motion, learned metrics, adaptive resolution, and camera/instrument
needs may change the final positions. If a domain is eagerly converted into one
fixed mesh before those decisions are known, the system loses both accuracy and
composability.

Nodus makes late evaluation more powerful because the graph can expose the
decision:

```text
domain ─┐
chart ──┼─> final-position query ─> signed field ─> topology case
pose ───┤                                  │
metric ─┘                                  └─> refine/interpolate ─> mesh
```

Every stage can have typed ports, explicit dependencies, capability tags,
profiling, and visible intermediate tensors.

## YoungMan as the first meaningful geometry integration

YoungMan is a good integration target because it exercises nearly every
boundary without requiring geometry to own the whole application.

### Geometry supplies

- cell-family definitions;
- local vertex and edge topology;
- topology-map compilation;
- signed-field sampling;
- edge intersection;
- optional centroid/metric refinement.

### Nodus supplies

- the executable stage graph;
- tensor handles between stages and plugins;
- scheduling and resource visibility;
- backend selection;
- inspection of bitmasks, active edges, triangles, and performance;
- publication of the resulting surface to consumers.

### Pluck supplies

- the ordinary OpenGL view;
- material and lighting presentation;
- interaction and instrumentation;
- a field/surface station through which the player can discover the process.

### Turing supplies

- the reference `AbstractTensor` implementation;
- metric and Laplace machinery;
- learned metric experiments;
- numerical agreement tests.

The current Turing YoungMan demonstration is therefore useful as a reference
prototype, not necessarily the final ownership location. It proves that:

- a domain can defer spatial resolution;
- the final signed field can be evaluated at requested positions;
- topology assembly can remain discrete;
- numeric interpolation can use a backend-agnostic tensor;
- Pluck can render the final triangle product.

## Geometry and Nodus K-path

Nodus K-path already treats rasterization, meshes, machining, painting, and
robotics as different integrations of a tool kernel along a metric-rich path.
A geometry service can unify this further:

- a parametric transform defines the path or surface chart;
- the metric defines meaningful distance and sampling density;
- an armature transform supplies final pose;
- a tool kernel supplies local occupancy or influence;
- a discretizer produces raster, mesh, cut, deposition, or collision products;
- Nodus schedules the shared program and routes products to backends.

This is a strong reason not to design geometry as “the mesh library.” Meshes are
only one realized product of the geometry program.

## Integration shape

The first integration should probably be a Nodus plugin or small native service,
not a wholesale link against the existing geometry repository.

Recommended boundary:

```text
Geometry plugin
├─ registers geometry-program and output types
├─ declares typed argument/return ports
├─ receives Nodus tensor handles
├─ evaluates a bounded geometry operation
├─ publishes new handles plus provenance
└─ advertises backend capabilities
```

Early plugins might include:

- `parametric_domain`;
- `transform_positions`;
- `signed_field_query`;
- `structured_cell_complex`;
- `youngman_topology`;
- `youngman_intersections`;
- `dec_incidence`;
- `metric_measure`;
- `surface_refine`.

Splitting these responsibilities allows Nodus to show the graph and schedule
them independently. A fused native kernel can be added later without erasing
the logical graph.

## Data identity comes before native geometry

The current `nodus_tensor_core` extraction work is a prerequisite. Geometry
plugins cannot safely exchange tensor data if the host and plugin have separate
backend registries, pools, or handle maps.

Before native geometry integration:

1. the shared tensor core must link and pass host/plugin handle-identity tests;
2. geometry input/output tensor specifications must be explicit;
3. topology tensors must use stable integer types and layouts;
4. plugin lifetime must not invalidate published tensor handles;
5. provenance must identify domain, transform, field, metric, and compiler
   versions;
6. capability negotiation must distinguish portable kernels from opaque native
   implementations.

Geometry should not work around handle identity with private allocations or
untracked pointers.

## Migration plan

### Phase 0 — preserve and specify

- Preserve the standalone `geometry/` history as archaeology.
- Do not treat its README roadmap as implemented behavior.
- Extract the useful `ParametricDomain` semantics into a language-neutral
  specification.
- Freeze reference outputs from Kakarot/SpeakToMe YoungMan and Turing Laplace.

### Phase 1 — reference geometry program

- Define a serializable geometry-program schema.
- Implement domain, transform, signed-field, and structured-cell reference
  operations in Turing.
- Maintain NumPy and pure-Python `AbstractTensor` agreement.
- Record topology and metric provenance.

### Phase 2 — Nodus type and plugin boundary

- Register geometry-program and topology types with Nodus.
- Create narrowly scoped plugins over the stable tensor core.
- Pass data through Nodus handles rather than copying ownership into plugin
  globals.
- Expose intermediate tensors to headless and browser inspection.

### Phase 3 — native kernels

- Port stable, tested operations to C++.
- Use Nodus tensor abstractions or explicit Eigen adapters.
- Add vectorized structured-cell and YoungMan kernels.
- Lower suitable numeric stages through `KernelIR`/SPIR-V.
- Preserve a reference backend for differential testing.

### Phase 4 — unstructured DEC and adaptive geometry

- Implement incidence and Hodge operations for explicit complexes.
- Support refinement driven by field error and metric distortion.
- Connect managed-time state for moving geometry.
- Define stitching and authority rules across independently refined domains.

### Phase 5 — Pluck and fabrication

- Present geometry programs as instruments and constructed stations.
- Render surface products through Pluck.
- Route the same programs to optical, acoustic, fabrication, and robotic
  consumers.
- Let users inspect the exact Nodus graph behind the world-space machinery.

## What not to do

- Do not import the entire current `geometry/` repository into Nodus as if it
  were a finished engine.
- Do not create a second graph authority inside geometry.
- Do not let geometry own Nodus tensor handles or plugin lifetimes.
- Do not define a mesh as the only durable representation of a manifold.
- Do not hide domain, transform, metric, or refinement provenance.
- Do not require every geometry program to be portable IR; capability tags are
  more honest.
- Do not mechanically translate Python classes before their data contracts and
  reference outputs are stable.
- Do not revive Kakarot's token locks as a geometry scheduler.

## Long-term result

When this relationship is mature, Nodus will be able to represent a continuous
geometric intention as an inspectable graph, defer its realization until the
final position and consumer are known, select an appropriate backend, and route
the resulting tensors to simulation, rendering, learning, or fabrication.

That gives the projects distinct and complementary identities:

- **Geometry** defines space and the lawful transition from continuous intent
  to discrete structure.
- **Nodus** defines how that work is connected, scheduled, identified, and
  inspected.
- **Turing** explores backend-agnostic numerical and learned geometric
  computation.
- **Pluck** makes the same machinery visible, interactive, and inhabitable.

Geometry becomes important to Nodus not because Nodus needs more shapes, but
because a general graph runtime eventually needs a shared meaning of space.
