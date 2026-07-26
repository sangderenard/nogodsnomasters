# Before Kakarot: the early lineage of the workspace

**Status:** historical and architectural reconstruction

**Date:** 2026-07-25

## Summary

The projects now gathered around SpeakToMe, Turing, Nodus, Pluck, and the
geometry work did not begin with Kakarot. An earlier browser-based JavaScript
system already contained embryonic versions of several ideas that later became
central:

- continuous interpolation over authored control points;
- arbitrary named data axes;
- vector-valued data processing;
- operations assembled into editable chains;
- visual panels connected into a process flow;
- asynchronous workers with progress, cancellation, and stale-work rejection;
- cached datasets refined in the background;
- derivative- and integral-aware resampling over a media span;
- reversible control histories;
- dense and sparse representations of the same underlying signal.

Kakarot and Alpha expanded those ideas from a media-processing application into
a much broader computational geometry, physics, scheduling, simulation, and
game architecture. SpeakToMe then inherited that architecture and developed
several of its mathematical and graph-oriented branches more explicitly.

The lineage is therefore not:

```text
Kakarot → an unrelated set of later projects
```

It is closer to:

```text
early interpolation and process-flow workspace
    → increasingly general data and execution abstractions
    → Kakarot / Alpha
    → SpeakToMe
    → Transmogrifier and Turing
    → Nodus, Pluck, and the current geometry program
```

This report describes that continuity without claiming that every later module
is a direct source-code port of an earlier one.

## The first recognizable system

The earliest recovered layer was a browser application for constructing,
interpolating, redistributing, visualizing, and exporting time-based workout
and media data.

On its surface it was a specialized application. Underneath, it had already
become a general experiment in representing a continuous process with editable
discrete controls.

Authored intervals supplied sparse control values over time. The system could
convert those intervals into continuous functions, inspect the functions,
densify them, redistribute sample locations, and convert the result back into
an interval program. It experimented with:

- linear interpolation;
- polynomial interpolation;
- cubic splines;
- B-splines;
- PCHIP;
- Akima interpolation;
- Bézier interpolation;
- Catmull–Rom interpolation;
- wavelet-like reconstruction;
- numerical slope estimation;
- numerical integration;
- repeated interpolation passes;
- slope-sensitive redistribution;
- sinusoidal and randomized redistribution;
- hill-climbing;
- fixed-point-count reduction and expansion.

The important historical fact is not that every implementation was complete or
correct. The important fact is the form of the problem:

> Preserve the meaning of a continuous process while changing how, where, and
> how densely it is discretized.

That question remains central to the current spline, geometry, YoungMan, mesh,
managed-time, and sampling work.

## Data structures before they were called tensors

The early system began with ordinary points and intervals, but its data model
grew beyond fixed two-dimensional samples.

A point could carry an open-ended collection of named coordinates. The system
distinguished:

- domain axes used to locate a sample;
- range axes whose values were functions of the domain;
- detail axes carrying color or secondary information;
- index axes preserving order and identity;
- inherited axes transferred between differently sampled datasets.

Datasets could calculate statistics independently for every axis, choose a
usable interpolation domain, interpolate several output axes at once, preserve
secondary channels across changes in resolution, and map between parent and
child point collections.

This was not yet a conventional tensor implementation. It had no formal
shape/stride algebra, broadcasting contract, or backend abstraction.
Nevertheless, it was already treating data as a labeled, multidimensional
object whose axes had distinct semantics. In hindsight, it is one ancestor of
the later AbstractTensor work.

The vector interpolator makes the relationship especially clear. Its effective
contract was:

```text
one domain coordinate
    → several synchronized range coordinates
```

The later geometry work generalizes the same idea:

```text
an intrinsic parameter vector
    → a point in an embedding space
    → derivatives and a metric induced by that map
```

## The early process-flow runtime

The browser system also evolved beyond a collection of mathematical
functions. It had an editable operation-chain runtime.

Operations were represented as chain links with:

- typed or labeled inputs;
- user-supplied parameters;
- cached input and output data;
- persistent operation-local state;
- preview and full-resolution modes;
- change detection;
- dynamically constructed operation classes;
- reordering and removal;
- recursive upstream evaluation;
- downstream propagation when an output changed.

This was already a form of incremental dataflow execution. A link did not need
to recompute merely because the interface asked for its output. It compared
inputs and parameters, reused cached state, and notified downstream work only
when its result changed.

A visual manager surrounded those chains with processing panels and
connections. Panels could be added, removed, ordered, connected, and supplied
with previews and controls. The interface and computation were not cleanly
separated by modern standards, but the architectural instinct was already
present:

> Make computation visible as editable structure.

That instinct later reappears in Kakarot's gameplay and scheduling ideas,
SpeakToMe's graph experiments, Transmogrifier's executable graph work, and
Nodus's explicit graph editor/runtime.

## Scheduling, cancellation, and process generations

The early worker system treated long-running computation as managed activity
rather than one blocking function call.

It supplied:

- asynchronous iteration;
- progress reporting;
- pause and cancellation state;
- parent and child process identities;
- several workers over one operation;
- cooperative scheduling;
- accumulated partial results;
- cancellation of superseded work.

The progress system compared process generations and could terminate an older
operation when a newer version of the same logical process appeared. That is a
primitive form of revision-aware execution and stale-result rejection.

This concern recurs throughout the later lineage:

- Kakarot's lock and scheduling experiments;
- graph execution and propagation;
- Turing's managed scientific time and transactional advancement;
- Nodus's thread, handle, and execution boundaries;
- live geometry and rendering snapshots;
- FIFO-fed spline and mesh generations.

The implementations differ, but the persistent problem is the same: useful
work must continue concurrently without allowing an obsolete computation to
overwrite newer state.

## Cached densification

One particularly revealing early component proposed a background cache that
would:

1. retain sparse datasets;
2. lock one dataset for an update;
3. densify it from control points;
4. publish the refined cache;
5. unlock it;
6. schedule another refinement pass.

That mechanism was incomplete, but its intended architecture closely resembles
the current live geometry direction:

```text
incoming control points
    → queued accumulation
    → continuous reconstruction
    → tolerance-driven refinement
    → immutable published generation
```

The modern version can replace coarse timer-and-lock coordination with FIFO
inputs, bounded snapshots, explicit generations, and atomic publication.
Conceptually, however, the desire was already there: a continuous object should
become more resolved in the background without interrupting its use.

## From the early system to Kakarot and Alpha

The early application was organized around a media span. Time was the primary
domain, values such as power were ranges, and operation chains transformed one
representation into another.

Kakarot and Alpha broadened the domain:

- points became geometry and simulation state;
- interpolation became parametric geometry and field evaluation;
- operation chains became scheduling and executable graph experiments;
- media intervals became more general spatial and temporal domains;
- cached refinement became dynamic model resolution;
- visualization became a possible game and interface;
- asynchronous processes became interacting simulated systems;
- arbitrary axes became geometric coordinates, fields, and metrics;
- discrete samples became meshes, cells, graphs, and physical networks.

Kakarot also introduced or consolidated ideas involving:

- parametric manifolds and transformations;
- signed or implicit geometry;
- discrete exterior calculus;
- mesh and isosurface generation;
- lock scheduling;
- network-mediated resolution of simulations;
- neural components;
- gameplay that reveals the machinery by interacting with it;
- systems whose computational organization is itself part of the world.

Alpha preserved multiple stages of that work and became an archaeological
record of experiments that were later separated, renamed, or rewritten.

The transition was not simply an increase in code size. It changed the ambition
from “edit and process one continuous media program” to:

> Build a world in which continuous models, discrete computation, execution
> resources, geometry, and interaction can all describe and modify one another.

## How Alpha and Kakarot begat SpeakToMe

SpeakToMe inherited both mathematical material and architectural habits from
the Alpha/Kakarot period.

Its visible descendants include:

- parametric geometry and transform experiments;
- metric tensors and Laplace–Beltrami machinery;
- discrete exterior calculus;
- grid and volume construction;
- YoungMan isosurface extraction;
- graph-shaped computation;
- multidimensional interpolation and field handling;
- model-driven and neural experiments;
- aggressive parallelism;
- visualization as an instrument for understanding computation.

SpeakToMe also reframed the older concern with continuous media as a concern
with language-model search.

Its bidirectional beam and FluxGraph work again asks how a continuous or
wide-valued process can be represented without prematurely collapsing it into
one discrete answer. Alternative branches remain alive, acquire fields and
resources, grow in parallel, and can be sampled or viewed as a structured
region rather than only as a winning sequence.

That is a conceptual continuation of the earliest interpolation system:

```text
do not confuse the current samples with the underlying process
```

In the early system, the underlying process was a workout or media curve. In
SpeakToMe, it is a region of model belief and possible language.

## Later branches

Several later projects specialize parts of this inheritance.

### Transmogrifier

Transmogrifier developed executable graphs, scheduling, SSA, memory models,
binding membranes, and compiler-like transformations. Its active form now
belongs within Turing.

### Turing

Turing formalizes backend-agnostic tensor execution, compiler and graph
research, metric-aware numerical methods, managed time, and the integrated
Transmogrifier work. AbstractTensor gives a deliberate name and contract to a
problem that the early named-axis datasets approached informally.

### Nodus

Nodus makes editable computation, durable graph identity, typed connections,
execution, plugins, handles, and inspection first-class. It is the clearest
modern continuation of the old visual process-panel and function-chain
instinct, though its implementation and scope are much more rigorous.

### Pluck

Pluck joins media, physical simulation, optics, cameras, instruments, and
rendering. It continues the early refusal to separate a computed process from
the interface through which it is perceived and manipulated.

### Geometry

The geometry program carries the continuous/discrete boundary forward:
parametric domains, delayed transforms, metric tensors, topology, refinement,
splines, and triangulation. It asks the generalized version of the original
resampling question:

> How should a continuous geometric program become discrete data at a requested
> tolerance, in a requested embedding, only when a consumer needs it?

## The recurring architecture

Across the lineage, the vocabulary changes but a common structure persists:

| Early form | Later form |
|---|---|
| authored workout controls | sparse control points or graph state |
| interpolation over time | parametric maps and learned/local charts |
| dense workout reconstruction | deferred geometric resolution |
| slope-aware redistribution | derivative- and curvature-driven refinement |
| arbitrary coordinate dictionaries | tensor axes and geometric coordinates |
| function chains | executable graphs and compiler IR |
| control panels and connections | graph editors and inspectable runtime nodes |
| worker generations | revisioned scheduling and managed time |
| dataset cache | immutable snapshots and durable tensor handles |
| media-span visualization | Pluck rendering and interactive simulation |
| export back to intervals | mesh, toolpath, language, or backend materialization |

This continuity matters because it identifies which ideas are foundational and
which are implementation accidents.

The foundational ideas are:

1. Preserve a continuous or wide-valued model behind its discrete samples.
2. Let resolution follow error, purpose, and available compute.
3. Treat coordinates and channels as meaningful axes.
4. Make processing an editable and inspectable graph.
5. Keep long-running work concurrent, cancellable, and generation-aware.
6. Permit several representations of the same underlying process.
7. Return refined results without stopping the system that requested them.
8. Make visualization and interaction part of computation, not an afterthought.

## Historical cautions

This is an architectural reconstruction, not a claim that every modern module
was copied directly from the early JavaScript system.

Some early algorithms are incomplete, duplicated, or mathematically uncertain.
They should be preserved as evidence and inspiration, not imported blindly.
Their value is that they reveal the shape of the problem before the project had
the vocabulary it uses today.

Likewise, Kakarot and Alpha are not one clean release boundary. They are a
period and collection of experiments whose components moved through several
repositories. “Begat” here means that SpeakToMe demonstrably continues their
ideas and contains descendants of their work, not that the lineage can always
be represented by one uninterrupted commit history.

## Conclusion

The earliest recovered work shows that this ecosystem began with a deceptively
simple concern: how to transform a sparse, time-based program without losing
the continuous process it represents.

That concern generated interpolation engines, multidimensional datasets,
operation chains, visual process management, worker scheduling, cached
densification, and reversible controls. Kakarot and Alpha generalized those
ideas into geometry, simulation, networks, scheduling, and a game-shaped
computational world. SpeakToMe inherited that expansion and carried it into
tensor mathematics, graph computation, physicalized search, and language-model
exploration.

The current spline, YoungMan, triangulation, metric, managed-time, Nodus, and
Pluck work is therefore not a collection of unrelated new ambitions. It is a
more explicit and technically mature return to the questions present at the
beginning.
