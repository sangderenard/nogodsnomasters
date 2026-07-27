# ProcessGraph translation frontier

Status: active, verified 2026-07-27.

## The governing idea

`ProcessGraph` is the high-level claim about a program: mathematical,
inspectable, role-aware, schedulable, and capable of a physical realization.
SSA and `FusedProgram` are progressively lower views of selected regions.
Execution tapes can supply evidence, profiling, and convenient straight-line
captures, but they are not authoritative for source control flow they never
observed.

The intended continuum is:

```text
Python AST or SymPy mathematics
        ↓
semantic ProcessGraph
  roles, source, control, domains, provenance, tensor and BitBit metadata
        ↓
optional Turing/BitOps proof expansion
        ↓
scheduled SSA and region selection
        ↓
AbstractTensor replay | one-call C | one-shader GLSL | Nodus GraphIR
        ↓
optional ProcessGraph physical materialization
```

The aim is not to make Turing and Nodus halves of one inseparable animal.
Each remains capable on its own; when both are present, the same program and
operation identity can travel between them.

## The recovered semantic/physical seam

The original `ProcessGraph` is backed by `BitTensorMemoryGraph`. Process nodes,
`DomainNode`s, interference graphs, allocation bins, the unified graph, and
the resulting data-flow graph constitute a physical analogue of the program:
atomic graphs in motion and communicating according to activity.

Those structures are not obsolete allocator noise. They connect an
inspectable mathematical program to material execution and quanta accounting.

The older `ProcessGraph.extract_full_process_graph()` is the remembered
semantic projection. It exports node type, label, expression, role-bearing
parents and children, schedule level, and roots without exporting the physical
memory graph, domains, bins, or moving buffers.

The newer `materialize_memory=False` option only lets compiler front ends defer
physical construction. It does not replace the two-layer design.

## What is working

### Source and mathematics

- `ProcessGraph.build_from_ast` accepts an AST, source string, or file.
- Its semantic mode establishes definition/use dataflow for the current Python
  subset and records source spans, constants, ordered input roles, attributes,
  tensor metadata, control metadata, and roots.
- Arithmetic syntax and canonical tensor calls both enter the graph. For
  example, `tanh((x + y).sin())` becomes
  `input, input, add, sin, tanh, return`.
- Simple `if` assignment is predicated into an explicit `select`.
- Unsupported Python is retained as `opaque_python`, not silently executed or
  discarded.
- The original SymPy path, recombinatorics, `OperatorDef`, `Correlator`, and
  handler registries remain important. They have not been superseded by the
  Python importer.

### BitOps and the physical bit calculus

`Turing.Hooks` defines eight mechanics over an opaque bitstring carrier:

`nand`, `sigma_L`, `sigma_R`, `concat`, `slice`, `mu`, `length`, and `zeros`.

Every Boolean and integer operation is derived from those mechanics. The
provenance wrapper records calls to the eight primitives, and the resulting
graph can be imported into `ProcessGraph`.

BitOps expansion currently replaces `bitand`, `bitor`, `bitxor`, `invert`,
`add`, `sub`, and `mul` with those recorded primitive graphs. Other operations
stay visible and are marked `bitops_status=unexpanded`.

AbstractTensor is now a real BitOps carrier. Its hook adapter composes ordinary
tensor arithmetic, concatenation, slicing, shape inspection, and same-backend
construction; it adds no bit-specific tensor methods. Derived AND, XOR,
ripple addition, and multiplication were verified on NumPy and the native C
backend. ProcessGraph expansion can select this carrier through a translator
factory.

### SSA and execution

- `process_graph_to_ssa_instrs` preserves canonical operation names, ordered
  roles, constants, scalar attributes, source spans, tensor dtype/shape/device,
  and BitBit accounting.
- `lower_ssa_to_fused_program` lowers compatible equal-shape regions into the
  established backend-neutral `FusedProgram`.
- The same `FusedProgram` can run through AbstractTensor, the one-boundary C
  executor, the GLSL whole-program emitter, and the Nodus calculator transport.
- `process_graph_to_nodus_graph_ir` separately exports high-level graph
  structure as AbstractTensor operation tools, typed ports, and connections.

A source-defined `add → sin → tanh` kernel was verified end to end:

```text
Python source
  → ProcessGraph
  → SSA
  → one FusedProgram
  → NumPy AbstractTensor
  → C AbstractTensor
  → one GLSL compute dispatch on an RTX 3060
```

The GLSL output agreed with the numerical reference within about `3.1e-7`.

## Operation-table state

The universal table has two complementary faces:

- Turing `operator_definitions`: roles, signatures, parameters, concurrency,
  in-place rules, and high-level handlers.
- Nodus `canonical_ops.json`: append-only cross-language IDs and exact
  correlations to Turing handlers, SymPy names, C operations, and KernelIR.

Neither should be replaced by another private switch statement. They should
eventually be generated or verified together.

Current verified counts:

| Surface | Count |
|---|---:|
| Canonical cross-language operations | 66 |
| GLSL canonical primitives | 56 |
| Whole-program equal-shape fused operations | 40 |
| CTensor opcodes reported by the catalog verifier | 40 |
| Nodus KernelIR-lowerable operations | 56 |

The ten canonical operations outside GLSL's primitive set are:

`arange`, `cat`, `gather`, `log_softmax`, `matmul`, `mean`, `pad`, `stack`,
`sum`, and `topk`.

Several already have standalone GLSL kernels. Their absence from the single
elementwise packet is a region-composition frontier, not proof that the backend
lacks them.

## What remains

### 1. Whole-program control flow

The repository already defines SSA `BasicBlock`, `Phi`, `Br`, `CondBr`, and
`Ret`, and ProcessGraph has detailed structural AST role schemas. The active
semantic importer still leaves `for` and `while` opaque, while the active SSA
builder emits a linear instruction sequence.

The correct next step is to connect those existing structures. Do not invent a
second control-flow graph and do not infer source control from a runtime tape.

### 2. Region partitioning

The whole-program GLSL emitter elegantly folds compatible elementwise
intermediates into shader locals: one dispatch, no intermediate buffers.
Shape changes, views, reductions, matmul, and other structural operations need
explicit neighboring regions.

A ProcessGraph partitioner should:

1. find maximal regions accepted by a backend;
2. fuse those regions;
3. route boundary nodes through existing specialized kernels;
4. preserve roles, shapes, provenance, and source across region edges;
5. allow a more capable backend to claim a larger region.

### 3. SymPy/Python convergence

SymPy recombinatorics and semantic Python import currently construct related
but partially parallel ProcessGraph vocabularies. Both should resolve through
the same operation definitions and correlation data so pure mathematics and
ordinary tensor source become interchangeable front ends.

### 4. Structural BitOps

Shape-changing primitives such as `zeros`, `concat`, `slice`, shifts, and
general `mu` cannot be squeezed honestly into the current equal-shape packet.
They already produce structured lowering issues. A region/view representation
should describe storage ranges and shapes rather than hiding copies or
flattening.

### 5. Nodus execution binding

Nodus receives both the narrow fused numerical transport and the richer
GraphIR tool graph. The next milestone is execution of a ProcessGraph-derived
multi-region program inside Nodus while retaining graph/tool inspectability,
not just replaying one elementwise calculator packet.

## Verification

Focused Turing checks:

```powershell
python -m pytest tests/test_abstract_tensor_bitops.py tests/test_ast_process_graph.py tests/test_bitops_process_graph.py tests/test_ssa_primitive_lowering.py tests/test_nodus_graph_ir.py -q
```

Current result: **16 passed**.

Relevant commits:

- Turing `5ce2d29` — initial semantic ProcessGraph translation spine.
- Turing `85659b6` — consolidation into the original ProcessGraph.
- Turing `d2456d8` — BitBit accounting and Nodus GraphIR export.
- Turing `a385d89` — AbstractTensor BitOps carrier, natural tensor-call import,
  and verified source-to-GLSL route.

The longer architectural record is in
`research/16_process_graph_tensor_continuum.md`.
