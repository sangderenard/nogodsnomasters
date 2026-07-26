# ProcessGraph translation frontier

Status: active audit and implementation handoff, 2026-07-26.

## Intended routes

```text
Python AST
    -> Turing ProcessGraph
    -> BitOps-expanded ProcessGraph
    -> SSA
    -> C / GLSL / Nodus KernelIR

Turing ProcessGraph
    -> Nodus graph of AbstractTensor tools
```

`ProcessGraph` should be the shared semantic and scheduling hub. BitOps is a
lowering pass over that graph, not a competing graph format. SSA is a lowered
execution form, not the place to recover metadata discarded earlier.

## What is real today

- `ProcessGraph.build_from_ast` accepts an AST, path, or source string and
  structurally reflects Python nodes.
- BitOps contains arithmetic built from the eight Turing primitives and records
  those primitive calls into a `ProvenanceGraph`.
- `process_graph_to_ssa_instrs` schedules a ProcessGraph and emits SSA.
- `TapeCompiler` consumes that SSA for the eight primitive analog-tape
  instructions: `nand`, `sigma_L`, `sigma_R`, `concat`, `slice`, `mu`,
  `length`, and `zeros`.
- The Nodus canonical operation catalog contains 66 tensor operations and is
  verified against Turing's C opcode header.
- Nodus has three useful receiving structures: `KernelIR` for executable
  kernels; `GraphIR` / `AbstractOpGraph` for operator-driven graph edits; and
  `ToolIR` plus table tools for executable graph UI components.
- Nodus already has a real AbstractTensor implementation, in-memory backend,
  tensor FIFO/edge infrastructure, tool stack execution, and KPath tools.

The provenance-to-ProcessGraph bridge is now functional. It imports existing
nodes and edges directly, preserving operation name, argument position, kwargs,
argument object identities, and output identity. Compiler-only ProcessGraphs can
now schedule without instantiating the experimental physical-memory substrate.
The formerly expected-failing provenance -> ProcessGraph -> SSA -> tape tests
are green.

The first semantic vertical slice is also functional:

```text
Python AST
  -> semantic ProcessGraph + ProcessOp
  -> symbolic BitOps ProcessGraph expansion using the real Turing algebra
  -> metadata-preserving SSA
  -> Nodus GraphIR AbstractTensor tool graph
```

The current AST slice covers function arguments, constants, assignment,
arithmetic/bitwise binary expressions, unary expressions, comparisons, calls,
returns, and `if` value merging through `select`. Unsupported Python remains an
explicit `opaque_python` node.

BitOps expansion currently handles `bitand`, `bitor`, `bitxor`, `invert`,
`add`, `sub`, and `mul`. It invokes the existing `Turing` derived operations
with a symbolic graph carrier, so NAND/ripple-add definitions still have one
home. Unsupported nodes remain tagged `bitops_status=unexpanded`.

Nodus now supplies an AbstractTensor GraphIR operator set. Turing exports each
ProcessGraph operation as an `abstract_tensor_tool` node with directional
ports, named roles, and connections. Nodus validates canonical operation names
against its generated catalog while retaining structural nodes explicitly.

Commits:

- Turing `5ce2d29` — semantic ProcessGraph and BitOps/SSA spine.
- Turing `d2456d8` — BitBit accounting and Nodus GraphIR export.
- Turing `61a4b30` — metadata-rich SSA to shared C/GLSL primitive programs.
- Turing `f5f8a1e` — C indexed assignment and composed sliced solve.
- Turing `5d787b1` — fused GLSL backend and five-way parity benchmark.
- Nodus `fc3c6f9` — AbstractTensor tool-graph receiver.
- Nodus `0f5aa7e` — 66 canonical IDs and corrected KernelIR BitOps selectors.

## BitBit quanta and provenance contract

Bit-level lowering must not reduce BitBit storage to an anonymous integer
width. `BitQuantaSpec` now carries:

- mask-plane quantum count;
- `bitsforbits` payload width per quantum;
- PID provenance-domain labels;
- source ProcessGraph node identities.

It can describe a live `BitBitBuffer` without copying its mask plane, data
plane, or UUID tables. Primitive BitOps nodes carry the accounting record;
SSA values retain it; Nodus exports it as `bitbit.quanta`,
`bitbit.bitsforbits`, and optional PID-domain metadata.

## The representations that still do not connect

### AST -> ProcessGraph is structural, not yet a Python semantics compiler

It recognizes Python syntax through generic AST introspection, but does not yet
establish symbol definition/use, lexical scope, branches, joins, or loop-carried
values as semantic dataflow. BitOps has a second AST-to-`ProcDAG` experiment
which recognizes only `Module`, `Assign`, `Name`, `BinOp`, `Constant`, `Call`,
`FunctionDef`, and `Return`. These paths should be consolidated by teaching the
ProcessGraph AST importer semantics, then deleting or adapting the private
`ProcDAG`.

### ProcessGraph -> BitOps ProcessGraph is partial

The new pass rewrites supported nodes into primitive ProcessGraph subgraphs.
Division, modulus, dynamic shifts, comparisons, and general control-flow
lowering remain. They require runtime-visible conditions or scalar parameters
rather than compile-time Python branching.

### ProcessGraph -> SSA currently loses information

The emitter preserves scheduled operation order and graph value ids. It does
not preserve constants, kwargs, argument roles beyond ordering, tensor
descriptors, device/backend, multiple outputs, source spans, names, basic
blocks, branch targets, or lexical scope.

Legacy nodes still emit their label verbatim. Semantic `ProcessOp` nodes carry
canonical names, roles, scalar constants, attributes, source spans, tensor
dtype/shape/device, and BitBit accounting into SSA. Symbolic labels, SSA
`Handler` spellings,
AbstractTensor operation names, C opcodes, and Nodus KernelIR opcodes are
related but not identical. Canonical operation identity must be attached to
each node before SSA emission.

### Nodus GraphIR bridge exists; execution binding remains

Nodus GraphIR receives the exported graph and emits AbstractTensor tool nodes
and ports. Its scalar `GraphIrValue` still cannot natively carry full tensor
descriptors, multi-output bundles, source spans, or control-flow blocks; the
current bridge encodes structured attributes as stable strings where needed.

## C and GLSL execution bridge

Metadata-rich SSA now lowers into the same `PrimitiveProgram` already consumed
by the one-call C executor and fused GLSL backend. Numeric scalar constants,
operand reversal, canonical unary/binary operations, `nand`, and tensor
`select` are supported. The C path is executed in tests; the same result adapts
to a validated fused GLSL shader.

The equal-shape packet cannot honestly represent `zeros`, `concat`, `slice`,
`sigma_L`, `sigma_R`, or general BitBit `mu` when their shapes differ. These
produce structured `LoweringIssue` records and no executable program. The next
backend packet must add views/regions and shape descriptors rather than hiding
those boundaries.

`ToolIR` is a callback bundle, not a computational IR. The table tensor tool is
currently a runtime placeholder. A tensor node therefore needs a generated
tool wrapper around a canonical operation plus typed ports; ToolIR itself
should not become the tensor instruction format. The `TranslationMatrix` is
still an unused backend-name-to-callback stub.

## Operation parity snapshot

The executable audit, using Nodus's 66-operation canonical catalog, reports:

| Surface | Count |
|---|---:|
| canonical operations | 66 |
| complete across audited C, GLSL, and Nodus lowering | 28 |
| C-native | 40 |
| GLSL | 28 |
| Nodus KernelIR-lowerable | 56 |

Missing in both C and GLSL:

`sign`, `invert`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`,
`cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `bitand`, `bitor`, `bitxor`,
`shl`, `shr`, `logical_and`, `logical_or`, `int_trunc`, `zext`, `sext`,
`fptoui`, and `uitofp`.

Missing only in GLSL among primitive/cast entries: `fptosi` and `sitofp`.

Cataloged high-level operations not represented as one GLSL or Nodus KernelIR
instruction:

`matmul`, `sum`, `mean`, `topk`, `log_softmax`, `pad`, `stack`, `cat`,
`gather`, and `arange`.

Those high-level operations need not all become primitive opcodes. Many should
remain canonical composite functions expanded into basic operators before
backend lowering. The catalog needs an explicit distinction between
“composite and expandable” and “unsupported”; its current `lowerable` boolean
cannot express that.

Beyond the 66-op catalog, the C backend hook comparison identifies these
NumPy-shaped entry points as absent:

`allclose_`, `bool_`, `diag_`, `double_`, `einsum_`, `float_`, `fold2d_`,
`int_`, `long_`, `nonzero_`, `pad_cat_`, and `unfold2d_`.

This is not yet full AbstractTensor parity. A complete audit must add
specialization modules and composite AbstractTensor methods, classify every
entry as primitive/composite/backend-specific, and exercise dtype, shape, and
device variants. Method-name equality alone overstates parity.

## Existing SSA vocabulary

Turing's `Handler` enum contains 42 entries across arithmetic, bitwise,
logical, comparison, memory/indexing, casts, control flow, and calls. Its name
map is valuable but lossy:

- elementary math collapses to `Call`;
- `floordiv` has no Handler;
- floating `trunc` and integer-width `Trunc` must not be merged;
- logical not and bitwise invert require dtype-aware lowering;
- the SSA helper registry is empty until external decorators are imported.

The canonical catalog already documents several of these traps. It should
become the executable correlation source for ProcessGraph node annotation and
SSA emission.

## Ordered implementation plan

1. **Done for the vertical slice:** define a serializable `ProcessOp` payload:
   canonical op id, ordered operands and roles, outputs, attributes, tensor
   descriptors, constants, control metadata, and source span.
2. **Partial:** make AST import semantic for expressions, assignment, call,
   return, branch, and loop/phi. Keep unknown constructs as explicit opaque
   nodes.
3. **Partial:** adapt BitOps to rewrite ProcessGraph nodes into primitive
   subgraphs, reusing its implementations and provenance recorder.
4. **Partial:** upgrade SSA emission to consume `ProcessOp`, preserving constants,
   attributes, types, blocks, and multiple results.
5. **One of two connected:** add independent Nodus consumers:
   - ProcessGraph/SSA -> KernelIR for fused backend execution;
   - ProcessGraph -> GraphIR edits -> generated AbstractTensor tools and ports.
6. Generate backend decisions from the canonical catalog. Composite ops expand
   before the backend frontier; native ops remain eligible for fusion.
7. Fill primitive C and GLSL holes, then use conformance vectors across NumPy,
   Torch, C, GLSL, and Nodus for dtype, broadcast, indexing, and edge cases.

## Immediate vertical test

```python
def kernel(x, y, n):
    z = (x + y) * 3
    if z > n:
        z = z ^ n
    return z
```

The test should assert preserved identity and metadata at AST, ProcessGraph,
BitOps-expanded ProcessGraph, SSA, and Nodus graph/tool stages, then compare
execution across NumPy, Torch, C, GLSL, and Nodus where supported. Unsupported
lowering must be structured data, never silently replaced or skipped.
